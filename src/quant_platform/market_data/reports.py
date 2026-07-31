"""Deterministic reporting for `quant_platform.market_data` (Milestones
10 Phase 1 and Phase 2) -- every section is recomputed FRESH from the
store's own raw entries each time (never a cached/stale derived value),
mirroring `portfolio_risk.reports.generate_portfolio_risk_session_report`'s
identical convention exactly.

Phase 2 adds one report function per required concern (ingestion batch
result, dataset manifest summary, recovery, reconciliation, verification,
export). The manifest/reconciliation/verification reports re-read current
repository state on every call (a dataset-state QUERY, always fresh); the
ingestion/recovery/export reports instead wrap the result object the
corresponding operation itself already returned -- for those three, "what
happened in that specific completed operation" IS the durable evidence
(the operation already performed real, durable store reads/writes to
produce that result); there is no separate "fresher" state to re-query."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.market_data.events import MarketEventStore, market_data_event_id
from quant_platform.market_data.export import ExportResult
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.ingestion import IngestionResult
from quant_platform.market_data.manifests import DatasetKey
from quant_platform.market_data.recovery import RecoveryReport
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.verification import verify_feature_store, verify_market_event_store
from quant_platform.ml.models import ValidationReport

__all__ = [
    "MarketDataReport",
    "generate_dataset_manifest_report",
    "generate_export_report",
    "generate_ingestion_batch_report",
    "generate_market_data_report",
    "generate_reconciliation_report",
    "generate_recovery_report",
    "generate_verification_report",
]


@dataclass(frozen=True, slots=True)
class MarketDataReport:
    instrument_id: str
    sections: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return {"instrument_id": self.instrument_id, "sections": self.sections}


def generate_market_data_report(
    *, event_store: MarketEventStore, provider: str, instrument_id: str, feature_store: FeatureStore | None = None,
    feature_partitions: tuple[tuple[str, int], ...] = (), report_time: datetime,
) -> MarketDataReport:
    """`feature_partitions` is a tuple of `(feature_name, feature_version)`
    pairs to include in the report's feature-store section -- the report
    itself has no way to enumerate every feature series that might exist
    for `instrument_id` (the store has no directory-listing API by
    design, matching the ledger-style stores elsewhere in this
    repository), so the caller states which ones it cares about."""
    events = event_store.read_events(provider, instrument_id)
    event_counts_by_kind: dict[str, int] = {}
    for event in events:
        kind = str(event.to_json_dict()["kind"])
        event_counts_by_kind[kind] = event_counts_by_kind.get(kind, 0) + 1

    event_verification = verify_market_event_store(store=event_store, provider=provider, instrument_id=instrument_id, as_of=report_time)

    feature_sections: dict[str, object] = {}
    if feature_store is not None:
        for feature_name, feature_version in feature_partitions:
            records = feature_store.read_records(feature_name, feature_version, instrument_id)
            feature_verification = verify_feature_store(
                store=feature_store, feature_name=feature_name, feature_version=feature_version, instrument_id=instrument_id, as_of=report_time,
            )
            feature_sections[f"{feature_name}_v{feature_version}"] = {
                "record_count": len(records),
                "first_timestamp": (None if not records else records[0].timestamp.isoformat()),
                "last_timestamp": (None if not records else records[-1].timestamp.isoformat()),
                "critical_issue_count": len(feature_verification.criticals),
            }

    sections: dict[str, object] = {
        "MarketEventSummary": {
            "provider": provider, "instrument_id": instrument_id, "total_events": len(events),
            "by_kind": event_counts_by_kind, "event_ids_sample": [market_data_event_id(e) for e in events[:10]],
        },
        "EventVerificationSummary": {
            "critical_count": len(event_verification.criticals), "total_issue_count": len(event_verification.issues),
            "generated_at": event_verification.generated_at,
        },
        "FeatureStoreSummary": feature_sections,
    }
    return MarketDataReport(instrument_id=instrument_id, sections=sections)


# --------------------------------------------------------------------------
# Milestone 10, Phase 2.
# --------------------------------------------------------------------------
def generate_ingestion_batch_report(result: IngestionResult) -> dict[str, object]:
    return {
        "batch_id": result.batch_id, "dataset_key": result.dataset_key.to_json_dict(), "content_digest": result.content_digest,
        "resulting_dataset_id": result.resulting_dataset_id, "appended_event_count": result.appended_event_count,
        "rebuilt_partition_keys": list(result.rebuilt_partition_keys), "was_idempotent_replay": result.was_idempotent_replay,
    }


def generate_dataset_manifest_report(*, repository: MarketDataRepository, dataset_key: DatasetKey) -> dict[str, object]:
    manifest = repository.manifest_store.read_current(dataset_key)
    history = repository.manifest_store.read_history(dataset_key)
    if manifest is None:
        return {"dataset_key": dataset_key.to_json_dict(), "exists": False, "version_count": 0}
    return {
        "dataset_key": dataset_key.to_json_dict(), "exists": True, "dataset_id": manifest.dataset_id, "version_count": len(history),
        "event_count": manifest.event_count,
        "first_event_time": (None if manifest.first_event_time is None else manifest.first_event_time.isoformat()),
        "last_event_time": (None if manifest.last_event_time is None else manifest.last_event_time.isoformat()),
        "partition_count": len(manifest.ordered_partition_ids), "raw_source_dataset_id": manifest.raw_source_dataset_id,
        "semantic_digest": manifest.semantic_digest, "physical_digest": manifest.physical_digest,
    }


def generate_recovery_report(result: RecoveryReport) -> dict[str, object]:
    return {
        "dataset_key": result.dataset_key.to_json_dict(), "recovery_time": result.recovery_time.isoformat(), "manifest_advanced": result.manifest_advanced,
        "resulting_dataset_id": result.resulting_dataset_id, "pending_batch_ids": list(result.pending_batch_ids),
        "discarded_truncated_tail": result.discarded_truncated_tail, "notes": list(result.notes),
    }


def _summarize_validation_report(report: ValidationReport, *, dataset_key: DatasetKey) -> dict[str, object]:
    return {
        "dataset_key": dataset_key.to_json_dict(), "generated_at": report.generated_at, "critical_count": len(report.criticals),
        "warning_count": len(report.warnings), "total_issue_count": len(report.issues),
        "issues": [i.to_json_dict() for i in report.issues],
    }


def generate_reconciliation_report(*, report: ValidationReport, dataset_key: DatasetKey) -> dict[str, object]:
    return _summarize_validation_report(report, dataset_key=dataset_key)


def generate_verification_report(*, report: ValidationReport, dataset_key: DatasetKey) -> dict[str, object]:
    return _summarize_validation_report(report, dataset_key=dataset_key)


def generate_export_report(result: ExportResult) -> dict[str, object]:
    return {
        "dataset_key": result.dataset_key.to_json_dict(), "destination": str(result.destination), "row_count": result.row_count,
        "export_semantic_digest": result.export_semantic_digest,
    }
