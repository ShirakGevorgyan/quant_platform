"""Deterministic reporting for the curated universe (Milestone 10,
Phase 4B) -- mirrors `collectors.reports`'s own convention exactly:
every function wraps an object this package already produces into a
stable, deterministic dict, never re-deriving new facts of its own.
`generate_quarantine_summary_report`/`generate_provenance_summary_report`
are RE-EXPORTED UNCHANGED from `collectors.reports` (already
`DatasetKey`-generic, so Phase 3/4A's own implementations serve curated
per-series scoping exactly as they are).

NO REPORT HERE MAY EVER CONTAIN THE FRED API KEY OR A COMPLETE REQUEST
URL WITH SECRETS: every dict is assembled purely from objects that are
themselves structurally secret-free (`CollectorRequestManifest`/
`CollectorResponseManifest`, `CuratedFredRegistry`, `CuratedMacroObservation`,
...); no function here ever accepts a raw credential as an argument."""

from __future__ import annotations

from quant_platform.market_data.collectors.curated.backfill import CuratedBackfillSpec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifest,
    ComponentDatasetManifest,
)
from quant_platform.market_data.collectors.curated.metadata import MetadataVerificationResult
from quant_platform.market_data.collectors.curated.orchestration import CuratedIngestionReport, SeriesOutcome
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry
from quant_platform.market_data.collectors.curated.update_plan import CuratedUpdatePlan
from quant_platform.market_data.collectors.reports import (
    generate_provenance_summary_report,
    generate_quarantine_summary_report,
)
from quant_platform.ml.models import ValidationReport

__all__ = [
    "generate_combined_universe_report",
    "generate_component_dataset_report",
    "generate_curated_backfill_plan_report",
    "generate_curated_ingestion_report",
    "generate_curated_reconciliation_report",
    "generate_curated_registry_report",
    "generate_curated_verification_report",
    "generate_metadata_compatibility_report",
    "generate_provenance_summary_report",
    "generate_quarantine_summary_report",
    "generate_series_collection_report",
    "generate_update_plan_report",
]


def generate_curated_registry_report(registry: CuratedFredRegistry) -> dict[str, object]:
    return {
        "registry_id": registry.registry_id, "registry_version": registry.registry_version, "series_count": len(registry.specs),
        "enabled_series_ids": list(registry.enabled_series_ids()),
        "series_by_tier": {
            tier: sorted(s.series_id for s in registry.specs if s.tier.value == tier)
            for tier in sorted({s.tier.value for s in registry.specs})
        },
    }


def generate_metadata_compatibility_report(result: MetadataVerificationResult) -> dict[str, object]:
    return result.to_json_dict()


def generate_curated_backfill_plan_report(spec: CuratedBackfillSpec) -> dict[str, object]:
    return spec.to_json_dict()


def generate_series_collection_report(outcome: SeriesOutcome) -> dict[str, object]:
    return {
        "series_id": outcome.series_id, "succeeded": outcome.succeeded, "failure_reason": outcome.failure_reason,
        "parsed_row_count": outcome.parsed_row_count, "valid_row_count": outcome.valid_row_count, "quarantined_row_count": outcome.quarantined_row_count,
        "missing_count": outcome.missing_count, "skipped_missing_count": outcome.skipped_missing_count,
        "committed_observation_count": outcome.committed_observation_count, "component_manifest_id": outcome.component_manifest_id,
    }


def generate_curated_ingestion_report(report: CuratedIngestionReport) -> dict[str, object]:
    return report.to_json_dict()


def generate_component_dataset_report(manifest: ComponentDatasetManifest) -> dict[str, object]:
    return manifest.to_json_dict()


def generate_combined_universe_report(manifest: CombinedUniverseManifest) -> dict[str, object]:
    return manifest.to_json_dict()


def generate_update_plan_report(plan: CuratedUpdatePlan) -> dict[str, object]:
    result = plan.to_json_dict()
    result["series_requiring_update"] = list(plan.series_requiring_update())
    result["is_exact_no_op"] = plan.is_exact_no_op()
    return result


def _summarize_validation_report(report: ValidationReport, *, scope: str) -> dict[str, object]:
    return {
        "scope": scope, "generated_at": report.generated_at, "critical_count": len(report.criticals), "warning_count": len(report.warnings),
        "total_issue_count": len(report.issues), "issues": [i.to_json_dict() for i in report.issues],
    }


def generate_curated_reconciliation_report(*, report: ValidationReport, target_dataset_namespace: str) -> dict[str, object]:
    return _summarize_validation_report(report, scope=target_dataset_namespace)


def generate_curated_verification_report(*, report: ValidationReport, target_dataset_namespace: str) -> dict[str, object]:
    return _summarize_validation_report(report, scope=target_dataset_namespace)
