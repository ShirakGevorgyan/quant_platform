"""Deterministic reporting for `quant_platform.market_data` (Milestone
10, Phase 1) -- every section is recomputed FRESH from the store's own
raw entries each time (never a cached/stale derived value), mirroring
`portfolio_risk.reports.generate_portfolio_risk_session_report`'s
identical convention exactly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.market_data.events import MarketEventStore, market_data_event_id
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.verification import verify_feature_store, verify_market_event_store

__all__ = ["MarketDataReport", "generate_market_data_report"]


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
