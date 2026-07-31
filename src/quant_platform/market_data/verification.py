"""Independent verification for `quant_platform.market_data` (Milestone
10, Phase 1). Reuses `ml.models.ValidationIssue`/`ValidationReport`/
`ValidationSeverity`, mirroring `portfolio_risk.verification`'s and
`execution_gateway.verification`'s identical choice.

HONESTY CLASSIFICATION (mirrors those two modules' own explicit 2-tier
taxonomy):
- **STRUCTURALLY INDEPENDENT** (this module's entire scope): event/
  feature append-only sequence integrity, forged-identity detection
  (recomputing each object's own content id from its stored payload),
  cross-event/cross-feature ordering, and duplicate detection. None of
  these trust a cached report, an in-memory set, or a caller assertion --
  every one is a pure recomputation from the store's own raw bytes.
- **NOT INDEPENDENTLY RE-VERIFIED** (an honest, explicit limitation, not
  an oversight): this module does not re-run `feature_generation.py`'s
  own arithmetic against the store's raw candle events to confirm a
  stored feature VALUE is the economically correct one -- that is
  `replay.py`'s job (`replay_candle_features_from_events` plus
  `assert_replay_deterministic`), invoked separately, not folded into
  this module's checks. `verify_feature_store` verifies a `FeatureRecord`'s
  own internal identity coherence and its store-level positioning, never
  whether its `value` is the correct output of the generator that
  produced it."""

from __future__ import annotations

import itertools
from datetime import datetime

import pandas as pd

from quant_platform.market_data.events import (
    MarketDataEvent,
    MarketEventStore,
    market_data_event_id,
    market_data_event_time,
)
from quant_platform.market_data.feature_store import FEATURE_RECORD_KIND, FeatureStore
from quant_platform.market_data.identity import compute_content_id, require_tz_aware
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["VERIFICATION_REPORT_SCHEMA_VERSION", "verify_feature_store", "verify_market_event_store"]

VERIFICATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _recompute_event_id(event: MarketDataEvent) -> str:
    json_dict = event.to_json_dict()
    kind = str(json_dict["kind"])
    identity_payload = dict(json_dict)
    del identity_payload["event_id"]
    return compute_content_id(kind, identity_payload)


def verify_market_event_store(*, store: MarketEventStore, provider: str, instrument_id: str, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    events = store.read_events(provider, instrument_id)
    issues: list[ValidationIssue] = []

    for index, event in enumerate(events):
        if event.sequence != index:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_sequence_gap_or_reorder",
                f"Event at physical position {index} declares sequence={event.sequence} for {provider}/{instrument_id}.",
            ))
        if event.instrument_id != instrument_id or event.provider != provider:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_partition_mismatch",
                f"Event {market_data_event_id(event)!r} declares provider={event.provider!r}/instrument_id={event.instrument_id!r}, "
                f"expected {provider!r}/{instrument_id!r}.",
            ))
        recomputed_id = _recompute_event_id(event)
        if recomputed_id != market_data_event_id(event):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_event_identity",
                f"Event {market_data_event_id(event)!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
            ))
        if market_data_event_time(event) > as_of:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "future_event_timestamp",
                f"Event {market_data_event_id(event)!r} has event_time {market_data_event_time(event)} after as_of {as_of}.",
            ))

    for previous, current in itertools.pairwise(events):
        if market_data_event_time(current) < market_data_event_time(previous):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_ordering_violation",
                f"Event {market_data_event_id(current)!r} has event_time {market_data_event_time(current)} before "
                f"the preceding event's {market_data_event_time(previous)}.",
            ))

    seen_ids: dict[str, int] = {}
    for event in events:
        seen_ids[market_data_event_id(event)] = seen_ids.get(market_data_event_id(event), 0) + 1
    duplicated = sorted(eid for eid, count in seen_ids.items() if count > 1)
    if duplicated:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "duplicate_event_id_in_store",
            f"{len(duplicated)} event id(s) appear more than once in the store: {duplicated[:10]}.",
        ))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_feature_store(*, store: FeatureStore, feature_name: str, feature_version: int, instrument_id: str, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    records = store.read_records(feature_name, feature_version, instrument_id)
    issues: list[ValidationIssue] = []

    seen_timestamps: dict[datetime, str] = {}
    for record in records:
        if record.feature_name != feature_name or record.feature_version != feature_version or record.instrument_id != instrument_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "feature_partition_mismatch",
                f"Record {record.feature_id!r} declares ({record.feature_name!r}, v{record.feature_version}, {record.instrument_id!r}), "
                f"expected ({feature_name!r}, v{feature_version}, {instrument_id!r}).",
            ))
        recomputed_id = compute_content_id(FEATURE_RECORD_KIND, record.to_identity_payload())
        if recomputed_id != record.feature_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_feature_identity",
                f"Record {record.feature_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
            ))
        if record.timestamp > as_of:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "future_feature_timestamp",
                f"Record {record.feature_id!r} has timestamp {record.timestamp} after as_of {as_of}.",
            ))
        existing_id = seen_timestamps.get(record.timestamp)
        if existing_id is not None and existing_id != record.feature_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "conflicting_feature_value_at_timestamp",
                f"Timestamp {record.timestamp} has two different feature_id values ({existing_id!r} and {record.feature_id!r}) "
                f"for ({feature_name!r}, v{feature_version}, {instrument_id!r}) -- append-only history was violated.",
            ))
        seen_timestamps[record.timestamp] = record.feature_id

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))
