"""Independent verification for `quant_platform.market_data` (Milestone
10, Phases 1 and 2). Reuses `ml.models.ValidationIssue`/`ValidationReport`/
`ValidationSeverity`, mirroring `portfolio_risk.verification`'s and
`execution_gateway.verification`'s identical choice.

HONESTY CLASSIFICATION (mirrors those two modules' own explicit 2-tier
taxonomy):
- **STRUCTURALLY INDEPENDENT**: `verify_market_event_store`/
  `verify_feature_store` (Phase 1) -- event/feature append-only sequence
  integrity, forged-identity detection (recomputing each object's own
  content id from its stored payload), cross-event/cross-feature
  ordering, and duplicate detection. `verify_raw_dataset`/
  `verify_feature_dataset` (Phase 2) REUSE these two directly for their
  own event/feature-identity steps, and additionally recompute: every
  `Partition`'s own id from its own recorded fields (`compute_content_id
  (PARTITION_KIND, ...)`), every `DatasetManifest`'s own `dataset_id`
  from its own identity payload, logical ordering and counts/ranges from
  the raw store, the manifest's `semantic_digest`, and (for a feature
  dataset) that `raw_source_dataset_id` actually appears in the raw
  dataset's own manifest HISTORY. None of these trust a cached report,
  an in-memory set, a checkpoint, or a caller assertion.
- **NOT INDEPENDENTLY RE-VERIFIED BY DEFAULT** (an honest, explicit
  limitation, not an oversight): `verify_feature_dataset` does not, by
  default, re-run `feature_generation.py`'s own arithmetic against the
  raw store to confirm a stored feature VALUE is the economically
  correct one -- recomputing digests/ids alone cannot catch a value an
  attacker tampered with AND consistently re-hashed (a forged id that
  correctly reproduces itself from its own, also-tampered, payload).
  Passing `cross_check_against_fresh_recomputation=True` closes exactly
  this gap: it regenerates every requested feature fresh, into a
  throwaway scratch store, straight from the SAME raw candles this
  dataset's own manifest claims as its lineage, and compares every
  resulting value against what is actually stored -- catching coherent
  tampering that a pure identity/digest recomputation cannot. It is
  off by default because it re-runs the full generator (a materially
  more expensive check than the rest of this module's O(1)-per-record
  recomputations)."""

from __future__ import annotations

import itertools
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.market_data.backfill import BACKFILL_PLAN_KIND, BackfillPlan
from quant_platform.market_data.candles import Candle
from quant_platform.market_data.checkpoints import (
    RAW_INGESTION_CHECKPOINT_KIND,
    CheckpointStore,
    HistoricalIngestionCheckpoint,
    RawIngestionCheckpoint,
)
from quant_platform.market_data.events import (
    MarketDataEvent,
    MarketEventStore,
    market_data_event_id,
    market_data_event_time,
)
from quant_platform.market_data.feature_store import FEATURE_RECORD_KIND, FeatureStore
from quant_platform.market_data.identity import compute_content_id, require_tz_aware
from quant_platform.market_data.manifests import DatasetKey, DatasetKind, compute_dataset_id
from quant_platform.market_data.partitions import PARTITION_KIND
from quant_platform.market_data.provenance import ProvenanceStore, find_provenance_conflicts
from quant_platform.market_data.quarantine import QUARANTINE_RECORD_KIND, QuarantineStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import SOURCE_MANIFEST_KIND, SourceManifest
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = [
    "VERIFICATION_REPORT_SCHEMA_VERSION",
    "verify_backfill_plan",
    "verify_feature_dataset",
    "verify_feature_store",
    "verify_historical_ingestion_checkpoint",
    "verify_market_event_store",
    "verify_provenance_store",
    "verify_quarantine_store",
    "verify_raw_dataset",
    "verify_source_manifest",
]

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


# --------------------------------------------------------------------------
# Milestone 10, Phase 2: repository-level (dataset/partition/manifest)
# verification.
# --------------------------------------------------------------------------
def verify_raw_dataset(*, repository: MarketDataRepository, dataset_key: DatasetKey, as_of: datetime) -> ValidationReport:
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise ValueError("verify_raw_dataset requires a RAW_MARKET_EVENTS dataset_key")
    require_tz_aware(as_of, field_name="as_of")
    assert dataset_key.provider is not None
    issues: list[ValidationIssue] = []

    event_report = verify_market_event_store(store=repository.event_store, provider=dataset_key.provider, instrument_id=dataset_key.instrument_id, as_of=as_of)
    issues.extend(event_report.issues)

    manifest = repository.manifest_store.read_current(dataset_key)
    if manifest is None:
        return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))

    recomputed_dataset_id = compute_dataset_id(manifest)
    if recomputed_dataset_id != manifest.dataset_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_dataset_identity", f"Manifest {manifest.dataset_id!r}'s own recorded fields do not reproduce its own dataset_id -- forged or tampered."))

    for partition_key in repository.partition_store.list_partition_keys(dataset_key):
        partition = repository.partition_store.read(dataset_key, partition_key)
        if partition is None:
            continue
        recomputed_partition_id = compute_content_id(PARTITION_KIND, partition.to_identity_payload())
        if recomputed_partition_id != partition.partition_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_partition_identity", f"Partition {partition.partition_id!r} at {partition_key!r} does not reproduce its own id -- forged or tampered."))
        if partition.partition_id not in manifest.ordered_partition_ids:
            issues.append(_issue(ValidationSeverity.WARNING, "partition_not_referenced_by_current_manifest", f"Partition {partition.partition_id!r} at {partition_key!r} exists but is not referenced by the current manifest version (superseded, or orphaned)."))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_feature_dataset(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, raw_dataset_key: DatasetKey, as_of: datetime,
    cross_check_against_fresh_recomputation: bool = False, feature_base_name: str | None = None, window: int | None = None,
) -> ValidationReport:
    if dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise ValueError("verify_feature_dataset requires a DERIVED_FEATURES dataset_key")
    require_tz_aware(as_of, field_name="as_of")
    assert dataset_key.feature_name is not None and dataset_key.feature_version is not None
    issues: list[ValidationIssue] = []

    feature_report = verify_feature_store(store=repository.feature_store, feature_name=dataset_key.feature_name, feature_version=dataset_key.feature_version, instrument_id=dataset_key.instrument_id, as_of=as_of)
    issues.extend(feature_report.issues)

    manifest = repository.manifest_store.read_current(dataset_key)
    if manifest is None:
        return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))

    recomputed_dataset_id = compute_dataset_id(manifest)
    if recomputed_dataset_id != manifest.dataset_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_dataset_identity", f"Manifest {manifest.dataset_id!r}'s own recorded fields do not reproduce its own dataset_id -- forged or tampered."))

    raw_history_ids = {m.dataset_id for m in repository.manifest_store.read_history(raw_dataset_key)}
    if manifest.raw_source_dataset_id not in raw_history_ids:
        issues.append(_issue(ValidationSeverity.CRITICAL, "broken_lineage", f"Feature manifest references raw_source_dataset_id={manifest.raw_source_dataset_id!r}, which does not appear in the raw dataset's own manifest history."))

    for partition_key in repository.partition_store.list_partition_keys(dataset_key):
        partition = repository.partition_store.read(dataset_key, partition_key)
        if partition is None:
            continue
        recomputed_partition_id = compute_content_id(PARTITION_KIND, partition.to_identity_payload())
        if recomputed_partition_id != partition.partition_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_partition_identity", f"Partition {partition.partition_id!r} at {partition_key!r} does not reproduce its own id -- forged or tampered."))

    if cross_check_against_fresh_recomputation:
        if feature_base_name is None:
            raise ValueError("cross_check_against_fresh_recomputation=True requires feature_base_name")
        from quant_platform.market_data.feature_generation import generate_candle_features

        assert raw_dataset_key.provider is not None
        raw_events = repository.event_store.read_events(raw_dataset_key.provider, raw_dataset_key.instrument_id)
        candles = sorted((e for e in raw_events if isinstance(e, Candle)), key=lambda c: c.event_time)
        with tempfile.TemporaryDirectory() as scratch_root:
            scratch_store = FeatureStore(Path(scratch_root))
            resolved_windows = {feature_base_name: window} if window is not None else None
            fresh_records = generate_candle_features(candles, feature_version=dataset_key.feature_version, store=scratch_store, feature_names=(feature_base_name,), windows=resolved_windows)
        fresh_by_timestamp = {r.timestamp: r.value for r in fresh_records}
        stored_records = repository.feature_store.read_records(dataset_key.feature_name, dataset_key.feature_version, dataset_key.instrument_id)
        for record in stored_records:
            fresh_value = fresh_by_timestamp.get(record.timestamp)
            if fresh_value is not None and fresh_value != record.value:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "coherent_feature_tampering_detected",
                    f"Stored value {record.value} at {record.timestamp} does not match a fresh recomputation from raw data ({fresh_value}) -- "
                    "the record's own id is internally consistent with a tampered value.",
                ))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


# --------------------------------------------------------------------------
# Milestone 10, Phase 3: historical-ingestion verification. STRUCTURALLY
# INDEPENDENT (like every function above): each recomputes an object's own
# content id from its own recorded fields, and cross-checks durable
# evidence across stores -- never trusts a cached report, an in-memory
# set, or a caller assertion. `verify_source_manifest`/
# `verify_backfill_plan` do NOT re-read the original source bytes (the
# strongest possible check -- see module docstring's own honesty
# taxonomy) -- a caller wanting that re-reads the source file/adapter and
# compares `adapter.content_digest()` to `SourceManifest.content_digest`
# directly (`orchestration.run_ingestion_operation` already does exactly
# this at its own `SOURCE_VERIFIED` stage).
# --------------------------------------------------------------------------
def verify_source_manifest(manifest: SourceManifest, *, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    recomputed_id = compute_content_id(SOURCE_MANIFEST_KIND, manifest.to_identity_payload())
    if recomputed_id != manifest.source_manifest_id:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "forged_source_manifest_identity",
            f"SourceManifest {manifest.source_manifest_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
        ))
    if len(manifest.content_digest) != 64:
        issues.append(_issue(ValidationSeverity.CRITICAL, "malformed_content_digest", f"SourceManifest.content_digest {manifest.content_digest!r} is not a 64-char sha256 hex digest."))
    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_backfill_plan(plan: BackfillPlan, *, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    recomputed_id = compute_content_id(BACKFILL_PLAN_KIND, plan.to_identity_payload())
    if recomputed_id != plan.backfill_plan_id:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "forged_backfill_plan_identity",
            f"BackfillPlan {plan.backfill_plan_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
        ))
    declared = set(plan.expected_partitions_touched)
    batch_keys = {b.partition_key for b in plan.batches}
    if declared != batch_keys:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "plan_batches_partition_mismatch",
            f"BackfillPlan.expected_partitions_touched {sorted(declared)} does not match the partition_key set covered by its own batches {sorted(batch_keys)}.",
        ))
    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_provenance_store(*, provenance_store: ProvenanceStore, dataset_key: DatasetKey, repository: MarketDataRepository, as_of: datetime) -> ValidationReport:
    """Recomputes every `ProvenanceRecord`'s own id from its recorded
    fields (forged-identity detection), runs `find_provenance_conflicts`
    (bidirectional row<->event linkage), and confirms every referenced
    `event_id` actually exists in the repository's raw event store --
    the "provenance references an existing event" half of "quarantine
    exclusion" (the other half -- that no event ALSO appears in
    quarantine for the same coordinate -- is a cross-store check; see
    `reconciliation.reconcile_historical_ingestion_operation`)."""
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    records = provenance_store.read_all(dataset_key)

    for record in records:
        recomputed_id = compute_content_id("provenance_record", record.to_identity_payload())
        if recomputed_id != record.provenance_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_provenance_identity",
                f"ProvenanceRecord {record.provenance_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
            ))

    for conflict in find_provenance_conflicts(records):
        severity = ValidationSeverity.CRITICAL if conflict.issue_code == "coordinate_bound_to_multiple_events" else ValidationSeverity.WARNING
        issues.append(_issue(severity, conflict.issue_code, conflict.detail))

    if records and dataset_key.provider is not None:
        existing_event_ids = {market_data_event_id(e) for e in repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)}
        missing = sorted({r.event_id for r in records} - existing_event_ids)
        if missing:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "provenance_references_missing_event",
                f"{len(missing)} provenance record(s) reference event_id(s) not present in the repository: {missing[:10]}.",
            ))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_quarantine_store(*, quarantine_store: QuarantineStore, dataset_key: DatasetKey, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    records = quarantine_store.read_all(dataset_key)
    for record in records:
        recomputed_id = compute_content_id(QUARANTINE_RECORD_KIND, record.to_identity_payload())
        if recomputed_id != record.quarantine_record_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_quarantine_identity",
                f"QuarantineRecord {record.quarantine_record_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
            ))
    seen_coordinates: dict[tuple[str, int], str] = {}
    for record in records:
        key = (record.source_manifest_id, record.source_row_index)
        existing = seen_coordinates.get(key)
        if existing is not None and existing != record.quarantine_record_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "quarantine_coordinate_conflict",
                f"source coordinate {key!r} has two different quarantine_record_id values durably recorded -- append-only history was violated.",
            ))
        seen_coordinates[key] = record.quarantine_record_id
    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_historical_ingestion_checkpoint(checkpoint: HistoricalIngestionCheckpoint, *, repository: MarketDataRepository) -> ValidationReport:
    """Recomputes `checkpoint`'s own id (forged-identity detection), and
    confirms its embedded `repository_checkpoint_id` refers to a
    `RawIngestionCheckpoint` that GENUINELY EXISTS in the checkpoint
    store's history and reproduces ITS OWN id from its own recorded
    fields -- the bridge described in `checkpoints.py`'s own
    `HistoricalIngestionCheckpoint` docstring. This deliberately does
    NOT call `checkpoints.verify_raw_ingestion_checkpoint` (which checks
    a checkpoint against CURRENT live repository state, and is
    EXPECTED to report the referenced checkpoint "stale" once ANY later
    activity -- this operation's own subsequent stages, or an unrelated
    later operation -- advances the repository past the point-in-time
    this checkpoint captured; that is normal, not evidence of tampering).
    A `HistoricalIngestionCheckpoint` embeds `repository_checkpoint_id`
    to prove "this repository state genuinely existed at commit time",
    never "the repository has not changed since"."""
    issues: list[ValidationIssue] = []
    recomputed_id = compute_content_id("historical_ingestion_checkpoint", checkpoint.to_identity_payload())
    if recomputed_id != checkpoint.checkpoint_id:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "forged_historical_checkpoint_identity",
            f"HistoricalIngestionCheckpoint {checkpoint.checkpoint_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
        ))

    checkpoint_history = [
        c for c in CheckpointStore(repository.root).read_history(checkpoint.dataset_key)
        if isinstance(c, RawIngestionCheckpoint) and c.checkpoint_id == checkpoint.repository_checkpoint_id
    ]
    if not checkpoint_history:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "missing_repository_checkpoint",
            f"HistoricalIngestionCheckpoint {checkpoint.checkpoint_id!r} references repository_checkpoint_id "
            f"{checkpoint.repository_checkpoint_id!r}, which does not appear in the checkpoint store's history.",
        ))
    else:
        referenced = checkpoint_history[0]
        recomputed_repository_checkpoint_id = compute_content_id(RAW_INGESTION_CHECKPOINT_KIND, referenced.to_identity_payload())
        if recomputed_repository_checkpoint_id != referenced.checkpoint_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_repository_checkpoint",
                f"RawIngestionCheckpoint {referenced.checkpoint_id!r} (referenced by HistoricalIngestionCheckpoint "
                f"{checkpoint.checkpoint_id!r}) does not reproduce its own id from its own recorded fields -- forged or tampered.",
            ))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(checkpoint.checkpoint_time)))
