"""Structured reconciliation across manifest, partitions, checkpoints,
and raw/feature data (Milestone 10, Phase 2). Reuses `ml.models.
ValidationIssue`/`ValidationReport`/`ValidationSeverity`, exactly like
`quality.py`/`verification.py` already do.

Ordinary inconsistencies (a missing/orphan partition, a wrong digest, a
stale checkpoint, a broken lineage reference) become structured CRITICAL
issues on the returned report -- this module never raises for them.
It raises `MarketDataReconciliationError` only when reconciliation itself
cannot proceed structurally (a referenced store cannot even be read),
never for a finding reconciliation was specifically designed to
surface."""

from __future__ import annotations

from datetime import datetime

from quant_platform.core.exceptions import (
    CheckpointError,
    MarketDataPersistenceError,
    MarketDataReconciliationError,
    StaleCheckpointError,
)
from quant_platform.market_data.checkpoints import (
    CheckpointStore,
    FeatureGenerationCheckpoint,
    RawIngestionCheckpoint,
    verify_feature_generation_checkpoint,
    verify_raw_ingestion_checkpoint,
)
from quant_platform.market_data.events import market_data_event_id, market_data_event_time
from quant_platform.market_data.ingestion import physical_digest_for_ids, semantic_digest_for_raw_events
from quant_platform.market_data.manifests import DatasetKey, DatasetKind, PartitioningSpec
from quant_platform.market_data.partitions import build_partition, partition_key_for
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity

__all__ = [
    "RECONCILIATION_REPORT_SCHEMA_VERSION",
    "reconcile_feature_dataset",
    "reconcile_historical_ingestion_operation",
    "reconcile_raw_dataset",
]

RECONCILIATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def reconcile_raw_dataset(*, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, generated_at: str) -> ValidationReport:
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise MarketDataReconciliationError("reconcile_raw_dataset requires a RAW_MARKET_EVENTS dataset_key")
    assert dataset_key.provider is not None
    issues: list[ValidationIssue] = []

    try:
        events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    except MarketDataPersistenceError as exc:
        raise MarketDataReconciliationError(f"could not read raw event store for {dataset_key!r}: {exc}") from exc
    manifest = repository.manifest_store.read_current(dataset_key)

    expected_partition_keys: dict[str, list[tuple[str, datetime]]] = {}
    member_partition_keys: dict[str, set[str]] = {}
    for event in events:
        key = partition_key_for(market_data_event_time(event), partitioning)
        expected_partition_keys.setdefault(key, []).append((market_data_event_id(event), market_data_event_time(event)))
        member_partition_keys.setdefault(market_data_event_id(event), set()).add(key)

    for member_id, keys in member_partition_keys.items():
        if len(keys) > 1:
            issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_event_across_partitions", f"Event {member_id!r} falls into more than one partition key: {sorted(keys)}."))

    for partition_key, members in sorted(expected_partition_keys.items()):
        stored = repository.partition_store.read(dataset_key, partition_key)
        if stored is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_partition", f"No partition file exists for partition_key {partition_key!r}, which {len(members)} event(s) currently belong to."))
            continue
        fresh = build_partition(dataset_key=dataset_key, partition_key=partition_key, spec=partitioning, members=members)
        if fresh.content_digest != stored.content_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_partition_digest", f"Partition {partition_key!r}'s stored content_digest does not match a fresh recompute from current raw events."))
        if set(stored.ordered_member_ids) != {m for m, _ in members}:
            issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_partition_membership", f"Partition {partition_key!r}'s stored membership does not match current raw events."))

    physical_keys = set(repository.partition_store.list_partition_keys(dataset_key))
    orphans = sorted(physical_keys - set(expected_partition_keys.keys()))
    for orphan_key in orphans:
        issues.append(_issue(ValidationSeverity.CRITICAL, "orphan_partition", f"Partition file exists for partition_key {orphan_key!r}, which no current raw event belongs to."))

    if manifest is None:
        if events:
            issues.append(_issue(ValidationSeverity.CRITICAL, "manifest_missing", f"No manifest exists for {dataset_key!r}, but {len(events)} raw event(s) are durably stored."))
    else:
        if manifest.event_count != len(events):
            issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_event_count", f"Manifest declares event_count={manifest.event_count}, actual raw store has {len(events)}."))
        if events:
            actual_first = min(market_data_event_time(e) for e in events)
            actual_last = max(market_data_event_time(e) for e in events)
            if manifest.first_event_time != actual_first or manifest.last_event_time != actual_last:
                issues.append(_issue(ValidationSeverity.CRITICAL, "manifest_range_mismatch", f"Manifest declares [{manifest.first_event_time}, {manifest.last_event_time}], actual raw store spans [{actual_first}, {actual_last}]."))
            expected_semantic_digest = semantic_digest_for_raw_events(events)
            if manifest.semantic_digest != expected_semantic_digest:
                issues.append(_issue(ValidationSeverity.CRITICAL, "semantic_digest_mismatch", "Manifest semantic_digest does not match a fresh recompute from current raw events."))
            expected_physical_digest = physical_digest_for_ids(tuple(sorted(market_data_event_id(e) for e in events)))
            if manifest.physical_digest != expected_physical_digest:
                issues.append(_issue(ValidationSeverity.CRITICAL, "physical_digest_mismatch", "Manifest physical_digest does not match a fresh recompute from current raw events."))

        checkpoint = CheckpointStore(repository.root).read_current(dataset_key)
        if isinstance(checkpoint, RawIngestionCheckpoint):
            try:
                verify_raw_ingestion_checkpoint(checkpoint, repository=repository)
            except StaleCheckpointError as exc:
                issues.append(_issue(ValidationSeverity.CRITICAL, "stale_checkpoint", str(exc)))
            except CheckpointError as exc:
                issues.append(_issue(ValidationSeverity.CRITICAL, "forged_checkpoint", str(exc)))

    return ValidationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=generated_at)


def reconcile_feature_dataset(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, raw_dataset_key: DatasetKey, partitioning: PartitioningSpec, generated_at: str,
) -> ValidationReport:
    if dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise MarketDataReconciliationError("reconcile_feature_dataset requires a DERIVED_FEATURES dataset_key")
    assert dataset_key.feature_name is not None and dataset_key.feature_version is not None
    issues: list[ValidationIssue] = []

    try:
        records = repository.feature_store.read_records(dataset_key.feature_name, dataset_key.feature_version, dataset_key.instrument_id)
    except MarketDataPersistenceError as exc:
        raise MarketDataReconciliationError(f"could not read feature store for {dataset_key!r}: {exc}") from exc
    manifest = repository.manifest_store.read_current(dataset_key)

    expected_partition_keys: dict[str, list[tuple[str, datetime]]] = {}
    seen_timestamps: dict[datetime, str] = {}
    for record in records:
        key = partition_key_for(record.timestamp, partitioning)
        expected_partition_keys.setdefault(key, []).append((record.feature_id, record.timestamp))
        existing = seen_timestamps.get(record.timestamp)
        if existing is not None and existing != record.feature_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "feature_coordinate_conflict", f"Timestamp {record.timestamp} has two different feature_id values."))
        elif existing is not None:
            issues.append(_issue(ValidationSeverity.WARNING, "feature_coordinate_duplicate", f"Timestamp {record.timestamp} has a byte-identical duplicate feature record."))
        seen_timestamps[record.timestamp] = record.feature_id

    for partition_key, members in sorted(expected_partition_keys.items()):
        stored = repository.partition_store.read(dataset_key, partition_key)
        if stored is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_partition", f"No partition file exists for partition_key {partition_key!r}."))
            continue
        fresh = build_partition(dataset_key=dataset_key, partition_key=partition_key, spec=partitioning, members=members)
        if fresh.content_digest != stored.content_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_partition_digest", f"Partition {partition_key!r}'s stored content_digest does not match a fresh recompute from current feature records."))

    physical_keys = set(repository.partition_store.list_partition_keys(dataset_key))
    orphans = sorted(physical_keys - set(expected_partition_keys.keys()))
    for orphan_key in orphans:
        issues.append(_issue(ValidationSeverity.CRITICAL, "orphan_partition", f"Partition file exists for partition_key {orphan_key!r}, which no current feature record belongs to."))

    if manifest is None:
        if records:
            issues.append(_issue(ValidationSeverity.CRITICAL, "manifest_missing", f"No manifest exists for {dataset_key!r}, but {len(records)} feature record(s) are durably stored."))
    else:
        if manifest.event_count != len(records):
            issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_event_count", f"Manifest declares event_count={manifest.event_count}, actual feature store has {len(records)}."))
        raw_history_ids = {m.dataset_id for m in repository.manifest_store.read_history(raw_dataset_key)}
        if manifest.raw_source_dataset_id not in raw_history_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "broken_lineage", f"Feature manifest references raw_source_dataset_id={manifest.raw_source_dataset_id!r}, which no longer appears in the raw dataset's own manifest history."))
        else:
            raw_current = repository.manifest_store.read_current(raw_dataset_key)
            if raw_current is not None and manifest.raw_source_dataset_id != raw_current.dataset_id:
                issues.append(_issue(ValidationSeverity.WARNING, "feature_dataset_behind_raw_version", "Feature dataset's raw_source_dataset_id is not the raw dataset's current version -- incremental generation has not yet caught up."))

        checkpoint = CheckpointStore(repository.root).read_current(dataset_key)
        if isinstance(checkpoint, FeatureGenerationCheckpoint):
            try:
                verify_feature_generation_checkpoint(checkpoint, repository=repository)
            except StaleCheckpointError as exc:
                issues.append(_issue(ValidationSeverity.CRITICAL, "stale_checkpoint", str(exc)))
            except CheckpointError as exc:
                issues.append(_issue(ValidationSeverity.CRITICAL, "forged_checkpoint", str(exc)))

    return ValidationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=generated_at)


# --------------------------------------------------------------------------
# Milestone 10, Phase 3: historical-ingestion reconciliation. Spans FOUR
# stores one operation touched (`orchestration.OperationStore`,
# `quarantine.QuarantineStore`, `provenance.ProvenanceStore`,
# `checkpoints.CheckpointStore`) plus the raw dataset manifest history --
# exactly the cross-store integrity `verification.py`'s Phase 3 functions
# (each scoped to ONE store) deliberately do not attempt alone.
# --------------------------------------------------------------------------
def reconcile_historical_ingestion_operation(*, repository: MarketDataRepository, dataset_key: DatasetKey, operation_id: str, generated_at: str) -> ValidationReport:
    from quant_platform.market_data.checkpoints import HistoricalIngestionCheckpoint
    from quant_platform.market_data.orchestration import IngestionStage, OperationStore
    from quant_platform.market_data.provenance import ProvenanceStore
    from quant_platform.market_data.quarantine import QuarantineStore
    from quant_platform.market_data.verification import verify_historical_ingestion_checkpoint

    issues: list[ValidationIssue] = []
    history = [r for r in OperationStore(repository.root).read_all(dataset_key) if r.operation_id == operation_id]
    if not history:
        issues.append(_issue(ValidationSeverity.CRITICAL, "operation_not_found", f"No operation ledger entry exists for operation_id {operation_id!r} under {dataset_key!r}."))
        return ValidationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=generated_at)

    latest = history[-1]
    if latest.stage is not IngestionStage.COMPLETED:
        issues.append(_issue(ValidationSeverity.WARNING, "operation_not_completed", f"operation_id {operation_id!r} is at stage {latest.stage.value!r}, not COMPLETED."))

    quarantine_records = [q for q in QuarantineStore(repository.root).read_all(dataset_key) if q.ingestion_batch_id == operation_id]
    provenance_records = [p for p in ProvenanceStore(repository.root).read_all(dataset_key) if p.ingestion_batch_id == operation_id]

    quarantined_coords = {(q.source_manifest_id, q.source_row_index) for q in quarantine_records}
    provenance_coords = {(p.source_manifest_id, p.source_row_index) for p in provenance_records}
    overlap = quarantined_coords & provenance_coords
    if overlap:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "row_both_quarantined_and_ingested",
            f"{len(overlap)} source row coordinate(s) appear in BOTH quarantine and provenance for operation_id {operation_id!r}: {sorted(overlap)[:10]}.",
        ))

    rows_parsed_record = next((r for r in history if r.stage is IngestionStage.ROWS_PARSED), None)
    rows_validated_record = next((r for r in history if r.stage is IngestionStage.ROWS_VALIDATED), None)
    source_verified_record = next((r for r in history if r.stage is IngestionStage.SOURCE_VERIFIED), None)
    if rows_parsed_record is not None and rows_validated_record is not None:
        parsed_count = int(str(rows_parsed_record.stage_evidence["parsed_row_count"]))
        valid_count = int(str(rows_validated_record.stage_evidence["valid_row_count"]))
        quarantined_count = int(str(rows_validated_record.stage_evidence["quarantined_row_count"]))
        if valid_count + quarantined_count != parsed_count:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "row_count_mismatch",
                f"operation_id {operation_id!r}: valid_row_count({valid_count}) + quarantined_row_count({quarantined_count}) != parsed_row_count({parsed_count}).",
            ))
        if len(provenance_records) != valid_count:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "provenance_count_mismatch",
                f"operation_id {operation_id!r}: {len(provenance_records)} provenance record(s) durably exist, expected valid_row_count={valid_count}.",
            ))

        # Coordinate-level, NOT batch-id-filtered: `QuarantineStore`
        # idempotently absorbs a later operation's rediscovery of a row
        # an EARLIER operation already quarantined for the identical
        # reason (see `quarantine.QuarantineRecord.to_identity_payload`'s
        # own docstring) -- the durably stored record's `ingestion_batch_id`
        # then still names whichever operation wrote it FIRST, so
        # completeness must be checked by COORDINATE, never by re-filtering
        # on this operation's own batch id.
        quarantined_row_indices_raw = rows_validated_record.stage_evidence.get("quarantined_row_indices")
        if quarantined_row_indices_raw is not None and source_verified_record is not None:
            assert isinstance(quarantined_row_indices_raw, list)
            expected_indices = {int(str(i)) for i in quarantined_row_indices_raw}
            source_manifest_id = str(source_verified_record.stage_evidence["source_manifest_id"])
            actual_indices = {
                q.source_row_index for q in QuarantineStore(repository.root).read_all(dataset_key) if q.source_manifest_id == source_manifest_id
            }
            missing_indices = sorted(expected_indices - actual_indices)
            if missing_indices:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "quarantine_count_mismatch",
                    f"operation_id {operation_id!r}: row(s) {missing_indices} were quarantined by this operation's own row processing "
                    "but have no corresponding durable QuarantineRecord for this source_manifest_id.",
                ))

    completed_record = next((r for r in history if r.stage is IngestionStage.COMPLETED), None)
    if completed_record is not None:
        resulting_dataset_id = str(completed_record.stage_evidence["resulting_dataset_id"])
        manifest_history_ids = {m.dataset_id for m in repository.manifest_store.read_history(dataset_key)}
        if resulting_dataset_id not in manifest_history_ids:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "resulting_dataset_id_not_in_manifest_history",
                f"operation_id {operation_id!r} claims resulting_dataset_id={resulting_dataset_id!r}, which does not appear in the dataset's own manifest history.",
            ))

    historical_checkpoints = [
        c for c in CheckpointStore(repository.root).read_history(dataset_key)
        if isinstance(c, HistoricalIngestionCheckpoint) and c.operation_id == operation_id
    ]
    if not historical_checkpoints:
        if latest.stage is IngestionStage.COMPLETED:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "missing_historical_checkpoint",
                f"operation_id {operation_id!r} is COMPLETED but no HistoricalIngestionCheckpoint exists for it.",
            ))
    else:
        checkpoint_report = verify_historical_ingestion_checkpoint(historical_checkpoints[-1], repository=repository)
        issues.extend(checkpoint_report.issues)

    return ValidationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=generated_at)
