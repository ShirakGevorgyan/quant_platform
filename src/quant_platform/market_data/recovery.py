"""Deterministic recovery for interrupted repository operations
(Milestone 10, Phase 2).

RECOVERY NEVER GUESSES. It reconstructs derived state (partitions,
manifests) EXCLUSIVELY from whatever the underlying append-only stores
(`MarketEventStore`/`FeatureStore`) durably, verifiably hold RIGHT NOW --
the same "always recompute fresh from current store state" functions
ingestion itself uses (`ingestion.rebuild_dataset_manifest_from_events`/
`feature_generation.rebuild_feature_dataset_manifest`), never a diff
against a prior manifest, and never a replay of an ingestion batch's
original (not durably retained) event list. A batch whose events were
never actually applied before a crash is reported as PENDING, never
fabricated -- the caller must resubmit that exact batch (idempotently
convergent either way, per `ingestion.py`).

TRUNCATED TRAILING RECORD -- THE ONE FORM OF "REPAIR" THIS MODULE
PERFORMS: a process killed mid-write of the LAST line of a `.jsonl` file
leaves a partial, unparseable JSON fragment as that file's final line,
with every EARLIER line still complete and valid.
`read_jsonl_tolerating_truncated_tail` parses every line strictly and,
ONLY if the FINAL line fails to parse, discards that one line and
reports it -- a failure on any NON-final line is genuine, unexplained
corruption and raises `RepositoryCorruptionError` rather than being
silently discarded."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.core.exceptions import RepositoryCorruptionError
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.candles import Candle
from quant_platform.market_data.checkpoints import CheckpointStore, FeatureGenerationCheckpoint
from quant_platform.market_data.events import (
    market_data_event_from_json_dict,
    market_data_event_id,
    market_data_event_time,
)
from quant_platform.market_data.feature_generation import (
    carry_window_size_for,
    generate_candle_features,
    rebuild_feature_dataset_manifest,
)
from quant_platform.market_data.feature_store import FeatureRecord
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.ingestion import (
    IngestionBatchStatus,
    IngestionBatchStore,
    rebuild_dataset_manifest_from_events,
    rebuild_touched_partitions,
)
from quant_platform.market_data.manifests import DatasetKey, DatasetKind, DatasetManifest, PartitioningSpec
from quant_platform.market_data.partitions import partition_key_for
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.persistence import parse_json_strict

__all__ = ["RecoveryReport", "read_jsonl_tolerating_truncated_tail", "recover_feature_dataset", "recover_raw_dataset"]


def _atomically_rewrite_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Rewrites `path` to contain EXACTLY `records` (one canonical JSON
    line each), via temp-file-then-rename -- the same atomicity guarantee
    `core.json.write_json_atomic` gives a single JSON file, applied here
    to a `.jsonl` stream. Used ONLY to physically remove an already-
    discarded truncated trailing line (never to drop or reorder a
    successfully parsed record) so that subsequent STRICT reads
    (`MarketEventStore.read_events`/`FeatureStore.read_records`) succeed
    again. Assumes no concurrent writer targets the same path during the
    recovery call -- a reasonable operational assumption for crash
    recovery (run before ordinary traffic resumes), documented here
    rather than left implicit."""
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_path.open("wb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_jsonl_tolerating_truncated_tail(path: Path) -> tuple[list[dict[str, object]], bool]:
    """Returns `(parsed_records, discarded_truncated_tail)`. Never used by
    normal read paths (`MarketEventStore.read_events` etc. remain strict
    -- a truncated trailing record must never be silently tolerated
    during ordinary operation, only during EXPLICIT recovery)."""
    if not path.is_file():
        return [], False
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [(i, line) for i, line in enumerate(lines) if line.strip()]
    if not non_empty:
        return [], False
    records: list[dict[str, object]] = []
    discarded = False
    last_index = non_empty[-1][0]
    for index, line in non_empty:
        try:
            raw = parse_json_strict(line)
        except ValueError as exc:
            if index == last_index:
                discarded = True
                continue
            raise RepositoryCorruptionError(f"Corrupted (non-trailing) record at line {index} of {path}: {exc}") from exc
        if not isinstance(raw, dict):
            if index == last_index:
                discarded = True
                continue
            raise RepositoryCorruptionError(f"Corrupted (non-trailing) record at line {index} of {path}: expected a JSON object")
        records.append(raw)
    return records, discarded


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    dataset_key: DatasetKey
    recovery_time: datetime
    manifest_advanced: bool
    resulting_dataset_id: str | None
    pending_batch_ids: tuple[str, ...]
    discarded_truncated_tail: bool
    notes: tuple[str, ...]


def _pending_batch_ids(batch_store: IngestionBatchStore, dataset_key: DatasetKey) -> tuple[str, ...]:
    latest_by_batch: dict[str, IngestionBatchStatus] = {}
    for record in batch_store.read_all(dataset_key):
        latest_by_batch[record.batch_id] = record.status
    return tuple(sorted(batch_id for batch_id, status in latest_by_batch.items() if status is IngestionBatchStatus.RESERVED))


def recover_raw_dataset(*, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, recovery_time: datetime) -> RecoveryReport:
    """Recovers a `RAW_MARKET_EVENTS` dataset: re-derives every partition
    and the manifest fresh from the event store's current (tolerant-read)
    content, and reports any batch left `RESERVED` without a matching
    `COMMITTED` entry as pending caller action."""
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise RepositoryCorruptionError("recover_raw_dataset requires a RAW_MARKET_EVENTS dataset_key")
    require_tz_aware(recovery_time, field_name="recovery_time")
    assert dataset_key.provider is not None
    notes: list[str] = []

    events_path = repository.event_store.events_path(dataset_key.provider, dataset_key.instrument_id)
    raw_records, discarded = read_jsonl_tolerating_truncated_tail(events_path)
    if discarded:
        notes.append(f"discarded 1 truncated trailing record at {events_path}")
        _atomically_rewrite_jsonl(events_path, raw_records)
    events = [market_data_event_from_json_dict(r) for r in raw_records]
    for index, event in enumerate(events):
        if event.sequence != index:
            raise RepositoryCorruptionError(f"Recovered event stream for {dataset_key!r} has a sequence gap/reorder at position {index}")

    all_members = [(market_data_event_id(e), market_data_event_time(e)) for e in events]
    touched_partition_keys = {partition_key_for(t, partitioning) for _, t in all_members}
    if touched_partition_keys:
        rebuild_touched_partitions(repository=repository, dataset_key=dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys)

    manifest_before = repository.manifest_store.read_current(dataset_key)
    manifest_after = rebuild_dataset_manifest_from_events(repository=repository, dataset_key=dataset_key, partitioning=partitioning, raw_source_dataset_id=None, creation_time=recovery_time)
    manifest_advanced = manifest_before is None or manifest_before.dataset_id != manifest_after.dataset_id

    batch_store = IngestionBatchStore(repository.root)
    pending = _pending_batch_ids(batch_store, dataset_key)
    if pending:
        notes.append(f"{len(pending)} batch(es) reserved but never committed -- caller must resubmit: {pending}")

    return RecoveryReport(
        dataset_key=dataset_key, recovery_time=recovery_time, manifest_advanced=manifest_advanced, resulting_dataset_id=manifest_after.dataset_id,
        pending_batch_ids=pending, discarded_truncated_tail=discarded, notes=tuple(notes),
    )


def _checkpoint_needs_completion(repository: MarketDataRepository, feature_dataset_key: DatasetKey, raw_manifest: DatasetManifest) -> bool:
    checkpoint = CheckpointStore(repository.root).read_current(feature_dataset_key)
    if not isinstance(checkpoint, FeatureGenerationCheckpoint):
        return True
    return checkpoint.raw_dataset_id != raw_manifest.dataset_id


def _complete_pending_feature_generation(
    *, repository: MarketDataRepository, raw_dataset_key: DatasetKey, feature_dataset_key: DatasetKey, feature_base_name: str,
    feature_version: int, window: int | None, partitioning: PartitioningSpec, last_processed: datetime | None, raw_manifest: DatasetManifest,
    recovery_time: datetime,
) -> DatasetManifest:
    assert raw_dataset_key.provider is not None
    assert feature_dataset_key.feature_name is not None and feature_dataset_key.feature_version is not None
    all_candles = sorted(
        (e for e in repository.event_store.read_events(raw_dataset_key.provider, raw_dataset_key.instrument_id) if isinstance(e, Candle)),
        key=lambda c: c.event_time,
    )
    carry_window_size = carry_window_size_for(feature_base_name, window=window)
    new_candles = all_candles if last_processed is None else [c for c in all_candles if c.event_time > last_processed]
    if last_processed is None or carry_window_size is None:
        working_set = all_candles
    else:
        cutoff_index = len(all_candles) - len(new_candles)
        context_start = max(0, cutoff_index - carry_window_size)
        working_set = all_candles[context_start:]

    resolved_windows = {feature_base_name: window} if window is not None else None
    only_persist = frozenset(c.event_time for c in new_candles)
    generate_candle_features(
        working_set, feature_version=feature_version, store=repository.feature_store, feature_names=(feature_base_name,), windows=resolved_windows,
        only_persist_timestamps=only_persist,
    )

    all_records = repository.feature_store.read_records(feature_dataset_key.feature_name, feature_dataset_key.feature_version, feature_dataset_key.instrument_id)
    touched = {partition_key_for(r.timestamp, partitioning) for r in all_records}
    if touched:
        rebuild_touched_partitions(
            repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning,
            all_members=[(r.feature_id, r.timestamp) for r in all_records], touched_partition_keys=touched,
        )
    return rebuild_feature_dataset_manifest(
        repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning, raw_source_dataset_id=raw_manifest.dataset_id,
        creation_time=recovery_time,
    )


def recover_feature_dataset(
    *, repository: MarketDataRepository, raw_dataset_key: DatasetKey, feature_dataset_key: DatasetKey, feature_base_name: str,
    feature_version: int, window: int | None, partitioning: PartitioningSpec, recovery_time: datetime,
) -> RecoveryReport:
    """Recovers a `DERIVED_FEATURES` dataset: re-derives every partition
    and the manifest fresh from the feature store's current (tolerant-
    read) content. If the raw dataset has genuinely new candles the last
    checkpoint never processed, this also completes that generation --
    the same idempotent, deterministic operation
    `generate_feature_dataset_incremental` performs, simply invoked here
    as part of recovery."""
    if feature_dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise RepositoryCorruptionError("recover_feature_dataset requires a DERIVED_FEATURES feature_dataset_key")
    require_tz_aware(recovery_time, field_name="recovery_time")
    assert feature_dataset_key.feature_name is not None and feature_dataset_key.feature_version is not None
    notes: list[str] = []

    records_path = repository.feature_store.records_path(feature_dataset_key.feature_name, feature_dataset_key.feature_version, feature_dataset_key.instrument_id)
    raw_records, discarded = read_jsonl_tolerating_truncated_tail(records_path)
    if discarded:
        notes.append(f"discarded 1 truncated trailing record at {records_path}")
        _atomically_rewrite_jsonl(records_path, raw_records)
    records = [FeatureRecord.from_json_dict(r) for r in raw_records]

    seen_timestamps: dict[datetime, str] = {}
    for record in records:
        existing = seen_timestamps.get(record.timestamp)
        if existing is not None and existing != record.feature_id:
            raise RepositoryCorruptionError(f"Recovered feature stream for {feature_dataset_key!r} has conflicting values at {record.timestamp}")
        seen_timestamps[record.timestamp] = record.feature_id

    all_members = [(r.feature_id, r.timestamp) for r in records]
    if all_members:
        touched_partition_keys = {partition_key_for(t, partitioning) for _, t in all_members}
        rebuild_touched_partitions(repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys)

    raw_manifest = repository.manifest_store.read_current(raw_dataset_key)
    manifest_before = repository.manifest_store.read_current(feature_dataset_key)

    if raw_manifest is None or raw_manifest.event_count == 0:
        return RecoveryReport(
            dataset_key=feature_dataset_key, recovery_time=recovery_time, manifest_advanced=False,
            resulting_dataset_id=(manifest_before.dataset_id if manifest_before is not None else None),
            pending_batch_ids=(), discarded_truncated_tail=discarded, notes=tuple(notes),
        )

    manifest_after = rebuild_feature_dataset_manifest(
        repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning, raw_source_dataset_id=raw_manifest.dataset_id,
        creation_time=recovery_time,
    )

    if _checkpoint_needs_completion(repository, feature_dataset_key, raw_manifest):
        last_processed = max((r.timestamp for r in records), default=None)
        manifest_after = _complete_pending_feature_generation(
            repository=repository, raw_dataset_key=raw_dataset_key, feature_dataset_key=feature_dataset_key, feature_base_name=feature_base_name,
            feature_version=feature_version, window=window, partitioning=partitioning, last_processed=last_processed, raw_manifest=raw_manifest,
            recovery_time=recovery_time,
        )
        notes.append("completed feature generation for raw candles the prior checkpoint had not yet processed")

    manifest_advanced = manifest_before is None or manifest_before.dataset_id != manifest_after.dataset_id
    return RecoveryReport(
        dataset_key=feature_dataset_key, recovery_time=recovery_time, manifest_advanced=manifest_advanced, resulting_dataset_id=manifest_after.dataset_id,
        pending_batch_ids=(), discarded_truncated_tail=discarded, notes=tuple(notes),
    )
