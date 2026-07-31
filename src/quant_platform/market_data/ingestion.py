"""Incremental raw-event ingestion (Milestone 10, Phase 2).

THREE DISTINCT ORDERINGS -- NEVER ASSUMED IDENTICAL (the specification's
own explicit requirement):

1. **Provider/source sequence**: whatever ordering a raw provider itself
   claims (e.g. a vendor's own tick sequence number) -- carried, if
   present at all, in an event's own `source_event_id`; this module never
   interprets or relies on it.
2. **Repository append sequence**: `MarketDataEvent.sequence` (Phase 1),
   assigned by the CALLER via `next_sequence_for` before constructing an
   event (never minted by this module -- `sequence` is baked into an
   event's own content identity, so this module cannot assign it after
   the fact without changing the event's own id). Events are appended to
   `MarketEventStore` in EXACTLY the order the caller submits them --
   this is ARRIVAL order, not event-time order.
3. **Event-time ordering**: `market_data_event_time(event)`. Partition
   membership (and therefore dataset semantic identity) is ordered by
   THIS, via `(member_time, member_id)` -- see `partitions.py`'s module
   docstring. A batch may freely arrive out of event-time order (a
   classic late-historical-backfill), and even out of provider-sequence
   order, without corrupting anything: the repository append sequence
   simply records arrival order as its own independent fact, while
   partitions/manifests always reflect the canonical event-time view.

LATE-ARRIVING HISTORICAL EVENTS -- THE CHOSEN, EXPLICIT MODEL: a
late-arriving event (whose `event_time` falls into an ALREADY-partitioned
window) is appended to the arrival-ordered raw store like any other event
(repository sequence keeps growing), and the ONE partition its
`event_time` belongs to is REBUILT from its current complete membership
(see `partitions.py`), producing a NEW `partition_id` -- which in turn
produces a NEW dataset manifest VERSION (new `dataset_id`) referencing
the updated partition. The event is never rejected, and no existing
manifest VERSION is ever mutated (each manifest version remains
immutable and independently retrievable via `DatasetManifestStore`'s
full history) -- only a NEW version is appended on top. This is
"append to an arrival-ordered raw store while changing canonical
event-time views," the second of the three models the specification
names, chosen because it is the only one of the three that neither loses
information (rejection) nor requires an expensive full-dataset rebuild
for a single late event (a brand-new dataset version for the whole
history)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    IngestionConflictError,
    IngestionError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.events import MarketDataEvent, market_data_event_id, market_data_event_time
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    DatasetManifest,
    PartitioningSpec,
    create_dataset_manifest,
)
from quant_platform.market_data.partitions import build_partition, partition_key_for
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import parse_json_strict

__all__ = [
    "INGESTION_BATCH_CONTENT_KIND",
    "IngestionBatchRecord",
    "IngestionBatchStatus",
    "IngestionBatchStore",
    "IngestionResult",
    "compute_batch_content_digest",
    "ingest_raw_events",
    "next_sequence_for",
    "physical_digest_for_ids",
    "rebuild_dataset_manifest_from_events",
    "rebuild_touched_partitions",
    "semantic_digest_for_raw_events",
]

INGESTION_BATCH_CONTENT_KIND = "ingestion_batch_content"


def next_sequence_for(repository: MarketDataRepository, dataset_key: DatasetKey) -> int:
    """The next REPOSITORY APPEND sequence a caller must use when
    constructing a new raw event for `dataset_key` -- callers determine
    this BEFORE construction, since `sequence` participates in an
    event's own content identity (Phase 1) and this module never mints
    one after the fact."""
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise IngestionError("next_sequence_for is only meaningful for RAW_MARKET_EVENTS dataset keys")
    assert dataset_key.provider is not None
    return repository.event_store.next_sequence(dataset_key.provider, dataset_key.instrument_id)


# --------------------------------------------------------------------------
# Batch idempotency ledger.
# --------------------------------------------------------------------------
class IngestionBatchStatus(Enum):
    RESERVED = "reserved"
    COMMITTED = "committed"


@dataclass(frozen=True, slots=True)
class IngestionBatchRecord:
    batch_id: str
    dataset_key: DatasetKey
    status: IngestionBatchStatus
    content_digest: str
    ingestion_time: datetime
    resulting_dataset_id: str | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id, "dataset_key": self.dataset_key.to_json_dict(), "status": self.status.value,
            "content_digest": self.content_digest, "ingestion_time": serialize_timestamp(self.ingestion_time, field_name="ingestion_time"),
            "resulting_dataset_id": self.resulting_dataset_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> IngestionBatchRecord:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            batch_id=str(raw["batch_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            status=IngestionBatchStatus(raw["status"]), content_digest=str(raw["content_digest"]),
            ingestion_time=deserialize_timestamp(raw["ingestion_time"], field_name="ingestion_time"),
            resulting_dataset_id=(None if raw.get("resulting_dataset_id") is None else str(raw["resulting_dataset_id"])),
        )


def compute_batch_content_digest(*, dataset_key: DatasetKey, ingestion_time: datetime, ordered_event_ids: tuple[str, ...]) -> str:
    return compute_content_id(
        INGESTION_BATCH_CONTENT_KIND,
        {"dataset_key": dataset_key.to_json_dict(), "ingestion_time": serialize_timestamp(ingestion_time, field_name="ingestion_time"), "ordered_event_ids": list(ordered_event_ids)},
    )


@contextmanager
def _batch_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire ingestion batch lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Ingestion batch lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class IngestionBatchStore:
    """Append-only ledger: `{storage_root}/repository/batches/
    {dataset_key_path}/batches.jsonl`. For one `batch_id`, the LATEST
    entry is current status; every entry for that `batch_id` must share
    the same `content_digest` (enforced by `reserve`) -- a durably
    recorded batch_id is permanently bound to one specific content,
    exactly like `client_order_id` idempotency elsewhere in this
    repository."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "batches" / Path(*dataset_key.storage_path_parts())

    def _batches_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "batches.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".batches.lock"

    def read_all(self, dataset_key: DatasetKey) -> list[IngestionBatchRecord]:
        path = self._batches_path(dataset_key)
        if not path.is_file():
            return []
        records: list[IngestionBatchRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted ingestion batch line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted ingestion batch line for dataset {dataset_key!r}: expected a JSON object")
            records.append(IngestionBatchRecord.from_json_dict(raw))
        return records

    def read_latest(self, dataset_key: DatasetKey, batch_id: str) -> IngestionBatchRecord | None:
        matching = [r for r in self.read_all(dataset_key) if r.batch_id == batch_id]
        return matching[-1] if matching else None

    def _append(self, dataset_key: DatasetKey, record: IngestionBatchRecord) -> None:
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        path = self._batches_path(dataset_key)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self, *, dataset_key: DatasetKey, batch_id: str, content_digest: str, ingestion_time: datetime) -> IngestionBatchRecord:
        """Idempotent: an existing RESERVED/COMMITTED record for
        `batch_id` with the SAME `content_digest` is returned unchanged
        (no new ledger entry). A DIFFERENT `content_digest` under the
        same `batch_id` raises `IngestionConflictError` -- fails closed,
        never silently processed as if it were new data."""
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _batch_store_lock(lock_path):
            existing = self.read_latest(dataset_key, batch_id)
            if existing is not None:
                if existing.content_digest != content_digest:
                    raise IngestionConflictError(
                        f"batch_id {batch_id!r} is already bound to content_digest {existing.content_digest!r}; "
                        f"a conflicting content_digest {content_digest!r} was submitted"
                    )
                return existing
            record = IngestionBatchRecord(
                batch_id=batch_id, dataset_key=dataset_key, status=IngestionBatchStatus.RESERVED, content_digest=content_digest,
                ingestion_time=ingestion_time, resulting_dataset_id=None,
            )
            self._append(dataset_key, record)
            return record

    def commit(self, *, dataset_key: DatasetKey, batch_id: str, resulting_dataset_id: str) -> IngestionBatchRecord:
        lock_path = self._lock_path(dataset_key)
        with _batch_store_lock(lock_path):
            existing = self.read_latest(dataset_key, batch_id)
            if existing is None:
                raise IngestionError(f"cannot commit unreserved batch_id {batch_id!r}")
            if existing.status is IngestionBatchStatus.COMMITTED:
                return existing  # idempotent no-op
            record = IngestionBatchRecord(
                batch_id=batch_id, dataset_key=dataset_key, status=IngestionBatchStatus.COMMITTED, content_digest=existing.content_digest,
                ingestion_time=existing.ingestion_time, resulting_dataset_id=resulting_dataset_id,
            )
            self._append(dataset_key, record)
            return record


# --------------------------------------------------------------------------
# Manifest/partition rebuild -- shared by raw ingestion (below) and
# incremental feature generation (`feature_generation.
# generate_candle_features_incremental`), since both reduce to the same
# "rebuild touched partitions, then recompute the manifest fresh from
# current store state" shape.
# --------------------------------------------------------------------------
def rebuild_touched_partitions(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, all_members: list[tuple[str, datetime]],
    touched_partition_keys: set[str],
) -> None:
    """`all_members` is the CURRENT COMPLETE membership of the dataset
    (every `(member_id, member_time)` pair the underlying store holds);
    only partitions in `touched_partition_keys` are rebuilt and written
    -- an untouched partition's physical file, and therefore its
    `partition_id`, never changes."""
    if not touched_partition_keys:
        return
    grouped: dict[str, list[tuple[str, datetime]]] = {}
    for member_id, member_time in all_members:
        key = partition_key_for(member_time, partitioning)
        if key in touched_partition_keys:
            grouped.setdefault(key, []).append((member_id, member_time))
    for partition_key in touched_partition_keys:
        members = grouped.get(partition_key, [])
        if not members:
            continue  # nothing currently falls in this window (should not happen if touched correctly, but never fabricate an empty partition)
        partition = build_partition(dataset_key=dataset_key, partition_key=partition_key, spec=partitioning, members=members)
        repository.partition_store.write(partition)


def semantic_digest_for_raw_events(events: list[MarketDataEvent]) -> str:
    ordered = sorted(events, key=lambda e: (market_data_event_time(e), market_data_event_id(e)))
    canonical = [{k: v for k, v in e.to_json_dict().items() if k != "event_id"} for e in ordered]
    return compute_content_id("raw_dataset_semantic_digest", {"events": canonical})


def physical_digest_for_ids(ordered_ids: tuple[str, ...]) -> str:
    return compute_content_id("dataset_physical_digest", {"member_ids": list(ordered_ids)})


def rebuild_dataset_manifest_from_events(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, raw_source_dataset_id: str | None,
    creation_time: datetime,
) -> DatasetManifest:
    """Recomputes a `RAW_MARKET_EVENTS` manifest FRESH from
    `repository.event_store`'s and `repository.partition_store`'s
    current durable state -- never from a diff against the prior
    manifest. Safe to call after ingestion OR after recovery: both
    reduce to "make the manifest agree with what the stores actually
    durably hold right now." Appends the new manifest version (idempotent
    if content is unchanged) and returns it."""
    assert dataset_key.provider is not None
    all_events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    if not all_events:
        manifest = create_dataset_manifest(
            dataset_key=dataset_key, schema_version=1, timeframe=None, partitioning=partitioning, first_event_time=None,
            last_event_time=None, event_count=0, ordered_partition_ids=(), raw_source_dataset_id=raw_source_dataset_id,
            semantic_digest=compute_content_id("raw_dataset_semantic_digest", {"events": []}), physical_digest=physical_digest_for_ids(()),
            creation_time=creation_time,
        )
        return repository.manifest_store.append(dataset_key, manifest)

    first_event_time = min(market_data_event_time(e) for e in all_events)
    last_event_time = max(market_data_event_time(e) for e in all_events)
    partition_keys = repository.partition_store.list_partition_keys(dataset_key)
    ordered_partitions = []
    for partition_key in partition_keys:
        partition = repository.partition_store.read(dataset_key, partition_key)
        if partition is None:
            raise IngestionError(f"listed partition_key {partition_key!r} has no readable partition file")
        ordered_partitions.append(partition)
    ordered_partition_ids = tuple(p.partition_id for p in ordered_partitions)
    semantic_digest = semantic_digest_for_raw_events(all_events)
    physical_digest = physical_digest_for_ids(tuple(sorted(market_data_event_id(e) for e in all_events)))
    manifest = create_dataset_manifest(
        dataset_key=dataset_key, schema_version=1, timeframe=None, partitioning=partitioning, first_event_time=first_event_time,
        last_event_time=last_event_time, event_count=len(all_events), ordered_partition_ids=ordered_partition_ids,
        raw_source_dataset_id=raw_source_dataset_id, semantic_digest=semantic_digest, physical_digest=physical_digest,
        creation_time=creation_time,
    )
    return repository.manifest_store.append(dataset_key, manifest)


# --------------------------------------------------------------------------
# Raw ingestion orchestration.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IngestionResult:
    batch_id: str
    dataset_key: DatasetKey
    content_digest: str
    resulting_dataset_id: str
    appended_event_count: int
    rebuilt_partition_keys: tuple[str, ...]
    was_idempotent_replay: bool


def ingest_raw_events(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, batch_id: str, ingestion_time: datetime,
    events: tuple[MarketDataEvent, ...], partitioning: PartitioningSpec,
) -> IngestionResult:
    """Ingests `events` (already constructed with the correct repository
    APPEND sequence -- see `next_sequence_for`) into `dataset_key`'s raw
    dataset, IN THE GIVEN ORDER (arrival order, never re-sorted by
    event_time). `events` may be empty (a legal, if unusual, no-op
    batch). See module docstring for the late-arrival model."""
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise IngestionError("ingest_raw_events requires a RAW_MARKET_EVENTS dataset_key")
    require_non_empty(batch_id, field_name="batch_id")
    require_tz_aware(ingestion_time, field_name="ingestion_time")
    assert dataset_key.provider is not None
    for event in events:
        if event.provider != dataset_key.provider or event.instrument_id != dataset_key.instrument_id:
            raise IngestionError(f"event {market_data_event_id(event)!r} does not belong to dataset_key {dataset_key!r}")

    ordered_event_ids = tuple(market_data_event_id(e) for e in events)
    content_digest = compute_batch_content_digest(dataset_key=dataset_key, ingestion_time=ingestion_time, ordered_event_ids=ordered_event_ids)

    batch_store = IngestionBatchStore(repository.root)
    reservation = batch_store.reserve(dataset_key=dataset_key, batch_id=batch_id, content_digest=content_digest, ingestion_time=ingestion_time)
    if reservation.status is IngestionBatchStatus.COMMITTED:
        assert reservation.resulting_dataset_id is not None
        return IngestionResult(
            batch_id=batch_id, dataset_key=dataset_key, content_digest=content_digest, resulting_dataset_id=reservation.resulting_dataset_id,
            appended_event_count=0, rebuilt_partition_keys=(), was_idempotent_replay=True,
        )

    appended_count = 0
    touched_partition_keys: set[str] = set()
    for event in events:
        before = repository.event_store.next_sequence(dataset_key.provider, dataset_key.instrument_id)
        repository.event_store.append(event)
        after = repository.event_store.next_sequence(dataset_key.provider, dataset_key.instrument_id)
        if after > before:
            appended_count += 1
        touched_partition_keys.add(partition_key_for(market_data_event_time(event), partitioning))

    all_events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    all_members = [(market_data_event_id(e), market_data_event_time(e)) for e in all_events]
    rebuild_touched_partitions(
        repository=repository, dataset_key=dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys,
    )
    manifest = rebuild_dataset_manifest_from_events(
        repository=repository, dataset_key=dataset_key, partitioning=partitioning, raw_source_dataset_id=None, creation_time=ingestion_time,
    )
    batch_store.commit(dataset_key=dataset_key, batch_id=batch_id, resulting_dataset_id=manifest.dataset_id)

    return IngestionResult(
        batch_id=batch_id, dataset_key=dataset_key, content_digest=content_digest, resulting_dataset_id=manifest.dataset_id,
        appended_event_count=appended_count, rebuilt_partition_keys=tuple(sorted(touched_partition_keys)), was_idempotent_replay=False,
    )
