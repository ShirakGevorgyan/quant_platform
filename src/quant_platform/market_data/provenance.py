"""Bidirectional source-row <-> event provenance (Milestone 10, Phase 3).

Every `ProvenanceRecord` binds ONE source row (`source_manifest_id` +
`source_row_index`, matching `adapters.SourceRowCoordinate`) to the ONE
resulting canonical event it produced -- plus every identity a caller
needs to independently answer "which source row produced this event",
"which event came from this source row", and "which mapping/
normalization rules were used" without touching a human-readable log
(the specification's own explicit requirement). `provenance_id` is
content-addressed from every field below EXCEPT itself and
`recorded_time` (operational, exactly like `creation_time` elsewhere in
this package) -- so an EXACT retry (identical source row, identical
mapping/normalization/batch context, identical resulting event)
reproduces the identical `provenance_id` and is idempotently absorbed by
`ProvenanceStore.append`; a CONFLICTING retry (same source coordinate,
anything else different) raises `ProvenanceError`, per `ingestion.
IngestionBatchStore`'s own reserve/commit conflict pattern -- this store
follows the identical append-only-JSONL-plus-file-lock shape, scoped
under the same target `DatasetKey`."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPersistenceError,
    ProvenanceError,
)
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.market_data.adapters import SourceRowCoordinate
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.manifests import DatasetKey
from quant_platform.ml.concurrency import experiment_lock

__all__ = [
    "PROVENANCE_RECORD_KIND",
    "ProvenanceConflict",
    "ProvenanceRecord",
    "ProvenanceStore",
    "create_provenance_record",
    "find_provenance_conflicts",
]

PROVENANCE_RECORD_KIND = "provenance_record"


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    provenance_id: str
    source_manifest_id: str
    source_row_index: int
    source_record_digest: str
    original_timestamp_text: str
    normalized_event_time: datetime
    instrument_mapping_id: str
    resolved_instrument_id: str
    timeframe_mapping_id: str | None
    timezone_policy_id: str
    ingestion_batch_id: str
    event_id: str
    dataset_id: str
    recorded_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.source_manifest_id, field_name="ProvenanceRecord.source_manifest_id")
        if self.source_row_index < 0:
            raise ProvenanceError(f"ProvenanceRecord.source_row_index must be >= 0, got {self.source_row_index}")
        require_non_empty(self.source_record_digest, field_name="ProvenanceRecord.source_record_digest")
        require_non_empty(self.instrument_mapping_id, field_name="ProvenanceRecord.instrument_mapping_id")
        require_non_empty(self.resolved_instrument_id, field_name="ProvenanceRecord.resolved_instrument_id")
        require_non_empty(self.timezone_policy_id, field_name="ProvenanceRecord.timezone_policy_id")
        require_non_empty(self.ingestion_batch_id, field_name="ProvenanceRecord.ingestion_batch_id")
        require_non_empty(self.event_id, field_name="ProvenanceRecord.event_id")
        require_non_empty(self.dataset_id, field_name="ProvenanceRecord.dataset_id")
        require_tz_aware(self.normalized_event_time, field_name="ProvenanceRecord.normalized_event_time")
        require_tz_aware(self.recorded_time, field_name="ProvenanceRecord.recorded_time")

    def source_coordinate(self) -> SourceRowCoordinate:
        return SourceRowCoordinate(source_manifest_id=self.source_manifest_id, row_index=self.source_row_index)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": PROVENANCE_RECORD_KIND, "provenance_id": self.provenance_id, "source_manifest_id": self.source_manifest_id,
            "source_row_index": self.source_row_index, "source_record_digest": self.source_record_digest,
            "original_timestamp_text": self.original_timestamp_text,
            "normalized_event_time": serialize_timestamp(self.normalized_event_time, field_name="normalized_event_time"),
            "instrument_mapping_id": self.instrument_mapping_id, "resolved_instrument_id": self.resolved_instrument_id,
            "timeframe_mapping_id": self.timeframe_mapping_id, "timezone_policy_id": self.timezone_policy_id,
            "ingestion_batch_id": self.ingestion_batch_id, "event_id": self.event_id, "dataset_id": self.dataset_id,
            "recorded_time": serialize_timestamp(self.recorded_time, field_name="recorded_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Excludes `dataset_id` in addition to `provenance_id`/
        `recorded_time`: the RESULTING dataset VERSION is a snapshot of
        repository state at commit time, which can legitimately drift
        for reasons that have nothing to do with THIS row (e.g. other,
        unrelated data landing in the same dataset between an original
        attempt and a later idempotent retry) -- see
        `orchestration.py`'s own "NO IN-MEMORY-ONLY CORRECTNESS STATE"
        discussion. Including it here would make an otherwise-exact
        retry spuriously conflict. `event_id` (the part that actually
        matters -- did this row produce the SAME economic event)
        remains in the identity payload."""
        payload = dict(self.to_json_dict())
        del payload["provenance_id"]
        del payload["recorded_time"]
        del payload["dataset_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ProvenanceRecord:
        return cls(
            provenance_id=str(raw["provenance_id"]), source_manifest_id=str(raw["source_manifest_id"]),
            source_row_index=int(str(raw["source_row_index"])), source_record_digest=str(raw["source_record_digest"]),
            original_timestamp_text=str(raw["original_timestamp_text"]),
            normalized_event_time=deserialize_timestamp(raw["normalized_event_time"], field_name="normalized_event_time"),
            instrument_mapping_id=str(raw["instrument_mapping_id"]), resolved_instrument_id=str(raw["resolved_instrument_id"]),
            timeframe_mapping_id=(None if raw.get("timeframe_mapping_id") is None else str(raw["timeframe_mapping_id"])),
            timezone_policy_id=str(raw["timezone_policy_id"]), ingestion_batch_id=str(raw["ingestion_batch_id"]),
            event_id=str(raw["event_id"]), dataset_id=str(raw["dataset_id"]),
            recorded_time=deserialize_timestamp(raw["recorded_time"], field_name="recorded_time"),
        )


def create_provenance_record(
    *, source_manifest_id: str, source_row_index: int, source_record_digest: str, original_timestamp_text: str,
    normalized_event_time: datetime, instrument_mapping_id: str, resolved_instrument_id: str, timeframe_mapping_id: str | None,
    timezone_policy_id: str, ingestion_batch_id: str, event_id: str, dataset_id: str, recorded_time: datetime,
) -> ProvenanceRecord:
    provisional = ProvenanceRecord(
        provenance_id="0" * 64, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
        source_record_digest=source_record_digest, original_timestamp_text=original_timestamp_text,
        normalized_event_time=normalized_event_time, instrument_mapping_id=instrument_mapping_id,
        resolved_instrument_id=resolved_instrument_id, timeframe_mapping_id=timeframe_mapping_id,
        timezone_policy_id=timezone_policy_id, ingestion_batch_id=ingestion_batch_id, event_id=event_id,
        dataset_id=dataset_id, recorded_time=recorded_time,
    )
    provenance_id = compute_content_id(PROVENANCE_RECORD_KIND, provisional.to_identity_payload())
    return ProvenanceRecord(
        provenance_id=provenance_id, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
        source_record_digest=source_record_digest, original_timestamp_text=original_timestamp_text,
        normalized_event_time=normalized_event_time, instrument_mapping_id=instrument_mapping_id,
        resolved_instrument_id=resolved_instrument_id, timeframe_mapping_id=timeframe_mapping_id,
        timezone_policy_id=timezone_policy_id, ingestion_batch_id=ingestion_batch_id, event_id=event_id,
        dataset_id=dataset_id, recorded_time=recorded_time,
    )


# --------------------------------------------------------------------------
# Cross-record conflict detection -- a pure function over already-read
# records, reused by both `ProvenanceStore.append` (a single incremental
# check against one coordinate's existing history) and reconciliation
# (a full-store audit; see `ProvenanceError`'s own docstring: "an index
# built from durable provenance evidence finds a genuine conflict").
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProvenanceConflict:
    issue_code: str
    detail: str


def find_provenance_conflicts(records: list[ProvenanceRecord]) -> tuple[ProvenanceConflict, ...]:
    """Two conflict shapes: (1) `coordinate_bound_to_multiple_events` --
    the SAME source row (`source_manifest_id`, `source_row_index`)
    durably bound to more than one DIFFERENT `event_id` -- always a
    genuine defect, since `ProvenanceStore.append` itself refuses to
    durably record this. (2) `event_bound_to_multiple_source_rows` --
    the SAME `event_id` produced by more than one DIFFERENT source row --
    reported, not raised, since exact-duplicate source rows legitimately
    normalize to the identical content-addressed event (the
    specification's own explicitly-supported case); a caller comparing
    each conflicting record's `source_record_digest` can distinguish a
    genuine collision (different digests) from expected duplicate
    absorption (identical digests)."""
    by_coordinate: dict[tuple[str, int], set[str]] = {}
    by_event: dict[str, set[tuple[str, int]]] = {}
    for record in records:
        coord_key = (record.source_manifest_id, record.source_row_index)
        by_coordinate.setdefault(coord_key, set()).add(record.event_id)
        by_event.setdefault(record.event_id, set()).add(coord_key)

    conflicts: list[ProvenanceConflict] = []
    for coord_key, event_ids in sorted(by_coordinate.items()):
        if len(event_ids) > 1:
            conflicts.append(ProvenanceConflict(
                issue_code="coordinate_bound_to_multiple_events",
                detail=f"source_manifest_id={coord_key[0]!r} row_index={coord_key[1]} is bound to {len(event_ids)} distinct event_id(s): {sorted(event_ids)}",
            ))
    for event_id, coord_keys in sorted(by_event.items()):
        if len(coord_keys) > 1:
            conflicts.append(ProvenanceConflict(
                issue_code="event_bound_to_multiple_source_rows",
                detail=f"event_id={event_id!r} is produced by {len(coord_keys)} distinct source row(s): {sorted(coord_keys)}",
            ))
    return tuple(conflicts)


# --------------------------------------------------------------------------
# Durable append-only store.
# --------------------------------------------------------------------------
@contextmanager
def _provenance_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire provenance lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Provenance lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class ProvenanceStore:
    """Append-only ledger: `{storage_root}/repository/provenance/
    {dataset_key_path}/provenance.jsonl`, one file per TARGET dataset
    (mirrors `ingestion.IngestionBatchStore`'s own layout exactly)."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "provenance" / Path(*dataset_key.storage_path_parts())

    def _provenance_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "provenance.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".provenance.lock"

    def read_all(self, dataset_key: DatasetKey) -> list[ProvenanceRecord]:
        path = self._provenance_path(dataset_key)
        if not path.is_file():
            return []
        records: list[ProvenanceRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted provenance line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted provenance line for dataset {dataset_key!r}: expected a JSON object")
            records.append(ProvenanceRecord.from_json_dict(raw))
        return records

    def read_by_source_coordinate(self, dataset_key: DatasetKey, coordinate: SourceRowCoordinate) -> ProvenanceRecord | None:
        matching = [
            r for r in self.read_all(dataset_key)
            if r.source_manifest_id == coordinate.source_manifest_id and r.source_row_index == coordinate.row_index
        ]
        return matching[-1] if matching else None

    def read_by_event_id(self, dataset_key: DatasetKey, event_id: str) -> list[ProvenanceRecord]:
        return [r for r in self.read_all(dataset_key) if r.event_id == event_id]

    def _append_line(self, dataset_key: DatasetKey, record: ProvenanceRecord) -> None:
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        path = self._provenance_path(dataset_key)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, dataset_key: DatasetKey, record: ProvenanceRecord) -> ProvenanceRecord:
        """Idempotent for an EXACT retry (same source coordinate, same
        `provenance_id`); raises `ProvenanceError` for a CONFLICTING
        retry (same source coordinate, different `provenance_id` -- i.e.
        anything else about the record changed)."""
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _provenance_store_lock(lock_path):
            existing = self.read_by_source_coordinate(dataset_key, record.source_coordinate())
            if existing is not None:
                if existing.provenance_id == record.provenance_id:
                    return existing
                raise ProvenanceError(
                    f"source row (source_manifest_id={record.source_manifest_id!r}, row_index={record.source_row_index}) is already "
                    f"bound to provenance_id {existing.provenance_id!r} (event_id={existing.event_id!r}); a conflicting "
                    f"provenance_id {record.provenance_id!r} (event_id={record.event_id!r}) was submitted"
                )
            self._append_line(dataset_key, record)
            return record
