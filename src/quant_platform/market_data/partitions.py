"""Deterministic time-based partitioning (Milestone 10, Phase 2).

A `Partition` binds a dataset's lineage context, a partition key/range, an
ORDERED member-id list, a schema version, and a content digest -- exactly
the five things the specification requires. Partition-building is a PURE
function of "every member currently known to fall within this partition's
[start, end) window" -- it is never incremental at the single-partition
level (a late-arriving event whose time falls into an already-built
partition causes that ONE partition to be rebuilt from its current full
membership, producing a new `partition_id`; see `ingestion.py`). This is
what makes "incremental at the dataset level" (only the affected
partition(s) get rebuilt, not the whole dataset) compatible with "each
partition's own content is always exactly and only what belongs there."

MEMBER ORDERING WITHIN A PARTITION is by `(member_time, member_id)` --
NEVER by raw ingestion/arrival order. This is a deliberate choice: it
makes partition (and therefore dataset) identity depend only on ECONOMIC
content and event-time ordering (the specification's own "ordering...
where ordering is meaningful" test), completely insulated from the
irrelevant, purely-operational detail of which order a caller happened to
submit ingestion batches in. Two datasets built from the same events
submitted in a different arrival order produce byte-identical partitions
and therefore byte-identical `dataset_id`s."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import (
    MarketDataPathSecurityError,
    MarketDataPersistenceError,
    PartitionError,
)
from quant_platform.core.json import write_json_atomic
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.manifests import DatasetKey, PartitionGranularity, PartitioningSpec
from quant_platform.ml.persistence import parse_json_strict

__all__ = ["PARTITION_KIND", "Partition", "PartitionStore", "build_partition", "partition_bounds", "partition_key_for"]

_SAFE_PARTITION_KEY = re.compile(r"^[0-9]{4}-[0-9]{2}(-[0-9]{2})?$")

PARTITION_KIND = "dataset_partition"


def partition_key_for(event_time: datetime, spec: PartitioningSpec) -> str:
    require_tz_aware(event_time, field_name="event_time")
    ts = pd.Timestamp(event_time).tz_convert("UTC")
    if spec.granularity is PartitionGranularity.DAILY:
        return ts.strftime("%Y-%m-%d")
    if spec.granularity is PartitionGranularity.MONTHLY:
        return ts.strftime("%Y-%m")
    raise PartitionError(f"Unsupported partition granularity: {spec.granularity!r}")


def partition_bounds(partition_key: str, spec: PartitioningSpec) -> tuple[datetime, datetime]:
    """The half-open `[start, end)` UTC interval `partition_key` covers."""
    if spec.granularity is PartitionGranularity.DAILY:
        try:
            start = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise PartitionError(f"Invalid daily partition_key {partition_key!r}: {exc}") from exc
        return start, start + timedelta(days=1)
    if spec.granularity is PartitionGranularity.MONTHLY:
        try:
            start = datetime.strptime(partition_key, "%Y-%m").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise PartitionError(f"Invalid monthly partition_key {partition_key!r}: {exc}") from exc
        end = (pd.Timestamp(start) + pd.DateOffset(months=1)).to_pydatetime()
        return start, end
    raise PartitionError(f"Unsupported partition granularity: {spec.granularity!r}")


@dataclass(frozen=True, slots=True)
class Partition:
    partition_id: str
    dataset_key: DatasetKey
    partition_key: str
    granularity: PartitionGranularity
    schema_version: int
    ordered_member_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    content_digest: str

    def __post_init__(self) -> None:
        require_non_empty(self.partition_key, field_name="Partition.partition_key")
        if self.schema_version < 1:
            raise PartitionError(f"Partition.schema_version must be >= 1, got {self.schema_version}")
        if not self.ordered_member_ids:
            raise PartitionError("Partition.ordered_member_ids must not be empty")
        if len(set(self.ordered_member_ids)) != len(self.ordered_member_ids):
            raise PartitionError("Partition.ordered_member_ids must not repeat a member id")
        require_tz_aware(self.start_time, field_name="Partition.start_time")
        require_tz_aware(self.end_time, field_name="Partition.end_time")
        if self.end_time <= self.start_time:
            raise PartitionError(f"Partition.end_time ({self.end_time}) must be after start_time ({self.start_time})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": PARTITION_KIND, "partition_id": self.partition_id, "dataset_key": self.dataset_key.to_json_dict(),
            "partition_key": self.partition_key, "granularity": self.granularity.value, "schema_version": self.schema_version,
            "ordered_member_ids": list(self.ordered_member_ids),
            "start_time": serialize_timestamp(self.start_time, field_name="start_time"),
            "end_time": serialize_timestamp(self.end_time, field_name="end_time"), "content_digest": self.content_digest,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["partition_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Partition:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        return cls(
            partition_id=str(raw["partition_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            partition_key=str(raw["partition_key"]), granularity=PartitionGranularity(raw["granularity"]),
            schema_version=int(str(raw["schema_version"])),
            ordered_member_ids=tuple(str(m) for m in as_json_list(raw.get("ordered_member_ids") or [], field_name="ordered_member_ids")),
            start_time=deserialize_timestamp(raw["start_time"], field_name="start_time"),
            end_time=deserialize_timestamp(raw["end_time"], field_name="end_time"), content_digest=str(raw["content_digest"]),
        )


def build_partition(*, dataset_key: DatasetKey, partition_key: str, spec: PartitioningSpec, members: list[tuple[str, datetime]]) -> Partition:
    """`members` is the CURRENT, COMPLETE set of `(member_id, member_time)`
    pairs known to fall within `partition_key`'s window -- callers pass
    the full membership every time (never a delta), and this function is
    a pure, order-independent function of that set (see module
    docstring). Raises `PartitionError` if any member's time falls
    outside the partition's own bounds -- a caller-side bug, never
    silently reassigned to the "closest" partition."""
    if not members:
        raise PartitionError(f"build_partition: no members supplied for partition_key {partition_key!r}")
    start_time, end_time = partition_bounds(partition_key, spec)
    for member_id, member_time in members:
        require_tz_aware(member_time, field_name=f"member_time for {member_id!r}")
        if not (start_time <= member_time < end_time):
            raise PartitionError(f"Member {member_id!r} at {member_time} falls outside partition {partition_key!r}'s bounds [{start_time}, {end_time})")
    ordered = sorted(members, key=lambda m: (m[1], m[0]))
    ordered_member_ids = tuple(member_id for member_id, _ in ordered)
    content_digest = compute_content_id(
        "partition_content_digest",
        {"members": [{"member_id": mid, "member_time": serialize_timestamp(mtime, field_name="member_time")} for mid, mtime in ordered]},
    )
    provisional = Partition(
        partition_id="0" * 64, dataset_key=dataset_key, partition_key=partition_key, granularity=spec.granularity,
        schema_version=spec.schema_version, ordered_member_ids=ordered_member_ids, start_time=start_time, end_time=end_time,
        content_digest=content_digest,
    )
    partition_id = compute_content_id(PARTITION_KIND, provisional.to_identity_payload())
    return Partition(
        partition_id=partition_id, dataset_key=dataset_key, partition_key=partition_key, granularity=spec.granularity,
        schema_version=spec.schema_version, ordered_member_ids=ordered_member_ids, start_time=start_time, end_time=end_time,
        content_digest=content_digest,
    )


class PartitionStore:
    """CURRENT-version-only storage for `Partition` objects -- storage
    layout: `{storage_root}/repository/partitions/{dataset_key_path}/
    {partition_key}.json`. Unlike `MarketEventStore`/`FeatureStore`
    (append-only primary facts), a partition is a DERIVED index over
    already-durable primary facts (raw events or feature records) --
    always fully reconstructible from them via `build_partition` -- so it
    is safe, and matches the specification's own compaction/rebuild
    model, for a new version to atomically REPLACE the previous physical
    file at the same `(dataset_key, partition_key)` path (never a partial
    write: `core.json.write_json_atomic`'s temp-then-rename means a
    reader never observes anything but the fully-old or fully-new
    content). Retained VERSION HISTORY lives one layer up, in
    `DatasetManifestStore` -- each manifest's own `ordered_partition_ids`
    snapshot records exactly which `partition_id` a given `partition_key`
    resolved to as of that manifest version, which is what
    `verification.py`'s "partition identity mutation detection" checks
    against across manifest versions."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "partitions" / Path(*dataset_key.storage_path_parts())

    def _partition_path(self, dataset_key: DatasetKey, partition_key: str) -> Path:
        if not _SAFE_PARTITION_KEY.match(partition_key):
            raise MarketDataPathSecurityError(f"Unsafe partition_key {partition_key!r} -- must match YYYY-MM or YYYY-MM-DD")
        path = (self._dataset_dir(dataset_key) / f"{partition_key}.json").resolve()
        dataset_dir = self._dataset_dir(dataset_key).resolve()
        if dataset_dir not in path.parents:
            raise MarketDataPathSecurityError(f"partition_key {partition_key!r} would resolve outside its dataset directory")
        return path

    def read(self, dataset_key: DatasetKey, partition_key: str) -> Partition | None:
        path = self._partition_path(dataset_key, partition_key)
        if not path.is_file():
            return None
        try:
            raw = parse_json_strict(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MarketDataPersistenceError(f"Corrupted partition file at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise MarketDataPersistenceError(f"Corrupted partition file at {path}: expected a JSON object")
        return Partition.from_json_dict(raw)

    def write(self, partition: Partition) -> Partition:
        path = self._partition_path(partition.dataset_key, partition.partition_key)
        write_json_atomic(path, partition.to_json_dict())
        return partition

    def list_partition_keys(self, dataset_key: DatasetKey) -> tuple[str, ...]:
        directory = self._dataset_dir(dataset_key)
        if not directory.is_dir():
            return ()
        keys = sorted(p.stem for p in directory.glob("*.json"))
        return tuple(keys)
