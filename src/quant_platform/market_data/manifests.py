"""Dataset identity, keys, and immutable manifests (Milestone 10, Phase 2).

TWO DISTINCT IDENTITY CONCEPTS, ANALOGOUS TO A GIT BRANCH VS A COMMIT:

- `DatasetKey` is the STABLE routing/lineage identity of "which dataset"
  an ingestion batch, checkpoint, or partition belongs to -- it never
  changes as more data is committed (e.g. "raw XAUUSD from mt5", or
  "sma_20 v1 for XAUUSD"). It is NOT content-addressed from economic
  data; it is a small, caller-declared composite (see below).
- `dataset_id` (on `DatasetManifest`) is a per-VERSION, content-addressed
  identity that changes every time new data is committed (a new batch,
  a late-arriving event, a new partition). Two manifests for the SAME
  `DatasetKey` at different points in time legitimately have different
  `dataset_id`s -- that is the whole point of "dataset versioning": each
  commit is its own immutable, independently verifiable version, exactly
  like each git commit under one branch has its own hash.

Because ingestion is incremental, `dataset_id` must be a pure function of
ECONOMIC content only -- never of the filesystem root, a temp path, a
lock file, a pid/hostname, operational logging metadata, or serialization
whitespace (`core.json.canonical_json_bytes` already guarantees the
last). `DatasetManifest.to_identity_payload()` therefore excludes
`dataset_id` itself, `physical_digest` (an explicitly physical-layout
signal -- see `compaction.py`'s own invariant: physical layout may change
without economic content changing), `creation_time` (a caller-supplied
OPERATIONAL label, exactly like `portfolio_risk.ledger.RiskLedgerEntry.
recorded_time` -- present in the durable JSON, never part of identity),
and `completion_status` (a manifest object is only ever constructed once
a commit is fully complete -- see `repository.py`'s atomicity discipline
-- so this field is always `"complete"` for any `DatasetManifest` that
exists at all; it is retained purely for schema completeness/future
staged-manifest use, never to encode economically meaningful state)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    DatasetManifestError,
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.core.types import Timeframe
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import parse_json_strict

__all__ = [
    "DATASET_MANIFEST_KIND",
    "REPOSITORY_SCHEMA_VERSION",
    "DatasetKey",
    "DatasetKind",
    "DatasetManifest",
    "DatasetManifestStore",
    "PartitionGranularity",
    "PartitioningSpec",
    "compute_dataset_id",
    "create_dataset_manifest",
]

REPOSITORY_SCHEMA_VERSION = 1
DATASET_MANIFEST_KIND = "dataset_manifest"


class DatasetKind(Enum):
    RAW_MARKET_EVENTS = "raw_market_events"
    DERIVED_FEATURES = "derived_features"
    MACRO_OBSERVATIONS = "macro_observations"
    """Milestone 10, Phase 4A: a `DatasetKey` SCOPING identity only, for
    stores that need one purely to compute a storage path/durable-record
    field (`checkpoints.OperationStore`-shaped ledgers, `provenance.
    ProvenanceStore`, `quarantine.QuarantineStore` -- none of which care
    what kind of economic record the operation ultimately produces).
    Macro observations do NOT participate in `DatasetManifest`/
    `PartitionStore` dataset-versioning (that machinery is shaped
    specifically for `MarketEventStore`-backed candle/tick/quote/trade
    data); the durable economic record store remains Phase 1's own
    `macro.MacroEventStore`, keyed by `(provider, series_id)` directly.
    `instrument_id` is repurposed to mean `series_id` for this kind --
    same field, since a macro series id (e.g. `"DGS10"`) is exactly as
    path-safe as an instrument id already must be."""

    CROSS_ASSET_MARKET_BARS = "cross_asset_market_bars"
    """Milestone 10, Phase 4C: the same scoping-only pattern
    `MACRO_OBSERVATIONS` established, for `collectors.cross_asset`'s own
    `MarketDriverBar` records -- these ALSO live in their own new store
    (`cross_asset.market_record.MarketDriverBarStore`), never
    `MarketEventStore`/`DatasetManifest`/`PartitionStore` versioning,
    reusing `ProvenanceStore`/`QuarantineStore` purely for scoping.
    `instrument_id` is repurposed to mean `f"{canonical_driver_id}__
    {instrument_form}"` -- a single provider may map a canonical driver
    through more than one instrument form (e.g. both an ETF proxy and a
    futures contract), and each must never share a provenance/quarantine
    scope with the other."""


class PartitionGranularity(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    """Phase 2 deliberately supports only these two, calendar-based
    granularities -- the "minimum useful set" the specification itself
    permits choosing. Fixed-event-count partitioning is a documented,
    out-of-scope extension (see `docs/market_data_architecture.md`'s
    Phase 2 Known Limitations): every requirement this phase must satisfy
    (boundary-timestamp correctness, deterministic membership, checkpoint
    carry windows) is fully exercised by time-based partitioning alone."""


@dataclass(frozen=True, slots=True)
class PartitioningSpec:
    granularity: PartitionGranularity
    schema_version: int = REPOSITORY_SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, object]:
        return {"granularity": self.granularity.value, "schema_version": self.schema_version}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PartitioningSpec:
        return cls(granularity=PartitionGranularity(raw["granularity"]), schema_version=int(str(raw["schema_version"])))


@dataclass(frozen=True, slots=True)
class DatasetKey:
    """Stable lineage identity -- see module docstring. `provider` is
    required (and `feature_name`/`feature_version` forbidden) for
    `RAW_MARKET_EVENTS`; `feature_name`/`feature_version` are required
    (and `provider` forbidden) for `DERIVED_FEATURES` -- a raw event
    stream has no single feature identity, and a `FeatureRecord` (Phase
    1) carries no `provider` field, so mixing the two vocabularies would
    silently paper over which kind a key describes."""

    dataset_kind: DatasetKind
    instrument_id: str
    provider: str | None = None
    feature_name: str | None = None
    feature_version: int | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.instrument_id, field_name="DatasetKey.instrument_id")
        if self.dataset_kind in (DatasetKind.RAW_MARKET_EVENTS, DatasetKind.MACRO_OBSERVATIONS, DatasetKind.CROSS_ASSET_MARKET_BARS):
            if not self.provider:
                raise DatasetManifestError(f"DatasetKey.provider is required for {self.dataset_kind.value}")
            if self.feature_name is not None or self.feature_version is not None:
                raise DatasetManifestError(f"DatasetKey.feature_name/feature_version must be None for {self.dataset_kind.value}")
        else:
            if self.provider is not None:
                raise DatasetManifestError("DatasetKey.provider must be None for DERIVED_FEATURES")
            if not self.feature_name:
                raise DatasetManifestError("DatasetKey.feature_name is required for DERIVED_FEATURES")
            if self.feature_version is None or self.feature_version < 1:
                raise DatasetManifestError("DatasetKey.feature_version must be >= 1 for DERIVED_FEATURES")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_kind": self.dataset_kind.value, "instrument_id": self.instrument_id, "provider": self.provider,
            "feature_name": self.feature_name, "feature_version": self.feature_version,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DatasetKey:
        return cls(
            dataset_kind=DatasetKind(raw["dataset_kind"]), instrument_id=str(raw["instrument_id"]),
            provider=(None if raw.get("provider") is None else str(raw["provider"])),
            feature_name=(None if raw.get("feature_name") is None else str(raw["feature_name"])),
            feature_version=(None if raw.get("feature_version") is None else int(str(raw["feature_version"]))),
        )

    def storage_path_parts(self) -> tuple[str, ...]:
        """Deterministic, path-safe directory components -- reused by
        `repository.py` for every store this key touches. `instrument_id`
        is already path-safe by construction (`normalization.
        derive_instrument_id` uses `__`, never `:` or `/`)."""
        if self.dataset_kind is DatasetKind.RAW_MARKET_EVENTS:
            assert self.provider is not None
            return ("raw_market_events", self.provider, self.instrument_id)
        if self.dataset_kind is DatasetKind.MACRO_OBSERVATIONS:
            assert self.provider is not None
            return ("macro_observations", self.provider, self.instrument_id)
        if self.dataset_kind is DatasetKind.CROSS_ASSET_MARKET_BARS:
            assert self.provider is not None
            return ("cross_asset_market_bars", self.provider, self.instrument_id)
        assert self.feature_name is not None and self.feature_version is not None
        return ("derived_features", self.feature_name, f"v{self.feature_version}", self.instrument_id)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    dataset_key: DatasetKey
    schema_version: int
    timeframe: Timeframe | None
    """Always `None` for `RAW_MARKET_EVENTS` (a raw stream interleaves
    timeframe-less ticks/quotes/trades with candles of a timeframe -- see
    module docstring); the candles' own timeframe for `DERIVED_FEATURES`."""
    partitioning: PartitioningSpec
    first_event_time: datetime | None
    last_event_time: datetime | None
    event_count: int
    ordered_partition_ids: tuple[str, ...]
    raw_source_dataset_id: str | None
    """The exact `dataset_id` of the raw dataset version this feature
    dataset was generated from -- `None` for `RAW_MARKET_EVENTS`,
    required for `DERIVED_FEATURES` (lineage)."""
    semantic_digest: str
    physical_digest: str
    creation_time: datetime
    completion_status: str = "complete"

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise DatasetManifestError(f"DatasetManifest.schema_version must be >= 1, got {self.schema_version}")
        if self.event_count < 0:
            raise DatasetManifestError(f"DatasetManifest.event_count must be >= 0, got {self.event_count}")
        if self.event_count == 0:
            if self.first_event_time is not None or self.last_event_time is not None or self.ordered_partition_ids:
                raise DatasetManifestError("DatasetManifest with event_count == 0 must have no event times and no partitions")
        else:
            if self.first_event_time is None or self.last_event_time is None:
                raise DatasetManifestError("DatasetManifest with event_count > 0 must have first_event_time and last_event_time")
            require_tz_aware(self.first_event_time, field_name="DatasetManifest.first_event_time")
            require_tz_aware(self.last_event_time, field_name="DatasetManifest.last_event_time")
            if self.last_event_time < self.first_event_time:
                raise DatasetManifestError(f"DatasetManifest.last_event_time ({self.last_event_time}) must be >= first_event_time ({self.first_event_time})")
            if not self.ordered_partition_ids:
                raise DatasetManifestError("DatasetManifest with event_count > 0 must reference at least one partition")
        if len(set(self.ordered_partition_ids)) != len(self.ordered_partition_ids):
            raise DatasetManifestError("DatasetManifest.ordered_partition_ids must not repeat a partition id")
        require_tz_aware(self.creation_time, field_name="DatasetManifest.creation_time")
        if self.dataset_key.dataset_kind is DatasetKind.RAW_MARKET_EVENTS:
            if self.raw_source_dataset_id is not None:
                raise DatasetManifestError("DatasetManifest.raw_source_dataset_id must be None for RAW_MARKET_EVENTS")
            if self.timeframe is not None:
                raise DatasetManifestError("DatasetManifest.timeframe must be None for RAW_MARKET_EVENTS")
        else:
            if self.raw_source_dataset_id is None:
                raise DatasetManifestError("DatasetManifest.raw_source_dataset_id is required for DERIVED_FEATURES")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id, "dataset_key": self.dataset_key.to_json_dict(), "schema_version": self.schema_version,
            "timeframe": (None if self.timeframe is None else self.timeframe.value), "partitioning": self.partitioning.to_json_dict(),
            "first_event_time": (None if self.first_event_time is None else serialize_timestamp(self.first_event_time, field_name="first_event_time")),
            "last_event_time": (None if self.last_event_time is None else serialize_timestamp(self.last_event_time, field_name="last_event_time")),
            "event_count": self.event_count, "ordered_partition_ids": list(self.ordered_partition_ids),
            "raw_source_dataset_id": self.raw_source_dataset_id, "semantic_digest": self.semantic_digest,
            "physical_digest": self.physical_digest, "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
            "completion_status": self.completion_status,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        for key in ("dataset_id", "physical_digest", "creation_time", "completion_status"):
            del payload[key]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DatasetManifest:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        return cls(
            dataset_id=str(raw["dataset_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            schema_version=int(str(raw["schema_version"])), timeframe=(None if raw.get("timeframe") is None else Timeframe(raw["timeframe"])),
            partitioning=PartitioningSpec.from_json_dict(as_json_dict(raw["partitioning"], field_name="partitioning")),
            first_event_time=(None if raw.get("first_event_time") is None else deserialize_timestamp(raw["first_event_time"], field_name="first_event_time")),
            last_event_time=(None if raw.get("last_event_time") is None else deserialize_timestamp(raw["last_event_time"], field_name="last_event_time")),
            event_count=int(str(raw["event_count"])),
            ordered_partition_ids=tuple(str(p) for p in as_json_list(raw.get("ordered_partition_ids") or [], field_name="ordered_partition_ids")),
            raw_source_dataset_id=(None if raw.get("raw_source_dataset_id") is None else str(raw["raw_source_dataset_id"])),
            semantic_digest=str(raw["semantic_digest"]), physical_digest=str(raw["physical_digest"]),
            creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
            completion_status=str(raw.get("completion_status", "complete")),
        )


@contextmanager
def manifest_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire manifest store lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Manifest store lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class DatasetManifestStore:
    """Append-only VERSION HISTORY of `DatasetManifest`s for one
    `DatasetKey` -- storage layout: `{storage_root}/repository/datasets/
    {dataset_key_path}/manifests.jsonl`. The LATEST entry is the dataset's
    current state; every earlier entry remains independently readable and
    verifiable -- "dataset version identities" (the milestone's own
    phrase) means every commit, not only the most recent one.

    Appending a manifest whose `dataset_id` equals the current latest
    entry's is an idempotent no-op (never records a duplicate "new"
    version for unchanged content) -- this is what makes `ingestion.py`'s
    exact-retry path safe to simply re-run the whole flow rather than
    needing its own separate short-circuit."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "datasets" / Path(*dataset_key.storage_path_parts())

    def _manifests_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "manifests.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".manifests.lock"

    def read_history(self, dataset_key: DatasetKey) -> list[DatasetManifest]:
        path = self._manifests_path(dataset_key)
        if not path.is_file():
            return []
        manifests: list[DatasetManifest] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted manifest history line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted manifest history line for dataset {dataset_key!r}: expected a JSON object")
            manifests.append(DatasetManifest.from_json_dict(raw))
        return manifests

    def read_current(self, dataset_key: DatasetKey) -> DatasetManifest | None:
        history = self.read_history(dataset_key)
        return history[-1] if history else None

    def append(self, dataset_key: DatasetKey, manifest: DatasetManifest) -> DatasetManifest:
        if manifest.dataset_key != dataset_key:
            raise DatasetManifestError(f"DatasetManifest.dataset_key {manifest.dataset_key!r} does not match target {dataset_key!r}")
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with manifest_store_lock(lock_path):
            current = self.read_current(dataset_key)
            if current is not None and current.dataset_id == manifest.dataset_id:
                return current  # idempotent no-op: content-identical version
            path = self._manifests_path(dataset_key)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(manifest.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return manifest


def compute_dataset_id(manifest: DatasetManifest) -> str:
    return compute_content_id(DATASET_MANIFEST_KIND, manifest.to_identity_payload())


def create_dataset_manifest(
    *, dataset_key: DatasetKey, schema_version: int, timeframe: Timeframe | None, partitioning: PartitioningSpec,
    first_event_time: datetime | None, last_event_time: datetime | None, event_count: int, ordered_partition_ids: tuple[str, ...],
    raw_source_dataset_id: str | None, semantic_digest: str, physical_digest: str, creation_time: datetime,
) -> DatasetManifest:
    provisional = DatasetManifest(
        dataset_id="0" * 64, dataset_key=dataset_key, schema_version=schema_version, timeframe=timeframe, partitioning=partitioning,
        first_event_time=first_event_time, last_event_time=last_event_time, event_count=event_count,
        ordered_partition_ids=ordered_partition_ids, raw_source_dataset_id=raw_source_dataset_id, semantic_digest=semantic_digest,
        physical_digest=physical_digest, creation_time=creation_time,
    )
    dataset_id = compute_dataset_id(provisional)
    return DatasetManifest(
        dataset_id=dataset_id, dataset_key=dataset_key, schema_version=schema_version, timeframe=timeframe, partitioning=partitioning,
        first_event_time=first_event_time, last_event_time=last_event_time, event_count=event_count,
        ordered_partition_ids=ordered_partition_ids, raw_source_dataset_id=raw_source_dataset_id, semantic_digest=semantic_digest,
        physical_digest=physical_digest, creation_time=creation_time,
    )
