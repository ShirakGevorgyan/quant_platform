"""Durable, self-verifying checkpoints (Milestone 10, Phase 2).

A checkpoint is NEVER the primary source of truth -- it is a durable
HINT for where incremental processing left off, always independently
re-derivable (and re-verified) from the underlying manifest/store state.
`verify_raw_ingestion_checkpoint`/`verify_feature_generation_checkpoint`
recompute a FRESH checkpoint from current durable evidence and compare it
against the stored one field-by-field; a caller who trusts a checkpoint
WITHOUT calling one of these first is using this module incorrectly (see
`recovery.py`, which always verifies before relying on one).

CARRY STATE, BY DESIGN CHOICE: `FeatureGenerationCheckpoint` stores a
`carry_window_size` (an int -- how many trailing raw candles incremental
generation must re-read before the new batch to give every rolling
indicator correct context), NOT cached partial sums/EMA state. The raw
event store is itself a durable, replayable source of truth; re-deriving
the needed trailing context from it via a bounded backward read is
strictly safer than trusting a cached numeric accumulator that could
silently drift from the source with no way to detect it. This makes
`FeatureGenerationCheckpoint` itself trivially self-verifying (it is pure
metadata, nothing to independently recompute except by re-running
generation, which `verify_feature_generation_checkpoint` does not do --
that full re-derivation is `replay.py`'s job)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.core.exceptions import (
    CheckpointError,
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPersistenceError,
    StaleCheckpointError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import parse_json_strict

__all__ = [
    "FEATURE_GENERATION_CHECKPOINT_KIND",
    "RAW_INGESTION_CHECKPOINT_KIND",
    "CheckpointStore",
    "FeatureGenerationCheckpoint",
    "RawIngestionCheckpoint",
    "compute_raw_ingestion_checkpoint",
    "create_feature_generation_checkpoint",
    "verify_feature_generation_checkpoint",
    "verify_raw_ingestion_checkpoint",
]

RAW_INGESTION_CHECKPOINT_KIND = "raw_ingestion_checkpoint"
FEATURE_GENERATION_CHECKPOINT_KIND = "feature_generation_checkpoint"


# --------------------------------------------------------------------------
# Raw ingestion checkpoint.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RawIngestionCheckpoint:
    checkpoint_id: str
    dataset_key: DatasetKey
    last_committed_sequence: int
    last_committed_batch_id: str | None
    last_canonical_partition_id: str | None
    semantic_digest: str
    checkpoint_time: datetime

    def __post_init__(self) -> None:
        if self.last_committed_sequence < 0:
            raise CheckpointError(f"RawIngestionCheckpoint.last_committed_sequence must be >= 0, got {self.last_committed_sequence}")
        require_tz_aware(self.checkpoint_time, field_name="RawIngestionCheckpoint.checkpoint_time")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": RAW_INGESTION_CHECKPOINT_KIND, "checkpoint_id": self.checkpoint_id, "dataset_key": self.dataset_key.to_json_dict(),
            "last_committed_sequence": self.last_committed_sequence, "last_committed_batch_id": self.last_committed_batch_id,
            "last_canonical_partition_id": self.last_canonical_partition_id, "semantic_digest": self.semantic_digest,
            "checkpoint_time": serialize_timestamp(self.checkpoint_time, field_name="checkpoint_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["checkpoint_id"]
        del payload["checkpoint_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RawIngestionCheckpoint:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            checkpoint_id=str(raw["checkpoint_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            last_committed_sequence=int(str(raw["last_committed_sequence"])),
            last_committed_batch_id=(None if raw.get("last_committed_batch_id") is None else str(raw["last_committed_batch_id"])),
            last_canonical_partition_id=(None if raw.get("last_canonical_partition_id") is None else str(raw["last_canonical_partition_id"])),
            semantic_digest=str(raw["semantic_digest"]), checkpoint_time=deserialize_timestamp(raw["checkpoint_time"], field_name="checkpoint_time"),
        )


def _create_raw_checkpoint(
    *, dataset_key: DatasetKey, last_committed_sequence: int, last_committed_batch_id: str | None, last_canonical_partition_id: str | None,
    semantic_digest: str, checkpoint_time: datetime,
) -> RawIngestionCheckpoint:
    provisional = RawIngestionCheckpoint(
        checkpoint_id="0" * 64, dataset_key=dataset_key, last_committed_sequence=last_committed_sequence,
        last_committed_batch_id=last_committed_batch_id, last_canonical_partition_id=last_canonical_partition_id,
        semantic_digest=semantic_digest, checkpoint_time=checkpoint_time,
    )
    checkpoint_id = compute_content_id(RAW_INGESTION_CHECKPOINT_KIND, provisional.to_identity_payload())
    return RawIngestionCheckpoint(
        checkpoint_id=checkpoint_id, dataset_key=dataset_key, last_committed_sequence=last_committed_sequence,
        last_committed_batch_id=last_committed_batch_id, last_canonical_partition_id=last_canonical_partition_id,
        semantic_digest=semantic_digest, checkpoint_time=checkpoint_time,
    )


def compute_raw_ingestion_checkpoint(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, last_committed_batch_id: str | None, checkpoint_time: datetime,
) -> RawIngestionCheckpoint:
    """Derives a checkpoint FRESH from `repository`'s current durable
    state -- the only supported way to produce one. `last_committed_batch_id`
    is caller-supplied (the batch that triggered this checkpoint), since
    the repository's own batch ledger records EVERY batch, not which one
    a particular checkpoint call is being taken in response to."""
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise CheckpointError("compute_raw_ingestion_checkpoint requires a RAW_MARKET_EVENTS dataset_key")
    assert dataset_key.provider is not None
    manifest = repository.manifest_store.read_current(dataset_key)
    last_committed_sequence = repository.event_store.next_sequence(dataset_key.provider, dataset_key.instrument_id)
    last_canonical_partition_id = manifest.ordered_partition_ids[-1] if manifest is not None and manifest.ordered_partition_ids else None
    semantic_digest = manifest.semantic_digest if manifest is not None else compute_content_id("raw_dataset_semantic_digest", {"events": []})
    return _create_raw_checkpoint(
        dataset_key=dataset_key, last_committed_sequence=last_committed_sequence, last_committed_batch_id=last_committed_batch_id,
        last_canonical_partition_id=last_canonical_partition_id, semantic_digest=semantic_digest, checkpoint_time=checkpoint_time,
    )


def verify_raw_ingestion_checkpoint(checkpoint: RawIngestionCheckpoint, *, repository: MarketDataRepository) -> None:
    """Raises `StaleCheckpointError` if `checkpoint` does not match
    `repository`'s CURRENT durable state (behind OR ahead), and
    `CheckpointError` if it is internally forged (does not reproduce its
    own `checkpoint_id`). Never returns a value to "trust" -- a clean
    return IS the confirmation."""
    recomputed_id = compute_content_id(RAW_INGESTION_CHECKPOINT_KIND, checkpoint.to_identity_payload())
    if recomputed_id != checkpoint.checkpoint_id:
        raise CheckpointError(f"RawIngestionCheckpoint {checkpoint.checkpoint_id!r} does not reproduce its own id -- forged or tampered")
    current = compute_raw_ingestion_checkpoint(
        repository=repository, dataset_key=checkpoint.dataset_key, last_committed_batch_id=checkpoint.last_committed_batch_id,
        checkpoint_time=checkpoint.checkpoint_time,
    )
    if current.last_committed_sequence != checkpoint.last_committed_sequence:
        raise StaleCheckpointError(
            f"RawIngestionCheckpoint claims last_committed_sequence={checkpoint.last_committed_sequence}, "
            f"repository currently has {current.last_committed_sequence}"
        )
    if current.semantic_digest != checkpoint.semantic_digest:
        raise StaleCheckpointError("RawIngestionCheckpoint semantic_digest does not match the repository's current semantic_digest")
    if current.last_canonical_partition_id != checkpoint.last_canonical_partition_id:
        raise StaleCheckpointError("RawIngestionCheckpoint last_canonical_partition_id does not match the repository's current state")


# --------------------------------------------------------------------------
# Feature-generation checkpoint.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureGenerationCheckpoint:
    checkpoint_id: str
    raw_dataset_key: DatasetKey
    raw_dataset_id: str
    feature_dataset_key: DatasetKey
    last_processed_raw_event_time: datetime | None
    carry_window_size: int | None
    """`None` means "unbounded -- the full raw history is re-read every
    time" (`vwap`/`ema`; see `feature_generation.py`'s own
    `_UNBOUNDED_CARRY_FEATURES` docstring for why), never a sentinel
    integer -- a sentinel would either collide with a genuine window size
    or fail this class's own non-negative validation, exactly the defect
    this phase's own adversarial testing caught and fixed (see the
    delivery report)."""
    resulting_feature_dataset_id: str | None
    semantic_digest: str
    checkpoint_time: datetime

    def __post_init__(self) -> None:
        if self.carry_window_size is not None and self.carry_window_size < 0:
            raise CheckpointError(f"FeatureGenerationCheckpoint.carry_window_size must be >= 0 or None, got {self.carry_window_size}")
        require_tz_aware(self.checkpoint_time, field_name="FeatureGenerationCheckpoint.checkpoint_time")
        if self.last_processed_raw_event_time is not None:
            require_tz_aware(self.last_processed_raw_event_time, field_name="FeatureGenerationCheckpoint.last_processed_raw_event_time")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": FEATURE_GENERATION_CHECKPOINT_KIND, "checkpoint_id": self.checkpoint_id, "raw_dataset_key": self.raw_dataset_key.to_json_dict(),
            "raw_dataset_id": self.raw_dataset_id, "feature_dataset_key": self.feature_dataset_key.to_json_dict(),
            "last_processed_raw_event_time": (None if self.last_processed_raw_event_time is None else serialize_timestamp(self.last_processed_raw_event_time, field_name="last_processed_raw_event_time")),
            "carry_window_size": self.carry_window_size, "resulting_feature_dataset_id": self.resulting_feature_dataset_id,
            "semantic_digest": self.semantic_digest, "checkpoint_time": serialize_timestamp(self.checkpoint_time, field_name="checkpoint_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["checkpoint_id"]
        del payload["checkpoint_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureGenerationCheckpoint:
        from quant_platform.ml.persistence import as_json_dict

        raw_last_time = raw.get("last_processed_raw_event_time")
        raw_carry_window_size = raw.get("carry_window_size")
        return cls(
            checkpoint_id=str(raw["checkpoint_id"]), raw_dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["raw_dataset_key"], field_name="raw_dataset_key")),
            raw_dataset_id=str(raw["raw_dataset_id"]), feature_dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["feature_dataset_key"], field_name="feature_dataset_key")),
            last_processed_raw_event_time=(None if raw_last_time is None else deserialize_timestamp(raw_last_time, field_name="last_processed_raw_event_time")),
            carry_window_size=(None if raw_carry_window_size is None else int(str(raw_carry_window_size))),
            resulting_feature_dataset_id=(None if raw.get("resulting_feature_dataset_id") is None else str(raw["resulting_feature_dataset_id"])),
            semantic_digest=str(raw["semantic_digest"]), checkpoint_time=deserialize_timestamp(raw["checkpoint_time"], field_name="checkpoint_time"),
        )


def create_feature_generation_checkpoint(
    *, raw_dataset_key: DatasetKey, raw_dataset_id: str, feature_dataset_key: DatasetKey, last_processed_raw_event_time: datetime | None,
    carry_window_size: int | None, resulting_feature_dataset_id: str | None, semantic_digest: str, checkpoint_time: datetime,
) -> FeatureGenerationCheckpoint:
    provisional = FeatureGenerationCheckpoint(
        checkpoint_id="0" * 64, raw_dataset_key=raw_dataset_key, raw_dataset_id=raw_dataset_id, feature_dataset_key=feature_dataset_key,
        last_processed_raw_event_time=last_processed_raw_event_time, carry_window_size=carry_window_size,
        resulting_feature_dataset_id=resulting_feature_dataset_id, semantic_digest=semantic_digest, checkpoint_time=checkpoint_time,
    )
    checkpoint_id = compute_content_id(FEATURE_GENERATION_CHECKPOINT_KIND, provisional.to_identity_payload())
    return FeatureGenerationCheckpoint(
        checkpoint_id=checkpoint_id, raw_dataset_key=raw_dataset_key, raw_dataset_id=raw_dataset_id, feature_dataset_key=feature_dataset_key,
        last_processed_raw_event_time=last_processed_raw_event_time, carry_window_size=carry_window_size,
        resulting_feature_dataset_id=resulting_feature_dataset_id, semantic_digest=semantic_digest, checkpoint_time=checkpoint_time,
    )


def verify_feature_generation_checkpoint(checkpoint: FeatureGenerationCheckpoint, *, repository: MarketDataRepository) -> None:
    """Raises `CheckpointError` if forged. Raises `StaleCheckpointError`
    if `checkpoint.raw_dataset_id` no longer matches the raw dataset's
    CURRENT manifest version (the raw data has moved on -- ahead of what
    this checkpoint was computed against) or if
    `checkpoint.resulting_feature_dataset_id` does not match the feature
    dataset's current manifest (the feature dataset has moved on
    independently, behind or ahead of what this checkpoint recorded)."""
    recomputed_id = compute_content_id(FEATURE_GENERATION_CHECKPOINT_KIND, checkpoint.to_identity_payload())
    if recomputed_id != checkpoint.checkpoint_id:
        raise CheckpointError(f"FeatureGenerationCheckpoint {checkpoint.checkpoint_id!r} does not reproduce its own id -- forged or tampered")
    current_raw_manifest = repository.manifest_store.read_current(checkpoint.raw_dataset_key)
    current_raw_id = current_raw_manifest.dataset_id if current_raw_manifest is not None else None
    if current_raw_id != checkpoint.raw_dataset_id:
        raise StaleCheckpointError(
            f"FeatureGenerationCheckpoint claims raw_dataset_id={checkpoint.raw_dataset_id!r}, repository's raw dataset is currently at {current_raw_id!r}"
        )
    current_feature_manifest = repository.manifest_store.read_current(checkpoint.feature_dataset_key)
    current_feature_id = current_feature_manifest.dataset_id if current_feature_manifest is not None else None
    if current_feature_id != checkpoint.resulting_feature_dataset_id:
        raise StaleCheckpointError(
            f"FeatureGenerationCheckpoint claims resulting_feature_dataset_id={checkpoint.resulting_feature_dataset_id!r}, "
            f"repository's feature dataset is currently at {current_feature_id!r}"
        )


# --------------------------------------------------------------------------
# Durable storage -- one append-only history per dataset_key, holding
# EITHER checkpoint kind (discriminated by the stored "kind" field).
# --------------------------------------------------------------------------
Checkpoint = RawIngestionCheckpoint | FeatureGenerationCheckpoint


@contextmanager
def _checkpoint_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire checkpoint store lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Checkpoint store lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class CheckpointStore:
    """Storage layout: `{storage_root}/repository/checkpoints/
    {dataset_key_path}/checkpoints.jsonl`. Append-only, full history
    retained; `read_current` returns the latest entry. Appending a
    checkpoint identical to the current latest is an idempotent no-op."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "checkpoints" / Path(*dataset_key.storage_path_parts())

    def _checkpoints_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "checkpoints.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".checkpoints.lock"

    def read_history(self, dataset_key: DatasetKey) -> list[Checkpoint]:
        path = self._checkpoints_path(dataset_key)
        if not path.is_file():
            return []
        checkpoints: list[Checkpoint] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted checkpoint line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted checkpoint line for dataset {dataset_key!r}: expected a JSON object")
            kind = raw.get("kind")
            if kind == RAW_INGESTION_CHECKPOINT_KIND:
                checkpoints.append(RawIngestionCheckpoint.from_json_dict(raw))
            elif kind == FEATURE_GENERATION_CHECKPOINT_KIND:
                checkpoints.append(FeatureGenerationCheckpoint.from_json_dict(raw))
            else:
                raise MarketDataPersistenceError(f"Corrupted checkpoint line for dataset {dataset_key!r}: unknown kind {kind!r}")
        return checkpoints

    def read_current(self, dataset_key: DatasetKey) -> Checkpoint | None:
        history = self.read_history(dataset_key)
        return history[-1] if history else None

    def append(self, dataset_key: DatasetKey, checkpoint: Checkpoint) -> Checkpoint:
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _checkpoint_store_lock(lock_path):
            current = self.read_current(dataset_key)
            if current is not None and current.checkpoint_id == checkpoint.checkpoint_id:
                return current
            path = self._checkpoints_path(dataset_key)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(checkpoint.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return checkpoint
