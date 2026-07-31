"""Deterministic, append-only feature storage (Milestone 10, Phase 1).

A `FeatureRecord` is keyed by its economic coordinate --
`(feature_name, feature_version, instrument_id, timeframe, timestamp)` --
not by physical append position. `FeatureStore.append` therefore checks
for a conflict by looking up that coordinate directly rather than
assuming records arrive in strict chronological order (unlike `events.
MarketEventStore`'s positional-sequence check): feature generation may
legitimately be re-run over an overlapping window (e.g. after a
gap-fill), and as long as it reproduces the SAME value at each
coordinate every time (Phase 1's core determinism guarantee), that
re-run must be an idempotent no-op rather than a spurious "gap" or
"conflict" error. A DIFFERENT value at an already-recorded coordinate is
never silently accepted -- deterministic generation means that can only
happen if something upstream changed without a new `feature_version`,
which is exactly the kind of silent corruption `FeatureStoreError` exists
to surface.

`FeatureStore.read_records` always returns records sorted by
`(timestamp, feature_id)` -- independent of physical append order -- so
two callers who generated the same feature set in a different order (or
re-ran a partial backfill) still observe an identical, deterministically
ordered result (Milestone 10's "identical ordering" replay requirement)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    FeatureStoreError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.core.types import Timeframe
from quant_platform.market_data.identity import (
    compute_content_id,
    decimal_to_json,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import as_json_dict, parse_json_strict

__all__ = ["FEATURE_RECORD_KIND", "FeatureRecord", "FeatureStore", "create_feature_record"]

FEATURE_RECORD_KIND = "feature_record"


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    feature_id: str
    feature_name: str
    feature_version: int
    instrument_id: str
    timestamp: datetime
    timeframe: Timeframe | None
    value: Decimal
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.feature_name, field_name="FeatureRecord.feature_name")
        if self.feature_version < 1:
            raise FeatureStoreError(f"FeatureRecord.feature_version must be >= 1, got {self.feature_version}")
        require_non_empty(self.instrument_id, field_name="FeatureRecord.instrument_id")
        require_tz_aware(self.timestamp, field_name="FeatureRecord.timestamp")
        if not self.value.is_finite():
            raise FeatureStoreError(f"FeatureRecord.value must be finite, got {self.value!r}")
        for key in self.metadata:
            if not isinstance(key, str):
                raise FeatureStoreError(f"FeatureRecord.metadata keys must be strings, got {type(key).__name__}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": FEATURE_RECORD_KIND, "feature_id": self.feature_id, "feature_name": self.feature_name,
            "feature_version": self.feature_version, "instrument_id": self.instrument_id,
            "timestamp": serialize_timestamp(self.timestamp, field_name="timestamp"),
            "timeframe": (None if self.timeframe is None else self.timeframe.value), "value": decimal_to_json(self.value),
            "metadata": dict(sorted(self.metadata.items())),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["feature_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureRecord:
        raw_timeframe = raw.get("timeframe")
        return cls(
            feature_id=str(raw["feature_id"]), feature_name=str(raw["feature_name"]), feature_version=int(str(raw["feature_version"])),
            instrument_id=str(raw["instrument_id"]), timestamp=deserialize_timestamp(raw["timestamp"], field_name="timestamp"),
            timeframe=(None if raw_timeframe is None else Timeframe(raw_timeframe)),
            value=parse_decimal(raw["value"], field_name="value"),
            metadata=dict(as_json_dict(raw.get("metadata") or {}, field_name="metadata")),
        )


def create_feature_record(
    *, feature_name: str, feature_version: int, instrument_id: str, timestamp: datetime, timeframe: Timeframe | None, value: Decimal,
    metadata: dict[str, object] | None = None,
) -> FeatureRecord:
    resolved_metadata = {} if metadata is None else metadata
    provisional = FeatureRecord(
        feature_id="0" * 64, feature_name=feature_name, feature_version=feature_version, instrument_id=instrument_id,
        timestamp=timestamp, timeframe=timeframe, value=value, metadata=resolved_metadata,
    )
    feature_id = compute_content_id(FEATURE_RECORD_KIND, provisional.to_identity_payload())
    return FeatureRecord(
        feature_id=feature_id, feature_name=feature_name, feature_version=feature_version, instrument_id=instrument_id,
        timestamp=timestamp, timeframe=timeframe, value=value, metadata=resolved_metadata,
    )


@contextmanager
def feature_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire feature store lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Feature store lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


def _partition_key(record: FeatureRecord) -> tuple[str, int, str]:
    return record.feature_name, record.feature_version, record.instrument_id


class FeatureStore:
    """Storage layout: `{storage_root}/features/{feature_name}/
    v{feature_version}/{instrument_id}/records.jsonl`."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _partition_dir(self, feature_name: str, feature_version: int, instrument_id: str) -> Path:
        return self._root / "features" / feature_name / f"v{feature_version}" / instrument_id

    def _records_path(self, feature_name: str, feature_version: int, instrument_id: str) -> Path:
        return self._partition_dir(feature_name, feature_version, instrument_id) / "records.jsonl"

    def _lock_path(self, feature_name: str, feature_version: int, instrument_id: str) -> Path:
        return self._partition_dir(feature_name, feature_version, instrument_id) / ".records.lock"

    def _read_raw(self, feature_name: str, feature_version: int, instrument_id: str) -> list[FeatureRecord]:
        """Records in physical append order (undeduplicated, unsorted) --
        used internally by `append` for its coordinate-conflict check;
        see `read_records` for the deterministically sorted public view."""
        path = self._records_path(feature_name, feature_version, instrument_id)
        if not path.is_file():
            return []
        records: list[FeatureRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted feature record line for {feature_name}/v{feature_version}/{instrument_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted feature record line for {feature_name}/v{feature_version}/{instrument_id}: expected a JSON object")
            records.append(FeatureRecord.from_json_dict(raw))
        return records

    def read_records(self, feature_name: str, feature_version: int, instrument_id: str) -> list[FeatureRecord]:
        records = self._read_raw(feature_name, feature_version, instrument_id)
        return sorted(records, key=lambda r: (r.timestamp, r.feature_id))

    def append(self, record: FeatureRecord) -> FeatureRecord:
        feature_name, feature_version, instrument_id = _partition_key(record)
        lock_path = self._lock_path(feature_name, feature_version, instrument_id)
        self._partition_dir(feature_name, feature_version, instrument_id).mkdir(parents=True, exist_ok=True)
        with feature_store_lock(lock_path):
            existing_records = self._read_raw(feature_name, feature_version, instrument_id)
            for existing in existing_records:
                if existing.timestamp != record.timestamp:
                    continue
                if existing.feature_id == record.feature_id:
                    return existing  # idempotent no-op: identical re-append
                raise FeatureStoreError(
                    f"Conflicting feature value at {feature_name}/v{feature_version}/{instrument_id}/{record.timestamp}: "
                    f"existing feature_id {existing.feature_id!r} != new feature_id {record.feature_id!r}"
                )
            path = self._records_path(feature_name, feature_version, instrument_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(record.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record
