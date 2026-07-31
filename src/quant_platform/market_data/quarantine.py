"""Append-only quarantine evidence for rejected source rows (Milestone
10, Phase 3). A `QuarantineRecord` stores the SAFE, already-text-only
`RawSourceRecord.raw_fields` (never raw bytes, never an arbitrary blob --
every adapter in this package already restricts a raw record to string
fields; quarantine simply persists that same safe shape, per the
specification's own "do not store secrets or arbitrary unsafe binary
blobs") alongside the stable, machine-readable issue code(s) that caused
rejection.

Keyed by the PHYSICAL `(source_manifest_id, source_row_index)`
coordinate -- not by content digest -- so "conflicting evidence under the
same identity" is actually reachable: if the same physical row is
re-quarantined with DIFFERENT evidence (different digest or different
issue codes) without a new source manifest, that is a genuine
inconsistency (`SourceQuarantineError`, fails closed), whereas an EXACT
repeat quarantine (identical evidence) is idempotently absorbed. This
mirrors `feature_store.FeatureStore.append`'s own coordinate-conflict
pattern from Phase 1/2, and `provenance.ProvenanceStore`'s identical
shape one module over. A later CORRECTED source file is a NEW
`SourceManifest` (new `source_manifest_id`, since its `content_digest`
changed) -- so it produces entirely new quarantine evidence under a new
coordinate namespace; this store never mutates or removes an existing
entry."""

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
    MarketDataLockError,
    MarketDataPersistenceError,
    SourceQuarantineError,
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
    "AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME",
    "CONFLICTING_SOURCE_SEQUENCE",
    "DUPLICATE_SOURCE_RECORD_DIGEST",
    "DUPLICATE_SOURCE_ROW_COORDINATE",
    "EMPTY_TIMESTAMP",
    "EXTRA_FORBIDDEN_COLUMN",
    "FUTURE_TIMESTAMP",
    "INVALID_DECIMAL",
    "INVALID_OHLC",
    "MALFORMED_TIMESTAMP",
    "MISSING_OBSERVATION_VALUE",
    "MISSING_REQUIRED_COLUMN",
    "NAIVE_TIMESTAMP_WITHOUT_POLICY",
    "NEGATIVE_VOLUME",
    "NON_FINITE_DECIMAL",
    "QUARANTINE_RECORD_KIND",
    "TIMESTAMP_OUTSIDE_DECLARED_RANGE",
    "UNKNOWN_SYMBOL",
    "UNKNOWN_TIMEFRAME",
    "VALIDATION_ISSUE_CODES",
    "QuarantineRecord",
    "QuarantineStore",
    "RetryEligibility",
    "create_quarantine_record",
    "default_retry_eligibility",
]

QUARANTINE_RECORD_KIND = "quarantine_record"

# --------------------------------------------------------------------------
# Stable, machine-readable row-validation issue codes -- the shared
# vocabulary `orchestration.py`'s row validator applies to a specific
# parsed row; centralized here since quarantine evidence is the durable
# artifact that names them (see module docstring). Never derived from
# free-text/human-readable log messages.
# --------------------------------------------------------------------------
MISSING_REQUIRED_COLUMN = "missing_required_column"
EXTRA_FORBIDDEN_COLUMN = "extra_forbidden_column"
EMPTY_TIMESTAMP = "empty_timestamp"
MALFORMED_TIMESTAMP = "malformed_timestamp"
NAIVE_TIMESTAMP_WITHOUT_POLICY = "naive_timestamp_without_policy"
AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME = "ambiguous_or_nonexistent_local_time"
INVALID_DECIMAL = "invalid_decimal"
NON_FINITE_DECIMAL = "non_finite_decimal"
NEGATIVE_VOLUME = "negative_volume"
INVALID_OHLC = "invalid_ohlc"
UNKNOWN_SYMBOL = "unknown_symbol"
UNKNOWN_TIMEFRAME = "unknown_timeframe"
DUPLICATE_SOURCE_ROW_COORDINATE = "duplicate_source_row_coordinate"
DUPLICATE_SOURCE_RECORD_DIGEST = "duplicate_source_record_digest"
CONFLICTING_SOURCE_SEQUENCE = "conflicting_source_sequence"
FUTURE_TIMESTAMP = "future_timestamp"
TIMESTAMP_OUTSIDE_DECLARED_RANGE = "timestamp_outside_declared_range"
MISSING_OBSERVATION_VALUE = "missing_observation_value"
"""Milestone 10, Phase 4A: the source EXPLICITLY declared "no value
published for this coordinate" (e.g. FRED's own `"."` convention for a
macro observation) -- a distinct concept from `INVALID_DECIMAL` (the
value was PRESENT but malformed): the source is not malfunctioning here,
it is faithfully reporting an absence. `macro.MacroEvent`'s own schema
(Phase 1) has no representation for "a committed observation with no
value," so this is quarantine's own STRUCTURED representation of a
missing observation, per the specification's own instruction never to
silently coerce a missing value to zero."""

VALIDATION_ISSUE_CODES: tuple[str, ...] = (
    MISSING_REQUIRED_COLUMN, EXTRA_FORBIDDEN_COLUMN, EMPTY_TIMESTAMP, MALFORMED_TIMESTAMP, NAIVE_TIMESTAMP_WITHOUT_POLICY,
    AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME, INVALID_DECIMAL, NON_FINITE_DECIMAL, NEGATIVE_VOLUME, INVALID_OHLC, UNKNOWN_SYMBOL,
    UNKNOWN_TIMEFRAME, DUPLICATE_SOURCE_ROW_COORDINATE, DUPLICATE_SOURCE_RECORD_DIGEST, CONFLICTING_SOURCE_SEQUENCE,
    FUTURE_TIMESTAMP, TIMESTAMP_OUTSIDE_DECLARED_RANGE, MISSING_OBSERVATION_VALUE,
)

# Issues fixable by supplying CORRECTED SOURCE CONTENT under the same
# operation config vs. issues that require a CONFIG change (a new
# mapping/timezone-policy version) -- source content alone cannot fix
# these. A caller may always override this default via
# `create_quarantine_record`'s explicit `retry_eligibility` parameter.
_PERMANENT_ISSUE_CODES = frozenset({
    NAIVE_TIMESTAMP_WITHOUT_POLICY, AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME, UNKNOWN_SYMBOL, UNKNOWN_TIMEFRAME, MISSING_OBSERVATION_VALUE,
})


class RetryEligibility(Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"


def default_retry_eligibility(issue_codes: tuple[str, ...]) -> RetryEligibility:
    """`PERMANENT` if ANY code among `issue_codes` requires a config
    change; `RETRYABLE` only if every code is fixable by corrected
    source content alone."""
    if any(code in _PERMANENT_ISSUE_CODES for code in issue_codes):
        return RetryEligibility.PERMANENT
    return RetryEligibility.RETRYABLE


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    quarantine_record_id: str
    source_manifest_id: str
    source_row_index: int
    raw_record_digest: str
    raw_fields: dict[str, str]
    validation_issue_codes: tuple[str, ...]
    ingestion_batch_id: str
    retry_eligibility: RetryEligibility
    event_time: datetime
    """Caller-supplied OPERATIONAL time (the same explicit
    `operation_time` every orchestration stage requires) -- never derived
    from wall-clock, and excluded from identity below exactly like
    `creation_time` elsewhere in this package."""

    def __post_init__(self) -> None:
        require_non_empty(self.source_manifest_id, field_name="QuarantineRecord.source_manifest_id")
        if self.source_row_index < 0:
            raise SourceQuarantineError(f"QuarantineRecord.source_row_index must be >= 0, got {self.source_row_index}")
        require_non_empty(self.raw_record_digest, field_name="QuarantineRecord.raw_record_digest")
        require_non_empty(self.ingestion_batch_id, field_name="QuarantineRecord.ingestion_batch_id")
        if not self.validation_issue_codes:
            raise SourceQuarantineError("QuarantineRecord.validation_issue_codes must not be empty")
        unknown_codes = [c for c in self.validation_issue_codes if c not in VALIDATION_ISSUE_CODES]
        if unknown_codes:
            raise SourceQuarantineError(f"QuarantineRecord.validation_issue_codes has unrecognized code(s): {unknown_codes}")
        require_tz_aware(self.event_time, field_name="QuarantineRecord.event_time")

    def source_coordinate(self) -> SourceRowCoordinate:
        return SourceRowCoordinate(source_manifest_id=self.source_manifest_id, row_index=self.source_row_index)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": QUARANTINE_RECORD_KIND, "quarantine_record_id": self.quarantine_record_id,
            "source_manifest_id": self.source_manifest_id, "source_row_index": self.source_row_index,
            "raw_record_digest": self.raw_record_digest, "raw_fields": dict(self.raw_fields),
            "validation_issue_codes": list(self.validation_issue_codes), "ingestion_batch_id": self.ingestion_batch_id,
            "retry_eligibility": self.retry_eligibility.value, "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Excludes `ingestion_batch_id` (operational: WHICH operation
        most recently rediscovered this row -- exactly like
        `creation_time` elsewhere in this package) in addition to
        `quarantine_record_id`/`event_time`. This is deliberate: the SAME
        physical source row, rejected for the SAME reason, is ONE piece
        of evidence regardless of how many independent operations
        (different `operation_id`s re-reading the same unmodified
        source) rediscover it -- excluding the batch id is what makes a
        second, later operation's rediscovery of an already-known-bad
        row an idempotent no-op instead of a spurious
        `SourceQuarantineError` (an operation-scoping detail is not
        evidentiary disagreement about the row itself)."""
        payload = dict(self.to_json_dict())
        del payload["quarantine_record_id"]
        del payload["event_time"]
        del payload["ingestion_batch_id"]
        payload["validation_issue_codes"] = sorted(self.validation_issue_codes)
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QuarantineRecord:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        raw_fields_obj = as_json_dict(raw["raw_fields"], field_name="raw_fields")
        return cls(
            quarantine_record_id=str(raw["quarantine_record_id"]), source_manifest_id=str(raw["source_manifest_id"]),
            source_row_index=int(str(raw["source_row_index"])), raw_record_digest=str(raw["raw_record_digest"]),
            raw_fields={str(k): str(v) for k, v in raw_fields_obj.items()},
            validation_issue_codes=tuple(str(c) for c in as_json_list(raw["validation_issue_codes"], field_name="validation_issue_codes")),
            ingestion_batch_id=str(raw["ingestion_batch_id"]), retry_eligibility=RetryEligibility(raw["retry_eligibility"]),
            event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_quarantine_record(
    *, source_manifest_id: str, source_row_index: int, raw_record_digest: str, raw_fields: dict[str, str],
    validation_issue_codes: tuple[str, ...], ingestion_batch_id: str, event_time: datetime,
    retry_eligibility: RetryEligibility | None = None,
) -> QuarantineRecord:
    resolved_retry_eligibility = default_retry_eligibility(validation_issue_codes) if retry_eligibility is None else retry_eligibility
    provisional = QuarantineRecord(
        quarantine_record_id="0" * 64, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
        raw_record_digest=raw_record_digest, raw_fields=raw_fields, validation_issue_codes=validation_issue_codes,
        ingestion_batch_id=ingestion_batch_id, retry_eligibility=resolved_retry_eligibility, event_time=event_time,
    )
    quarantine_record_id = compute_content_id(QUARANTINE_RECORD_KIND, provisional.to_identity_payload())
    return QuarantineRecord(
        quarantine_record_id=quarantine_record_id, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
        raw_record_digest=raw_record_digest, raw_fields=raw_fields, validation_issue_codes=validation_issue_codes,
        ingestion_batch_id=ingestion_batch_id, retry_eligibility=resolved_retry_eligibility, event_time=event_time,
    )


# --------------------------------------------------------------------------
# Durable append-only store.
# --------------------------------------------------------------------------
@contextmanager
def _quarantine_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire quarantine lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Quarantine lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class QuarantineStore:
    """Append-only ledger: `{storage_root}/repository/quarantine/
    {dataset_key_path}/quarantine.jsonl`, one file per TARGET dataset
    (mirrors `provenance.ProvenanceStore`'s own layout exactly)."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "quarantine" / Path(*dataset_key.storage_path_parts())

    def _quarantine_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "quarantine.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".quarantine.lock"

    def read_all(self, dataset_key: DatasetKey) -> list[QuarantineRecord]:
        path = self._quarantine_path(dataset_key)
        if not path.is_file():
            return []
        records: list[QuarantineRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted quarantine line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted quarantine line for dataset {dataset_key!r}: expected a JSON object")
            records.append(QuarantineRecord.from_json_dict(raw))
        return records

    def read_by_source_coordinate(self, dataset_key: DatasetKey, coordinate: SourceRowCoordinate) -> QuarantineRecord | None:
        matching = [
            r for r in self.read_all(dataset_key)
            if r.source_manifest_id == coordinate.source_manifest_id and r.source_row_index == coordinate.row_index
        ]
        return matching[-1] if matching else None

    def is_quarantined(self, dataset_key: DatasetKey, coordinate: SourceRowCoordinate) -> bool:
        return self.read_by_source_coordinate(dataset_key, coordinate) is not None

    def _append_line(self, dataset_key: DatasetKey, record: QuarantineRecord) -> None:
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        path = self._quarantine_path(dataset_key)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, dataset_key: DatasetKey, record: QuarantineRecord) -> QuarantineRecord:
        """Idempotent for an EXACT repeat quarantine (same source
        coordinate, same `quarantine_record_id`); raises
        `SourceQuarantineError` for CONFLICTING evidence under the same
        coordinate (same row, different digest or issue codes)."""
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _quarantine_store_lock(lock_path):
            existing = self.read_by_source_coordinate(dataset_key, record.source_coordinate())
            if existing is not None:
                if existing.quarantine_record_id == record.quarantine_record_id:
                    return existing
                raise SourceQuarantineError(
                    f"source row (source_manifest_id={record.source_manifest_id!r}, row_index={record.source_row_index}) is already "
                    f"quarantined under quarantine_record_id {existing.quarantine_record_id!r} with issue codes "
                    f"{list(existing.validation_issue_codes)!r}; conflicting evidence "
                    f"(quarantine_record_id={record.quarantine_record_id!r}, issue codes {list(record.validation_issue_codes)!r}) was submitted"
                )
            self._append_line(dataset_key, record)
            return record
