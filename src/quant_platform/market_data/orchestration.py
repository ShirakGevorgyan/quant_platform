"""Deterministic historical-ingestion orchestration (Milestone 10, Phase
3) -- the layer that OWNS normalization, mapping resolution, row
validation, canonical event construction, repository ingestion,
provenance, checkpointing, and reporting; an adapter never does any of
this (see `adapters.py`'s own docstring).

THE SMALLEST HONEST STAGE MACHINE: `IngestionStage` names eleven stages
(`SOURCE_VERIFIED` through `COMPLETED`); `OperationRecord`/`OperationStore`
give each one durable, independently-readable evidence via a SINGLE
uniform mechanism -- `OperationStore.advance` -- rather than eleven
bespoke stores. Every operation is identified by a caller-supplied
`operation_id` bound, at first use, to a `content_digest` computed from
everything that defines what the operation IS (source manifest, backfill
plan, mapping/normalization spec ids, row-failure policy, target
dataset): an EXACT retry (same `operation_id`, same `content_digest`,
same stage, same evidence) is idempotently absorbed; a CONFLICTING retry
(same `operation_id`, anything about the inputs or a stage's outcome
different) raises `OrchestrationConflictError`; an illegal transition
(skipping a stage, regressing) raises `OrchestrationStateError`. No
completed status is reachable before `REPOSITORY_COMMITTED` and
`PROVENANCE_COMMITTED` both durably exist, because `VERIFIED` (the stage
immediately before `COMPLETED`) explicitly re-reads both and cross-checks
them (see `_verify_repository_provenance_agreement`).

NO IN-MEMORY-ONLY CORRECTNESS STATE: `run_ingestion_operation` re-derives
EVERYTHING -- parsed rows, validation outcomes, normalized events -- fresh
from the adapter and specs on every call, rather than trusting anything
carried only in a Python variable across a crash. The one piece of state
that genuinely cannot be recomputed identically from the repository's
current condition alone -- the REPOSITORY APPEND SEQUENCE this
operation's events must use -- is durably pinned the first time
`BATCH_RESERVED` is recorded (`_resolve_sequence_start`) and always
reused verbatim on every subsequent call for the same `operation_id`,
specifically so a resumed operation assigns the SAME sequence numbers
(and therefore the SAME content-addressed `event_id`s) as the original
attempt, even though the repository's own "next sequence" may have moved
on for unrelated reasons in between. A concurrent, unrelated operation
against the SAME `dataset_key` racing this one to commit first is a real,
acknowledged limitation (it surfaces as a safe, fail-closed
`MarketDataPersistenceError`/`IngestionConflictError` at the Phase 2
commit layer -- see `docs/milestone10_phase3_delivery_report.md`'s Known
Limitations), not a silent corruption.

FAIL-FAST VS QUARANTINE: `RowFailurePolicy.FAIL_FAST` raises
`RowValidationError` on the FIRST invalid row found, before any
repository/provenance/checkpoint write for this call happens.
`RowFailurePolicy.QUARANTINE` (the default) durably quarantines every
invalid row (via `quarantine.QuarantineStore`, idempotent) and proceeds
with the remaining valid rows.

DRY RUN: `dry_run=True` performs the identical parse/validate/normalize
computation (so its report is genuinely accurate) but writes NOTHING --
no operation ledger entry, no quarantine record, no repository event, no
provenance record, no checkpoint. Its `resulting_dataset_id` PREVIEW is
computed by reusing `ingestion.semantic_digest_for_raw_events`/
`physical_digest_for_ids`/`manifests.create_dataset_manifest` (all pure)
against the union of currently-durable events (read-only) and this
call's newly-normalized events, and `partitions.build_partition` (also
pure) for any partition the new events would touch -- an untouched
partition's existing `partition_id` is read, never recomputed -- so the
preview is the EXACT id a real commit would produce, never an
approximation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    HistoricalIngestionError,
    InstrumentMappingError,
    MarketDataLockError,
    MarketDataPersistenceError,
    OrchestrationConflictError,
    OrchestrationError,
    OrchestrationStateError,
    ProvenanceError,
    RowValidationError,
    TimeframeMappingError,
    TimezoneError,
)
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.core.types import OrderSide, Timeframe
from quant_platform.market_data.adapters import HistoricalSourceAdapter, RawSourceRecord, SourceRowCoordinate
from quant_platform.market_data.backfill import BackfillPlan
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.checkpoints import (
    CheckpointStore,
    compute_raw_ingestion_checkpoint,
    create_historical_ingestion_checkpoint,
)
from quant_platform.market_data.events import (
    MarketDataEvent,
    create_quote,
    create_trade,
    market_data_event_id,
    market_data_event_time,
)
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.ingestion import (
    ingest_raw_events,
    next_sequence_for,
    physical_digest_for_ids,
    semantic_digest_for_raw_events,
)
from quant_platform.market_data.jsonl_adapter import schema_for_record_kind
from quant_platform.market_data.manifests import DatasetKey, DatasetKind, create_dataset_manifest
from quant_platform.market_data.mappings import (
    InstrumentMappingSpec,
    TimeframeMappingSpec,
    resolve_instrument_id,
    resolve_timeframe,
)
from quant_platform.market_data.partitions import build_partition, partition_key_for
from quant_platform.market_data.provenance import ProvenanceStore, create_provenance_record
from quant_platform.market_data.quarantine import (
    AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME,
    CONFLICTING_SOURCE_SEQUENCE,
    DUPLICATE_SOURCE_RECORD_DIGEST,
    DUPLICATE_SOURCE_ROW_COORDINATE,
    EMPTY_TIMESTAMP,
    EXTRA_FORBIDDEN_COLUMN,
    FUTURE_TIMESTAMP,
    INVALID_DECIMAL,
    INVALID_OHLC,
    MALFORMED_TIMESTAMP,
    MISSING_REQUIRED_COLUMN,
    NAIVE_TIMESTAMP_WITHOUT_POLICY,
    NEGATIVE_VOLUME,
    NON_FINITE_DECIMAL,
    TIMESTAMP_OUTSIDE_DECLARED_RANGE,
    UNKNOWN_SYMBOL,
    UNKNOWN_TIMEFRAME,
    QuarantineStore,
    create_quarantine_record,
)
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import (
    RecordKind,
    SourceManifest,
    compute_timestamp_policy_id,
)
from quant_platform.market_data.source_normalization import (
    TimestampParsingPolicy,
    normalize_signed_zero,
    parse_source_timestamp,
)
from quant_platform.market_data.ticks import create_tick
from quant_platform.ml.concurrency import experiment_lock

__all__ = [
    "OPERATION_RECORD_KIND",
    "IngestionOperationReport",
    "IngestionStage",
    "OperationRecord",
    "OperationStore",
    "RowFailurePolicy",
    "replay_ingestion_operation",
    "run_ingestion_operation",
]

OPERATION_RECORD_KIND = "historical_ingestion_operation_record"


class IngestionStage(Enum):
    SOURCE_VERIFIED = "source_verified"
    PLAN_CREATED = "plan_created"
    BATCH_RESERVED = "batch_reserved"
    ROWS_PARSED = "rows_parsed"
    ROWS_VALIDATED = "rows_validated"
    EVENTS_NORMALIZED = "events_normalized"
    REPOSITORY_COMMITTED = "repository_committed"
    PROVENANCE_COMMITTED = "provenance_committed"
    CHECKPOINT_COMMITTED = "checkpoint_committed"
    VERIFIED = "verified"
    COMPLETED = "completed"


_STAGE_ORDER: tuple[IngestionStage, ...] = tuple(IngestionStage)
_STAGE_RANK: dict[IngestionStage, int] = {stage: rank for rank, stage in enumerate(_STAGE_ORDER)}


class RowFailurePolicy(Enum):
    QUARANTINE = "quarantine"
    FAIL_FAST = "fail_fast"


# --------------------------------------------------------------------------
# Durable operation ledger.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    dataset_key: DatasetKey
    stage: IngestionStage
    content_digest: str
    stage_evidence: dict[str, object]
    operation_time: datetime

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": OPERATION_RECORD_KIND, "operation_id": self.operation_id, "dataset_key": self.dataset_key.to_json_dict(),
            "stage": self.stage.value, "content_digest": self.content_digest, "stage_evidence": dict(self.stage_evidence),
            "operation_time": serialize_timestamp(self.operation_time, field_name="operation_time"),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OperationRecord:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            operation_id=str(raw["operation_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            stage=IngestionStage(raw["stage"]), content_digest=str(raw["content_digest"]),
            stage_evidence=as_json_dict(raw["stage_evidence"], field_name="stage_evidence"),
            operation_time=deserialize_timestamp(raw["operation_time"], field_name="operation_time"),
        )


@contextmanager
def _operation_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire operation ledger lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Operation ledger lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class OperationStore:
    """Append-only ledger: `{storage_root}/repository/operations/
    {dataset_key_path}/operations.jsonl`. `advance` is the ONLY mutator
    -- see module docstring for its idempotent/conflict/illegal-transition
    semantics."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "operations" / Path(*dataset_key.storage_path_parts())

    def _operations_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "operations.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".operations.lock"

    def read_all(self, dataset_key: DatasetKey) -> list[OperationRecord]:
        path = self._operations_path(dataset_key)
        if not path.is_file():
            return []
        records: list[OperationRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted operation ledger line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted operation ledger line for dataset {dataset_key!r}: expected a JSON object")
            records.append(OperationRecord.from_json_dict(raw))
        return records

    def read_latest(self, dataset_key: DatasetKey, operation_id: str) -> OperationRecord | None:
        matching = [r for r in self.read_all(dataset_key) if r.operation_id == operation_id]
        return matching[-1] if matching else None

    def _append(self, dataset_key: DatasetKey, record: OperationRecord) -> None:
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        path = self._operations_path(dataset_key)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())

    def advance(
        self, *, dataset_key: DatasetKey, operation_id: str, content_digest: str, stage: IngestionStage,
        stage_evidence: dict[str, object], operation_time: datetime,
    ) -> OperationRecord:
        """Idempotent not only for a retry of the LATEST stage, but for a
        retry of ANY previously-recorded stage -- e.g. a full re-run of
        an already-`COMPLETED` operation re-plays every stage from
        `SOURCE_VERIFIED` onward, and each call must match its own
        HISTORICAL record (not the current latest, which by then is
        `COMPLETED`) to be recognized as an exact retry rather than an
        illegal regression."""
        lock_path = self._lock_path(dataset_key)
        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _operation_store_lock(lock_path):
            history = [r for r in self.read_all(dataset_key) if r.operation_id == operation_id]
            new_record = OperationRecord(
                operation_id=operation_id, dataset_key=dataset_key, stage=stage, content_digest=content_digest,
                stage_evidence=stage_evidence, operation_time=operation_time,
            )
            if not history:
                if stage is not IngestionStage.SOURCE_VERIFIED:
                    raise OrchestrationStateError(f"first stage recorded for a new operation_id must be SOURCE_VERIFIED, got {stage.value!r}")
                self._append(dataset_key, new_record)
                return new_record
            if history[0].content_digest != content_digest:
                raise OrchestrationConflictError(
                    f"operation_id {operation_id!r} is already bound to content_digest {history[0].content_digest!r}; "
                    f"a conflicting content_digest {content_digest!r} was submitted"
                )
            latest_rank = _STAGE_RANK[history[-1].stage]
            new_rank = _STAGE_RANK[stage]
            if new_rank <= latest_rank:
                recorded = next(r for r in reversed(history) if r.stage is stage)
                if recorded.stage_evidence == stage_evidence:
                    return recorded
                raise OrchestrationConflictError(
                    f"operation_id {operation_id!r} stage {stage.value!r} is already durably recorded with different evidence"
                )
            if new_rank != latest_rank + 1:
                raise OrchestrationStateError(
                    f"operation_id {operation_id!r} cannot advance from {history[-1].stage.value!r} to {stage.value!r} "
                    "(stages must advance exactly one step at a time, never skip or regress)"
                )
            self._append(dataset_key, new_record)
            return new_record


def _resolve_sequence_start(operation_store: OperationStore, repository: MarketDataRepository, dataset_key: DatasetKey, operation_id: str) -> int:
    """Durably pins the FIRST-observed repository append sequence for
    this `operation_id` -- see module docstring's "NO IN-MEMORY-ONLY
    CORRECTNESS STATE" section for why this one value cannot simply be
    recomputed fresh on every call."""
    for record in operation_store.read_all(dataset_key):
        if record.operation_id == operation_id and record.stage is IngestionStage.BATCH_RESERVED:
            return int(str(record.stage_evidence["sequence_start"]))
    return next_sequence_for(repository, dataset_key)


# --------------------------------------------------------------------------
# Row validation, normalization, and event construction.
# --------------------------------------------------------------------------
_DECIMAL_FIELDS_BY_KIND: dict[RecordKind, tuple[str, ...]] = {
    RecordKind.CANDLE: ("open", "high", "low", "close", "volume"),
    RecordKind.TICK: ("price", "volume"),
    RecordKind.QUOTE: ("bid", "ask", "bid_size", "ask_size"),
    RecordKind.TRADE: ("price", "size"),
}


def _expected_row_fields(record_kind: RecordKind) -> tuple[frozenset[str], frozenset[str]]:
    """Reuses `jsonl_adapter.schema_for_record_kind`'s table (the same
    canonical field vocabulary CSV/JSONL/in-memory adapters all target)
    minus `"kind"`, which is a JSONL-envelope-only marker never present
    in a CSV or in-memory `RawSourceRecord.raw_fields`. Re-validating
    row shape here (rather than trusting the adapter) is NOT redundant:
    `InMemorySourceAdapter` performs no schema enforcement of its own,
    and `read_csv_candle_adapter(strict_columns=False)` deliberately lets
    extra columns through -- both reach this check.

    `"timeframe"` is always treated as OPTIONAL here (even though
    JSONL's own per-line schema requires it): a CSV candle file has no
    per-row timeframe column at all -- one file always shares a single
    timeframe, declared once on `SourceManifest.expected_timeframe` --
    so `_process_row` falls back to `expected_timeframe` whenever a row
    supplies no `"timeframe"` field, and only a JSONL row that supplies
    one explicitly is resolved through `timeframe_mapping`."""
    required, optional = schema_for_record_kind(record_kind)
    required_set = frozenset(f for f in required if f not in ("kind", "timeframe"))
    optional_set = frozenset(optional) | {"timeframe"}
    return required_set, optional_set


def _classify_decimal(raw: str) -> tuple[Decimal | None, str | None]:
    stripped = raw.strip()
    if "," in stripped:
        return None, INVALID_DECIMAL
    try:
        value = Decimal(stripped)
    except InvalidOperation:
        return None, INVALID_DECIMAL
    if not value.is_finite():
        return None, NON_FINITE_DECIMAL
    return normalize_signed_zero(value), None


def _detect_batch_level_issues(records: tuple[RawSourceRecord, ...]) -> dict[int, list[str]]:
    """Cross-row checks that no single record can determine on its own:
    a repeated `row_index` (structurally should never happen through any
    real adapter -- every adapter assigns it via `enumerate()` -- kept as
    a defensive check reachable only via a hand-constructed pathological
    record list), a repeated `record_digest()` (an exact duplicate row
    within the same batch), and a source-declared `sequence` value
    claimed by two rows with DIFFERENT content."""
    issues: dict[int, list[str]] = {}
    seen_indices: set[int] = set()
    seen_digests: dict[str, int] = {}
    seen_sequences: dict[str, str] = {}
    for record in records:
        if record.row_index in seen_indices:
            issues.setdefault(record.row_index, []).append(DUPLICATE_SOURCE_ROW_COORDINATE)
        seen_indices.add(record.row_index)

        digest = record.record_digest()
        if digest in seen_digests:
            issues.setdefault(record.row_index, []).append(DUPLICATE_SOURCE_RECORD_DIGEST)
        else:
            seen_digests[digest] = record.row_index

        sequence_text = record.raw_fields.get("sequence")
        if sequence_text is not None:
            prior_digest = seen_sequences.get(sequence_text)
            if prior_digest is not None:
                if prior_digest != digest:
                    issues.setdefault(record.row_index, []).append(CONFLICTING_SOURCE_SEQUENCE)
            else:
                seen_sequences[sequence_text] = digest
    return issues


@dataclass(frozen=True, slots=True)
class _RowOutcome:
    record: RawSourceRecord
    issue_codes: tuple[str, ...]
    event: MarketDataEvent | None
    original_timestamp_text: str
    normalized_event_time: datetime | None
    resolved_instrument_id: str | None


def _process_row(
    record: RawSourceRecord, *, record_kind: RecordKind, required_fields: frozenset[str], optional_fields: frozenset[str],
    timestamp_policy: TimestampParsingPolicy, instrument_mapping: InstrumentMappingSpec, timeframe_mapping: TimeframeMappingSpec | None,
    default_provider: str, instrument_id_for_dataset: str, reference_time: datetime | None, expected_start: datetime | None,
    expected_end: datetime | None, expected_timeframe: Timeframe | None, sequence_provider: Callable[[], int],
) -> _RowOutcome:
    issue_codes: list[str] = []
    fields = record.raw_fields
    present = set(fields.keys())

    missing = required_fields - present
    if missing:
        issue_codes.append(MISSING_REQUIRED_COLUMN)
    extra = present - (required_fields | optional_fields)
    if extra:
        issue_codes.append(EXTRA_FORBIDDEN_COLUMN)
    if issue_codes:
        return _RowOutcome(
            record=record, issue_codes=tuple(issue_codes), event=None, original_timestamp_text=fields.get("timestamp", ""),
            normalized_event_time=None, resolved_instrument_id=None,
        )

    raw_timestamp = fields.get("timestamp", "")
    if not raw_timestamp.strip():
        issue_codes.append(EMPTY_TIMESTAMP)

    normalized_event_time: datetime | None = None
    if not issue_codes:
        try:
            normalized_event_time = parse_source_timestamp(raw_timestamp, policy=timestamp_policy)
        except TimezoneError:
            issue_codes.append(NAIVE_TIMESTAMP_WITHOUT_POLICY if timestamp_policy.source_timezone is None else AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME)
        except HistoricalIngestionError:
            issue_codes.append(MALFORMED_TIMESTAMP)

    if normalized_event_time is not None:
        if reference_time is not None and normalized_event_time > reference_time:
            issue_codes.append(FUTURE_TIMESTAMP)
        if expected_start is not None and normalized_event_time < expected_start:
            issue_codes.append(TIMESTAMP_OUTSIDE_DECLARED_RANGE)
        if expected_end is not None and normalized_event_time >= expected_end:
            issue_codes.append(TIMESTAMP_OUTSIDE_DECLARED_RANGE)

    decimals: dict[str, Decimal] = {}
    for field_name in _DECIMAL_FIELDS_BY_KIND[record_kind]:
        raw_value = fields.get(field_name)
        if raw_value is None:
            continue
        value, code = _classify_decimal(raw_value)
        if code is not None:
            issue_codes.append(code)
        else:
            assert value is not None
            decimals[field_name] = value

    # Positivity/ordering checks every event constructor (`candles.py`/
    # `ticks.py`/`events.py`) would otherwise raise `MarketDataEventError`
    # for -- pre-checked here so a bad row is quarantined, never an
    # unhandled exception. `NEGATIVE_VOLUME`/`INVALID_OHLC` are reused
    # for TICK/QUOTE/TRADE's own positivity/ordering rules; the
    # specification's 17-code vocabulary has no dedicated code for these,
    # and both are the closest existing semantic match.
    if record_kind is RecordKind.CANDLE:
        volume = decimals.get("volume")
        if volume is not None and volume < 0:
            issue_codes.append(NEGATIVE_VOLUME)
        if all(k in decimals for k in ("open", "high", "low", "close")):
            o, h, low_, c = decimals["open"], decimals["high"], decimals["low"], decimals["close"]
            if h < low_ or not (low_ <= o <= h) or not (low_ <= c <= h):
                issue_codes.append(INVALID_OHLC)
    elif record_kind is RecordKind.TICK:
        volume = decimals.get("volume")
        if volume is not None and volume < 0:
            issue_codes.append(NEGATIVE_VOLUME)
        price = decimals.get("price")
        if price is not None and price <= 0:
            issue_codes.append(NEGATIVE_VOLUME)
    elif record_kind is RecordKind.QUOTE:
        bid = decimals.get("bid")
        ask = decimals.get("ask")
        if bid is not None and bid <= 0:
            issue_codes.append(NEGATIVE_VOLUME)
        if ask is not None and ask <= 0:
            issue_codes.append(NEGATIVE_VOLUME)
        if bid is not None and ask is not None and ask < bid:
            issue_codes.append(INVALID_OHLC)
        for size_field in ("bid_size", "ask_size"):
            size_value = decimals.get(size_field)
            if size_value is not None and size_value < 0:
                issue_codes.append(NEGATIVE_VOLUME)
    else:
        assert record_kind is RecordKind.TRADE
        price = decimals.get("price")
        if price is not None and price <= 0:
            issue_codes.append(NEGATIVE_VOLUME)
        size_value = decimals.get("size")
        if size_value is not None and size_value <= 0:
            issue_codes.append(NEGATIVE_VOLUME)

    raw_symbol = fields.get("symbol", "")
    row_provider = fields.get("provider") or default_provider
    resolved_instrument_id: str | None = None
    if raw_symbol:
        try:
            resolved_instrument_id = resolve_instrument_id(instrument_mapping, source_symbol=raw_symbol, provider=row_provider)
        except InstrumentMappingError:
            issue_codes.append(UNKNOWN_SYMBOL)
        else:
            if resolved_instrument_id != instrument_id_for_dataset:
                issue_codes.append(UNKNOWN_SYMBOL)
    else:
        issue_codes.append(UNKNOWN_SYMBOL)

    resolved_timeframe: Timeframe | None = None
    if record_kind is RecordKind.CANDLE:
        raw_timeframe = fields.get("timeframe", "")
        if raw_timeframe:
            if timeframe_mapping is None:
                issue_codes.append(UNKNOWN_TIMEFRAME)
            else:
                try:
                    resolved_timeframe = resolve_timeframe(timeframe_mapping, source_label=raw_timeframe)
                except TimeframeMappingError:
                    issue_codes.append(UNKNOWN_TIMEFRAME)
                else:
                    if expected_timeframe is not None and resolved_timeframe is not expected_timeframe:
                        issue_codes.append(UNKNOWN_TIMEFRAME)
        elif expected_timeframe is not None:
            resolved_timeframe = expected_timeframe
        else:
            issue_codes.append(UNKNOWN_TIMEFRAME)

    if issue_codes:
        return _RowOutcome(
            record=record, issue_codes=tuple(issue_codes), event=None, original_timestamp_text=raw_timestamp,
            normalized_event_time=normalized_event_time, resolved_instrument_id=resolved_instrument_id,
        )

    assert normalized_event_time is not None and resolved_instrument_id is not None
    sequence = sequence_provider()
    event: MarketDataEvent
    if record_kind is RecordKind.CANDLE:
        assert resolved_timeframe is not None
        event = create_candle(
            instrument_id=resolved_instrument_id, provider=row_provider, symbol=raw_symbol, event_time=normalized_event_time,
            timeframe=resolved_timeframe, sequence=sequence, open=decimals["open"], high=decimals["high"], low=decimals["low"],
            close=decimals["close"], volume=decimals.get("volume"),
        )
    elif record_kind is RecordKind.TICK:
        event = create_tick(
            instrument_id=resolved_instrument_id, provider=row_provider, symbol=raw_symbol, event_time=normalized_event_time,
            sequence=sequence, price=decimals["price"], volume=decimals.get("volume"),
        )
    elif record_kind is RecordKind.QUOTE:
        event = create_quote(
            instrument_id=resolved_instrument_id, provider=row_provider, symbol=raw_symbol, event_time=normalized_event_time,
            sequence=sequence, bid=decimals["bid"], ask=decimals["ask"], bid_size=decimals.get("bid_size"), ask_size=decimals.get("ask_size"),
        )
    else:
        assert record_kind is RecordKind.TRADE
        raw_side = fields.get("side")
        side = OrderSide(raw_side) if raw_side else None
        event = create_trade(
            instrument_id=resolved_instrument_id, provider=row_provider, symbol=raw_symbol, event_time=normalized_event_time,
            sequence=sequence, price=decimals["price"], size=decimals["size"], side=side,
        )

    return _RowOutcome(
        record=record, issue_codes=(), event=event, original_timestamp_text=raw_timestamp,
        normalized_event_time=normalized_event_time, resolved_instrument_id=resolved_instrument_id,
    )


def _make_sequence_provider(start: int) -> Callable[[], int]:
    counter = [start]

    def _next() -> int:
        value = counter[0]
        counter[0] += 1
        return value

    return _next


# --------------------------------------------------------------------------
# Dry-run dataset-id preview -- pure computation plus READS only, never a
# write. See module docstring's "DRY RUN" section.
# --------------------------------------------------------------------------
def _preview_resulting_dataset_id(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, new_events: list[MarketDataEvent], backfill_plan: BackfillPlan,
    creation_time: datetime,
) -> str:
    assert dataset_key.provider is not None
    current_events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    existing_ids = {market_data_event_id(e) for e in current_events}
    preview_events = list(current_events) + [e for e in new_events if market_data_event_id(e) not in existing_ids]

    if not preview_events:
        manifest = create_dataset_manifest(
            dataset_key=dataset_key, schema_version=1, timeframe=None, partitioning=backfill_plan.partitioning, first_event_time=None,
            last_event_time=None, event_count=0, ordered_partition_ids=(), raw_source_dataset_id=None,
            semantic_digest=compute_content_id("raw_dataset_semantic_digest", {"events": []}), physical_digest=physical_digest_for_ids(()),
            creation_time=creation_time,
        )
        return manifest.dataset_id

    touched_partition_keys = {partition_key_for(market_data_event_time(e), backfill_plan.partitioning) for e in new_events}
    all_members = [(market_data_event_id(e), market_data_event_time(e)) for e in preview_events]
    grouped: dict[str, list[tuple[str, datetime]]] = {}
    for member_id, member_time in all_members:
        grouped.setdefault(partition_key_for(member_time, backfill_plan.partitioning), []).append((member_id, member_time))

    existing_partition_keys = set(repository.partition_store.list_partition_keys(dataset_key))
    ordered_partition_ids: list[str] = []
    for partition_key in sorted(set(grouped) | existing_partition_keys):
        members = grouped.get(partition_key, [])
        if partition_key in touched_partition_keys or partition_key not in existing_partition_keys:
            if not members:
                continue
            partition = build_partition(dataset_key=dataset_key, partition_key=partition_key, spec=backfill_plan.partitioning, members=members)
            ordered_partition_ids.append(partition.partition_id)
        else:
            existing_partition = repository.partition_store.read(dataset_key, partition_key)
            if existing_partition is not None:
                ordered_partition_ids.append(existing_partition.partition_id)

    first_event_time = min(market_data_event_time(e) for e in preview_events)
    last_event_time = max(market_data_event_time(e) for e in preview_events)
    semantic_digest = semantic_digest_for_raw_events(preview_events)
    physical_digest = physical_digest_for_ids(tuple(sorted(market_data_event_id(e) for e in preview_events)))
    manifest = create_dataset_manifest(
        dataset_key=dataset_key, schema_version=1, timeframe=None, partitioning=backfill_plan.partitioning, first_event_time=first_event_time,
        last_event_time=last_event_time, event_count=len(preview_events), ordered_partition_ids=tuple(ordered_partition_ids),
        raw_source_dataset_id=None, semantic_digest=semantic_digest, physical_digest=physical_digest, creation_time=creation_time,
    )
    return manifest.dataset_id


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IngestionOperationReport:
    operation_id: str
    dataset_key: DatasetKey
    source_manifest_id: str
    backfill_plan_id: str
    stage: IngestionStage
    is_dry_run: bool
    parsed_row_count: int
    valid_row_count: int
    quarantined_row_count: int
    quarantine_issue_counts: dict[str, int]
    normalized_event_count: int
    normalized_events_digest: str
    expected_partitions_touched: tuple[str, ...]
    overlapping_interval_count: int
    gap_interval_count: int
    resulting_dataset_id: str | None
    warnings: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id, "dataset_key": self.dataset_key.to_json_dict(), "source_manifest_id": self.source_manifest_id,
            "backfill_plan_id": self.backfill_plan_id, "stage": self.stage.value, "is_dry_run": self.is_dry_run,
            "parsed_row_count": self.parsed_row_count, "valid_row_count": self.valid_row_count,
            "quarantined_row_count": self.quarantined_row_count, "quarantine_issue_counts": dict(self.quarantine_issue_counts),
            "normalized_event_count": self.normalized_event_count, "normalized_events_digest": self.normalized_events_digest,
            "expected_partitions_touched": list(self.expected_partitions_touched), "overlapping_interval_count": self.overlapping_interval_count,
            "gap_interval_count": self.gap_interval_count, "resulting_dataset_id": self.resulting_dataset_id, "warnings": list(self.warnings),
        }


def _verify_repository_provenance_agreement(
    *, repository: MarketDataRepository, provenance_store: ProvenanceStore, dataset_key: DatasetKey, operation_id: str, expected_event_count: int,
) -> None:
    assert dataset_key.provider is not None
    provenance_records = [r for r in provenance_store.read_all(dataset_key) if r.ingestion_batch_id == operation_id]
    if len(provenance_records) != expected_event_count:
        raise OrchestrationStateError(
            f"operation_id {operation_id!r}: provenance has {len(provenance_records)} record(s) for this operation, expected {expected_event_count}"
        )
    if not provenance_records:
        return
    existing_event_ids = {market_data_event_id(e) for e in repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)}
    missing = [r.event_id for r in provenance_records if r.event_id not in existing_event_ids]
    if missing:
        raise OrchestrationStateError(f"operation_id {operation_id!r}: provenance references event_id(s) not present in the repository: {missing}")


def run_ingestion_operation(
    *,
    repository: MarketDataRepository,
    adapter: HistoricalSourceAdapter,
    source_manifest: SourceManifest,
    backfill_plan: BackfillPlan,
    instrument_mapping: InstrumentMappingSpec,
    timeframe_mapping: TimeframeMappingSpec | None,
    timestamp_policy: TimestampParsingPolicy,
    operation_id: str,
    operation_time: datetime,
    on_invalid_row: RowFailurePolicy = RowFailurePolicy.QUARANTINE,
    reference_time: datetime | None = None,
    dry_run: bool = False,
) -> IngestionOperationReport:
    dataset_key = backfill_plan.target_dataset_key
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise OrchestrationError("run_ingestion_operation currently supports only RAW_MARKET_EVENTS dataset keys")
    assert dataset_key.provider is not None
    require_non_empty(operation_id, field_name="operation_id")
    require_tz_aware(operation_time, field_name="operation_time")
    if backfill_plan.source_manifest_id != source_manifest.source_manifest_id:
        raise OrchestrationError("backfill_plan.source_manifest_id does not match source_manifest.source_manifest_id")
    if not backfill_plan.is_admissible:
        raise OrchestrationError(f"backfill_plan {backfill_plan.backfill_plan_id!r} is not admissible: {list(backfill_plan.warnings)}")

    record_kind = source_manifest.record_kind
    required_fields, optional_fields = _expected_row_fields(record_kind)
    timezone_policy_id = compute_timestamp_policy_id(timestamp_policy)

    operation_store = OperationStore(repository.root)
    quarantine_store = QuarantineStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)
    checkpoint_store = CheckpointStore(repository.root)

    content_digest = compute_content_id(
        "historical_ingestion_operation",
        {
            "source_manifest_id": source_manifest.source_manifest_id, "backfill_plan_id": backfill_plan.backfill_plan_id,
            "instrument_mapping_id": instrument_mapping.mapping_id,
            "timeframe_mapping_id": (None if timeframe_mapping is None else timeframe_mapping.mapping_id),
            "timezone_policy_id": timezone_policy_id, "on_invalid_row": on_invalid_row.value, "dataset_key": dataset_key.to_json_dict(),
        },
    )

    if adapter.content_digest() != source_manifest.content_digest:
        raise OrchestrationError(
            f"adapter content_digest {adapter.content_digest()!r} does not match "
            f"source_manifest.content_digest {source_manifest.content_digest!r} -- source content changed"
        )

    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.SOURCE_VERIFIED,
            stage_evidence={"source_manifest_id": source_manifest.source_manifest_id, "adapter_content_digest": adapter.content_digest()},
            operation_time=operation_time,
        )
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.PLAN_CREATED,
            stage_evidence={"backfill_plan_id": backfill_plan.backfill_plan_id}, operation_time=operation_time,
        )

    sequence_start = _resolve_sequence_start(operation_store, repository, dataset_key, operation_id) if not dry_run \
        else repository.event_store.next_sequence(dataset_key.provider, dataset_key.instrument_id)
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.BATCH_RESERVED,
            stage_evidence={"sequence_start": sequence_start}, operation_time=operation_time,
        )

    records = tuple(adapter.iter_records())
    parsed_row_count = len(records)
    parsed_digest = compute_content_id("parsed_rows_digest", {"digests": [r.record_digest() for r in records]})
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.ROWS_PARSED,
            stage_evidence={"parsed_row_count": parsed_row_count, "parsed_digest": parsed_digest}, operation_time=operation_time,
        )

    batch_issues = _detect_batch_level_issues(records)
    sequence_provider = _make_sequence_provider(sequence_start)
    outcomes: list[_RowOutcome] = []
    for record in records:
        outcome = _process_row(
            record, record_kind=record_kind, required_fields=required_fields, optional_fields=optional_fields,
            timestamp_policy=timestamp_policy, instrument_mapping=instrument_mapping, timeframe_mapping=timeframe_mapping,
            default_provider=dataset_key.provider, instrument_id_for_dataset=dataset_key.instrument_id, reference_time=reference_time,
            expected_start=source_manifest.expected_start, expected_end=source_manifest.expected_end,
            expected_timeframe=source_manifest.expected_timeframe, sequence_provider=sequence_provider,
        )
        extra_issues = batch_issues.get(record.row_index, [])
        if extra_issues:
            outcome = _RowOutcome(
                record=outcome.record, issue_codes=outcome.issue_codes + tuple(extra_issues), event=None,
                original_timestamp_text=outcome.original_timestamp_text, normalized_event_time=outcome.normalized_event_time,
                resolved_instrument_id=outcome.resolved_instrument_id,
            )
        outcomes.append(outcome)

    invalid_outcomes = [o for o in outcomes if o.issue_codes]
    valid_outcomes = [o for o in outcomes if not o.issue_codes]

    if invalid_outcomes and on_invalid_row is RowFailurePolicy.FAIL_FAST:
        first = invalid_outcomes[0]
        raise RowValidationError(
            f"row {first.record.row_index} failed validation with issue codes {list(first.issue_codes)} and on_invalid_row=FAIL_FAST"
        )

    quarantine_issue_counts: dict[str, int] = {}
    for outcome in invalid_outcomes:
        for code in outcome.issue_codes:
            quarantine_issue_counts[code] = quarantine_issue_counts.get(code, 0) + 1

    if not dry_run:
        for outcome in invalid_outcomes:
            quarantine_record = create_quarantine_record(
                source_manifest_id=source_manifest.source_manifest_id, source_row_index=outcome.record.row_index,
                raw_record_digest=outcome.record.record_digest(), raw_fields=dict(outcome.record.raw_fields),
                validation_issue_codes=outcome.issue_codes, ingestion_batch_id=operation_id, event_time=operation_time,
            )
            quarantine_store.append(dataset_key, quarantine_record)
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.ROWS_VALIDATED,
            stage_evidence={
                "valid_row_count": len(valid_outcomes), "quarantined_row_count": len(invalid_outcomes),
                "quarantined_row_indices": sorted(o.record.row_index for o in invalid_outcomes),
            },
            operation_time=operation_time,
        )

    events = [o.event for o in valid_outcomes if o.event is not None]
    normalized_events_digest = compute_content_id("normalized_events_digest", {"event_ids": sorted(market_data_event_id(e) for e in events)})
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.EVENTS_NORMALIZED,
            stage_evidence={"event_count": len(events), "normalized_events_digest": normalized_events_digest}, operation_time=operation_time,
        )

    warnings: list[str] = list(backfill_plan.warnings)
    resulting_dataset_id: str | None

    if dry_run:
        resulting_dataset_id = _preview_resulting_dataset_id(
            repository=repository, dataset_key=dataset_key, new_events=events, backfill_plan=backfill_plan, creation_time=operation_time,
        )
        return IngestionOperationReport(
            operation_id=operation_id, dataset_key=dataset_key, source_manifest_id=source_manifest.source_manifest_id,
            backfill_plan_id=backfill_plan.backfill_plan_id, stage=IngestionStage.EVENTS_NORMALIZED, is_dry_run=True,
            parsed_row_count=parsed_row_count, valid_row_count=len(valid_outcomes), quarantined_row_count=len(invalid_outcomes),
            quarantine_issue_counts=quarantine_issue_counts, normalized_event_count=len(events), normalized_events_digest=normalized_events_digest,
            expected_partitions_touched=backfill_plan.expected_partitions_touched, overlapping_interval_count=len(backfill_plan.overlapping_intervals),
            gap_interval_count=len(backfill_plan.gap_intervals), resulting_dataset_id=resulting_dataset_id, warnings=tuple(warnings),
        )

    # Pre-flight, BEFORE any repository write: refuse to commit if any
    # valid row would conflict with ALREADY-durable provenance at its
    # own coordinate (a different operation's event_id already occupies
    # that source row). Checking this only AFTER `ingest_raw_events`
    # would leave an ORPHAN event durably committed with no provenance
    # and no way to ever acquire one -- see `ProvenanceStore.append`'s
    # own conflict rule and this module's "NO IN-MEMORY-ONLY CORRECTNESS
    # STATE" docstring section.
    for outcome in valid_outcomes:
        assert outcome.event is not None
        existing_provenance = provenance_store.read_by_source_coordinate(
            dataset_key, SourceRowCoordinate(source_manifest_id=source_manifest.source_manifest_id, row_index=outcome.record.row_index),
        )
        if existing_provenance is not None and existing_provenance.event_id != market_data_event_id(outcome.event):
            raise ProvenanceError(
                f"source row (source_manifest_id={source_manifest.source_manifest_id!r}, row_index={outcome.record.row_index}) is already "
                f"bound to event_id {existing_provenance.event_id!r} by an earlier operation; operation_id {operation_id!r} would produce "
                f"conflicting event_id {market_data_event_id(outcome.event)!r} -- aborting before any repository write. Reuse the SAME "
                "operation_id to retry the original operation, or use a non-overlapping backfill plan."
            )

    ingestion_result = ingest_raw_events(
        repository=repository, dataset_key=dataset_key, batch_id=operation_id, ingestion_time=operation_time, events=tuple(events),
        partitioning=backfill_plan.partitioning,
    )
    resulting_dataset_id = ingestion_result.resulting_dataset_id
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.REPOSITORY_COMMITTED,
        stage_evidence={"resulting_dataset_id": resulting_dataset_id}, operation_time=operation_time,
    )

    provenance_ids: list[str] = []
    for outcome in valid_outcomes:
        assert outcome.event is not None and outcome.normalized_event_time is not None and outcome.resolved_instrument_id is not None
        provenance_record = create_provenance_record(
            source_manifest_id=source_manifest.source_manifest_id, source_row_index=outcome.record.row_index,
            source_record_digest=outcome.record.record_digest(), original_timestamp_text=outcome.original_timestamp_text,
            normalized_event_time=outcome.normalized_event_time, instrument_mapping_id=instrument_mapping.mapping_id,
            resolved_instrument_id=outcome.resolved_instrument_id,
            timeframe_mapping_id=(None if timeframe_mapping is None else timeframe_mapping.mapping_id), timezone_policy_id=timezone_policy_id,
            ingestion_batch_id=operation_id, event_id=market_data_event_id(outcome.event), dataset_id=resulting_dataset_id,
            recorded_time=operation_time,
        )
        committed = provenance_store.append(dataset_key, provenance_record)
        provenance_ids.append(committed.provenance_id)
    provenance_digest = compute_content_id("provenance_batch_digest", {"provenance_ids": sorted(provenance_ids)})
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.PROVENANCE_COMMITTED,
        stage_evidence={"provenance_digest": provenance_digest, "provenance_count": len(provenance_ids)}, operation_time=operation_time,
    )

    repository_checkpoint = compute_raw_ingestion_checkpoint(
        repository=repository, dataset_key=dataset_key, last_committed_batch_id=operation_id, checkpoint_time=operation_time,
    )
    checkpoint_store.append(dataset_key, repository_checkpoint)
    quarantine_ids = sorted(q.quarantine_record_id for q in quarantine_store.read_all(dataset_key) if q.ingestion_batch_id == operation_id)
    quarantine_digest = compute_content_id("quarantine_batch_digest", {"quarantine_ids": quarantine_ids})
    historical_checkpoint = create_historical_ingestion_checkpoint(
        dataset_key=dataset_key, source_manifest_id=source_manifest.source_manifest_id, backfill_plan_id=backfill_plan.backfill_plan_id,
        operation_id=operation_id, last_processed_source_row_index=(records[-1].row_index if records else None),
        committed_event_ids_digest=normalized_events_digest, quarantine_digest=quarantine_digest, resulting_dataset_id=resulting_dataset_id,
        provenance_digest=provenance_digest, repository_checkpoint_id=repository_checkpoint.checkpoint_id,
        instrument_mapping_id=instrument_mapping.mapping_id, timeframe_mapping_id=(None if timeframe_mapping is None else timeframe_mapping.mapping_id),
        timezone_policy_id=timezone_policy_id, checkpoint_time=operation_time,
    )
    checkpoint_store.append(dataset_key, historical_checkpoint)
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.CHECKPOINT_COMMITTED,
        stage_evidence={"checkpoint_id": historical_checkpoint.checkpoint_id}, operation_time=operation_time,
    )

    _verify_repository_provenance_agreement(
        repository=repository, provenance_store=provenance_store, dataset_key=dataset_key, operation_id=operation_id,
        expected_event_count=len(valid_outcomes),
    )
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.VERIFIED,
        stage_evidence={"verified_event_count": len(valid_outcomes)}, operation_time=operation_time,
    )
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=IngestionStage.COMPLETED,
        stage_evidence={"resulting_dataset_id": resulting_dataset_id}, operation_time=operation_time,
    )

    return IngestionOperationReport(
        operation_id=operation_id, dataset_key=dataset_key, source_manifest_id=source_manifest.source_manifest_id,
        backfill_plan_id=backfill_plan.backfill_plan_id, stage=IngestionStage.COMPLETED, is_dry_run=False, parsed_row_count=parsed_row_count,
        valid_row_count=len(valid_outcomes), quarantined_row_count=len(invalid_outcomes), quarantine_issue_counts=quarantine_issue_counts,
        normalized_event_count=len(events), normalized_events_digest=normalized_events_digest,
        expected_partitions_touched=backfill_plan.expected_partitions_touched, overlapping_interval_count=len(backfill_plan.overlapping_intervals),
        gap_interval_count=len(backfill_plan.gap_intervals), resulting_dataset_id=resulting_dataset_id, warnings=tuple(warnings),
    )


def replay_ingestion_operation(
    *,
    repository: MarketDataRepository,
    adapter: HistoricalSourceAdapter,
    source_manifest: SourceManifest,
    backfill_plan: BackfillPlan,
    instrument_mapping: InstrumentMappingSpec,
    timeframe_mapping: TimeframeMappingSpec | None,
    timestamp_policy: TimestampParsingPolicy,
    operation_id: str,
    operation_time: datetime,
    on_invalid_row: RowFailurePolicy = RowFailurePolicy.QUARANTINE,
    reference_time: datetime | None = None,
) -> IngestionOperationReport:
    """Deterministic replay -- a thin, explicitly-named alias for
    `run_ingestion_operation`. There is no separate replay mechanism to
    write: given the SAME source bytes/records, manifest, mappings,
    normalization spec, backfill plan, and operation times, `run_
    ingestion_operation` is already pure-deterministic (every identity in
    this package is content-addressed; see module docstring), so calling
    it again against a FRESH `repository` (a different, empty
    `storage_root` -- filesystem root/temp paths never participate in
    any identity here) reproduces identical event ids, dataset lineage/
    version, provenance ids, quarantine ids, and checkpoint identities.
    This function exists purely so replay intent is expressed
    explicitly at call sites and in tests, never to add behavior
    `run_ingestion_operation` does not already have."""
    return run_ingestion_operation(
        repository=repository, adapter=adapter, source_manifest=source_manifest, backfill_plan=backfill_plan,
        instrument_mapping=instrument_mapping, timeframe_mapping=timeframe_mapping, timestamp_policy=timestamp_policy,
        operation_id=operation_id, operation_time=operation_time, on_invalid_row=on_invalid_row, reference_time=reference_time,
        dry_run=False,
    )
