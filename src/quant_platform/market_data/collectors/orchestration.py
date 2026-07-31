"""Collector-side ingestion orchestration (Milestone 10, Phase 4A) -- the
smallest honest stage machine for "fetch, persist, verify, and commit one
FRED macro series over one interval," mirroring `market_data.
orchestration.run_ingestion_operation`'s own discipline (idempotent exact
retry, fail-closed conflicting retry, pre-flight provenance-conflict
check before any repository write, durable per-stage evidence) without
literally reusing its `OperationStore`/`IngestionStage` (which are typed
specifically to `IngestionStage` and to committing `MarketDataEvent`s via
`ingest_raw_events`, neither of which fits a `MacroEvent` commit to
`macro.MacroEventStore`) -- `CollectorOperationStore` below duplicates
that PROVEN small pattern (~60 lines) rather than widen Phase 3's own
already-shipped, already-tested code's blast radius for a materially
different commit target.

Required flow (never short-circuited): `REQUEST_PLANNED ->
REQUEST_MANIFEST_COMMITTED -> RESPONSE_DOWNLOADED ->
RAW_RESPONSE_COMMITTED -> RESPONSE_VERIFIED -> SOURCE_MANIFEST_CREATED ->
SOURCE_PARSED -> NORMALIZED_RECORDS_PRODUCED ->
REPOSITORY_INGESTION_COMMITTED -> PROVENANCE_COMMITTED ->
VERIFICATION_COMPLETED`. `market_data.provenance.ProvenanceStore`/
`market_data.quarantine.QuarantineStore` (Phase 3, genuinely
record-kind-agnostic) ARE reused directly, scoped via a `DatasetKey`
using the Phase 4A `DatasetKind.MACRO_OBSERVATIONS` kind purely as a
storage-path/ledger-scoping identity -- the durable ECONOMIC record
store remains Phase 1's own `macro.MacroEventStore`, never Phase 2's
`MarketEventStore`/dataset-manifest/partition machinery (which is
shaped specifically for candle/tick/quote/trade data).

`FetchMode.CACHED_REPLAY` NEVER calls a transport -- it reads an
ALREADY-CACHED response by `request_manifest_id` (or a caller-pinned
`reference_response_manifest_id`), re-hashing on every read via
`cache.RawResponseCache.read_bytes(verify=True)`. A caller wanting a
STRUCTURAL guarantee that this code path performs zero network calls
passes `transport=None` (the default) -- it is never touched in
`CACHED_REPLAY` mode, and `FetchMode.FRESH` raises immediately if it IS
`None`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    CollectorError,
    CollectorOrchestrationConflictError,
    CollectorOrchestrationStateError,
    ProvenanceError,
    ResponseIntegrityError,
    RowValidationError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.adapters import RawSourceRecord, SourceRowCoordinate
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred import (
    build_fred_request_manifest,
    execute_fred_request,
    load_fred_adapter_from_cache,
)
from quant_platform.market_data.collectors.macro_normalization import (
    NormalizedMacroObservation,
    UnitMappingSpec,
    fred_timezone_policy_id,
    normalize_macro_row,
)
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.response_manifest import compute_raw_content_digest
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.macro import MacroEvent, MacroEventStore, create_macro_event
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.orchestration import RowFailurePolicy
from quant_platform.market_data.provenance import ProvenanceStore, create_provenance_record
from quant_platform.market_data.quarantine import QuarantineStore, create_quarantine_record
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, create_source_manifest

__all__ = [
    "COLLECTOR_OPERATION_RECORD_KIND",
    "CollectorIngestionReport",
    "CollectorOperationRecord",
    "CollectorOperationStage",
    "CollectorOperationStore",
    "FetchMode",
    "run_fred_macro_ingestion_operation",
]

COLLECTOR_OPERATION_RECORD_KIND = "collector_operation_record"


class CollectorOperationStage(Enum):
    REQUEST_PLANNED = "request_planned"
    REQUEST_MANIFEST_COMMITTED = "request_manifest_committed"
    RESPONSE_DOWNLOADED = "response_downloaded"
    RAW_RESPONSE_COMMITTED = "raw_response_committed"
    RESPONSE_VERIFIED = "response_verified"
    SOURCE_MANIFEST_CREATED = "source_manifest_created"
    SOURCE_PARSED = "source_parsed"
    NORMALIZED_RECORDS_PRODUCED = "normalized_records_produced"
    REPOSITORY_INGESTION_COMMITTED = "repository_ingestion_committed"
    PROVENANCE_COMMITTED = "provenance_committed"
    VERIFICATION_COMPLETED = "verification_completed"


_STAGE_ORDER: tuple[CollectorOperationStage, ...] = tuple(CollectorOperationStage)
_STAGE_RANK: dict[CollectorOperationStage, int] = {stage: rank for rank, stage in enumerate(_STAGE_ORDER)}


class FetchMode(Enum):
    FRESH = "fresh"
    CACHED_REPLAY = "cached_replay"


@dataclass(frozen=True, slots=True)
class CollectorOperationRecord:
    operation_id: str
    dataset_key: DatasetKey
    stage: CollectorOperationStage
    content_digest: str
    stage_evidence: dict[str, object]
    operation_time: datetime

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": COLLECTOR_OPERATION_RECORD_KIND, "operation_id": self.operation_id, "dataset_key": self.dataset_key.to_json_dict(),
            "stage": self.stage.value, "content_digest": self.content_digest, "stage_evidence": dict(self.stage_evidence),
            "operation_time": serialize_timestamp(self.operation_time, field_name="operation_time"),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CollectorOperationRecord:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            operation_id=str(raw["operation_id"]), dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["dataset_key"], field_name="dataset_key")),
            stage=CollectorOperationStage(raw["stage"]), content_digest=str(raw["content_digest"]),
            stage_evidence=as_json_dict(raw["stage_evidence"], field_name="stage_evidence"),
            operation_time=deserialize_timestamp(raw["operation_time"], field_name="operation_time"),
        )


class CollectorOperationStore:
    """Append-only ledger: `{storage_root}/repository/collector_operations/
    {dataset_key_path}/operations.jsonl`. `advance` mirrors `market_data.
    orchestration.OperationStore.advance` exactly: idempotent for a retry
    of ANY previously-recorded stage (not only the latest -- a full
    re-play from `REQUEST_PLANNED` matches its own history at every
    stage), fails closed (`CollectorOrchestrationConflictError`) for a
    conflicting retry, and (`CollectorOrchestrationStateError`) for an
    illegal jump/regression."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dataset_dir(self, dataset_key: DatasetKey) -> Path:
        return self._root / "repository" / "collector_operations" / Path(*dataset_key.storage_path_parts())

    def _operations_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / "operations.jsonl"

    def _lock_path(self, dataset_key: DatasetKey) -> Path:
        return self._dataset_dir(dataset_key) / ".operations.lock"

    def read_all(self, dataset_key: DatasetKey) -> list[CollectorOperationRecord]:
        from quant_platform.core.exceptions import MarketDataPersistenceError
        from quant_platform.ml.persistence import parse_json_strict

        path = self._operations_path(dataset_key)
        if not path.is_file():
            return []
        records: list[CollectorOperationRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted collector operation ledger line for dataset {dataset_key!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted collector operation ledger line for dataset {dataset_key!r}: expected a JSON object")
            records.append(CollectorOperationRecord.from_json_dict(raw))
        return records

    def read_latest(self, dataset_key: DatasetKey, operation_id: str) -> CollectorOperationRecord | None:
        matching = [r for r in self.read_all(dataset_key) if r.operation_id == operation_id]
        return matching[-1] if matching else None

    def _append(self, dataset_key: DatasetKey, record: CollectorOperationRecord) -> None:
        import os

        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        path = self._operations_path(dataset_key)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def advance(
        self, *, dataset_key: DatasetKey, operation_id: str, content_digest: str, stage: CollectorOperationStage,
        stage_evidence: dict[str, object], operation_time: datetime,
    ) -> CollectorOperationRecord:
        from contextlib import contextmanager

        from quant_platform.core.exceptions import ExperimentLockError, MarketDataLockError
        from quant_platform.ml.concurrency import experiment_lock

        @contextmanager
        def _lock():  # type: ignore[no-untyped-def]
            lock_path = self._lock_path(dataset_key)
            try:
                with experiment_lock(lock_path):
                    yield
            except ExperimentLockError as exc:
                raise MarketDataLockError(f"Could not acquire collector operation ledger lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
            except OSError as exc:
                raise MarketDataLockError(f"Collector operation ledger lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc

        self._dataset_dir(dataset_key).mkdir(parents=True, exist_ok=True)
        with _lock():
            history = [r for r in self.read_all(dataset_key) if r.operation_id == operation_id]
            new_record = CollectorOperationRecord(
                operation_id=operation_id, dataset_key=dataset_key, stage=stage, content_digest=content_digest,
                stage_evidence=stage_evidence, operation_time=operation_time,
            )
            if not history:
                if stage is not CollectorOperationStage.REQUEST_PLANNED:
                    raise CollectorOrchestrationStateError(f"first stage recorded for a new operation_id must be REQUEST_PLANNED, got {stage.value!r}")
                self._append(dataset_key, new_record)
                return new_record
            if history[0].content_digest != content_digest:
                raise CollectorOrchestrationConflictError(
                    f"operation_id {operation_id!r} is already bound to content_digest {history[0].content_digest!r}; "
                    f"a conflicting content_digest {content_digest!r} was submitted"
                )
            latest_rank = _STAGE_RANK[history[-1].stage]
            new_rank = _STAGE_RANK[stage]
            if new_rank <= latest_rank:
                recorded = next(r for r in reversed(history) if r.stage is stage)
                if recorded.stage_evidence == stage_evidence:
                    return recorded
                raise CollectorOrchestrationConflictError(
                    f"operation_id {operation_id!r} stage {stage.value!r} is already durably recorded with different evidence"
                )
            if new_rank != latest_rank + 1:
                raise CollectorOrchestrationStateError(
                    f"operation_id {operation_id!r} cannot advance from {history[-1].stage.value!r} to {stage.value!r} "
                    "(stages must advance exactly one step at a time, never skip or regress)"
                )
            self._append(dataset_key, new_record)
            return new_record


def _resolve_macro_sequence_start(operation_store: CollectorOperationStore, macro_store: MacroEventStore, *, dataset_key: DatasetKey, provider: str, series_id: str, operation_id: str) -> int:
    for record in operation_store.read_all(dataset_key):
        if record.operation_id == operation_id and record.stage is CollectorOperationStage.NORMALIZED_RECORDS_PRODUCED:
            evidence_start = record.stage_evidence.get("macro_sequence_start")
            if evidence_start is not None:
                return int(str(evidence_start))
    return macro_store.next_sequence(provider, series_id)


@dataclass(frozen=True, slots=True)
class CollectorIngestionReport:
    operation_id: str
    dataset_key: DatasetKey
    request_manifest_id: str
    response_manifest_id: str
    source_manifest_id: str
    stage: CollectorOperationStage
    is_dry_run: bool
    fetch_mode: FetchMode
    parsed_row_count: int
    valid_row_count: int
    quarantined_row_count: int
    quarantine_issue_counts: dict[str, int]
    committed_event_count: int
    normalized_events_digest: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id, "dataset_key": self.dataset_key.to_json_dict(), "request_manifest_id": self.request_manifest_id,
            "response_manifest_id": self.response_manifest_id, "source_manifest_id": self.source_manifest_id, "stage": self.stage.value,
            "is_dry_run": self.is_dry_run, "fetch_mode": self.fetch_mode.value, "parsed_row_count": self.parsed_row_count,
            "valid_row_count": self.valid_row_count, "quarantined_row_count": self.quarantined_row_count,
            "quarantine_issue_counts": dict(self.quarantine_issue_counts), "committed_event_count": self.committed_event_count,
            "normalized_events_digest": self.normalized_events_digest,
        }


def run_fred_macro_ingestion_operation(
    *,
    repository: MarketDataRepository,
    cache: RawResponseCache,
    operation_id: str,
    operation_time: datetime,
    series_id: str,
    provider: str,
    unit_mapping: UnitMappingSpec,
    fetch_mode: FetchMode,
    observation_start: datetime | None = None,
    observation_end: datetime | None = None,
    response_format: str = "json",
    credential_mode: CredentialMode = CredentialMode.ANONYMOUS,
    api_key: str | None = None,
    retry_policy: RetryPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    rate_limit_state: TokenBucketState | None = None,
    timeout_policy_id: str = "0" * 64,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    max_response_bytes: int = 10_000_000,
    transport: HistoricalHttpTransport | None = None,
    reference_response_manifest_id: str | None = None,
    on_invalid_row: RowFailurePolicy = RowFailurePolicy.QUARANTINE,
    dry_run: bool = False,
) -> CollectorIngestionReport:
    require_non_empty(operation_id, field_name="operation_id")
    require_tz_aware(operation_time, field_name="operation_time")
    dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider=provider, instrument_id=series_id)

    if fetch_mode is FetchMode.FRESH and (transport is None or retry_policy is None or rate_limit_policy is None or rate_limit_state is None):
        raise CollectorError("FetchMode.FRESH requires transport, retry_policy, rate_limit_policy, and rate_limit_state")

    operation_store = CollectorOperationStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)
    quarantine_store = QuarantineStore(repository.root)
    macro_store = MacroEventStore(repository.root)

    resolved_retry_policy_id = retry_policy.retry_policy_id if retry_policy is not None else "0" * 64
    resolved_rate_limit_policy_id = rate_limit_policy.rate_limit_policy_id if rate_limit_policy is not None else "0" * 64

    # `fetch_mode` and `reference_response_manifest_id` are deliberately EXCLUDED from
    # identity: they describe HOW this operation_id is being executed THIS particular
    # time (fetch fresh vs. replay an already-cached response), not WHAT the operation
    # semantically is (which series, interval, mapping, policies). This is what lets an
    # "exact retry" of the same operation_id switch from FetchMode.FRESH to
    # FetchMode.CACHED_REPLAY (or vice versa) and still be recognized as the same
    # operation -- while a FRESH retry that happens to pull genuinely different response
    # bytes than what is already recorded is caught downstream, as a per-stage evidence
    # conflict at RESPONSE_DOWNLOADED, not collapsed into this top-level digest.
    content_digest = compute_content_id(
        "collector_macro_ingestion_operation",
        {
            "series_id": series_id, "provider": provider, "unit_mapping_id": unit_mapping.unit_mapping_id,
            "response_format": response_format, "retry_policy_id": resolved_retry_policy_id, "rate_limit_policy_id": resolved_rate_limit_policy_id,
            "on_invalid_row": on_invalid_row.value, "dataset_key": dataset_key.to_json_dict(),
        },
    )

    # ---- Stage 1/2: REQUEST_PLANNED / REQUEST_MANIFEST_COMMITTED ----
    request_manifest = build_fred_request_manifest(
        series_id=series_id, observation_start=observation_start, observation_end=observation_end, response_format=response_format,
        timeout_policy_id=timeout_policy_id, retry_policy_id=resolved_retry_policy_id, rate_limit_policy_id=resolved_rate_limit_policy_id,
        credential_mode=credential_mode, request_time=operation_time,
    )
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.REQUEST_PLANNED,
            stage_evidence={"request_manifest_id": request_manifest.request_manifest_id}, operation_time=operation_time,
        )
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.REQUEST_MANIFEST_COMMITTED,
            stage_evidence={"request_manifest_id": request_manifest.request_manifest_id}, operation_time=operation_time,
        )

    # ---- Stage 3: RESPONSE_DOWNLOADED ----
    if fetch_mode is FetchMode.FRESH:
        assert transport is not None and retry_policy is not None and rate_limit_policy is not None and rate_limit_state is not None
        execution, _new_rate_limit_state = execute_fred_request(
            transport=transport, request_manifest=request_manifest, api_key=api_key, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=rate_limit_state, connect_timeout=connect_timeout, read_timeout=read_timeout, max_response_bytes=max_response_bytes,
            operation_time=operation_time,
        )
        response_manifest = execution.response_manifest
        raw_bytes = execution.raw_bytes
    else:
        if reference_response_manifest_id is not None:
            found = cache.read_manifest(reference_response_manifest_id)
            if found is None:
                raise CollectorError(f"no cached response for reference_response_manifest_id {reference_response_manifest_id!r}")
            response_manifest = found
        else:
            latest = cache.read_latest_response_for_request(request_manifest.request_manifest_id)
            if latest is None:
                raise CollectorError(f"no cached response available for offline replay of request_manifest_id {request_manifest.request_manifest_id!r}")
            response_manifest = latest
        raw_bytes = cache.read_bytes(response_manifest.response_manifest_id, verify=True)

    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.RESPONSE_DOWNLOADED,
            stage_evidence={"response_manifest_id": response_manifest.response_manifest_id}, operation_time=operation_time,
        )

    # ---- Stage 4: RAW_RESPONSE_COMMITTED ----
    # The cache write itself is UNCONDITIONAL, even under dry_run: "persist raw response
    # bytes before parsing" (below, Stage 7 reads exclusively from `cache`) is a structural
    # invariant, not a business-record commit -- the cache is a content-addressed, idempotent
    # store, so writing to it carries no semantic weight of "this operation happened" the way
    # appending to the operation ledger / quarantine / provenance / macro event stores does.
    cache.store(response_manifest, raw_bytes)
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.RAW_RESPONSE_COMMITTED,
            stage_evidence={"raw_content_digest": response_manifest.raw_content_digest}, operation_time=operation_time,
        )

    # ---- Stage 5: RESPONSE_VERIFIED ----
    actual_digest = compute_raw_content_digest(raw_bytes)
    if actual_digest != response_manifest.raw_content_digest:
        raise ResponseIntegrityError(
            f"re-hash mismatch for response_manifest_id {response_manifest.response_manifest_id!r}: {actual_digest!r} != {response_manifest.raw_content_digest!r}"
        )
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.RESPONSE_VERIFIED,
            stage_evidence={"verified_digest": actual_digest}, operation_time=operation_time,
        )

    # ---- Stage 6: SOURCE_MANIFEST_CREATED ----
    timezone_policy_id = fred_timezone_policy_id()
    source_manifest = create_source_manifest(
        source_name=f"fred:{series_id}", source_kind=SourceKind.FRED_API, source_schema_version=1, record_kind=RecordKind.MACRO_OBSERVATION,
        source_label=f"fred:{series_id}:{response_manifest.response_manifest_id[:16]}", content_digest=response_manifest.raw_content_digest,
        byte_size=response_manifest.byte_length, encoding=response_manifest.encoding or "utf-8", instrument_mapping_id=unit_mapping.unit_mapping_id,
        timezone_policy_id=timezone_policy_id, unit_normalization_version=unit_mapping.unit_mapping_version, creation_time=operation_time,
        expected_start=observation_start, expected_end=observation_end,
    )
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.SOURCE_MANIFEST_CREATED,
            stage_evidence={"source_manifest_id": source_manifest.source_manifest_id}, operation_time=operation_time,
        )

    # ---- Stage 7: SOURCE_PARSED ----
    adapter = load_fred_adapter_from_cache(cache, response_manifest.response_manifest_id, series_id=series_id, response_format=response_format)
    if adapter.content_digest() != source_manifest.content_digest:
        raise ResponseIntegrityError(
            f"adapter content_digest {adapter.content_digest()!r} does not match source_manifest.content_digest {source_manifest.content_digest!r}"
        )
    records = tuple(adapter.iter_records())
    parsed_row_count = len(records)
    if not dry_run:
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.SOURCE_PARSED,
            stage_evidence={"parsed_row_count": parsed_row_count}, operation_time=operation_time,
        )

    # ---- Stage 8: NORMALIZED_RECORDS_PRODUCED ----
    valid_outcomes: list[tuple[RawSourceRecord, NormalizedMacroObservation]] = []
    invalid_outcomes: list[tuple[RawSourceRecord, tuple[str, ...]]] = []
    for record in records:
        observation, issue_codes = normalize_macro_row(record.raw_fields, series_id=series_id, unit_mapping=unit_mapping)
        if issue_codes:
            invalid_outcomes.append((record, issue_codes))
        else:
            assert observation is not None
            valid_outcomes.append((record, observation))

    if invalid_outcomes and on_invalid_row is RowFailurePolicy.FAIL_FAST:
        first_record, first_codes = invalid_outcomes[0]
        raise RowValidationError(f"row {first_record.row_index} failed validation with issue codes {list(first_codes)} and on_invalid_row=FAIL_FAST")

    quarantine_issue_counts: dict[str, int] = {}
    for _record, codes in invalid_outcomes:
        for code in codes:
            quarantine_issue_counts[code] = quarantine_issue_counts.get(code, 0) + 1

    sequence_start = (
        _resolve_macro_sequence_start(operation_store, macro_store, dataset_key=dataset_key, provider=provider, series_id=series_id, operation_id=operation_id)
        if not dry_run else macro_store.next_sequence(provider, series_id)
    )

    if not dry_run:
        for record, codes in invalid_outcomes:
            quarantine_record = create_quarantine_record(
                source_manifest_id=source_manifest.source_manifest_id, source_row_index=record.row_index,
                raw_record_digest=record.record_digest(), raw_fields=dict(record.raw_fields), validation_issue_codes=codes,
                ingestion_batch_id=operation_id, event_time=operation_time,
            )
            quarantine_store.append(dataset_key, quarantine_record)
        operation_store.advance(
            dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.NORMALIZED_RECORDS_PRODUCED,
            stage_evidence={
                "valid_row_count": len(valid_outcomes), "quarantined_row_count": len(invalid_outcomes),
                "quarantined_row_indices": sorted(r.row_index for r, _c in invalid_outcomes),
                "macro_sequence_start": sequence_start,
            },
            operation_time=operation_time,
        )

    events: list[tuple[RawSourceRecord, MacroEvent]] = []
    for index, (record, observation) in enumerate(valid_outcomes):
        event = create_macro_event(
            series_id=series_id, provider=provider, event_time=observation.event_time, sequence=sequence_start + index,
            value=observation.value, unit=observation.unit.value, source_event_id=observation.source_event_id,
        )
        events.append((record, event))
    normalized_events_digest = compute_content_id("collector_normalized_events_digest", {"event_ids": sorted(e.event_id for _r, e in events)})

    if dry_run:
        return CollectorIngestionReport(
            operation_id=operation_id, dataset_key=dataset_key, request_manifest_id=request_manifest.request_manifest_id,
            response_manifest_id=response_manifest.response_manifest_id, source_manifest_id=source_manifest.source_manifest_id,
            stage=CollectorOperationStage.NORMALIZED_RECORDS_PRODUCED, is_dry_run=True, fetch_mode=fetch_mode, parsed_row_count=parsed_row_count,
            valid_row_count=len(valid_outcomes), quarantined_row_count=len(invalid_outcomes), quarantine_issue_counts=quarantine_issue_counts,
            committed_event_count=0, normalized_events_digest=normalized_events_digest,
        )

    # Pre-flight provenance conflict check -- BEFORE any repository write,
    # mirroring `market_data.orchestration`'s own fix for the identical
    # "orphan committed event with no provenance" hazard.
    for record, event in events:
        existing = provenance_store.read_by_source_coordinate(dataset_key, SourceRowCoordinate(source_manifest_id=source_manifest.source_manifest_id, row_index=record.row_index))
        if existing is not None and existing.event_id != event.event_id:
            raise ProvenanceError(
                f"source row (source_manifest_id={source_manifest.source_manifest_id!r}, row_index={record.row_index}) is already bound to "
                f"event_id {existing.event_id!r}; operation_id {operation_id!r} would produce conflicting event_id {event.event_id!r} -- "
                "aborting before any repository write"
            )

    # ---- Stage 9: REPOSITORY_INGESTION_COMMITTED ----
    for _record, event in events:
        macro_store.append(event)
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.REPOSITORY_INGESTION_COMMITTED,
        stage_evidence={"committed_event_count": len(events)}, operation_time=operation_time,
    )

    # ---- Stage 10: PROVENANCE_COMMITTED ----
    provenance_ids: list[str] = []
    for record, event in events:
        provenance_record = create_provenance_record(
            source_manifest_id=source_manifest.source_manifest_id, source_row_index=record.row_index, source_record_digest=record.record_digest(),
            original_timestamp_text=record.raw_fields.get("date", ""), normalized_event_time=event.event_time,
            instrument_mapping_id=unit_mapping.unit_mapping_id, resolved_instrument_id=series_id, timeframe_mapping_id=None,
            timezone_policy_id=timezone_policy_id, ingestion_batch_id=operation_id, event_id=event.event_id, dataset_id=response_manifest.response_manifest_id,
            recorded_time=operation_time,
        )
        committed = provenance_store.append(dataset_key, provenance_record)
        provenance_ids.append(committed.provenance_id)
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.PROVENANCE_COMMITTED,
        stage_evidence={"provenance_digest": compute_content_id("collector_provenance_batch_digest", {"provenance_ids": sorted(provenance_ids)})},
        operation_time=operation_time,
    )

    # ---- Stage 11: VERIFICATION_COMPLETED ----
    committed_event_ids = {e.event_id for _r, e in events}
    existing_event_ids = {e.event_id for e in macro_store.read_events(provider, series_id)}
    missing = committed_event_ids - existing_event_ids
    if missing:
        raise CollectorOrchestrationStateError(f"operation_id {operation_id!r}: {len(missing)} committed event id(s) not found in macro store after commit: {sorted(missing)}")
    operation_store.advance(
        dataset_key=dataset_key, operation_id=operation_id, content_digest=content_digest, stage=CollectorOperationStage.VERIFICATION_COMPLETED,
        stage_evidence={"verified_event_count": len(events)}, operation_time=operation_time,
    )

    return CollectorIngestionReport(
        operation_id=operation_id, dataset_key=dataset_key, request_manifest_id=request_manifest.request_manifest_id,
        response_manifest_id=response_manifest.response_manifest_id, source_manifest_id=source_manifest.source_manifest_id,
        stage=CollectorOperationStage.VERIFICATION_COMPLETED, is_dry_run=False, fetch_mode=fetch_mode, parsed_row_count=parsed_row_count,
        valid_row_count=len(valid_outcomes), quarantined_row_count=len(invalid_outcomes), quarantine_issue_counts=quarantine_issue_counts,
        committed_event_count=len(events), normalized_events_digest=normalized_events_digest,
    )
