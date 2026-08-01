"""Multi-series curated backfill orchestration (Milestone 10, Phase
4B) -- the 12-stage state machine over an entire `CuratedBackfillSpec`.
`CuratedOperationStage`/`CuratedOperationStore` are a THIRD, self-
contained, small stage-machine implementation (after Phase 3's
`OperationStore` and Phase 4A's `CollectorOperationStore`), duplicating
the SAME proven idempotent/conflict/monotonic-progression algorithm --
scoped to `target_dataset_namespace` (the natural multi-series analogue
of Phase 4A's single `operation_id`+`DatasetKey` scoping), because this
operation spans MANY series at once, a materially different shape than
either of the two existing stage machines commit.

STORAGE SCOPE DECISION: curated observations live in their OWN new
`CuratedObservationStore`/`ComponentDatasetManifestStore`/
`CombinedUniverseManifestStore` (see `datasets.py`/`macro_observation.py`),
NOT funneled into Phase 1's `macro.MacroEventStore` -- `CuratedMacroObservation`
is a materially richer record (native vs. normalized unit, full
vintage/availability lineage) than `MacroEvent` was ever shaped to
hold. `provenance.ProvenanceStore`/`quarantine.QuarantineStore` (Phase
3, genuinely record-kind-agnostic) ARE reused directly, scoped via the
SAME `DatasetKind.MACRO_OBSERVATIONS` Phase 4A already established for
`(provider="fred", instrument_id=series_id)`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    CollectorOrchestrationStateError,
    ProvenanceError,
    ResponseIntegrityError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.adapters import RawSourceRecord, SourceRowCoordinate
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.curated.availability import (
    AvailabilityPolicy,
    resolve_availability_time,
)
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, CuratedBackfillSpec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    CompletenessStatus,
    ComponentDatasetManifest,
    ComponentDatasetManifestStore,
    create_combined_universe_manifest,
    create_component_dataset_manifest,
)
from quant_platform.market_data.collectors.curated.macro_observation import (
    CuratedMacroObservation,
    CuratedObservationStore,
    create_curated_macro_observation,
)
from quant_platform.market_data.collectors.curated.metadata import verify_series_metadata
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry, MissingValuePolicy
from quant_platform.market_data.collectors.curated.revision_policy import (
    RevisionPolicy,
    resolve_fred_request_overrides,
)
from quant_platform.market_data.collectors.fred import (
    _observation_to_raw_fields,
    build_fred_request_manifest,
    execute_fred_request,
)
from quant_platform.market_data.collectors.fred_schemas import (
    FredObservation,
    is_missing_value,
    parse_fred_json_response,
)
from quant_platform.market_data.collectors.fred_series_metadata import (
    build_fred_series_metadata_request_manifest,
    execute_fred_series_metadata_request,
    parse_fred_series_metadata_response,
)
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.response_manifest import compute_raw_content_digest
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.identity import compute_content_id, require_non_empty, require_tz_aware
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore, create_provenance_record
from quant_platform.market_data.quarantine import (
    EMPTY_TIMESTAMP,
    INVALID_DECIMAL,
    MISSING_OBSERVATION_VALUE,
    QuarantineStore,
    create_quarantine_record,
)
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, create_source_manifest

__all__ = [
    "CURATED_OPERATION_RECORD_KIND",
    "CuratedIngestionReport",
    "CuratedOperationRecord",
    "CuratedOperationStage",
    "CuratedOperationStore",
    "SeriesOutcome",
    "run_curated_backfill_operation",
]

CURATED_OPERATION_RECORD_KIND = "curated_operation_record"


class CuratedOperationStage(Enum):
    REGISTRY_VERIFIED = "registry_verified"
    PLAN_CREATED = "plan_created"
    SERIES_METADATA_VERIFIED = "series_metadata_verified"
    REQUESTS_COMMITTED = "requests_committed"
    RESPONSES_COMMITTED = "responses_committed"
    OBSERVATIONS_PARSED = "observations_parsed"
    AVAILABILITY_RESOLVED = "availability_resolved"
    SERIES_DATASETS_COMMITTED = "series_datasets_committed"
    COMBINED_MANIFEST_COMMITTED = "combined_manifest_committed"
    RECONCILED = "reconciled"
    VERIFIED = "verified"
    COMPLETED = "completed"


_STAGE_ORDER: tuple[CuratedOperationStage, ...] = tuple(CuratedOperationStage)
_STAGE_RANK: dict[CuratedOperationStage, int] = {stage: rank for rank, stage in enumerate(_STAGE_ORDER)}


@dataclass(frozen=True, slots=True)
class CuratedOperationRecord:
    operation_id: str
    target_dataset_namespace: str
    stage: CuratedOperationStage
    content_digest: str
    stage_evidence: dict[str, object]
    operation_time: datetime

    def to_json_dict(self) -> dict[str, object]:
        from quant_platform.market_data.identity import serialize_timestamp

        return {
            "kind": CURATED_OPERATION_RECORD_KIND, "operation_id": self.operation_id, "target_dataset_namespace": self.target_dataset_namespace,
            "stage": self.stage.value, "content_digest": self.content_digest, "stage_evidence": dict(self.stage_evidence),
            "operation_time": serialize_timestamp(self.operation_time, field_name="operation_time"),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedOperationRecord:
        from quant_platform.market_data.identity import deserialize_timestamp
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            operation_id=str(raw["operation_id"]), target_dataset_namespace=str(raw["target_dataset_namespace"]), stage=CuratedOperationStage(raw["stage"]),
            content_digest=str(raw["content_digest"]), stage_evidence=as_json_dict(raw["stage_evidence"], field_name="stage_evidence"),
            operation_time=deserialize_timestamp(raw["operation_time"], field_name="operation_time"),
        )


class CuratedOperationStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, target_dataset_namespace: str) -> Path:
        return self._root / "collectors" / "curated" / "operations" / target_dataset_namespace

    def _path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / "operations.jsonl"

    def _lock_path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / ".operations.lock"

    def read_all(self, target_dataset_namespace: str) -> list[CuratedOperationRecord]:
        from quant_platform.core.exceptions import MarketDataPersistenceError
        from quant_platform.ml.persistence import parse_json_strict

        path = self._path(target_dataset_namespace)
        if not path.is_file():
            return []
        records: list[CuratedOperationRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted curated operation ledger line for namespace {target_dataset_namespace!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted curated operation ledger line for namespace {target_dataset_namespace!r}: expected a JSON object")
            records.append(CuratedOperationRecord.from_json_dict(raw))
        return records

    def _append(self, target_dataset_namespace: str, record: CuratedOperationRecord) -> None:
        import os

        self._dir(target_dataset_namespace).mkdir(parents=True, exist_ok=True)
        path = self._path(target_dataset_namespace)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def advance(
        self, *, target_dataset_namespace: str, operation_id: str, content_digest: str, stage: CuratedOperationStage,
        stage_evidence: dict[str, object], operation_time: datetime,
    ) -> CuratedOperationRecord:
        from contextlib import contextmanager

        from quant_platform.core.exceptions import (
            CollectorOrchestrationConflictError,
            ExperimentLockError,
            MarketDataLockError,
        )
        from quant_platform.ml.concurrency import experiment_lock

        @contextmanager
        def _lock():  # type: ignore[no-untyped-def]
            lock_path = self._lock_path(target_dataset_namespace)
            try:
                with experiment_lock(lock_path):
                    yield
            except ExperimentLockError as exc:
                raise MarketDataLockError(f"Could not acquire curated operation ledger lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
            except OSError as exc:
                raise MarketDataLockError(f"Curated operation ledger lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc

        self._dir(target_dataset_namespace).mkdir(parents=True, exist_ok=True)
        with _lock():
            history = [r for r in self.read_all(target_dataset_namespace) if r.operation_id == operation_id]
            new_record = CuratedOperationRecord(
                operation_id=operation_id, target_dataset_namespace=target_dataset_namespace, stage=stage, content_digest=content_digest,
                stage_evidence=stage_evidence, operation_time=operation_time,
            )
            if not history:
                if stage is not CuratedOperationStage.REGISTRY_VERIFIED:
                    raise CollectorOrchestrationStateError(f"first stage recorded for a new operation_id must be REGISTRY_VERIFIED, got {stage.value!r}")
                self._append(target_dataset_namespace, new_record)
                return new_record
            if history[0].content_digest != content_digest:
                raise CollectorOrchestrationConflictError(
                    f"operation_id {operation_id!r} is already bound to content_digest {history[0].content_digest!r}; a conflicting content_digest {content_digest!r} was submitted"
                )
            latest_rank = _STAGE_RANK[history[-1].stage]
            new_rank = _STAGE_RANK[stage]
            if new_rank <= latest_rank:
                recorded = next(r for r in reversed(history) if r.stage is stage)
                if recorded.stage_evidence == stage_evidence:
                    return recorded
                raise CollectorOrchestrationConflictError(f"operation_id {operation_id!r} stage {stage.value!r} is already durably recorded with different evidence")
            if new_rank != latest_rank + 1:
                raise CollectorOrchestrationStateError(
                    f"operation_id {operation_id!r} cannot advance from {history[-1].stage.value!r} to {stage.value!r} (stages must advance exactly one step at a time)"
                )
            self._append(target_dataset_namespace, new_record)
            return new_record


@dataclass(frozen=True, slots=True)
class SeriesOutcome:
    series_id: str
    succeeded: bool
    failure_reason: str | None
    request_manifest_id: str | None
    response_manifest_id: str | None
    source_manifest_id: str | None
    parsed_row_count: int
    valid_row_count: int
    quarantined_row_count: int
    missing_count: int
    skipped_missing_count: int
    committed_observation_count: int
    component_manifest_id: str | None


@dataclass(frozen=True, slots=True)
class CuratedIngestionReport:
    operation_id: str
    target_dataset_namespace: str
    backfill_plan_id: str
    stage: CuratedOperationStage
    completeness_status: str
    series_outcomes: tuple[SeriesOutcome, ...]
    combined_manifest_id: str | None
    is_dry_run: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id, "target_dataset_namespace": self.target_dataset_namespace, "backfill_plan_id": self.backfill_plan_id,
            "stage": self.stage.value, "completeness_status": self.completeness_status,
            "series_outcomes": [
                {
                    "series_id": o.series_id, "succeeded": o.succeeded, "failure_reason": o.failure_reason, "request_manifest_id": o.request_manifest_id,
                    "response_manifest_id": o.response_manifest_id, "source_manifest_id": o.source_manifest_id, "parsed_row_count": o.parsed_row_count,
                    "valid_row_count": o.valid_row_count, "quarantined_row_count": o.quarantined_row_count, "missing_count": o.missing_count,
                    "skipped_missing_count": o.skipped_missing_count, "committed_observation_count": o.committed_observation_count,
                    "component_manifest_id": o.component_manifest_id,
                }
                for o in self.series_outcomes
            ],
            "combined_manifest_id": self.combined_manifest_id, "is_dry_run": self.is_dry_run,
        }


@dataclass(frozen=True, slots=True)
class _EffectiveRequestParams:
    realtime_start: datetime | None
    realtime_end: datetime | None
    output_type: int | None


def _resolve_effective_request_params(spec: CuratedBackfillSpec, revision_policy: RevisionPolicy) -> _EffectiveRequestParams:
    overrides = resolve_fred_request_overrides(revision_policy)
    realtime_start = overrides["realtime_start"]
    realtime_end = overrides["realtime_end"]
    output_type = overrides["output_type"]
    assert realtime_start is None or isinstance(realtime_start, datetime)
    assert realtime_end is None or isinstance(realtime_end, datetime)
    assert output_type is None or isinstance(output_type, int)
    return _EffectiveRequestParams(
        realtime_start=spec.realtime_start if spec.realtime_start is not None else realtime_start,
        realtime_end=spec.realtime_end if spec.realtime_end is not None else realtime_end,
        output_type=spec.output_type if spec.output_type is not None else output_type,
    )


class _SeriesFailureError(Exception):
    def __init__(self, series_id: str, reason: str) -> None:
        super().__init__(reason)
        self.series_id = series_id
        self.reason = reason


def _normalize_curated_row(
    obs: FredObservation, *, spec_series_id: str, canonical_series_name: str, target_macro_instrument_id: str, normalized_unit: str,
    native_unit: str, native_frequency: str, unit_conversion: Decimal, missing_value_policy: MissingValuePolicy, availability_policy: AvailabilityPolicy,
    availability_policy_id: str, request_manifest_id: str, response_manifest_id: str, source_manifest_id: str,
) -> tuple[CuratedMacroObservation | None, tuple[str, ...], bool]:
    """Pure. Returns `(observation_or_none, quarantine_issue_codes,
    skip_only)`: `skip_only=True` means SKIP_AND_REPORT excluded this
    row WITHOUT quarantining it (counted, never silent)."""
    from quant_platform.market_data.source_normalization import parse_source_decimal

    missing = is_missing_value(obs.value_text)
    value: Decimal | None = None
    if not missing:
        try:
            value = parse_source_decimal(obs.value_text, field_name="value")
        except Exception:
            return None, (INVALID_DECIMAL,), False

    if missing and missing_value_policy is MissingValuePolicy.SKIP_AND_REPORT:
        return None, (), True

    try:
        availability_time = resolve_availability_time(availability_policy, observation_date_text=obs.observation_date, realtime_start_text=obs.realtime_start)
    except Exception:
        return None, (EMPTY_TIMESTAMP,), False

    if missing:
        if missing_value_policy is MissingValuePolicy.QUARANTINE:
            return None, (MISSING_OBSERVATION_VALUE,), False
        assert missing_value_policy is MissingValuePolicy.STORE_AS_MISSING_FACT
        observation = create_curated_macro_observation(
            series_id=spec_series_id, canonical_series_name=canonical_series_name, target_macro_instrument_id=target_macro_instrument_id,
            observation_date=obs.observation_date, value=None, is_missing=True, normalized_unit=normalized_unit, native_unit=native_unit, native_frequency=native_frequency,
            realtime_start=obs.realtime_start, realtime_end=obs.realtime_end, availability_time=availability_time, availability_policy_id=availability_policy_id,
            request_manifest_id=request_manifest_id, response_manifest_id=response_manifest_id, source_manifest_id=source_manifest_id, source_row_index=obs.row_index,
        )
        return observation, (), False

    assert value is not None
    from quant_platform.market_data.source_normalization import normalize_signed_zero

    scaled_value = normalize_signed_zero(value * unit_conversion)
    observation = create_curated_macro_observation(
        series_id=spec_series_id, canonical_series_name=canonical_series_name, target_macro_instrument_id=target_macro_instrument_id,
        observation_date=obs.observation_date, value=scaled_value, is_missing=False, normalized_unit=normalized_unit, native_unit=native_unit, native_frequency=native_frequency,
        realtime_start=obs.realtime_start, realtime_end=obs.realtime_end, availability_time=availability_time, availability_policy_id=availability_policy_id,
        request_manifest_id=request_manifest_id, response_manifest_id=response_manifest_id, source_manifest_id=source_manifest_id, source_row_index=obs.row_index,
    )
    return observation, (), False


def run_curated_backfill_operation(
    *,
    repository: MarketDataRepository,
    cache: RawResponseCache,
    registry: CuratedFredRegistry,
    backfill_spec: CuratedBackfillSpec,
    availability_policies: dict[str, AvailabilityPolicy],
    revision_policy: RevisionPolicy,
    operation_id: str,
    operation_time: datetime,
    transport: HistoricalHttpTransport | None = None,
    api_key: str | None = None,
    retry_policy: RetryPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    rate_limit_state: TokenBucketState | None = None,
    timeout_policy_id: str = "0" * 64,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    max_response_bytes: int = 10_000_000,
    credential_mode: CredentialMode = CredentialMode.ANONYMOUS,
    dry_run: bool = False,
) -> CuratedIngestionReport:
    """`backfill_spec.fail_fast=True` raises `_SeriesFailureError` (wrapped
    into a `CollectorOrchestrationStateError`-style stop) the moment any
    series fails ANY stage -- nothing beyond that point is committed for
    ANY series, and `COMPLETED` is never reached. `fail_fast=False`
    (partial success) records each series' own outcome independently;
    the combined manifest's `completeness_status` is `PARTIAL` the
    moment even one series failed, and can never be mistaken for a
    complete universe (see `datasets.CompletenessStatus`)."""
    require_non_empty(operation_id, field_name="operation_id")
    require_tz_aware(operation_time, field_name="operation_time")
    if backfill_spec.curated_registry_id != registry.registry_id:
        raise CollectorOrchestrationStateError(f"backfill_spec.curated_registry_id={backfill_spec.curated_registry_id!r} does not match the supplied registry's own id={registry.registry_id!r}")
    # `backfill_spec.cache_policy` (PREFER_CACHE / FORCE_FRESH) and the presence
    # of transport/retry/rate-limit objects are both consulted PER SERIES below
    # (each series independently decides FRESH vs. CACHED_REPLAY), never here.

    operation_store = CuratedOperationStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)
    quarantine_store = QuarantineStore(repository.root)
    observation_store = CuratedObservationStore(repository.root)
    component_store = ComponentDatasetManifestStore(repository.root)
    combined_store = CombinedUniverseManifestStore(repository.root)

    content_digest = compute_content_id("curated_backfill_operation", {"backfill_plan_id": backfill_spec.backfill_plan_id, "revision_policy_id": revision_policy.revision_policy_id})
    ns = backfill_spec.target_dataset_namespace

    def _advance(stage: CuratedOperationStage, evidence: dict[str, object]) -> None:
        if not dry_run:
            operation_store.advance(target_dataset_namespace=ns, operation_id=operation_id, content_digest=content_digest, stage=stage, stage_evidence=evidence, operation_time=operation_time)

    # ---- Stage 1: REGISTRY_VERIFIED ----
    _advance(CuratedOperationStage.REGISTRY_VERIFIED, {"registry_id": registry.registry_id})
    # ---- Stage 2: PLAN_CREATED ----
    _advance(CuratedOperationStage.PLAN_CREATED, {"backfill_plan_id": backfill_spec.backfill_plan_id})

    effective_params = _resolve_effective_request_params(backfill_spec, revision_policy)
    outcomes: list[SeriesOutcome] = []
    component_manifests: dict[str, ComponentDatasetManifest] = {}
    availability_ids_used: dict[str, str] = {}

    for series_id in backfill_spec.selected_series_ids:
        spec = registry.get(series_id)
        assert spec is not None
        availability_policy = availability_policies[series_id]
        dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id=series_id)

        try:
            # ---- Stage 3 (per series): metadata verification ----
            metadata_request = build_fred_series_metadata_request_manifest(
                series_id=series_id, timeout_policy_id=timeout_policy_id, retry_policy_id=(retry_policy.retry_policy_id if retry_policy else "0" * 64),
                rate_limit_policy_id=(rate_limit_policy.rate_limit_policy_id if rate_limit_policy else "0" * 64), credential_mode=credential_mode,
                request_time=operation_time,
            )
            if backfill_spec.cache_policy is CachePolicy.PREFER_CACHE:
                cached_meta = cache.read_latest_response_for_request(metadata_request.request_manifest_id)
            else:
                cached_meta = None
            if cached_meta is not None:
                meta_response_manifest = cached_meta
                meta_raw_bytes = cache.read_bytes(cached_meta.response_manifest_id, verify=True)
            else:
                if transport is None or retry_policy is None or rate_limit_policy is None or rate_limit_state is None:
                    raise _SeriesFailureError(series_id, "FRESH metadata fetch required but transport/retry/rate-limit not supplied")
                meta_execution, rate_limit_state = execute_fred_series_metadata_request(
                    transport=transport, request_manifest=metadata_request, api_key=api_key, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                    rate_limit_state=rate_limit_state, connect_timeout=connect_timeout, read_timeout=read_timeout, max_response_bytes=max_response_bytes,
                    operation_time=operation_time,
                )
                meta_response_manifest = meta_execution.response_manifest
                meta_raw_bytes = meta_execution.raw_bytes
                cache.store(meta_response_manifest, meta_raw_bytes)
            fred_metadata = parse_fred_series_metadata_response(meta_raw_bytes)
            drift_result = verify_series_metadata(
                spec, fred_metadata, requested_observation_start=backfill_spec.observation_start.strftime("%Y-%m-%d"),
                requested_observation_end=backfill_spec.observation_end.strftime("%Y-%m-%d"),
            )
            if not drift_result.passed:
                fail_codes = [f.code for f in drift_result.findings if f.severity == "fail_closed"]
                raise _SeriesFailureError(series_id, f"metadata drift (fail-closed): {fail_codes}")

            # ---- Stage 4 (per series): request committed ----
            request_manifest = build_fred_request_manifest(
                series_id=series_id, observation_start=backfill_spec.observation_start, observation_end=backfill_spec.observation_end,
                response_format="json", timeout_policy_id=timeout_policy_id, retry_policy_id=(retry_policy.retry_policy_id if retry_policy else "0" * 64),
                rate_limit_policy_id=(rate_limit_policy.rate_limit_policy_id if rate_limit_policy else "0" * 64), credential_mode=credential_mode,
                request_time=operation_time, realtime_start=effective_params.realtime_start, realtime_end=effective_params.realtime_end,
                limit=backfill_spec.page_size, sort_order="asc", units=spec.request_units, frequency=spec.request_frequency,
                aggregation_method=spec.aggregation_method, output_type=effective_params.output_type,
            )

            # ---- Stage 5 (per series): response committed ----
            if backfill_spec.cache_policy is CachePolicy.PREFER_CACHE:
                cached_obs = cache.read_latest_response_for_request(request_manifest.request_manifest_id)
            else:
                cached_obs = None
            if cached_obs is not None:
                response_manifest = cached_obs
                raw_bytes = cache.read_bytes(cached_obs.response_manifest_id, verify=True)
            else:
                if transport is None or retry_policy is None or rate_limit_policy is None or rate_limit_state is None:
                    raise _SeriesFailureError(series_id, "FRESH observation fetch required but transport/retry/rate-limit not supplied")
                execution, rate_limit_state = execute_fred_request(
                    transport=transport, request_manifest=request_manifest, api_key=api_key, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                    rate_limit_state=rate_limit_state, connect_timeout=connect_timeout, read_timeout=read_timeout, max_response_bytes=max_response_bytes,
                    operation_time=operation_time,
                )
                response_manifest = execution.response_manifest
                raw_bytes = execution.raw_bytes
                cache.store(response_manifest, raw_bytes)

            actual_digest = compute_raw_content_digest(raw_bytes)
            if actual_digest != response_manifest.raw_content_digest:
                raise ResponseIntegrityError(f"re-hash mismatch for {series_id!r} response_manifest_id {response_manifest.response_manifest_id!r}")

            # ---- Stage 6 (per series): observations parsed ----
            observations_raw = parse_fred_json_response(raw_bytes, series_id=series_id)
            if len(observations_raw) > backfill_spec.max_observations_per_series:
                raise _SeriesFailureError(series_id, f"{len(observations_raw)} observations exceeds max_observations_per_series={backfill_spec.max_observations_per_series}")

            normalization_ref_id = compute_content_id("curated_normalization_ref", {"normalization_kind": spec.normalization_kind.value, "unit_conversion": str(spec.unit_conversion)})
            source_manifest = create_source_manifest(
                source_name=f"fred:{series_id}", source_kind=SourceKind.FRED_API, source_schema_version=1, record_kind=RecordKind.MACRO_OBSERVATION,
                source_label=f"fred:{series_id}:{response_manifest.response_manifest_id[:16]}", content_digest=response_manifest.raw_content_digest,
                byte_size=response_manifest.byte_length, encoding=response_manifest.encoding or "utf-8", instrument_mapping_id=normalization_ref_id,
                timezone_policy_id=availability_policy.availability_policy_id, unit_normalization_version=1, creation_time=operation_time,
                expected_start=backfill_spec.observation_start, expected_end=backfill_spec.observation_end, row_count=len(observations_raw),
            )

            # ---- Stage 7 (per series): availability resolved ----
            valid_rows: list[tuple[FredObservation, CuratedMacroObservation]] = []
            quarantine_rows: list[tuple[FredObservation, tuple[str, ...]]] = []
            skipped_missing = 0
            for obs in observations_raw:
                observation, issue_codes, skip_only = _normalize_curated_row(
                    obs, spec_series_id=series_id, canonical_series_name=spec.canonical_series_name, target_macro_instrument_id=spec.target_macro_instrument_id,
                    normalized_unit=spec.normalization_kind.value, native_unit=fred_metadata.units_short, native_frequency=fred_metadata.frequency_short,
                    unit_conversion=spec.unit_conversion, missing_value_policy=spec.missing_value_policy,
                    availability_policy=availability_policy, availability_policy_id=availability_policy.availability_policy_id,
                    request_manifest_id=request_manifest.request_manifest_id, response_manifest_id=response_manifest.response_manifest_id,
                    source_manifest_id=source_manifest.source_manifest_id,
                )
                if skip_only:
                    skipped_missing += 1
                elif observation is not None:
                    valid_rows.append((obs, observation))
                else:
                    quarantine_rows.append((obs, issue_codes))

            if not dry_run:
                for obs, issue_codes in quarantine_rows:
                    record = RawSourceRecord(row_index=obs.row_index, raw_fields=_observation_to_raw_fields(obs), raw_text=f"{obs.observation_date},{obs.value_text}")
                    quarantine_record = create_quarantine_record(
                        source_manifest_id=source_manifest.source_manifest_id, source_row_index=obs.row_index, raw_record_digest=record.record_digest(),
                        raw_fields=dict(record.raw_fields), validation_issue_codes=issue_codes, ingestion_batch_id=operation_id, event_time=operation_time,
                    )
                    quarantine_store.append(dataset_key, quarantine_record)

            missing_count = sum(1 for _obs, observation in valid_rows if observation.is_missing) + sum(1 for obs, _codes in quarantine_rows if is_missing_value(obs.value_text)) + skipped_missing

            # ---- Pre-flight provenance-conflict check (Phase 3/4A pattern) ----
            if not dry_run:
                for obs, observation in valid_rows:
                    existing = provenance_store.read_by_source_coordinate(dataset_key, SourceRowCoordinate(source_manifest_id=source_manifest.source_manifest_id, row_index=obs.row_index))
                    if existing is not None and existing.event_id != observation.observation_id:
                        raise ProvenanceError(
                            f"series {series_id!r} source row (source_manifest_id={source_manifest.source_manifest_id!r}, row_index={obs.row_index}) is already bound to "
                            f"observation_id {existing.event_id!r}; operation {operation_id!r} would produce conflicting observation_id {observation.observation_id!r}"
                        )

            # ---- Stage 8 (per series): series dataset committed ----
            if not dry_run:
                # Append-then-read as ONE atomic, lock-held operation (never a separate
                # append() loop followed by a separate, unlocked read_observations() call)
                # -- under concurrent writers to the same series, that two-call pattern has
                # a real race window between a write completing and a read observing it,
                # which can silently compute a component manifest from a stale/incomplete
                # snapshot (confirmed via Milestone 10 Phase 4B concurrency testing).
                all_observations = observation_store.append_many_and_read_all("fred", series_id, (observation for _obs, observation in valid_rows))
                component_manifest = create_component_dataset_manifest(
                    series_id=series_id, canonical_series_name=spec.canonical_series_name, observations=tuple(all_observations), missing_count=missing_count,
                    creation_time=operation_time,
                )
                component_store.append("fred", component_manifest)
                component_manifests[series_id] = component_manifest
                availability_ids_used[series_id] = availability_policy.availability_policy_id

                for obs, observation in valid_rows:
                    record = RawSourceRecord(row_index=obs.row_index, raw_fields=_observation_to_raw_fields(obs), raw_text=f"{obs.observation_date},{obs.value_text}")
                    provenance_record = create_provenance_record(
                        source_manifest_id=source_manifest.source_manifest_id, source_row_index=obs.row_index, source_record_digest=record.record_digest(),
                        original_timestamp_text=obs.observation_date, normalized_event_time=observation.availability_time, instrument_mapping_id=normalization_ref_id,
                        resolved_instrument_id=series_id, timeframe_mapping_id=None, timezone_policy_id=availability_policy.availability_policy_id,
                        ingestion_batch_id=operation_id, event_id=observation.observation_id, dataset_id=component_manifest.component_manifest_id, recorded_time=operation_time,
                    )
                    provenance_store.append(dataset_key, provenance_record)
            else:
                component_manifest = None

            outcomes.append(SeriesOutcome(
                series_id=series_id, succeeded=True, failure_reason=None, request_manifest_id=request_manifest.request_manifest_id,
                response_manifest_id=response_manifest.response_manifest_id, source_manifest_id=source_manifest.source_manifest_id,
                parsed_row_count=len(observations_raw), valid_row_count=len(valid_rows), quarantined_row_count=len(quarantine_rows),
                missing_count=missing_count, skipped_missing_count=skipped_missing, committed_observation_count=(0 if dry_run else len(valid_rows)),
                component_manifest_id=(None if component_manifest is None else component_manifest.component_manifest_id),
            ))
        except _SeriesFailureError as failure:
            if backfill_spec.fail_fast:
                raise CollectorOrchestrationStateError(f"series {failure.series_id!r} failed under fail_fast policy: {failure.reason}") from failure
            outcomes.append(SeriesOutcome(
                series_id=series_id, succeeded=False, failure_reason=failure.reason, request_manifest_id=None, response_manifest_id=None,
                source_manifest_id=None, parsed_row_count=0, valid_row_count=0, quarantined_row_count=0, missing_count=0, skipped_missing_count=0,
                committed_observation_count=0, component_manifest_id=None,
            ))

    succeeded_series = [o for o in outcomes if o.succeeded]
    completeness_status = CompletenessStatus.COMPLETE if len(succeeded_series) == len(backfill_spec.selected_series_ids) else CompletenessStatus.PARTIAL

    if not succeeded_series:
        raise CollectorOrchestrationStateError(f"operation {operation_id!r}: no series succeeded -- nothing to commit")

    _advance(CuratedOperationStage.SERIES_METADATA_VERIFIED, {"series_count": len(backfill_spec.selected_series_ids)})
    _advance(CuratedOperationStage.REQUESTS_COMMITTED, {"request_manifest_ids": sorted(o.request_manifest_id for o in succeeded_series if o.request_manifest_id)})
    _advance(CuratedOperationStage.RESPONSES_COMMITTED, {"response_manifest_ids": sorted(o.response_manifest_id for o in succeeded_series if o.response_manifest_id)})
    _advance(CuratedOperationStage.OBSERVATIONS_PARSED, {"total_parsed": sum(o.parsed_row_count for o in outcomes)})
    _advance(CuratedOperationStage.AVAILABILITY_RESOLVED, {"total_valid": sum(o.valid_row_count for o in outcomes)})
    _advance(CuratedOperationStage.SERIES_DATASETS_COMMITTED, {"component_manifest_ids": sorted(o.component_manifest_id for o in succeeded_series if o.component_manifest_id)})

    combined_manifest_id = None
    if not dry_run:
        combined_manifest = create_combined_universe_manifest(
            curated_registry_id=registry.registry_id, backfill_plan_id=backfill_spec.backfill_plan_id, target_dataset_namespace=ns,
            component_manifests=component_manifests, availability_policy_ids_by_series=availability_ids_used, revision_policy_id=revision_policy.revision_policy_id,
            completeness_status=completeness_status, creation_time=operation_time,
        )
        combined_store.append(combined_manifest)
        combined_manifest_id = combined_manifest.combined_manifest_id
        _advance(CuratedOperationStage.COMBINED_MANIFEST_COMMITTED, {"combined_manifest_id": combined_manifest_id})
        _advance(CuratedOperationStage.RECONCILED, {"succeeded_series_count": len(succeeded_series)})
        _advance(CuratedOperationStage.VERIFIED, {"succeeded_series_count": len(succeeded_series)})
        _advance(CuratedOperationStage.COMPLETED, {"completeness_status": completeness_status})
        final_stage = CuratedOperationStage.COMPLETED
    else:
        final_stage = CuratedOperationStage.SERIES_DATASETS_COMMITTED

    return CuratedIngestionReport(
        operation_id=operation_id, target_dataset_namespace=ns, backfill_plan_id=backfill_spec.backfill_plan_id, stage=final_stage,
        completeness_status=completeness_status, series_outcomes=tuple(outcomes), combined_manifest_id=combined_manifest_id, is_dry_run=dry_run,
    )
