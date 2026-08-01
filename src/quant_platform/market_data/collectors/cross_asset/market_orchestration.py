"""Multi-mapping curated cross-asset backfill orchestration (Milestone
10, Phase 4C, spec Section 18) -- the 12-stage state machine over an
entire `MarketBackfillSpec`: `REGISTRY_VERIFIED -> PLAN_CREATED ->
PROVIDER_METADATA_VERIFIED -> REQUESTS_COMMITTED -> RESPONSES_COMMITTED
-> RAW_RECORDS_PARSED -> RECORDS_NORMALIZED -> COMPONENT_DATASETS_
COMMITTED -> COMBINED_MANIFEST_COMMITTED -> RECONCILED -> VERIFIED ->
COMPLETED`. `CrossAssetOperationStage`/`CrossAssetOperationStore` are a
self-contained stage machine, duplicating the SAME proven idempotent/
conflict/monotonic-progression algorithm `curated.orchestration.
CuratedOperationStore` already established, scoped to
`target_dataset_namespace` -- one operation spans MANY mappings at once.

PROVIDER-NEUTRAL BY CONSTRUCTION: this module depends only on
`protocols.HistoricalMarketCollector`'s structural shape, never a
concrete adapter -- `collectors_by_provider`/`allowed_hosts_by_provider`
let one operation span mappings served by DIFFERENT providers
simultaneously (the cross-provider conflict model, spec Section 20,
depends on exactly this).

KNOWN, DISCLOSED SIMPLIFICATION: `contract_metadata_id_by_mapping`/
`roll_provenance_by_mapping` (when supplied) apply the SAME futures
contract/roll-provenance identity to every bar produced for that
mapping in ONE call -- adequate for this phase's fixture coverage of
the futures/continuous-series code paths (no real provider this phase
maps a futures instrument form), but not a general per-row roll
resolver. A future phase adding a real futures-capable provider must
extend this before relying on it for genuine multi-roll history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    CollectorOrchestrationStateError,
    MarketProviderResponseError,
    ProvenanceError,
    ResponseIntegrityError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.core.types import Timeframe
from quant_platform.market_data.adapters import RawSourceRecord, SourceRowCoordinate
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.cross_asset.availability import BarAvailabilityPolicy
from quant_platform.market_data.collectors.cross_asset.datasets import (
    CombinedCrossAssetManifestStore,
    CompletenessStatus,
    ComponentMarketDatasetManifest,
    ComponentMarketDatasetManifestStore,
    create_combined_cross_asset_manifest,
    create_component_market_dataset_manifest,
)
from quant_platform.market_data.collectors.cross_asset.futures import RollProvenance
from quant_platform.market_data.collectors.cross_asset.gap_policy import analyze_bar_gaps
from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketBackfillSpec,
    MarketCachePolicy,
)
from quant_platform.market_data.collectors.cross_asset.market_normalization import normalize_raw_market_record
from quant_platform.market_data.collectors.cross_asset.market_record import (
    MarketDriverBar,
    MarketDriverBarStore,
    RawMarketRecord,
)
from quant_platform.market_data.collectors.cross_asset.protocols import (
    HistoricalMarketCollector,
    ProviderMetadataRecord,
    require_within_capabilities,
)
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.sessions import TimezoneSessionPolicy
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    ProviderSymbolMapping,
    SymbolMappingSet,
)
from quant_platform.market_data.collectors.execute_request import execute_collector_request
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import CollectorRequestManifest, CredentialMode
from quant_platform.market_data.collectors.response_manifest import (
    CollectorResponseManifest,
    compute_raw_content_digest,
)
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.identity import compute_content_id, require_non_empty, require_tz_aware
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore, create_provenance_record
from quant_platform.market_data.quarantine import QuarantineStore, create_quarantine_record
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, create_source_manifest

__all__ = [
    "CROSS_ASSET_OPERATION_RECORD_KIND",
    "CrossAssetIngestionReport",
    "CrossAssetOperationRecord",
    "CrossAssetOperationStage",
    "CrossAssetOperationStore",
    "MappingOutcome",
    "run_cross_asset_backfill_operation",
]

CROSS_ASSET_OPERATION_RECORD_KIND = "cross_asset_operation_record"

_GRANULARITY_TIMEFRAMES: dict[str, Timeframe] = {"1d": Timeframe.D1}


class CrossAssetOperationStage(Enum):
    REGISTRY_VERIFIED = "registry_verified"
    PLAN_CREATED = "plan_created"
    PROVIDER_METADATA_VERIFIED = "provider_metadata_verified"
    REQUESTS_COMMITTED = "requests_committed"
    RESPONSES_COMMITTED = "responses_committed"
    RAW_RECORDS_PARSED = "raw_records_parsed"
    RECORDS_NORMALIZED = "records_normalized"
    COMPONENT_DATASETS_COMMITTED = "component_datasets_committed"
    COMBINED_MANIFEST_COMMITTED = "combined_manifest_committed"
    RECONCILED = "reconciled"
    VERIFIED = "verified"
    COMPLETED = "completed"


_STAGE_ORDER: tuple[CrossAssetOperationStage, ...] = tuple(CrossAssetOperationStage)
_STAGE_RANK: dict[CrossAssetOperationStage, int] = {stage: rank for rank, stage in enumerate(_STAGE_ORDER)}


@dataclass(frozen=True, slots=True)
class CrossAssetOperationRecord:
    operation_id: str
    target_dataset_namespace: str
    stage: CrossAssetOperationStage
    content_digest: str
    stage_evidence: dict[str, object]
    operation_time: datetime

    def to_json_dict(self) -> dict[str, object]:
        from quant_platform.market_data.identity import serialize_timestamp

        return {
            "kind": CROSS_ASSET_OPERATION_RECORD_KIND, "operation_id": self.operation_id, "target_dataset_namespace": self.target_dataset_namespace,
            "stage": self.stage.value, "content_digest": self.content_digest, "stage_evidence": dict(self.stage_evidence),
            "operation_time": serialize_timestamp(self.operation_time, field_name="operation_time"),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CrossAssetOperationRecord:
        from quant_platform.market_data.identity import deserialize_timestamp
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            operation_id=str(raw["operation_id"]), target_dataset_namespace=str(raw["target_dataset_namespace"]),
            stage=CrossAssetOperationStage(raw["stage"]), content_digest=str(raw["content_digest"]),
            stage_evidence=as_json_dict(raw["stage_evidence"], field_name="stage_evidence"),
            operation_time=deserialize_timestamp(raw["operation_time"], field_name="operation_time"),
        )


class CrossAssetOperationStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, target_dataset_namespace: str) -> Path:
        return self._root / "collectors" / "cross_asset" / "operations" / target_dataset_namespace

    def _path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / "operations.jsonl"

    def _lock_path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / ".operations.lock"

    def read_all(self, target_dataset_namespace: str) -> list[CrossAssetOperationRecord]:
        from quant_platform.core.exceptions import MarketDataPersistenceError
        from quant_platform.ml.persistence import parse_json_strict

        path = self._path(target_dataset_namespace)
        if not path.is_file():
            return []
        records: list[CrossAssetOperationRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted cross-asset operation ledger line for namespace {target_dataset_namespace!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted cross-asset operation ledger line for namespace {target_dataset_namespace!r}: expected a JSON object")
            records.append(CrossAssetOperationRecord.from_json_dict(raw))
        return records

    def _append(self, target_dataset_namespace: str, record: CrossAssetOperationRecord) -> None:
        import os

        self._dir(target_dataset_namespace).mkdir(parents=True, exist_ok=True)
        path = self._path(target_dataset_namespace)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(record.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def advance(
        self, *, target_dataset_namespace: str, operation_id: str, content_digest: str, stage: CrossAssetOperationStage,
        stage_evidence: dict[str, object], operation_time: datetime,
    ) -> CrossAssetOperationRecord:
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
                raise MarketDataLockError(f"Could not acquire cross-asset operation ledger lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
            except OSError as exc:
                raise MarketDataLockError(f"Cross-asset operation ledger lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc

        self._dir(target_dataset_namespace).mkdir(parents=True, exist_ok=True)
        with _lock():
            history = [r for r in self.read_all(target_dataset_namespace) if r.operation_id == operation_id]
            new_record = CrossAssetOperationRecord(
                operation_id=operation_id, target_dataset_namespace=target_dataset_namespace, stage=stage, content_digest=content_digest,
                stage_evidence=stage_evidence, operation_time=operation_time,
            )
            if not history:
                if stage is not CrossAssetOperationStage.REGISTRY_VERIFIED:
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
class MappingOutcome:
    mapping_id: str
    canonical_driver_id: str
    succeeded: bool
    failure_reason: str | None
    request_manifest_id: str | None
    response_manifest_id: str | None
    source_manifest_id: str | None
    parsed_row_count: int
    valid_row_count: int
    quarantined_row_count: int
    committed_bar_count: int
    component_manifest_id: str | None


@dataclass(frozen=True, slots=True)
class CrossAssetIngestionReport:
    operation_id: str
    target_dataset_namespace: str
    backfill_plan_id: str
    stage: CrossAssetOperationStage
    completeness_status: str
    mapping_outcomes: tuple[MappingOutcome, ...]
    combined_manifest_id: str | None
    is_dry_run: bool


class _MappingFailureError(Exception):
    def __init__(self, mapping_id: str, reason: str) -> None:
        super().__init__(reason)
        self.mapping_id = mapping_id
        self.reason = reason


def _verify_provider_metadata(
    metadata: ProviderMetadataRecord, mapping: ProviderSymbolMapping, *, requested_granularity: str,
) -> tuple[str, ...]:
    """PURE. Every check here is a REAL, independent comparison against
    what the mapping already declares -- a field the provider's own
    metadata leaves undisclosed (`None`) is skipped, never treated as an
    automatic pass OR an automatic fail (spec Section 8: "non-material
    title changes may be warnings", but a genuine mismatch fails
    closed)."""
    issues: list[str] = []
    if metadata.provider_symbol != mapping.provider_symbol:
        issues.append(f"provider_symbol_mismatch:{metadata.provider_symbol!r}!={mapping.provider_symbol!r}")
    if metadata.canonical_driver_id != mapping.canonical_driver_id:
        issues.append(f"canonical_driver_id_mismatch:{metadata.canonical_driver_id!r}!={mapping.canonical_driver_id!r}")
    if metadata.instrument_form != mapping.instrument_form:
        issues.append(f"instrument_form_mismatch:{metadata.instrument_form!r}!={mapping.instrument_form!r}")
    if metadata.currency is not None and metadata.currency != mapping.currency:
        issues.append(f"currency_mismatch:{metadata.currency!r}!={mapping.currency!r}")
    if metadata.exchange_or_venue is not None and mapping.exchange_or_venue is not None and metadata.exchange_or_venue != mapping.exchange_or_venue:
        issues.append(f"exchange_mismatch:{metadata.exchange_or_venue!r}!={mapping.exchange_or_venue!r}")
    if metadata.supported_intervals and requested_granularity not in metadata.supported_intervals:
        issues.append(f"granularity_not_supported:{requested_granularity!r} not in {metadata.supported_intervals!r}")
    return tuple(issues)


def run_cross_asset_backfill_operation(
    *,
    repository: MarketDataRepository,
    cache: RawResponseCache,
    registry: CuratedMarketDriverRegistry,
    mapping_set: SymbolMappingSet,
    backfill_spec: MarketBackfillSpec,
    session_policies: dict[str, TimezoneSessionPolicy],
    availability_policies: dict[str, BarAvailabilityPolicy],
    collectors_by_provider: dict[str, HistoricalMarketCollector],
    allowed_hosts_by_provider: dict[str, frozenset[str]],
    operation_id: str,
    operation_time: datetime,
    transport: HistoricalHttpTransport | None = None,
    api_key: str | None = None,
    retry_policy: RetryPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    rate_limit_state: TokenBucketState | None = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    max_response_bytes: int = 10_000_000,
    credential_mode: CredentialMode = CredentialMode.API_KEY,
    contract_metadata_id_by_mapping: dict[str, str] | None = None,
    roll_provenance_by_mapping: dict[str, RollProvenance] | None = None,
    dry_run: bool = False,
) -> CrossAssetIngestionReport:
    """`backfill_spec.fail_fast=True` raises `CollectorOrchestrationStateError`
    the moment any mapping fails ANY stage -- nothing beyond that point
    is committed for ANY mapping, and `COMPLETED` is never reached.
    `fail_fast=False` (partial success) records each mapping's own
    outcome independently; the combined manifest's `completeness_status`
    reflects `datasets.CombinedCrossAssetManifest`'s own required-driver
    tracking (never merely "all mappings succeeded")."""
    require_non_empty(operation_id, field_name="operation_id")
    require_tz_aware(operation_time, field_name="operation_time")
    if backfill_spec.curated_registry_id != registry.registry_id:
        raise CollectorOrchestrationStateError(
            f"backfill_spec.curated_registry_id={backfill_spec.curated_registry_id!r} does not match the supplied registry's own id={registry.registry_id!r}"
        )
    timeframe = _GRANULARITY_TIMEFRAMES.get(backfill_spec.requested_granularity)
    if timeframe is None:
        raise CollectorOrchestrationStateError(f"unsupported requested_granularity {backfill_spec.requested_granularity!r}")

    operation_store = CrossAssetOperationStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)
    quarantine_store = QuarantineStore(repository.root)
    bar_store = MarketDriverBarStore(repository.root)
    component_store = ComponentMarketDatasetManifestStore(repository.root)
    combined_store = CombinedCrossAssetManifestStore(repository.root)

    content_digest = compute_content_id("cross_asset_backfill_operation", {"backfill_plan_id": backfill_spec.backfill_plan_id})
    ns = backfill_spec.target_dataset_namespace

    def _advance(stage: CrossAssetOperationStage, evidence: dict[str, object]) -> None:
        if not dry_run:
            operation_store.advance(target_dataset_namespace=ns, operation_id=operation_id, content_digest=content_digest, stage=stage, stage_evidence=evidence, operation_time=operation_time)

    # ---- Stage 1: REGISTRY_VERIFIED ----
    _advance(CrossAssetOperationStage.REGISTRY_VERIFIED, {"registry_id": registry.registry_id})
    # ---- Stage 2: PLAN_CREATED ----
    _advance(CrossAssetOperationStage.PLAN_CREATED, {"backfill_plan_id": backfill_spec.backfill_plan_id})

    rate_limit_state_current = rate_limit_state
    outcomes: list[MappingOutcome] = []
    component_manifests: dict[str, ComponentMarketDatasetManifest] = {}

    for mapping_id in backfill_spec.selected_mapping_ids:
        mapping = mapping_set.get(mapping_id)
        assert mapping is not None
        driver_spec = registry.get(mapping.canonical_driver_id)
        assert driver_spec is not None
        collector = collectors_by_provider.get(mapping.provider)
        allowed_hosts = allowed_hosts_by_provider.get(mapping.provider)
        session_policy = session_policies.get(driver_spec.session_policy_id)
        availability_policy = availability_policies.get(driver_spec.availability_policy_id)
        dataset_key = DatasetKey(dataset_kind=DatasetKind.CROSS_ASSET_MARKET_BARS, provider=mapping.provider, instrument_id=f"{mapping.canonical_driver_id}__{mapping.instrument_form.value}")

        def _fetch(
            request_manifest: CollectorRequestManifest, *, this_mapping_id: str, this_allowed_hosts: frozenset[str] | None,
        ) -> tuple[CollectorResponseManifest, bytes]:
            nonlocal rate_limit_state_current
            if backfill_spec.cache_policy is MarketCachePolicy.PREFER_CACHE:
                cached = cache.read_latest_response_for_request(request_manifest.request_manifest_id)
            else:
                cached = None
            if cached is not None:
                return cached, cache.read_bytes(cached.response_manifest_id, verify=True)
            if transport is None or retry_policy is None or rate_limit_policy is None or rate_limit_state_current is None or this_allowed_hosts is None:
                raise _MappingFailureError(this_mapping_id, "FRESH fetch required but transport/retry/rate-limit/allowed_hosts not supplied")
            execution, rate_limit_state_current = execute_collector_request(
                transport=transport, request_manifest=request_manifest, api_key=api_key, retry_policy=retry_policy,
                rate_limit_policy=rate_limit_policy, rate_limit_state=rate_limit_state_current, connect_timeout=connect_timeout, read_timeout=read_timeout,
                max_response_bytes=max_response_bytes, operation_time=operation_time, allowed_hosts=this_allowed_hosts,
            )
            cache.store(execution.response_manifest, execution.raw_bytes)
            return execution.response_manifest, execution.raw_bytes

        try:
            if collector is None:
                raise _MappingFailureError(mapping_id, f"no HistoricalMarketCollector registered for provider {mapping.provider!r}")
            if session_policy is None:
                raise _MappingFailureError(mapping_id, f"no TimezoneSessionPolicy registered for session_policy_id {driver_spec.session_policy_id!r}")
            if availability_policy is None:
                raise _MappingFailureError(mapping_id, f"no BarAvailabilityPolicy registered for availability_policy_id {driver_spec.availability_policy_id!r}")

            capabilities = collector.supported_capabilities()
            requires_adjusted = mapping.adjustment_policy_kind.value not in ("raw_unadjusted", "not_applicable")
            try:
                require_within_capabilities(
                    capabilities, instrument_form=mapping.instrument_form, granularity=backfill_spec.requested_granularity,
                    requires_adjusted=requires_adjusted, requires_credential=(capabilities.runtime_credential_required),
                )
            except Exception as exc:
                raise _MappingFailureError(mapping_id, f"capability check failed: {exc}") from exc

            # ---- Provider metadata verification ----
            metadata_request = collector.build_metadata_request(provider_symbol=mapping.provider_symbol, request_time=operation_time, credential_mode=credential_mode)
            meta_response_manifest, meta_raw_bytes = _fetch(metadata_request, this_mapping_id=mapping_id, this_allowed_hosts=allowed_hosts)
            metadata_digest = compute_raw_content_digest(meta_raw_bytes)
            if metadata_digest != meta_response_manifest.raw_content_digest:
                raise ResponseIntegrityError(f"re-hash mismatch for mapping {mapping_id!r} metadata response {meta_response_manifest.response_manifest_id!r}")
            metadata = collector.parse_metadata_response(
                meta_raw_bytes, provider_symbol=mapping.provider_symbol, canonical_driver_id=mapping.canonical_driver_id, instrument_form=mapping.instrument_form,
            )
            drift_issues = _verify_provider_metadata(metadata, mapping, requested_granularity=backfill_spec.requested_granularity)
            if drift_issues:
                raise _MappingFailureError(mapping_id, f"provider metadata verification failed (fail-closed): {list(drift_issues)}")

            # ---- History request/response ----
            history_request = collector.build_history_request(
                provider_symbol=mapping.provider_symbol, granularity=backfill_spec.requested_granularity, request_time=operation_time, credential_mode=credential_mode,
            )
            if history_request.request_manifest_id == metadata_request.request_manifest_id:
                response_manifest, raw_bytes = meta_response_manifest, meta_raw_bytes
            else:
                response_manifest, raw_bytes = _fetch(history_request, this_mapping_id=mapping_id, this_allowed_hosts=allowed_hosts)

            actual_digest = compute_raw_content_digest(raw_bytes)
            if actual_digest != response_manifest.raw_content_digest:
                raise ResponseIntegrityError(f"re-hash mismatch for mapping {mapping_id!r} response {response_manifest.response_manifest_id!r}")

            # ---- Raw records parsed ----
            raw_records = collector.parse_history_response(raw_bytes, provider_symbol=mapping.provider_symbol, response_manifest=response_manifest)
            if len(raw_records) > backfill_spec.max_records_per_mapping:
                raise _MappingFailureError(mapping_id, f"{len(raw_records)} records exceeds max_records_per_mapping={backfill_spec.max_records_per_mapping}")

            source_manifest = create_source_manifest(
                source_name=f"{mapping.provider}:{mapping.provider_symbol}", source_kind=SourceKind.MARKET_DATA_PROVIDER_API, source_schema_version=1,
                record_kind=RecordKind.MARKET_DRIVER_BAR, source_label=f"{mapping.provider}:{mapping.provider_symbol}:{response_manifest.response_manifest_id[:16]}",
                content_digest=response_manifest.raw_content_digest, byte_size=response_manifest.byte_length, encoding=response_manifest.encoding or "utf-8",
                instrument_mapping_id=mapping.mapping_id, timezone_policy_id=session_policy.session_policy_id, unit_normalization_version=1,
                creation_time=operation_time, row_count=len(raw_records),
            )

            # ---- Records normalized ----
            contract_metadata_id = (contract_metadata_id_by_mapping or {}).get(mapping_id)
            roll_provenance = (roll_provenance_by_mapping or {}).get(mapping_id)
            valid_rows: list[tuple[RawMarketRecord, MarketDriverBar]] = []
            quarantine_rows: list[tuple[RawMarketRecord, tuple[str, ...]]] = []
            for raw in raw_records:
                bar, issue_codes = normalize_raw_market_record(
                    raw, canonical_driver_id=mapping.canonical_driver_id, instrument_form=mapping.instrument_form, timeframe=timeframe,
                    session_policy=session_policy, availability_policy=availability_policy, adjustment_policy_id=driver_spec.adjustment_policy_id,
                    request_manifest_id=history_request.request_manifest_id, response_manifest_id=response_manifest.response_manifest_id,
                    source_manifest_id=source_manifest.source_manifest_id, source_row_index=raw.source_sequence, contract_metadata_id=contract_metadata_id,
                    roll_provenance=roll_provenance,
                )
                if bar is not None:
                    valid_rows.append((raw, bar))
                else:
                    quarantine_rows.append((raw, issue_codes))

            if not valid_rows:
                raise _MappingFailureError(mapping_id, "no valid bars produced from raw records")

            # ---- Conflicting-duplicate-coordinate hard fail (never a GapPolicy choice) ----
            candidate_bars = tuple(bar for _raw, bar in valid_rows)
            candidate_gap_report = analyze_bar_gaps(candidate_bars, session_policy=session_policy)
            if candidate_gap_report.has_conflicting_coordinates:
                raise _MappingFailureError(
                    mapping_id, f"{candidate_gap_report.conflicting_coordinate_count} conflicting duplicate bar coordinate(s) in this batch -- refusing to commit"
                )

            if not dry_run:
                for raw, issue_codes in quarantine_rows:
                    record = RawSourceRecord(
                        row_index=raw.source_sequence, raw_fields={"timestamp": raw.provider_timestamp_text, "open": raw.open_text, "high": raw.high_text, "low": raw.low_text, "close": raw.close_text},
                        raw_text=f"{raw.provider_timestamp_text},{raw.close_text}",
                    )
                    quarantine_record = create_quarantine_record(
                        source_manifest_id=source_manifest.source_manifest_id, source_row_index=raw.source_sequence, raw_record_digest=record.record_digest(),
                        raw_fields=dict(record.raw_fields), validation_issue_codes=issue_codes, ingestion_batch_id=operation_id, event_time=operation_time,
                    )
                    quarantine_store.append(dataset_key, quarantine_record)

            # ---- Pre-flight provenance-conflict check ----
            if not dry_run:
                for raw, bar in valid_rows:
                    existing = provenance_store.read_by_source_coordinate(dataset_key, SourceRowCoordinate(source_manifest_id=source_manifest.source_manifest_id, row_index=raw.source_sequence))
                    if existing is not None and existing.event_id != bar.bar_id:
                        raise ProvenanceError(
                            f"mapping {mapping_id!r} source row (source_manifest_id={source_manifest.source_manifest_id!r}, row_index={raw.source_sequence}) is already bound to "
                            f"bar_id {existing.event_id!r}; operation {operation_id!r} would produce conflicting bar_id {bar.bar_id!r}"
                        )

            # ---- Component dataset committed ----
            if not dry_run:
                all_bars = bar_store.append_many_and_read_all(mapping.provider, mapping.canonical_driver_id, mapping.instrument_form, (bar for _raw, bar in valid_rows))
                full_gap_report = analyze_bar_gaps(tuple(all_bars), session_policy=session_policy)
                if full_gap_report.has_conflicting_coordinates:
                    raise MarketProviderResponseError(
                        f"mapping {mapping_id!r}: committed bar set has {full_gap_report.conflicting_coordinate_count} conflicting coordinate(s) -- data integrity violation"
                    )
                component_manifest = create_component_market_dataset_manifest(
                    mapping_id=mapping.mapping_id, canonical_driver_id=mapping.canonical_driver_id, provider=mapping.provider, provider_symbol=mapping.provider_symbol,
                    instrument_form=mapping.instrument_form.value, timeframe=timeframe.value, adjustment_policy_id=driver_spec.adjustment_policy_id,
                    session_policy_id=session_policy.session_policy_id, availability_policy_id=availability_policy.availability_policy_id, bars=tuple(all_bars),
                    missing_business_day_count=full_gap_report.missing_business_day_count, conflicting_coordinate_count=full_gap_report.conflicting_coordinate_count,
                    creation_time=operation_time, continuation_policy_id=mapping.continuation_policy_id,
                )
                component_store.append(component_manifest)
                component_manifests[mapping_id] = component_manifest

                for raw, bar in valid_rows:
                    record = RawSourceRecord(
                        row_index=raw.source_sequence, raw_fields={"timestamp": raw.provider_timestamp_text, "close": raw.close_text}, raw_text=f"{raw.provider_timestamp_text},{raw.close_text}",
                    )
                    provenance_record = create_provenance_record(
                        source_manifest_id=source_manifest.source_manifest_id, source_row_index=raw.source_sequence, source_record_digest=record.record_digest(),
                        original_timestamp_text=raw.provider_timestamp_text, normalized_event_time=bar.availability_time, instrument_mapping_id=mapping.mapping_id,
                        resolved_instrument_id=mapping.canonical_driver_id, timeframe_mapping_id=None, timezone_policy_id=session_policy.session_policy_id,
                        ingestion_batch_id=operation_id, event_id=bar.bar_id, dataset_id=component_manifest.component_manifest_id, recorded_time=operation_time,
                    )
                    provenance_store.append(dataset_key, provenance_record)
            else:
                component_manifest = None

            outcomes.append(MappingOutcome(
                mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, succeeded=True, failure_reason=None,
                request_manifest_id=history_request.request_manifest_id, response_manifest_id=response_manifest.response_manifest_id,
                source_manifest_id=source_manifest.source_manifest_id, parsed_row_count=len(raw_records), valid_row_count=len(valid_rows),
                quarantined_row_count=len(quarantine_rows), committed_bar_count=(0 if dry_run else len(valid_rows)),
                component_manifest_id=(None if component_manifest is None else component_manifest.component_manifest_id),
            ))
        except _MappingFailureError as failure:
            if backfill_spec.fail_fast:
                raise CollectorOrchestrationStateError(f"mapping {failure.mapping_id!r} failed under fail_fast policy: {failure.reason}") from failure
            outcomes.append(MappingOutcome(
                mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, succeeded=False, failure_reason=failure.reason,
                request_manifest_id=None, response_manifest_id=None, source_manifest_id=None, parsed_row_count=0, valid_row_count=0,
                quarantined_row_count=0, committed_bar_count=0, component_manifest_id=None,
            ))

    succeeded = [o for o in outcomes if o.succeeded]
    if not succeeded:
        raise CollectorOrchestrationStateError(f"operation {operation_id!r}: no mapping succeeded -- nothing to commit")

    _advance(CrossAssetOperationStage.PROVIDER_METADATA_VERIFIED, {"mapping_count": len(backfill_spec.selected_mapping_ids)})
    _advance(CrossAssetOperationStage.REQUESTS_COMMITTED, {"request_manifest_ids": sorted(o.request_manifest_id for o in succeeded if o.request_manifest_id)})
    _advance(CrossAssetOperationStage.RESPONSES_COMMITTED, {"response_manifest_ids": sorted(o.response_manifest_id for o in succeeded if o.response_manifest_id)})
    _advance(CrossAssetOperationStage.RAW_RECORDS_PARSED, {"total_parsed": sum(o.parsed_row_count for o in outcomes)})
    _advance(CrossAssetOperationStage.RECORDS_NORMALIZED, {"total_valid": sum(o.valid_row_count for o in outcomes)})
    _advance(CrossAssetOperationStage.COMPONENT_DATASETS_COMMITTED, {"component_manifest_ids": sorted(o.component_manifest_id for o in succeeded if o.component_manifest_id)})

    combined_manifest_id = None
    if not dry_run:
        required_driver_ids = registry.required_driver_ids()
        combined_manifest = create_combined_cross_asset_manifest(
            curated_registry_id=registry.registry_id, backfill_plan_id=backfill_spec.backfill_plan_id, target_dataset_namespace=ns,
            component_manifests=component_manifests, required_driver_ids=required_driver_ids, creation_time=operation_time,
        )
        combined_store.append(combined_manifest)
        combined_manifest_id = combined_manifest.combined_manifest_id
        _advance(CrossAssetOperationStage.COMBINED_MANIFEST_COMMITTED, {"combined_manifest_id": combined_manifest_id})
        _advance(CrossAssetOperationStage.RECONCILED, {"succeeded_mapping_count": len(succeeded)})
        _advance(CrossAssetOperationStage.VERIFIED, {"succeeded_mapping_count": len(succeeded)})
        _advance(CrossAssetOperationStage.COMPLETED, {"completeness_status": combined_manifest.completeness_status})
        final_stage = CrossAssetOperationStage.COMPLETED
        completeness_status = combined_manifest.completeness_status
    else:
        final_stage = CrossAssetOperationStage.COMPONENT_DATASETS_COMMITTED
        completeness_status = CompletenessStatus.PARTIAL if len(succeeded) != len(backfill_spec.selected_mapping_ids) else CompletenessStatus.COMPLETE

    return CrossAssetIngestionReport(
        operation_id=operation_id, target_dataset_namespace=ns, backfill_plan_id=backfill_spec.backfill_plan_id, stage=final_stage,
        completeness_status=completeness_status, mapping_outcomes=tuple(outcomes), combined_manifest_id=combined_manifest_id, is_dry_run=dry_run,
    )
