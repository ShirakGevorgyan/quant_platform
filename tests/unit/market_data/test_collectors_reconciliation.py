"""Reconciliation tests (Milestone 10, Phase 4A) -- cross-store
consistency scanning purely from durable evidence (no original
construction parameters needed), detecting stalled operations, missing
raw responses, digest mismatches, and orphaned repository records."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
    fred_json_body,
)

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    create_unit_mapping_spec,
)
from quant_platform.market_data.collectors.orchestration import (
    CollectorOperationStage,
    CollectorOperationStore,
    FetchMode,
    run_fred_macro_ingestion_operation,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.reconciliation import reconcile_fred_macro_dataset
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.macro import MacroEventStore, create_macro_event
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.repository import MarketDataRepository

UNIT_MAPPING = create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


def _env():
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def _run(repository, cache, operation_id: str, observations: list[dict]):
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    transport = FakeTransport(responses=[
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body(observations), final_url="https://api.stlouisfed.org/fred/series/observations"),
    ])
    return run_fred_macro_ingestion_operation(
        repository=repository, cache=cache, operation_id=operation_id, operation_time=T0, series_id="DGS10", provider="fred",
        unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
        credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport,
    )


class TestCleanOperation:
    def test_reconciles_with_zero_issues(self) -> None:
        _root, repository, cache = _env()
        _run(repository, cache, "op-recon-1", [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}])
        result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
        assert result.issues == ()


class TestStaleCheckpointAnalogue:
    def test_operation_never_reaching_verification_completed_is_stalled(self) -> None:
        root, repository, cache = _env()
        dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        CollectorOperationStore(root).advance(
            dataset_key=dataset_key, operation_id="op-stalled", content_digest="a" * 64, stage=CollectorOperationStage.REQUEST_PLANNED,
            stage_evidence={"request_manifest_id": "b" * 64}, operation_time=T0,
        )
        result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
        assert any(i.code == "stalled_operation" for i in result.warnings)


class TestMissingResponseAndDigestMismatch:
    def test_tampered_raw_bytes_produce_digest_mismatch(self) -> None:
        root, repository, cache = _env()
        report = _run(repository, cache, "op-recon-2", [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}])
        candidates = [p for p in root.rglob("body.bin") if report.response_manifest_id in str(p.parent)]
        raw_bytes_path = candidates[0]
        original = raw_bytes_path.read_bytes()
        raw_bytes_path.write_bytes(original + b"TAMPERED")
        try:
            result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
            assert any(i.code in ("digest_mismatch", "truncated_payload") for i in result.criticals)
        finally:
            raw_bytes_path.write_bytes(original)


class TestRepositoryRecordWithoutSourceEvidence:
    def test_orphaned_macro_event_is_detected(self) -> None:
        root, repository, cache = _env()
        _run(repository, cache, "op-recon-3", [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}])
        macro_store = MacroEventStore(root)
        next_seq = macro_store.next_sequence("fred", "DGS10")
        orphan = create_macro_event(series_id="DGS10", provider="fred", event_time=T0, sequence=next_seq, value=Decimal("1.23"), source_event_id="fabricated")
        macro_store.append(orphan)
        result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
        assert any(i.code == "repository_record_without_source_evidence" for i in result.criticals)


class TestReplayConsistency:
    def test_an_exact_retry_via_cached_replay_still_reconciles_cleanly(self) -> None:
        """Same `operation_id`, `FetchMode.CACHED_REPLAY` -- an idempotent
        exact retry, not a second independent operation (which would
        instead hit the provenance-conflict guard; see
        `test_collectors_orchestration.py`'s own coverage of that case)."""
        from quant_platform.market_data.collectors.orchestration import (
            run_fred_macro_ingestion_operation as run_op,
        )

        _root, repository, cache = _env()
        _run(repository, cache, "op-recon-4", [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}])
        run_op(
            repository=repository, cache=cache, operation_id="op-recon-4", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=default_retry_policy(), rate_limit_policy=default_rate_limit_policy(),
        )
        result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
        assert result.criticals == ()
