"""Reporting tests (Milestone 10, Phase 4A) -- every report is
deterministic, secret-free, and re-exports of Phase 3's own
dataset-generic report functions correctly serve the macro DatasetKind."""

from __future__ import annotations

import json
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
from quant_platform.market_data.collectors.fred import build_fred_request_manifest
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    create_unit_mapping_spec,
)
from quant_platform.market_data.collectors.orchestration import FetchMode, run_fred_macro_ingestion_operation
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import create_rate_limit_policy, initial_bucket_state
from quant_platform.market_data.collectors.reconciliation import reconcile_fred_macro_dataset
from quant_platform.market_data.collectors.reports import (
    generate_fred_ingestion_operation_report,
    generate_provenance_summary_report,
    generate_quarantine_summary_report,
    generate_rate_limit_status_report,
    generate_reconciliation_report,
    generate_replay_comparison_report,
    generate_request_manifest_report,
    generate_response_manifest_report,
    generate_retry_attempts_report,
    generate_verification_report,
)
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.verification import verify_fred_macro_operation
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.quarantine import QuarantineStore
from quant_platform.market_data.repository import MarketDataRepository

SECRET = "SECRET_KEY_VALUE"
UNIT_MAPPING = create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


def _env():
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def _run(repository, cache, operation_id: str = "op-reports-1"):
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    transport = FakeTransport(responses=[
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
            {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
        ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
    ])
    report = run_fred_macro_ingestion_operation(
        repository=repository, cache=cache, operation_id=operation_id, operation_time=T0, series_id="DGS10", provider="fred",
        unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
        credential_mode=CredentialMode.API_KEY, api_key=SECRET, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport,
    )
    return report, retry_policy, rate_limit_policy, cache


class TestManifestReportsAreSecretFree:
    def test_request_and_response_manifest_reports(self) -> None:
        _root, repository, cache = _env()
        report, retry_policy, rate_limit_policy, cache = _run(repository, cache)
        request_manifest = build_fred_request_manifest(
            series_id="DGS10", observation_start=T0, observation_end=T0, response_format="json", timeout_policy_id="0" * 64,
            retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
            credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        response_manifest = cache.read_manifest(report.response_manifest_id)

        req_report = generate_request_manifest_report(request_manifest)
        resp_report = generate_response_manifest_report(response_manifest)
        assert SECRET not in json.dumps(req_report)
        assert SECRET not in json.dumps(resp_report)
        assert req_report["request_manifest_id"] == request_manifest.request_manifest_id


class TestRetryAttemptsReport:
    def test_empty_attempts(self) -> None:
        result = generate_retry_attempts_report(())
        assert result["attempt_count"] == 0
        assert result["attempts"] == []


class TestRateLimitStatusReport:
    def test_fresh_bucket_reports_full_tokens(self) -> None:
        policy = create_rate_limit_policy(max_tokens=Decimal(10), refill_rate_per_second=Decimal(1))
        state = initial_bucket_state(policy, now=T0)
        result = generate_rate_limit_status_report(state=state, policy=policy)
        assert result["current_tokens"] == "10"


class TestIngestionOperationReport:
    def test_wraps_the_report_and_is_secret_free(self) -> None:
        _root, repository, cache = _env()
        report, *_rest = _run(repository, cache)
        op_report = generate_fred_ingestion_operation_report(report)
        assert op_report["operation_id"] == "op-reports-1"
        assert SECRET not in json.dumps(op_report)


class TestReplayComparisonReport:
    def test_exact_retry_replay_is_identical(self) -> None:
        _root, repository, cache = _env()
        report, retry_policy, rate_limit_policy, cache = _run(repository, cache)
        replayed = run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-reports-1", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.API_KEY, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        )
        result = generate_replay_comparison_report(original=report, replayed=replayed)
        assert result["identical"] is True
        assert result["differences"] == {}

    def test_a_genuine_difference_is_reported(self) -> None:
        _root, repository, cache = _env()
        report, *_rest = _run(repository, cache)
        import dataclasses

        modified = dataclasses.replace(report, committed_event_count=999)
        result = generate_replay_comparison_report(original=report, replayed=modified)
        assert result["identical"] is False
        assert "committed_event_count" in result["differences"]


class TestReExportedGenericReports:
    def test_quarantine_and_provenance_summaries_work_against_macro_dataset_key(self) -> None:
        _root, repository, cache = _env()
        _run(repository, cache)
        dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        q_report = generate_quarantine_summary_report(quarantine_store=QuarantineStore(_root), dataset_key=dataset_key)
        p_report = generate_provenance_summary_report(provenance_store=ProvenanceStore(_root), dataset_key=dataset_key)
        assert q_report["total_quarantined"] == 0
        assert p_report["total_provenance_records"] == 1

    def test_reconciliation_and_verification_reports(self) -> None:
        _root, repository, cache = _env()
        _report, retry_policy, rate_limit_policy, cache = _run(repository, cache)
        dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")

        recon_result = reconcile_fred_macro_dataset(repository=repository, cache=cache, provider="fred", series_id="DGS10", as_of=T0)
        recon_report = generate_reconciliation_report(report=recon_result, dataset_key=dataset_key)
        assert recon_report["critical_count"] == 0

        verif_result = verify_fred_macro_operation(
            repository=repository, cache=cache, operation_id="op-reports-1", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
            as_of=T0, observation_start=T0, observation_end=T0, credential_mode=CredentialMode.API_KEY, request_time=T0,
            retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
        )
        verif_report = generate_verification_report(report=verif_result, dataset_key=dataset_key)
        assert verif_report["critical_count"] == 0
