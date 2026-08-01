"""Tests for `reconcile_cross_asset_universe`, `verify_cross_asset_universe`,
and `market_reports.py` (Milestone 10, Phase 4C, spec Sections 26-28)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _collectors_test_helpers import FakeTransport
from _cross_asset_test_helpers import (
    build_default_registry_and_mappings,
    default_rate_limit_policy,
    default_retry_policy,
    fresh_repository_and_cache,
)

from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    run_cross_asset_backfill_operation,
)
from quant_platform.market_data.collectors.cross_asset.market_reconciliation import (
    PRICE_TOLERANCE_RATIO,
    reconcile_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.market_reports import (
    generate_capability_assessment_report,
    generate_combined_cross_asset_report,
    generate_component_market_dataset_report,
    generate_cross_asset_backfill_plan_report,
    generate_cross_asset_ingestion_report,
    generate_cross_asset_reconciliation_report,
    generate_cross_asset_registry_report,
    generate_cross_asset_verification_report,
    generate_gap_analysis_report,
    generate_mapping_collection_report,
    generate_symbol_mapping_report,
)
from quant_platform.market_data.collectors.cross_asset.market_verification import (
    verify_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    AlphaVantageCollector,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


def _http_ok(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="https://www.alphavantage.co/query")


def _gld_body() -> bytes:
    import json as _json

    return _json.dumps({
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "GLD", "3. Last Refreshed": "2024-01-05", "4. Output Size": "Full size", "5. Time Zone": "US/Eastern"},
        "Time Series (Daily)": {
            "2024-01-03": {"1. open": "190.00", "2. high": "191.50", "3. low": "189.80", "4. close": "191.00", "5. volume": "1000000"},
            "2024-01-04": {"1. open": "191.10", "2. high": "192.00", "3. low": "190.50", "4. close": "191.80", "5. volume": "900000"},
            "2024-01-05": {"1. open": "191.90", "2. high": "193.00", "3. low": "191.50", "4. close": "192.50", "5. volume": "1100000"},
        },
    }).encode("utf-8")


def _run_and_get(*, repository, cache):
    registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
    mapping = mapping_set.for_driver("gold_reference")[0]
    backfill_spec = create_market_backfill_spec(
        registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference",), selected_mapping_ids=(mapping.mapping_id,),
        start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="test_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=True,
    )
    transport = FakeTransport(responses=[_http_ok(_gld_body())])
    rate_limit_policy = default_rate_limit_policy()
    report = run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
        session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
        collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
        operation_id="op1", operation_time=T0, transport=transport, api_key="demo", retry_policy=default_retry_policy(),
        rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
    )
    return report, registry, mapping_set, session, avail, backfill_spec


class TestReconciliation:
    def test_clean_universe_has_no_criticals(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        _report, registry, mapping_set, session, _avail, _spec = _run_and_get(repository=repository, cache=cache)
        result = reconcile_cross_asset_universe(
            repository=repository, registry=registry, mapping_set=mapping_set, target_dataset_namespace="test_ns",
            session_policies={session.session_policy_id: session}, as_of=T0,
        )
        assert not result.criticals

    def test_missing_combined_manifest_is_critical(self) -> None:
        _root, repository, _cache = fresh_repository_and_cache()
        registry, mapping_set, session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        result = reconcile_cross_asset_universe(
            repository=repository, registry=registry, mapping_set=mapping_set, target_dataset_namespace="nonexistent_ns",
            session_policies={session.session_policy_id: session}, as_of=T0,
        )
        assert any(i.code == "combined_manifest_missing" for i in result.criticals)

    def test_price_tolerance_ratio_is_conservative(self) -> None:
        assert __import__("decimal").Decimal("0.005") == PRICE_TOLERANCE_RATIO


class TestVerification:
    def test_clean_universe_has_no_criticals(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        report, registry, mapping_set, session, avail, backfill_spec = _run_and_get(repository=repository, cache=cache)
        result = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        assert not result.criticals

    def test_forged_registry_identity_detected(self) -> None:
        from dataclasses import replace

        _root, repository, cache = fresh_repository_and_cache()
        report, registry, mapping_set, session, avail, backfill_spec = _run_and_get(repository=repository, cache=cache)
        forged_registry = replace(registry, registry_id="f" * 64)
        result = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=forged_registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        assert any(i.code == "forged_registry_identity" for i in result.criticals)

    def test_corrupted_raw_bytes_with_coherent_rehash_detected(self) -> None:
        """Adversarial audit item: tampered raw bytes whose OUTER manifest
        still self-verifies must be caught by the re-hash check (13),
        never silently accepted."""
        _root, repository, cache = fresh_repository_and_cache()
        report, registry, mapping_set, session, avail, backfill_spec = _run_and_get(repository=repository, cache=cache)
        outcome = report.mapping_outcomes[0]
        assert outcome.response_manifest_id is not None
        response_dir = repository.root / "collectors" / "raw_responses" / outcome.response_manifest_id
        body_path = response_dir / "body.bin"
        original = body_path.read_bytes()
        body_path.write_bytes(original + b"TAMPERED")
        result = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        assert any(i.code == "raw_content_digest_mismatch" for i in result.criticals)
        body_path.write_bytes(original)


class TestReports:
    def test_registry_report_never_contains_credentials(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        _report, registry, _mapping_set, _session, _avail, _spec = _run_and_get(repository=repository, cache=cache)
        report_dict = generate_cross_asset_registry_report(registry)
        assert "demo" not in json.dumps(report_dict)
        assert report_dict["driver_count"] == 10

    def test_symbol_mapping_report_shape(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        _report, _registry, mapping_set, _session, _avail, _spec = _run_and_get(repository=repository, cache=cache)
        mapping = mapping_set.for_driver("gold_reference")[0]
        report_dict = generate_symbol_mapping_report(mapping)
        assert report_dict["provider_symbol"] == "GLD"

    def test_capability_report_shape(self) -> None:
        collector = AlphaVantageCollector()
        report_dict = generate_capability_assessment_report("alpha_vantage", collector.supported_capabilities())
        assert report_dict["provider"] == "alpha_vantage"
        assert "GC" not in json.dumps(report_dict)

    def test_backfill_plan_report_never_leaks_credentials(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        _report, _registry, _mapping_set, _session, _avail, backfill_spec = _run_and_get(repository=repository, cache=cache)
        report_dict = generate_cross_asset_backfill_plan_report(backfill_spec)
        assert "demo" not in json.dumps(report_dict)

    def test_ingestion_report_shape(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        report, *_rest = _run_and_get(repository=repository, cache=cache)
        report_dict = generate_cross_asset_ingestion_report(report)
        assert report_dict["stage"] == "completed"
        assert len(report_dict["mapping_outcomes"]) == 1
        assert "demo" not in json.dumps(report_dict)

    def test_mapping_collection_report_shape(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        report, *_rest = _run_and_get(repository=repository, cache=cache)
        report_dict = generate_mapping_collection_report(report.mapping_outcomes[0])
        assert report_dict["committed_bar_count"] == 3

    def test_component_and_combined_reports_never_leak_credentials(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.datasets import (
            CombinedCrossAssetManifestStore,
            ComponentMarketDatasetManifestStore,
        )

        _root, repository, cache = fresh_repository_and_cache()
        _report, _registry, mapping_set, *_rest = _run_and_get(repository=repository, cache=cache)
        mapping = mapping_set.for_driver("gold_reference")[0]
        component = ComponentMarketDatasetManifestStore(repository.root).read_current(mapping.mapping_id)
        combined = CombinedCrossAssetManifestStore(repository.root).read_current("test_ns")
        assert component is not None and combined is not None
        component_dict = generate_component_market_dataset_report(component)
        combined_dict = generate_combined_cross_asset_report(combined)
        assert "demo" not in json.dumps(component_dict)
        assert "demo" not in json.dumps(combined_dict)

    def test_gap_analysis_report_shape(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.gap_policy import analyze_bar_gaps
        from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore

        _root, repository, cache = fresh_repository_and_cache()
        _report, _registry, mapping_set, session, *_rest = _run_and_get(repository=repository, cache=cache)
        mapping = mapping_set.for_driver("gold_reference")[0]
        bars = MarketDriverBarStore(repository.root).read_bars(mapping.provider, mapping.canonical_driver_id, InstrumentForm.ETF)
        gap_report = analyze_bar_gaps(tuple(bars), session_policy=session)
        report_dict = generate_gap_analysis_report(gap_report)
        assert report_dict["conflicting_coordinate_count"] == 0

    def test_reconciliation_and_verification_report_wrappers(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        report, registry, mapping_set, session, avail, backfill_spec = _run_and_get(repository=repository, cache=cache)
        recon_result = reconcile_cross_asset_universe(
            repository=repository, registry=registry, mapping_set=mapping_set, target_dataset_namespace="test_ns",
            session_policies={session.session_policy_id: session}, as_of=T0,
        )
        verify_result = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        recon_dict = generate_cross_asset_reconciliation_report(report=recon_result, target_dataset_namespace="test_ns")
        verify_dict = generate_cross_asset_verification_report(report=verify_result, target_dataset_namespace="test_ns")
        assert recon_dict["critical_count"] == 0
        assert verify_dict["critical_count"] == 0
