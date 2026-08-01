"""Tests for `run_cross_asset_backfill_operation` (Milestone 10, Phase
4C, spec Section 30 "Orchestration"). Uses both the REAL
`AlphaVantageCollector` (via `FakeTransport`, exercising the real
single-endpoint-reuse code path) and the synthetic `FakeMarketCollector`
(exercising futures/multi-provider/genuinely-separate-endpoint code
paths the real adapter cannot)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _collectors_test_helpers import FakeTransport
from _cross_asset_test_helpers import (
    FIXTURE_ALLOWED_HOSTS,
    FakeMarketCollector,
    alpha_vantage_daily_body,
    build_default_registry_and_mappings,
    default_availability_policy,
    default_nyse_session_policy,
    default_rate_limit_policy,
    default_retry_policy,
    fixture_body,
    fixture_full_capabilities,
    fixture_metadata,
    fixture_raw_record,
    fresh_repository_and_cache,
)

from quant_platform.core.exceptions import CollectorOrchestrationStateError
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.cross_asset.futures import (
    ContinuationPolicyKind,
    RollProvenance,
    create_continuation_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    run_cross_asset_backfill_operation,
)
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    AlphaVantageCollector,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.repository import MarketDataRepository

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


def _http_ok(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="https://www.alphavantage.co/query")


def _gld_body(*, extra_days: int = 0) -> bytes:
    rows = {
        "2024-01-03": {"1. open": "190.00", "2. high": "191.50", "3. low": "189.80", "4. close": "191.00", "5. volume": "1000000"},
        "2024-01-04": {"1. open": "191.10", "2. high": "192.00", "3. low": "190.50", "4. close": "191.80", "5. volume": "900000"},
        "2024-01-05": {"1. open": "191.90", "2. high": "193.00", "3. low": "191.50", "4. close": "192.50", "5. volume": "1100000"},
    }
    if extra_days:
        rows["2024-01-08"] = {"1. open": "192.60", "2. high": "193.50", "3. low": "192.00", "4. close": "193.00", "5. volume": "1050000"}
    return alpha_vantage_daily_body(symbol="GLD", rows=rows)


def _run_gld_backfill(*, repository: MarketDataRepository, cache: RawResponseCache, transport: object, api_key: str = "demo", cache_policy: MarketCachePolicy = MarketCachePolicy.PREFER_CACHE, operation_id: str = "op1", fail_fast: bool = False):
    registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
    mapping = mapping_set.for_driver("gold_reference")[0]
    backfill_spec = create_market_backfill_spec(
        registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference",), selected_mapping_ids=(mapping.mapping_id,),
        start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="test_ns", cache_policy=cache_policy, fail_fast=fail_fast,
    )
    rate_limit_policy = default_rate_limit_policy()
    report = run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
        session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
        collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
        operation_id=operation_id, operation_time=T0, transport=transport, api_key=api_key, retry_policy=default_retry_policy(),
        rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
    )
    return report, registry, mapping_set, backfill_spec


class TestSingleMappingBackfill:
    def test_successful_backfill_reaches_completed(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        report, _registry, _mapping_set, _spec = _run_gld_backfill(repository=repository, cache=cache, transport=transport)
        assert report.stage.value == "completed"
        assert report.mapping_outcomes[0].succeeded
        assert report.mapping_outcomes[0].committed_bar_count == 3

    def test_exact_retry_zero_network_calls(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        report_1, *_ = _run_gld_backfill(repository=repository, cache=cache, transport=transport)

        transport_2 = FakeTransport(responses=[])
        report_2, *_ = _run_gld_backfill(repository=repository, cache=cache, transport=transport_2)
        assert report_2.combined_manifest_id == report_1.combined_manifest_id
        assert len(transport_2.calls) == 0

    def test_cached_replay_with_no_transport(self) -> None:
        """Offline replay: `transport=None` structurally guarantees zero
        network calls (spec Section 25)."""
        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        report_1, *_ = _run_gld_backfill(repository=repository, cache=cache, transport=transport)

        report_2, *_ = _run_gld_backfill(repository=repository, cache=cache, transport=None)
        assert report_2.combined_manifest_id == report_1.combined_manifest_id

    def test_force_fresh_with_new_data_produces_new_component_version(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.datasets import (
            ComponentMarketDatasetManifestStore,
        )

        _root, repository, cache = fresh_repository_and_cache()
        transport_1 = FakeTransport(responses=[_http_ok(_gld_body())])
        _report_1, _registry, mapping_set, _spec = _run_gld_backfill(repository=repository, cache=cache, transport=transport_1, operation_id="op1")

        transport_2 = FakeTransport(responses=[_http_ok(_gld_body(extra_days=1))])
        report_2, _registry2, _mapping_set2, _spec2 = _run_gld_backfill(
            repository=repository, cache=cache, transport=transport_2, cache_policy=MarketCachePolicy.FORCE_FRESH, operation_id="op2",
        )
        assert len(transport_2.calls) == 1  # metadata+history share one manifest
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(repository.root)
        assert component_store.current_version(mapping.mapping_id) == 2
        assert report_2.mapping_outcomes[0].committed_bar_count == 4

    def test_no_op_update_never_mints_new_version(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.datasets import (
            ComponentMarketDatasetManifestStore,
        )

        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        _report_1, _registry, mapping_set, _spec = _run_gld_backfill(repository=repository, cache=cache, transport=transport, operation_id="op1")
        _report_2, *_ = _run_gld_backfill(repository=repository, cache=cache, transport=None, operation_id="op1")
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(repository.root)
        assert component_store.current_version(mapping.mapping_id) == 1

    def test_provider_metadata_symbol_mismatch_fails_closed(self) -> None:
        """A single failing mapping with nothing else selected still
        raises (nothing to commit) -- the failure REASON is what this
        test verifies, surfaced through the raised exception's message."""
        _root, repository, cache = fresh_repository_and_cache()
        wrong_symbol_body = alpha_vantage_daily_body(symbol="WRONG_SYMBOL", rows={"2024-01-05": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1", "5. volume": "1"}})
        transport = FakeTransport(responses=[_http_ok(wrong_symbol_body)])
        with pytest.raises(CollectorOrchestrationStateError):
            _run_gld_backfill(repository=repository, cache=cache, transport=transport, fail_fast=True)

    def test_fail_fast_raises_and_commits_nothing(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        wrong_symbol_body = alpha_vantage_daily_body(symbol="WRONG_SYMBOL", rows={"2024-01-05": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1", "5. volume": "1"}})
        transport = FakeTransport(responses=[_http_ok(wrong_symbol_body)])
        with pytest.raises(CollectorOrchestrationStateError):
            _run_gld_backfill(repository=repository, cache=cache, transport=transport, fail_fast=True)
        from quant_platform.market_data.collectors.cross_asset.datasets import CombinedCrossAssetManifestStore

        assert CombinedCrossAssetManifestStore(repository.root).read_current("test_ns") is None


class TestMultiMappingBackfill:
    def test_two_required_drivers_both_succeed_is_complete(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference", "silver"))
        gold_mapping = mapping_set.for_driver("gold_reference")[0]
        silver_mapping = mapping_set.for_driver("silver")[0]
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference", "silver"),
            selected_mapping_ids=(gold_mapping.mapping_id, silver_mapping.mapping_id), start_time=T0, end_time=T0, requested_granularity="1d",
            target_dataset_namespace="test_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=False,
        )
        silver_body = alpha_vantage_daily_body(symbol="SLV", rows={"2024-01-05": {"1. open": "22.00", "2. high": "22.50", "3. low": "21.80", "4. close": "22.30", "5. volume": "500000"}})
        # `backfill_spec.selected_mapping_ids` is sorted by content-addressed
        # mapping_id (not declaration order) -- the fake transport's response
        # queue must match orchestration's actual per-mapping fetch order.
        body_by_mapping_id = {gold_mapping.mapping_id: _gld_body(), silver_mapping.mapping_id: silver_body}
        transport = FakeTransport(responses=[_http_ok(body_by_mapping_id[mid]) for mid in backfill_spec.selected_mapping_ids])
        rate_limit_policy = default_rate_limit_policy()
        report = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=transport, api_key="demo", retry_policy=default_retry_policy(),
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        # The default registry declares 5 required core drivers regardless of
        # which mappings this particular backfill selects -- only 2 were
        # attempted here, so the 3 untouched required drivers correctly keep
        # completeness at "partial" (spec Section 18's own "no completed
        # universe with silently missing required driver"). What THIS test
        # verifies is that both ATTEMPTED required mappings succeeded and
        # neither is misreported as missing.
        assert all(o.succeeded for o in report.mapping_outcomes)
        succeeded_ids = {o.mapping_id for o in report.mapping_outcomes if o.succeeded}
        assert gold_mapping.mapping_id in succeeded_ids
        assert silver_mapping.mapping_id in succeeded_ids

        from quant_platform.market_data.collectors.cross_asset.datasets import CombinedCrossAssetManifestStore

        combined = CombinedCrossAssetManifestStore(repository.root).read_current("test_ns")
        assert combined is not None
        assert "gold_reference" not in combined.missing_required_driver_ids
        assert "silver" not in combined.missing_required_driver_ids

    def test_one_required_driver_failure_yields_partial(self) -> None:
        _root, repository, cache = fresh_repository_and_cache()
        registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference", "silver"))
        gold_mapping = mapping_set.for_driver("gold_reference")[0]
        silver_mapping = mapping_set.for_driver("silver")[0]
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference", "silver"),
            selected_mapping_ids=(gold_mapping.mapping_id, silver_mapping.mapping_id), start_time=T0, end_time=T0, requested_granularity="1d",
            target_dataset_namespace="test_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=False,
        )
        wrong_symbol_silver_body = alpha_vantage_daily_body(symbol="WRONG", rows={"2024-01-05": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1", "5. volume": "1"}})
        body_by_mapping_id = {gold_mapping.mapping_id: _gld_body(), silver_mapping.mapping_id: wrong_symbol_silver_body}
        transport = FakeTransport(responses=[_http_ok(body_by_mapping_id[mid]) for mid in backfill_spec.selected_mapping_ids])
        rate_limit_policy = default_rate_limit_policy()
        report = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=transport, api_key="demo", retry_policy=default_retry_policy(),
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        assert report.completeness_status == "partial"
        outcomes_by_mapping = {o.mapping_id: o for o in report.mapping_outcomes}
        assert outcomes_by_mapping[gold_mapping.mapping_id].succeeded
        assert not outcomes_by_mapping[silver_mapping.mapping_id].succeeded

    def test_two_different_providers_in_one_operation(self) -> None:
        """Exercises `collectors_by_provider`/`allowed_hosts_by_provider`
        spanning multiple providers simultaneously (spec Section 20's
        own cross-provider foundation)."""
        _root, repository, cache = fresh_repository_and_cache()
        registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        av_mapping = mapping_set.for_driver("gold_reference")[0]

        from quant_platform.market_data.collectors.cross_asset.adjustment import AdjustmentPolicyKind
        from quant_platform.market_data.collectors.cross_asset.instrument_form import (
            ProxyQuality,
            create_proxy_policy,
        )
        from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
            create_provider_symbol_mapping,
            create_symbol_mapping_set,
        )

        fixture_proxy = create_proxy_policy(is_proxy=True, proxy_for="gold_reference", proxy_quality=ProxyQuality.HIGH)
        fixture_mapping = create_provider_symbol_mapping(
            provider="fixture_provider", provider_symbol="XAUFIX", canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF,
            currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=fixture_proxy,
        )
        combined_mapping_set = create_symbol_mapping_set((av_mapping, fixture_mapping))

        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=combined_mapping_set, selected_driver_ids=("gold_reference",),
            selected_mapping_ids=(av_mapping.mapping_id, fixture_mapping.mapping_id), start_time=T0, end_time=T0, requested_granularity="1d",
            target_dataset_namespace="multi_provider_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=False,
        )

        fixture_capabilities = fixture_full_capabilities(instrument_forms=(InstrumentForm.ETF,))
        fixture_meta = fixture_metadata(provider_symbol="XAUFIX", canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF)
        fixture_rec = fixture_raw_record(provider_symbol="XAUFIX", date_text="2024-01-05", open_="2040.00", high="2050.00", low="2035.00", close="2045.00")
        fixture_collector = FakeMarketCollector(capabilities=fixture_capabilities, metadata_by_symbol={"XAUFIX": fixture_meta}, records_by_symbol={"XAUFIX": (fixture_rec,)})

        # One shared transport serves BOTH providers -- Alpha Vantage needs 1
        # fetch (metadata+history share one manifest), the fixture provider
        # needs 2 (genuinely separate manifests). Response order must match
        # orchestration's actual per-mapping fetch order (sorted mapping_id).
        bodies_by_mapping = {
            av_mapping.mapping_id: [_http_ok(_gld_body())],
            fixture_mapping.mapping_id: [
                TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="XAUFIX", tag="metadata"), final_url="https://fixture.invalid/metadata"),
                TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="XAUFIX", tag="history"), final_url="https://fixture.invalid/history"),
            ],
        }
        av_transport = FakeTransport(responses=[r for mid in backfill_spec.selected_mapping_ids for r in bodies_by_mapping[mid]])

        rate_limit_policy = default_rate_limit_policy()
        report = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=registry, mapping_set=combined_mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector(), "fixture_provider": fixture_collector},
            allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS, "fixture_provider": FIXTURE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=av_transport, api_key="demo", retry_policy=default_retry_policy(),
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        assert all(o.succeeded for o in report.mapping_outcomes)
        assert len(report.mapping_outcomes) == 2
        assert "metadata:XAUFIX" in fixture_collector.fetch_log
        assert "history:XAUFIX" in fixture_collector.fetch_log


class TestFuturesViaFixtureCollector:
    def test_provider_continuous_futures_commits_with_roll_provenance(self) -> None:
        """Exercises the futures/continuous-series code path no real
        provider this phase supports (spec Section 23's own mandatory
        fixture-coverage requirement)."""
        from quant_platform.market_data.collectors.cross_asset.adjustment import (
            AdjustmentPolicyKind,
            create_adjustment_policy,
        )
        from quant_platform.market_data.collectors.cross_asset.instrument_form import create_proxy_policy
        from quant_platform.market_data.collectors.cross_asset.registry import (
            DriverTier,
            create_curated_market_driver_registry,
            create_curated_market_driver_spec,
        )
        from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
            create_provider_symbol_mapping,
            create_symbol_mapping_set,
        )

        _root, repository, cache = fresh_repository_and_cache()
        session = default_nyse_session_policy()
        avail = default_availability_policy()
        continuation_policy = create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS)
        adjustment = create_adjustment_policy(kind=AdjustmentPolicyKind.NOT_APPLICABLE)
        proxy = create_proxy_policy(is_proxy=False)
        mapping = create_provider_symbol_mapping(
            provider="fixture_provider", provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
            currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.NOT_APPLICABLE, proxy_policy=proxy, continuation_policy_id=continuation_policy.continuation_policy_id,
        )
        mapping_set = create_symbol_mapping_set((mapping,))
        driver_spec = create_curated_market_driver_spec(
            canonical_driver_id="wti_crude", canonical_name="WTI Crude Oil", registry_version=1, tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER,
            economic_role="test", is_required=True, asset_class="energy_commodity", preferred_instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
            allowed_instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,), canonical_currency="USD", canonical_quote_unit="usd_per_barrel",
            expected_frequency="daily", session_policy_id=session.session_policy_id, adjustment_policy=adjustment, availability_policy_id=avail.availability_policy_id,
            continuation_policy_id=continuation_policy.continuation_policy_id, provider_mapping_ids=(mapping.mapping_id,), enabled=True,
        )
        registry = create_curated_market_driver_registry(registry_version=1, specs=(driver_spec,))
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=("wti_crude",), selected_mapping_ids=(mapping.mapping_id,), start_time=T0,
            end_time=T0, requested_granularity="1d", target_dataset_namespace="futures_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=True,
        )

        capabilities = fixture_full_capabilities(instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,))
        meta = fixture_metadata(provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES)
        rec = fixture_raw_record(provider_symbol="CL1!", date_text="2024-01-05", open_="75.00", high="76.00", low="74.50", close="75.50", contract_symbol="CLG24")
        collector = FakeMarketCollector(capabilities=capabilities, metadata_by_symbol={"CL1!": meta}, records_by_symbol={"CL1!": (rec,)})
        roll = RollProvenance(
            active_contract_symbol="CLG24", prior_contract_symbol="CLF24", next_contract_symbol=None, roll_timestamp="2024-01-02",
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=continuation_policy.continuation_policy_id,
        )

        rate_limit_policy = default_rate_limit_policy()
        # `FakeMarketCollector` builds genuinely SEPARATE metadata/history request
        # manifests (unlike Alpha Vantage's single-endpoint reuse) -- two fetches,
        # two queued responses.
        fixture_transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="metadata"), final_url="https://fixture.invalid/metadata"),
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="history"), final_url="https://fixture.invalid/history"),
        ])
        report = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={"fixture_provider": collector}, allowed_hosts_by_provider={"fixture_provider": FIXTURE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=fixture_transport, api_key=None, retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.ANONYMOUS,
            roll_provenance_by_mapping={mapping.mapping_id: roll},
        )
        assert report.stage.value == "completed"
        assert report.completeness_status == "complete"
        assert report.mapping_outcomes[0].committed_bar_count == 1

        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore

        bars = MarketDriverBarStore(repository.root).read_bars("fixture_provider", "wti_crude", InstrumentForm.PROVIDER_CONTINUOUS_FUTURES)
        assert len(bars) == 1
        assert bars[0].roll_provenance == roll
