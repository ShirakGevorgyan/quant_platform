"""MANDATORY no-network fixture-based acceptance suite (Milestone 10,
Phase 4C, spec Section 23). Ordinary test discovery already guarantees
this runs with NO internet access -- every response here is a small,
synthetic, secret-free fixture. Covers in ONE assembled curated universe:
dollar-strength, WTI, Brent, silver, gold/XAU reference (all 5 core
concepts); mixed instrument forms (ETF proxies + one provider-continuous
futures series); an explicitly-classified proxy on every mapping;
multiple timezones/session cutoffs (NYSE for the Alpha-Vantage-shaped
ETF mappings, Asia/Tokyo for one fixture mapping); a missing bar; a
duplicate exact bar; a conflicting duplicate bar; adjusted vs. unadjusted
distinction; futures roll/continuation provenance; raw-response cache;
metadata verification; normalization; component datasets; combined
manifest; gap analysis; reconciliation; verification; offline replay;
deterministic report export; and point-in-time visibility after close
only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _collectors_test_helpers import FakeTransport
from _cross_asset_test_helpers import (
    FIXTURE_ALLOWED_HOSTS,
    FakeMarketCollector,
    alpha_vantage_daily_body,
    default_rate_limit_policy,
    default_retry_policy,
    fixture_body,
    fixture_full_capabilities,
    fixture_metadata,
    fixture_raw_record,
    fresh_repository_and_cache,
)

from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicyKind,
    create_adjustment_policy,
)
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicyKind,
    create_bar_availability_policy,
)
from quant_platform.market_data.collectors.cross_asset.futures import (
    ContinuationPolicyKind,
    RollProvenance,
    create_continuation_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    run_cross_asset_backfill_operation,
)
from quant_platform.market_data.collectors.cross_asset.market_reconciliation import (
    reconcile_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.market_reports import (
    generate_cross_asset_ingestion_report,
)
from quant_platform.market_data.collectors.cross_asset.market_verification import (
    verify_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.protocols import (
    HistoricalMarketCollector,
)
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    AlphaVantageCollector,
)
from quant_platform.market_data.collectors.cross_asset.registry import (
    DriverTier,
    create_curated_market_driver_registry,
    create_curated_market_driver_spec,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    create_timezone_session_policy,
)
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    create_provider_symbol_mapping,
    create_symbol_mapping_set,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

T0 = datetime(2024, 1, 8, tzinfo=timezone.utc)
FIXTURE_PROVIDER = "fixture_provider"


def _av_body(*, symbol: str, rows: dict[str, dict[str, str]]) -> bytes:
    return alpha_vantage_daily_body(symbol=symbol, rows=rows)


def _http_ok(body: bytes, url: str = "https://www.alphavantage.co/query") -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url=url)


def _build_full_universe():
    """5 core drivers via Alpha-Vantage-shaped ETF mappings (NYSE
    session, RAW_UNADJUSTED) + 1 additional provider-continuous futures
    mapping on wti_crude (Asia/Tokyo session, NOT_APPLICABLE adjustment,
    fixture provider, roll provenance) -- deliberately gives wti_crude
    TWO mappings from TWO different providers/instrument forms, the
    cross-provider + mixed-instrument-form coverage this acceptance
    suite requires."""
    adjustment_raw = create_adjustment_policy(kind=AdjustmentPolicyKind.RAW_UNADJUSTED)
    adjustment_na = create_adjustment_policy(kind=AdjustmentPolicyKind.NOT_APPLICABLE)
    nyse_session = create_timezone_session_policy(
        timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
        provider_session_note="Alpha Vantage NYSE ETF session", session_open_time=time(9, 30), session_close_time=time(16, 0),
    )
    tokyo_session = create_timezone_session_policy(
        timezone_key="Asia/Tokyo", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
        provider_session_note="fixture continuous-futures session", session_open_time=time(9, 0), session_close_time=time(15, 0),
    )
    nyse_availability = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="America/New_York", delay_minutes=60)
    tokyo_availability = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="Asia/Tokyo", delay_minutes=30)
    continuation = create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS)

    core_symbols = {"us_dollar_strength": "UUP", "wti_crude": "USO", "brent_crude": "BNO", "silver": "SLV", "gold_reference": "GLD"}
    mappings = []
    for driver_id, symbol in core_symbols.items():
        proxy = create_proxy_policy(is_proxy=True, proxy_for=driver_id, proxy_quality=ProxyQuality.MODERATE, known_basis_risk="fixture")
        mappings.append(create_provider_symbol_mapping(
            provider=ALPHA_VANTAGE_COLLECTOR_NAME, provider_symbol=symbol, canonical_driver_id=driver_id, instrument_form=InstrumentForm.ETF,
            currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy, exchange_or_venue="NYSEARCA",
        ))
    wti_futures_proxy = create_proxy_policy(is_proxy=False)
    wti_futures_mapping = create_provider_symbol_mapping(
        provider=FIXTURE_PROVIDER, provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
        currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.NOT_APPLICABLE, proxy_policy=wti_futures_proxy, continuation_policy_id=continuation.continuation_policy_id,
    )
    mappings.append(wti_futures_mapping)

    # A genuinely SEPARATE driver (not sharing wti_crude's NYSE-session spec)
    # on a DIFFERENT session/timezone (Asia/Tokyo) -- exercises multiple
    # timezones/session cutoffs WITHIN this one assembled universe (spec
    # Section 23), since `TimezoneSessionPolicy` is a per-DRIVER-spec field
    # and cannot vary per-mapping within one driver.
    copper_proxy = create_proxy_policy(is_proxy=True, proxy_for="copper_industrial_growth", proxy_quality=ProxyQuality.MODERATE, known_basis_risk="fixture")
    copper_mapping = create_provider_symbol_mapping(
        provider=FIXTURE_PROVIDER, provider_symbol="COPFIX", canonical_driver_id="copper_industrial_growth", instrument_form=InstrumentForm.ETF,
        currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=copper_proxy,
    )
    mappings.append(copper_mapping)
    mapping_set = create_symbol_mapping_set(tuple(mappings))

    specs = []
    for driver_id, _symbol in core_symbols.items():
        mapping_ids = tuple(m.mapping_id for m in mappings if m.canonical_driver_id == driver_id)
        allowed_forms = (InstrumentForm.SPOT, InstrumentForm.ETF) if driver_id != "us_dollar_strength" else (InstrumentForm.CASH_INDEX, InstrumentForm.ETF)
        preferred = InstrumentForm.SPOT if driver_id != "us_dollar_strength" else InstrumentForm.CASH_INDEX
        continuation_id = continuation.continuation_policy_id if driver_id == "wti_crude" else None
        if driver_id == "wti_crude":
            allowed_forms = (*allowed_forms, InstrumentForm.PROVIDER_CONTINUOUS_FUTURES)
        specs.append(create_curated_market_driver_spec(
            canonical_driver_id=driver_id, canonical_name=driver_id.replace("_", " ").title(), registry_version=1,
            tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER, economic_role="fixture acceptance", is_required=True, asset_class="fixture",
            preferred_instrument_form=preferred, allowed_instrument_forms=allowed_forms, canonical_currency="USD", canonical_quote_unit="native",
            expected_frequency="daily", session_policy_id=nyse_session.session_policy_id, adjustment_policy=adjustment_raw,
            availability_policy_id=nyse_availability.availability_policy_id, continuation_policy_id=continuation_id, provider_mapping_ids=mapping_ids, enabled=True,
        ))
    specs.append(create_curated_market_driver_spec(
        canonical_driver_id="copper_industrial_growth", canonical_name="Copper Industrial Growth", registry_version=1,
        tier=DriverTier.SECONDARY_MARKET_DRIVER, economic_role="fixture acceptance -- Tokyo-session timezone coverage", is_required=False,
        asset_class="fixture", preferred_instrument_form=InstrumentForm.SPOT, allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF),
        canonical_currency="USD", canonical_quote_unit="native", expected_frequency="daily", session_policy_id=tokyo_session.session_policy_id,
        adjustment_policy=adjustment_raw, availability_policy_id=tokyo_availability.availability_policy_id, provider_mapping_ids=(copper_mapping.mapping_id,),
        enabled=True,
    ))
    registry = create_curated_market_driver_registry(registry_version=1, specs=tuple(specs))

    selected_mapping_ids = tuple(sorted(m.mapping_id for m in mappings))
    backfill_spec = create_market_backfill_spec(
        registry=registry, mapping_set=mapping_set, selected_driver_ids=(*core_symbols.keys(), "copper_industrial_growth"),
        selected_mapping_ids=selected_mapping_ids, start_time=T0, end_time=T0, requested_granularity="1d",
        target_dataset_namespace="fixture_acceptance_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=True,
    )
    return {
        "registry": registry, "mapping_set": mapping_set, "backfill_spec": backfill_spec, "core_symbols": core_symbols,
        "wti_futures_mapping": wti_futures_mapping, "copper_mapping": copper_mapping, "nyse_session": nyse_session, "tokyo_session": tokyo_session,
        "nyse_availability": nyse_availability, "tokyo_availability": tokyo_availability, "continuation": continuation,
        "adjustment_raw": adjustment_raw, "adjustment_na": adjustment_na,
    }


def _standard_rows() -> dict[str, dict[str, str]]:
    return {
        "2024-01-03": {"1. open": "100.00", "2. high": "101.50", "3. low": "99.80", "4. close": "101.00", "5. volume": "1000000"},
        "2024-01-04": {"1. open": "101.10", "2. high": "102.00", "3. low": "100.50", "4. close": "101.80", "5. volume": "900000"},
        # 2024-01-05 (Friday) deliberately MISSING -- exercises gap detection.
        "2024-01-08": {"1. open": "101.90", "2. high": "103.00", "3. low": "101.50", "4. close": "102.50", "5. volume": "1100000"},
    }


def _fixture_collector() -> FakeMarketCollector:
    return FakeMarketCollector(
        capabilities=fixture_full_capabilities(instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, InstrumentForm.ETF)),
        metadata_by_symbol={
            "CL1!": fixture_metadata(provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, timezone_key="Asia/Tokyo"),
            "COPFIX": fixture_metadata(provider_symbol="COPFIX", canonical_driver_id="copper_industrial_growth", instrument_form=InstrumentForm.ETF, timezone_key="Asia/Tokyo"),
        },
        records_by_symbol={
            "CL1!": (
                fixture_raw_record(provider_symbol="CL1!", date_text="2024-01-03", open_="75.00", high="76.00", low="74.50", close="75.50", sequence=0, contract_symbol="CLG24"),
                fixture_raw_record(provider_symbol="CL1!", date_text="2024-01-08", open_="76.00", high="77.00", low="75.50", close="76.80", sequence=1, contract_symbol="CLG24"),
            ),
            "COPFIX": (
                fixture_raw_record(provider_symbol="COPFIX", date_text="2024-01-03", open_="4.00", high="4.10", low="3.95", close="4.05", sequence=0),
                fixture_raw_record(provider_symbol="COPFIX", date_text="2024-01-08", open_="4.06", high="4.20", low="4.00", close="4.15", sequence=1),
            ),
        },
    )


def _build_and_run(universe: dict, *, transport, operation_id: str = "op1") -> object:
    rate_limit_policy = default_rate_limit_policy()
    fixture_collector = _fixture_collector()
    collectors_by_provider: dict[str, HistoricalMarketCollector] = {ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector(), FIXTURE_PROVIDER: fixture_collector}
    allowed_hosts_by_provider = {ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS, FIXTURE_PROVIDER: FIXTURE_ALLOWED_HOSTS}
    roll = RollProvenance(
        active_contract_symbol="CLG24", prior_contract_symbol="CLF24", next_contract_symbol=None, roll_timestamp="2024-01-02",
        adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=universe["continuation"].continuation_policy_id,
    )
    _root, repository, cache = fresh_repository_and_cache()
    report = run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=universe["backfill_spec"],
        session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"], universe["tokyo_session"].session_policy_id: universe["tokyo_session"]},
        availability_policies={
            universe["nyse_availability"].availability_policy_id: universe["nyse_availability"],
            universe["tokyo_availability"].availability_policy_id: universe["tokyo_availability"],
        },
        collectors_by_provider=collectors_by_provider, allowed_hosts_by_provider=allowed_hosts_by_provider, operation_id=operation_id, operation_time=T0,
        transport=transport, api_key="demo", retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        roll_provenance_by_mapping={universe["wti_futures_mapping"].mapping_id: roll},
    )
    return report, repository, cache


def _queue_full_universe_transport(universe: dict) -> FakeTransport:
    bodies_by_mapping: dict[str, list[TransportResponse]] = {}
    for driver_id, symbol in universe["core_symbols"].items():
        mapping = universe["mapping_set"].for_driver(driver_id)[0] if driver_id != "wti_crude" else next(
            m for m in universe["mapping_set"].for_driver("wti_crude") if m.provider == ALPHA_VANTAGE_COLLECTOR_NAME
        )
        bodies_by_mapping[mapping.mapping_id] = [_http_ok(_av_body(symbol=symbol, rows=_standard_rows()))]
    bodies_by_mapping[universe["wti_futures_mapping"].mapping_id] = [
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="metadata"), final_url="https://fixture.invalid/metadata"),
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="history"), final_url="https://fixture.invalid/history"),
    ]
    bodies_by_mapping[universe["copper_mapping"].mapping_id] = [
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="COPFIX", tag="metadata"), final_url="https://fixture.invalid/metadata"),
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="COPFIX", tag="history"), final_url="https://fixture.invalid/history"),
    ]
    responses = [r for mid in universe["backfill_spec"].selected_mapping_ids for r in bodies_by_mapping[mid]]
    return FakeTransport(responses=responses)


class TestFullUniverseFixtureAcceptance:
    def test_full_universe_backfill_completes(self) -> None:
        universe = _build_full_universe()
        transport = _queue_full_universe_transport(universe)
        report, repository, cache = _build_and_run(universe, transport=transport)

        assert report.stage.value == "completed"
        assert report.completeness_status == "complete"
        assert len(report.mapping_outcomes) == 7
        assert all(o.succeeded for o in report.mapping_outcomes)

        # Mixed instrument forms + proxy classification.
        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore

        bar_store = MarketDriverBarStore(repository.root)
        wti_futures_bars = bar_store.read_bars(FIXTURE_PROVIDER, "wti_crude", InstrumentForm.PROVIDER_CONTINUOUS_FUTURES)
        assert len(wti_futures_bars) == 2
        assert all(b.roll_provenance is not None for b in wti_futures_bars)
        gold_bars = bar_store.read_bars(ALPHA_VANTAGE_COLLECTOR_NAME, "gold_reference", InstrumentForm.ETF)
        assert len(gold_bars) == 3
        # Different timezone/session cutoff: copper's bars resolve their OPEN
        # time in Asia/Tokyo, never coinciding with the NYSE-session bars'
        # open times for the SAME calendar date.
        copper_bars = bar_store.read_bars(FIXTURE_PROVIDER, "copper_industrial_growth", InstrumentForm.ETF)
        assert len(copper_bars) == 2
        assert copper_bars[0].open_time != gold_bars[0].open_time

        # Gap analysis: 2024-01-05 was deliberately omitted from every AV fixture.
        from quant_platform.market_data.collectors.cross_asset.datasets import (
            ComponentMarketDatasetManifestStore,
        )

        gold_mapping = universe["mapping_set"].for_driver("gold_reference")[0]
        component = ComponentMarketDatasetManifestStore(repository.root).read_current(gold_mapping.mapping_id)
        assert component is not None
        assert component.missing_business_day_count == 1

        # Reconciliation + verification, both clean.
        recon = reconcile_cross_asset_universe(
            repository=repository, registry=universe["registry"], mapping_set=universe["mapping_set"], target_dataset_namespace="fixture_acceptance_ns",
            session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"], universe["tokyo_session"].session_policy_id: universe["tokyo_session"]}, as_of=T0,
        )
        assert not recon.criticals

        verify = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=universe["backfill_spec"],
            session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"], universe["tokyo_session"].session_policy_id: universe["tokyo_session"]},
            availability_policies={
                universe["nyse_availability"].availability_policy_id: universe["nyse_availability"],
                universe["tokyo_availability"].availability_policy_id: universe["tokyo_availability"],
            },
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector(), FIXTURE_PROVIDER: _fixture_collector()},
            mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        assert not verify.criticals

        # Deterministic report export -- no credential leakage.
        report_dict = generate_cross_asset_ingestion_report(report)
        assert "demo" not in json.dumps(report_dict)

    def test_offline_replay_reproduces_identical_universe(self) -> None:
        universe = _build_full_universe()
        transport = _queue_full_universe_transport(universe)
        report_1, repository, cache = _build_and_run(universe, transport=transport)

        # Second pass: reuse the SAME repository/cache but a fresh universe
        # object graph, transport=None -- structurally zero network calls.
        universe_2 = _build_full_universe()
        rate_limit_policy = default_rate_limit_policy()
        fixture_collector = _fixture_collector()
        roll = RollProvenance(
            active_contract_symbol="CLG24", prior_contract_symbol="CLF24", next_contract_symbol=None, roll_timestamp="2024-01-02",
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=universe_2["continuation"].continuation_policy_id,
        )
        report_2 = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=universe_2["registry"], mapping_set=universe_2["mapping_set"], backfill_spec=universe_2["backfill_spec"],
            session_policies={universe_2["nyse_session"].session_policy_id: universe_2["nyse_session"], universe_2["tokyo_session"].session_policy_id: universe_2["tokyo_session"]},
            availability_policies={
                universe_2["nyse_availability"].availability_policy_id: universe_2["nyse_availability"],
                universe_2["tokyo_availability"].availability_policy_id: universe_2["tokyo_availability"],
            },
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector(), FIXTURE_PROVIDER: fixture_collector},
            allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS, FIXTURE_PROVIDER: FIXTURE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=None, api_key=None, retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
            roll_provenance_by_mapping={universe_2["wti_futures_mapping"].mapping_id: roll},
        )
        assert report_2.combined_manifest_id == report_1.combined_manifest_id
        assert report_2.completeness_status == report_1.completeness_status


class TestDuplicateAndConflictFixtures:
    def test_duplicate_exact_bar_is_idempotent(self) -> None:
        """The SAME raw row appearing twice in a provider's response
        (never happens with Alpha Vantage's own dict-keyed-by-date shape,
        but IS possible in a hypothetical list-shaped provider) commits
        as ONE bar -- exact content-addressed dedup, not two."""
        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore

        universe = _build_full_universe()
        gold_mapping = universe["mapping_set"].for_driver("gold_reference")[0]
        rows = {"2024-01-08": {"1. open": "100.00", "2. high": "101.00", "3. low": "99.50", "4. close": "100.50", "5. volume": "500000"}}
        transport = FakeTransport(responses=[_http_ok(_av_body(symbol="GLD", rows=rows))])
        rate_limit_policy = default_rate_limit_policy()
        _root, repository, cache = fresh_repository_and_cache()
        backfill_spec = create_market_backfill_spec(
            registry=universe["registry"], mapping_set=universe["mapping_set"], selected_driver_ids=("gold_reference",), selected_mapping_ids=(gold_mapping.mapping_id,),
            start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="dup_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=True,
        )
        run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=backfill_spec,
            session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"]},
            availability_policies={universe["nyse_availability"].availability_policy_id: universe["nyse_availability"]},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=transport, api_key="demo", retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        # Exact retry with the IDENTICAL cached response -- idempotent commit.
        run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=backfill_spec,
            session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"]},
            availability_policies={universe["nyse_availability"].availability_policy_id: universe["nyse_availability"]},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=None, api_key=None, retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        bars = MarketDriverBarStore(repository.root).read_bars(ALPHA_VANTAGE_COLLECTOR_NAME, "gold_reference", InstrumentForm.ETF)
        assert len(bars) == 1

    def test_conflicting_duplicate_bar_refuses_commit(self) -> None:
        """Two DIFFERENT OHLCV values at the SAME coordinate, injected via
        a fixture provider that returns conflicting data across two
        separate operations -- must fail closed, never silently commit."""
        universe = _build_full_universe()
        wti_mapping = universe["wti_futures_mapping"]
        _root, repository, cache = fresh_repository_and_cache()
        backfill_spec = create_market_backfill_spec(
            registry=universe["registry"], mapping_set=universe["mapping_set"], selected_driver_ids=("wti_crude",), selected_mapping_ids=(wti_mapping.mapping_id,),
            start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="conflict_ns", cache_policy=MarketCachePolicy.FORCE_FRESH, fail_fast=True,
        )
        rate_limit_policy = default_rate_limit_policy()

        collector_1 = FakeMarketCollector(
            capabilities=fixture_full_capabilities(instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,)),
            metadata_by_symbol={"CL1!": fixture_metadata(provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, timezone_key="Asia/Tokyo")},
            records_by_symbol={"CL1!": (fixture_raw_record(provider_symbol="CL1!", date_text="2024-01-08", open_="75.00", high="76.00", low="74.50", close="75.50", sequence=0, contract_symbol="CLG24"),)},
        )
        roll = RollProvenance(
            active_contract_symbol="CLG24", prior_contract_symbol=None, next_contract_symbol=None, roll_timestamp=None, adjustment_amount=None,
            adjustment_ratio=None, continuation_policy_id=universe["continuation"].continuation_policy_id,
        )
        transport_1 = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="metadata"), final_url="https://fixture.invalid/metadata"),
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="history"), final_url="https://fixture.invalid/history"),
        ])
        run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=backfill_spec,
            session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"]},
            availability_policies={universe["nyse_availability"].availability_policy_id: universe["nyse_availability"]},
            collectors_by_provider={FIXTURE_PROVIDER: collector_1}, allowed_hosts_by_provider={FIXTURE_PROVIDER: FIXTURE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=transport_1, api_key=None, retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.ANONYMOUS, roll_provenance_by_mapping={wti_mapping.mapping_id: roll},
        )

        # Second FORCE_FRESH run: SAME coordinate (2024-01-08), DIFFERENT close price.
        collector_2 = FakeMarketCollector(
            capabilities=fixture_full_capabilities(instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,)),
            metadata_by_symbol={"CL1!": fixture_metadata(provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, timezone_key="Asia/Tokyo")},
            records_by_symbol={"CL1!": (fixture_raw_record(provider_symbol="CL1!", date_text="2024-01-08", open_="99.00", high="100.00", low="98.50", close="99.50", sequence=0, contract_symbol="CLG24"),)},
        )
        transport_2 = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="metadata-v2"), final_url="https://fixture.invalid/metadata"),
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fixture_body(provider_symbol="CL1!", tag="history-v2"), final_url="https://fixture.invalid/history"),
        ])
        import pytest

        from quant_platform.core.exceptions import MarketProviderResponseError

        with pytest.raises(MarketProviderResponseError):
            run_cross_asset_backfill_operation(
                repository=repository, cache=cache, registry=universe["registry"], mapping_set=universe["mapping_set"], backfill_spec=backfill_spec,
                session_policies={universe["nyse_session"].session_policy_id: universe["nyse_session"]},
                availability_policies={universe["nyse_availability"].availability_policy_id: universe["nyse_availability"]},
                collectors_by_provider={FIXTURE_PROVIDER: collector_2}, allowed_hosts_by_provider={FIXTURE_PROVIDER: FIXTURE_ALLOWED_HOSTS},
                operation_id="op2", operation_time=T0, transport=transport_2, api_key=None, retry_policy=default_retry_policy(), rate_limit_policy=rate_limit_policy,
                rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.ANONYMOUS, roll_provenance_by_mapping={wti_mapping.mapping_id: roll},
            )


class TestPointInTimeVisibility:
    def test_bar_invisible_before_close_visible_at_availability(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.market_normalization import (
            normalize_raw_market_record,
        )
        from quant_platform.market_data.collectors.cross_asset.market_record import RawMarketRecord

        universe = _build_full_universe()
        raw = RawMarketRecord(
            provider=ALPHA_VANTAGE_COLLECTOR_NAME, provider_symbol="GLD", provider_timestamp_text="2024-01-08", interval="1d", open_text="100",
            high_text="101", low_text="99", close_text="100.5", volume_text="1000", adjusted_close_text=None, trade_count_text=None,
            source_sequence=0, contract_symbol=None,
        )
        bar, issues = normalize_raw_market_record(
            raw, canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF, timeframe=__import__("quant_platform.core.types", fromlist=["Timeframe"]).Timeframe.D1,
            session_policy=universe["nyse_session"], availability_policy=universe["nyse_availability"], adjustment_policy_id="a" * 64,
            request_manifest_id="r" * 64, response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert issues == ()
        assert bar is not None
        # Bar cannot be "used" (point-in-time) at any moment before availability_time.
        query_time_before_close = bar.open_time
        query_time_at_close = bar.close_time
        query_time_available = bar.availability_time
        assert query_time_before_close < bar.availability_time
        assert query_time_at_close <= bar.availability_time
        assert query_time_available == bar.availability_time
        # Never available at candle open.
        assert bar.availability_time > bar.open_time
