"""Shared, explicitly-imported test doubles/fixtures for the Phase 4C
cross-asset test suite (Milestone 10) -- mirrors `_curated_test_helpers.
py`'s own "no conftest.py, explicit imports" convention.

`FakeMarketCollector` is a fully synthetic `HistoricalMarketCollector`
double -- NOT shaped like Alpha Vantage's real schema -- used to exercise
code paths the one real shipped adapter cannot (futures contracts,
provider-continuous series with roll provenance, cash-index/spot forms,
adjusted-close series, multiple timezones/session cutoffs). It builds
GENUINELY SEPARATE metadata/history request manifests (unlike Alpha
Vantage's single-endpoint optimization), exercising orchestration's
two-fetch code path as well."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicy,
    AdjustmentPolicyKind,
    create_adjustment_policy,
)
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicy,
    BarAvailabilityPolicyKind,
    create_bar_availability_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_record import RawMarketRecord
from quant_platform.market_data.collectors.cross_asset.protocols import (
    MarketCollectorCapabilities,
    ProviderMetadataRecord,
)
from quant_platform.market_data.collectors.cross_asset.registry import (
    CuratedMarketDriverRegistry,
    create_curated_market_driver_registry,
    default_core_market_driver_specs,
    default_optional_market_driver_specs,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    TimezoneSessionPolicy,
    create_timezone_session_policy,
)
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    ProviderSymbolMapping,
    SymbolMappingSet,
    create_provider_symbol_mapping,
    create_symbol_mapping_set,
)
from quant_platform.market_data.collectors.rate_limit import create_rate_limit_policy
from quant_platform.market_data.collectors.request_manifest import (
    CollectorRequestManifest,
    CredentialMode,
    create_request_manifest,
)
from quant_platform.market_data.collectors.response_manifest import CollectorResponseManifest
from quant_platform.market_data.collectors.retry import create_retry_policy
from quant_platform.market_data.repository import MarketDataRepository

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)

ALPHA_VANTAGE_ETF_SYMBOLS: dict[str, tuple[str, ProxyQuality]] = {
    "us_dollar_strength": ("UUP", ProxyQuality.MODERATE),
    "wti_crude": ("USO", ProxyQuality.MODERATE),
    "brent_crude": ("BNO", ProxyQuality.MODERATE),
    "silver": ("SLV", ProxyQuality.HIGH),
    "gold_reference": ("GLD", ProxyQuality.HIGH),
    "us_equity_market_stress": ("VIXY", ProxyQuality.LOW),
    "broad_commodity_index": ("DBC", ProxyQuality.MODERATE),
    "copper_industrial_growth": ("CPER", ProxyQuality.MODERATE),
    "gold_miner_equity": ("GDX", ProxyQuality.MODERATE),
}
"""9 of the 10 curated concepts -- matches `registry.py`'s own default
spec factories. `treasury_volatility` is deliberately absent (no viable
ETF proxy this phase -- see `registry.default_optional_market_driver_specs`'s
own docstring)."""


def fresh_repository_and_cache() -> tuple[Path, MarketDataRepository, RawResponseCache]:
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def default_retry_policy():
    return create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0, 2.0))


def default_rate_limit_policy():
    return create_rate_limit_policy(max_tokens=Decimal(20), refill_rate_per_second=Decimal(5))


def default_adjustment_policy() -> AdjustmentPolicy:
    return create_adjustment_policy(kind=AdjustmentPolicyKind.RAW_UNADJUSTED)


def default_nyse_session_policy() -> TimezoneSessionPolicy:
    return create_timezone_session_policy(
        timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
        provider_session_note="test fixture NYSE ETF session", session_open_time=time(9, 30), session_close_time=time(16, 0),
    )


def default_availability_policy(*, delay_minutes: int = 60) -> BarAvailabilityPolicy:
    return create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="America/New_York", delay_minutes=delay_minutes)


def build_default_alpha_vantage_mappings(*, driver_ids: tuple[str, ...] | None = None) -> tuple[ProviderSymbolMapping, ...]:
    """Real Alpha Vantage ETF-proxy mappings for the given driver ids
    (default: all 9 supported concepts) -- mirrors `providers/
    alpha_vantage.py`'s own module-docstring-declared scope exactly."""
    selected = driver_ids if driver_ids is not None else tuple(ALPHA_VANTAGE_ETF_SYMBOLS.keys())
    mappings = []
    for driver_id in selected:
        symbol, quality = ALPHA_VANTAGE_ETF_SYMBOLS[driver_id]
        proxy = create_proxy_policy(is_proxy=True, proxy_for=driver_id, proxy_quality=quality, known_basis_risk="test fixture")
        mappings.append(create_provider_symbol_mapping(
            provider="alpha_vantage", provider_symbol=symbol, canonical_driver_id=driver_id, instrument_form=InstrumentForm.ETF,
            currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy, exchange_or_venue="NYSEARCA",
        ))
    return tuple(mappings)


def build_default_registry_and_mappings(
    *, driver_ids: tuple[str, ...] | None = None,
) -> tuple[CuratedMarketDriverRegistry, SymbolMappingSet, TimezoneSessionPolicy, BarAvailabilityPolicy]:
    """The full populated 10-concept registry (5 core + 5 optional, with
    `treasury_volatility` honestly UNSUPPORTED) wired to real Alpha
    Vantage ETF mappings for `driver_ids` (default: all 9 supported)."""
    adjustment = default_adjustment_policy()
    session = default_nyse_session_policy()
    availability = default_availability_policy()

    mappings = build_default_alpha_vantage_mappings(driver_ids=driver_ids)
    mapping_set = create_symbol_mapping_set(mappings)
    mapping_ids_by_driver = {m.canonical_driver_id: (m.mapping_id,) for m in mappings}

    core_specs = default_core_market_driver_specs(
        registry_version=1, adjustment_policy=adjustment, session_policy_id=session.session_policy_id, availability_policy_id=availability.availability_policy_id,
        provider_mapping_ids_by_driver=mapping_ids_by_driver,
    )
    optional_specs = default_optional_market_driver_specs(
        registry_version=1, adjustment_policy=adjustment, session_policy_id=session.session_policy_id, availability_policy_id=availability.availability_policy_id,
        provider_mapping_ids_by_driver=mapping_ids_by_driver,
    )
    registry = create_curated_market_driver_registry(registry_version=1, specs=core_specs + optional_specs)
    return registry, mapping_set, session, availability


def alpha_vantage_daily_body(*, symbol: str, rows: dict[str, dict[str, str]]) -> bytes:
    """`rows` keyed by `"YYYY-MM-DD"` -> `{"1. open": ..., "2. high": ..., "3. low": ..., "4. close": ..., "5. volume": ...}`."""
    return json.dumps({
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": symbol, "3. Last Refreshed": "2024-01-05", "4. Output Size": "Full size", "5. Time Zone": "US/Eastern"},
        "Time Series (Daily)": rows,
    }).encode("utf-8")


def alpha_vantage_error_body(*, key: str = "Note", message: str = "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day.") -> bytes:
    return json.dumps({key: message}).encode("utf-8")


# --------------------------------------------------------------------------
# `FakeMarketCollector` -- synthetic provider double, see module docstring.
# --------------------------------------------------------------------------
FIXTURE_PROVIDER_NAME = "fixture_provider"
FIXTURE_ALLOWED_HOSTS = frozenset({"fixture.invalid"})


def fixture_metadata(
    *, provider_symbol: str, canonical_driver_id: str, instrument_form: InstrumentForm, currency: str = "USD", timezone_key: str = "UTC",
    supported_intervals: tuple[str, ...] = ("1d",), exchange_or_venue: str | None = "FIXTURE", adjustment_supported: bool = False,
) -> ProviderMetadataRecord:
    digest = hashlib.sha256(f"{provider_symbol}:{canonical_driver_id}".encode()).hexdigest()
    return ProviderMetadataRecord(
        provider=FIXTURE_PROVIDER_NAME, provider_symbol=provider_symbol, canonical_driver_id=canonical_driver_id,
        provider_instrument_name=f"Fixture {provider_symbol}", asset_class="fixture", instrument_form=instrument_form,
        exchange_or_venue=exchange_or_venue, currency=currency, quote_unit="native", timezone_key=timezone_key,
        supported_intervals=supported_intervals, first_available_timestamp=None, last_available_timestamp=None,
        adjustment_supported=adjustment_supported, provider_metadata_digest=digest,
    )


def fixture_raw_record(
    *, provider_symbol: str, date_text: str, open_: str, high: str, low: str, close: str, volume: str | None = "1000", sequence: int = 0,
    contract_symbol: str | None = None, adjusted_close: str | None = None,
) -> RawMarketRecord:
    return RawMarketRecord(
        provider=FIXTURE_PROVIDER_NAME, provider_symbol=provider_symbol, provider_timestamp_text=date_text, interval="1d", open_text=open_,
        high_text=high, low_text=low, close_text=close, volume_text=volume, adjusted_close_text=adjusted_close, trade_count_text=None,
        source_sequence=sequence, contract_symbol=contract_symbol,
    )


@dataclass
class FakeMarketCollector:
    capabilities: MarketCollectorCapabilities
    metadata_by_symbol: dict[str, ProviderMetadataRecord] = field(default_factory=dict)
    records_by_symbol: dict[str, tuple[RawMarketRecord, ...]] = field(default_factory=dict)
    fetch_log: list[str] = field(default_factory=list)

    def provider_metadata(self) -> MarketCollectorCapabilities:
        return self.capabilities

    def supported_capabilities(self) -> MarketCollectorCapabilities:
        return self.capabilities

    def build_metadata_request(self, *, provider_symbol: str, request_time: datetime, credential_mode: CredentialMode) -> CollectorRequestManifest:
        return create_request_manifest(
            collector_name=FIXTURE_PROVIDER_NAME, collector_version="1.0.0", endpoint_host="fixture.invalid", endpoint_path="/metadata",
            canonical_query_params={"symbol": provider_symbol}, canonical_headers={}, requested_series_or_dataset=provider_symbol,
            response_format="json", timeout_policy_id="0" * 64, retry_policy_id="0" * 64, rate_limit_policy_id="0" * 64,
            credential_mode=credential_mode, request_time=request_time,
        )

    def build_history_request(self, *, provider_symbol: str, granularity: str, request_time: datetime, credential_mode: CredentialMode) -> CollectorRequestManifest:
        return create_request_manifest(
            collector_name=FIXTURE_PROVIDER_NAME, collector_version="1.0.0", endpoint_host="fixture.invalid", endpoint_path="/history",
            canonical_query_params={"symbol": provider_symbol, "granularity": granularity}, canonical_headers={},
            requested_series_or_dataset=provider_symbol, response_format="json", timeout_policy_id="0" * 64, retry_policy_id="0" * 64,
            rate_limit_policy_id="0" * 64, credential_mode=credential_mode, request_time=request_time,
        )

    def parse_metadata_response(
        self, raw_bytes: bytes, *, provider_symbol: str, canonical_driver_id: str, instrument_form: InstrumentForm,
    ) -> ProviderMetadataRecord:
        self.fetch_log.append(f"metadata:{provider_symbol}")
        return self.metadata_by_symbol[provider_symbol]

    def parse_history_response(self, raw_bytes: bytes, *, provider_symbol: str, response_manifest: CollectorResponseManifest) -> tuple[RawMarketRecord, ...]:
        self.fetch_log.append(f"history:{provider_symbol}")
        return self.records_by_symbol[provider_symbol]


def fixture_body(*, provider_symbol: str, tag: str) -> bytes:
    """Content only needs to be non-empty and deterministic -- `FakeMarketCollector`'s
    own parse methods ignore it and return pre-built records/metadata."""
    return json.dumps({"symbol": provider_symbol, "tag": tag}).encode("utf-8")


def fixture_full_capabilities(*, instrument_forms: tuple[InstrumentForm, ...]) -> MarketCollectorCapabilities:
    return MarketCollectorCapabilities(
        provider=FIXTURE_PROVIDER_NAME, candles_supported=True, quotes_supported=False, trades_supported=False, adjusted_data_supported=True,
        unadjusted_data_supported=True, corporate_actions_supported=False, futures_contracts_supported=True, continuous_futures_supported=True,
        pagination_supported=False, anonymous_access_supported=True, runtime_credential_required=False, max_interval_days_per_request=None,
        max_rows_per_page=None, supported_granularities=("1d",), supported_instrument_forms=instrument_forms,
    )
