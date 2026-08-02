"""Shared fixture-building helpers for `tests/unit/features/
test_market_data_bridge_*.py` -- mirrors `tests/unit/market_data/
_collectors_test_helpers.py`'s own established naming/role: not a test
file itself (no `Test*`/`test_*` collected here), just deterministic
factories every bridge test file needs to avoid re-deriving durable
`market_data` repositories from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.bindings import (
    BaseAssetDatasetBinding,
    CrossAssetDatasetBinding,
    MacroDatasetBinding,
    create_base_asset_binding,
    create_cross_asset_dataset_binding,
    create_macro_dataset_binding,
)
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.collectors.cross_asset.datasets import (
    ComponentMarketDatasetManifest,
    ComponentMarketDatasetManifestStore,
    create_component_market_dataset_manifest,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_record import (
    MarketDriverBar,
    MarketDriverBarStore,
    create_market_driver_bar,
)
from quant_platform.market_data.collectors.curated.datasets import (
    ComponentDatasetManifest,
    ComponentDatasetManifestStore,
    create_component_dataset_manifest,
)
from quant_platform.market_data.collectors.curated.macro_observation import (
    CuratedMacroObservation,
    CuratedObservationStore,
    create_curated_macro_observation,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind
from quant_platform.market_data.ingestion import IngestionResult, ingest_raw_events
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.repository import MarketDataRepository

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
PARTITIONING = PartitioningSpec(granularity=PartitionGranularity.DAILY)


def base_dataset_key(*, instrument_id: str = "XAUUSD", provider: str = "mt5") -> DatasetKey:
    return DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id=instrument_id, provider=provider)


def ingest_base_candles(
    repository: MarketDataRepository, *, instrument_id: str = "XAUUSD", provider: str = "mt5", timeframe: Timeframe = Timeframe.H1,
    hours: int = 48, batch_id: str = "batch1", start_sequence: int = 0,
) -> IngestionResult:
    key = base_dataset_key(instrument_id=instrument_id, provider=provider)
    candles = []
    for i in range(hours):
        close_val = 2000 + (i % 10) * 0.4
        candles.append(create_candle(
            instrument_id=instrument_id, provider=provider, symbol=instrument_id, event_time=BASE_TIME + timedelta(hours=i),
            timeframe=timeframe, sequence=start_sequence + i, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"),
            close=Decimal(str(close_val)), volume=Decimal("100"),
        ))
    return ingest_raw_events(repository=repository, dataset_key=key, batch_id=batch_id, ingestion_time=BASE_TIME, events=tuple(candles), partitioning=PARTITIONING)


def make_base_binding(repository: MarketDataRepository, *, instrument_id: str = "XAUUSD", provider: str = "mt5", timeframe: Timeframe = Timeframe.H1, hours: int = 48) -> BaseAssetDatasetBinding:
    result = ingest_base_candles(repository, instrument_id=instrument_id, provider=provider, timeframe=timeframe, hours=hours)
    return create_base_asset_binding(canonical_instrument_id=instrument_id, provider=provider, pinned_dataset_id=result.resulting_dataset_id, timeframe=timeframe)


@dataclass(frozen=True, slots=True)
class MacroFixture:
    observation_store: CuratedObservationStore
    manifest_store: ComponentDatasetManifestStore
    manifest: ComponentDatasetManifest
    observations: list[CuratedMacroObservation]
    binding: MacroDatasetBinding


def make_macro_fixture(
    root: Path, *, series_id: str = "DFII10", provider: str = "fred", days: int = 10,
    revision_policy_kind: RevisionPolicyKind = RevisionPolicyKind.VINTAGE_SERIES, with_missing_day: int | None = None,
    with_revision_on_day: int | None = None,
) -> MacroFixture:
    obs_store = CuratedObservationStore(str(root))
    manifest_store = ComponentDatasetManifestStore(str(root))
    observations: list[CuratedMacroObservation] = []
    for d in range(days):
        date = f"2024-01-{d + 1:02d}"
        avail = BASE_TIME + timedelta(days=d, hours=6)
        is_missing = with_missing_day is not None and d == with_missing_day
        observations.append(create_curated_macro_observation(
            series_id=series_id, canonical_series_name="10Y TIPS", target_macro_instrument_id=series_id.lower(),
            observation_date=date, value=(None if is_missing else Decimal("1.5")), is_missing=is_missing,
            normalized_unit="percent", native_unit="percent", native_frequency="daily", realtime_start=date, realtime_end=None,
            availability_time=avail, availability_policy_id="ap1", request_manifest_id="r" * 10, response_manifest_id="p" * 10,
            source_manifest_id="s" * 10, source_row_index=d,
        ))
        if with_revision_on_day is not None and d == with_revision_on_day:
            observations.append(create_curated_macro_observation(
                series_id=series_id, canonical_series_name="10Y TIPS", target_macro_instrument_id=series_id.lower(),
                observation_date=date, value=Decimal("1.75"), is_missing=False, normalized_unit="percent", native_unit="percent",
                native_frequency="daily", realtime_start=f"2024-02-{d + 1:02d}", realtime_end=None,
                availability_time=BASE_TIME + timedelta(days=30 + d), availability_policy_id="ap1", request_manifest_id="r" * 10,
                response_manifest_id="p" * 10, source_manifest_id="s" * 10, source_row_index=1000 + d,
            ))
    all_obs = obs_store.append_many_and_read_all(provider, series_id, observations)
    missing_count = sum(1 for o in all_obs if o.is_missing)
    manifest = create_component_dataset_manifest(
        series_id=series_id, canonical_series_name="10Y TIPS", observations=tuple(all_obs), missing_count=missing_count, creation_time=datetime.now(timezone.utc)
    )
    manifest_store.append(provider, manifest)
    binding = create_macro_dataset_binding(
        curated_registry_id="r" * 64, combined_universe_manifest_id="c" * 64, series_id=series_id, canonical_series_name="10Y TIPS",
        provider=provider, component_manifest_id=manifest.component_manifest_id, revision_policy_id="p" * 64,
        revision_policy_kind=revision_policy_kind, availability_policy_id="ap1", native_frequency="daily", normalized_unit="percent",
    )
    return MacroFixture(observation_store=obs_store, manifest_store=manifest_store, manifest=manifest, observations=all_obs, binding=binding)


@dataclass(frozen=True, slots=True)
class CrossAssetFixture:
    bar_store: MarketDriverBarStore
    manifest_store: ComponentMarketDatasetManifestStore
    manifest: ComponentMarketDatasetManifest
    bars: list[MarketDriverBar]
    binding: CrossAssetDatasetBinding


def make_cross_asset_fixture(
    root: Path, *, canonical_driver_id: str = "us_dollar_strength", provider: str = "alpha_vantage", provider_symbol: str = "UUP",
    mapping_id: str = "mapping1", days: int = 10, extra_delay: timedelta = timedelta(days=1),
) -> CrossAssetFixture:
    bar_store = MarketDriverBarStore(str(root))
    manifest_store = ComponentMarketDatasetManifestStore(str(root))
    bars = []
    for d in range(days):
        open_time = BASE_TIME + timedelta(days=d)
        avail = open_time + timedelta(days=1) + extra_delay
        close_val = 28 + (d % 10) * 0.04
        bars.append(create_market_driver_bar(
            canonical_driver_id=canonical_driver_id, provider=provider, provider_symbol=provider_symbol, instrument_form=InstrumentForm.ETF,
            open_time=open_time, timeframe=Timeframe.D1, open=Decimal("28"), high=Decimal("28.5"), low=Decimal("27.5"),
            close=Decimal(str(close_val)), volume=Decimal("500000"), volume_unit="shares", availability_time=avail,
            availability_policy_id="close_plus_conservative_delay", session_policy_id="nyse", adjustment_policy_id="raw_unadjusted",
            request_manifest_id="r" * 10, response_manifest_id="p" * 10, source_manifest_id="s" * 10, source_row_index=d,
        ))
    all_bars = bar_store.append_many_and_read_all(provider, canonical_driver_id, InstrumentForm.ETF, bars)
    manifest = create_component_market_dataset_manifest(
        mapping_id=mapping_id, canonical_driver_id=canonical_driver_id, provider=provider, provider_symbol=provider_symbol,
        instrument_form=InstrumentForm.ETF.value, timeframe=Timeframe.D1.value, adjustment_policy_id="raw_unadjusted",
        session_policy_id="nyse", availability_policy_id="close_plus_conservative_delay", bars=tuple(all_bars),
        missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=datetime.now(timezone.utc),
    )
    manifest_store.append(manifest)
    proxy = create_proxy_policy(is_proxy=True, proxy_for=canonical_driver_id, proxy_quality=ProxyQuality.MODERATE)
    binding = create_cross_asset_dataset_binding(
        curated_registry_id="r" * 64, combined_manifest_id="c" * 64, canonical_driver_id=canonical_driver_id, mapping_id=mapping_id,
        provider=provider, provider_symbol=provider_symbol, component_manifest_id=manifest.component_manifest_id,
        instrument_form=InstrumentForm.ETF, proxy_policy=proxy, adjustment_policy_id="raw_unadjusted", continuation_policy_id=None,
        session_policy_id="nyse", availability_policy_id="close_plus_conservative_delay", timeframe=Timeframe.D1,
    )
    return CrossAssetFixture(bar_store=bar_store, manifest_store=manifest_store, manifest=manifest, bars=all_bars, binding=binding)


def open_repository(root: Path) -> MarketDataRepository:
    return MarketDataRepository.open(str(root))
