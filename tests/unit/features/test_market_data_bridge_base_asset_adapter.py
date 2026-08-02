"""`features.market_data_bridge.base_asset_adapter`: pin verification,
conflicting-coordinate rejection, and `HistoricalDatasetLoaderProtocol`
conformance (spec Section 5)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest
from _market_data_bridge_test_helpers import (
    BASE_TIME,
    PARTITIONING,
    base_dataset_key,
    ingest_base_candles,
    make_base_binding,
    open_repository,
)

from quant_platform.core.exceptions import SourceVerificationError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.features.market_data_bridge.base_asset_adapter import (
    MarketDataBaseAssetLoader,
    resolve_base_asset_dataframe,
    verify_base_asset_binding,
)
from quant_platform.historical.loader import HistoricalDatasetLoaderProtocol, LoadRequest
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.ingestion import ingest_raw_events


class TestVerifyBaseAssetBinding:
    def test_resolves_expected_candle_count(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=24)
        candles = verify_base_asset_binding(repo, binding)
        assert len(candles) == 24
        assert [c.event_time for c in candles] == sorted(c.event_time for c in candles)

    def test_stale_pin_fails_closed(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=24)
        ingest_base_candles(repo, hours=1, batch_id="batch2", start_sequence=24)
        with pytest.raises(SourceVerificationError):
            verify_base_asset_binding(repo, binding)

    def test_no_manifest_at_all_fails_closed(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=1)
        other_binding = replace(binding, canonical_instrument_id="EURUSD", binding_id="")
        with pytest.raises(SourceVerificationError):
            verify_base_asset_binding(repo, other_binding)

    def test_zero_matching_timeframe_fails_closed(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5, timeframe=Timeframe.H1)
        wrong_timeframe_binding = replace(binding, timeframe=Timeframe.D1)
        # pinned_dataset_id still matches (same event stream), but no D1 candles exist
        with pytest.raises(SourceVerificationError):
            verify_base_asset_binding(repo, wrong_timeframe_binding)

    def test_conflicting_candle_at_same_open_time_is_rejected(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        key = base_dataset_key()
        c1 = create_candle(
            instrument_id="XAUUSD", provider="mt5", symbol="XAUUSD", event_time=BASE_TIME, timeframe=Timeframe.H1, sequence=0,
            open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"),
        )
        c2 = create_candle(
            instrument_id="XAUUSD", provider="mt5", symbol="XAUUSD", event_time=BASE_TIME, timeframe=Timeframe.H1, sequence=1,
            open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2002"), volume=Decimal("1"),
        )
        result = ingest_raw_events(repository=repo, dataset_key=key, batch_id="b1", ingestion_time=BASE_TIME, events=(c1, c2), partitioning=PARTITIONING)
        from quant_platform.features.market_data_bridge.bindings import create_base_asset_binding

        binding = create_base_asset_binding(canonical_instrument_id="XAUUSD", provider="mt5", pinned_dataset_id=result.resulting_dataset_id, timeframe=Timeframe.H1)
        with pytest.raises(SourceVerificationError):
            verify_base_asset_binding(repo, binding)


class TestResolveBaseAssetDataframe:
    def test_shape_and_range_filtering(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=48)
        df = resolve_base_asset_dataframe(repo, binding, start=BASE_TIME, end=BASE_TIME + timedelta(hours=10))
        assert list(df.columns) == list(OHLCV_COLUMNS)
        assert len(df) == 10
        assert df["open_time"].is_monotonic_increasing
        assert not df["open_time"].duplicated().any()

    def test_naive_timestamps_rejected(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        with pytest.raises(SourceVerificationError):
            resolve_base_asset_dataframe(repo, binding, start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))


class TestMarketDataBaseAssetLoaderProtocolConformance:
    def test_loader_satisfies_the_protocol(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        loader = MarketDataBaseAssetLoader(repo, binding)
        assert isinstance(loader, HistoricalDatasetLoaderProtocol)

    def test_resolve_manifest_returns_stable_dataset_id(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        loader = MarketDataBaseAssetLoader(repo, binding)
        req = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.H1, start=BASE_TIME, end=BASE_TIME + timedelta(hours=5))
        manifest_a = loader.resolve_manifest(req)
        manifest_b = loader.resolve_manifest(req)
        assert manifest_a.dataset_id == manifest_b.dataset_id == binding.pinned_dataset_id
        assert manifest_a.content_checksum == manifest_b.content_checksum

    def test_mismatched_symbol_is_rejected(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        loader = MarketDataBaseAssetLoader(repo, binding)
        req = LoadRequest(symbol="EURUSD", timeframe=Timeframe.H1, start=BASE_TIME, end=BASE_TIME + timedelta(hours=5))
        with pytest.raises(SourceVerificationError):
            loader.resolve_manifest(req)

    def test_mismatched_timeframe_is_rejected(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        loader = MarketDataBaseAssetLoader(repo, binding)
        req = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.D1, start=BASE_TIME, end=BASE_TIME + timedelta(hours=5))
        with pytest.raises(SourceVerificationError):
            loader.resolve_manifest(req)

    def test_mismatched_dataset_version_is_rejected(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=5)
        loader = MarketDataBaseAssetLoader(repo, binding)
        req = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.H1, start=BASE_TIME, end=BASE_TIME + timedelta(hours=5), dataset_version="z" * 64)
        with pytest.raises(SourceVerificationError):
            loader.resolve_manifest(req)

    def test_load_for_engine_returns_ohlcv_shape(self, tmp_path) -> None:
        repo = open_repository(tmp_path)
        binding = make_base_binding(repo, hours=10)
        loader = MarketDataBaseAssetLoader(repo, binding)
        req = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.H1, start=BASE_TIME, end=BASE_TIME + timedelta(hours=10))
        df = loader.load_for_engine(req)
        assert list(df.columns) == list(OHLCV_COLUMNS)
        assert len(df) == 10
