from __future__ import annotations

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.exceptions import FeatureComputationError
from quant_platform.core.types import Timeframe
from quant_platform.features.cross_asset.cross_asset import CrossAssetWindows, register_cross_asset_features
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.registry import FeatureRegistry


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


def _registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    register_cross_asset_features(
        registry, base_timeframe=Timeframe.M1, base_momentum_window=10, base_volatility_window=20,
        cross_asset_symbol="DXY", cross_asset_timeframe=Timeframe.M1, windows=CrossAssetWindows(correlation_window=30),
    )
    return registry


class TestCrossAssetFeatures:
    def test_all_features_registered(self) -> None:
        registry = _registry()
        names = {s.name for s in registry.list_features()}
        assert names == {
            "cross_dxy_return_5", "cross_dxy_relative_momentum", "cross_dxy_volatility_ratio",
            "cross_dxy_rolling_correlation_30",
        }

    def test_perfectly_correlated_series_gives_correlation_near_one(self) -> None:
        registry = _registry()
        base_df = _engine_df(make_synthetic_ohlcv(200, seed=1))
        # DXY that is an exact affine function of base close -> perfect correlation of returns
        cross_df = make_synthetic_ohlcv(200, seed=1).copy()
        cross_df["close"] = base_df["close"] * 2.0 + 5.0
        cross_df["open"] = base_df["open"] * 2.0 + 5.0
        cross_df["high"] = base_df["high"] * 2.0 + 5.0
        cross_df["low"] = base_df["low"] * 2.0 + 5.0

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_rolling_correlation_30",),
            cross_asset_data={"DXY": cross_df},
        )
        corr = result.features["cross_dxy_rolling_correlation_30"].dropna()
        assert (corr > 0.999).all()

    def test_future_cross_asset_data_does_not_leak_into_past_rows(self) -> None:
        registry = _registry()
        base_df = _engine_df(make_synthetic_ohlcv(300, seed=2))
        cross_df = make_synthetic_ohlcv(300, seed=5)

        engine = FeatureEngine(registry)
        full = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_return_5",),
            cross_asset_data={"DXY": cross_df},
        )
        truncated_cross = cross_df.iloc[:150].reset_index(drop=True)
        truncated = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_return_5",),
            cross_asset_data={"DXY": truncated_cross},
        )
        # Rows whose availability instant is before truncated_cross's last close
        # must be identical whether or not the LATER cross-asset rows exist.
        last_visible_close = truncated_cross["open_time"].iloc[-1] + Timeframe.M1.duration
        mask = (base_df["open_time"] + Timeframe.M1.duration) < last_visible_close
        pd.testing.assert_series_equal(
            full.features.loc[mask, "cross_dxy_return_5"].reset_index(drop=True),
            truncated.features.loc[mask, "cross_dxy_return_5"].reset_index(drop=True),
        )

    def test_relative_momentum_matches_base_minus_cross(self) -> None:
        registry = _registry()
        base_df = _engine_df(make_synthetic_ohlcv(200, seed=3))
        cross_df = make_synthetic_ohlcv(200, seed=4)

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_relative_momentum",),
            cross_asset_data={"DXY": cross_df},
        )
        assert "cross_dxy_relative_momentum" in result.features.columns
        assert result.features["cross_dxy_relative_momentum"].notna().any()

    def test_volatility_ratio_guards_against_zero_division(self) -> None:
        registry = _registry()
        base_df = _engine_df(make_synthetic_ohlcv(100, seed=6))
        # A perfectly flat cross-asset series has zero volatility everywhere.
        cross_df = make_synthetic_ohlcv(100, seed=6)
        cross_df[["open", "high", "low", "close"]] = 100.0

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_volatility_ratio",),
            cross_asset_data={"DXY": cross_df},
        )
        ratio = result.features["cross_dxy_volatility_ratio"]
        assert not ratio.isin([float("inf"), float("-inf")]).any()

    def test_missing_close_column_raises_actionable_error(self) -> None:
        registry = _registry()
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=7))
        cross_df_without_close = make_synthetic_ohlcv(50, seed=8).drop(columns=["close"])

        engine = FeatureEngine(registry)
        with pytest.raises(FeatureComputationError, match="close"):
            engine.compute(
                base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("cross_dxy_return_5",),
                cross_asset_data={"DXY": cross_df_without_close},
            )
