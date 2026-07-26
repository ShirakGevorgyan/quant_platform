from __future__ import annotations

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.multi_timeframe import (
    MultiTimeframeWindows,
    _augmented_higher_df,
    register_multi_timeframe_features,
)
from quant_platform.features.registry import FeatureRegistry


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


class TestRegistrationValidation:
    def test_higher_timeframe_must_be_strictly_coarser(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ValueError, match="strictly coarser"):
            register_multi_timeframe_features(registry, base_timeframe=Timeframe.H1, higher_timeframe=Timeframe.M1)

    def test_equal_timeframes_rejected(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ValueError):
            register_multi_timeframe_features(registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.M1)


class TestIncompleteBarNeverVisible:
    def test_still_forming_h1_bar_not_reflected_in_close(self) -> None:
        registry = FeatureRegistry()
        register_multi_timeframe_features(registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.H1)
        base_df = _engine_df(make_synthetic_ohlcv(200, freq_minutes=1, start="2024-01-01T08:00:00", seed=1))
        higher_df = _engine_df(make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00", seed=2))

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("htf_H1_close",),
            higher_timeframe_data={Timeframe.H1: higher_df},
        )
        # base row at 09:59 (close 10:00) must see the 08:00-09:00 or 09:00-10:00 bar
        # depending on exact minute; specifically at 09:00 (close 09:01), the bar opened
        # at 09:00 has NOT closed yet (closes 10:00), so the visible close must be the
        # PRIOR (08:00) bar's close, not the still-forming 09:00 bar's.
        row_at_0900 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T09:00:00Z")][0]
        visible_close = result.features["htf_H1_close"].iloc[row_at_0900]
        still_forming_bar_close = higher_df.loc[higher_df["open_time"] == pd.Timestamp("2024-01-01T09:00:00Z"), "close"].iloc[0]
        assert visible_close != still_forming_bar_close

    def test_bar_becomes_visible_exactly_at_its_close(self) -> None:
        registry = FeatureRegistry()
        register_multi_timeframe_features(registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.H1)
        base_df = _engine_df(make_synthetic_ohlcv(200, freq_minutes=1, start="2024-01-01T08:00:00", seed=1))
        higher_df = _engine_df(make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00", seed=2))

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("htf_H1_close",),
            higher_timeframe_data={Timeframe.H1: higher_df},
        )
        # base row opening at 09:59 closes at 10:00, exactly when the 09:00 H1 bar closes.
        row_at_0959 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T09:59:00Z")][0]
        visible_close = result.features["htf_H1_close"].iloc[row_at_0959]
        bar_0900_close = higher_df.loc[higher_df["open_time"] == pd.Timestamp("2024-01-01T09:00:00Z"), "close"].iloc[0]
        assert visible_close == bar_0900_close


class TestReturnComputedAtNativeCadence:
    def test_return_reflects_htf_bar_to_bar_change_not_base_bar_repeats(self) -> None:
        """The bug this design deliberately avoids: computing a return on
        the UPSAMPLED (repeated-until-next-update) aligned series would
        measure change over N BASE bars of mostly-repeated values, not N
        HTF bars. Hand-craft H1 closes with a known 3-bar-ago return and
        confirm the feature reports THAT, not something diluted by
        repetition."""
        higher_df = _engine_df(make_synthetic_ohlcv(6, freq_minutes=60, start="2024-01-01T00:00:00", seed=3))
        # Force known close values.
        higher_df["close"] = [100.0, 101.0, 102.0, 103.0, 110.0, 111.0]
        windows = MultiTimeframeWindows(return_window=3, volatility_window=3, trend_window=3)
        augmented = _augmented_higher_df(higher_df, windows)
        # return_3 at row 4 (close=110) vs row 1 (close=101): (110-101)/101
        expected = (110.0 - 101.0) / 101.0
        assert augmented["return_3"].iloc[4] == pytest.approx(expected)


class TestSecondsSinceClose:
    def test_elapsed_time_grows_within_the_same_covering_bar(self) -> None:
        registry = FeatureRegistry()
        register_multi_timeframe_features(registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.H1)
        base_df = _engine_df(make_synthetic_ohlcv(180, freq_minutes=1, start="2024-01-01T08:00:00", seed=1))
        higher_df = _engine_df(make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00", seed=2))

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("htf_H1_seconds_since_close",),
            higher_timeframe_data={Timeframe.H1: higher_df},
        )
        elapsed = result.features["htf_H1_seconds_since_close"]
        # Both rows are covered by the SAME H1 bar (08:00-09:00, visible from
        # 09:00 through just before 10:00) -- elapsed must grow monotonically
        # within that window.
        row_0901 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T09:01:00Z")][0]
        row_0930 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T09:30:00Z")][0]
        assert elapsed.iloc[row_0901] < elapsed.iloc[row_0930]

    def test_elapsed_time_resets_when_a_new_bar_closes(self) -> None:
        registry = FeatureRegistry()
        register_multi_timeframe_features(registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.H1)
        base_df = _engine_df(make_synthetic_ohlcv(180, freq_minutes=1, start="2024-01-01T08:00:00", seed=1))
        higher_df = _engine_df(make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00", seed=2))

        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("htf_H1_seconds_since_close",),
            higher_timeframe_data={Timeframe.H1: higher_df},
        )
        elapsed = result.features["htf_H1_seconds_since_close"]
        # Row "09:59" closes at 10:00 -- exactly when the 09:00-10:00 H1 bar
        # closes, so elapsed resets to 0 there. Row "10:00" closes at 10:01,
        # one base bar into the NEW covering window, so elapsed grows again.
        row_0959 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T09:59:00Z")][0]
        row_1000 = base_df.index[base_df["open_time"] == pd.Timestamp("2024-01-01T10:00:00Z")][0]
        assert elapsed.iloc[row_0959] == 0.0
        assert elapsed.iloc[row_1000] == 60.0
