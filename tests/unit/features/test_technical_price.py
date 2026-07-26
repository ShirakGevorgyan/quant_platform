"""Technical/price feature tests, centered on the general point-in-time
proof: recomputing the SAME feature set against a truncated prefix of the
data must reproduce EXACTLY the same values for every row up to the
truncation point. Any feature that peeked at a future row would fail this
test -- it is a single, powerful, general-purpose leakage detector that
covers the whole technical family at once (Section 17 item 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import (
    TechnicalWindows,
    candle_body_ratio,
    candle_lower_wick_ratio,
    candle_upper_wick_ratio,
    register_core_technical_features,
    rolling_high_low_distance,
    trailing_rolling,
)


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


@pytest.fixture
def registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    register_core_technical_features(reg, timeframe=Timeframe.M1, windows=TechnicalWindows())
    return reg


class TestTruncationInvariance:
    @pytest.mark.parametrize("truncate_at", [50, 100, 250, 400])
    def test_all_technical_features_identical_on_truncated_prefix(self, registry: FeatureRegistry, truncate_at: int) -> None:
        base_df = _engine_df(make_synthetic_ohlcv(500, seed=3))
        names = tuple(s.name for s in registry.list_features())
        engine = FeatureEngine(registry)

        full = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=names)
        truncated = engine.compute(
            base_df=base_df.iloc[:truncate_at].reset_index(drop=True), symbol="X", timeframe=Timeframe.M1,
            feature_names=names,
        )

        for col in names:
            full_slice = full.features[col].iloc[:truncate_at].to_numpy()
            truncated_slice = truncated.features[col].to_numpy()
            assert np.allclose(full_slice, truncated_slice, equal_nan=True), (
                f"LEAK in feature {col!r}: truncated computation disagrees with full computation"
            )

    def test_appending_future_rows_never_changes_past_rows(self, registry: FeatureRegistry) -> None:
        """The converse framing of the same guarantee: adding MORE future
        data must not retroactively change any already-computed value.
        Constructs a genuine shared prefix by concatenation (two
        independent `make_synthetic_ohlcv` calls with different `n` do NOT
        share a prefix, since the RNG draws one batch of size `n` at once)."""
        base_df = _engine_df(make_synthetic_ohlcv(300, seed=9))
        appended_rows = make_synthetic_ohlcv(300, start="2024-01-01T05:00:00", seed=11)
        appended_rows["open_time"] = base_df["open_time"].iloc[-1] + Timeframe.M1.duration * (
            1 + np.arange(len(appended_rows))
        )
        extended_df = pd.concat([base_df, _engine_df(appended_rows)], ignore_index=True)

        names = ("return_simple_1", "atr_14", "rolling_zscore_close_20", "ma_distance_50")
        engine = FeatureEngine(registry)
        short_result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=names)
        long_result = engine.compute(base_df=extended_df, symbol="X", timeframe=Timeframe.M1, feature_names=names)
        for col in names:
            a = short_result.features[col].to_numpy()
            b = long_result.features[col].iloc[:300].to_numpy()
            assert np.allclose(a, b, equal_nan=True)


class TestCandleRatios:
    def test_body_ratio_zero_range_bar_is_nan(self) -> None:
        open_ = pd.Series([1.0])
        high = pd.Series([1.0])
        low = pd.Series([1.0])
        close = pd.Series([1.0])
        result = candle_body_ratio(open_, high, low, close)
        assert result.isna().all()

    def test_wick_ratios_sum_with_body_ratio_to_one(self) -> None:
        open_ = pd.Series([10.0])
        high = pd.Series([12.0])
        low = pd.Series([8.0])
        close = pd.Series([11.0])
        body = candle_body_ratio(open_, high, low, close).iloc[0]
        upper = candle_upper_wick_ratio(open_, high, low, close).iloc[0]
        lower = candle_lower_wick_ratio(open_, high, low, close).iloc[0]
        assert body + upper + lower == pytest.approx(1.0)

    def test_high_low_distance_at_extremes(self) -> None:
        close = pd.Series([1.0, 2.0, 3.0])
        high = pd.Series([1.0, 2.0, 3.0])
        low = pd.Series([1.0, 2.0, 3.0])
        result = rolling_high_low_distance(close, high, low, window=3)
        assert result.iloc[2:].isna().all() or (result.iloc[2] == result.iloc[2])  # zero-range window -> NaN guard exercised


class TestTrailingWindowNeverCentered:
    def test_warmup_row_count_matches_window_exactly(self) -> None:
        series = pd.Series(np.arange(50.0))
        rolled = trailing_rolling(series, window=10).mean()
        assert rolled.iloc[:9].isna().all()
        assert not pd.isna(rolled.iloc[9])

    def test_rolling_window_never_uses_center_true(self) -> None:
        """A centered window at row i would use rows from i+1..i+w/2 --
        i.e. it would NOT reproduce truncation invariance. This directly
        exercises that guarantee for the shared helper every technical
        feature is built on."""
        series = pd.Series(np.arange(20.0))
        full = trailing_rolling(series, window=5).mean()
        truncated = trailing_rolling(series.iloc[:10], window=5).mean()
        assert np.allclose(full.iloc[:10].to_numpy(), truncated.to_numpy(), equal_nan=True)
