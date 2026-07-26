"""Tests for `historical.resampling.resample_ohlcv`.

Golden-master expected values below are computed BY HAND from the
hand-constructed input series (arithmetic sequences with known closed-form
min/max/first/last), not by re-deriving or re-running the production
aggregation logic -- see each test's inline comment for the hand
computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_platform.core.exceptions import ResamplingError
from quant_platform.core.types import Timeframe
from quant_platform.historical.resampling import DerivedBarPolicy, resample_ohlcv

UTC = "UTC"


def _linear_frame(n: int, start: str = "2024-01-03T00:00:00") -> pd.DataFrame:
    """A deterministic M1 series where every field is a simple linear
    function of row index, so aggregates have an easy closed-form value."""
    ot = pd.date_range(start, periods=n, freq="1min", tz=UTC)
    opens = np.arange(n, dtype=np.float64) + 2000.0
    return pd.DataFrame(
        {
            "open_time": ot,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.5,
            "tick_volume": np.full(n, 10, dtype=np.int64),
            "real_volume": np.full(n, 3, dtype=np.int64),
            "spread": np.full(n, 20, dtype=np.int64),
        }
    )


class TestGoldenMasterSingleHour:
    """120 minutes of M1 -> 2 complete H1 bars. For hour 0 (rows 0-59):
    open = opens[0] = 2000.0 (hand-picked: row index 0 -> 2000 + 0).
    high = max(highs[0:60]) = highs[59] = 2000 + 59 + 1 = 2060.0 (the
    series is strictly increasing, so the max is always the last element).
    low = min(lows[0:60]) = lows[0] = 2000 - 1 = 1999.0 (strictly
    increasing series -> min is always the first element).
    close = closes[59] = 2000 + 59 + 0.5 = 2059.5.
    tick_volume = 60 bars * 10 = 600. real_volume = 60 * 3 = 180.
    spread = mean(20, 20, ..., 20) = 20 (constant input).
    """

    def test_hour_zero_aggregation_matches_hand_computed_values(self) -> None:
        df = _linear_frame(120)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        row0 = result.iloc[0]
        assert row0["open_time"] == pd.Timestamp("2024-01-03T00:00:00", tz=UTC)
        assert row0["open"] == 2000.0
        assert row0["high"] == 2060.0
        assert row0["low"] == 1999.0
        assert row0["close"] == 2059.5
        assert row0["tick_volume"] == 600
        assert row0["real_volume"] == 180
        assert row0["spread"] == 20
        assert row0["source_bar_count"] == 60
        assert row0["is_complete"]

    def test_hour_one_aggregation_matches_hand_computed_values(self) -> None:
        # Rows 60-119: open=opens[60]=2060, high=highs[119]=2000+119+1=2120,
        # low=lows[60]=2000+60-1=2059, close=closes[119]=2000+119+0.5=2119.5.
        df = _linear_frame(120)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        row1 = result.iloc[1]
        assert row1["open"] == 2060.0
        assert row1["high"] == 2120.0
        assert row1["low"] == 2059.0
        assert row1["close"] == 2119.5

    def test_two_full_hours_produces_exactly_two_bars(self) -> None:
        df = _linear_frame(120)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert len(result) == 2


class TestIncompleteTrailingBucket:
    def test_reject_incomplete_drops_the_partial_trailing_hour(self) -> None:
        df = _linear_frame(150)  # 2 full hours + 30 extra minutes
        result = resample_ohlcv(
            df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1,
            policy=DerivedBarPolicy.REJECT_INCOMPLETE,
        )
        assert len(result) == 2
        assert result["is_complete"].all()

    def test_retain_incomplete_keeps_it_flagged(self) -> None:
        df = _linear_frame(150)
        result = resample_ohlcv(
            df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1,
            policy=DerivedBarPolicy.RETAIN_INCOMPLETE,
        )
        assert len(result) == 3
        assert result["is_complete"].tolist() == [True, True, False]
        assert result.iloc[2]["source_bar_count"] == 30

    def test_exactly_one_full_hour_is_complete_with_no_trailing_partial(self) -> None:
        df = _linear_frame(60)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert len(result) == 1
        assert result.iloc[0]["is_complete"]

    def test_less_than_one_bucket_of_data_yields_no_complete_bars(self) -> None:
        df = _linear_frame(30)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert len(result) == 0


class TestSessionGapTransparency:
    def test_bucket_spanning_a_gap_is_still_complete_but_thinly_populated(self) -> None:
        # A bucket whose TIME window has fully elapsed is complete even if
        # it aggregates fewer than the "expected" 60 M1 bars -- e.g. a
        # session/maintenance gap removed 20 of them. This must not be
        # confused with an INCOMPLETE (still-forming) trailing bucket.
        before = _linear_frame(40, "2024-01-03T00:00:00")
        after = _linear_frame(60, "2024-01-03T01:00:00")  # next full hour, unaffected
        df = pd.concat([before, after]).reset_index(drop=True)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert len(result) == 2
        assert result.iloc[0]["source_bar_count"] == 40
        assert bool(result.iloc[0]["is_complete"]) is True


class TestValidationErrors:
    def test_target_not_coarser_than_source_raises(self) -> None:
        df = _linear_frame(10)
        with pytest.raises(ResamplingError, match="strictly coarser"):
            resample_ohlcv(df, source_timeframe=Timeframe.H1, target_timeframe=Timeframe.M1)

    def test_equal_timeframes_raises(self) -> None:
        df = _linear_frame(10)
        with pytest.raises(ResamplingError, match="strictly coarser"):
            resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.M1)

    def test_unsorted_source_raises(self) -> None:
        df = _linear_frame(10)
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        with pytest.raises(ResamplingError, match="sorted ascending"):
            resample_ohlcv(shuffled, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)

    def test_duplicate_timestamps_raise(self) -> None:
        df = _linear_frame(10)
        df.loc[1, "open_time"] = df.loc[0, "open_time"]
        with pytest.raises(ResamplingError, match="duplicate"):
            resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)

    def test_empty_source_returns_empty_result_with_correct_schema(self) -> None:
        df = _linear_frame(1).iloc[0:0]
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert len(result) == 0
        assert "source_bar_count" in result.columns
        assert "is_complete" in result.columns


class TestVolumeConservation:
    @given(n=st.integers(min_value=60, max_value=600))
    def test_tick_and_real_volume_are_conserved_across_complete_buckets(self, n: int) -> None:
        df = _linear_frame(n)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        complete_bar_minutes = len(result) * 60
        assert result["tick_volume"].sum() == complete_bar_minutes * 10
        assert result["real_volume"].sum() == complete_bar_minutes * 3


class TestOHLCAggregationInvariants:
    @given(n=st.integers(min_value=60, max_value=300))
    def test_derived_high_is_never_less_than_derived_low(self, n: int) -> None:
        df = _linear_frame(n)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert (result["high"] >= result["low"]).all()

    @given(n=st.integers(min_value=60, max_value=300))
    def test_derived_open_and_close_are_within_high_low_range(self, n: int) -> None:
        df = _linear_frame(n)
        result = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        assert (result["open"] >= result["low"]).all() and (result["open"] <= result["high"]).all()
        assert (result["close"] >= result["low"]).all() and (result["close"] <= result["high"]).all()


class TestDeterminism:
    def test_resampling_twice_produces_identical_output(self) -> None:
        df = _linear_frame(180)
        result_a = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        result_b = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        pd.testing.assert_frame_equal(result_a, result_b)


class TestChunkedVsOneShotEquivalence:
    def test_chunking_exactly_on_a_bucket_boundary_matches_one_shot(self) -> None:
        df = _linear_frame(180)  # exactly 3 hours
        one_shot = resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)

        chunk_a = df.iloc[:120].reset_index(drop=True)  # first 2 full hours
        chunk_b = df.iloc[120:].reset_index(drop=True)  # last full hour
        result_a = resample_ohlcv(chunk_a, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        result_b = resample_ohlcv(chunk_b, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        chunked = pd.concat([result_a, result_b]).reset_index(drop=True)

        pd.testing.assert_frame_equal(one_shot, chunked)

    def test_chunking_mid_bucket_never_fabricates_a_complete_bar_early(self) -> None:
        # A chunk boundary falling MID-bucket must not let either half
        # claim that bucket as complete -- this is the batch-resampling
        # analogue of "no look-ahead": neither chunk alone has the full
        # picture, so neither should assert completeness for it.
        df = _linear_frame(90)  # 1 full hour + 30 minutes into the next
        first_chunk = df.iloc[:75].reset_index(drop=True)  # ends mid-second-hour
        result = resample_ohlcv(
            first_chunk, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1,
            policy=DerivedBarPolicy.RETAIN_INCOMPLETE,
        )
        assert result.iloc[-1]["is_complete"] is np.False_ or result.iloc[-1]["is_complete"] is False


class TestTimezoneRepresentationEquivalence:
    def test_same_instants_expressed_via_different_but_equivalent_utc_construction_resample_identically(self) -> None:
        df_a = _linear_frame(120)
        df_b = df_a.copy()
        # Route the same instants through a tz_convert round-trip via a
        # different (but equivalent) zone to prove resampling depends only
        # on the underlying UTC instant, never on how it was constructed.
        df_b["open_time"] = df_b["open_time"].dt.tz_convert("America/New_York").dt.tz_convert("UTC")
        result_a = resample_ohlcv(df_a, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        result_b = resample_ohlcv(df_b, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        pd.testing.assert_frame_equal(result_a, result_b)
