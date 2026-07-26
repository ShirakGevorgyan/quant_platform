"""Correctness proof for `TimeframeCursor`'s core guarantee: a bar is never
revealed before its close time.

This is the single most important test in the repository. The bug it
guards against (revealing a higher-timeframe bar based on its OPEN time
instead of its CLOSE time) previously shipped in a production backtest and
silently inflated results by ~3x before being caught by manual timestamp
inspection. It must never regress.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_platform.core.exceptions import SchemaError
from quant_platform.core.types import Timeframe
from quant_platform.multiframe.cursor import TimeframeCursor

UTC = timezone.utc


def _make_ohlcv_frame(start: datetime, periods: int, timeframe: Timeframe) -> pd.DataFrame:
    """Deterministic, minimal OHLCV frame for cursor-isolation tests. Prices
    are irrelevant to look-ahead correctness, so they are held constant."""
    open_times = [start + i * timeframe.duration for i in range(periods)]
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.5] * periods,
            "volume": [1000.0] * periods,
        }
    )


class TestSchemaValidation:
    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "open": [1.0]})
        with pytest.raises(SchemaError, match="missing required OHLCV columns"):
            TimeframeCursor(df, Timeframe.M15)

    def test_rejects_non_monotonic_open_time(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=3, timeframe=Timeframe.M15)
        df.loc[1, "open_time"] = df.loc[0, "open_time"] - timedelta(minutes=1)
        with pytest.raises(SchemaError, match="monotonically increasing"):
            TimeframeCursor(df, Timeframe.M15)

    def test_accepts_empty_frame(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=0, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        assert not cursor.has_current_bar
        assert cursor.current_bar is None


class TestNoLookaheadDeterministic:
    """Exact regression test for the specific bug that was shipped: a bar
    open at 09:00 on M15 must NOT be visible at 09:00, 09:05, ..., 09:14,
    and MUST be visible starting exactly at 09:15."""

    def test_bar_not_visible_before_its_close_time(self) -> None:
        bar_open = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
        df = _make_ohlcv_frame(bar_open, periods=1, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)

        # Every instant strictly before the bar's close (09:15) must NOT reveal it.
        for minute_offset in range(0, 15):
            as_of = pd.Timestamp(bar_open + timedelta(minutes=minute_offset))
            cursor.advance_to(as_of)
            assert cursor.current_bar is None, (
                f"Bar opened at {bar_open} leaked at as_of={as_of}, "
                f"{15 - minute_offset} minutes before its true close time"
            )

    def test_bar_visible_exactly_at_its_close_time(self) -> None:
        bar_open = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
        df = _make_ohlcv_frame(bar_open, periods=1, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)

        as_of = pd.Timestamp(bar_open + timedelta(minutes=15))
        cursor.advance_to(as_of)

        assert cursor.current_bar is not None
        assert cursor.current_bar.open_time == pd.Timestamp(bar_open)
        assert cursor.current_bar.close_time == as_of

    @pytest.mark.parametrize("timeframe", list(Timeframe))
    def test_no_leak_across_all_supported_timeframes(self, timeframe: Timeframe) -> None:
        bar_open = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)  # a Monday, for D1 sanity
        df = _make_ohlcv_frame(bar_open, periods=1, timeframe=timeframe)
        cursor = TimeframeCursor(df, timeframe)

        just_before_close = pd.Timestamp(bar_open + timeframe.duration - timedelta(seconds=1))
        cursor.advance_to(just_before_close)
        assert cursor.current_bar is None, f"{timeframe} leaked 1 second before close"

        at_close = pd.Timestamp(bar_open + timeframe.duration)
        cursor.advance_to(at_close)
        assert cursor.current_bar is not None, f"{timeframe} failed to reveal at exact close"


class TestCursorMonotonicity:
    def test_advance_to_rejects_decreasing_as_of(self) -> None:
        """The cursor models a forward-only simulation clock. A caller
        passing a decreasing `as_of` almost certainly has an iteration-order
        bug, so this must fail loudly rather than silently no-op."""
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=10, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)

        far_future = pd.Timestamp(datetime(2024, 1, 2, tzinfo=UTC))
        cursor.advance_to(far_future)
        assert cursor.current_index == 9

        earlier = pd.Timestamp(datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            cursor.advance_to(earlier)

    def test_repeated_same_as_of_is_idempotent(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=5, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        as_of = pd.Timestamp(datetime(2024, 1, 1, 0, 15, tzinfo=UTC))

        first = cursor.advance_to(as_of)
        second = cursor.advance_to(as_of)

        assert first is True
        assert second is False
        assert cursor.current_index == 0


class TestWindow:
    def test_window_before_any_bar_revealed_is_empty(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=5, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        window = cursor.window(size=3)
        assert len(window) == 0
        assert list(window.columns) == list(df.columns)

    def test_window_returns_most_recent_n_bars(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        df = _make_ohlcv_frame(start, periods=10, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        cursor.advance_to(pd.Timestamp(start + 7 * Timeframe.M15.duration))

        window = cursor.window(size=3)

        assert len(window) == 3
        expected_opens = [start + i * Timeframe.M15.duration for i in (4, 5, 6)]
        assert list(window["open_time"]) == [pd.Timestamp(t) for t in expected_opens]

    def test_window_returns_fewer_rows_during_warmup(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        df = _make_ohlcv_frame(start, periods=10, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        cursor.advance_to(pd.Timestamp(start + 1 * Timeframe.M15.duration))  # only bar 0 revealed

        window = cursor.window(size=5)
        assert len(window) == 1

    def test_window_is_a_copy_not_a_view(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        df = _make_ohlcv_frame(start, periods=5, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)
        cursor.advance_to(pd.Timestamp(start + 4 * Timeframe.M15.duration))

        window = cursor.window(size=5)
        window.loc[0, "close"] = -999.0

        assert cursor.window(size=5).loc[0, "close"] != -999.0


# --------------------------------------------------------------------------
# Property-based adversarial test: the invariant must hold for ANY
# combination of series length, start time, and clock-stepping pattern.
# --------------------------------------------------------------------------
@given(
    n_bars=st.integers(min_value=1, max_value=200),
    start_offset_minutes=st.integers(min_value=0, max_value=10_000),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=150, deadline=None)
def test_property_no_bar_ever_revealed_before_its_close_time(
    n_bars: int, start_offset_minutes: int, seed: int
) -> None:
    timeframe = Timeframe.M15
    start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=start_offset_minutes)
    df = _make_ohlcv_frame(start, periods=n_bars, timeframe=timeframe)
    cursor = TimeframeCursor(df, timeframe)

    total_span_minutes = int(n_bars * timeframe.minutes) + 30
    rng = np.random.default_rng(seed)
    # A monotonically increasing, irregular (sub-bar-granularity) clock --
    # deliberately NOT aligned to the timeframe grid, to stress exactly the
    # boundary conditions a real M1-driven base clock would produce.
    step_minutes = rng.integers(low=1, high=7, size=total_span_minutes)
    clock_offsets = np.cumsum(step_minutes)

    for offset in clock_offsets:
        as_of = pd.Timestamp(start + timedelta(minutes=int(offset)))
        cursor.advance_to(as_of)

        if cursor.has_current_bar:
            current = cursor.current_bar
            assert current is not None
            assert current.close_time <= as_of, (
                f"INVARIANT VIOLATION: bar closing at {current.close_time} was visible "
                f"at as_of={as_of}"
            )

        next_idx = cursor.current_index + 1
        if next_idx < n_bars:
            next_close_time = pd.Timestamp(df.loc[next_idx, "open_time"]) + timeframe.duration
            assert next_close_time > as_of, (
                f"INVARIANT VIOLATION: bar at index {next_idx} closing at "
                f"{next_close_time} should have been revealed by as_of={as_of} but was not"
            )


class TestNaiveTimestampRejection:
    """Golden-master regression test for a critical bug found during
    adversarial audit: `TimeframeCursor` used to accept naive (tz-less)
    `open_time` columns, silently treating them as "already UTC". If a
    caller fed one naive-timestamped timeframe alongside a properly
    tz-aware one to the same `BacktestEngine`, the cross-timeframe clock
    desynchronized -- concretely demonstrated with a naive series
    mislabeling a UTC+9 wall clock: an H1 bar that truly closes at 01:00
    UTC was revealed at the very first base bar, when true elapsed time
    into that bar was only 15 minutes of its 60-minute window. This is a
    genuine look-ahead leak, not just a display/formatting quirk.
    """

    def test_rejects_naive_open_time_column(self) -> None:
        naive_df = _make_ohlcv_frame(datetime(2024, 1, 1), periods=3, timeframe=Timeframe.M15)
        assert naive_df["open_time"].dt.tz is None  # sanity: genuinely naive
        with pytest.raises(SchemaError, match="tz-aware"):
            TimeframeCursor(naive_df, Timeframe.M15)

    def test_rejects_naive_as_of_passed_to_advance_to(self) -> None:
        aware_df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=3, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(aware_df, Timeframe.M15)
        with pytest.raises(ValueError, match="tz-aware"):
            cursor.advance_to(pd.Timestamp(datetime(2024, 1, 1, 0, 15)))  # naive, no tzinfo

    def test_mixed_naive_and_aware_timeframes_no_longer_desynchronize(self) -> None:
        """End-to-end reproduction of the originally-discovered leak,
        confirming the fix: constructing the naive series now fails fast
        at the cursor boundary instead of silently misaligning the clock."""
        base_open_naive = [datetime(2024, 1, 1, 9, 0) + i * timedelta(minutes=15) for i in range(4)]
        naive_base = pd.DataFrame(
            {
                "open_time": base_open_naive,
                "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
                "close": [100.5] * 4, "volume": [10.0] * 4,
            }
        )
        aware_h1 = pd.DataFrame(
            {
                "open_time": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "open": [500.0], "high": [510.0], "low": [490.0], "close": [505.0], "volume": [999.0],
            }
        )
        with pytest.raises(SchemaError, match="tz-aware"):
            TimeframeCursor(naive_base, Timeframe.M15)
        # The properly tz-aware series is unaffected and still constructs fine.
        TimeframeCursor(aware_h1, Timeframe.H1)


class TestDuplicateTimestampRejection:
    """Golden-master regression test for a bug found during adversarial
    audit: pandas' `is_monotonic_increasing` means non-decreasing, so it
    alone does not reject duplicate (tied) `open_time` values. A duplicate
    used to construct successfully and silently resolve to whichever row
    won `advance_to`'s `<=` comparison (the later-indexed one), discarding
    the other bar's OHLC data with no warning at all.
    """

    def test_rejects_duplicate_open_time(self) -> None:
        df = pd.DataFrame(
            {
                "open_time": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),  # exact duplicate of the row above
                    datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                ],
                "open": [100.0, 999.0, 101.0], "high": [101.0, 1000.0, 102.0],
                "low": [99.0, 998.0, 100.0], "close": [100.5, 999.5, 101.5],
                "volume": [10.0, 10.0, 10.0],
            }
        )
        assert df["open_time"].is_monotonic_increasing  # sanity: pandas permits this
        with pytest.raises(SchemaError, match="duplicate value"):
            TimeframeCursor(df, Timeframe.M15)

    def test_accepts_strictly_increasing_timestamps(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=3, timeframe=Timeframe.M15)
        TimeframeCursor(df, Timeframe.M15)  # must not raise


class TestDefensiveCopy:
    """Golden-master regression test: `TimeframeCursor` must be independent
    of the caller's original DataFrame regardless of pandas' copy-on-write
    configuration, which is version/setting dependent and should not be an
    implicit safety dependency for the platform's most safety-critical
    component."""

    def test_mutating_the_original_dataframe_after_construction_does_not_affect_the_cursor(self) -> None:
        df = _make_ohlcv_frame(datetime(2024, 1, 1, tzinfo=UTC), periods=3, timeframe=Timeframe.M15)
        cursor = TimeframeCursor(df, Timeframe.M15)

        df.loc[0, "close"] = -999.0  # mutate the ORIGINAL after the cursor was built

        cursor.advance_to(pd.Timestamp(datetime(2024, 1, 1, 0, 15, tzinfo=UTC)))
        assert cursor.current_bar is not None
        assert cursor.current_bar.close != -999.0
