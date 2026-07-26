"""Point-in-time alignment tests -- the highest-priority correctness
surface in Milestone 3. `align_higher_timeframe` is cross-validated against
`multiframe.cursor.TimeframeCursor` (the platform's existing, already-
adversarially-tested no-look-ahead primitive) bar-by-bar, plus explicit
boundary cases: 59 minutes into an H1 candle, the exact H1 close instant,
a daily close crossing a UTC day boundary, sparse/missing higher-timeframe
bars.
"""

from __future__ import annotations

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.exceptions import PointInTimeViolationError, SchemaError, TimezoneError
from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe, as_of_join_external
from quant_platform.multiframe.cursor import TimeframeCursor


def _cursor_compatible(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


def _cross_validate(base_close_times: pd.Series, higher_df: pd.DataFrame, timeframe: Timeframe) -> None:
    """For every base row, the vectorized alignment result must agree
    EXACTLY (same open_time / bar_index) with what a `TimeframeCursor`
    driven one row at a time, in order, reveals."""
    aligned = align_higher_timeframe(base_close_times, higher_df, timeframe)
    cursor = TimeframeCursor(_cursor_compatible(higher_df), timeframe)
    prefix = f"htf_{timeframe.value}_"

    for i, as_of in enumerate(base_close_times):
        cursor.advance_to(as_of)
        expected_index = cursor.current_index
        actual_index = int(aligned[f"{prefix}bar_index"].iloc[i])
        assert actual_index == expected_index, f"row {i}: expected bar_index={expected_index}, got {actual_index}"
        if expected_index >= 0:
            expected_open_time = cursor.current_bar.open_time
            actual_open_time = aligned[f"{prefix}open_time"].iloc[i]
            assert actual_open_time == expected_open_time


class TestCrossValidationAgainstTimeframeCursor:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_random_m1_h1_alignment_matches_cursor(self, seed: int) -> None:
        base_df = make_synthetic_ohlcv(600, freq_minutes=1, seed=seed)
        higher_df = make_synthetic_ohlcv(20, freq_minutes=60, seed=seed + 100)
        base_close_times = base_df["open_time"] + Timeframe.M1.duration
        _cross_validate(base_close_times, higher_df, Timeframe.H1)

    def test_random_m5_d1_alignment_matches_cursor(self) -> None:
        base_df = make_synthetic_ohlcv(2000, freq_minutes=5, seed=7)
        higher_df = make_synthetic_ohlcv(10, freq_minutes=24 * 60, seed=8)
        base_close_times = base_df["open_time"] + Timeframe.M5.duration
        _cross_validate(base_close_times, higher_df, Timeframe.D1)


class TestBoundaryCases:
    def test_59_minutes_into_h1_candle_not_yet_visible(self) -> None:
        """An H1 bar opening at 10:00 must NOT be visible at 10:59 (its
        close is 11:00) -- only the PREVIOUS H1 bar may be visible."""
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00Z", seed=1)
        as_of = pd.Series([pd.Timestamp("2024-01-01T10:59:00Z")])
        aligned = align_higher_timeframe(as_of, higher_df, Timeframe.H1)
        visible_open = aligned["htf_H1_open_time"].iloc[0]
        assert visible_open == pd.Timestamp("2024-01-01T09:00:00Z")
        assert visible_open != pd.Timestamp("2024-01-01T10:00:00Z")

    def test_exact_h1_close_reveals_the_bar(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00Z", seed=1)
        as_of = pd.Series([pd.Timestamp("2024-01-01T11:00:00Z")])
        aligned = align_higher_timeframe(as_of, higher_df, Timeframe.H1)
        assert aligned["htf_H1_open_time"].iloc[0] == pd.Timestamp("2024-01-01T10:00:00Z")

    def test_one_second_before_close_does_not_reveal_the_bar(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00Z", seed=1)
        as_of = pd.Series([pd.Timestamp("2024-01-01T10:59:59Z")])
        aligned = align_higher_timeframe(as_of, higher_df, Timeframe.H1)
        assert aligned["htf_H1_open_time"].iloc[0] == pd.Timestamp("2024-01-01T09:00:00Z")

    def test_daily_close_crossing_utc_boundary(self) -> None:
        """A D1 bar opening 2024-01-01T00:00Z closes exactly at
        2024-01-02T00:00Z -- must not be revealed a moment before that UTC
        midnight boundary."""
        higher_df = make_synthetic_ohlcv(5, freq_minutes=24 * 60, start="2024-01-01T00:00:00Z", seed=2)
        just_before = pd.Series([pd.Timestamp("2024-01-01T23:59:59Z")])
        at_close = pd.Series([pd.Timestamp("2024-01-02T00:00:00Z")])
        aligned_before = align_higher_timeframe(just_before, higher_df, Timeframe.D1)
        aligned_at = align_higher_timeframe(at_close, higher_df, Timeframe.D1)
        assert aligned_before["htf_D1_bar_index"].iloc[0] == -1
        assert aligned_at["htf_D1_open_time"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")

    def test_no_higher_timeframe_bar_closed_yet_gives_warmup_sentinel(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00Z", seed=1)
        as_of = pd.Series([pd.Timestamp("2024-01-01T08:30:00Z")])
        aligned = align_higher_timeframe(as_of, higher_df, Timeframe.H1)
        assert aligned["htf_H1_bar_index"].iloc[0] == -1
        assert pd.isna(aligned["htf_H1_open_time"].iloc[0])

    def test_sparse_missing_higher_timeframe_bar_carries_forward_not_fabricated(self) -> None:
        """If hour 10's H1 bar is simply absent (a real data gap), a base
        row at hour 11's close must still see hour 9's bar -- never a
        fabricated hour-10 bar, and never hour 11's (which hasn't closed
        relative to hour 10's own absence -- there IS no hour-10 bar to
        close)."""
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, start="2024-01-01T08:00:00Z", seed=1)
        # Drop the bar opening at 10:00 (index 2).
        higher_sparse = pd.concat([higher_df.iloc[:2], higher_df.iloc[3:]]).reset_index(drop=True)
        as_of = pd.Series([pd.Timestamp("2024-01-01T11:00:00Z")])
        aligned = align_higher_timeframe(as_of, higher_sparse, Timeframe.H1)
        # last bar with open_time <= (11:00 - 1h) among {08:00,09:00,11:00} is 09:00
        assert aligned["htf_H1_open_time"].iloc[0] == pd.Timestamp("2024-01-01T09:00:00Z")

    def test_empty_higher_df_yields_all_warmup_sentinels(self) -> None:
        empty = make_synthetic_ohlcv(0, freq_minutes=60, seed=1)
        as_of = pd.Series([pd.Timestamp("2024-01-01T11:00:00Z"), pd.Timestamp("2024-01-01T12:00:00Z")])
        aligned = align_higher_timeframe(as_of, empty, Timeframe.H1)
        assert (aligned["htf_H1_bar_index"] == -1).all()
        assert aligned["htf_H1_close_time"].isna().all()


class TestSchemaEnforcement:
    def test_missing_open_time_column_rejected(self) -> None:
        higher_df = pd.DataFrame({"close": [1.0, 2.0]})
        as_of = pd.Series([pd.Timestamp("2024-01-01T00:00:00Z")])
        with pytest.raises(SchemaError):
            align_higher_timeframe(as_of, higher_df, Timeframe.H1)

    def test_unsorted_higher_df_rejected(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, seed=1)
        shuffled = higher_df.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
        as_of = pd.Series([pd.Timestamp("2024-01-01T12:00:00Z")])
        with pytest.raises(PointInTimeViolationError):
            align_higher_timeframe(as_of, shuffled, Timeframe.H1)

    def test_duplicate_open_time_rejected(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, seed=1)
        duplicated = pd.concat([higher_df, higher_df.iloc[[0]]]).reset_index(drop=True)
        as_of = pd.Series([pd.Timestamp("2024-01-01T12:00:00Z")])
        with pytest.raises(PointInTimeViolationError):
            align_higher_timeframe(as_of, duplicated, Timeframe.H1)

    def test_naive_higher_df_open_time_rejected(self) -> None:
        """Adversarial self-audit (Section 20 'timezone mismatches'): a
        higher_df with naive timestamps must never be silently assumed to
        be UTC."""
        naive_higher_df = pd.DataFrame(
            {"open_time": pd.date_range("2024-01-01", periods=3, freq="1h"), "close": [1.0, 2.0, 3.0]}
        )
        as_of = pd.Series([pd.Timestamp("2024-01-01T02:00:00Z")])
        with pytest.raises(TimezoneError):
            align_higher_timeframe(as_of, naive_higher_df, Timeframe.H1)

    def test_naive_base_close_times_rejected(self) -> None:
        higher_df = make_synthetic_ohlcv(5, freq_minutes=60, seed=1)
        naive_as_of = pd.Series(pd.date_range("2024-01-01T12:00:00", periods=1, freq="1h"))
        with pytest.raises(TimezoneError):
            align_higher_timeframe(naive_as_of, higher_df, Timeframe.H1)


class TestAsOfJoinExternal:
    def test_naive_release_time_rejected(self) -> None:
        """Adversarial self-audit: a macro source with naive release
        timestamps must never be silently assumed to be UTC."""
        external = pd.DataFrame(
            {"release_time": pd.date_range("2024-01-01", periods=1, freq="1D"), "value": [1.0]}
        )
        as_of = pd.Series([pd.Timestamp("2024-01-01T00:00:00Z")])
        with pytest.raises(TimezoneError):
            as_of_join_external(as_of, external, value_column="value")


    def test_value_unavailable_before_release(self) -> None:
        external = pd.DataFrame(
            {"release_time": [pd.Timestamp("2024-02-01T18:00:00Z")], "value": [5.5]}
        )
        as_of_january = pd.Series([pd.Timestamp("2024-01-31T23:59:59Z")])
        result = as_of_join_external(as_of_january, external, value_column="value")
        assert pd.isna(result["value"].iloc[0])
        assert result["value_is_stale"].iloc[0]

    def test_value_available_immediately_after_release(self) -> None:
        external = pd.DataFrame(
            {"release_time": [pd.Timestamp("2024-02-01T18:00:00Z")], "value": [5.5]}
        )
        as_of_after = pd.Series([pd.Timestamp("2024-02-01T18:00:01Z")])
        result = as_of_join_external(as_of_after, external, value_column="value")
        assert result["value"].iloc[0] == 5.5
        assert not result["value_is_stale"].iloc[0]

    def test_exact_release_instant_is_visible(self) -> None:
        external = pd.DataFrame(
            {"release_time": [pd.Timestamp("2024-02-01T18:00:00Z")], "value": [5.5]}
        )
        as_of_exact = pd.Series([pd.Timestamp("2024-02-01T18:00:00Z")])
        result = as_of_join_external(as_of_exact, external, value_column="value")
        assert result["value"].iloc[0] == 5.5

    def test_tolerance_marks_stale_value_as_missing(self) -> None:
        external = pd.DataFrame(
            {"release_time": [pd.Timestamp("2024-01-01T00:00:00Z")], "value": [5.5]}
        )
        far_future = pd.Series([pd.Timestamp("2024-06-01T00:00:00Z")])
        result = as_of_join_external(far_future, external, value_column="value", tolerance=pd.Timedelta(days=45))
        assert pd.isna(result["value"].iloc[0])
        assert result["value_is_stale"].iloc[0]

    def test_duplicate_release_timestamps_resolve_deterministically(self) -> None:
        """Two releases at the EXACT same instant (e.g. an immediate
        correction) -- the later one in input order (the more
        authoritative vintage) must win for any query at or after that
        instant."""
        release_time = pd.Timestamp("2024-02-01T18:00:00Z")
        external = pd.DataFrame({"release_time": [release_time, release_time], "value": [1.0, 2.0]})
        as_of = pd.Series([release_time])
        result = as_of_join_external(as_of, external, value_column="value")
        assert result["value"].iloc[0] == 2.0

    def test_empty_external_df_yields_all_missing(self) -> None:
        external = pd.DataFrame({"release_time": pd.Series([], dtype="datetime64[ns, UTC]"), "value": pd.Series([], dtype="float64")})
        as_of = pd.Series([pd.Timestamp("2024-01-01T00:00:00Z")])
        result = as_of_join_external(as_of, external, value_column="value")
        assert pd.isna(result["value"].iloc[0])
        assert result["value_is_stale"].iloc[0]

    def test_revision_only_visible_after_its_own_release(self) -> None:
        """The core Section 6 proof: a January-observed value first
        released (preliminary) in early February, then REVISED in March,
        must show the preliminary value throughout February and the
        revised value only from March onward -- never the revised value
        retroactively applied to February rows."""
        external = pd.DataFrame(
            {
                "observation_time": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")],
                "release_time": [pd.Timestamp("2024-02-05T00:00:00Z"), pd.Timestamp("2024-03-05T00:00:00Z")],
                "value": [100.0, 105.0],
            }
        )
        as_of = pd.Series(
            [
                pd.Timestamp("2024-01-15T00:00:00Z"),  # before any release
                pd.Timestamp("2024-02-10T00:00:00Z"),  # after preliminary, before revision
                pd.Timestamp("2024-03-10T00:00:00Z"),  # after revision
            ]
        )
        result = as_of_join_external(as_of, external, value_column="value")
        assert pd.isna(result["value"].iloc[0])
        assert result["value"].iloc[1] == 100.0
        assert result["value"].iloc[2] == 105.0
