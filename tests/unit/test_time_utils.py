"""Tests for pure timestamp/timeframe arithmetic utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quant_platform.core.time_utils import compute_close_time, ensure_utc, to_pandas_freq
from quant_platform.core.types import Timeframe

UTC = timezone.utc


class TestComputeCloseTime:
    @pytest.mark.parametrize("timeframe", list(Timeframe))
    def test_close_time_is_open_time_plus_duration(self, timeframe: Timeframe) -> None:
        open_time = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
        close_time = compute_close_time(open_time, timeframe)
        assert close_time == open_time + timeframe.duration

    def test_works_with_naive_datetimes_too(self) -> None:
        open_time = datetime(2024, 3, 1, 9, 0)
        close_time = compute_close_time(open_time, Timeframe.M15)
        assert close_time == datetime(2024, 3, 1, 9, 15)


class TestToPandasFreq:
    @pytest.mark.parametrize(
        ("timeframe", "expected"),
        [
            (Timeframe.M1, "1min"),
            (Timeframe.M5, "5min"),
            (Timeframe.M15, "15min"),
            (Timeframe.M30, "30min"),
            (Timeframe.H1, "1h"),
            (Timeframe.H4, "4h"),
            (Timeframe.D1, "1D"),
        ],
    )
    def test_returns_expected_alias(self, timeframe: Timeframe, expected: str) -> None:
        assert to_pandas_freq(timeframe) == expected

    @pytest.mark.parametrize("timeframe", list(Timeframe))
    def test_alias_is_usable_by_pandas_date_range(self, timeframe: Timeframe) -> None:
        freq = to_pandas_freq(timeframe)
        index = pd.date_range(start="2024-01-01", periods=3, freq=freq)
        assert len(index) == 3
        assert (index[1] - index[0]) == timeframe.duration


class TestEnsureUtc:
    def test_naive_timestamp_is_localized_to_utc(self) -> None:
        result = ensure_utc(datetime(2024, 1, 1, 12, 0))
        assert result.tzinfo is not None
        assert str(result.tzinfo) == "UTC"
        assert result.hour == 12  # localization, not conversion -- wall-clock value unchanged

    def test_utc_timestamp_passes_through_unchanged(self) -> None:
        original = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        result = ensure_utc(original)
        assert result == pd.Timestamp(original)

    def test_non_utc_aware_timestamp_is_converted(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        original = datetime(2024, 1, 1, 12, 0, tzinfo=eastern)
        result = ensure_utc(original)
        assert result.hour == 17  # 12:00 -05:00 == 17:00 UTC
        assert str(result.tzinfo) == "UTC"

    def test_accepts_pandas_timestamp_input(self) -> None:
        result = ensure_utc(pd.Timestamp("2024-01-01 12:00", tz="Asia/Tokyo"))
        assert str(result.tzinfo) == "UTC"
