"""Unit tests for `market_data.calendar`: reuse of `historical.calendar.
TradingCalendar` plus `enumerate_expected_open_times`'s deterministic bar
enumeration."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest

from quant_platform.core.exceptions import MarketCalendarError, MarketDataError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import FixedOffsetTimezone
from quant_platform.market_data.calendar import (
    TradingCalendar,
    WeeklySession,
    enumerate_expected_open_times,
)

_UTC_TZ = FixedOffsetTimezone(offset=timedelta(0))


def _always_open_calendar() -> TradingCalendar:
    return TradingCalendar(local_tz=_UTC_TZ, weekly_sessions=(WeeklySession(open_weekday=0, open_time=time(0, 0), close_weekday=6, close_time=time(23, 59, 59)),), name="always_open")


class TestEnumerateExpectedOpenTimes:
    def test_returns_one_open_time_per_bar_when_market_always_open(self) -> None:
        calendar = _always_open_calendar()
        start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # Monday
        end = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
        result = enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=start, end=end)
        assert result == tuple(datetime(2026, 1, 5, h, 0, tzinfo=timezone.utc) for h in range(4))

    def test_two_calls_with_identical_inputs_produce_identical_output(self) -> None:
        calendar = _always_open_calendar()
        start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        first = enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=start, end=end)
        second = enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=start, end=end)
        assert first == second

    def test_weekend_closure_is_excluded(self) -> None:
        # Session: Monday 00:00 through Friday 23:59:59 only -- Saturday is closed.
        calendar = TradingCalendar(
            local_tz=_UTC_TZ,
            weekly_sessions=(WeeklySession(open_weekday=0, open_time=time(0, 0), close_weekday=4, close_time=time(23, 59, 59)),),
            name="weekday_only",
        )
        saturday_start = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)  # Saturday
        saturday_end = datetime(2026, 1, 10, 3, 0, tzinfo=timezone.utc)
        result = enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=saturday_start, end=saturday_end)
        assert result == ()

    def test_end_before_start_is_rejected(self) -> None:
        calendar = _always_open_calendar()
        start = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(MarketCalendarError):
            enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=start, end=end)

    def test_naive_start_is_rejected(self) -> None:
        calendar = _always_open_calendar()
        with pytest.raises(MarketDataError):
            enumerate_expected_open_times(calendar, timeframe=Timeframe.H1, start=datetime(2026, 1, 5), end=datetime(2026, 1, 5, 1, tzinfo=timezone.utc))

    def test_excessively_large_range_is_rejected(self) -> None:
        calendar = _always_open_calendar()
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(MarketCalendarError):
            enumerate_expected_open_times(calendar, timeframe=Timeframe.M1, start=start, end=end)
