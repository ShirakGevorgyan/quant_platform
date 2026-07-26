"""Tests for `historical.calendar`. Session/maintenance-break/holiday
boundaries below are hand-picked against a known reference week
(2024-01-01 is a Monday) rather than re-deriving the implementation.
"""

from __future__ import annotations

from datetime import time, timedelta

import pandas as pd
import pytest

from quant_platform.core.exceptions import ConfigurationError
from quant_platform.historical.calendar import (
    DailyMaintenanceBreak,
    HolidayClosure,
    TradingCalendar,
    WeeklySession,
    default_xauusd_calendar,
)
from quant_platform.historical.timezones import FixedOffsetTimezone

UTC0 = FixedOffsetTimezone(timedelta(0), name="UTC0")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestTradingCalendarConstruction:
    def test_requires_at_least_one_weekly_session(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one weekly session"):
            TradingCalendar(local_tz=UTC0, weekly_sessions=())


class TestDefaultXauusdCalendarWeeklySession:
    """Session: opens Sunday 23:00, closes Friday 23:00 (server time)."""

    cal = default_xauusd_calendar(UTC0)

    @pytest.mark.parametrize(
        ("moment", "expected_open"),
        [
            ("2024-01-07 23:30:00", True),   # Sunday, just after weekly open
            ("2024-01-06 12:00:00", False),  # Saturday noon: closed
            ("2024-01-05 23:30:00", False),  # Friday 23:30: after Friday close
            ("2024-01-05 22:59:00", True),   # Friday 22:59: just before close
            ("2024-01-03 12:00:00", True),   # Wednesday noon: open
            ("2024-01-01 00:00:00", True),   # Monday midnight: open
        ],
    )
    def test_weekly_session_boundaries(self, moment: str, expected_open: bool) -> None:
        ts = _ts(moment).tz_convert(self.cal.local_tz.to_tzinfo())
        assert self.cal._is_open_at_local(ts) is expected_open


class TestDefaultXauusdCalendarMaintenanceBreak:
    cal = default_xauusd_calendar(UTC0)

    def test_inside_daily_break_is_closed(self) -> None:
        ts = _ts("2024-01-03 23:00:30").tz_convert(self.cal.local_tz.to_tzinfo())
        assert self.cal._is_open_at_local(ts) is False

    def test_just_before_daily_break_is_open(self) -> None:
        ts = _ts("2024-01-03 22:59:59").tz_convert(self.cal.local_tz.to_tzinfo())
        assert self.cal._is_open_at_local(ts) is True

    def test_just_after_daily_break_is_open(self) -> None:
        ts = _ts("2024-01-03 23:01:01").tz_convert(self.cal.local_tz.to_tzinfo())
        assert self.cal._is_open_at_local(ts) is True


class TestHolidayClosure:
    def test_holiday_overrides_an_otherwise_open_weekday(self) -> None:
        cal = TradingCalendar(
            local_tz=UTC0,
            weekly_sessions=(WeeklySession(open_weekday=6, open_time=time(23, 0), close_weekday=4, close_time=time(23, 0)),),
            holidays=(HolidayClosure(closure_date=_ts("2024-01-03 00:00:00").date(), description="test holiday"),),
        )
        ts = _ts("2024-01-03 12:00:00").tz_convert(cal.local_tz.to_tzinfo())
        assert cal._is_open_at_local(ts) is False


class TestIsExpectedClosure:
    cal = default_xauusd_calendar(UTC0)

    def test_full_weekend_gap_is_an_expected_closure(self) -> None:
        assert self.cal.is_expected_closure(_ts("2024-01-05 23:00:00"), _ts("2024-01-07 23:00:00")) is True

    def test_midweek_gap_spanning_open_hours_is_not_expected(self) -> None:
        assert self.cal.is_expected_closure(_ts("2024-01-03 10:00:00"), _ts("2024-01-03 14:00:00")) is False

    def test_daily_maintenance_break_gap_is_expected(self) -> None:
        assert self.cal.is_expected_closure(_ts("2024-01-03 23:00:00"), _ts("2024-01-03 23:01:00")) is True

    def test_rejects_non_increasing_interval(self) -> None:
        with pytest.raises(ValueError, match="must be after"):
            self.cal.is_expected_closure(_ts("2024-01-03 10:00:00"), _ts("2024-01-03 09:00:00"))


class TestMaintenanceBreakWrapsMidnight:
    def test_break_spanning_midnight_is_closed_on_both_sides(self) -> None:
        cal = TradingCalendar(
            local_tz=UTC0,
            weekly_sessions=(WeeklySession(open_weekday=0, open_time=time(0, 0), close_weekday=6, close_time=time(23, 59, 59)),),
            maintenance_breaks=(DailyMaintenanceBreak(start=time(23, 30), end=time(0, 30)),),
        )
        before_midnight = _ts("2024-01-03 23:45:00").tz_convert(cal.local_tz.to_tzinfo())
        after_midnight = _ts("2024-01-04 00:15:00").tz_convert(cal.local_tz.to_tzinfo())
        clearly_open = _ts("2024-01-03 12:00:00").tz_convert(cal.local_tz.to_tzinfo())
        assert cal._is_open_at_local(before_midnight) is False
        assert cal._is_open_at_local(after_midnight) is False
        assert cal._is_open_at_local(clearly_open) is True
