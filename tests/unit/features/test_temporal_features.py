from __future__ import annotations

from datetime import time as time_of_day
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import ConfigurationError
from quant_platform.core.types import Timeframe
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.temporal.calendar_features import (
    cyclical_encode,
    day_of_week,
    hour_of_day,
    market_open_proximity,
    minutes_since_session_open,
    month_of_year,
    register_core_temporal_features,
    session_open_flag,
)
from quant_platform.historical.calendar import TradingCalendar, WeeklySession
from quant_platform.historical.timezones import FixedOffsetTimezone, NamedZoneTimezone


def _calendar(**overrides) -> TradingCalendar:
    base = {
        "local_tz": FixedOffsetTimezone(timedelta(0), name="UTC"),
        "weekly_sessions": (
            WeeklySession(open_weekday=6, open_time=time_of_day(23, 0), close_weekday=4, close_time=time_of_day(23, 0)),
        ),
    }
    base.update(overrides)
    return TradingCalendar(**base)


class TestBasicTemporalValues:
    def test_hour_of_day(self) -> None:
        open_time = pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T13:30:00Z"]))
        result = hour_of_day(open_time)
        assert result.tolist() == [0.0, 13.0]

    def test_day_of_week_monday_is_zero(self) -> None:
        open_time = pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"]))  # a Monday
        assert day_of_week(open_time).iloc[0] == 0.0

    def test_month_of_year(self) -> None:
        open_time = pd.Series(pd.to_datetime(["2024-03-15T00:00:00Z"]))
        assert month_of_year(open_time).iloc[0] == 3.0

    def test_cyclical_encode_matches_known_points(self) -> None:
        hours = pd.Series([0.0, 6.0, 12.0, 18.0])
        sin_vals, cos_vals = cyclical_encode(hours, 24.0)
        assert sin_vals.iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert cos_vals.iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert sin_vals.iloc[1] == pytest.approx(1.0, abs=1e-9)
        assert cos_vals.iloc[2] == pytest.approx(-1.0, abs=1e-9)

    def test_cyclical_encoding_makes_hour_23_and_hour_0_adjacent(self) -> None:
        """The whole point of cyclical encoding: hour 23 and hour 0 should
        be close in encoded space, unlike raw integers 23 and 0."""
        hours = pd.Series([23.0, 0.0])
        sin_vals, cos_vals = cyclical_encode(hours, 24.0)
        distance = np.hypot(sin_vals.iloc[0] - sin_vals.iloc[1], cos_vals.iloc[0] - cos_vals.iloc[1])
        assert distance < 0.3


class TestSessionFlags:
    def test_open_during_session(self) -> None:
        calendar = _calendar()
        open_time = pd.Series(pd.to_datetime(["2024-01-02T12:00:00Z"]))  # Tuesday, inside the session
        assert session_open_flag(open_time, calendar).iloc[0]

    def test_closed_during_weekend(self) -> None:
        calendar = _calendar()
        open_time = pd.Series(pd.to_datetime(["2024-01-06T12:00:00Z"]))  # Saturday
        assert not session_open_flag(open_time, calendar).iloc[0]

    def test_multiple_weekly_sessions_rejects_session_relative_features(self) -> None:
        calendar = _calendar(
            weekly_sessions=(
                WeeklySession(open_weekday=0, open_time=time_of_day(0, 0), close_weekday=0, close_time=time_of_day(12, 0)),
                WeeklySession(open_weekday=1, open_time=time_of_day(0, 0), close_weekday=1, close_time=time_of_day(12, 0)),
            )
        )
        open_time = pd.Series(pd.to_datetime(["2024-01-01T06:00:00Z"]))
        with pytest.raises(ConfigurationError):
            minutes_since_session_open(open_time, calendar)

    def test_minutes_since_session_open_at_the_open_instant(self) -> None:
        calendar = _calendar()
        # session opens Sunday 23:00 UTC
        open_time = pd.Series(pd.to_datetime(["2024-01-07T23:00:00Z"]))  # a Sunday
        result = minutes_since_session_open(open_time, calendar)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_minutes_since_session_open_grows_through_the_week(self) -> None:
        calendar = _calendar()
        open_time = pd.Series(pd.to_datetime(["2024-01-07T23:00:00Z", "2024-01-08T23:00:00Z", "2024-01-09T23:00:00Z"]))
        result = minutes_since_session_open(open_time, calendar)
        assert result.iloc[0] < result.iloc[1] < result.iloc[2]

    def test_market_open_proximity_small_near_boundaries(self) -> None:
        calendar = _calendar()
        near_open = pd.Series(pd.to_datetime(["2024-01-07T23:05:00Z"]))
        mid_week = pd.Series(pd.to_datetime(["2024-01-10T12:00:00Z"]))
        proximity_near_open = market_open_proximity(near_open, calendar).iloc[0]
        proximity_mid_week = market_open_proximity(mid_week, calendar).iloc[0]
        assert proximity_near_open < proximity_mid_week


class TestDSTHandling:
    """Session-relative features convert through `calendar.local_tz` --
    exactly where a DST-observing zone could introduce ambiguity. These
    tests use `America/New_York` (observes DST) across both the "spring
    forward" and "fall back" transitions of 2024 and require: no crash, no
    NaN, and no wildly discontinuous jump beyond the expected +/-1 hour
    around the transition."""

    def _ny_calendar(self) -> TradingCalendar:
        return _calendar(local_tz=NamedZoneTimezone("America/New_York"))

    def test_spring_forward_transition_2024(self) -> None:
        calendar = self._ny_calendar()
        # 2024-03-10 02:00 local (EST) springs forward to 03:00 (EDT) in America/New_York
        around_transition = pd.Series(
            pd.to_datetime(
                ["2024-03-10T06:00:00Z", "2024-03-10T07:00:00Z", "2024-03-10T08:00:00Z"]
            )
        )
        result = minutes_since_session_open(around_transition, calendar)
        assert not result.isna().any()
        # consecutive hourly UTC steps should advance by ~60 minutes each, never by ~120 or ~0
        diffs = result.diff().dropna()
        assert (diffs.abs().between(0, 120)).all()

    def test_fall_back_transition_2024(self) -> None:
        calendar = self._ny_calendar()
        around_transition = pd.Series(
            pd.to_datetime(
                ["2024-11-03T04:00:00Z", "2024-11-03T05:00:00Z", "2024-11-03T06:00:00Z", "2024-11-03T07:00:00Z"]
            )
        )
        result = minutes_since_session_open(around_transition, calendar)
        assert not result.isna().any()
        diffs = result.diff().dropna()
        assert (diffs.abs().between(0, 120)).all()

    def test_session_open_flag_deterministic_across_dst(self) -> None:
        calendar = self._ny_calendar()
        open_time = pd.Series(
            pd.to_datetime(["2024-03-09T12:00:00Z", "2024-03-10T12:00:00Z", "2024-03-11T12:00:00Z"])
        )
        result = session_open_flag(open_time, calendar)
        assert result.dtype == bool
        assert not result.isna().any()


class TestRegistration:
    def test_registers_base_features_without_calendar(self) -> None:
        registry = FeatureRegistry()
        register_core_temporal_features(registry, timeframe=Timeframe.M1, calendar=None)
        names = {s.name for s in registry.list_features()}
        assert "hour_of_day" in names
        assert "session_open_flag" not in names

    def test_registers_session_features_with_calendar(self) -> None:
        registry = FeatureRegistry()
        register_core_temporal_features(registry, timeframe=Timeframe.M1, calendar=_calendar())
        names = {s.name for s in registry.list_features()}
        assert "session_open_flag" in names
        assert "minutes_since_session_open" in names
