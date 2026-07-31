"""Market session/trading-calendar support (Milestone 10, Phase 1).

This module deliberately does NOT re-implement session/holiday/
maintenance-break arithmetic: `historical.calendar.TradingCalendar`
already models exactly this (weekly sessions, daily maintenance breaks,
holiday closures, explicit local-timezone-to-UTC conversion) and is
already tested. Re-deriving the same DST-sensitive arithmetic a second
time in this package would be the kind of duplicated infrastructure this
repository's own convention explicitly avoids (see `portfolio_risk.
identity`'s identical "do not duplicate infrastructure that already
exists" reasoning for `compute_content_id`). This module re-exports
`TradingCalendar` and its constituent types so a `market_data` caller
never has to reach into `historical` directly -- this package is meant to
be the single entry point for market/calendar data -- and adds the one
capability `historical.calendar` does not provide: deterministically
enumerating the sequence of candle OPEN times a `timeframe` is expected
to produce over a date range, which `quality.py` needs for missing-candle
and timeframe-gap detection."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import MarketCalendarError
from quant_platform.core.types import Timeframe
from quant_platform.historical.calendar import (
    DailyMaintenanceBreak,
    HolidayClosure,
    TradingCalendar,
    WeeklySession,
    default_xauusd_calendar,
)
from quant_platform.market_data.identity import require_tz_aware

__all__ = [
    "DailyMaintenanceBreak",
    "HolidayClosure",
    "TradingCalendar",
    "WeeklySession",
    "default_xauusd_calendar",
    "enumerate_expected_open_times",
]

_MAX_ENUMERATED_BARS = 500_000
"""A generous but finite bound -- this function walks bar-by-bar (it
cannot use a closed-form arithmetic shortcut once a calendar is
involved), so an unbounded caller-supplied range must fail loudly rather
than silently hang or exhaust memory."""


def enumerate_expected_open_times(
    calendar: TradingCalendar, *, timeframe: Timeframe, start: datetime, end: datetime,
) -> tuple[datetime, ...]:
    """Every `timeframe`-aligned open time in `[start, end)` at which the
    market is open per `calendar` -- i.e. every timestamp a complete,
    non-anomalous candle series is expected to have a bar for. Purely a
    function of `calendar`/`timeframe`/`start`/`end`: calling it twice
    with the same inputs always returns the same sequence, which is what
    lets `quality.py`'s missing-candle/timeframe-gap detection be
    deterministic rather than dependent on wall-clock state."""
    require_tz_aware(start, field_name="start")
    require_tz_aware(end, field_name="end")
    if end <= start:
        raise MarketCalendarError(f"end ({end}) must be after start ({start})")
    duration = timeframe.duration
    total_bars = (end - start) // duration
    if total_bars > _MAX_ENUMERATED_BARS:
        raise MarketCalendarError(
            f"enumerate_expected_open_times: range [{start}, {end}) at {timeframe.value} would enumerate "
            f"{total_bars} bars, exceeding the {_MAX_ENUMERATED_BARS} bound -- this function is intended for "
            "gap-sized/report-sized ranges, not multi-year bulk enumeration."
        )
    open_times: list[datetime] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    step = pd.Timedelta(duration)
    while cursor < end_ts:
        if calendar.is_open_at(cursor):
            open_times.append(cursor.to_pydatetime())
        cursor += step
    return tuple(open_times)
