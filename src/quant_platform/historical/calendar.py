"""Market session/trading-calendar abstraction.

XAUUSD (spot gold, typically traded as a CFD against USD) is not a 24/7
instrument: most brokers close it for a weekly maintenance break (commonly
Friday evening through Sunday evening in the broker's server time) and many
also run a short daily maintenance break. Naively treating every absence of
a bar as a data-quality "gap" would flag these entirely expected closures as
corruption on every single week of history. This module gives the
validation layer (`historical.quality`) and the resampler
(`historical.resampling`) a way to distinguish "the market was expectedly
closed" from "a bar is missing that should be there."

Deliberately NOT hardcoded to one broker's schedule: every rule (weekly
session windows, daily maintenance breaks, ad-hoc holiday closures) is
configuration or injected data, because different brokers genuinely run
different XAUUSD schedules (server timezone, exact break times, holiday
calendars) and treating any one of them as universal would silently
misclassify real gaps as expected (or vice versa) for every other broker.

No external market-calendar dependency is used here: the full generality of
exchange-traded futures/equity calendars (product-specific rolls, exchange
holiday tables spanning decades, half-days) is out of scope for a spot/CFD
instrument whose actual closure schedule is "a weekly window plus a short
daily maintenance break plus occasional broker-announced holiday closures,"
which this small, explicit, testable model captures completely. This is a
known, documented limitation (see README) should a future milestone add an
exchange-traded instrument with a genuinely complex published calendar.

All session boundaries are defined in a single explicit timezone (never
inferred) and are normalized to UTC internally so every comparison against
canonical (UTC) bar timestamps is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta

import pandas as pd

from quant_platform.core.exceptions import ConfigurationError
from quant_platform.historical.timezones import SourceTimezone


@dataclass(frozen=True, slots=True)
class WeeklySession:
    """One weekly trading window, e.g. "opens Sunday 22:00, closes Friday
    22:00" (server time). `open_weekday`/`close_weekday` use Python's
    `date.weekday()` convention: Monday=0 .. Sunday=6."""

    open_weekday: int
    open_time: time
    close_weekday: int
    close_time: time

    def __post_init__(self) -> None:
        if not (0 <= self.open_weekday <= 6):
            raise ValueError(f"open_weekday must be 0-6, got {self.open_weekday}")
        if not (0 <= self.close_weekday <= 6):
            raise ValueError(f"close_weekday must be 0-6, got {self.close_weekday}")


@dataclass(frozen=True, slots=True)
class DailyMaintenanceBreak:
    """A recurring daily closure window (e.g. a 1-hour "rollover" break),
    in the calendar's configured local time. `start`/`end` may wrap past
    midnight (e.g. start=23:59, end=00:15) -- handled explicitly, not by
    silently truncating at day boundaries."""

    start: time
    end: time
    weekdays: frozenset[int] = field(default_factory=lambda: frozenset(range(7)))


@dataclass(frozen=True, slots=True)
class HolidayClosure:
    """An exceptional full-day (or explicit sub-day) closure on top of the
    regular weekly schedule, e.g. a broker-announced Christmas closure."""

    closure_date: date
    start: time = time(0, 0)
    end: time = time(23, 59, 59, 999999)
    description: str = ""


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """A complete, explicit XAUUSD (or other instrument) trading schedule
    for one broker/venue. Every rule is expressed in `local_tz` and
    converted to UTC on demand -- never assumed."""

    local_tz: SourceTimezone
    weekly_sessions: tuple[WeeklySession, ...]
    maintenance_breaks: tuple[DailyMaintenanceBreak, ...] = ()
    holidays: tuple[HolidayClosure, ...] = ()
    name: str = "default"

    def __post_init__(self) -> None:
        if not self.weekly_sessions:
            raise ConfigurationError(
                f"TradingCalendar {self.name!r} must define at least one weekly session"
            )

    def is_expected_closure(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> bool:
        """True if the half-open UTC interval [start_utc, end_utc) falls
        entirely within an expected closure (outside every weekly session,
        inside a maintenance break, or inside a holiday closure) -- i.e. a
        missing bar spanning exactly this interval is NOT an anomaly."""
        if end_utc <= start_utc:
            raise ValueError(f"end_utc ({end_utc}) must be after start_utc ({start_utc})")

        tzinfo_obj = self.local_tz.to_tzinfo()
        local_start = start_utc.tz_convert(tzinfo_obj)
        local_end = end_utc.tz_convert(tzinfo_obj)

        cursor = local_start
        # Walk the interval in coarse steps and verify every sub-point is
        # covered by *some* closure rule. A closure is "entire interval
        # closed" only if there is no instant in [local_start, local_end)
        # that falls inside an open weekly session AND outside every
        # maintenance break/holiday.
        step = timedelta(minutes=1)
        # Bound the number of iterations for pathologically large intervals
        # (this function is intended for gap-sized intervals, i.e. a small
        # number of missing bars, not multi-year ranges).
        max_steps = 100_000
        steps = 0
        while cursor < local_end and steps < max_steps:
            if self._is_open_at_local(cursor):
                return False
            cursor += step
            steps += 1
        if steps >= max_steps:
            raise ValueError(
                "is_expected_closure interval too large for per-minute evaluation "
                f"({start_utc} -> {end_utc}); this function is intended for gap-sized "
                "intervals, not bulk range queries."
            )
        return True

    def _is_open_at_local(self, moment: pd.Timestamp) -> bool:
        if self._in_any_holiday(moment):
            return False
        if self._in_any_maintenance_break(moment):
            return False
        return self._in_any_weekly_session(moment)

    def _in_any_holiday(self, moment: pd.Timestamp) -> bool:
        moment_date = moment.date()
        moment_time = moment.time()
        for holiday in self.holidays:
            if holiday.closure_date == moment_date and holiday.start <= moment_time <= holiday.end:
                return True
        return False

    def _in_any_maintenance_break(self, moment: pd.Timestamp) -> bool:
        weekday = moment.weekday()
        moment_time = moment.time()
        for brk in self.maintenance_breaks:
            if weekday not in brk.weekdays and (weekday - 1) % 7 not in brk.weekdays:
                # Also check the previous weekday for breaks that wrap past
                # midnight (e.g. start=23:30 on weekday W, end=00:30, which
                # is "still weekday W's break" even after local midnight).
                continue
            if brk.start <= brk.end:
                if brk.start <= moment_time < brk.end and weekday in brk.weekdays:
                    return True
            else:
                # Wraps midnight: active from `start` to 24:00 on `weekday`,
                # and from 00:00 to `end` on `weekday + 1`.
                if moment_time >= brk.start and weekday in brk.weekdays:
                    return True
                if moment_time < brk.end and (weekday - 1) % 7 in brk.weekdays:
                    return True
        return False

    def _in_any_weekly_session(self, moment: pd.Timestamp) -> bool:
        return any(self._within_weekly_session(moment, session) for session in self.weekly_sessions)

    @staticmethod
    def _within_weekly_session(moment: pd.Timestamp, session: WeeklySession) -> bool:
        # Express both the session boundaries and `moment` as minute-offsets
        # from the start of a fixed reference week (Monday 00:00 = 0), so a
        # session that spans a week wrap (e.g. Sun 22:00 -> Fri 22:00, which
        # is the *complement* of a short weekend closure, not a literal
        # forward span) is handled by comparing on a circular (mod
        # 7*1440-minute) timeline rather than needing separate wrap-around
        # branches for every case.
        week_minutes = 7 * 24 * 60
        moment_offset = moment.weekday() * 24 * 60 + moment.hour * 60 + moment.minute
        open_offset = session.open_weekday * 24 * 60 + session.open_time.hour * 60 + session.open_time.minute
        close_offset = session.close_weekday * 24 * 60 + session.close_time.hour * 60 + session.close_time.minute

        span = (close_offset - open_offset) % week_minutes
        position = (moment_offset - open_offset) % week_minutes
        return position < span


def default_xauusd_calendar(local_tz: SourceTimezone) -> TradingCalendar:
    """A commonly-used illustrative XAUUSD schedule: opens Sunday 23:00,
    closes Friday 23:00 (broker server time), with a daily 23:00-23:01
    settlement break. This is NOT universal -- it is provided as a
    reasonable, explicit, override-everything default so a caller who has
    not yet obtained their specific broker's published schedule still gets
    correct weekend-gap handling rather than none at all; production use
    should supply the actual broker-published schedule via
    `TradingCalendar` directly."""
    return TradingCalendar(
        local_tz=local_tz,
        weekly_sessions=(
            WeeklySession(
                open_weekday=6, open_time=time(23, 0),
                close_weekday=4, close_time=time(23, 0),
            ),
        ),
        maintenance_breaks=(
            DailyMaintenanceBreak(start=time(23, 0), end=time(23, 1)),
        ),
        name="illustrative_default_xauusd",
    )


__all__ = [
    "DailyMaintenanceBreak",
    "HolidayClosure",
    "TradingCalendar",
    "WeeklySession",
    "default_xauusd_calendar",
]
