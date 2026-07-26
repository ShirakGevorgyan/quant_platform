"""Temporal features -- computed purely from a bar's own `open_time` (and,
for session-relative features, an explicit `historical.calendar.
TradingCalendar`). Zero leakage risk by construction: a bar's `open_time`
is known no later than the bar itself opens, which is strictly before its
close/availability instant -- there is no rolling window, no future lookup,
nothing that could reach past the current row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import ConfigurationError
from quant_platform.core.types import Timeframe
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicySpec
from quant_platform.features.registry import FeatureRegistry
from quant_platform.historical.calendar import TradingCalendar, WeeklySession

_WEEK_MINUTES = 7 * 24 * 60


def hour_of_day(open_time: pd.Series) -> pd.Series:
    result: pd.Series = open_time.dt.hour.astype("float64")
    return result


def day_of_week(open_time: pd.Series) -> pd.Series:
    result: pd.Series = open_time.dt.dayofweek.astype("float64")
    return result


def month_of_year(open_time: pd.Series) -> pd.Series:
    result: pd.Series = open_time.dt.month.astype("float64")
    return result


def cyclical_encode(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2.0 * np.pi * values / period
    return pd.Series(np.sin(angle), index=values.index), pd.Series(np.cos(angle), index=values.index)


def _single_weekly_session(calendar: TradingCalendar) -> WeeklySession:
    if len(calendar.weekly_sessions) != 1:
        raise ConfigurationError(
            "Session-relative temporal features (minutes_since_session_open, "
            "market_open_proximity) require a TradingCalendar with EXACTLY ONE weekly session; "
            f"{calendar.name!r} defines {len(calendar.weekly_sessions)}. This is a deliberate, "
            "documented scope limitation -- see docs/feature_engineering.md.",
            context={"calendar": calendar.name, "session_count": len(calendar.weekly_sessions)},
        )
    return calendar.weekly_sessions[0]


def _local_minute_offset(open_time: pd.Series, calendar: TradingCalendar) -> pd.Series:
    local = open_time.dt.tz_convert(calendar.local_tz.to_tzinfo())
    result: pd.Series = (local.dt.dayofweek * 24 * 60 + local.dt.hour * 60 + local.dt.minute).astype("float64")
    return result


def minutes_since_session_open(open_time: pd.Series, calendar: TradingCalendar) -> pd.Series:
    """Minutes elapsed since the most recent occurrence of the calendar's
    (single) weekly session open, using the same circular (mod one week)
    arithmetic as `TradingCalendar._within_weekly_session` -- so a bar just
    after the weekly open reports a small value and one just before the
    NEXT open (a full week later) reports a value approaching one week,
    with no special-cased wraparound branch needed."""
    session = _single_weekly_session(calendar)
    moment_offset = _local_minute_offset(open_time, calendar)
    open_offset = session.open_weekday * 24 * 60 + session.open_time.hour * 60 + session.open_time.minute
    result: pd.Series = (moment_offset - open_offset) % _WEEK_MINUTES
    return result


def market_open_proximity(open_time: pd.Series, calendar: TradingCalendar) -> pd.Series:
    """Minutes to the NEAREST session boundary (whichever of "since open" or
    "until close" is smaller) -- small near either edge of the trading
    week, largest mid-session."""
    session = _single_weekly_session(calendar)
    open_offset = session.open_weekday * 24 * 60 + session.open_time.hour * 60 + session.open_time.minute
    close_offset = session.close_weekday * 24 * 60 + session.close_time.hour * 60 + session.close_time.minute
    session_length = (close_offset - open_offset) % _WEEK_MINUTES
    since_open = minutes_since_session_open(open_time, calendar)
    until_close = session_length - since_open
    result: pd.Series = pd.concat([since_open, until_close], axis=1).min(axis=1)
    return result


def session_open_flag(open_time: pd.Series, calendar: TradingCalendar) -> pd.Series:
    result: pd.Series = open_time.map(calendar.is_open_at).astype(bool)
    return result


def register_core_temporal_features(
    registry: FeatureRegistry, *, timeframe: Timeframe, calendar: TradingCalendar | None = None
) -> None:
    """Register the core temporal feature family. `hour_of_day`/
    `day_of_week`/`month_of_year`/their cyclical encodings are always
    registered (no calendar needed); `session_open_flag`/
    `minutes_since_session_open`/`market_open_proximity` are only
    registered if `calendar` is given (and, for the latter two, only if it
    defines exactly one weekly session -- see `_single_weekly_session`)."""
    null_policy = MissingPolicySpec()

    def _register(name: str, *, description: str, compute_fn: object) -> None:
        spec = FeatureSpec(
            name=name, version="1", description=description, category=FeatureCategory.TEMPORAL,
            required_inputs=("open_time",), source_symbols=(), source_timeframe=timeframe,
            output_dtype="float64", lookback_bars=0, warmup_bars=0, null_policy=null_policy,
        )
        registry.register(FeatureDefinition(spec=spec, compute=compute_fn))  # type: ignore[arg-type]

    _register(
        "hour_of_day", description="UTC hour of the bar's own open_time (0-23).",
        compute_fn=lambda ctx: hour_of_day(ctx.base_df["open_time"]),
    )
    _register(
        "day_of_week", description="Day of week of open_time (Monday=0 .. Sunday=6).",
        compute_fn=lambda ctx: day_of_week(ctx.base_df["open_time"]),
    )
    _register(
        "month_of_year", description="Calendar month of open_time (1-12).",
        compute_fn=lambda ctx: month_of_year(ctx.base_df["open_time"]),
    )
    _register(
        "hour_of_day_sin", description="sin(2*pi*hour/24).",
        compute_fn=lambda ctx: cyclical_encode(hour_of_day(ctx.base_df["open_time"]), 24.0)[0],
    )
    _register(
        "hour_of_day_cos", description="cos(2*pi*hour/24).",
        compute_fn=lambda ctx: cyclical_encode(hour_of_day(ctx.base_df["open_time"]), 24.0)[1],
    )
    _register(
        "day_of_week_sin", description="sin(2*pi*day_of_week/7).",
        compute_fn=lambda ctx: cyclical_encode(day_of_week(ctx.base_df["open_time"]), 7.0)[0],
    )
    _register(
        "day_of_week_cos", description="cos(2*pi*day_of_week/7).",
        compute_fn=lambda ctx: cyclical_encode(day_of_week(ctx.base_df["open_time"]), 7.0)[1],
    )

    if calendar is None:
        return

    _register(
        "session_open_flag", description=f"True if the bar's open_time falls inside calendar {calendar.name!r}'s open session.",
        compute_fn=lambda ctx: session_open_flag(ctx.base_df["open_time"], calendar),
    )
    if len(calendar.weekly_sessions) == 1:
        _register(
            "minutes_since_session_open",
            description=f"Minutes since the most recent weekly session open, per calendar {calendar.name!r}.",
            compute_fn=lambda ctx: minutes_since_session_open(ctx.base_df["open_time"], calendar),
        )
        _register(
            "market_open_proximity",
            description=f"Minutes to the nearest session boundary (open or close), per calendar {calendar.name!r}.",
            compute_fn=lambda ctx: market_open_proximity(ctx.base_df["open_time"], calendar),
        )


__all__ = [
    "cyclical_encode",
    "day_of_week",
    "hour_of_day",
    "market_open_proximity",
    "minutes_since_session_open",
    "month_of_year",
    "register_core_temporal_features",
    "session_open_flag",
]
