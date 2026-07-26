"""Pure timestamp/timeframe arithmetic.

This module contains no engine state and no I/O -- every function is a pure,
deterministic transformation of its inputs, which keeps it trivially unit
testable and safe to use from both the hot backtest loop and offline data
validation tooling.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.core.types import Timeframe

_PANDAS_FREQ_ALIASES: dict[Timeframe, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.H12: "12h",
    Timeframe.D1: "1D",
}


def compute_close_time(open_time: datetime, timeframe: Timeframe) -> datetime:
    """The instant a bar's data is actually fully known: its open time plus
    the timeframe's duration. This is the ONLY correct value to compare
    against "now" when deciding whether a bar may be revealed to a strategy
    -- comparing against `open_time` directly is the exact class of bug that
    causes look-ahead bias in naive multi-timeframe backtesters."""
    return open_time + timeframe.duration


def to_pandas_freq(timeframe: Timeframe) -> str:
    """Pandas offset-alias string for `timeframe`, for use in `resample()`
    and `date_range()` calls in the data-validation and synthetic-data code."""
    return _PANDAS_FREQ_ALIASES[timeframe]


def ensure_utc(timestamp: pd.Timestamp | datetime) -> pd.Timestamp:
    """Normalize a timestamp to a timezone-aware, UTC `pandas.Timestamp`.
    Naive timestamps are assumed to already represent UTC (the platform's
    single internal time standard) and are localized rather than converted."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
