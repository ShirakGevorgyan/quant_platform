"""The engine's core no-look-ahead primitive.

`TimeframeCursor` provides point-in-time access to a higher-timeframe OHLCV
series while a base (finer) timeframe clock advances through a backtest.

THE INVARIANT THIS CLASS EXISTS TO ENFORCE
--------------------------------------------------------------------------
A bar is only ever revealed once its CLOSE time -- open_time + timeframe
duration -- has elapsed relative to the current instant. It is never
revealed based on its OPEN time.

This is not a stylistic preference: comparing against open time is a real,
previously-shipped bug class. A candle labeled "09:00" on a 15-minute chart
spans 09:00-09:15 and its OHLC values are not fully known until 09:15. A
naive cursor that reveals the bar as soon as the clock reaches "09:00"
leaks up to (duration - 1 base-bar) of future information into every
signal evaluation that depends on it -- silently inflating backtested
performance. See `tests/unit/test_cursor_no_lookahead.py` for a property
based adversarial test that proves this implementation does not do that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LookaheadViolationError, SchemaError
from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import OHLCV_COLUMNS, Bar, Timeframe


def _to_utc_datetime64(timestamp: pd.Timestamp) -> np.datetime64:
    """Explicitly convert a tz-aware Timestamp to the naive-UTC
    `datetime64[ns]` representation used internally for fast comparisons.
    Doing this explicitly -- rather than relying on numpy's implicit (and
    warning-emitting) tz handling in `np.datetime64(timestamp)` -- keeps the
    conversion intent auditable and independent of numpy version behavior.

    Requires `timestamp` to be tz-aware. A naive timestamp is silently
    ambiguous about which real-world instant it represents, and this
    function has no way to know whether the caller intended it as UTC, a
    local wall-clock time, or something else -- guessing "naive means
    already UTC" is exactly the bug class `TimeframeCursor.__init__`
    rejects naive `open_time` columns to prevent (see its docstring): two
    DataFrames using different implicit tz conventions but both technically
    "monotonic" can silently desynchronize the cross-timeframe clock and
    leak future data. Failing loudly here, at the point of use, catches
    the case where an already-validated cursor is driven by a caller-
    constructed `as_of` that wasn't put through the same discipline.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError(
            f"TimeframeCursor requires tz-aware timestamps; got naive timestamp {ts}. "
            "Localize it explicitly (e.g. via quant_platform.core.time_utils.ensure_utc) "
            "so the intended real-world instant is unambiguous."
        )
    return ts.tz_convert("UTC").tz_localize(None).to_datetime64()


class TimeframeCursor:
    """Tracks how far into a higher-timeframe DataFrame a base-timeframe
    clock has advanced. `advance_to()` only ever moves forward -- it has no
    mechanism to reveal a bar out of order or ahead of its close time.
    """

    __slots__ = ("_close_times", "_data", "_idx", "_last_as_of", "_timeframe")

    def __init__(self, data: pd.DataFrame, timeframe: Timeframe) -> None:
        missing = set(OHLCV_COLUMNS) - set(data.columns)
        if missing:
            raise SchemaError(
                f"DataFrame is missing required OHLCV columns: {sorted(missing)}",
                context={"timeframe": timeframe.value},
            )

        if len(data) > 0:
            if not pd.api.types.is_datetime64_any_dtype(data["open_time"]):
                raise SchemaError(
                    "DataFrame open_time column must be a datetime64 dtype",
                    context={"timeframe": timeframe.value, "actual_dtype": str(data["open_time"].dtype)},
                )
            if data["open_time"].dt.tz is None:
                # Naive timestamps are ambiguous about which real-world instant
                # they represent. Silently treating "naive" as "already UTC" --
                # while a DIFFERENT timeframe fed to the same BacktestEngine is
                # properly tz-aware -- desynchronizes the cross-timeframe clock
                # and can leak future data into a signal evaluation (confirmed
                # via adversarial audit: a naive series mislabeled as a
                # UTC+9-equivalent wall clock revealed an hourly bar up to 45
                # minutes before it had actually closed in true UTC time).
                # Requiring tz-awareness here removes the ambiguity at its root.
                raise SchemaError(
                    "DataFrame open_time column must be tz-aware (UTC); naive timestamps are "
                    "ambiguous about which real-world instant they represent and risk silent "
                    "cross-timeframe misalignment. Localize explicitly first, e.g. via "
                    "quant_platform.core.time_utils.ensure_utc.",
                    context={"timeframe": timeframe.value},
                )

        if len(data) > 1:
            if not data["open_time"].is_monotonic_increasing:
                raise SchemaError(
                    "DataFrame open_time column must be monotonically increasing",
                    context={"timeframe": timeframe.value},
                )
            if data["open_time"].duplicated().any():
                # `is_monotonic_increasing` in pandas means non-decreasing, so
                # it alone does NOT reject duplicate (tied) open_time values;
                # a duplicate silently collapses to whichever row wins
                # `advance_to`'s `<=` comparison, discarding the other bar's
                # OHLC data with no warning (confirmed via adversarial audit).
                duplicate_count = int(data["open_time"].duplicated().sum())
                raise SchemaError(
                    f"DataFrame open_time column contains {duplicate_count} duplicate value(s); "
                    "each bar must have a unique open_time",
                    context={"timeframe": timeframe.value},
                )

        # Defensive copy: guarantees independence from the caller's original
        # DataFrame regardless of pandas' copy-on-write configuration, which
        # is version/setting dependent and not something this safety-critical
        # component should rely on implicitly.
        self._data: pd.DataFrame = data.reset_index(drop=True).copy()
        self._timeframe = timeframe

        # Internal fast-path representation for advance_to()'s hot loop only.
        # Converting tz-aware pandas Timestamps to numpy datetime64 yields
        # the correct UTC instant but drops the tz label; the public
        # `current_bar` property never reads from this array for that
        # reason -- it recomputes close_time via pandas Timestamp
        # arithmetic on the original (tz-aware) open_time instead.
        duration_ns = np.timedelta64(int(timeframe.duration.total_seconds()), "s")
        open_times = self._data["open_time"].to_numpy(dtype="datetime64[ns]")
        self._close_times: np.ndarray = open_times + duration_ns
        self._idx: int = -1
        self._last_as_of: np.datetime64 | None = None

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def current_index(self) -> int:
        """Index into the underlying DataFrame of the most recently
        revealed (fully closed) bar, or -1 if none has closed yet."""
        return self._idx

    @property
    def has_current_bar(self) -> bool:
        return self._idx >= 0

    def advance_to(self, as_of: pd.Timestamp) -> bool:
        """Reveal every bar whose close time is <= `as_of`. Returns True if
        the cursor moved (a new bar became visible this call).

        `as_of` must be monotonically non-decreasing across successive
        calls -- this cursor models a forward-only simulation clock, and a
        decreasing `as_of` almost always indicates a bug in the caller's
        iteration order rather than a legitimate use case.
        """
        as_of_np = _to_utc_datetime64(as_of)
        if self._last_as_of is not None and as_of_np < self._last_as_of:
            raise ValueError(
                f"TimeframeCursor.advance_to called with as_of={as_of} which is earlier "
                f"than the previous call's as_of={self._last_as_of}; the cursor clock "
                "must be monotonically non-decreasing."
            )
        self._last_as_of = as_of_np

        n = len(self._data)
        close_times = self._close_times
        idx = self._idx
        moved = False

        while idx + 1 < n and close_times[idx + 1] <= as_of_np:
            idx += 1
            moved = True

        self._idx = idx

        if idx >= 0 and close_times[idx] > as_of_np:
            # Structurally unreachable given the loop condition above; kept
            # as an explicit, cheap guard on the engine's core guarantee.
            raise LookaheadViolationError(
                "TimeframeCursor revealed a bar whose close time is after `as_of`",
                context={
                    "timeframe": self._timeframe.value,
                    "as_of": str(as_of),
                    "bar_close_time": str(close_times[idx]),
                },
            )
        return moved

    @property
    def current_bar(self) -> Bar | None:
        """The most recently revealed bar, or None if the cursor has not
        advanced past any bar yet (e.g. warm-up period)."""
        if self._idx < 0:
            return None
        row = self._data.iloc[self._idx]
        open_time = row["open_time"]
        return Bar(
            open_time=open_time,
            close_time=compute_close_time(open_time, self._timeframe),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def window(self, size: int) -> pd.DataFrame:
        """The last `size` revealed bars, oldest first, as a fresh DataFrame
        (never a view into the source, so callers may freely mutate it).
        Returns fewer than `size` rows during warm-up, and an empty
        DataFrame if no bar has been revealed yet."""
        if size <= 0:
            raise ValueError(f"window size must be positive, got {size}")
        if self._idx < 0:
            return self._data.iloc[0:0].copy()
        start = max(0, self._idx - size + 1)
        return self._data.iloc[start : self._idx + 1].reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self._data)
