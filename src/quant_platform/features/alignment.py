"""Point-in-time alignment: the ONLY sanctioned way a feature may pull in
data from a higher timeframe, a cross-asset instrument, or an external/macro
source. Every function here is a deliberately narrow, explicit API --
never a raw `pd.merge`/index-align -- specifically because an ordinary join
has no concept of "as of this instant" and silently permits future
information to leak in through misaligned or coincidentally-matching keys.

Two distinct alignment problems, two functions:

`align_higher_timeframe`
    For each base-timeframe row's availability instant (its CLOSE time),
    find the most recent higher-timeframe bar whose OWN close time is
    `<=` that instant. This is exactly `multiframe.cursor.TimeframeCursor`'s
    "reveal by close time, never by open time" rule, reimplemented here as
    a vectorized (`numpy.searchsorted`) batch operation for dataset-building
    throughput rather than a live streaming cursor -- see
    `tests/unit/features/test_alignment_boundaries.py` for the adversarial
    proof that this vectorized implementation and `TimeframeCursor`,
    driven bar-by-bar over the same data, always agree.

`as_of_join_external`
    For each base row's availability instant, find the most recently
    RELEASED external observation (backward-looking `pandas.merge_asof`,
    keyed on an explicit release/availability timestamp -- never the
    observation's own reference period). A value "for January" published
    in February is invisible to every row before that February release
    instant; once released, it remains visible until superseded by a later
    release, which is exactly the vintage-aware behavior real point-in-time
    macro research requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import PointInTimeViolationError, SchemaError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import require_utc


def align_higher_timeframe(
    base_close_times: pd.Series, higher_df: pd.DataFrame, higher_timeframe: Timeframe
) -> pd.DataFrame:
    """Row-aligned (position, not label, aligned -- same length/order as
    `base_close_times`) higher-timeframe snapshot: every `higher_df` column
    (prefixed `htf_{timeframe}_`), plus `_close_time`, `_bar_index` (-1
    during warm-up, before any higher-timeframe bar has closed), and
    `_seconds_since_close`. `higher_df` must have a tz-aware UTC `open_time`
    column, sorted ascending with no duplicates -- a caller contract,
    enforced by raising rather than silently sorting/deduplicating (this
    module never repairs data, only aligns it)."""
    require_utc(base_close_times, context="align_higher_timeframe.base_close_times")
    if "open_time" not in higher_df.columns:
        raise SchemaError("align_higher_timeframe: higher_df is missing an 'open_time' column")

    higher_sorted = higher_df.reset_index(drop=True)
    if len(higher_sorted) > 0:
        require_utc(higher_sorted["open_time"], context="align_higher_timeframe.higher_df")
    if len(higher_sorted) > 1:
        if not higher_sorted["open_time"].is_monotonic_increasing:
            raise PointInTimeViolationError(
                "align_higher_timeframe: higher_df.open_time must be sorted ascending",
                context={"timeframe": higher_timeframe.value},
            )
        if higher_sorted["open_time"].duplicated().any():
            raise PointInTimeViolationError(
                "align_higher_timeframe: higher_df.open_time contains duplicate values",
                context={"timeframe": higher_timeframe.value},
            )

    n = len(base_close_times)
    prefix = f"htf_{higher_timeframe.value}_"
    if len(higher_sorted) == 0:
        result = higher_sorted.add_prefix(prefix).reindex(range(n))
        result[f"{prefix}close_time"] = pd.Series(pd.NaT, index=range(n))
        result[f"{prefix}bar_index"] = -1
        result[f"{prefix}seconds_since_close"] = np.nan
        return result

    # `.astype("int64")` on a tz-aware datetime64 column returns the epoch
    # value in THAT column's own storage resolution (ns vs us vs ms differ
    # by construction path, e.g. `pd.date_range` vs a Parquet round-trip --
    # see `historical.models.coerce_historical_dtypes`'s identical concern).
    # Both sides must be normalized to the SAME resolution before the
    # int64 comparison below, or `searchsorted` silently compares
    # differently-scaled integers -- a wrong-answer bug, not a crash.
    higher_close_ns = (
        (higher_sorted["open_time"] + higher_timeframe.duration).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    )
    base_close_ns = (
        pd.Series(base_close_times).reset_index(drop=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    )

    # Index of the LAST higher-timeframe bar whose close time is <= this
    # base row's availability instant -- the vectorized equivalent of
    # `TimeframeCursor.advance_to(as_of)` for every row at once.
    positions = np.searchsorted(higher_close_ns, base_close_ns, side="right") - 1
    valid = positions >= 0
    safe_positions = np.where(valid, positions, 0)

    selected = higher_sorted.iloc[safe_positions].reset_index(drop=True)
    valid_series = pd.Series(valid, index=selected.index)
    selected = selected.mask(~valid_series)

    result = selected.add_prefix(prefix)
    result[f"{prefix}close_time"] = result[f"{prefix}open_time"] + higher_timeframe.duration
    result[f"{prefix}bar_index"] = pd.Series(np.where(valid, positions, -1), index=result.index)
    elapsed = pd.Series(base_close_times.to_numpy(), index=result.index) - result[f"{prefix}close_time"]
    result[f"{prefix}seconds_since_close"] = elapsed.dt.total_seconds()
    return result


def as_of_join_external(
    base_availability_times: pd.Series,
    external_df: pd.DataFrame,
    *,
    value_column: str,
    release_time_column: str = "release_time",
    tolerance: pd.Timedelta | None = None,
    output_name: str | None = None,
) -> pd.DataFrame:
    """Row-aligned as-of snapshot of one external/macro value column: for
    every base row, the most recently RELEASED observation as of that
    instant (backward-looking, `release_time <= availability_time`),
    subject to an optional maximum staleness `tolerance`. Returns
    `{name}` (the value, NaN if none released yet or beyond tolerance),
    `{name}_release_time`, `{name}_age_seconds`, and `{name}_is_stale`
    (True if no qualifying release exists at all -- distinct from merely
    "old but within tolerance").

    Duplicate `release_time_column` values (e.g. a same-instant correction)
    are resolved deterministically: `external_df` is stable-sorted by
    release time first, so among exactly-tied releases the one appearing
    LATER in the input (assumed to be the more authoritative/final one,
    consistent with typical vintage ordering) is what a query at or after
    that instant observes -- see
    `tests/unit/features/test_macro_alignment.py::test_duplicate_release_timestamps_resolve_deterministically`.
    """
    require_utc(base_availability_times, context="as_of_join_external.base_availability_times")
    missing = {value_column, release_time_column} - set(external_df.columns)
    if missing:
        raise SchemaError(f"as_of_join_external: external_df is missing column(s) {sorted(missing)}")
    if len(external_df) > 0:
        require_utc(external_df[release_time_column], context="as_of_join_external.external_df")

    name = output_name if output_name is not None else value_column
    n = len(base_availability_times)
    if len(external_df) == 0:
        return pd.DataFrame(
            {
                name: pd.Series([np.nan] * n),
                f"{name}_release_time": pd.Series([pd.NaT] * n),
                f"{name}_age_seconds": pd.Series([np.nan] * n),
                f"{name}_is_stale": pd.Series([True] * n),
            }
        )

    external_sorted = external_df.sort_values(release_time_column, kind="mergesort").reset_index(drop=True)
    # Normalize both merge keys to the same datetime64 storage resolution
    # before merge_asof -- see the identical concern/fix in
    # `align_higher_timeframe` above; `merge_asof` refuses to run at all on
    # mismatched resolutions (raises `MergeError`), so this is required for
    # correctness, not just an optimization.
    left = pd.DataFrame(
        {"_at": pd.Series(base_availability_times).reset_index(drop=True).astype("datetime64[ns, UTC]")}
    )
    right = external_sorted[[release_time_column, value_column]].rename(
        columns={release_time_column: "_release", value_column: "_value"}
    )
    right["_release"] = right["_release"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(
        left, right, left_on="_at", right_on="_release", direction="backward",
        tolerance=tolerance, allow_exact_matches=True,
    )
    age = merged["_at"] - merged["_release"]
    return pd.DataFrame(
        {
            name: merged["_value"],
            f"{name}_release_time": merged["_release"],
            f"{name}_age_seconds": age.dt.total_seconds(),
            f"{name}_is_stale": merged["_release"].isna(),
        }
    )


__all__ = ["align_higher_timeframe", "as_of_join_external"]
