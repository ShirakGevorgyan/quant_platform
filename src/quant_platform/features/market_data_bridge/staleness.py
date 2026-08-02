"""Staleness reporting (Milestone 10, Phase 4D, spec Section 11):
source-specific staleness measured from each source's OWN availability
proof (a macro observation's `release_time`; a cross-asset bar's
availability-shifted `open_time`) -- never one global threshold across
unlike frequencies (a daily yield and monthly CPI are never compared
against the same cutoff).

REUSES THE EXISTING AS-OF/CLOSE-TIME MACHINERY, ADDS NO NEW ALIGNMENT
LOGIC. `features.alignment.as_of_join_external`/`align_higher_timeframe`
already compute `{name}_age_seconds`/`{name}_is_stale`/
`{prefix}seconds_since_close` as a side effect of the join itself (see
that module) -- this module's only job is running that SAME join
independently of whichever features a caller actually registered (so a
staleness report is available even if, say, no `macro_{source}_is_stale`
feature was requested) and applying a source-specific age THRESHOLD on
top of the join's own release/close-boundedness check, producing one
deterministic `StalenessFinding` per source. It never carries a value
forward beyond what the as-of join itself already does (no new
forward-fill), and never mutates a source frame."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe, as_of_join_external

__all__ = ["StalenessFinding", "evaluate_cross_asset_staleness", "evaluate_macro_staleness"]


@dataclass(frozen=True, slots=True)
class StalenessFinding:
    source_kind: str
    """`"macro"` or `"cross_asset"`."""
    source_name: str
    total_row_count: int
    unavailable_row_count: int
    """Rows with no qualifying release/close at all as of that row's
    availability instant (warm-up, or a source with no data yet)."""
    stale_row_count: int
    """Rows with a qualifying release/close, but older than `threshold`
    (0 if `threshold` is `None` -- no threshold was configured)."""
    stale_fraction: float
    max_observed_age_seconds: float | None
    threshold_seconds: float | None


def evaluate_macro_staleness(
    base_availability_times: pd.Series, macro_df: pd.DataFrame, *, source_name: str, threshold: pd.Timedelta | None,
) -> StalenessFinding:
    joined = as_of_join_external(base_availability_times, macro_df, value_column="value", output_name="level")
    n = len(joined)
    unavailable = int(joined["level_is_stale"].sum())
    age_seconds = joined["level_age_seconds"]
    threshold_seconds = threshold.total_seconds() if threshold is not None else None
    if threshold_seconds is None:
        stale = 0
    else:
        stale = int(((age_seconds > threshold_seconds) & ~joined["level_is_stale"]).sum())
    max_age = float(age_seconds.max()) if n and age_seconds.notna().any() else None
    return StalenessFinding(
        source_kind="macro", source_name=source_name, total_row_count=n, unavailable_row_count=unavailable,
        stale_row_count=stale, stale_fraction=((unavailable + stale) / n if n else 0.0), max_observed_age_seconds=max_age,
        threshold_seconds=threshold_seconds,
    )


def evaluate_cross_asset_staleness(
    base_availability_times: pd.Series, cross_asset_df: pd.DataFrame, *, source_name: str, timeframe: Timeframe,
    threshold: pd.Timedelta | None,
) -> StalenessFinding:
    aligned = align_higher_timeframe(base_availability_times, cross_asset_df, timeframe)
    prefix = f"htf_{timeframe.value}_"
    n = len(aligned)
    unavailable = int((aligned[f"{prefix}bar_index"] == -1).sum())
    age_seconds = aligned[f"{prefix}seconds_since_close"]
    threshold_seconds = threshold.total_seconds() if threshold is not None else None
    if threshold_seconds is None:
        stale = 0
    else:
        stale = int(((age_seconds > threshold_seconds) & (aligned[f"{prefix}bar_index"] != -1)).sum())
    max_age = float(age_seconds.max()) if n and age_seconds.notna().any() else None
    return StalenessFinding(
        source_kind="cross_asset", source_name=source_name, total_row_count=n, unavailable_row_count=unavailable,
        stale_row_count=stale, stale_fraction=((unavailable + stale) / n if n else 0.0), max_observed_age_seconds=max_age,
        threshold_seconds=threshold_seconds,
    )
