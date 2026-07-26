"""Leak-free resampling of canonical OHLCV bars to a coarser timeframe.

THE CORE CORRECTNESS CONCERN THIS MODULE EXISTS TO ADDRESS: a derived
(higher-timeframe) bar must never be presented as "complete" if the source
data does not actually cover its entire bucket window. This is the offline,
batch-resampling counterpart to `multiframe.cursor.TimeframeCursor`'s
"reveal by close time, never by open time" invariant -- the failure mode
here is not a live look-ahead leak (there is no live clock in a batch
resampling pass), but the same underlying mistake in a different guise: if
the tail of the source history ends mid-hour and this module built an H1
bar for that partial hour anyway, anything downstream that trusts "this bar
is closed and final" would be acting on data that was not actually fully
observed -- exactly the kind of silent-completeness bug the rest of this
platform is built to make structurally impossible.

Design choices made deliberately, not by accident:

  * Bucket boundaries are computed by pure UTC-epoch arithmetic
    (`epoch_ns // target_duration_ns`), never via `pandas.DataFrame.resample`.
    `resample()`'s `label`/`closed` defaults are a well-known source of
    off-by-one ambiguity (whether a bucket's label is its start or end, and
    whether the boundary bar belongs to the bucket before or after it) --
    sidestepping the API removes that whole ambiguity class rather than
    configuring around it.
  * A derived bar is "complete" if and only if its bucket's close time
    (`bucket_start + target_duration`) is `<=` the source data's own
    coverage end (`last source open_time + source duration`) -- exactly
    mirroring `TimeframeCursor`'s close-time-based reveal rule. Incomplete
    trailing buckets are, by default, dropped entirely
    (`DerivedBarPolicy.REJECT_INCOMPLETE`); a caller that explicitly wants
    to see the still-forming current bar (e.g. a live/near-real-time view)
    can request `RETAIN_INCOMPLETE`, but every returned row -- under either
    policy -- carries an explicit `is_complete` column, so "this looks like
    a normal bar" can never be confused with "this bar is actually done."
  * Session/maintenance-break-spanning buckets are NOT specially split or
    dropped -- a bucket that happens to span an expected weekend/maintenance
    closure simply aggregates fewer underlying source bars than a fully
    "loaded" bucket would, which is both correct (real markets have that
    gap too) and fully transparent: every derived bar carries a
    `source_bar_count` column recording exactly how many source bars
    contributed, so a caller can distinguish a normal bucket from a thin
    one caused by an expected closure without this module having to
    duplicate `historical.calendar`'s session logic.
  * Standard OHLCV aggregation: open=first, high=max, low=min, close=last,
    tick_volume=sum, real_volume=sum (both are genuinely additive counts).
    `spread` is NOT additive -- it is a point-in-time quoted cost, and
    summing or taking-first/last would each be an arbitrary choice with no
    clear justification. This module uses the MEAN spread across the
    bucket's constituent bars (rounded to the nearest integer point, since
    spread is stored in whole points) as the most representative single
    figure for downstream cost modeling; this is a deliberate, documented
    choice, not an oversight -- a caller with a different requirement
    (e.g. max spread for a conservative cost estimate) can compute it
    directly from the source data instead.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import ResamplingError
from quant_platform.core.types import Timeframe
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS, validate_historical_schema

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = (*RAW_HISTORICAL_COLUMNS, "source_bar_count", "is_complete")


class DerivedBarPolicy(Enum):
    REJECT_INCOMPLETE = "REJECT_INCOMPLETE"
    """Drop any trailing bucket whose window has not fully elapsed within
    the source data's coverage. The default, and the only safe choice for
    building canonical derived-timeframe storage."""
    RETAIN_INCOMPLETE = "RETAIN_INCOMPLETE"
    """Keep every bucket, including a still-forming trailing one, with
    `is_complete=False` on the affected row(s). Intended for callers that
    explicitly want visibility into the current, not-yet-closed bar (e.g.
    a live dashboard) and will check `is_complete` themselves."""


def resample_ohlcv(
    source_df: pd.DataFrame,
    *,
    source_timeframe: Timeframe,
    target_timeframe: Timeframe,
    policy: DerivedBarPolicy = DerivedBarPolicy.REJECT_INCOMPLETE,
) -> pd.DataFrame:
    """Aggregate `source_df` (canonical `RAW_HISTORICAL_COLUMNS` bars at
    `source_timeframe`) into `target_timeframe` bars. `source_df` must
    already be sorted ascending by `open_time` with no duplicates -- this
    is a caller contract (enforced by raising `ResamplingError`, not
    silently sorted/deduplicated here) since resampling already-canonical,
    already-validated data should never need to repair it again."""
    started_at = time.perf_counter()
    if target_timeframe.duration <= source_timeframe.duration:
        raise ResamplingError(
            f"target timeframe {target_timeframe.value} must be strictly coarser than "
            f"source timeframe {source_timeframe.value}",
            context={"source": source_timeframe.value, "target": target_timeframe.value},
        )
    source_seconds = int(source_timeframe.duration.total_seconds())
    target_seconds = int(target_timeframe.duration.total_seconds())
    if target_seconds % source_seconds != 0:
        raise ResamplingError(
            f"target timeframe {target_timeframe.value} duration is not an exact multiple of "
            f"source timeframe {source_timeframe.value} duration",
            context={"source": source_timeframe.value, "target": target_timeframe.value},
        )

    validate_historical_schema(source_df, context="resample_ohlcv")
    if len(source_df) == 0:
        logger.info(
            "resample complete: source_timeframe=%s target_timeframe=%s policy=%s rows_in=0 rows_out=0 "
            "duration_s=%.3f",
            source_timeframe.value, target_timeframe.value, policy.value, time.perf_counter() - started_at,
        )
        return _empty_result()

    open_time = source_df["open_time"]
    if not open_time.is_monotonic_increasing:
        raise ResamplingError("source_df.open_time must be sorted ascending before resampling")
    if open_time.duplicated().any():
        raise ResamplingError("source_df.open_time contains duplicate values")

    epoch_ns = open_time.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    target_ns = target_seconds * 1_000_000_000
    bucket_start_ns = (epoch_ns // target_ns) * target_ns

    working = source_df.copy()
    working["_bucket"] = pd.to_datetime(bucket_start_ns, utc=True)

    grouped = working.groupby("_bucket", sort=True)
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        real_volume=("real_volume", "sum"),
        spread=("spread", "mean"),
        source_bar_count=("open", "size"),
    ).reset_index().rename(columns={"_bucket": "open_time"})

    result["tick_volume"] = result["tick_volume"].astype(np.int64)
    result["real_volume"] = result["real_volume"].astype(np.int64)
    result["spread"] = result["spread"].round().astype(np.int64)
    result["source_bar_count"] = result["source_bar_count"].astype(np.int64)

    last_source_coverage_end = pd.Timestamp(open_time.iloc[-1]) + source_timeframe.duration
    bucket_close = result["open_time"] + target_timeframe.duration
    result["is_complete"] = bucket_close <= last_source_coverage_end

    if policy is DerivedBarPolicy.REJECT_INCOMPLETE:
        result = result.loc[result["is_complete"]].reset_index(drop=True)

    result = result[list(_OUTPUT_COLUMNS)]
    logger.info(
        "resample complete: source_timeframe=%s target_timeframe=%s policy=%s rows_in=%d rows_out=%d "
        "duration_s=%.3f",
        source_timeframe.value, target_timeframe.value, policy.value, len(source_df), len(result),
        time.perf_counter() - started_at,
    )
    return result


def _empty_result() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name in RAW_HISTORICAL_COLUMNS:
        columns[name] = pd.Series([], dtype="datetime64[ns, UTC]" if name == "open_time" else np.float64)
    columns["source_bar_count"] = pd.Series([], dtype=np.int64)
    columns["is_complete"] = pd.Series([], dtype=bool)
    return pd.DataFrame(columns)


__all__ = ["DerivedBarPolicy", "resample_ohlcv"]
