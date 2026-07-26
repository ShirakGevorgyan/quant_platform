"""Canonical domain model for the historical OHLCV data pipeline.

This defines the schema every stage of the pipeline (raw snapshots,
normalization, validation, canonical storage, resampling, the dataset
loader) agrees on. It is deliberately a DataFrame-level schema, not a new
per-row dataclass: `core.types.Bar` already exists as the lightweight,
per-step value the *engine* passes between components one bar at a time
(see its module docstring -- "never used to represent bulk series"), and
bulk historical data is exactly the "millions of rows, vectorized" case
that docstring says belongs in a DataFrame. Introducing a second,
differently-shaped per-row type for the same concept would fragment the
domain model rather than extend it; the boundary between the two is
handled explicitly in `historical.loader`, which projects the wider
canonical schema below down to `core.types.OHLCV_COLUMNS` for consumption
by `TimeframeCursor`/`BacktestEngine`.

Column semantics
----------------
open_time
    The bar's OPEN instant, timezone-aware UTC (enforced by
    `historical.timezones.require_utc`). This is the ONLY timestamp
    persisted. `close_time` is deliberately NOT stored: it is always
    exactly `open_time + timeframe.duration` (see
    `core.time_utils.compute_close_time`), and persisting a second,
    independently-computed column would create a class of bug where the
    stored close_time and the derived one silently disagree (e.g. after a
    timeframe-configuration change). `timeframe` itself is dataset-level
    metadata (one canonical dataset is always exactly one symbol and one
    timeframe -- consistent with how `core.types.OHLCV_COLUMNS` already
    omits a timeframe column for the same reason), not a per-row value.
open, high, low, close
    Trade/quote prices in the instrument's native price units, as `float64`.
    Decimal is deliberately NOT used: XAUUSD-scale prices (order of
    magnitude 10^3-10^4) with the 2-3 decimal places brokers actually
    publish need on the order of 6-7 significant decimal digits of
    precision, while `float64` carries ~15-17 -- more than an order of
    magnitude of headroom, so no realistic representable price loses
    precision (see `tests/unit/historical/test_models.py::TestFloatPrecision`
    for an explicit round-trip proof at realistic XAUUSD price/decimal
    scale). Introducing `Decimal` here would also force a lossy conversion
    right back to `float` at the `historical.loader` boundary, since
    `core.types.Bar`/the entire M1 engine is `float`-typed throughout --
    so the case for Decimal (exact decimal arithmetic) would be undermined
    by the very next stage regardless.
tick_volume
    Number of price ticks (quote changes) observed during the bar. This is
    what MT5's `copy_rates_range` always populates and what most CFD/spot
    FX and metals brokers report as their only volume proxy, since they
    have no visibility into a centralized exchange's true traded volume.
    Stored as `int64` (a tick count is an exact whole number).
real_volume
    Actual traded volume during the bar, when the source/broker genuinely
    reports it (typically 0 for OTC/CFD instruments like spot XAUUSD on
    most retail brokers -- MT5 populates this field but it is commonly 0
    for exactly that reason). Stored as `int64`. This pipeline never
    fabricates a real_volume value when the source does not provide one;
    an unavailable real_volume is `0`, not estimated from tick_volume.
spread
    The broker's quoted spread for the bar, in RAW BROKER SPREAD UNITS --
    for MT5 this is an integer number of "points" (the smallest quoted
    price increment for the symbol), NOT a price-unit value. Converting to
    a price-unit spread requires the symbol's point size (`trade_tick_size`
    in MT5 terms), which is broker/symbol/account-type specific and not
    always known at ingestion time; this pipeline stores the raw points
    value as-is (`int64`) and provides `spread_points_to_price` as an
    explicit, opt-in conversion a caller performs only once it has that
    point size, rather than silently guessing it during ingestion.

`RAW_HISTORICAL_COLUMNS` is the schema used by the raw snapshot store, the
normalization stage, and canonical storage. It intentionally does NOT
include `symbol`, `timeframe`, `source`, or broker identity as per-row
columns -- those are dataset-level lineage recorded once in the raw
snapshot metadata (`historical.raw_store.SnapshotMetadata`) and the dataset
manifest (`historical.manifest.DatasetManifest`), not duplicated across
every row of what may be millions of rows of history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import SchemaError
from quant_platform.historical.timezones import require_utc

RAW_HISTORICAL_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread",
)

_FLOAT_COLUMNS = ("open", "high", "low", "close")
_INT_COLUMNS = ("tick_volume", "real_volume", "spread")


def validate_historical_schema(df: pd.DataFrame, *, context: str) -> None:
    """Raise `SchemaError`/`TimezoneError` unless `df` conforms to
    `RAW_HISTORICAL_COLUMNS` with tz-aware UTC `open_time`. Does not mutate
    or coerce -- this is a strict boundary check, not a normalization step
    (normalization is `historical.timezones.localize_broker_timestamps`
    plus explicit dtype casting performed once, deliberately, by the
    source adapter that first constructs the DataFrame)."""
    missing = set(RAW_HISTORICAL_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(
            f"{context}: DataFrame is missing required historical columns: {sorted(missing)}",
            context={"missing": sorted(missing)},
        )
    if len(df) > 0:
        require_utc(df["open_time"], context=context)
        for column in _FLOAT_COLUMNS:
            if not pd.api.types.is_float_dtype(df[column]):
                raise SchemaError(
                    f"{context}: column {column!r} must be float64, got {df[column].dtype}"
                )
        for column in _INT_COLUMNS:
            if not pd.api.types.is_integer_dtype(df[column]):
                raise SchemaError(
                    f"{context}: column {column!r} must be an integer dtype, got {df[column].dtype}"
                )


def coerce_historical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast `df`'s OHLCV-adjacent columns to the canonical dtypes
    (`RAW_HISTORICAL_COLUMNS`). Does NOT touch `open_time` timezone
    handling -- callers must have already produced a tz-aware UTC
    `open_time` column via `historical.timezones.localize_broker_timestamps`
    before calling this; this function only fixes up numeric dtypes
    (source adapters commonly hand back platform-native integer widths,
    e.g. MT5 returns `tick_volume`/`real_volume` as unsigned 64-bit and
    `spread` as signed 32-bit, which this normalizes to a single consistent
    `int64` used throughout the rest of the pipeline).

    Also normalizes `open_time` to a single canonical `datetime64[ns, UTC]`
    storage resolution. Different construction paths produce different
    (but semantically identical) resolutions -- e.g. `pd.to_datetime(...,
    unit="s")` on the MT5 adapter's integer-second timestamps yields
    `datetime64[s, UTC]` -- and a Parquet round-trip does not reliably
    preserve whichever one was used (observed: a `datetime64[s, UTC]`
    column came back as `datetime64[ms, UTC]` after `to_parquet`/
    `read_parquet`, purely a pyarrow/pandas storage-representation detail
    with no data difference at all). Pinning one resolution here, at the
    single normalization boundary every source ultimately passes through,
    stops that representation drift from ever reaching `schema_fingerprint`
    -- see its own docstring for the belt-and-braces resolution-independent
    comparison it also uses, and
    `tests/unit/historical/test_models.py::TestDatetimeResolutionNormalization`
    for the regression test this fixes."""
    df = df.copy()
    df["open_time"] = df["open_time"].astype("datetime64[ns, UTC]")
    for column in _FLOAT_COLUMNS:
        df[column] = df[column].astype(np.float64)
    for column in _INT_COLUMNS:
        df[column] = df[column].astype(np.int64)
    return df[list(RAW_HISTORICAL_COLUMNS)]


def spread_points_to_price(spread_points: pd.Series | int, point_size: float) -> pd.Series | float:
    """Explicit, opt-in conversion of a raw broker spread (in points) to a
    price-unit spread, given the symbol's point size. Never applied
    automatically during ingestion -- see module docstring."""
    if point_size <= 0:
        raise ValueError(f"point_size must be positive, got {point_size}")
    return spread_points * point_size


def _fingerprint_dtype_str(series: pd.Series) -> str:
    """`str(dtype)` for a tz-aware datetime64 column includes its storage
    resolution (ns/us/ms/s), which a Parquet write/read round-trip does
    NOT reliably preserve even though the data is semantically unchanged
    (see `coerce_historical_dtypes`) -- normalizing that away here means
    `schema_fingerprint` reflects a genuine schema difference (wrong dtype,
    missing/wrong timezone) rather than an incidental storage detail,
    independently of whether the caller already normalized resolution via
    `coerce_historical_dtypes`."""
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return f"datetime64[tz={dtype.tz}]"
    return str(dtype)


def schema_fingerprint(df: pd.DataFrame) -> str:
    """A short, deterministic fingerprint of `df`'s column names/order and
    dtypes (NOT its data). Used by the raw snapshot store and dataset
    manifest to detect schema drift/corruption between when a Parquet file
    was written and when it is later read back -- e.g. a pyarrow/pandas
    version change silently reinterpreting a column's dtype, or a
    truncated/corrupted file that happens to still parse but with the
    wrong shape."""
    signature = ",".join(f"{name}:{_fingerprint_dtype_str(df[name])}" for name in df.columns)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes, shared by every storage layer
    (raw snapshots, canonical partitions) that needs a content checksum
    without loading the whole file into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RAW_HISTORICAL_COLUMNS",
    "coerce_historical_dtypes",
    "schema_fingerprint",
    "sha256_file",
    "spread_points_to_price",
    "validate_historical_schema",
]
