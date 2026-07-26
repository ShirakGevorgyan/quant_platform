"""Data source abstraction.

Every concrete data source (CSV, Parquet, and eventually live broker feeds
or a market-data database) implements this single interface, so the
backtesting engine and validation tooling never depend on where bars
physically come from. New venues/formats are added by implementing this
interface, never by modifying the engine (Open/Closed Principle).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import DataSourceError, SchemaError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe

_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")


class DataSource(ABC):
    """Loads canonical-schema OHLCV data for a symbol/timeframe/date range.

    Implementations MUST return a DataFrame that:
      * contains all of `OHLCV_COLUMNS` (additional columns are permitted)
      * has `open_time` as a timezone-aware (UTC) monotonically increasing,
        duplicate-free column
      * is sorted ascending by `open_time`

    `load()` is expected to enforce these guarantees itself (via
    `_finalize`) rather than trust the raw underlying storage format --
    concrete sources should call `self._finalize(df)` as their last step.
    """

    @abstractmethod
    def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Load bars for `symbol`/`timeframe` within [start, end)."""
        raise NotImplementedError

    @staticmethod
    def sanitize_identifier(value: str, *, field_name: str = "symbol") -> str:
        """Validate that `value` (a symbol name or similar identifier used
        to build a filesystem path) contains only a safe character set.

        This is a security boundary, not a convenience helper: `symbol`
        values may ultimately originate from configuration files or
        upstream services outside this process's control, and are
        interpolated into file paths by concrete `DataSource`
        implementations. Rejecting anything containing path separators or
        traversal sequences here prevents path-injection regardless of
        what any individual source implementation does downstream.
        """
        if not value or not _SAFE_IDENTIFIER_PATTERN.match(value):
            raise DataSourceError(
                f"Invalid {field_name} {value!r}: only letters, digits, '_', '-', and '.' "
                "are permitted (path separators and traversal sequences are rejected)."
            )
        return value

    @staticmethod
    def _finalize(df: pd.DataFrame, *, source_description: str) -> pd.DataFrame:
        """Shared post-processing every concrete source must apply: schema
        check, UTC localization, dedup, and sort. Centralizing this here
        (rather than duplicating it in every source) keeps the hygiene
        guarantee consistent and DRY across current and future sources."""
        missing = set(OHLCV_COLUMNS) - set(df.columns)
        if missing:
            raise SchemaError(
                f"{source_description} is missing required OHLCV columns: {sorted(missing)}"
            )
        if df.empty:
            return df.loc[:, list(OHLCV_COLUMNS)].copy()

        df = df.copy()
        open_time = pd.to_datetime(df["open_time"], utc=True)
        df["open_time"] = open_time

        df = df.drop_duplicates(subset="open_time", keep="last")
        df = df.sort_values("open_time").reset_index(drop=True)

        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise DataSourceError(
                f"{source_description} contains non-numeric OHLCV values after coercion"
            )

        ordered_columns = list(OHLCV_COLUMNS) + [c for c in df.columns if c not in OHLCV_COLUMNS]
        return df[ordered_columns]
