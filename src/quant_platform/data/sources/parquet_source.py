"""Local-filesystem Parquet data source.

Parquet's columnar layout and row-group statistics let the underlying
pyarrow engine skip whole row groups that fall outside the requested date
range without reading them into memory, which is the difference between a
tractable and an intractable query once a single symbol's history reaches
the "hundreds of GB" scale this platform is designed for.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import DataSourceError
from quant_platform.core.time_utils import ensure_utc
from quant_platform.core.types import Timeframe
from quant_platform.data.interfaces import DataSource


class ParquetDataSource(DataSource):
    """Reads OHLCV bars from one Parquet file per symbol/timeframe, using
    predicate pushdown on `open_time` so only relevant row groups are read.
    """

    def __init__(
        self, root_dir: Path | str, filename_template: str = "{symbol}_{timeframe}.parquet"
    ) -> None:
        self._root_dir = Path(root_dir)
        self._filename_template = filename_template

    def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        safe_symbol = self.sanitize_identifier(symbol, field_name="symbol")
        path = self._resolve_path(safe_symbol, timeframe)

        if not path.is_file():
            raise DataSourceError(
                f"Parquet data file not found: {path}",
                context={"symbol": symbol, "timeframe": timeframe.value},
            )

        start_utc, end_utc = ensure_utc(start), ensure_utc(end)

        try:
            # `engine` deliberately omitted: pandas auto-detects and uses
            # pyarrow (our hard dependency, see pyproject.toml) automatically,
            # and pandas-stubs' explicit engine="pyarrow" overload requires a
            # mandatory `to_pandas_kwargs` we have no use for.
            raw = pd.read_parquet(
                path,
                filters=[("open_time", ">=", start_utc), ("open_time", "<", end_utc)],
            )
        except Exception as exc:
            raise DataSourceError(
                f"Failed to read Parquet file {path}: {exc}",
                context={"symbol": symbol, "timeframe": timeframe.value},
            ) from exc

        return self._finalize(raw, source_description=f"Parquet file {path}")

    def _resolve_path(self, symbol: str, timeframe: Timeframe) -> Path:
        filename = self._filename_template.format(symbol=symbol, timeframe=timeframe.value)
        return self._root_dir / Path(filename).name
