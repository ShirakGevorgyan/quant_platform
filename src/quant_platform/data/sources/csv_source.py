"""Local-filesystem CSV data source."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import DataSourceError
from quant_platform.core.time_utils import ensure_utc
from quant_platform.core.types import Timeframe
from quant_platform.data.interfaces import DataSource


class CsvDataSource(DataSource):
    """Reads OHLCV bars from one CSV file per symbol/timeframe.

    Files are located as `{root_dir}/{symbol}_{timeframe}.csv` by default;
    pass a different `filename_template` to match an existing directory
    layout. `{symbol}` and `{timeframe}` are the only substitution fields.
    """

    def __init__(self, root_dir: Path | str, filename_template: str = "{symbol}_{timeframe}.csv") -> None:
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
                f"CSV data file not found: {path}",
                context={"symbol": symbol, "timeframe": timeframe.value},
            )

        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            raise DataSourceError(
                f"Failed to read CSV file {path}: {exc}",
                context={"symbol": symbol, "timeframe": timeframe.value},
            ) from exc

        df = self._finalize(raw, source_description=f"CSV file {path}")

        start_utc, end_utc = ensure_utc(start), ensure_utc(end)
        mask = (df["open_time"] >= start_utc) & (df["open_time"] < end_utc)
        return df.loc[mask].reset_index(drop=True)

    def _resolve_path(self, symbol: str, timeframe: Timeframe) -> Path:
        filename = self._filename_template.format(symbol=symbol, timeframe=timeframe.value)
        # `Path(filename).name` strips any directory components a malformed
        # template or substitution could otherwise introduce -- belt and
        # braces on top of `sanitize_identifier`.
        return self._root_dir / Path(filename).name
