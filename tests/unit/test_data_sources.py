"""Tests for CsvDataSource and ParquetDataSource, including the path-
injection security boundary shared by both via `DataSource.sanitize_identifier`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.core.exceptions import DataSourceError, SchemaError
from quant_platform.core.types import Timeframe
from quant_platform.data.sources import CsvDataSource, ParquetDataSource
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv

UTC = timezone.utc


def _synthetic_frame(periods: int = 100):
    return generate_ohlcv(
        SyntheticDataConfig(
            start=datetime(2024, 1, 1, tzinfo=UTC), periods=periods, timeframe=Timeframe.M15, seed=5
        )
    )


class TestCsvDataSource:
    def test_loads_and_filters_by_date_range(self, tmp_path: Path) -> None:
        df = _synthetic_frame(periods=100)
        df.to_csv(tmp_path / "EURUSD_M15.csv", index=False)

        source = CsvDataSource(root_dir=tmp_path)
        start = df["open_time"].iloc[10]
        end = df["open_time"].iloc[40]
        result = source.load("EURUSD", Timeframe.M15, start.to_pydatetime(), end.to_pydatetime())

        assert len(result) == 30  # [start, end) -> indices 10..39
        assert result["open_time"].iloc[0] == start
        assert result["open_time"].min() >= start
        assert result["open_time"].max() < end

    def test_missing_file_raises_data_source_error(self, tmp_path: Path) -> None:
        source = CsvDataSource(root_dir=tmp_path)
        with pytest.raises(DataSourceError, match="not found"):
            source.load("NOPE", Timeframe.M15, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))

    def test_missing_columns_raise_schema_error(self, tmp_path: Path) -> None:
        df = _synthetic_frame(periods=10).drop(columns=["volume"])
        df.to_csv(tmp_path / "BAD_M15.csv", index=False)
        source = CsvDataSource(root_dir=tmp_path)
        with pytest.raises(SchemaError):
            source.load("BAD", Timeframe.M15, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))

    def test_deduplicates_and_sorts(self, tmp_path: Path) -> None:
        df = _synthetic_frame(periods=20)
        shuffled = pd.concat([df, df.iloc[[3]]]).sample(frac=1.0, random_state=0)
        shuffled.to_csv(tmp_path / "DUP_M15.csv", index=False)

        source = CsvDataSource(root_dir=tmp_path)
        result = source.load(
            "DUP", Timeframe.M15, datetime(2020, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC)
        )
        assert len(result) == 20
        assert result["open_time"].is_monotonic_increasing
        assert not result["open_time"].duplicated().any()

    @pytest.mark.parametrize(
        "malicious_symbol",
        ["../secrets", "..\\secrets", "/etc/passwd", "a/b", "a\\b", ""],
    )
    def test_rejects_path_traversal_in_symbol(self, tmp_path: Path, malicious_symbol: str) -> None:
        source = CsvDataSource(root_dir=tmp_path)
        with pytest.raises(DataSourceError, match="Invalid symbol"):
            source.load(
                malicious_symbol, Timeframe.M15, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
            )


class TestParquetDataSource:
    def test_loads_and_filters_by_date_range(self, tmp_path: Path) -> None:
        df = _synthetic_frame(periods=100)
        df.to_parquet(tmp_path / "EURUSD_M15.parquet", engine="pyarrow", index=False)

        source = ParquetDataSource(root_dir=tmp_path)
        start = df["open_time"].iloc[10]
        end = df["open_time"].iloc[40]
        result = source.load("EURUSD", Timeframe.M15, start.to_pydatetime(), end.to_pydatetime())

        assert len(result) == 30
        assert result["open_time"].min() >= start
        assert result["open_time"].max() < end

    def test_missing_file_raises_data_source_error(self, tmp_path: Path) -> None:
        source = ParquetDataSource(root_dir=tmp_path)
        with pytest.raises(DataSourceError, match="not found"):
            source.load("NOPE", Timeframe.M15, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))

    def test_rejects_path_traversal_in_symbol(self, tmp_path: Path) -> None:
        source = ParquetDataSource(root_dir=tmp_path)
        with pytest.raises(DataSourceError, match="Invalid symbol"):
            source.load(
                "../secrets", Timeframe.M15, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
            )
