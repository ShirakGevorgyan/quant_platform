"""Data loading, synthetic generation, and quality validation."""

from quant_platform.data.interfaces import DataSource
from quant_platform.data.sources import CsvDataSource, ParquetDataSource
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.data.validation import (
    DataQualityReport,
    Gap,
    Overlap,
    detect_gaps,
    detect_overlaps,
    validate_ohlcv,
)

__all__ = [
    "CsvDataSource",
    "DataQualityReport",
    "DataSource",
    "Gap",
    "Overlap",
    "ParquetDataSource",
    "SyntheticDataConfig",
    "detect_gaps",
    "detect_overlaps",
    "generate_ohlcv",
    "validate_ohlcv",
]
