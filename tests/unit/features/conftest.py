"""Shared fixtures for the Milestone 3 feature engineering test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.historical import PIPELINE_VERSION
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS
from quant_platform.historical.update_pipeline import apply_incremental_update


def make_synthetic_ohlcv(n: int, *, start: str = "2024-01-01", freq_minutes: int = 1, seed: int = 0) -> pd.DataFrame:
    """A deterministic synthetic RAW_HISTORICAL_COLUMNS OHLCV series --
    monotonic, tz-aware UTC, no duplicates, geometric-random-walk prices."""
    rng = np.random.default_rng(seed)
    open_time = pd.date_range(start, periods=n, freq=f"{freq_minutes}min", tz="UTC")
    returns = rng.normal(0, 0.0005, size=n)
    close = 2000.0 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    if n > 0:
        open_[0] = 2000.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, size=n)))
    tick_volume = rng.integers(10, 1000, size=n)
    real_volume = np.zeros(n, dtype=np.int64)
    spread = rng.integers(1, 30, size=n)
    return pd.DataFrame(
        {
            "open_time": open_time, "open": open_, "high": high, "low": low, "close": close,
            "tick_volume": tick_volume, "real_volume": real_volume, "spread": spread,
        }
    )[list(RAW_HISTORICAL_COLUMNS)]


@pytest.fixture
def synthetic_m1_df() -> pd.DataFrame:
    return make_synthetic_ohlcv(2000, seed=1)


@pytest.fixture
def synthetic_h1_df() -> pd.DataFrame:
    return make_synthetic_ohlcv(100, freq_minutes=60, seed=2)


def seed_canonical_dataset(
    root, df: pd.DataFrame, *, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1
) -> None:
    canonical_store = CanonicalStore(root)
    manifest_store = ManifestStore(root)
    apply_incremental_update(
        canonical_store, manifest_store, df, symbol=symbol, timeframe=timeframe, source_name="synthetic",
        broker="test", pipeline_version=PIPELINE_VERSION, parent_snapshot_ids=(),
        requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + timeframe.duration,
    )


@pytest.fixture
def seeded_loader(tmp_path, synthetic_m1_df):
    seed_canonical_dataset(tmp_path, synthetic_m1_df)
    canonical_store = CanonicalStore(tmp_path)
    manifest_store = ManifestStore(tmp_path)
    return DatasetLoader(canonical_store, manifest_store)


__all__ = ["make_synthetic_ohlcv", "seed_canonical_dataset"]
