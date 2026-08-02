"""Shared fixtures for the Milestone 11 Phase 1 qualification test suite.

Mirrors `tests/unit/features/conftest.py`'s own synthetic-OHLCV/
seeded-loader pattern (duplicated here rather than imported across test
directories, matching this repo's existing per-directory-conftest
convention -- `tests/unit/robustness/` has no conftest of its own
either) and `tests/unit/features/test_dataset_builder.py`'s `_trend_
registry`/`_builder`/`_request` helpers, so every qualification test
exercises the REAL, unmodified `ResearchDatasetBuilder` -- never a
second builder or a hand-rolled manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.features.models import FeatureCategory, FeatureSpec
from quant_platform.features.registry import FeatureRegistry
from quant_platform.historical import PIPELINE_VERSION
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS
from quant_platform.historical.update_pipeline import apply_incremental_update


def make_synthetic_ohlcv(n: int, *, start: str = "2024-01-01", freq_minutes: int = 1, seed: int = 0) -> pd.DataFrame:
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


def seed_canonical_dataset(root, df: pd.DataFrame, *, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1) -> None:
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


def trend_registry() -> FeatureRegistry:
    """A tiny registry with one deterministic, linearly-trending feature
    -- deliberately monotonic so `stability`'s PSI-based drift check has
    something genuine (and severe) to detect between train and eval
    splits, proving that dimension isn't a stub."""
    registry = FeatureRegistry()
    spec = FeatureSpec(
        name="trend", version="1", description="row index as a float", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(
        FeatureDefinition(spec=spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64")))
    )
    return registry


def two_feature_registry() -> FeatureRegistry:
    """`trend_registry()` plus a genuinely constant feature -- used by
    Part 2's deep-diagnostics tests to exercise zero-variance/mutable-
    alias detection, which need a feature with zero standard deviation
    to have anything real to find."""
    registry = trend_registry()
    spec = FeatureSpec(
        name="const", version="1", description="constant feature", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(FeatureDefinition(spec=spec, compute=lambda ctx: pd.Series(np.full(len(ctx.base_df), 7.0))))
    return registry


def build_request(**overrides) -> ResearchDatasetBuildRequest:
    base: dict[str, object] = {
        "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "start": pd.Timestamp("2024-01-01", tz="UTC"),
        "end": pd.Timestamp("2024-01-01", tz="UTC") + Timeframe.M1.duration * 2000,
        "feature_names": ("trend",), "label_definition": LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        "split_strategy": "chronological",
        "split_params": {"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    }
    base.update(overrides)
    return ResearchDatasetBuildRequest(**base)


@pytest.fixture
def research_store(tmp_path) -> ResearchDatasetStore:
    return ResearchDatasetStore(tmp_path / "research")


@pytest.fixture
def qualified_manifest(tmp_path, seeded_loader, research_store) -> ResearchDatasetManifest:
    """A REAL `ResearchDatasetManifest` produced by the real, unmodified
    `ResearchDatasetBuilder` -- the fixture every qualification test
    builds on top of."""
    registry = trend_registry()
    builder = ResearchDatasetBuilder(
        historical_loader=seeded_loader, registry=registry, research_store=research_store,
        manifest_store=ResearchManifestStore(tmp_path / "research"),
    )
    return builder.build(build_request())


@pytest.fixture
def two_feature_manifest(tmp_path, seeded_loader, research_store) -> ResearchDatasetManifest:
    registry = two_feature_registry()
    builder = ResearchDatasetBuilder(
        historical_loader=seeded_loader, registry=registry, research_store=research_store,
        manifest_store=ResearchManifestStore(tmp_path / "research"),
    )
    return builder.build(build_request(feature_names=("trend", "const")))


__all__ = ["build_request", "make_synthetic_ohlcv", "seed_canonical_dataset", "trend_registry", "two_feature_registry"]
