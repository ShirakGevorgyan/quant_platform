"""Shared fixtures for `tests/unit/label_validation/`. Builds real
`labels.builder.LabelBundle`s via Milestone 11 Phase 3 Part B's own
concrete label family generators (Next Return, Direction, Triple
Barrier) -- this package evaluates already-generated labels, so its own
tests need genuinely generated ones, never a synthetic stand-in."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.direction import build_direction_specification, generate_direction_labels
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.records import LabelRecord, materialize_label_records
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
    generate_triple_barrier_labels,
)
from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

OHLCV_ROW_COUNT = 300


def ohlcv_dataframe(*, row_count: int = OHLCV_ROW_COUNT, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    open_time = pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC")
    returns = rng.normal(0, 0.003, size=row_count)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=row_count)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=row_count)))
    return pd.DataFrame({"open_time": open_time, "open": open_, "high": high, "low": low, "close": close})


@pytest.fixture
def ohlcv_source_data() -> pd.DataFrame:
    return ohlcv_dataframe()


@pytest.fixture
def source_content_id() -> str:
    return "test-source-content-id-0001"


@pytest.fixture
def dataset_id() -> str:
    return "ds1"


@pytest.fixture
def timeframe() -> Timeframe:
    return Timeframe.M1


# -- Next Return (continuous family) ----------------------------------------------------------------


@pytest.fixture
def next_return_definition() -> LabelDefinition:
    spec = build_next_return_specification(
        price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1",
    )
    return LabelDefinition(specification=spec, generate=generate_next_return_labels)


@pytest.fixture
def next_return_bundle(next_return_definition: LabelDefinition, ohlcv_source_data: pd.DataFrame, source_content_id: str) -> LabelBundle:
    return LabelBuilder().build(next_return_definition, ohlcv_source_data, source_content_id=source_content_id)


@pytest.fixture
def next_return_manifest(next_return_definition: LabelDefinition) -> LabelManifest:
    return build_label_manifest(next_return_definition.specification, generation_timestamp=format_utc_timestamp(utc_now()))


@pytest.fixture
def next_return_records(next_return_bundle: LabelBundle, ohlcv_source_data: pd.DataFrame) -> tuple[LabelRecord, ...]:
    return materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)


# Generic aliases most single-bundle tests use.
@pytest.fixture
def bundle(next_return_bundle: LabelBundle) -> LabelBundle:
    return next_return_bundle


@pytest.fixture
def manifest(next_return_manifest: LabelManifest) -> LabelManifest:
    return next_return_manifest


@pytest.fixture
def definition(next_return_definition: LabelDefinition) -> LabelDefinition:
    return next_return_definition


@pytest.fixture
def records(next_return_records: tuple[LabelRecord, ...]) -> tuple[LabelRecord, ...]:
    return next_return_records


# -- Direction (discrete family) ---------------------------------------------------------------------


@pytest.fixture
def direction_definition() -> LabelDefinition:
    spec = build_direction_specification(
        price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
    )
    return LabelDefinition(specification=spec, generate=generate_direction_labels)


@pytest.fixture
def direction_bundle(direction_definition: LabelDefinition, ohlcv_source_data: pd.DataFrame, source_content_id: str) -> LabelBundle:
    return LabelBuilder().build(direction_definition, ohlcv_source_data, source_content_id=source_content_id)


@pytest.fixture
def direction_manifest(direction_definition: LabelDefinition) -> LabelManifest:
    return build_label_manifest(direction_definition.specification, generation_timestamp=format_utc_timestamp(utc_now()))


# -- Triple Barrier (discrete family, {-1, 0, 1}) -----------------------------------------------------


@pytest.fixture
def triple_barrier_definition() -> LabelDefinition:
    spec = build_triple_barrier_specification(
        profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
        volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
    )
    return LabelDefinition(specification=spec, generate=generate_triple_barrier_labels)


@pytest.fixture
def triple_barrier_bundle(triple_barrier_definition: LabelDefinition, ohlcv_source_data: pd.DataFrame, source_content_id: str) -> LabelBundle:
    return LabelBuilder().build(triple_barrier_definition, ohlcv_source_data, source_content_id=source_content_id)


@pytest.fixture
def triple_barrier_manifest(triple_barrier_definition: LabelDefinition) -> LabelManifest:
    return build_label_manifest(triple_barrier_definition.specification, generation_timestamp=format_utc_timestamp(utc_now()))


def _rebuild_bundle_with_values(bundle: LabelBundle, values: pd.Series) -> LabelBundle:
    """Constructs a bundle with genuinely DIFFERENT values while keeping
    `identity` self-consistent with them -- for tests that want to
    simulate "a differently-valued but otherwise honest bundle" (e.g. a
    degeneracy/balance/stability scenario) WITHOUT incidentally tripping
    `leakage.validate_leakage`'s identity self-consistency check, which
    is a real, intentional detector for tampered bundles, not a bug to
    route around. Mirrors `feature_discovery`'s own established
    `_rebuild_bundle_from_tampered_snapshot` test-helper pattern."""
    fresh_identity = compute_label_identity(bundle.specification.label_specification_id, values, source_content_id=bundle.identity.source_content_id)
    return replace(bundle, values=values, identity=fresh_identity, valid_count=int(values.notna().sum()))


@pytest.fixture
def rebuild_bundle_with_values_fn():
    """Exposes `_rebuild_bundle_with_values` as an injected fixture
    rather than requiring `from conftest import ...` -- avoids the
    `sys.modules["conftest"]` collision class of defect (see
    `tests/unit/qualification/conftest.py`'s own identical fixtures for
    the fully-documented rationale)."""
    return _rebuild_bundle_with_values
