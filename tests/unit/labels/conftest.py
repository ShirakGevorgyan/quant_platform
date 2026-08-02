"""Shared fixtures for `tests/unit/labels/`. Every generator function
defined here is deliberately a trivial, synthetic STRUCTURAL fixture
(a row-position marker with a trailing-NaN tail) -- never a recognizable
financial return/direction/barrier/volatility formula, so nothing in
this test suite could be mistaken for one of the 6 named label
families' real implementation (which belongs to a later phase)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

SOURCE_ROW_COUNT = 20
TRAILING_UNRESOLVED = 3
"""How many trailing rows `marker_generator` leaves NaN, simulating "not
enough future data yet" near the end of a source frame -- the only
legitimate NaN shape this infrastructure-only phase reasons about."""


def source_dataframe(*, row_count: int = SOURCE_ROW_COUNT) -> pd.DataFrame:
    open_time = pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC")
    close = 100.0 + np.arange(row_count, dtype="float64")
    return pd.DataFrame({"open_time": open_time, "close": close, "high": close + 0.5, "low": close - 0.5})


def marker_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    """A synthetic, non-financial "structural marker": row position as a
    float, with a trailing run of NaN -- the shape a real forward-looking
    label would have near the end of the frame, without computing
    anything resembling a return/direction/barrier/volatility value."""
    n = len(source_data)
    values = np.arange(n, dtype="float64")
    if TRAILING_UNRESOLVED > 0:
        values[max(0, n - TRAILING_UNRESOLVED) :] = np.nan
    return pd.Series(values, index=source_data.index)


def constant_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    return pd.Series(np.full(len(source_data), 1.0), index=source_data.index)


def wrong_length_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    return pd.Series(np.zeros(len(source_data) - 1), index=source_data.index[:-1])


def non_series_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> object:
    return list(range(len(source_data)))


def non_numeric_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    return pd.Series(["a"] * len(source_data), index=source_data.index)


def aliasing_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    """Deliberately returns the SAME underlying memory as a source
    column -- the exact mistake `builder._assert_no_mutable_alias` must
    catch."""
    return source_data["close"]


def non_trailing_nan_generator(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    """A NaN in the middle followed by valid data -- the malformed shape
    `diagnostics._trailing_nan_tail_is_well_formed` flags."""
    n = len(source_data)
    values = np.arange(n, dtype="float64")
    values[n // 2] = np.nan
    return pd.Series(values, index=source_data.index)


@pytest.fixture
def source_data() -> pd.DataFrame:
    return source_dataframe()


@pytest.fixture
def source_content_id() -> str:
    return "test-source-content-id-0001"


@pytest.fixture
def specification() -> LabelSpecification:
    return build_label_specification(
        label_family=LabelFamily.NEXT_RETURN, generation_version="v1", price_basis="close",
        prediction_horizon="5 bars", availability_rule="available at reference_time + 5 bars", reference_price="close at event_time",
        event_time_rule="bar close time", generation_rule="structural test fixture -- not a real generation rule",
        created_from_dataset="dataset-0001", created_from_manifest="manifest-0001", parameters={"horizon_bars": 5},
    )


@pytest.fixture
def other_family_specification() -> LabelSpecification:
    return build_label_specification(
        label_family=LabelFamily.DIRECTION, generation_version="v1", price_basis="close",
        prediction_horizon="10 bars", availability_rule="available at reference_time + 10 bars", reference_price="close at event_time",
        event_time_rule="bar close time", generation_rule="structural test fixture -- not a real generation rule",
        created_from_dataset="dataset-0001", created_from_manifest="manifest-0001", parameters={"threshold": 0.0},
    )


@pytest.fixture
def definition(specification: LabelSpecification) -> LabelDefinition:
    return LabelDefinition(specification=specification, generate=marker_generator)


@pytest.fixture
def bundle(definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str) -> LabelBundle:
    return LabelBuilder().build(definition, source_data, source_content_id=source_content_id)


@pytest.fixture
def marker_generator_fn():
    """Exposes `marker_generator` as an injected fixture rather than
    requiring `from conftest import marker_generator` -- a bare import of
    this directory's `conftest.py` is ambiguous once more than one test
    directory's own (identically-named) `conftest.py` has been imported
    into the SAME `pytest` session's `sys.modules` cache (see
    `tests/unit/qualification/conftest.py`'s own identical fixtures for
    the fully-documented rationale)."""
    return marker_generator


@pytest.fixture
def constant_generator_fn():
    return constant_generator


@pytest.fixture
def wrong_length_generator_fn():
    return wrong_length_generator


@pytest.fixture
def non_series_generator_fn():
    return non_series_generator


@pytest.fixture
def non_numeric_generator_fn():
    return non_numeric_generator


@pytest.fixture
def aliasing_generator_fn():
    return aliasing_generator


@pytest.fixture
def non_trailing_nan_generator_fn():
    return non_trailing_nan_generator


@pytest.fixture
def manifest(specification: LabelSpecification) -> LabelManifest:
    return build_label_manifest(
        specification, generation_timestamp=format_utc_timestamp(utc_now()), feature_identity="feature-fingerprint-0001",
        qualification_identity="qualification-report-0001",
    )
