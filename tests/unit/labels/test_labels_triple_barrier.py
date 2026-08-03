from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
    generate_triple_barrier_labels,
)
from quant_platform.labels.volatility import REALIZED_PARKINSON_ESTIMATOR_NAME, REALIZED_STDDEV_ESTIMATOR_NAME


class TestBuildTripleBarrierSpecification:
    def test_family_is_triple_barrier(self) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert spec.label_family is LabelFamily.TRIPLE_BARRIER
        assert spec.parameters["volatility_estimator_reference"] == REALIZED_STDDEV_ESTIMATOR_NAME

    def test_non_positive_multipliers_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_triple_barrier_specification(
                profit_multiplier=0.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
                volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_non_positive_max_holding_bars_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_triple_barrier_specification(
                profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=0, volatility_window_bars=20,
                volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_unknown_estimator_reference_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_triple_barrier_specification(
                profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
                volatility_estimator_reference="not_a_real_estimator", created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_different_estimator_reference_is_a_different_specification(self) -> None:
        a = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        b = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_PARKINSON_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert a.label_specification_id != b.label_specification_id


class TestGenerateTripleBarrierLabels:
    def test_only_valid_outcomes_appear(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_triple_barrier_labels(ohlcv_source_data, spec)
        assert set(values.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_produces_genuine_variety_of_outcomes(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=1.0, loss_multiplier=1.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_triple_barrier_labels(ohlcv_source_data, spec)
        assert len(values.dropna().unique()) >= 2

    def test_missing_high_low_rejected(self, source_data: pd.DataFrame) -> None:
        stripped = source_data.drop(columns=["high", "low"])
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=5, volatility_window_bars=5,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        with pytest.raises(LabelRequestError):
            generate_triple_barrier_labels(stripped, spec)

    def test_warmup_rows_without_trailing_volatility_are_nan(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_triple_barrier_labels(ohlcv_source_data, spec)
        assert values.iloc[:19].isna().all()

    def test_usable_through_label_builder(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definition = LabelDefinition(specification=spec, generate=generate_triple_barrier_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        assert bundle.row_count == len(ohlcv_source_data)
