from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.forward_volatility import (
    build_forward_volatility_specification,
    generate_forward_volatility_labels,
)
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.volatility import REALIZED_PARKINSON_ESTIMATOR_NAME, REALIZED_STDDEV_ESTIMATOR_NAME


class TestBuildForwardVolatilitySpecification:
    def test_family_is_forward_volatility(self) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert spec.label_family is LabelFamily.FORWARD_VOLATILITY

    def test_non_positive_horizon_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_forward_volatility_specification(
                horizon_bars=0, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_unknown_estimator_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_forward_volatility_specification(
                horizon_bars=10, volatility_estimator_reference="not_a_real_estimator", created_from_dataset="ds1", created_from_manifest="m1",
            )

    @pytest.mark.parametrize("estimator_name", [REALIZED_STDDEV_ESTIMATOR_NAME, REALIZED_PARKINSON_ESTIMATOR_NAME])
    def test_no_estimator_is_privileged_both_are_buildable(self, estimator_name: str) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=estimator_name, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert spec.parameters["volatility_estimator_reference"] == estimator_name


class TestGenerateForwardVolatilityLabels:
    def test_never_negative(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_forward_volatility_labels(ohlcv_source_data, spec)
        assert (values.dropna() >= 0).all()

    def test_trailing_tail_is_nan_never_reaches_beyond_horizon(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_forward_volatility_labels(ohlcv_source_data, spec)
        assert values.iloc[-10:].isna().all()

    def test_different_estimators_give_different_values(self, ohlcv_source_data: pd.DataFrame) -> None:
        stddev_spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        parkinson_spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_PARKINSON_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        stddev_values = generate_forward_volatility_labels(ohlcv_source_data, stddev_spec)
        parkinson_values = generate_forward_volatility_labels(ohlcv_source_data, parkinson_spec)
        assert not stddev_values.dropna().equals(parkinson_values.dropna())

    def test_usable_through_label_builder(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definition = LabelDefinition(specification=spec, generate=generate_forward_volatility_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        assert bundle.row_count == len(ohlcv_source_data)
