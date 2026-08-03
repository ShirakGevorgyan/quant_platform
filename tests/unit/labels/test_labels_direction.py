from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.direction import (
    Direction,
    build_direction_specification,
    generate_direction_labels,
)
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.pricing import PriceBasis


class TestBuildDirectionSpecification:
    def test_family_is_direction(self) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert spec.label_family is LabelFamily.DIRECTION
        assert spec.parameters["neutral_threshold"] == 0.001

    def test_neutral_threshold_has_no_hidden_default(self) -> None:
        with pytest.raises(TypeError):
            build_direction_specification(  # type: ignore[call-arg]
                price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_direction_specification(
                price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=-0.1, created_from_dataset="ds1", created_from_manifest="m1",
            )

    def test_threshold_participates_in_identity(self) -> None:
        a = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.0, created_from_dataset="ds1", created_from_manifest="m1")
        b = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.01, created_from_dataset="ds1", created_from_manifest="m1")
        assert a.label_specification_id != b.label_specification_id


class TestGenerateDirectionLabels:
    def test_only_valid_encoded_values_appear(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_direction_labels(ohlcv_source_data, spec)
        allowed = {Direction.UP.value, Direction.DOWN.value, Direction.NEUTRAL.value}
        assert set(values.dropna().unique()).issubset(allowed)

    def test_zero_threshold_never_produces_neutral_for_nonzero_returns(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=1, neutral_threshold=0.0, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_direction_labels(ohlcv_source_data, spec)
        assert Direction.NEUTRAL.value not in values.dropna().unique()

    def test_large_threshold_produces_all_neutral(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=10.0, created_from_dataset="ds1", created_from_manifest="m1",
        )
        values = generate_direction_labels(ohlcv_source_data, spec)
        assert set(values.dropna().unique()) == {Direction.NEUTRAL.value}

    def test_usable_through_label_builder(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.002, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definition = LabelDefinition(specification=spec, generate=generate_direction_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        assert bundle.valid_count == len(ohlcv_source_data) - 5
