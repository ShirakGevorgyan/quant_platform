from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis


class TestBuildNextReturnSpecification:
    def test_family_is_next_return(self) -> None:
        spec = build_next_return_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert spec.label_family is LabelFamily.NEXT_RETURN
        assert spec.parameters["price_basis"] == "close_to_close"
        assert spec.parameters["horizon_bars"] == 5

    def test_price_basis_has_no_hidden_default(self) -> None:
        with pytest.raises(TypeError):
            build_next_return_specification(horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")  # type: ignore[call-arg]

    @pytest.mark.parametrize("basis", list(PriceBasis))
    def test_every_price_basis_produces_a_distinct_specification(self, basis: PriceBasis) -> None:
        spec = build_next_return_specification(price_basis=basis, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        assert spec.parameters["price_basis"] == basis.value


class TestGenerateNextReturnLabels:
    def test_matches_pricing_helper_directly(self, ohlcv_source_data: pd.DataFrame) -> None:
        from quant_platform.labels.pricing import compute_forward_return

        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        values = generate_next_return_labels(ohlcv_source_data, spec)
        expected = compute_forward_return(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE, 5)
        pd.testing.assert_series_equal(values, expected, check_names=False)

    def test_usable_through_label_builder(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_next_return_specification(price_basis=PriceBasis.OPEN_TO_CLOSE, horizon_bars=10, created_from_dataset="ds1", created_from_manifest="m1")
        definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        assert bundle.row_count == len(ohlcv_source_data)
        assert bundle.valid_count == len(ohlcv_source_data) - 10

    def test_different_horizons_produce_different_specifications(self) -> None:
        a = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        b = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=10, created_from_dataset="ds1", created_from_manifest="m1")
        assert a.label_specification_id != b.label_specification_id
