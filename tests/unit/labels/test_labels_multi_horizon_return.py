from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.multi_horizon_return import (
    MULTI_HORIZON_RETURN_MINIMUM_HORIZONS,
    build_multi_horizon_return_specifications,
    generate_multi_horizon_return_labels,
)
from quant_platform.labels.pricing import PriceBasis


class TestBuildMultiHorizonReturnSpecifications:
    def test_minimum_required_horizons_constant(self) -> None:
        assert MULTI_HORIZON_RETURN_MINIMUM_HORIZONS == (1, 5, 10, 20, 50, 100)

    def test_one_specification_per_horizon(self) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=MULTI_HORIZON_RETURN_MINIMUM_HORIZONS, price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert len(specs) == 6
        assert all(s.label_family is LabelFamily.MULTI_HORIZON_RETURN for s in specs)
        assert [s.parameters["horizon_bars"] for s in specs] == list(MULTI_HORIZON_RETURN_MINIMUM_HORIZONS)

    def test_every_horizon_gets_a_distinct_identity(self) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=MULTI_HORIZON_RETURN_MINIMUM_HORIZONS, price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert len({s.label_specification_id for s in specs}) == 6

    def test_arbitrary_non_standard_horizons_are_supported(self) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=(3, 7, 250), price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert [s.parameters["horizon_bars"] for s in specs] == [3, 7, 250]

    def test_empty_horizons_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_multi_horizon_return_specifications(horizons=(), price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1")


class TestGenerateMultiHorizonReturnLabels:
    def test_each_horizon_builds_independently_through_label_builder(self, ohlcv_source_data: pd.DataFrame) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=(1, 5, 20), price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        bundles = []
        for spec in specs:
            definition = LabelDefinition(specification=spec, generate=generate_multi_horizon_return_labels)
            bundles.append(LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1"))
        valid_counts = [b.valid_count for b in bundles]
        assert valid_counts == [len(ohlcv_source_data) - h for h in (1, 5, 20)]

    def test_different_horizons_produce_different_content_ids(self, ohlcv_source_data: pd.DataFrame) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=(5, 10), price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        bundles = [
            LabelBuilder().build(LabelDefinition(specification=s, generate=generate_multi_horizon_return_labels), ohlcv_source_data, source_content_id="src1")
            for s in specs
        ]
        assert bundles[0].identity.content_id != bundles[1].identity.content_id
