from __future__ import annotations

import pytest

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.label_validation.horizon import HorizonComparisonReport, compare_horizons
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.multi_horizon_return import (
    MULTI_HORIZON_RETURN_MINIMUM_HORIZONS,
    build_multi_horizon_return_specifications,
    generate_multi_horizon_return_labels,
)
from quant_platform.labels.pricing import PriceBasis


@pytest.fixture
def multi_horizon_bundles(ohlcv_source_data):
    specs = build_multi_horizon_return_specifications(
        horizons=MULTI_HORIZON_RETURN_MINIMUM_HORIZONS, price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
    )
    bundles = []
    for spec in specs:
        definition = LabelDefinition(specification=spec, generate=generate_multi_horizon_return_labels)
        bundles.append(LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1"))
    return tuple(bundles)


class TestCompareHorizons:
    def test_covers_all_six_minimum_horizons(self, multi_horizon_bundles) -> None:
        report = compare_horizons(multi_horizon_bundles)
        assert [h.horizon_bars for h in report.horizons] == list(MULTI_HORIZON_RETURN_MINIMUM_HORIZONS)

    def test_sorted_ascending_regardless_of_input_order(self, multi_horizon_bundles) -> None:
        shuffled = tuple(reversed(multi_horizon_bundles))
        report = compare_horizons(shuffled)
        horizons = [h.horizon_bars for h in report.horizons]
        assert horizons == sorted(horizons)

    def test_coverage_decreases_as_horizon_grows(self, multi_horizon_bundles) -> None:
        report = compare_horizons(multi_horizon_bundles)
        coverages = [h.coverage_fraction for h in report.horizons]
        assert coverages == sorted(coverages, reverse=True)

    def test_never_ranks_a_best_horizon(self, multi_horizon_bundles) -> None:
        report = compare_horizons(multi_horizon_bundles)
        # purely descriptive: no field claims to rank/score "best"
        assert not hasattr(report, "best_horizon")
        assert not hasattr(report, "recommended_horizon")

    def test_empty_bundles_rejected(self) -> None:
        with pytest.raises(LabelValidationRequestError):
            compare_horizons(())

    def test_mixed_families_rejected(self, multi_horizon_bundles, direction_bundle) -> None:
        with pytest.raises(LabelValidationRequestError):
            compare_horizons((*multi_horizon_bundles, direction_bundle))

    def test_json_round_trip(self, multi_horizon_bundles) -> None:
        report = compare_horizons(multi_horizon_bundles)
        restored = HorizonComparisonReport.from_json_dict(report.to_json_dict())
        assert restored == report
