from __future__ import annotations

import pytest

from quant_platform.core.exceptions import FeatureError
from quant_platform.core.types import Timeframe
from quant_platform.features.lineage import FeatureLineage, build_lineage, render_lineage_report
from quant_platform.features.models import FeatureCategory, FeatureSpec


def _spec(**overrides) -> FeatureSpec:
    base = {
        "name": "return_simple_1", "version": "1", "description": "simple return", "category": FeatureCategory.PRICE,
        "required_inputs": ("close",), "source_symbols": (), "source_timeframe": Timeframe.M1,
        "output_dtype": "float64", "lookback_bars": 1, "warmup_bars": 1,
    }
    base.update(overrides)
    return FeatureSpec(**base)


class TestBuildLineage:
    def test_captures_spec_fields(self) -> None:
        spec = _spec(deterministic_params={"window": 1}, feature_dependencies=())
        lineage = build_lineage(spec, source_dataset_manifest_id="hist123", transformation="simple return")
        assert lineage.feature_name == "return_simple_1"
        assert lineage.source_dataset_manifest_id == "hist123"
        assert lineage.parameters == {"window": 1}
        assert lineage.spec_fingerprint == spec.fingerprint()

    def test_none_manifest_id_preserved(self) -> None:
        lineage = build_lineage(_spec(), source_dataset_manifest_id=None, transformation="t")
        assert lineage.source_dataset_manifest_id is None


class TestJsonRoundTrip:
    def test_round_trip_preserves_all_fields(self) -> None:
        spec = _spec(deterministic_params={"window": 5}, feature_dependencies=("dep_a",))
        lineage = build_lineage(spec, source_dataset_manifest_id="hist1", transformation="t")
        restored = FeatureLineage.from_json_dict(lineage.to_json_dict())
        assert restored == lineage

    def test_round_trip_with_none_manifest_id(self) -> None:
        lineage = build_lineage(_spec(), source_dataset_manifest_id=None, transformation="t")
        restored = FeatureLineage.from_json_dict(lineage.to_json_dict())
        assert restored.source_dataset_manifest_id is None

    def test_from_json_dict_rejects_non_list_source_symbols(self) -> None:
        raw = build_lineage(_spec(), source_dataset_manifest_id=None, transformation="t").to_json_dict()
        raw["source_symbols"] = "not-a-list"
        with pytest.raises(FeatureError):
            FeatureLineage.from_json_dict(raw)

    def test_from_json_dict_rejects_non_dict_parameters(self) -> None:
        raw = build_lineage(_spec(), source_dataset_manifest_id=None, transformation="t").to_json_dict()
        raw["parameters"] = "not-a-dict"
        with pytest.raises(FeatureError):
            FeatureLineage.from_json_dict(raw)

    def test_from_json_dict_defaults_missing_optional_fields(self) -> None:
        raw = {
            "feature_name": "f", "feature_version": "1", "category": "price", "source_dataset_manifest_id": None,
            "source_symbols": [], "source_timeframe": "M1", "required_inputs": [], "transformation": "t",
        }
        restored = FeatureLineage.from_json_dict(raw)
        assert restored.lookback_bars == 0
        assert restored.parameters == {}
        assert restored.feature_dependencies == ()


class TestRenderLineageReport:
    def test_report_contains_feature_name_and_category(self) -> None:
        lineage = build_lineage(_spec(), source_dataset_manifest_id="hist1", transformation="simple return")
        report = render_lineage_report([lineage])
        assert "return_simple_1" in report
        assert "price" in report
        assert "Feature lineage report (1 feature(s))" in report

    def test_report_includes_parameters_when_present(self) -> None:
        spec = _spec(name="derived", deterministic_params={"k": 1})
        lineage = build_lineage(spec, source_dataset_manifest_id=None, transformation="t")
        report = render_lineage_report([lineage])
        assert "parameters: {'k': 1}" in report

    def test_report_includes_availability_delay_when_nonzero(self) -> None:
        import pandas as pd

        spec = _spec(availability_delay=pd.Timedelta(minutes=5))
        lineage = build_lineage(spec, source_dataset_manifest_id=None, transformation="t")
        report = render_lineage_report([lineage])
        assert "availability_delay=300.0s" in report

    def test_report_sorted_by_feature_name(self) -> None:
        lineage_b = build_lineage(_spec(name="b_feature"), source_dataset_manifest_id=None, transformation="t")
        lineage_a = build_lineage(_spec(name="a_feature"), source_dataset_manifest_id=None, transformation="t")
        report = render_lineage_report([lineage_b, lineage_a])
        assert report.index("a_feature") < report.index("b_feature")

    def test_report_with_feature_dependencies(self) -> None:
        # A FeatureLineage can be constructed directly (independent of a live
        # registry) to exercise the "depends on features" rendering branch.
        lineage = FeatureLineage(
            feature_name="composed", feature_version="1", category="price", source_dataset_manifest_id=None,
            source_symbols=(), source_timeframe="M1", required_inputs=(), transformation="t",
            feature_dependencies=("return_simple_1",),
        )
        report = render_lineage_report([lineage])
        assert "depends on features: ['return_simple_1']" in report
