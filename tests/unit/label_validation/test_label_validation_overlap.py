from __future__ import annotations

from quant_platform.label_validation.overlap import OverlapReport, detect_overlap
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.multi_horizon_return import (
    build_multi_horizon_return_specifications,
    generate_multi_horizon_return_labels,
)
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis


class TestDetectOverlap:
    def test_no_findings_among_unrelated_bundles(self, next_return_bundle, direction_bundle) -> None:
        report = detect_overlap((next_return_bundle, direction_bundle))
        assert report.findings == ()

    def test_duplicate_target_detected_for_identical_content_id(self, next_return_bundle) -> None:
        report = detect_overlap((next_return_bundle, next_return_bundle))
        assert any(f.kind == "duplicate_target" for f in report.findings)

    def test_redundant_target_detected_for_value_identical_but_differently_identified_bundles(self, next_return_bundle, ohlcv_source_data) -> None:
        # Same math (compute_forward_return, close_to_close, horizon=5), but
        # under a DIFFERENT family -> different label_specification_id and
        # content_id, yet the raw VALUES are identical -- exactly the case
        # duplicate_target (identity-based) cannot catch but redundant_target
        # (value-based) can.
        mh_specs = build_multi_horizon_return_specifications(
            horizons=(5,), price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        mh_definition = LabelDefinition(specification=mh_specs[0], generate=generate_multi_horizon_return_labels)
        mh_bundle = LabelBuilder().build(mh_definition, ohlcv_source_data, source_content_id="test-source-content-id-0001")
        assert mh_bundle.identity.content_id != next_return_bundle.identity.content_id

        report = detect_overlap((next_return_bundle, mh_bundle))
        assert any(f.kind == "redundant_target" for f in report.findings)
        assert not any(f.kind == "duplicate_target" for f in report.findings)

    def test_horizon_overlap_requires_same_family(self, next_return_bundle, ohlcv_source_data) -> None:
        same_horizon_different_basis_spec = build_next_return_specification(
            price_basis=PriceBasis.OPEN_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definition = LabelDefinition(specification=same_horizon_different_basis_spec, generate=generate_next_return_labels)
        other_bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="test-source-content-id-0001")
        report = detect_overlap((next_return_bundle, other_bundle))
        assert any(f.kind == "horizon_overlap" for f in report.findings)

    def test_json_round_trip(self, next_return_bundle) -> None:
        report = detect_overlap((next_return_bundle, next_return_bundle))
        restored = OverlapReport.from_json_dict(report.to_json_dict())
        assert restored == report
