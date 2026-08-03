from __future__ import annotations

from dataclasses import replace

import numpy as np

from quant_platform.label_validation.coverage import LabelCoverage, compute_label_coverage
from quant_platform.labels.builder import LabelBundle


class TestComputeLabelCoverage:
    def test_basic_fields(self, next_return_bundle: LabelBundle) -> None:
        coverage = compute_label_coverage(next_return_bundle)
        assert coverage.row_count == next_return_bundle.row_count
        assert coverage.valid_count == next_return_bundle.valid_count
        assert coverage.missing_count == coverage.row_count - coverage.valid_count
        assert abs(coverage.coverage_fraction - coverage.valid_count / coverage.row_count) < 1e-9

    def test_trailing_only_nan_has_no_interior_missing(self, next_return_bundle: LabelBundle) -> None:
        # next_return_bundle's own trailing-NaN tail (from horizon_bars=5) is
        # the ONLY legitimate missing-data shape a concrete family produces.
        coverage = compute_label_coverage(next_return_bundle)
        assert coverage.interior_missing_count == 0
        assert coverage.trailing_unresolved_count == 5

    def test_interior_hole_detected(self, next_return_bundle: LabelBundle) -> None:
        tampered = next_return_bundle.values.copy()
        tampered.iloc[len(tampered) // 2] = np.nan  # a NaN surrounded by valid rows
        tampered_bundle = replace(next_return_bundle, values=tampered)
        coverage = compute_label_coverage(tampered_bundle)
        assert coverage.interior_missing_count >= 1
        assert any("interior" in e.finding for e in coverage.evidence)

    def test_low_coverage_flagged(self, next_return_bundle: LabelBundle) -> None:
        mostly_nan = next_return_bundle.values.copy()
        mostly_nan.iloc[: int(len(mostly_nan) * 0.9)] = np.nan
        sparse_bundle = replace(next_return_bundle, values=mostly_nan)
        coverage = compute_label_coverage(sparse_bundle)
        assert coverage.coverage_fraction < 0.5
        assert any("coverage" in e.finding.lower() for e in coverage.evidence)

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        coverage = compute_label_coverage(next_return_bundle)
        restored = LabelCoverage.from_json_dict(coverage.to_json_dict())
        assert restored == coverage
