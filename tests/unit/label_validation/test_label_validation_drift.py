from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.label_validation.drift import LabelDrift, compute_label_drift
from quant_platform.labels.builder import LabelBundle


class TestComputeLabelDrift:
    def test_identical_bundles_have_zero_drift(self, next_return_bundle: LabelBundle) -> None:
        drift = compute_label_drift(next_return_bundle, next_return_bundle)
        assert drift.psi is not None
        assert drift.psi == pytest.approx(0.0, abs=1e-9)
        assert drift.kl_divergence == pytest.approx(0.0, abs=1e-9)
        assert drift.js_divergence == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_has_positive_psi(self, next_return_bundle: LabelBundle) -> None:
        shifted_values = next_return_bundle.values * 5.0 + 1.0
        shifted_bundle = replace(next_return_bundle, values=shifted_values)
        drift = compute_label_drift(next_return_bundle, shifted_bundle)
        assert drift.psi is not None
        assert drift.psi > 0.0

    def test_different_families_raise(self, next_return_bundle: LabelBundle, direction_bundle: LabelBundle) -> None:
        with pytest.raises(LabelValidationRequestError):
            compute_label_drift(next_return_bundle, direction_bundle)

    def test_class_drift_sums_reasonably(self, next_return_bundle: LabelBundle) -> None:
        shifted_values = next_return_bundle.values * 3.0
        shifted_bundle = replace(next_return_bundle, values=shifted_values)
        drift = compute_label_drift(next_return_bundle, shifted_bundle)
        assert len(drift.class_drift) > 0

    def test_rolling_drift_is_a_sequence(self, next_return_bundle: LabelBundle) -> None:
        shifted_values = next_return_bundle.values * 2.0
        shifted_bundle = replace(next_return_bundle, values=shifted_values)
        drift = compute_label_drift(next_return_bundle, shifted_bundle, rolling_window_bars=20)
        assert isinstance(drift.rolling_drift, tuple)

    def test_significant_drift_produces_warning_evidence(self, next_return_bundle: LabelBundle) -> None:
        extreme_values = next_return_bundle.values * 1000.0 - 500.0
        extreme_bundle = replace(next_return_bundle, values=extreme_values)
        drift = compute_label_drift(next_return_bundle, extreme_bundle)
        assert any(e.severity.value == "WARNING" for e in drift.evidence)

    def test_empty_bundle_has_none_metrics(self, next_return_bundle: LabelBundle) -> None:
        empty_bundle = replace(next_return_bundle, values=np.nan * next_return_bundle.values)
        drift = compute_label_drift(next_return_bundle, empty_bundle)
        assert drift.psi is None

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        shifted_values = next_return_bundle.values * 2.0
        shifted_bundle = replace(next_return_bundle, values=shifted_values)
        drift = compute_label_drift(next_return_bundle, shifted_bundle)
        restored = LabelDrift.from_json_dict(drift.to_json_dict())
        assert restored == drift
