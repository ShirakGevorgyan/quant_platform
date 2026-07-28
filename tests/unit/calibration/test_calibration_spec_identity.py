"""Release audit Section 9: `CalibrationSpec.bin_support_minimum_samples`
must participate in `calibration_id` -- it materially changes persisted
`calibration_bin_support`/`bin_support_uncertainty` outputs for otherwise
-identical inputs (see `calibration.runner._confidence_and_uncertainty_for_row`),
so two specs differing only here must never collide on one calibration_id.
Also covers the field's own validation and round-trip."""

from __future__ import annotations

import pytest

from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
    compute_calibration_identity,
)
from quant_platform.core.exceptions import CalibrationValidationError
from quant_platform.ml.models import ObjectiveType
from quant_platform.optimization.inner_splits import InnerSplitConfig


def _make_spec(*, bin_support_minimum_samples: int = 20) -> CalibrationSpec:
    return CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION,
        positive_class_label=1.0, source_experiment_id="a" * 64, base_model_definition_identity="constant_test_model:1",
        dataset_content_id="b" * 64, split_plan_fingerprint="c" * 64,
        calibration_method_candidates=(CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=10, minimum_samples_per_class=2,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15, embargo_bars=1),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.NONE),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy",), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
        seed=42, determinism_policy=DeterminismPolicy.STRICT, bin_support_minimum_samples=bin_support_minimum_samples,
    )


class TestBinSupportMinimumSamplesParticipatesInIdentity:
    def test_default_is_twenty(self) -> None:
        spec = _make_spec()
        assert spec.bin_support_minimum_samples == 20

    def test_different_bin_support_minimum_produces_different_calibration_id(self) -> None:
        spec_a = _make_spec(bin_support_minimum_samples=20)
        spec_b = _make_spec(bin_support_minimum_samples=50)
        identity_a = compute_calibration_identity(spec_a)
        identity_b = compute_calibration_identity(spec_b)
        assert identity_a.calibration_id != identity_b.calibration_id

    def test_identical_bin_support_minimum_produces_identical_calibration_id(self) -> None:
        spec_a = _make_spec(bin_support_minimum_samples=33)
        spec_b = _make_spec(bin_support_minimum_samples=33)
        assert compute_calibration_identity(spec_a).calibration_id == compute_calibration_identity(spec_b).calibration_id

    def test_field_is_present_in_to_json_dict(self) -> None:
        spec = _make_spec(bin_support_minimum_samples=17)
        assert spec.to_json_dict()["bin_support_minimum_samples"] == 17
        assert spec.to_identity_payload()["bin_support_minimum_samples"] == 17

    def test_round_trips_through_json(self) -> None:
        spec = _make_spec(bin_support_minimum_samples=17)
        reloaded = CalibrationSpec.from_json_dict(spec.to_json_dict())
        assert reloaded.bin_support_minimum_samples == 17
        assert compute_calibration_identity(reloaded).calibration_id == compute_calibration_identity(spec).calibration_id

    def test_from_json_dict_defaults_to_twenty_when_field_absent(self) -> None:
        """An older persisted spec payload (written before this field
        existed) must still load, defaulting to the value that was the
        hardcoded behavior before this field was promoted out of
        `calibration.runner`'s module constant."""
        spec = _make_spec()
        raw = spec.to_json_dict()
        del raw["bin_support_minimum_samples"]
        reloaded = CalibrationSpec.from_json_dict(raw)
        assert reloaded.bin_support_minimum_samples == 20

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match="bin_support_minimum_samples"):
            _make_spec(bin_support_minimum_samples=0)

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match="bin_support_minimum_samples"):
            _make_spec(bin_support_minimum_samples=-5)
