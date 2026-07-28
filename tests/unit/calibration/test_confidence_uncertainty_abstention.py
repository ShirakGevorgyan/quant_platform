"""Confidence, uncertainty, and abstention tests (Milestone 4E, Section
32): scores finite in [0, 1], monotonicity properties that ARE
mathematically guaranteed, missing-component handling never silently
zero-filled, confidence categories matching configured boundaries, and
abstention decisions matching the frozen policy exactly."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from quant_platform.calibration.abstention import decide, evaluate_selective_prediction
from quant_platform.calibration.confidence import (
    compute_confidence,
    distance_from_threshold_component,
    probability_extremity_component,
)
from quant_platform.calibration.models import AbstentionPolicyKind, AbstentionReasonCode, Decision
from quant_platform.calibration.specs import AbstentionSpec, ConfidenceSpec, UncertaintySpec
from quant_platform.calibration.uncertainty import (
    bin_support_uncertainty_component,
    compute_uncertainty,
    entropy_component,
    margin_component,
)
from quant_platform.core.exceptions import (
    CalibrationDataError,
    CalibrationValidationError,
    ConfidencePolicyError,
    UncertaintyPolicyError,
)


class TestEntropyComponentMonotonicity:
    """Section 17: "higher entropy never produces lower entropy-based
    uncertainty" -- entropy_component IS the entropy-based uncertainty
    proxy, so this is a direct monotonicity property of the function."""

    def test_maximal_at_probability_one_half(self) -> None:
        assert entropy_component(0.5) == pytest.approx(1.0)

    def test_zero_at_the_extremes(self) -> None:
        assert entropy_component(0.0) == pytest.approx(0.0)
        assert entropy_component(1.0) == pytest.approx(0.0)

    def test_monotonically_decreases_moving_away_from_one_half(self) -> None:
        values = [entropy_component(p) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)]
        assert all(a >= b for a, b in pairwise(values))

    def test_symmetric_around_one_half(self) -> None:
        for p in (0.1, 0.3, 0.45):
            assert entropy_component(p) == pytest.approx(entropy_component(1.0 - p))


class TestMarginComponentMonotonicity:
    """Section 17: "larger threshold margin never produces greater
    margin uncertainty" -- margin_component is 1 - distance_from_
    threshold, so a LARGER margin (distance) must give a SMALLER (never
    greater) margin-uncertainty value."""

    def test_zero_at_the_threshold_itself(self) -> None:
        assert margin_component(0.5, 0.5) == pytest.approx(1.0)

    def test_decreases_as_distance_from_threshold_grows(self) -> None:
        threshold = 0.5
        distances_increasing = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        values = [margin_component(p, threshold) for p in distances_increasing]
        assert all(a >= b for a, b in pairwise(values))

    def test_is_the_exact_complement_of_distance_from_threshold(self) -> None:
        for p, threshold in ((0.7, 0.4), (0.2, 0.6), (0.9, 0.1)):
            assert margin_component(p, threshold) == pytest.approx(1.0 - distance_from_threshold_component(p, threshold))


class TestDistanceFromThresholdAndExtremity:
    def test_distance_from_threshold_is_zero_at_the_threshold(self) -> None:
        assert distance_from_threshold_component(0.37, 0.37) == pytest.approx(0.0)

    def test_distance_from_threshold_hand_computed_example(self) -> None:
        # normalizer = max(0.5, 1 - 0.5) = 0.5; |0.9 - 0.5| / 0.5 = 0.8
        assert distance_from_threshold_component(0.9, 0.5) == pytest.approx(0.8)

    def test_probability_extremity_hand_computed(self) -> None:
        assert probability_extremity_component(0.5) == pytest.approx(0.0)
        assert probability_extremity_component(0.9) == pytest.approx(0.8)
        assert probability_extremity_component(0.0) == pytest.approx(1.0)


class TestBinSupportUncertaintyComponent:
    def test_zero_once_support_meets_the_minimum(self) -> None:
        assert bin_support_uncertainty_component(20, minimum_support=20) == pytest.approx(0.0)
        assert bin_support_uncertainty_component(50, minimum_support=20) == pytest.approx(0.0)

    def test_maximal_at_zero_samples(self) -> None:
        assert bin_support_uncertainty_component(0, minimum_support=20) == pytest.approx(1.0)

    def test_scales_linearly_below_the_minimum(self) -> None:
        assert bin_support_uncertainty_component(10, minimum_support=20) == pytest.approx(0.5)


class TestConfidenceComposition:
    def test_single_component_heuristic_mode(self) -> None:
        spec = ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8)
        result = compute_confidence({"distance_from_threshold": 0.9}, spec=spec)
        assert result.score == pytest.approx(0.9)
        assert result.category == "very_high"
        assert result.provenance.value == "heuristic"

    def test_composite_weighted_average(self) -> None:
        spec = ConfidenceSpec(
            very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8,
            component_weights={"distance_from_threshold": 0.5, "probability_extremity": 0.5},
        )
        result = compute_confidence({"distance_from_threshold": 0.8, "probability_extremity": 0.4}, spec=spec)
        assert result.score == pytest.approx(0.6)
        assert result.provenance.value == "composite"

    def test_missing_component_is_renormalized_not_zero_filled(self) -> None:
        """Section 15/17: 'do not silently replace missing components
        with zero.' A missing component must be EXCLUDED from the
        weighted average (renormalized over what remains), never
        implicitly treated as contributing a 0.0."""
        spec = ConfidenceSpec(
            very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8,
            component_weights={"distance_from_threshold": 0.5, "calibration_bin_support": 0.5},
        )
        result = compute_confidence({"distance_from_threshold": 0.8, "calibration_bin_support": None}, spec=spec)
        # If zero-filled, score would be 0.4 (0.5*0.8 + 0.5*0.0); renormalized-only gives 0.8.
        assert result.score == pytest.approx(0.8)
        assert result.component_availability["calibration_bin_support"] is False
        assert any("calibration_bin_support" in code for code in result.reason_codes)

    def test_all_weighted_components_unavailable_raises(self) -> None:
        spec = ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8, component_weights={"x": 1.0})
        with pytest.raises(ConfidencePolicyError):
            compute_confidence({"x": None}, spec=spec)

    @pytest.mark.parametrize(
        ("score", "expected_category"),
        [(0.1, "very_low"), (0.3, "low"), (0.5, "medium"), (0.7, "high"), (0.9, "very_high")],
    )
    def test_category_matches_configured_boundaries(self, score: float, expected_category: str) -> None:
        spec = ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8)
        result = compute_confidence({"distance_from_threshold": score}, spec=spec)
        assert result.category == expected_category

    def test_score_is_always_finite_and_in_unit_interval(self) -> None:
        spec = ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8)
        for raw in (0.0, 0.5, 1.0):
            result = compute_confidence({"distance_from_threshold": raw}, spec=spec)
            assert math.isfinite(result.score)
            assert 0.0 <= result.score <= 1.0


class TestUncertaintyComposition:
    def test_missing_component_excluded_not_zero_filled(self) -> None:
        spec = UncertaintySpec(components=("entropy", "model_disagreement"), aggregation="mean")
        result = compute_uncertainty({"entropy": 0.6, "model_disagreement": None}, spec=spec)
        # If zero-filled: mean(0.6, 0.0) = 0.3; excluded: just 0.6.
        assert result.total_uncertainty == pytest.approx(0.6)
        assert result.component_availability["model_disagreement"] is False

    def test_missing_component_is_explicit_in_reason_codes(self) -> None:
        spec = UncertaintySpec(components=("entropy", "model_disagreement"), aggregation="mean")
        result = compute_uncertainty({"entropy": 0.6, "model_disagreement": None}, spec=spec)
        assert any("model_disagreement" in code for code in result.reason_codes)

    def test_all_components_unavailable_raises(self) -> None:
        spec = UncertaintySpec(components=("model_disagreement",), aggregation="mean")
        with pytest.raises(UncertaintyPolicyError):
            compute_uncertainty({"model_disagreement": None}, spec=spec)

    def test_missing_declared_key_entirely_raises(self) -> None:
        """The components mapping must contain an entry (even if None)
        for EVERY name in spec.components -- a name absent from the
        mapping entirely is a caller contract violation, not "missing
        data" to gracefully handle."""
        spec = UncertaintySpec(components=("entropy", "margin"), aggregation="mean")
        with pytest.raises(UncertaintyPolicyError):
            compute_uncertainty({"entropy": 0.5}, spec=spec)

    def test_max_aggregation(self) -> None:
        spec = UncertaintySpec(components=("entropy", "margin"), aggregation="max")
        result = compute_uncertainty({"entropy": 0.3, "margin": 0.7}, spec=spec)
        assert result.total_uncertainty == pytest.approx(0.7)

    def test_total_uncertainty_always_finite_and_in_unit_interval(self) -> None:
        spec = UncertaintySpec(components=("entropy",), aggregation="mean")
        for raw in (0.0, 0.5, 1.0):
            result = compute_uncertainty({"entropy": raw}, spec=spec)
            assert math.isfinite(result.total_uncertainty)
            assert 0.0 <= result.total_uncertainty <= 1.0


class TestAbstentionDecisionsMatchFrozenPolicyExactly:
    def test_none_policy_never_abstains(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.NONE)
        decision, reason = decide(0.5, 0.5, spec=spec)
        assert decision is Decision.POSITIVE  # boundary is >=
        assert reason is AbstentionReasonCode.NOT_ABSTAINED

    def test_symmetric_band_abstains_strictly_inside_the_band(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.SYMMETRIC_BAND, band_half_width=0.1)
        inside, reason_inside = decide(0.55, 0.5, spec=spec)
        assert inside is Decision.ABSTAIN
        assert reason_inside is AbstentionReasonCode.INSIDE_UNCERTAINTY_BAND
        outside, reason_outside = decide(0.65, 0.5, spec=spec)
        assert outside is Decision.POSITIVE
        assert reason_outside is AbstentionReasonCode.NOT_ABSTAINED

    def test_min_confidence_abstains_below_the_floor(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.MIN_CONFIDENCE, min_confidence=0.6)
        below, reason_below = decide(0.9, 0.5, spec=spec, confidence=0.5)
        assert below is Decision.ABSTAIN
        assert reason_below is AbstentionReasonCode.BELOW_CONFIDENCE_FLOOR
        above, reason_above = decide(0.9, 0.5, spec=spec, confidence=0.7)
        assert above is Decision.POSITIVE
        assert reason_above is AbstentionReasonCode.NOT_ABSTAINED

    def test_min_confidence_without_a_confidence_score_raises(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.MIN_CONFIDENCE, min_confidence=0.6)
        with pytest.raises(CalibrationValidationError):
            decide(0.9, 0.5, spec=spec)

    def test_max_uncertainty_abstains_above_the_limit(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.MAX_UNCERTAINTY, max_uncertainty=0.4)
        above, reason_above = decide(0.9, 0.5, spec=spec, uncertainty=0.5)
        assert above is Decision.ABSTAIN
        assert reason_above is AbstentionReasonCode.UNCERTAINTY_ABOVE_LIMIT
        below, _reason_below = decide(0.9, 0.5, spec=spec, uncertainty=0.3)
        assert below is Decision.POSITIVE

    def test_class_specific_boundaries(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.CLASS_SPECIFIC_BOUNDARIES, negative_upper_bound=0.3, positive_lower_bound=0.7)
        negative, _ = decide(0.2, 0.5, spec=spec)
        assert negative is Decision.NEGATIVE
        positive, _ = decide(0.8, 0.5, spec=spec)
        assert positive is Decision.POSITIVE
        abstain, reason = decide(0.5, 0.5, spec=spec)
        assert abstain is Decision.ABSTAIN
        assert reason is AbstentionReasonCode.INSIDE_UNCERTAINTY_BAND

    def test_invalid_probability_fails_closed_unconditionally(self) -> None:
        spec = AbstentionSpec(policy=AbstentionPolicyKind.NONE)
        with pytest.raises(CalibrationDataError):
            decide(1.5, 0.5, spec=spec)
        with pytest.raises(CalibrationDataError):
            decide(math.nan, 0.5, spec=spec)


class TestSelectivePredictionEvaluation:
    def test_coverage_and_accuracy_are_always_reported_together(self) -> None:
        """Section 14: never present improved accepted-sample accuracy
        without the reduced coverage -- both fields always exist
        together on the same result object."""
        decisions = [Decision.POSITIVE, Decision.ABSTAIN, Decision.NEGATIVE, Decision.POSITIVE]
        labels = [1.0, 0.0, 0.0, 0.0]
        result = evaluate_selective_prediction(decisions, labels)
        assert result.n_total == 4
        assert result.n_accepted == 3
        assert result.coverage == pytest.approx(0.75)
        assert result.abstention_rate == pytest.approx(0.25)
        # accepted: (POSITIVE,1.0)=correct, (NEGATIVE,0.0)=correct, (POSITIVE,0.0)=wrong -> 2/3
        assert result.accuracy_on_accepted == pytest.approx(2 / 3)

    def test_zero_accepted_predictions_leaves_accuracy_and_risk_as_none(self) -> None:
        """Never fabricate a 0.0 or 1.0 accuracy for zero accepted
        predictions -- structurally enforced by `SelectivePredictionEvaluation.__post_init__`."""
        decisions = [Decision.ABSTAIN, Decision.ABSTAIN]
        labels = [1.0, 0.0]
        result = evaluate_selective_prediction(decisions, labels)
        assert result.n_accepted == 0
        assert result.accuracy_on_accepted is None
        assert result.selective_risk is None

    def test_all_accepted_matches_coverage_one(self) -> None:
        decisions = [Decision.POSITIVE, Decision.NEGATIVE]
        labels = [1.0, 0.0]
        result = evaluate_selective_prediction(decisions, labels)
        assert result.coverage == pytest.approx(1.0)
        assert result.accuracy_on_accepted == pytest.approx(1.0)

    def test_requires_at_least_one_sample(self) -> None:
        with pytest.raises(CalibrationDataError):
            evaluate_selective_prediction([], [])
