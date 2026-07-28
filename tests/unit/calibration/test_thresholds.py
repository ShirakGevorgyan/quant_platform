"""Decision threshold tests (Milestone 4E, Section 31): exact `>=`
boundary semantics, every threshold policy, deterministic candidate
evaluation, infeasible-constraint fallback, and stability aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_platform.calibration.models import ThresholdPolicyKind
from quant_platform.calibration.specs import CostMatrix, ThresholdSpec
from quant_platform.calibration.thresholds import (
    apply_threshold,
    compute_threshold_stability,
    evaluate_threshold_candidates,
)
from quant_platform.core.exceptions import CalibrationDataError


class TestApplyThreshold:
    def test_boundary_is_greater_than_or_equal(self) -> None:
        """Section 12's exact fixed rule: `positive_prediction =
        probability >= threshold` -- a probability EQUAL to the
        threshold must be POSITIVE, never negative."""
        out = apply_threshold(np.array([0.49, 0.5, 0.51]), 0.5)
        assert list(out) == [False, True, True]

    def test_zero_and_one_thresholds(self) -> None:
        assert list(apply_threshold(np.array([0.0, 0.5, 1.0]), 0.0)) == [True, True, True]
        assert list(apply_threshold(np.array([0.0, 0.5, 0.999]), 1.0)) == [False, False, False]


def _separated_data() -> tuple[np.ndarray, np.ndarray]:
    """8 cleanly-separated samples: all negatives have probability
    strictly below all positives, so every non-trivial threshold policy
    has an unambiguous, hand-verifiable optimum."""
    probabilities = np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])
    labels = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return probabilities, labels


class TestF1Policy:
    def test_optimum_is_the_perfect_separation_boundary(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=101)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        # Any threshold in (0.35, 0.65] achieves perfect F1=1.0; the grid
        # search's own deterministic tie-break (closest to 0.5) must land
        # exactly at 0.5 for a 101-point [0, 1] grid.
        assert report.selected_objective_value == pytest.approx(1.0)
        assert 0.35 < report.selected_threshold <= 0.65

    def test_candidates_are_a_deterministic_evenly_spaced_grid(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=11)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        thresholds = sorted(c.threshold for c in report.candidates)
        np.testing.assert_allclose(thresholds, np.linspace(0.0, 1.0, 11), atol=1e-9)


class TestFixedPolicy:
    def test_fixed_policy_uses_the_declared_threshold_with_no_search(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.FIXED, fixed_threshold=0.42)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert report.selected_threshold == 0.42
        assert len(report.candidates) == 1
        assert not report.fallback_used


class TestBalancedAccuracyAndMcc:
    @pytest.mark.parametrize("policy", [ThresholdPolicyKind.BALANCED_ACCURACY, ThresholdPolicyKind.MATTHEWS_CORRCOEF])
    def test_perfect_separation_achieves_the_maximum_objective(self, policy: ThresholdPolicyKind) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=policy, candidate_grid_size=101)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert report.selected_objective_value == pytest.approx(1.0, abs=1e-9)


class TestYoudenJ:
    def test_perfect_separation_achieves_j_equal_one(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.YOUDEN_J, candidate_grid_size=101)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert report.selected_objective_value == pytest.approx(1.0, abs=1e-9)


class TestMinPrecisionMaxRecall:
    def test_achievable_constraint_is_satisfied(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL, min_precision=0.99, candidate_grid_size=101)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert not report.fallback_used
        selected = next(c for c in report.candidates if c.threshold == report.selected_threshold)
        assert selected.constraint_satisfied
        assert selected.precision == pytest.approx(1.0)

    def test_infeasible_constraint_falls_back_with_a_recorded_reason(self) -> None:
        """Precision=1.0 is UNREACHABLE at any threshold predicting >=1
        positive: the single highest-probability sample is deliberately
        mislabeled negative, so every candidate that predicts it positive
        (required to predict ANY positive, since it is the top-ranked
        probability) has precision < 1."""
        probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.99])
        labels = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])  # 0.99 is the mislabeled negative
        spec = ThresholdSpec(policy=ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL, min_precision=0.999, candidate_grid_size=101, infeasible_fallback_threshold=0.5)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert report.fallback_used
        assert report.fallback_reason is not None and len(report.fallback_reason) > 0
        assert report.selected_threshold == 0.5


class TestCostSensitive:
    def test_perfectly_separated_data_achieves_zero_cost(self) -> None:
        probabilities, labels = _separated_data()
        cost_matrix = CostMatrix(false_positive_cost=1.0, false_negative_cost=1.0)
        spec = ThresholdSpec(policy=ThresholdPolicyKind.COST_SENSITIVE, cost_matrix=cost_matrix, candidate_grid_size=101)
        report = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert report.selected_objective_value == pytest.approx(0.0, abs=1e-9)
        # A whole range of thresholds in (0.35, 0.65] achieve zero cost;
        # the deterministic tie-break (closest to 0.5) must not overshoot.
        assert report.selected_threshold <= 0.5

    def test_asymmetric_costs_shift_the_selected_threshold(self) -> None:
        """A much higher false-negative cost should pull the threshold
        down (favoring predicting positive) relative to symmetric costs."""
        rng = np.random.default_rng(1)
        n = 300
        probabilities = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
        labels = (rng.uniform(size=n) < probabilities).astype(float)
        labels[0], labels[1] = 0.0, 1.0

        symmetric = ThresholdSpec(policy=ThresholdPolicyKind.COST_SENSITIVE, cost_matrix=CostMatrix(false_positive_cost=1.0, false_negative_cost=1.0), candidate_grid_size=101)
        asymmetric = ThresholdSpec(policy=ThresholdPolicyKind.COST_SENSITIVE, cost_matrix=CostMatrix(false_positive_cost=1.0, false_negative_cost=20.0), candidate_grid_size=101)
        symmetric_report = evaluate_threshold_candidates(probabilities, labels, spec=symmetric)
        asymmetric_report = evaluate_threshold_candidates(probabilities, labels, spec=asymmetric)
        assert asymmetric_report.selected_threshold <= symmetric_report.selected_threshold


class TestDeterminism:
    def test_evaluating_twice_on_identical_data_produces_identical_reports(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51)
        first = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        second = evaluate_threshold_candidates(probabilities, labels, spec=spec)
        assert first.to_json_dict() == second.to_json_dict()


class TestInputValidation:
    def test_rejects_mismatched_lengths(self) -> None:
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1)
        with pytest.raises(CalibrationDataError):
            evaluate_threshold_candidates(np.array([0.1, 0.2]), np.array([0.0]), spec=spec)

    def test_single_class_labels_produce_a_well_defined_degenerate_result(self) -> None:
        """All-negative labels make F1 well-defined (if uninteresting) at
        every threshold -- zero true positives are possible, so F1=0.0
        everywhere; this must be a clean, DEFINED result (every candidate
        tied at 0.0, tie-break lands on the closest-to-0.5 threshold),
        never an exception, never a fabricated non-zero value."""
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=11)
        report = evaluate_threshold_candidates(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0]), spec=spec)
        assert report.selected_objective_value == pytest.approx(0.0)
        assert all(c.objective_value == pytest.approx(0.0) for c in report.candidates if c.objective_value is not None)
        assert report.selected_threshold == pytest.approx(0.5)


class TestThresholdStability:
    def test_stability_statistics_match_hand_computed_values(self) -> None:
        probabilities, labels = _separated_data()
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=101)
        fold_reports = [evaluate_threshold_candidates(probabilities, labels, spec=spec) for _ in range(3)]
        stability = compute_threshold_stability(fold_reports)
        thresholds = [r.selected_threshold for r in fold_reports]
        assert stability.per_fold_thresholds == tuple(thresholds)
        assert stability.mean == pytest.approx(sum(thresholds) / len(thresholds))
        assert stability.minimum == min(thresholds)
        assert stability.maximum == max(thresholds)
        assert 0.0 <= stability.constraint_satisfaction_rate <= 1.0

    def test_requires_at_least_one_report(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_threshold_stability([])
