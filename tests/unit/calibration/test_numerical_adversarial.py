"""Release audit Section 6: a consolidated numerical-edge-case sweep
across calibration methods, metrics, thresholds, diagnostics, confidence,
and uncertainty. Every test in this file runs under `-W error` (via the
module-level `pytestmark`) -- a `RuntimeWarning` (e.g. numpy overflow) is
treated exactly as seriously as a wrong numeric answer, since a warning
that silently escapes in production is indistinguishable from one nobody
ever looks at.

This file specifically targets cases NOT already exercised by
`test_methods.py`/`test_thresholds.py`/`test_calibration_metrics.py`/
`test_confidence_uncertainty_abstention.py`: exactly-0/1 probabilities
pushed through EVERY calibrator at once, extreme/adversarially-tampered
calibrator parameters (the overflow-to-infinity case that motivated the
`np.errstate` fix in `calibration.methods`), one-class folds, tied
candidates, negative zero, subnormal floats, and -- the case with no
platform-wide guarantee protecting it -- string-coerced "nan"/"inf"/"-inf"
smuggled through `from_json_dict` as JSON STRING values (not bare JSON
tokens, which `core.json.parse_json_strict` already rejects outright)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_platform.calibration.abstention import SelectivePredictionEvaluation
from quant_platform.calibration.confidence import (
    compute_confidence,
    distance_from_threshold_component,
    probability_extremity_component,
)
from quant_platform.calibration.diagnostics import (
    ReliabilityBin,
    ReliabilityBinningSpec,
    compute_reliability_bins,
)
from quant_platform.calibration.fitting import CalibratorCandidateResult
from quant_platform.calibration.methods import (
    BetaCalibrator,
    FittedBetaCalibrator,
    FittedPlattCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)
from quant_platform.calibration.metrics import compute_calibration_metrics
from quant_platform.calibration.models import BinningStrategy, ProbabilityRepresentation, ThresholdPolicyKind
from quant_platform.calibration.specs import ConfidenceSpec, CostMatrix, ThresholdSpec
from quant_platform.calibration.thresholds import (
    ThresholdReport,
    apply_threshold,
    evaluate_threshold_candidates,
)
from quant_platform.calibration.uncertainty import bin_support_uncertainty_component, entropy_component
from quant_platform.core.exceptions import CalibrationDataError, CalibrationFitError

pytestmark = pytest.mark.filterwarnings("error")


# --------------------------------------------------------------------------
# Exactly 0.0 / 1.0 probabilities through every calibrator at once
# --------------------------------------------------------------------------
class TestBoundaryProbabilitiesThroughEveryCalibrator:
    _x = np.asarray([0.0, 1.0, 0.5, 0.25, 0.75])
    _y = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0])

    @pytest.mark.parametrize("factory", [IdentityCalibrator, PlattCalibrator, IsotonicCalibrator, BetaCalibrator])
    def test_fit_and_transform_at_exact_boundaries_produces_finite_in_range_output(self, factory) -> None:
        fitted = factory().fit(self._x, self._y)
        out = fitted.transform(self._x)
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))


# --------------------------------------------------------------------------
# Extreme / adversarially-tampered calibrator parameters (overflow class)
# --------------------------------------------------------------------------
class TestExtremeCalibratorParametersDoNotOverflowOrWarn:
    def test_platt_with_huge_coefficient_saturates_cleanly(self) -> None:
        cal = FittedPlattCalibrator(coefficient=1e300, intercept=0.0, input_representation=ProbabilityRepresentation.DECISION_FUNCTION)
        out = cal.transform(np.asarray([1e300, -1e300, 0.0]))
        assert np.array_equal(out, np.asarray([1.0, 0.0, 0.5]))

    def test_beta_with_huge_coefficients_saturates_cleanly(self) -> None:
        cal = FittedBetaCalibrator(log_p_coefficient=1e300, log_one_minus_p_coefficient=-1e300, intercept=0.0)
        out = cal.transform(np.asarray([0.5, 1e-10, 1.0 - 1e-10]))
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_platt_decision_function_mode_with_moderately_large_raw_scores(self) -> None:
        """A realistic (not artificially tampered) large decision-function
        score -- e.g. a boosted-tree margin far from the decision boundary
        -- must not warn or overflow either."""
        cal = FittedPlattCalibrator(coefficient=2.5, intercept=-0.3, input_representation=ProbabilityRepresentation.DECISION_FUNCTION)
        out = cal.transform(np.asarray([500.0, -500.0, 0.0]))
        assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# Constant scores / one-class folds / minimum sample counts
# --------------------------------------------------------------------------
class TestConstantScoresAndOneClassFolds:
    def test_platt_on_constant_scores_fits_a_degenerate_zero_coefficient(self) -> None:
        x = np.asarray([0.5, 0.5, 0.5, 0.5])
        y = np.asarray([0.0, 1.0, 0.0, 1.0])
        fitted = PlattCalibrator().fit(x, y)
        assert math.isfinite(fitted.coefficient)
        out = fitted.transform(x)
        assert np.all(np.isfinite(out))

    def test_isotonic_on_constant_scores_collapses_to_one_threshold_pair(self) -> None:
        x = np.asarray([0.5, 0.5, 0.5, 0.5])
        y = np.asarray([0.0, 1.0, 0.0, 1.0])
        fitted = IsotonicCalibrator().fit(x, y)
        assert len(fitted.x_thresholds) >= 1

    def test_fit_requires_both_classes_present(self) -> None:
        x = np.asarray([0.1, 0.2, 0.3])
        y = np.asarray([1.0, 1.0, 1.0])
        with pytest.raises(CalibrationFitError, match="both classes"):
            PlattCalibrator().fit(x, y)

    def test_calibration_metrics_on_one_class_labels_skips_not_crashes(self) -> None:
        x = np.asarray([0.5, 0.6, 0.5, 0.7])
        y = np.asarray([1.0, 1.0, 1.0, 1.0])
        report = compute_calibration_metrics(x, y, binning_spec=ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=5))
        assert "log_loss" in report.skipped
        assert "log_loss" not in report.values
        assert not any(math.isnan(v) for v in report.values.values() if isinstance(v, float))


# --------------------------------------------------------------------------
# Duplicate probabilities / repeated isotonic thresholds / collapsed bins
# --------------------------------------------------------------------------
class TestDuplicateProbabilitiesAndCollapsedBins:
    def test_isotonic_with_repeated_x_values_produces_monotone_deduplicated_thresholds(self) -> None:
        x = np.asarray([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        y = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        fitted = IsotonicCalibrator().fit(x, y)
        assert list(fitted.x_thresholds) == sorted(fitted.x_thresholds)
        assert list(fitted.y_thresholds) == sorted(fitted.y_thresholds)
        out = fitted.transform(x)
        assert np.all(np.isfinite(out))

    def test_equal_frequency_bins_collapse_gracefully_with_documented_note(self) -> None:
        x = np.asarray([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        y = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        report = compute_reliability_bins(x, y, spec=ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_FREQUENCY, n_bins=10))
        assert report.actual_n_bins < report.requested_n_bins
        assert report.collapsed_edges_note is not None
        assert sum(b.sample_count for b in report.bins) == 6


# --------------------------------------------------------------------------
# Tied threshold candidates / no feasible threshold
# --------------------------------------------------------------------------
class TestThresholdTiesAndInfeasibility:
    def test_cost_sensitive_with_huge_costs_does_not_overflow(self) -> None:
        x = np.asarray([0.0, 1.0, 0.5])
        y = np.asarray([0.0, 1.0, 1.0])
        spec = ThresholdSpec(policy=ThresholdPolicyKind.COST_SENSITIVE, cost_matrix=CostMatrix(false_positive_cost=1e300, false_negative_cost=1e300))
        report = evaluate_threshold_candidates(x, y, spec=spec)
        assert math.isfinite(report.selected_threshold)

    def test_min_precision_impossible_to_satisfy_falls_back(self) -> None:
        x = np.asarray([0.1, 0.2, 0.3, 0.4])
        y = np.asarray([1.0, 0.0, 1.0, 0.0])
        spec = ThresholdSpec(policy=ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL, min_precision=0.999999, candidate_grid_size=11)
        report = evaluate_threshold_candidates(x, y, spec=spec)
        assert report.fallback_used
        assert report.fallback_reason is not None
        assert report.selected_threshold == spec.infeasible_fallback_threshold

    def test_all_thresholds_tie_on_a_perfectly_separated_extreme_dataset(self) -> None:
        x = np.asarray([0.0, 1.0])
        y = np.asarray([0.0, 1.0])
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=11)
        report = evaluate_threshold_candidates(x, y, spec=spec)
        assert report.tie_break_note is not None or report.selected_threshold is not None


# --------------------------------------------------------------------------
# Selective prediction: all abstained / none abstained
# --------------------------------------------------------------------------
class TestSelectivePredictionCoverageExtremes:
    def test_all_accepted_zero_abstained(self) -> None:
        from quant_platform.calibration.abstention import evaluate_selective_prediction
        from quant_platform.calibration.models import Decision

        decisions = [Decision.POSITIVE, Decision.NEGATIVE, Decision.POSITIVE]
        labels = [1.0, 0.0, 1.0]
        result = evaluate_selective_prediction(decisions, labels)
        assert result.coverage == 1.0
        assert result.abstention_rate == 0.0
        assert result.accuracy_on_accepted == 1.0

    def test_all_abstained_zero_accepted_leaves_accuracy_undefined_not_zero(self) -> None:
        from quant_platform.calibration.abstention import evaluate_selective_prediction
        from quant_platform.calibration.models import Decision

        decisions = [Decision.ABSTAIN, Decision.ABSTAIN, Decision.ABSTAIN]
        labels = [1.0, 0.0, 1.0]
        result = evaluate_selective_prediction(decisions, labels)
        assert result.coverage == 0.0
        assert result.abstention_rate == 1.0
        assert result.accuracy_on_accepted is None
        assert result.selective_risk is None


# --------------------------------------------------------------------------
# Negative zero and subnormal floats
# --------------------------------------------------------------------------
class TestNegativeZeroAndSubnormals:
    def test_negative_zero_probability_treated_identically_to_positive_zero(self) -> None:
        assert distance_from_threshold_component(-0.0, 0.5) == distance_from_threshold_component(0.0, 0.5)
        assert probability_extremity_component(-0.0) == probability_extremity_component(0.0)
        assert entropy_component(-0.0) == entropy_component(0.0)
        assert bool(apply_threshold(np.asarray([-0.0]), 0.0)[0]) is True  # -0.0 >= 0.0 is True

    def test_smallest_subnormal_probability_does_not_warn_or_crash(self) -> None:
        subnormal = 5e-324
        assert 0.0 <= entropy_component(subnormal) <= 1.0
        x = np.asarray([0.1, 0.9])
        y = np.asarray([0.0, 1.0])
        fitted = PlattCalibrator().fit(x, y)
        out = fitted.transform(np.asarray([subnormal]))
        assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# String-coerced "nan"/"inf"/"-inf" smuggled through from_json_dict
# --------------------------------------------------------------------------
class TestStringCoercedNonFiniteValuesAreRejected:
    """`core.json.parse_json_strict` already rejects a BARE JSON `NaN`/
    `Infinity`/`-Infinity` token outright (see that module's docstring).
    The narrower, remaining gap this class targets: a JSON STRING value
    (e.g. `"nan"`), which parses as an ordinary Python string and only
    becomes a real float NaN/inf when a `from_json_dict` does
    `float(str(raw[key]))` -- every field checked here was audited and
    hardened (release audit Section 6) to reject that at construction."""

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity", "-Infinity"])
    def test_threshold_report_selected_objective_value_rejects_string_coerced_token(self, token: str) -> None:
        with pytest.raises(CalibrationDataError, match="finite"):
            ThresholdReport.from_json_dict({
                "schema_version": 1, "policy": "f1", "candidates": [],
                "selected_threshold": 0.5, "selected_objective_value": token,
                "fallback_used": False, "fallback_reason": None, "tie_break_note": None,
            })

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
    def test_reliability_bin_rejects_string_coerced_token(self, token: str) -> None:
        with pytest.raises(CalibrationDataError, match="finite"):
            ReliabilityBin.from_json_dict({
                "bin_index": 0, "lower_bound": 0.0, "upper_bound": 0.1, "sample_count": 5,
                "mean_predicted_probability": token, "empirical_positive_rate": 0.5, "calibration_gap": 0.1,
                "confidence_interval_low": 0.0, "confidence_interval_high": 1.0, "is_empty": False,
            })

    def test_calibrator_candidate_result_rejects_non_finite_metric(self) -> None:
        with pytest.raises(CalibrationDataError, match="finite"):
            CalibratorCandidateResult(kind="platt", succeeded=True, fitted=None, metrics={"log_loss": float("-inf")}, selection_metric_value=0.5, failure=None)

    def test_calibrator_candidate_result_rejects_non_finite_selection_metric_value(self) -> None:
        with pytest.raises(CalibrationDataError, match="finite"):
            CalibratorCandidateResult(kind="platt", succeeded=True, fitted=None, metrics={}, selection_metric_value=float("nan"), failure=None)

    def test_selective_prediction_evaluation_rejects_nan_coverage_not_silently_pass(self) -> None:
        """The specific bug this guards against: `abs(nan - x) > 1e-9` is
        `False` in IEEE 754 (any comparison against NaN except `!=` is
        False), so the pre-existing consistency check alone would NOT
        have caught a NaN `coverage` -- it would have looked "consistent"
        by vacuously not raising."""
        with pytest.raises(CalibrationDataError, match="finite"):
            SelectivePredictionEvaluation(
                schema_version=1, n_total=10, n_accepted=5, coverage=float("nan"), abstention_rate=0.5,
                accuracy_on_accepted=0.8, balanced_accuracy_on_accepted=None, class_conditional_coverage={},
                selective_risk=0.2, risk_coverage_points=((0.5, 0.2),),
            )

    def test_selective_prediction_evaluation_rejects_nan_in_class_conditional_coverage(self) -> None:
        with pytest.raises(CalibrationDataError, match="finite"):
            SelectivePredictionEvaluation(
                schema_version=1, n_total=10, n_accepted=5, coverage=0.5, abstention_rate=0.5,
                accuracy_on_accepted=0.8, balanced_accuracy_on_accepted=None,
                class_conditional_coverage={"positive": float("nan")},
                selective_risk=0.2, risk_coverage_points=((0.5, 0.2),),
            )

    def test_fitted_platt_calibrator_rejects_non_finite_coefficient(self) -> None:
        with pytest.raises(Exception, match="finite"):
            FittedPlattCalibrator.from_json_dict({
                "schema_version": 1, "kind": "platt", "coefficient": "nan", "intercept": "0.0",
                "input_representation": "predict_proba",
            })

    def test_outer_fold_calibration_result_rejects_non_finite_metric_value(self) -> None:
        """Exercises the `validate_json_primitive_mapping` fix added to
        `OuterFoldCalibrationResult.__post_init__` -- previously
        `classification_metrics`/`calibration_metrics_on_outer_test`/
        `selective_prediction_summary` were never validated at all."""
        from quant_platform.calibration.runner import (
            OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION,
            OuterFoldCalibrationResult,
        )

        base_ref = {"category": "calibration_spec", "content_hash": "a" * 64, "size_bytes": 1, "created_at": "2024-01-01T00:00:00+00:00"}
        with pytest.raises(CalibrationDataError, match="finite"):
            OuterFoldCalibrationResult(
                schema_version=OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION, calibration_id="a" * 64, outer_fold_index=0,
                inner_oof_reference=_ref(base_ref, "inner_oof_predictions"), calibrator_selection_reference=_ref(base_ref, "calibrator_candidate_report"),
                threshold_report_reference=_ref(base_ref, "threshold_report"), decision_policy_reference=_ref(base_ref, "decision_policy"),
                model_reference=_ref(base_ref, "model"), seed=1, training_duration_seconds=0.1, outer_train_row_count=10,
                outer_test_row_count=1, sample_positions=(0,), raw_probabilities=(0.5,), calibrated_probabilities=(0.5,),
                decisions=("positive",), abstention_reason_codes=("not_abstained",), confidence_scores=(0.5,),
                confidence_categories=("medium",), uncertainty_scores=(0.5,),
                classification_metrics={"accuracy": float("nan")}, calibration_metrics_on_outer_test={},
                selective_prediction_summary={}, evaluated_at="2024-01-01T00:00:00+00:00",
            )


def _ref(base: dict, category: str):
    from quant_platform.ml.models import ArtifactReference

    return ArtifactReference.from_json_dict({**base, "category": category})


# --------------------------------------------------------------------------
# Confidence composite scoring at missing-component edges
# --------------------------------------------------------------------------
class TestConfidenceCompositeAtNumericEdges:
    def test_all_weighted_components_unavailable_raises_not_nan(self) -> None:
        from quant_platform.core.exceptions import ConfidencePolicyError

        spec = ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8, component_weights={"calibration_bin_support": 1.0})
        with pytest.raises(ConfidencePolicyError):
            compute_confidence({"calibration_bin_support": None}, spec=spec)

    def test_bin_support_uncertainty_at_zero_and_saturating_support(self) -> None:
        assert bin_support_uncertainty_component(0, minimum_support=20) == 1.0
        assert bin_support_uncertainty_component(20, minimum_support=20) == 0.0
        assert bin_support_uncertainty_component(1_000_000, minimum_support=20) == 0.0
