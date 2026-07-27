from __future__ import annotations

import numpy as np
import pytest

from quant_platform.ml.metrics import (
    CLASSIFICATION_METRIC_NAMES,
    REGRESSION_METRIC_NAMES,
    aggregate_fold_metrics,
    aggregate_metric_values,
    compute_classification_metrics,
    compute_metrics,
    compute_regression_metrics,
)
from quant_platform.ml.models import ObjectiveType


@pytest.mark.filterwarnings("error")
class TestComputeClassificationMetrics:
    """BLOCKER 1 (warning-cleanliness audit): every test in this class
    turns ANY warning into a hard test failure -- `compute_classification_
    metrics` must never let `sklearn` warn about a degenerate input this
    platform claims to support (see `ml.metrics`'s own module docstring:
    every such case is decided BEFORE calling `sklearn`, never after)."""

    def test_all_metrics_present_for_normal_input(self) -> None:
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.6, 0.8, 0.4, 0.2, 0.7, 0.3])
        report = compute_classification_metrics(y_true, y_pred, y_proba)
        assert set(report.values) == set(CLASSIFICATION_METRIC_NAMES)
        assert report.skipped == {}
        for value in report.values.values():
            assert np.isfinite(value)

    def test_no_proba_skips_auc_metrics_only(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        report = compute_classification_metrics(y_true, y_pred, None)
        assert "roc_auc" in report.skipped
        assert "pr_auc" in report.skipped
        assert set(report.values) == set(CLASSIFICATION_METRIC_NAMES) - {"roc_auc", "pr_auc"}

    def test_constant_true_labels_skips_class_balance_dependent_metrics_but_computes_the_rest(self) -> None:
        """A single-class `y_true` makes ROC AUC/PR AUC/balanced accuracy/
        MCC all undefined (each needs both classes actually present in
        `y_true` to mean anything) -- all four are OMITTED with an
        explicit reason, never computed to a degenerate value AND never
        left to `sklearn` to warn about internally. Accuracy/precision/
        recall/F1 remain always computable."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_proba = np.array([0.9, 0.8, 0.3, 0.7])
        report = compute_classification_metrics(y_true, y_pred, y_proba)
        assert set(report.values) == {"accuracy", "precision", "recall", "f1"}
        assert set(report.skipped) == {"roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef"}
        assert "only 1 distinct class" in report.skipped["roc_auc"]
        assert "y_true contains only 1 distinct class" in report.skipped["balanced_accuracy"]
        assert "y_true contains only 1 distinct class" in report.skipped["matthews_corrcoef"]
        assert np.isfinite(report.values["accuracy"])

    def test_perfect_predictions_score_maximally(self) -> None:
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = y_true.copy()
        y_proba = np.array([0.01, 0.99, 0.02, 0.98, 0.97])
        report = compute_classification_metrics(y_true, y_pred, y_proba)
        assert report.values["accuracy"] == 1.0
        assert report.values["f1"] == 1.0
        assert report.values["matthews_corrcoef"] == 1.0
        assert report.values["roc_auc"] == 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_classification_metrics(np.array([0, 1, 0]), np.array([0, 1]))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_classification_metrics(np.array([]), np.array([]))

    def test_nan_in_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compute_classification_metrics(np.array([0, 1]), np.array([0, np.nan]))

    def test_nan_in_true_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compute_classification_metrics(np.array([0, np.nan]), np.array([0, 1]))

    def test_infinite_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compute_classification_metrics(np.array([0, 1]), np.array([0, np.inf]))

    def test_infinite_true_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compute_classification_metrics(np.array([0, np.inf]), np.array([0, 1]))

    def test_infinite_proba_raises(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compute_classification_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0.1, np.inf, 0.2, 0.9]))


@pytest.mark.filterwarnings("error")
class TestDegenerateClassificationLabelDomains:
    """Milestone 4C release-readiness audit, "BLOCKER 1 -- ZERO UNEXPECTED
    METRIC WARNINGS": one test per audited degenerate-input bullet,
    each asserting BOTH the chosen semantic (compute/omit/reject) AND
    that no `sklearn` warning is emitted (`filterwarnings("error")` at
    the class level turns any into a hard failure)."""

    def test_ytrue_one_class_ypred_same_one_class(self) -> None:
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.9, 0.8, 0.7, 0.95]))
        assert report.values["accuracy"] == 1.0
        for name in ("balanced_accuracy", "matthews_corrcoef", "roc_auc", "pr_auc"):
            assert name in report.skipped

    def test_ytrue_one_class_ypred_has_an_additional_unseen_class(self) -> None:
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 1.0, 0.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.9, 0.8, 0.4, 0.95]))
        assert report.values["accuracy"] == 0.75
        for name in ("balanced_accuracy", "matthews_corrcoef", "roc_auc", "pr_auc"):
            assert name in report.skipped

    def test_ytrue_and_ypred_contain_different_single_classes(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.9, 0.8, 0.7, 0.95]))
        assert report.values["accuracy"] == 0.0
        for name in ("balanced_accuracy", "matthews_corrcoef", "roc_auc", "pr_auc"):
            assert name in report.skipped

    def test_positive_class_absent_from_ytrue(self) -> None:
        """`y_true` is entirely the NEGATIVE class (`0.0`) -- the
        positive class (`1.0`) never occurs in this fold's ground truth."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.1, 0.6, 0.2, 0.3, 0.7]))
        assert report.values["accuracy"] == 0.6
        for name in ("balanced_accuracy", "matthews_corrcoef", "roc_auc", "pr_auc"):
            assert name in report.skipped

    def test_negative_class_absent_from_ytrue(self) -> None:
        """`y_true` is entirely the POSITIVE class (`1.0`) -- the
        negative class (`0.0`) never occurs in this fold's ground truth."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.9, 0.4, 0.8, 0.7, 0.3]))
        assert report.values["accuracy"] == 0.6
        for name in ("balanced_accuracy", "matthews_corrcoef", "roc_auc", "pr_auc"):
            assert name in report.skipped

    def test_valid_varied_probabilities_but_auc_still_undefined(self) -> None:
        """Genuinely informative, varied predicted probabilities do NOT
        rescue ROC AUC/PR AUC when `y_true` itself has only one class --
        the metric is about ranking POSITIVES ahead of NEGATIVES, which
        is undefined when one of the two does not exist in `y_true`."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_proba = np.array([0.05, 0.99, 0.42, 0.77, 0.31])  # genuinely varied, not degenerate itself
        report = compute_classification_metrics(y_true, (y_proba >= 0.5).astype(float), y_proba)
        assert "roc_auc" in report.skipped
        assert "pr_auc" in report.skipped

    def test_balanced_accuracy_with_a_class_appearing_only_in_predictions(self) -> None:
        """Same underlying condition as the "additional unseen class"
        case above, named to match the audit's own bullet exactly:
        `y_pred` contains a class that never appears in `y_true`."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 1.0, 0.0, 0.0])
        report = compute_classification_metrics(y_true, y_pred, np.array([0.2, 0.6, 0.3, 0.1]))
        assert "balanced_accuracy" in report.skipped

    def test_mcc_both_arrays_constant_same_class(self) -> None:
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred)
        assert "matthews_corrcoef" in report.skipped

    def test_mcc_both_arrays_constant_different_classes(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        report = compute_classification_metrics(y_true, y_pred)
        assert "matthews_corrcoef" in report.skipped

    def test_mcc_ytrue_constant_ypred_varies(self) -> None:
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        report = compute_classification_metrics(y_true, y_pred)
        assert "matthews_corrcoef" in report.skipped

    def test_mcc_ypred_constant_ytrue_varies(self) -> None:
        """`y_true` has both classes (balanced accuracy IS computed) but
        `y_pred` is constant -- MCC is still undefined (the Pearson
        correlation between the two label vectors needs variance in
        BOTH), even though `sklearn.metrics.matthews_corrcoef` itself
        does not warn for this specific shape (verified empirically) and
        would otherwise silently return `0.0` by internal convention."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])
        report = compute_classification_metrics(y_true, y_pred)
        assert "matthews_corrcoef" in report.skipped
        assert "balanced_accuracy" in report.values

    def test_proba_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_classification_metrics(np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([0.1, 0.9]))


@pytest.mark.filterwarnings("error")
class TestComputeRegressionMetrics:
    def test_all_metrics_present_for_normal_input(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        report = compute_regression_metrics(y_true, y_pred)
        assert set(report.values) == set(REGRESSION_METRIC_NAMES)
        assert report.skipped == {}

    def test_perfect_predictions_score_maximally(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        report = compute_regression_metrics(y_true, y_true.copy())
        assert report.values["mae"] == 0.0
        assert report.values["rmse"] == 0.0
        assert report.values["r2"] == 1.0

    def test_constant_true_labels_skips_r2_but_computes_the_rest(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 4.0, 6.0])
        report = compute_regression_metrics(y_true, y_pred)
        assert "r2" in report.skipped
        assert "distinct values" in report.skipped["r2"]
        assert set(report.values) == set(REGRESSION_METRIC_NAMES) - {"r2"}

    def test_zero_in_true_labels_does_not_raise_mape_stays_finite(self) -> None:
        """MAPE is ill-defined at y_true=0 (division by zero) -- sklearn
        clips the denominator rather than raising; this module must not
        crash, and the resulting (large but finite) value must still be
        JSON-serializable (see `MetricComputationReport.__post_init__`'s
        `validate_json_primitive_mapping` finiteness check)."""
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        report = compute_regression_metrics(y_true, y_pred)
        assert np.isfinite(report.values["mape"])

    def test_single_sample_skips_r2(self) -> None:
        report = compute_regression_metrics(np.array([1.0]), np.array([1.5]))
        assert "r2" in report.skipped
        assert set(report.values) == set(REGRESSION_METRIC_NAMES) - {"r2"}

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_regression_metrics(np.array([]), np.array([]))

    def test_nan_predictions_raise(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0, np.nan]))

    def test_infinite_predictions_raise(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0, np.inf]))

    def test_infinite_true_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compute_regression_metrics(np.array([1.0, -np.inf]), np.array([1.0, 2.0]))


class TestComputeMetricsDispatch:
    def test_regression_dispatch(self) -> None:
        report = compute_metrics(ObjectiveType.REGRESSION, np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))
        assert set(report.values) <= set(REGRESSION_METRIC_NAMES)

    def test_binary_classification_dispatch(self) -> None:
        report = compute_metrics(
            ObjectiveType.BINARY_CLASSIFICATION, np.array([0, 1, 0]), np.array([0, 1, 1]), np.array([0.2, 0.8, 0.6]),
        )
        assert "roc_auc" in report.values

    def test_regression_with_proba_raises(self) -> None:
        with pytest.raises(ValueError, match="y_proba"):
            compute_metrics(ObjectiveType.REGRESSION, np.array([1.0]), np.array([1.0]), np.array([0.5]))

    def test_multiclass_not_yet_supported_raises(self) -> None:
        with pytest.raises(ValueError, match="does not yet support"):
            compute_metrics(ObjectiveType.MULTICLASS_CLASSIFICATION, np.array([0, 1, 2]), np.array([0, 1, 2]))


class TestAggregateMetricValues:
    def test_basic_aggregation(self) -> None:
        agg = aggregate_metric_values([1.0, 2.0, 3.0, 4.0, 5.0])
        assert agg.mean == 3.0
        assert agg.median == 3.0
        assert agg.min == 1.0
        assert agg.max == 5.0
        assert agg.n == 5
        assert agg.std > 0

    def test_single_value_has_zero_std_never_nan(self) -> None:
        agg = aggregate_metric_values([0.7])
        assert agg.std == 0.0
        assert agg.mean == 0.7
        assert agg.n == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_metric_values([])

    def test_round_trip(self) -> None:
        from quant_platform.ml.metrics import MetricAggregate

        agg = aggregate_metric_values([1.0, 2.0, 3.0])
        restored = MetricAggregate.from_json_dict(agg.to_json_dict())
        assert restored == agg


class TestAggregateFoldMetrics:
    def test_aggregates_across_folds_sharing_a_metric(self) -> None:
        fold_metrics = [{"accuracy": 0.8, "roc_auc": 0.9}, {"accuracy": 0.6}, {"accuracy": 0.7, "roc_auc": 0.95}]
        aggregated = aggregate_fold_metrics(fold_metrics)
        assert aggregated["accuracy"].n == 3
        assert aggregated["roc_auc"].n == 2  # only 2 of 3 folds reported it -- never padded with a zero
        assert aggregated["accuracy"].mean == pytest.approx(0.7)

    def test_empty_fold_list_yields_empty_aggregate(self) -> None:
        assert aggregate_fold_metrics([]) == {}

    def test_fold_with_no_metrics_at_all_is_harmless(self) -> None:
        """A FAILED fold contributes an EMPTY `metrics` mapping (see
        `execution.results.FoldResult`) -- must never crash aggregation,
        and must never be silently counted as a zero for any metric."""
        aggregated = aggregate_fold_metrics([{"accuracy": 0.8}, {}, {"accuracy": 0.6}])
        assert aggregated["accuracy"].n == 2
        assert aggregated["accuracy"].mean == pytest.approx(0.7)

    def test_non_numeric_or_bool_values_are_ignored(self) -> None:
        """`FoldResult.metrics` is typed as `Mapping[str, JsonPrimitive]`
        -- a non-numeric (e.g. accidentally a string) or boolean value
        must never silently corrupt a numeric aggregate."""
        aggregated = aggregate_fold_metrics([{"accuracy": 0.8, "note": "ok", "flag": True}, {"accuracy": 0.6}])
        assert "note" not in aggregated
        assert "flag" not in aggregated
        assert aggregated["accuracy"].n == 2
