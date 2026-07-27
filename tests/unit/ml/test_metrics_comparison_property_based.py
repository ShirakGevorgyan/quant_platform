"""Property-based tests for Milestone 4C's `ml.metrics`/`ml.comparison`
most safety-critical invariants: computed metric values are never NaN/
inf and always fall within their mathematically-guaranteed bounds; every
declared metric name is accounted for in EXACTLY ONE of `values`/
`skipped`, never both, never neither; fold aggregation's `n` always
matches the actual number of folds that reported a given metric; the
comparison engine never fabricates a statistical verdict from too few
paired folds or an all-identical sample; `outperforms_all_baselines` is
`True` if and only if every baseline comparison for that metric is
`CANDIDATE_SIGNIFICANTLY_BETTER`; `rank_candidates` always orders by the
primary metric's own direction (higher-is-better vs. lower-is-better)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quant_platform.ml.comparison import (
    BaselineComparisonReport,
    ComparisonOutcome,
    MetricComparison,
    ModelComparisonReport,
    ModelFoldMetrics,
    compare_to_baseline,
    rank_candidates,
)
from quant_platform.ml.metrics import (
    CLASSIFICATION_METRIC_NAMES,
    REGRESSION_METRIC_NAMES,
    aggregate_fold_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
)

_UNIT_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_BOUNDED_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_EPS = 1e-9


# --------------------------------------------------------------------------
# ml.metrics: compute_classification_metrics
# --------------------------------------------------------------------------
def _force_variance(values: list[float]) -> None:
    """Mutates `values` in place so both `0.0` and `1.0` are present --
    shared by every composite below that needs to force variance in
    either `y_true` or `y_pred` specifically (never both accidentally
    forced into the SAME two positions when both are needed)."""
    if len(set(values)) < 2:
        values[0], values[1] = 0.0, 1.0


@st.composite
def _classification_case(draw, *, force_both_classes: bool):
    n = draw(st.integers(min_value=2 if force_both_classes else 1, max_value=60))
    y_true = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    if force_both_classes:
        _force_variance(y_true)
    y_pred = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    y_proba = draw(st.lists(_UNIT_FLOAT, min_size=n, max_size=n))
    return y_true, y_pred, y_proba


@st.composite
def _classification_case_both_sides_vary(draw):
    """Both `y_true` AND `y_pred` are forced to have variance -- the only
    shape under which NOTHING in `CLASSIFICATION_METRIC_NAMES` is ever
    skipped (balanced accuracy needs `y_true` variance; MCC needs BOTH
    `y_true` and `y_pred` variance, being the Pearson correlation between
    the two label vectors)."""
    n = draw(st.integers(min_value=2, max_value=60))
    y_true = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    _force_variance(y_true)
    y_pred = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    _force_variance(y_pred)
    y_proba = draw(st.lists(_UNIT_FLOAT, min_size=n, max_size=n))
    return y_true, y_pred, y_proba


@st.composite
def _single_class_case(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    class_value = draw(st.sampled_from([0.0, 1.0]))
    y_pred = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    y_proba = draw(st.lists(_UNIT_FLOAT, min_size=n, max_size=n))
    return class_value, n, y_pred, y_proba


@st.composite
def _constant_pred_both_true_classes_case(draw):
    """`y_true` has both classes (so balanced accuracy/ROC AUC/PR AUC are
    all defined), but `y_pred` is forced CONSTANT -- MCC specifically
    must still be skipped (undefined: zero variance in one of the two
    vectors a correlation coefficient is computed between)."""
    n = draw(st.integers(min_value=2, max_value=60))
    y_true = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    _force_variance(y_true)
    constant = draw(st.sampled_from([0.0, 1.0]))
    y_pred = [constant] * n
    y_proba = draw(st.lists(_UNIT_FLOAT, min_size=n, max_size=n))
    return y_true, y_pred, y_proba


@pytest.mark.filterwarnings("error")
@given(_classification_case_both_sides_vary())
@settings(max_examples=200)
def test_classification_metrics_with_proba_and_variance_on_both_sides_never_skips_and_stays_in_bounds(case) -> None:
    y_true, y_pred, y_proba = case
    report = compute_classification_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba))

    assert set(report.values) == set(CLASSIFICATION_METRIC_NAMES)
    assert report.skipped == {}
    for name, value in report.values.items():
        assert not math.isnan(value), name
        assert not math.isinf(value), name
    for name in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"):
        assert -_EPS <= report.values[name] <= 1.0 + _EPS, name
    assert -1.0 - _EPS <= report.values["matthews_corrcoef"] <= 1.0 + _EPS


@pytest.mark.filterwarnings("error")
@given(_classification_case(force_both_classes=False), st.booleans())
@settings(max_examples=200)
def test_classification_metrics_every_declared_metric_is_in_exactly_one_of_values_or_skipped(case, supply_proba: bool) -> None:
    y_true, y_pred, y_proba = case
    report = compute_classification_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba) if supply_proba else None)

    assert set(report.values) | set(report.skipped) == set(CLASSIFICATION_METRIC_NAMES)
    assert set(report.values) & set(report.skipped) == set()
    for name, value in report.values.items():
        assert not math.isnan(value), name
        assert not math.isinf(value), name


@pytest.mark.filterwarnings("error")
@given(_single_class_case())
@settings(max_examples=100)
def test_classification_metrics_single_class_ytrue_always_skips_class_balance_dependent_metrics(case) -> None:
    """A single-class `y_true` makes ROC AUC/PR AUC/balanced accuracy/MCC
    ALL undefined -- not just the AUC pair."""
    class_value, n, y_pred, y_proba = case
    y_true = np.full(n, class_value, dtype="float64")
    report = compute_classification_metrics(y_true, np.asarray(y_pred), np.asarray(y_proba))

    for name in ("roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef"):
        assert name in report.skipped, name
        assert name not in report.values, name


@pytest.mark.filterwarnings("error")
@given(_constant_pred_both_true_classes_case())
@settings(max_examples=100)
def test_classification_metrics_constant_ypred_always_skips_mcc_even_with_both_true_classes(case) -> None:
    """MCC is skipped whenever `y_pred` is constant, even though `y_true`
    has both classes (so balanced accuracy/ROC AUC/PR AUC remain
    defined and computed) -- MCC alone is sensitive to variance in
    EITHER vector, being the Pearson correlation between them."""
    y_true, y_pred, y_proba = case
    report = compute_classification_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba))

    assert "matthews_corrcoef" in report.skipped
    assert "matthews_corrcoef" not in report.values
    assert "balanced_accuracy" in report.values
    assert "roc_auc" in report.values
    assert "pr_auc" in report.values


# --------------------------------------------------------------------------
# ml.metrics: compute_regression_metrics
# --------------------------------------------------------------------------
@st.composite
def _regression_case(draw, *, min_n: int = 1):
    n = draw(st.integers(min_value=min_n, max_value=60))
    y_true = draw(st.lists(_BOUNDED_FLOAT, min_size=n, max_size=n))
    y_pred = draw(st.lists(_BOUNDED_FLOAT, min_size=n, max_size=n))
    return y_true, y_pred


@st.composite
def _constant_ytrue_regression_case(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    constant = draw(_BOUNDED_FLOAT)
    y_pred = draw(st.lists(_BOUNDED_FLOAT, min_size=n, max_size=n))
    return constant, n, y_pred


@given(_regression_case())
@settings(max_examples=200)
def test_regression_metrics_never_nan_or_inf_and_every_metric_accounted_for(case) -> None:
    y_true, y_pred = case
    report = compute_regression_metrics(np.asarray(y_true), np.asarray(y_pred))

    assert set(report.values) | set(report.skipped) == set(REGRESSION_METRIC_NAMES)
    assert set(report.values) & set(report.skipped) == set()
    for name, value in report.values.items():
        assert not math.isnan(value), name
        assert not math.isinf(value), name
    assert report.values["mae"] >= -_EPS
    assert report.values["rmse"] >= -_EPS
    assert report.values["mape"] >= -_EPS
    if "r2" in report.values:
        assert report.values["r2"] <= 1.0 + _EPS


@given(_constant_ytrue_regression_case())
@settings(max_examples=100)
def test_regression_metrics_constant_ytrue_always_skips_r2(case) -> None:
    constant, n, y_pred = case
    y_true = np.full(n, constant, dtype="float64")
    report = compute_regression_metrics(y_true, np.asarray(y_pred))

    assert "r2" in report.skipped
    assert "r2" not in report.values


# --------------------------------------------------------------------------
# ml.metrics: aggregate_fold_metrics
# --------------------------------------------------------------------------
_fold_metrics_dict = st.dictionaries(
    st.sampled_from(["accuracy", "f1", "mae", "custom_metric"]), _BOUNDED_FLOAT, max_size=4,
)
_fold_metrics_list = st.lists(_fold_metrics_dict, min_size=1, max_size=12)


@given(_fold_metrics_list)
@settings(max_examples=200)
def test_aggregate_fold_metrics_n_matches_actual_reporting_fold_count(fold_metrics_list: list[dict[str, float]]) -> None:
    result = aggregate_fold_metrics(fold_metrics_list)
    for metric_name, aggregate in result.items():
        expected_n = sum(1 for fm in fold_metrics_list if metric_name in fm)
        assert aggregate.n == expected_n
        assert aggregate.min - _EPS <= aggregate.mean <= aggregate.max + _EPS
        assert aggregate.min - _EPS <= aggregate.median <= aggregate.max + _EPS
        assert aggregate.std >= 0.0
        if aggregate.n == 1:
            assert aggregate.std == 0.0


# --------------------------------------------------------------------------
# ml.comparison: compare_to_baseline's honesty about small/degenerate samples
# --------------------------------------------------------------------------
@st.composite
def _insufficient_fold_case(draw):
    n_folds = draw(st.integers(min_value=1, max_value=4))  # always < _MIN_FOLDS_FOR_SIGNIFICANCE
    values_a = draw(st.lists(_UNIT_FLOAT, min_size=n_folds, max_size=n_folds))
    values_b = draw(st.lists(_UNIT_FLOAT, min_size=n_folds, max_size=n_folds))
    candidate = ModelFoldMetrics(model_name="candidate", per_fold_metrics=tuple({"accuracy": v} for v in values_a))
    baseline = ModelFoldMetrics(model_name="baseline", per_fold_metrics=tuple({"accuracy": v} for v in values_b))
    return candidate, baseline


@st.composite
def _identical_values_case(draw):
    n_folds = draw(st.integers(min_value=5, max_value=15))
    values = draw(st.lists(_UNIT_FLOAT, min_size=n_folds, max_size=n_folds))
    candidate = ModelFoldMetrics(model_name="candidate", per_fold_metrics=tuple({"accuracy": v} for v in values))
    baseline = ModelFoldMetrics(model_name="baseline", per_fold_metrics=tuple({"accuracy": v} for v in values))
    return candidate, baseline


@st.composite
def _sufficient_nonidentical_case(draw):
    n_folds = draw(st.integers(min_value=5, max_value=15))
    values_a = draw(st.lists(_UNIT_FLOAT, min_size=n_folds, max_size=n_folds))
    values_b = draw(st.lists(_UNIT_FLOAT, min_size=n_folds, max_size=n_folds))
    assume(any(a != b for a, b in zip(values_a, values_b, strict=True)))
    candidate = ModelFoldMetrics(model_name="candidate", per_fold_metrics=tuple({"accuracy": v} for v in values_a))
    baseline = ModelFoldMetrics(model_name="baseline", per_fold_metrics=tuple({"accuracy": v} for v in values_b))
    return candidate, baseline


@given(_insufficient_fold_case())
@settings(max_examples=150)
def test_fewer_than_minimum_paired_folds_always_skips_with_no_p_value(case) -> None:
    candidate, baseline = case
    comparison = compare_to_baseline(candidate, baseline).comparison_for("accuracy")
    assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
    assert comparison.p_value is None
    assert comparison.reason


@given(_identical_values_case())
@settings(max_examples=100)
def test_identical_paired_values_always_skips_rather_than_fabricating_a_verdict(case) -> None:
    candidate, baseline = case
    comparison = compare_to_baseline(candidate, baseline).comparison_for("accuracy")
    assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
    assert comparison.p_value is None


@given(_sufficient_nonidentical_case())
@settings(max_examples=150)
def test_sufficient_nonidentical_folds_always_produce_a_real_verdict_with_a_valid_p_value(case) -> None:
    candidate, baseline = case
    comparison = compare_to_baseline(candidate, baseline).comparison_for("accuracy")
    assert comparison.outcome in (
        ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER,
        ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_WORSE,
        ComparisonOutcome.NO_SIGNIFICANT_DIFFERENCE,
    )
    assert comparison.p_value is not None
    assert 0.0 <= comparison.p_value <= 1.0


@given(_sufficient_nonidentical_case())
@settings(max_examples=50)
def test_compare_to_baseline_is_deterministic(case) -> None:
    candidate, baseline = case
    first = compare_to_baseline(candidate, baseline).comparison_for("accuracy")
    second = compare_to_baseline(candidate, baseline).comparison_for("accuracy")
    assert first.outcome is second.outcome
    assert first.p_value == second.p_value


# --------------------------------------------------------------------------
# ml.comparison: the "never declare success" gate itself
# --------------------------------------------------------------------------
_outcome_strategy = st.sampled_from(list(ComparisonOutcome))


@given(st.lists(_outcome_strategy, min_size=0, max_size=6))
@settings(max_examples=200)
def test_outperforms_all_baselines_is_true_iff_every_outcome_is_significantly_better(outcomes: list[ComparisonOutcome]) -> None:
    baseline_reports = tuple(
        BaselineComparisonReport(
            baseline_name=f"baseline_{i}",
            metric_comparisons=(
                MetricComparison(
                    metric_name="accuracy", candidate_aggregate=None, baseline_aggregate=None,
                    paired_n=0, p_value=None, outcome=outcome,
                ),
            ),
        )
        for i, outcome in enumerate(outcomes)
    )
    report = ModelComparisonReport(candidate_name="candidate", baseline_reports=baseline_reports)
    expected = bool(outcomes) and all(o is ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER for o in outcomes)
    assert report.outperforms_all_baselines("accuracy") == expected


@given(st.lists(_outcome_strategy, min_size=1, max_size=6))
@settings(max_examples=100)
def test_outperforms_all_baselines_false_when_metric_missing_from_any_baseline(outcomes: list[ComparisonOutcome]) -> None:
    """Even if every PRESENT comparison is significantly-better, a
    baseline that never reported the primary metric at all must still
    make the gate return `False` -- a missing comparison is never
    silently treated as a pass."""
    baseline_reports = [
        BaselineComparisonReport(
            baseline_name=f"baseline_{i}",
            metric_comparisons=(
                MetricComparison(
                    metric_name="accuracy", candidate_aggregate=None, baseline_aggregate=None,
                    paired_n=0, p_value=None, outcome=ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER,
                ),
            ),
        )
        for i in range(len(outcomes))
    ]
    # One baseline reports a DIFFERENT metric only -- "accuracy" is absent.
    baseline_reports.append(BaselineComparisonReport(
        baseline_name="baseline_missing_metric",
        metric_comparisons=(
            MetricComparison(
                metric_name="rmse", candidate_aggregate=None, baseline_aggregate=None,
                paired_n=0, p_value=None, outcome=ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER,
            ),
        ),
    ))
    report = ModelComparisonReport(candidate_name="candidate", baseline_reports=tuple(baseline_reports))
    assert report.outperforms_all_baselines("accuracy") is False


# --------------------------------------------------------------------------
# ml.comparison: rank_candidates orders by the metric's own direction
# --------------------------------------------------------------------------
@st.composite
def _distinct_candidates_case(draw):
    n = draw(st.integers(min_value=2, max_value=5))
    values = draw(st.lists(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n, unique=True))
    return tuple(
        ModelFoldMetrics(model_name=f"model_{i}", per_fold_metrics=({"accuracy": v, "mae": v},))
        for i, v in enumerate(values)
    )


@given(_distinct_candidates_case())
@settings(max_examples=100)
def test_rank_candidates_orders_descending_for_a_higher_is_better_metric(candidates) -> None:
    report = rank_candidates(candidates, baselines=[], primary_metric="accuracy")
    means = [m.primary_metric_mean for m in report.ranked_models]
    assert means == sorted(means, reverse=True)
    assert len(report.ranked_models) == len(candidates)
    assert {m.model_name for m in report.ranked_models} == {c.model_name for c in candidates}


@given(_distinct_candidates_case())
@settings(max_examples=100)
def test_rank_candidates_orders_ascending_for_a_lower_is_better_metric(candidates) -> None:
    report = rank_candidates(candidates, baselines=[], primary_metric="mae")
    means = [m.primary_metric_mean for m in report.ranked_models]
    assert means == sorted(means)


@given(_distinct_candidates_case())
@settings(max_examples=50)
def test_rank_candidates_with_no_baselines_never_claims_outperformance(candidates) -> None:
    """`compare_to_baselines(candidate, [])` has nothing to outperform --
    `outperforms_all_baselines` must be `False` for every ranked model,
    never vacuously `True` just because there was nothing to fail against."""
    report = rank_candidates(candidates, baselines=[], primary_metric="accuracy")
    assert all(m.outperforms_all_baselines is False for m in report.ranked_models)
