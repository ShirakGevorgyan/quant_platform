"""Model-performance metrics (Milestone 4C, Sections "METRICS"/"FOLD
AGGREGATION") -- the first place in this codebase that computes a REAL
performance score from predictions vs. truth. Milestones 4A/4B
deliberately left this as an empty placeholder (`FoldResult.metrics`, `
execution.executor`'s module docstring) precisely because choosing a
metric definition is itself a modeling decision, out of scope until a
real model existed to evaluate -- this module is that decision, made
explicitly and narrowly: standard, well-known metric definitions only,
computed via `sklearn.metrics` (a well-tested, widely-audited
implementation) rather than hand-rolled formulas, wrapped just enough to
make every edge case this module's own docstrings describe behave
predictably rather than raising or silently producing `NaN`/`inf` that a
caller did not ask for.

FAIL-DEFINED, NOT FAIL-CLOSED, FOR UNDEFINED METRICS
--------------------------------------------------------------------------
Some metrics are mathematically UNDEFINED for certain (rare but real)
inputs -- ROC AUC/PR AUC/balanced accuracy when `y_true` has only one
class present; MCC when `y_true` OR `y_pred` has only one class present
(MCC is the Pearson correlation coefficient between the two label
vectors, undefined when either has zero variance); R^2 when `y_true` has
zero variance. Rather than raising (which would abort an entire fold for
a metric that is merely inapplicable to it), silently returning `0.0`/
`NaN` unlabeled, OR letting `sklearn` compute a same-shaped but degenerate
value while emitting a `UserWarning` about it, every such metric is
OMITTED from the returned mapping and the omission is explained in the
returned `MetricComputationReport.skipped` mapping (metric name ->
reason) -- a caller can always tell the difference between "this metric
is 0.0" and "this metric could not be computed for this fold". Fold
aggregation (`aggregate_fold_metrics`) then aggregates only over the
folds where a given metric was actually computed, never treating a skip
as a zero.

LABEL DOMAIN, BINARY CLASSIFICATION
--------------------------------------------------------------------------
For `BINARY_CLASSIFICATION` (this module's only classification objective
today): allowed true labels and allowed predicted hard labels are both
exactly `{0.0, 1.0}` (enforced upstream by `ml.model_validation.
validate_training_data`'s `non_binary_label_values` check, and by every
`ml.model_zoo` model's `predict()` contract); the POSITIVE class is
always `1.0`, the NEGATIVE class always `0.0`. A fold whose `y_true`
happens to contain only ONE of these two classes is a LEGITIMATE,
EXPECTED outcome of time-series walk-forward evaluation (a short or
unlucky test window can genuinely see only one class) -- it is never
rejected outright, but it does make some metrics below undefined for
THAT fold specifically (see `_ClassificationLabelDomain`). This is
distinct from, and never conflated with, a class being unsupported by
the model/objective overall (impossible here: `ObjectiveType.
BINARY_CLASSIFICATION` structurally has no third class to support). A
model may expose `class_labels` in either order (`(0, 1)` or `(1, 0)`)
-- callers extract the positive-class probability column via `class_
labels.index(1)`, never by assuming a fixed column position.

WHY NO CUSTOM THRESHOLD LOGIC HERE
--------------------------------------------------------------------------
Precision/recall/F1/MCC/balanced-accuracy are all computed from HARD
class predictions (`Predictor.predict()`'s output, per this milestone's
convention -- see `ml.model_zoo`'s module docstring) -- this module never
re-thresholds a probability score itself (e.g. "predict positive if
proba > 0.5"); doing so would be a calibration/threshold-selection
decision, explicitly out of scope (see the milestone's own "DO NOT
IMPLEMENT: Calibration").
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import sklearn.metrics as skm  # type: ignore[import-untyped]

from quant_platform.ml.models import JsonPrimitive, ObjectiveType, validate_json_primitive_mapping

CLASSIFICATION_METRIC_NAMES: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef",
)
REGRESSION_METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "mape", "r2")


@dataclass(frozen=True, slots=True)
class MetricComputationReport:
    """`values` is ready to hand straight to `FoldResult(metrics=...)` (a
    plain `Mapping[str, JsonPrimitive]`, never a bespoke type -- Milestone
    4B's `FoldResult.metrics` field shape is unchanged). `skipped` records
    WHY a metric this objective normally reports is absent from `values`
    for this particular fold -- never silently missing with no
    explanation."""

    values: Mapping[str, float]
    skipped: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_json_primitive_mapping(self.values, field_name="MetricComputationReport.values")


def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    return arr


def _reject_non_finite(arr: np.ndarray, *, name: str) -> None:
    """Rejects NaN AND +/-infinity, with the SAME clear, array-named
    message either way. Every `sklearn.metrics` function called below
    would ALSO reject infinity internally (its own input validation), but
    with a generic, unnamed message ("Input contains infinity...") that
    does not say whether `y_true`/`y_pred`/`y_proba` was the culprit --
    this explicit, named check runs first so a caller always gets the
    same actionable diagnosis this module already gives for NaN, rather
    than relying on incidental behavior that could differ across the
    several different `sklearn.metrics` functions this module calls."""
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN value(s) -- metrics cannot be computed over undefined predictions/labels")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite value(s) -- metrics cannot be computed over an unbounded prediction/label")


@dataclass(frozen=True, slots=True)
class _ClassificationLabelDomain:
    """Label-domain facts computed ONCE per `compute_classification_
    metrics` call and reused for every degenerate-case decision below --
    never re-deriving `np.unique(...)` per metric, and never left to
    `sklearn` to discover (and warn about) internally."""

    true_classes: frozenset[float]
    pred_classes: frozenset[float]

    @property
    def n_true_classes(self) -> int:
        return len(self.true_classes)

    @property
    def n_pred_classes(self) -> int:
        return len(self.pred_classes)


def _classification_label_domain(true_arr: np.ndarray, pred_arr: np.ndarray) -> _ClassificationLabelDomain:
    return _ClassificationLabelDomain(true_classes=frozenset(true_arr.tolist()), pred_classes=frozenset(pred_arr.tolist()))


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None,
) -> MetricComputationReport:
    """`y_true`/`y_pred` are HARD binary labels (`{0, 1}`, any dtype
    coercible to that). `y_proba`, if given, is the predicted probability
    of the POSITIVE (`1`) class only -- a 1-D array, not the full
    `predict_proba` 2-column matrix (see `ml.model_zoo`'s convention:
    `ProbabilisticPredictor.class_labels` establishes column order; the
    caller extracts the positive-class column before calling this).

    Every degenerate case below is decided BEFORE calling `sklearn` --
    never by calling it first and reacting to whatever `UserWarning` it
    happens to emit. A missing class in THIS fold's `y_true` is expected
    and legitimate (see module docstring); it changes which metrics are
    DEFINED for this fold, never whether the fold itself is accepted."""
    true_arr = _as_1d_float_array(y_true, name="y_true")
    pred_arr = _as_1d_float_array(y_pred, name="y_pred")
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"y_true shape {true_arr.shape} does not match y_pred shape {pred_arr.shape}")
    if len(true_arr) == 0:
        raise ValueError("y_true/y_pred must not be empty")
    _reject_non_finite(true_arr, name="y_true")
    _reject_non_finite(pred_arr, name="y_pred")

    domain = _classification_label_domain(true_arr, pred_arr)

    # Accuracy/precision/recall/F1 are always defined for any non-empty,
    # binary-valued input -- `zero_division=0.0` makes precision/recall/F1
    # explicit (never a warning) for the "predicted positive never
    # occurs"/"actual positive never occurs" sub-cases.
    values: dict[str, float] = {
        "accuracy": float(skm.accuracy_score(true_arr, pred_arr)),
        "precision": float(skm.precision_score(true_arr, pred_arr, zero_division=0.0)),
        "recall": float(skm.recall_score(true_arr, pred_arr, zero_division=0.0)),
        "f1": float(skm.f1_score(true_arr, pred_arr, zero_division=0.0)),
    }
    skipped: dict[str, str] = {}

    # Balanced accuracy = mean(per-true-class recall) -- undefined the
    # moment y_true itself has fewer than 2 classes (one of the two
    # per-class recalls has no actual instances to measure against);
    # well-defined for ANY y_pred content otherwise (a constant or
    # partially-wrong y_pred still yields a meaningful, finite average).
    if domain.n_true_classes < 2:
        skipped["balanced_accuracy"] = (
            f"y_true contains only {domain.n_true_classes} distinct class(es) -- balanced accuracy "
            "(the mean of per-true-class recall) is undefined"
        )
    else:
        values["balanced_accuracy"] = float(skm.balanced_accuracy_score(true_arr, pred_arr))

    # MCC is the Pearson correlation coefficient between the two label
    # vectors -- undefined (zero variance in one of the two inputs to a
    # correlation) whenever EITHER y_true OR y_pred is itself constant,
    # not only when both happen to be the identical single class.
    # `sklearn` only WARNS for the narrower "single label across both
    # arrays combined" case and silently returns `0.0` by convention for
    # the wider one (e.g. y_true has both classes but y_pred is
    # constant) -- this module chooses NOT to inherit that narrower
    # convention: any zero-variance side makes MCC equally undefined.
    if domain.n_true_classes < 2 or domain.n_pred_classes < 2:
        reasons = []
        if domain.n_true_classes < 2:
            reasons.append(f"y_true contains only {domain.n_true_classes} distinct class(es)")
        if domain.n_pred_classes < 2:
            reasons.append(f"y_pred contains only {domain.n_pred_classes} distinct class(es)")
        skipped["matthews_corrcoef"] = (
            " and ".join(reasons) + " -- Matthews correlation coefficient (equivalent to the Pearson "
            "correlation between the two label vectors) is undefined when either vector has zero variance"
        )
    else:
        values["matthews_corrcoef"] = float(skm.matthews_corrcoef(true_arr, pred_arr))

    if y_proba is None:
        skipped["roc_auc"] = "no predicted probability column supplied for this fold"
        skipped["pr_auc"] = "no predicted probability column supplied for this fold"
    elif domain.n_true_classes < 2:
        skipped["roc_auc"] = f"y_true contains only {domain.n_true_classes} distinct class(es) -- ROC AUC is undefined"
        skipped["pr_auc"] = f"y_true contains only {domain.n_true_classes} distinct class(es) -- PR AUC is undefined"
    else:
        proba_arr = _as_1d_float_array(y_proba, name="y_proba")
        if proba_arr.shape != true_arr.shape:
            raise ValueError(f"y_proba shape {proba_arr.shape} does not match y_true shape {true_arr.shape}")
        _reject_non_finite(proba_arr, name="y_proba")
        values["roc_auc"] = float(skm.roc_auc_score(true_arr, proba_arr))
        values["pr_auc"] = float(skm.average_precision_score(true_arr, proba_arr))

    return MetricComputationReport(values=values, skipped=skipped)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricComputationReport:
    """MAPE ZERO-TARGET POLICY: `mean_absolute_percentage_error` is
    mathematically undefined at `y_true == 0` (division by zero), but
    (unlike ROC AUC/PR AUC/R^2's genuine undefined-ness elsewhere in this
    module) this is never treated as a skip -- `sklearn` clips the
    denominator to `max(abs(y_true), np.finfo(np.float64).eps)`, giving a
    large but well-defined, finite value instead (still subject to this
    module's own `_reject_non_finite` guard on the INPUTS, and to
    `MetricComputationReport`'s finiteness check on the OUTPUT). A near-
    zero true return is a routine, expected occurrence in this platform's
    domain (a bar with ~0% price change), not a degenerate edge case
    worth suppressing MAPE entirely for."""
    true_arr = _as_1d_float_array(y_true, name="y_true")
    pred_arr = _as_1d_float_array(y_pred, name="y_pred")
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"y_true shape {true_arr.shape} does not match y_pred shape {pred_arr.shape}")
    if len(true_arr) == 0:
        raise ValueError("y_true/y_pred must not be empty")
    _reject_non_finite(true_arr, name="y_true")
    _reject_non_finite(pred_arr, name="y_pred")

    values: dict[str, float] = {
        "mae": float(skm.mean_absolute_error(true_arr, pred_arr)),
        "rmse": float(skm.root_mean_squared_error(true_arr, pred_arr)),
        "mape": float(skm.mean_absolute_percentage_error(true_arr, pred_arr)),
    }
    skipped: dict[str, str] = {}

    if len(true_arr) < 2 or len(np.unique(true_arr)) < 2:
        # Checking distinct-VALUE count rather than a computed `np.var(...)
        # == 0.0` -- summing/dividing even bit-identical float64 values can
        # round to a tiny nonzero variance (e.g. ~1e-26), which would let a
        # genuinely-constant y_true slip past an exact-equality check and
        # produce a meaningless, wildly out-of-range R^2 instead of being
        # honestly skipped.
        skipped["r2"] = "y_true has fewer than 2 samples or fewer than 2 distinct values -- R^2 is undefined"
    else:
        values["r2"] = float(skm.r2_score(true_arr, pred_arr))

    return MetricComputationReport(values=values, skipped=skipped)


def compute_metrics(
    objective: ObjectiveType, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None,
) -> MetricComputationReport:
    """Dispatches on `objective` -- the one entry point `execution.
    executor`'s production `FoldExecutor` calls; never imports a
    model-specific module, keeping metrics computation exactly as
    model-agnostic as the execution engine itself."""
    if objective is ObjectiveType.REGRESSION:
        if y_proba is not None:
            raise ValueError("y_proba must not be supplied for a REGRESSION objective")
        return compute_regression_metrics(y_true, y_pred)
    if objective is ObjectiveType.BINARY_CLASSIFICATION:
        return compute_classification_metrics(y_true, y_pred, y_proba)
    raise ValueError(f"compute_metrics does not yet support objective {objective.value!r}")  # pragma: no cover - MULTICLASS_CLASSIFICATION is declared but no registered model targets it yet


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """`mean`/`std`/`min`/`max`/`median` across every fold that actually
    reported this metric (see module docstring: a fold that SKIPPED a
    metric contributes nothing to its aggregate, never a zero). `std` is
    the sample standard deviation (`ddof=1`); `0.0` when only one fold
    contributed (never `NaN` -- `statistics.stdev` would raise for `n<2`,
    so `n==1` is handled explicitly)."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"MetricAggregate.n must be >= 1, got {self.n}")
        for name in ("mean", "std", "min", "max", "median"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"MetricAggregate.{name} must be finite, got {value!r}")
        if self.std < 0:
            raise ValueError(f"MetricAggregate.std must be >= 0 (it is a standard deviation), got {self.std}")
        if self.min > self.max:
            raise ValueError(f"MetricAggregate.min ({self.min}) must be <= MetricAggregate.max ({self.max})")

    def to_json_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max, "median": self.median, "n": self.n}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MetricAggregate:
        return cls(
            mean=float(str(raw["mean"])), std=float(str(raw["std"])), min=float(str(raw["min"])),
            max=float(str(raw["max"])), median=float(str(raw["median"])), n=int(str(raw["n"])),
        )


def aggregate_metric_values(values: Sequence[float]) -> MetricAggregate:
    if not values:
        raise ValueError("aggregate_metric_values requires at least one value")
    floats = [float(v) for v in values]
    return MetricAggregate(
        mean=statistics.fmean(floats), std=(statistics.stdev(floats) if len(floats) > 1 else 0.0),
        min=min(floats), max=max(floats), median=statistics.median(floats), n=len(floats),
    )


def aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, JsonPrimitive]]) -> dict[str, MetricAggregate]:
    """Aggregates metric values across folds, keyed by metric name. A
    metric name present in only SOME folds' `metrics` mapping (because it
    was skipped -- see `MetricComputationReport`, or because a `FoldResult`
    with `status=FAILED` contributes an empty `metrics` mapping entirely)
    is aggregated only over the folds that actually reported it --
    `MetricAggregate.n` records exactly how many that was, so a caller
    never mistakes "aggregated over 2 of 5 folds" for "aggregated over
    all 5"."""
    by_metric: dict[str, list[float]] = {}
    for metrics in fold_metrics:
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            by_metric.setdefault(name, []).append(float(value))
    return {name: aggregate_metric_values(values) for name, values in by_metric.items()}


__all__ = [
    "CLASSIFICATION_METRIC_NAMES",
    "REGRESSION_METRIC_NAMES",
    "MetricAggregate",
    "MetricComputationReport",
    "aggregate_fold_metrics",
    "aggregate_metric_values",
    "compute_classification_metrics",
    "compute_metrics",
    "compute_regression_metrics",
]
