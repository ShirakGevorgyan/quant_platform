"""Calibration metrics (Milestone 4E, Section 10) -- log loss, Brier
score, expected/maximum calibration error, and calibration slope/
intercept, computed over a `(probabilities, labels)` pair.

FAIL-DEFINED, NOT FAIL-CLOSED, FOR UNDEFINED METRICS
--------------------------------------------------------------------------
Mirrors `ml.metrics`'s own established convention exactly (see that
module's docstring): a metric that is mathematically undefined for a
particular input (calibration slope/intercept when every probability is
identical or only one label class is present) is OMITTED from `values`
and explained in `skipped`, never silently stored as `NaN` or a
fabricated `0.0`.

CLIPPED VS. UNCLIPPED PROBABILITIES -- CALLER'S EXPLICIT CHOICE
--------------------------------------------------------------------------
No function in this module clips its `probabilities` input itself --
Section 9 requires "metrics document whether they consume unclipped or
clipped probabilities", which this module satisfies by taking that
decision away from itself entirely: the CALLER (`calibration.fitting`/
`calibration.runner`) decides which variant (raw or `ProbabilityClippingPolicy`
-clipped) to pass, and `CalibrationMetricReport`/persisted artifacts
record which one was used. `calibration_slope_intercept` is the one
metric mathematically sensitive to this choice (`logit(0)`/`logit(1)`
are undefined) -- passed exact 0/1 probabilities, it reports itself
UNAVAILABLE with an explicit reason rather than silently clipping
internally.
"""

from __future__ import annotations

import numpy as np
import sklearn.metrics as skm  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from quant_platform.calibration.diagnostics import compute_reliability_bins
from quant_platform.calibration.specs import ReliabilityBinningSpec
from quant_platform.core.exceptions import CalibrationDataError
from quant_platform.ml.metrics import MetricComputationReport

CALIBRATION_METRIC_NAMES: tuple[str, ...] = (
    "log_loss", "brier_score", "expected_calibration_error", "maximum_calibration_error",
    "calibration_slope", "calibration_intercept", "sharpness", "resolution",
)


def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise CalibrationDataError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    return arr


def _validate_inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _as_1d_float_array(probabilities, name="probabilities")
    labs = _as_1d_float_array(labels, name="labels")
    if probs.shape != labs.shape:
        raise CalibrationDataError(f"probabilities shape {probs.shape} does not match labels shape {labs.shape}")
    if len(probs) == 0:
        raise CalibrationDataError("probabilities/labels must not be empty")
    if not np.all(np.isfinite(probs)):
        raise CalibrationDataError("probabilities contains non-finite value(s)")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise CalibrationDataError("probabilities contains value(s) outside [0, 1]")
    if not np.all(np.isfinite(labs)):
        raise CalibrationDataError("labels contains non-finite value(s)")
    if not np.all(np.isin(labs, (0.0, 1.0))):
        raise CalibrationDataError("labels must be binary (0.0/1.0) valued -- outside the binary-classification task contract")
    return probs, labs


def compute_calibration_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, binning_spec: ReliabilityBinningSpec,
) -> MetricComputationReport:
    """Computes every metric in `CALIBRATION_METRIC_NAMES` for one
    `(probabilities, labels)` pair, skipping (with an explicit reason)
    whichever are undefined for this particular input."""
    probs, labs = _validate_inputs(probabilities, labels)
    n_true_classes = len(set(labs.tolist()))

    values: dict[str, float] = {}
    skipped: dict[str, str] = {}

    if n_true_classes < 2:
        skipped["log_loss"] = f"labels contains only {n_true_classes} distinct class(es) -- log loss requires both classes to be meaningful"
    else:
        values["log_loss"] = float(skm.log_loss(labs, probs, labels=[0.0, 1.0]))

    values["brier_score"] = float(skm.brier_score_loss(labs, probs))

    reliability = compute_reliability_bins(probs, labs, spec=binning_spec)
    non_empty = [b for b in reliability.bins if not b.is_empty]
    if not non_empty:
        skipped["expected_calibration_error"] = "no non-empty reliability bin was produced"
        skipped["maximum_calibration_error"] = "no non-empty reliability bin was produced"
    else:
        total = sum(b.sample_count for b in non_empty)
        assert reliability.n_samples == total
        ece = sum((b.sample_count / total) * (b.calibration_gap or 0.0) for b in non_empty)
        mce = max(b.calibration_gap or 0.0 for b in non_empty)
        values["expected_calibration_error"] = float(ece)
        values["maximum_calibration_error"] = float(mce)

    slope, intercept, slope_skip_reason = _calibration_slope_intercept(probs, labs)
    if slope_skip_reason is not None:
        skipped["calibration_slope"] = slope_skip_reason
        skipped["calibration_intercept"] = slope_skip_reason
    else:
        assert slope is not None and intercept is not None
        values["calibration_slope"] = slope
        values["calibration_intercept"] = intercept

    values["sharpness"] = float(np.var(probs))
    overall_rate = float(labs.mean())
    resolution = 0.0
    for b in non_empty:
        assert b.empirical_positive_rate is not None  # guaranteed by ReliabilityBin.__post_init__ for a non-empty bin
        resolution += (b.sample_count / reliability.n_samples) * (b.empirical_positive_rate - overall_rate) ** 2
    values["resolution"] = float(resolution)

    return MetricComputationReport(values=values, skipped=skipped)


def _calibration_slope_intercept(probs: np.ndarray, labs: np.ndarray) -> tuple[float | None, float | None, str | None]:
    """Cox calibration regression: fit `y ~ intercept + slope * logit(p)`
    via a 1-feature logistic regression. `slope == 1, intercept == 0` is
    perfect calibration; `slope < 1` indicates predictions are too
    extreme (overconfident), `slope > 1` too conservative."""
    n_true_classes = len(set(labs.tolist()))
    if n_true_classes < 2:
        return None, None, f"labels contains only {n_true_classes} distinct class(es) -- calibration slope/intercept requires both"
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        return None, None, "probabilities contains value(s) at the [0, 1] boundary -- logit(p) is undefined there; supply clipped probabilities to compute calibration slope/intercept"
    if float(np.ptp(probs)) == 0.0:
        return None, None, "probabilities are all identical -- calibration slope is undefined with zero predictor variance"
    logit_p = np.log(probs / (1.0 - probs)).reshape(-1, 1)
    # `C=np.inf` (rather than the deprecated `penalty=None`) is sklearn's
    # own documented replacement for "no regularization" -- an
    # effectively-infinite inverse regularization strength makes the
    # penalty term a no-op, giving the same plain-MLE fit this
    # calibration-slope diagnostic requires.
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit_p, labs)
    return float(model.coef_[0][0]), float(model.intercept_[0]), None


__all__ = ["CALIBRATION_METRIC_NAMES", "compute_calibration_metrics"]
