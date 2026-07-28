"""Decision-threshold selection (Milestone 4E, Sections 12/13) --
deliberately distinct from probability calibration itself (Section 12:
"Prediction calibration and decision thresholding are distinct
concerns"). Every function here operates on already-calibrated
probabilities and training-side (inner OOF) labels only.

THE ONE FIXED BOUNDARY RULE
--------------------------------------------------------------------------
`positive_prediction = probability >= threshold` -- `apply_threshold` is
the SOLE place this platform decides that boundary, called identically
by threshold candidate evaluation, `calibration.runner`'s outer-test
step, `calibration.verification`, and the CLI. No other module
re-implements a `>` vs. `>=` comparison.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sklearn.metrics as skm  # type: ignore[import-untyped]

from quant_platform.calibration.models import ThresholdPolicyKind
from quant_platform.calibration.specs import ThresholdSpec
from quant_platform.core.exceptions import CalibrationDataError, ThresholdSelectionError
from quant_platform.ml.persistence import as_json_list, require_schema_version

THRESHOLD_REPORT_SCHEMA_VERSION = 1
THRESHOLD_STABILITY_SCHEMA_VERSION = 1

_MAXIMIZING_POLICIES = frozenset({
    ThresholdPolicyKind.BALANCED_ACCURACY, ThresholdPolicyKind.F1, ThresholdPolicyKind.MATTHEWS_CORRCOEF,
    ThresholdPolicyKind.YOUDEN_J, ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL, ThresholdPolicyKind.MIN_RECALL_MAX_PRECISION,
})


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """`positive_prediction = probability >= threshold` -- this
    framework's ONE fixed boundary rule (Section 12)."""
    probs = np.asarray(probabilities, dtype="float64")
    if not np.all(np.isfinite(probs)):
        raise CalibrationDataError("apply_threshold: probabilities contains non-finite value(s)")
    if not math.isfinite(threshold):
        raise CalibrationDataError(f"apply_threshold: threshold must be finite, got {threshold!r}")
    return (probs >= threshold).astype("float64")


@dataclass(frozen=True, slots=True)
class ThresholdCandidateResult:
    threshold: float
    objective_value: float | None
    objective_undefined_reason: str | None
    constraint_satisfied: bool
    precision: float | None
    recall: float | None

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise CalibrationDataError(f"ThresholdCandidateResult.threshold must be in [0, 1], got {self.threshold}")
        if self.objective_value is not None and not math.isfinite(self.objective_value):
            raise CalibrationDataError(f"ThresholdCandidateResult.objective_value must be finite if set, got {self.objective_value!r}")
        if (self.objective_value is None) != (self.objective_undefined_reason is not None):
            raise CalibrationDataError(
                "ThresholdCandidateResult: objective_undefined_reason must be set if and only if objective_value is None"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold, "objective_value": self.objective_value,
            "objective_undefined_reason": self.objective_undefined_reason, "constraint_satisfied": self.constraint_satisfied,
            "precision": self.precision, "recall": self.recall,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ThresholdCandidateResult:
        def _opt(key: str) -> float | None:
            return None if raw.get(key) is None else float(str(raw[key]))

        return cls(
            threshold=float(str(raw["threshold"])), objective_value=_opt("objective_value"),
            objective_undefined_reason=(None if raw.get("objective_undefined_reason") is None else str(raw["objective_undefined_reason"])),
            constraint_satisfied=bool(raw["constraint_satisfied"]), precision=_opt("precision"), recall=_opt("recall"),
        )


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    schema_version: int
    policy: ThresholdPolicyKind
    candidates: tuple[ThresholdCandidateResult, ...]
    selected_threshold: float
    selected_objective_value: float | None
    fallback_used: bool
    fallback_reason: str | None
    tie_break_note: str | None

    def __post_init__(self) -> None:
        if not (0.0 <= self.selected_threshold <= 1.0):
            raise CalibrationDataError(f"ThresholdReport.selected_threshold must be in [0, 1], got {self.selected_threshold}")
        if self.selected_objective_value is not None and not math.isfinite(self.selected_objective_value):
            raise CalibrationDataError(f"ThresholdReport.selected_objective_value must be finite if set, got {self.selected_objective_value!r}")
        if self.fallback_used and not self.fallback_reason:
            raise CalibrationDataError("ThresholdReport.fallback_reason is required when fallback_used=True")
        if not self.fallback_used and self.fallback_reason is not None:
            raise CalibrationDataError("ThresholdReport.fallback_reason must be None unless fallback_used=True")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "policy": self.policy.value,
            "candidates": [c.to_json_dict() for c in self.candidates], "selected_threshold": self.selected_threshold,
            "selected_objective_value": self.selected_objective_value, "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason, "tie_break_note": self.tie_break_note,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ThresholdReport:
        require_schema_version(raw, supported=THRESHOLD_REPORT_SCHEMA_VERSION, context="ThresholdReport")
        return cls(
            schema_version=THRESHOLD_REPORT_SCHEMA_VERSION, policy=ThresholdPolicyKind(raw["policy"]),
            candidates=tuple(
                ThresholdCandidateResult.from_json_dict(c) for c in as_json_list(raw["candidates"], field_name="candidates")
            ),
            selected_threshold=float(str(raw["selected_threshold"])),
            selected_objective_value=(None if raw.get("selected_objective_value") is None else float(str(raw["selected_objective_value"]))),
            fallback_used=bool(raw["fallback_used"]),
            fallback_reason=(None if raw.get("fallback_reason") is None else str(raw["fallback_reason"])),
            tie_break_note=(None if raw.get("tie_break_note") is None else str(raw["tie_break_note"])),
        )


def _objective_at(policy: ThresholdPolicyKind, probs: np.ndarray, labels: np.ndarray, threshold: float, spec: ThresholdSpec) -> ThresholdCandidateResult:
    preds = apply_threshold(probs, threshold)
    n_pred_classes = len(set(preds.tolist()))
    n_true_classes = len(set(labels.tolist()))
    precision = float(skm.precision_score(labels, preds, zero_division=0.0))
    recall = float(skm.recall_score(labels, preds, zero_division=0.0))

    if policy is ThresholdPolicyKind.BALANCED_ACCURACY:
        if n_true_classes < 2:
            return ThresholdCandidateResult(threshold, None, "labels contains only one distinct class -- balanced accuracy is undefined", False, precision, recall)
        value = float(skm.balanced_accuracy_score(labels, preds))
        return ThresholdCandidateResult(threshold, value, None, True, precision, recall)

    if policy is ThresholdPolicyKind.F1:
        value = float(skm.f1_score(labels, preds, zero_division=0.0))
        return ThresholdCandidateResult(threshold, value, None, True, precision, recall)

    if policy is ThresholdPolicyKind.MATTHEWS_CORRCOEF:
        if n_true_classes < 2 or n_pred_classes < 2:
            return ThresholdCandidateResult(threshold, None, "labels or predictions have zero variance -- MCC is undefined", False, precision, recall)
        value = float(skm.matthews_corrcoef(labels, preds))
        return ThresholdCandidateResult(threshold, value, None, True, precision, recall)

    if policy is ThresholdPolicyKind.YOUDEN_J:
        if n_true_classes < 2:
            return ThresholdCandidateResult(threshold, None, "labels contains only one distinct class -- Youden's J is undefined", False, precision, recall)
        tn, fp, fn, tp = skm.confusion_matrix(labels, preds, labels=[0.0, 1.0]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return ThresholdCandidateResult(threshold, float(sensitivity + specificity - 1.0), None, True, precision, recall)

    if policy is ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL:
        assert spec.min_precision is not None
        satisfied = precision >= spec.min_precision
        return ThresholdCandidateResult(threshold, (recall if satisfied else None), (None if satisfied else f"precision {precision:.6g} < required minimum {spec.min_precision:.6g}"), satisfied, precision, recall)

    if policy is ThresholdPolicyKind.MIN_RECALL_MAX_PRECISION:
        assert spec.min_recall is not None
        satisfied = recall >= spec.min_recall
        return ThresholdCandidateResult(threshold, (precision if satisfied else None), (None if satisfied else f"recall {recall:.6g} < required minimum {spec.min_recall:.6g}"), satisfied, precision, recall)

    if policy is ThresholdPolicyKind.COST_SENSITIVE:
        assert spec.cost_matrix is not None
        tn, fp, fn, tp = skm.confusion_matrix(labels, preds, labels=[0.0, 1.0]).ravel()
        cost = (
            fp * spec.cost_matrix.false_positive_cost + fn * spec.cost_matrix.false_negative_cost
            + tp * spec.cost_matrix.true_positive_cost + tn * spec.cost_matrix.true_negative_cost
        )
        return ThresholdCandidateResult(threshold, float(cost), None, True, precision, recall)

    raise ThresholdSelectionError(f"Unsupported threshold policy for candidate evaluation: {policy!r}")  # pragma: no cover - exhaustive


def _tie_break_key(policy: ThresholdPolicyKind, candidate: ThresholdCandidateResult) -> tuple[float, float, float]:
    """Section 12's deterministic tie-break: (1) best objective value
    (direction depends on policy -- COST_SENSITIVE minimizes, every other
    policy maximizes, encoded here via a signed rank so `min()` always
    finds the best); (2) smallest distance from 0.5; (3) smallest
    threshold value."""
    assert candidate.objective_value is not None
    sign = 1.0 if policy is ThresholdPolicyKind.COST_SENSITIVE else -1.0
    return (sign * candidate.objective_value, abs(candidate.threshold - 0.5), candidate.threshold)


def evaluate_threshold_candidates(probabilities: np.ndarray, labels: np.ndarray, *, spec: ThresholdSpec) -> ThresholdReport:
    """Section 12: deterministic candidate evaluation on training-side
    (inner OOF) probabilities/labels ONLY -- the caller is responsible
    for never passing outer-test data here (see `calibration.runner`'s
    module docstring for the structural guarantee)."""
    probs = np.asarray(probabilities, dtype="float64")
    labs = np.asarray(labels, dtype="float64")
    if probs.shape != labs.shape:
        raise CalibrationDataError(f"probabilities shape {probs.shape} does not match labels shape {labs.shape}")
    if len(probs) == 0:
        raise CalibrationDataError("probabilities/labels must not be empty")
    if not np.all(np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise CalibrationDataError("probabilities must be finite and in [0, 1]")
    if not np.all(np.isin(labs, (0.0, 1.0))):
        raise CalibrationDataError("labels must be binary (0.0/1.0) valued")

    if spec.policy is ThresholdPolicyKind.FIXED:
        assert spec.fixed_threshold is not None
        result = _objective_at(ThresholdPolicyKind.F1, probs, labs, spec.fixed_threshold, spec)  # F1 computed only for reporting context
        candidate = ThresholdCandidateResult(spec.fixed_threshold, None, "policy=FIXED -- no objective search performed", True, result.precision, result.recall)
        return ThresholdReport(
            schema_version=THRESHOLD_REPORT_SCHEMA_VERSION, policy=spec.policy, candidates=(candidate,),
            selected_threshold=spec.fixed_threshold, selected_objective_value=None, fallback_used=False, fallback_reason=None,
            tie_break_note="policy=FIXED: the declared fixed_threshold is used directly, no candidate search",
        )

    grid = np.linspace(0.0, 1.0, spec.candidate_grid_size)
    candidates = tuple(_objective_at(spec.policy, probs, labs, float(t), spec) for t in grid)
    feasible = [c for c in candidates if c.objective_value is not None]

    if not feasible:
        return ThresholdReport(
            schema_version=THRESHOLD_REPORT_SCHEMA_VERSION, policy=spec.policy, candidates=candidates,
            selected_threshold=spec.infeasible_fallback_threshold, selected_objective_value=None, fallback_used=True,
            fallback_reason=f"no candidate threshold satisfied policy={spec.policy.value!r}'s constraint/objective -- falling back to infeasible_fallback_threshold={spec.infeasible_fallback_threshold}",
            tie_break_note=None,
        )

    best = min(feasible, key=lambda c: _tie_break_key(spec.policy, c))
    tie_count = sum(1 for c in feasible if _tie_break_key(spec.policy, c)[0] == _tie_break_key(spec.policy, best)[0])
    note = (
        f"{tie_count} candidate(s) tied on the primary objective; broke the tie by preferring the threshold "
        "closest to 0.5, then the smallest threshold" if tie_count > 1 else None
    )
    return ThresholdReport(
        schema_version=THRESHOLD_REPORT_SCHEMA_VERSION, policy=spec.policy, candidates=candidates,
        selected_threshold=best.threshold, selected_objective_value=best.objective_value, fallback_used=False,
        fallback_reason=None, tie_break_note=note,
    )


@dataclass(frozen=True, slots=True)
class ThresholdStabilityReport:
    schema_version: int
    per_fold_thresholds: tuple[float, ...]
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    interquartile_range: float
    objective_dispersion: float | None
    constraint_satisfaction_rate: float

    def __post_init__(self) -> None:
        if not self.per_fold_thresholds:
            raise CalibrationDataError("ThresholdStabilityReport.per_fold_thresholds must not be empty")
        for name, value in (
            ("mean", self.mean), ("median", self.median), ("std", self.std), ("minimum", self.minimum),
            ("maximum", self.maximum), ("interquartile_range", self.interquartile_range),
        ):
            if not math.isfinite(value):
                raise CalibrationDataError(f"ThresholdStabilityReport.{name} must be finite, got {value!r}")
        for i, v in enumerate(self.per_fold_thresholds):
            if not math.isfinite(v):
                raise CalibrationDataError(f"ThresholdStabilityReport.per_fold_thresholds[{i}] must be finite, got {v!r}")
        if self.objective_dispersion is not None and not math.isfinite(self.objective_dispersion):
            raise CalibrationDataError(f"ThresholdStabilityReport.objective_dispersion must be finite if set, got {self.objective_dispersion!r}")
        if not (0.0 <= self.constraint_satisfaction_rate <= 1.0):
            raise CalibrationDataError(f"ThresholdStabilityReport.constraint_satisfaction_rate must be in [0, 1], got {self.constraint_satisfaction_rate}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "per_fold_thresholds": list(self.per_fold_thresholds), "mean": self.mean,
            "median": self.median, "std": self.std, "minimum": self.minimum, "maximum": self.maximum,
            "interquartile_range": self.interquartile_range, "objective_dispersion": self.objective_dispersion,
            "constraint_satisfaction_rate": self.constraint_satisfaction_rate,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ThresholdStabilityReport:
        require_schema_version(raw, supported=THRESHOLD_STABILITY_SCHEMA_VERSION, context="ThresholdStabilityReport")
        return cls(
            schema_version=THRESHOLD_STABILITY_SCHEMA_VERSION,
            per_fold_thresholds=tuple(float(v) for v in as_json_list(raw["per_fold_thresholds"], field_name="per_fold_thresholds")),
            mean=float(str(raw["mean"])), median=float(str(raw["median"])), std=float(str(raw["std"])),
            minimum=float(str(raw["minimum"])), maximum=float(str(raw["maximum"])),
            interquartile_range=float(str(raw["interquartile_range"])),
            objective_dispersion=(None if raw.get("objective_dispersion") is None else float(str(raw["objective_dispersion"]))),
            constraint_satisfaction_rate=float(str(raw["constraint_satisfaction_rate"])),
        )


def compute_threshold_stability(per_fold_reports: Sequence[ThresholdReport]) -> ThresholdStabilityReport:
    """Section 13: threshold stability across INNER folds -- never
    across outer folds, and never influenced by outer-test in any way
    (every `ThresholdReport` supplied here must itself have been computed
    from one inner fold's own training-side predictions)."""
    if not per_fold_reports:
        raise CalibrationDataError("compute_threshold_stability requires at least one ThresholdReport")
    thresholds = [r.selected_threshold for r in per_fold_reports]
    objective_values = [r.selected_objective_value for r in per_fold_reports if r.selected_objective_value is not None]
    satisfaction_rate = sum(1 for r in per_fold_reports if not r.fallback_used) / len(per_fold_reports)
    quartiles = statistics.quantiles(thresholds, n=4, method="inclusive") if len(thresholds) > 1 else [thresholds[0]] * 3
    return ThresholdStabilityReport(
        schema_version=THRESHOLD_STABILITY_SCHEMA_VERSION, per_fold_thresholds=tuple(thresholds),
        mean=statistics.fmean(thresholds), median=statistics.median(thresholds),
        std=(statistics.pstdev(thresholds) if len(thresholds) > 1 else 0.0), minimum=min(thresholds), maximum=max(thresholds),
        interquartile_range=(quartiles[2] - quartiles[0]),
        objective_dispersion=(statistics.pstdev(objective_values) if len(objective_values) > 1 else (0.0 if objective_values else None)),
        constraint_satisfaction_rate=satisfaction_rate,
    )


__all__ = [
    "THRESHOLD_REPORT_SCHEMA_VERSION",
    "THRESHOLD_STABILITY_SCHEMA_VERSION",
    "ThresholdCandidateResult",
    "ThresholdReport",
    "ThresholdStabilityReport",
    "apply_threshold",
    "compute_threshold_stability",
    "evaluate_threshold_candidates",
]
