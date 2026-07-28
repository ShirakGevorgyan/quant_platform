"""Abstention / selective prediction (Milestone 4E, Section 14).

INVALID PREDICTIONS FAIL CLOSED, UNCONDITIONALLY
--------------------------------------------------------------------------
Section 14: "Invalid predictions should normally fail closed rather than
be converted to abstain unless the spec explicitly permits it." This
module implements the "normally" case as the ONLY case: `decide` raises
`CalibrationDataError` for a non-finite or out-of-range probability --
there is no spec flag to relax this. `RawPredictionSet.__post_init__`
already guarantees every probability reaching this module is valid; a
value that isn't indicates a deeper contract violation upstream, which
silently mapping to `ABSTAIN` would hide rather than surface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quant_platform.calibration.models import AbstentionPolicyKind, AbstentionReasonCode, Decision
from quant_platform.calibration.specs import AbstentionSpec
from quant_platform.calibration.thresholds import apply_threshold
from quant_platform.core.exceptions import CalibrationDataError, CalibrationValidationError
from quant_platform.ml.persistence import require_schema_version

SELECTIVE_EVALUATION_SCHEMA_VERSION = 1


def decide(
    probability: float, threshold: float, *, spec: AbstentionSpec, confidence: float | None = None, uncertainty: float | None = None,
) -> tuple[Decision, AbstentionReasonCode]:
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise CalibrationDataError(f"abstention.decide: probability must be a finite value in [0, 1], got {probability!r}")
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise CalibrationDataError(f"abstention.decide: threshold must be a finite value in [0, 1], got {threshold!r}")

    # Delegates the actual boundary comparison to `apply_threshold` --
    # the ONE place this platform decides `probability >= threshold` --
    # rather than re-implementing it inline here.
    base_decision = Decision.POSITIVE if bool(apply_threshold(np.asarray([probability]), threshold)[0]) else Decision.NEGATIVE

    if spec.policy is AbstentionPolicyKind.NONE:
        return base_decision, AbstentionReasonCode.NOT_ABSTAINED

    if spec.policy is AbstentionPolicyKind.SYMMETRIC_BAND:
        assert spec.band_half_width is not None
        if abs(probability - threshold) < spec.band_half_width:
            return Decision.ABSTAIN, AbstentionReasonCode.INSIDE_UNCERTAINTY_BAND
        return base_decision, AbstentionReasonCode.NOT_ABSTAINED

    if spec.policy is AbstentionPolicyKind.MIN_CONFIDENCE:
        assert spec.min_confidence is not None
        if confidence is None:
            raise CalibrationValidationError("abstention.decide: policy=MIN_CONFIDENCE requires a confidence score")
        if not (0.0 <= confidence <= 1.0):
            raise CalibrationDataError(f"abstention.decide: confidence must be in [0, 1], got {confidence!r}")
        if confidence < spec.min_confidence:
            return Decision.ABSTAIN, AbstentionReasonCode.BELOW_CONFIDENCE_FLOOR
        return base_decision, AbstentionReasonCode.NOT_ABSTAINED

    if spec.policy is AbstentionPolicyKind.MAX_UNCERTAINTY:
        assert spec.max_uncertainty is not None
        if uncertainty is None:
            raise CalibrationValidationError("abstention.decide: policy=MAX_UNCERTAINTY requires an uncertainty score")
        if not (0.0 <= uncertainty <= 1.0):
            raise CalibrationDataError(f"abstention.decide: uncertainty must be in [0, 1], got {uncertainty!r}")
        if uncertainty > spec.max_uncertainty:
            return Decision.ABSTAIN, AbstentionReasonCode.UNCERTAINTY_ABOVE_LIMIT
        return base_decision, AbstentionReasonCode.NOT_ABSTAINED

    if spec.policy is AbstentionPolicyKind.CLASS_SPECIFIC_BOUNDARIES:
        assert spec.negative_upper_bound is not None and spec.positive_lower_bound is not None
        if probability <= spec.negative_upper_bound:
            return Decision.NEGATIVE, AbstentionReasonCode.NOT_ABSTAINED
        if probability >= spec.positive_lower_bound:
            return Decision.POSITIVE, AbstentionReasonCode.NOT_ABSTAINED
        return Decision.ABSTAIN, AbstentionReasonCode.INSIDE_UNCERTAINTY_BAND

    raise CalibrationValidationError(f"abstention.decide: unsupported policy {spec.policy!r}")  # pragma: no cover - exhaustive enum


@dataclass(frozen=True, slots=True)
class SelectivePredictionEvaluation:
    schema_version: int
    n_total: int
    n_accepted: int
    coverage: float
    abstention_rate: float
    accuracy_on_accepted: float | None
    balanced_accuracy_on_accepted: float | None
    class_conditional_coverage: dict[str, float]
    selective_risk: float | None
    risk_coverage_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.n_total < 1:
            raise CalibrationDataError(f"SelectivePredictionEvaluation.n_total must be >= 1, got {self.n_total}")
        if not (0 <= self.n_accepted <= self.n_total):
            raise CalibrationDataError(f"SelectivePredictionEvaluation.n_accepted must be in [0, {self.n_total}], got {self.n_accepted}")
        # Explicit finiteness checks BEFORE the `abs(...) > 1e-9` consistency
        # comparisons below: any comparison against NaN (other than `!=`)
        # evaluates to False in IEEE 754, so `abs(nan - x) > 1e-9` would
        # silently NOT raise for a NaN `coverage`/`abstention_rate` -- the
        # consistency check would be vacuously "satisfied" instead of
        # catching the corruption.
        for name, value in (
            ("coverage", self.coverage), ("abstention_rate", self.abstention_rate),
            ("accuracy_on_accepted", self.accuracy_on_accepted), ("balanced_accuracy_on_accepted", self.balanced_accuracy_on_accepted),
            ("selective_risk", self.selective_risk),
        ):
            if value is not None and not math.isfinite(value):
                raise CalibrationDataError(f"SelectivePredictionEvaluation.{name} must be finite if set, got {value!r}")
        for name, value in self.class_conditional_coverage.items():
            if not math.isfinite(value):
                raise CalibrationDataError(f"SelectivePredictionEvaluation.class_conditional_coverage[{name!r}] must be finite, got {value!r}")
        for i, point in enumerate(self.risk_coverage_points):
            if not all(math.isfinite(v) for v in point):
                raise CalibrationDataError(f"SelectivePredictionEvaluation.risk_coverage_points[{i}] must contain only finite values, got {point!r}")
        if abs(self.coverage - self.n_accepted / self.n_total) > 1e-9:
            raise CalibrationDataError("SelectivePredictionEvaluation.coverage must equal n_accepted / n_total")
        if abs(self.abstention_rate - (1.0 - self.coverage)) > 1e-9:
            raise CalibrationDataError("SelectivePredictionEvaluation.abstention_rate must equal 1 - coverage")
        if self.n_accepted == 0 and (self.accuracy_on_accepted is not None or self.selective_risk is not None):
            raise CalibrationDataError(
                "SelectivePredictionEvaluation: accuracy_on_accepted/selective_risk must be None when n_accepted == 0 "
                "(zero accepted predictions -- undefined, never fabricated as 0.0 or 1.0)"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "n_total": self.n_total, "n_accepted": self.n_accepted,
            "coverage": self.coverage, "abstention_rate": self.abstention_rate, "accuracy_on_accepted": self.accuracy_on_accepted,
            "balanced_accuracy_on_accepted": self.balanced_accuracy_on_accepted,
            "class_conditional_coverage": dict(sorted(self.class_conditional_coverage.items())),
            "selective_risk": self.selective_risk, "risk_coverage_points": [list(p) for p in self.risk_coverage_points],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SelectivePredictionEvaluation:
        require_schema_version(raw, supported=SELECTIVE_EVALUATION_SCHEMA_VERSION, context="SelectivePredictionEvaluation")

        def _opt(key: str) -> float | None:
            return None if raw.get(key) is None else float(str(raw[key]))

        ccc_raw = raw.get("class_conditional_coverage") or {}
        if not isinstance(ccc_raw, dict):
            raise CalibrationDataError("SelectivePredictionEvaluation.class_conditional_coverage must be a JSON object")
        points_raw = raw.get("risk_coverage_points") or []
        if not isinstance(points_raw, list):
            raise CalibrationDataError("SelectivePredictionEvaluation.risk_coverage_points must be a JSON array")
        return cls(
            schema_version=SELECTIVE_EVALUATION_SCHEMA_VERSION, n_total=int(str(raw["n_total"])), n_accepted=int(str(raw["n_accepted"])),
            coverage=float(str(raw["coverage"])), abstention_rate=float(str(raw["abstention_rate"])),
            accuracy_on_accepted=_opt("accuracy_on_accepted"), balanced_accuracy_on_accepted=_opt("balanced_accuracy_on_accepted"),
            class_conditional_coverage={str(k): float(v) for k, v in ccc_raw.items()}, selective_risk=_opt("selective_risk"),
            risk_coverage_points=tuple((float(p[0]), float(p[1])) for p in points_raw),
        )


def evaluate_selective_prediction(decisions: Sequence[Decision], labels: Sequence[float]) -> SelectivePredictionEvaluation:
    """Section 14: "Do not present improved accepted-sample accuracy
    without also presenting the reduced coverage." -- `coverage` and
    `accuracy_on_accepted` are always computed and returned together, by
    construction, from the same function call."""
    from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-untyped]

    if len(decisions) != len(labels):
        raise CalibrationDataError(f"evaluate_selective_prediction: decisions length {len(decisions)} does not match labels length {len(labels)}")
    n_total = len(decisions)
    if n_total == 0:
        raise CalibrationDataError("evaluate_selective_prediction requires at least one sample")
    for lab in labels:
        if lab not in (0.0, 1.0):
            raise CalibrationDataError(f"evaluate_selective_prediction: labels must be binary (0.0/1.0) valued, got {lab!r}")

    accepted_idx = [i for i, d in enumerate(decisions) if d is not Decision.ABSTAIN]
    n_accepted = len(accepted_idx)
    coverage = n_accepted / n_total

    accuracy = None
    balanced_accuracy = None
    selective_risk = None
    if n_accepted > 0:
        correct = 0
        accepted_preds = []
        accepted_labels = []
        for i in accepted_idx:
            predicted_positive = decisions[i] is Decision.POSITIVE
            true_positive = labels[i] == 1.0
            if predicted_positive == true_positive:
                correct += 1
            accepted_preds.append(1.0 if predicted_positive else 0.0)
            accepted_labels.append(labels[i])
        accuracy = correct / n_accepted
        selective_risk = 1.0 - accuracy
        if len(set(accepted_labels)) >= 2:
            balanced_accuracy = float(balanced_accuracy_score(accepted_labels, accepted_preds))

    class_conditional: dict[str, float] = {}
    for class_label, class_name in ((0.0, "negative"), (1.0, "positive")):
        class_total = sum(1 for lab in labels if lab == class_label)
        if class_total > 0:
            class_accepted = sum(1 for i in accepted_idx if labels[i] == class_label)
            class_conditional[class_name] = class_accepted / class_total

    risk_coverage_points = _risk_coverage_curve(decisions, labels)

    return SelectivePredictionEvaluation(
        schema_version=SELECTIVE_EVALUATION_SCHEMA_VERSION, n_total=n_total, n_accepted=n_accepted, coverage=coverage,
        abstention_rate=1.0 - coverage, accuracy_on_accepted=accuracy, balanced_accuracy_on_accepted=balanced_accuracy,
        class_conditional_coverage=class_conditional, selective_risk=selective_risk, risk_coverage_points=risk_coverage_points,
    )


def _risk_coverage_curve(decisions: Sequence[Decision], labels: Sequence[float]) -> tuple[tuple[float, float], ...]:
    """One `(coverage, risk)` point for the ACTUAL accept/abstain
    partition this policy produced -- a single-point "curve" (this
    module does not sweep a separate confidence-ranking threshold to
    trace a full curve; `calibration.fitting` may call this repeatedly
    across several policies/parameterizations to build a genuine multi-
    point curve for a report). Deterministically ordered: `(coverage,
    risk)` ascending by coverage."""
    accepted = [(d, lab) for d, lab in zip(decisions, labels, strict=True) if d is not Decision.ABSTAIN]
    if not accepted:
        return ((0.0, 0.0),)
    n_total = len(decisions)
    coverage = len(accepted) / n_total
    incorrect = sum(1 for d, lab in accepted if (d is Decision.POSITIVE) != (lab == 1.0))
    risk = incorrect / len(accepted)
    return ((coverage, risk),)


__all__ = [
    "SELECTIVE_EVALUATION_SCHEMA_VERSION",
    "SelectivePredictionEvaluation",
    "decide",
    "evaluate_selective_prediction",
]
