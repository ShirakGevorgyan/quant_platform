"""Shared enums and the authoritative raw-prediction contract for the
calibration/thresholding/confidence/uncertainty framework (Milestone 4E).

WHY A SEPARATE RAW-PREDICTION CONTRACT, NOT A BARE DATAFRAME
--------------------------------------------------------------------------
Every other module in this package (`fitting`, `methods`, `thresholds`,
`confidence`, `uncertainty`) consumes `RawPredictionSet`, never a raw
`pandas.DataFrame` with feature/label/prediction columns mixed together.
This is a structural leakage guard, not just a style preference: a
`RawPredictionSet` for training-side (inner OOF) data and one for
outer-test data are constructed through DIFFERENT code paths
(`calibration.fitting` vs. `calibration.runner`'s outer-test evaluation
step), so calibration-fitting code has no DataFrame in scope that even
COULD contain outer-test labels -- see `calibration.runner`'s module
docstring for the full step-by-step isolation argument.

CLASS ORDERING IS NEVER INFERRED
--------------------------------------------------------------------------
`RawPredictionSet.class_labels`/`positive_class_index` are always
extracted from the fitted model's own `ProbabilisticPredictor.
class_labels` (see `ml.interfaces`), exactly like `ml.metrics`/
`optimization.outer_fold` already do (`class_labels.index(1)`) -- never
assumed to be `(0, 1)` by incidental array position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from quant_platform.core.exceptions import CalibrationDataError
from quant_platform.ml.models import ObjectiveType
from quant_platform.ml.persistence import as_json_list, parse_utc_timestamp, require_schema_version

RAW_PREDICTION_SET_SCHEMA_VERSION = 1

_SUPPORTED_OBJECTIVES: tuple[ObjectiveType, ...] = (
    ObjectiveType.BINARY_CLASSIFICATION,
    ObjectiveType.MULTICLASS_CLASSIFICATION,
)
"""Section 2: "Support the existing classification task contracts... Do
not invent regression calibration." `ObjectiveType.REGRESSION` is
therefore never accepted anywhere in this package. Multiclass is
STRUCTURALLY represented throughout (`RawPredictionSet`, specs) but every
higher-level entry point (`calibration.specs.CalibrationSpec.__post_init__`,
`calibration.runner`) fails closed on it today: no model registered in
`ml.model_zoo` actually declares `ObjectiveType.MULTICLASS_CLASSIFICATION`
support end-to-end (see `ml.metrics.compute_metrics`'s own identical,
pre-existing "declared but no registered model targets it yet" note) --
binary classification is this milestone's fully supported reference
implementation, per the milestone spec's own Section 2."""

_POSITIVE_CLASS_LABEL = 1.0
"""The platform-wide fixed convention `ml.metrics` already establishes
("the POSITIVE class is always 1.0, the NEGATIVE class always 0.0" for
binary classification) -- `CalibrationSpec.positive_class_label` is still
an explicit, validated spec field (never a silent hardcode invisible to a
reader of a persisted spec), but is currently constrained to equal this
one value so calibration's notion of "positive" can never silently
diverge from what `ml.metrics.compute_classification_metrics` already
assumes when this framework's own metrics are compared against it."""


class CalibrationMethodKind(Enum):
    IDENTITY = "identity"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    BETA = "beta"


class ProbabilityRepresentation(Enum):
    """Which raw model output one calibration method's `fit`/`transform`
    actually consumes -- persisted explicitly per Section 5's "The
    framework must explicitly record which raw representation is used by
    each calibration method." requirement, never left implicit."""

    DECISION_FUNCTION = "decision_function"
    PREDICT_PROBA = "predict_proba"


class SelectionMetric(Enum):
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"
    EXPECTED_CALIBRATION_ERROR = "expected_calibration_error"
    MAXIMUM_CALIBRATION_ERROR = "maximum_calibration_error"


class CalibrationTieBreakPolicy(Enum):
    CANONICAL = "canonical"
    """The ONE authoritative, fixed tie-break chain (Section 8): (1)
    primary selection metric: (2) simpler method preference (`identity` <
    `platt` < `beta` < `isotonic`, fewer fitted parameters first); (3)
    secondary metric (log loss if the primary was Brier-family, Brier
    score otherwise); (4) lexical method identifier. Not caller-
    configurable -- this enum exists so a persisted `CalibrationSpec`
    always names its tie-break policy explicitly (Section 8: "The exact
    order must be specified and persisted"), while the actual chain stays
    a single, auditable implementation in `calibration.fitting`, exactly
    like `optimization.candidates.rank_trials`'s own fixed chain."""


class BinningStrategy(Enum):
    EQUAL_WIDTH = "equal_width"
    EQUAL_FREQUENCY = "equal_frequency"


class ThresholdPolicyKind(Enum):
    FIXED = "fixed"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    YOUDEN_J = "youden_j"
    MIN_PRECISION_MAX_RECALL = "min_precision_max_recall"
    MIN_RECALL_MAX_PRECISION = "min_recall_max_precision"
    COST_SENSITIVE = "cost_sensitive"


class AbstentionPolicyKind(Enum):
    NONE = "none"
    SYMMETRIC_BAND = "symmetric_band"
    MIN_CONFIDENCE = "min_confidence"
    MAX_UNCERTAINTY = "max_uncertainty"
    CLASS_SPECIFIC_BOUNDARIES = "class_specific_boundaries"


class Decision(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSTAIN = "abstain"


class AbstentionReasonCode(Enum):
    NOT_ABSTAINED = "not_abstained"
    BELOW_CONFIDENCE_FLOOR = "below_confidence_floor"
    INSIDE_UNCERTAINTY_BAND = "inside_uncertainty_band"
    UNCERTAINTY_ABOVE_LIMIT = "uncertainty_above_limit"
    INSUFFICIENT_CALIBRATION_SUPPORT = "insufficient_calibration_support"
    INVALID_PREDICTION = "invalid_prediction"


class ConfidenceCategory(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ScoreProvenance(Enum):
    """Section 15: "whether the score is empirical, heuristic, or
    composite" -- attached to both confidence and uncertainty outputs."""

    EMPIRICAL = "empirical"
    HEURISTIC = "heuristic"
    COMPOSITE = "composite"


class CalibrationStage(Enum):
    """Section 23's exact, closed stage list for ONE outer fold's journey
    through `calibration.runner`'s 16-step sequence. `EVALUATED` loops
    back to `INNER_PREDICTIONS_READY` to begin the NEXT outer fold
    (`CalibrationManifest.current_outer_fold_index` advances on that same
    transition) -- exactly `optimization.models.OptimizationStage.
    STORING_RESULTS -> RUNNING_OUTER_FOLD`'s identical loop-back-to-an-
    earlier-name-for-the-next-unit precedent -- or forward to `VERIFIED`
    once every outer fold is done.

    NO `RECOVERABLE_FAILURE` STAGE, UNLIKE `OptimizationStage` -- BUT
    EVERY MID-FOLD STAGE CAN RESTART THAT FOLD
    --------------------------------------------------------------------------
    Section 23 gives this CLOSED list of names with no recoverable-
    failure entry, and calibration's per-outer-fold sequence is a single
    deterministic linear pipeline (unlike optimization's trial SEARCH
    loop, which can fail one trial and continue trying others -- the
    reason `OptimizationStage` needed a dedicated intermediate stage).
    `calibration.runner.run_outer_fold_calibration` computes Section 18's
    steps 2-14 as ONE atomic, pure function of already-fixed inputs
    (spec + outer fold + timeline + seed), so a crash at ANY point during
    or immediately after that call -- leaving the manifest at
    `CALIBRATORS_EVALUATED`, `CALIBRATOR_SELECTED`, `THRESHOLD_SELECTED`,
    `POLICIES_FROZEN`, or `OUTER_PREDICTIONS_READY` -- is always safe to
    resolve by simply REDOING that entire fold from scratch: every one of
    these five stages therefore has a legal edge straight back to
    `INNER_PREDICTIONS_READY`, mirroring `optimization.runner`'s own
    `RECOVERABLE_FAILURE -> RUNNING_OUTER_FOLD` restart edge, just
    without a dedicated intermediate stage name to route through. This is
    still "legal monotonic transitions": each edge either advances
    toward completing the CURRENT fold, or starts a fresh, independent
    attempt at it -- never regresses an already-COMPLETED fold (recorded
    separately via `CalibrationManifest.completed_outer_fold_indices`,
    which this state machine never touches)."""

    CREATED = "created"
    INNER_PREDICTIONS_READY = "inner_predictions_ready"
    CALIBRATORS_EVALUATED = "calibrators_evaluated"
    CALIBRATOR_SELECTED = "calibrator_selected"
    THRESHOLD_SELECTED = "threshold_selected"
    POLICIES_FROZEN = "policies_frozen"
    OUTER_PREDICTIONS_READY = "outer_predictions_ready"
    EVALUATED = "evaluated"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_CALIBRATION_STAGES: frozenset[CalibrationStage] = frozenset({CalibrationStage.COMPLETED, CalibrationStage.FAILED})
_MID_FOLD_CALIBRATION_STAGES: frozenset[CalibrationStage] = frozenset({
    CalibrationStage.CALIBRATORS_EVALUATED, CalibrationStage.CALIBRATOR_SELECTED, CalibrationStage.THRESHOLD_SELECTED,
    CalibrationStage.POLICIES_FROZEN, CalibrationStage.OUTER_PREDICTIONS_READY,
})
"""Every stage strictly between `INNER_PREDICTIONS_READY` and
`EVALUATED` for one outer fold -- each can legally restart straight back
to `INNER_PREDICTIONS_READY` (see `CalibrationStage`'s own docstring)."""

_LEGAL_CALIBRATION_TRANSITIONS: dict[CalibrationStage, frozenset[CalibrationStage]] = {
    CalibrationStage.CREATED: frozenset({CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.INNER_PREDICTIONS_READY: frozenset({CalibrationStage.CALIBRATORS_EVALUATED, CalibrationStage.FAILED}),
    CalibrationStage.CALIBRATORS_EVALUATED: frozenset({CalibrationStage.CALIBRATOR_SELECTED, CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.CALIBRATOR_SELECTED: frozenset({CalibrationStage.THRESHOLD_SELECTED, CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.THRESHOLD_SELECTED: frozenset({CalibrationStage.POLICIES_FROZEN, CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.POLICIES_FROZEN: frozenset({CalibrationStage.OUTER_PREDICTIONS_READY, CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.OUTER_PREDICTIONS_READY: frozenset({CalibrationStage.EVALUATED, CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.FAILED}),
    CalibrationStage.EVALUATED: frozenset({
        # Loop back for the NEXT outer fold, or move on to whole-run verification.
        CalibrationStage.INNER_PREDICTIONS_READY, CalibrationStage.VERIFIED, CalibrationStage.FAILED,
    }),
    CalibrationStage.VERIFIED: frozenset({CalibrationStage.COMPLETED, CalibrationStage.FAILED}),
    CalibrationStage.COMPLETED: frozenset(),
    CalibrationStage.FAILED: frozenset(),
}


def is_legal_calibration_transition(current: CalibrationStage, target: CalibrationStage) -> bool:
    return target in _LEGAL_CALIBRATION_TRANSITIONS[current]


def is_terminal_calibration_stage(stage: CalibrationStage) -> bool:
    return stage in TERMINAL_CALIBRATION_STAGES


class DeterminismPolicy(Enum):
    STRICT = "strict"
    """Resume fails closed (`CalibrationResumeError`) if the recorded
    environment snapshot's tracked package versions differ from the
    currently installed ones -- mirrors `optimization.runner._require_
    compatible_optuna_version`'s fail-closed policy exactly."""
    WARN = "warn"
    """Resume proceeds with a logged warning on environment mismatch --
    an explicit, opt-in relaxation, never the default."""


def _require_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class RawPredictionSet:
    """One outer fold's (or one inner fold's) complete set of raw model
    outputs, in deterministic ascending `sample_positions` order. Every
    array is parallel (same length, same order) -- see `__post_init__`
    for the full raw-prediction contract this enforces (Section 5).

    `true_labels` is `None` whenever "evaluation context permits" does
    NOT hold -- specifically, this is how outer-test labels are kept
    structurally unavailable to any code path that must not see them:
    `calibration.runner` never constructs a `RawPredictionSet` with
    `true_labels` populated until its own explicit, late "final
    evaluation" step (see that module's docstring)."""

    schema_version: int
    outer_fold_index: int
    inner_fold_index: int | None
    sample_positions: tuple[int, ...]
    timestamps: tuple[str, ...]
    raw_scores: tuple[float, ...] | None
    raw_probabilities: tuple[float, ...] | None
    class_labels: tuple[float, ...]
    positive_class_index: int
    source_model_identity: str
    source_experiment_id: str
    true_labels: tuple[float, ...] | None = None
    fitted_on_rows: tuple[int, ...] | None = None
    """Defense-in-depth provenance (Section 6): the exact row positions
    the model that PRODUCED these predictions was fit on -- `None` only
    for the final outer-test prediction set (produced by a model fit on
    outer-train, evaluated on outer-test; row-disjointness from
    `sample_positions` is instead guaranteed structurally by
    `calibration.runner`'s separate-object design, see its docstring).
    When set (every inner OOF prediction), `calibration.fitting` verifies
    `set(fitted_on_rows) & set(sample_positions) == set()` for every
    single row before concatenating it into training-side calibration
    data."""

    def __post_init__(self) -> None:
        if self.outer_fold_index < 0:
            raise CalibrationDataError(f"RawPredictionSet.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if self.inner_fold_index is not None and self.inner_fold_index < 0:
            raise CalibrationDataError(f"RawPredictionSet.inner_fold_index must be >= 0 if set, got {self.inner_fold_index}")
        n = len(self.sample_positions)
        if n == 0:
            raise CalibrationDataError("RawPredictionSet must contain at least one sample")
        for name, arr in (
            ("timestamps", self.timestamps), ("raw_scores", self.raw_scores),
            ("raw_probabilities", self.raw_probabilities), ("true_labels", self.true_labels),
        ):
            if arr is not None and len(arr) != n:
                raise CalibrationDataError(
                    f"RawPredictionSet.{name} has length {len(arr)}, expected {n} (== len(sample_positions))"
                )
        if self.raw_scores is None and self.raw_probabilities is None:
            raise CalibrationDataError("RawPredictionSet requires at least one of raw_scores/raw_probabilities")
        if len(set(self.sample_positions)) != n:
            raise CalibrationDataError("RawPredictionSet.sample_positions must not contain duplicate sample identities")
        if list(self.sample_positions) != sorted(self.sample_positions):
            raise CalibrationDataError("RawPredictionSet.sample_positions must be strictly ascending")
        if list(self.timestamps) != sorted(self.timestamps):
            raise CalibrationDataError("RawPredictionSet.timestamps must be non-decreasing in sample_positions order")
        if not self.class_labels:
            raise CalibrationDataError("RawPredictionSet.class_labels must not be empty")
        if len(set(self.class_labels)) != len(self.class_labels):
            raise CalibrationDataError("RawPredictionSet.class_labels must not contain duplicates")
        if not (0 <= self.positive_class_index < len(self.class_labels)):
            raise CalibrationDataError(
                f"RawPredictionSet.positive_class_index ({self.positive_class_index}) is out of range for "
                f"class_labels of length {len(self.class_labels)}"
            )
        if self.class_labels[self.positive_class_index] != _POSITIVE_CLASS_LABEL:
            raise CalibrationDataError(
                f"RawPredictionSet.class_labels[positive_class_index] must equal {_POSITIVE_CLASS_LABEL!r} "
                f"(this platform's fixed positive-class convention), got {self.class_labels[self.positive_class_index]!r}"
            )
        if not self.source_model_identity:
            raise CalibrationDataError("RawPredictionSet.source_model_identity must not be empty")
        if not self.source_experiment_id:
            raise CalibrationDataError("RawPredictionSet.source_experiment_id must not be empty")
        for ts in self.timestamps:
            parse_utc_timestamp(ts)
        if self.raw_scores is not None:
            for v in self.raw_scores:
                _require_finite(v, field_name="RawPredictionSet.raw_scores[]")
        if self.raw_probabilities is not None:
            for v in self.raw_probabilities:
                _require_finite(v, field_name="RawPredictionSet.raw_probabilities[]")
                if not (0.0 <= v <= 1.0):
                    raise CalibrationDataError(f"RawPredictionSet.raw_probabilities[] must be in [0, 1], got {v!r}")
        if self.true_labels is not None:
            allowed = frozenset(self.class_labels)
            for v in self.true_labels:
                _require_finite(v, field_name="RawPredictionSet.true_labels[]")
                if v not in allowed:
                    raise CalibrationDataError(
                        f"RawPredictionSet.true_labels[] must be one of {sorted(allowed)}, got {v!r}"
                    )
        if self.fitted_on_rows is not None:
            overlap = set(self.fitted_on_rows) & set(self.sample_positions)
            if overlap:
                raise CalibrationDataError(
                    f"RawPredictionSet: {len(overlap)} sample position(s) were both predicted AND present in "
                    f"fitted_on_rows (the model that produced these predictions trained on its own evaluation "
                    f"rows) -- this is a leakage violation: {sorted(overlap)[:10]}"
                )

    @property
    def n_samples(self) -> int:
        return len(self.sample_positions)

    def positive_probabilities(self) -> tuple[float, ...] | None:
        """Convenience accessor mirroring `ml.metrics`/`optimization.
        outer_fold`'s `class_labels.index(1)` convention -- since
        `positive_class_index` always indexes a two-column
        (`predict_proba`-shaped) probability, this is only meaningful
        (and only ever populated) for binary classification."""
        return self.raw_probabilities

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index,
            "inner_fold_index": self.inner_fold_index, "sample_positions": list(self.sample_positions),
            "timestamps": list(self.timestamps),
            "raw_scores": (None if self.raw_scores is None else list(self.raw_scores)),
            "raw_probabilities": (None if self.raw_probabilities is None else list(self.raw_probabilities)),
            "class_labels": list(self.class_labels), "positive_class_index": self.positive_class_index,
            "source_model_identity": self.source_model_identity, "source_experiment_id": self.source_experiment_id,
            "true_labels": (None if self.true_labels is None else list(self.true_labels)),
            "fitted_on_rows": (None if self.fitted_on_rows is None else list(self.fitted_on_rows)),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RawPredictionSet:
        require_schema_version(raw, supported=RAW_PREDICTION_SET_SCHEMA_VERSION, context="RawPredictionSet")

        def _opt_float_tuple(key: str) -> tuple[float, ...] | None:
            value = raw.get(key)
            return None if value is None else tuple(float(v) for v in as_json_list(value, field_name=key))

        def _opt_int_tuple(key: str) -> tuple[int, ...] | None:
            value = raw.get(key)
            return None if value is None else tuple(int(v) for v in as_json_list(value, field_name=key))

        return cls(
            schema_version=RAW_PREDICTION_SET_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            inner_fold_index=(None if raw.get("inner_fold_index") is None else int(str(raw["inner_fold_index"]))),
            sample_positions=tuple(int(v) for v in as_json_list(raw["sample_positions"], field_name="sample_positions")),
            timestamps=tuple(str(v) for v in as_json_list(raw["timestamps"], field_name="timestamps")),
            raw_scores=_opt_float_tuple("raw_scores"), raw_probabilities=_opt_float_tuple("raw_probabilities"),
            class_labels=tuple(float(v) for v in as_json_list(raw["class_labels"], field_name="class_labels")),
            positive_class_index=int(str(raw["positive_class_index"])),
            source_model_identity=str(raw["source_model_identity"]), source_experiment_id=str(raw["source_experiment_id"]),
            true_labels=_opt_float_tuple("true_labels"), fitted_on_rows=_opt_int_tuple("fitted_on_rows"),
        )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """A trained model's declared identity -- name/version/objective plus
    a content fingerprint of its `ModelDefinition` -- persisted alongside
    every `RawPredictionSet` (`source_model_identity`) and independently
    re-derivable, never trusted as a bare opaque string."""

    model_name: str
    model_version: str
    objective: ObjectiveType
    definition_fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("ModelIdentity.model_name must not be empty")
        if not self.model_version:
            raise ValueError("ModelIdentity.model_version must not be empty")
        if self.objective not in _SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"ModelIdentity.objective must be one of {[o.value for o in _SUPPORTED_OBJECTIVES]}, got {self.objective.value!r}"
            )
        if not self.definition_fingerprint:
            raise ValueError("ModelIdentity.definition_fingerprint must not be empty")

    def qualified(self) -> str:
        return f"{self.model_name}@{self.model_version}#{self.definition_fingerprint[:12]}"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name, "model_version": self.model_version, "objective": self.objective.value,
            "definition_fingerprint": self.definition_fingerprint,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ModelIdentity:
        return cls(
            model_name=str(raw["model_name"]), model_version=str(raw["model_version"]),
            objective=ObjectiveType(raw["objective"]), definition_fingerprint=str(raw["definition_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class FailedCandidateReason:
    """Section 8: "Persist the failure reason for every rejected method."
    Reused by calibrator candidates AND threshold candidates."""

    identifier: str
    reason: str
    context: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("FailedCandidateReason.identifier must not be empty")
        if not self.reason:
            raise ValueError("FailedCandidateReason.reason must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {"identifier": self.identifier, "reason": self.reason, "context": dict(sorted(self.context.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FailedCandidateReason:
        context_raw = raw.get("context") or {}
        if not isinstance(context_raw, dict):
            raise ValueError("FailedCandidateReason.context must be a JSON object")
        return cls(identifier=str(raw["identifier"]), reason=str(raw["reason"]), context={str(k): str(v) for k, v in context_raw.items()})


__all__ = [
    "RAW_PREDICTION_SET_SCHEMA_VERSION",
    "TERMINAL_CALIBRATION_STAGES",
    "AbstentionPolicyKind",
    "AbstentionReasonCode",
    "BinningStrategy",
    "CalibrationMethodKind",
    "CalibrationStage",
    "CalibrationTieBreakPolicy",
    "ConfidenceCategory",
    "Decision",
    "DeterminismPolicy",
    "FailedCandidateReason",
    "ModelIdentity",
    "ProbabilityRepresentation",
    "RawPredictionSet",
    "ScoreProvenance",
    "SelectionMetric",
    "ThresholdPolicyKind",
    "is_legal_calibration_transition",
    "is_terminal_calibration_stage",
]
