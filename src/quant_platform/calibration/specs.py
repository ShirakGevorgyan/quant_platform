"""Immutable specifications for the calibration/threshold/confidence/
uncertainty framework (Milestone 4E, Sections 4/9/12/14/15/16).

Every spec in this module is a frozen, slotted dataclass with an
explicit `__post_init__` validator -- nothing here is an arbitrary,
unvalidated dict at a public boundary (Section 3). `CalibrationSpec` is
the top-level, content-addressed specification; every other spec in this
module is embedded within it and therefore participates in its identity.

`InnerSplitConfig` (inner out-of-fold prediction policy) is imported
directly from `quant_platform.optimization.inner_splits` -- Section 6:
"reuse the existing inner split machinery where possible" -- rather than
a second, parallel definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.core.exceptions import CalibrationValidationError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ObjectiveType
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration, SeedDomain, derive_seed
from quant_platform.optimization.inner_splits import InnerSplitConfig

CALIBRATION_SPEC_SCHEMA_VERSION = 1
_SUPPORTED_TASKS: tuple[ObjectiveType, ...] = (ObjectiveType.BINARY_CLASSIFICATION, ObjectiveType.MULTICLASS_CLASSIFICATION)
_FULLY_SUPPORTED_TASKS: tuple[ObjectiveType, ...] = (ObjectiveType.BINARY_CLASSIFICATION,)
"""Section 2: binary classification is the fully supported reference
implementation. Multiclass is structurally representable in every spec
in this module (never hardcoded out at the dataclass level) but
`CalibrationSpec.__post_init__` fails closed on it -- see that class's
own docstring for why."""
_POSITIVE_CLASS_LABEL = 1.0


def _finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise CalibrationValidationError(f"{field_name} must be finite, got {value!r}")


def _unit_interval(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if not (0.0 <= value <= 1.0):
        raise CalibrationValidationError(f"{field_name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class ProbabilityClippingPolicy:
    """Section 9. `enabled=False` (`epsilon` ignored) is the explicit "no
    clipping" policy; `enabled=True` clips to `[epsilon, 1 - epsilon]`
    -- applied ONLY after the calibration transformation itself (never
    before), unless a specific method's own docstring formally requires
    another order (none does in this milestone)."""

    enabled: bool
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.enabled:
            _finite(self.epsilon, field_name="ProbabilityClippingPolicy.epsilon")
            if not (0.0 < self.epsilon < 0.5):
                raise CalibrationValidationError(
                    f"ProbabilityClippingPolicy.epsilon must be in (0, 0.5) when enabled, got {self.epsilon!r}"
                )

    def apply(self, probability: float) -> float:
        if not self.enabled:
            return probability
        return min(max(probability, self.epsilon), 1.0 - self.epsilon)

    def to_json_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled, "epsilon": self.epsilon}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ProbabilityClippingPolicy:
        return cls(enabled=bool(raw["enabled"]), epsilon=float(str(raw.get("epsilon", 1e-6))))


@dataclass(frozen=True, slots=True)
class ReliabilityBinningSpec:
    """Section 10/11: an immutable, identity-bearing binning
    configuration -- one requested reliability diagnostic."""

    strategy: BinningStrategy
    n_bins: int

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise CalibrationValidationError(f"ReliabilityBinningSpec.n_bins must be >= 1, got {self.n_bins}")
        if self.n_bins > 1000:
            raise CalibrationValidationError(f"ReliabilityBinningSpec.n_bins must be <= 1000, got {self.n_bins}")

    def to_json_dict(self) -> dict[str, object]:
        return {"strategy": self.strategy.value, "n_bins": self.n_bins}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReliabilityBinningSpec:
        return cls(strategy=BinningStrategy(raw["strategy"]), n_bins=int(str(raw["n_bins"])))


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """Section 12H: an explicit, immutable cost matrix for cost-sensitive
    thresholding. Costs are non-negative "penalty" magnitudes (never
    negative -- a negative cost would silently invert the optimization
    direction); at least one of the two misclassification costs must be
    strictly positive, or every candidate threshold would tie at zero
    cost and selection would be meaningless."""

    false_positive_cost: float
    false_negative_cost: float
    true_positive_cost: float = 0.0
    true_negative_cost: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("false_positive_cost", self.false_positive_cost), ("false_negative_cost", self.false_negative_cost),
            ("true_positive_cost", self.true_positive_cost), ("true_negative_cost", self.true_negative_cost),
        ):
            _finite(value, field_name=f"CostMatrix.{name}")
            if value < 0:
                raise CalibrationValidationError(f"CostMatrix.{name} must be >= 0, got {value!r}")
        if self.false_positive_cost == 0 and self.false_negative_cost == 0:
            raise CalibrationValidationError(
                "CostMatrix: at least one of false_positive_cost/false_negative_cost must be > 0, "
                "otherwise every threshold candidate ties at zero cost"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "false_positive_cost": self.false_positive_cost, "false_negative_cost": self.false_negative_cost,
            "true_positive_cost": self.true_positive_cost, "true_negative_cost": self.true_negative_cost,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostMatrix:
        return cls(
            false_positive_cost=float(str(raw["false_positive_cost"])), false_negative_cost=float(str(raw["false_negative_cost"])),
            true_positive_cost=float(str(raw.get("true_positive_cost", 0.0))), true_negative_cost=float(str(raw.get("true_negative_cost", 0.0))),
        )


_CONSTRAINED_POLICIES = (ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL, ThresholdPolicyKind.MIN_RECALL_MAX_PRECISION)


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    """Section 12. `positive_prediction = probability >= threshold` is
    this framework's ONE fixed boundary rule (Section 12's explicit
    example), enforced uniformly by `calibration.thresholds.apply_
    threshold` -- never re-decided per policy, per training/verification/
    CLI call site."""

    policy: ThresholdPolicyKind
    fixed_threshold: float | None = None
    min_precision: float | None = None
    min_recall: float | None = None
    cost_matrix: CostMatrix | None = None
    infeasible_fallback_threshold: float = 0.5
    candidate_grid_size: int = 101
    """Number of evenly-spaced threshold candidates in `[0, 1]` evaluated
    for every non-FIXED policy (Section 12: "Threshold candidates must be
    evaluated deterministically") -- e.g. 101 -> `0.00, 0.01, ..., 1.00`."""

    def __post_init__(self) -> None:
        if self.policy is ThresholdPolicyKind.FIXED:
            if self.fixed_threshold is None:
                raise CalibrationValidationError("ThresholdSpec.fixed_threshold is required when policy=FIXED")
            _unit_interval(self.fixed_threshold, field_name="ThresholdSpec.fixed_threshold")
        elif self.fixed_threshold is not None:
            raise CalibrationValidationError("ThresholdSpec.fixed_threshold must be None unless policy=FIXED")

        if self.policy is ThresholdPolicyKind.MIN_PRECISION_MAX_RECALL:
            if self.min_precision is None:
                raise CalibrationValidationError("ThresholdSpec.min_precision is required when policy=MIN_PRECISION_MAX_RECALL")
            _unit_interval(self.min_precision, field_name="ThresholdSpec.min_precision")
        elif self.min_precision is not None:
            raise CalibrationValidationError("ThresholdSpec.min_precision must be None unless policy=MIN_PRECISION_MAX_RECALL")

        if self.policy is ThresholdPolicyKind.MIN_RECALL_MAX_PRECISION:
            if self.min_recall is None:
                raise CalibrationValidationError("ThresholdSpec.min_recall is required when policy=MIN_RECALL_MAX_PRECISION")
            _unit_interval(self.min_recall, field_name="ThresholdSpec.min_recall")
        elif self.min_recall is not None:
            raise CalibrationValidationError("ThresholdSpec.min_recall must be None unless policy=MIN_RECALL_MAX_PRECISION")

        if self.policy is ThresholdPolicyKind.COST_SENSITIVE:
            if self.cost_matrix is None:
                raise CalibrationValidationError("ThresholdSpec.cost_matrix is required when policy=COST_SENSITIVE")
        elif self.cost_matrix is not None:
            raise CalibrationValidationError("ThresholdSpec.cost_matrix must be None unless policy=COST_SENSITIVE")

        _unit_interval(self.infeasible_fallback_threshold, field_name="ThresholdSpec.infeasible_fallback_threshold")
        if self.candidate_grid_size < 3:
            raise CalibrationValidationError(f"ThresholdSpec.candidate_grid_size must be >= 3, got {self.candidate_grid_size}")
        if self.candidate_grid_size > 100_001:
            raise CalibrationValidationError(f"ThresholdSpec.candidate_grid_size must be <= 100001, got {self.candidate_grid_size}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.value, "fixed_threshold": self.fixed_threshold, "min_precision": self.min_precision,
            "min_recall": self.min_recall, "cost_matrix": (None if self.cost_matrix is None else self.cost_matrix.to_json_dict()),
            "infeasible_fallback_threshold": self.infeasible_fallback_threshold, "candidate_grid_size": self.candidate_grid_size,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ThresholdSpec:
        cost_matrix_raw = raw.get("cost_matrix")
        return cls(
            policy=ThresholdPolicyKind(raw["policy"]),
            fixed_threshold=(None if raw.get("fixed_threshold") is None else float(str(raw["fixed_threshold"]))),
            min_precision=(None if raw.get("min_precision") is None else float(str(raw["min_precision"]))),
            min_recall=(None if raw.get("min_recall") is None else float(str(raw["min_recall"]))),
            cost_matrix=(None if cost_matrix_raw is None else CostMatrix.from_json_dict(as_json_dict(cost_matrix_raw, field_name="cost_matrix"))),
            infeasible_fallback_threshold=float(str(raw.get("infeasible_fallback_threshold", 0.5))),
            candidate_grid_size=int(str(raw.get("candidate_grid_size", 101))),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceSpec:
    """Section 15. Category boundaries are strictly increasing bounds on
    `[0, 1)`; `very_high` is implicitly everything above `high_max`, up
    to and including `1.0`. `component_weights` (empty by default,
    meaning "single-component: distance from decision threshold only")
    declares which named components (Section 15's list) participate in a
    COMPOSITE confidence score and their relative weight -- never
    silently defaulting an unlisted, unweighted component into the mix."""

    very_low_max: float
    low_max: float
    medium_max: float
    high_max: float
    component_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bounds = (self.very_low_max, self.low_max, self.medium_max, self.high_max)
        for name, value in zip(("very_low_max", "low_max", "medium_max", "high_max"), bounds, strict=True):
            _unit_interval(value, field_name=f"ConfidenceSpec.{name}")
        if not (self.very_low_max < self.low_max < self.medium_max < self.high_max < 1.0):
            raise CalibrationValidationError(
                f"ConfidenceSpec boundaries must satisfy very_low_max < low_max < medium_max < high_max < 1.0, "
                f"got {bounds}"
            )
        for key, weight in self.component_weights.items():
            if not key:
                raise CalibrationValidationError("ConfidenceSpec.component_weights keys must not be empty")
            _finite(weight, field_name=f"ConfidenceSpec.component_weights[{key!r}]")
            if weight < 0:
                raise CalibrationValidationError(f"ConfidenceSpec.component_weights[{key!r}] must be >= 0, got {weight!r}")
        if self.component_weights and sum(self.component_weights.values()) <= 0:
            raise CalibrationValidationError("ConfidenceSpec.component_weights must sum to a positive value when non-empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "very_low_max": self.very_low_max, "low_max": self.low_max, "medium_max": self.medium_max,
            "high_max": self.high_max, "component_weights": dict(sorted(self.component_weights.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ConfidenceSpec:
        weights_raw = raw.get("component_weights") or {}
        if not isinstance(weights_raw, dict):
            raise CalibrationValidationError("ConfidenceSpec.component_weights must be a JSON object")
        return cls(
            very_low_max=float(str(raw["very_low_max"])), low_max=float(str(raw["low_max"])),
            medium_max=float(str(raw["medium_max"])), high_max=float(str(raw["high_max"])),
            component_weights={str(k): float(v) for k, v in weights_raw.items()},
        )

    def category_for(self, confidence: float) -> str:
        from quant_platform.calibration.models import ConfidenceCategory

        if not (0.0 <= confidence <= 1.0):
            raise CalibrationValidationError(f"confidence must be in [0, 1] to categorize, got {confidence!r}")
        if confidence <= self.very_low_max:
            return ConfidenceCategory.VERY_LOW.value
        if confidence <= self.low_max:
            return ConfidenceCategory.LOW.value
        if confidence <= self.medium_max:
            return ConfidenceCategory.MEDIUM.value
        if confidence <= self.high_max:
            return ConfidenceCategory.HIGH.value
        return ConfidenceCategory.VERY_HIGH.value


_KNOWN_UNCERTAINTY_COMPONENTS = frozenset({"entropy", "margin", "model_disagreement", "calibrator_disagreement", "bin_support"})
_KNOWN_AGGREGATIONS = frozenset({"mean", "max"})


@dataclass(frozen=True, slots=True)
class UncertaintySpec:
    """Section 16. `components` names which of the five documented,
    transparent proxies (Section 16 A-E) are computed; `aggregation`
    defines how the AVAILABLE ones combine into `total_uncertainty`
    (never silently replacing a missing/unavailable component with
    zero -- see `calibration.uncertainty`)."""

    components: tuple[str, ...]
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not self.components:
            raise CalibrationValidationError("UncertaintySpec.components must not be empty")
        if len(set(self.components)) != len(self.components):
            raise CalibrationValidationError("UncertaintySpec.components must not contain duplicates")
        unknown = set(self.components) - _KNOWN_UNCERTAINTY_COMPONENTS
        if unknown:
            raise CalibrationValidationError(f"UncertaintySpec.components contains unknown component(s): {sorted(unknown)}")
        if self.aggregation not in _KNOWN_AGGREGATIONS:
            raise CalibrationValidationError(
                f"UncertaintySpec.aggregation must be one of {sorted(_KNOWN_AGGREGATIONS)}, got {self.aggregation!r}"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {"components": list(self.components), "aggregation": self.aggregation}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> UncertaintySpec:
        return cls(
            components=tuple(str(c) for c in as_json_list(raw["components"], field_name="components")),
            aggregation=str(raw.get("aggregation", "mean")),
        )


@dataclass(frozen=True, slots=True)
class AbstentionSpec:
    """Section 14. Exactly one of the policy-specific field groups below
    is populated, matching `policy`."""

    policy: AbstentionPolicyKind
    band_half_width: float | None = None
    min_confidence: float | None = None
    max_uncertainty: float | None = None
    negative_upper_bound: float | None = None
    positive_lower_bound: float | None = None

    def __post_init__(self) -> None:
        if self.policy is AbstentionPolicyKind.SYMMETRIC_BAND:
            if self.band_half_width is None:
                raise CalibrationValidationError("AbstentionSpec.band_half_width is required when policy=SYMMETRIC_BAND")
            _finite(self.band_half_width, field_name="AbstentionSpec.band_half_width")
            if not (0.0 < self.band_half_width < 0.5):
                raise CalibrationValidationError(f"AbstentionSpec.band_half_width must be in (0, 0.5), got {self.band_half_width!r}")
        elif self.band_half_width is not None:
            raise CalibrationValidationError("AbstentionSpec.band_half_width must be None unless policy=SYMMETRIC_BAND")

        if self.policy is AbstentionPolicyKind.MIN_CONFIDENCE:
            if self.min_confidence is None:
                raise CalibrationValidationError("AbstentionSpec.min_confidence is required when policy=MIN_CONFIDENCE")
            _unit_interval(self.min_confidence, field_name="AbstentionSpec.min_confidence")
        elif self.min_confidence is not None:
            raise CalibrationValidationError("AbstentionSpec.min_confidence must be None unless policy=MIN_CONFIDENCE")

        if self.policy is AbstentionPolicyKind.MAX_UNCERTAINTY:
            if self.max_uncertainty is None:
                raise CalibrationValidationError("AbstentionSpec.max_uncertainty is required when policy=MAX_UNCERTAINTY")
            _unit_interval(self.max_uncertainty, field_name="AbstentionSpec.max_uncertainty")
        elif self.max_uncertainty is not None:
            raise CalibrationValidationError("AbstentionSpec.max_uncertainty must be None unless policy=MAX_UNCERTAINTY")

        if self.policy is AbstentionPolicyKind.CLASS_SPECIFIC_BOUNDARIES:
            if self.negative_upper_bound is None or self.positive_lower_bound is None:
                raise CalibrationValidationError(
                    "AbstentionSpec.negative_upper_bound/positive_lower_bound are both required when "
                    "policy=CLASS_SPECIFIC_BOUNDARIES"
                )
            _unit_interval(self.negative_upper_bound, field_name="AbstentionSpec.negative_upper_bound")
            _unit_interval(self.positive_lower_bound, field_name="AbstentionSpec.positive_lower_bound")
            if self.negative_upper_bound >= self.positive_lower_bound:
                raise CalibrationValidationError(
                    f"AbstentionSpec.negative_upper_bound ({self.negative_upper_bound}) must be < "
                    f"positive_lower_bound ({self.positive_lower_bound})"
                )
        elif self.negative_upper_bound is not None or self.positive_lower_bound is not None:
            raise CalibrationValidationError(
                "AbstentionSpec.negative_upper_bound/positive_lower_bound must both be None unless "
                "policy=CLASS_SPECIFIC_BOUNDARIES"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.value, "band_half_width": self.band_half_width, "min_confidence": self.min_confidence,
            "max_uncertainty": self.max_uncertainty, "negative_upper_bound": self.negative_upper_bound,
            "positive_lower_bound": self.positive_lower_bound,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AbstentionSpec:
        def _opt(key: str) -> float | None:
            return None if raw.get(key) is None else float(str(raw[key]))

        return cls(
            policy=AbstentionPolicyKind(raw["policy"]), band_half_width=_opt("band_half_width"),
            min_confidence=_opt("min_confidence"), max_uncertainty=_opt("max_uncertainty"),
            negative_upper_bound=_opt("negative_upper_bound"), positive_lower_bound=_opt("positive_lower_bound"),
        )


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """The top-level, content-addressed specification for one calibration
    run (Section 4). Its `calibration_id` (see `calibration.specs.
    compute_calibration_identity`) is a pure function of every field
    below except `schema_version` itself -- two independently constructed
    `CalibrationSpec`s with identical field values always produce the
    same `calibration_id`, regardless of process, machine, or wall-clock
    time."""

    schema_version: int
    task: ObjectiveType
    positive_class_label: float
    source_experiment_id: str
    base_model_definition_identity: str
    dataset_content_id: str
    split_plan_fingerprint: str
    calibration_method_candidates: tuple[CalibrationMethodKind, ...]
    calibration_selection_metric: SelectionMetric
    calibration_tie_break_policy: CalibrationTieBreakPolicy
    minimum_calibration_sample_count: int
    minimum_samples_per_class: int
    inner_oof_policy: InnerSplitConfig
    threshold_spec: ThresholdSpec
    abstention_spec: AbstentionSpec
    confidence_spec: ConfidenceSpec
    uncertainty_spec: UncertaintySpec
    probability_clipping: ProbabilityClippingPolicy
    reliability_binning_specs: tuple[ReliabilityBinningSpec, ...]
    seed: int
    determinism_policy: DeterminismPolicy
    source_optimization_id: str | None = None
    bin_support_minimum_samples: int = 20
    """The reliability-bin sample count at or above which `calibration.
    confidence`'s `calibration_bin_support` component and `calibration.
    uncertainty`'s `bin_support_uncertainty_component` treat a bin's
    empirical rate as fully supported (Section 15/16). Explicit and
    identity-bearing -- unlike an internal fitting hyperparameter, this
    value materially changes persisted confidence/uncertainty OUTPUTS for
    otherwise-identical inputs, so two `CalibrationSpec`s that differ only
    here must never collide on the same `calibration_id` (see
    `compute_calibration_identity`)."""

    def __post_init__(self) -> None:
        if self.bin_support_minimum_samples < 1:
            raise CalibrationValidationError(
                f"CalibrationSpec.bin_support_minimum_samples must be >= 1, got {self.bin_support_minimum_samples}"
            )
        if self.task not in _SUPPORTED_TASKS:
            raise CalibrationValidationError(
                f"CalibrationSpec.task must be one of {[t.value for t in _SUPPORTED_TASKS]}, got {self.task.value!r}"
            )
        if self.task not in _FULLY_SUPPORTED_TASKS:
            raise CalibrationValidationError(
                f"CalibrationSpec.task={self.task.value!r} is not yet a fully supported reference implementation "
                f"(only {[t.value for t in _FULLY_SUPPORTED_TASKS]} is) -- failing closed rather than running an "
                "undertested code path (Milestone 4E Section 2: 'Fail closed for unsupported task/model combinations')"
            )
        if self.positive_class_label != _POSITIVE_CLASS_LABEL:
            raise CalibrationValidationError(
                f"CalibrationSpec.positive_class_label must equal {_POSITIVE_CLASS_LABEL!r} (this platform's fixed "
                f"binary-classification convention, matching ml.metrics), got {self.positive_class_label!r}"
            )
        if not is_valid_sha256_hex(self.source_experiment_id):
            raise CalibrationValidationError(
                f"CalibrationSpec.source_experiment_id must be a 64-character lowercase hex SHA-256 digest, "
                f"got {self.source_experiment_id!r}"
            )
        if self.source_optimization_id is not None and not is_valid_sha256_hex(self.source_optimization_id):
            raise CalibrationValidationError(
                f"CalibrationSpec.source_optimization_id must be a 64-character lowercase hex SHA-256 digest or "
                f"None, got {self.source_optimization_id!r}"
            )
        if not self.base_model_definition_identity:
            raise CalibrationValidationError("CalibrationSpec.base_model_definition_identity must not be empty")
        if not is_valid_sha256_hex(self.dataset_content_id):
            raise CalibrationValidationError(
                f"CalibrationSpec.dataset_content_id must be a 64-character lowercase hex SHA-256 digest, "
                f"got {self.dataset_content_id!r}"
            )
        if not is_valid_sha256_hex(self.split_plan_fingerprint):
            raise CalibrationValidationError(
                f"CalibrationSpec.split_plan_fingerprint must be a 64-character lowercase hex SHA-256 digest, "
                f"got {self.split_plan_fingerprint!r}"
            )
        if not self.calibration_method_candidates:
            raise CalibrationValidationError("CalibrationSpec.calibration_method_candidates must not be empty")
        if len(set(self.calibration_method_candidates)) != len(self.calibration_method_candidates):
            raise CalibrationValidationError("CalibrationSpec.calibration_method_candidates must not contain duplicates")
        if CalibrationMethodKind.IDENTITY not in self.calibration_method_candidates:
            raise CalibrationValidationError(
                "CalibrationSpec.calibration_method_candidates must always include IDENTITY (Section 8: "
                "'The identity method must always be available as a baseline')"
            )
        if self.minimum_calibration_sample_count < 2:
            raise CalibrationValidationError(
                f"CalibrationSpec.minimum_calibration_sample_count must be >= 2, got {self.minimum_calibration_sample_count}"
            )
        if self.minimum_samples_per_class < 1:
            raise CalibrationValidationError(
                f"CalibrationSpec.minimum_samples_per_class must be >= 1, got {self.minimum_samples_per_class}"
            )
        if self.minimum_calibration_sample_count < 2 * self.minimum_samples_per_class:
            raise CalibrationValidationError(
                f"CalibrationSpec.minimum_calibration_sample_count ({self.minimum_calibration_sample_count}) must "
                f"be >= 2 * minimum_samples_per_class ({self.minimum_samples_per_class}) -- an impossible class "
                "requirement (binary classification needs both classes represented)"
            )
        if not self.reliability_binning_specs:
            raise CalibrationValidationError("CalibrationSpec.reliability_binning_specs must not be empty")
        if self.seed < 0:
            raise CalibrationValidationError(f"CalibrationSpec.seed must be >= 0, got {self.seed}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "task": self.task.value, "positive_class_label": self.positive_class_label,
            "source_experiment_id": self.source_experiment_id, "source_optimization_id": self.source_optimization_id,
            "base_model_definition_identity": self.base_model_definition_identity, "dataset_content_id": self.dataset_content_id,
            "split_plan_fingerprint": self.split_plan_fingerprint,
            "calibration_method_candidates": [m.value for m in self.calibration_method_candidates],
            "calibration_selection_metric": self.calibration_selection_metric.value,
            "calibration_tie_break_policy": self.calibration_tie_break_policy.value,
            "minimum_calibration_sample_count": self.minimum_calibration_sample_count,
            "minimum_samples_per_class": self.minimum_samples_per_class,
            "inner_oof_policy": self.inner_oof_policy.to_json_dict(), "threshold_spec": self.threshold_spec.to_json_dict(),
            "abstention_spec": self.abstention_spec.to_json_dict(), "confidence_spec": self.confidence_spec.to_json_dict(),
            "uncertainty_spec": self.uncertainty_spec.to_json_dict(), "probability_clipping": self.probability_clipping.to_json_dict(),
            "reliability_binning_specs": [b.to_json_dict() for b in self.reliability_binning_specs],
            "seed": self.seed, "determinism_policy": self.determinism_policy.value,
            "bin_support_minimum_samples": self.bin_support_minimum_samples,
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Everything except `schema_version` -- identical in spirit to
        `ExperimentSpec.to_identity_payload`/`OptimizationSpec.
        to_identity_payload`: `schema_version` describes the CODE version
        that can read this payload, not a scientific choice, so it is
        added separately (as `identity_schema_version`) by `compute_
        calibration_identity`, never mixed into the hashed content here."""
        payload = self.to_json_dict()
        del payload["schema_version"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CalibrationSpec:
        require_schema_version(raw, supported=CALIBRATION_SPEC_SCHEMA_VERSION, context="CalibrationSpec")
        return cls(
            schema_version=CALIBRATION_SPEC_SCHEMA_VERSION, task=ObjectiveType(raw["task"]),
            positive_class_label=float(str(raw["positive_class_label"])), source_experiment_id=str(raw["source_experiment_id"]),
            source_optimization_id=(None if raw.get("source_optimization_id") is None else str(raw["source_optimization_id"])),
            base_model_definition_identity=str(raw["base_model_definition_identity"]), dataset_content_id=str(raw["dataset_content_id"]),
            split_plan_fingerprint=str(raw["split_plan_fingerprint"]),
            calibration_method_candidates=tuple(
                CalibrationMethodKind(m) for m in as_json_list(raw["calibration_method_candidates"], field_name="calibration_method_candidates")
            ),
            calibration_selection_metric=SelectionMetric(raw["calibration_selection_metric"]),
            calibration_tie_break_policy=CalibrationTieBreakPolicy(raw["calibration_tie_break_policy"]),
            minimum_calibration_sample_count=int(str(raw["minimum_calibration_sample_count"])),
            minimum_samples_per_class=int(str(raw["minimum_samples_per_class"])),
            inner_oof_policy=InnerSplitConfig.from_json_dict(as_json_dict(raw["inner_oof_policy"], field_name="inner_oof_policy")),
            threshold_spec=ThresholdSpec.from_json_dict(as_json_dict(raw["threshold_spec"], field_name="threshold_spec")),
            abstention_spec=AbstentionSpec.from_json_dict(as_json_dict(raw["abstention_spec"], field_name="abstention_spec")),
            confidence_spec=ConfidenceSpec.from_json_dict(as_json_dict(raw["confidence_spec"], field_name="confidence_spec")),
            uncertainty_spec=UncertaintySpec.from_json_dict(as_json_dict(raw["uncertainty_spec"], field_name="uncertainty_spec")),
            probability_clipping=ProbabilityClippingPolicy.from_json_dict(as_json_dict(raw["probability_clipping"], field_name="probability_clipping")),
            reliability_binning_specs=tuple(
                ReliabilityBinningSpec.from_json_dict(as_json_dict(b, field_name="reliability_binning_specs[]"))
                for b in as_json_list(raw["reliability_binning_specs"], field_name="reliability_binning_specs")
            ),
            seed=int(str(raw["seed"])), determinism_policy=DeterminismPolicy(raw["determinism_policy"]),
            bin_support_minimum_samples=int(str(raw.get("bin_support_minimum_samples", 20))),
        )


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    schema_version: int
    calibration_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "calibration_id": self.calibration_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CalibrationIdentity:
        require_schema_version(raw, supported=1, context="CalibrationIdentity")
        return cls(schema_version=1, calibration_id=str(raw["calibration_id"]))


def compute_calibration_identity(spec: CalibrationSpec) -> CalibrationIdentity:
    """Pure function: `CalibrationSpec` -> `CalibrationIdentity`. Mirrors
    `experiment_identity.compute_experiment_identity`/`optimization.
    models.compute_optimization_identity` exactly."""
    from quant_platform.ml.fingerprints import fingerprint_json

    payload = dict(spec.to_identity_payload())
    payload["identity_schema_version"] = 1
    calibration_id = fingerprint_json(payload)
    return CalibrationIdentity(schema_version=1, calibration_id=calibration_id)


def calibration_inner_fold_seed(seed_configuration: SeedConfiguration, *, outer_fold_index: int, inner_fold_index: int) -> int:
    """Deterministic per-inner-fold seed for the FRESH base model trained
    to generate that inner fold's out-of-fold predictions -- branched
    from `SeedDomain.CALIBRATION` (already reserved in `ml.seeds`),
    exactly like `optimization.models`'s own `inner_fold_seed` branches
    from `SeedDomain.HYPERPARAMETER_SEARCH`."""
    branch = derive_seed(seed_configuration.master_seed, SeedDomain.CALIBRATION)
    outer_branch = derive_seed(branch, f"outer_fold:{outer_fold_index}")
    return derive_seed(outer_branch, f"inner_fold:{inner_fold_index}")


def calibration_outer_refit_seed(seed_configuration: SeedConfiguration, *, outer_fold_index: int) -> int:
    """Deterministic seed for the FINAL base-model refit on complete
    outer-train data -- its own distinct branch, never coupled to any
    one inner fold's seed."""
    branch = derive_seed(seed_configuration.master_seed, SeedDomain.CALIBRATION)
    return derive_seed(branch, f"outer_fold_refit:{outer_fold_index}")


__all__ = [
    "CALIBRATION_SPEC_SCHEMA_VERSION",
    "AbstentionSpec",
    "CalibrationIdentity",
    "CalibrationSpec",
    "ConfidenceSpec",
    "CostMatrix",
    "ProbabilityClippingPolicy",
    "ReliabilityBinningSpec",
    "ThresholdSpec",
    "UncertaintySpec",
    "compute_calibration_identity",
]
