"""Pydantic configuration schemas for the leakage-safe calibration,
thresholding, confidence, and uncertainty framework (Milestone 4E). Same
conventions as `config.optimization_schemas`: frozen, `extra="forbid"`, a
`.build()` factory per schema turning validated config into the runtime
object it describes.

WHY THIS CONFIG REFERENCES AN EXISTING `source_experiment_id`, NEVER A
FRESH EXPERIMENT CONFIG FILE
--------------------------------------------------------------------------
Unlike `OptimizationConfig` (which prepares hyperparameter/feature search
against a parent experiment), `CalibrationConfig` post-processes an
ALREADY-PREPARED experiment's (optionally, an already-COMPLETED
optimization's) raw outputs -- there is no new experiment to build here.
`CalibrationConfig.build()` therefore takes an already-loaded
`ExperimentManifest`/`ModelDefinition` and derives `CalibrationSpec`'s
identity-relevant `dataset_content_id`/`split_plan_fingerprint`/
`base_model_definition_identity` fields directly from them -- never
re-typed by a human into this config file, where they could silently
drift from the actual bound experiment (see `calibration.runner.
resolve_calibration_inputs`'s identical cross-check at RUN time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    CostMatrix,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
)
from quant_platform.config.optimization_schemas import OptimizationInnerSplitConfigSchema
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.models import ObjectiveType
from quant_platform.ml.registry import ModelDefinition


class CalibrationCostMatrixSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    false_positive_cost: float = Field(ge=0.0)
    false_negative_cost: float = Field(ge=0.0)
    true_positive_cost: float = Field(default=0.0, ge=0.0)
    true_negative_cost: float = Field(default=0.0, ge=0.0)

    def build(self) -> CostMatrix:
        return CostMatrix(
            false_positive_cost=self.false_positive_cost, false_negative_cost=self.false_negative_cost,
            true_positive_cost=self.true_positive_cost, true_negative_cost=self.true_negative_cost,
        )


class CalibrationThresholdConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Literal[
        "fixed", "balanced_accuracy", "f1", "matthews_corrcoef", "youden_j",
        "min_precision_max_recall", "min_recall_max_precision", "cost_sensitive",
    ]
    fixed_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    min_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_matrix: CalibrationCostMatrixSchema | None = None
    infeasible_fallback_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_grid_size: int = Field(default=101, ge=2)

    def build(self) -> ThresholdSpec:
        return ThresholdSpec(
            policy=ThresholdPolicyKind(self.policy), fixed_threshold=self.fixed_threshold,
            min_precision=self.min_precision, min_recall=self.min_recall,
            cost_matrix=(None if self.cost_matrix is None else self.cost_matrix.build()),
            infeasible_fallback_threshold=self.infeasible_fallback_threshold, candidate_grid_size=self.candidate_grid_size,
        )


class CalibrationAbstentionConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Literal["none", "symmetric_band", "min_confidence", "max_uncertainty", "class_specific_boundaries"] = "none"
    band_half_width: float | None = Field(default=None, gt=0.0, lt=0.5)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    negative_upper_bound: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_lower_bound: float | None = Field(default=None, ge=0.0, le=1.0)

    def build(self) -> AbstentionSpec:
        return AbstentionSpec(
            policy=AbstentionPolicyKind(self.policy), band_half_width=self.band_half_width, min_confidence=self.min_confidence,
            max_uncertainty=self.max_uncertainty, negative_upper_bound=self.negative_upper_bound, positive_lower_bound=self.positive_lower_bound,
        )


class CalibrationConfidenceConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    very_low_max: float = Field(gt=0.0, lt=1.0)
    low_max: float = Field(gt=0.0, lt=1.0)
    medium_max: float = Field(gt=0.0, lt=1.0)
    high_max: float = Field(gt=0.0, lt=1.0)
    component_weights: dict[str, float] = Field(default_factory=dict)

    def build(self) -> ConfidenceSpec:
        return ConfidenceSpec(
            very_low_max=self.very_low_max, low_max=self.low_max, medium_max=self.medium_max, high_max=self.high_max,
            component_weights=dict(self.component_weights),
        )


class CalibrationUncertaintyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    components: list[Literal["entropy", "margin", "model_disagreement", "calibrator_disagreement", "bin_support"]] = Field(min_length=1)
    aggregation: Literal["mean", "max"] = "mean"

    def build(self) -> UncertaintySpec:
        return UncertaintySpec(components=tuple(self.components), aggregation=self.aggregation)


class CalibrationClippingConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    epsilon: float = Field(default=1e-6, gt=0.0, lt=0.5)

    def build(self) -> ProbabilityClippingPolicy:
        return ProbabilityClippingPolicy(enabled=self.enabled, epsilon=self.epsilon)


class CalibrationReliabilityBinningConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["equal_width", "equal_frequency"] = "equal_width"
    n_bins: int = Field(default=10, ge=1)

    def build(self) -> ReliabilityBinningSpec:
        return ReliabilityBinningSpec(strategy=BinningStrategy(self.strategy), n_bins=self.n_bins)


class CalibrationSeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    master_seed: int = Field(ge=0)


class CalibrationConfig(BaseModel):
    """The top-level config for one calibration run -- everything
    `quant_platform.ml_cli`'s `run-calibration`/`resume-calibration`
    commands need, all in one validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    research_storage_root: Path
    """Unlike `OptimizationConfig.experiment_config_path`, this config
    binds directly to an ALREADY-prepared `source_experiment_id` -- there
    is no fresh `MLExperimentConfig` to read a research storage root from,
    so it is declared explicitly here instead."""
    source_experiment_id: str = Field(min_length=64, max_length=64)
    source_optimization_id: str | None = Field(default=None, min_length=64, max_length=64)
    calibration_method_candidates: list[Literal["identity", "platt", "isotonic", "beta"]] = Field(min_length=1)
    calibration_selection_metric: Literal["log_loss", "brier_score", "expected_calibration_error", "maximum_calibration_error"] = "log_loss"
    minimum_calibration_sample_count: int = Field(default=30, ge=2)
    minimum_samples_per_class: int = Field(default=5, ge=1)
    inner_oof_policy: OptimizationInnerSplitConfigSchema
    threshold: CalibrationThresholdConfigSchema
    abstention: CalibrationAbstentionConfigSchema = Field(default_factory=CalibrationAbstentionConfigSchema)
    confidence: CalibrationConfidenceConfigSchema
    uncertainty: CalibrationUncertaintyConfigSchema
    probability_clipping: CalibrationClippingConfigSchema = Field(default_factory=CalibrationClippingConfigSchema)
    reliability_binning: list[CalibrationReliabilityBinningConfigSchema] = Field(min_length=1)
    seeds: CalibrationSeedConfig
    determinism_policy: Literal["strict", "warn"] = "strict"
    bin_support_minimum_samples: int = Field(default=20, ge=1)

    def build(self, *, experiment_spec: ExperimentSpec, model_definition: ModelDefinition) -> CalibrationSpec:
        if experiment_spec.objective is not ObjectiveType.BINARY_CLASSIFICATION:
            raise ValueError(
                f"CalibrationConfig: source experiment objective={experiment_spec.objective.value!r} is not "
                "supported -- only binary_classification is a fully supported reference implementation (Section 2)"
            )
        return CalibrationSpec(
            schema_version=1, task=experiment_spec.objective, positive_class_label=1.0,
            source_experiment_id=self.source_experiment_id, source_optimization_id=self.source_optimization_id,
            base_model_definition_identity=model_definition.fingerprint(),
            dataset_content_id=experiment_spec.dataset_binding.content_id,
            split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
            calibration_method_candidates=tuple(CalibrationMethodKind(m) for m in self.calibration_method_candidates),
            calibration_selection_metric=SelectionMetric(self.calibration_selection_metric),
            calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
            minimum_calibration_sample_count=self.minimum_calibration_sample_count,
            minimum_samples_per_class=self.minimum_samples_per_class, inner_oof_policy=self.inner_oof_policy.build(),
            threshold_spec=self.threshold.build(), abstention_spec=self.abstention.build(), confidence_spec=self.confidence.build(),
            uncertainty_spec=self.uncertainty.build(), probability_clipping=self.probability_clipping.build(),
            reliability_binning_specs=tuple(b.build() for b in self.reliability_binning),
            seed=self.seeds.master_seed, determinism_policy=DeterminismPolicy(self.determinism_policy),
            bin_support_minimum_samples=self.bin_support_minimum_samples,
        )


__all__ = [
    "CalibrationAbstentionConfigSchema",
    "CalibrationClippingConfigSchema",
    "CalibrationConfidenceConfigSchema",
    "CalibrationConfig",
    "CalibrationCostMatrixSchema",
    "CalibrationReliabilityBinningConfigSchema",
    "CalibrationSeedConfig",
    "CalibrationThresholdConfigSchema",
    "CalibrationUncertaintyConfigSchema",
]
