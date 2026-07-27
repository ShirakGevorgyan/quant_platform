"""Pydantic configuration schemas for the leakage-safe feature-selection
and hyperparameter-optimization engine (Milestone 4D). Same conventions
as `config.ml_schemas`: frozen, `extra="forbid"`, a `.build()` factory per
schema turning validated config into the runtime object it describes.

WHY THIS CONFIG REFERENCES A PARENT EXPERIMENT CONFIG FILE, NEVER
RE-TYPED DATASET/SPLIT BINDINGS
--------------------------------------------------------------------------
`OptimizationConfig.experiment_config_path` points at the SAME
`MLExperimentConfig` JSON `prepare-experiment` already consumes.
`ml_cli.cmd_optimize` loads and builds that config exactly as
`prepare-experiment` does, then hands the resulting `ExperimentSpec` to
`OptimizationConfig.build()` -- `build_optimization_spec` derives
`dataset_binding`/`outer_split_binding`/`feature_universe_fingerprint`
from it directly (see `optimization.models`' own docstring). A human
never hand-copies a dataset id or split strategy into two separate config
files that could silently drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import JsonPrimitive
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.feature_selection import FeatureSelectionSpec, FeatureSelectionStrategy
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    OptimizationSpec,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
)
from quant_platform.optimization.search_space import (
    BooleanParameter,
    CategoricalParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    ParameterDefinition,
    SearchSpace,
    build_search_space,
    default_search_space_for_model,
)


class OptimizationInnerSplitConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["expanding_walk_forward", "rolling_walk_forward"] = "expanding_walk_forward"
    n_splits: int = Field(ge=1)
    test_size_fraction: float = Field(gt=0.0, lt=1.0)
    embargo_bars: int = Field(default=0, ge=0)
    max_train_size_fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    def build(self) -> InnerSplitConfig:
        return InnerSplitConfig(
            strategy=self.strategy, n_splits=self.n_splits, test_size_fraction=self.test_size_fraction,
            embargo_bars=self.embargo_bars, max_train_size_fraction=self.max_train_size_fraction,
        )


class OptimizationFeatureSelectionConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["none", "variance_filter", "correlation_filter", "univariate", "model_native_importance", "stability_selection"]
    params: dict[str, JsonPrimitive] = Field(default_factory=dict)

    def build(self) -> FeatureSelectionSpec:
        return FeatureSelectionSpec(strategy=FeatureSelectionStrategy(self.strategy), params=dict(self.params))


class OptimizationSearchParameterSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["integer", "float", "categorical", "boolean", "fixed"]
    name: str = Field(min_length=1)
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[JsonPrimitive] | None = None
    value: JsonPrimitive = None

    def build(self) -> ParameterDefinition:
        if self.kind == "integer":
            if self.low is None or self.high is None:
                raise ValueError(f"search-space parameter {self.name!r}: kind='integer' requires low/high")
            return IntegerParameter(
                name=self.name, low=int(self.low), high=int(self.high), step=(int(self.step) if self.step is not None else 1), log=self.log,
            )
        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"search-space parameter {self.name!r}: kind='float' requires low/high")
            return FloatParameter(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"search-space parameter {self.name!r}: kind='categorical' requires non-empty choices")
            return CategoricalParameter(name=self.name, choices=tuple(self.choices))
        if self.kind == "boolean":
            return BooleanParameter(name=self.name)
        return FixedParameter(name=self.name, value=self.value)


class OptimizationSearchSpaceConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    use_default_for_model: bool = False
    """When `True`, `parameters` must be empty and `default_search_space_
    for_model(model_name)` is used instead -- never both at once, so a
    config can never silently ignore hand-declared parameters."""
    parameters: list[OptimizationSearchParameterSchema] = Field(default_factory=list)

    def build(self, *, model_name: str) -> SearchSpace:
        if self.use_default_for_model:
            if self.parameters:
                raise ValueError("search_space.use_default_for_model=True requires search_space.parameters to be empty")
            return default_search_space_for_model(model_name)
        if not self.parameters:
            raise ValueError("search_space.parameters must not be empty unless use_default_for_model=True")
        return build_search_space([p.build() for p in self.parameters])


class OptimizationPruningConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "median_stopping"] = "none"
    min_completed_inner_folds: int = Field(default=1, ge=1)

    def build(self) -> PruningConfig:
        return PruningConfig(kind=PruningKind(self.kind), min_completed_inner_folds=self.min_completed_inner_folds)


class OptimizationEarlyStoppingConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    patience: int | None = Field(default=None, ge=1)
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    final_round_policy: Literal["median_best_iteration", "fixed"] = "median_best_iteration"

    def build(self) -> EarlyStoppingConfig:
        return EarlyStoppingConfig(
            enabled=self.enabled, patience=self.patience, validation_fraction=self.validation_fraction,
            final_round_policy=self.final_round_policy,
        )


class OptimizationSeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    master_seed: int = Field(ge=0)

    def build(self) -> SeedConfiguration:
        return SeedConfiguration(master_seed=self.master_seed)


class OptimizationConfig(BaseModel):
    """The top-level config for one optimization run -- everything
    `quant_platform.ml_cli`'s `optimize`/`resume-optimization` commands
    need, all in one validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    experiment_config_path: Path
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    inner_split: OptimizationInnerSplitConfigSchema
    feature_selection: OptimizationFeatureSelectionConfigSchema
    search_space: OptimizationSearchSpaceConfigSchema
    sampler: Literal["tpe", "random"] = "tpe"
    pruning: OptimizationPruningConfigSchema = Field(default_factory=OptimizationPruningConfigSchema)
    early_stopping: OptimizationEarlyStoppingConfigSchema = Field(default_factory=OptimizationEarlyStoppingConfigSchema)
    max_trials: int = Field(gt=0)
    min_successful_inner_folds: int = Field(default=1, ge=1)
    seeds: OptimizationSeedConfig
    timeout_seconds: int | None = Field(default=None, ge=1)
    max_failed_trials: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    def build(self, *, experiment: ExperimentSpec, parent_experiment_id: str, model_registry: ModelRegistry) -> OptimizationSpec:
        return build_optimization_spec(
            experiment=experiment, parent_experiment_id=parent_experiment_id, model_name=self.model_name,
            model_version=self.model_version, primary_metric=self.primary_metric,
            inner_split_config=self.inner_split.build(), feature_selection_spec=self.feature_selection.build(),
            search_space=self.search_space.build(model_name=self.model_name), sampler_kind=SamplerKind(self.sampler),
            pruning_config=self.pruning.build(), early_stopping_config=self.early_stopping.build(),
            max_trials=self.max_trials, min_successful_inner_folds=self.min_successful_inner_folds,
            seed_configuration=self.seeds.build(), timeout_seconds=self.timeout_seconds,
            max_failed_trials=self.max_failed_trials, tags=tuple(self.tags), notes=self.notes,
            model_registry=model_registry,
        )


__all__ = [
    "OptimizationConfig",
    "OptimizationEarlyStoppingConfigSchema",
    "OptimizationFeatureSelectionConfigSchema",
    "OptimizationInnerSplitConfigSchema",
    "OptimizationPruningConfigSchema",
    "OptimizationSearchParameterSchema",
    "OptimizationSearchSpaceConfigSchema",
    "OptimizationSeedConfig",
]
