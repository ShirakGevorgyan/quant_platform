"""Pydantic configuration schemas for the ML core infrastructure platform
(Milestone 4A). Same conventions as `config.feature_schemas`/
`config.historical_schemas`: frozen, `extra="forbid"`, a `.build()`
factory per schema turning validated config into the runtime object it
describes.

WHY THIS CONFIG DOES NOT ASK FOR `content_id`/`feature_registry_fingerprint`
DIRECTLY
--------------------------------------------------------------------------
`MLDatasetConfig` only asks for `dataset_id` + `manifest_version` (a
human picks WHICH research dataset version to bind to) -- the actual
`content_id`, `feature_versions`, `feature_registry_fingerprint`,
`preprocessing_definition`, and `fitted_preprocessing_fingerprint` are
all derived live from the loaded `ResearchDatasetManifest` by
`ml_cli.py`, never hand-typed into a config file. A human mistyping a
64-character hex hash is exactly the kind of avoidable error this
platform's "derive, don't hand-copy" philosophy exists to prevent.

THIS MILESTONE TRAINS NO MODEL -- THE EXAMPLE CONFIG REGISTERS ONLY THE
TEST-ONLY MODEL
--------------------------------------------------------------------------
`MLModelConfig.name`/`.version` are validated against whatever
`ModelRegistry` the CLI builds -- in this milestone, that registry only
ever contains `ml.testing.ConstantTestModelFactory` under an explicitly
test-labeled name. No config shipped with this milestone claims or
implies trading profitability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.ml.models import (
    LabelBinding,
    LabelType,
    ModelHyperparameters,
    ObjectiveType,
    SplitBinding,
)
from quant_platform.ml.seeds import SeedConfiguration


class MLDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    manifest_version: str = Field(min_length=1)
    """Pinned to an EXACT research dataset manifest version -- never
    "latest" implicitly, so preparing the same config twice always binds
    to the same dataset content."""
    research_storage_root: Path
    feature_names: list[str] | None = None
    """`None` means "use every feature the research dataset manifest
    records, in its recorded order." A caller may instead supply an
    explicit ORDERED subset -- order is preserved exactly as given."""


class MLLabelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    horizon_bars: int = Field(gt=0)
    label_type: Literal["continuous", "binary", "multiclass"]
    params: dict[str, float] = Field(default_factory=dict)

    def build(self) -> LabelBinding:
        return LabelBinding(
            name=self.name, kind=self.kind, horizon_bars=self.horizon_bars,
            label_type=LabelType(self.label_type), params=dict(self.params),
        )


class MLSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)

    def build(self) -> SplitBinding:
        return SplitBinding(strategy=self.strategy, params=dict(self.params))


class MLModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    objective: Literal["binary_classification", "multiclass_classification", "regression"]
    hyperparameters: dict[str, float | int | str | bool] = Field(default_factory=dict)

    def build_objective(self) -> ObjectiveType:
        return ObjectiveType(self.objective)

    def build_hyperparameters(self) -> ModelHyperparameters:
        return ModelHyperparameters(values=dict(self.hyperparameters))


class MLSeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    master_seed: int = Field(ge=0)

    def build(self) -> SeedConfiguration:
        return SeedConfiguration(master_seed=self.master_seed)


class MLExperimentConfig(BaseModel):
    """The top-level config for one experiment preparation --
    everything `quant_platform.ml_cli`'s `prepare-experiment`/
    `validate-experiment` commands need, all in one validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    dataset: MLDatasetConfig
    label: MLLabelConfig
    split: MLSplitConfig
    model: MLModelConfig
    seeds: MLSeedConfig
    primary_metric: str = Field(min_length=1)
    environment_requirements: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


__all__ = [
    "MLDatasetConfig",
    "MLExperimentConfig",
    "MLLabelConfig",
    "MLModelConfig",
    "MLSeedConfig",
    "MLSplitConfig",
]
