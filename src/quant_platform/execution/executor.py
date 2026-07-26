"""`FoldExecutor`: the pluggable "how do we actually run one fold"
contract (Milestone 4B), and `DeterministicFoldExecutor`, the ONE
concrete implementation this milestone ships.

WHY THIS EXECUTOR CALLS REAL `fit`/`predict`, AND WHY THAT IS STILL "NO
MODEL TRAINING" IN THE SENSE THIS MILESTONE FORBIDS
--------------------------------------------------------------------------
The milestone explicitly forbids LightGBM/XGBoost/CatBoost/Random Forest/
Logistic Regression/Elastic Net/neural networks/feature selection/
hyperparameter optimization/SHAP/calibration/ensembles -- REAL predictive
algorithms and REAL model-selection machinery. It does not forbid
exercising `ml.interfaces`' `TrainableModel.fit`/`FittedModel.predict`
contract end-to-end, and doing so is exactly how Milestone 4A validated
its OWN artifact/serialization plumbing (via `ml.testing.
ConstantTestModel`, a model that always predicts the training label mean/
positive rate regardless of input -- deliberately incapable of learning
anything). `DeterministicFoldExecutor` continues that same philosophy:
it calls the REAL `ModelFactory.create` -> `TrainableModel.fit` ->
`FittedModel.predict` -> `ModelSerializer.serialize` pipeline, so the
walk-forward engine's orchestration is genuinely, end-to-end exercised --
but the ONLY model ever registered anywhere in this codebase is the
deterministic test-only one. "No model training yet" (Section 5) means
no REAL predictive algorithm trains; "metrics placeholder" (Section 7)
means this executor computes NO real performance score from predictions
vs. truth (that requires choosing a metric definition, itself a modeling
decision explicitly out of scope) -- `FoldExecutionOutcome.metrics` is
always empty here, reserved for a future milestone to populate.

WHY `execute()` DOES NOT BUILD THE FINAL `FoldResult` ITSELF
--------------------------------------------------------------------------
A `FoldExecutor` has no need to know about `execution.splitters.Fold`'s
time bounds/row counts -- the RUNNER already holds those (from the
`FoldPlan` it built) and is responsible for combining them with this
executor's OUTCOME (artifacts written, metrics, duration) into the final,
persisted `FoldResult` -- including deciding `FoldStatus.FAILED` and a
`failure_reason` when `execute()` raises, since only the runner knows
whether a given failure should be treated as recoverable.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from quant_platform.execution.context import FoldExecutionContext
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory, ModelSerializer
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    JsonPrimitive,
    ModelHyperparameters,
    ObjectiveType,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import canonical_json_bytes
from quant_platform.ml.seeds import SeedConfiguration


@dataclass(frozen=True, slots=True)
class FoldData:
    """Already-sliced, already-column-selected data for one fold. Feature
    frames carry ONLY the declared feature columns (in schema order);
    label series carry only the target."""

    train_features: pd.DataFrame
    train_labels: pd.Series
    test_features: pd.DataFrame
    test_labels: pd.Series
    validation_features: pd.DataFrame | None = None
    validation_labels: pd.Series | None = None

    def __post_init__(self) -> None:
        if len(self.train_features) != len(self.train_labels):
            raise ValueError("FoldData: train_features and train_labels must have matching length")
        if len(self.test_features) != len(self.test_labels):
            raise ValueError("FoldData: test_features and test_labels must have matching length")
        if (self.validation_features is None) != (self.validation_labels is None):
            raise ValueError("FoldData: validation_features and validation_labels must both be set or both be None")
        if self.validation_features is not None and len(self.validation_features) != len(self.validation_labels):  # type: ignore[arg-type]
            raise ValueError("FoldData: validation_features and validation_labels must have matching length")


@dataclass(frozen=True, slots=True)
class FoldExecutionOutcome:
    artifact_references: tuple[ArtifactReference, ...]
    duration_seconds: float
    metrics: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError(f"FoldExecutionOutcome.duration_seconds must be >= 0, got {self.duration_seconds}")
        validate_json_primitive_mapping(self.metrics, field_name="FoldExecutionOutcome.metrics")


@runtime_checkable
class FoldExecutor(Protocol):
    """Pluggable "run one fold" contract. A future milestone that adds a
    real model swaps in a different `FoldExecutor` implementation; the
    runner's orchestration (splitting, validation, locking, resume,
    artifact/manifest bookkeeping) needs no change."""

    def execute(
        self,
        context: FoldExecutionContext,
        *,
        model_factory: ModelFactory,
        hyperparameters: ModelHyperparameters,
        feature_schema: FeatureSchema,
        objective: ObjectiveType,
        serializer: ModelSerializer,
        data: FoldData,
    ) -> FoldExecutionOutcome: ...


class DeterministicFoldExecutor:
    """The ONE `FoldExecutor` this milestone ships. See module docstring
    for why calling real `fit`/`predict` on the registry's sole,
    deterministic test-only model is within scope."""

    def execute(
        self,
        context: FoldExecutionContext,
        *,
        model_factory: ModelFactory,
        hyperparameters: ModelHyperparameters,
        feature_schema: FeatureSchema,
        objective: ObjectiveType,
        serializer: ModelSerializer,
        data: FoldData,
    ) -> FoldExecutionOutcome:
        started = time.perf_counter()
        model = model_factory.create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
        fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=context.seed))
        predictions = fitted.predict(data.test_features)

        model_ref = context.artifact_store.write_artifact(serializer.serialize(fitted), category=ArtifactCategory.MODEL)
        predictions_ref = context.artifact_store.write_artifact(
            canonical_json_bytes({"schema_version": 1, "predictions": [float(p) for p in predictions]}),
            category=ArtifactCategory.PREDICTIONS,
        )
        artifact_references = (model_ref, predictions_ref)

        duration = time.perf_counter() - started
        return FoldExecutionOutcome(artifact_references=artifact_references, duration_seconds=duration, metrics={})


__all__ = ["DeterministicFoldExecutor", "FoldData", "FoldExecutionOutcome", "FoldExecutor"]
