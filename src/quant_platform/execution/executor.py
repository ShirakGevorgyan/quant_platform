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

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import TrainingDataValidationError
from quant_platform.execution.context import FoldExecutionContext
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory, ModelSerializer, ProbabilisticPredictor
from quant_platform.ml.metrics import compute_metrics
from quant_platform.ml.model_validation import validate_training_data
from quant_platform.ml.model_zoo.common import feature_schema_fingerprint
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    JsonPrimitive,
    ModelHyperparameters,
    ObjectiveType,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.training_metadata import TrainingMetadata


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


class MetricsFoldExecutor:
    """The production `FoldExecutor` Milestone 4C ships -- a NEW,
    additional implementation of the SAME `FoldExecutor` Protocol
    `DeterministicFoldExecutor` already implements. That class is left
    completely UNCHANGED (still used wherever a caller wants the
    lighter-weight, metrics-free path, and by every pre-existing 4B
    test); nothing in `execution.runner`/`ExecutionRunner` changes
    either -- this is exactly the extension point this module's own
    docstring names: "A future milestone that adds a real model swaps in
    a different FoldExecutor implementation; the runner's orchestration
    ... needs no change."

    On top of the fit -> predict -> serialize pipeline
    `DeterministicFoldExecutor` already exercises, this executor:

      1. Runs `ml.model_validation.validate_training_data` against the
         freshly-created (not yet fit) model's OWN metadata and the
         fold's actual train partition immediately after `ModelFactory.
         create` -- raising `TrainingDataValidationError` (never a raw
         library exception) if the report is not ready, BEFORE `fit` is
         ever called. `execution.runner` needs no change to handle this:
         it already converts ANY exception raised during one fold into
         that fold's `FoldResult(status=FAILED, failure_reason=...)`,
         continuing to the remaining folds -- exactly the treatment a
         bad-data fold should get.
      2. Calls `predict_proba` too, when the fitted model supports it for
         this objective, persisting the full probability matrix as its
         own `ArtifactCategory.PROBABILITIES` artifact (declared, but
         unused, since Milestone 4A).
      3. Computes REAL performance metrics (`ml.metrics.compute_metrics`)
         from the fold's actual test predictions vs. test labels --
         `FoldExecutionOutcome.metrics` is no longer the empty
         placeholder `DeterministicFoldExecutor` still returns (see that
         class's own docstring: this was always "reserved for a future
         milestone to populate" -- this milestone).
      4. Builds and persists a `TrainingMetadata` provenance record
         (training duration, seed, library name/version, hyperparameters,
         feature-schema fingerprint, dataset content id, experiment id)
         as its own `ArtifactCategory.TRAINING_METADATA` artifact.
    """

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

        validation_report = validate_training_data(
            metadata=model.metadata, features=data.train_features, labels=data.train_labels,
            preprocessing_binding=context.manifest.spec.preprocessing_binding,
        )
        if not validation_report.is_ready:
            blocking = [*validation_report.criticals, *validation_report.errors]
            summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            raise TrainingDataValidationError(
                f"Fold {context.fold_index}: training data failed pre-fit validation: {summary}",
                context={"fold_index": context.fold_index, "experiment_id": context.experiment_id},
            )

        fit_started = time.perf_counter()
        fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=context.seed))
        training_duration = time.perf_counter() - fit_started

        predictions = fitted.predict(data.test_features)
        artifact_references: list[ArtifactReference] = []

        model_ref = context.artifact_store.write_artifact(serializer.serialize(fitted), category=ArtifactCategory.MODEL)
        artifact_references.append(model_ref)

        predictions_ref = context.artifact_store.write_artifact(
            canonical_json_bytes({"schema_version": 1, "predictions": [float(p) for p in predictions]}),
            category=ArtifactCategory.PREDICTIONS,
        )
        artifact_references.append(predictions_ref)

        y_true = data.test_labels.to_numpy(dtype="float64")
        y_proba_positive: np.ndarray | None = None
        can_predict_proba = (
            objective is not ObjectiveType.REGRESSION
            and fitted.metadata.capabilities.supports_predict_proba
            and isinstance(fitted, ProbabilisticPredictor)
        )
        if can_predict_proba:
            assert isinstance(fitted, ProbabilisticPredictor)  # narrows for type-checking; already checked above
            proba = fitted.predict_proba(data.test_features)
            class_labels = list(fitted.class_labels)
            positive_index = class_labels.index(1)
            y_proba_positive = proba[:, positive_index]
            probabilities_ref = context.artifact_store.write_artifact(
                canonical_json_bytes({
                    "schema_version": 1, "class_labels": [str(c) for c in class_labels],
                    "probabilities": [[float(v) for v in row] for row in proba],
                }),
                category=ArtifactCategory.PROBABILITIES,
            )
            artifact_references.append(probabilities_ref)

        metric_report = compute_metrics(objective, y_true, predictions, y_proba_positive)

        training_metadata = TrainingMetadata(
            schema_version=1, experiment_id=context.experiment_id, fold_index=context.fold_index,
            model_name=fitted.metadata.name, model_version=fitted.metadata.version,
            library_name=fitted.metadata.capabilities.library_name,
            library_version=str(
                context.environment.package_versions.get(fitted.metadata.capabilities.library_name) or "unknown"
            ),
            seed=context.seed, training_duration_seconds=training_duration,
            feature_schema_fingerprint=feature_schema_fingerprint(feature_schema),
            dataset_content_id=context.dataset_content_id, fitted_at=format_utc_timestamp(utc_now()),
            hyperparameters=dict(hyperparameters.values),
        )
        training_metadata_ref = context.artifact_store.write_artifact(
            canonical_json_bytes(training_metadata.to_json_dict()), category=ArtifactCategory.TRAINING_METADATA,
        )
        artifact_references.append(training_metadata_ref)

        duration = time.perf_counter() - started
        return FoldExecutionOutcome(artifact_references=tuple(artifact_references), duration_seconds=duration, metrics=metric_report.values)


__all__ = ["DeterministicFoldExecutor", "FoldData", "FoldExecutionOutcome", "FoldExecutor", "MetricsFoldExecutor"]
