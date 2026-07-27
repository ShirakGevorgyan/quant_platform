from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import TrainingDataValidationError
from quant_platform.execution.context import FoldExecutionContext
from quant_platform.execution.executor import FoldData, MetricsFoldExecutor
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.model_zoo import baselines as b
from quant_platform.ml.model_zoo import default_serializer_registry, register_default_models
from quant_platform.ml.models import ArtifactCategory, ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.tracking import ExperimentEventStore
from quant_platform.ml.training_metadata import TrainingMetadata

EID = "a" * 64
_SCHEMA = FeatureSchema(feature_names=("f1", "f2"))


def _context(tmp_path: Path, *, seed: int = 1) -> FoldExecutionContext:
    from tests.unit.execution.conftest import (
        build_registry,
        make_experiment_spec_kwargs,
        write_synthetic_research_dataset,
    )

    from quant_platform.ml.experiment_manager import ExperimentPreparer
    from quant_platform.ml.experiment_spec import ExperimentSpec

    dataset_manifest, _research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
    manifest = preparer.prepare(spec)
    return FoldExecutionContext(
        experiment_id=manifest.identity.experiment_id, fold_index=2, split_id="fold:2",
        dataset_content_id=dataset_manifest.content_id, manifest=manifest, seed=seed,
        environment=capture_environment_snapshot(), artifact_store=MLArtifactStore(tmp_path / "ml"),
        event_store=ExperimentEventStore(tmp_path / "ml"), artifacts_root=tmp_path / "ml",
        started_at=format_utc_timestamp(utc_now()),
    )


def _clf_data(n: int = 40) -> FoldData:
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    train_labels = pd.Series((train_features["f1"] + 0.3 * train_features["f2"] > 0).astype(int))
    test_features = pd.DataFrame({"f1": rng.normal(size=10), "f2": rng.normal(size=10)})
    test_labels = pd.Series((test_features["f1"] > 0).astype(int))
    return FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)


def _reg_data(n: int = 40) -> FoldData:
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    train_labels = pd.Series(train_features["f1"] * 2 - train_features["f2"])
    test_features = pd.DataFrame({"f1": rng.normal(size=10), "f2": rng.normal(size=10)})
    test_labels = pd.Series(test_features["f1"] * 2 - test_features["f2"])
    return FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)


@dataclass(frozen=True, slots=True)
class _ReversedClassOrderFittedModel:
    """A deliberately adversarial, test-only fitted model whose
    `class_labels` is REVERSED (`(1, 0)`, positive class FIRST) --
    proves `MetricsFoldExecutor` extracts the positive-class probability
    column via `class_labels.index(1)`, never by blindly assuming
    "column 1 is always the positive class"."""

    metadata: ModelMetadata

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return (1, 0)

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        return (features["f1"].to_numpy() > 0).astype("float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        # Column 0 = P(class label class_labels[0] == 1) -- perfectly
        # correlated with the SAME f1>0 rule predict()/test_labels use.
        p_class_1 = np.where(features["f1"].to_numpy() > 0, 0.95, 0.05)
        return np.column_stack([p_class_1, 1.0 - p_class_1])


@dataclass(frozen=True, slots=True)
class _ReversedClassOrderTrainableModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: object) -> _ReversedClassOrderFittedModel:
        return _ReversedClassOrderFittedModel(metadata=self.metadata)


class _ReversedClassOrderModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), supports_predict_proba=True, library_name="test-stub")

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> _ReversedClassOrderTrainableModel:
        metadata = ModelMetadata(
            name="reversed_class_order_stub", version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return _ReversedClassOrderTrainableModel(metadata=metadata)


class _ReversedClassOrderModelSerializer:
    def serialize(self, model: _ReversedClassOrderFittedModel) -> bytes:
        return b"{}"


class TestPositiveClassSelectionUsesFittedClassMapping:
    def test_reversed_class_labels_still_selects_the_correct_positive_class_column(self, tmp_path: Path) -> None:
        data = _clf_data()
        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=_ReversedClassOrderModelFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=_ReversedClassOrderModelSerializer(), data=data,
        )
        # `class_labels == (1, 0)` means column 0 is P(class=1). If the
        # executor correctly reads `class_labels.index(1) == 0`, it
        # extracts column 0 (perfectly correlated with test_labels via
        # the same f1>0 rule) and roc_auc is near 1.0. If it instead
        # assumed "column 1 is always positive" (the bug this guards
        # against), it would extract `1 - p_class_1` -- an inverted
        # signal giving roc_auc near 0.0.
        assert outcome.metrics["roc_auc"] > 0.9


class TestMetricsUseTestPartitionOnly:
    """"Prove that fold metrics use out-of-sample test labels and
    predictions; training labels are never used to compute reported fold
    performance." Uses `ConstantPredictor` (predicts a fixed, declared
    constant regardless of ANY data) so the correct metric value is
    computable analytically from `test_labels` ALONE -- `train_labels` is
    deliberately set to a DIFFERENT, easily-distinguishable distribution;
    if it leaked into the computed metric at all, the assertion below
    would fail."""

    def test_classification_accuracy_reflects_test_labels_not_train_labels(self, tmp_path: Path) -> None:
        train_features = pd.DataFrame({"f1": [0.0] * 20, "f2": [0.0] * 20})
        train_labels = pd.Series([0.0] * 19 + [1.0])  # train: overwhelmingly 0.0 (never constant, to pass the separate constant-labels gate)
        test_features = pd.DataFrame({"f1": [0.0] * 10, "f2": [0.0] * 10})
        test_labels = pd.Series([1.0] * 7 + [0.0] * 3)  # test: 7 ones, 3 zeros
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)

        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=b.ConstantPredictorFactory(),
            hyperparameters=ModelHyperparameters(values={"constant": 1.0}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.ConstantPredictorSerializer(), data=data,
        )
        # Predictions are always 1.0 (the declared constant); accuracy
        # against the TEST labels (7 of 10 are 1.0) is exactly 0.7 -- if
        # TRAIN labels (all 0.0) had leaked in instead, accuracy would be
        # 0.0.
        assert outcome.metrics["accuracy"] == pytest.approx(0.7)

    def test_regression_mae_reflects_test_labels_not_train_labels(self, tmp_path: Path) -> None:
        train_features = pd.DataFrame({"f1": [0.0] * 20, "f2": [0.0] * 20})
        train_labels = pd.Series([100.0] * 20)  # train: constant, far from the declared predicted constant
        test_features = pd.DataFrame({"f1": [0.0] * 10, "f2": [0.0] * 10})
        test_labels = pd.Series([5.0] * 7 + [3.0] * 3)  # test: known, mixed values
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)

        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=b.ConstantPredictorFactory(),
            hyperparameters=ModelHyperparameters(values={"constant": 5.0}),
            feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
            serializer=b.ConstantPredictorSerializer(), data=data,
        )
        # Predictions are always 5.0; MAE against the TEST labels is
        # (|5-5|*7 + |5-3|*3) / 10 = 0.6 -- if TRAIN labels (all 100.0)
        # had leaked in, MAE would be 95.0.
        assert outcome.metrics["mae"] == pytest.approx(0.6)


class TestMetricsFoldExecutorClassification:
    def test_produces_all_four_artifact_categories(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        factory = b.MajorityPredictorFactory()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=factory, hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.MajorityPredictorSerializer(), data=data,
        )
        categories = [r.category for r in outcome.artifact_references]
        assert categories == [
            ArtifactCategory.MODEL, ArtifactCategory.PREDICTIONS, ArtifactCategory.PROBABILITIES, ArtifactCategory.TRAINING_METADATA,
        ]

    def test_computes_real_classification_metrics(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=b.MajorityPredictorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.MajorityPredictorSerializer(), data=data,
        )
        assert "accuracy" in outcome.metrics
        # `_clf_data()`'s fixed seed gives both classes in `test_labels`,
        # and `MajorityPredictorFactory` supports `predict_proba` -- so
        # ROC AUC/PR AUC are always actually computed here, never skipped.
        assert "roc_auc" in outcome.metrics
        assert "pr_auc" in outcome.metrics
        assert isinstance(outcome.metrics["accuracy"], float)

    def test_persisted_training_metadata_has_correct_provenance(self, tmp_path: Path) -> None:
        context = _context(tmp_path, seed=99)
        data = _clf_data()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=b.MajorityPredictorFactory(), hyperparameters=ModelHyperparameters(values={"unused": 1}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.MajorityPredictorSerializer(), data=data,
        )
        ref = next(r for r in outcome.artifact_references if r.category is ArtifactCategory.TRAINING_METADATA)
        raw = context.artifact_store.read_artifact(ref.content_hash)
        tm = TrainingMetadata.from_json_dict(json.loads(raw.decode("utf-8")))
        assert tm.experiment_id == context.experiment_id
        assert tm.fold_index == 2
        assert tm.seed == 99
        assert tm.model_name == "majority_predictor"
        assert tm.library_name == "quant-platform"
        assert tm.dataset_content_id == context.dataset_content_id
        assert tm.hyperparameters == {"unused": 1}
        assert tm.training_duration_seconds >= 0
        assert len(tm.feature_schema_fingerprint) == 64

    def test_persisted_probabilities_artifact_is_decodable(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=b.MajorityPredictorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.MajorityPredictorSerializer(), data=data,
        )
        ref = next(r for r in outcome.artifact_references if r.category is ArtifactCategory.PROBABILITIES)
        raw = json.loads(context.artifact_store.read_artifact(ref.content_hash).decode("utf-8"))
        assert raw["class_labels"] == ["0", "1"]
        assert len(raw["probabilities"]) == len(data.test_features)
        for row in raw["probabilities"]:
            assert pytest.approx(sum(row)) == 1.0


class TestMetricsFoldExecutorRegression:
    def test_no_probabilities_artifact_for_regression(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _reg_data()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=b.DummyMeanRegressorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
            serializer=b.DummyMeanRegressorSerializer(), data=data,
        )
        categories = [r.category for r in outcome.artifact_references]
        assert categories == [ArtifactCategory.MODEL, ArtifactCategory.PREDICTIONS, ArtifactCategory.TRAINING_METADATA]

    def test_computes_real_regression_metrics(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _reg_data()
        outcome = MetricsFoldExecutor().execute(
            context, model_factory=b.DummyMeanRegressorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
            serializer=b.DummyMeanRegressorSerializer(), data=data,
        )
        assert set(outcome.metrics) >= {"mae", "rmse", "mape"}


class TestMetricsFoldExecutorPreFitValidationGate:
    def test_constant_labels_rejected_before_fit_for_classification(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        constant_train_labels = pd.Series([1] * len(data.train_features))
        bad_data = FoldData(
            train_features=data.train_features, train_labels=constant_train_labels,
            test_features=data.test_features, test_labels=data.test_labels,
        )
        with pytest.raises(TrainingDataValidationError, match="constant_labels"):
            MetricsFoldExecutor().execute(
                context, model_factory=b.RandomPredictorFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=b.RandomPredictorSerializer(), data=bad_data,
            )

    def test_zero_training_samples_rejected_before_fit(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        empty_features = data.train_features.iloc[:0]
        empty_labels = data.train_labels.iloc[:0]
        bad_data = FoldData(
            train_features=empty_features, train_labels=empty_labels,
            test_features=data.test_features, test_labels=data.test_labels,
        )
        with pytest.raises(TrainingDataValidationError, match="zero_training_samples"):
            MetricsFoldExecutor().execute(
                context, model_factory=b.RandomPredictorFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=b.RandomPredictorSerializer(), data=bad_data,
            )

    def test_missing_values_rejected_before_fit_for_a_model_that_declares_no_support(self, tmp_path: Path) -> None:
        """`logistic_regression` declares `supports_missing_values=False`
        -- a fold's train partition with a NaN feature must be rejected
        by the pre-fit gate, never reaching sklearn's own (less
        actionable) `ValueError: Input X contains NaN`."""
        registry = register_default_models()
        definition = registry.get("logistic_regression", "1")
        context = _context(tmp_path)
        data = _clf_data()
        data.train_features.loc[0, "f1"] = np.nan
        with pytest.raises(TrainingDataValidationError, match="missing_values_unsupported"):
            MetricsFoldExecutor().execute(
                context, model_factory=definition.factory, hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=default_serializer_registry()[definition.serializer_id][0], data=data,
            )

    def test_no_fold_result_artifacts_written_when_validation_rejects(self, tmp_path: Path) -> None:
        """A rejected fold must not leave a partially-written MODEL/
        PREDICTIONS artifact behind -- the validation gate runs BEFORE
        `model.fit`/any `write_artifact` call, never after a partial
        write."""
        context = _context(tmp_path)
        data = _clf_data()
        constant_train_labels = pd.Series([0] * len(data.train_features))
        bad_data = FoldData(
            train_features=data.train_features, train_labels=constant_train_labels,
            test_features=data.test_features, test_labels=data.test_labels,
        )

        def _written_content_files() -> set[str]:
            content_root = context.artifact_store.root / "content"
            if not content_root.is_dir():
                return set()
            return {str(p.relative_to(content_root)) for p in content_root.rglob("*") if p.is_file()}

        before = _written_content_files()
        with pytest.raises(TrainingDataValidationError):
            MetricsFoldExecutor().execute(
                context, model_factory=b.MajorityPredictorFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=b.MajorityPredictorSerializer(), data=bad_data,
            )
        after = _written_content_files()
        assert after == before


class TestMetricsFoldExecutorDeterminism:
    def test_same_seed_produces_identical_model_artifact(self, tmp_path: Path) -> None:
        context1 = _context(tmp_path / "a", seed=11)
        context2 = _context(tmp_path / "b", seed=11)
        data = _clf_data()

        outcome1 = MetricsFoldExecutor().execute(
            context1, model_factory=b.RandomPredictorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.RandomPredictorSerializer(), data=data,
        )
        outcome2 = MetricsFoldExecutor().execute(
            context2, model_factory=b.RandomPredictorFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=b.RandomPredictorSerializer(), data=data,
        )
        model_hash1 = next(r.content_hash for r in outcome1.artifact_references if r.category is ArtifactCategory.MODEL)
        model_hash2 = next(r.content_hash for r in outcome2.artifact_references if r.category is ArtifactCategory.MODEL)
        assert model_hash1 == model_hash2
        assert outcome1.metrics == outcome2.metrics
