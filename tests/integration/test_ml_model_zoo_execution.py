"""End-to-end Milestone 4C integration test: real synthetic historical
data -> a real Milestone 3 research dataset -> a real Milestone 4A
prepared experiment -> a real Milestone 4B/4C walk-forward execution,
using the ACTUAL production `ml.model_zoo` models (not `ml.testing.
ConstantTestModel`) through `MetricsFoldExecutor` and `ml.comparison`.
Mirrors `tests/integration/test_execution_engine.py`'s conventions
exactly, but proves the model_zoo layer (registered models, real
serializers, real metrics, real training metadata, real comparison)
works together against the genuine execution engine -- not just each
module in isolation, and not just through the CLI (see
`tests/unit/test_ml_cli.py` for the CLI-level equivalent)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.exceptions import ExperimentLockError
from quant_platform.core.types import Timeframe
from quant_platform.execution.executor import DeterministicFoldExecutor, MetricsFoldExecutor
from quant_platform.execution.results import FoldResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.verification import verify_execution
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.comparison import ModelFoldMetrics, compare_to_baselines
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.metrics import CLASSIFICATION_METRIC_NAMES, REGRESSION_METRIC_NAMES
from quant_platform.ml.models import (
    CodeRevisionBinding,
    DatasetBinding,
    ExperimentStatus,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.training_metadata import TrainingMetadata

_WALK_FORWARD_SPLIT = {"strategy": "expanding_walk_forward", "params": {"n_splits": 3, "test_size": 100, "purge_bars": 5, "embargo_bars": 2}}

_LABEL_BY_OBJECTIVE: dict[ObjectiveType, tuple[str, LabelKind, LabelType]] = {
    ObjectiveType.BINARY_CLASSIFICATION: ("fut5", LabelKind.BINARY_DIRECTION, LabelType.BINARY),
    ObjectiveType.REGRESSION: ("fut5", LabelKind.FUTURE_RETURN, LabelType.CONTINUOUS),
}


def _build_real_research_dataset(tmp_path: Path, *, label_kind: LabelKind, n_rows: int = 2000, seed: int = 11, horizon_bars: int = 5):
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(n_rows, seed=seed)
    seed_canonical_dataset(historical_root, df)

    canonical_store = CanonicalStore(historical_root)
    manifest_store = ManifestStore(historical_root)
    historical_loader = DatasetLoader(canonical_store, manifest_store)

    registry = FeatureRegistry()
    register_core_technical_features(
        registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1, 5), momentum_windows=(10,), atr_window=14)
    )

    research_store = ResearchDatasetStore(research_root)
    research_manifest_store = ResearchManifestStore(research_root)
    builder = ResearchDatasetBuilder(
        historical_loader=historical_loader, registry=registry, research_store=research_store,
        manifest_store=research_manifest_store,
    )
    feature_names = tuple(spec.name for spec in registry.list_features())
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1,
        start=start, end=end, feature_names=feature_names,
        label_definition=LabelDefinition(name="fut5", kind=label_kind, horizon_bars=horizon_bars),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        preprocessing={},  # type: ignore[arg-type]
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, research_store


@pytest.fixture(scope="module")
def classification_dataset(tmp_path_factory: pytest.TempPathFactory):
    """Module-scoped: dataset construction (real feature pipeline over
    2000 synthetic bars) is read-only once built, and rebuilding it once
    per test would dominate this file's runtime for no correctness
    benefit -- every test below only WRITES into its own function-scoped
    `ml_artifacts_root`, never into the dataset itself."""
    return _build_real_research_dataset(tmp_path_factory.mktemp("clf_dataset"), label_kind=LabelKind.BINARY_DIRECTION)


@pytest.fixture(scope="module")
def regression_dataset(tmp_path_factory: pytest.TempPathFactory):
    return _build_real_research_dataset(tmp_path_factory.mktemp("reg_dataset"), label_kind=LabelKind.FUTURE_RETURN)


def _spec_for(dataset_manifest, *, model_name: str, objective: ObjectiveType, hyperparameters: dict[str, object] | None = None, model_version: str = "1", master_seed: int = 1, **overrides: object) -> ExperimentSpec:
    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value,
        source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_binding = FeatureBinding(
        feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    label_name, label_kind, label_type = _LABEL_BY_OBJECTIVE[objective]
    base: dict[str, object] = {
        "dataset_binding": dataset_binding, "feature_binding": feature_binding,
        "label_binding": LabelBinding(name=label_name, kind=label_kind.value, horizon_bars=5, label_type=label_type),
        "split_binding": SplitBinding(strategy=_WALK_FORWARD_SPLIT["strategy"], params=_WALK_FORWARD_SPLIT["params"]),  # type: ignore[arg-type]
        "preprocessing_binding": preprocessing_binding,
        "model_name": model_name, "model_version": model_version,
        "hyperparameters": ModelHyperparameters(values=hyperparameters or {}),
        "objective": objective,
        "seed_configuration": SeedConfiguration(master_seed=master_seed),
        "code_revision_binding": CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        "primary_metric": "accuracy" if objective is ObjectiveType.BINARY_CLASSIFICATION else "rmse",
    }
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


def _prepare_ready_experiment(tmp_path: Path, dataset_manifest, research_manifest_store, *, model_registry: ModelRegistry, **spec_overrides: object):
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry,
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(dataset_manifest, **spec_overrides)
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary
    return manifest


def _runner(tmp_path: Path, research_manifest_store, research_store, *, model_registry: ModelRegistry, fold_executor=None) -> ExecutionRunner:
    return ExecutionRunner(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry,
        research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        fold_executor=fold_executor if fold_executor is not None else MetricsFoldExecutor(),
        additional_serializers=mz.default_serializer_registry(),
    )


def _fold_result(runner: ExecutionRunner, experiment_id: str, fold_index: int) -> FoldResult:
    execution_manifest = runner.execution_manifest_store.load(experiment_id)
    ref = execution_manifest.fold_result_references[fold_index]
    raw = runner.artifact_store.read_artifact(ref.content_hash)
    return FoldResult.from_json_dict(json.loads(raw.decode("utf-8")))


class TestRealModelWalkForwardExecution:
    def test_full_walk_forward_execution_with_real_lightgbm_classifier(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name="lightgbm", objective=ObjectiveType.BINARY_CLASSIFICATION,
            hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(manifest.identity.experiment_id)

        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
        assert outcome.aggregate.failed_fold_indices == ()

        for fold_index in outcome.aggregate.completed_fold_indices:
            fold_result = _fold_result(runner, manifest.identity.experiment_id, fold_index)
            assert fold_result.status.value == "completed"
            for metric_name in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "matthews_corrcoef"):
                assert metric_name in fold_result.metrics, f"fold {fold_index} missing {metric_name}"

            categories = {ar.category.value for ar in fold_result.artifact_references}
            assert categories == {"model", "predictions", "probabilities", "training_metadata"}

            training_metadata_ref = next(ar for ar in fold_result.artifact_references if ar.category.value == "training_metadata")
            raw = runner.artifact_store.read_artifact(training_metadata_ref.content_hash)
            training_metadata = TrainingMetadata.from_json_dict(json.loads(raw.decode("utf-8")))
            assert training_metadata.experiment_id == manifest.identity.experiment_id
            assert training_metadata.fold_index == fold_index
            assert training_metadata.model_name == "lightgbm"
            assert training_metadata.library_name == "lightgbm"
            assert training_metadata.training_duration_seconds >= 0

    def test_verify_execution_succeeds_for_real_model_with_full_artifact_set(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name="catboost", objective=ObjectiveType.BINARY_CLASSIFICATION,
            hyperparameters={"iterations": 20},
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED

        report = verify_execution(
            manifest.identity.experiment_id,
            execution_manifest_store=runner.execution_manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner.artifact_store,
            event_store=runner.event_store,
        )
        assert report.is_ready


class TestResumeAndArtifactCompatibility:
    def test_resume_compatibility_with_real_model_and_metrics_executor(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name="lightgbm", objective=ObjectiveType.BINARY_CLASSIFICATION,
            hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )

        class InterruptOnFoldTwo:
            def __init__(self) -> None:
                self.attempts: dict[int, int] = {}

            def execute(self, context, **kwargs):
                self.attempts[context.fold_index] = self.attempts.get(context.fold_index, 0) + 1
                if context.fold_index == 2 and self.attempts[context.fold_index] == 1:
                    raise ExperimentLockError("simulated interruption")
                return MetricsFoldExecutor().execute(context, **kwargs)

        flaky = InterruptOnFoldTwo()
        runner = _runner(tmp_path, rms, rds, model_registry=registry, fold_executor=flaky)
        with pytest.raises(ExperimentLockError):
            runner.run(manifest.identity.experiment_id)

        exec_manifest_before = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest_before.stage is ExecutionStage.RECOVERABLE_FAILURE
        assert exec_manifest_before.completed_fold_indices == (0, 1)
        fold_0_ref_before = exec_manifest_before.fold_result_references[0]
        fold_1_ref_before = exec_manifest_before.fold_result_references[1]

        outcome = runner.resume(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
        # Folds 0 and 1 (already carrying real metrics/probabilities/
        # training-metadata artifacts from the ORIGINAL run) must never be
        # recomputed by MetricsFoldExecutor on resume.
        assert flaky.attempts == {0: 1, 1: 1, 2: 2}

        exec_manifest_after = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest_after.fold_result_references[0].content_hash == fold_0_ref_before.content_hash
        assert exec_manifest_after.fold_result_references[1].content_hash == fold_1_ref_before.content_hash

        for fold_index in (0, 1, 2):
            fold_result = _fold_result(runner, manifest.identity.experiment_id, fold_index)
            assert "accuracy" in fold_result.metrics
            assert any(ar.category.value == "training_metadata" for ar in fold_result.artifact_references)

    def test_execute_and_train_produce_identical_model_and_predictions_for_same_seed(self, tmp_path: Path, classification_dataset) -> None:
        """`execute` (`DeterministicFoldExecutor`, no metrics) and `train`
        (`MetricsFoldExecutor`, full metrics/probabilities/training-
        metadata) must fit the IDENTICAL model for the same experiment
        spec and seed -- proving Milestone 4C's additional validation/
        metrics/probability/provenance work is purely additive and never
        perturbs the fit path `DeterministicFoldExecutor` already
        exercised in Milestone 4B."""
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()

        root_a, root_b = tmp_path / "root_a", tmp_path / "root_b"

        def _prepare_in(root: Path) -> object:
            preparer = ExperimentPreparer(ml_artifacts_root=root, model_registry=registry, research_manifest_store=rms)
            spec = _spec_for(
                dataset_manifest, model_name="lightgbm", objective=ObjectiveType.BINARY_CLASSIFICATION,
                hyperparameters={"num_boost_round": 20, "num_leaves": 7},
            )
            manifest = preparer.prepare(spec)
            assert manifest.status is ExperimentStatus.READY, manifest.failure_summary
            return manifest

        manifest_a = _prepare_in(root_a)
        manifest_b = _prepare_in(root_b)
        assert manifest_a.identity.experiment_id == manifest_b.identity.experiment_id  # storage root is not part of experiment identity

        runner_a = ExecutionRunner(
            ml_artifacts_root=root_a, model_registry=registry, research_manifest_store=rms, research_dataset_store=rds,
            fold_executor=DeterministicFoldExecutor(), additional_serializers=mz.default_serializer_registry(),
        )
        runner_b = ExecutionRunner(
            ml_artifacts_root=root_b, model_registry=registry, research_manifest_store=rms, research_dataset_store=rds,
            fold_executor=MetricsFoldExecutor(), additional_serializers=mz.default_serializer_registry(),
        )
        outcome_a = runner_a.run(manifest_a.identity.experiment_id)
        outcome_b = runner_b.run(manifest_b.identity.experiment_id)
        assert outcome_a.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome_b.aggregate.overall_status is ExecutionStage.COMPLETED

        for fold_index in outcome_a.aggregate.completed_fold_indices:
            fold_a = _fold_result(runner_a, manifest_a.identity.experiment_id, fold_index)
            fold_b = _fold_result(runner_b, manifest_b.identity.experiment_id, fold_index)
            model_ref_a = next(ar for ar in fold_a.artifact_references if ar.category.value == "model")
            model_ref_b = next(ar for ar in fold_b.artifact_references if ar.category.value == "model")
            assert model_ref_a.content_hash == model_ref_b.content_hash
            predictions_ref_a = next(ar for ar in fold_a.artifact_references if ar.category.value == "predictions")
            predictions_ref_b = next(ar for ar in fold_b.artifact_references if ar.category.value == "predictions")
            assert predictions_ref_a.content_hash == predictions_ref_b.content_hash


class TestComparisonAgainstRealBaselines:
    def test_baseline_and_real_model_are_comparable_via_comparison_module(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()

        candidate_manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name="lightgbm", objective=ObjectiveType.BINARY_CLASSIFICATION,
            hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        baseline_manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name="majority_predictor", objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome_candidate = runner.run(candidate_manifest.identity.experiment_id)
        outcome_baseline = runner.run(baseline_manifest.identity.experiment_id)
        assert outcome_candidate.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome_baseline.aggregate.overall_status is ExecutionStage.COMPLETED

        def _all_fold_metrics(experiment_id: str) -> tuple[dict, ...]:
            execution_manifest = runner.execution_manifest_store.load(experiment_id)
            return tuple(
                _fold_result(runner, experiment_id, fold_index).metrics
                for fold_index in sorted(execution_manifest.fold_result_references)
            )

        candidate_metrics = ModelFoldMetrics(model_name="lightgbm", per_fold_metrics=_all_fold_metrics(candidate_manifest.identity.experiment_id))
        baseline_metrics = ModelFoldMetrics(model_name="majority_predictor", per_fold_metrics=_all_fold_metrics(baseline_manifest.identity.experiment_id))

        report = compare_to_baselines(candidate_metrics, [baseline_metrics])
        assert report.candidate_name == "lightgbm"
        assert len(report.baseline_reports) == 1
        metric_names = {mc.metric_name for mc in report.baseline_reports[0].metric_comparisons}
        assert "accuracy" in metric_names

        outperforms = report.outperforms_all_baselines("accuracy")
        assert isinstance(outperforms, bool)


_ALL_MODEL_CASES: list[tuple[str, ObjectiveType, dict[str, object]]] = [
    ("constant_predictor", ObjectiveType.BINARY_CLASSIFICATION, {"constant": 1.0}),
    ("random_predictor", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("majority_predictor", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("lightgbm", ObjectiveType.BINARY_CLASSIFICATION, {"num_boost_round": 20, "num_leaves": 7}),
    ("xgboost", ObjectiveType.BINARY_CLASSIFICATION, {"num_boost_round": 20, "max_depth": 3}),
    ("catboost", ObjectiveType.BINARY_CLASSIFICATION, {"iterations": 20}),
    ("dummy_mean_regressor", ObjectiveType.REGRESSION, {}),
]
"""Every model NOT declaring `requires_scaled_numeric_features` -- always
completable through the real engine, whose bound dataset never has
proven scaling (see `TestScaleSensitiveModelsRejectedByTheRealEngine`
just below for the two that do)."""

_SCALE_SENSITIVE_MODEL_CASES: list[tuple[str, ObjectiveType, dict[str, object]]] = [
    ("logistic_regression", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("elastic_net", ObjectiveType.REGRESSION, {}),
]


class TestEveryModelZooModelAgainstTheRealEngine:
    @pytest.mark.parametrize("model_name, objective, hyperparameters", _ALL_MODEL_CASES, ids=[c[0] for c in _ALL_MODEL_CASES])
    def test_every_model_zoo_model_trains_successfully_end_to_end(
        self, tmp_path: Path, classification_dataset, regression_dataset,
        model_name: str, objective: ObjectiveType, hyperparameters: dict[str, object],
    ) -> None:
        """7 of the 9 mandatory Milestone 4C models (3 gradient-boosting
        learners + 4 baselines -- everything that does NOT declare
        `requires_scaled_numeric_features`), each run once through the
        REAL `ExecutionRunner` + `MetricsFoldExecutor` + real `ml.
        model_zoo` serializers -- not a unit-level fit/predict call, the
        genuine walk-forward engine. Logistic Regression/Elastic Net are
        covered separately below, since BLOCKER 2's preprocessing-proof
        gate means they are EXPECTED to be rejected here, not completed."""
        dataset_manifest, rms, rds = classification_dataset if objective is ObjectiveType.BINARY_CLASSIFICATION else regression_dataset
        registry = mz.register_default_models()
        manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name=model_name, objective=objective, hyperparameters=hyperparameters,
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(manifest.identity.experiment_id)

        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED, outcome.aggregate.failed_fold_indices
        assert outcome.aggregate.failed_fold_indices == ()

        allowed_metric_names = REGRESSION_METRIC_NAMES if objective is ObjectiveType.REGRESSION else CLASSIFICATION_METRIC_NAMES
        for fold_index in outcome.aggregate.completed_fold_indices:
            fold_result = _fold_result(runner, manifest.identity.experiment_id, fold_index)
            assert fold_result.metrics, f"{model_name} fold {fold_index}: no metrics recorded"
            assert set(fold_result.metrics).issubset(allowed_metric_names)
            assert any(ar.category.value == "model" for ar in fold_result.artifact_references)
            assert any(ar.category.value == "training_metadata" for ar in fold_result.artifact_references)


class TestScaleSensitiveModelsRejectedByTheRealEngine:
    """BLOCKER 2, "REQUIRED PREPROCESSING MUST BE ENFORCED": Logistic
    Regression and Elastic Net both declare `ModelCapabilities.
    requires_scaled_numeric_features=True`. No dataset the real execution
    engine will ever run against can carry PROOF of safe, train-only
    scaling today (`execution.runner.
    assert_preprocessing_is_safe_for_execution` already fails an entire
    execution closed the moment a bound dataset shows ANY sign of fitted
    preprocessing, since a globally-fit transform cannot be proven safe
    against this engine's own, independently-chosen fold boundaries) --
    so every fold of every experiment for these two models must be
    rejected by `ml.model_validation.validate_training_data`'s
    `required_preprocessing_unproven` check, BEFORE `fit`, never silently
    trained on unscaled data (which previously surfaced only as an
    sklearn `ConvergenceWarning`, not a hard failure)."""

    @pytest.mark.parametrize("model_name, objective, hyperparameters", _SCALE_SENSITIVE_MODEL_CASES, ids=[c[0] for c in _SCALE_SENSITIVE_MODEL_CASES])
    def test_model_is_rejected_on_every_fold_before_any_fit(
        self, tmp_path: Path, classification_dataset, regression_dataset,
        model_name: str, objective: ObjectiveType, hyperparameters: dict[str, object],
    ) -> None:
        dataset_manifest, rms, rds = classification_dataset if objective is ObjectiveType.BINARY_CLASSIFICATION else regression_dataset
        registry = mz.register_default_models()
        manifest = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry,
            model_name=model_name, objective=objective, hyperparameters=hyperparameters,
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(manifest.identity.experiment_id)

        assert outcome.aggregate.overall_status is ExecutionStage.FAILED
        assert outcome.aggregate.completed_fold_indices == ()
        assert outcome.aggregate.failed_fold_indices == (0, 1, 2)

        for fold_index in outcome.aggregate.failed_fold_indices:
            fold_result = _fold_result(runner, manifest.identity.experiment_id, fold_index)
            assert fold_result.status.value == "failed"
            assert fold_result.metrics == {}
            assert fold_result.artifact_references == ()  # no MODEL/PREDICTIONS artifact written on rejection
            assert fold_result.failure_reason is not None
            assert "required_preprocessing_unproven" in fold_result.failure_reason
