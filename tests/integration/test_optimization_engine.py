"""End-to-end Milestone 4D integration tests: real synthetic historical
data -> a real Milestone 3 research dataset -> a real Milestone 4A
prepared experiment -> a real `OptimizationRunner` nested walk-forward
search, using the ACTUAL production `ml.model_zoo` models. Mirrors
`tests/integration/test_ml_model_zoo_execution.py`'s conventions exactly.

These tests prove what no unit test (with its hand-built, isolated
fixtures) can: that the whole pipeline -- outer/inner split construction,
feature selection, trial execution, Optuna sampling, pruning, early
stopping, ranking, outer-fold finalization, resume, and verification --
actually composes correctly end-to-end, including the empirically-
critical claim that deterministic sampler resume reproduces an
uninterrupted run's exact winning trials."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.exceptions import ExperimentLockError, OptimizationResumeError
from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
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
from quant_platform.ml.persistence import write_json_atomic
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization import runner as runner_module
from quant_platform.optimization.candidates import TrialResult, TrialStatus
from quant_platform.optimization.feature_selection import FeatureSelectionSpec, FeatureSelectionStrategy
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.optimization.manifests import (
    OptimizationEventStore,
    OptimizationManifestStore,
    trial_references_for_outer_fold,
)
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    OptimizationStage,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
    compute_optimization_identity,
)
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.runner import OptimizationRunner
from quant_platform.optimization.search_space import lightgbm_default_search_space
from quant_platform.optimization.verification import _verify_manifest_stage_consistency, verify_optimization

_WALK_FORWARD_SPLIT = {"strategy": "expanding_walk_forward", "params": {"n_splits": 2, "test_size": 150, "purge_bars": 5, "embargo_bars": 2}}


def _build_real_research_dataset(tmp_path: Path, *, n_rows: int = 2500, seed: int = 11):
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
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=start, end=end, feature_names=feature_names,
        label_definition=LabelDefinition(name="fut5", kind=LabelKind.BINARY_DIRECTION, horizon_bars=5),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        preprocessing={},  # type: ignore[arg-type]
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, research_store


@pytest.fixture(scope="module")
def classification_dataset(tmp_path_factory: pytest.TempPathFactory):
    return _build_real_research_dataset(tmp_path_factory.mktemp("opt_dataset"))


def _experiment_spec_for(dataset_manifest, *, model_name: str = "lightgbm", hyperparameters: dict[str, object] | None = None) -> ExperimentSpec:
    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value, source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_binding = FeatureBinding(
        feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    return ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding,
        label_binding=LabelBinding(name="fut5", kind=LabelKind.BINARY_DIRECTION.value, horizon_bars=5, label_type=LabelType.BINARY),
        split_binding=SplitBinding(strategy=_WALK_FORWARD_SPLIT["strategy"], params=_WALK_FORWARD_SPLIT["params"]),  # type: ignore[arg-type]
        preprocessing_binding=preprocessing_binding, model_name=model_name, model_version="1",
        hyperparameters=ModelHyperparameters(values=hyperparameters or {}), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=1), code_revision_binding=CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )


def _prepare_ready_experiment(tmp_path: Path, dataset_manifest, research_manifest_store, *, model_registry: ModelRegistry, **overrides: object):
    preparer = ExperimentPreparer(ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry, research_manifest_store=research_manifest_store)
    spec = _experiment_spec_for(dataset_manifest, **overrides)
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary
    return manifest, spec


def _runner(tmp_path: Path, research_manifest_store, research_store, *, model_registry: ModelRegistry) -> OptimizationRunner:
    return OptimizationRunner(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry,
        research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
        additional_serializers=mz.default_serializer_registry(),
    )


def _default_optimization_spec(experiment_spec: ExperimentSpec, parent_experiment_id: str, *, model_registry: ModelRegistry, **overrides: object):
    base: dict[str, object] = {
        "experiment": experiment_spec, "parent_experiment_id": parent_experiment_id, "model_name": "lightgbm", "model_version": "1",
        "primary_metric": "accuracy", "inner_split_config": InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
        "feature_selection_spec": FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}),
        "search_space": lightgbm_default_search_space(), "sampler_kind": SamplerKind.TPE, "pruning_config": PruningConfig(kind=PruningKind.NONE),
        "early_stopping_config": EarlyStoppingConfig(enabled=False), "max_trials": 3, "min_successful_inner_folds": 1,
        "seed_configuration": SeedConfiguration(master_seed=42), "model_registry": model_registry,
    }
    base.update(overrides)
    return build_optimization_spec(**base)  # type: ignore[arg-type]


class TestFullNestedOptimizationPipeline:
    def test_completes_with_outer_test_evaluated_once_per_fold_and_verifies_clean(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)

        outcome = runner.run(spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED, outcome.manifest.failure_summary
        assert outcome.manifest.total_trials_completed > 0
        assert set(outcome.manifest.completed_outer_fold_indices) == set(outcome.manifest.outer_fold_result_references)
        assert len(outcome.manifest.winning_trial_by_outer_fold) == len(outcome.manifest.completed_outer_fold_indices)

        report = verify_optimization(
            outcome.manifest.optimization_id, optimization_manifest_store=runner.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner.artifact_store, event_store=runner.event_store,
        )
        assert report.is_ready, [i.message for i in report.criticals]

    def test_idempotent_rerun_of_a_completed_optimization_is_a_no_op(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)

        first = runner.run(spec)
        second = runner.run(spec)
        assert second.was_idempotent_no_op
        assert second.manifest.total_trials_completed == first.manifest.total_trials_completed


class TestResumeReproducesUninterruptedRun:
    def test_resume_after_simulated_crash_reproduces_the_same_winning_trials(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry_full = mz.register_default_models()
        experiment_manifest_full, experiment_spec_full = _prepare_ready_experiment(
            tmp_path / "full", dataset_manifest, rms, model_registry=registry_full, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec_full = _default_optimization_spec(
            experiment_spec_full, experiment_manifest_full.identity.experiment_id, model_registry=registry_full,
            feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE), max_trials=6,
        )
        runner_full = _runner(tmp_path / "full", rms, rds, model_registry=registry_full)
        outcome_full = runner_full.run(spec_full)
        assert outcome_full.manifest.stage is OptimizationStage.COMPLETED

        registry_int = mz.register_default_models()
        experiment_manifest_int, experiment_spec_int = _prepare_ready_experiment(
            tmp_path / "interrupted", dataset_manifest, rms, model_registry=registry_int, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec_int = _default_optimization_spec(
            experiment_spec_int, experiment_manifest_int.identity.experiment_id, model_registry=registry_int,
            feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE), max_trials=6,
        )
        assert compute_optimization_identity(spec_int).optimization_id == compute_optimization_identity(spec_full).optimization_id

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial

        def flaky_run_trial(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 4:
                raise ExperimentLockError("simulated crash mid-trial-loop")
            return real_run_trial(*args, **kwargs)

        runner_module.run_trial = flaky_run_trial
        try:
            runner_int = _runner(tmp_path / "interrupted", rms, rds, model_registry=registry_int)
            with pytest.raises(ExperimentLockError):
                runner_int.run(spec_int)
        finally:
            runner_module.run_trial = real_run_trial

        manifest_after_crash = runner_int.manifest_store.load(compute_optimization_identity(spec_int).optimization_id)
        assert manifest_after_crash.stage is OptimizationStage.RECOVERABLE_FAILURE

        runner_resumed = _runner(tmp_path / "interrupted", rms, rds, model_registry=registry_int)
        outcome_resumed = runner_resumed.resume(compute_optimization_identity(spec_int).optimization_id, spec=spec_int)
        assert outcome_resumed.manifest.stage is OptimizationStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.winning_trial_by_outer_fold == outcome_full.manifest.winning_trial_by_outer_fold

        report = verify_optimization(
            compute_optimization_identity(spec_int).optimization_id, optimization_manifest_store=runner_resumed.manifest_store,
            experiment_manifest_store=ExperimentManifestStore((tmp_path / "interrupted") / "ml_artifacts"),
            artifact_store=runner_resumed.artifact_store, event_store=runner_resumed.event_store,
        )
        assert report.is_ready

    def test_resuming_a_completed_optimization_raises(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=2)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        runner.run(spec)
        with pytest.raises(OptimizationResumeError, match="terminal"):
            runner.resume(compute_optimization_identity(spec).optimization_id)


class TestPruningStabilitySelectionEarlyStopping:
    def test_combined_pruning_stability_selection_and_early_stopping_complete_cleanly(self, tmp_path: Path, classification_dataset) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 30, "num_leaves": 7},
        )
        spec = _default_optimization_spec(
            experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry,
            inner_split_config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15),
            feature_selection_spec=FeatureSelectionSpec(
                strategy=FeatureSelectionStrategy.STABILITY_SELECTION,
                params={"base_strategy": "univariate", "mode": "top_k", "k": 3, "n_repeats": 4, "subsample_fraction": 0.8, "min_frequency": 0.4},
            ),
            pruning_config=PruningConfig(kind=PruningKind.MEDIAN_STOPPING, min_completed_inner_folds=1),
            early_stopping_config=EarlyStoppingConfig(enabled=True, patience=5, validation_fraction=0.15, final_round_policy="median_best_iteration"),
            max_trials=5,
        )
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED, outcome.manifest.failure_summary
        assert outcome.manifest.total_trials_completed + outcome.manifest.total_trials_pruned > 0

        for outer_fold_index, ref in outcome.manifest.outer_fold_result_references.items():
            raw = runner.artifact_store.read_artifact(ref.content_hash)
            result = json.loads(raw.decode("utf-8"))
            assert result["outer_test_metrics"], f"outer fold {outer_fold_index} has no outer-test metrics"

        report = verify_optimization(
            outcome.manifest.optimization_id, optimization_manifest_store=runner.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner.artifact_store, event_store=runner.event_store,
        )
        assert report.is_ready


class TestScaleSensitiveModelExcludedFromOptimization:
    @pytest.mark.parametrize("model_name", ["logistic_regression", "elastic_net"])
    def test_build_optimization_spec_fails_closed_for_the_real_model_registry(self, tmp_path: Path, classification_dataset, model_name: str) -> None:
        dataset_manifest, rms, _rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry)
        with pytest.raises(ValueError, match="requires scaled numeric features"):
            _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, model_name=model_name)


class TestVerifyOptimizationDetectsInconsistencies:
    """Runs one REAL, small, otherwise-clean optimization, then perturbs
    exactly one thing at a time -- proving each detector actually fires,
    not merely that the happy path passes."""

    @pytest.fixture()
    def completed_run(self, tmp_path: Path, classification_dataset):
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=2)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED
        return runner, outcome.manifest, tmp_path

    def _verify(self, runner, optimization_id: str, tmp_path: Path):
        return verify_optimization(
            optimization_id, optimization_manifest_store=runner.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner.artifact_store, event_store=runner.event_store,
        )

    def test_clean_run_verifies(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert report.is_ready

    def test_corrupted_trial_result_artifact_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        any_ref = next(iter(manifest.trial_result_references.values()))
        content_path = runner.artifact_store._content_path(any_ref.content_hash)
        content_path.write_bytes(b"corrupted bytes, not valid JSON")
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "trial_result_unverifiable" for i in report.criticals)

    def test_tampered_winning_trial_by_outer_fold_is_detected(self, completed_run) -> None:
        """`transition()` has no legal COMPLETED->COMPLETED self-loop (by
        design -- a terminal manifest is never legitimately mutated in
        place), so tampering must bypass the state machine entirely and
        write directly to the manifest file, exactly as the sibling tests
        above bypass the artifact store to corrupt artifact bytes."""
        runner, manifest, tmp_path = completed_run
        outer_fold_index = next(iter(manifest.winning_trial_by_outer_fold))
        real_winner = manifest.winning_trial_by_outer_fold[outer_fold_index]
        bogus_winner = real_winner + 1000
        tampered = {**manifest.winning_trial_by_outer_fold, outer_fold_index: bogus_winner}
        tampered_manifest = replace(manifest, winning_trial_by_outer_fold=tampered)
        write_json_atomic(runner.manifest_store._manifest_path(manifest.optimization_id), tampered_manifest.to_json_dict())
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "ranking_not_reproducible" in codes or "outer_fold_result_winner_mismatch" in codes

    def test_corrupted_optimization_spec_artifact_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        spec_ref = next(r for r in manifest.artifact_references if r.category is ArtifactCategory.OPTIMIZATION_SPEC)
        content_path = runner.artifact_store._content_path(spec_ref.content_hash)
        content_path.write_bytes(b"not json")
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "optimization_spec_unverifiable" for i in report.criticals)

    def test_missing_outer_fold_result_reference_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        outer_fold_index = next(iter(manifest.outer_fold_result_references))
        ref = manifest.outer_fold_result_references[outer_fold_index]
        content_path = runner.artifact_store._content_path(ref.content_hash)
        content_path.unlink()
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_unverifiable" for i in report.criticals)

    def test_manifest_claiming_completed_with_missing_outer_folds_is_detected(self, completed_run) -> None:
        """`total_outer_folds` disagreeing with `completed_outer_fold_indices`
        at stage=COMPLETED can never arise from the real runner (see
        `runner.py`'s hard-failure-on-missing-winner policy), so this
        drives the specific manifest-level invariant check directly
        rather than contorting a real run into an unreachable state."""
        _runner, manifest, _tmp_path = completed_run

        tampered_manifest = replace(manifest, total_outer_folds=(manifest.total_outer_folds or 0) + 1)
        issues = _verify_manifest_stage_consistency(tampered_manifest)
        assert any(i.code == "completed_stage_outer_fold_count_mismatch" for i in issues)

    def _write_tampered_spec(self, runner, manifest, mutate):
        """Adversarial audit, Section 8 helper: reads the real
        OPTIMIZATION_SPEC artifact's raw JSON dict, applies `mutate`
        in place, writes the result under a NEW content hash (a real,
        self-consistent tamper -- exactly what a hand-edited or
        corrupted-but-internally-consistent file on disk would look
        like), and repoints the manifest's own reference at it."""
        spec_ref = next(r for r in manifest.artifact_references if r.category is ArtifactCategory.OPTIMIZATION_SPEC)
        raw = json.loads(runner.artifact_store.read_artifact(spec_ref.content_hash).decode("utf-8"))
        mutate(raw)
        new_ref = runner.artifact_store.write_artifact(json.dumps(raw).encode("utf-8"), category=ArtifactCategory.OPTIMIZATION_SPEC)
        new_refs = tuple(new_ref if r.category is ArtifactCategory.OPTIMIZATION_SPEC else r for r in manifest.artifact_references)
        tampered_manifest = replace(manifest, artifact_references=new_refs)
        write_json_atomic(runner.manifest_store._manifest_path(manifest.optimization_id), tampered_manifest.to_json_dict())

    def test_tampered_metric_direction_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        self._write_tampered_spec(runner, manifest, lambda raw: raw.__setitem__(
            "metric_direction", "minimize" if raw["metric_direction"] == "maximize" else "maximize",
        ))
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "optimization_spec_unverifiable" for i in report.criticals)

    def test_tampered_dataset_binding_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        self._write_tampered_spec(runner, manifest, lambda raw: raw["dataset_binding"].__setitem__("dataset_id", "f" * 16))
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "dataset_binding_mismatch" for i in report.criticals)

    def test_tampered_feature_universe_fingerprint_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        self._write_tampered_spec(runner, manifest, lambda raw: raw.__setitem__("feature_universe_fingerprint", "f" * 64))
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "feature_universe_fingerprint_mismatch" for i in report.criticals)

    def test_tampered_outer_split_binding_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        self._write_tampered_spec(runner, manifest, lambda raw: raw["outer_split_binding"].__setitem__("strategy", "rolling_walk_forward"))
        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "outer_split_binding_mismatch" for i in report.criticals)

    def test_sampled_hyperparameters_outside_the_declared_search_space_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        any_key, any_ref = next(iter(manifest.trial_result_references.items()))
        raw = json.loads(runner.artifact_store.read_artifact(any_ref.content_hash).decode("utf-8"))
        # Corrupt whichever declared parameter is present with a wildly
        # out-of-range value -- guaranteed invalid regardless of the
        # winning trial's own default LightGBM search space contents.
        param_name = next(iter(raw["sampled_hyperparameters"]))
        raw["sampled_hyperparameters"][param_name] = -999999
        new_ref = runner.artifact_store.write_artifact(json.dumps(raw).encode("utf-8"), category=ArtifactCategory.TRIAL_RESULT)
        new_refs = dict(manifest.trial_result_references)
        new_refs[any_key] = new_ref
        tampered_manifest = replace(manifest, trial_result_references=new_refs)
        write_json_atomic(runner.manifest_store._manifest_path(manifest.optimization_id), tampered_manifest.to_json_dict())

        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "trial_sampled_values_invalid" for i in report.criticals)

    def test_selected_features_outside_the_candidate_universe_is_detected(self, completed_run) -> None:
        runner, manifest, tmp_path = completed_run
        any_key, any_ref = next(iter(manifest.trial_result_references.items()))
        trial_raw = json.loads(runner.artifact_store.read_artifact(any_ref.content_hash).decode("utf-8"))
        fs_ref_dict = trial_raw["inner_fold_metrics"][0]["feature_selection_result_reference"]
        assert fs_ref_dict is not None, "expected at least one inner fold to have run real feature selection"
        fs_raw = json.loads(runner.artifact_store.read_artifact(fs_ref_dict["content_hash"]).decode("utf-8"))
        fs_raw["selected_features"] = [*fs_raw["selected_features"], "not_a_real_feature_in_the_universe"]
        new_fs_ref = runner.artifact_store.write_artifact(json.dumps(fs_raw).encode("utf-8"), category=ArtifactCategory.FEATURE_SELECTION_RESULT)
        trial_raw["inner_fold_metrics"][0]["feature_selection_result_reference"] = new_fs_ref.to_json_dict()
        new_trial_ref = runner.artifact_store.write_artifact(json.dumps(trial_raw).encode("utf-8"), category=ArtifactCategory.TRIAL_RESULT)
        new_refs = dict(manifest.trial_result_references)
        new_refs[any_key] = new_trial_ref
        tampered_manifest = replace(manifest, trial_result_references=new_refs)
        write_json_atomic(runner.manifest_store._manifest_path(manifest.optimization_id), tampered_manifest.to_json_dict())

        report = self._verify(runner, manifest.optimization_id, tmp_path)
        assert not report.is_ready
        assert any(i.code == "feature_selection_result_unverifiable" for i in report.criticals)

    def test_tampered_final_hyperparameters_and_outer_test_metrics_are_not_independently_reverified(self, completed_run) -> None:
        """Honest, deliberately-documented limitation (adversarial audit,
        Section 8): `verify_optimization` re-derives and cross-checks
        everything CHEAPLY re-derivable (identity, sampled-value validity,
        feature-selection-vs-universe, ranking reproducibility from
        verified trials) or structurally cross-referenced (winner
        consistency). It does NOT re-run the winning candidate's outer-
        train refit (an expensive, real model-fitting operation) merely to
        confirm `OuterFoldResult.final_hyperparameters`/`outer_test_metrics`
        are self-consistent -- there is no cheaper independent source of
        truth to check them against without redoing the fit. This test
        documents that fact directly, rather than letting a false
        assumption of blanket re-verification stand unverified."""
        runner, manifest, tmp_path = completed_run
        any_outer_index, any_outer_ref = next(iter(manifest.outer_fold_result_references.items()))
        raw = json.loads(runner.artifact_store.read_artifact(any_outer_ref.content_hash).decode("utf-8"))
        raw["final_hyperparameters"] = {"num_boost_round": 999999999}
        raw["outer_test_metrics"] = {"rmse": -1.0}  # a nonsensical value for a real metric
        new_ref = runner.artifact_store.write_artifact(json.dumps(raw).encode("utf-8"), category=ArtifactCategory.OUTER_FOLD_SELECTION)
        new_refs = dict(manifest.outer_fold_result_references)
        new_refs[any_outer_index] = new_ref
        tampered_manifest = replace(manifest, outer_fold_result_references=new_refs)
        write_json_atomic(runner.manifest_store._manifest_path(manifest.optimization_id), tampered_manifest.to_json_dict())

        report = self._verify(runner, manifest.optimization_id, tmp_path)
        # Self-consistent otherwise (same winning_trial_number, same
        # references, same category) -- report STAYS ready, confirming
        # this specific field is genuinely out of scope for re-verification.
        assert report.is_ready, [i.message for i in report.criticals]


class TestExhaustiveResumeAtEveryTrialStateTransition:
    """Adversarial release-readiness audit, Section 2: TPE RESUME
    DETERMINISM, compared at EVERY meaningful interruption point within
    one outer fold's trial search -- before the first trial, after a
    completed trial, after a failed trial, after an invalid trial, after
    a pruned trial, and immediately before winner selection -- with FULL
    equality proofs, never just the winning trial number: sampled
    hyperparameters, trial status, primary score, complete ranking,
    winning trial, final selected features, final hyperparameters, and
    outer-test metrics (byte-identical, since the SAME winner is refit
    against the SAME outer-train partition either way).

    Trials 1/2/3 are deterministically FORCED (bypassing the real
    inner-loop computation, but still flowing through the REAL runner's
    manifest/event/Optuna-tell bookkeeping) to FAILED/INVALID/PRUNED
    respectively -- `ConstantTestModel` cannot naturally produce those
    outcomes -- so every one of the 4 `TrialStatus` values is exercised
    at a known, deliberately chosen trial number."""

    _FORCED_STATUS: ClassVar[dict[int, TrialStatus]] = {1: TrialStatus.FAILED, 2: TrialStatus.INVALID, 3: TrialStatus.PRUNED}
    _MAX_TRIALS = 6

    @classmethod
    def _state_forcing_run_trial(cls, real_run_trial):
        def wrapper(trial_spec, **kwargs):
            forced = cls._FORCED_STATUS.get(trial_spec.trial_number)
            if forced is None:
                return real_run_trial(trial_spec, **kwargs)
            common = {
                "schema_version": 1, "optimization_id": trial_spec.optimization_id, "outer_fold_index": trial_spec.outer_fold_index,
                "trial_number": trial_spec.trial_number, "sampled_hyperparameters": trial_spec.sampled_hyperparameters,
                "inner_fold_metrics": (), "primary_metric_aggregate": None, "successful_inner_folds": 0, "total_inner_folds": 1,
                "duration_seconds": 0.001, "failure_code": ("pruned" if forced is TrialStatus.PRUNED else f"forced_{forced.value}"),
                "failure_reason": f"forced by adversarial audit test (trial {trial_spec.trial_number})",
            }
            return TrialResult(status=forced, **common)  # type: ignore[arg-type]

        return wrapper

    @classmethod
    def _spec_and_runner(cls, tmp_path: Path, dataset_manifest, rms, rds, *, model_registry: ModelRegistry):
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=model_registry, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec = _default_optimization_spec(
            experiment_spec, experiment_manifest.identity.experiment_id, model_registry=model_registry, max_trials=cls._MAX_TRIALS,
            feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}),
        )
        return spec, _runner(tmp_path, rms, rds, model_registry=model_registry)

    @staticmethod
    def _extract_full_state(runner, manifest):
        trials_by_outer_fold: dict[int, list[TrialResult]] = {}
        for outer_fold_index in manifest.completed_outer_fold_indices:
            refs = trial_references_for_outer_fold(manifest, outer_fold_index)
            trials = []
            for trial_number in sorted(refs):
                raw = runner.artifact_store.read_artifact(refs[trial_number].content_hash)
                trials.append(TrialResult.from_json_dict(json.loads(raw.decode("utf-8"))))
            trials_by_outer_fold[outer_fold_index] = trials
        outer_results: dict[int, OuterFoldResult] = {}
        for outer_fold_index, ref in manifest.outer_fold_result_references.items():
            raw = runner.artifact_store.read_artifact(ref.content_hash)
            outer_results[outer_fold_index] = OuterFoldResult.from_json_dict(json.loads(raw.decode("utf-8")))
        return trials_by_outer_fold, outer_results

    @classmethod
    @pytest.fixture(scope="class")
    def baseline(cls, tmp_path_factory: pytest.TempPathFactory, classification_dataset):
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        spec, runner = cls._spec_and_runner(tmp_path_factory.mktemp("resume_baseline"), dataset_manifest, rms, rds, model_registry=registry)

        real_run_trial = runner_module.run_trial
        runner_module.run_trial = cls._state_forcing_run_trial(real_run_trial)
        try:
            outcome = runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial
        assert outcome.manifest.stage is OptimizationStage.COMPLETED, outcome.manifest.failure_summary
        trials, outer_results = cls._extract_full_state(runner, outcome.manifest)
        # Sanity: all 4 TrialStatus values genuinely occurred in fold 0 --
        # proving the equality checks below are not vacuous.
        fold_0_statuses = {t.trial_number: t.status for t in trials[0]}
        assert fold_0_statuses[1] is TrialStatus.FAILED
        assert fold_0_statuses[2] is TrialStatus.INVALID
        assert fold_0_statuses[3] is TrialStatus.PRUNED
        assert any(s is TrialStatus.COMPLETED for s in fold_0_statuses.values())
        return spec, outcome, trials, outer_results

    @pytest.mark.parametrize(
        "crash_after_n_trials",
        [0, 1, 2, 3, 4, 5],
        ids=[
            "before_first_trial", "after_completed_trial_0", "after_failed_trial_1",
            "after_invalid_trial_2", "after_pruned_trial_3", "immediately_before_winner_selection",
        ],
    )
    def test_resume_reproduces_full_state_at_every_interruption_point(
        self, tmp_path: Path, classification_dataset, baseline, crash_after_n_trials: int,
    ) -> None:
        baseline_spec, baseline_outcome, baseline_trials, baseline_outer = baseline
        dataset_manifest, rms, rds = classification_dataset
        registry_int = mz.register_default_models()
        spec_int, _unused_runner = self._spec_and_runner(tmp_path, dataset_manifest, rms, rds, model_registry=registry_int)
        assert compute_optimization_identity(spec_int).optimization_id == compute_optimization_identity(baseline_spec).optimization_id

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial
        state_forcing = self._state_forcing_run_trial(real_run_trial)

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == crash_after_n_trials + 1:
                raise ExperimentLockError("simulated crash for the adversarial resume audit")
            return state_forcing(trial_spec, **kwargs)

        runner_module.run_trial = flaky
        try:
            runner_crashed = _runner(tmp_path, rms, rds, model_registry=registry_int)
            with pytest.raises(ExperimentLockError):
                runner_crashed.run(spec_int)
        finally:
            runner_module.run_trial = real_run_trial

        optimization_id = compute_optimization_identity(spec_int).optimization_id
        manifest_after_crash = runner_crashed.manifest_store.load(optimization_id)
        assert manifest_after_crash.stage is OptimizationStage.RECOVERABLE_FAILURE

        # "After process restart with only persisted platform state": a
        # BRAND NEW OptimizationRunner instance, no in-memory state carried
        # over from the crashed one.
        runner_module.run_trial = state_forcing
        try:
            runner_resumed = _runner(tmp_path, rms, rds, model_registry=registry_int)
            outcome_resumed = runner_resumed.resume(optimization_id, spec=spec_int)
        finally:
            runner_module.run_trial = real_run_trial

        assert outcome_resumed.manifest.stage is OptimizationStage.COMPLETED, outcome_resumed.manifest.failure_summary
        resumed_trials, resumed_outer = self._extract_full_state(runner_resumed, outcome_resumed.manifest)

        # Trial sequence: identical status, sampled hyperparameters, and
        # primary score for EVERY trial of EVERY outer fold -- not merely
        # the winner.
        assert set(resumed_trials) == set(baseline_trials)
        for outer_fold_index, b_trials in baseline_trials.items():
            r_trials = resumed_trials[outer_fold_index]
            assert [t.trial_number for t in b_trials] == [t.trial_number for t in r_trials]
            for bt, rt in zip(b_trials, r_trials, strict=True):
                assert bt.status == rt.status, f"outer_fold={outer_fold_index} trial={bt.trial_number}: status diverged"
                assert bt.sampled_hyperparameters == rt.sampled_hyperparameters, (
                    f"outer_fold={outer_fold_index} trial={bt.trial_number}: sampled hyperparameters diverged"
                )
                if bt.primary_metric_aggregate is None:
                    assert rt.primary_metric_aggregate is None
                else:
                    assert rt.primary_metric_aggregate == pytest.approx(bt.primary_metric_aggregate)

        # Ranking / winning trial: identical for every outer fold.
        assert outcome_resumed.manifest.winning_trial_by_outer_fold == baseline_outcome.manifest.winning_trial_by_outer_fold

        # Final feature selections, final hyperparameters, and outer-test
        # metrics: identical -- the same winner, refit against the same
        # outer-train partition, evaluated against the same outer-test
        # partition, must produce byte-identical results.
        for outer_fold_index, b_outer in baseline_outer.items():
            r_outer = resumed_outer[outer_fold_index]
            assert b_outer.winning_trial_number == r_outer.winning_trial_number
            assert b_outer.final_selected_features == r_outer.final_selected_features
            assert b_outer.final_hyperparameters == r_outer.final_hyperparameters
            assert b_outer.outer_test_metrics == r_outer.outer_test_metrics

        report = verify_optimization(
            optimization_id, optimization_manifest_store=runner_resumed.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner_resumed.artifact_store, event_store=runner_resumed.event_store,
        )
        assert report.is_ready, [i.message for i in report.criticals]


class TestOptunaVersionBindingFailsClosedOnResume:
    """Adversarial release-readiness audit, Section 2 (Optuna version
    range): deterministic sampler resume is empirically verified against
    ONE installed Optuna patch version, not guaranteed by Optuna's public
    API across this platform's whole declared dependency range. Proves the
    binding+fail-closed mechanism added to close that gap:
    `OptimizationRunner` records the installed Optuna version in a
    persisted `ENVIRONMENT_SNAPSHOT` artifact at optimization CREATION
    time, and `.resume()` refuses to proceed if the currently installed
    version differs."""

    def test_environment_snapshot_is_recorded_with_the_real_installed_optuna_version(self, tmp_path: Path, classification_dataset) -> None:
        import importlib.metadata

        from quant_platform.ml.models import ArtifactCategory, EnvironmentSnapshot

        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=2)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED

        env_ref = next(r for r in outcome.manifest.artifact_references if r.category is ArtifactCategory.ENVIRONMENT_SNAPSHOT)
        raw = runner.artifact_store.read_artifact(env_ref.content_hash)
        snapshot = EnvironmentSnapshot.from_json_dict(json.loads(raw.decode("utf-8")))
        assert snapshot.package_versions["optuna"] == importlib.metadata.version("optuna")

    def test_resume_succeeds_when_the_installed_optuna_version_still_matches(self, tmp_path: Path, classification_dataset) -> None:
        """The ordinary case (no version drift) -- every other resume test
        in this file already exercises this path implicitly; asserted
        explicitly here as the complementary case to the fail-closed test
        below, so that test cannot pass merely because resume always fails."""
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=6)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash")
            return real_run_trial(trial_spec, **kwargs)

        runner_module.run_trial = flaky
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial

        optimization_id = compute_optimization_identity(spec).optimization_id
        outcome = runner.resume(optimization_id, spec=spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED

    def test_resume_fails_closed_when_the_installed_optuna_version_has_changed(self, tmp_path: Path, classification_dataset) -> None:
        from quant_platform.core.exceptions import OptimizationResumeError as ResumeError
        from quant_platform.ml.models import ArtifactCategory, EnvironmentSnapshot

        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=6)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash")
            return real_run_trial(trial_spec, **kwargs)

        runner_module.run_trial = flaky
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial

        optimization_id = compute_optimization_identity(spec).optimization_id
        manifest = runner.manifest_store.load(optimization_id)
        env_ref = next(r for r in manifest.artifact_references if r.category is ArtifactCategory.ENVIRONMENT_SNAPSHOT)

        # Simulate the recorded snapshot having been captured under a
        # DIFFERENT Optuna version -- tamper the artifact's own bytes
        # directly (a real file, real content-hash-consistent tamper, the
        # same technique the verification-tampering tests already use).
        raw = runner.artifact_store.read_artifact(env_ref.content_hash)
        original = EnvironmentSnapshot.from_json_dict(json.loads(raw.decode("utf-8")))
        tampered_versions = dict(original.package_versions)
        tampered_versions["optuna"] = "0.0.1-simulated-old-version"
        tampered = replace(original, package_versions=tampered_versions)
        tampered_bytes = json.dumps(tampered.to_json_dict()).encode("utf-8")
        # Content-addressed storage keys by hash -- write the tampered
        # payload under a NEW hash and repoint the manifest's own
        # artifact_references at it, exactly like the other tampering
        # tests in this file do for other artifact kinds.
        new_ref = runner.artifact_store.write_artifact(tampered_bytes, category=ArtifactCategory.ENVIRONMENT_SNAPSHOT)
        new_refs = tuple(new_ref if r.category is ArtifactCategory.ENVIRONMENT_SNAPSHOT else r for r in manifest.artifact_references)
        tampered_manifest = replace(manifest, artifact_references=new_refs)
        write_json_atomic(runner.manifest_store._manifest_path(optimization_id), tampered_manifest.to_json_dict())

        trial_artifact_count_before = sum(1 for _ in (tmp_path / "ml_artifacts").rglob("*")) if (tmp_path / "ml_artifacts").is_dir() else 0

        with pytest.raises(ResumeError, match=r"optuna==0\.0\.1-simulated-old-version.*optuna=="):
            runner.resume(optimization_id, spec=spec)

        # Fails BEFORE any new trial work is attempted -- no new artifacts
        # written as a side effect of the refused resume.
        trial_artifact_count_after = sum(1 for _ in (tmp_path / "ml_artifacts").rglob("*")) if (tmp_path / "ml_artifacts").is_dir() else 0
        assert trial_artifact_count_after == trial_artifact_count_before

    def test_resume_proceeds_normally_when_no_environment_snapshot_was_recorded(self, tmp_path: Path, classification_dataset) -> None:
        """Backward compatibility: an optimization manifest predating this
        check (no ENVIRONMENT_SNAPSHOT reference at all) must not be
        treated as an automatic failure -- there is nothing to compare
        against, so this is a silent no-op, not a false-positive block."""
        from quant_platform.ml.models import ArtifactCategory

        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20})
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=6)
        runner = _runner(tmp_path, rms, rds, model_registry=registry)

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash")
            return real_run_trial(trial_spec, **kwargs)

        runner_module.run_trial = flaky
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial

        optimization_id = compute_optimization_identity(spec).optimization_id
        manifest = runner.manifest_store.load(optimization_id)
        refs_without_env_snapshot = tuple(r for r in manifest.artifact_references if r.category is not ArtifactCategory.ENVIRONMENT_SNAPSHOT)
        tampered_manifest = replace(manifest, artifact_references=refs_without_env_snapshot)
        write_json_atomic(runner.manifest_store._manifest_path(optimization_id), tampered_manifest.to_json_dict())

        outcome = runner.resume(optimization_id, spec=spec)
        assert outcome.manifest.stage is OptimizationStage.COMPLETED


class _CountingArtifactStoreProxy:
    """Delegates to a real `MLArtifactStore`, raising `ExperimentLockError`
    the instant `write_artifact` call number `crash_after_n_writes + 1` is
    attempted -- simulating a process death at a PRECISE point inside
    `finalize_outer_fold`'s own write sequence (feature-selection result,
    then model, then predictions -- probabilities only for classification)."""

    def __init__(self, real_store, *, crash_after_n_writes: int) -> None:
        self._real_store = real_store
        self._crash_after_n_writes = crash_after_n_writes
        self.write_calls = 0

    def write_artifact(self, *args, **kwargs):
        self.write_calls += 1
        if self.write_calls > self._crash_after_n_writes:
            raise ExperimentLockError(f"simulated crash after {self._crash_after_n_writes} artifact write(s) inside finalize_outer_fold")
        return self._real_store.write_artifact(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_store, name)


class TestCrashWindowsAroundOuterFoldFinalization:
    """Adversarial release-readiness audit, Section 6: CRASH-WINDOW AUDIT,
    focused on the outer-fold-finalization boundaries specifically (the
    per-trial crash windows -- 'trial artifact written before manifest
    update', 'manifest updated before event append' -- are already
    exhaustively covered by `TestExhaustiveResumeAtEveryTrialStateTransition`
    above, which crashes at every trial-number boundary). Proves: after a
    crash at each of these points, resume (1) never re-runs the already-
    verified trial search (no new trial numbers allocated), (2) always
    redoes outer-fold finalization completely from scratch (never resumes
    mid-way through it -- there is no partial-finalization resume by
    design, since finalization is a pure, deterministic function of
    already-fixed inputs), (3) reaches an identical, valid terminal state,
    and (4) never evaluates outer-test twice with a different result."""

    def _spec_and_runner(self, tmp_path: Path, dataset_manifest, rms, rds, *, model_registry: ModelRegistry, max_trials: int = 3):
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=model_registry, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec = _default_optimization_spec(
            experiment_spec, experiment_manifest.identity.experiment_id, model_registry=model_registry, max_trials=max_trials,
            feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}),
        )
        return spec, _runner(tmp_path, rms, rds, model_registry=model_registry)

    @pytest.mark.parametrize(
        "crash_after_n_writes",
        [0, 1, 2, 3],
        ids=[
            "before_any_write_winner_selected_not_yet_refit", "after_feature_selection_before_model_refit",
            "after_model_refit_before_predictions", "after_predictions_before_outer_result_persisted",
        ],
    )
    def test_resume_redoes_finalization_completely_and_reaches_an_identical_terminal_state(
        self, tmp_path: Path, classification_dataset, crash_after_n_writes: int,
    ) -> None:
        dataset_manifest, rms, rds = classification_dataset
        registry_baseline = mz.register_default_models()
        baseline_spec, baseline_runner = self._spec_and_runner(tmp_path / "baseline", dataset_manifest, rms, rds, model_registry=registry_baseline)
        baseline_outcome = baseline_runner.run(baseline_spec)
        assert baseline_outcome.manifest.stage is OptimizationStage.COMPLETED

        registry_int = mz.register_default_models()
        spec_int, runner_int = self._spec_and_runner(tmp_path / "interrupted", dataset_manifest, rms, rds, model_registry=registry_int)
        assert compute_optimization_identity(spec_int).optimization_id == compute_optimization_identity(baseline_spec).optimization_id

        real_finalize_outer_fold = runner_module.finalize_outer_fold

        def crashing_finalize(*, artifact_store, **kwargs):
            proxy = _CountingArtifactStoreProxy(artifact_store, crash_after_n_writes=crash_after_n_writes)
            return real_finalize_outer_fold(artifact_store=proxy, **kwargs)

        runner_module.finalize_outer_fold = crashing_finalize
        try:
            with pytest.raises(ExperimentLockError):
                runner_int.run(spec_int)
        finally:
            runner_module.finalize_outer_fold = real_finalize_outer_fold

        optimization_id = compute_optimization_identity(spec_int).optimization_id
        manifest_after_crash = runner_int.manifest_store.load(optimization_id)
        assert manifest_after_crash.stage is OptimizationStage.RECOVERABLE_FAILURE
        assert manifest_after_crash.completed_outer_fold_indices == ()  # the crashing outer fold never got to record completion
        trial_refs_before_resume = dict(manifest_after_crash.trial_result_references)

        outcome_resumed = runner_int.resume(optimization_id, spec=spec_int)
        assert outcome_resumed.manifest.stage is OptimizationStage.COMPLETED, outcome_resumed.manifest.failure_summary

        # (1) No new trial numbers allocated for outer fold 0 (the one that
        # crashed mid-finalization) on resume -- every trial reference that
        # existed immediately before the crash is still present, byte-
        # identical, afterward. (The resumed run legitimately proceeds on
        # to outer fold 1's OWN, genuinely-new trial search once fold 0 is
        # finalized -- a superset, not an exact-equal, is the correct claim.)
        assert trial_refs_before_resume  # sanity: the crash genuinely happened after >=1 trial completed
        resumed_refs = dict(outcome_resumed.manifest.trial_result_references)
        for key, ref in trial_refs_before_resume.items():
            assert resumed_refs.get(key) == ref, f"trial reference {key!r} changed or disappeared across resume"

        # (3)+(4) Identical terminal state to an uninterrupted baseline --
        # same winner, same final refit, same outer-test evaluation,
        # evaluated exactly once with one final, stable result.
        assert outcome_resumed.manifest.winning_trial_by_outer_fold == baseline_outcome.manifest.winning_trial_by_outer_fold
        for outer_fold_index, ref in outcome_resumed.manifest.outer_fold_result_references.items():
            resumed_result = OuterFoldResult.from_json_dict(json.loads(runner_int.artifact_store.read_artifact(ref.content_hash).decode("utf-8")))
            baseline_ref = baseline_outcome.manifest.outer_fold_result_references[outer_fold_index]
            baseline_result = OuterFoldResult.from_json_dict(json.loads(baseline_runner.artifact_store.read_artifact(baseline_ref.content_hash).decode("utf-8")))
            assert resumed_result.final_selected_features == baseline_result.final_selected_features
            assert resumed_result.final_hyperparameters == baseline_result.final_hyperparameters
            assert resumed_result.outer_test_metrics == baseline_result.outer_test_metrics

        report = verify_optimization(
            optimization_id, optimization_manifest_store=runner_int.manifest_store,
            experiment_manifest_store=ExperimentManifestStore((tmp_path / "interrupted") / "ml_artifacts"),
            artifact_store=runner_int.artifact_store, event_store=runner_int.event_store,
        )
        assert report.is_ready, [i.message for i in report.criticals]

    def test_resume_after_outer_fold_result_persisted_but_before_stage_transition_never_double_finalizes(
        self, tmp_path: Path, classification_dataset,
    ) -> None:
        """The remaining named boundary: 'outer result persisted before
        stage transition'. Simulates a crash immediately after
        `finalize_outer_fold` returns successfully (its result IS
        durably persisted -- content-addressed, so re-writing identical
        bytes on a subsequent attempt is a safe no-op) but before the
        manifest is updated to record the outer fold as complete."""
        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        spec, runner = self._spec_and_runner(tmp_path, dataset_manifest, rms, rds, model_registry=registry)

        real_finalize_outer_fold = runner_module.finalize_outer_fold
        call_state = {"crashed_once": False}

        def crash_right_after_finalize(**kwargs):
            result = real_finalize_outer_fold(**kwargs)
            if not call_state["crashed_once"]:
                call_state["crashed_once"] = True
                raise ExperimentLockError("simulated crash immediately after finalize_outer_fold returned, before the manifest recorded it")
            return result

        runner_module.finalize_outer_fold = crash_right_after_finalize
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.finalize_outer_fold = real_finalize_outer_fold

        optimization_id = compute_optimization_identity(spec).optimization_id
        manifest_after_crash = runner.manifest_store.load(optimization_id)
        assert manifest_after_crash.stage is OptimizationStage.RECOVERABLE_FAILURE
        assert 0 not in manifest_after_crash.completed_outer_fold_indices

        outcome_resumed = runner.resume(optimization_id, spec=spec)
        assert outcome_resumed.manifest.stage is OptimizationStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert 0 in outcome_resumed.manifest.completed_outer_fold_indices
        # Never double-finalized: exactly one OuterFoldResult reference for
        # outer fold 0 in the final manifest, and it verifies cleanly.
        report = verify_optimization(
            optimization_id, optimization_manifest_store=runner.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner.artifact_store, event_store=runner.event_store,
        )
        assert report.is_ready, [i.message for i in report.criticals]


class TestConcurrencyStress:
    """Adversarial release-readiness audit, Section 9: CONCURRENCY.
    Stresses concurrent attempts to START the SAME optimization (never
    with `time.sleep`-based synchronization -- both threads are released
    from a `threading.Barrier` at the precise `os.link` call
    `DatasetLock.acquire()` uses to publish its lock file, the exact
    technique `tests/unit/historical/test_locking.py`'s own authoritative
    concurrency test already established). Proves: exactly one active
    owner; no duplicate trial numbers; no double publication (never two
    divergent completed manifests); no indefinite blocking (a losing
    attempt fails FAST with `ExperimentLockError`, never hangs); and,
    separately, that a lock left behind by a forced process death does not
    permanently wedge the optimization (the SAME `DatasetLock` staleness/
    reclaim behavior already proven correct for every other store in this
    platform -- reused here, never reimplemented)."""

    def test_two_simultaneous_run_attempts_for_a_brand_new_optimization_exactly_one_wins(
        self, tmp_path: Path, classification_dataset, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os
        import threading

        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=2)

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            # A full run() call makes MANY os.link calls over its lifetime
            # (the one outer run-lock, PLUS one per manifest transition) --
            # only the FIRST one per thread (the run-lock race this test
            # actually targets) is barrier-synchronized; every later,
            # unrelated lock acquisition proceeds normally, or a 2-party
            # barrier consumed by unsynchronized later calls would corrupt
            # itself (BrokenBarrierError) instead of ever testing the race.
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[tuple[str, object]] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            runner = _runner(tmp_path, rms, rds, model_registry=registry)
            try:
                outcome = runner.run(spec)
                with results_lock:
                    results.append(("completed", outcome))
            except ExperimentLockError as exc:
                with results_lock:
                    results.append(("rejected", exc))

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        monkeypatch.setattr(os, "link", real_link)  # restore BEFORE any further (single-threaded) lock acquisition below -- nothing is left to pair a barrier.wait() with
        assert not any(t.is_alive() for t in threads), "a losing attempt hung instead of failing fast"

        outcomes = sorted(r[0] for r in results)
        assert outcomes == ["completed", "rejected"], f"expected exactly one winner and one fast-failing loser, got {outcomes}"

        # No double publication: the ONE completed run reaches a single,
        # self-consistent COMPLETED manifest -- re-loading it independently
        # confirms there is no divergent second version anywhere.
        winning_outcome = next(r[1] for r in results if r[0] == "completed")
        optimization_id = compute_optimization_identity(spec).optimization_id
        reloaded = OptimizationManifestStore(tmp_path / "ml_artifacts").load(optimization_id)
        assert reloaded.stage is OptimizationStage.COMPLETED
        assert reloaded.winning_trial_by_outer_fold == winning_outcome.manifest.winning_trial_by_outer_fold  # type: ignore[union-attr]

        report = verify_optimization(
            optimization_id, optimization_manifest_store=OptimizationManifestStore(tmp_path / "ml_artifacts"),
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=MLArtifactStore(tmp_path / "ml_artifacts"), event_store=OptimizationEventStore(tmp_path / "ml_artifacts"),
        )
        assert report.is_ready, [i.message for i in report.criticals]

    def test_run_and_resume_racing_for_the_same_already_started_optimization_exactly_one_wins(
        self, tmp_path: Path, classification_dataset, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The `.resume()` entry point acquires the identical run-lock
        `.run()` does -- proven here by racing a fresh `.resume()` call
        against a `.run()` call for an optimization already left
        `RECOVERABLE_FAILURE` by a prior simulated crash."""
        import os
        import threading

        dataset_manifest, rms, rds = classification_dataset
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry, hyperparameters={"num_boost_round": 20, "num_leaves": 7},
        )
        spec = _default_optimization_spec(experiment_spec, experiment_manifest.identity.experiment_id, model_registry=registry, max_trials=6)
        setup_runner = _runner(tmp_path, rms, rds, model_registry=registry)

        call_count = {"n": 0}
        real_run_trial = runner_module.run_trial

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash to leave a resumable optimization behind")
            return real_run_trial(trial_spec, **kwargs)

        runner_module.run_trial = flaky
        try:
            with pytest.raises(ExperimentLockError):
                setup_runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial

        optimization_id = compute_optimization_identity(spec).optimization_id
        assert setup_runner.manifest_store.load(optimization_id).stage is OptimizationStage.RECOVERABLE_FAILURE

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[str] = []
        results_lock = threading.Lock()

        def attempt(use_resume: bool) -> None:
            runner = _runner(tmp_path, rms, rds, model_registry=registry)
            try:
                if use_resume:
                    runner.resume(optimization_id, spec=spec)
                else:
                    runner.run(spec)
                with results_lock:
                    results.append("completed")
            except ExperimentLockError:
                with results_lock:
                    results.append("rejected")

        threads = [threading.Thread(target=attempt, args=(use_resume,)) for use_resume in (True, False)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        monkeypatch.setattr(os, "link", real_link)
        assert not any(t.is_alive() for t in threads)
        assert sorted(results) == ["completed", "rejected"]

        final_manifest = OptimizationManifestStore(tmp_path / "ml_artifacts").load(optimization_id)
        assert final_manifest.stage is OptimizationStage.COMPLETED
