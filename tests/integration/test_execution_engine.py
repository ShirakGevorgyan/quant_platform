"""End-to-end Milestone 4B integration test: real synthetic historical
data -> a real Milestone 3 research dataset (via `ResearchDatasetBuilder`)
-> a real Milestone 4A prepared experiment (via `ExperimentPreparer`) ->
a real Milestone 4B walk-forward execution (via `ExecutionRunner`),
proving the full stack works together, not just each module in
isolation. Mirrors `tests/integration/test_ml_experiment_preparation.py`'s
conventions exactly."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.exceptions import ExperimentLockError
from quant_platform.core.types import Timeframe
from quant_platform.execution.executor import DeterministicFoldExecutor
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
    CodeRevisionBinding,
    DatasetBinding,
    ExperimentStatus,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory


def _build_real_research_dataset(
    tmp_path: Path, *, n_rows: int = 3000, seed: int = 11, horizon_bars: int = 5,
    preprocessing: dict[str, object] | None = None,
):
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
        label_definition=LabelDefinition(name="fut5", kind=LabelKind.FUTURE_RETURN, horizon_bars=horizon_bars),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        preprocessing=preprocessing or {},  # type: ignore[arg-type]
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, research_store


def _model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            name="constant_test_model", version="1", description="test",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION), supports_predict_proba=True),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        )
    )
    return registry


def _spec_for(dataset_manifest, **overrides: object) -> ExperimentSpec:
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
    base: dict[str, object] = {
        "dataset_binding": dataset_binding, "feature_binding": feature_binding,
        "label_binding": LabelBinding(name="fut5", kind="future_return", horizon_bars=5, label_type=LabelType.CONTINUOUS),
        "split_binding": SplitBinding(
            strategy="expanding_walk_forward", params={"n_splits": 3, "test_size": 100, "purge_bars": 5, "embargo_bars": 2},
        ),
        "preprocessing_binding": preprocessing_binding,
        "model_name": "constant_test_model", "model_version": "1",
        "hyperparameters": ModelHyperparameters(),
        "objective": ObjectiveType.REGRESSION,
        "seed_configuration": SeedConfiguration(master_seed=1),
        "code_revision_binding": CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        "primary_metric": "rmse",
    }
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


def _prepare_ready_experiment(tmp_path: Path, **spec_overrides: object):
    dataset_manifest, research_manifest_store, research_store = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(dataset_manifest, **spec_overrides)
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary
    return manifest, research_manifest_store, research_store


def _runner(tmp_path: Path, research_manifest_store, research_store, *, fold_executor=None) -> ExecutionRunner:
    return ExecutionRunner(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        fold_executor=fold_executor,
    )


def test_full_walk_forward_execution_against_real_research_dataset(tmp_path: Path) -> None:
    manifest, rms, rds = _prepare_ready_experiment(tmp_path)
    runner = _runner(tmp_path, rms, rds)
    outcome = runner.run(manifest.identity.experiment_id)

    assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
    assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
    assert outcome.aggregate.failed_fold_indices == ()

    reloaded_experiment = ExperimentManifestStore(tmp_path / "ml_artifacts").load(manifest.identity.experiment_id)
    assert reloaded_experiment.status is ExperimentStatus.COMPLETED

    events = [e.event_type.value for e in runner.event_store.read_events(manifest.identity.experiment_id)]
    assert events == [
        "experiment_created", "validation_started", "validation_passed", "run_started",
        "fold_started", "fold_completed", "fold_started", "fold_completed", "fold_started", "fold_completed",
        "run_completed",
    ]


def test_resume_after_simulated_interruption_against_real_dataset(tmp_path: Path) -> None:
    manifest, rms, rds = _prepare_ready_experiment(tmp_path)

    class InterruptOnFoldTwo:
        def __init__(self) -> None:
            self.attempts: dict[int, int] = {}

        def execute(self, context, **kwargs):
            self.attempts[context.fold_index] = self.attempts.get(context.fold_index, 0) + 1
            if context.fold_index == 2 and self.attempts[context.fold_index] == 1:
                raise ExperimentLockError("simulated interruption")
            return DeterministicFoldExecutor().execute(context, **kwargs)

    flaky = InterruptOnFoldTwo()
    runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
    with pytest.raises(ExperimentLockError):
        runner.run(manifest.identity.experiment_id)

    exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    assert exec_manifest.stage is ExecutionStage.RECOVERABLE_FAILURE
    assert exec_manifest.completed_fold_indices == (0, 1)

    outcome = runner.resume(manifest.identity.experiment_id)
    assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
    assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
    # Folds 0 and 1 must never have been re-executed.
    assert flaky.attempts == {0: 1, 1: 1, 2: 2}


def test_duplicate_parallel_execution_never_corrupts_real_dataset_run(tmp_path: Path) -> None:
    manifest, rms, rds = _prepare_ready_experiment(tmp_path)
    runner_a = _runner(tmp_path, rms, rds)
    runner_b = _runner(tmp_path, rms, rds)
    results: list[object] = []
    lock = threading.Lock()

    def run(runner: ExecutionRunner) -> None:
        try:
            outcome = runner.run(manifest.identity.experiment_id)
            with lock:
                results.append(outcome)
        except ExperimentLockError as exc:
            with lock:
                results.append(exc)

    threads = [threading.Thread(target=run, args=(r,)) for r in (runner_a, runner_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    final = runner_a.execution_manifest_store.load(manifest.identity.experiment_id)
    assert final.stage is ExecutionStage.COMPLETED
    assert final.completed_fold_indices == (0, 1, 2)
    assert len(final.fold_result_references) == 3


def test_purge_below_real_dataset_label_horizon_is_rejected_end_to_end(tmp_path: Path) -> None:
    """Full-stack reproduction of the audit's PRIMARY BLOCKER example
    (horizon_bars=12, purge_bars=0, embargo_bars=0) against a REAL
    `ResearchDatasetBuilder`-built dataset -- not the lighter synthetic
    fixture `test_runner.py` uses. Must be rejected before any fold is
    fit/predicted."""
    dataset_manifest, research_manifest_store, research_store = _build_real_research_dataset(tmp_path, horizon_bars=12)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(
        dataset_manifest,
        label_binding=LabelBinding(name="fut5", kind="future_return", horizon_bars=12, label_type=LabelType.CONTINUOUS),
        split_binding=SplitBinding(
            strategy="expanding_walk_forward", params={"n_splits": 3, "test_size": 100, "purge_bars": 0, "embargo_bars": 0},
        ),
    )
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary

    runner = _runner(tmp_path, research_manifest_store, research_store)
    with pytest.raises(Exception, match="insufficient_label_horizon_purge"):
        runner.run(manifest.identity.experiment_id)

    exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    assert exec_manifest.stage is ExecutionStage.FAILED
    assert dict(exec_manifest.fold_result_references) == {}
    events = [e.event_type.value for e in runner.event_store.read_events(manifest.identity.experiment_id)]
    assert "fold_started" not in events


def test_unsafe_globally_fitted_preprocessing_is_rejected_before_any_fold_runs(tmp_path: Path) -> None:
    """A dataset built with a REAL, globally-fitted `TransformPipeline`
    (`return_simple_1` standard-scaled) must be refused by the execution
    engine -- Milestone 4B's independent re-splitting does not align
    with whatever fold-group boundaries Milestone 3 fit that transform
    against (see `execution.runner.
    assert_preprocessing_is_safe_for_execution`)."""
    from quant_platform.features.normalization import TransformKind

    dataset_manifest, research_manifest_store, research_store = _build_real_research_dataset(
        tmp_path, preprocessing={"return_simple_1": TransformKind.STANDARD_SCALE},
    )
    assert dataset_manifest.fitted_preprocessing_fingerprint is not None
    assert dataset_manifest.preprocessing_definition == {"return_simple_1": "standard_scale"}

    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(
        dataset_manifest,
        preprocessing_binding=PreprocessingBinding(
            preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
            fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
        ),
    )
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary

    runner = _runner(tmp_path, research_manifest_store, research_store)
    with pytest.raises(Exception, match="fitted preprocessing"):
        runner.run(manifest.identity.experiment_id)

    exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    assert exec_manifest.stage is ExecutionStage.FAILED
    assert dict(exec_manifest.fold_result_references) == {}


def test_causal_only_dataset_explicitly_satisfies_preprocessing_safety_check(tmp_path: Path) -> None:
    """Contrast case for the rejection test above, using the SAME real
    `ResearchDatasetBuilder` pipeline with no `preprocessing` requested:
    proves the safe state is the DEFAULT real-pipeline output, not a
    hand-constructed special case."""
    from quant_platform.execution.runner import assert_preprocessing_is_safe_for_execution

    dataset_manifest, _, _ = _build_real_research_dataset(tmp_path)
    assert dataset_manifest.preprocessing_definition == {}
    assert dataset_manifest.fitted_preprocessing_fingerprint is None
    assert_preprocessing_is_safe_for_execution(dataset_manifest)  # must not raise


def test_corrupted_fold_artifact_forces_rerun_on_resume(tmp_path: Path) -> None:
    """Simulates the "corrupted fold artifact" recovery case (Section 10):
    a fold that the manifest CLAIMS is complete, but whose content has
    been corrupted on disk, must be verified as needing a rerun, never
    trusted blindly -- exercised here via the real `ExecutionRunner`/
    `MLArtifactStore` stack, not just `execution.resume`'s unit tests."""
    manifest, rms, rds = _prepare_ready_experiment(tmp_path)
    runner = _runner(tmp_path, rms, rds)
    runner.run(manifest.identity.experiment_id)

    exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    corrupted_ref = exec_manifest.fold_result_references[0]
    content_path = runner.artifact_store._content_path(corrupted_ref.content_hash)
    content_path.write_bytes(b"CORRUPTED FOLD RESULT")

    # A completed execution is terminal -- corruption discovered post-hoc
    # is reported via verify-execution/read_artifact, not silently healed.
    from quant_platform.core.exceptions import ArtifactCorruptionError

    with pytest.raises(ArtifactCorruptionError):
        runner.artifact_store.read_artifact(corrupted_ref.content_hash)


def test_resume_replaces_a_corrupted_completed_fold_and_rebuilds_the_aggregate(tmp_path: Path) -> None:
    """RESUME REPLACEMENT AUDIT, end-to-end: a NON-terminal execution
    (stopped by a simulated transient failure on fold 2, after folds 0
    and 1 completed) has fold 0's artifact corrupted on disk BEFORE
    `resume()` is called. Proves, against the real stack: fold 0 is
    verified-corrupted and rerun (never trusted); fold 1 is verified-
    intact and skipped (never needlessly rerun); the final aggregate is
    rebuilt from the CURRENTLY verified fold results (fold 0's fresh
    reference, not the corrupted one); completed/failed sets stay
    disjoint; no fold index is duplicated."""
    manifest, rms, rds = _prepare_ready_experiment(tmp_path)

    class InterruptOnFoldTwo:
        def __init__(self) -> None:
            self.attempts: dict[int, int] = {}

        def execute(self, context, **kwargs):
            self.attempts[context.fold_index] = self.attempts.get(context.fold_index, 0) + 1
            if context.fold_index == 2 and self.attempts[context.fold_index] == 1:
                raise ExperimentLockError("simulated interruption")
            return DeterministicFoldExecutor().execute(context, **kwargs)

    flaky = InterruptOnFoldTwo()
    runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
    with pytest.raises(ExperimentLockError):
        runner.run(manifest.identity.experiment_id)

    exec_manifest_before = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    assert exec_manifest_before.stage is ExecutionStage.RECOVERABLE_FAILURE
    assert exec_manifest_before.completed_fold_indices == (0, 1)
    old_fold_0_ref = exec_manifest_before.fold_result_references[0]
    runner.artifact_store._content_path(old_fold_0_ref.content_hash).write_bytes(b"CORRUPTED FOLD RESULT")

    outcome = runner.resume(manifest.identity.experiment_id)

    assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
    assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
    assert outcome.aggregate.failed_fold_indices == ()
    # Fold 0 ran twice in total (once in the original run, once more here
    # because its artifact was corrupted); fold 2 ran twice (the first
    # attempt raised); fold 1 (verified intact both times) ran only once,
    # in the original run -- proving it was correctly skipped here.
    assert flaky.attempts == {0: 2, 1: 1, 2: 2}

    exec_manifest_after = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    # Exactly one entry per fold index -- no duplicates, disjoint sets.
    assert sorted(exec_manifest_after.fold_result_references) == [0, 1, 2]
    assert set(exec_manifest_after.completed_fold_indices) & set(exec_manifest_after.failed_fold_indices) == set()
    new_fold_0_ref = exec_manifest_after.fold_result_references[0]
    assert new_fold_0_ref.content_hash != old_fold_0_ref.content_hash  # old reference was replaced, not reused

    # The replacement is genuinely readable/decodable (the corrupted one was not).
    import json as _json

    from quant_platform.execution.results import FoldResult

    raw = runner.artifact_store.read_artifact(new_fold_0_ref.content_hash)
    fresh_fold_0 = FoldResult.from_json_dict(_json.loads(raw.decode("utf-8")))
    assert fresh_fold_0.fold_index == 0
    assert fresh_fold_0.status.value == "completed"

    # verify-execution confirms the rebuilt state is fully consistent.
    from quant_platform.execution.verification import verify_execution
    from quant_platform.ml.manifests import ExperimentManifestStore

    report = verify_execution(
        manifest.identity.experiment_id,
        execution_manifest_store=runner.execution_manifest_store,
        experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
        artifact_store=runner.artifact_store,
        event_store=runner.event_store,
    )
    assert report.is_ready
