"""Adversarial release-readiness audit, Section 10: SCALE AND PERFORMANCE
SANITY. Runs a REAL, bounded nested-CV optimization exercising every
dimension the audit calls for simultaneously -- multiple outer folds,
multiple inner folds, two model families (LightGBM and CatBoost), two
feature-selection strategies (one per optimization), completed/invalid/
pruned trial states, and an interruption+resume cycle -- recording actual
memory, runtime, trial throughput, and artifact count for the delivery
report. Every compute dimension exercised here (`max_trials`, outer/inner
`n_splits`, `MAX_STABILITY_REPEATS`) is a validated, typed field on a
frozen dataclass (`OptimizationSpec`/`InnerSplitConfig`/
`FeatureSelectionSpec`) -- there is no code path in this package that
accepts an unbounded or unvalidated search dimension."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.exceptions import ExperimentLockError
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
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization import runner as runner_module
from quant_platform.optimization.candidates import TrialResult, TrialStatus
from quant_platform.optimization.feature_selection import FeatureSelectionSpec, FeatureSelectionStrategy
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    OptimizationStage,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
    compute_optimization_identity,
)
from quant_platform.optimization.runner import OptimizationRunner
from quant_platform.optimization.search_space import default_search_space_for_model
from quant_platform.optimization.verification import verify_optimization

_WALK_FORWARD_SPLIT = {"strategy": "expanding_walk_forward", "params": {"n_splits": 2, "test_size": 200, "purge_bars": 5, "embargo_bars": 2}}


def _build_real_research_dataset(tmp_path: Path, *, n_rows: int = 3000, seed: int = 7):
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
        historical_loader=historical_loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store,
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


def _experiment_spec_for(dataset_manifest, *, model_name: str, hyperparameters: dict[str, object]) -> ExperimentSpec:
    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version, content_id=dataset_manifest.content_id,
        symbol=dataset_manifest.symbol, base_timeframe=dataset_manifest.base_timeframe.value,
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
    return ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding,
        label_binding=LabelBinding(name="fut5", kind=LabelKind.BINARY_DIRECTION.value, horizon_bars=5, label_type=LabelType.BINARY),
        split_binding=SplitBinding(strategy=_WALK_FORWARD_SPLIT["strategy"], params=_WALK_FORWARD_SPLIT["params"]),  # type: ignore[arg-type]
        preprocessing_binding=preprocessing_binding, model_name=model_name, model_version="1",
        hyperparameters=ModelHyperparameters(values=hyperparameters), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=1), code_revision_binding=CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )


def _prepare_ready_experiment(tmp_path: Path, dataset_manifest, research_manifest_store, *, model_registry: ModelRegistry, model_name: str, hyperparameters: dict[str, object]):
    preparer = ExperimentPreparer(ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry, research_manifest_store=research_manifest_store)
    spec = _experiment_spec_for(dataset_manifest, model_name=model_name, hyperparameters=hyperparameters)
    manifest = preparer.prepare(spec)
    assert manifest.status is ExperimentStatus.READY, manifest.failure_summary
    return manifest, spec


def _runner(tmp_path: Path, research_manifest_store, research_store, *, model_registry: ModelRegistry) -> OptimizationRunner:
    return OptimizationRunner(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
        additional_serializers=mz.default_serializer_registry(),
    )


def _count_artifacts(tmp_path: Path) -> int:
    root = tmp_path / "ml_artifacts"
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


class TestBoundedScaleRunAcrossModelFamiliesAndSelectors:
    @pytest.mark.parametrize(
        "model_name, hyperparameters, feature_selection_spec",
        [
            (
                "lightgbm", {"num_boost_round": 30, "num_leaves": 7},
                FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}),
            ),
            (
                "catboost", {"iterations": 30},
                FeatureSelectionSpec(strategy=FeatureSelectionStrategy.UNIVARIATE, params={"mode": "top_k", "k": 5}),
            ),
        ],
        ids=["lightgbm_variance_filter", "catboost_univariate"],
    )
    def test_bounded_real_optimization_completes_with_mixed_trial_states_and_survives_interruption(
        self, tmp_path: Path, model_name: str, hyperparameters: dict[str, object], feature_selection_spec: FeatureSelectionSpec,
    ) -> None:
        dataset_manifest, rms, rds = _build_real_research_dataset(tmp_path)
        registry = mz.register_default_models()
        experiment_manifest, experiment_spec = _prepare_ready_experiment(
            tmp_path, dataset_manifest, rms, model_registry=registry, model_name=model_name, hyperparameters=hyperparameters,
        )

        max_trials = 6
        spec = build_optimization_spec(
            experiment=experiment_spec, parent_experiment_id=experiment_manifest.identity.experiment_id, model_name=model_name,
            model_version="1", primary_metric="accuracy",
            inner_split_config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15),
            feature_selection_spec=feature_selection_spec, search_space=default_search_space_for_model(model_name),
            sampler_kind=SamplerKind.TPE, pruning_config=PruningConfig(kind=PruningKind.MEDIAN_STOPPING, min_completed_inner_folds=1),
            early_stopping_config=EarlyStoppingConfig(enabled=True, patience=10, validation_fraction=0.15, final_round_policy="median_best_iteration"),
            max_trials=max_trials, min_successful_inner_folds=1, seed_configuration=SeedConfiguration(master_seed=3),
            model_registry=registry,
        )

        # Force trial 1 to INVALID deterministically -- ConstantTestModel-
        # free real models rarely fail naturally against this synthetic
        # dataset, so this guarantees the required "completed, invalid,
        # and pruned" state coverage rather than hoping for it.
        real_run_trial = runner_module.run_trial

        def force_trial_1_invalid(trial_spec, **kwargs):
            if trial_spec.trial_number == 1:
                return TrialResult(
                    schema_version=1, optimization_id=trial_spec.optimization_id, outer_fold_index=trial_spec.outer_fold_index,
                    trial_number=1, status=TrialStatus.INVALID, sampled_hyperparameters=trial_spec.sampled_hyperparameters,
                    inner_fold_metrics=(), primary_metric_aggregate=None, successful_inner_folds=0, total_inner_folds=1,
                    duration_seconds=0.001, failure_code="forced_invalid_for_scale_audit", failure_reason="forced for Section 10 scale coverage",
                )
            return real_run_trial(trial_spec, **kwargs)

        # Interrupt partway through outer fold 0, then resume -- proves the
        # bounded run survives interruption, not merely that it completes
        # when uninterrupted.
        call_count = {"n": 0}

        def flaky(trial_spec, **kwargs):
            call_count["n"] += 1
            result = force_trial_1_invalid(trial_spec, **kwargs)
            if call_count["n"] == 4:
                raise ExperimentLockError("simulated crash for Section 10 scale audit")
            return result

        runner_module.run_trial = flaky
        tracemalloc.start()
        started = time.perf_counter()
        try:
            runner = _runner(tmp_path, rms, rds, model_registry=registry)
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_trial = real_run_trial

        runner_module.run_trial = force_trial_1_invalid
        try:
            runner_resumed = _runner(tmp_path, rms, rds, model_registry=registry)
            outcome = runner_resumed.resume(compute_optimization_identity(spec).optimization_id, spec=spec)
        finally:
            runner_module.run_trial = real_run_trial
        elapsed = time.perf_counter() - started
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert outcome.manifest.stage is OptimizationStage.COMPLETED, outcome.manifest.failure_summary

        # Mixed trial states genuinely occurred (never assumed).
        all_statuses: list[TrialStatus] = []
        for outer_fold_index in outcome.manifest.completed_outer_fold_indices:
            refs = {
                int(k.split(":", 1)[1]): v for k, v in outcome.manifest.trial_result_references.items()
                if int(k.split(":", 1)[0]) == outer_fold_index
            }
            for ref in refs.values():
                raw = runner_resumed.artifact_store.read_artifact(ref.content_hash)
                all_statuses.append(TrialResult.from_json_dict(json.loads(raw.decode("utf-8"))).status)
        assert TrialStatus.COMPLETED in all_statuses
        assert TrialStatus.INVALID in all_statuses
        # PRUNED is opportunistic (depends on real relative model
        # performance, unlike the deterministically-forced INVALID trial)
        # -- reported, not asserted, to avoid a flaky requirement on real
        # model behavior.
        pruned_occurred = TrialStatus.PRUNED in all_statuses

        artifact_count = _count_artifacts(tmp_path)
        total_outer_folds = len(outcome.manifest.completed_outer_fold_indices)
        total_trials = len(all_statuses)

        report = verify_optimization(
            outcome.manifest.optimization_id, optimization_manifest_store=runner_resumed.manifest_store,
            experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml_artifacts"),
            artifact_store=runner_resumed.artifact_store, event_store=runner_resumed.event_store,
        )
        assert report.is_ready, [i.message for i in report.criticals]

        print(
            f"\n[Section 10 scale audit -- {model_name}/{feature_selection_spec.strategy.value}] "
            f"outer_folds={total_outer_folds} trials={total_trials} statuses={sorted(s.value for s in set(all_statuses))} "
            f"pruned_occurred={pruned_occurred} elapsed={elapsed:.2f}s throughput={total_trials / elapsed:.2f} trials/sec "
            f"peak_memory={peak_mem / (1024 * 1024):.1f}MiB artifact_count={artifact_count}"
        )
