"""Performance benchmarks for the Milestone 4D leakage-safe feature-
selection/hyperparameter-optimization engine. Same philosophy as
`test_execution_throughput.py`: conservative floors (roughly 10x-100x
below measured numbers on reference hardware) to catch a severe
accidental regression without being flaky on a slower CI runner -- these
are NOT production throughput guarantees, and no safety check (leakage
validation, artifact hash re-verification, ranking recomputation) is ever
skipped to make a number look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%):
  - `build_inner_fold_plan` (1000-row timeline, one outer fold's ~700-row
    train partition, 3 inner splits), 500 iterations: 0.227ms/iter
    median, ~4,400 iter/sec.
  - `run_trial` (one trial, 3 inner folds, `ConstantTestModelFactory`,
    `FeatureSelectionStrategy.NONE`), 100 iterations: 18.3ms/iter median,
    ~55 iter/sec.
  - `OptimizationRunner.run` (full nested pipeline: 2 outer folds x 3
    trials x 2 inner folds each, `constant_test_model`, against a fresh,
    DISTINCT optimization every iteration, never an idempotent no-op), 15
    iterations: 332ms/iter median, ~3/sec.
  - `ask_next_trial` (TPE sampler, 6-parameter mixed search space, pure
    sampling, no fit/predict), 500 iterations: 8.0ms/iter median,
    ~124 iter/sec.
  - `rank_trials` (20 trials x 3 inner folds each, in-memory only, no
    I/O), 1000 iterations: 0.616ms/iter median, ~1,620 iter/sec.
  - `MLArtifactStore.write_artifact` for a `TrialResult` JSON payload
    (unique content per call), 500 iterations: 2.64ms/iter median,
    ~380/sec (dominated by the same per-call file-lock/rename overhead
    documented in the Milestone 4A/4B/4C benchmarks, not payload size).
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)
from tests.unit.optimization.conftest import make_optimization_spec

from quant_platform.execution.splitters import Fold, required_label_purge_bars_for
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, CodeRevisionBinding, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.candidates import (
    InnerFoldTrialMetrics,
    TrialResult,
    TrialSpec,
    TrialStatus,
    rank_trials,
)
from quant_platform.optimization.feature_selection import (
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
)
from quant_platform.optimization.inner_splits import (
    InnerFold,
    InnerFoldPlan,
    InnerSplitConfig,
    build_inner_fold_plan,
)
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
)
from quant_platform.optimization.runner import OptimizationRunner
from quant_platform.optimization.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    build_search_space,
)
from quant_platform.optimization.study import ask_next_trial, create_study
from quant_platform.optimization.trial_executor import run_trial

pytestmark = pytest.mark.performance

_TS = pd.Timestamp("2024-01-01")


def _timed_iterations(fn, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> float:
    median = statistics.median(timings)
    rate = 1.0 / median if median > 0 else float("inf")
    print(f"\n{label}: n={len(timings)} median={median * 1000:.3f}ms p95={sorted(timings)[int(len(timings) * 0.95)] * 1000:.3f}ms rate={rate:,.0f}/sec")
    return median


class TestInnerSplitConstructionThroughput:
    def test_build_inner_fold_plan(self) -> None:
        timeline = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=1000, freq="1min", tz="UTC")})
        outer_fold = Fold(
            fold_index=0, train_indices=np.arange(0, 700), test_indices=np.arange(700, 900),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[699],
            test_start=timeline["open_time"].iloc[700], test_end=timeline["open_time"].iloc[899],
        )
        config = InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15)
        median = _report(
            "build_inner_fold_plan (1000-row timeline, 700-row outer-train, 3 inner splits)",
            _timed_iterations(lambda: build_inner_fold_plan(outer_fold, config=config, label_horizon_bars=5, timeline=timeline, timestamp_column="open_time"), 500),
        )
        assert median < 0.03, "building a 3-split inner fold plan over a 700-row partition should not take >30ms (100x the measured floor)"


def _trial_timeline(n_rows: int = 200, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows), "label": rng.normal(size=n_rows)})


def _trial_inner_plan() -> InnerFoldPlan:
    purge = required_label_purge_bars_for(1)
    folds = tuple(
        InnerFold(
            inner_fold_index=i, train_indices=np.arange(0, 100 + i * 20), validation_indices=np.arange(100 + i * 20, 120 + i * 20),
            train_start=_TS, train_end=_TS, validation_start=_TS, validation_end=_TS,
        )
        for i in range(3)
    )
    return InnerFoldPlan(
        schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=folds, purge_bars=purge,
        embargo_bars=0, label_horizon_bars=1, required_label_purge_bars=purge, outer_train_row_count=200,
    )


class TestTrialExecutionThroughput:
    def test_run_trial(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        universe = FeatureUniverse(feature_names=("f1", "f2"), fingerprint="a" * 64)
        plan = _trial_inner_plan()
        timeline = _trial_timeline()
        counter = {"i": 0}

        def run() -> None:
            counter["i"] += 1
            spec = TrialSpec(
                schema_version=1, optimization_id="a" * 64, outer_fold_index=0, trial_number=counter["i"],
                sampled_hyperparameters={"alpha": 0.1}, feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE),
                trial_seed=counter["i"], inner_split_plan_fingerprint="b" * 64, model_definition_fingerprint="c" * 64,
                objective=ObjectiveType.REGRESSION, primary_metric="rmse",
            )
            run_trial(
                spec, inner_fold_plan=plan, timeline=timeline, feature_universe=universe, model_name="constant_test_model",
                model_factory=ConstantTestModelFactory(), seed_configuration=SeedConfiguration(master_seed=counter["i"]),
                min_successful_inner_folds=1, early_stopping_config=EarlyStoppingConfig(enabled=False), artifact_store=store,
            )

        median = _report("run_trial (3 inner folds, ConstantTestModelFactory, NONE feature selection)", _timed_iterations(run, 100))
        assert median < 0.5, "a single trial's 3-inner-fold loop should not take >500ms (~100x the measured floor)"


class TestOptimizationRunThroughput:
    def test_run_distinct_optimizations(self, tmp_path: Path) -> None:
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        registry = build_registry()
        experiment_manifest_store = ExperimentManifestStore(tmp_path / "ml")
        preparer = ExperimentPreparer(ml_artifacts_root=tmp_path / "ml", model_registry=registry, research_manifest_store=research_manifest_store)
        runner = OptimizationRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=registry, research_manifest_store=research_manifest_store,
            research_dataset_store=research_store, experiment_manifest_store=experiment_manifest_store,
        )
        search_space = build_search_space([IntegerParameter(name="dummy", low=1, high=10)])
        counter = {"i": 0}

        def run() -> None:
            counter["i"] += 1
            exp_spec = ExperimentSpec(**make_experiment_spec_kwargs(
                dataset_manifest=dataset_manifest, split_params={"n_splits": 2, "test_size": 100, "purge_bars": 5, "embargo_bars": 2},
                code_revision_binding=CodeRevisionBinding(revision=f"{counter['i']:040x}", source="git", is_dirty=True),
            ))
            experiment_manifest = preparer.prepare(exp_spec)
            opt_spec = build_optimization_spec(
                experiment=exp_spec, parent_experiment_id=experiment_manifest.identity.experiment_id, model_name="constant_test_model",
                model_version="1", primary_metric="rmse", inner_split_config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
                feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE), search_space=search_space,
                sampler_kind=SamplerKind.TPE, pruning_config=PruningConfig(kind=PruningKind.NONE), early_stopping_config=EarlyStoppingConfig(enabled=False),
                max_trials=3, min_successful_inner_folds=1, seed_configuration=SeedConfiguration(master_seed=counter["i"]),
            )
            runner.run(opt_spec)

        median = _report("OptimizationRunner.run (distinct optimizations, 2 outer folds x 3 trials x 2 inner folds)", _timed_iterations(run, 15))
        assert median < 3.5, "a full small nested-CV optimization run should not take >3.5s (~10x the measured floor)"


class TestOptunaSamplingThroughput:
    def test_ask_next_trial(self) -> None:
        space = build_search_space([
            IntegerParameter(name="num_leaves", low=4, high=64), FloatParameter(name="learning_rate", low=1e-4, high=1.0, log=True),
            CategoricalParameter(name="boosting", choices=("gbdt", "dart", "goss")), IntegerParameter(name="max_depth", low=2, high=12),
            FloatParameter(name="subsample", low=0.5, high=1.0), IntegerParameter(name="num_boost_round", low=50, high=500),
        ])
        spec = make_optimization_spec(search_space=space)
        study = create_study(spec)

        def ask() -> None:
            trial, _values = ask_next_trial(study, space)
            study.tell(trial, 0.5)

        median = _report("ask_next_trial (TPE sampler, 6-parameter mixed search space)", _timed_iterations(ask, 500))
        assert median < 0.1, "a single TPE suggestion over a 6-parameter space should not take >100ms (~12x the measured floor)"


def _trial_result(trial_number: int, *, value: float) -> TrialResult:
    metrics = tuple(
        InnerFoldTrialMetrics(inner_fold_index=i, primary_metric_value=value + i * 0.01, secondary_metrics={}, selected_feature_count=5, feature_selection_result_reference=None, best_iteration=None, duration_seconds=0.1)
        for i in range(3)
    )
    return TrialResult(
        schema_version=1, optimization_id="a" * 64, outer_fold_index=0, trial_number=trial_number, status=TrialStatus.COMPLETED,
        sampled_hyperparameters={"num_boost_round": 100 + trial_number}, inner_fold_metrics=metrics, primary_metric_aggregate=value,
        successful_inner_folds=3, total_inner_folds=3, duration_seconds=1.0,
    )


class TestRankingThroughput:
    def test_rank_trials(self) -> None:
        rng = np.random.default_rng(0)
        trials = [_trial_result(i, value=float(rng.normal())) for i in range(20)]
        median = _report("rank_trials (20 trials x 3 inner folds each, in-memory)", _timed_iterations(lambda: rank_trials(trials, primary_metric="rmse"), 1000))
        assert median < 0.01, "ranking 20 trials should not take >10ms (~100x the measured floor)"


class TestArtifactWriteThroughput:
    def test_write_trial_result_artifact(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        counter = {"i": 0}

        def write() -> None:
            counter["i"] += 1
            result = _trial_result(counter["i"], value=0.5)
            store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)

        median = _report("MLArtifactStore.write_artifact (TrialResult JSON, unique content)", _timed_iterations(write, 500))
        assert median < 0.05, "a single trial-result artifact write should not take >50ms"
