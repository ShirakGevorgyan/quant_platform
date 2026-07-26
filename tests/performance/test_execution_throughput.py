"""Performance benchmarks for the Milestone 4B execution engine (Section
17). Same philosophy as `test_ml_infrastructure_throughput.py`:
conservative floors (roughly 10x-100x below measured numbers on
reference hardware) to catch a severe accidental regression without
being flaky on a slower CI runner -- these are NOT production throughput
guarantees, and no integrity verification (content-hash recomputation on
every artifact read, fold-plan validation before every run) is ever
skipped to make a number look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%):
  - `generate_expanding_folds` (2000-row timeline, 5 folds, purge+embargo),
    500 iterations: 0.177ms/iter median, ~5,660 iter/sec.
  - `ExecutionRunner.run` (full pipeline: reconstruct timeline, build+
    validate fold plan, run 3 folds via `DeterministicFoldExecutor`
    -- real fit/predict/serialize/write per fold -- aggregate, all
    against a fresh, DISTINCT experiment every iteration, never an
    idempotent no-op), 20 iterations: 96.9ms/iter median, ~10/sec.
  - `ExecutionRunner.resume` (idempotent no-op path: execution already
    COMPLETED, resume only re-reads the stored aggregate), 200
    iterations: 2.60ms/iter median, ~384/sec.
  - `MLArtifactStore.write_artifact` for a `FoldResult` JSON payload
    (unique content per call), 500 iterations: 2.21ms/iter median,
    ~453/sec (dominated by the same per-call file-lock/rename overhead
    documented in the Milestone 4A benchmarks, not the payload size).
  - `AggregatedExecutionResult` construction + canonical serialization
    (in-memory only, no I/O), 2000 iterations: 0.010ms/iter median,
    ~100,000/sec.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    make_timeline,
    write_synthetic_research_dataset,
)

from quant_platform.execution.results import AggregatedExecutionResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.splitters import generate_expanding_folds
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import ArtifactCategory, CodeRevisionBinding
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now

pytestmark = pytest.mark.performance


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


class TestSplitGenerationThroughput:
    def test_generate_expanding_folds(self) -> None:
        timeline = make_timeline(2000)
        median = _report(
            "generate_expanding_folds (2000 rows, 5 folds)",
            _timed_iterations(
                lambda: generate_expanding_folds(
                    timeline["open_time"], n_splits=5, test_size=100, purge_bars=5, embargo_bars=2, label_horizon_bars=5,
                ),
                500,
            ),
        )
        assert median < 0.02, "generating 5 folds over 2000 rows should not take >20ms (100x the measured floor)"


class TestWalkForwardExecutionThroughput:
    def test_run_distinct_experiments(self, tmp_path: Path) -> None:
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store,
        )
        counter = {"i": 0}

        def run() -> None:
            counter["i"] += 1
            spec = ExperimentSpec(**make_experiment_spec_kwargs(
                dataset_manifest=dataset_manifest,
                code_revision_binding=CodeRevisionBinding(revision=f"{counter['i']:040x}", source="git", is_dirty=True),
            ))
            manifest = preparer.prepare(spec)
            runner.run(manifest.identity.experiment_id)

        median = _report("ExecutionRunner.run (distinct experiments, full pipeline, 3 folds each)", _timed_iterations(run, 20))
        assert median < 1.0, "a full 3-fold walk-forward execution should not take >1s (~10x the measured floor)"


class TestResumeThroughput:
    def test_resume_idempotent_no_op(self, tmp_path: Path) -> None:
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
        manifest = preparer.prepare(spec)
        runner.run(manifest.identity.experiment_id)

        median = _report(
            "ExecutionRunner.resume (idempotent no-op, COMPLETED)",
            _timed_iterations(lambda: runner.resume(manifest.identity.experiment_id), 200),
        )
        assert median < 0.1, "an idempotent resume of a completed execution should not take >100ms"


class TestArtifactWriteThroughput:
    def test_write_fold_result_artifact(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        counter = {"i": 0}

        def write() -> None:
            counter["i"] += 1
            payload = {
                "schema_version": 1, "fold_index": counter["i"], "train_start": format_utc_timestamp(utc_now()),
                "train_end": format_utc_timestamp(utc_now()), "test_start": format_utc_timestamp(utc_now()),
                "test_end": format_utc_timestamp(utc_now()), "train_size": 100, "validation_size": 0,
                "test_size": 20, "status": "completed", "duration_seconds": 0.1, "validation_start": None,
                "validation_end": None, "artifact_references": [], "metrics": {}, "failure_reason": None,
            }
            store.write_artifact(canonical_json_bytes(payload), category=ArtifactCategory.FOLD_RESULT)

        median = _report("MLArtifactStore.write_artifact (FoldResult JSON, unique content)", _timed_iterations(write, 500))
        assert median < 0.05, "a single fold-result artifact write should not take >50ms"


class TestAggregateCreationThroughput:
    def test_aggregate_construction_and_serialization(self) -> None:
        now = format_utc_timestamp(utc_now())

        def build() -> None:
            aggregate = AggregatedExecutionResult(
                schema_version=1, experiment_id="a" * 64, total_folds=5,
                completed_fold_indices=(0, 1, 2, 3, 4), failed_fold_indices=(),
                overall_status=ExecutionStage.COMPLETED, started_at=now, completed_at=now,
                execution_duration_seconds=12.3,
            )
            canonical_json_bytes(aggregate.to_json_dict())

        median = _report("AggregatedExecutionResult construction + canonical serialization", _timed_iterations(build, 2000))
        assert median < 0.005, "in-memory aggregate construction+serialization should not take >5ms (100x the measured floor)"
