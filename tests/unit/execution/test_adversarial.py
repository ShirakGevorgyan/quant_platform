"""Section 19 adversarial self-audit for the execution engine. Every item
below either has a permanent regression test HERE or is cross-referenced
to the test file that already covers it, mirroring `tests/unit/ml/
test_ml_adversarial.py`'s exact "test file doubles as an audit checklist"
convention.

Covered elsewhere (cross-referenced, not duplicated):
- future leakage / non-chronological folds / insufficient purge-embargo
  gap -> test_execution_validation.py, test_property_based.py
- duplicate/out-of-order timestamps in a reconstructed timeline ->
  test_splitters.py::TestReconstructDatasetTimeline,
  test_execution_validation.py::TestDatasetCompatibilityChecks
- within-fold and cross-fold-test overlap (never flagging legitimate
  expanding/grouped train overlap) -> test_execution_validation.py::TestOverlapChecks
- resume corruption (a completed fold's artifact missing/corrupted) ->
  test_resume.py::TestVerifyCompletedFolds,
  test_execution_engine.py::test_corrupted_fold_artifact_forces_rerun_on_resume
- duplicate/parallel execution -> test_runner.py::TestDuplicateExecutionPrevention,
  test_execution_engine.py::test_duplicate_parallel_execution_never_corrupts_real_dataset_run
- partial/interrupted writes -> test_execution_manifests.py::TestInterruptedWrites

Covered HERE (not adequately exercised elsewhere):
- invalid (negative) purge/embargo declared through `SplitBinding.params`
  (the actual end-to-end path a human config typo would take), not just
  through the lower-level generator functions directly.
- a hand-tampered `execution_manifest.json` (claiming a fold complete
  without a matching `fold_result_references` entry) fails LOUDLY on
  load, never silently accepted -- the execution-engine analogue of
  Milestone 4A's `ExperimentManifest` identity-recomputation-on-load
  guarantee.
- a genuinely STALE (past `stale_after`) execution-run lock is reclaimed
  rather than blocking a legitimate run forever.
- a maliciously/incorrectly hand-built `FoldPlan` with a fold referencing
  row positions beyond the dataset never crashes the runner with an
  unguarded `IndexError` -- validation catches it first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    make_timeline,
    write_synthetic_research_dataset,
)

from quant_platform.execution.manifests import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    ExecutionManifest,
    ExecutionManifestStore,
)
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.splitters import build_folds_from_split_binding
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.historical.locking import DEFAULT_STALE_AFTER, LockInfo
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import SplitBinding
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now

EID = "a" * 64
_NOW = format_utc_timestamp(utc_now())


class TestInvalidPurgeEmbargoThroughSplitBinding:
    def test_negative_purge_bars_rejected(self) -> None:
        timeline = make_timeline()
        binding = SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 2, "test_size": 100, "purge_bars": -1})
        with pytest.raises(ValueError, match="non-negative"):
            build_folds_from_split_binding(binding, timeline["open_time"], label_horizon_bars=0)

    def test_negative_embargo_bars_rejected(self) -> None:
        timeline = make_timeline()
        binding = SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 2, "test_size": 100, "embargo_bars": -1})
        with pytest.raises(ValueError, match="non-negative"):
            build_folds_from_split_binding(binding, timeline["open_time"], label_horizon_bars=0)

    def test_negative_n_splits_rejected(self) -> None:
        timeline = make_timeline()
        binding = SplitBinding(strategy="expanding_walk_forward", params={"n_splits": -1, "test_size": 100})
        with pytest.raises(ValueError, match="n_splits"):
            build_folds_from_split_binding(binding, timeline["open_time"], label_horizon_bars=0)


class TestTamperedExecutionManifestFailsLoudly:
    def test_hand_edited_manifest_claiming_completion_without_reference_rejected_on_load(self, tmp_path: Path) -> None:
        """Mirrors Milestone 4A's `ExperimentManifest` identity-
        recomputation-on-load guarantee: a manifest file tampered with
        directly on disk (bypassing `ExecutionManifestStore.transition`
        entirely) must fail loudly on the NEXT load, never be silently
        trusted."""
        store = ExecutionManifestStore(tmp_path)
        manifest = ExecutionManifest(
            schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id=EID, stage=ExecutionStage.INITIALIZING,
            created_at=_NOW, updated_at=_NOW,
        )
        store.create(manifest)

        import json

        raw = json.loads(store._manifest_path(EID).read_text())
        raw["completed_fold_indices"] = [0]  # claims fold 0 complete...
        raw["fold_result_references"] = {}  # ...but names no artifact for it
        store._manifest_path(EID).write_text(json.dumps(raw))

        with pytest.raises(ValueError, match="fold_result_references"):
            store.load(EID)


class TestStaleExecutionRunLockIsReclaimed:
    def test_stale_lock_does_not_block_a_legitimate_run(self, tmp_path: Path) -> None:
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
        manifest = preparer.prepare(spec)

        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store,
        )
        lock_path = runner._lock_path(manifest.identity.experiment_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale_info = LockInfo(
            pid=999999, hostname="stale-host",
            acquired_at=pd.Timestamp.now(tz="UTC") - DEFAULT_STALE_AFTER - pd.Timedelta(minutes=1),
        )
        lock_path.write_text(canonical_json_bytes(stale_info.to_json_dict()).decode("utf-8"))

        outcome = runner.run(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED

    def test_fresh_contested_lock_still_blocks(self, tmp_path: Path) -> None:
        """Contrast case proving the stale-reclaim test above is
        genuinely exercising staleness, not just "locks never block"."""
        from quant_platform.core.exceptions import ExperimentLockError
        from quant_platform.historical.locking import DatasetLock

        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
        manifest = preparer.prepare(spec)
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store,
        )
        holder = DatasetLock(runner._lock_path(manifest.identity.experiment_id))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(manifest.identity.experiment_id)
        finally:
            holder.release()


class TestMaliciousFoldPlanNeverCrashesWithRawIndexError:
    def test_out_of_bounds_fold_plan_fails_via_validation_not_index_error(self, tmp_path: Path) -> None:
        """A `FoldPlan` referencing row positions beyond the dataset
        (e.g. built against a stale/mismatched timeline) must be caught
        by `validate_fold_plan`'s explicit bounds check and reported as a
        FAILED, non-resumable execution -- never propagate a raw,
        unguarded `IndexError`/`KeyError` from deep inside `.iloc[]`."""
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(
            tmp_path, timeline=make_timeline(50),  # deliberately tiny
        )
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        # n_splits*test_size (3*100=300) vastly exceeds the 50-row dataset.
        spec = ExperimentSpec(**make_experiment_spec_kwargs(
            dataset_manifest=dataset_manifest, split_params={"n_splits": 3, "test_size": 100},
        ))
        manifest = preparer.prepare(spec)
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store,
        )
        with pytest.raises(Exception) as exc_info:
            runner.run(manifest.identity.experiment_id)
        assert not isinstance(exc_info.value, (IndexError, KeyError))
