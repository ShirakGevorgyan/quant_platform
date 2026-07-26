from __future__ import annotations

import threading
from pathlib import Path

import pytest

from quant_platform.core.exceptions import ArtifactNotFoundError, ExecutionStateError, ExperimentLockError
from quant_platform.execution.manifests import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    ExecutionManifest,
    ExecutionManifestStore,
)
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

EID = "a" * 64
_NOW = format_utc_timestamp(utc_now())


def _manifest(**overrides: object) -> ExecutionManifest:
    base: dict[str, object] = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION, "experiment_id": EID,
        "stage": ExecutionStage.INITIALIZING, "created_at": _NOW, "updated_at": _NOW,
    }
    base.update(overrides)
    return ExecutionManifest(**base)  # type: ignore[arg-type]


def _ref(hash_char: str = "b") -> ArtifactReference:
    return ArtifactReference(category=ArtifactCategory.FOLD_RESULT, content_hash=hash_char * 64, size_bytes=10, created_at=_NOW)


class TestExecutionManifestConstruction:
    def test_valid_manifest_builds(self) -> None:
        _manifest()

    def test_invalid_experiment_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="experiment_id"):
            _manifest(experiment_id="not-a-hash")

    def test_duplicate_completed_indices_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            _manifest(completed_fold_indices=(0, 0), fold_result_references={0: _ref()})

    def test_overlap_between_completed_and_failed_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be both"):
            _manifest(completed_fold_indices=(0,), failed_fold_indices=(0,), fold_result_references={0: _ref()})

    def test_negative_resume_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="resume_count"):
            _manifest(resume_count=-1)

    def test_failure_summary_required_when_failed(self) -> None:
        with pytest.raises(ValueError, match="failure_summary is required"):
            _manifest(stage=ExecutionStage.FAILED, completed_at=_NOW)

    def test_failure_summary_forbidden_outside_failed(self) -> None:
        with pytest.raises(ValueError, match="must be None unless"):
            _manifest(failure_summary="x")

    def test_completed_fold_without_reference_rejected(self) -> None:
        """A fold cannot be claimed COMPLETE without a corresponding
        `fold_result_references` entry -- this is what makes
        `execution.resume.verify_completed_folds` possible at all."""
        with pytest.raises(ValueError, match="fold_result_references"):
            _manifest(completed_fold_indices=(0,))

    def test_round_trip(self) -> None:
        manifest = _manifest(
            stage=ExecutionStage.COMPLETED, completed_at=_NOW, completed_fold_indices=(0, 1),
            fold_result_references={0: _ref("b"), 1: _ref("c")}, artifact_references=(_ref("d"),),
        )
        assert ExecutionManifest.from_json_dict(manifest.to_json_dict()) == manifest


class TestExecutionManifestStoreCreate:
    def test_create_and_load(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        manifest = _manifest()
        store.create(manifest)
        assert store.load(EID) == manifest

    def test_create_requires_initializing_stage(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        with pytest.raises(ExecutionStateError, match="INITIALIZING"):
            store.create(_manifest(stage=ExecutionStage.LOADING_DATASET))

    def test_duplicate_create_rejected(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        with pytest.raises(ExecutionStateError, match="already exists"):
            store.create(_manifest())

    def test_load_missing_raises_not_found(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.load(EID)

    def test_load_if_exists_returns_none_when_absent(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        assert store.load_if_exists(EID) is None
        assert not store.exists(EID)


class TestExecutionManifestStoreTransitions:
    def test_full_legal_sequence(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=_NOW, fold_plan_strategy="expanding_walk_forward", total_folds=3)
        m = store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW)
        assert m.fold_plan_strategy == "expanding_walk_forward"
        assert m.total_folds == 3
        m = store.transition(
            EID, new_stage=ExecutionStage.STORING_RESULTS, updated_at=_NOW,
            completed_fold_indices=(0,), fold_result_references={0: _ref("b")},
        )
        m = store.transition(EID, new_stage=ExecutionStage.COMPLETED, updated_at=_NOW, completed_at=_NOW)
        assert m.stage is ExecutionStage.COMPLETED

    def test_illegal_transition_rejected(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        with pytest.raises(ExecutionStateError):
            store.transition(EID, new_stage=ExecutionStage.COMPLETED, updated_at=_NOW)

    def test_no_self_transition(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        with pytest.raises(ExecutionStateError):
            store.transition(EID, new_stage=ExecutionStage.INITIALIZING, updated_at=_NOW)

    def test_terminal_state_cannot_be_edited_again(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(
            EID, new_stage=ExecutionStage.FAILED, updated_at=_NOW, completed_at=_NOW, failure_summary="nope",
        )
        for target in ExecutionStage:
            with pytest.raises(ExecutionStateError):
                store.transition(EID, new_stage=target, updated_at=_NOW)

    def test_omitting_optional_fields_carries_them_forward_unchanged(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(
            EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW,
        )
        store.transition(EID, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=_NOW, total_folds=7, fold_plan_strategy="x")
        m2 = store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW)
        assert m2.total_folds == 7
        assert m2.fold_plan_strategy == "x"

    def test_current_fold_index_sentinel_distinguishes_unchanged_from_explicit_none(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=_NOW)
        m = store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW, current_fold_index=3)
        assert m.current_fold_index == 3
        # Omitted entirely -- unchanged.
        m2 = store.transition(EID, new_stage=ExecutionStage.STORING_RESULTS, updated_at=_NOW)
        assert m2.current_fold_index == 3
        # Explicitly cleared to None.
        m3 = store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW, current_fold_index=None)
        assert m3.current_fold_index is None

    def test_fold_result_references_carry_forward_when_omitted(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW)
        store.transition(
            EID, new_stage=ExecutionStage.STORING_RESULTS, updated_at=_NOW,
            completed_fold_indices=(0,), fold_result_references={0: _ref("b")},
        )
        store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW)
        m = store.transition(EID, new_stage=ExecutionStage.STORING_RESULTS, updated_at=_NOW)
        assert m.fold_result_references == {0: _ref("b")}


class TestBumpResumeCount:
    def test_increments_without_changing_stage(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        m = store.bump_resume_count(EID)
        assert m.resume_count == 1
        assert m.stage is ExecutionStage.LOADING_DATASET
        m2 = store.bump_resume_count(EID)
        assert m2.resume_count == 2

    def test_works_from_recoverable_failure(self, tmp_path: Path) -> None:
        """Regression test for a real bug found via smoke testing: resume
        used to call `transition(new_stage=current.stage, ...)` to bump
        the counter, which is illegal for EVERY stage (no self-loops
        exist in the transition table) -- `bump_resume_count` bypasses
        the transition-legality check entirely instead."""
        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=_NOW)
        store.transition(EID, new_stage=ExecutionStage.RECOVERABLE_FAILURE, updated_at=_NOW)
        m = store.bump_resume_count(EID)
        assert m.stage is ExecutionStage.RECOVERABLE_FAILURE
        assert m.resume_count == 1


class TestInterruptedWrites:
    def test_interrupted_create_leaves_no_file(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        store = ExecutionManifestStore(tmp_path)
        with patch.object(Path, "replace", side_effect=OSError("simulated crash")), pytest.raises(OSError, match="simulated crash"):
            store.create(_manifest())
        assert not store.exists(EID)

    def test_interrupted_transition_leaves_previous_manifest_intact(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        with patch.object(Path, "replace", side_effect=OSError("simulated crash")), pytest.raises(OSError, match="simulated crash"):
            store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
        reloaded = store.load(EID)
        assert reloaded.stage is ExecutionStage.INITIALIZING


class TestExecutionLockTranslation:
    def test_transition_translates_contested_lock_deterministically(self, tmp_path: Path) -> None:
        from quant_platform.core.exceptions import DatasetLockError
        from quant_platform.historical.locking import DatasetLock

        store = ExecutionManifestStore(tmp_path)
        store.create(_manifest())
        holder = DatasetLock(store._lock_path(EID))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError) as exc_info:
                store.transition(EID, new_stage=ExecutionStage.LOADING_DATASET, updated_at=_NOW)
            assert isinstance(exc_info.value.__cause__, DatasetLockError)
        finally:
            holder.release()
        assert store.load(EID).stage is ExecutionStage.INITIALIZING


class TestConcurrency:
    def test_concurrent_create_is_safe(self, tmp_path: Path) -> None:
        store = ExecutionManifestStore(tmp_path)
        manifest = _manifest()
        results: list[BaseException | None] = []
        lock = threading.Lock()

        def create() -> None:
            try:
                store.create(manifest)
                with lock:
                    results.append(None)
            except (ExecutionStateError, ExperimentLockError) as exc:
                with lock:
                    results.append(exc)

        threads = [threading.Thread(target=create) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4
        assert len([r for r in results if r is None]) == 1
        assert store.load(EID) == manifest
