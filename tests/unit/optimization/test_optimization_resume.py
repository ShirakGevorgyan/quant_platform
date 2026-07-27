"""Milestone 4D: resume planning -- verified-artifact trial-resume
planning (corrupted/missing/wrong-key artifacts truncate the resumable
prefix, never silently trusted) and outer-fold completion verification."""

from __future__ import annotations

import pytest

from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory
from quant_platform.ml.persistence import canonical_json_bytes
from quant_platform.optimization.candidates import TrialResult, TrialStatus
from quant_platform.optimization.manifests import (
    OPTIMIZATION_MANIFEST_SCHEMA_VERSION,
    OptimizationManifest,
)
from quant_platform.optimization.models import OptimizationStage
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.resume import (
    build_trial_resume_plan,
    can_resume,
    require_resumable,
    verify_completed_outer_folds,
)

OPT_ID = "a" * 64
PARENT_ID = "b" * 64


def _trial_result(trial_number: int, *, outer_fold_index: int = 0, optimization_id: str = OPT_ID) -> TrialResult:
    return TrialResult(
        schema_version=1, optimization_id=optimization_id, outer_fold_index=outer_fold_index, trial_number=trial_number,
        status=TrialStatus.COMPLETED, sampled_hyperparameters={"lr": 0.1}, inner_fold_metrics=(),
        primary_metric_aggregate=0.5, successful_inner_folds=2, total_inner_folds=2, duration_seconds=1.0,
    )


def _write_trial(store: MLArtifactStore, trial_number: int, **overrides) -> object:
    result = _trial_result(trial_number, **overrides)
    return store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)


class TestBuildTrialResumePlanHappyPath:
    def test_all_verified_trials_form_the_prefix(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        refs = {i: _write_trial(store, i) for i in range(3)}
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == (0, 1, 2)
        assert plan.discarded_trial_numbers == frozenset()
        assert plan.next_trial_number == 3

    def test_empty_claimed_references_gives_a_fresh_start(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        plan = build_trial_resume_plan({}, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == ()
        assert plan.next_trial_number == 0


class TestBuildTrialResumePlanTruncatesAtFirstFailure:
    def test_missing_trial_in_the_middle_truncates_everything_after_it(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        refs = {0: _write_trial(store, 0), 2: _write_trial(store, 2)}  # trial 1 missing entirely
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == (0,)
        assert 2 in plan.discarded_trial_numbers
        assert plan.next_trial_number == 1

    def test_corrupted_artifact_bytes_truncates_from_that_point(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        refs = {0: _write_trial(store, 0), 1: _write_trial(store, 1), 2: _write_trial(store, 2)}
        # Corrupt trial 1's content on disk directly (bit flip).
        content_path = store._content_path(refs[1].content_hash)
        content_path.write_bytes(b"not the original bytes at all")
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == (0,)
        assert 1 in plan.discarded_trial_numbers
        assert 2 in plan.discarded_trial_numbers  # even though trial 2's OWN bytes are perfectly intact

    def test_wrong_category_reference_is_rejected(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        good = _write_trial(store, 0)
        wrong_category_ref = store.write_artifact(b"some other content", category=ArtifactCategory.METRICS)
        refs = {0: good, 1: wrong_category_ref}
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == (0,)
        assert 1 in plan.discarded_trial_numbers

    def test_wrong_trial_number_filed_under_a_different_key_is_rejected(self, tmp_path) -> None:
        """A content hash that verifies cleanly but decodes to a
        TrialResult whose OWN trial_number does not match the manifest
        key it was filed under -- a valid hash proves the bytes are
        intact, never that they were filed correctly."""
        store = MLArtifactStore(tmp_path)
        actually_trial_5 = store.write_artifact(canonical_json_bytes(_trial_result(5).to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)
        refs = {0: _write_trial(store, 0), 1: actually_trial_5}  # filed under key 1, but decodes as trial_number=5
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == (0,)
        assert 1 in plan.discarded_trial_numbers

    def test_wrong_outer_fold_index_is_rejected(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        wrong_fold = store.write_artifact(canonical_json_bytes(_trial_result(0, outer_fold_index=1).to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)
        refs = {0: wrong_fold}
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == ()
        assert 0 in plan.discarded_trial_numbers

    def test_wrong_optimization_id_is_rejected(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        wrong_opt = store.write_artifact(canonical_json_bytes(_trial_result(0, optimization_id="c" * 64).to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)
        refs = {0: wrong_opt}
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == ()

    def test_gap_at_trial_zero_discards_everything(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        refs = {1: _write_trial(store, 1)}  # trial 0 missing -- nothing can ever be trusted
        plan = build_trial_resume_plan(refs, optimization_id=OPT_ID, outer_fold_index=0, artifact_store=store)
        assert plan.verified_prefix_trial_numbers == ()
        assert plan.next_trial_number == 0
        assert 1 in plan.discarded_trial_numbers


class TestCanResumeAndRequireResumable:
    def _manifest(self, stage: OptimizationStage, **overrides) -> OptimizationManifest:
        base: dict[str, object] = {
            "schema_version": OPTIMIZATION_MANIFEST_SCHEMA_VERSION, "optimization_id": OPT_ID, "parent_experiment_id": PARENT_ID,
            "stage": stage, "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00",
        }
        base.update(overrides)
        return OptimizationManifest(**base)  # type: ignore[arg-type]

    def test_none_manifest_cannot_resume(self) -> None:
        assert not can_resume(None)

    def test_terminal_stage_cannot_resume(self) -> None:
        assert not can_resume(self._manifest(OptimizationStage.COMPLETED, completed_at="2024-01-01T00:00:00+00:00"))

    def test_non_terminal_stage_can_resume(self) -> None:
        assert can_resume(self._manifest(OptimizationStage.RUNNING_OUTER_FOLD))

    def test_require_resumable_raises_for_none(self) -> None:
        from quant_platform.core.exceptions import OptimizationResumeError

        with pytest.raises(OptimizationResumeError, match="nothing to resume"):
            require_resumable(None, optimization_id=OPT_ID)

    def test_require_resumable_raises_for_terminal(self) -> None:
        from quant_platform.core.exceptions import OptimizationResumeError

        with pytest.raises(OptimizationResumeError, match="terminal"):
            require_resumable(self._manifest(OptimizationStage.FAILED, failure_summary="x"), optimization_id=OPT_ID)


class TestVerifyCompletedOuterFolds:
    def _outer_result(self, outer_fold_index: int, optimization_id: str = OPT_ID) -> OuterFoldResult:
        return OuterFoldResult(
            schema_version=1, optimization_id=optimization_id, outer_fold_index=outer_fold_index, winning_trial_number=0,
            final_selected_features=("a",), final_hyperparameters={}, final_round_source=None, seed=1,
            training_duration_seconds=1.0, outer_train_row_count=10, outer_test_row_count=5, outer_test_metrics={"accuracy": 0.5},
            feature_selection_result_reference=_ref_stub(), model_reference=_ref_stub(), predictions_reference=_ref_stub(),
            evaluated_at="2024-01-01T00:00:00+00:00",
        )

    def test_verified_outer_fold_is_trusted(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        result = self._outer_result(0)
        ref = store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.OUTER_FOLD_SELECTION)
        manifest = OptimizationManifest(
            schema_version=OPTIMIZATION_MANIFEST_SCHEMA_VERSION, optimization_id=OPT_ID, parent_experiment_id=PARENT_ID,
            stage=OptimizationStage.STORING_RESULTS, created_at="2024-01-01T00:00:00+00:00", updated_at="2024-01-01T00:00:00+00:00",
            completed_outer_fold_indices=(0,), outer_fold_result_references={0: ref},
        )
        verified, needs_rerun = verify_completed_outer_folds(manifest, artifact_store=store)
        assert verified == frozenset({0})
        assert needs_rerun == frozenset()

    def test_corrupted_outer_fold_result_needs_rerun(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        result = self._outer_result(0)
        ref = store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.OUTER_FOLD_SELECTION)
        store._content_path(ref.content_hash).write_bytes(b"corrupted")
        manifest = OptimizationManifest(
            schema_version=OPTIMIZATION_MANIFEST_SCHEMA_VERSION, optimization_id=OPT_ID, parent_experiment_id=PARENT_ID,
            stage=OptimizationStage.STORING_RESULTS, created_at="2024-01-01T00:00:00+00:00", updated_at="2024-01-01T00:00:00+00:00",
            completed_outer_fold_indices=(0,), outer_fold_result_references={0: ref},
        )
        verified, needs_rerun = verify_completed_outer_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({0})


def _ref_stub():
    from quant_platform.ml.models import ArtifactReference

    return ArtifactReference(category=ArtifactCategory.MODEL, content_hash="c" * 64, size_bytes=1, created_at="2024-01-01T00:00:00+00:00")
