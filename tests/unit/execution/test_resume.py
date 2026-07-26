from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import ExecutionResumeError
from quant_platform.execution.manifests import EXECUTION_MANIFEST_SCHEMA_VERSION, ExecutionManifest
from quant_platform.execution.results import FoldResult, FoldStatus
from quant_platform.execution.resume import build_resume_plan, can_resume, verify_completed_folds
from quant_platform.execution.splitters import Fold, FoldPlan
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now

EID = "a" * 64
_NOW = format_utc_timestamp(utc_now())
_TS = pd.Timestamp("2024-01-01", tz="UTC")


def _fold(index: int) -> Fold:
    return Fold(
        fold_index=index, train_indices=np.arange(0, 10), test_indices=np.arange(20, 30),
        train_start=_TS, train_end=_TS, test_start=_TS, test_end=_TS,
    )


def _plan(n: int) -> FoldPlan:
    return FoldPlan(
        strategy="x", purge_bars=0, embargo_bars=0, total_rows=100, folds=tuple(_fold(i) for i in range(n)),
        label_horizon_bars=0, required_label_purge_bars=0,
    )


def _manifest(**overrides: object) -> ExecutionManifest:
    base: dict[str, object] = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION, "experiment_id": EID,
        "stage": ExecutionStage.RUNNING_FOLD, "created_at": _NOW, "updated_at": _NOW,
    }
    base.update(overrides)
    return ExecutionManifest(**base)  # type: ignore[arg-type]


def _write_fold_result(store: MLArtifactStore, *, fold_index: int, category: ArtifactCategory = ArtifactCategory.FOLD_RESULT) -> ArtifactReference:
    """Writes a REAL, decodable `FoldResult` -- not an arbitrary JSON blob
    -- so `verify_completed_folds`'s decode-and-cross-check step has
    genuine `FoldResult` content to work with. `category` defaults to the
    correct one; pass `TIMELINE`/etc. to prove a category mismatch alone
    is enough to demote a fold to `needs_rerun`."""
    result = FoldResult(
        schema_version=1, fold_index=fold_index, train_start=_NOW, train_end=_NOW, test_start=_NOW, test_end=_NOW,
        train_size=10, test_size=10, status=FoldStatus.COMPLETED, duration_seconds=0.1,
    )
    return store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=category)


class TestCanResume:
    def test_none_manifest_cannot_resume(self) -> None:
        assert not can_resume(None)

    @pytest.mark.parametrize("stage", [ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED])
    def test_terminal_stages_cannot_resume(self, stage: ExecutionStage) -> None:
        kwargs: dict[str, object] = {"stage": stage}
        if stage is not ExecutionStage.CANCELLED:
            kwargs["completed_at"] = _NOW
        if stage is ExecutionStage.FAILED:
            kwargs["failure_summary"] = "x"
        assert not can_resume(_manifest(**kwargs))

    @pytest.mark.parametrize(
        "stage",
        [
            ExecutionStage.INITIALIZING, ExecutionStage.LOADING_DATASET, ExecutionStage.BUILDING_SPLITS,
            ExecutionStage.RUNNING_FOLD, ExecutionStage.STORING_RESULTS, ExecutionStage.RECOVERABLE_FAILURE,
        ],
    )
    def test_non_terminal_stages_can_resume(self, stage: ExecutionStage) -> None:
        assert can_resume(_manifest(stage=stage))


class TestVerifyCompletedFolds:
    def test_verified_when_artifact_intact(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset({0})
        assert needs_rerun == frozenset()

    def test_needs_rerun_when_reference_missing(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        manifest = _manifest()  # no completed folds, no references at all -- nothing to verify
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset()

    def test_needs_rerun_when_artifact_content_missing_from_store(self, tmp_path: Path) -> None:
        """The manifest CLAIMS fold 0 is complete and names a content
        hash, but that content was never actually written (or was
        deleted) -- must be treated as needing rerun, never trusted."""
        store = MLArtifactStore(tmp_path)
        fake_ref = ArtifactReference(category=ArtifactCategory.FOLD_RESULT, content_hash="b" * 64, size_bytes=1, created_at=_NOW)
        manifest = ExecutionManifest(
            schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id=EID, stage=ExecutionStage.RUNNING_FOLD,
            created_at=_NOW, updated_at=_NOW, completed_fold_indices=(0,), fold_result_references={0: fake_ref},
        )
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({0})

    def test_needs_rerun_when_artifact_corrupted(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0)
        # Corrupt the content in place.
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(b"TAMPERED")
        manifest = ExecutionManifest(
            schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id=EID, stage=ExecutionStage.RUNNING_FOLD,
            created_at=_NOW, updated_at=_NOW, completed_fold_indices=(0,), fold_result_references={0: ref},
        )
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({0})

    def test_needs_rerun_when_artifact_content_is_not_valid_json(self, tmp_path: Path) -> None:
        """A hash CAN verify against non-JSON bytes (the store only
        checksums bytes, it does not parse them) -- proves the decode
        step fails closed rather than letting a raw JSON decode
        exception escape (required test: 'no raw... generic JSON
        exception escapes from these validation paths')."""
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"not json at all {{{", category=ArtifactCategory.FOLD_RESULT)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({0})

    def test_needs_rerun_when_artifact_category_is_not_fold_result(self, tmp_path: Path) -> None:
        """A valid, decodable `FoldResult` filed under the WRONG artifact
        category must still be rejected -- category is checked
        independently of content hash and decodability."""
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0, category=ArtifactCategory.TIMELINE)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({0})

    def test_needs_rerun_when_decoded_fold_index_does_not_match_manifest_key(self, tmp_path: Path) -> None:
        """The PRIMARY new guarantee: a `FOLD_RESULT` artifact with a
        perfectly valid hash, correct category, and clean decode is still
        rejected if its OWN `fold_index` field does not match the
        `fold_result_references` dict key it was filed under -- e.g. fold
        3's real result mistakenly (or maliciously) filed under key 5."""
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=3)  # content says fold_index=3...
        manifest = _manifest(completed_fold_indices=(5,), fold_result_references={5: ref})  # ...filed under key 5
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=store)
        assert verified == frozenset()
        assert needs_rerun == frozenset({5})


class TestBuildResumePlan:
    def test_terminal_manifest_raises(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        manifest = _manifest(stage=ExecutionStage.FAILED, completed_at=_NOW, failure_summary="x")
        with pytest.raises(ExecutionResumeError, match="terminal"):
            build_resume_plan(manifest, _plan(3), artifact_store=store)

    def test_all_verified_complete_yields_empty_remaining(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        plan = build_resume_plan(manifest, _plan(1), artifact_store=store)
        assert plan.remaining_folds == ()
        assert plan.verified_complete == frozenset({0})

    def test_unverified_and_new_folds_are_remaining(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        plan = build_resume_plan(manifest, _plan(3), artifact_store=store)
        assert [f.fold_index for f in plan.remaining_folds] == [1, 2]

    def test_force_rerun_folds_reruns_a_verified_complete_fold(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = _write_fold_result(store, fold_index=0)
        manifest = _manifest(completed_fold_indices=(0,), fold_result_references={0: ref})
        plan = build_resume_plan(manifest, _plan(1), artifact_store=store, force_rerun_folds=frozenset({0}))
        assert [f.fold_index for f in plan.remaining_folds] == [0]
        assert plan.verified_complete == frozenset()

    def test_force_rerun_never_affects_folds_not_named(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref0 = _write_fold_result(store, fold_index=0)
        ref1 = _write_fold_result(store, fold_index=1)
        manifest = _manifest(completed_fold_indices=(0, 1), fold_result_references={0: ref0, 1: ref1})
        plan = build_resume_plan(manifest, _plan(2), artifact_store=store, force_rerun_folds=frozenset({0}))
        assert [f.fold_index for f in plan.remaining_folds] == [0]
        assert plan.verified_complete == frozenset({1})

    def test_corrupted_completed_fold_is_included_in_remaining_folds_to_rerun(self, tmp_path: Path) -> None:
        """End-to-end (at the resume-plan level) proof that a corrupted
        completed fold is neither silently trusted NOR simply dropped --
        it reappears in `remaining_folds` so the runner re-executes it
        and writes a fresh replacement reference."""
        store = MLArtifactStore(tmp_path)
        good_ref = _write_fold_result(store, fold_index=0)
        bad_ref = _write_fold_result(store, fold_index=1)
        store._content_path(bad_ref.content_hash).write_bytes(b"TAMPERED")
        manifest = _manifest(completed_fold_indices=(0, 1), fold_result_references={0: good_ref, 1: bad_ref})
        plan = build_resume_plan(manifest, _plan(2), artifact_store=store)
        assert plan.verified_complete == frozenset({0})
        assert plan.needs_rerun == frozenset({1})
        assert [f.fold_index for f in plan.remaining_folds] == [1]
