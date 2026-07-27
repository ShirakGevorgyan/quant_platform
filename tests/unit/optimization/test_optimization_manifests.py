"""Milestone 4D: `OptimizationManifest`/`OptimizationManifestStore` and
`OptimizationEventStore` -- construction validation, legal-transition
enforcement, atomic writes, append-only event log crash-tolerance, and
concurrent-duplicate-creation prevention (reusing the already-corrected,
already-tested `DatasetLock`/`experiment_lock` -- this file proves THIS
store uses that lock correctly, not that the lock itself is atomic; see
`tests/unit/historical/test_locking.py` for that)."""

from __future__ import annotations

import os
import threading

import pytest

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ExperimentLockError,
    OptimizationStateError,
)
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.optimization.manifests import (
    OPTIMIZATION_MANIFEST_SCHEMA_VERSION,
    OptimizationEventStore,
    OptimizationEventType,
    OptimizationManifest,
    OptimizationManifestStore,
    trial_reference_key,
    trial_references_for_outer_fold,
)
from quant_platform.optimization.models import OptimizationStage

OPT_ID = "a" * 64
PARENT_ID = "b" * 64


def _manifest(**overrides: object) -> OptimizationManifest:
    base: dict[str, object] = {
        "schema_version": OPTIMIZATION_MANIFEST_SCHEMA_VERSION, "optimization_id": OPT_ID, "parent_experiment_id": PARENT_ID,
        "stage": OptimizationStage.INITIALIZING, "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return OptimizationManifest(**base)  # type: ignore[arg-type]


def _ref(content_hash: str, category: ArtifactCategory = ArtifactCategory.TRIAL_RESULT) -> ArtifactReference:
    return ArtifactReference(category=category, content_hash=content_hash, size_bytes=10, created_at="2024-01-01T00:00:00+00:00")


class TestOptimizationManifestValidation:
    def test_invalid_optimization_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="optimization_id"):
            _manifest(optimization_id="not-a-hash")

    def test_duplicate_completed_outer_fold_indices_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            _manifest(completed_outer_fold_indices=(0, 0))

    def test_failed_stage_requires_failure_summary(self) -> None:
        with pytest.raises(ValueError, match="failure_summary is required"):
            _manifest(stage=OptimizationStage.FAILED)

    def test_non_failed_stage_must_not_carry_failure_summary(self) -> None:
        with pytest.raises(ValueError, match="must be None"):
            _manifest(stage=OptimizationStage.INITIALIZING, failure_summary="oops")

    def test_completed_outer_fold_without_result_reference_rejected(self) -> None:
        with pytest.raises(ValueError, match="outer_fold_result_references"):
            _manifest(completed_outer_fold_indices=(0,), outer_fold_result_references={})

    def test_round_trip(self) -> None:
        manifest = _manifest(
            stage=OptimizationStage.COMPLETED, completed_at="2024-01-01T00:00:00+00:00",
            completed_outer_fold_indices=(0,), outer_fold_result_references={0: _ref("c" * 64, ArtifactCategory.OUTER_FOLD_SELECTION)},
            trial_result_references={trial_reference_key(0, 0): _ref("d" * 64)},
            winning_trial_by_outer_fold={0: 3},
        )
        assert OptimizationManifest.from_json_dict(manifest.to_json_dict()) == manifest


class TestTrialReferenceKeyHelpers:
    def test_key_format(self) -> None:
        assert trial_reference_key(2, 7) == "2:7"

    def test_references_for_outer_fold_filters_correctly(self) -> None:
        manifest = _manifest(trial_result_references={
            trial_reference_key(0, 0): _ref("a" * 64), trial_reference_key(0, 1): _ref("b" * 64),
            trial_reference_key(1, 0): _ref("c" * 64),
        })
        refs = trial_references_for_outer_fold(manifest, 0)
        assert set(refs) == {0, 1}
        refs_1 = trial_references_for_outer_fold(manifest, 1)
        assert set(refs_1) == {0}


class TestOptimizationManifestStore:
    def test_create_requires_initializing_stage(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        with pytest.raises(OptimizationStateError, match="INITIALIZING"):
            store.create(_manifest(stage=OptimizationStage.RUNNING_OUTER_FOLD))

    def test_create_then_load_round_trips(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        loaded = store.load(OPT_ID)
        assert loaded.optimization_id == OPT_ID
        assert loaded.stage is OptimizationStage.INITIALIZING

    def test_create_refuses_to_overwrite_existing_manifest(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        with pytest.raises(OptimizationStateError, match="already exists"):
            store.create(_manifest())

    def test_load_if_exists_returns_none_when_absent(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        assert store.load_if_exists(OPT_ID) is None

    def test_transition_rejects_illegal_target(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        with pytest.raises(OptimizationStateError, match="Illegal"):
            store.transition(OPT_ID, new_stage=OptimizationStage.COMPLETED, updated_at="2024-01-01T00:00:00+00:00")

    def test_transition_persists_and_is_visible_on_reload(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(OPT_ID, new_stage=OptimizationStage.LOADING_EXPERIMENT, updated_at="2024-01-01T00:01:00+00:00")
        reloaded = store.load(OPT_ID)
        assert reloaded.stage is OptimizationStage.LOADING_EXPERIMENT

    def test_current_outer_fold_index_sentinel_distinguishes_unchanged_from_explicit_none(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        store.transition(
            OPT_ID, new_stage=OptimizationStage.LOADING_EXPERIMENT, updated_at="2024-01-01T00:01:00+00:00", current_outer_fold_index=2,
        )
        assert store.load(OPT_ID).current_outer_fold_index == 2
        store.transition(OPT_ID, new_stage=OptimizationStage.BUILDING_OUTER_PLAN, updated_at="2024-01-01T00:02:00+00:00")
        assert store.load(OPT_ID).current_outer_fold_index == 2  # omitted -> unchanged
        store.transition(
            OPT_ID, new_stage=OptimizationStage.RUNNING_OUTER_FOLD, updated_at="2024-01-01T00:03:00+00:00", current_outer_fold_index=None,
        )
        assert store.load(OPT_ID).current_outer_fold_index is None  # explicit None -> cleared

    def test_bump_resume_count_does_not_change_stage(self, tmp_path) -> None:
        store = OptimizationManifestStore(tmp_path)
        store.create(_manifest())
        updated = store.bump_resume_count(OPT_ID)
        assert updated.stage is OptimizationStage.INITIALIZING
        assert updated.resume_count == 1

    def test_concurrent_create_attempts_never_both_succeed(self, tmp_path, monkeypatch) -> None:
        """Two threads racing to `create()` the SAME optimization_id --
        exactly one must succeed, the other must fail, never both silently
        succeeding with divergent content. Mirrors `tests/unit/historical/
        test_locking.py::test_exactly_one_winner_when_two_threads_link_at_
        the_same_instant`'s own technique EXACTLY: barrier the two threads
        at the precise `os.link` call `DatasetLock.acquire()` uses to
        publish its lock file (the actual atomic decision point), not at
        the coarser `store.create()` call -- barriering there left a
        timing gap where one thread could win the lock ENTIRELY (acquire,
        write, release) before the other even attempted acquisition,
        making the loser's exception type (`OptimizationStateError` from
        the post-lock existence check, vs `ExperimentLockError` from
        losing lock acquisition itself) flaky rather than deterministic --
        both are legitimate 'lost the race' outcomes of the SAME safe,
        mutually-exclusive critical section, but a non-deterministic test
        is still a bug in the test."""
        store = OptimizationManifestStore(tmp_path)
        real_link = os.link
        barrier = threading.Barrier(2, timeout=5)

        def synchronized_link(src: str, dst: str) -> None:
            barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[str] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            try:
                store.create(_manifest())
                with results_lock:
                    results.append("created")
            except (OptimizationStateError, ExperimentLockError):
                with results_lock:
                    results.append("rejected")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not any(t.is_alive() for t in threads)
        assert sorted(results) == ["created", "rejected"]
        assert store.exists(OPT_ID)


class TestOptimizationEventStore:
    def test_append_assigns_gapless_sequence_numbers(self, tmp_path) -> None:
        store = OptimizationEventStore(tmp_path)
        store.append(OPT_ID, OptimizationEventType.OPTIMIZATION_CREATED)
        store.append(OPT_ID, OptimizationEventType.RUN_STARTED)
        events = store.read_events(OPT_ID)
        assert [e.sequence for e in events] == [1, 2]

    def test_details_round_trip(self, tmp_path) -> None:
        store = OptimizationEventStore(tmp_path)
        store.append(OPT_ID, OptimizationEventType.TRIAL_STARTED, details={"outer_fold_index": 0, "trial_number": 3})
        events = store.read_events(OPT_ID)
        assert events[0].details == {"outer_fold_index": 0, "trial_number": 3}

    def test_truncated_final_line_is_repaired_not_fatal(self, tmp_path) -> None:
        store = OptimizationEventStore(tmp_path)
        store.append(OPT_ID, OptimizationEventType.OPTIMIZATION_CREATED)
        store.append(OPT_ID, OptimizationEventType.RUN_STARTED)
        events_path = tmp_path / "optimizations" / OPT_ID / "events.jsonl"
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        events_path.write_text(lines[0] + "\n" + lines[1][:10], encoding="utf-8")  # truncate the final line

        events = store.read_events(OPT_ID)
        assert len(events) == 1  # the truncated final line was repaired away, not fatal
        # A subsequent append continues cleanly from the repaired state.
        store.append(OPT_ID, OptimizationEventType.OUTER_FOLD_STARTED)
        assert [e.sequence for e in store.read_events(OPT_ID)] == [1, 2]

    def test_corrupted_non_final_line_is_fatal(self, tmp_path) -> None:
        store = OptimizationEventStore(tmp_path)
        store.append(OPT_ID, OptimizationEventType.OPTIMIZATION_CREATED)
        store.append(OPT_ID, OptimizationEventType.RUN_STARTED)
        events_path = tmp_path / "optimizations" / OPT_ID / "events.jsonl"
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        events_path.write_text("not valid json at all\n" + lines[1] + "\n", encoding="utf-8")

        with pytest.raises(ArtifactCorruptionError):
            store.read_events(OPT_ID)
