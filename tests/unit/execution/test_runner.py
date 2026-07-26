from __future__ import annotations

import threading
from pathlib import Path

import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)

from quant_platform.core.exceptions import ExecutionResumeError, ExperimentLockError
from quant_platform.execution.executor import DeterministicFoldExecutor
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.historical.locking import DatasetLock
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import ExperimentStatus, FeatureBinding


def _prepared(tmp_path: Path, **spec_overrides: object):
    dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest, **spec_overrides))
    manifest = preparer.prepare(spec)
    return manifest, research_manifest_store, research_store


def _runner(tmp_path: Path, research_manifest_store, research_store, *, fold_executor=None) -> ExecutionRunner:
    return ExecutionRunner(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, fold_executor=fold_executor,
    )


class TestHappyPath:
    def test_fresh_run_completes_all_folds(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        assert manifest.status is ExperimentStatus.READY
        runner = _runner(tmp_path, rms, rds)
        outcome = runner.run(manifest.identity.experiment_id)
        assert not outcome.was_idempotent_no_op
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
        assert outcome.aggregate.failed_fold_indices == ()
        assert outcome.aggregate.total_folds == 3

    def test_experiment_manifest_transitions_to_completed(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        from quant_platform.ml.manifests import ExperimentManifestStore

        reloaded = ExperimentManifestStore(tmp_path / "ml").load(manifest.identity.experiment_id)
        assert reloaded.status is ExperimentStatus.COMPLETED
        assert reloaded.completed_at is not None

    def test_execution_manifest_records_expected_state(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.COMPLETED
        assert exec_manifest.total_folds == 3
        assert set(exec_manifest.fold_result_references) == {0, 1, 2}
        assert len(exec_manifest.artifact_references) == 2  # timeline + execution_summary

    def test_events_recorded_in_expected_order(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        events = [e.event_type.value for e in runner.event_store.read_events(manifest.identity.experiment_id)]
        assert events == [
            "experiment_created", "validation_started", "validation_passed", "run_started",
            "fold_started", "fold_completed", "fold_started", "fold_completed", "fold_started", "fold_completed",
            "run_completed",
        ]

    def test_fold_artifacts_are_readable_and_verifiable(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        for ref in exec_manifest.fold_result_references.values():
            runner.artifact_store.read_artifact(ref.content_hash)  # must not raise


class TestIdempotency:
    def test_second_run_is_idempotent_no_op(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        outcome1 = runner.run(manifest.identity.experiment_id)
        outcome2 = runner.run(manifest.identity.experiment_id)
        assert outcome2.was_idempotent_no_op
        assert outcome2.aggregate == outcome1.aggregate

    def test_idempotent_rerun_does_not_duplicate_events_or_artifacts(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        events_before = runner.event_store.read_events(manifest.identity.experiment_id)
        runner.run(manifest.identity.experiment_id)
        events_after = runner.event_store.read_events(manifest.identity.experiment_id)
        assert len(events_before) == len(events_after)


class TestFoldLevelFailureIsolation:
    class _FailFoldTwoExecutor:
        def execute(self, context, **kwargs):
            if context.fold_index == 1:
                raise ValueError("simulated bad fold data")
            return DeterministicFoldExecutor().execute(context, **kwargs)

    def test_one_bad_fold_does_not_abort_other_folds(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds, fold_executor=self._FailFoldTwoExecutor())
        outcome = runner.run(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.FAILED
        assert outcome.aggregate.completed_fold_indices == (0, 2)
        assert outcome.aggregate.failed_fold_indices == (1,)

    def test_failed_fold_result_records_failure_reason(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds, fold_executor=self._FailFoldTwoExecutor())
        runner.run(manifest.identity.experiment_id)
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        import json

        from quant_platform.execution.results import FoldResult

        ref = exec_manifest.fold_result_references[1]
        raw = runner.artifact_store.read_artifact(ref.content_hash)
        result = FoldResult.from_json_dict(json.loads(raw.decode("utf-8")))
        assert result.status.value == "failed"
        assert "simulated bad fold data" in (result.failure_reason or "")

    def test_experiment_manifest_marked_failed_when_any_fold_fails(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds, fold_executor=self._FailFoldTwoExecutor())
        runner.run(manifest.identity.experiment_id)
        from quant_platform.ml.manifests import ExperimentManifestStore

        reloaded = ExperimentManifestStore(tmp_path / "ml").load(manifest.identity.experiment_id)
        assert reloaded.status is ExperimentStatus.FAILED
        assert reloaded.failure_summary


class TestRecoverableFailureAndResume:
    class _FlakyOnceExecutor:
        def __init__(self) -> None:
            self.attempts: dict[int, int] = {}

        def execute(self, context, **kwargs):
            self.attempts[context.fold_index] = self.attempts.get(context.fold_index, 0) + 1
            if context.fold_index == 1 and self.attempts[context.fold_index] == 1:
                raise ExperimentLockError("simulated transient resource contention")
            return DeterministicFoldExecutor().execute(context, **kwargs)

    def test_lock_error_stops_run_and_marks_recoverable(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        flaky = self._FlakyOnceExecutor()
        runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
        with pytest.raises(ExperimentLockError):
            runner.run(manifest.identity.experiment_id)
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.RECOVERABLE_FAILURE
        assert exec_manifest.completed_fold_indices == (0,)

    def test_experiment_manifest_stays_running_during_recoverable_failure(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        flaky = self._FlakyOnceExecutor()
        runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
        with pytest.raises(ExperimentLockError):
            runner.run(manifest.identity.experiment_id)
        from quant_platform.ml.manifests import ExperimentManifestStore

        reloaded = ExperimentManifestStore(tmp_path / "ml").load(manifest.identity.experiment_id)
        assert reloaded.status is ExperimentStatus.RUNNING

    def test_resume_completes_remaining_folds_and_skips_verified_complete(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        flaky = self._FlakyOnceExecutor()
        runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
        with pytest.raises(ExperimentLockError):
            runner.run(manifest.identity.experiment_id)

        outcome = runner.resume(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome.aggregate.completed_fold_indices == (0, 1, 2)
        # Fold 0 must never have been re-executed (verified complete, skipped).
        assert flaky.attempts[0] == 1

    def test_resume_records_resumed_event_and_bumps_resume_count(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        flaky = self._FlakyOnceExecutor()
        runner = _runner(tmp_path, rms, rds, fold_executor=flaky)
        with pytest.raises(ExperimentLockError):
            runner.run(manifest.identity.experiment_id)
        runner.resume(manifest.identity.experiment_id)
        events = [e.event_type.value for e in runner.event_store.read_events(manifest.identity.experiment_id)]
        assert "execution_resumed" in events
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.resume_count == 1

    def test_resume_without_prior_execution_raises(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(ExecutionResumeError, match="nothing to resume"):
            runner.resume(manifest.identity.experiment_id)

    def test_resume_after_full_completion_is_idempotent(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        outcome = runner.resume(manifest.identity.experiment_id)
        assert outcome.was_idempotent_no_op


class TestForceRerun:
    def test_without_force_a_completed_execution_is_an_idempotent_no_op(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)

        # Nothing left to do without force -- idempotent no-op, no fold reruns.
        outcome2 = runner.run(manifest.identity.experiment_id)
        assert outcome2.was_idempotent_no_op

    def test_force_rerun_folds_only_affects_a_non_terminal_execution(self, tmp_path: Path) -> None:
        """Once COMPLETED, an execution is terminal -- `force_rerun_folds`
        cannot reopen it (see `execution.resume`'s module docstring: this
        milestone does not support restarting a terminal execution in
        place). `force_rerun_folds` only matters while resuming a
        NON-terminal (e.g. `RECOVERABLE_FAILURE`) execution -- exercised
        end-to-end in `TestRecoverableFailureAndResume`."""
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)

        outcome = runner.run(manifest.identity.experiment_id, force_rerun_folds=frozenset({0}))
        assert outcome.was_idempotent_no_op  # still terminal -- force is a no-op here


class TestDuplicateExecutionPrevention:
    def test_concurrent_execution_attempts_serialize_never_corrupt(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
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

        t1 = threading.Thread(target=run, args=(runner_a,))
        t2 = threading.Thread(target=run, args=(runner_b,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        # Whether both succeed (serialized one after another) or one hits
        # a contested lock, the FINAL state must be exactly one clean
        # completed execution -- never corrupted, never double-run.
        final = runner_a.execution_manifest_store.load(manifest.identity.experiment_id)
        assert final.stage is ExecutionStage.COMPLETED
        assert final.completed_fold_indices == (0, 1, 2)

    def test_held_lock_blocks_a_second_run_deterministically(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        holder = DatasetLock(runner._lock_path(manifest.identity.experiment_id))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(manifest.identity.experiment_id)
        finally:
            holder.release()
        # Never created an execution manifest -- the lock was contested
        # before any state was written.
        assert runner.execution_manifest_store.load_if_exists(manifest.identity.experiment_id) is None


class TestFoldPlanValidationFailure:
    def test_invalid_split_params_fail_execution_immediately(self, tmp_path: Path) -> None:
        # n_splits*test_size vastly exceeds available rows -> ValidationSplitError inside splitters,
        # surfaced as a failed, non-resumable execution.
        manifest, rms, rds = _prepared(tmp_path, split_params={"n_splits": 50, "test_size": 100})
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(Exception):  # noqa: B017 - propagates the underlying split error, wrapped or not
            runner.run(manifest.identity.experiment_id)


class TestResolveSerializer:
    def test_unknown_serializer_id_raises_actionably(self) -> None:
        from quant_platform.core.exceptions import UnknownModelDefinitionError
        from quant_platform.execution.runner import resolve_serializer

        with pytest.raises(UnknownModelDefinitionError, match="no_such_serializer"):
            resolve_serializer("no_such_serializer")


class TestAlreadyTerminalExecutionCannotBeRerun:
    def test_running_an_already_failed_execution_raises(self, tmp_path: Path) -> None:
        """A `FAILED` (fold-plan-validation-failure) execution is
        terminal -- a subsequent `.run()`/`.resume()` call must not
        silently re-attempt it, and must not be confused with the
        `COMPLETED` idempotent-no-op path."""
        manifest, rms, rds = _prepared(tmp_path, split_params={"n_splits": 50, "test_size": 100})
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(Exception):  # noqa: B017 - the first attempt's own failure mode
            runner.run(manifest.identity.experiment_id)

        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.FAILED

        with pytest.raises(ExecutionResumeError, match="terminal"):
            runner.run(manifest.identity.experiment_id)


class TestResumeWithNothingLeftPassesThroughStoringResults:
    def test_resume_with_all_folds_already_verified_still_reaches_completed(self, tmp_path: Path) -> None:
        """Regression test for the fix documented in `runner.py`'s own
        module docstring: the legal-transition table only allows the
        final terminal transition FROM `STORING_RESULTS`, never directly
        from `RUNNING_FOLD` -- so a resume that finds EVERY fold already
        verified-complete (nothing left to run) must still explicitly
        pass through `STORING_RESULTS` before reaching `COMPLETED`,
        rather than getting stuck trying an illegal direct transition."""

        from quant_platform.execution.manifests import EXECUTION_MANIFEST_SCHEMA_VERSION, ExecutionManifest
        from quant_platform.execution.results import FoldResult, FoldStatus
        from quant_platform.execution.splitters import (
            build_folds_from_split_binding,
            reconstruct_dataset_timeline,
        )
        from quant_platform.ml.models import ArtifactCategory
        from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now

        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)
        eid = manifest.identity.experiment_id

        from quant_platform.execution.runner import extract_label_horizon_bars

        timeline = reconstruct_dataset_timeline(
            rds, dataset_id=manifest.spec.dataset_binding.dataset_id, content_id=manifest.spec.dataset_binding.content_id,
        )
        dataset_manifest = rms.load(manifest.spec.dataset_binding.dataset_id, manifest.spec.dataset_binding.manifest_version)
        fold_plan = build_folds_from_split_binding(
            manifest.spec.split_binding, timeline["open_time"],
            label_horizon_bars=extract_label_horizon_bars(dataset_manifest),
        )

        now = format_utc_timestamp(utc_now())
        fold_refs = {}
        for fold in fold_plan.folds:
            payload = {
                "schema_version": 1, "fold_index": fold.fold_index, "train_start": now, "train_end": now,
                "test_start": now, "test_end": now, "train_size": len(fold.train_indices), "validation_size": 0,
                "test_size": len(fold.test_indices), "status": FoldStatus.COMPLETED.value, "duration_seconds": 0.1,
                "validation_start": None, "validation_end": None, "artifact_references": [], "metrics": {},
                "failure_reason": None,
            }
            FoldResult.from_json_dict(payload)  # sanity: must itself be a valid FoldResult
            fold_refs[fold.fold_index] = runner.artifact_store.write_artifact(
                canonical_json_bytes(payload), category=ArtifactCategory.FOLD_RESULT,
            )

        exec_manifest = ExecutionManifest(
            schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id=eid, stage=ExecutionStage.INITIALIZING,
            created_at=now, updated_at=now,
        )
        runner.execution_manifest_store.create(exec_manifest)
        runner.execution_manifest_store.transition(eid, new_stage=ExecutionStage.LOADING_DATASET, updated_at=now)
        runner.execution_manifest_store.transition(eid, new_stage=ExecutionStage.BUILDING_SPLITS, updated_at=now)
        runner.execution_manifest_store.transition(
            eid, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=now,
            fold_plan_strategy=fold_plan.strategy, total_folds=len(fold_plan.folds),
            completed_fold_indices=tuple(fold_refs), fold_result_references=fold_refs,
        )

        outcome = runner.resume(eid)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
        assert outcome.aggregate.completed_fold_indices == tuple(sorted(fold_refs))


class TestFoldPlanReportedNotReadyByValidation:
    def test_validation_reporting_not_ready_fails_the_execution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinct from `TestFoldPlanValidationFailure` (where fold
        GENERATION itself raises): this proves the runner correctly
        fails the execution when `validate_fold_plan` reports a
        CRITICAL/ERROR issue despite the plan having been generated
        successfully -- the defense-in-depth path `execution_validation`'s
        own module docstring describes."""
        import quant_platform.execution.runner as runner_module
        from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
        from quant_platform.ml.persistence import format_utc_timestamp, utc_now

        manifest, rms, rds = _prepared(tmp_path)
        runner = _runner(tmp_path, rms, rds)

        fake_report = ValidationReport(
            schema_version=1,
            issues=(ValidationIssue(severity=ValidationSeverity.CRITICAL, code="fake_failure", message="synthetic"),),
            generated_at=format_utc_timestamp(utc_now()),
        )
        monkeypatch.setattr(runner_module, "validate_fold_plan", lambda *_args, **_kwargs: fake_report)

        with pytest.raises(ExecutionResumeError, match="fake_failure"):
            runner.run(manifest.identity.experiment_id)

        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.FAILED
        from quant_platform.ml.manifests import ExperimentManifestStore

        assert ExperimentManifestStore(tmp_path / "ml").load(manifest.identity.experiment_id).status is ExperimentStatus.FAILED


class TestLabelHorizonPurgeEnforcement:
    """Runner-level, end-to-end proof of the Milestone 4B leakage-audit
    fix: an experiment whose declared purge_bars is below the bound
    dataset's real label horizon is REJECTED before a single fold is
    fit/predicted -- the PRIMARY BLOCKER the audit identified there
    illustrated with horizon_bars=12/purge_bars=0/embargo_bars=0; proven
    here against `write_synthetic_research_dataset`'s fixture dataset,
    whose `label_definition.horizon_bars` is 5 (see `conftest.py`), with
    an equally insufficient purge_bars=0/embargo_bars=0 declaration --
    the identical leakage mechanism, different concrete numbers."""

    def test_purge_below_required_label_horizon_is_rejected_before_any_fold_runs(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 0, "embargo_bars": 0})
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(ExecutionResumeError, match="insufficient_label_horizon_purge"):
            runner.run(manifest.identity.experiment_id)

        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.FAILED
        # No fold ever ran: no fold_started event, no fold_result references.
        events = [e.event_type.value for e in runner.event_store.read_events(manifest.identity.experiment_id)]
        assert "fold_started" not in events
        assert dict(exec_manifest.fold_result_references) == {}

    def test_purge_one_below_required_minimum_is_rejected(self, tmp_path: Path) -> None:
        # Fixture dataset's label horizon is 5 -- required minimum is 5, so 4 is one below it.
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 4, "embargo_bars": 0})
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(ExecutionResumeError, match="insufficient_label_horizon_purge"):
            runner.run(manifest.identity.experiment_id)

    def test_purge_exactly_at_required_minimum_passes(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 5, "embargo_bars": 0})
        runner = _runner(tmp_path, rms, rds)
        outcome = runner.run(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED

    def test_purge_above_required_minimum_passes(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 50, "embargo_bars": 0})
        runner = _runner(tmp_path, rms, rds)
        outcome = runner.run(manifest.identity.experiment_id)
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED

    def test_large_embargo_cannot_mask_insufficient_purge(self, tmp_path: Path) -> None:
        """Embargo is additional protection, never a substitute for the
        label-information purge -- proven end-to-end: even an embargo far
        larger than the required label horizon (5) does not let an
        insufficient purge_bars=0 pass. (embargo_bars is kept small
        enough that `PurgedWalkForwardSplitter` can still build folds at
        all against the fixture's 1000-row timeline -- an embargo large
        enough to exhaust all training rows would fail for an unrelated
        reason before ever reaching this check.)"""
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 0, "embargo_bars": 50})
        runner = _runner(tmp_path, rms, rds)
        with pytest.raises(ExecutionResumeError, match="insufficient_label_horizon_purge"):
            runner.run(manifest.identity.experiment_id)

    def test_dataset_manifest_is_the_source_not_split_binding_params(self, tmp_path: Path) -> None:
        """A caller cannot smuggle a fabricated `label_horizon_bars` into
        `SplitBinding.params` to escape (or falsely trigger) the check --
        the runner never reads such a key; only the bound
        `ResearchDatasetManifest`'s own `label_definition` is consulted."""
        manifest, rms, rds = _prepared(
            tmp_path, split_params={"purge_bars": 5, "embargo_bars": 0, "label_horizon_bars": 0},
        )
        runner = _runner(tmp_path, rms, rds)
        outcome = runner.run(manifest.identity.experiment_id)  # succeeds: real horizon is 5, purge=5 suffices
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED

        manifest2, rms2, rds2 = _prepared(
            tmp_path, split_params={"purge_bars": 0, "embargo_bars": 0, "label_horizon_bars": 999},
        )
        runner2 = _runner(tmp_path, rms2, rds2)
        with pytest.raises(ExecutionResumeError, match="insufficient_label_horizon_purge"):
            runner2.run(manifest2.identity.experiment_id)  # still rejected: params['label_horizon_bars'] is ignored

    def test_required_purge_derived_metadata_persisted_on_execution_manifest(self, tmp_path: Path) -> None:
        manifest, rms, rds = _prepared(tmp_path, split_params={"purge_bars": 8, "embargo_bars": 2})
        runner = _runner(tmp_path, rms, rds)
        runner.run(manifest.identity.experiment_id)
        exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
        assert exec_manifest.declared_purge_bars == 8
        assert exec_manifest.required_label_purge_bars == 5
        assert exec_manifest.effective_purge_bars == 8
        assert exec_manifest.embargo_bars == 2
        assert exec_manifest.label_horizon_source == "research_dataset_manifest"
        assert exec_manifest.split_policy == "reject_if_declared_purge_below_required_label_horizon"


class TestExperimentNotReady:
    def test_running_a_non_ready_experiment_raises(self, tmp_path: Path) -> None:
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        bad_spec = ExperimentSpec(**make_experiment_spec_kwargs(
            dataset_manifest=dataset_manifest,
            feature_binding=FeatureBinding(
                feature_names=("nope",), feature_versions={"nope": "1"}, feature_registry_fingerprint="b" * 64,
            ),
        ))
        manifest = preparer.prepare(bad_spec)
        assert manifest.status is ExperimentStatus.FAILED
        runner = _runner(tmp_path, research_manifest_store, research_store)
        with pytest.raises(ExecutionResumeError, match="only a READY"):
            runner.run(manifest.identity.experiment_id)
