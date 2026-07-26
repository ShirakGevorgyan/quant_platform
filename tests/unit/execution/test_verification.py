"""Unit tests for `execution.verification.verify_execution` -- built on a
REAL, end-to-end COMPLETED execution (via `ExecutionRunner`, exactly like
`tests/unit/execution/test_runner.py`), then selectively hand-tampering
ONE store at a time (mirroring `test_adversarial.py`'s established
"tamper after a real run" convention) to prove each cross-store
consistency check actually fires. Never mocks away the scientific
boundary or the four real stores this module cross-checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)

from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.verification import verify_execution
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ValidationReport


def _completed_execution(tmp_path: Path) -> tuple[str, ExecutionRunner]:
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
    outcome = runner.run(manifest.identity.experiment_id)
    assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
    return manifest.identity.experiment_id, runner


def _verify(tmp_path: Path, runner: ExecutionRunner, experiment_id: str) -> ValidationReport:
    return verify_execution(
        experiment_id,
        execution_manifest_store=runner.execution_manifest_store,
        experiment_manifest_store=ExperimentManifestStore(tmp_path / "ml"),
        artifact_store=runner.artifact_store,
        event_store=runner.event_store,
    )


class TestCleanExecutionVerifiesReady:
    def test_clean_completed_execution_is_ready(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        report = _verify(tmp_path, runner, experiment_id)
        assert report.is_ready
        codes = {i.code for i in report.infos}
        assert {
            "all_fold_results_verified", "aggregate_verified", "timeline_verified",
            "terminal_stage_compatible_with_aggregate", "experiment_status_compatible",
            "event_sequence_has_no_impossible_transitions", "terminal_event_present",
        } <= codes


class TestAggregateExecutionManifestDisagreement:
    def test_completed_indices_mismatch_detected(self, tmp_path: Path) -> None:
        """Hand-edit the (plain, mutable) execution manifest file to claim
        FEWER completed folds than the immutable, already-written
        aggregate artifact actually records -- proves the aggregate/
        manifest cross-check (required guarantee: 'verify-execution
        detects aggregate/manifest disagreement')."""
        experiment_id, runner = _completed_execution(tmp_path)
        manifest_path = runner.execution_manifest_store._manifest_path(experiment_id)
        raw = json.loads(manifest_path.read_text())
        raw["completed_fold_indices"] = raw["completed_fold_indices"][:-1]
        manifest_path.write_text(json.dumps(raw))

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "aggregate_completed_indices_mismatch" for i in report.criticals)

    def test_total_folds_mismatch_detected(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        manifest_path = runner.execution_manifest_store._manifest_path(experiment_id)
        raw = json.loads(manifest_path.read_text())
        raw["total_folds"] = raw["total_folds"] + 1
        manifest_path.write_text(json.dumps(raw))

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "aggregate_total_folds_mismatch" for i in report.criticals)


class TestExperimentExecutionManifestDisagreement:
    def test_experiment_status_incompatible_with_terminal_execution_stage_detected(self, tmp_path: Path) -> None:
        """Hand-edit the Milestone 4A `ExperimentManifest` file so its
        `status` no longer matches the (terminal, COMPLETED) execution
        stage -- proves the required guarantee: 'verify-execution detects
        ExperimentManifest/ExecutionManifest disagreement'."""
        experiment_id, runner = _completed_execution(tmp_path)
        experiment_manifest_path = ExperimentManifestStore(runner.execution_manifest_store.root)._manifest_path(experiment_id)
        raw = json.loads(experiment_manifest_path.read_text())
        raw["status"] = "running"
        raw["completed_at"] = None  # ExperimentManifest.__post_init__ requires this pairing
        experiment_manifest_path.write_text(json.dumps(raw))

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        issue = next(i for i in report.criticals if i.code == "experiment_status_execution_stage_incompatible")
        assert "completed" in issue.message
        assert "running" in issue.message


class TestMissingExpectedEventDetected:
    def test_missing_terminal_run_completed_event_is_a_warning_not_a_failure(self, tmp_path: Path) -> None:
        """Simulates the documented manifest-write-before-event-append
        crash window: the terminal ExecutionManifest write succeeded, but
        the closing RUN_COMPLETED event never got appended. Must be
        reported as an explicit WARNING (never silently ignored) while
        `is_ready` stays True (recoverable incompleteness, not
        corruption) -- required guarantee: 'verify-execution detects a
        missing expected terminal/fold event'."""
        experiment_id, runner = _completed_execution(tmp_path)
        events_path = runner.event_store._events_path(experiment_id)
        lines = [line for line in events_path.read_text(encoding="utf-8").split("\n") if line]
        assert json.loads(lines[-1])["event_type"] == "run_completed"
        events_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        report = _verify(tmp_path, runner, experiment_id)
        assert report.is_ready  # a WARNING alone must not block readiness
        issue = next(i for i in report.warnings if i.code == "terminal_manifest_missing_terminal_event")
        assert "run_completed" in issue.message

    def test_fold_completion_event_without_matching_start_detected(self, tmp_path: Path) -> None:
        """A `FOLD_COMPLETED` event whose fold_index has no earlier
        `FOLD_STARTED` is an impossible transition -- proves the event-
        sequence 'impossible transitions' check."""
        experiment_id, runner = _completed_execution(tmp_path)
        events_path = runner.event_store._events_path(experiment_id)
        lines = [line for line in events_path.read_text(encoding="utf-8").split("\n") if line]
        records = [json.loads(line) for line in lines]
        for record in records:
            if record["event_type"] == "fold_started" and record["details"].get("fold_index") == 0:
                record["details"] = {}  # erase the fold_index so fold 0's completed event has no matching start
        events_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_completion_event_without_start" for i in report.criticals)


class TestFoldResultCrossChecks:
    def test_fold_result_index_mismatch_detected(self, tmp_path: Path) -> None:
        """Hand-tamper the execution manifest so fold 0's REAL artifact
        reference (whose decoded FoldResult.fold_index is genuinely 0) is
        instead filed under key 1 as well -- proves the same fold_index
        cross-check `resume.py` enforces is ALSO enforced here, with a
        richer per-fold report."""
        experiment_id, runner = _completed_execution(tmp_path)
        manifest_path = runner.execution_manifest_store._manifest_path(experiment_id)
        raw = json.loads(manifest_path.read_text())
        raw["fold_result_references"]["1"] = raw["fold_result_references"]["0"]
        manifest_path.write_text(json.dumps(raw))

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        issue = next(i for i in report.criticals if i.code == "fold_result_index_mismatch")
        assert issue.context["manifest_fold_index"] == 1
        assert issue.context["decoded_fold_index"] == 0

    def test_fold_result_content_corrupted_detected(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        exec_manifest = runner.execution_manifest_store.load(experiment_id)
        ref = exec_manifest.fold_result_references[0]
        runner.artifact_store._content_path(ref.content_hash).write_bytes(b"TAMPERED")

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_unverifiable" for i in report.criticals)


class TestNonTerminalExecutionNeverFalselyFlagged:
    def test_recoverable_failure_with_no_completed_folds_is_clean(self, tmp_path: Path) -> None:
        """A non-terminal execution (stopped by a transient failure
        before ANY fold completed) must never be flagged for a "missing"
        aggregate/timeline -- neither was ever supposed to exist yet.
        Verify-execution must recognize this as a genuinely clean,
        still-in-progress state, not corruption."""
        from quant_platform.core.exceptions import ExperimentLockError

        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
        prepared = preparer.prepare(spec)
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
            research_dataset_store=research_store, fold_executor=_AlwaysLockErrorOnFirstAttempt(),
        )
        with pytest.raises(ExperimentLockError):
            runner.run(prepared.identity.experiment_id)

        exec_manifest = runner.execution_manifest_store.load(prepared.identity.experiment_id)
        assert exec_manifest.stage is ExecutionStage.RECOVERABLE_FAILURE
        assert exec_manifest.completed_fold_indices == ()

        report = _verify(tmp_path, runner, prepared.identity.experiment_id)
        assert report.is_ready
        assert not any(i.code in ("aggregate_missing", "timeline_missing") for i in report.issues)


class _AlwaysLockErrorOnFirstAttempt:
    def __init__(self) -> None:
        self._raised = False

    def execute(self, context, **kwargs):
        if not self._raised:
            self._raised = True
            from quant_platform.core.exceptions import ExperimentLockError

            raise ExperimentLockError("simulated contention on the very first fold")
        from quant_platform.execution.executor import DeterministicFoldExecutor

        return DeterministicFoldExecutor().execute(context, **kwargs)


class TestNoRawExceptionsEscape:
    def test_malformed_aggregate_json_fails_closed_not_raw(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        exec_manifest = runner.execution_manifest_store.load(experiment_id)
        from quant_platform.ml.models import ArtifactCategory

        summary_ref = next(r for r in exec_manifest.artifact_references if r.category is ArtifactCategory.EXECUTION_SUMMARY)
        runner.artifact_store._content_path(summary_ref.content_hash).write_bytes(b"not json {{{")

        report = _verify(tmp_path, runner, experiment_id)  # must not raise
        assert not report.is_ready
        assert any(i.code == "aggregate_unverifiable" for i in report.criticals)
