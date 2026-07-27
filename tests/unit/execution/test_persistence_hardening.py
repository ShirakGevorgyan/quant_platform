"""Milestone 4D.1 (cross-package durable-persistence and verification
hardening) -- execution-package-specific regression tests.

Built on the SAME "real, end-to-end COMPLETED execution, then selectively
hand-tamper ONE artifact at a time" convention `test_verification.py`
already established, extended here to cover the specific defect class this
milestone targets: unsafe `json.loads` swapped for `quant_platform.ml.
persistence.parse_json_strict` in `execution.verification`, `execution.
runner`, and `execution.resume`, plus the schema/envelope identity checks
(artifact category, fold_index, experiment_id) added alongside it.

Every tamper here writes a NEW, internally self-consistent payload under
its own correctly-computed content hash (via `artifact_store.write_artifact`
directly) -- the realistic threat model this platform's content-addressed
storage actually has to defend against: a hash mismatch is already caught
unconditionally by `MLArtifactStore.read_artifact`; what these tests prove
is that a *self-consistently hashed but semantically wrong* payload is
still caught.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)

from quant_platform.core.exceptions import ExecutionResumeError
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult, FoldStatus
from quant_platform.execution.resume import verify_completed_folds
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.verification import verify_execution
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, ValidationReport

_NOW = "2024-01-01T00:00:00+00:00"


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


def _repoint_fold_result_reference(
    runner: ExecutionRunner, experiment_id: str, *, fold_index: int, new_content_hash: str,
    new_category: ArtifactCategory = ArtifactCategory.FOLD_RESULT,
) -> None:
    """Hand-edits the (plain, mutable) execution manifest so
    `fold_result_references[fold_index]` points at a DIFFERENT,
    already-written, self-consistently-hashed artifact -- the same
    manifest-file-tampering convention `test_verification.py` uses, just
    targeting a fold-result reference instead of the claimed-complete
    index list."""
    manifest_path = runner.execution_manifest_store._manifest_path(experiment_id)
    raw = json.loads(manifest_path.read_text())
    raw["fold_result_references"][str(fold_index)] = {
        "category": new_category.value, "content_hash": new_content_hash,
        "size_bytes": 1, "created_at": _NOW,
    }
    manifest_path.write_text(json.dumps(raw))


def _repoint_execution_summary_reference(runner: ExecutionRunner, experiment_id: str, *, new_content_hash: str) -> None:
    manifest_path = runner.execution_manifest_store._manifest_path(experiment_id)
    raw = json.loads(manifest_path.read_text())
    for ref in raw["artifact_references"]:
        if ref["category"] == ArtifactCategory.EXECUTION_SUMMARY.value:
            ref["content_hash"] = new_content_hash
    manifest_path.write_text(json.dumps(raw))


class TestCorruptedFoldResultArtifactNeverEscapesRaw:
    """`verify_execution`, `resume.verify_completed_folds`, and
    `ExecutionRunner._load_all_fold_results` each independently read the
    same FOLD_RESULT artifacts -- every one of the three must convert a
    corrupted artifact into its own domain-appropriate outcome, never a
    raw `json.JSONDecodeError`/`KeyError`/`ValueError`/`TypeError`."""

    def test_malformed_json_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.FOLD_RESULT)
        _repoint_fold_result_reference(runner, experiment_id, fold_index=0, new_content_hash=bad_ref.content_hash)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_unverifiable" for i in report.criticals)

    def test_nan_metric_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        """A bare `NaN` token: `json.loads` (the old, unsafe path) would
        have silently accepted this; `parse_json_strict` rejects it, so it
        surfaces as the SAME `fold_result_unverifiable` critical as any
        other malformed payload -- proving the parser swap is actually
        wired into the read path, not just present in the module."""
        experiment_id, runner = _completed_execution(tmp_path)
        poisoned = b'{"schema_version": 1, "fold_index": 0, "train_start": "2024-01-01T00:00:00+00:00", "train_end": "2024-01-01T00:00:00+00:00", "test_start": "2024-01-01T00:00:00+00:00", "test_end": "2024-01-01T00:00:00+00:00", "train_size": 1, "test_size": 1, "status": "completed", "duration_seconds": NaN, "validation_size": 0, "artifact_references": [], "metrics": {}, "failure_reason": null}'
        bad_ref = runner.artifact_store.write_artifact(poisoned, category=ArtifactCategory.FOLD_RESULT)
        _repoint_fold_result_reference(runner, experiment_id, fold_index=0, new_content_hash=bad_ref.content_hash)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_unverifiable" for i in report.criticals)

    def test_invalid_utf8_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"\xff\xfe\x00bad utf8 \x80\x81", category=ArtifactCategory.FOLD_RESULT)
        _repoint_fold_result_reference(runner, experiment_id, fold_index=0, new_content_hash=bad_ref.content_hash)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_unverifiable" for i in report.criticals)

    def test_wrong_category_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        other_ref = runner.artifact_store.write_artifact(b'{"unrelated": true}', category=ArtifactCategory.TIMELINE)
        _repoint_fold_result_reference(
            runner, experiment_id, fold_index=0, new_content_hash=other_ref.content_hash, new_category=ArtifactCategory.TIMELINE,
        )

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_wrong_category" for i in report.criticals)

    def test_wrong_fold_index_identity_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        """A genuinely valid FOLD_RESULT artifact -- just filed under the
        WRONG key: fold 1's real result substituted in for fold 0's. A
        valid content hash proves the bytes are intact, not that they were
        filed under the correct key."""
        experiment_id, runner = _completed_execution(tmp_path)
        manifest = runner.execution_manifest_store.load(experiment_id)
        fold_1_ref = manifest.fold_result_references[1]
        _repoint_fold_result_reference(runner, experiment_id, fold_index=0, new_content_hash=fold_1_ref.content_hash)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "fold_result_index_mismatch" for i in report.criticals)

    def test_corrupted_claimed_complete_fold_demoted_to_needs_rerun_not_raised(self, tmp_path: Path) -> None:
        """`resume.verify_completed_folds`'s whole contract: a corrupted
        claimed-complete fold must never crash resume planning -- it is
        silently (but honestly, via its return value) demoted to
        needs_rerun instead."""
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.FOLD_RESULT)
        _repoint_fold_result_reference(runner, experiment_id, fold_index=0, new_content_hash=bad_ref.content_hash)

        manifest = runner.execution_manifest_store.load(experiment_id)
        verified, needs_rerun = verify_completed_folds(manifest, artifact_store=runner.artifact_store)
        assert 0 in needs_rerun
        assert 0 not in verified

    def test_corrupted_fold_result_raises_execution_resume_error_not_raw_exception_on_finalize(
        self, tmp_path: Path,
    ) -> None:
        """`ExecutionRunner._load_all_fold_results` (called while
        finalizing a run, reading back every fold result including ones
        this same call just wrote) must translate a corrupted artifact
        into `ExecutionResumeError`, never a raw JSON/Key/Type/Value
        error escaping a public `ExecutionRunner` method."""
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.FOLD_RESULT)
        manifest = runner.execution_manifest_store.load(experiment_id)
        with pytest.raises(ExecutionResumeError, match="could not be read and decoded"):
            runner._load_all_fold_results({0: bad_ref, **{k: v for k, v in manifest.fold_result_references.items() if k != 0}})

    def test_wrong_category_fold_result_raises_execution_resume_error_on_finalize(self, tmp_path: Path) -> None:
        _experiment_id, runner = _completed_execution(tmp_path)
        wrong_category_ref = runner.artifact_store.write_artifact(b'{"unrelated": true}', category=ArtifactCategory.TIMELINE)
        with pytest.raises(ExecutionResumeError, match="expected 'fold_result'"):
            runner._load_all_fold_results({0: wrong_category_ref})


class TestCorruptedExecutionSummaryArtifactNeverEscapesRaw:
    """`ExecutionRunner._load_existing_aggregate` -- the idempotent-resume
    path for an already-COMPLETED execution -- and `verify_execution`'s
    `_verify_aggregate` both independently read the EXECUTION_SUMMARY
    artifact."""

    def test_malformed_json_raises_execution_resume_error_on_idempotent_resume(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.EXECUTION_SUMMARY)
        _repoint_execution_summary_reference(runner, experiment_id, new_content_hash=bad_ref.content_hash)

        with pytest.raises(ExecutionResumeError, match="could not be read and decoded"):
            runner.run(experiment_id)

    def test_malformed_json_reported_as_critical_by_verify_execution(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.EXECUTION_SUMMARY)
        _repoint_execution_summary_reference(runner, experiment_id, new_content_hash=bad_ref.content_hash)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "aggregate_unverifiable" for i in report.criticals)

    def test_wrong_experiment_id_raises_execution_resume_error_on_idempotent_resume(self, tmp_path: Path) -> None:
        """A genuinely valid, hash-correct AggregatedExecutionResult --
        just belonging to a DIFFERENT experiment_id. A valid content hash
        proves the bytes are intact, not that this is genuinely THIS
        execution's own summary."""
        experiment_id, runner = _completed_execution(tmp_path)
        manifest = runner.execution_manifest_store.load(experiment_id)
        foreign_aggregate = AggregatedExecutionResult(
            schema_version=1, experiment_id="f" * 64, total_folds=3,
            completed_fold_indices=(0, 1, 2), failed_fold_indices=(), overall_status=ExecutionStage.COMPLETED,
            started_at=_NOW, completed_at=_NOW, execution_duration_seconds=1.0,
        )
        from quant_platform.ml.persistence import canonical_json_bytes

        foreign_ref = runner.artifact_store.write_artifact(
            canonical_json_bytes(foreign_aggregate.to_json_dict()), category=ArtifactCategory.EXECUTION_SUMMARY,
        )
        _repoint_execution_summary_reference(runner, experiment_id, new_content_hash=foreign_ref.content_hash)

        with pytest.raises(ExecutionResumeError, match="decodes to experiment_id"):
            runner.run(experiment_id)

        report = _verify(tmp_path, runner, experiment_id)
        assert not report.is_ready
        assert any(i.code == "aggregate_experiment_id_mismatch" for i in report.criticals)

        del manifest  # only used to establish the pre-tamper baseline above


class TestConcurrencyRegressionAfterCorruptionRaise:
    def test_run_lock_is_released_after_execution_resume_error_mid_lock(self, tmp_path: Path) -> None:
        """Proves `ExecutionResumeError` raised from inside
        `_load_existing_aggregate` (itself inside `with experiment_lock(...)`)
        does not leave the run lock stuck -- a second call must fail on
        the SAME corruption again, never on a stale lock."""
        experiment_id, runner = _completed_execution(tmp_path)
        bad_ref = runner.artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.EXECUTION_SUMMARY)
        _repoint_execution_summary_reference(runner, experiment_id, new_content_hash=bad_ref.content_hash)

        with pytest.raises(ExecutionResumeError, match="could not be read and decoded"):
            runner.run(experiment_id)
        with pytest.raises(ExecutionResumeError, match="could not be read and decoded"):
            runner.run(experiment_id)


class TestFiniteNumberInvariantsRejectNonFiniteDurations:
    def test_fold_result_rejects_nan_duration(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FoldResult(
                schema_version=1, fold_index=0, train_start=_NOW, train_end=_NOW, test_start=_NOW, test_end=_NOW,
                train_size=1, test_size=1, status=FoldStatus.COMPLETED, duration_seconds=math.nan,
            )

    def test_fold_result_rejects_infinite_duration(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FoldResult(
                schema_version=1, fold_index=0, train_start=_NOW, train_end=_NOW, test_start=_NOW, test_end=_NOW,
                train_size=1, test_size=1, status=FoldStatus.COMPLETED, duration_seconds=math.inf,
            )

    def test_fold_result_accepts_zero_duration(self) -> None:
        result = FoldResult(
            schema_version=1, fold_index=0, train_start=_NOW, train_end=_NOW, test_start=_NOW, test_end=_NOW,
            train_size=1, test_size=1, status=FoldStatus.COMPLETED, duration_seconds=0.0,
        )
        assert result.duration_seconds == 0.0

    def test_aggregated_execution_result_rejects_nan_duration(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AggregatedExecutionResult(
                schema_version=1, experiment_id="e" * 64, total_folds=1, completed_fold_indices=(0,),
                failed_fold_indices=(), overall_status=ExecutionStage.COMPLETED, started_at=_NOW, completed_at=_NOW,
                execution_duration_seconds=math.nan,
            )

    def test_aggregated_execution_result_rejects_infinite_duration(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AggregatedExecutionResult(
                schema_version=1, experiment_id="e" * 64, total_folds=1, completed_fold_indices=(0,),
                failed_fold_indices=(), overall_status=ExecutionStage.COMPLETED, started_at=_NOW, completed_at=_NOW,
                execution_duration_seconds=math.inf,
            )

    def test_string_coerced_nan_duration_is_rejected_on_from_json_dict(self) -> None:
        """The residual bypass `parse_json_strict`'s bare-token rejection
        does not cover: a JSON STRING `"nan"` survives strict parsing (it
        is a legitimate string), then gets silently accepted by Python's
        own `float("nan")` coercion inside `from_json_dict`. Proves the
        `__post_init__` finiteness check (not the parser) is what closes
        this specific gap."""
        raw = {
            "schema_version": 1, "fold_index": 0, "train_start": _NOW, "train_end": _NOW,
            "test_start": _NOW, "test_end": _NOW, "train_size": 1, "test_size": 1,
            "status": "completed", "duration_seconds": "nan",
        }
        with pytest.raises(ValueError, match="finite"):
            FoldResult.from_json_dict(raw)


class TestBackwardCompatibilityValidArtifactsRemainReadable:
    def test_a_legitimately_completed_execution_from_before_this_hardening_still_verifies_ready(
        self, tmp_path: Path,
    ) -> None:
        """No tampering at all -- proves the `parse_json_strict` swap and
        the new category/identity/finite-number checks do not reject any
        legitimately-produced execution; only genuinely corrupted or
        semantically-wrong data is newly caught."""
        experiment_id, runner = _completed_execution(tmp_path)
        report = _verify(tmp_path, runner, experiment_id)
        assert report.is_ready

    def test_resuming_an_untampered_completed_execution_is_still_a_clean_idempotent_no_op(self, tmp_path: Path) -> None:
        experiment_id, runner = _completed_execution(tmp_path)
        outcome = runner.run(experiment_id)
        assert outcome.was_idempotent_no_op
        assert outcome.aggregate.overall_status is ExecutionStage.COMPLETED
