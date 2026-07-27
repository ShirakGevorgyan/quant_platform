"""`verify_execution`: an independent, read-only re-audit of everything an
execution has recorded across its FOUR separate durable stores --
`ExecutionManifest`, `ExperimentManifest` (Milestone 4A), the artifact
store (fold results, aggregate, timeline), and the append-only event log
-- proving they still agree with each other, not just that each one is
independently well-formed.

WHY THIS EXISTS, SEPARATELY FROM `execution.resume`
--------------------------------------------------------------------------
`execution.resume.verify_completed_folds` answers a NARROW, forward-
looking question for one still-resumable execution: "which claimed-
complete folds can I trust well enough to skip re-running?" This module
answers a BROADER, backward-looking question for ANY execution (still
running, terminal, or long finished): "is everything this execution ever
wrote still mutually consistent?" The two deliberately do not share one
implementation -- `resume` needs a fast binary partition to plan future
work; this module needs a detailed, per-check audit trail a human (or
`ml_cli.py`'s `verify-execution` command) can read.

DO NOT REQUIRE FALSE ATOMICITY ACROSS FILES
--------------------------------------------------------------------------
`ExecutionManifest`, `ExperimentManifest`, artifacts, and the event log
are FOUR separate files/directories, written by separate, non-atomic
operations, never inside one cross-file transaction. `execution.runner.
ExecutionRunner._execute_pipeline` writes them in a specific, DELIBERATE
order at each step, and a process crash between any two writes is always
possible. This module's job is to recognize the SPECIFIC, KNOWN crash
windows this codebase's own write ordering creates and report them
HONESTLY as recoverable incompleteness (`ValidationSeverity.WARNING` --
`ValidationReport.is_ready` stays true) -- never silently ignored, and
never escalated to a false claim of corruption:

  * Per-fold, the EVENT (`FOLD_STARTED`/`FOLD_COMPLETED`/`FOLD_FAILED`)
    is appended BEFORE the `ExecutionManifest` transition that folds it
    into `completed_fold_indices`/`failed_fold_indices` -- so a crash
    here can leave the event log MOMENTARILY AHEAD of the manifest. This
    is normal, benign, and self-heals on the next call (the manifest
    transition either completes normally, or a future `resume()`
    re-verifies the fold independently) -- NOT checked here as an error
    condition for a still-running execution.
  * At the very end, the terminal `ExecutionManifest` transition (and
    then the terminal `ExperimentManifest` transition) is written BEFORE
    the describing `RUN_COMPLETED`/`RUN_FAILED` event is appended -- so a
    crash here can leave a terminal, authoritative manifest whose event
    log never got its closing entry. THIS is the crash window this
    module explicitly detects and reports (`terminal_manifest_missing_terminal_event`,
    WARNING) -- the manifest remains authoritative; only the event log's
    history of the run is incomplete.
"""

from __future__ import annotations

from quant_platform.core.exceptions import ArtifactCorruptionError, ArtifactNotFoundError, SchemaVersionError
from quant_platform.execution.manifests import ExecutionManifest, ExecutionManifestStore
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult, FoldStatus
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.timeline import Timeline
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
    ExperimentStatus,
    JsonPrimitive,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_json_strict, utc_now
from quant_platform.ml.tracking import EventRecord, EventType, ExperimentEventStore

_SCHEMA_VERSION = 1

_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError,
)
"""Every failure mode reading+decoding a content-addressed artifact can
legitimately hit. All treated identically here: report a CRITICAL
`ValidationIssue`, never let a raw exception escape this module (required
guarantee: no raw `KeyError`/`ValueError`/`TypeError`/JSON-decode
exception escapes execution verification)."""

_TERMINAL_STAGE_TO_RUN_EVENT: dict[ExecutionStage, EventType] = {
    ExecutionStage.COMPLETED: EventType.RUN_COMPLETED,
    ExecutionStage.FAILED: EventType.RUN_FAILED,
}

_TERMINAL_STAGE_TO_EXPERIMENT_STATUS: dict[ExecutionStage, ExperimentStatus] = {
    ExecutionStage.COMPLETED: ExperimentStatus.COMPLETED,
    ExecutionStage.FAILED: ExperimentStatus.FAILED,
    ExecutionStage.CANCELLED: ExperimentStatus.CANCELLED,
}
"""`ExperimentManifest.status` this milestone's runner sets AT THE SAME
TIME it moves `ExecutionManifest.stage` to the matching terminal value
(see `ExecutionRunner._transition_experiment_to_terminal`). Every
NON-terminal `ExecutionStage` maps to `ExperimentStatus.RUNNING` -- set
once, before an `ExecutionManifest` is even created (see
`ExecutionRunner._run_locked`), and never changed again until a terminal
execution stage is reached."""


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def verify_execution(
    experiment_id: str,
    *,
    execution_manifest_store: ExecutionManifestStore,
    experiment_manifest_store: ExperimentManifestStore,
    artifact_store: MLArtifactStore,
    event_store: ExperimentEventStore,
) -> ValidationReport:
    """Loads everything itself (never trusts a caller-supplied, possibly
    stale copy of any of the four stores' state) and returns ONE
    `ValidationReport` -- never raises for an inconsistent-but-loadable
    execution; only a missing `ExecutionManifest`/`ExperimentManifest`
    entirely (nothing to verify at all) propagates the underlying store's
    own `ArtifactNotFoundError`, exactly like `ml_cli.py`'s other
    `load`-based commands."""
    execution_manifest = execution_manifest_store.load(experiment_id)
    experiment_manifest = experiment_manifest_store.load(experiment_id)

    issues: list[ValidationIssue] = []
    issues += _verify_fold_results(execution_manifest, artifact_store=artifact_store)
    aggregate, aggregate_issues = _verify_aggregate(execution_manifest, artifact_store=artifact_store)
    issues += aggregate_issues
    issues += _verify_timeline(execution_manifest, artifact_store=artifact_store)
    issues += _verify_terminal_stage_matches_aggregate(execution_manifest, aggregate)
    issues += _verify_experiment_manifest_compatibility(execution_manifest, experiment_manifest)
    issues += _verify_events(execution_manifest, event_store=event_store)
    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _plan_was_built(execution_manifest: ExecutionManifest) -> bool:
    """True once a fold plan was successfully built and validated (the
    `BUILDING_SPLITS -> RUNNING_FOLD` transition recorded `total_folds`)
    -- distinguishes a `FAILED` execution that never got that far (no
    aggregate/timeline artifact was ever supposed to exist) from one that
    ran folds and then failed (which DOES have both)."""
    return execution_manifest.total_folds is not None


def _verify_fold_results(execution_manifest: ExecutionManifest, *, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    completed = set(execution_manifest.completed_fold_indices)
    failed = set(execution_manifest.failed_fold_indices)

    # `ExecutionManifest.__post_init__` already structurally guarantees
    # every completed_fold_indices entry has a fold_result_references
    # entry -- but NOT every failed_fold_indices entry (no such
    # constructor-level check exists for the failed set). Checked
    # explicitly here so a manifest that somehow violates it (a hand
    # edit, or a future bug) is still caught, not silently assumed away.
    missing_failed_refs = failed - set(execution_manifest.fold_result_references)
    if missing_failed_refs:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "failed_fold_result_reference_missing",
            f"failed_fold_indices names fold(s) {sorted(missing_failed_refs)} with no corresponding "
            "fold_result_references entry",
            fold_indices=", ".join(str(i) for i in sorted(missing_failed_refs)),
        ))

    for fold_index, reference in sorted(execution_manifest.fold_result_references.items()):
        if reference.category is not ArtifactCategory.FOLD_RESULT:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_result_wrong_category",
                f"Fold {fold_index}: artifact reference has category={reference.category.value!r}, "
                f"expected {ArtifactCategory.FOLD_RESULT.value!r}",
                fold_index=fold_index,
            ))
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = FoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_result_unverifiable",
                f"Fold {fold_index}: artifact could not be read and decoded as a FoldResult: {exc}",
                fold_index=fold_index,
            ))
            continue
        if decoded.fold_index != fold_index:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_result_index_mismatch",
                f"Fold {fold_index}: decoded FoldResult.fold_index={decoded.fold_index} does not match the "
                "manifest's own fold_result_references key -- a valid content hash proves the bytes are "
                "intact, not that they were filed under the correct key",
                manifest_fold_index=fold_index, decoded_fold_index=decoded.fold_index,
            ))
            continue
        expected_status = FoldStatus.COMPLETED if fold_index in completed else FoldStatus.FAILED if fold_index in failed else None
        if expected_status is not None and decoded.status is not expected_status:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_result_status_mismatch",
                f"Fold {fold_index}: manifest classifies it as "
                f"{'completed' if expected_status is FoldStatus.COMPLETED else 'failed'}, but the decoded "
                f"FoldResult.status is {decoded.status.value!r}",
                fold_index=fold_index, decoded_status=decoded.status.value,
            ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "all_fold_results_verified",
            f"All {len(execution_manifest.fold_result_references)} recorded fold result artifact(s) verified: "
            "correct category, decodable, matching internal fold_index, and status-consistent",
        ))
    return issues


def _verify_aggregate(
    execution_manifest: ExecutionManifest, *, artifact_store: MLArtifactStore,
) -> tuple[AggregatedExecutionResult | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    summary_ref = next(
        (r for r in execution_manifest.artifact_references if r.category is ArtifactCategory.EXECUTION_SUMMARY), None,
    )
    if summary_ref is None:
        if execution_manifest.stage in _TERMINAL_STAGE_TO_RUN_EVENT and _plan_was_built(execution_manifest):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "aggregate_missing",
                f"ExecutionManifest.stage={execution_manifest.stage.value!r} is terminal and a fold plan was "
                "built, but no EXECUTION_SUMMARY artifact reference is recorded",
            ))
        return None, issues

    try:
        raw = artifact_store.read_artifact(summary_ref.content_hash)
        aggregate = AggregatedExecutionResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "aggregate_unverifiable",
            f"EXECUTION_SUMMARY artifact could not be read and decoded as an AggregatedExecutionResult: {exc}",
        ))
        return None, issues

    if aggregate.experiment_id != execution_manifest.experiment_id:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "aggregate_experiment_id_mismatch",
            f"Aggregate.experiment_id={aggregate.experiment_id!r} does not match "
            f"ExecutionManifest.experiment_id={execution_manifest.experiment_id!r} -- a valid content hash "
            "proves the bytes are intact, not that this is genuinely THIS execution's own summary",
        ))
    if aggregate.completed_fold_indices != execution_manifest.completed_fold_indices:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "aggregate_completed_indices_mismatch",
            f"Aggregate.completed_fold_indices={list(aggregate.completed_fold_indices)} does not match "
            f"ExecutionManifest.completed_fold_indices={list(execution_manifest.completed_fold_indices)}",
        ))
    if aggregate.failed_fold_indices != execution_manifest.failed_fold_indices:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "aggregate_failed_indices_mismatch",
            f"Aggregate.failed_fold_indices={list(aggregate.failed_fold_indices)} does not match "
            f"ExecutionManifest.failed_fold_indices={list(execution_manifest.failed_fold_indices)}",
        ))
    if execution_manifest.total_folds is not None and aggregate.total_folds != execution_manifest.total_folds:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "aggregate_total_folds_mismatch",
            f"Aggregate.total_folds={aggregate.total_folds} does not match "
            f"ExecutionManifest.total_folds={execution_manifest.total_folds}",
        ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "aggregate_verified", "EXECUTION_SUMMARY artifact verified and consistent with the execution manifest"))
    return aggregate, issues


def _verify_timeline(execution_manifest: ExecutionManifest, *, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    timeline_ref = next((r for r in execution_manifest.artifact_references if r.category is ArtifactCategory.TIMELINE), None)
    if timeline_ref is None:
        if execution_manifest.stage in _TERMINAL_STAGE_TO_RUN_EVENT and _plan_was_built(execution_manifest):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "timeline_missing",
                f"ExecutionManifest.stage={execution_manifest.stage.value!r} is terminal and a fold plan was "
                "built, but no TIMELINE artifact reference is recorded",
            ))
        return issues
    try:
        raw = artifact_store.read_artifact(timeline_ref.content_hash)
        Timeline.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "timeline_unverifiable",
            f"TIMELINE artifact could not be read and decoded as a Timeline: {exc}",
        ))
        return issues
    issues.append(_issue(ValidationSeverity.INFO, "timeline_verified", "TIMELINE artifact verified and decodes cleanly"))
    return issues


def _verify_terminal_stage_matches_aggregate(
    execution_manifest: ExecutionManifest, aggregate: AggregatedExecutionResult | None,
) -> list[ValidationIssue]:
    if aggregate is None or execution_manifest.stage not in _TERMINAL_STAGE_TO_RUN_EVENT:
        return []
    if aggregate.overall_status is not execution_manifest.stage:
        return [_issue(
            ValidationSeverity.CRITICAL, "terminal_stage_aggregate_outcome_mismatch",
            f"ExecutionManifest.stage={execution_manifest.stage.value!r} does not match "
            f"Aggregate.overall_status={aggregate.overall_status.value!r}",
        )]
    return [_issue(
        ValidationSeverity.INFO, "terminal_stage_compatible_with_aggregate",
        f"ExecutionManifest.stage matches Aggregate.overall_status ({execution_manifest.stage.value!r})",
    )]


def _verify_experiment_manifest_compatibility(
    execution_manifest: ExecutionManifest, experiment_manifest: ExperimentManifest,
) -> list[ValidationIssue]:
    expected = _TERMINAL_STAGE_TO_EXPERIMENT_STATUS.get(execution_manifest.stage, ExperimentStatus.RUNNING)
    if experiment_manifest.status is not expected:
        return [_issue(
            ValidationSeverity.CRITICAL, "experiment_status_execution_stage_incompatible",
            f"ExecutionManifest.stage={execution_manifest.stage.value!r} expects "
            f"ExperimentManifest.status={expected.value!r}, but it is {experiment_manifest.status.value!r}",
        )]
    return [_issue(
        ValidationSeverity.INFO, "experiment_status_compatible",
        f"ExperimentManifest.status={experiment_manifest.status.value!r} is compatible with "
        f"ExecutionManifest.stage={execution_manifest.stage.value!r}",
    )]


def _verify_events(execution_manifest: ExecutionManifest, *, event_store: ExperimentEventStore) -> list[ValidationIssue]:
    try:
        events = event_store.read_events(execution_manifest.experiment_id)
    except ArtifactCorruptionError as exc:
        return [_issue(ValidationSeverity.CRITICAL, "event_log_corrupted", f"Event log could not be read: {exc}")]
    issues: list[ValidationIssue] = []
    issues += _verify_no_impossible_event_transitions(events)
    issues += _verify_manifest_not_behind_latest_event(execution_manifest, events)
    return issues


def _event_fold_index(event: EventRecord) -> int | None:
    raw = event.details.get("fold_index")
    return None if raw is None else int(str(raw))


def _verify_no_impossible_event_transitions(events: tuple[EventRecord, ...]) -> list[ValidationIssue]:
    """A concrete, checkable subset of "no impossible transitions": at
    most one `RUN_STARTED` per experiment; `RUN_COMPLETED`/`RUN_FAILED`
    (if present) is always the LAST event (a run reaching a terminal
    outcome is never legitimately followed by anything else); every
    `FOLD_COMPLETED`/`FOLD_FAILED` has an earlier `FOLD_STARTED` for the
    SAME fold_index; `EXECUTION_RESUMED` never precedes the first
    `RUN_STARTED`."""
    issues: list[ValidationIssue] = []

    run_started = [e for e in events if e.event_type is EventType.RUN_STARTED]
    if len(run_started) > 1:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "multiple_run_started_events",
            f"{len(run_started)} RUN_STARTED events recorded for one experiment_id -- expected at most 1",
            sequences=", ".join(str(e.sequence) for e in run_started),
        ))

    for index, event in enumerate(events):
        if event.event_type in _TERMINAL_STAGE_TO_RUN_EVENT.values() and index != len(events) - 1:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_after_terminal_run_event",
                f"{event.event_type.value!r} at sequence={event.sequence} is not the last event in the log "
                f"({len(events)} total) -- a run reaching a terminal outcome cannot be legitimately followed "
                "by further events",
                sequence=event.sequence, event_type=event.event_type.value,
            ))

    started_folds: set[int] = set()
    for event in events:
        if event.event_type is EventType.FOLD_STARTED:
            fold_index = _event_fold_index(event)
            if fold_index is not None:
                started_folds.add(fold_index)
        elif event.event_type in (EventType.FOLD_COMPLETED, EventType.FOLD_FAILED):
            fold_index = _event_fold_index(event)
            if fold_index is not None and fold_index not in started_folds:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "fold_completion_event_without_start",
                    f"{event.event_type.value!r} for fold_index={fold_index} at sequence={event.sequence} has "
                    "no earlier FOLD_STARTED event for the same fold_index",
                    fold_index=fold_index, sequence=event.sequence,
                ))

    first_run_started_sequence = run_started[0].sequence if run_started else None
    for event in events:
        if event.event_type is EventType.EXECUTION_RESUMED and (
            first_run_started_sequence is None or event.sequence < first_run_started_sequence
        ):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "execution_resumed_before_run_started",
                f"EXECUTION_RESUMED at sequence={event.sequence} precedes the first RUN_STARTED event "
                "(or none exists) -- a resume presupposes a run was already started",
                sequence=event.sequence,
            ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "event_sequence_has_no_impossible_transitions",
            f"All {len(events)} event(s) form a plausible sequence: at most one RUN_STARTED, no event follows "
            "a terminal RUN_COMPLETED/RUN_FAILED, every fold completion has a matching start, and no resume "
            "precedes its run's start",
        ))
    return issues


def _verify_manifest_not_behind_latest_event(
    execution_manifest: ExecutionManifest, events: tuple[EventRecord, ...],
) -> list[ValidationIssue]:
    """The ONE crash window this codebase's manifest-write-BEFORE-event-
    append ordering can actually leave behind (see module docstring): a
    terminal `ExecutionManifest` whose closing `RUN_COMPLETED`/
    `RUN_FAILED` event never got appended. Reported as a WARNING --
    recoverable incompleteness, not corruption; `ValidationReport.
    is_ready` stays true, but the gap is never silently dropped."""
    expected_event = _TERMINAL_STAGE_TO_RUN_EVENT.get(execution_manifest.stage)
    if expected_event is None:
        return []  # not a terminal stage this check applies to (includes CANCELLED: no event type exists for it yet)
    last_event = events[-1] if events else None
    if last_event is None or last_event.event_type is not expected_event:
        observed = "no events at all" if last_event is None else f"{last_event.event_type.value!r}"
        return [_issue(
            ValidationSeverity.WARNING, "terminal_manifest_missing_terminal_event",
            f"ExecutionManifest.stage={execution_manifest.stage.value!r} is terminal, expecting the event "
            f"log's last event to be {expected_event.value!r}, but observed {observed}. Consistent with a "
            "process crash between the terminal manifest write and its describing event append -- the "
            "manifest remains authoritative; only the event log's history of this run is incomplete.",
            expected_event=expected_event.value,
        )]
    return [_issue(
        ValidationSeverity.INFO, "terminal_event_present",
        f"Event log's last event is the expected {expected_event.value!r}, matching the terminal "
        f"ExecutionManifest.stage={execution_manifest.stage.value!r}",
    )]


__all__ = ["verify_execution"]
