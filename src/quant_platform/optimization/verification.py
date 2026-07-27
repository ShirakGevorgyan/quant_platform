"""`verify_optimization`: an independent, read-only re-audit of everything
an optimization has recorded -- its own `OptimizationManifest`, the parent
`ExperimentManifest`, every trial/feature-selection/outer-fold artifact,
and the append-only event log -- proving they still agree with each
other. Mirrors `execution.verification.verify_execution`'s exact
philosophy and structure, extended for this milestone's own specific
leakage-safety concerns (outer-test isolation, sampled-parameter
validity, ranking reproducibility).

WHY THIS EXISTS, SEPARATELY FROM `optimization.resume`
--------------------------------------------------------------------------
`optimization.resume` answers a NARROW, forward-looking question for one
still-resumable optimization: "which claimed-complete trials/outer folds
can I trust well enough to skip re-running?" This module answers a
BROADER, backward-looking question for ANY optimization (running,
terminal, or long finished): "is everything this optimization ever wrote
still mutually consistent, and does it actually prove outer-test isolation
was respected?" Reports warnings separately from fatal (CRITICAL/ERROR)
consistency errors -- `ValidationReport.is_ready` is the single
authoritative "did this pass" gate; nothing here raises for an
inconsistent-but-loadable optimization.
"""

from __future__ import annotations

from collections.abc import Mapping

from quant_platform.core.exceptions import ArtifactCorruptionError, ArtifactNotFoundError, SchemaVersionError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
    ExperimentStatus,
    JsonPrimitive,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_json_strict, utc_now
from quant_platform.optimization.candidates import TrialResult, rank_trials
from quant_platform.optimization.feature_selection import (
    FeatureSelectionResult,
    FeatureUniverse,
    validate_feature_selection_result,
)
from quant_platform.optimization.manifests import (
    OptimizationEventStore,
    OptimizationEventType,
    OptimizationManifest,
    OptimizationManifestStore,
)
from quant_platform.optimization.models import (
    OptimizationSpec,
    OptimizationStage,
    compute_optimization_identity,
)
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.search_space import validate_sampled_values

_SCHEMA_VERSION = 1

_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError,
)

_TERMINAL_STAGE_TO_RUN_EVENT: dict[OptimizationStage, OptimizationEventType] = {
    OptimizationStage.COMPLETED: OptimizationEventType.RUN_COMPLETED,
    OptimizationStage.FAILED: OptimizationEventType.RUN_FAILED,
}
_TRIAL_OUTCOME_EVENTS = frozenset({
    OptimizationEventType.TRIAL_COMPLETED, OptimizationEventType.TRIAL_FAILED,
    OptimizationEventType.TRIAL_INVALID, OptimizationEventType.TRIAL_PRUNED,
})


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def verify_optimization(
    optimization_id: str,
    *,
    optimization_manifest_store: OptimizationManifestStore,
    experiment_manifest_store: ExperimentManifestStore,
    artifact_store: MLArtifactStore,
    event_store: OptimizationEventStore,
) -> ValidationReport:
    manifest = optimization_manifest_store.load(optimization_id)

    issues: list[ValidationIssue] = []
    spec, spec_issues = _verify_optimization_spec(manifest, artifact_store=artifact_store)
    issues += spec_issues

    feature_universe: FeatureUniverse | None = None
    if spec is not None:
        experiment_issues, feature_universe = _verify_parent_experiment(spec, manifest, experiment_manifest_store=experiment_manifest_store)
        issues += experiment_issues

    trials_by_outer_fold: dict[int, list[TrialResult]] = {}
    if spec is not None:
        trial_issues, trials_by_outer_fold = _verify_trial_results(manifest, spec, artifact_store=artifact_store)
        issues += trial_issues
        if feature_universe is not None:
            issues += _verify_feature_selection_results(trials_by_outer_fold, feature_universe, artifact_store=artifact_store)
        issues += _verify_ranking_reproducibility(manifest, trials_by_outer_fold, spec)

    issues += _verify_outer_fold_results(manifest, artifact_store=artifact_store)
    issues += _verify_summary_references(manifest, artifact_store=artifact_store)
    issues += _verify_manifest_stage_consistency(manifest)
    issues += _verify_events(manifest, event_store=event_store)

    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _verify_optimization_spec(manifest: OptimizationManifest, *, artifact_store: MLArtifactStore) -> tuple[OptimizationSpec | None, list[ValidationIssue]]:
    reference = next((r for r in manifest.artifact_references if r.category is ArtifactCategory.OPTIMIZATION_SPEC), None)
    if reference is None:
        return None, [_issue(ValidationSeverity.CRITICAL, "optimization_spec_missing", "No OPTIMIZATION_SPEC artifact reference is recorded on the manifest")]
    try:
        raw = artifact_store.read_artifact(reference.content_hash)
        spec = OptimizationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return None, [_issue(ValidationSeverity.CRITICAL, "optimization_spec_unverifiable", f"OPTIMIZATION_SPEC artifact could not be read and decoded: {exc}")]

    identity = compute_optimization_identity(spec)
    if identity.optimization_id != manifest.optimization_id:
        return spec, [_issue(
            ValidationSeverity.CRITICAL, "optimization_identity_mismatch",
            f"Recomputed optimization_id={identity.optimization_id!r} does not match "
            f"OptimizationManifest.optimization_id={manifest.optimization_id!r} -- the stored spec does not "
            "produce the identity it is filed under",
        )]
    if spec.parent_experiment_id != manifest.parent_experiment_id:
        return spec, [_issue(
            ValidationSeverity.CRITICAL, "parent_experiment_id_mismatch",
            f"OptimizationSpec.parent_experiment_id={spec.parent_experiment_id!r} does not match "
            f"OptimizationManifest.parent_experiment_id={manifest.parent_experiment_id!r}",
        )]
    return spec, [_issue(ValidationSeverity.INFO, "optimization_spec_verified", "OPTIMIZATION_SPEC artifact verified and its recomputed identity matches the manifest")]


def _verify_parent_experiment(
    spec: OptimizationSpec, manifest: OptimizationManifest, *, experiment_manifest_store: ExperimentManifestStore,
) -> tuple[list[ValidationIssue], FeatureUniverse | None]:
    try:
        experiment_manifest = experiment_manifest_store.load(manifest.parent_experiment_id)
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(ValidationSeverity.CRITICAL, "parent_experiment_unverifiable", f"Parent ExperimentManifest could not be loaded: {exc}")], None

    issues: list[ValidationIssue] = []
    if experiment_manifest.spec.dataset_binding != spec.dataset_binding:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "dataset_binding_mismatch",
            "OptimizationSpec.dataset_binding does not match the parent ExperimentSpec.dataset_binding",
        ))
    if experiment_manifest.spec.split_binding != spec.outer_split_binding:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "outer_split_binding_mismatch",
            "OptimizationSpec.outer_split_binding does not match the parent ExperimentSpec.split_binding",
        ))
    if experiment_manifest.status is ExperimentStatus.CANCELLED:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "parent_experiment_cancelled",
            "Parent experiment status is CANCELLED -- an optimization must not be built against a cancelled experiment",
        ))
    elif experiment_manifest.status is ExperimentStatus.FAILED:
        issues.append(_issue(
            ValidationSeverity.WARNING, "parent_experiment_failed",
            "Parent experiment status is FAILED -- its own execution did not complete successfully, though its "
            "dataset/feature/split bindings may still be independently valid for optimization",
        ))

    feature_universe = FeatureUniverse.from_experiment_spec(experiment_manifest.spec)
    if feature_universe.fingerprint != spec.feature_universe_fingerprint:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "feature_universe_fingerprint_mismatch",
            f"OptimizationSpec.feature_universe_fingerprint={spec.feature_universe_fingerprint!r} does not match "
            f"the fingerprint recomputed from the parent experiment's current feature binding "
            f"({feature_universe.fingerprint!r})",
        ))
    if not any(i.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR) for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "parent_experiment_verified", "Parent experiment bindings and feature universe fingerprint are consistent"))
    return issues, feature_universe


def _verify_trial_results(
    manifest: OptimizationManifest, spec: OptimizationSpec, *, artifact_store: MLArtifactStore,
) -> tuple[list[ValidationIssue], dict[int, list[TrialResult]]]:
    issues: list[ValidationIssue] = []
    trials_by_outer_fold: dict[int, list[TrialResult]] = {}
    checked = 0
    for key, reference in manifest.trial_result_references.items():
        outer_fold_index_str, trial_number_str = key.split(":", 1)
        outer_fold_index, trial_number = int(outer_fold_index_str), int(trial_number_str)
        if reference.category is not ArtifactCategory.TRIAL_RESULT:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "trial_result_wrong_category",
                f"Trial (outer_fold={outer_fold_index}, trial={trial_number}): artifact reference has category="
                f"{reference.category.value!r}, expected {ArtifactCategory.TRIAL_RESULT.value!r}",
                outer_fold_index=outer_fold_index, trial_number=trial_number,
            ))
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = TrialResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "trial_result_unverifiable",
                f"Trial (outer_fold={outer_fold_index}, trial={trial_number}): artifact could not be read and "
                f"decoded as a TrialResult: {exc}", outer_fold_index=outer_fold_index, trial_number=trial_number,
            ))
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.trial_number != trial_number or decoded.optimization_id != manifest.optimization_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "trial_result_key_mismatch",
                f"Trial filed under (outer_fold={outer_fold_index}, trial={trial_number}) decodes with "
                f"outer_fold_index={decoded.outer_fold_index}, trial_number={decoded.trial_number}, "
                f"optimization_id={decoded.optimization_id!r} -- a valid content hash proves the bytes are "
                "intact, not that they were filed under the correct key",
                outer_fold_index=outer_fold_index, trial_number=trial_number,
            ))
            continue
        try:
            validate_sampled_values(spec.search_space, dict(decoded.sampled_hyperparameters))
        except ValueError as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "trial_sampled_values_invalid",
                f"Trial (outer_fold={outer_fold_index}, trial={trial_number}): sampled hyperparameters do not "
                f"satisfy OptimizationSpec.search_space: {exc}", outer_fold_index=outer_fold_index, trial_number=trial_number,
            ))
            continue
        trials_by_outer_fold.setdefault(outer_fold_index, []).append(decoded)
        checked += 1

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "all_trial_results_verified", f"All {checked} recorded trial result artifact(s) verified"))
    return issues, trials_by_outer_fold


def _verify_feature_selection_results(
    trials_by_outer_fold: Mapping[int, list[TrialResult]], feature_universe: FeatureUniverse,
    *, artifact_store: MLArtifactStore,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    checked = 0
    for outer_fold_index, trials in trials_by_outer_fold.items():
        for trial in trials:
            for inner_metric in trial.inner_fold_metrics:
                reference = inner_metric.feature_selection_result_reference
                if reference is None:
                    continue
                if reference.category is not ArtifactCategory.FEATURE_SELECTION_RESULT:
                    issues.append(_issue(
                        ValidationSeverity.CRITICAL, "feature_selection_result_wrong_category",
                        f"Trial (outer_fold={outer_fold_index}, trial={trial.trial_number}, inner_fold="
                        f"{inner_metric.inner_fold_index}): FeatureSelectionResult reference has wrong category",
                        outer_fold_index=outer_fold_index, trial_number=trial.trial_number,
                    ))
                    continue
                try:
                    raw = artifact_store.read_artifact(reference.content_hash)
                    decoded = FeatureSelectionResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
                    validate_feature_selection_result(decoded, feature_universe)
                except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                    issues.append(_issue(
                        ValidationSeverity.CRITICAL, "feature_selection_result_unverifiable",
                        f"Trial (outer_fold={outer_fold_index}, trial={trial.trial_number}, inner_fold="
                        f"{inner_metric.inner_fold_index}): FeatureSelectionResult could not be verified: {exc}",
                        outer_fold_index=outer_fold_index, trial_number=trial.trial_number,
                    ))
                    continue
                checked += 1
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "all_feature_selection_results_verified", f"All {checked} recorded FeatureSelectionResult artifact(s) verified against the feature universe"))
    return issues


def _verify_ranking_reproducibility(
    manifest: OptimizationManifest, trials_by_outer_fold: Mapping[int, list[TrialResult]], spec: OptimizationSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for outer_fold_index, trials in sorted(trials_by_outer_fold.items()):
        table = rank_trials(trials, primary_metric=spec.primary_metric)
        recorded_winner = manifest.winning_trial_by_outer_fold.get(outer_fold_index)
        if recorded_winner is None:
            continue
        recomputed_winner = table.winner
        if recomputed_winner is None or recomputed_winner.trial_number != recorded_winner:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "ranking_not_reproducible",
                f"Outer fold {outer_fold_index}: recomputing the ranking policy over every verified trial "
                f"produces winner={(recomputed_winner.trial_number if recomputed_winner else None)!r}, but the "
                f"manifest records winning_trial_by_outer_fold[{outer_fold_index}]={recorded_winner!r}",
                outer_fold_index=outer_fold_index,
            ))
        else:
            issues.append(_issue(ValidationSeverity.INFO, "ranking_reproducible", f"Outer fold {outer_fold_index}: recorded winner matches the recomputed ranking policy", outer_fold_index=outer_fold_index))
    return issues


def _verify_outer_fold_results(manifest: OptimizationManifest, *, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    checked = 0
    for outer_fold_index, reference in manifest.outer_fold_result_references.items():
        if reference.category is not ArtifactCategory.OUTER_FOLD_SELECTION:
            issues.append(_issue(ValidationSeverity.CRITICAL, "outer_fold_result_wrong_category", f"Outer fold {outer_fold_index}: OuterFoldResult reference has wrong category", outer_fold_index=outer_fold_index))
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "outer_fold_result_unverifiable", f"Outer fold {outer_fold_index}: OuterFoldResult could not be read and decoded: {exc}", outer_fold_index=outer_fold_index))
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.optimization_id != manifest.optimization_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "outer_fold_result_key_mismatch", f"Outer fold {outer_fold_index}: decoded OuterFoldResult does not match its own filing key", outer_fold_index=outer_fold_index))
            continue
        recorded_winner = manifest.winning_trial_by_outer_fold.get(outer_fold_index)
        if recorded_winner is not None and decoded.winning_trial_number != recorded_winner:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_winner_mismatch",
                f"Outer fold {outer_fold_index}: OuterFoldResult.winning_trial_number={decoded.winning_trial_number} "
                f"does not match manifest.winning_trial_by_outer_fold[{outer_fold_index}]={recorded_winner}",
                outer_fold_index=outer_fold_index,
            ))
        for ref_name, ref in (
            ("model_reference", decoded.model_reference), ("predictions_reference", decoded.predictions_reference),
            ("feature_selection_result_reference", decoded.feature_selection_result_reference),
        ):
            try:
                artifact_store.read_artifact(ref.content_hash)
            except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "outer_fold_result_dependent_artifact_unverifiable",
                    f"Outer fold {outer_fold_index}: {ref_name} could not be verified: {exc}", outer_fold_index=outer_fold_index,
                ))
        checked += 1
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "all_outer_fold_results_verified", f"All {checked} recorded OuterFoldResult artifact(s) verified"))
    return issues


def _verify_summary_references(manifest: OptimizationManifest, *, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for reference in manifest.summary_references:
        try:
            artifact_store.read_artifact(reference.content_hash)
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "summary_reference_unverifiable", f"Summary artifact (category={reference.category.value!r}) could not be verified: {exc}"))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "summary_references_verified", f"All {len(manifest.summary_references)} recorded summary artifact reference(s) verified"))
    return issues


def _verify_manifest_stage_consistency(manifest: OptimizationManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.stage is OptimizationStage.COMPLETED and (
        manifest.total_outer_folds is None or len(manifest.completed_outer_fold_indices) != manifest.total_outer_folds
    ):
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "completed_stage_outer_fold_count_mismatch",
            f"stage=COMPLETED but completed_outer_fold_indices has {len(manifest.completed_outer_fold_indices)} "
            f"entries, expected total_outer_folds={manifest.total_outer_folds!r}",
        ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "manifest_stage_consistent", f"OptimizationManifest.stage={manifest.stage.value!r} is consistent with its own recorded progress fields"))
    return issues


def _event_int_detail(event_details: Mapping[str, JsonPrimitive], key: str) -> int | None:
    raw = event_details.get(key)
    return None if raw is None else int(str(raw))


def _verify_events(manifest: OptimizationManifest, *, event_store: OptimizationEventStore) -> list[ValidationIssue]:
    try:
        events = event_store.read_events(manifest.optimization_id)
    except ArtifactCorruptionError as exc:
        return [_issue(ValidationSeverity.CRITICAL, "event_log_corrupted", f"Event log could not be read: {exc}")]

    issues: list[ValidationIssue] = []
    run_started = [e for e in events if e.event_type is OptimizationEventType.RUN_STARTED]
    if len(run_started) > 1:
        issues.append(_issue(ValidationSeverity.CRITICAL, "multiple_run_started_events", f"{len(run_started)} RUN_STARTED events recorded -- expected at most 1"))

    for index, event in enumerate(events):
        if event.event_type in _TERMINAL_STAGE_TO_RUN_EVENT.values() and index != len(events) - 1:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_after_terminal_run_event",
                f"{event.event_type.value!r} at sequence={event.sequence} is not the last event ({len(events)} total)",
                sequence=event.sequence,
            ))

    started_trials: set[tuple[int, int]] = set()
    for event in events:
        outer_fold_index = _event_int_detail(event.details, "outer_fold_index")
        trial_number = _event_int_detail(event.details, "trial_number")
        if event.event_type is OptimizationEventType.TRIAL_STARTED and outer_fold_index is not None and trial_number is not None:
            started_trials.add((outer_fold_index, trial_number))
        elif (
            event.event_type in _TRIAL_OUTCOME_EVENTS and outer_fold_index is not None and trial_number is not None
            and (outer_fold_index, trial_number) not in started_trials
        ):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "trial_outcome_event_without_start",
                f"{event.event_type.value!r} for (outer_fold={outer_fold_index}, trial={trial_number}) at "
                f"sequence={event.sequence} has no earlier TRIAL_STARTED for the same trial",
                sequence=event.sequence, outer_fold_index=outer_fold_index, trial_number=trial_number,
            ))

    # THE outer-test-isolation event-ordering check: candidate selection
    # must precede outer-fold finalization (which is when outer-test is
    # ever evaluated) -- for every outer fold that has both events.
    selected_sequence: dict[int, int] = {}
    for event in events:
        if event.event_type is OptimizationEventType.CANDIDATE_SELECTED:
            outer_fold_index = _event_int_detail(event.details, "outer_fold_index")
            if outer_fold_index is not None:
                selected_sequence[outer_fold_index] = event.sequence
    for event in events:
        if event.event_type is OptimizationEventType.OUTER_FOLD_FINALIZED:
            outer_fold_index = _event_int_detail(event.details, "outer_fold_index")
            if outer_fold_index is None:
                continue
            select_seq = selected_sequence.get(outer_fold_index)
            if select_seq is None or select_seq >= event.sequence:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "outer_test_evaluated_before_candidate_selection",
                    f"Outer fold {outer_fold_index}: OUTER_FOLD_FINALIZED (sequence={event.sequence}, this is "
                    "when outer-test is evaluated) does not have an earlier CANDIDATE_SELECTED event -- candidate "
                    "selection must complete strictly before outer-test is ever observed",
                    outer_fold_index=outer_fold_index, sequence=event.sequence,
                ))

    first_run_started_sequence = run_started[0].sequence if run_started else None
    for event in events:
        if event.event_type is OptimizationEventType.OPTIMIZATION_RESUMED and (
            first_run_started_sequence is None or event.sequence < first_run_started_sequence
        ):
            issues.append(_issue(ValidationSeverity.CRITICAL, "resumed_before_run_started", f"OPTIMIZATION_RESUMED at sequence={event.sequence} precedes the first RUN_STARTED event", sequence=event.sequence))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "event_sequence_consistent",
            f"All {len(events)} event(s) form a plausible sequence, including outer-test isolation (candidate "
            "selection always precedes outer-fold finalization for every outer fold with both events)",
        ))
    return issues


__all__ = ["verify_optimization"]
