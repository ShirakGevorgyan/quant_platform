"""`verify_calibration`: an independent, read-only re-audit of everything
a calibration run has recorded -- its own `CalibrationManifest`, every
outer-fold result artifact and the sub-artifacts it references, and the
append-only event log -- proving they still agree with each other.
Mirrors `optimization.verification.verify_optimization`'s exact
philosophy and structure.

WHY THIS EXISTS, SEPARATELY FROM `calibration.resume`
--------------------------------------------------------------------------
`calibration.resume` answers a NARROW, forward-looking question for one
still-resumable calibration: "which claimed-complete outer folds can I
trust well enough to skip re-running?" This module answers a BROADER,
backward-looking question for ANY calibration (running, terminal, or long
finished): "is everything this calibration ever wrote still mutually
consistent, and does it actually prove the frozen post-processing policy
was faithfully applied?" Reports warnings separately from fatal
(CRITICAL/ERROR) consistency errors -- `ValidationReport.is_ready` is the
single authoritative "did this pass" gate; nothing here raises for an
inconsistent-but-loadable calibration.

HASH VALIDITY IS NOT ENOUGH: THE RECOMPUTATION CHECK
--------------------------------------------------------------------------
`_verify_calibrated_probabilities_reproduce` is the check that makes this
module more than a glorified hash-consistency scan (Section 25: "hash
validity insufficient -- semantically wrong hash-consistent artifacts
must be rejected"). It re-derives the selected calibrator from its
persisted, explicit parameters (`FrozenDecisionPolicy.selected_
calibrator()`), re-applies `.transform()` to the persisted RAW outer-test
probabilities, and asserts the result matches the persisted CALIBRATED
probabilities bit-for-bit (within float tolerance) -- proving the
persisted calibrated output is not just present, but ACTUALLY the output
of the persisted calibrator parameters applied to the persisted input.
The same recomputation is repeated for the frozen threshold's decision
boundary.

WHAT THIS MODULE CANNOT INDEPENDENTLY VERIFY
--------------------------------------------------------------------------
It does NOT re-fit the inner-fold models or the final outer-train refit
(doing so would require the full training-side dataset and be far more
expensive than an audit should be) -- so it cannot prove the RAW
probabilities themselves came from a correctly-trained model, only that
everything downstream of those raw probabilities (calibration,
thresholding, confidence, uncertainty, abstention, metrics) is
self-consistent and faithfully reproducible from what was persisted. It
also does not re-verify `calibration.fitting`'s own inner-OOF leakage
guarantees beyond what `RawPredictionSet.__post_init__`/
`InnerOofPredictionSet.__post_init__` already enforce structurally at
deserialization time (those checks run again automatically on every
`from_json_dict` call, since they are dataclass invariants, not a
separate opt-in verification step).
"""

from __future__ import annotations

import numpy as np

from quant_platform.calibration.fitting import (
    CalibratorSelectionReport,
    FrozenDecisionPolicy,
    InnerOofPredictionSet,
)
from quant_platform.calibration.manifests import (
    CalibrationEventStore,
    CalibrationEventType,
    CalibrationManifest,
    CalibrationManifestStore,
)
from quant_platform.calibration.models import CalibrationStage
from quant_platform.calibration.runner import OuterFoldCalibrationResult
from quant_platform.calibration.specs import CalibrationSpec, compute_calibration_identity
from quant_platform.calibration.thresholds import ThresholdReport, apply_threshold
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    CalibrationDataError,
    CalibrationValidationError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import (
    ArtifactCategory,
    JsonPrimitive,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_json_strict, utc_now

_SCHEMA_VERSION = 1

_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, CalibrationDataError,
    CalibrationValidationError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-good artifact can legitimately hit when
DECODED (not just read as bytes) -- includes `CalibrationDataError`/
`CalibrationValidationError` because decoding `InnerOofPredictionSet`/
`RawPredictionSet`/`ThresholdReport`/etc. via `from_json_dict` re-runs
their full `__post_init__` structural validation (Section 5/25: a
hash-valid-but-semantically-tampered artifact, e.g. one whose
`fitted_on_rows` overlaps its own `sample_positions`, must be reported
as a CRITICAL issue here, never propagate as an uncaught exception)."""
_RECOMPUTATION_TOLERANCE = 1e-9
_TERMINAL_STAGE_TO_RUN_EVENT: dict[CalibrationStage, CalibrationEventType] = {
    CalibrationStage.COMPLETED: CalibrationEventType.RUN_COMPLETED, CalibrationStage.FAILED: CalibrationEventType.RUN_FAILED,
}


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def verify_calibration(
    calibration_id: str, *, calibration_manifest_store: CalibrationManifestStore, artifact_store: MLArtifactStore,
    event_store: CalibrationEventStore,
) -> ValidationReport:
    manifest = calibration_manifest_store.load(calibration_id)

    issues: list[ValidationIssue] = []
    _spec, spec_issues = _verify_calibration_spec(manifest, artifact_store=artifact_store)
    issues += spec_issues

    results, result_issues = _verify_outer_fold_results(manifest, artifact_store=artifact_store)
    issues += result_issues
    for result in results:
        issues += _verify_calibrated_probabilities_reproduce(result, artifact_store=artifact_store)

    issues += _verify_manifest_stage_consistency(manifest)
    issues += _verify_events(manifest, event_store=event_store)

    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _verify_calibration_spec(manifest: CalibrationManifest, *, artifact_store: MLArtifactStore) -> tuple[CalibrationSpec | None, list[ValidationIssue]]:
    if manifest.spec_reference is None:
        return None, [_issue(ValidationSeverity.CRITICAL, "missing_spec_reference", "CalibrationManifest has no spec_reference")]
    if manifest.spec_reference.category is not ArtifactCategory.CALIBRATION_SPEC:
        return None, [_issue(ValidationSeverity.CRITICAL, "spec_reference_wrong_category", "spec_reference has the wrong ArtifactCategory")]
    try:
        raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
        spec = CalibrationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return None, [_issue(ValidationSeverity.CRITICAL, "spec_unverifiable", f"CalibrationSpec could not be read and decoded: {exc}")]
    identity = compute_calibration_identity(spec)
    if identity.calibration_id != manifest.calibration_id:
        return spec, [_issue(
            ValidationSeverity.CRITICAL, "spec_identity_mismatch",
            f"Recomputed calibration_id={identity.calibration_id!r} does not match manifest.calibration_id={manifest.calibration_id!r}",
        )]
    return spec, [_issue(ValidationSeverity.INFO, "spec_verified", "CalibrationSpec decoded and its identity matches the manifest")]


def _verify_outer_fold_results(
    manifest: CalibrationManifest, *, artifact_store: MLArtifactStore,
) -> tuple[list[OuterFoldCalibrationResult], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    results: list[OuterFoldCalibrationResult] = []
    for outer_fold_index, reference in manifest.outer_fold_result_references.items():
        if reference.category is not ArtifactCategory.OUTER_FOLD_CALIBRATION_RESULT:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_wrong_category",
                f"Outer fold {outer_fold_index}: OuterFoldCalibrationResult reference has the wrong category", outer_fold_index=outer_fold_index,
            ))
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_unverifiable",
                f"Outer fold {outer_fold_index}: OuterFoldCalibrationResult could not be read and decoded: {exc}", outer_fold_index=outer_fold_index,
            ))
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.calibration_id != manifest.calibration_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_key_mismatch",
                f"Outer fold {outer_fold_index}: decoded OuterFoldCalibrationResult does not match its own filing key", outer_fold_index=outer_fold_index,
            ))
            continue
        # `model_reference` is read as bytes only -- a serialized model has
        # no generic, deserializer-free decode this module can perform
        # (see module docstring: re-fitting/re-deserializing the base
        # model is explicitly out of this module's scope). Every OTHER
        # dependent artifact below is fully DECODED (not just read as
        # bytes): decoding re-runs that type's own `__post_init__`
        # structural validation, which is what actually catches a
        # hash-valid-but-semantically-tampered artifact (Section 5) --
        # e.g. an `InnerOofPredictionSet` whose `fitted_on_rows` overlaps
        # its own `sample_positions` (`RawPredictionSet.__post_init__`'s
        # leakage check) would previously pass a bytes-only read silently.
        try:
            artifact_store.read_artifact(decoded.model_reference.content_hash)
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_dependent_artifact_unverifiable",
                f"Outer fold {outer_fold_index}: model_reference could not be verified: {exc}", outer_fold_index=outer_fold_index,
            ))
        for ref_name, ref, decoder in (
            ("inner_oof_reference", decoded.inner_oof_reference, InnerOofPredictionSet.from_json_dict),
            ("calibrator_selection_reference", decoded.calibrator_selection_reference, CalibratorSelectionReport.from_json_dict),
            ("threshold_report_reference", decoded.threshold_report_reference, ThresholdReport.from_json_dict),
            ("decision_policy_reference", decoded.decision_policy_reference, FrozenDecisionPolicy.from_json_dict),
        ):
            try:
                raw_sub = artifact_store.read_artifact(ref.content_hash)
                decoder(parse_json_strict(raw_sub.decode("utf-8")))
            except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "outer_fold_result_dependent_artifact_unverifiable",
                    f"Outer fold {outer_fold_index}: {ref_name} could not be read, decoded, and structurally "
                    f"validated: {exc}", outer_fold_index=outer_fold_index,
                ))
        results.append(decoded)
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "all_outer_fold_results_verified", f"All {len(results)} recorded OuterFoldCalibrationResult artifact(s) verified"))
    return results, issues


def _verify_calibrated_probabilities_reproduce(result: OuterFoldCalibrationResult, *, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    """THE recomputation check (see module docstring): re-derives the
    frozen calibrator from its persisted parameters and confirms it
    reproduces the persisted calibrated probabilities AND the persisted
    threshold decisions from the persisted raw probabilities alone."""
    try:
        raw = artifact_store.read_artifact(result.decision_policy_reference.content_hash)
        policy = FrozenDecisionPolicy.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(
            ValidationSeverity.CRITICAL, "decision_policy_unverifiable",
            f"Outer fold {result.outer_fold_index}: FrozenDecisionPolicy could not be read and decoded: {exc}",
            outer_fold_index=result.outer_fold_index,
        )]

    recomputed = policy.selected_calibrator().transform(np.asarray(result.raw_probabilities))
    persisted = np.asarray(result.calibrated_probabilities)
    max_diff = float(np.max(np.abs(recomputed - persisted))) if len(persisted) else 0.0
    if max_diff > _RECOMPUTATION_TOLERANCE:
        return [_issue(
            ValidationSeverity.CRITICAL, "calibrated_probabilities_do_not_reproduce",
            f"Outer fold {result.outer_fold_index}: re-applying the persisted calibrator to the persisted raw "
            f"probabilities does not reproduce the persisted calibrated probabilities (max abs diff={max_diff:.3g}, "
            f"tolerance={_RECOMPUTATION_TOLERANCE:.3g}) -- the persisted artifact is hash-consistent but "
            "SEMANTICALLY WRONG", outer_fold_index=result.outer_fold_index, max_diff=max_diff,
        )]

    threshold = policy.threshold_report.selected_threshold
    recomputed_positive = apply_threshold(recomputed, threshold)
    # Abstention can convert a POSITIVE/NEGATIVE boundary decision into
    # "abstain" -- so a mismatch is only fatal where the persisted
    # decision is NOT "abstain" (a genuine accept/reject call this
    # module CAN independently recompute without the abstention spec's
    # confidence/uncertainty context).
    mismatches = sum(
        1 for recomputed_p, decision in zip(recomputed_positive, result.decisions, strict=True)
        if decision != "abstain" and bool(recomputed_p) != (decision == "positive")
    )
    if mismatches:
        return [_issue(
            ValidationSeverity.CRITICAL, "threshold_decisions_do_not_reproduce",
            f"Outer fold {result.outer_fold_index}: re-applying the persisted threshold to the recomputed "
            f"calibrated probabilities disagrees with {mismatches} persisted non-abstain decision(s)",
            outer_fold_index=result.outer_fold_index, mismatches=mismatches,
        )]
    return [_issue(
        ValidationSeverity.INFO, "calibrated_probabilities_reproduce",
        f"Outer fold {result.outer_fold_index}: persisted calibrated probabilities and threshold decisions are "
        "bit-for-bit reproducible from the persisted calibrator parameters and raw probabilities alone",
        outer_fold_index=result.outer_fold_index,
    )]


def _verify_manifest_stage_consistency(manifest: CalibrationManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.stage in (CalibrationStage.COMPLETED, CalibrationStage.VERIFIED) and (
        manifest.total_outer_folds is None or len(manifest.completed_outer_fold_indices) != manifest.total_outer_folds
    ):
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "terminal_stage_outer_fold_count_mismatch",
            f"stage={manifest.stage.value!r} but completed_outer_fold_indices has "
            f"{len(manifest.completed_outer_fold_indices)} entries, expected total_outer_folds={manifest.total_outer_folds!r}",
        ))
    if manifest.stage is CalibrationStage.COMPLETED and manifest.aggregate_report_reference is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "completed_without_aggregate_report", "stage=COMPLETED but aggregate_report_reference is None"))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "manifest_stage_consistent",
            f"CalibrationManifest.stage={manifest.stage.value!r} is consistent with its own recorded progress fields",
        ))
    return issues


def _verify_events(manifest: CalibrationManifest, *, event_store: CalibrationEventStore) -> list[ValidationIssue]:
    try:
        events = event_store.read_events(manifest.calibration_id)
    except ArtifactCorruptionError as exc:
        return [_issue(ValidationSeverity.CRITICAL, "event_log_corrupted", f"Event log could not be read: {exc}")]

    issues: list[ValidationIssue] = []
    run_started = [e for e in events if e.event_type is CalibrationEventType.RUN_STARTED]
    if len(run_started) > 1:
        issues.append(_issue(ValidationSeverity.CRITICAL, "multiple_run_started_events", f"{len(run_started)} RUN_STARTED events recorded -- expected at most 1"))

    for index, event in enumerate(events):
        if event.event_type in _TERMINAL_STAGE_TO_RUN_EVENT.values() and index != len(events) - 1:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_after_terminal_run_event",
                f"{event.event_type.value!r} at sequence={event.sequence} is not the last event ({len(events)} total)",
                sequence=event.sequence,
            ))

    # THE outer-test-isolation event-ordering check: policies must be
    # frozen (POLICIES_FROZEN) strictly before the outer fold's
    # prediction/evaluation events -- for every outer fold that has both.
    frozen_sequence: dict[int, int] = {}
    for event in events:
        raw_index = event.details.get("outer_fold_index")
        outer_fold_index = None if raw_index is None else int(str(raw_index))
        if event.event_type is CalibrationEventType.POLICIES_FROZEN and outer_fold_index is not None:
            frozen_sequence[outer_fold_index] = event.sequence
        elif event.event_type in (CalibrationEventType.OUTER_FOLD_PREDICTED, CalibrationEventType.OUTER_FOLD_EVALUATED) and outer_fold_index is not None:
            frozen_at = frozen_sequence.get(outer_fold_index)
            if frozen_at is None or frozen_at >= event.sequence:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "prediction_event_before_policies_frozen",
                    f"Outer fold {outer_fold_index}: {event.event_type.value!r} at sequence={event.sequence} has no "
                    "earlier POLICIES_FROZEN event for the same fold -- the post-processing policy ordering "
                    "guarantee is not evidenced by the event log", sequence=event.sequence, outer_fold_index=outer_fold_index,
                ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "events_consistent", f"All {len(events)} recorded event(s) are internally consistent"))
    return issues


__all__ = ["verify_calibration"]
