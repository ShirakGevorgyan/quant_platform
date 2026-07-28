"""Resume support (Milestone 4E) -- discovering which outer folds of an
interrupted calibration run are genuinely, verifiably complete. Mirrors
`optimization.resume`'s "never trust the manifest's claim alone"
philosophy exactly, simplified for calibration's shape: there is no
sequential-sampler ordering dependency between outer folds (each is an
independent, deterministic function of already-fixed inputs -- see
`calibration.models.CalibrationStage`'s own docstring), so a corrupted
outer fold `i` never invalidates outer fold `i+1`'s own, independently
verified result, and there is no `optimization.resume.
build_trial_resume_plan`-style "everything from the first failure onward
is discarded" cascade to reproduce here.
"""

from __future__ import annotations

from quant_platform.calibration.manifests import CalibrationManifest
from quant_platform.calibration.models import (
    _MID_FOLD_CALIBRATION_STAGES,
    CalibrationStage,
    is_terminal_calibration_stage,
)
from quant_platform.calibration.runner import OuterFoldCalibrationResult
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    CalibrationDataError,
    CalibrationResumeError,
    CalibrationValidationError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory
from quant_platform.ml.persistence import parse_json_strict

_UNSAFE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, CalibrationDataError,
    CalibrationValidationError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-complete artifact can legitimately hit --
identical in spirit to `optimization.resume._UNSAFE_DECODE_ERRORS`, plus
`CalibrationDataError`/`CalibrationValidationError`: `OuterFoldCalibrationResult.
from_json_dict` re-runs `__post_init__`'s structural checks (e.g. array-
length-parallel-to-sample_positions), which raise `CalibrationDataError`,
not one of the generic decode errors below -- a tampered-but-otherwise-
well-formed artifact must demote that fold to "needs rerun", never crash
the whole resume attempt. All treated identically: fail closed, demote to
"needs rerun", never propagate raw."""


def can_resume(manifest: CalibrationManifest | None) -> bool:
    if manifest is None:
        return False
    return not is_terminal_calibration_stage(manifest.stage)


def verify_completed_calibration_outer_folds(
    manifest: CalibrationManifest, *, artifact_store: MLArtifactStore,
) -> tuple[frozenset[int], frozenset[int]]:
    """`(verified_complete, needs_rerun)` outer-fold indices -- re-checks
    each of `manifest.completed_outer_fold_indices`'s claimed
    `OUTER_FOLD_CALIBRATION_RESULT` artifact by content hash, category,
    and decoded self-identity."""
    verified: set[int] = set()
    needs_rerun: set[int] = set()
    for outer_fold_index in manifest.completed_outer_fold_indices:
        reference = manifest.outer_fold_result_references.get(outer_fold_index)
        if reference is None or reference.category is not ArtifactCategory.OUTER_FOLD_CALIBRATION_RESULT:
            needs_rerun.add(outer_fold_index)
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNSAFE_DECODE_ERRORS:
            needs_rerun.add(outer_fold_index)
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.calibration_id != manifest.calibration_id:
            needs_rerun.add(outer_fold_index)
            continue
        verified.add(outer_fold_index)
    return frozenset(verified), frozenset(needs_rerun)


def require_calibration_resumable(manifest: CalibrationManifest | None, *, calibration_id: str) -> CalibrationManifest:
    if manifest is None:
        raise CalibrationResumeError(
            f"No calibration manifest exists for calibration_id={calibration_id!r} -- nothing to resume",
            context={"calibration_id": calibration_id},
        )
    if not can_resume(manifest):
        raise CalibrationResumeError(
            f"Calibration {calibration_id!r} already reached a terminal stage {manifest.stage.value!r} -- "
            "it cannot be resumed or restarted in place",
            context={"calibration_id": calibration_id, "stage": manifest.stage.value},
        )
    return manifest


def resolve_resume_start_stage(manifest: CalibrationManifest) -> CalibrationStage:
    """Where `CalibrationRunner._execute_pipeline`'s outer-fold loop
    should treat `manifest.stage` as being, for resume purposes: any
    stage strictly between `INNER_PREDICTIONS_READY` and `EVALUATED`
    (inclusive of `INNER_PREDICTIONS_READY`) collapses to
    `INNER_PREDICTIONS_READY` -- since `run_outer_fold_calibration`
    recomputes that whole span atomically, there is nothing to preserve
    from a partial attempt (see `CalibrationStage`'s own docstring for
    why every one of those stages has a legal edge back to
    `INNER_PREDICTIONS_READY`). `CREATED`/`EVALUATED`/`VERIFIED` are
    returned unchanged."""
    if manifest.stage in _MID_FOLD_CALIBRATION_STAGES:
        return CalibrationStage.INNER_PREDICTIONS_READY
    return manifest.stage


__all__ = [
    "can_resume",
    "require_calibration_resumable",
    "resolve_resume_start_stage",
    "verify_completed_calibration_outer_folds",
]
