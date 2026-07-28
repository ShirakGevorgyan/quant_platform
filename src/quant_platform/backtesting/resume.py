"""Resume support (Milestone 5) -- discovering which outer folds of an
interrupted backtest run are genuinely, verifiably complete. Mirrors
`calibration.resume`'s "never trust the manifest's claim alone"
philosophy exactly: there is no sequential-sampler ordering dependency
between outer folds here either (each is an independent, deterministic
function of already-fixed inputs -- see `backtesting.models.BacktestStage`'s
own docstring), so a corrupted outer fold `i` never invalidates outer
fold `i+1`'s own, independently verified result."""

from __future__ import annotations

from quant_platform.backtesting.manifests import BacktestManifest
from quant_platform.backtesting.models import (
    _MID_FOLD_BACKTEST_STAGES,
    BacktestStage,
    is_terminal_backtest_stage,
)
from quant_platform.backtesting.runner import OuterFoldBacktestResult
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    BacktestResumeError,
    BacktestValidationError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory
from quant_platform.ml.persistence import parse_json_strict

_UNSAFE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, BacktestValidationError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-complete artifact can legitimately hit --
identical in spirit to `calibration.resume._UNSAFE_DECODE_ERRORS`:
`OuterFoldBacktestResult.from_json_dict` re-runs `__post_init__`'s
structural checks, which raise `BacktestValidationError`, not one of the
generic decode errors below -- a tampered-but-otherwise-well-formed
artifact must demote that fold to "needs rerun", never crash the whole
resume attempt."""


def can_resume(manifest: BacktestManifest | None) -> bool:
    if manifest is None:
        return False
    return not is_terminal_backtest_stage(manifest.stage)


def verify_completed_backtest_outer_folds(
    manifest: BacktestManifest, *, artifact_store: MLArtifactStore,
) -> tuple[frozenset[int], frozenset[int]]:
    """`(verified_complete, needs_rerun)` outer-fold indices -- re-checks
    each of `manifest.completed_outer_fold_indices`'s claimed
    `OUTER_FOLD_BACKTEST_RESULT` artifact by content hash, category, and
    decoded self-identity."""
    verified: set[int] = set()
    needs_rerun: set[int] = set()
    for outer_fold_index in manifest.completed_outer_fold_indices:
        reference = manifest.outer_fold_result_references.get(outer_fold_index)
        if reference is None or reference.category is not ArtifactCategory.OUTER_FOLD_BACKTEST_RESULT:
            needs_rerun.add(outer_fold_index)
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNSAFE_DECODE_ERRORS:
            needs_rerun.add(outer_fold_index)
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.backtest_id != manifest.backtest_id:
            needs_rerun.add(outer_fold_index)
            continue
        verified.add(outer_fold_index)
    return frozenset(verified), frozenset(needs_rerun)


def require_backtest_resumable(manifest: BacktestManifest | None, *, backtest_id: str) -> BacktestManifest:
    if manifest is None:
        raise BacktestResumeError(
            f"No backtest manifest exists for backtest_id={backtest_id!r} -- nothing to resume",
            context={"backtest_id": backtest_id},
        )
    if not can_resume(manifest):
        raise BacktestResumeError(
            f"Backtest {backtest_id!r} already reached a terminal stage {manifest.stage.value!r} -- it cannot be "
            "resumed or restarted in place",
            context={"backtest_id": backtest_id, "stage": manifest.stage.value},
        )
    return manifest


def resolve_resume_start_stage(manifest: BacktestManifest) -> BacktestStage:
    """Where `BacktestRunner._execute_pipeline`'s outer-fold loop should
    treat `manifest.stage` as being, for resume purposes: any stage
    strictly between `SOURCES_VERIFIED` and `REPORTS_READY` (inclusive of
    `SOURCES_VERIFIED`) collapses to `SOURCES_VERIFIED` -- since
    `run_outer_fold_backtest` recomputes that whole span atomically, there
    is nothing to preserve from a partial attempt. `CREATED`/
    `REPORTS_READY`/`VERIFIED` are returned unchanged."""
    if manifest.stage in _MID_FOLD_BACKTEST_STAGES:
        return BacktestStage.SOURCES_VERIFIED
    return manifest.stage


__all__ = [
    "can_resume",
    "require_backtest_resumable",
    "resolve_resume_start_stage",
    "verify_completed_backtest_outer_folds",
]
