"""Resume support (Milestone 4D) -- discovering which outer folds and
trials of an interrupted optimization are genuinely, verifiably complete.
Mirrors `execution.resume`'s "never trust the manifest's claim alone"
philosophy exactly, extended one level (outer fold containing trials)
deeper.

WHY A CORRUPTED/MISSING TRIAL TRUNCATES THE *ENTIRE REMAINDER*, NOT JUST
ITSELF
--------------------------------------------------------------------------
`execution.resume` can safely discard just ONE corrupted fold and rerun
only that one, because outer-fold-level walk-forward execution has no
ordering dependency between folds' RESULTS (each fold is independently
computed from the same fixed `FoldPlan`). Trial numbering within one
outer fold's search is different: `optimization.study`'s sampler resume
mechanism (see that module's docstring) depends on replaying EVERY prior
trial, IN EXACT ORDER, to reproduce the sampler's RNG trajectory. If
trial `k`'s recorded artifact is missing or corrupted, there is no way to
know what its sampled hyperparameters actually were, so trial `k` cannot
be replayed -- and neither can any trial `k+1, k+2, ...`, even if THEIR
OWN artifacts still look perfectly intact, because the sampler state
those later trials were originally drawn from can never be reconstructed
without first replaying trial `k` itself. `build_trial_resume_plan`
therefore stops at the FIRST verification failure (gap, missing artifact,
category mismatch, corruption, or a fold_index/trial_number/
optimization_id mismatch) it encounters, scanning trial numbers in
ascending order from 0, and treats every trial number from that point
onward as needing to be freshly (re-)sampled -- even ones that exist and
verify individually. This is a documented, deliberate consequence of
using a genuinely sequential sampler; it is announced loudly (via
`TrialResumePlan.discarded_trial_numbers`), never silently patched over.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    OptimizationResumeError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import parse_json_strict
from quant_platform.optimization.candidates import TrialResult
from quant_platform.optimization.manifests import OptimizationManifest
from quant_platform.optimization.models import is_terminal_optimization_stage
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.study import HistoricalTrialRecord

_UNSAFE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-complete artifact can legitimately hit --
identical to `execution.resume._UNSAFE_DECODE_ERRORS`. All treated
identically: fail closed, demote to "needs rerun", never propagate raw."""


def can_resume(manifest: OptimizationManifest | None) -> bool:
    if manifest is None:
        return False
    return not is_terminal_optimization_stage(manifest.stage)


@dataclass(frozen=True, slots=True)
class TrialResumePlan:
    verified_prefix_trial_numbers: tuple[int, ...]
    discarded_trial_numbers: frozenset[int]
    next_trial_number: int


def build_trial_resume_plan(
    claimed_trial_result_references: Mapping[int, ArtifactReference], *, optimization_id: str, outer_fold_index: int,
    artifact_store: MLArtifactStore,
) -> TrialResumePlan:
    """Scans claimed trial numbers `0, 1, 2, ...` in order, verifying each
    one's artifact by content hash, category, and decoded self-identity
    (`TrialResult.optimization_id`/`outer_fold_index`/`trial_number` must
    match the context it was filed under) -- stopping at the first
    failure (see module docstring for why everything from that point
    onward is discarded, not just the one failing entry)."""
    if not claimed_trial_result_references:
        return TrialResumePlan(verified_prefix_trial_numbers=(), discarded_trial_numbers=frozenset(), next_trial_number=0)

    max_claimed = max(claimed_trial_result_references)
    verified_prefix: list[int] = []
    discarded: set[int] = set()
    broken = False

    for trial_number in range(max_claimed + 1):
        if broken:
            if trial_number in claimed_trial_result_references:
                discarded.add(trial_number)
            continue
        reference = claimed_trial_result_references.get(trial_number)
        if reference is None:
            broken = True
            continue
        if reference.category is not ArtifactCategory.TRIAL_RESULT:
            broken = True
            discarded.add(trial_number)
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = TrialResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNSAFE_DECODE_ERRORS:
            broken = True
            discarded.add(trial_number)
            continue
        if decoded.trial_number != trial_number or decoded.outer_fold_index != outer_fold_index or decoded.optimization_id != optimization_id:
            broken = True
            discarded.add(trial_number)
            continue
        verified_prefix.append(trial_number)

    return TrialResumePlan(
        verified_prefix_trial_numbers=tuple(verified_prefix), discarded_trial_numbers=frozenset(discarded),
        next_trial_number=len(verified_prefix),
    )


def load_historical_trial_records(
    plan: TrialResumePlan, *, claimed_trial_result_references: Mapping[int, ArtifactReference], artifact_store: MLArtifactStore,
) -> list[HistoricalTrialRecord]:
    """Re-reads (and, via `MLArtifactStore.read_artifact`, re-verifies)
    every trial in `plan.verified_prefix_trial_numbers`, projecting each
    into the minimal `HistoricalTrialRecord` shape `optimization.study.
    rebuild_study_from_history` needs. Never trusts `plan` alone without
    re-reading the actual bytes -- `plan` only decided WHICH trial numbers
    are safe to use; this function still goes back to the content-
    addressed store for the authoritative content."""
    records: list[HistoricalTrialRecord] = []
    for trial_number in plan.verified_prefix_trial_numbers:
        reference = claimed_trial_result_references[trial_number]
        raw = artifact_store.read_artifact(reference.content_hash)
        decoded = TrialResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        records.append(HistoricalTrialRecord(
            trial_number=decoded.trial_number, sampled_hyperparameters=decoded.sampled_hyperparameters,
            status=decoded.status, primary_metric_aggregate=decoded.primary_metric_aggregate,
        ))
    return records


def verify_completed_outer_folds(
    manifest: OptimizationManifest, *, artifact_store: MLArtifactStore,
) -> tuple[frozenset[int], frozenset[int]]:
    """`(verified_complete, needs_rerun)` outer-fold indices -- re-checks
    each of `manifest.completed_outer_fold_indices`'s claimed
    `OUTER_FOLD_SELECTION` artifact by content hash, category, and decoded
    self-identity, exactly like `build_trial_resume_plan` does for trials.
    Unlike trials, outer folds have NO sequential-sampler ordering
    dependency -- a corrupted outer fold `i` does not invalidate outer
    fold `i+1`'s own, independently verified result."""
    verified: set[int] = set()
    needs_rerun: set[int] = set()
    for outer_fold_index in manifest.completed_outer_fold_indices:
        reference = manifest.outer_fold_result_references.get(outer_fold_index)
        if reference is None or reference.category is not ArtifactCategory.OUTER_FOLD_SELECTION:
            needs_rerun.add(outer_fold_index)
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNSAFE_DECODE_ERRORS:
            needs_rerun.add(outer_fold_index)
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.optimization_id != manifest.optimization_id:
            needs_rerun.add(outer_fold_index)
            continue
        verified.add(outer_fold_index)
    return frozenset(verified), frozenset(needs_rerun)


def require_resumable(manifest: OptimizationManifest | None, *, optimization_id: str) -> OptimizationManifest:
    if manifest is None:
        raise OptimizationResumeError(
            f"No optimization manifest exists for optimization_id={optimization_id!r} -- nothing to resume",
            context={"optimization_id": optimization_id},
        )
    if not can_resume(manifest):
        raise OptimizationResumeError(
            f"Optimization {optimization_id!r} already reached a terminal stage {manifest.stage.value!r} -- "
            "it cannot be resumed or restarted in place",
            context={"optimization_id": optimization_id, "stage": manifest.stage.value},
        )
    return manifest


__all__ = [
    "TrialResumePlan",
    "build_trial_resume_plan",
    "can_resume",
    "load_historical_trial_records",
    "require_resumable",
    "verify_completed_outer_folds",
]
