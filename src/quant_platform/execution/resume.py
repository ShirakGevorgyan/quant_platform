"""Resume support (Milestone 4B, Section 9) -- discovering which folds of
an interrupted execution are genuinely, verifiably complete, and which
remain to run.

NEVER TRUST THE MANIFEST'S CLAIM ALONE
--------------------------------------------------------------------------
`ExecutionManifest.completed_fold_indices` is what the manifest CLAIMS.
`verify_completed_folds` re-checks each claim against the artifact store
itself before trusting it, in THREE independent ways, any one of which is
enough to demote a claimed-complete fold to `needs_rerun`:

  1. `MLArtifactStore.read_artifact` re-verifies the SHA-256 content hash
     on every read -- never skipped. A fold whose recorded `FOLD_RESULT`
     artifact is missing or corrupted is NOT resumed as "already done"
     merely because the manifest says so.
  2. The reference's OWN recorded `category` must actually be
     `FOLD_RESULT` -- a reference pointing at a hash that happens to
     verify but was written under a different artifact category is
     rejected before its bytes are even decoded.
  3. The artifact's bytes must decode as a `FoldResult` whose OWN
     internal `fold_index` field matches the `fold_result_references`
     dict KEY it was filed under. A valid SHA-256 hash proves the BYTES
     are intact; it says nothing about whether those bytes were filed
     under the CORRECT key. A `fold_result_references` entry that (via a
     hand-edited manifest, a future caller's copy-paste bug, or any
     other means) points a genuine fold-3 result at key 5 must be
     rejected here even though its hash checks out cleanly -- silently
     trusting it would let a stale or misplaced result stand in for a
     fold that was never actually verified.

This is the same "recompute, don't trust a stale claim" philosophy
`ml.manifests.ExperimentManifest.__post_init__` already applies to its
own recorded `identity`. Every failure mode above -- missing/corrupted
artifact, wrong category, undecodable JSON, or a fold_index mismatch --
is caught explicitly and converted into "needs rerun"; none of them ever
escapes this module as a raw `KeyError`/`ValueError`/`TypeError`/JSON
decode exception.

WHAT "FORCE" MEANS HERE
--------------------------------------------------------------------------
This milestone does not support restarting an already-TERMINAL
(`COMPLETED`/`FAILED`/`CANCELLED`) execution from scratch in place --
doing so safely would require deciding how to archive/version the prior
terminal record, a real design question left to a future milestone.
"Never rerun completed folds unless explicitly forced" is honored at a
narrower, still-genuinely-useful scope: `force_rerun_folds` lets a caller
resuming a NON-terminal (in particular, `RECOVERABLE_FAILURE`) execution
explicitly name specific fold indices to rerun even if they are verified-
complete (e.g. a fold whose result is suspected stale for a reason the
hash check cannot catch, such as a code change) -- every OTHER verified-
complete fold is still skipped, and a terminal execution remains
permanently terminal regardless of `force_rerun_folds`.

WHY A REPLACED FOLD NEEDS NO EXPLICIT "REMOVE THE OLD REFERENCE" STEP
--------------------------------------------------------------------------
`execution.runner.ExecutionRunner._execute_pipeline` seeds its working
`fold_result_refs` dict ONLY from `resume_plan.verified_complete` (i.e.
folds THIS function actually verified) -- a fold demoted to
`needs_rerun` is therefore simply ABSENT from that seed dict, not present
-with-a-stale-value. When the fold loop reruns it, `fold_result_refs[
fold.fold_index] = ref` writes the one and only entry that dict will
ever hold for that index. There is no separate "delete the old
reference" step because the old one was never carried forward -- see
`tests/unit/execution/test_resume.py::TestFoldIndexMismatchRejected` and
`tests/integration/test_execution_engine.py` for end-to-end proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ExecutionResumeError,
    SchemaVersionError,
)
from quant_platform.execution.manifests import ExecutionManifest
from quant_platform.execution.results import FoldResult
from quant_platform.execution.splitters import Fold, FoldPlan
from quant_platform.execution.state_machine import is_terminal_stage
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory

_UNSAFE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-complete fold's artifact can legitimately
hit: missing/corrupted content (`ArtifactNotFoundError`/
`ArtifactCorruptionError`), an unreadable schema (`SchemaVersionError`),
malformed JSON (`json.JSONDecodeError`, a `ValueError` subclass), or a
`FoldResult.from_json_dict`/`__post_init__` field problem (`KeyError` for
a missing key, `ValueError`/`TypeError` for a malformed one). All treated
identically: fail closed, demote to `needs_rerun`, never propagate raw."""


@dataclass(frozen=True, slots=True)
class ResumePlan:
    verified_complete: frozenset[int]
    needs_rerun: frozenset[int]
    remaining_folds: tuple[Fold, ...]


def can_resume(execution_manifest: ExecutionManifest | None) -> bool:
    """True iff there is a prior, NON-terminal execution manifest to
    resume. `False` for "never started" (`None`) and for every terminal
    stage -- see module docstring for why a terminal execution cannot be
    force-restarted in place."""
    if execution_manifest is None:
        return False
    return not is_terminal_stage(execution_manifest.stage)


def verify_completed_folds(
    execution_manifest: ExecutionManifest, *, artifact_store: MLArtifactStore,
) -> tuple[frozenset[int], frozenset[int]]:
    """`(verified_complete, needs_rerun)` -- partitions
    `execution_manifest.completed_fold_indices` by whether that fold's
    recorded `FOLD_RESULT` artifact is still genuinely present, intact,
    correctly categorized, AND decodes to a `FoldResult` whose own
    `fold_index` matches the manifest dict key it was filed under (see
    module docstring for why each of these is checked independently)."""
    verified: set[int] = set()
    needs_rerun: set[int] = set()
    for fold_index in execution_manifest.completed_fold_indices:
        reference = execution_manifest.fold_result_references.get(fold_index)
        if reference is None:
            needs_rerun.add(fold_index)
            continue
        if reference.category is not ArtifactCategory.FOLD_RESULT:
            needs_rerun.add(fold_index)
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = FoldResult.from_json_dict(json.loads(raw.decode("utf-8")))
        except _UNSAFE_DECODE_ERRORS:
            needs_rerun.add(fold_index)
            continue
        if decoded.fold_index != fold_index:
            needs_rerun.add(fold_index)
            continue
        verified.add(fold_index)
    return frozenset(verified), frozenset(needs_rerun)


def build_resume_plan(
    execution_manifest: ExecutionManifest,
    fold_plan: FoldPlan,
    *,
    artifact_store: MLArtifactStore,
    force_rerun_folds: frozenset[int] = frozenset(),
) -> ResumePlan:
    """Combines `verify_completed_folds` with `force_rerun_folds` into
    the final set of folds that must actually run this time, in
    ascending fold order -- never re-executing a verified-complete fold
    unless it is explicitly named in `force_rerun_folds`."""
    if not can_resume(execution_manifest):
        raise ExecutionResumeError(
            f"Execution for experiment_id={execution_manifest.experiment_id!r} is already terminal "
            f"(stage={execution_manifest.stage.value!r}) -- a terminal execution cannot be resumed",
            context={"experiment_id": execution_manifest.experiment_id, "stage": execution_manifest.stage.value},
        )
    verified, needs_rerun = verify_completed_folds(execution_manifest, artifact_store=artifact_store)
    truly_verified = verified - force_rerun_folds
    remaining = tuple(f for f in fold_plan.folds if f.fold_index not in truly_verified)
    return ResumePlan(
        verified_complete=frozenset(truly_verified), needs_rerun=frozenset(needs_rerun | (verified & force_rerun_folds)),
        remaining_folds=remaining,
    )


__all__ = ["ResumePlan", "build_resume_plan", "can_resume", "verify_completed_folds"]
