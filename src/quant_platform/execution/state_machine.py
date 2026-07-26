"""Deterministic execution-lifecycle state machine (Milestone 4B).

`ExecutionStage` tracks the FINE-GRAINED progress of actually RUNNING an
already-`ready` experiment -- distinct from, and layered ON TOP OF,
`ml.models.ExperimentStatus` (Milestone 4A), which this module never
modifies: `ExperimentManifestStore.transition()` still only ever sees the
manifest go `READY -> RUNNING` once (when execution begins) and
`RUNNING -> COMPLETED`/`RUNNING -> FAILED` once (when execution ends).
Everything in between -- initializing, loading the dataset, building
folds, running each fold, storing results -- is tracked HERE, in a
separate `ExecutionManifest` (`execution/manifests.py`), keyed by the
same `experiment_id`. This layering is deliberate: it lets this milestone
add arbitrarily fine-grained execution bookkeeping without touching a
single line of Milestone 4A's manifest/status code.

WHY `RUNNING_FOLD` AND `STORING_RESULTS` ALTERNATE RATHER THAN APPEARING
ONCE EACH
--------------------------------------------------------------------------
The milestone's own lifecycle diagram (`READY -> INITIALIZING ->
LOADING_DATASET -> BUILDING_SPLITS -> RUNNING_FOLD -> STORING_RESULTS ->
COMPLETED`) is illustrative of the OVERALL shape, not a claim that each
stage is visited exactly once: a walk-forward execution runs N folds,
storing each fold's result as it completes, before finally aggregating.
The legal-transition table below therefore allows `RUNNING_FOLD ->
STORING_RESULTS -> RUNNING_FOLD -> ... -> STORING_RESULTS -> COMPLETED`,
looping once per fold, while still making the terminal states
(`COMPLETED`, `FAILED`, `CANCELLED`) structurally un-exitable, exactly
like `ml.models.is_legal_transition`'s table does for `ExperimentStatus`.

`RECOVERABLE_FAILURE` VS. `FAILED`
--------------------------------------------------------------------------
`RECOVERABLE_FAILURE` means "this specific attempt stopped, but a future
`resume()` call may retry the remaining work" (e.g. a single fold's
artifact write hit a transient lock contention). `FAILED` means "this
execution is done, permanently, and must not be resumed" (e.g. the fold
plan itself failed time-safety validation -- retrying would reproduce the
identical failure). `execution.resume` is the only code that transitions
OUT of `RECOVERABLE_FAILURE`.
"""

from __future__ import annotations

from enum import Enum


class ExecutionStage(Enum):
    INITIALIZING = "initializing"
    LOADING_DATASET = "loading_dataset"
    BUILDING_SPLITS = "building_splits"
    RUNNING_FOLD = "running_fold"
    STORING_RESULTS = "storing_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERABLE_FAILURE = "recoverable_failure"


TERMINAL_STAGES: frozenset[ExecutionStage] = frozenset(
    {ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED}
)

_LEGAL_TRANSITIONS: dict[ExecutionStage, frozenset[ExecutionStage]] = {
    ExecutionStage.INITIALIZING: frozenset(
        {ExecutionStage.LOADING_DATASET, ExecutionStage.FAILED, ExecutionStage.CANCELLED}
    ),
    ExecutionStage.LOADING_DATASET: frozenset(
        {ExecutionStage.BUILDING_SPLITS, ExecutionStage.FAILED, ExecutionStage.CANCELLED}
    ),
    ExecutionStage.BUILDING_SPLITS: frozenset(
        {ExecutionStage.RUNNING_FOLD, ExecutionStage.FAILED, ExecutionStage.CANCELLED}
    ),
    ExecutionStage.RUNNING_FOLD: frozenset(
        {
            ExecutionStage.STORING_RESULTS,
            ExecutionStage.RECOVERABLE_FAILURE,
            ExecutionStage.FAILED,
            ExecutionStage.CANCELLED,
        }
    ),
    ExecutionStage.STORING_RESULTS: frozenset(
        {
            ExecutionStage.RUNNING_FOLD,
            ExecutionStage.COMPLETED,
            ExecutionStage.FAILED,
            ExecutionStage.CANCELLED,
        }
    ),
    ExecutionStage.RECOVERABLE_FAILURE: frozenset(
        {ExecutionStage.RUNNING_FOLD, ExecutionStage.FAILED, ExecutionStage.CANCELLED}
    ),
    ExecutionStage.COMPLETED: frozenset(),
    ExecutionStage.FAILED: frozenset(),
    ExecutionStage.CANCELLED: frozenset(),
}


def is_legal_execution_transition(current: ExecutionStage, target: ExecutionStage) -> bool:
    return target in _LEGAL_TRANSITIONS[current]


def is_terminal_stage(stage: ExecutionStage) -> bool:
    return stage in TERMINAL_STAGES


__all__ = [
    "TERMINAL_STAGES",
    "ExecutionStage",
    "is_legal_execution_transition",
    "is_terminal_stage",
]
