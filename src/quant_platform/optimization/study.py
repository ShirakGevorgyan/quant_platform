"""Optuna integration (Milestone 4D): deterministic TPE/Random sampling,
typed-`SearchSpace`-to-Optuna-suggestion translation, and the empirically
verified sampler-resume mechanism. Also home to this milestone's ONE
deterministic, platform-owned pruning rule (median-stopping over inner-
fold primary-metric values) -- deliberately never routed through Optuna's
own `trial.report()`/`optuna.pruners` machinery, since "platform-owned
state" (this platform's own persisted `TrialResult`s) is what the spec
prefers when it integrates more safely with the existing content-
addressed artifact model, and it sidesteps needing to reconstruct
Optuna's own pruner internal state as part of resume at all.

WHY MANUAL `ask()`/`tell()`, NEVER `study.optimize()`
--------------------------------------------------------------------------
`study.optimize(objective, n_trials=N)` is Optuna's own convenience loop;
nothing in this package ever calls it. Manual `study.ask()`/`study.tell()`
is required for three independent reasons: (1) this platform must persist
a `TrialResult` artifact BETWEEN sampling a trial's parameters and
reporting its outcome back to the sampler -- `optimize()`'s single-
callback shape has no seam for that; (2) resume must replay a specific,
already-known sequence of historical trials before asking for anything
new, which `optimize()`'s "run N more trials" semantics cannot express;
(3) a trial this platform marks `INVALID`/`FAILED`/`PRUNED` must be told
to Optuna via `state=TrialState.FAIL`/`PRUNED` (no value) so it never
poisons future TPE proposals with a fabricated numeric value --
`optimize()`'s default exception-propagating behavior does not fit this
platform's own explicit `TrialStatus` model at all.

EMPIRICAL VERIFICATION OF DETERMINISTIC RESUME (against installed
`optuna==4.9.0`, see the delivery report for the exact scripts used)
--------------------------------------------------------------------------
The commonly suggested "replay completed trials via `study.add_trial(...)`"
approach was tried FIRST and found NOT to reproduce identical future
suggestions: `add_trial` bypasses `ask()` entirely, so the sampler's
internal RNG is never actually consumed for the replayed trials --
diverging from an uninterrupted run's RNG trajectory, which DOES consume
it once per `ask()`/`suggest_*()` call. The mechanism this module
actually implements -- calling `study.ask()` for every historical trial
too, re-executing the EXACT same ordered sequence of `suggest_*()` calls
(fixed entirely by `SearchSpace.parameters`' declared order), then
`study.tell()` with that trial's already-known final state/value -- WAS
verified to reproduce byte-identical subsequent suggestions, across both
`TPESampler` and `RandomSampler`, across `int`/`float` (including log-
scale)/`categorical`/`boolean` parameters, and across `COMPLETE`/`FAIL`/
`PRUNED` historical trial states. `replay_trial` additionally ASSERTS
(never silently trusts) that each replayed suggestion matches the
originally-recorded `TrialSpec.sampled_hyperparameters` -- any mismatch
(a search-space change, an Optuna version change, or corrupted stored
data) raises `OptimizationResumeError` rather than silently continuing
with a diverged sampler. "Do not reconstruct a TPE study from only the
current best trial" is satisfied structurally: every historical trial,
not merely the best one, is replayed.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import optuna

from quant_platform.core.exceptions import OptimizationResumeError
from quant_platform.ml.models import JsonPrimitive
from quant_platform.optimization.candidates import InnerFoldTrialMetrics, TrialStatus
from quant_platform.optimization.models import OptimizationSpec, PruningConfig, PruningKind, sampler_seed
from quant_platform.optimization.objectives import metric_direction_multiplier
from quant_platform.optimization.search_space import (
    BooleanParameter,
    CategoricalParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
    validate_sampled_values,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
"""This platform's own event store/manifest/reporting is the durable
record of trial progress -- Optuna's own INFO-level per-trial console
logging would be redundant chatter, not additional information, in every
CLI invocation. Warnings/errors still surface."""


def build_sampler(kind: str, seed: int) -> optuna.samplers.BaseSampler:
    from quant_platform.optimization.models import SamplerKind

    if kind == SamplerKind.TPE.value:
        return optuna.samplers.TPESampler(seed=seed)
    if kind == SamplerKind.RANDOM.value:
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler kind {kind!r}")


def create_study(spec: OptimizationSpec) -> optuna.Study:
    """A fresh, LOCAL, in-memory Optuna study (`storage=None`, Optuna's
    own default) -- "do not use distributed Optuna storage in this
    milestone" is honored by never passing a `storage=` argument at all.
    This platform's own manifests/artifacts are the durable record; the
    `optuna.Study` object itself is reconstructed fresh (via `create_
    study` + `rebuild_study_from_history` on resume) every time a process
    starts, never persisted or loaded as Optuna's own file/database."""
    seed = sampler_seed(spec.seed_configuration)
    sampler = build_sampler(spec.sampler_kind.value, seed)
    return optuna.create_study(direction=spec.metric_direction, sampler=sampler)


def suggest_hyperparameters(trial: optuna.trial.Trial, space: SearchSpace) -> dict[str, JsonPrimitive]:
    """Translates one typed `SearchSpace` into a full round of Optuna
    `suggest_*` calls, IN THE SEARCH SPACE'S OWN DECLARED ORDER -- the
    exact order every replay must also use (see module docstring).
    `FixedParameter`s never touch Optuna's suggestion machinery at all
    (they consume no randomness and never vary), so they cannot
    contribute to -- or be a source of divergence in -- RNG consumption."""
    values: dict[str, JsonPrimitive] = {}
    for parameter in space.parameters:
        if isinstance(parameter, IntegerParameter):
            values[parameter.name] = trial.suggest_int(parameter.name, parameter.low, parameter.high, step=parameter.step, log=parameter.log)
        elif isinstance(parameter, FloatParameter):
            if parameter.step is not None:
                values[parameter.name] = trial.suggest_float(parameter.name, parameter.low, parameter.high, step=parameter.step)
            else:
                values[parameter.name] = trial.suggest_float(parameter.name, parameter.low, parameter.high, log=parameter.log)
        elif isinstance(parameter, CategoricalParameter):
            values[parameter.name] = trial.suggest_categorical(parameter.name, list(parameter.choices))
        elif isinstance(parameter, BooleanParameter):
            values[parameter.name] = bool(trial.suggest_categorical(parameter.name, [True, False]))
        elif isinstance(parameter, FixedParameter):
            values[parameter.name] = parameter.value
        else:  # pragma: no cover - exhaustive over ParameterDefinition's closed union
            raise TypeError(f"Unknown parameter definition type {type(parameter).__name__}")
    validate_sampled_values(space, values)
    return values


def ask_next_trial(study: optuna.Study, space: SearchSpace) -> tuple[optuna.trial.Trial, dict[str, JsonPrimitive]]:
    """The ONE entry point a live (non-resume) run uses to get its next
    trial number and sampled hyperparameters."""
    trial = study.ask()
    values = suggest_hyperparameters(trial, space)
    return trial, values


def tell_trial_outcome(study: optuna.Study, trial: optuna.trial.Trial, *, status: TrialStatus, value: float | None) -> None:
    if status is TrialStatus.COMPLETED:
        if value is None:
            raise ValueError("tell_trial_outcome: status=COMPLETED requires a non-None value")
        study.tell(trial, value)
    elif status is TrialStatus.PRUNED:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    else:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


@dataclass(frozen=True, slots=True)
class HistoricalTrialRecord:
    """The minimal projection of a persisted `TrialResult` that resume
    needs to replay it -- deliberately narrow (not the whole
    `TrialResult`) so `optimization.resume`'s own module stays the one
    place that decides how to build this from verified artifacts."""

    trial_number: int
    sampled_hyperparameters: Mapping[str, JsonPrimitive]
    status: TrialStatus
    primary_metric_aggregate: float | None


def replay_trial(study: optuna.Study, space: SearchSpace, record: HistoricalTrialRecord) -> None:
    trial = study.ask()
    if trial.number != record.trial_number:
        raise OptimizationResumeError(
            f"Replay trial-number mismatch: expected trial {record.trial_number}, but the sampler produced "
            f"trial {trial.number} -- historical trials must be replayed in exact, gapless trial-number order",
            context={"expected": record.trial_number, "actual": trial.number},
        )
    replayed_values = suggest_hyperparameters(trial, space)
    if dict(replayed_values) != dict(record.sampled_hyperparameters):
        raise OptimizationResumeError(
            f"Deterministic sampler resume failed for trial {record.trial_number}: replayed sampled "
            f"hyperparameters {replayed_values!r} do not match the originally recorded "
            f"{dict(record.sampled_hyperparameters)!r}. This can happen if the search space, Optuna version, "
            "or sampler seed changed since the original run -- refusing to resume with a diverged sampler.",
            context={"trial_number": record.trial_number},
        )
    value = record.primary_metric_aggregate if record.status is TrialStatus.COMPLETED else None
    tell_trial_outcome(study, trial, status=record.status, value=value)


def rebuild_study_from_history(spec: OptimizationSpec, history: Sequence[HistoricalTrialRecord]) -> optuna.Study:
    """Builds a fresh `optuna.Study` (same sampler class + same derived
    seed as the original run) and replays every historical trial, in
    strict ascending trial-number order, so the next `ask_next_trial`
    call reproduces exactly what an uninterrupted run would have
    produced at that trial number. "Do not reconstruct a TPE study from
    only the current best trial" -- `history` must contain every trial,
    which `optimization.resume` is responsible for supplying complete and
    artifact-hash-verified."""
    study = create_study(spec)
    ordered = sorted(history, key=lambda r: r.trial_number)
    expected_numbers = list(range(len(ordered)))
    actual_numbers = [r.trial_number for r in ordered]
    if actual_numbers != expected_numbers:
        raise OptimizationResumeError(
            f"Cannot rebuild study: historical trial numbers must be exactly 0..{len(ordered) - 1} with no "
            f"gaps or duplicates, got {actual_numbers}",
            context={"trial_numbers": str(actual_numbers)},
        )
    for record in ordered:
        replay_trial(study, spec.search_space, record)
    return study


def evaluate_median_stopping(
    pruning_config: PruningConfig, *, primary_metric: str,
    other_trials_metrics: Sequence[tuple[InnerFoldTrialMetrics, ...]], current_trial_metrics: tuple[InnerFoldTrialMetrics, ...],
) -> bool:
    """This milestone's ONE pruning rule. Deterministic median-stopping:
    once the current trial has completed at least `pruning_config.
    min_completed_inner_folds` inner folds, it is pruned iff its OWN
    running primary-metric aggregate (mean over its own non-`None`
    values so far) is worse -- per `primary_metric`'s authoritative
    direction -- than the MEDIAN of every OTHER trial's running aggregate
    at the SAME inner-fold count (only trials that themselves reached at
    least that many completed folds are eligible comparison points).
    Returns `False` (never prunes) when `pruning_config.kind is
    PruningKind.NONE` -- the mandatory no-pruning control -- or when
    there is not yet enough information (too few of this trial's own
    folds, or no eligible comparison trials) to make any decision at all.
    Never reads, and structurally cannot see, outer-test performance."""
    if pruning_config.kind is PruningKind.NONE:
        return False
    n = len(current_trial_metrics)
    if n < pruning_config.min_completed_inner_folds:
        return False
    current_values = [m.primary_metric_value for m in current_trial_metrics if m.primary_metric_value is not None]
    if not current_values:
        return False
    current_aggregate = statistics.fmean(current_values)

    comparison_aggregates: list[float] = []
    for other in other_trials_metrics:
        if len(other) < n:
            continue
        prefix_values = [m.primary_metric_value for m in other[:n] if m.primary_metric_value is not None]
        if prefix_values:
            comparison_aggregates.append(statistics.fmean(prefix_values))
    if not comparison_aggregates:
        return False

    median = statistics.median(comparison_aggregates)
    direction = metric_direction_multiplier(primary_metric)
    return (direction * current_aggregate) < (direction * median)


__all__ = [
    "HistoricalTrialRecord",
    "ask_next_trial",
    "build_sampler",
    "create_study",
    "evaluate_median_stopping",
    "rebuild_study_from_history",
    "replay_trial",
    "suggest_hyperparameters",
    "tell_trial_outcome",
]
