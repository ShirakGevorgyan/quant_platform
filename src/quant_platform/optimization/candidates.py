"""`TrialSpec`/`TrialResult` (Milestone 4D) and the ONE deterministic
candidate-ranking policy every outer fold's trial set is ranked with.

"One trial must never mutate another trial's state. Trial execution must
be isolated and idempotent." -- `TrialSpec` is frozen and fully self-
describing (it duplicates the optimization's fixed `feature_selection_
spec` and carries its own `inner_split_plan_fingerprint`/`model_
definition_fingerprint` rather than requiring a reader to dereference the
parent `OptimizationSpec`/`ModelRegistry` to interpret it), and
`TrialResult` is built exactly once, from data that cannot change
afterward -- there is no mutable "trial state updated in place" anywhere
in this module.

RANKING: ONE FIXED, DETERMINISTIC POLICY (`RANKING_POLICY_VERSION`)
--------------------------------------------------------------------------
"Do not use statistical tests between every trial unless justified and
computationally bounded" -- ranking here is a plain, transparent, total
order (never a pairwise significance test), broken down into exactly
these ordered tie-break criteria, each one only consulted when every
earlier one ties:

  1. Valid (COMPLETED with a defined primary-metric aggregate) before
     invalid/failed/pruned -- an invalid trial can never outrank a valid
     one, regardless of any other criterion.
  2. Primary metric aggregate, better-first per `ml.comparison.
     is_higher_better`'s authoritative direction for this optimization's
     `primary_metric` -- never a re-decided direction.
  3. More successful inner folds is better (more evidence backing the
     same aggregate value is preferred).
  4. Lower dispersion (population std) of the primary metric across the
     SAME inner folds that produced the aggregate -- a trial whose score
     is more consistent across inner folds is preferred over one that is
     equally good on average but wildly inconsistent.
  5. Fewer selected features on average across inner folds -- prefers
     the simpler feature set when nothing else distinguishes two trials.
  6. Lower `estimate_model_complexity` (see that function's own
     docstring for exactly what it measures, and when it is `None`
     -- i.e. skipped -- entirely).
  7. Lower trial number -- the final, always-available, fully
     deterministic tie-break.

Outer-test performance is NEVER an input to any of the above -- nothing
in this module can even see it (no parameter here accepts anything
resembling an outer-test metric)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from quant_platform.ml.models import (
    ArtifactReference,
    JsonPrimitive,
    ObjectiveType,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.optimization.feature_selection import FeatureSelectionSpec
from quant_platform.optimization.objectives import metric_direction_multiplier
from quant_platform.optimization.search_space import SearchSpace, validate_sampled_values

TRIAL_SPEC_SCHEMA_VERSION = 1
TRIAL_RESULT_SCHEMA_VERSION = 1

_ROUNDS_KEYS = ("num_boost_round", "iterations")
_DEPTH_KEYS = ("max_depth", "depth", "num_leaves")


class TrialStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"
    PRUNED = "pruned"


@dataclass(frozen=True, slots=True)
class TrialSpec:
    schema_version: int
    optimization_id: str
    outer_fold_index: int
    trial_number: int
    sampled_hyperparameters: Mapping[str, JsonPrimitive]
    feature_selection_spec: FeatureSelectionSpec
    trial_seed: int
    inner_split_plan_fingerprint: str
    model_definition_fingerprint: str
    objective: ObjectiveType
    primary_metric: str

    def __post_init__(self) -> None:
        if not self.optimization_id:
            raise ValueError("TrialSpec.optimization_id must not be empty")
        if self.outer_fold_index < 0:
            raise ValueError(f"TrialSpec.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if self.trial_number < 0:
            raise ValueError(f"TrialSpec.trial_number must be >= 0, got {self.trial_number}")
        if self.trial_seed < 0:
            raise ValueError(f"TrialSpec.trial_seed must be >= 0, got {self.trial_seed}")
        if not self.inner_split_plan_fingerprint:
            raise ValueError("TrialSpec.inner_split_plan_fingerprint must not be empty")
        if not self.model_definition_fingerprint:
            raise ValueError("TrialSpec.model_definition_fingerprint must not be empty")
        if not self.primary_metric:
            raise ValueError("TrialSpec.primary_metric must not be empty")
        validate_json_primitive_mapping(self.sampled_hyperparameters, field_name="TrialSpec.sampled_hyperparameters")

    def validate_against_search_space(self, space: SearchSpace) -> None:
        validate_sampled_values(space, dict(self.sampled_hyperparameters))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "outer_fold_index": self.outer_fold_index, "trial_number": self.trial_number,
            "sampled_hyperparameters": dict(sorted(self.sampled_hyperparameters.items())),
            "feature_selection_spec": self.feature_selection_spec.to_json_dict(), "trial_seed": self.trial_seed,
            "inner_split_plan_fingerprint": self.inner_split_plan_fingerprint,
            "model_definition_fingerprint": self.model_definition_fingerprint, "objective": self.objective.value,
            "primary_metric": self.primary_metric,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TrialSpec:
        require_schema_version(raw, supported=TRIAL_SPEC_SCHEMA_VERSION, context="TrialSpec")
        return cls(
            schema_version=TRIAL_SPEC_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            outer_fold_index=int(str(raw["outer_fold_index"])), trial_number=int(str(raw["trial_number"])),
            sampled_hyperparameters=as_json_dict(raw.get("sampled_hyperparameters") or {}, field_name="sampled_hyperparameters"),
            feature_selection_spec=FeatureSelectionSpec.from_json_dict(as_json_dict(raw["feature_selection_spec"], field_name="feature_selection_spec")),
            trial_seed=int(str(raw["trial_seed"])), inner_split_plan_fingerprint=str(raw["inner_split_plan_fingerprint"]),
            model_definition_fingerprint=str(raw["model_definition_fingerprint"]), objective=ObjectiveType(raw["objective"]),
            primary_metric=str(raw["primary_metric"]),
        )


@dataclass(frozen=True, slots=True)
class InnerFoldTrialMetrics:
    """One inner fold's contribution to a trial -- computed against that
    inner fold's OWN validation partition only."""

    inner_fold_index: int
    primary_metric_value: float | None
    secondary_metrics: Mapping[str, float]
    selected_feature_count: int
    feature_selection_result_reference: ArtifactReference | None
    best_iteration: int | None
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.inner_fold_index < 0:
            raise ValueError(f"InnerFoldTrialMetrics.inner_fold_index must be >= 0, got {self.inner_fold_index}")
        if self.selected_feature_count < 1:
            raise ValueError(f"InnerFoldTrialMetrics.selected_feature_count must be >= 1, got {self.selected_feature_count}")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError(f"InnerFoldTrialMetrics.duration_seconds must be a finite number >= 0, got {self.duration_seconds}")
        validate_json_primitive_mapping(self.secondary_metrics, field_name="InnerFoldTrialMetrics.secondary_metrics")
        if self.best_iteration is not None and self.best_iteration < 0:
            raise ValueError(f"InnerFoldTrialMetrics.best_iteration must be >= 0 if set, got {self.best_iteration}")
        if self.primary_metric_value is not None and not math.isfinite(self.primary_metric_value):
            raise ValueError(
                f"InnerFoldTrialMetrics.primary_metric_value must be finite if set, got {self.primary_metric_value!r} "
                "-- a NaN/Infinity metric must never enter ranking; the inner fold that produced it should report "
                "None (skipped) instead"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "inner_fold_index": self.inner_fold_index, "primary_metric_value": self.primary_metric_value,
            "secondary_metrics": dict(sorted(self.secondary_metrics.items())),
            "selected_feature_count": self.selected_feature_count,
            "feature_selection_result_reference": (
                None if self.feature_selection_result_reference is None else self.feature_selection_result_reference.to_json_dict()
            ),
            "best_iteration": self.best_iteration, "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InnerFoldTrialMetrics:
        ref_raw = raw.get("feature_selection_result_reference")
        best_iteration_raw = raw.get("best_iteration")
        return cls(
            inner_fold_index=int(str(raw["inner_fold_index"])),
            primary_metric_value=(None if raw.get("primary_metric_value") is None else float(str(raw["primary_metric_value"]))),
            secondary_metrics={str(k): float(v) for k, v in as_json_dict(raw.get("secondary_metrics") or {}, field_name="secondary_metrics").items()},
            selected_feature_count=int(str(raw["selected_feature_count"])),
            feature_selection_result_reference=(None if ref_raw is None else ArtifactReference.from_json_dict(as_json_dict(ref_raw, field_name="feature_selection_result_reference"))),
            best_iteration=(None if best_iteration_raw is None else int(str(best_iteration_raw))),
            duration_seconds=float(str(raw["duration_seconds"])),
        )


@dataclass(frozen=True, slots=True)
class TrialResult:
    schema_version: int
    optimization_id: str
    outer_fold_index: int
    trial_number: int
    status: TrialStatus
    sampled_hyperparameters: Mapping[str, JsonPrimitive]
    inner_fold_metrics: tuple[InnerFoldTrialMetrics, ...]
    primary_metric_aggregate: float | None
    successful_inner_folds: int
    total_inner_folds: int
    duration_seconds: float
    artifact_references: tuple[ArtifactReference, ...] = ()
    environment_snapshot_reference: ArtifactReference | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outer_fold_index < 0:
            raise ValueError(f"TrialResult.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if self.trial_number < 0:
            raise ValueError(f"TrialResult.trial_number must be >= 0, got {self.trial_number}")
        if self.successful_inner_folds < 0 or self.total_inner_folds < 0:
            raise ValueError("TrialResult.successful_inner_folds/total_inner_folds must be >= 0")
        if self.successful_inner_folds > self.total_inner_folds:
            raise ValueError(
                f"TrialResult.successful_inner_folds ({self.successful_inner_folds}) cannot exceed "
                f"total_inner_folds ({self.total_inner_folds})"
            )
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError(f"TrialResult.duration_seconds must be a finite number >= 0, got {self.duration_seconds}")
        validate_json_primitive_mapping(self.sampled_hyperparameters, field_name="TrialResult.sampled_hyperparameters")
        if self.status is TrialStatus.COMPLETED:
            if self.primary_metric_aggregate is None:
                raise ValueError("TrialResult.status=COMPLETED requires a non-None primary_metric_aggregate")
            if self.failure_code is not None or self.failure_reason is not None:
                raise ValueError("TrialResult.status=COMPLETED must not carry a failure_code/failure_reason")
        else:
            if self.failure_reason is None:
                raise ValueError(f"TrialResult.status={self.status.value!r} requires a non-empty failure_reason")
        if self.primary_metric_aggregate is not None and not math.isfinite(self.primary_metric_aggregate):
            raise ValueError(
                f"TrialResult.primary_metric_aggregate must be finite if set, got {self.primary_metric_aggregate!r} "
                "-- a NaN/Infinity aggregate must never enter ranking; this outcome should be reported as "
                "INVALID with a failure_reason instead"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "outer_fold_index": self.outer_fold_index, "trial_number": self.trial_number, "status": self.status.value,
            "sampled_hyperparameters": dict(sorted(self.sampled_hyperparameters.items())),
            "inner_fold_metrics": [m.to_json_dict() for m in self.inner_fold_metrics],
            "primary_metric_aggregate": self.primary_metric_aggregate, "successful_inner_folds": self.successful_inner_folds,
            "total_inner_folds": self.total_inner_folds, "duration_seconds": self.duration_seconds,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
            "environment_snapshot_reference": (None if self.environment_snapshot_reference is None else self.environment_snapshot_reference.to_json_dict()),
            "failure_code": self.failure_code, "failure_reason": self.failure_reason, "warnings": list(self.warnings),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TrialResult:
        require_schema_version(raw, supported=TRIAL_RESULT_SCHEMA_VERSION, context="TrialResult")
        env_ref_raw = raw.get("environment_snapshot_reference")
        return cls(
            schema_version=TRIAL_RESULT_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            outer_fold_index=int(str(raw["outer_fold_index"])), trial_number=int(str(raw["trial_number"])),
            status=TrialStatus(raw["status"]),
            sampled_hyperparameters=as_json_dict(raw.get("sampled_hyperparameters") or {}, field_name="sampled_hyperparameters"),
            inner_fold_metrics=tuple(
                InnerFoldTrialMetrics.from_json_dict(as_json_dict(m, field_name="inner_fold_metrics[]"))
                for m in as_json_list(raw.get("inner_fold_metrics") or [], field_name="inner_fold_metrics")
            ),
            primary_metric_aggregate=(None if raw.get("primary_metric_aggregate") is None else float(str(raw["primary_metric_aggregate"]))),
            successful_inner_folds=int(str(raw["successful_inner_folds"])), total_inner_folds=int(str(raw["total_inner_folds"])),
            duration_seconds=float(str(raw["duration_seconds"])),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
            environment_snapshot_reference=(None if env_ref_raw is None else ArtifactReference.from_json_dict(as_json_dict(env_ref_raw, field_name="environment_snapshot_reference"))),
            failure_code=(None if raw.get("failure_code") is None else str(raw["failure_code"])),
            failure_reason=(None if raw.get("failure_reason") is None else str(raw["failure_reason"])),
            warnings=tuple(str(w) for w in as_json_list(raw.get("warnings") or [], field_name="warnings")),
        )

    @property
    def is_valid_candidate(self) -> bool:
        return self.status is TrialStatus.COMPLETED and self.primary_metric_aggregate is not None


def estimate_model_complexity(sampled_hyperparameters: Mapping[str, JsonPrimitive]) -> float | None:
    """A DELIBERATELY minimal, typed complexity proxy: `(boosting rounds)
    x (a depth-like parameter, if present, else 1)`, read from whichever
    of a small, fixed set of recognized key names is present. Returns
    `None` (never a fabricated value) when no recognized "rounds" key is
    present at all -- e.g. every baseline's fixed search space, and any
    future model this measure was not extended for -- so ranking treats
    "no known complexity measure" as a tie on this criterion, exactly as
    the spec's own "if a typed complexity measure exists" wording implies.
    This is intentionally NOT a claim of comparability ACROSS different
    model families' hyperparameter scales; it only ever breaks a tie
    among trials that already tied on every earlier, more important
    criterion."""
    rounds: float | None = None
    for key in _ROUNDS_KEYS:
        if key in sampled_hyperparameters:
            value = sampled_hyperparameters[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rounds = float(value)
                break
    if rounds is None:
        return None
    depth: float = 1.0
    for key in _DEPTH_KEYS:
        if key in sampled_hyperparameters:
            value = sampled_hyperparameters[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                depth = float(value)
                break
    return rounds * depth


def _metric_dispersion(trial: TrialResult) -> float | None:
    values = [m.primary_metric_value for m in trial.inner_fold_metrics if m.primary_metric_value is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _mean_selected_feature_count(trial: TrialResult) -> float | None:
    counts = [m.selected_feature_count for m in trial.inner_fold_metrics]
    return statistics.fmean(counts) if counts else None


def _ranking_key(trial: TrialResult, *, primary_metric: str) -> tuple[int, float, int, float, float, float, int]:
    if not trial.is_valid_candidate:
        return (1, float("inf"), 0, float("inf"), float("inf"), float("inf"), trial.trial_number)
    assert trial.primary_metric_aggregate is not None
    direction = metric_direction_multiplier(primary_metric)
    metric_rank = -direction * trial.primary_metric_aggregate
    successful_folds_rank = -trial.successful_inner_folds
    dispersion = _metric_dispersion(trial)
    dispersion_rank = dispersion if dispersion is not None else float("inf")
    feature_count = _mean_selected_feature_count(trial)
    feature_count_rank = feature_count if feature_count is not None else float("inf")
    complexity = estimate_model_complexity(trial.sampled_hyperparameters)
    complexity_rank = complexity if complexity is not None else float("inf")
    return (0, metric_rank, successful_folds_rank, dispersion_rank, feature_count_rank, complexity_rank, trial.trial_number)


@dataclass(frozen=True, slots=True)
class RankingEntry:
    rank: int
    trial_number: int
    is_valid: bool
    primary_metric_aggregate: float | None
    successful_inner_folds: int
    metric_dispersion: float | None
    mean_selected_feature_count: float | None
    model_complexity: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank, "trial_number": self.trial_number, "is_valid": self.is_valid,
            "primary_metric_aggregate": self.primary_metric_aggregate, "successful_inner_folds": self.successful_inner_folds,
            "metric_dispersion": self.metric_dispersion, "mean_selected_feature_count": self.mean_selected_feature_count,
            "model_complexity": self.model_complexity,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RankingEntry:
        return cls(
            rank=int(str(raw["rank"])), trial_number=int(str(raw["trial_number"])), is_valid=bool(raw["is_valid"]),
            primary_metric_aggregate=(None if raw.get("primary_metric_aggregate") is None else float(str(raw["primary_metric_aggregate"]))),
            successful_inner_folds=int(str(raw["successful_inner_folds"])),
            metric_dispersion=(None if raw.get("metric_dispersion") is None else float(str(raw["metric_dispersion"]))),
            mean_selected_feature_count=(None if raw.get("mean_selected_feature_count") is None else float(str(raw["mean_selected_feature_count"]))),
            model_complexity=(None if raw.get("model_complexity") is None else float(str(raw["model_complexity"]))),
        )


@dataclass(frozen=True, slots=True)
class RankingTable:
    optimization_id: str
    outer_fold_index: int
    primary_metric: str
    entries: tuple[RankingEntry, ...] = field(default_factory=tuple)

    @property
    def winner(self) -> RankingEntry | None:
        if not self.entries:
            return None
        top = self.entries[0]
        return top if top.is_valid else None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "optimization_id": self.optimization_id, "outer_fold_index": self.outer_fold_index,
            "primary_metric": self.primary_metric, "entries": [e.to_json_dict() for e in self.entries],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RankingTable:
        return cls(
            optimization_id=str(raw["optimization_id"]), outer_fold_index=int(str(raw["outer_fold_index"])),
            primary_metric=str(raw["primary_metric"]),
            entries=tuple(
                RankingEntry.from_json_dict(as_json_dict(e, field_name="entries[]"))
                for e in as_json_list(raw.get("entries") or [], field_name="entries")
            ),
        )


def rank_trials(trials: Sequence[TrialResult], *, primary_metric: str) -> RankingTable:
    """Builds the complete, deterministic ranking table for every trial of
    ONE outer fold -- see module docstring for the exact tie-break chain.
    Every trial supplied is required to share the same `optimization_id`/
    `outer_fold_index` (defense-in-depth: never silently rank trials from
    different outer folds or optimizations against each other)."""
    if not trials:
        raise ValueError("rank_trials requires at least one TrialResult")
    optimization_ids = {t.optimization_id for t in trials}
    outer_fold_indices = {t.outer_fold_index for t in trials}
    if len(optimization_ids) != 1 or len(outer_fold_indices) != 1:
        raise ValueError(
            f"rank_trials requires every trial to share one optimization_id/outer_fold_index, got "
            f"optimization_ids={optimization_ids}, outer_fold_indices={outer_fold_indices}"
        )
    ordered = sorted(trials, key=lambda t: _ranking_key(t, primary_metric=primary_metric))
    entries = tuple(
        RankingEntry(
            rank=index + 1, trial_number=t.trial_number, is_valid=t.is_valid_candidate,
            primary_metric_aggregate=t.primary_metric_aggregate, successful_inner_folds=t.successful_inner_folds,
            metric_dispersion=_metric_dispersion(t), mean_selected_feature_count=_mean_selected_feature_count(t),
            model_complexity=estimate_model_complexity(t.sampled_hyperparameters),
        )
        for index, t in enumerate(ordered)
    )
    return RankingTable(
        optimization_id=trials[0].optimization_id, outer_fold_index=trials[0].outer_fold_index,
        primary_metric=primary_metric, entries=entries,
    )


__all__ = [
    "TRIAL_RESULT_SCHEMA_VERSION",
    "TRIAL_SPEC_SCHEMA_VERSION",
    "InnerFoldTrialMetrics",
    "RankingEntry",
    "RankingTable",
    "TrialResult",
    "TrialSpec",
    "TrialStatus",
    "estimate_model_complexity",
    "rank_trials",
]
