"""Feature-stability and hyperparameter-stability analysis (Milestone 4D)
-- summarizes how consistent feature selection and winning hyperparameters
are across inner folds, trials, and outer folds. Purely a READ of already-
computed `FeatureSelectionResult`/`TrialResult`/`OuterFoldResult` data;
nothing here fits anything or touches outer-test.

"Do not equate model-native importance magnitude across different model
families without normalization and explicit documentation" -- this module
never compares `per_feature_score` values across DIFFERENT selector
strategies or model families at all; `FeatureStabilityEntry.mean_score`/
`score_std` are computed only from scores that share one `FeatureSelectionResult.
strategy` (a caller comparing two optimizations that used different
strategies is responsible for treating their score summaries as
non-comparable, which is documented here and in `docs/optimization_engine.md`).

"Flag unstable winning feature sets. Do not automatically remove unstable
features after observing outer-test results." This module only ever
PRODUCES a report and warnings -- there is no code path anywhere in it
(or called by it) that removes a feature, reruns a trial, or otherwise
acts on its own findings; a human (or a future, separate milestone)
decides what to do with an "unstable" flag.

"Do not automatically expand the search space inside the same optimization
identity." This module never touches `SearchSpace` construction -- it
only reports where WITHIN an already-fixed space winners tend to land.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from quant_platform.ml.models import JsonPrimitive
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.optimization.candidates import RankingTable
from quant_platform.optimization.feature_selection import FeatureSelectionResult
from quant_platform.optimization.search_space import (
    BooleanParameter,
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)

_BOUNDARY_FRACTION = 0.01
_BOUNDARY_HIT_WARNING_THRESHOLD = 0.5
_UNSTABLE_JACCARD_THRESHOLD = 0.5
_UNSTABLE_COEFFICIENT_OF_VARIATION = 0.5
_NEAR_TIE_EPSILON_FRACTION = 0.01
FEATURE_STABILITY_SCHEMA_VERSION = 1
HYPERPARAMETER_STABILITY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class JaccardSummary:
    mean: float
    std: float
    min: float
    max: float
    n_pairs: int

    def to_json_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max, "n_pairs": self.n_pairs}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> JaccardSummary:
        return cls(
            mean=float(str(raw["mean"])), std=float(str(raw["std"])), min=float(str(raw["min"])),
            max=float(str(raw["max"])), n_pairs=int(str(raw["n_pairs"])),
        )


def pairwise_jaccard_similarity(feature_sets: Sequence[frozenset[str]]) -> JaccardSummary:
    if len(feature_sets) < 2:
        raise ValueError("pairwise_jaccard_similarity requires at least 2 feature sets")
    values: list[float] = []
    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            a, b = feature_sets[i], feature_sets[j]
            union = a | b
            values.append(len(a & b) / len(union) if union else 1.0)
    return JaccardSummary(mean=statistics.fmean(values), std=(statistics.pstdev(values) if len(values) > 1 else 0.0), min=min(values), max=max(values), n_pairs=len(values))


@dataclass(frozen=True, slots=True)
class FeatureStabilityEntry:
    feature_name: str
    times_selected: int
    eligible_evaluations: int
    selection_frequency: float
    selected_in_winning_candidate_frequency: float
    mean_rank: float | None = None
    rank_std: float | None = None
    mean_score: float | None = None
    score_std: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name, "times_selected": self.times_selected,
            "eligible_evaluations": self.eligible_evaluations, "selection_frequency": self.selection_frequency,
            "selected_in_winning_candidate_frequency": self.selected_in_winning_candidate_frequency,
            "mean_rank": self.mean_rank, "rank_std": self.rank_std, "mean_score": self.mean_score, "score_std": self.score_std,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureStabilityEntry:
        return cls(
            feature_name=str(raw["feature_name"]), times_selected=int(str(raw["times_selected"])),
            eligible_evaluations=int(str(raw["eligible_evaluations"])), selection_frequency=float(str(raw["selection_frequency"])),
            selected_in_winning_candidate_frequency=float(str(raw["selected_in_winning_candidate_frequency"])),
            mean_rank=(None if raw.get("mean_rank") is None else float(str(raw["mean_rank"]))),
            rank_std=(None if raw.get("rank_std") is None else float(str(raw["rank_std"]))),
            mean_score=(None if raw.get("mean_score") is None else float(str(raw["mean_score"]))),
            score_std=(None if raw.get("score_std") is None else float(str(raw["score_std"]))),
        )


@dataclass(frozen=True, slots=True)
class FeatureStabilityReport:
    schema_version: int
    optimization_id: str
    total_evaluations: int
    entries: tuple[FeatureStabilityEntry, ...]
    feature_count_distribution: Mapping[int, int]
    pairwise_jaccard: JaccardSummary | None = None
    warnings: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "total_evaluations": self.total_evaluations, "entries": [e.to_json_dict() for e in self.entries],
            "feature_count_distribution": {str(k): v for k, v in sorted(self.feature_count_distribution.items())},
            "pairwise_jaccard": (None if self.pairwise_jaccard is None else self.pairwise_jaccard.to_json_dict()),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureStabilityReport:
        require_schema_version(raw, supported=FEATURE_STABILITY_SCHEMA_VERSION, context="FeatureStabilityReport")
        jaccard_raw = raw.get("pairwise_jaccard")
        return cls(
            schema_version=FEATURE_STABILITY_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            total_evaluations=int(str(raw["total_evaluations"])),
            entries=tuple(
                FeatureStabilityEntry.from_json_dict(as_json_dict(e, field_name="entries[]"))
                for e in as_json_list(raw.get("entries") or [], field_name="entries")
            ),
            feature_count_distribution={
                int(k): int(v) for k, v in as_json_dict(raw.get("feature_count_distribution") or {}, field_name="feature_count_distribution").items()
            },
            pairwise_jaccard=(None if jaccard_raw is None else JaccardSummary.from_json_dict(as_json_dict(jaccard_raw, field_name="pairwise_jaccard"))),
            warnings=tuple(str(w) for w in as_json_list(raw.get("warnings") or [], field_name="warnings")),
        )


def summarize_feature_stability(
    *, optimization_id: str, feature_selection_results: Sequence[FeatureSelectionResult], winning_feature_sets: Sequence[tuple[str, ...]],
) -> FeatureStabilityReport:
    """`feature_selection_results` is every evaluated `FeatureSelectionResult`
    across every inner fold of every trial (of one outer fold, or of the
    whole optimization -- the caller decides the scope). `winning_feature_sets`
    is one entry per OUTER FOLD's `OuterFoldResult.final_selected_features`
    -- used for `selected_in_winning_candidate_frequency` and the pairwise
    Jaccard summary (comparing what the FINAL answer looked like across
    outer folds, the single most decision-relevant stability question)."""
    if not feature_selection_results:
        raise ValueError("summarize_feature_stability requires at least one FeatureSelectionResult")
    total = len(feature_selection_results)
    universe_names: set[str] = set()
    for r in feature_selection_results:
        universe_names |= set(r.selected_features) | set(r.rejected_features)

    times_selected = dict.fromkeys(universe_names, 0)
    ranks_by_feature: dict[str, list[int]] = {name: [] for name in universe_names}
    scores_by_feature: dict[str, list[float]] = {name: [] for name in universe_names}
    feature_count_distribution: dict[int, int] = {}

    for r in feature_selection_results:
        for name in r.selected_features:
            times_selected[name] += 1
        if r.per_feature_rank:
            for name, rank in r.per_feature_rank.items():
                if name in ranks_by_feature:
                    ranks_by_feature[name].append(rank)
        if r.per_feature_score:
            for name, score in r.per_feature_score.items():
                if name in scores_by_feature:
                    scores_by_feature[name].append(score)
        count = len(r.selected_features)
        feature_count_distribution[count] = feature_count_distribution.get(count, 0) + 1

    winning_count = len(winning_feature_sets)
    times_in_winner = dict.fromkeys(universe_names, 0)
    for feature_set in winning_feature_sets:
        for name in feature_set:
            if name in times_in_winner:
                times_in_winner[name] += 1

    entries = tuple(
        FeatureStabilityEntry(
            feature_name=name, times_selected=times_selected[name], eligible_evaluations=total,
            selection_frequency=times_selected[name] / total,
            selected_in_winning_candidate_frequency=(times_in_winner[name] / winning_count if winning_count else 0.0),
            mean_rank=(statistics.fmean(ranks_by_feature[name]) if ranks_by_feature[name] else None),
            rank_std=(statistics.pstdev(ranks_by_feature[name]) if len(ranks_by_feature[name]) > 1 else (0.0 if ranks_by_feature[name] else None)),
            mean_score=(statistics.fmean(scores_by_feature[name]) if scores_by_feature[name] else None),
            score_std=(statistics.pstdev(scores_by_feature[name]) if len(scores_by_feature[name]) > 1 else (0.0 if scores_by_feature[name] else None)),
        )
        for name in sorted(universe_names)
    )

    jaccard = pairwise_jaccard_similarity([frozenset(s) for s in winning_feature_sets]) if len(winning_feature_sets) >= 2 else None
    warnings: list[str] = []
    if jaccard is not None and jaccard.mean < _UNSTABLE_JACCARD_THRESHOLD:
        warnings.append(
            f"winning feature sets across outer folds show LOW stability (mean pairwise Jaccard similarity "
            f"{jaccard.mean:.2f} < {_UNSTABLE_JACCARD_THRESHOLD}) -- treat the selected feature set as "
            "provisional, not a single confirmed answer"
        )
    return FeatureStabilityReport(
        schema_version=FEATURE_STABILITY_SCHEMA_VERSION, optimization_id=optimization_id, total_evaluations=total, entries=entries,
        feature_count_distribution=feature_count_distribution, pairwise_jaccard=jaccard, warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class NumericParameterStability:
    parameter_name: str
    values: tuple[float, ...]
    mean: float
    std: float
    min: float
    max: float
    low_bound: float
    high_bound: float
    boundary_hit_frequency: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "parameter_name": self.parameter_name, "values": list(self.values), "mean": self.mean, "std": self.std,
            "min": self.min, "max": self.max, "low_bound": self.low_bound, "high_bound": self.high_bound,
            "boundary_hit_frequency": self.boundary_hit_frequency,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> NumericParameterStability:
        return cls(
            parameter_name=str(raw["parameter_name"]),
            values=tuple(float(v) for v in as_json_list(raw["values"], field_name="values")),
            mean=float(str(raw["mean"])), std=float(str(raw["std"])), min=float(str(raw["min"])), max=float(str(raw["max"])),
            low_bound=float(str(raw["low_bound"])), high_bound=float(str(raw["high_bound"])),
            boundary_hit_frequency=float(str(raw["boundary_hit_frequency"])),
        )


@dataclass(frozen=True, slots=True)
class CategoricalParameterStability:
    parameter_name: str
    choice_frequencies: Mapping[str, float]

    def to_json_dict(self) -> dict[str, object]:
        return {"parameter_name": self.parameter_name, "choice_frequencies": dict(sorted(self.choice_frequencies.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CategoricalParameterStability:
        return cls(
            parameter_name=str(raw["parameter_name"]),
            choice_frequencies={
                str(k): float(v) for k, v in as_json_dict(raw.get("choice_frequencies") or {}, field_name="choice_frequencies").items()
            },
        )


@dataclass(frozen=True, slots=True)
class HyperparameterStabilityReport:
    schema_version: int
    optimization_id: str
    outer_fold_count: int
    numeric_parameters: tuple[NumericParameterStability, ...] = field(default_factory=tuple)
    categorical_parameters: tuple[CategoricalParameterStability, ...] = field(default_factory=tuple)
    best_iteration_distribution: Mapping[str, float] | None = None
    trial_score_dispersion: float | None = None
    warnings: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "outer_fold_count": self.outer_fold_count,
            "numeric_parameters": [p.to_json_dict() for p in self.numeric_parameters],
            "categorical_parameters": [p.to_json_dict() for p in self.categorical_parameters],
            "best_iteration_distribution": (None if self.best_iteration_distribution is None else dict(self.best_iteration_distribution)),
            "trial_score_dispersion": self.trial_score_dispersion, "warnings": list(self.warnings),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HyperparameterStabilityReport:
        require_schema_version(raw, supported=HYPERPARAMETER_STABILITY_SCHEMA_VERSION, context="HyperparameterStabilityReport")
        best_iteration_raw = raw.get("best_iteration_distribution")
        trial_score_dispersion_raw = raw.get("trial_score_dispersion")
        return cls(
            schema_version=HYPERPARAMETER_STABILITY_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            outer_fold_count=int(str(raw["outer_fold_count"])),
            numeric_parameters=tuple(
                NumericParameterStability.from_json_dict(as_json_dict(p, field_name="numeric_parameters[]"))
                for p in as_json_list(raw.get("numeric_parameters") or [], field_name="numeric_parameters")
            ),
            categorical_parameters=tuple(
                CategoricalParameterStability.from_json_dict(as_json_dict(p, field_name="categorical_parameters[]"))
                for p in as_json_list(raw.get("categorical_parameters") or [], field_name="categorical_parameters")
            ),
            best_iteration_distribution=(
                None if best_iteration_raw is None
                else {str(k): float(v) for k, v in as_json_dict(best_iteration_raw, field_name="best_iteration_distribution").items()}
            ),
            trial_score_dispersion=(None if trial_score_dispersion_raw is None else float(str(trial_score_dispersion_raw))),
            warnings=tuple(str(w) for w in as_json_list(raw.get("warnings") or [], field_name="warnings")),
        )


def summarize_hyperparameter_stability(
    *, optimization_id: str, search_space: SearchSpace, winning_hyperparameters: Sequence[Mapping[str, JsonPrimitive]],
    winning_primary_metric_values: Sequence[float], best_iterations: Sequence[int] = (),
) -> HyperparameterStabilityReport:
    """`winning_hyperparameters`/`winning_primary_metric_values` are one
    entry per outer fold (that outer fold's winning trial's sampled
    hyperparameters and outer-test-independent inner aggregate score --
    NEVER outer-test performance; this analysis is entirely about search
    behavior, not final evaluation)."""
    numeric_parameters: list[NumericParameterStability] = []
    categorical_parameters: list[CategoricalParameterStability] = []
    warnings: list[str] = []

    for parameter in search_space.parameters:
        if isinstance(parameter, (IntegerParameter, FloatParameter)):
            values = [float(str(w[parameter.name])) for w in winning_hyperparameters if parameter.name in w]
            if not values:
                continue
            low, high = float(parameter.low), float(parameter.high)
            epsilon = (high - low) * _BOUNDARY_FRACTION if high > low else 0.0
            hits = sum(1 for v in values if v <= low + epsilon or v >= high - epsilon)
            frequency = hits / len(values)
            numeric_parameters.append(NumericParameterStability(
                parameter_name=parameter.name, values=tuple(values), mean=statistics.fmean(values),
                std=(statistics.pstdev(values) if len(values) > 1 else 0.0), min=min(values), max=max(values),
                low_bound=low, high_bound=high, boundary_hit_frequency=frequency,
            ))
            if frequency >= _BOUNDARY_HIT_WARNING_THRESHOLD:
                warnings.append(
                    f"parameter {parameter.name!r} winners repeatedly lie on the search-space boundary "
                    f"({frequency:.0%} of outer folds within {_BOUNDARY_FRACTION:.0%} of [{low}, {high}]) -- "
                    "consider widening the search space in a NEW OptimizationSpec"
                )
        elif isinstance(parameter, (CategoricalParameter, BooleanParameter)):
            choices_seen = [w[parameter.name] for w in winning_hyperparameters if parameter.name in w]
            if not choices_seen:
                continue
            counts: dict[str, int] = {}
            for choice in choices_seen:
                counts[str(choice)] = counts.get(str(choice), 0) + 1
            categorical_parameters.append(CategoricalParameterStability(
                parameter_name=parameter.name, choice_frequencies={k: v / len(choices_seen) for k, v in counts.items()},
            ))

    best_iteration_distribution = None
    if best_iterations:
        best_iteration_distribution = {
            "mean": statistics.fmean(best_iterations), "std": (statistics.pstdev(best_iterations) if len(best_iterations) > 1 else 0.0),
            "min": float(min(best_iterations)), "max": float(max(best_iterations)),
        }

    trial_score_dispersion: float | None = None
    if winning_primary_metric_values:
        trial_score_dispersion = statistics.pstdev(winning_primary_metric_values) if len(winning_primary_metric_values) > 1 else 0.0
        if len(winning_primary_metric_values) >= 2:
            mean_score = statistics.fmean(winning_primary_metric_values)
            if mean_score != 0 and abs(trial_score_dispersion / mean_score) > _UNSTABLE_COEFFICIENT_OF_VARIATION:
                warnings.append(
                    "the winning candidate's inner-validation aggregate score varies substantially across outer "
                    f"folds (coefficient of variation {abs(trial_score_dispersion / mean_score):.2f} > "
                    f"{_UNSTABLE_COEFFICIENT_OF_VARIATION}) -- treat the winning configuration as UNSTABLE across "
                    "time, not a single confirmed best choice"
                )

    return HyperparameterStabilityReport(
        schema_version=HYPERPARAMETER_STABILITY_SCHEMA_VERSION, optimization_id=optimization_id, outer_fold_count=len(winning_hyperparameters),
        numeric_parameters=tuple(numeric_parameters), categorical_parameters=tuple(categorical_parameters),
        best_iteration_distribution=best_iteration_distribution, trial_score_dispersion=trial_score_dispersion,
        warnings=tuple(warnings),
    )


def flag_near_tied_top_candidates(ranking_table: RankingTable, *, epsilon_fraction: float = _NEAR_TIE_EPSILON_FRACTION) -> tuple[str, ...]:
    """A DELIBERATELY narrow, honest implementation of "materially
    different parameters produce indistinguishable scores": compares
    ONLY the top two valid entries of one outer fold's ranking table. If
    their primary-metric aggregates differ by less than `epsilon_fraction`
    of the winner's own magnitude, returns one warning naming both trial
    numbers. This is NOT a full pairwise statistical-equivalence test
    across every trial (that would require the "computationally bounded"
    justification the spec asks for and which this milestone does not
    attempt) -- it is a cheap, transparent proximity check on the two
    trials that matter most (the winner and its closest competitor)."""
    valid_entries = [e for e in ranking_table.entries if e.is_valid]
    if len(valid_entries) < 2:
        return ()
    top, runner_up = valid_entries[0], valid_entries[1]
    if top.primary_metric_aggregate is None or runner_up.primary_metric_aggregate is None:
        return ()
    scale = max(abs(top.primary_metric_aggregate), 1e-12)
    if abs(top.primary_metric_aggregate - runner_up.primary_metric_aggregate) <= epsilon_fraction * scale:
        return (
            f"trial {top.trial_number} (winner) and trial {runner_up.trial_number} (runner-up) have "
            f"near-indistinguishable primary-metric aggregates ({top.primary_metric_aggregate:.6g} vs "
            f"{runner_up.primary_metric_aggregate:.6g}) -- the winning hyperparameters may not be meaningfully "
            "better than the runner-up's, only its ranking tie-break more favorable",
        )
    return ()


__all__ = [
    "FEATURE_STABILITY_SCHEMA_VERSION",
    "HYPERPARAMETER_STABILITY_SCHEMA_VERSION",
    "CategoricalParameterStability",
    "FeatureStabilityEntry",
    "FeatureStabilityReport",
    "HyperparameterStabilityReport",
    "JaccardSummary",
    "NumericParameterStability",
    "flag_near_tied_top_candidates",
    "pairwise_jaccard_similarity",
    "summarize_feature_stability",
    "summarize_hyperparameter_stability",
]
