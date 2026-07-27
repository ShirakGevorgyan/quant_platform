"""Model comparison and ranking (Milestone 4C, "MODEL COMPARISON") --
"Never declare a model successful unless it statistically outperforms
every baseline."

INPUT SHAPE: PER-FOLD METRIC VALUES, ALREADY COMPUTED
--------------------------------------------------------------------------
This module never re-runs a model or re-computes a metric from raw
predictions -- it consumes `ModelFoldMetrics` (one per candidate/baseline
model: a name plus one `Mapping[str, float]` per fold, exactly the shape
`execution.results.FoldResult.metrics` already has for every fold of an
experiment run through `execution.executor.MetricsFoldExecutor`).
Comparing two models is only meaningful when both were evaluated against
the SAME folds of the SAME dataset -- this module trusts its caller to
have arranged THAT (e.g. by running the candidate and every baseline as
separate `ExperimentSpec`s bound to the identical `dataset_binding`/
`split_binding`), but it never trusts raw list POSITION to prove it:
`ModelFoldMetrics.fold_identities` (each entry's `FoldResult.fold_index`)
is what `_compare_one_metric` actually pairs on -- only folds present,
under the SAME identity, on BOTH sides are ever paired; folds present on
only one side (different fold counts, different split configs, results
loaded out of order) are simply excluded from the paired sample, which
can turn into `SKIPPED_INSUFFICIENT_DATA` if too few (or zero) remain.
`fold_identities` defaults to sequential `0..n-1` when omitted -- correct
whenever a caller built both sides from the same configuration in the
same order (most of this module's own tests), but `ml_cli.cmd_compare`'s
real caller always supplies the genuine fold indices it read results
from.

STATISTICAL TEST: PAIRED, NONPARAMETRIC, HONEST ABOUT SMALL FOLD COUNTS
--------------------------------------------------------------------------
`scipy.stats.wilcoxon` (a paired, distribution-free test on the per-fold
DIFFERENCES between candidate and baseline) is used rather than a paired
t-test -- walk-forward cross-validation typically produces few folds
(single digits), where a t-test's normality assumption is not defensible
and Wilcoxon's rank-based approach is more robust. Wilcoxon itself,
though, still needs enough non-tied differences to say anything
meaningful: with fewer than `_MIN_FOLDS_FOR_SIGNIFICANCE` folds, or when
every fold's difference is exactly zero (`scipy.stats.wilcoxon` raises for
an all-zero-difference input), this module reports the comparison as
`SKIPPED_INSUFFICIENT_DATA` rather than fabricating a p-value from a
sample too small to support one -- "never declare success" cuts both
ways: an underpowered test must never be silently treated as "passed".

DIRECTION-AWARE: "BETTER" DEPENDS ON THE METRIC
--------------------------------------------------------------------------
Accuracy/precision/recall/F1/ROC AUC/PR AUC/balanced accuracy/MCC/R^2 are
HIGHER-is-better; MAE/RMSE/MAPE are LOWER-is-better. `_HIGHER_IS_BETTER`
is the one place this directionality is declared; every comparison in
this module reads it from there rather than re-deciding per call site.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from quant_platform.ml.metrics import MetricAggregate, aggregate_metric_values

_MIN_FOLDS_FOR_SIGNIFICANCE = 5
"""Below this many PAIRED, non-tied observations, `scipy.stats.wilcoxon`'s
own result is not reported as a pass/fail significance call -- see module
docstring. Chosen as a conservative, documented floor, not a guarantee
that 5 folds gives strong statistical power (it does not); it is the
point below which this module refuses to pretend the test means
anything at all."""
_DEFAULT_ALPHA = 0.05

_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef", "r2",
})
_LOWER_IS_BETTER: frozenset[str] = frozenset({"mae", "rmse", "mape"})


def _is_higher_better(metric_name: str) -> bool:
    if metric_name in _HIGHER_IS_BETTER:
        return True
    if metric_name in _LOWER_IS_BETTER:
        return False
    raise ValueError(
        f"Unknown metric {metric_name!r} -- comparison direction (higher-is-better vs. lower-is-better) is "
        "not declared for it; add it to ml.comparison._HIGHER_IS_BETTER/_LOWER_IS_BETTER explicitly"
    )


def is_higher_better(metric_name: str) -> bool:
    """The PUBLIC accessor for this module's one authoritative metric-
    direction registry -- added for Milestone 4D (`optimization.objectives`),
    which must derive ranking/pruning/early-stopping direction from THIS
    registry and must never define a second, competing one. Raises
    `ValueError` for any metric name not declared in `_HIGHER_IS_BETTER`/
    `_LOWER_IS_BETTER`, exactly like every internal call site already
    does -- there is no silent default direction."""
    return _is_higher_better(metric_name)


def known_metric_names() -> frozenset[str]:
    """Every metric name this registry declares a direction for -- the
    complete, authoritative set `optimization.objectives` validates a
    caller-declared `primary_metric` against."""
    return _HIGHER_IS_BETTER | _LOWER_IS_BETTER


@dataclass(frozen=True, slots=True)
class ModelFoldMetrics:
    """One model's (candidate OR baseline) per-fold metric values.
    `per_fold_metrics[i]` need not contain every metric name (a fold that
    skipped a metric -- see `ml.metrics.MetricComputationReport` --
    simply omits it, exactly like `FoldResult.metrics` already can).

    `fold_identities`, when given, is `FoldResult.fold_index` for each
    entry of `per_fold_metrics`, in the SAME order -- `values_by_fold_for`
    uses it to pair folds by IDENTITY, never by raw list position, when
    comparing two `ModelFoldMetrics` built from potentially-different
    sources (see `_compare_one_metric`: "never compare unrelated folds by
    list position unless fold identity matches"). Omitted (`None`, the
    default), it is treated as sequential `0..n-1` -- correct whenever
    both sides of a comparison were built from the SAME walk-forward
    configuration in the SAME order (true of most of this module's own
    tests), but a REAL caller comparing two independently-loaded
    experiments (`ml_cli.cmd_compare`'s `_load_fold_metrics`) always
    supplies the actual fold indices it read them from, so a genuine
    misalignment (different fold counts, different split configs,
    results loaded out of order) is DETECTED rather than silently paired
    by position."""

    model_name: str
    per_fold_metrics: tuple[Mapping[str, float], ...]
    fold_identities: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("ModelFoldMetrics.model_name must not be empty")
        if not self.per_fold_metrics:
            raise ValueError("ModelFoldMetrics.per_fold_metrics must not be empty")
        if self.fold_identities is not None:
            if len(self.fold_identities) != len(self.per_fold_metrics):
                raise ValueError(
                    f"ModelFoldMetrics.fold_identities ({len(self.fold_identities)}) must have the same length "
                    f"as per_fold_metrics ({len(self.per_fold_metrics)})"
                )
            if len(set(self.fold_identities)) != len(self.fold_identities):
                raise ValueError(f"ModelFoldMetrics.fold_identities must not contain duplicates, got {self.fold_identities}")

    def _identities(self) -> tuple[int, ...]:
        return self.fold_identities if self.fold_identities is not None else tuple(range(len(self.per_fold_metrics)))

    def values_for(self, metric_name: str) -> list[float]:
        return [float(m[metric_name]) for m in self.per_fold_metrics if metric_name in m]

    def values_by_fold_for(self, metric_name: str) -> dict[int, float]:
        """`fold_identity -> value`, for every fold (by IDENTITY, from
        `fold_identities`/its sequential default) that reported this
        metric. The identity-aware counterpart to `values_for`, used
        wherever two `ModelFoldMetrics` must be paired fold-for-fold."""
        return {
            fold_id: float(m[metric_name])
            for fold_id, m in zip(self._identities(), self.per_fold_metrics, strict=True)
            if metric_name in m
        }


class ComparisonOutcome(Enum):
    CANDIDATE_SIGNIFICANTLY_BETTER = "candidate_significantly_better"
    CANDIDATE_SIGNIFICANTLY_WORSE = "candidate_significantly_worse"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """`candidate_aggregate`/`baseline_aggregate` reuse `ml.metrics.
    MetricAggregate` (mean/std/min/max/median/n) directly -- the SAME
    "FOLD AGGREGATION" this milestone already requires elsewhere, never a
    second, parallel summary-statistics type computed just for
    comparison reports. `None` when that side never reported this metric
    in any fold at all (see `reason`)."""

    metric_name: str
    candidate_aggregate: MetricAggregate | None
    baseline_aggregate: MetricAggregate | None
    paired_n: int
    p_value: float | None
    outcome: ComparisonOutcome
    reason: str | None = None
    """Populated ONLY when `outcome is SKIPPED_INSUFFICIENT_DATA` --
    explains WHY (too few paired folds, all-zero differences, a metric
    missing from one side entirely), never silently blank."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "candidate_aggregate": (None if self.candidate_aggregate is None else self.candidate_aggregate.to_json_dict()),
            "baseline_aggregate": (None if self.baseline_aggregate is None else self.baseline_aggregate.to_json_dict()),
            "paired_n": self.paired_n, "p_value": self.p_value, "outcome": self.outcome.value, "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BaselineComparisonReport:
    baseline_name: str
    metric_comparisons: tuple[MetricComparison, ...]

    def comparison_for(self, metric_name: str) -> MetricComparison:
        for c in self.metric_comparisons:
            if c.metric_name == metric_name:
                return c
        raise KeyError(f"No comparison recorded for metric {metric_name!r} against baseline {self.baseline_name!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {"baseline_name": self.baseline_name, "metric_comparisons": [c.to_json_dict() for c in self.metric_comparisons]}


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """One candidate model's comparison against EVERY baseline it was
    compared to, for every metric BOTH sides reported in at least one
    fold. `outperforms_all_baselines(primary_metric)` is the single,
    explicit gate "never declare a model successful unless..." means --
    never inferred implicitly from scanning `baseline_reports` yourself."""

    candidate_name: str
    baseline_reports: tuple[BaselineComparisonReport, ...]

    def outperforms_all_baselines(self, primary_metric: str) -> bool:
        """True iff, for `primary_metric` specifically, the candidate is
        `CANDIDATE_SIGNIFICANTLY_BETTER` than EVERY baseline compared
        against. `SKIPPED_INSUFFICIENT_DATA` or a missing comparison
        entry is treated as "not proven better" (never silently ignored
        as a pass) -- exactly the "never declare success" requirement."""
        if not self.baseline_reports:
            return False
        for report in self.baseline_reports:
            try:
                comparison = report.comparison_for(primary_metric)
            except KeyError:
                return False
            if comparison.outcome is not ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER:
                return False
        return True

    def to_json_dict(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "baseline_reports": [b.to_json_dict() for b in self.baseline_reports],
        }


def compare_to_baseline(
    candidate: ModelFoldMetrics, baseline: ModelFoldMetrics, *, alpha: float = _DEFAULT_ALPHA,
) -> BaselineComparisonReport:
    """Compares `candidate` against ONE `baseline`, across every metric
    name either side reports in at least one fold."""
    metric_names = sorted({name for fold in (*candidate.per_fold_metrics, *baseline.per_fold_metrics) for name in fold})
    comparisons = tuple(_compare_one_metric(candidate, baseline, metric_name, alpha=alpha) for metric_name in metric_names)
    return BaselineComparisonReport(baseline_name=baseline.model_name, metric_comparisons=comparisons)


def compare_to_baselines(
    candidate: ModelFoldMetrics, baselines: Sequence[ModelFoldMetrics], *, alpha: float = _DEFAULT_ALPHA,
) -> ModelComparisonReport:
    reports = tuple(compare_to_baseline(candidate, baseline, alpha=alpha) for baseline in baselines)
    return ModelComparisonReport(candidate_name=candidate.model_name, baseline_reports=reports)


def _compare_one_metric(
    candidate: ModelFoldMetrics, baseline: ModelFoldMetrics, metric_name: str, *, alpha: float,
) -> MetricComparison:
    candidate_values = candidate.values_for(metric_name)
    baseline_values = baseline.values_for(metric_name)
    candidate_aggregate = aggregate_metric_values(candidate_values) if candidate_values else None
    baseline_aggregate = aggregate_metric_values(baseline_values) if baseline_values else None

    if candidate_aggregate is None or baseline_aggregate is None:
        return MetricComparison(
            metric_name=metric_name, candidate_aggregate=candidate_aggregate, baseline_aggregate=baseline_aggregate,
            paired_n=0, p_value=None, outcome=ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA,
            reason=f"metric {metric_name!r} was never reported by "
            f"{'the candidate' if candidate_aggregate is None else 'the baseline'}",
        )

    # Pairing by FOLD IDENTITY, never raw list position: only folds where
    # BOTH sides reported this metric under the SAME fold identity are
    # paired -- a candidate/baseline built from different fold counts,
    # different split configurations, or results loaded out of order are
    # never silently compared position-for-position (see `ModelFoldMetrics.
    # fold_identities`'s own docstring).
    candidate_by_fold = candidate.values_by_fold_for(metric_name)
    baseline_by_fold = baseline.values_by_fold_for(metric_name)
    aligned_fold_ids = sorted(set(candidate_by_fold) & set(baseline_by_fold))
    paired_n = len(aligned_fold_ids)

    if paired_n < _MIN_FOLDS_FOR_SIGNIFICANCE:
        if not aligned_fold_ids and candidate_by_fold and baseline_by_fold:
            reason = (
                f"no overlapping fold identities reported {metric_name!r} on both sides -- candidate folds "
                f"{sorted(candidate_by_fold)}, baseline folds {sorted(baseline_by_fold)}"
            )
        else:
            reason = (
                f"only {paired_n} fold(s) with a matching fold identity available; a statistical test needs "
                f"at least {_MIN_FOLDS_FOR_SIGNIFICANCE} to report a meaningful result"
            )
        return MetricComparison(
            metric_name=metric_name, candidate_aggregate=candidate_aggregate, baseline_aggregate=baseline_aggregate,
            paired_n=paired_n, p_value=None, outcome=ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA, reason=reason,
        )

    candidate_paired = [candidate_by_fold[fold_id] for fold_id in aligned_fold_ids]
    baseline_paired = [baseline_by_fold[fold_id] for fold_id in aligned_fold_ids]
    differences = np.asarray(candidate_paired, dtype="float64") - np.asarray(baseline_paired, dtype="float64")
    if np.all(differences == 0.0):
        return MetricComparison(
            metric_name=metric_name, candidate_aggregate=candidate_aggregate, baseline_aggregate=baseline_aggregate,
            paired_n=paired_n, p_value=None, outcome=ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA,
            reason="candidate and baseline produced IDENTICAL values in every paired fold -- Wilcoxon is "
            "undefined for an all-zero-difference sample",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # scipy warns about ties/small samples; handled explicitly above
        _statistic, p_value = stats.wilcoxon(differences)
    p_value = float(p_value)

    higher_is_better = _is_higher_better(metric_name)
    candidate_is_better_direction = (
        (candidate_aggregate.mean > baseline_aggregate.mean) if higher_is_better
        else (candidate_aggregate.mean < baseline_aggregate.mean)
    )

    if p_value >= alpha:
        outcome = ComparisonOutcome.NO_SIGNIFICANT_DIFFERENCE
    elif candidate_is_better_direction:
        outcome = ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER
    else:
        outcome = ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_WORSE

    return MetricComparison(
        metric_name=metric_name, candidate_aggregate=candidate_aggregate, baseline_aggregate=baseline_aggregate,
        paired_n=paired_n, p_value=p_value, outcome=outcome,
    )


@dataclass(frozen=True, slots=True)
class RankedModel:
    model_name: str
    primary_metric_mean: float
    outperforms_all_baselines: bool
    comparison_report: ModelComparisonReport

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name, "primary_metric_mean": self.primary_metric_mean,
            "outperforms_all_baselines": self.outperforms_all_baselines,
            "comparison_report": self.comparison_report.to_json_dict(),
        }


@dataclass(frozen=True, slots=True)
class RankingReport:
    primary_metric: str
    ranked_models: tuple[RankedModel, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, object]:
        return {"primary_metric": self.primary_metric, "ranked_models": [m.to_json_dict() for m in self.ranked_models]}


def rank_candidates(
    candidates: Sequence[ModelFoldMetrics], baselines: Sequence[ModelFoldMetrics], *,
    primary_metric: str, alpha: float = _DEFAULT_ALPHA,
) -> RankingReport:
    """Ranks every candidate by `primary_metric`'s mean (best first, per
    that metric's own direction), each annotated with its full
    baseline-comparison report and the single `outperforms_all_baselines`
    gate. Ranking by a metric's raw mean is a simple, transparent, total
    order -- it does NOT itself imply statistical significance; that is
    exactly what `outperforms_all_baselines` is for, kept as a SEPARATE,
    explicit field so a reader never conflates "ranked first" with
    "proven better"."""
    higher_is_better = _is_higher_better(primary_metric)
    ranked: list[RankedModel] = []
    for candidate in candidates:
        values = candidate.values_for(primary_metric)
        if not values:
            raise ValueError(f"Candidate {candidate.model_name!r} never reported primary_metric {primary_metric!r} in any fold")
        comparison_report = compare_to_baselines(candidate, baselines, alpha=alpha)
        ranked.append(RankedModel(
            model_name=candidate.model_name, primary_metric_mean=float(np.mean(values)),
            outperforms_all_baselines=comparison_report.outperforms_all_baselines(primary_metric),
            comparison_report=comparison_report,
        ))
    ranked.sort(key=lambda m: m.primary_metric_mean, reverse=higher_is_better)
    return RankingReport(primary_metric=primary_metric, ranked_models=tuple(ranked))


__all__ = [
    "BaselineComparisonReport",
    "ComparisonOutcome",
    "MetricComparison",
    "ModelComparisonReport",
    "ModelFoldMetrics",
    "RankedModel",
    "RankingReport",
    "aggregate_metric_values",
    "compare_to_baseline",
    "compare_to_baselines",
    "rank_candidates",
]
