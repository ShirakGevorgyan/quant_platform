"""Objective/primary-metric policy (Milestone 4D).

"Reuse the Milestone 4C authoritative metric-direction registry. Do not
create a conflicting second registry." Every direction decision in this
package flows through `ml.comparison.is_higher_better`/`known_metric_names`
-- this module adds compatibility validation (which metric names make
sense for which objective) and the undefined-primary-metric-on-a-trial
policy on top of that registry, never a second one.

UNDEFINED METRICS ARE NEVER CONVERTED TO ZERO
--------------------------------------------------------------------------
`ml.metrics.compute_metrics` already OMITS (never zero-fills) a metric
that is mathematically undefined for a given fold's label domain (see
that module's own docstring: ROC AUC/PR AUC/balanced accuracy/MCC/R^2 can
all be legitimately absent for a walk-forward fold with degenerate
labels). `aggregate_primary_metric` below extends that same discipline to
the INNER-FOLD-to-TRIAL aggregation step: an inner fold that never
produced the primary metric contributes NOTHING to the trial's aggregate
-- never a `0.0` silently dragging the mean in one direction. If too FEW
inner folds produced it (below `min_successful_inner_folds`), the whole
trial is invalid, per explicit policy, not partially scored.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from quant_platform.ml.comparison import is_higher_better, known_metric_names
from quant_platform.ml.metrics import CLASSIFICATION_METRIC_NAMES, REGRESSION_METRIC_NAMES
from quant_platform.ml.models import ObjectiveType

_METRIC_NAMES_BY_OBJECTIVE: dict[ObjectiveType, frozenset[str]] = {
    ObjectiveType.BINARY_CLASSIFICATION: frozenset(CLASSIFICATION_METRIC_NAMES),
    ObjectiveType.MULTICLASS_CLASSIFICATION: frozenset(CLASSIFICATION_METRIC_NAMES),
    ObjectiveType.REGRESSION: frozenset(REGRESSION_METRIC_NAMES),
}


def validate_primary_metric(objective: ObjectiveType, primary_metric: str) -> None:
    """Raises `ValueError` unless `primary_metric` is BOTH declared in the
    authoritative `ml.comparison` direction registry AND compatible with
    `objective` (a regression metric for a classification objective, or
    vice versa, is always a caller error, never silently accepted)."""
    if primary_metric not in known_metric_names():
        raise ValueError(
            f"primary_metric {primary_metric!r} is not declared in the authoritative ml.comparison metric-"
            f"direction registry -- known metrics: {sorted(known_metric_names())}"
        )
    allowed = _METRIC_NAMES_BY_OBJECTIVE.get(objective)
    if allowed is not None and primary_metric not in allowed:
        raise ValueError(
            f"primary_metric {primary_metric!r} is not compatible with objective {objective.value!r} -- "
            f"expected one of {sorted(allowed)}"
        )


def metric_direction_multiplier(primary_metric: str) -> int:
    """`+1` for a higher-is-better metric, `-1` for lower-is-better --
    lets ranking/pruning code write one direction-agnostic comparison
    (`multiplier * value`) instead of branching on direction at every
    call site."""
    return 1 if is_higher_better(primary_metric) else -1


@dataclass(frozen=True, slots=True)
class PrimaryMetricAggregationOutcome:
    is_valid: bool
    aggregate_value: float | None
    successful_inner_folds: int
    total_inner_folds: int
    reason: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid, "aggregate_value": self.aggregate_value,
            "successful_inner_folds": self.successful_inner_folds, "total_inner_folds": self.total_inner_folds,
            "reason": self.reason,
        }


def aggregate_primary_metric(
    per_inner_fold_values: Sequence[float | None], *, min_successful_inner_folds: int,
) -> PrimaryMetricAggregationOutcome:
    """`per_inner_fold_values[i]` is `None` exactly when that inner fold's
    `MetricComputationReport` skipped the primary metric (undefined for
    that fold's label domain) -- see module docstring. A trial is valid
    iff at least `min_successful_inner_folds` inner folds produced a
    value; the aggregate is the plain mean over exactly those folds."""
    if not per_inner_fold_values:
        raise ValueError("aggregate_primary_metric requires at least one inner-fold value slot")
    successful = [v for v in per_inner_fold_values if v is not None]
    if len(successful) < min_successful_inner_folds:
        return PrimaryMetricAggregationOutcome(
            is_valid=False, aggregate_value=None, successful_inner_folds=len(successful),
            total_inner_folds=len(per_inner_fold_values),
            reason=f"only {len(successful)}/{len(per_inner_fold_values)} inner fold(s) produced the primary "
            f"metric, below the required minimum of {min_successful_inner_folds}",
        )
    return PrimaryMetricAggregationOutcome(
        is_valid=True, aggregate_value=statistics.fmean(successful), successful_inner_folds=len(successful),
        total_inner_folds=len(per_inner_fold_values),
    )


__all__ = [
    "PrimaryMetricAggregationOutcome",
    "aggregate_primary_metric",
    "metric_direction_multiplier",
    "validate_primary_metric",
]
