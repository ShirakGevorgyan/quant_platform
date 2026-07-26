"""Explicit missing-value policies for feature columns.

Every policy is enumerated in `models.MissingPolicyKind`; there is
deliberately NO backward-fill option anywhere in this module -- "unrestricted
backward fill" (Milestone 3 Section 7's first forbidden item) is not a
policy this function forgot to restrict, it structurally does not exist as
a callable code path.

`TRAINING_STATISTIC_FILL` never computes its own statistic: `apply_missing_
policy` REQUIRES a `fitted_statistic` argument for that policy, and this
module provides exactly one function (`compute_training_statistic`) capable
of producing one. Nothing in this module's types can stop a careless caller
from computing that statistic over a validation/test slice by mistake --
the guarantee comes from `dataset_builder.ResearchDatasetBuilder`'s own
fit/apply ordering (fit only ever runs against the train split; see
`tests/unit/features/test_missing.py::test_training_statistic_fill_never_reflects_validation_data`
for the adversarial proof of that ordering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_platform.features.models import MissingPolicyKind, MissingPolicySpec


@dataclass(frozen=True, slots=True)
class MissingApplicationResult:
    values: pd.Series
    missing_indicator: pd.Series | None
    rows_to_drop: pd.Index | None
    fitted_statistic_used: float | None


def compute_training_statistic(train_series: pd.Series, *, statistic: Literal["mean", "median"] = "mean") -> float:
    """The ONLY function in this module that computes a fill value from
    data. Callers must pass ONLY a training-partition slice -- see module
    docstring."""
    clean = train_series.dropna()
    if statistic == "mean":
        return float(clean.mean()) if len(clean) else 0.0
    if statistic == "median":
        return float(clean.median()) if len(clean) else 0.0
    raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")


def _forward_fill_with_max_age(series: pd.Series, max_age_bars: int) -> pd.Series:
    """Forward-fill NaNs, but null back out any fill that is more than
    `max_age_bars` positions stale relative to the last actually-observed
    value. `age == 0` always refers to the original observation itself
    (never nulled); `age == k` is the k-th consecutive bar since that
    observation."""
    valid_mask = series.notna()
    group_id = valid_mask.cumsum()
    age = series.groupby(group_id).cumcount()
    filled = series.ffill()
    too_old = age > max_age_bars
    return filled.where(~too_old)


def apply_missing_policy(
    series: pd.Series, *, policy: MissingPolicySpec, fitted_statistic: float | None = None
) -> MissingApplicationResult:
    """Apply `policy` to `series`, returning the (possibly filled) values
    plus an optional missing-indicator series and, for `DROP_ROW`, the
    positional index labels the caller should drop consistently across
    every column of the dataset (this function only ever operates on one
    column at a time; it is the dataset builder's job to drop the same
    rows everywhere so the feature matrix stays rectangular)."""
    if policy.kind is MissingPolicyKind.PRESERVE_NULL:
        filled = series
        stat_used = None
    elif policy.kind is MissingPolicyKind.FORWARD_FILL_MAX_AGE:
        assert policy.max_age_bars is not None
        filled = _forward_fill_with_max_age(series, policy.max_age_bars)
        stat_used = None
    elif policy.kind is MissingPolicyKind.CONSTANT_FILL:
        assert policy.constant_value is not None
        filled = series.fillna(policy.constant_value)
        stat_used = None
    elif policy.kind is MissingPolicyKind.TRAINING_STATISTIC_FILL:
        if fitted_statistic is None:
            raise ValueError(
                "apply_missing_policy: fitted_statistic is required for TRAINING_STATISTIC_FILL. "
                "It must be computed via compute_training_statistic() against a training-partition "
                "slice ONLY, and passed in here explicitly -- this function never computes it itself."
            )
        filled = series.fillna(fitted_statistic)
        stat_used = fitted_statistic
    elif policy.kind is MissingPolicyKind.DROP_ROW:
        filled = series
        stat_used = None
    else:  # pragma: no cover - exhaustive over MissingPolicyKind
        raise AssertionError(f"Unhandled MissingPolicyKind: {policy.kind}")

    indicator = series.isna() if policy.add_missing_indicator else None
    rows_to_drop = series.index[series.isna()] if policy.kind is MissingPolicyKind.DROP_ROW else None

    return MissingApplicationResult(
        values=filled, missing_indicator=indicator, rows_to_drop=rows_to_drop, fitted_statistic_used=stat_used
    )


def missingness_summary(df: pd.DataFrame) -> dict[str, object]:
    """A dataset-level missing-data summary: per-column null count/fraction
    plus the overall row count -- used both by `dataset_builder` (recorded
    in the manifest) and `validation` (excessive-missingness checks)."""
    null_counts = df.isna().sum()
    row_count = len(df)
    return {
        "row_count": row_count,
        "columns": {
            str(col): {
                "null_count": int(null_counts[col]),
                "null_fraction": float(null_counts[col]) / row_count if row_count else 0.0,
            }
            for col in df.columns
        },
    }


__all__ = ["MissingApplicationResult", "apply_missing_policy", "compute_training_statistic", "missingness_summary"]
