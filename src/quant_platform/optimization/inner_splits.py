"""Nested (inner) walk-forward split construction and independent
validation (Milestone 4D).

REUSE, NOT REIMPLEMENTATION
--------------------------------------------------------------------------
Every bar of purge/embargo/expanding-window arithmetic here is the SAME
code Milestone 4B's `execution.splitters` already uses: an inner split
plan is built by slicing one outer fold's `train_indices` into its own
chronologically-ordered sub-timeline, then calling `execution.splitters.
generate_expanding_folds`/`generate_rolling_folds` -- the exact functions
`execution.runner.ExecutionRunner` calls for OUTER folds -- a SECOND time,
against that sub-timeline. Nothing in this module hand-rolls a new gap
calculation; `validation.walk_forward.PurgedWalkForwardSplitter` (via
`execution.splitters`) is still the one splitting engine in this
codebase. Local positions the inner splitter returns (relative to the
length of the outer-train sub-timeline) are mapped back to GLOBAL
positions (relative to the full dataset timeline) via one line of numpy
fancy-indexing: `outer_fold.train_indices[local_position]`.

WHY INNER PURGE HAS NO SEPARATE, CALLER-DECLARED KNOB
--------------------------------------------------------------------------
Purge exists to satisfy one fact: a training row's label depends on
`label_horizon_bars` bars of future price data (see `execution.splitters.
required_label_purge_bars_for`'s exact off-by-one proof). That fact is
identical for inner splits and outer splits -- it comes from the SAME
bound research dataset's label definition, never a user-editable
parameter. `InnerSplitConfig` therefore has no `purge_bars` field at all;
`build_inner_fold_plan` always uses exactly `required_label_purge_bars_
for(label_horizon_bars)` as the inner purge, and only EMBARGO (a genuine
policy choice -- how much extra serial-correlation buffer beyond the
label horizon) is caller-configurable. This mirrors the outer engine's
own policy of REJECTING an insufficient declared purge rather than
silently widening it, taken one step further: for inner splits there is
no "declared" value to be insufficient in the first place.

THE ONE LEAKAGE CHECK THIS MODULE EXISTS FOR
--------------------------------------------------------------------------
`validate_nested_plan` is independent, defense-in-depth verification that
no inner fold -- not one row of it -- ever touches the outer fold's TEST
partition (or, for symmetry, the outer fold's own optional validation
carve-out, which is likewise reserved). This is checked by SET
INTERSECTION against `outer_fold.test_indices`/`outer_fold.
validation_indices` directly, never inferred from timestamp ordering
alone: a bug that let inner positions leak past the outer-train boundary
must be caught here even if it happened to preserve chronological order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FoldValidationError
from quant_platform.execution.splitters import (
    Fold,
    generate_expanding_folds,
    generate_rolling_folds,
    required_label_purge_bars_for,
)
from quant_platform.ml.models import JsonPrimitive, ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

INNER_SPLIT_SCHEMA_VERSION = 1

_ALLOWED_STRATEGIES = ("expanding_walk_forward", "rolling_walk_forward")


@dataclass(frozen=True, slots=True)
class InnerSplitConfig:
    """The identity-relevant, durable description of how inner splits are
    constructed from one outer fold's training partition. `test_size_
    fraction` (rather than a fixed row count) is deliberate: outer folds
    in an expanding-window plan have GROWING training partitions, so a
    fixed inner test size that fits a late outer fold's huge train
    partition would be far too large -- or entirely infeasible -- for an
    early outer fold's small one. A fraction scales automatically with
    each outer fold's own train size."""

    strategy: str
    n_splits: int
    test_size_fraction: float
    embargo_bars: int = 0
    max_train_size_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(f"InnerSplitConfig.strategy must be one of {_ALLOWED_STRATEGIES}, got {self.strategy!r}")
        if self.n_splits < 1:
            raise ValueError(f"InnerSplitConfig.n_splits must be >= 1, got {self.n_splits}")
        if not (0.0 < self.test_size_fraction < 1.0):
            raise ValueError(f"InnerSplitConfig.test_size_fraction must be in (0, 1), got {self.test_size_fraction}")
        if self.embargo_bars < 0:
            raise ValueError(f"InnerSplitConfig.embargo_bars must be >= 0, got {self.embargo_bars}")
        if self.max_train_size_fraction is not None and not (0.0 < self.max_train_size_fraction <= 1.0):
            raise ValueError(f"InnerSplitConfig.max_train_size_fraction must be in (0, 1], got {self.max_train_size_fraction}")
        if self.strategy != "rolling_walk_forward" and self.max_train_size_fraction is not None:
            raise ValueError("InnerSplitConfig.max_train_size_fraction is only meaningful for strategy='rolling_walk_forward'")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy, "n_splits": self.n_splits, "test_size_fraction": self.test_size_fraction,
            "embargo_bars": self.embargo_bars, "max_train_size_fraction": self.max_train_size_fraction,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InnerSplitConfig:
        max_train_raw = raw.get("max_train_size_fraction")
        return cls(
            strategy=str(raw["strategy"]), n_splits=int(str(raw["n_splits"])),
            test_size_fraction=float(str(raw["test_size_fraction"])), embargo_bars=int(str(raw.get("embargo_bars", 0))),
            max_train_size_fraction=(None if max_train_raw is None else float(str(max_train_raw))),
        )


@dataclass(frozen=True, slots=True)
class InnerFold:
    """One inner (nested) fold's row positions -- ALWAYS in GLOBAL
    (full-dataset-timeline) positions, never local-to-the-outer-train-
    partition ones, so a persisted `InnerFold` is independently auditable
    against the dataset's own timeline without needing the outer fold's
    own indices to decode it."""

    inner_fold_index: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.inner_fold_index < 0:
            raise FoldValidationError(f"InnerFold.inner_fold_index must be >= 0, got {self.inner_fold_index}")
        if len(self.train_indices) == 0:
            raise FoldValidationError(f"InnerFold {self.inner_fold_index}: train_indices must not be empty")
        if len(self.validation_indices) == 0:
            raise FoldValidationError(f"InnerFold {self.inner_fold_index}: validation_indices must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "inner_fold_index": self.inner_fold_index,
            "train_indices": [int(i) for i in self.train_indices.tolist()],
            "validation_indices": [int(i) for i in self.validation_indices.tolist()],
            "train_start": self.train_start.isoformat(), "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(), "validation_end": self.validation_end.isoformat(),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InnerFold:
        return cls(
            inner_fold_index=int(str(raw["inner_fold_index"])),
            train_indices=np.asarray([int(i) for i in as_json_list(raw["train_indices"], field_name="train_indices")], dtype=np.int64),
            validation_indices=np.asarray(
                [int(i) for i in as_json_list(raw["validation_indices"], field_name="validation_indices")], dtype=np.int64
            ),
            train_start=pd.Timestamp(str(raw["train_start"])), train_end=pd.Timestamp(str(raw["train_end"])),
            validation_start=pd.Timestamp(str(raw["validation_start"])), validation_end=pd.Timestamp(str(raw["validation_end"])),
        )


@dataclass(frozen=True, slots=True)
class InnerFoldPlan:
    schema_version: int
    outer_fold_index: int
    strategy: str
    inner_folds: tuple[InnerFold, ...]
    purge_bars: int
    embargo_bars: int
    label_horizon_bars: int
    required_label_purge_bars: int
    outer_train_row_count: int

    def __post_init__(self) -> None:
        if not self.inner_folds:
            raise FoldValidationError(f"InnerFoldPlan for outer fold {self.outer_fold_index}: must contain at least one inner fold")
        indices = [f.inner_fold_index for f in self.inner_folds]
        if indices != list(range(len(indices))):
            raise FoldValidationError(f"InnerFoldPlan.inner_folds inner_fold_index values must be exactly 0..N-1 in order, got {indices}")
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise FoldValidationError("InnerFoldPlan.purge_bars/embargo_bars must be >= 0")
        if self.required_label_purge_bars != required_label_purge_bars_for(self.label_horizon_bars):
            raise FoldValidationError("InnerFoldPlan.required_label_purge_bars is inconsistent with label_horizon_bars")
        if self.purge_bars != self.required_label_purge_bars:
            raise FoldValidationError(
                f"InnerFoldPlan.purge_bars ({self.purge_bars}) must exactly equal required_label_purge_bars "
                f"({self.required_label_purge_bars}) -- inner splits have no separate, caller-declared purge"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "strategy": self.strategy,
            "inner_folds": [f.to_json_dict() for f in self.inner_folds], "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars, "label_horizon_bars": self.label_horizon_bars,
            "required_label_purge_bars": self.required_label_purge_bars, "outer_train_row_count": self.outer_train_row_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InnerFoldPlan:
        require_schema_version(raw, supported=INNER_SPLIT_SCHEMA_VERSION, context="InnerFoldPlan")
        return cls(
            schema_version=INNER_SPLIT_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            strategy=str(raw["strategy"]),
            inner_folds=tuple(
                InnerFold.from_json_dict(as_json_dict(f, field_name="inner_folds[]"))
                for f in as_json_list(raw["inner_folds"], field_name="inner_folds")
            ),
            purge_bars=int(str(raw["purge_bars"])), embargo_bars=int(str(raw["embargo_bars"])),
            label_horizon_bars=int(str(raw["label_horizon_bars"])), required_label_purge_bars=int(str(raw["required_label_purge_bars"])),
            outer_train_row_count=int(str(raw["outer_train_row_count"])),
        )


def build_inner_fold_plan(
    outer_fold: Fold, *, config: InnerSplitConfig, label_horizon_bars: int, timeline: pd.DataFrame, timestamp_column: str = "open_time",
) -> InnerFoldPlan:
    """Builds one outer fold's complete inner (nested) split plan. `timeline`
    is the SAME fully-reconstructed dataset timeline the outer `FoldPlan`
    was built against (`execution.splitters.reconstruct_dataset_timeline`);
    only `outer_fold.train_indices` rows of it are ever touched here --
    `timeline.iloc[outer_fold.test_indices]` (and, if present, `.
    validation_indices`) are never read by this function at all, which is
    itself one layer of the outer-test isolation guarantee (the bytes are
    simply never in scope, not merely "checked and then ignored")."""
    outer_train_positions = outer_fold.train_indices
    n = len(outer_train_positions)
    sub_timestamps = timeline[timestamp_column].iloc[outer_train_positions].reset_index(drop=True)

    required_purge = required_label_purge_bars_for(label_horizon_bars)
    test_size = max(1, round(n * config.test_size_fraction))

    if config.strategy == "expanding_walk_forward":
        local_plan = generate_expanding_folds(
            sub_timestamps, n_splits=config.n_splits, test_size=test_size, label_horizon_bars=label_horizon_bars,
            purge_bars=required_purge, embargo_bars=config.embargo_bars,
        )
    else:
        max_train_size = None if config.max_train_size_fraction is None else max(1, round(n * config.max_train_size_fraction))
        if max_train_size is None:
            raise ValueError("InnerSplitConfig.strategy='rolling_walk_forward' requires max_train_size_fraction to be set")
        local_plan = generate_rolling_folds(
            sub_timestamps, n_splits=config.n_splits, test_size=test_size, max_train_size=max_train_size,
            label_horizon_bars=label_horizon_bars, purge_bars=required_purge, embargo_bars=config.embargo_bars,
        )

    inner_folds = tuple(
        InnerFold(
            inner_fold_index=local_fold.fold_index,
            train_indices=outer_train_positions[local_fold.train_indices],
            validation_indices=outer_train_positions[local_fold.test_indices],
            train_start=local_fold.train_start, train_end=local_fold.train_end,
            validation_start=local_fold.test_start, validation_end=local_fold.test_end,
        )
        for local_fold in local_plan.folds
    )
    return InnerFoldPlan(
        schema_version=INNER_SPLIT_SCHEMA_VERSION, outer_fold_index=outer_fold.fold_index, strategy=config.strategy,
        inner_folds=inner_folds, purge_bars=required_purge, embargo_bars=config.embargo_bars,
        label_horizon_bars=label_horizon_bars, required_label_purge_bars=required_purge, outer_train_row_count=n,
    )


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def validate_nested_plan(outer_fold: Fold, inner_plan: InnerFoldPlan) -> ValidationReport:
    """Independent, defense-in-depth verification of one outer fold's
    `InnerFoldPlan` -- re-checked from the persisted `InnerFold` row
    positions alone, never trusting that `build_inner_fold_plan`'s own
    construction was correct. See module docstring for the single most
    important guarantee this enforces: no inner row ever reaches the
    outer fold's test (or validation) partition."""
    issues: list[ValidationIssue] = []
    issues += _validate_subset_of_outer_train(outer_fold, inner_plan)
    issues += _validate_no_outer_test_leakage(outer_fold, inner_plan)
    issues += _validate_chronology_and_gaps(inner_plan)
    issues += _validate_no_cross_fold_validation_overlap(inner_plan)
    return ValidationReport(schema_version=1, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _validate_subset_of_outer_train(outer_fold: Fold, inner_plan: InnerFoldPlan) -> list[ValidationIssue]:
    outer_train_set = set(outer_fold.train_indices.tolist())
    issues: list[ValidationIssue] = []
    for f in inner_plan.inner_folds:
        train_extra = set(f.train_indices.tolist()) - outer_train_set
        validation_extra = set(f.validation_indices.tolist()) - outer_train_set
        if train_extra:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_train_outside_outer_train",
                f"Inner fold {f.inner_fold_index}: {len(train_extra)} train row position(s) are not part of "
                f"outer fold {outer_fold.fold_index}'s train partition", inner_fold_index=f.inner_fold_index,
            ))
        if validation_extra:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_validation_outside_outer_train",
                f"Inner fold {f.inner_fold_index}: {len(validation_extra)} validation row position(s) are not "
                f"part of outer fold {outer_fold.fold_index}'s train partition", inner_fold_index=f.inner_fold_index,
            ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "inner_rows_subset_of_outer_train",
            f"Every inner row of all {len(inner_plan.inner_folds)} inner fold(s) is a subset of outer fold "
            f"{outer_fold.fold_index}'s train partition",
        ))
    return issues


def _validate_no_outer_test_leakage(outer_fold: Fold, inner_plan: InnerFoldPlan) -> list[ValidationIssue]:
    """THE single most safety-critical check in this module: no inner
    row -- train or validation -- may ever coincide with the outer
    fold's OWN test partition (or its optional validation carve-out,
    likewise reserved and never available to inner splitting)."""
    outer_reserved = set(outer_fold.test_indices.tolist()) | set(outer_fold.validation_indices.tolist())
    issues: list[ValidationIssue] = []
    for f in inner_plan.inner_folds:
        leaked_train = set(f.train_indices.tolist()) & outer_reserved
        leaked_validation = set(f.validation_indices.tolist()) & outer_reserved
        if leaked_train:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_train_touches_outer_reserved_rows",
                f"Inner fold {f.inner_fold_index}: train partition contains {len(leaked_train)} row position(s) "
                f"reserved for outer fold {outer_fold.fold_index}'s test/validation partition -- this is a "
                "leakage violation", inner_fold_index=f.inner_fold_index,
            ))
        if leaked_validation:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_validation_touches_outer_reserved_rows",
                f"Inner fold {f.inner_fold_index}: validation partition contains {len(leaked_validation)} row "
                f"position(s) reserved for outer fold {outer_fold.fold_index}'s test/validation partition -- "
                "this is a leakage violation", inner_fold_index=f.inner_fold_index,
            ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "no_outer_test_leakage",
            f"No inner fold of outer fold {outer_fold.fold_index} touches any row reserved for that outer "
            "fold's test/validation partition",
        ))
    return issues


def _validate_chronology_and_gaps(inner_plan: InnerFoldPlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_gap = inner_plan.purge_bars + inner_plan.embargo_bars
    for f in inner_plan.inner_folds:
        if not np.all(np.diff(f.train_indices) > 0) or not np.all(np.diff(f.validation_indices) > 0):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_fold_not_chronological",
                f"Inner fold {f.inner_fold_index}: train/validation row positions are not strictly ascending",
                inner_fold_index=f.inner_fold_index,
            ))
            continue
        gap = int(f.validation_indices.min()) - int(f.train_indices.max()) - 1
        if int(f.train_indices.max()) >= int(f.validation_indices.min()):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_train_not_before_validation",
                f"Inner fold {f.inner_fold_index}: train data is not entirely before validation data",
                inner_fold_index=f.inner_fold_index,
            ))
        elif gap < required_gap:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "inner_insufficient_purge_embargo_gap",
                f"Inner fold {f.inner_fold_index}: train/validation gap is {gap} bar(s), less than the "
                f"required purge+embargo of {required_gap} bar(s)", inner_fold_index=f.inner_fold_index,
                actual_gap=gap, required_gap=required_gap,
            ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "inner_chronology_and_gaps_consistent",
            f"All {len(inner_plan.inner_folds)} inner fold(s) are chronologically ordered with a sufficient "
            f"train/validation gap (>= {required_gap} bar(s))",
        ))
    return issues


def _validate_no_cross_fold_validation_overlap(inner_plan: InnerFoldPlan) -> list[ValidationIssue]:
    """"No duplicate inner validation rows unless explicitly allowed" --
    this milestone allows none: every inner validation row must belong to
    exactly one inner fold's validation partition, mirroring `execution.
    execution_validation._validate_no_overlap`'s identical cross-fold
    test-set-disjointness check at the outer level."""
    seen: dict[int, int] = {}
    issues: list[ValidationIssue] = []
    for f in inner_plan.inner_folds:
        for position in f.validation_indices.tolist():
            if position in seen:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "cross_inner_fold_validation_overlap",
                    f"Row position {position} appears in the validation set of both inner fold {seen[position]} "
                    f"and inner fold {f.inner_fold_index}", position=position, first_fold=seen[position], second_fold=f.inner_fold_index,
                ))
            else:
                seen[position] = f.inner_fold_index
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "no_cross_inner_fold_validation_overlap",
            f"No row position appears in more than one inner fold's validation partition across all "
            f"{len(inner_plan.inner_folds)} inner fold(s)",
        ))
    return issues


__all__ = [
    "INNER_SPLIT_SCHEMA_VERSION",
    "InnerFold",
    "InnerFoldPlan",
    "InnerSplitConfig",
    "build_inner_fold_plan",
    "validate_nested_plan",
]
