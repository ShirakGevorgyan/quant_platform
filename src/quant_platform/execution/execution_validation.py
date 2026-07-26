"""Fold-plan validation (Milestone 4B, Section 13) -- does a `FoldPlan`
ACTUALLY satisfy the time-safety guarantees its generator claims, checked
independently of how it was built. Mirrors `ml.validation`'s own
philosophy exactly: every check contributes one `ValidationIssue`
(reusing `ml.models.ValidationIssue`/`ValidationReport`/
`ValidationSeverity` rather than inventing a parallel reporting type),
`ValidationReport.is_ready` is the single authoritative gate, and this
function NEVER raises for a bad fold plan -- only genuinely unusable
input (e.g. a fold plan built against the wrong timeline entirely)
surfaces as a `CRITICAL` issue rather than an exception.

This is DEFENSE IN DEPTH, not redundant paranoia: `execution.splitters`'
generator functions already construct chronologically-safe folds by
construction (via the already-tested `PurgedWalkForwardSplitter`) -- this
module exists so a `FoldPlan` reaching the runner from ANY source (a
generator, a future alternative constructor, a corrupted/tampered
resume checkpoint) is re-verified before a single fold actually executes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.execution.splitters import Fold, FoldPlan
from quant_platform.ml.models import JsonPrimitive, ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def validate_fold_plan(
    plan: FoldPlan, *, timeline: pd.DataFrame, timestamp_column: str = "open_time"
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    issues += _validate_chronology(plan)
    issues += _validate_label_horizon_purge(plan)
    issues += _validate_purge_embargo(plan)
    issues += _validate_no_overlap(plan)
    issues += _validate_no_empty_folds(plan)
    issues += _validate_dataset_compatibility(plan, timeline=timeline, timestamp_column=timestamp_column)
    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _is_sorted_ascending(indices: np.ndarray) -> bool:
    return len(indices) <= 1 or bool(np.all(np.diff(indices) > 0))


def _validate_chronology(plan: FoldPlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for f in plan.folds:
        if not _is_sorted_ascending(f.train_indices):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_train_not_chronological",
                f"Fold {f.fold_index}: train_indices are not strictly ascending", fold_index=f.fold_index,
            ))
        if len(f.validation_indices) and not _is_sorted_ascending(f.validation_indices):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_validation_not_chronological",
                f"Fold {f.fold_index}: validation_indices are not strictly ascending", fold_index=f.fold_index,
            ))
        if not _is_sorted_ascending(f.test_indices):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_test_not_chronological",
                f"Fold {f.fold_index}: test_indices are not strictly ascending", fold_index=f.fold_index,
            ))
        if int(f.train_indices.max()) >= int(f.test_indices.min()):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_train_not_before_test",
                f"Fold {f.fold_index}: train data is not entirely before test data "
                f"(train max position {int(f.train_indices.max())} >= test min position {int(f.test_indices.min())})",
                fold_index=f.fold_index,
            ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "fold_chronology_consistent",
            f"All {len(plan.folds)} fold(s) have strictly chronological, non-overlapping train/test ordering",
        ))
    return issues


def _validate_label_horizon_purge(plan: FoldPlan) -> list[ValidationIssue]:
    """The label-information leakage gate (the fix for the audit finding
    that this engine used to derive its required gap ONLY from
    user-declared `purge_bars`/`embargo_bars`, never checking the real
    dataset's label horizon). Unlike `_validate_purge_embargo` below
    (which checks the ACTUAL gap the constructed folds deliver against
    the plan's OWN declared `purge_bars`/`embargo_bars` -- a
    construction-integrity check), this checks the DECLARED `purge_bars`
    itself against the dataset's TRUE, manifest-derived
    `required_label_purge_bars` (see `splitters.
    required_label_purge_bars_for` for the exact off-by-one proof) -- a
    scientific-sufficiency check. Together the two checks transitively
    guarantee: actual fold gap >= declared purge_bars >=
    required_label_purge_bars, for every partition boundary
    `_validate_purge_embargo` covers (train/test, train/validation,
    validation/test).

    This milestone's chosen policy is REJECTION, not silent widening: if
    the user-declared `purge_bars` is insufficient for the bound
    dataset's real label horizon, the execution is refused outright
    (CRITICAL) rather than silently run with a larger, different purge
    than the one the `ExperimentSpec` identity payload describes -- see
    `docs/execution_engine.md`'s "Label-information purge" section for
    the full policy rationale (the alternative, `effective_purge_bars =
    max(declared, required)`, was deliberately rejected: it would let
    two runs sharing the same identity payload execute materially
    different split semantics whenever `required_label_purge_bars`
    exceeded the declared value)."""
    issues: list[ValidationIssue] = []
    if plan.purge_bars < plan.required_label_purge_bars:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "insufficient_label_horizon_purge",
            f"Declared purge_bars={plan.purge_bars} is less than this dataset's required label-information "
            f"purge of {plan.required_label_purge_bars} bar(s) (label_horizon_bars={plan.label_horizon_bars}) -- "
            "a training sample's label would depend on data at or after the first validation/test row. "
            "Increase SplitBinding.params['purge_bars'] to at least the required value; this execution is "
            "rejected rather than silently run with a larger purge than declared.",
            declared_purge_bars=plan.purge_bars, required_label_purge_bars=plan.required_label_purge_bars,
            label_horizon_bars=plan.label_horizon_bars,
        ))
    else:
        issues.append(_issue(
            ValidationSeverity.INFO, "label_horizon_purge_sufficient",
            f"Declared purge_bars={plan.purge_bars} meets or exceeds the required label-information purge of "
            f"{plan.required_label_purge_bars} bar(s) for label_horizon_bars={plan.label_horizon_bars}",
        ))
    return issues


def _validate_purge_embargo(plan: FoldPlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_train_test_gap = plan.purge_bars + plan.embargo_bars
    for f in plan.folds:
        train_test_gap = int(f.test_indices.min()) - int(f.train_indices.max()) - 1
        if train_test_gap < required_train_test_gap:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "insufficient_purge_embargo_gap",
                f"Fold {f.fold_index}: train/test gap is {train_test_gap} bar(s), less than the "
                f"required purge+embargo of {required_train_test_gap} bar(s)",
                fold_index=f.fold_index, actual_gap=train_test_gap, required_gap=required_train_test_gap,
            ))
        if len(f.validation_indices):
            train_val_gap = int(f.validation_indices.min()) - int(f.train_indices.max()) - 1
            if train_val_gap < plan.purge_bars:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "insufficient_purge_gap_train_validation",
                    f"Fold {f.fold_index}: train/validation gap is {train_val_gap} bar(s), less than "
                    f"the required purge of {plan.purge_bars} bar(s)",
                    fold_index=f.fold_index, actual_gap=train_val_gap, required_gap=plan.purge_bars,
                ))
            val_test_gap = int(f.test_indices.min()) - int(f.validation_indices.max()) - 1
            if val_test_gap < required_train_test_gap:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "insufficient_purge_embargo_gap_validation_test",
                    f"Fold {f.fold_index}: validation/test gap is {val_test_gap} bar(s), less than the "
                    f"required purge+embargo of {required_train_test_gap} bar(s)",
                    fold_index=f.fold_index, actual_gap=val_test_gap, required_gap=required_train_test_gap,
                ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "purge_embargo_gaps_sufficient",
            f"Every fold's train/validation/test gap meets or exceeds purge_bars={plan.purge_bars} "
            f"+ embargo_bars={plan.embargo_bars}",
        ))
    return issues


def _fold_sets(f: Fold) -> tuple[set[int], set[int], set[int]]:
    return set(f.train_indices.tolist()), set(f.validation_indices.tolist()), set(f.test_indices.tolist())


def _validate_no_overlap(plan: FoldPlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for f in plan.folds:
        train_set, validation_set, test_set = _fold_sets(f)
        if train_set & validation_set:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_train_validation_overlap",
                f"Fold {f.fold_index}: train and validation row positions overlap", fold_index=f.fold_index,
            ))
        if train_set & test_set:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_train_test_overlap",
                f"Fold {f.fold_index}: train and test row positions overlap", fold_index=f.fold_index,
            ))
        if validation_set & test_set:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "fold_validation_test_overlap",
                f"Fold {f.fold_index}: validation and test row positions overlap", fold_index=f.fold_index,
            ))

    # Test blocks across DIFFERENT folds must never overlap each other --
    # unlike train blocks, which are EXPECTED to overlap across folds for
    # expanding/grouped strategies (each fold's train legitimately grows
    # to include earlier folds' train+test history) and are therefore
    # deliberately NOT checked here.
    seen: dict[int, int] = {}
    for f in plan.folds:
        for position in f.test_indices.tolist():
            if position in seen:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "cross_fold_test_overlap",
                    f"Row position {position} appears in the test set of both fold {seen[position]} "
                    f"and fold {f.fold_index}",
                    position=position, first_fold=seen[position], second_fold=f.fold_index,
                ))
            else:
                seen[position] = f.fold_index
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "fold_partitions_disjoint",
            "No within-fold train/validation/test overlap and no cross-fold test-set overlap detected",
        ))
    return issues


def _validate_no_empty_folds(plan: FoldPlan) -> list[ValidationIssue]:
    # `Fold.__post_init__` already refuses to construct an empty
    # train/test split -- reported here (INFO, confirmed-not-violated)
    # for a complete audit trail, exactly like `ml.validation`'s
    # already-structurally-guaranteed checks.
    empty = [f.fold_index for f in plan.folds if len(f.train_indices) == 0 or len(f.test_indices) == 0]
    if empty:
        return [_issue(  # pragma: no cover - unreachable given Fold's own construction-time guarantee
            ValidationSeverity.CRITICAL, "empty_fold", f"Fold(s) {empty} have an empty train or test set",
            fold_indices=", ".join(str(i) for i in empty),
        )]
    return [_issue(
        ValidationSeverity.INFO, "no_empty_folds", f"All {len(plan.folds)} fold(s) have non-empty train and test sets",
    )]


def _validate_dataset_compatibility(
    plan: FoldPlan, *, timeline: pd.DataFrame, timestamp_column: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if plan.total_rows != len(timeline):
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "fold_plan_row_count_mismatch",
            f"FoldPlan.total_rows={plan.total_rows} does not match the provided timeline's "
            f"{len(timeline)} row(s) -- this plan was not built against this timeline",
            plan_total_rows=plan.total_rows, timeline_row_count=len(timeline),
        ))
    max_position = max(
        (int(f.test_indices.max()) for f in plan.folds), default=-1,
    )
    if max_position >= len(timeline):
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "fold_plan_position_out_of_bounds",
            f"A fold references row position {max_position}, but the timeline only has {len(timeline)} row(s)",
            max_referenced_position=max_position, timeline_row_count=len(timeline),
        ))
    if timestamp_column not in timeline.columns:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "timeline_missing_timestamp_column",
            f"timeline is missing the expected {timestamp_column!r} column",
        ))
    else:
        ts = timeline[timestamp_column]
        dup_count = int(ts.duplicated().sum())
        if dup_count > 0:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "timeline_duplicate_timestamps",
                f"timeline has {dup_count} duplicate {timestamp_column!r} value(s)", duplicate_count=dup_count,
            ))
        if len(ts) > 1 and not pd.DatetimeIndex(pd.Index(ts)).is_monotonic_increasing:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "timeline_out_of_order",
                f"timeline's {timestamp_column!r} column is not monotonically increasing",
            ))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "fold_plan_dataset_compatible",
            f"FoldPlan is compatible with the provided timeline ({len(timeline)} row(s), no duplicate/"
            "out-of-order timestamps)",
        ))
    return issues


__all__ = ["validate_fold_plan"]
