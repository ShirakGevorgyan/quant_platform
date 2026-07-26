from __future__ import annotations

import numpy as np
import pandas as pd
from tests.unit.execution.conftest import make_timeline

from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.splitters import (
    Fold,
    FoldPlan,
    generate_expanding_folds,
    required_label_purge_bars_for,
)
from quant_platform.ml.models import ValidationSeverity

_TS = pd.Timestamp("2024-01-01", tz="UTC")


def _fold(fold_index: int, train: np.ndarray, test: np.ndarray, *, validation: np.ndarray | None = None) -> Fold:
    return Fold(
        fold_index=fold_index, train_indices=train, test_indices=test,
        train_start=_TS, train_end=_TS, test_start=_TS, test_end=_TS,
        validation_indices=(validation if validation is not None else np.array([], dtype=np.int64)),
        validation_start=(None if validation is None or len(validation) == 0 else _TS),
        validation_end=(None if validation is None or len(validation) == 0 else _TS),
    )


def _plan(*, purge_bars: int, embargo_bars: int, total_rows: int, folds: tuple[Fold, ...], label_horizon_bars: int = 0) -> FoldPlan:
    """Builds a `FoldPlan` for tests that are NOT about the label-horizon-
    purge check itself -- `label_horizon_bars` defaults to 0 (whose
    required purge is always 0, per `required_label_purge_bars_for`), so
    every pre-existing purge/embargo/chronology/overlap/compatibility
    check below stays decoupled from the label-horizon check's own,
    separately-tested behavior (see `TestLabelHorizonPurgeCheck`)."""
    return FoldPlan(
        strategy="x", purge_bars=purge_bars, embargo_bars=embargo_bars, total_rows=total_rows, folds=folds,
        label_horizon_bars=label_horizon_bars, required_label_purge_bars=required_label_purge_bars_for(label_horizon_bars),
    )


def _timeline(n: int = 1000) -> pd.DataFrame:
    return make_timeline(n)


class TestValidPlanPassesCleanly:
    def test_generator_output_always_validates_clean(self) -> None:
        timeline = _timeline()
        plan = generate_expanding_folds(
            timeline["open_time"], n_splits=4, test_size=100, purge_bars=5, embargo_bars=2,
            validation_fraction=0.2, label_horizon_bars=5,
        )
        report = validate_fold_plan(plan, timeline=timeline)
        assert report.is_ready
        codes = {i.code for i in report.infos}
        assert {
            "fold_chronology_consistent", "purge_embargo_gaps_sufficient", "fold_partitions_disjoint",
            "no_empty_folds", "fold_plan_dataset_compatible", "label_horizon_purge_sufficient",
        } <= codes


class TestChronologyChecks:
    def test_unsorted_train_indices_flagged(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.array([5, 3, 4]), np.arange(10, 20)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "fold_train_not_chronological" for i in report.criticals)

    def test_unsorted_test_indices_flagged(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.arange(0, 5), np.array([12, 10, 11])),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert any(i.code == "fold_test_not_chronological" for i in report.criticals)

    def test_train_overlapping_test_position_flagged(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.arange(0, 10), np.arange(5, 15)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "fold_train_not_before_test" for i in report.criticals)


class TestPurgeEmbargoChecks:
    def test_insufficient_gap_flagged(self) -> None:
        plan = _plan(
            purge_bars=10, embargo_bars=5, total_rows=1000,
            folds=(_fold(0, np.arange(0, 100), np.arange(102, 200)),),  # gap of 1, needs 15
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "insufficient_purge_embargo_gap" for i in report.criticals)

    def test_exact_required_gap_passes(self) -> None:
        # gap = test_start - train_end - 1 = 115 - 100 - 1 = 14... use exact match to required=15
        plan = _plan(
            purge_bars=10, embargo_bars=5, total_rows=1000,
            folds=(_fold(0, np.arange(0, 100), np.arange(115, 200)),),  # gap = 115-99-1=15
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not any(i.code == "insufficient_purge_embargo_gap" for i in report.issues)

    def test_insufficient_train_validation_gap_flagged(self) -> None:
        plan = _plan(
            purge_bars=10, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.arange(0, 100), np.arange(115, 200), validation=np.arange(101, 105)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "insufficient_purge_gap_train_validation" for i in report.criticals)

    def test_insufficient_validation_test_gap_flagged(self) -> None:
        plan = _plan(
            purge_bars=5, embargo_bars=5, total_rows=1000,
            folds=(_fold(0, np.arange(0, 90), np.arange(200, 300), validation=np.arange(95, 195)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline(400))
        assert not report.is_ready
        assert any(i.code == "insufficient_purge_embargo_gap_validation_test" for i in report.criticals)


class TestOverlapChecks:
    def test_train_validation_overlap_flagged(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.arange(0, 50), np.arange(100, 150), validation=np.arange(40, 60)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert any(i.code == "fold_train_validation_overlap" for i in report.criticals)

    def test_validation_test_overlap_flagged(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000,
            folds=(_fold(0, np.arange(0, 50), np.arange(90, 150), validation=np.arange(80, 100)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert any(i.code == "fold_validation_test_overlap" for i in report.criticals)

    def test_cross_fold_test_overlap_flagged(self) -> None:
        f0 = _fold(0, np.arange(0, 50), np.arange(60, 100))
        f1 = _fold(1, np.arange(0, 80), np.arange(90, 130))  # test overlaps fold 0's test [90,100)
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=1000, folds=(f0, f1))
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "cross_fold_test_overlap" for i in report.criticals)

    def test_expanding_train_overlap_across_folds_is_not_flagged(self) -> None:
        """Expanding-window train sets legitimately overlap each other
        across folds -- this must never be reported as an error."""
        timeline = _timeline()
        plan = generate_expanding_folds(
            timeline["open_time"], n_splits=4, test_size=100, purge_bars=2, embargo_bars=1, label_horizon_bars=0,
        )
        report = validate_fold_plan(plan, timeline=timeline)
        assert not any("overlap" in i.code and i.severity is ValidationSeverity.CRITICAL for i in report.issues)


class TestDatasetCompatibilityChecks:
    def test_row_count_mismatch_flagged(self) -> None:
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=9999, folds=(_fold(0, np.arange(0, 10), np.arange(20, 30)),))
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        assert any(i.code == "fold_plan_row_count_mismatch" for i in report.criticals)

    def test_out_of_bounds_position_flagged(self) -> None:
        timeline = _timeline(50)
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=50, folds=(_fold(0, np.arange(0, 10), np.arange(40, 60)),))
        report = validate_fold_plan(plan, timeline=timeline)
        assert any(i.code == "fold_plan_position_out_of_bounds" for i in report.criticals)

    def test_missing_timestamp_column_flagged(self) -> None:
        timeline = _timeline(50).drop(columns=["open_time"])
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=50, folds=(_fold(0, np.arange(0, 10), np.arange(20, 30)),))
        report = validate_fold_plan(plan, timeline=timeline)
        assert any(i.code == "timeline_missing_timestamp_column" for i in report.criticals)

    def test_duplicate_timestamps_flagged(self) -> None:
        timeline = _timeline(50)
        timeline.loc[10, "open_time"] = timeline.loc[9, "open_time"]
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=50, folds=(_fold(0, np.arange(0, 10), np.arange(20, 30)),))
        report = validate_fold_plan(plan, timeline=timeline)
        assert any(i.code == "timeline_duplicate_timestamps" for i in report.criticals)

    def test_out_of_order_timestamps_flagged(self) -> None:
        timeline = _timeline(50)
        timeline.loc[10, "open_time"] = timeline.loc[10, "open_time"] - pd.Timedelta(days=1)
        plan = _plan(purge_bars=0, embargo_bars=0, total_rows=50, folds=(_fold(0, np.arange(0, 10), np.arange(20, 30)),))
        report = validate_fold_plan(plan, timeline=timeline)
        assert any(i.code == "timeline_out_of_order" for i in report.criticals)


class TestLabelHorizonPurgeCheck:
    """Directly tests `_validate_label_horizon_purge` -- the fix for the
    audit finding that this engine used to derive its required gap ONLY
    from user-declared `purge_bars`/`embargo_bars`, never checking the
    real dataset's label horizon. Complements
    `tests/unit/execution/test_label_horizon_purge.py`'s
    `required_label_purge_bars_for`-level proof and
    `tests/unit/execution/test_runner.py`'s end-to-end (manifest ->
    rejection) proof: this class pins the `FoldPlan`-level policy gate
    itself, in isolation from both layers around it."""

    def test_declared_purge_one_below_required_is_rejected(self) -> None:
        plan = _plan(
            purge_bars=11, embargo_bars=0, total_rows=1000, label_horizon_bars=12,
            folds=(_fold(0, np.arange(0, 100), np.arange(112, 200)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not report.is_ready
        issue = next(i for i in report.criticals if i.code == "insufficient_label_horizon_purge")
        assert issue.context["declared_purge_bars"] == 11
        assert issue.context["required_label_purge_bars"] == 12

    def test_declared_purge_exactly_at_required_passes(self) -> None:
        plan = _plan(
            purge_bars=12, embargo_bars=0, total_rows=1000, label_horizon_bars=12,
            folds=(_fold(0, np.arange(0, 100), np.arange(113, 200)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not any(i.code == "insufficient_label_horizon_purge" for i in report.issues)
        assert any(i.code == "label_horizon_purge_sufficient" for i in report.infos)

    def test_declared_purge_above_required_passes(self) -> None:
        plan = _plan(
            purge_bars=50, embargo_bars=0, total_rows=1000, label_horizon_bars=12,
            folds=(_fold(0, np.arange(0, 100), np.arange(151, 200)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not any(i.code == "insufficient_label_horizon_purge" for i in report.issues)

    def test_embargo_alone_cannot_mask_insufficient_label_purge(self) -> None:
        """A large `embargo_bars` must NOT compensate for an insufficient
        `purge_bars` in this check -- only `purge_bars` itself is compared
        against `required_label_purge_bars`. `_validate_purge_embargo`
        (the pre-existing, still-separate check) is what cares about
        `purge_bars + embargo_bars`; this check deliberately does not."""
        plan = _plan(
            purge_bars=0, embargo_bars=1000, total_rows=1000, label_horizon_bars=12,
            folds=(_fold(0, np.arange(0, 100), np.arange(1100, 1000 + 200)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline(1300))
        issue = next(i for i in report.criticals if i.code == "insufficient_label_horizon_purge")
        assert issue.context["declared_purge_bars"] == 0
        assert issue.context["required_label_purge_bars"] == 12

    def test_zero_label_horizon_requires_no_purge(self) -> None:
        plan = _plan(
            purge_bars=0, embargo_bars=0, total_rows=1000, label_horizon_bars=0,
            folds=(_fold(0, np.arange(0, 100), np.arange(100, 200)),),
        )
        report = validate_fold_plan(plan, timeline=_timeline())
        assert not any(i.code == "insufficient_label_horizon_purge" for i in report.issues)

    def test_train_validation_boundary_is_label_horizon_safe_end_to_end(self) -> None:
        """REQUIRED SEMANTICS boundary proof, train -> validation: with
        validation_fraction > 0 and purge_bars set to EXACTLY the
        required label-horizon minimum, the ACTUAL train/validation gap a
        real generator produces still satisfies that minimum -- proving
        `_validate_label_horizon_purge` (declared >= required) combined
        with `_validate_purge_embargo` (actual gap >= declared)
        transitively guarantees `actual train/validation gap >=
        required_label_purge_bars`, not just the train/test gap."""
        timeline = _timeline()
        label_horizon_bars = 12
        plan = generate_expanding_folds(
            timeline["open_time"], n_splits=3, test_size=100, purge_bars=label_horizon_bars, embargo_bars=0,
            validation_fraction=0.2, label_horizon_bars=label_horizon_bars,
        )
        report = validate_fold_plan(plan, timeline=timeline)
        assert report.is_ready
        for fold in plan.folds:
            assert len(fold.validation_indices) > 0
            train_val_gap = int(fold.validation_indices.min()) - int(fold.train_indices.max()) - 1
            assert train_val_gap >= label_horizon_bars


def test_never_raises_for_a_maximally_broken_plan() -> None:
    """Every check must produce a `ValidationIssue`, never a raised
    exception, mirroring `ml.validation`'s own guarantee."""
    plan = _plan(
        purge_bars=999, embargo_bars=999, total_rows=5,
        folds=(_fold(0, np.array([9, 1]), np.array([1, 0])),),
    )
    report = validate_fold_plan(plan, timeline=_timeline(5))
    assert not report.is_ready
    assert len(report.criticals) >= 2
