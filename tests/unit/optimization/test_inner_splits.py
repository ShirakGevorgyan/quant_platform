"""Milestone 4D: nested (inner) walk-forward split construction and its
independent validator -- the single most safety-critical module in this
package. Every test in `TestNoOuterTestLeakage` proves a real, hand-
crafted leakage attempt is CAUGHT, not merely that the happy path works."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.execution.splitters import Fold
from quant_platform.optimization.inner_splits import (
    InnerFold,
    InnerFoldPlan,
    InnerSplitConfig,
    build_inner_fold_plan,
    validate_nested_plan,
)


def _timeline(n: int = 400) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": times, "label": np.zeros(n)})


def _outer_fold(*, train_end: int, test_start: int, test_end: int, fold_index: int = 0) -> Fold:
    timeline = _timeline()
    train_indices = np.arange(0, train_end)
    test_indices = np.arange(test_start, test_end)
    return Fold(
        fold_index=fold_index, train_indices=train_indices, test_indices=test_indices,
        train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[train_end - 1],
        test_start=timeline["open_time"].iloc[test_start], test_end=timeline["open_time"].iloc[test_end - 1],
    )


class TestInnerSplitConfigValidation:
    def test_unknown_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            InnerSplitConfig(strategy="bogus", n_splits=2, test_size_fraction=0.2)

    def test_n_splits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            InnerSplitConfig(strategy="expanding_walk_forward", n_splits=0, test_size_fraction=0.2)

    def test_test_size_fraction_must_be_in_open_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="test_size_fraction"):
            InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=1.0)

    def test_max_train_size_fraction_requires_rolling_strategy(self) -> None:
        with pytest.raises(ValueError, match="max_train_size_fraction"):
            InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2, max_train_size_fraction=0.5)

    def test_round_trip(self) -> None:
        config = InnerSplitConfig(strategy="rolling_walk_forward", n_splits=3, test_size_fraction=0.1, embargo_bars=2, max_train_size_fraction=0.5)
        assert InnerSplitConfig.from_json_dict(config.to_json_dict()) == config


class TestBuildInnerFoldPlan:
    def test_inner_folds_are_subset_of_outer_train(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        plan = build_inner_fold_plan(
            outer_fold, config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
            label_horizon_bars=3, timeline=_timeline(),
        )
        outer_train_set = set(outer_fold.train_indices.tolist())
        for inner_fold in plan.inner_folds:
            assert set(inner_fold.train_indices.tolist()) <= outer_train_set
            assert set(inner_fold.validation_indices.tolist()) <= outer_train_set

    def test_purge_always_equals_required_label_horizon(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        plan = build_inner_fold_plan(
            outer_fold, config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
            label_horizon_bars=7, timeline=_timeline(),
        )
        assert plan.purge_bars == 7
        assert plan.required_label_purge_bars == 7

    def test_rolling_strategy_requires_max_train_size_fraction(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        with pytest.raises(ValueError, match="max_train_size_fraction"):
            build_inner_fold_plan(
                outer_fold, config=InnerSplitConfig(strategy="rolling_walk_forward", n_splits=2, test_size_fraction=0.2),
                label_horizon_bars=3, timeline=_timeline(),
            )

    def test_inner_fold_indices_are_ordered_0_to_n_minus_1(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        plan = build_inner_fold_plan(
            outer_fold, config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15),
            label_horizon_bars=2, timeline=_timeline(),
        )
        assert [f.inner_fold_index for f in plan.inner_folds] == list(range(len(plan.inner_folds)))

    def test_round_trip(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        plan = build_inner_fold_plan(
            outer_fold, config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
            label_horizon_bars=3, timeline=_timeline(),
        )
        decoded = InnerFoldPlan.from_json_dict(plan.to_json_dict())
        assert decoded.purge_bars == plan.purge_bars
        for original, restored in zip(plan.inner_folds, decoded.inner_folds, strict=True):
            assert np.array_equal(original.train_indices, restored.train_indices)
            assert np.array_equal(original.validation_indices, restored.validation_indices)


class TestValidateNestedPlanHappyPath:
    def test_a_correctly_built_plan_passes_validation(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        plan = build_inner_fold_plan(
            outer_fold, config=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
            label_horizon_bars=3, timeline=_timeline(),
        )
        report = validate_nested_plan(outer_fold, plan)
        assert report.is_ready


class TestNoOuterTestLeakage:
    """Hand-craft a `InnerFoldPlan` whose inner rows reach into the outer
    fold's TEST partition -- the validator must catch this every time,
    regardless of which inner fold or which side (train/validation) it
    happens on."""

    def _base_plan(self, outer_fold: Fold) -> tuple[InnerFold, ...]:
        return (
            InnerFold(
                inner_fold_index=0, train_indices=outer_fold.train_indices[:100], validation_indices=outer_fold.train_indices[105:120],
                train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
                validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
            ),
        )

    def test_inner_train_touching_outer_test_is_caught(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        # Sneak 5 outer-TEST row positions into the inner-train set.
        leaked_train = np.concatenate([outer_fold.train_indices[:100], outer_fold.test_indices[:5]])
        bad_inner = InnerFold(
            inner_fold_index=0, train_indices=leaked_train, validation_indices=outer_fold.train_indices[105:120],
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(bad_inner,),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "inner_train_touches_outer_reserved_rows" in codes

    def test_inner_validation_touching_outer_test_is_caught(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        leaked_validation = np.concatenate([outer_fold.train_indices[105:115], outer_fold.test_indices[:3]])
        bad_inner = InnerFold(
            inner_fold_index=0, train_indices=outer_fold.train_indices[:100], validation_indices=leaked_validation,
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(bad_inner,),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "inner_validation_touches_outer_reserved_rows" in codes

    def test_inner_row_outside_outer_train_entirely_is_caught(self) -> None:
        """Even a row that is not literally in outer-test but ALSO not in
        outer-train (e.g. an out-of-range fabricated position) must be
        rejected -- "subset of outer-train" is checked independently of
        the outer-test-specific check."""
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        bogus_train = np.concatenate([outer_fold.train_indices[:100], np.array([99999])])
        bad_inner = InnerFold(
            inner_fold_index=0, train_indices=bogus_train, validation_indices=outer_fold.train_indices[105:120],
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(bad_inner,),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "inner_train_outside_outer_train" in codes

    def test_insufficient_purge_gap_between_inner_train_and_validation_is_caught(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        # Train ends at position 99, validation starts at 100 -- ZERO gap, but purge_bars=3 is declared.
        bad_inner = InnerFold(
            inner_fold_index=0, train_indices=np.arange(0, 100), validation_indices=np.arange(100, 120),
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(bad_inner,),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "inner_insufficient_purge_embargo_gap" in codes

    def test_non_chronological_inner_train_is_caught(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        shuffled = outer_fold.train_indices[:100][::-1]
        bad_inner = InnerFold(
            inner_fold_index=0, train_indices=shuffled, validation_indices=outer_fold.train_indices[105:120],
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(bad_inner,),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "inner_fold_not_chronological" in codes

    def test_duplicate_validation_rows_across_inner_folds_is_caught(self) -> None:
        outer_fold = _outer_fold(train_end=300, test_start=300, test_end=400)
        fold_0 = InnerFold(
            inner_fold_index=0, train_indices=np.arange(0, 100), validation_indices=np.arange(110, 130),
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        fold_1 = InnerFold(
            # Overlaps fold_0's validation set (120..135 vs 110..130).
            inner_fold_index=1, train_indices=np.arange(0, 150), validation_indices=np.arange(120, 135),
            train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        plan = InnerFoldPlan(
            schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(fold_0, fold_1),
            purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=300,
        )
        report = validate_nested_plan(outer_fold, plan)
        assert not report.is_ready
        codes = {i.code for i in report.criticals}
        assert "cross_inner_fold_validation_overlap" in codes


class TestInnerFoldPlanConstructionInvariants:
    def test_purge_bars_must_equal_required_label_purge_bars(self) -> None:
        with pytest.raises(Exception, match="purge_bars"):
            InnerFoldPlan(
                schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward",
                inner_folds=(InnerFold(
                    inner_fold_index=0, train_indices=np.arange(0, 10), validation_indices=np.arange(15, 20),
                    train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
                    validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
                ),),
                purge_bars=5, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=20,
            )

    def test_empty_inner_folds_rejected(self) -> None:
        with pytest.raises(Exception, match="at least one"):
            InnerFoldPlan(
                schema_version=1, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(),
                purge_bars=3, embargo_bars=0, label_horizon_bars=3, required_label_purge_bars=3, outer_train_row_count=20,
            )

    def test_inner_fold_with_empty_train_rejected(self) -> None:
        with pytest.raises(Exception, match="must not be empty"):
            InnerFold(
                inner_fold_index=0, train_indices=np.array([], dtype=np.int64), validation_indices=np.arange(0, 5),
                train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-01", tz="UTC"),
                validation_start=pd.Timestamp("2024-01-01", tz="UTC"), validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
            )
