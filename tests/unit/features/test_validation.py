from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quant_platform.features.splitting import DatasetSplit, SplitPlan
from quant_platform.features.validation import ValidationThresholds, validate_research_dataset


def _timestamps(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))


class TestStructuralChecks:
    def test_duplicate_timestamps_flagged_critical(self) -> None:
        timestamps = _timestamps(10)
        timestamps.iloc[5] = timestamps.iloc[4]
        features = pd.DataFrame({"x": np.arange(10.0)})
        report = validate_research_dataset(features, timestamps=timestamps)
        assert not report.is_valid
        assert any(i.issue_type.value == "DUPLICATE_TIMESTAMP" for i in report.critical_issues)

    def test_non_monotonic_timestamps_flagged_critical(self) -> None:
        timestamps = _timestamps(10)
        timestamps.iloc[3], timestamps.iloc[4] = timestamps.iloc[4], timestamps.iloc[3]
        features = pd.DataFrame({"x": np.arange(10.0)})
        report = validate_research_dataset(features, timestamps=timestamps)
        assert not report.is_valid

    def test_clean_data_is_valid(self) -> None:
        timestamps = _timestamps(50)
        features = pd.DataFrame({"x": np.random.default_rng(1).normal(size=50)})
        report = validate_research_dataset(features, timestamps=timestamps)
        assert report.is_valid

    def test_excessive_missingness_flagged(self) -> None:
        timestamps = _timestamps(20)
        features = pd.DataFrame({"x": [np.nan] * 15 + [1.0] * 5})
        report = validate_research_dataset(
            features, timestamps=timestamps, thresholds=ValidationThresholds(max_missing_fraction=0.5)
        )
        assert any(i.issue_type.value == "EXCESSIVE_MISSINGNESS" for i in report.warnings)

    def test_constant_feature_flagged(self) -> None:
        timestamps = _timestamps(20)
        features = pd.DataFrame({"x": [5.0] * 20})
        report = validate_research_dataset(features, timestamps=timestamps)
        assert any(i.issue_type.value == "CONSTANT_FEATURE" for i in report.warnings)

    def test_infinite_value_flagged_critical(self) -> None:
        timestamps = _timestamps(10)
        features = pd.DataFrame({"x": [1.0] * 9 + [np.inf]})
        report = validate_research_dataset(features, timestamps=timestamps)
        assert not report.is_valid
        assert any(i.issue_type.value == "INFINITE_VALUE" for i in report.critical_issues)

    def test_extreme_outlier_flagged_info(self) -> None:
        timestamps = _timestamps(30)
        values = [0.0] * 29 + [1000.0]
        features = pd.DataFrame({"x": values})
        report = validate_research_dataset(features, timestamps=timestamps, thresholds=ValidationThresholds(outlier_zscore_threshold=2.0))
        assert any(i.issue_type.value == "EXTREME_OUTLIER" for i in report.infos)

    def test_stale_external_feature_flagged(self) -> None:
        timestamps = _timestamps(10)
        features = pd.DataFrame({"x_is_stale": [1.0] * 10})
        report = validate_research_dataset(features, timestamps=timestamps, thresholds=ValidationThresholds(max_stale_fraction=0.5))
        assert any(i.issue_type.value == "STALE_EXTERNAL_FEATURE" for i in report.warnings)

    def test_missing_lineage_flagged(self) -> None:
        timestamps = _timestamps(10)
        features = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
        report = validate_research_dataset(features, timestamps=timestamps, lineage_feature_names={"a"})
        assert any(i.issue_type.value == "MISSING_FEATURE_METADATA" for i in report.warnings)

    def test_target_leakage_suspected_when_feature_equals_label(self) -> None:
        timestamps = _timestamps(30)
        label = pd.Series(np.random.default_rng(2).normal(size=30))
        features = pd.DataFrame({"suspicious": label.to_numpy()})
        report = validate_research_dataset(features, timestamps=timestamps, labels=label)
        assert any(i.issue_type.value == "TARGET_LEAKAGE_SUSPECTED" for i in report.critical_issues)

    def test_train_test_overlap_detected(self) -> None:
        timestamps = _timestamps(20)
        features = pd.DataFrame({"x": np.arange(20.0)})
        plan = SplitPlan(
            strategy="chronological",
            splits=(
                DatasetSplit(name="train", indices=np.arange(0, 12), start=timestamps.iloc[0], end=timestamps.iloc[11]),
                DatasetSplit(name="test", indices=np.arange(10, 20), start=timestamps.iloc[10], end=timestamps.iloc[19]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not report.is_valid
        assert any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.critical_issues)

    def test_walk_forward_train_train_overlap_across_folds_is_not_flagged(self) -> None:
        """Expanding-window walk-forward folds legitimately share train
        rows (fold_1's train is a superset of fold_0's) -- this must NEVER
        be flagged as a leak."""
        timestamps = _timestamps(100)
        features = pd.DataFrame({"x": np.arange(100.0)})
        plan = SplitPlan(
            strategy="expanding_walk_forward",
            splits=(
                DatasetSplit(name="fold_0_train", indices=np.arange(0, 50), start=timestamps.iloc[0], end=timestamps.iloc[49]),
                DatasetSplit(name="fold_0_test", indices=np.arange(50, 60), start=timestamps.iloc[50], end=timestamps.iloc[59]),
                DatasetSplit(name="fold_1_train", indices=np.arange(0, 70), start=timestamps.iloc[0], end=timestamps.iloc[69]),
                DatasetSplit(name="fold_1_test", indices=np.arange(70, 80), start=timestamps.iloc[70], end=timestamps.iloc[79]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.issues)

    def test_earlier_folds_test_becoming_later_folds_train_is_not_flagged(self) -> None:
        """fold_0's test rows (50-59) legitimately become part of fold_1's
        (larger) train range -- this is the defining behavior of expanding
        walk-forward CV, not a leak."""
        timestamps = _timestamps(100)
        features = pd.DataFrame({"x": np.arange(100.0)})
        plan = SplitPlan(
            strategy="expanding_walk_forward",
            splits=(
                DatasetSplit(name="fold_0_train", indices=np.arange(0, 50), start=timestamps.iloc[0], end=timestamps.iloc[49]),
                DatasetSplit(name="fold_0_test", indices=np.arange(50, 60), start=timestamps.iloc[50], end=timestamps.iloc[59]),
                DatasetSplit(name="fold_1_train", indices=np.arange(0, 65), start=timestamps.iloc[0], end=timestamps.iloc[64]),
                DatasetSplit(name="fold_1_test", indices=np.arange(70, 80), start=timestamps.iloc[70], end=timestamps.iloc[79]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.issues)

    def test_two_test_folds_overlapping_each_other_is_still_flagged(self) -> None:
        """A genuine bug (two test folds accidentally covering the same
        rows) must still be caught."""
        timestamps = _timestamps(100)
        features = pd.DataFrame({"x": np.arange(100.0)})
        plan = SplitPlan(
            strategy="expanding_walk_forward",
            splits=(
                DatasetSplit(name="fold_0_train", indices=np.arange(0, 40), start=timestamps.iloc[0], end=timestamps.iloc[39]),
                DatasetSplit(name="fold_0_test", indices=np.arange(40, 60), start=timestamps.iloc[40], end=timestamps.iloc[59]),
                DatasetSplit(name="fold_1_train", indices=np.arange(0, 40), start=timestamps.iloc[0], end=timestamps.iloc[39]),
                DatasetSplit(name="fold_1_test", indices=np.arange(50, 70), start=timestamps.iloc[50], end=timestamps.iloc[69]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not report.is_valid
        assert any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.critical_issues)

    def test_train_and_test_overlapping_within_the_same_fold_is_flagged(self) -> None:
        """Defense-in-depth: train/test overlap WITHIN the same fold should
        never happen given `splitting.py`'s own purge/embargo construction,
        but if it somehow did, this must still be caught."""
        timestamps = _timestamps(100)
        features = pd.DataFrame({"x": np.arange(100.0)})
        plan = SplitPlan(
            strategy="chronological",
            splits=(
                DatasetSplit(name="train", indices=np.arange(0, 55), start=timestamps.iloc[0], end=timestamps.iloc[54]),
                DatasetSplit(name="test", indices=np.arange(50, 100), start=timestamps.iloc[50], end=timestamps.iloc[99]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not report.is_valid
        assert any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.critical_issues)

    def test_no_overlap_when_splits_are_disjoint(self) -> None:
        timestamps = _timestamps(20)
        features = pd.DataFrame({"x": np.arange(20.0)})
        plan = SplitPlan(
            strategy="chronological",
            splits=(
                DatasetSplit(name="train", indices=np.arange(0, 10), start=timestamps.iloc[0], end=timestamps.iloc[9]),
                DatasetSplit(name="test", indices=np.arange(10, 20), start=timestamps.iloc[10], end=timestamps.iloc[19]),
            ),
            purge_bars=0, embargo_bars=0, gap_bars=0,
        )
        report = validate_research_dataset(features, timestamps=timestamps, split_plan=plan)
        assert not any(i.issue_type.value == "TRAIN_TEST_OVERLAP" for i in report.issues)


class TestNoSpuriousRuntimeWarnings:
    """A zero-variance or infinite-value input is an EXPECTED diagnostic
    case (constant/infinite features happen in real data), not an error --
    `validate_research_dataset` must report it correctly without emitting
    a `RuntimeWarning` as a side effect of computing a statistic that is
    mathematically undefined for that input (dividing by a zero/NaN
    stddev, or subtracting infinities). `warnings.simplefilter("error")`
    turns any such warning into a test failure."""

    def test_infinite_value_column_produces_no_warning(self) -> None:
        timestamps = _timestamps(10)
        features = pd.DataFrame({"x": [1.0] * 9 + [np.inf]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = validate_research_dataset(features, timestamps=timestamps)
        assert any(i.issue_type.value == "INFINITE_VALUE" for i in report.critical_issues)

    def test_constant_feature_vs_varying_label_produces_no_warning(self) -> None:
        timestamps = _timestamps(50)
        features = pd.DataFrame({"const": [1.0] * 50})
        labels = pd.Series(np.arange(50.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = validate_research_dataset(features, timestamps=timestamps, labels=labels)
        assert not any(i.issue_type.value == "TARGET_LEAKAGE_SUSPECTED" for i in report.issues)

    def test_constant_label_vs_varying_feature_produces_no_warning(self) -> None:
        timestamps = _timestamps(50)
        features = pd.DataFrame({"x": np.arange(50.0)})
        labels = pd.Series([1.0] * 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = validate_research_dataset(features, timestamps=timestamps, labels=labels)
        assert not any(i.issue_type.value == "TARGET_LEAKAGE_SUSPECTED" for i in report.issues)

    def test_constant_feature_and_constant_label_together_produce_no_warning(self) -> None:
        timestamps = _timestamps(50)
        features = pd.DataFrame({"const": [1.0] * 50})
        labels = pd.Series([2.0] * 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = validate_research_dataset(features, timestamps=timestamps, labels=labels)
        assert any(i.issue_type.value == "CONSTANT_FEATURE" for i in report.issues)
        assert not any(i.issue_type.value == "TARGET_LEAKAGE_SUSPECTED" for i in report.issues)


class TestReportSummary:
    def test_summary_is_a_readable_string(self) -> None:
        timestamps = _timestamps(10)
        features = pd.DataFrame({"x": np.arange(10.0)})
        report = validate_research_dataset(features, timestamps=timestamps)
        summary = report.summary()
        assert "ResearchDatasetValidationReport" in summary
