from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features.missing import (
    apply_missing_policy,
    compute_training_statistic,
    missingness_summary,
)
from quant_platform.features.models import MissingPolicyKind, MissingPolicySpec


class TestPreserveNull:
    def test_values_unchanged(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        result = apply_missing_policy(series, policy=MissingPolicySpec(kind=MissingPolicyKind.PRESERVE_NULL))
        pd.testing.assert_series_equal(result.values, series)
        assert result.rows_to_drop is None


class TestForwardFillMaxAge:
    def test_fills_within_max_age(self) -> None:
        series = pd.Series([1.0, np.nan, np.nan, 4.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE, max_age_bars=2)
        result = apply_missing_policy(series, policy=policy)
        assert result.values.tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_nulls_beyond_max_age_remain_null(self) -> None:
        series = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan])
        policy = MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE, max_age_bars=2)
        result = apply_missing_policy(series, policy=policy)
        assert result.values.iloc[0] == 1.0
        assert result.values.iloc[1] == 1.0
        assert result.values.iloc[2] == 1.0
        assert pd.isna(result.values.iloc[3])
        assert pd.isna(result.values.iloc[4])

    def test_leading_nulls_before_any_valid_value_stay_null(self) -> None:
        series = pd.Series([np.nan, np.nan, 3.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE, max_age_bars=5)
        result = apply_missing_policy(series, policy=policy)
        assert pd.isna(result.values.iloc[0])
        assert pd.isna(result.values.iloc[1])
        assert result.values.iloc[2] == 3.0

    def test_no_backward_fill_leaks_future_value_into_leading_nulls(self) -> None:
        """Adversarial: a large max_age must never reach BACKWARD -- a
        leading null has no PRIOR value to fill from, regardless of
        max_age_bars, and must never pick up a later value."""
        series = pd.Series([np.nan, np.nan, 100.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE, max_age_bars=100)
        result = apply_missing_policy(series, policy=policy)
        assert pd.isna(result.values.iloc[0])
        assert pd.isna(result.values.iloc[1])


class TestConstantFill:
    def test_fills_with_constant(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.CONSTANT_FILL, constant_value=-1.0)
        result = apply_missing_policy(series, policy=policy)
        assert result.values.tolist() == [1.0, -1.0, 3.0]


class TestTrainingStatisticFill:
    def test_requires_fitted_statistic(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.TRAINING_STATISTIC_FILL)
        with pytest.raises(ValueError, match="fitted_statistic"):
            apply_missing_policy(series, policy=policy)

    def test_uses_supplied_statistic_verbatim(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        policy = MissingPolicySpec(kind=MissingPolicyKind.TRAINING_STATISTIC_FILL)
        result = apply_missing_policy(series, policy=policy, fitted_statistic=42.0)
        assert result.values.tolist() == [1.0, 42.0, 3.0]
        assert result.fitted_statistic_used == 42.0

    def test_compute_training_statistic_never_reflects_data_outside_what_it_is_given(self) -> None:
        """The structural leakage guard: `compute_training_statistic` has
        no way to see any data except what's explicitly passed to it --
        proving that if a caller passes ONLY a train slice, the result
        cannot possibly reflect validation/test rows, however different
        their distribution is."""
        train_slice = pd.Series([1.0, 1.0, 1.0])
        validation_slice_with_wildly_different_values = pd.Series([1000.0, 2000.0, 3000.0])
        stat = compute_training_statistic(train_slice, statistic="mean")
        assert stat == 1.0
        assert stat not in validation_slice_with_wildly_different_values.tolist()

    def test_mean_and_median_statistics(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 100.0])
        assert compute_training_statistic(series, statistic="mean") == pytest.approx(26.5)
        assert compute_training_statistic(series, statistic="median") == pytest.approx(2.5)

    def test_invalid_statistic_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_training_statistic(pd.Series([1.0]), statistic="mode")  # type: ignore[arg-type]


class TestDropRow:
    def test_reports_rows_to_drop(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0, np.nan])
        policy = MissingPolicySpec(kind=MissingPolicyKind.DROP_ROW)
        result = apply_missing_policy(series, policy=policy)
        assert list(result.rows_to_drop) == [1, 3]
        # DROP_ROW does not itself mutate values -- dropping is the caller's job
        pd.testing.assert_series_equal(result.values, series)


class TestMissingIndicator:
    def test_indicator_generated_when_requested(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        policy = MissingPolicySpec(add_missing_indicator=True)
        result = apply_missing_policy(series, policy=policy)
        assert result.missing_indicator.tolist() == [False, True, False]

    def test_no_indicator_by_default(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        result = apply_missing_policy(series, policy=MissingPolicySpec())
        assert result.missing_indicator is None


class TestMissingnessSummary:
    def test_reports_null_counts_and_fractions(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
        summary = missingness_summary(df)
        assert summary["row_count"] == 4
        assert summary["columns"]["a"]["null_count"] == 2
        assert summary["columns"]["a"]["null_fraction"] == 0.5
        assert summary["columns"]["b"]["null_count"] == 0
