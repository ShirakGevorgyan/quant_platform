from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.feature_discovery.statistics import (
    FeatureStatistics,
    compute_feature_statistics,
    shannon_entropy,
)


class TestComputeFeatureStatistics:
    def test_constant_feature(self) -> None:
        series = pd.Series(np.full(500, 7.0))
        stats = compute_feature_statistics(series, feature_name="const")
        assert stats.constant_ratio == 1.0
        assert stats.near_constant_ratio == 1.0
        assert stats.cardinality == 1
        assert stats.entropy == 0.0
        assert stats.variance == 0.0

    def test_fully_unique_trending_feature(self) -> None:
        series = pd.Series(np.arange(500, dtype="float64"))
        stats = compute_feature_statistics(series, feature_name="trend")
        assert stats.cardinality == 500
        assert stats.unique_ratio == 1.0
        assert stats.constant_ratio == 0.0
        assert stats.entropy > 0.0

    def test_missing_values_are_reflected_in_missing_ratio_and_count(self) -> None:
        series = pd.Series(np.arange(500, dtype="float64"))
        series.iloc[:50] = np.nan
        stats = compute_feature_statistics(series, feature_name="warmup")
        assert stats.count == 450
        assert abs(stats.missing_ratio - 0.1) < 1e-9

    def test_all_null_feature_does_not_crash_and_reports_full_missingness(self) -> None:
        series = pd.Series(np.full(500, np.nan))
        stats = compute_feature_statistics(series, feature_name="empty")
        assert stats.missing_ratio == 1.0
        assert stats.count == 0
        assert stats.cardinality == 0

    def test_empty_series_does_not_crash(self) -> None:
        stats = compute_feature_statistics(pd.Series([], dtype="float64"), feature_name="nothing")
        assert stats.total_rows == 0
        assert stats.count == 0

    def test_zero_ratio(self) -> None:
        series = pd.Series([0.0] * 250 + [1.0] * 250)
        stats = compute_feature_statistics(series, feature_name="mixed")
        assert abs(stats.zero_ratio - 0.5) < 1e-9

    def test_json_round_trip(self) -> None:
        series = pd.Series(np.arange(500, dtype="float64"))
        stats = compute_feature_statistics(series, feature_name="trend")
        assert FeatureStatistics.from_json_dict(stats.to_json_dict()) == stats

    def test_rolling_missingness_reflects_a_late_appearing_null_run(self) -> None:
        series = pd.Series(np.arange(1000, dtype="float64"))
        series.iloc[500:600] = np.nan
        stats = compute_feature_statistics(series, feature_name="late_gap")
        assert stats.rolling_missingness_max_swing > 0.0


class TestShannonEntropy:
    def test_single_value_has_zero_entropy(self) -> None:
        assert shannon_entropy(np.full(100, 3.0)) == 0.0

    def test_uniform_distribution_has_higher_entropy_than_skewed(self) -> None:
        uniform = np.tile([1.0, 2.0, 3.0, 4.0], 250)
        skewed = np.concatenate([np.ones(996), [2.0, 3.0, 4.0, 5.0]])
        assert shannon_entropy(uniform) > shannon_entropy(skewed)

    def test_empty_and_single_element_arrays_return_zero(self) -> None:
        assert shannon_entropy(np.array([], dtype="float64")) == 0.0
        assert shannon_entropy(np.array([1.0])) == 0.0

    def test_large_cardinality_uses_binning_without_crashing(self) -> None:
        rng = np.random.default_rng(0)
        continuous = rng.normal(0, 1, size=5000)
        entropy = shannon_entropy(continuous)
        assert entropy > 0.0
