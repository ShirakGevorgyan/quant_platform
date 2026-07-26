from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.features.drift import compare_splits, population_stability_index, summarize_column


class TestSummarizeColumn:
    def test_basic_statistics(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = summarize_column(series)
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.minimum == 1.0
        assert stats.maximum == 5.0

    def test_all_null_column(self) -> None:
        series = pd.Series([np.nan, np.nan])
        stats = summarize_column(series)
        assert stats.count == 0
        assert stats.null_fraction == 1.0


class TestPopulationStabilityIndex:
    def test_identical_distributions_have_near_zero_psi(self) -> None:
        rng = np.random.default_rng(0)
        expected = pd.Series(rng.normal(size=2000))
        actual = pd.Series(rng.normal(size=2000))
        psi = population_stability_index(expected, actual)
        assert psi < 0.05

    def test_shifted_distribution_has_high_psi(self) -> None:
        rng = np.random.default_rng(0)
        expected = pd.Series(rng.normal(loc=0, size=2000))
        actual = pd.Series(rng.normal(loc=5, size=2000))
        psi = population_stability_index(expected, actual)
        assert psi > 0.25

    def test_no_variation_in_expected_returns_zero(self) -> None:
        expected = pd.Series([1.0] * 100)
        actual = pd.Series([1.0, 2.0, 3.0])
        assert population_stability_index(expected, actual) == 0.0

    def test_empty_series_returns_nan(self) -> None:
        expected = pd.Series([], dtype="float64")
        actual = pd.Series([1.0])
        assert pd.isna(population_stability_index(expected, actual))


class TestCompareSplits:
    def test_reports_per_feature_drift(self) -> None:
        rng = np.random.default_rng(1)
        reference = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        comparison = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(loc=10, size=200)})
        report = compare_splits(reference, comparison)
        feature_names = {fr.feature_name for fr in report.feature_reports}
        assert feature_names == {"a", "b"}
        b_report = next(fr for fr in report.feature_reports if fr.feature_name == "b")
        a_report = next(fr for fr in report.feature_reports if fr.feature_name == "a")
        assert b_report.population_stability_index > a_report.population_stability_index

    def test_detects_highly_correlated_pair(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(size=500)
        reference = pd.DataFrame({"a": base, "b": base * 2 + 1, "c": rng.normal(size=500)})
        comparison = reference.copy()
        report = compare_splits(reference, comparison, correlation_threshold=0.95)
        pairs = {(p[0], p[1]) for p in report.highly_correlated_pairs}
        assert ("a", "b") in pairs

    def test_detects_constant_and_near_constant_features(self) -> None:
        reference = pd.DataFrame({"const": [5.0] * 100, "near_const": np.full(100, 1.0) + np.linspace(0, 1e-8, 100)})
        comparison = reference.copy()
        report = compare_splits(reference, comparison)
        assert "const" in report.constant_features
        assert "near_const" in report.near_constant_features
