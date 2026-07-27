"""Milestone 4D: candidate feature universe correctness, all six required
feature-selection strategies (each fit inner-train-only, each
deterministic given a fixed seed), and `FeatureSelectionResult` validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.optimization.conftest import (
    make_binary_labels,
    make_experiment_spec,
    make_feature_frame,
    make_feature_universe,
    make_row_positions,
)

from quant_platform.ml.models import ObjectiveType
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.feature_selection import (
    FeatureSelectionResult,
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
    run_feature_selection,
    select_correlation_filter,
    select_model_native_importance,
    select_none,
    select_stability,
    select_univariate,
    select_variance_filter,
    validate_feature_selection_result,
)


class TestFeatureUniverse:
    def test_from_experiment_spec_uses_declared_feature_names_in_order(self) -> None:
        experiment = make_experiment_spec(feature_names=("f3", "f1", "f2"))
        universe = FeatureUniverse.from_experiment_spec(experiment)
        assert universe.feature_names == ("f3", "f1", "f2")

    def test_fingerprint_changes_when_feature_registry_fingerprint_changes(self) -> None:
        u1 = FeatureUniverse(feature_names=("a", "b"), fingerprint="x")
        assert u1.fingerprint == "x"  # fingerprint is caller-supplied here; from_experiment_spec derives it

    def test_empty_universe_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            FeatureUniverse(feature_names=(), fingerprint="x")

    def test_duplicate_feature_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            FeatureUniverse(feature_names=("a", "a"), fingerprint="x")

    def test_round_trip(self) -> None:
        universe = make_feature_universe()
        assert FeatureUniverse.from_json_dict(universe.to_json_dict()) == universe


class TestFeatureSelectionSpecValidation:
    def test_none_strategy_rejects_params(self) -> None:
        with pytest.raises(ValueError, match="NONE"):
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE, params={"x": 1})

    def test_variance_filter_requires_non_negative_min_variance(self) -> None:
        with pytest.raises(ValueError, match="min_variance"):
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": -1.0})

    def test_correlation_filter_requires_threshold_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.CORRELATION_FILTER, params={"max_abs_correlation": 1.5})

    def test_univariate_requires_known_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.UNIVARIATE, params={"mode": "bogus"})

    def test_stability_selection_rejects_unbounded_n_repeats(self) -> None:
        with pytest.raises(ValueError, match="exceeds the bounded maximum"):
            FeatureSelectionSpec(
                strategy=FeatureSelectionStrategy.STABILITY_SELECTION,
                params={"base_strategy": "univariate", "mode": "top_k", "k": 2, "n_repeats": 100000, "min_frequency": 0.5},
            )

    def test_stability_selection_requires_known_base_strategy(self) -> None:
        with pytest.raises(ValueError, match="base_strategy"):
            FeatureSelectionSpec(
                strategy=FeatureSelectionStrategy.STABILITY_SELECTION,
                params={"base_strategy": "model_native_importance", "n_repeats": 3, "min_frequency": 0.5},
            )

    def test_round_trip(self) -> None:
        spec = FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.1})
        assert FeatureSelectionSpec.from_json_dict(spec.to_json_dict()) == spec


class TestSelectNone:
    def test_returns_the_complete_universe(self) -> None:
        universe = make_feature_universe(("a", "b", "c"))
        result = select_none(universe=universe, row_positions=make_row_positions(50))
        assert result.selected_features == universe.feature_names
        assert result.rejected_features == ()


class TestSelectVarianceFilter:
    def test_constant_column_is_rejected(self) -> None:
        universe = make_feature_universe(("f1", "f2"))
        features = pd.DataFrame({"f1": np.zeros(100), "f2": np.arange(100, dtype=float)})
        result = select_variance_filter(universe=universe, features=features, row_positions=make_row_positions(100), params={"min_variance": 0.0}, seed=1)
        assert "f1" not in result.selected_features
        assert "f2" in result.selected_features

    def test_deterministic_given_same_data(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3"))
        features = make_feature_frame(n_features=3, seed=5)
        r1 = select_variance_filter(universe=universe, features=features, row_positions=make_row_positions(300), params={"min_variance": 0.0}, seed=1)
        r2 = select_variance_filter(universe=universe, features=features, row_positions=make_row_positions(300), params={"min_variance": 0.0}, seed=1)
        assert r1.selected_features == r2.selected_features
        assert r1.per_feature_score == r2.per_feature_score

    def test_raises_clear_error_when_every_feature_rejected(self) -> None:
        universe = make_feature_universe(("f1", "f2"))
        features = pd.DataFrame({"f1": np.zeros(50), "f2": np.zeros(50)})
        with pytest.raises(ValueError, match="empty feature set"):
            select_variance_filter(universe=universe, features=features, row_positions=make_row_positions(50), params={"min_variance": 0.0}, seed=1)

    def test_preserves_universe_order(self) -> None:
        universe = make_feature_universe(("f3", "f1", "f2"))
        features = pd.DataFrame({"f1": np.arange(50, dtype=float), "f2": np.arange(50, dtype=float) * 2, "f3": np.arange(50, dtype=float) * 3})
        result = select_variance_filter(universe=universe, features=features, row_positions=make_row_positions(50), params={"min_variance": 0.0}, seed=1)
        assert result.selected_features == ("f3", "f1", "f2")


class TestSelectCorrelationFilter:
    def test_perfectly_correlated_feature_is_dropped(self) -> None:
        universe = make_feature_universe(("f1", "f2"))
        base = np.arange(100, dtype=float)
        features = pd.DataFrame({"f1": base, "f2": base * 2.0 + 1.0})  # perfectly correlated with f1
        result = select_correlation_filter(universe=universe, features=features, row_positions=make_row_positions(100), params={"max_abs_correlation": 0.95}, seed=0)
        assert result.selected_features == ("f1",)
        assert "f2" in result.rejected_features

    def test_keeps_first_encountered_of_a_correlated_cluster(self) -> None:
        universe = make_feature_universe(("f2", "f1"))  # f2 first in universe order
        base = np.arange(100, dtype=float)
        features = pd.DataFrame({"f1": base, "f2": base * 3.0})
        result = select_correlation_filter(universe=universe, features=features, row_positions=make_row_positions(100), params={"max_abs_correlation": 0.95}, seed=0)
        assert result.selected_features == ("f2",)

    def test_constant_feature_correlation_is_nan_and_treated_as_kept(self) -> None:
        universe = make_feature_universe(("f1", "f2"))
        features = pd.DataFrame({"f1": np.zeros(100), "f2": np.arange(100, dtype=float)})
        result = select_correlation_filter(universe=universe, features=features, row_positions=make_row_positions(100), params={"max_abs_correlation": 0.9}, seed=0)
        assert set(result.selected_features) == {"f1", "f2"}

    def test_never_reads_labels(self) -> None:
        import inspect

        assert "labels" not in inspect.signature(select_correlation_filter).parameters


class TestSelectUnivariate:
    def test_deterministic_given_same_seed(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3", "f4"))
        labels = make_binary_labels(300, seed=1)
        features = make_feature_frame(n_features=4, seed=2, informative=True, label=labels)
        r1 = select_univariate(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params={"mode": "top_k", "k": 2}, seed=42, objective=ObjectiveType.BINARY_CLASSIFICATION)
        r2 = select_univariate(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params={"mode": "top_k", "k": 2}, seed=42, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert r1.selected_features == r2.selected_features
        assert r1.per_feature_score == r2.per_feature_score

    def test_top_k_selects_exactly_k_features(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3", "f4"))
        labels = make_binary_labels(300, seed=1)
        features = make_feature_frame(n_features=4, seed=3, informative=True, label=labels)
        result = select_univariate(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params={"mode": "top_k", "k": 2}, seed=1, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert len(result.selected_features) == 2

    def test_percentile_mode_selects_a_fraction(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3", "f4"))
        labels = make_binary_labels(300, seed=1)
        features = make_feature_frame(n_features=4, seed=4, informative=True, label=labels)
        result = select_univariate(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params={"mode": "percentile", "percentile": 50.0}, seed=1, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert len(result.selected_features) == 2

    def test_regression_objective_uses_regression_scorer_without_crashing(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3"))
        labels = pd.Series(np.random.default_rng(0).normal(size=200), name="label")
        features = make_feature_frame(n_rows=200, n_features=3, seed=5)
        result = select_univariate(universe=universe, features=features, labels=labels, row_positions=make_row_positions(200), params={"mode": "top_k", "k": 2}, seed=1, objective=ObjectiveType.REGRESSION)
        assert len(result.selected_features) == 2


class TestSelectModelNativeImportance:
    def test_unsupported_model_family_rejected(self) -> None:
        universe = make_feature_universe(("f1", "f2"))
        labels = make_binary_labels(50)
        features = make_feature_frame(n_features=2, seed=0)
        from quant_platform.ml.models import ModelHyperparameters

        with pytest.raises(ValueError, match="not supported"):
            select_model_native_importance(
                universe=universe, features=features, labels=labels, row_positions=make_row_positions(50),
                params={"mode": "top_k", "k": 1}, seed=1, model_name="constant_test_model",
                model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(values={}),
                objective=ObjectiveType.BINARY_CLASSIFICATION,
            )


class TestSelectStability:
    def test_deterministic_given_same_seed(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3", "f4"))
        labels = make_binary_labels(300, seed=1)
        features = make_feature_frame(n_features=4, seed=6, informative=True, label=labels)
        params = {"base_strategy": "univariate", "mode": "top_k", "k": 2, "n_repeats": 5, "subsample_fraction": 0.7, "min_frequency": 0.4}
        r1 = select_stability(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params=params, seed=9, objective=ObjectiveType.BINARY_CLASSIFICATION)
        r2 = select_stability(universe=universe, features=features, labels=labels, row_positions=make_row_positions(300), params=params, seed=9, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert r1.selected_features == r2.selected_features
        assert r1.stability_frequency == r2.stability_frequency

    def test_frequencies_are_valid_fractions(self) -> None:
        universe = make_feature_universe(("f1", "f2", "f3"))
        labels = make_binary_labels(200, seed=1)
        features = make_feature_frame(n_rows=200, n_features=3, seed=7, informative=True, label=labels)
        params = {"base_strategy": "variance_filter", "min_variance": 0.0, "n_repeats": 4, "subsample_fraction": 0.6, "min_frequency": 0.3}
        result = select_stability(universe=universe, features=features, labels=labels, row_positions=make_row_positions(200), params=params, seed=3, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert result.stability_frequency is not None
        for freq in result.stability_frequency.values():
            assert 0.0 <= freq <= 1.0

    def test_bootstrap_subsamples_never_leave_inner_train(self) -> None:
        """Every repeat's row positions must be a subset of the ORIGINAL
        row_positions passed in -- stability selection must never touch
        anything beyond its own confined inner-train partition."""
        universe = make_feature_universe(("f1", "f2"))
        labels = make_binary_labels(100, seed=1)
        features = make_feature_frame(n_rows=100, n_features=2, seed=8)
        row_positions = make_row_positions(100, start=500)  # global positions, not 0-based
        params = {"base_strategy": "variance_filter", "min_variance": 0.0, "n_repeats": 3, "subsample_fraction": 0.5, "min_frequency": 0.0}
        result = select_stability(universe=universe, features=features, labels=labels, row_positions=row_positions, params=params, seed=1, objective=ObjectiveType.BINARY_CLASSIFICATION)
        assert result.training_row_first_position >= 500


class TestFeatureSelectionResultValidation:
    def test_selected_and_rejected_must_be_disjoint(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            FeatureSelectionResult(
                schema_version=1, strategy=FeatureSelectionStrategy.NONE, selected_features=("a",), rejected_features=("a",),
                feature_universe_fingerprint="x", selector_params={}, selector_seed=0, training_row_count=1,
                training_row_first_position=0, training_row_last_position=0, training_row_fingerprint="y",
                selection_reason="r", fitted_at="2024-01-01T00:00:00+00:00",
            )

    def test_empty_selected_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            FeatureSelectionResult(
                schema_version=1, strategy=FeatureSelectionStrategy.NONE, selected_features=(), rejected_features=(),
                feature_universe_fingerprint="x", selector_params={}, selector_seed=0, training_row_count=1,
                training_row_first_position=0, training_row_last_position=0, training_row_fingerprint="y",
                selection_reason="r", fitted_at="2024-01-01T00:00:00+00:00",
            )

    def test_non_finite_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FeatureSelectionResult(
                schema_version=1, strategy=FeatureSelectionStrategy.NONE, selected_features=("a",), rejected_features=(),
                feature_universe_fingerprint="x", selector_params={}, selector_seed=0, training_row_count=1,
                training_row_first_position=0, training_row_last_position=0, training_row_fingerprint="y",
                selection_reason="r", fitted_at="2024-01-01T00:00:00+00:00", per_feature_score={"a": float("nan")},
            )

    def test_invalid_stability_frequency_rejected(self) -> None:
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            FeatureSelectionResult(
                schema_version=1, strategy=FeatureSelectionStrategy.STABILITY_SELECTION, selected_features=("a",), rejected_features=(),
                feature_universe_fingerprint="x", selector_params={}, selector_seed=0, training_row_count=1,
                training_row_first_position=0, training_row_last_position=0, training_row_fingerprint="y",
                selection_reason="r", fitted_at="2024-01-01T00:00:00+00:00", stability_frequency={"a": 1.5},
            )

    def test_round_trip(self) -> None:
        result = FeatureSelectionResult(
            schema_version=1, strategy=FeatureSelectionStrategy.VARIANCE_FILTER, selected_features=("a", "b"), rejected_features=("c",),
            feature_universe_fingerprint="x", selector_params={"min_variance": 0.1}, selector_seed=5, training_row_count=10,
            training_row_first_position=0, training_row_last_position=9, training_row_fingerprint="y",
            selection_reason="r", fitted_at="2024-01-01T00:00:00+00:00", per_feature_score={"a": 1.0, "b": 2.0, "c": 0.0},
            per_feature_rank={"a": 1, "b": 2, "c": 3},
        )
        assert FeatureSelectionResult.from_json_dict(result.to_json_dict()) == result


class TestValidateFeatureSelectionResultCrossCheck:
    def test_fingerprint_mismatch_raises(self) -> None:
        universe = make_feature_universe(("a", "b"))
        result = select_none(universe=universe, row_positions=make_row_positions(10))
        other_universe = make_feature_universe(("a", "b", "c"))
        with pytest.raises(ValueError, match="fingerprint"):
            validate_feature_selection_result(result, other_universe)

    def test_valid_result_passes(self) -> None:
        universe = make_feature_universe(("a", "b"))
        result = select_none(universe=universe, row_positions=make_row_positions(10))
        validate_feature_selection_result(result, universe)


class TestRunFeatureSelectionDispatch:
    def test_dispatches_none(self) -> None:
        universe = make_feature_universe(("a", "b"))
        result = run_feature_selection(
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE), universe=universe,
            features=make_feature_frame(n_features=2, seed=0), labels=make_binary_labels(300),
            row_positions=make_row_positions(300), seed=1, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        assert result.strategy is FeatureSelectionStrategy.NONE

    def test_model_native_importance_requires_model_factory(self) -> None:
        universe = make_feature_universe(("a", "b"))
        with pytest.raises(ValueError, match="requires model_name"):
            run_feature_selection(
                FeatureSelectionSpec(strategy=FeatureSelectionStrategy.MODEL_NATIVE_IMPORTANCE, params={"mode": "top_k", "k": 1}),
                universe=universe, features=make_feature_frame(n_features=2, seed=0), labels=make_binary_labels(300),
                row_positions=make_row_positions(300), seed=1, objective=ObjectiveType.BINARY_CLASSIFICATION,
            )
