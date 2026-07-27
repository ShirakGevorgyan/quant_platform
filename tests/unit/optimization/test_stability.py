"""Milestone 4D: deterministic feature-stability (Jaccard similarity,
selection-frequency) and hyperparameter-stability (boundary-hit,
coefficient-of-variation) calculations."""

from __future__ import annotations

import pytest

from quant_platform.optimization.candidates import RankingEntry, RankingTable
from quant_platform.optimization.feature_selection import FeatureSelectionResult, FeatureSelectionStrategy
from quant_platform.optimization.search_space import FloatParameter, IntegerParameter, build_search_space
from quant_platform.optimization.stability import (
    flag_near_tied_top_candidates,
    pairwise_jaccard_similarity,
    summarize_feature_stability,
    summarize_hyperparameter_stability,
)


def _fs_result(selected: tuple[str, ...], universe_fp: str = "u1", **overrides) -> FeatureSelectionResult:
    base = {
        "schema_version": 1, "strategy": FeatureSelectionStrategy.VARIANCE_FILTER, "selected_features": selected,
        "rejected_features": tuple(n for n in ("a", "b", "c", "d") if n not in selected),
        "feature_universe_fingerprint": universe_fp, "selector_params": {}, "selector_seed": 0, "training_row_count": 10,
        "training_row_first_position": 0, "training_row_last_position": 9, "training_row_fingerprint": "x",
        "selection_reason": "r", "fitted_at": "2024-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return FeatureSelectionResult(**base)  # type: ignore[arg-type]


class TestPairwiseJaccardSimilarity:
    def test_identical_sets_have_similarity_1(self) -> None:
        summary = pairwise_jaccard_similarity([frozenset({"a", "b"}), frozenset({"a", "b"})])
        assert summary.mean == pytest.approx(1.0)

    def test_disjoint_sets_have_similarity_0(self) -> None:
        summary = pairwise_jaccard_similarity([frozenset({"a", "b"}), frozenset({"c", "d"})])
        assert summary.mean == pytest.approx(0.0)

    def test_known_partial_overlap(self) -> None:
        # {a,b,c} vs {b,c,d}: intersection=2 (b,c), union=4 (a,b,c,d) -> 0.5
        summary = pairwise_jaccard_similarity([frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})])
        assert summary.mean == pytest.approx(0.5)

    def test_requires_at_least_two_sets(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            pairwise_jaccard_similarity([frozenset({"a"})])

    def test_deterministic(self) -> None:
        sets = [frozenset({"a", "b"}), frozenset({"b", "c"}), frozenset({"a", "c"})]
        assert pairwise_jaccard_similarity(sets).mean == pairwise_jaccard_similarity(sets).mean


class TestSummarizeFeatureStability:
    def test_selection_frequency_is_fraction_of_evaluations(self) -> None:
        results = [_fs_result(("a", "b")), _fs_result(("a",)), _fs_result(("a", "b"))]
        report = summarize_feature_stability(optimization_id="opt1", feature_selection_results=results, winning_feature_sets=[("a", "b")])
        by_name = {e.feature_name: e for e in report.entries}
        assert by_name["a"].selection_frequency == pytest.approx(1.0)
        assert by_name["b"].selection_frequency == pytest.approx(2 / 3)

    def test_selected_in_winning_candidate_frequency(self) -> None:
        results = [_fs_result(("a", "b"))]
        report = summarize_feature_stability(optimization_id="opt1", feature_selection_results=results, winning_feature_sets=[("a",), ("a", "b")])
        by_name = {e.feature_name: e for e in report.entries}
        assert by_name["a"].selected_in_winning_candidate_frequency == pytest.approx(1.0)
        assert by_name["b"].selected_in_winning_candidate_frequency == pytest.approx(0.5)

    def test_low_jaccard_across_winning_sets_produces_a_warning(self) -> None:
        results = [_fs_result(("a",)), _fs_result(("b",))]
        report = summarize_feature_stability(optimization_id="opt1", feature_selection_results=results, winning_feature_sets=[("a",), ("b",)])
        assert report.pairwise_jaccard is not None and report.pairwise_jaccard.mean == pytest.approx(0.0)
        assert any("LOW stability" in w for w in report.warnings)

    def test_requires_at_least_one_result(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            summarize_feature_stability(optimization_id="opt1", feature_selection_results=[], winning_feature_sets=[])

    def test_scores_averaged_only_within_same_strategy_family_is_documented_not_enforced_across_calls(self) -> None:
        # This module never mixes strategies WITHIN one call -- a caller
        # supplying mixed-strategy results is responsible for that choice.
        results = [_fs_result(("a",), per_feature_score={"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})]
        report = summarize_feature_stability(optimization_id="opt1", feature_selection_results=results, winning_feature_sets=[("a",)])
        by_name = {e.feature_name: e for e in report.entries}
        assert by_name["a"].mean_score == pytest.approx(1.0)


class TestSummarizeHyperparameterStability:
    def test_boundary_hit_frequency_flags_repeated_boundary_winners(self) -> None:
        space = build_search_space([IntegerParameter(name="depth", low=1, high=10)])
        winners = [{"depth": 10}, {"depth": 10}, {"depth": 1}]
        report = summarize_hyperparameter_stability(
            optimization_id="opt1", search_space=space, winning_hyperparameters=winners, winning_primary_metric_values=[0.5, 0.5, 0.5],
        )
        numeric = report.numeric_parameters[0]
        assert numeric.boundary_hit_frequency == pytest.approx(1.0)
        assert any("boundary" in w for w in report.warnings)

    def test_interior_values_do_not_trigger_boundary_warning(self) -> None:
        space = build_search_space([FloatParameter(name="lr", low=0.0, high=1.0)])
        winners = [{"lr": 0.5}, {"lr": 0.4}, {"lr": 0.6}]
        report = summarize_hyperparameter_stability(
            optimization_id="opt1", search_space=space, winning_hyperparameters=winners, winning_primary_metric_values=[0.5, 0.5, 0.5],
        )
        numeric = report.numeric_parameters[0]
        assert numeric.boundary_hit_frequency == pytest.approx(0.0)

    def test_high_score_dispersion_across_outer_folds_flagged(self) -> None:
        space = build_search_space([FloatParameter(name="lr", low=0.0, high=1.0)])
        winners = [{"lr": 0.5}, {"lr": 0.5}]
        report = summarize_hyperparameter_stability(
            optimization_id="opt1", search_space=space, winning_hyperparameters=winners, winning_primary_metric_values=[0.1, 0.9],
        )
        assert any("UNSTABLE" in w for w in report.warnings)

    def test_round_trip(self) -> None:
        space = build_search_space([IntegerParameter(name="depth", low=1, high=10)])
        report = summarize_hyperparameter_stability(
            optimization_id="opt1", search_space=space, winning_hyperparameters=[{"depth": 5}], winning_primary_metric_values=[0.5],
        )
        assert type(report).from_json_dict(report.to_json_dict()) == report


class TestFlagNearTiedTopCandidates:
    def _table(self, aggregates: list[float | None], valid: list[bool]) -> RankingTable:
        entries = tuple(
            RankingEntry(
                rank=i + 1, trial_number=i, is_valid=v, primary_metric_aggregate=a, successful_inner_folds=2,
                metric_dispersion=0.0, mean_selected_feature_count=3.0, model_complexity=None,
            )
            for i, (a, v) in enumerate(zip(aggregates, valid, strict=True))
        )
        return RankingTable(optimization_id="a" * 64, outer_fold_index=0, primary_metric="accuracy", entries=entries)

    def test_near_tied_top_two_flagged(self) -> None:
        table = self._table([0.700, 0.699], [True, True])
        warnings = flag_near_tied_top_candidates(table, epsilon_fraction=0.01)
        assert len(warnings) == 1
        assert "trial 0" in warnings[0] and "trial 1" in warnings[0]

    def test_clearly_separated_top_two_not_flagged(self) -> None:
        table = self._table([0.9, 0.1], [True, True])
        assert flag_near_tied_top_candidates(table) == ()

    def test_fewer_than_two_valid_entries_returns_no_warning(self) -> None:
        table = self._table([0.9, None], [True, False])
        assert flag_near_tied_top_candidates(table) == ()
