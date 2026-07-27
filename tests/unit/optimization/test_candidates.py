"""Milestone 4D: `TrialResult`/`TrialSpec` validation and the ONE
deterministic candidate-ranking policy -- direction-awareness, the full
tie-break chain, and that outer-test performance can never be an input
(structurally: no function here accepts anything resembling it)."""

from __future__ import annotations

import math

import pytest

from quant_platform.optimization.candidates import (
    InnerFoldTrialMetrics,
    TrialResult,
    TrialStatus,
    estimate_model_complexity,
    rank_trials,
)

OPT_ID = "a" * 64


def _trial(
    trial_number: int, *, status: TrialStatus = TrialStatus.COMPLETED, primary_metric_aggregate: float | None = 0.5,
    successful_inner_folds: int = 2, total_inner_folds: int = 2, sampled_hyperparameters: dict | None = None,
    inner_values: tuple[float | None, ...] = (0.5, 0.5), selected_feature_count: int = 3,
    failure_reason: str | None = None,
) -> TrialResult:
    inner_fold_metrics = tuple(
        InnerFoldTrialMetrics(
            inner_fold_index=i, primary_metric_value=v, secondary_metrics={}, selected_feature_count=selected_feature_count,
            feature_selection_result_reference=None, best_iteration=None, duration_seconds=0.1,
        )
        for i, v in enumerate(inner_values)
    )
    return TrialResult(
        schema_version=1, optimization_id=OPT_ID, outer_fold_index=0, trial_number=trial_number, status=status,
        sampled_hyperparameters=sampled_hyperparameters or {}, inner_fold_metrics=inner_fold_metrics,
        primary_metric_aggregate=primary_metric_aggregate, successful_inner_folds=successful_inner_folds,
        total_inner_folds=total_inner_folds, duration_seconds=1.0,
        failure_reason=(failure_reason if status is not TrialStatus.COMPLETED else None),
        failure_code=("x" if status is not TrialStatus.COMPLETED else None),
    )


class TestTrialResultValidation:
    def test_completed_requires_primary_metric_aggregate(self) -> None:
        with pytest.raises(ValueError, match="COMPLETED"):
            _trial(0, status=TrialStatus.COMPLETED, primary_metric_aggregate=None)

    def test_completed_must_not_carry_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="COMPLETED"):
            TrialResult(
                schema_version=1, optimization_id=OPT_ID, outer_fold_index=0, trial_number=0, status=TrialStatus.COMPLETED,
                sampled_hyperparameters={}, inner_fold_metrics=(), primary_metric_aggregate=0.5, successful_inner_folds=1,
                total_inner_folds=1, duration_seconds=1.0, failure_reason="should not be here",
            )

    def test_non_completed_requires_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty failure_reason"):
            TrialResult(
                schema_version=1, optimization_id=OPT_ID, outer_fold_index=0, trial_number=0, status=TrialStatus.FAILED,
                sampled_hyperparameters={}, inner_fold_metrics=(), primary_metric_aggregate=None, successful_inner_folds=0,
                total_inner_folds=1, duration_seconds=1.0,
            )

    def test_successful_cannot_exceed_total_inner_folds(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            _trial(0, successful_inner_folds=5, total_inner_folds=2)

    def test_is_valid_candidate_true_only_for_completed_with_value(self) -> None:
        assert _trial(0).is_valid_candidate
        assert not _trial(1, status=TrialStatus.PRUNED, primary_metric_aggregate=0.4, failure_reason="pruned").is_valid_candidate
        assert not _trial(2, status=TrialStatus.INVALID, primary_metric_aggregate=None, failure_reason="invalid").is_valid_candidate

    def test_round_trip(self) -> None:
        trial = _trial(3, sampled_hyperparameters={"lr": 0.1})
        assert TrialResult.from_json_dict(trial.to_json_dict()) == trial


class TestNonFiniteMetricsAreRejectedAtConstruction:
    """Adversarial audit, Section 5: 'Undefined, NaN, and infinite scores
    must never enter ordering.' Enforced at the EARLIEST possible point --
    construction -- so a NaN/Infinity primary metric can never even be
    represented by a `TrialResult`/`InnerFoldTrialMetrics`, let alone reach
    `rank_trials`'s sort key. Matches the identical, already-established
    `FeatureSelectionResult.per_feature_score` finiteness convention."""

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_trial_result_rejects_non_finite_primary_metric_aggregate(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _trial(0, primary_metric_aggregate=bad_value, inner_values=(bad_value, bad_value))

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_inner_fold_trial_metrics_rejects_non_finite_primary_metric_value(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            InnerFoldTrialMetrics(
                inner_fold_index=0, primary_metric_value=bad_value, secondary_metrics={}, selected_feature_count=1,
                feature_selection_result_reference=None, best_iteration=None, duration_seconds=0.1,
            )

    def test_none_primary_metric_value_remains_valid_the_skip_semantics_are_unaffected(self) -> None:
        """The finiteness check must not regress the OTHER established
        invariant this same field carries -- `None` means "this inner fold
        did not produce the metric" (never zero-filled) and stays legal."""
        metrics = InnerFoldTrialMetrics(
            inner_fold_index=0, primary_metric_value=None, secondary_metrics={}, selected_feature_count=1,
            feature_selection_result_reference=None, best_iteration=None, duration_seconds=0.1,
        )
        assert metrics.primary_metric_value is None

    def test_a_trial_can_never_be_constructed_in_a_state_that_would_corrupt_ranking(self) -> None:
        """End-to-end within this module: since NaN/Infinity trials can no
        longer be constructed at all, `rank_trials` is structurally
        protected -- there is no longer any input shape through which a
        non-finite key could reach `sorted()`."""
        with pytest.raises(ValueError, match="must be finite"):
            _trial(0, primary_metric_aggregate=math.nan)
        # The only trials `rank_trials` can ever receive are therefore
        # already finite-or-None by construction -- re-confirm ordinary
        # ranking still behaves correctly on the remaining, valid inputs.
        finite_trial = _trial(1, primary_metric_aggregate=0.7, inner_values=(0.7, 0.7))
        table = rank_trials([finite_trial], primary_metric="accuracy")
        assert table.winner is not None and table.winner.trial_number == 1

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_trial_result_rejects_non_finite_duration_seconds(self, bad_value: float) -> None:
        """Milestone 4D.1 regression: `duration_seconds < 0` alone does
        not reject NaN (`nan < 0` is `False`), so this earlier check
        silently accepted a NaN duration despite the class's own `>= 0`
        intent."""
        with pytest.raises(ValueError, match="must be a finite number"):
            TrialResult(
                schema_version=1, optimization_id=OPT_ID, outer_fold_index=0, trial_number=0, status=TrialStatus.COMPLETED,
                sampled_hyperparameters={}, inner_fold_metrics=(), primary_metric_aggregate=0.5, successful_inner_folds=1,
                total_inner_folds=1, duration_seconds=bad_value,
            )

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_inner_fold_trial_metrics_rejects_non_finite_duration_seconds(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="must be a finite number"):
            InnerFoldTrialMetrics(
                inner_fold_index=0, primary_metric_value=0.5, secondary_metrics={}, selected_feature_count=1,
                feature_selection_result_reference=None, best_iteration=None, duration_seconds=bad_value,
            )


class TestEstimateModelComplexity:
    def test_none_when_no_recognized_rounds_key(self) -> None:
        assert estimate_model_complexity({"unrelated_param": 5}) is None

    def test_rounds_times_depth_when_both_present(self) -> None:
        assert estimate_model_complexity({"num_boost_round": 100, "max_depth": 5}) == 500.0

    def test_rounds_alone_when_no_depth_key(self) -> None:
        assert estimate_model_complexity({"num_boost_round": 100}) == 100.0

    def test_iterations_key_recognized_for_catboost(self) -> None:
        assert estimate_model_complexity({"iterations": 50, "depth": 4}) == 200.0


class TestRankTrialsRequiresConsistentScope:
    def test_empty_trial_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            rank_trials([], primary_metric="accuracy")

    def test_mixed_outer_fold_indices_rejected(self) -> None:
        t0 = _trial(0)
        t1 = TrialResult(
            schema_version=1, optimization_id=OPT_ID, outer_fold_index=1, trial_number=1, status=TrialStatus.COMPLETED,
            sampled_hyperparameters={}, inner_fold_metrics=(), primary_metric_aggregate=0.5, successful_inner_folds=1,
            total_inner_folds=1, duration_seconds=1.0,
        )
        with pytest.raises(ValueError, match="one optimization_id/outer_fold_index"):
            rank_trials([t0, t1], primary_metric="accuracy")


class TestRankTrialsDirectionAwareness:
    def test_higher_is_better_metric_ranks_higher_value_first(self) -> None:
        low = _trial(0, primary_metric_aggregate=0.3, inner_values=(0.3, 0.3))
        high = _trial(1, primary_metric_aggregate=0.9, inner_values=(0.9, 0.9))
        table = rank_trials([low, high], primary_metric="accuracy")
        assert table.entries[0].trial_number == 1

    def test_lower_is_better_metric_ranks_lower_value_first(self) -> None:
        low = _trial(0, primary_metric_aggregate=0.3, inner_values=(0.3, 0.3))
        high = _trial(1, primary_metric_aggregate=0.9, inner_values=(0.9, 0.9))
        table = rank_trials([low, high], primary_metric="rmse")
        assert table.entries[0].trial_number == 0


class TestRankTrialsTieBreakChain:
    def test_valid_trial_always_outranks_invalid_regardless_of_score(self) -> None:
        invalid = _trial(0, status=TrialStatus.INVALID, primary_metric_aggregate=None, failure_reason="bad", successful_inner_folds=0)
        valid = _trial(1, primary_metric_aggregate=0.1, inner_values=(0.1, 0.1))
        table = rank_trials([invalid, valid], primary_metric="accuracy")
        assert table.entries[0].trial_number == 1
        assert table.winner is not None and table.winner.trial_number == 1

    def test_more_successful_inner_folds_breaks_a_score_tie(self) -> None:
        fewer = _trial(0, primary_metric_aggregate=0.5, successful_inner_folds=1, total_inner_folds=2, inner_values=(0.5, None))
        more = _trial(1, primary_metric_aggregate=0.5, successful_inner_folds=2, total_inner_folds=2, inner_values=(0.5, 0.5))
        table = rank_trials([fewer, more], primary_metric="accuracy")
        assert table.entries[0].trial_number == 1

    def test_lower_dispersion_breaks_a_further_tie(self) -> None:
        volatile = _trial(0, primary_metric_aggregate=0.5, inner_values=(0.2, 0.8))
        stable = _trial(1, primary_metric_aggregate=0.5, inner_values=(0.5, 0.5))
        table = rank_trials([volatile, stable], primary_metric="accuracy")
        assert table.entries[0].trial_number == 1

    def test_fewer_selected_features_breaks_a_further_tie(self) -> None:
        many_features = _trial(0, primary_metric_aggregate=0.5, inner_values=(0.5, 0.5), selected_feature_count=10)
        few_features = _trial(1, primary_metric_aggregate=0.5, inner_values=(0.5, 0.5), selected_feature_count=2)
        table = rank_trials([many_features, few_features], primary_metric="accuracy")
        assert table.entries[0].trial_number == 1

    def test_lower_trial_number_is_the_final_tie_break(self) -> None:
        a = _trial(5, primary_metric_aggregate=0.5, inner_values=(0.5, 0.5))
        b = _trial(2, primary_metric_aggregate=0.5, inner_values=(0.5, 0.5))
        table = rank_trials([a, b], primary_metric="accuracy")
        assert table.entries[0].trial_number == 2

    def test_ranking_is_deterministic_across_repeated_calls(self) -> None:
        trials = [_trial(i, primary_metric_aggregate=0.1 * i, inner_values=(0.1 * i, 0.1 * i)) for i in range(10)]
        table_a = rank_trials(trials, primary_metric="accuracy")
        table_b = rank_trials(list(reversed(trials)), primary_metric="accuracy")
        assert [e.trial_number for e in table_a.entries] == [e.trial_number for e in table_b.entries]

    def test_winner_is_none_when_no_valid_trial_exists(self) -> None:
        only_invalid = _trial(0, status=TrialStatus.INVALID, primary_metric_aggregate=None, failure_reason="bad", successful_inner_folds=0)
        table = rank_trials([only_invalid], primary_metric="accuracy")
        assert table.winner is None

    def test_ranking_table_never_references_outer_test_data(self) -> None:
        import inspect

        assert "outer_test" not in inspect.signature(rank_trials).parameters
        source = inspect.getsource(rank_trials)
        assert "outer_test" not in source
