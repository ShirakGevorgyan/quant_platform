from __future__ import annotations

import math

import numpy as np
import pytest

from quant_platform.ml.comparison import (
    ComparisonOutcome,
    ModelComparisonReport,
    ModelFoldMetrics,
    RankedModel,
    compare_to_baseline,
    compare_to_baselines,
    rank_candidates,
)


def _metrics(values: list[float], *, name: str = "accuracy") -> ModelFoldMetrics:
    return ModelFoldMetrics(model_name=name, per_fold_metrics=tuple({"accuracy": v} for v in values))


class TestModelFoldMetricsConstruction:
    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ModelFoldMetrics(model_name="", per_fold_metrics=({"a": 1.0},))

    def test_empty_per_fold_metrics_rejected(self) -> None:
        with pytest.raises(ValueError, match="per_fold_metrics"):
            ModelFoldMetrics(model_name="x", per_fold_metrics=())

    def test_values_for_skips_folds_missing_the_metric(self) -> None:
        m = ModelFoldMetrics(model_name="x", per_fold_metrics=({"a": 1.0}, {"b": 2.0}, {"a": 3.0}))
        assert m.values_for("a") == [1.0, 3.0]

    def test_fold_identities_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_identities"):
            ModelFoldMetrics(model_name="x", per_fold_metrics=({"a": 1.0}, {"a": 2.0}), fold_identities=(0,))

    def test_duplicate_fold_identities_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ModelFoldMetrics(model_name="x", per_fold_metrics=({"a": 1.0}, {"a": 2.0}), fold_identities=(0, 0))

    def test_values_by_fold_for_defaults_to_sequential_identity(self) -> None:
        m = ModelFoldMetrics(model_name="x", per_fold_metrics=({"a": 1.0}, {"a": 2.0}, {"a": 3.0}))
        assert m.values_by_fold_for("a") == {0: 1.0, 1: 2.0, 2: 3.0}

    def test_values_by_fold_for_uses_explicit_fold_identities(self) -> None:
        m = ModelFoldMetrics(model_name="x", per_fold_metrics=({"a": 1.0}, {"a": 2.0}, {"a": 3.0}), fold_identities=(5, 7, 9))
        assert m.values_by_fold_for("a") == {5: 1.0, 7: 2.0, 9: 3.0}


class TestFoldIdentityAlignment:
    """"Never compare unrelated folds by list position unless fold
    identity matches. Reject or explicitly skip comparisons when fold
    identities do not align." -- proves `_compare_one_metric` pairs by
    `ModelFoldMetrics.fold_identities`, never by raw list index."""

    def test_partially_overlapping_fold_identities_pair_only_the_overlap(self) -> None:
        # Candidate has folds 0-9; baseline has folds 4-13. Only folds
        # 4-9 (6 of them) exist on BOTH sides under the SAME identity --
        # enough to clear _MIN_FOLDS_FOR_SIGNIFICANCE AND for Wilcoxon to
        # reach two-sided significance on a constant difference (5 tied
        # differences tops out at p=0.0625, just above alpha=0.05; 6
        # reaches p=0.03125).
        candidate = ModelFoldMetrics(
            model_name="candidate", fold_identities=tuple(range(10)),
            per_fold_metrics=tuple({"accuracy": 0.9} for _ in range(10)),
        )
        baseline = ModelFoldMetrics(
            model_name="baseline", fold_identities=tuple(range(4, 14)),
            per_fold_metrics=tuple({"accuracy": 0.1} for _ in range(10)),
        )
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.paired_n == 6
        assert comparison.outcome is ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER

    def test_disjoint_fold_identities_skip_with_an_explicit_reason(self) -> None:
        candidate = ModelFoldMetrics(
            model_name="candidate", fold_identities=(0, 1, 2, 3, 4, 5),
            per_fold_metrics=tuple({"accuracy": 0.9} for _ in range(6)),
        )
        baseline = ModelFoldMetrics(
            model_name="baseline", fold_identities=(100, 101, 102, 103, 104, 105),
            per_fold_metrics=tuple({"accuracy": 0.1} for _ in range(6)),
        )
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
        assert comparison.paired_n == 0
        assert comparison.p_value is None
        assert "no overlapping fold identities" in comparison.reason

    def test_reordered_fold_identities_still_pair_correctly_not_by_position(self) -> None:
        """Candidate reports folds in order (0, 1, 2); baseline reports
        the SAME three folds but in a DIFFERENT list order (2, 0, 1),
        with each fold's value chosen so that pairing by raw POSITION
        would produce the OPPOSITE conclusion from pairing by identity."""
        candidate = ModelFoldMetrics(
            model_name="candidate", fold_identities=(0, 1, 2, 3, 4),
            per_fold_metrics=({"accuracy": 0.9}, {"accuracy": 0.9}, {"accuracy": 0.9}, {"accuracy": 0.9}, {"accuracy": 0.9}),
        )
        # Baseline's own list order is reversed relative to its identities
        # -- if pairing were positional, fold "0"'s candidate value (0.9)
        # would incorrectly pair against fold "4"'s baseline value (0.9)
        # instead of fold "0"'s own baseline value (0.1).
        baseline = ModelFoldMetrics(
            model_name="baseline", fold_identities=(4, 3, 2, 1, 0),
            per_fold_metrics=({"accuracy": 0.9}, {"accuracy": 0.9}, {"accuracy": 0.1}, {"accuracy": 0.1}, {"accuracy": 0.1}),
        )
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.paired_n == 5
        # Correct (identity-based) pairing: folds 0,1,2 -> candidate 0.9 vs
        # baseline 0.1 (candidate better); folds 3,4 -> candidate 0.9 vs
        # baseline 0.9 (tied). Net: candidate is better overall.
        assert comparison.candidate_aggregate is not None
        assert comparison.baseline_aggregate is not None
        assert comparison.candidate_aggregate.mean == pytest.approx(0.9)
        assert comparison.baseline_aggregate.mean == pytest.approx((0.9 + 0.9 + 0.1 + 0.1 + 0.1) / 5)


class TestMetricDirection:
    """`compare_to_baseline`/`rank_candidates` are the only public entry
    points that ever need to know a metric's direction -- exercised here
    end to end rather than importing the module's private direction
    table directly."""

    @pytest.mark.parametrize("name", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef", "r2"])
    def test_higher_is_better_metrics_rank_the_larger_mean_first(self, name: str) -> None:
        high = ModelFoldMetrics(model_name="high", per_fold_metrics=({name: 0.9},) * 8)
        low = ModelFoldMetrics(model_name="low", per_fold_metrics=({name: 0.1},) * 8)
        ranking = rank_candidates([low, high], [low], primary_metric=name)
        assert ranking.ranked_models[0].model_name == "high"

    @pytest.mark.parametrize("name", ["mae", "rmse", "mape"])
    def test_lower_is_better_metrics_rank_the_smaller_mean_first(self, name: str) -> None:
        high = ModelFoldMetrics(model_name="high", per_fold_metrics=({name: 0.9},) * 8)
        low = ModelFoldMetrics(model_name="low", per_fold_metrics=({name: 0.1},) * 8)
        ranking = rank_candidates([low, high], [high], primary_metric=name)
        assert ranking.ranked_models[0].model_name == "low"

    def test_unknown_metric_raises(self) -> None:
        candidate = ModelFoldMetrics(model_name="c", per_fold_metrics=({"not_a_real_metric": 0.5},) * 6)
        baseline = ModelFoldMetrics(model_name="b", per_fold_metrics=({"not_a_real_metric": 0.4},) * 6)
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_to_baseline(candidate, baseline)


class TestCompareToBaseline:
    def test_candidate_clearly_better_higher_is_better_metric(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.70 + rng.normal(0, 0.01) for _ in range(8)], name="candidate")
        baseline = _metrics([0.50 + rng.normal(0, 0.01) for _ in range(8)], name="baseline")
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER
        assert comparison.p_value is not None
        assert comparison.p_value < 0.05

    def test_candidate_clearly_worse_higher_is_better_metric(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.50 + rng.normal(0, 0.01) for _ in range(8)], name="candidate")
        baseline = _metrics([0.70 + rng.normal(0, 0.01) for _ in range(8)], name="baseline")
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_WORSE

    def test_candidate_clearly_better_lower_is_better_metric(self) -> None:
        rng = np.random.default_rng(0)
        candidate = ModelFoldMetrics(model_name="c", per_fold_metrics=tuple({"mae": 0.1 + rng.normal(0, 0.01)} for _ in range(8)))
        baseline = ModelFoldMetrics(model_name="b", per_fold_metrics=tuple({"mae": 0.5 + rng.normal(0, 0.01)} for _ in range(8)))
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("mae")
        assert comparison.outcome is ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER

    def test_no_meaningful_difference(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.60 + rng.normal(0, 0.02) for _ in range(8)], name="candidate")
        baseline = _metrics([0.60 + rng.normal(0, 0.02) for _ in range(8)], name="baseline")
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome in (ComparisonOutcome.NO_SIGNIFICANT_DIFFERENCE, ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_BETTER, ComparisonOutcome.CANDIDATE_SIGNIFICANTLY_WORSE)

    def test_too_few_folds_is_skipped_not_a_false_pass(self) -> None:
        candidate = _metrics([0.9, 0.9])
        baseline = _metrics([0.1, 0.1])
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
        assert comparison.p_value is None
        assert "matching fold identity" in comparison.reason

    def test_identical_values_every_fold_is_skipped_not_a_crash(self) -> None:
        candidate = _metrics([0.5] * 6)
        baseline = _metrics([0.5] * 6)
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
        assert "IDENTICAL" in comparison.reason

    def test_metric_missing_from_one_side_is_skipped(self) -> None:
        candidate = ModelFoldMetrics(model_name="c", per_fold_metrics=({"accuracy": 0.8}, {"accuracy": 0.7}))
        baseline = ModelFoldMetrics(model_name="b", per_fold_metrics=({"f1": 0.5}, {"f1": 0.4}))
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.outcome is ComparisonOutcome.SKIPPED_INSUFFICIENT_DATA
        assert comparison.candidate_aggregate is not None
        assert comparison.baseline_aggregate is None

    def test_unpaired_length_uses_shorter_common_prefix(self) -> None:
        candidate = _metrics([0.9] * 10)
        baseline = _metrics([0.1] * 6)
        report = compare_to_baseline(candidate, baseline)
        comparison = report.comparison_for("accuracy")
        assert comparison.paired_n == 6


class TestOutperformsAllBaselines:
    def test_true_when_better_than_every_baseline(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.80 + rng.normal(0, 0.01) for _ in range(8)], name="candidate")
        baseline1 = _metrics([0.50 + rng.normal(0, 0.01) for _ in range(8)], name="b1")
        baseline2 = _metrics([0.55 + rng.normal(0, 0.01) for _ in range(8)], name="b2")
        report = compare_to_baselines(candidate, [baseline1, baseline2])
        assert report.outperforms_all_baselines("accuracy") is True

    def test_false_when_worse_than_one_baseline(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.80 + rng.normal(0, 0.01) for _ in range(8)], name="candidate")
        baseline1 = _metrics([0.50 + rng.normal(0, 0.01) for _ in range(8)], name="b1")
        baseline2 = _metrics([0.95 + rng.normal(0, 0.005) for _ in range(8)], name="b2")  # candidate loses to this one
        report = compare_to_baselines(candidate, [baseline1, baseline2])
        assert report.outperforms_all_baselines("accuracy") is False

    def test_false_when_no_baselines_at_all(self) -> None:
        candidate = _metrics([0.8] * 8)
        report = compare_to_baselines(candidate, [])
        assert report.outperforms_all_baselines("accuracy") is False

    def test_false_when_comparison_skipped_never_silently_true(self) -> None:
        """A model must NOT be declared successful merely because the
        statistical test could not run (too few folds) -- SKIPPED must
        count as 'not proven better', never as an implicit pass."""
        candidate = _metrics([0.9, 0.9])  # only 2 folds -> always skipped
        baseline = _metrics([0.1, 0.1])
        report = compare_to_baselines(candidate, [baseline])
        assert report.outperforms_all_baselines("accuracy") is False

    def test_false_when_primary_metric_never_reported(self) -> None:
        candidate = ModelFoldMetrics(model_name="c", per_fold_metrics=({"f1": 0.8},) * 8)
        baseline = ModelFoldMetrics(model_name="b", per_fold_metrics=({"f1": 0.5},) * 8)
        report = compare_to_baselines(candidate, [baseline])
        assert report.outperforms_all_baselines("accuracy") is False


class TestRankCandidates:
    def test_ranks_best_first_for_higher_is_better_metric(self) -> None:
        rng = np.random.default_rng(0)
        good = _metrics([0.9 + rng.normal(0, 0.005) for _ in range(8)], name="good")
        bad = _metrics([0.5 + rng.normal(0, 0.005) for _ in range(8)], name="bad")
        baseline = _metrics([0.3] * 8, name="baseline")
        ranking = rank_candidates([bad, good], [baseline], primary_metric="accuracy")
        assert [m.model_name for m in ranking.ranked_models] == ["good", "bad"]

    def test_ranks_best_first_for_lower_is_better_metric(self) -> None:
        rng = np.random.default_rng(0)
        good = ModelFoldMetrics(model_name="good", per_fold_metrics=tuple({"mae": 0.1 + rng.normal(0, 0.005)} for _ in range(8)))
        bad = ModelFoldMetrics(model_name="bad", per_fold_metrics=tuple({"mae": 0.5 + rng.normal(0, 0.005)} for _ in range(8)))
        baseline = ModelFoldMetrics(model_name="baseline", per_fold_metrics=({"mae": 0.9},) * 8)
        ranking = rank_candidates([bad, good], [baseline], primary_metric="mae")
        assert [m.model_name for m in ranking.ranked_models] == ["good", "bad"]

    def test_candidate_missing_primary_metric_raises(self) -> None:
        candidate = ModelFoldMetrics(model_name="c", per_fold_metrics=({"f1": 0.8},))
        baseline = _metrics([0.5])
        with pytest.raises(ValueError, match="never reported"):
            rank_candidates([candidate], [baseline], primary_metric="accuracy")

    def test_ranked_model_carries_its_own_comparison_report(self) -> None:
        rng = np.random.default_rng(0)
        candidate = _metrics([0.9 + rng.normal(0, 0.005) for _ in range(8)], name="candidate")
        baseline = _metrics([0.3] * 8, name="baseline")
        ranking = rank_candidates([candidate], [baseline], primary_metric="accuracy")
        ranked = ranking.ranked_models[0]
        assert ranked.comparison_report.candidate_name == "candidate"
        assert ranked.outperforms_all_baselines == ranked.comparison_report.outperforms_all_baselines("accuracy")


class TestRankedModelFiniteNumberInvariant:
    """Regression test for Milestone 4D.1: `RankedModel` had no
    `__post_init__` at all -- a non-finite `primary_metric_mean` would
    silently corrupt `rank_candidates`' `ranked.sort(key=lambda m: m.
    primary_metric_mean, ...)` (Python's `sort` has no defined behavior
    for a NaN key; a NaN-scored model is not guaranteed to sort to either
    end, so it could be silently reported as the winner)."""

    @staticmethod
    def _empty_report(name: str = "candidate") -> ModelComparisonReport:
        return ModelComparisonReport(candidate_name=name, baseline_reports=())

    def test_rejects_nan_primary_metric_mean(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            RankedModel(
                model_name="m", primary_metric_mean=math.nan, outperforms_all_baselines=False,
                comparison_report=self._empty_report(),
            )

    def test_rejects_infinite_primary_metric_mean(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            RankedModel(
                model_name="m", primary_metric_mean=math.inf, outperforms_all_baselines=False,
                comparison_report=self._empty_report(),
            )

    def test_accepts_finite_primary_metric_mean(self) -> None:
        ranked = RankedModel(
            model_name="m", primary_metric_mean=0.75, outperforms_all_baselines=True,
            comparison_report=self._empty_report(),
        )
        assert ranked.primary_metric_mean == 0.75


class TestJsonSerialization:
    def test_metric_comparison_to_json_dict(self) -> None:
        candidate = _metrics([0.9] * 6)
        baseline = _metrics([0.1] * 6)
        report = compare_to_baseline(candidate, baseline)
        raw = report.to_json_dict()
        assert raw["baseline_name"] == baseline.model_name
        assert isinstance(raw["metric_comparisons"], list)

    def test_ranking_report_to_json_dict(self) -> None:
        candidate = _metrics([0.9] * 6, name="c")
        baseline = _metrics([0.1] * 6, name="b")
        ranking = rank_candidates([candidate], [baseline], primary_metric="accuracy")
        raw = ranking.to_json_dict()
        assert raw["primary_metric"] == "accuracy"
        assert len(raw["ranked_models"]) == 1
