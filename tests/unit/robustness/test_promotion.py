"""Milestone 6, Section 16: promotion-gate fail-closed behavior. Proves
all four `PromotionDecisionKind` branches are individually reachable with
their documented precedence (REJECTED > MANUAL_REVIEW_REQUIRED >
RESEARCH_ONLY > ELIGIBLE_FOR_PAPER_TRADING), that a skipped MANDATORY
gate never silently passes, that an unknown policy gate name fails
closed rather than being ignored, and that `ELIGIBLE_FOR_LIVE_TRADING`
cannot even be constructed."""

from __future__ import annotations

import pytest

from quant_platform.robustness.bootstrap import BootstrapEstimate, BootstrapReport, DownsideAnalysisReport
from quant_platform.robustness.models import PromotionDecisionKind, StressAxisKind
from quant_platform.robustness.promotion import (
    DEFAULT_PROMOTION_POLICY,
    DISCLAIMER,
    PromotionEvidence,
    evaluate_promotion,
)
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.specs import DEFAULT_PROMOTION_GATES, PromotionGateSpec, PromotionPolicySpec
from quant_platform.robustness.stability import ConcentrationReport, FoldStabilityReport
from quant_platform.robustness.stress import StressReport, StressScenarioResult


def _source_verification(*, passed: bool) -> SourceVerificationReport:
    return SourceVerificationReport(
        schema_version=1, source_backtest_id="a" * 64, verify_backtest_is_ready=passed, verify_backtest_critical_count=0,
        verify_backtest_issue_codes=(), dataset_content_id_matches=passed, split_plan_fingerprint_matches=passed,
        instrument_identity_matches=passed, bar_interval_matches=passed, total_outer_folds=6, generated_at="2026-01-01T00:00:00Z",
    )


def _bootstrap(lower_bound: float) -> BootstrapReport:
    return BootstrapReport(
        schema_version=1, series_kind="stitched_bar_net", method="stationary", repetitions=500, confidence_level=0.95, seed=1,
        estimates=(BootstrapEstimate(statistic_name="total_return", point_estimate=lower_bound + 0.05, lower_bound=lower_bound, upper_bound=lower_bound + 0.1, valid_repetitions=500, skipped_repetitions=0, failure_reasons={}),),
        generated_at="2026-01-01T00:00:00Z",
    )


def _downside(*, prob_loss: float) -> DownsideAnalysisReport:
    return DownsideAnalysisReport(
        schema_version=1, probability_total_net_return_non_positive=prob_loss, probability_mean_return_non_positive=prob_loss,
        probability_sharpe_non_positive=prob_loss, probability_underperforms_always_flat=None, probability_underperforms_always_long=None,
        probability_cost_stressed_unprofitable=None, probability_maximum_drawdown_exceeds_limit=None, drawdown_limit=None, repetitions=500,
        seed=1, generated_at="2026-01-01T00:00:00Z",
    )


def _fold_stability(*, profitable_fraction: float, worst_fold: float, max_dd: float) -> FoldStabilityReport:
    concentration = ConcentrationReport(
        schema_version=1, single_fold_profit_concentration=0.3, single_trade_profit_concentration=0.3, single_day_profit_concentration=0.3,
        single_direction_profit_concentration=0.3, single_confidence_bucket_profit_concentration=0.3, warning_codes=(),
    )
    return FoldStabilityReport(
        schema_version=1, fold_count=6, profitable_fold_fraction=profitable_fraction, positive_sharpe_fold_fraction=0.6, median_fold_return=0.02,
        worst_fold_return=worst_fold, fold_return_stdev=0.01, fold_sharpe_dispersion=0.1, maximum_fold_drawdown=max_dd, worst_fold_cost_drag=0.001,
        fold_trade_count_dispersion=1.0, fold_exposure_dispersion=0.1, direction_consistency=0.8, benchmark_outperformance_fraction=0.6,
        concentration=concentration, generated_at="2026-01-01T00:00:00Z",
    )


def _stress(*, base_return: float) -> StressReport:
    scenarios = (
        StressScenarioResult(
            name="base_cost", axis=StressAxisKind.BASE_COST, named_profile=None, status="evaluated", skip_reason=None, total_net_return=base_return,
            total_gross_return=base_return + 0.01, closed_trade_count=40, maximum_drawdown=0.05, is_profitable=base_return > 0.0, net_return_degradation_vs_baseline=0.0,
        ),
    )
    return StressReport(
        schema_version=1, source_backtest_id="a" * 64, baseline_total_net_return=base_return, baseline_total_gross_return=base_return + 0.01,
        baseline_closed_trade_count=40, baseline_maximum_drawdown=0.05, scenario_results=scenarios, breakeven_results=(), generated_at="2026-01-01T00:00:00Z",
    )


def _evidence(**overrides: object) -> PromotionEvidence:
    defaults: dict[str, object] = {
        "robustness_id": "a" * 64, "source_backtest_id": "a" * 64, "source_verification": _source_verification(passed=True), "total_outer_folds": 6,
        "observation_count": 500, "effective_sample_count": 200.0, "total_closed_trade_count": 40, "bootstrap": _bootstrap(0.02), "downside": _downside(prob_loss=0.1),
        "fold_stability": _fold_stability(profitable_fraction=0.8, worst_fold=0.0, max_dd=0.1), "stress": _stress(base_return=0.03), "sensitivity": None, "regime": None,
    }
    defaults.update(overrides)
    return PromotionEvidence(**defaults)  # type: ignore[arg-type]


class TestFourDecisionBranches:
    def test_clean_candidate_is_eligible_for_paper_trading(self) -> None:
        decision = evaluate_promotion(evidence=_evidence())
        assert decision.decision is PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING
        assert decision.disclaimer == DISCLAIMER

    def test_mandatory_gate_failing_outright_is_rejected(self) -> None:
        decision = evaluate_promotion(evidence=_evidence(fold_stability=_fold_stability(profitable_fraction=0.8, worst_fold=0.0, max_dd=0.9)))
        assert decision.decision is PromotionDecisionKind.REJECTED
        assert "maximum_drawdown" in decision.decision_reason

    def test_unmeasurable_mandatory_gate_requires_manual_review_not_a_pass(self) -> None:
        policy = PromotionPolicySpec(gates=(*DEFAULT_PROMOTION_GATES[:-2], PromotionGateSpec(name="no_parameter_cliff", mandatory=True, minimum_value=1.0), DEFAULT_PROMOTION_GATES[-1]))
        decision = evaluate_promotion(evidence=_evidence(sensitivity=None), policy=policy)
        assert decision.decision is PromotionDecisionKind.MANUAL_REVIEW_REQUIRED
        assert "no_parameter_cliff" in decision.decision_reason

    def test_advisory_gate_failing_with_clean_mandatory_gates_is_research_only(self) -> None:
        decision = evaluate_promotion(evidence=_evidence(fold_stability=_fold_stability(profitable_fraction=0.8, worst_fold=-0.9, max_dd=0.1)))
        assert decision.decision is PromotionDecisionKind.RESEARCH_ONLY
        assert "worst_fold_return" in decision.decision_reason

    def test_precedence_rejected_beats_manual_review_when_both_conditions_present(self) -> None:
        """A mandatory gate fails outright AND another mandatory gate is
        unmeasurable at the same time -- REJECTED must win, per the
        documented precedence order."""
        policy = PromotionPolicySpec(gates=(*DEFAULT_PROMOTION_GATES[:-2], PromotionGateSpec(name="no_parameter_cliff", mandatory=True, minimum_value=1.0), DEFAULT_PROMOTION_GATES[-1]))
        decision = evaluate_promotion(
            evidence=_evidence(sensitivity=None, fold_stability=_fold_stability(profitable_fraction=0.8, worst_fold=0.0, max_dd=0.9)), policy=policy,
        )
        assert decision.decision is PromotionDecisionKind.REJECTED


class TestFailClosedOnUnknownGateName:
    def test_unknown_gate_name_raises_rather_than_being_silently_skipped(self) -> None:
        bad_policy = PromotionPolicySpec(gates=(PromotionGateSpec(name="not_a_real_gate", mandatory=True, minimum_value=1.0),))
        with pytest.raises(Exception, match="not_a_real_gate"):
            evaluate_promotion(evidence=_evidence(), policy=bad_policy)


class TestNeverLiveTrading:
    def test_eligible_for_live_trading_is_not_a_member_of_the_enum(self) -> None:
        assert not hasattr(PromotionDecisionKind, "ELIGIBLE_FOR_LIVE_TRADING")
        assert "eligible_for_live_trading" not in {m.value for m in PromotionDecisionKind}


class TestDefaultPolicyMandatoryGatesAreAllMeasurable:
    def test_no_mandatory_gate_is_skipped_against_full_evidence(self) -> None:
        decision = evaluate_promotion(evidence=_evidence(), policy=DEFAULT_PROMOTION_POLICY)
        skipped_mandatory = [g.gate_name for g in decision.gate_evaluations if g.outcome.value == "skip" and g.mandatory]
        assert not skipped_mandatory
        assert decision.decision is PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING


class TestJsonRoundTrip:
    def test_promotion_decision_round_trips(self) -> None:
        decision = evaluate_promotion(evidence=_evidence())
        assert type(decision).from_json_dict(decision.to_json_dict()) == decision
