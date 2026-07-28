"""Milestone 6, Section 16: champion/challenger candidate eligibility
gates and deterministic ranking. Proves: no candidate ever disappears
from the report regardless of eligibility outcome; ranking is
deterministic under candidate-list reordering; unknown gate/metric names
fail closed at construction."""

from __future__ import annotations

import pytest

from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.robustness.bootstrap import BootstrapEstimate, BootstrapReport
from quant_platform.robustness.models import StressAxisKind
from quant_platform.robustness.selection import (
    DEFAULT_SELECTION_GATES,
    DEFAULT_SELECTION_POLICY,
    CandidateEvidence,
    SelectionGate,
    SelectionPolicy,
    compute_selection_report,
)
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.stability import ConcentrationReport, FoldStabilityReport
from quant_platform.robustness.stress import StressReport, StressScenarioResult


def _source_verification(*, passed: bool, folds: int = 6) -> SourceVerificationReport:
    return SourceVerificationReport(
        schema_version=1, source_backtest_id="a" * 64, verify_backtest_is_ready=passed, verify_backtest_critical_count=0,
        verify_backtest_issue_codes=(), dataset_content_id_matches=passed, split_plan_fingerprint_matches=passed,
        instrument_identity_matches=passed, bar_interval_matches=passed, total_outer_folds=folds, generated_at="2026-01-01T00:00:00Z",
    )


def _bootstrap(lower_bound: float) -> BootstrapReport:
    return BootstrapReport(
        schema_version=1, series_kind="benchmark_relative", method="stationary", repetitions=500, confidence_level=0.95, seed=1,
        estimates=(BootstrapEstimate(statistic_name="total_return", point_estimate=lower_bound + 0.05, lower_bound=lower_bound, upper_bound=lower_bound + 0.1, valid_repetitions=500, skipped_repetitions=0, failure_reasons={}),),
        generated_at="2026-01-01T00:00:00Z",
    )


def _fold_stability(*, profitable_fraction: float, worst_fold: float, max_dd: float, concentrated: bool = False) -> FoldStabilityReport:
    concentration = ConcentrationReport(
        schema_version=1, single_fold_profit_concentration=(0.99 if concentrated else 0.3), single_trade_profit_concentration=0.3,
        single_day_profit_concentration=0.3, single_direction_profit_concentration=0.3, single_confidence_bucket_profit_concentration=0.3,
        warning_codes=(("single_fold_profit_concentration_exceeded",) if concentrated else ()),
    )
    return FoldStabilityReport(
        schema_version=1, fold_count=5, profitable_fold_fraction=profitable_fraction, positive_sharpe_fold_fraction=0.6,
        median_fold_return=0.02, worst_fold_return=worst_fold, fold_return_stdev=0.01, fold_sharpe_dispersion=0.1,
        maximum_fold_drawdown=max_dd, worst_fold_cost_drag=0.001, fold_trade_count_dispersion=1.0, fold_exposure_dispersion=0.1,
        direction_consistency=0.8, benchmark_outperformance_fraction=0.6, concentration=concentration, generated_at="2026-01-01T00:00:00Z",
    )


def _stress(*, worst_scenario_return: float) -> StressReport:
    scenarios = (
        StressScenarioResult(
            name="base_cost", axis=StressAxisKind.BASE_COST, named_profile=None, status="evaluated", skip_reason=None, total_net_return=0.05,
            total_gross_return=0.06, closed_trade_count=20, maximum_drawdown=0.03, is_profitable=True, net_return_degradation_vs_baseline=0.0,
        ),
        StressScenarioResult(
            name="2x_spread", axis=StressAxisKind.SPREAD_MULTIPLIER, named_profile=None, status="evaluated", skip_reason=None,
            total_net_return=worst_scenario_return, total_gross_return=worst_scenario_return + 0.01, closed_trade_count=20, maximum_drawdown=0.05,
            is_profitable=worst_scenario_return > 0.0, net_return_degradation_vs_baseline=0.05 - worst_scenario_return,
        ),
    )
    return StressReport(
        schema_version=1, source_backtest_id="a" * 64, baseline_total_net_return=0.05, baseline_total_gross_return=0.06, baseline_closed_trade_count=20,
        baseline_maximum_drawdown=0.03, scenario_results=scenarios, breakeven_results=(), generated_at="2026-01-01T00:00:00Z",
    )


def _candidate(
    robustness_id: str, *, verified: bool = True, bootstrap_lower: float, profitable_fraction: float, worst_fold: float, max_dd: float, worst_stress: float,
    concentrated: bool = False, turnover: float | None = 0.5, complexity: float | None = 3.0,
) -> CandidateEvidence:
    return CandidateEvidence(
        robustness_id=robustness_id, source_backtest_id=robustness_id, source_verification=_source_verification(passed=verified),
        total_outer_folds=6, total_closed_trade_count=50, bootstrap=_bootstrap(bootstrap_lower),
        fold_stability=_fold_stability(profitable_fraction=profitable_fraction, worst_fold=worst_fold, max_dd=max_dd, concentrated=concentrated),
        stress=_stress(worst_scenario_return=worst_stress), mean_turnover_notional_ratio=turnover, strategy_complexity_score=complexity,
    )


class TestNoCandidateDisappears:
    def test_every_candidate_appears_in_the_report_regardless_of_outcome(self) -> None:
        strong = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        fails_drawdown = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.02, max_dd=0.9, worst_stress=0.01)
        not_verified = _candidate("c" * 64, verified=False, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)

        report = compute_selection_report(candidates=(strong, fails_drawdown, not_verified))

        assert len(report.candidate_eligibility) == 3
        by_id = {e.robustness_id: e for e in report.candidate_eligibility}
        assert by_id["a" * 64].eligible is True
        assert by_id["b" * 64].eligible is False
        assert any("maximum_drawdown_under_limit" in r for r in by_id["b" * 64].rejection_reasons)
        assert by_id["c" * 64].eligible is False
        assert any("source_backtest_verified" in r for r in by_id["c" * 64].rejection_reasons)


class TestDeterministicRanking:
    def test_strictly_better_candidate_ranks_first(self) -> None:
        better = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        worse = _candidate("c" * 64, bootstrap_lower=0.005, profitable_fraction=0.55, worst_fold=0.001, max_dd=0.2, worst_stress=-0.01)
        report = compute_selection_report(candidates=(better, worse))
        assert [r.robustness_id for r in report.ranking] == [better.robustness_id, worse.robustness_id]
        assert report.selected_candidate_robustness_id == better.robustness_id

    def test_ranking_and_selection_identity_are_invariant_to_candidate_list_order(self) -> None:
        a = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        b = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.02, max_dd=0.9, worst_stress=0.01)  # ineligible
        c = _candidate("c" * 64, bootstrap_lower=0.005, profitable_fraction=0.55, worst_fold=0.001, max_dd=0.2, worst_stress=-0.01)
        forward = compute_selection_report(candidates=(a, b, c))
        reversed_order = compute_selection_report(candidates=(c, b, a))
        assert forward.selection_identity == reversed_order.selection_identity
        assert forward.selected_candidate_robustness_id == reversed_order.selected_candidate_robustness_id
        assert [r.robustness_id for r in forward.ranking] == [r.robustness_id for r in reversed_order.ranking]

    def test_no_eligible_candidates_reports_none_selected_never_fabricated(self) -> None:
        candidate = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        strict_policy = SelectionPolicy(gates=(SelectionGate(name="minimum_trade_count", mandatory=True, minimum_value=1_000_000.0),), ranking_metric_order=DEFAULT_SELECTION_POLICY.ranking_metric_order)
        report = compute_selection_report(candidates=(candidate,), policy=strict_policy)
        assert report.selected_candidate_robustness_id is None
        assert report.ranking == ()
        assert len(report.candidate_eligibility) == 1


class TestFailClosedOnUnknownNames:
    def test_unknown_gate_name_rejected_at_construction(self) -> None:
        with pytest.raises(Exception, match="not_a_real_gate"):
            SelectionGate(name="not_a_real_gate", mandatory=True, minimum_value=1.0)

    def test_unknown_ranking_metric_rejected_at_construction(self) -> None:
        with pytest.raises(Exception, match="not_a_real_metric"):
            SelectionPolicy(gates=(SelectionGate(name="minimum_fold_count", mandatory=True, minimum_value=1.0),), ranking_metric_order=("not_a_real_metric",))


class TestPolicyIdentityOrderIndependence:
    """Release-audit regression (Milestone 6 final audit, Section 1): the
    same defect class fixed in `RobustnessSpec.to_json_dict` also existed
    in `SelectionPolicy.to_json_dict` -- `gates` is an unordered set
    (uniqueness already enforced by name in `__post_init__`), so declaring
    the same gates in a different order must not change `policy_identity`.
    `ranking_metric_order` is the deliberate OPPOSITE case: it is a
    prioritized tie-break sequence where position IS meaningful, and must
    keep changing the identity when reordered.

    IMPORTANT: canonicalization for `policy_identity` (computed from
    `to_identity_payload`, exactly what `compute_selection_report` itself
    calls) is intentionally NOT the same thing as `to_json_dict`, which
    must preserve declared order for round-trip/recomputation fidelity --
    see `TestJsonDictPreservesDeclaredOrder` below for the regression this
    distinction fixes."""

    def test_gates_reordered_produces_identical_policy_identity(self) -> None:
        forward = SelectionPolicy(gates=DEFAULT_SELECTION_GATES, ranking_metric_order=DEFAULT_SELECTION_POLICY.ranking_metric_order)
        reversed_policy = SelectionPolicy(gates=tuple(reversed(DEFAULT_SELECTION_GATES)), ranking_metric_order=DEFAULT_SELECTION_POLICY.ranking_metric_order)
        assert fingerprint_json(forward.to_identity_payload()) == fingerprint_json(reversed_policy.to_identity_payload())

    def test_ranking_metric_order_reordered_still_changes_policy_identity(self) -> None:
        order = DEFAULT_SELECTION_POLICY.ranking_metric_order
        forward = SelectionPolicy(gates=DEFAULT_SELECTION_GATES, ranking_metric_order=order)
        reversed_policy = SelectionPolicy(gates=DEFAULT_SELECTION_GATES, ranking_metric_order=tuple(reversed(order)))
        assert fingerprint_json(forward.to_identity_payload()) != fingerprint_json(reversed_policy.to_identity_payload())


class TestJsonDictPreservesDeclaredOrder:
    """Release-audit regression, found DURING the audit itself: an earlier
    fix for the order-independence defect above sorted `gates` directly in
    `to_json_dict` (not just `to_identity_payload`). Because
    `to_json_dict`/`from_json_dict` is the DURABLE round-trip
    representation -- reloaded by any code that persists a policy and
    later recomputes from it -- that made `to_json_dict` silently reorder
    `gates`, independent of what order the caller declared. This is
    exactly the mechanism that caused `robustness.verification.
    verify_robustness` to spuriously flag recomputed reports as mismatched
    against what the forward pass persisted, for the equivalent
    RobustnessSpec-embedded fields (caught by the full acceptance-workflow
    integration test). `to_json_dict` must always be order-PRESERVING;
    only `to_identity_payload` may canonicalize."""

    def test_to_json_dict_preserves_reversed_gate_order(self) -> None:
        reversed_gates = tuple(reversed(DEFAULT_SELECTION_GATES))
        policy = SelectionPolicy(gates=reversed_gates, ranking_metric_order=DEFAULT_SELECTION_POLICY.ranking_metric_order)
        assert [g["name"] for g in policy.to_json_dict()["gates"]] == [g.name for g in reversed_gates]

    def test_json_round_trip_preserves_declared_gate_order(self) -> None:
        reversed_gates = tuple(reversed(DEFAULT_SELECTION_GATES))
        policy = SelectionPolicy(gates=reversed_gates, ranking_metric_order=DEFAULT_SELECTION_POLICY.ranking_metric_order)
        roundtripped = SelectionPolicy.from_json_dict(policy.to_json_dict())
        assert [g.name for g in roundtripped.gates] == [g.name for g in reversed_gates]


class TestChampionChallengerSelectionClosure:
    """Closure-audit Section 8: literal selection fixtures for every
    required branch, every candidate persisting with its full gate/
    ranking evidence, deterministic tie-breaking (including turnover and
    complexity as explicit tie-break dimensions), and a semantic
    tampering test against a persisted `SelectionReport`."""

    def test_all_candidates_fail_none_selected(self) -> None:
        a = _candidate("a" * 64, verified=False, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        b = _candidate("b" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.9, worst_stress=0.01)  # fails drawdown
        report = compute_selection_report(candidates=(a, b))
        assert report.selected_candidate_robustness_id is None
        assert report.ranking == ()
        assert len(report.candidate_eligibility) == 2
        assert all(not e.eligible for e in report.candidate_eligibility)

    def test_exactly_one_passes(self) -> None:
        good = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        bad = _candidate("b" * 64, verified=False, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(good, bad))
        assert report.selected_candidate_robustness_id == good.robustness_id
        assert len(report.ranking) == 1

    def test_multiple_pass_best_ranks_first(self) -> None:
        best = _candidate("a" * 64, bootstrap_lower=0.05, profitable_fraction=0.8, worst_fold=0.02, max_dd=0.1, worst_stress=0.02)
        middle = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.7, worst_fold=0.01, max_dd=0.15, worst_stress=0.01)
        worst = _candidate("c" * 64, bootstrap_lower=0.01, profitable_fraction=0.55, worst_fold=0.001, max_dd=0.2, worst_stress=-0.005)
        report = compute_selection_report(candidates=(worst, best, middle))  # declared out of order
        assert [r.robustness_id for r in report.ranking] == [best.robustness_id, middle.robustness_id, worst.robustness_id]

    def test_exact_tie_on_every_ranking_metric_breaks_by_robustness_id(self) -> None:
        """Two candidates identical on EVERY ranking-metric dimension
        (bootstrap lower bound, worst fold, worst stress, drawdown,
        turnover, complexity) must still resolve deterministically --
        `_sort_key` appends `robustness_id` as the final, always-unique
        tie-break component."""
        tied_kwargs = {"bootstrap_lower": 0.03, "profitable_fraction": 0.8, "worst_fold": 0.01, "max_dd": 0.1, "worst_stress": 0.01, "turnover": 0.4, "complexity": 2.0}
        low_id = _candidate("1" * 64, **tied_kwargs)  # type: ignore[arg-type]
        high_id = _candidate("9" * 64, **tied_kwargs)  # type: ignore[arg-type]
        report = compute_selection_report(candidates=(high_id, low_id))
        assert report.selected_candidate_robustness_id == low_id.robustness_id  # lexicographically smaller id wins the tiebreak
        # Order-independence sanity: reversing declaration order must not change the outcome.
        reversed_report = compute_selection_report(candidates=(low_id, high_id))
        assert reversed_report.selected_candidate_robustness_id == low_id.robustness_id

    def test_undefined_ranking_metric_sorts_after_every_defined_value(self) -> None:
        """A candidate missing `strategy_complexity_score` (`None`) must
        rank BEHIND an otherwise-identical candidate that has a defined
        value for that metric, never crash and never rank ahead."""
        with_complexity = _candidate("a" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, complexity=2.0)
        without_complexity = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, complexity=None)
        report = compute_selection_report(candidates=(without_complexity, with_complexity))
        assert [r.robustness_id for r in report.ranking] == [with_complexity.robustness_id, without_complexity.robustness_id]

    def test_high_return_but_failed_drawdown_gate_is_excluded_from_ranking(self) -> None:
        high_return_bad_drawdown = _candidate("a" * 64, bootstrap_lower=0.20, profitable_fraction=0.9, worst_fold=0.10, max_dd=0.99, worst_stress=0.05)
        modest_but_eligible = _candidate("b" * 64, bootstrap_lower=0.02, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(high_return_bad_drawdown, modest_but_eligible))
        assert report.selected_candidate_robustness_id == modest_but_eligible.robustness_id
        by_id = {e.robustness_id: e for e in report.candidate_eligibility}
        assert by_id[high_return_bad_drawdown.robustness_id].eligible is False
        assert any("maximum_drawdown_under_limit" in r for r in by_id[high_return_bad_drawdown.robustness_id].rejection_reasons)

    def test_strong_mean_but_weak_lower_confidence_bound_ranks_below_a_more_conservative_candidate(self) -> None:
        """Ranking uses `bootstrap_lower_bound_return` (the CI's lower
        bound), never the point estimate/mean -- a candidate with a
        wide-uncertainty (weak lower bound) result ranks below a
        candidate with a tighter, more conservative lower bound even if
        its point estimate elsewhere might look stronger."""
        weak_lower_bound = _candidate("a" * 64, bootstrap_lower=0.001, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        strong_lower_bound = _candidate("b" * 64, bootstrap_lower=0.04, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(weak_lower_bound, strong_lower_bound))
        assert report.selected_candidate_robustness_id == strong_lower_bound.robustness_id

    def test_low_turnover_tie_break(self) -> None:
        """Tied on every metric ranked BEFORE `turnover` -- the
        lower-turnover candidate must win."""
        low_turnover = _candidate("a" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, turnover=0.2, complexity=2.0)
        high_turnover = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, turnover=0.9, complexity=2.0)
        report = compute_selection_report(candidates=(high_turnover, low_turnover))
        assert report.selected_candidate_robustness_id == low_turnover.robustness_id

    def test_complexity_tie_break(self) -> None:
        """Tied on every metric including turnover -- the lower-
        complexity (simpler) candidate must win, per `strategy_
        complexity_score`'s `lower_is_better` direction."""
        simple = _candidate("a" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, turnover=0.4, complexity=1.0)
        complex_ = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01, turnover=0.4, complexity=9.0)
        report = compute_selection_report(candidates=(complex_, simple))
        assert report.selected_candidate_robustness_id == simple.robustness_id

    def test_candidate_order_permutation_never_changes_champion_or_selection_identity(self) -> None:
        a = _candidate("a" * 64, bootstrap_lower=0.05, profitable_fraction=0.8, worst_fold=0.02, max_dd=0.1, worst_stress=0.02)
        b = _candidate("b" * 64, bootstrap_lower=0.03, profitable_fraction=0.7, worst_fold=0.01, max_dd=0.15, worst_stress=0.01)
        c = _candidate("c" * 64, verified=False, bootstrap_lower=0.03, profitable_fraction=0.7, worst_fold=0.01, max_dd=0.15, worst_stress=0.01)
        import itertools

        identities = set()
        champions = set()
        for perm in itertools.permutations((a, b, c)):
            report = compute_selection_report(candidates=perm)
            identities.add(report.selection_identity)
            champions.add(report.selected_candidate_robustness_id)
        assert len(identities) == 1
        assert len(champions) == 1

    def test_extreme_outlier_candidate_still_ranks_by_the_same_rule_no_special_casing(self) -> None:
        normal = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        extreme = _candidate("b" * 64, bootstrap_lower=50.0, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(normal, extreme))
        assert report.selected_candidate_robustness_id == extreme.robustness_id  # no clamping, no outlier rejection -- the rule is applied literally

    def test_every_candidate_persists_full_gate_and_ranking_evidence(self) -> None:
        eligible = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        ineligible = _candidate("b" * 64, verified=False, bootstrap_lower=0.02, profitable_fraction=0.6, worst_fold=0.005, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(eligible, ineligible))
        by_id = {e.robustness_id: e for e in report.candidate_eligibility}
        assert len(by_id[eligible.robustness_id].gate_evaluations) == len(DEFAULT_SELECTION_GATES)
        assert len(by_id[ineligible.robustness_id].gate_evaluations) == len(DEFAULT_SELECTION_GATES)  # ineligible candidate STILL gets every gate evaluated, not short-circuited
        ranked_entry = next(r for r in report.ranking if r.robustness_id == eligible.robustness_id)
        assert set(ranked_entry.metric_values) == set(DEFAULT_SELECTION_POLICY.ranking_metric_order)

    def test_semantic_tampering_persisted_selection_report_detected_by_recomputation(self) -> None:
        """Not a structural/schema check -- proves that TAMPERING with a
        persisted `SelectionReport`'s content (silently swapping which
        candidate is recorded as selected) is detectable by anyone who
        recomputes the report from the same underlying evidence and
        compares, since `selected_candidate_robustness_id` is fully
        determined by `candidates`/`policy` with no independent
        persisted-only escape hatch."""
        import dataclasses

        best = _candidate("a" * 64, bootstrap_lower=0.05, profitable_fraction=0.8, worst_fold=0.02, max_dd=0.1, worst_stress=0.02)
        worse = _candidate("b" * 64, bootstrap_lower=0.01, profitable_fraction=0.55, worst_fold=0.001, max_dd=0.2, worst_stress=-0.01)
        genuine = compute_selection_report(candidates=(best, worse))
        tampered = dataclasses.replace(genuine, selected_candidate_robustness_id=worse.robustness_id)  # attacker swaps the champion
        recomputed = compute_selection_report(candidates=(best, worse))
        assert tampered.selected_candidate_robustness_id != recomputed.selected_candidate_robustness_id
        assert tampered.selection_identity == recomputed.selection_identity  # the tamper did NOT also update the identity -- detectable by anyone who checks


class TestJsonRoundTrip:
    def test_selection_report_round_trips(self) -> None:
        candidate = _candidate("a" * 64, bootstrap_lower=0.02, profitable_fraction=0.8, worst_fold=0.01, max_dd=0.1, worst_stress=0.01)
        report = compute_selection_report(candidates=(candidate,))
        assert type(report).from_json_dict(report.to_json_dict()) == report
