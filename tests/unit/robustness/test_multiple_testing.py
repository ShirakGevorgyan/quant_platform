"""Milestone 6, Section 16: multiple-testing correction reference values
(hand-computed against Bonferroni/Holm/Benjamini-Hochberg textbook
definitions) and probabilistic/deflated Sharpe ratio / minimum track
record length fail-closed-under-missing-assumptions behavior.

PSR is checked against an EXACT closed-form reference value derived from
a symmetric (zero-skewness) dataset, where the kurtosis term algebraically
vanishes regardless of its actual value whenever the observed Sharpe
equals the benchmark Sharpe -- computed independently here via `math.erf`,
never by calling the module's own `_standard_normal_cdf`."""

from __future__ import annotations

import math

import pytest

from quant_platform.core.exceptions import MultipleTestingError
from quant_platform.robustness.models import MultipleTestingCorrectionKind
from quant_platform.robustness.multiple_testing import (
    apply_correction,
    benjamini_hochberg_correction,
    bonferroni_correction,
    build_strategy_family,
    compute_deflated_sharpe_ratio,
    compute_minimum_track_record_length,
    compute_probabilistic_sharpe_ratio,
    holm_correction,
)


class TestBonferroniHandComputed:
    def test_reference_values(self) -> None:
        adjusted = bonferroni_correction((0.005, 0.01, 0.03, 0.04, 0.20))
        expected = (0.025, 0.05, 0.15, 0.20, 1.0)
        for a, e in zip(adjusted, expected, strict=True):
            assert a == pytest.approx(e, abs=1e-12)

    def test_clips_at_one(self) -> None:
        assert bonferroni_correction((0.5, 0.5, 0.5)) == (1.0, 1.0, 1.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(MultipleTestingError):
            bonferroni_correction(())

    def test_rejects_out_of_range_p_value(self) -> None:
        with pytest.raises(MultipleTestingError):
            bonferroni_correction((1.5,))


class TestHolmHandComputed:
    def test_reference_values(self) -> None:
        """p=[0.005, 0.01, 0.03, 0.04, 0.20], m=5. Sorted (already
        ascending), raw = p_(k) * (m-k+1): [0.025, 0.04, 0.09, 0.08,
        0.20]; cumulative max enforces monotonicity: [0.025, 0.04, 0.09,
        0.09, 0.20]."""
        adjusted = holm_correction((0.005, 0.01, 0.03, 0.04, 0.20))
        expected = (0.025, 0.04, 0.09, 0.09, 0.20)
        for a, e in zip(adjusted, expected, strict=True):
            assert a == pytest.approx(e, abs=1e-12)

    def test_holm_is_never_less_conservative_than_bonferroni_elementwise(self) -> None:
        p_values = (0.001, 0.02, 0.03, 0.15, 0.4)
        holm = holm_correction(p_values)
        bonferroni = bonferroni_correction(p_values)
        assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni, strict=True))


class TestBenjaminiHochbergHandComputed:
    def test_reference_values(self) -> None:
        """p=[0.005, 0.01, 0.03, 0.04, 0.20], m=5. raw q(k)=p_(k)*m/k:
        [0.025, 0.025, 0.05, 0.05, 0.20]; cumulative min from the largest
        p-value downward: [0.025, 0.025, 0.05, 0.05, 0.20] (already
        monotonic in this example)."""
        adjusted = benjamini_hochberg_correction((0.005, 0.01, 0.03, 0.04, 0.20))
        expected = (0.025, 0.025, 0.05, 0.05, 0.20)
        for a, e in zip(adjusted, expected, strict=True):
            assert a == pytest.approx(e, abs=1e-12)

    def test_bh_is_never_more_conservative_than_holm_elementwise(self) -> None:
        p_values = (0.001, 0.02, 0.03, 0.15, 0.4)
        bh = benjamini_hochberg_correction(p_values)
        holm = holm_correction(p_values)
        assert all(b <= h + 1e-12 for b, h in zip(bh, holm, strict=True))

    def test_apply_correction_dispatches_correctly(self) -> None:
        p_values = (0.01, 0.02, 0.03)
        assert apply_correction(MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG, p_values) == benjamini_hochberg_correction(p_values)
        assert apply_correction(MultipleTestingCorrectionKind.HOLM, p_values) == holm_correction(p_values)
        assert apply_correction(MultipleTestingCorrectionKind.BONFERRONI, p_values) == bonferroni_correction(p_values)


class TestProbabilisticSharpeRatioClosedFormReference:
    _SYMMETRIC_RETURNS = (-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0)  # skew=0 exactly, sr_hat=0 exactly

    def test_psr_is_exactly_one_half_when_observed_equals_benchmark(self) -> None:
        """sr_hat=0 (symmetric series, mean 0) and benchmark_sharpe=0 ->
        z=0 -> PSR=Phi(0)=0.5 exactly, regardless of kurtosis (the
        kurtosis term is multiplied by sr_hat^2=0, so it algebraically
        cannot affect this particular result)."""
        result = compute_probabilistic_sharpe_ratio(self._SYMMETRIC_RETURNS, benchmark_sharpe=0.0)
        assert result.observed_sharpe == pytest.approx(0.0, abs=1e-12)
        assert result.skewness == pytest.approx(0.0, abs=1e-9)
        assert result.probabilistic_sharpe_ratio == pytest.approx(0.5, abs=1e-9)

    def test_psr_matches_independently_computed_phi_of_three(self) -> None:
        """benchmark_sharpe=-1 -> z = (0 - (-1)) * sqrt(9) / sqrt(1) = 3
        -> PSR = Phi(3), computed here independently via math.erf, never
        by calling the module's own `_standard_normal_cdf`."""
        result = compute_probabilistic_sharpe_ratio(self._SYMMETRIC_RETURNS, benchmark_sharpe=-1.0)
        independently_computed_phi_of_3 = 0.5 * (1.0 + math.erf(3.0 / math.sqrt(2.0)))
        assert result.probabilistic_sharpe_ratio == pytest.approx(independently_computed_phi_of_3, abs=1e-9)

    def test_requires_at_least_10_observations(self) -> None:
        with pytest.raises(MultipleTestingError, match="10"):
            compute_probabilistic_sharpe_ratio((0.01, 0.02, 0.03))

    def test_zero_variance_series_fails_closed(self) -> None:
        with pytest.raises(MultipleTestingError, match="variance"):
            compute_probabilistic_sharpe_ratio((0.01,) * 10)


class TestDeflatedSharpeRatioFailsClosedWithoutRealAssumptions:
    _RETURNS = (0.01, 0.02, -0.01, 0.015, 0.005, 0.02, -0.005, 0.01, 0.015, 0.02)

    def test_fewer_than_two_family_sharpes_fails_closed(self) -> None:
        with pytest.raises(MultipleTestingError, match="2"):
            compute_deflated_sharpe_ratio(self._RETURNS, sharpe_ratios_across_family=(1.2,))

    def test_zero_variance_family_sharpes_fails_closed(self) -> None:
        with pytest.raises(MultipleTestingError, match="standard deviation"):
            compute_deflated_sharpe_ratio(self._RETURNS, sharpe_ratios_across_family=(1.0, 1.0, 1.0))

    def test_sigma_sr_matches_independently_computed_population_stdev(self) -> None:
        import statistics

        family_sharpes = (0.5, 1.2, 0.8, 1.5, -0.2)
        result = compute_deflated_sharpe_ratio(self._RETURNS, sharpe_ratios_across_family=family_sharpes)
        assert result.sharpe_std_across_trials == pytest.approx(statistics.pstdev(family_sharpes), abs=1e-12)
        assert 0.0 <= result.deflated_sharpe_ratio <= 1.0


class TestMinimumTrackRecordLengthFailsClosed:
    def test_observed_sharpe_not_exceeding_benchmark_fails_closed(self) -> None:
        flat_returns = (0.0,) * 10
        with pytest.raises(MultipleTestingError):
            compute_minimum_track_record_length(flat_returns, benchmark_sharpe=0.0)

    def test_requires_at_least_10_observations(self) -> None:
        with pytest.raises(MultipleTestingError, match="10"):
            compute_minimum_track_record_length((0.01, 0.02, 0.03), benchmark_sharpe=0.0)

    def test_positive_case_returns_a_finite_value_at_least_one(self) -> None:
        """MinTRL = 1 + (non-negative term when the PSR variance term is
        well-formed) -- a full closed-form hand check is deferred (it
        would require re-deriving the kurtosis term independently, risking
        a correlated error with the implementation); this is a shape/
        bounds sanity check, not a claim of exact-value verification."""
        returns = (3.0, 4.0, 5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        result = compute_minimum_track_record_length(returns, benchmark_sharpe=0.0)
        assert math.isfinite(result)
        assert result >= 1.0


class TestStrategyFamilyDeterministicIdentity:
    def test_identical_inputs_produce_identical_family_id(self) -> None:
        kwargs: dict[str, object] = {"candidate_backtest_ids": ("a" * 64, "b" * 64), "search_space_identity": "grid_v1", "selection_metric": "total_net_return"}
        first = build_strategy_family(**kwargs)  # type: ignore[arg-type]
        second = build_strategy_family(**kwargs)  # type: ignore[arg-type]
        assert first.family_id == second.family_id

    def test_candidate_order_does_not_affect_family_id(self) -> None:
        """`to_identity_payload` sorts candidate id tuples -- a family
        built from the same candidates in a different order must produce
        the identical family_id."""
        a = build_strategy_family(candidate_backtest_ids=("a" * 64, "b" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        b = build_strategy_family(candidate_backtest_ids=("b" * 64, "a" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        assert a.family_id == b.family_id

    def test_empty_candidates_rejected(self) -> None:
        with pytest.raises(Exception, match="candidate_backtest_ids"):
            build_strategy_family(candidate_backtest_ids=(), search_space_identity="grid_v1", selection_metric="total_net_return")


class TestStrategyFamilyDeepAudit:
    """Closure-audit Section 7: duplicate rejection, content-sensitivity
    (not merely order), one-candidate families, and every declared
    identity-bearing field's participation in `family_id`."""

    def test_duplicate_candidate_backtest_id_rejected(self) -> None:
        with pytest.raises(MultipleTestingError, match="duplicate"):
            build_strategy_family(candidate_backtest_ids=("a" * 64, "a" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")

    def test_non_hex_candidate_id_rejected(self) -> None:
        with pytest.raises(MultipleTestingError, match="sha256"):
            build_strategy_family(candidate_backtest_ids=("not-a-valid-id",), search_space_identity="grid_v1", selection_metric="total_net_return")

    def test_one_candidate_family_is_legal(self) -> None:
        family = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return")
        assert family.candidate_count == 1

    def test_family_identity_changes_when_candidate_set_content_changes(self) -> None:
        """Distinct from the order-independence test above: REPLACING one
        candidate (not merely reordering the same set) must change
        `family_id`."""
        a = build_strategy_family(candidate_backtest_ids=("a" * 64, "b" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        b = build_strategy_family(candidate_backtest_ids=("a" * 64, "c" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        assert a.family_id != b.family_id

    def test_family_identity_changes_when_a_candidate_is_removed(self) -> None:
        """The `family_id`-is-a-pure-hash-of-content design makes "silent
        reduction after observing results" structurally impossible to do
        UNDETECTED: dropping a candidate always produces a DIFFERENT
        family_id, never the same one -- there is no way to shrink a
        family and keep its identity."""
        full = build_strategy_family(candidate_backtest_ids=("a" * 64, "b" * 64, "c" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        reduced = build_strategy_family(candidate_backtest_ids=("a" * 64, "c" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        assert full.family_id != reduced.family_id

    def test_search_space_identity_participates_in_family_id(self) -> None:
        a = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return")
        b = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v2", selection_metric="total_net_return")
        assert a.family_id != b.family_id

    def test_selection_metric_participates_in_family_id(self) -> None:
        a = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return")
        b = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="sharpe")
        assert a.family_id != b.family_id

    def test_eligibility_rules_description_participates_in_family_id(self) -> None:
        a = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return", eligibility_rules_description="min 30 trades")
        b = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return", eligibility_rules_description="min 50 trades")
        assert a.family_id != b.family_id

    def test_candidate_experiment_calibration_optimization_ids_participate_in_family_id(self) -> None:
        base = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return")
        with_experiment = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return", candidate_experiment_ids=("e" * 64,))
        with_calibration = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return", candidate_calibration_ids=("c" * 64,))
        with_optimization = build_strategy_family(candidate_backtest_ids=("a" * 64,), search_space_identity="grid_v1", selection_metric="total_net_return", candidate_optimization_identities=("o" * 64,))
        assert len({base.family_id, with_experiment.family_id, with_calibration.family_id, with_optimization.family_id}) == 4

    def test_json_round_trip_preserves_every_candidate_id_field(self) -> None:
        family = build_strategy_family(
            candidate_backtest_ids=("a" * 64, "b" * 64), search_space_identity="grid_v1", selection_metric="total_net_return",
            candidate_experiment_ids=("e" * 64,), candidate_calibration_ids=("c" * 64,), candidate_optimization_identities=("o" * 64,),
            eligibility_rules_description="min 30 trades",
        )
        from quant_platform.robustness.multiple_testing import StrategyFamily

        roundtripped = StrategyFamily.from_json_dict(family.to_json_dict())
        assert roundtripped == family

    def test_heterogeneous_dataset_instrument_split_plan_bar_interval_are_not_validated(self) -> None:
        """CLOSURE-AUDIT FINDING (Section 7): `StrategyFamily`/`build_
        strategy_family` receive only opaque ID STRINGS for each
        candidate -- never the candidates' own `BacktestSpec` (dataset_
        content_id/split_plan_fingerprint/instrument_identity/
        bar_interval) -- so there is NO structural validation anywhere in
        this module that every candidate in a family shares the same
        dataset, split plan, instrument, or bar interval. A family mixing
        (e.g.) an XAUUSD candidate with a EURUSD candidate, or candidates
        evaluated on different historical partitions, is constructed
        successfully with no error and no warning. This is a documented,
        NON-BLOCKING LIMITATION (not a defect introduced by this
        milestone's own logic, and not silently mis-stated anywhere as
        validated) -- comparing statistically incompatible candidates
        within one family/DSR/selection computation is a real risk left
        entirely to OPERATOR DISCIPLINE. This test exists so that
        behavior is explicit and tracked, not merely assumed."""
        # Two families sharing zero semantic relationship construct
        # successfully -- proving no cross-candidate compatibility check
        # exists anywhere in this constructor.
        family = build_strategy_family(candidate_backtest_ids=("a" * 64, "b" * 64), search_space_identity="grid_v1", selection_metric="total_net_return")
        assert family.candidate_count == 2  # no rejection, no warning field anywhere on StrategyFamily
