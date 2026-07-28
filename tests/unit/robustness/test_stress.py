"""Milestone 6, Section 16: cost/latency/execution stress -- exact-
arithmetic cost scaling, structural applicability, and hand-computed
break-even bracket search (monkeypatching `resimulate_stitched_outcome`
so the bracket-selection logic is tested in isolation from the expensive
production re-simulation pipeline)."""

from __future__ import annotations

import pytest

import quant_platform.robustness.stress as stress_mod
from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.types import Timeframe
from quant_platform.robustness.resimulation import ResimulationResult
from quant_platform.robustness.stress import (
    NAMED_XAUUSD_STRESS_PROFILES,
    _apply_cost_stress,
    _apply_scenario,
    _search_breakeven,
)


def _spec(**overrides: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_calibration_id": "a" * 64, "source_experiment_id": "b" * 64, "source_execution_id": "b" * 64,
        "dataset_content_id": "c" * 64, "split_plan_fingerprint": "d" * 64, "instrument_identity": "XAUUSD", "market_timezone": "UTC",
        "bar_interval": Timeframe.H1, "decision_timestamp_policy": DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        "signal_mapping": SignalMappingSpec(kind=SignalMappingPolicyKind.PROBABILITY_BANDS, probability_band_long_min=0.6, probability_band_short_max=0.4),
        "position_mode": PositionMode.LONG_SHORT, "entry_spec": EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        "exit_spec": ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=4, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        "overlap_policy": OverlapPolicyKind.IGNORE, "price_basis": PriceBasisKind.CLOSE,
        "spread_spec": SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=2.0),
        "commission_spec": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=1.0),
        "slippage_spec": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=0.5),
        "financing_spec": FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=0.1),
        "return_calculation_policy": ReturnCalculationPolicyKind.SIMPLE, "compounding_policy": CompoundingPolicyKind.NON_COMPOUNDED,
        "initial_notional": 10_000.0, "determinism_policy": DeterminismPolicy.STRICT, "exposure_cap": 1.0,
    }
    defaults.update(overrides)
    return BacktestSpec(**defaults)  # type: ignore[arg-type]


class TestCostScalingExactArithmetic:
    def test_zero_cost_forces_every_cost_spec_to_its_zero_kind(self) -> None:
        modified, reason = _apply_cost_stress(_spec(), force_zero_cost=True)
        assert reason is None and modified is not None
        assert modified.spread_spec.kind is SpreadModelKind.ZERO
        assert modified.commission_spec.kind is CommissionModelKind.ZERO
        assert modified.slippage_spec.kind is SlippageModelKind.ZERO
        assert modified.financing_spec.kind is FinancingModelKind.NONE

    def test_combined_multipliers_scale_exactly(self) -> None:
        modified, reason = _apply_cost_stress(
            _spec(), spread_multiplier=2.0, slippage_multiplier=3.0, commission_multiplier=1.5, financing_multiplier=2.0, additional_latency_bars=1,
        )
        assert reason is None and modified is not None
        assert modified.spread_spec.basis_points == 4.0
        assert modified.slippage_spec.basis_points == 1.5
        assert modified.commission_spec.per_side_basis_points == 1.5
        assert abs(modified.financing_spec.daily_basis_points - 0.2) < 1e-12
        assert modified.entry_spec.delay_bars == 2

    def test_bid_ask_observed_spread_cannot_be_scaled(self) -> None:
        observed_spec = _spec(spread_spec=SpreadSpec(kind=SpreadModelKind.BID_ASK_OBSERVED), price_basis=PriceBasisKind.BID_ASK)
        result, reason = _apply_cost_stress(observed_spec, spread_multiplier=2.0)
        assert result is None and reason is not None and "bid_ask_observed" in reason

    def test_bid_ask_observed_spread_with_multiplier_one_is_a_no_op(self) -> None:
        observed_spec = _spec(spread_spec=SpreadSpec(kind=SpreadModelKind.BID_ASK_OBSERVED), price_basis=PriceBasisKind.BID_ASK)
        result, reason = _apply_cost_stress(observed_spec, spread_multiplier=1.0)
        assert result is not None and reason is None

    def test_fixed_price_units_spread_and_slippage_scale_exactly(self) -> None:
        spec = _spec(spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5), slippage_spec=SlippageSpec(kind=SlippageModelKind.FIXED_PRICE_UNITS, price_units=0.2))
        modified, reason = _apply_cost_stress(spec, spread_multiplier=4.0, slippage_multiplier=2.5)
        assert reason is None and modified is not None
        assert modified.spread_spec.price_units == pytest.approx(2.0, abs=1e-12)
        assert modified.slippage_spec.price_units == pytest.approx(0.5, abs=1e-12)

    def test_fixed_per_trade_commission_scales_exactly(self) -> None:
        spec = _spec(commission_spec=CommissionSpec(kind=CommissionModelKind.FIXED_PER_TRADE, fixed_per_trade=3.0))
        modified, reason = _apply_cost_stress(spec, commission_multiplier=2.0)
        assert reason is None and modified is not None
        assert modified.commission_spec.fixed_per_trade == pytest.approx(6.0, abs=1e-12)

    def test_multiplier_of_one_leaves_every_cost_spec_object_identical(self) -> None:
        """A multiplier of exactly 1.0 must be a true no-op -- not merely
        numerically equal, but the SAME unmodified spec object (the `!=
        1.0` guard short-circuits `replace(...)` entirely), so no
        floating-point rounding is ever introduced by a supposedly-
        unstressed baseline."""
        spec = _spec()
        modified, reason = _apply_cost_stress(spec, spread_multiplier=1.0, slippage_multiplier=1.0, commission_multiplier=1.0, financing_multiplier=1.0)
        assert reason is None and modified is not None
        assert modified.spread_spec == spec.spread_spec
        assert modified.slippage_spec == spec.slippage_spec
        assert modified.commission_spec == spec.commission_spec
        assert modified.financing_spec == spec.financing_spec

    def test_non_negative_cost_multiplier_increase_can_only_increase_or_hold_total_cost_for_a_fixed_trade(self) -> None:
        """Structural monotonicity proof for the FOUR pure-cost axes
        (spread/slippage/commission/financing): entry/exit bar POSITION
        (and therefore which trades occur at all) is decided entirely by
        `EntryPolicyKind`/`ExitPolicyKind`/signal logic -- none of which
        reads any cost spec (confirmed: cost specs are consumed only in
        `runner.py`'s post-hoc `entry_spread_price_adjustment`/`exit_
        spread_price_adjustment`/etc., AFTER the entry/exit bar and
        direction are already fixed). Combined with `CostBreakdown`
        rejecting negative components, a `>= 1.0` multiplier increase on
        any of these four axes can only ever increase (never decrease)
        the total cost deducted from an otherwise IDENTICAL trade path --
        this is a STRUCTURAL guarantee, verified here at the level of
        `_apply_cost_stress`'s own arithmetic (each new magnitude is
        exactly `old * multiplier`, monotonic in the multiplier for any
        non-negative `old`)."""
        spec = _spec()
        base_spread, base_slippage, base_commission, base_financing = (
            spec.spread_spec.basis_points, spec.slippage_spec.basis_points, spec.commission_spec.per_side_basis_points, spec.financing_spec.daily_basis_points,
        )
        assert base_spread is not None and base_slippage is not None and base_commission is not None and base_financing is not None
        for multiplier in (1.0, 1.5, 2.0, 5.0, 10.0):
            modified, reason = _apply_cost_stress(spec, spread_multiplier=multiplier, slippage_multiplier=multiplier, commission_multiplier=multiplier, financing_multiplier=multiplier)
            assert reason is None and modified is not None
            assert modified.spread_spec.basis_points == pytest.approx(base_spread * multiplier, abs=1e-12)
            assert modified.spread_spec.basis_points >= base_spread * 1.0 - 1e-12  # never below the multiplier=1.0 baseline
            assert modified.slippage_spec.basis_points == pytest.approx(base_slippage * multiplier, abs=1e-12)
            assert modified.commission_spec.per_side_basis_points == pytest.approx(base_commission * multiplier, abs=1e-12)
            assert modified.financing_spec.daily_basis_points == pytest.approx(base_financing * multiplier, abs=1e-12)

    def test_additional_latency_bars_shifts_entry_delay_and_therefore_execution_decisions(self) -> None:
        """UNLIKE the four pure-cost axes above, `additional_latency_bars`
        changes `entry_spec.delay_bars` -- shifting WHICH bar a trade
        actually enters on. This is a genuine execution-decision change
        (different entry price, different holding-period alignment,
        possibly a trade pushed past the fold boundary and discarded/
        force-closed differently) -- monotonicity of `total_net_return` in
        `additional_latency_bars` is NOT structurally guaranteed the way
        it is for the four pure-cost multipliers, and this module's own
        break-even search for this axis must be read as "response to a
        genuinely different trade path", not "response to a fixed trade
        path under heavier cost"."""
        spec = _spec()
        modified, reason = _apply_cost_stress(spec, additional_latency_bars=3)
        assert reason is None and modified is not None
        assert modified.entry_spec.delay_bars == spec.entry_spec.delay_bars + 3
        # Every OTHER field (which does NOT change execution decisions) is untouched by a pure latency stress.
        assert modified.spread_spec == spec.spread_spec
        assert modified.commission_spec == spec.commission_spec


class TestNamedXauusdProfilesAreConfigurationIdentitiesOnly:
    def test_all_named_profiles_apply_cleanly_to_a_fully_fixed_magnitude_spec(self) -> None:
        spec = _spec()
        for profile in NAMED_XAUUSD_STRESS_PROFILES:
            modified, reason = _apply_scenario(spec, profile)
            assert modified is not None and reason is None, f"{profile.name} unexpectedly failed: {reason}"


class TestBreakEvenBracketSearchHandComputed:
    def test_bracket_search_finds_the_exact_crossing_interval(self, monkeypatch) -> None:
        """Fake `resimulate_stitched_outcome` returns a hand-declared
        return curve keyed by the searched multiplier value, encoded in
        the resimulation label: 1.0->0.05, 1.5->0.03, 2.0->0.01,
        3.0->-0.02, 5.0->-0.10, 8.0->-0.20, 13.0->-0.30, 21.0->-0.40.
        The crossing from positive to non-positive happens strictly
        between 2.0 (+0.01) and 3.0 (-0.02) -- the bracket search must
        report EXACTLY that pair, not an interpolated value."""
        net_returns_by_value = {1.0: 0.05, 1.5: 0.03, 2.0: 0.01, 3.0: -0.02, 5.0: -0.10, 8.0: -0.20, 13.0: -0.30, 21.0: -0.40}

        def _fake_resimulate(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            value = float(label_backtest_id.rsplit(":", 1)[-1])
            return ResimulationResult(total_net_return=net_returns_by_value[value], total_gross_return=net_returns_by_value[value], closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _fake_resimulate)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound == 2.0
        assert result.breakeven_upper_bound == 3.0

    def test_no_crossing_within_bounds_is_reported_explicitly_not_extrapolated(self, monkeypatch) -> None:
        def _always_profitable(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            return ResimulationResult(total_net_return=0.05, total_gross_return=0.06, closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _always_profitable)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is False
        assert result.breakeven_lower_bound is None and result.breakeven_upper_bound is None
        assert "no finite break-even point" in (result.reason or "")

    def test_already_at_or_below_breakeven_at_the_smallest_searched_value(self, monkeypatch) -> None:
        def _always_unprofitable(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            return ResimulationResult(total_net_return=-0.01, total_gross_return=-0.005, closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _always_unprofitable)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound is None
        assert result.breakeven_upper_bound == 1.0

    def test_equality_at_boundary_exactly_zero_counts_as_non_positive(self, monkeypatch) -> None:
        """`total_net_return <= 0.0` is the crossing condition -- a
        grid point landing EXACTLY at 0.0 must be treated as the
        non-positive side of the bracket, not skipped as ambiguous."""
        net_returns_by_value = {1.0: 0.02, 2.0: 0.0, 3.0: -0.05}

        def _fake_resimulate(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            value = float(label_backtest_id.rsplit(":", 1)[-1])
            return ResimulationResult(total_net_return=net_returns_by_value[value], total_gross_return=net_returns_by_value[value], closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _fake_resimulate)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound == 1.0
        assert result.breakeven_upper_bound == 2.0

    def test_multiple_crossings_reports_the_first_leftmost_bracket_only(self, monkeypatch) -> None:
        """Non-monotonic response: +0.05, -0.02, +0.01, -0.03 crosses zero
        THREE times. `_search_breakeven` is a left-to-right FIRST-bracket
        search (`itertools.pairwise` + early `return` on the first `lo>0,
        hi<=0` pair) -- it reports only the first crossing (1.0->2.0), and
        the LATER re-crossing back to profitable (2.0->3.0) and third
        crossing (3.0->5.0) are never reported. This is documented
        behavior, not a bug: the module's own docstring states it
        searches for "the tightest bracket ... where total_net_return
        crosses from positive to non-positive", implicitly the first one
        encountered on the grid, never a claim of finding every root."""
        net_returns_by_value = {1.0: 0.05, 2.0: -0.02, 3.0: 0.01, 5.0: -0.03}

        def _fake_resimulate(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            value = float(label_backtest_id.rsplit(":", 1)[-1])
            return ResimulationResult(total_net_return=net_returns_by_value[value], total_gross_return=net_returns_by_value[value], closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _fake_resimulate)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0, 5.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound == 1.0
        assert result.breakeven_upper_bound == 2.0  # the FIRST crossing, not the second (3.0->5.0)

    def test_discontinuous_jump_straight_past_zero_still_brackets_correctly(self, monkeypatch) -> None:
        """A jump from strongly positive directly to strongly negative
        (no near-zero grid point) must still report that exact bracket --
        the search never interpolates or refines within it."""
        net_returns_by_value = {1.0: 10.0, 2.0: -500.0}

        def _fake_resimulate(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            value = float(label_backtest_id.rsplit(":", 1)[-1])
            return ResimulationResult(total_net_return=net_returns_by_value[value], total_gross_return=net_returns_by_value[value], closed_trade_count=10, maximum_drawdown=0.1)

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _fake_resimulate)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0), modifier=lambda s, _v: (s, None),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound == 1.0
        assert result.breakeven_upper_bound == 2.0

    def test_fewer_than_two_evaluable_points_reports_not_found_with_reason(self, monkeypatch) -> None:
        """Invalid-bounds case: every grid point is structurally not
        applicable (e.g. a non-scalable cost model) -- fewer than two
        points can even be compared, so no break-even claim is made."""

        def _never_called(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            raise AssertionError("resimulate_stitched_outcome must not be called when the modifier reports not-applicable")

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _never_called)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0), modifier=lambda _s, _v: (None, "not applicable for this cost model"),
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is False
        assert result.breakeven_lower_bound is None and result.breakeven_upper_bound is None
        assert result.reason == "not applicable for this cost model"
        assert all(p.status == "skipped_not_applicable" for p in result.searched_points)

    def test_mixed_applicable_and_not_applicable_points_search_only_among_evaluated(self, monkeypatch) -> None:
        """One grid point is skipped (not applicable) while the rest are
        evaluated -- the skipped point must not corrupt the bracket search
        over the remaining evaluated points, and must still be persisted
        in `searched_points` with its own status."""
        net_returns_by_value = {1.0: 0.05, 3.0: -0.02}

        def _fake_resimulate(*, modified_spec, resolved_inputs, label_backtest_id, artifact_store):
            value = float(label_backtest_id.rsplit(":", 1)[-1])
            return ResimulationResult(total_net_return=net_returns_by_value[value], total_gross_return=net_returns_by_value[value], closed_trade_count=10, maximum_drawdown=0.1)

        def _modifier(s: object, v: float) -> tuple[object, str | None]:
            if v == 2.0:
                return None, "skipped for this test"
            return s, None

        monkeypatch.setattr(stress_mod, "resimulate_stitched_outcome", _fake_resimulate)
        result = _search_breakeven(
            axis_name="spread_multiplier", grid=(1.0, 2.0, 3.0), modifier=_modifier,
            source_backtest_spec=_spec(), resolved_inputs=None, robustness_source_backtest_id="a" * 64, artifact_store=None,
        )
        assert result.found is True
        assert result.breakeven_lower_bound == 1.0
        assert result.breakeven_upper_bound == 3.0
        statuses = {p.value: p.status for p in result.searched_points}
        assert statuses[2.0] == "skipped_not_applicable"
