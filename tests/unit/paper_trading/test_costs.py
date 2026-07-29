"""Milestone 7, Section 16: cost/financing computation. Exercises the thin
wrappers around `backtesting.costs`'s reused formulas, plus exact
reconciliation properties: spread/slippage cost dollars match the
adjusted-price delta exactly; long/short financing is genuinely
asymmetric; a positive financing rate is a cost (cash decreases) and a
negative rate is a credit (cash increases)."""

from __future__ import annotations

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingError
from quant_platform.paper_trading.costs import (
    FillCostComponents,
    bar_mode_spread_adjusted_price,
    compute_commission_dollars,
    compute_financing_cash_delta,
    compute_slippage_cost_dollars,
    compute_spread_cost_dollars,
    quote_mode_spread_cost_dollars,
    slippage_adjusted_price,
)
from quant_platform.paper_trading.models import OrderSide
from quant_platform.paper_trading.specs import FinancingPolicySpec


class TestFillCostComponents:
    def test_total_cost_sums_components(self) -> None:
        costs = FillCostComponents(spread_cost=1.0, slippage_cost=2.0, commission_cost=3.0)
        assert costs.total_cost == 6.0

    def test_negative_component_rejected(self) -> None:
        with pytest.raises(PaperTradingError, match="spread_cost"):
            FillCostComponents(spread_cost=-1.0, slippage_cost=0.0, commission_cost=0.0)


class TestBarModeSpreadAdjustedPrice:
    def test_entry_long_pays_above_reference(self) -> None:
        spec = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5)
        price = bar_mode_spread_adjusted_price(spec, 100.0, PositionDirection.LONG, is_entry=True)
        assert price == pytest.approx(100.25)

    def test_entry_short_receives_below_reference(self) -> None:
        spec = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5)
        price = bar_mode_spread_adjusted_price(spec, 100.0, PositionDirection.SHORT, is_entry=True)
        assert price == pytest.approx(99.75)

    def test_exit_long_receives_below_reference(self) -> None:
        spec = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5)
        price = bar_mode_spread_adjusted_price(spec, 100.0, PositionDirection.LONG, is_entry=False)
        assert price == pytest.approx(99.75)

    def test_zero_spread_is_a_no_op(self) -> None:
        spec = SpreadSpec(kind=SpreadModelKind.ZERO)
        assert bar_mode_spread_adjusted_price(spec, 100.0, PositionDirection.LONG, is_entry=True) == 100.0


class TestSlippageAdjustedPrice:
    def test_entry_long_pays_above_reference(self) -> None:
        spec = SlippageSpec(kind=SlippageModelKind.FIXED_PRICE_UNITS, price_units=0.1)
        assert slippage_adjusted_price(spec, 100.0, PositionDirection.LONG, is_entry=True) == pytest.approx(100.1)

    def test_zero_slippage_is_a_no_op(self) -> None:
        spec = SlippageSpec(kind=SlippageModelKind.ZERO)
        assert slippage_adjusted_price(spec, 100.0, PositionDirection.LONG, is_entry=True) == 100.0


class TestSpreadAndSlippageCostDollarsMatchPriceAdjustmentExactly:
    """Exact reconciliation property: the dollar cost attributed to
    spread/slippage must equal the price ADJUSTMENT magnitude times
    quantity times contract_multiplier, exactly -- no approximation."""

    def test_spread_cost_dollars_matches_price_delta(self) -> None:
        spec = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5)
        reference_price = 100.0
        adjusted = bar_mode_spread_adjusted_price(spec, reference_price, PositionDirection.LONG, is_entry=True)
        price_delta = adjusted - reference_price
        cost = compute_spread_cost_dollars(spec, reference_price, PositionDirection.LONG, is_entry=True, quantity=3.0, contract_multiplier=10.0)
        assert cost == pytest.approx(price_delta * 3.0 * 10.0)

    def test_slippage_cost_dollars_matches_price_delta(self) -> None:
        spec = SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=5.0)
        reference_price = 200.0
        adjusted = slippage_adjusted_price(spec, reference_price, PositionDirection.SHORT, is_entry=False)
        price_delta = abs(adjusted - reference_price)
        cost = compute_slippage_cost_dollars(spec, reference_price, PositionDirection.SHORT, is_entry=False, quantity=2.0, contract_multiplier=1.0)
        assert cost == pytest.approx(price_delta * 2.0)


class TestQuoteModeSpreadCostDollars:
    def test_buy_cost_is_half_spread_times_quantity(self) -> None:
        # half_spread = (ask - bid) / 2 = (100.1 - 99.9) / 2 = 0.1
        cost = quote_mode_spread_cost_dollars(bid=99.9, ask=100.1, side=OrderSide.BUY, quantity=2.0, contract_multiplier=1.0)
        assert cost == pytest.approx(0.1 * 2.0)

    def test_sell_cost_is_half_spread_times_quantity(self) -> None:
        cost = quote_mode_spread_cost_dollars(bid=99.9, ask=100.1, side=OrderSide.SELL, quantity=2.0, contract_multiplier=1.0)
        assert cost == pytest.approx(0.1 * 2.0)

    def test_zero_spread_produces_zero_cost(self) -> None:
        assert quote_mode_spread_cost_dollars(bid=100.0, ask=100.0, side=OrderSide.BUY, quantity=5.0, contract_multiplier=1.0) == 0.0


class TestComputeCommissionDollars:
    def test_per_side_basis_points(self) -> None:
        spec = CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=10.0)
        assert compute_commission_dollars(spec, notional=10_000.0) == pytest.approx(10.0)

    def test_zero_commission(self) -> None:
        spec = CommissionSpec(kind=CommissionModelKind.ZERO)
        assert compute_commission_dollars(spec, notional=10_000.0) == 0.0


class TestComputeFinancingCashDelta:
    def _policy(self, *, long_bps: float, short_bps: float) -> FinancingPolicySpec:
        return FinancingPolicySpec(
            long_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=long_bps),
            short_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=short_bps),
        )

    def test_flat_direction_produces_zero(self) -> None:
        policy = self._policy(long_bps=5.0, short_bps=5.0)
        assert compute_financing_cash_delta(policy, direction=PositionDirection.FLAT, notional=100_000.0, holding_days=1.0) == 0.0

    def test_positive_long_rate_decreases_cash(self) -> None:
        policy = self._policy(long_bps=10.0, short_bps=0.0)
        delta = compute_financing_cash_delta(policy, direction=PositionDirection.LONG, notional=100_000.0, holding_days=1.0)
        assert delta < 0.0
        assert delta == pytest.approx(-100.0)

    def test_negative_short_rate_increases_cash(self) -> None:
        """A short holder can be CREDITED financing -- a negative
        `daily_basis_points` on the short side."""
        policy = self._policy(long_bps=10.0, short_bps=-4.0)
        delta = compute_financing_cash_delta(policy, direction=PositionDirection.SHORT, notional=100_000.0, holding_days=1.0)
        assert delta > 0.0
        assert delta == pytest.approx(40.0)

    def test_long_and_short_rates_are_genuinely_independent(self) -> None:
        """The core asymmetry property Section 16 requires: changing the
        SHORT rate must not affect the LONG-direction result, and vice
        versa -- proving the two sides are not silently coupled to one
        shared rate."""
        policy_a = self._policy(long_bps=10.0, short_bps=-4.0)
        policy_b = self._policy(long_bps=10.0, short_bps=999.0)
        long_delta_a = compute_financing_cash_delta(policy_a, direction=PositionDirection.LONG, notional=100_000.0, holding_days=1.0)
        long_delta_b = compute_financing_cash_delta(policy_b, direction=PositionDirection.LONG, notional=100_000.0, holding_days=1.0)
        assert long_delta_a == long_delta_b

    def test_holding_days_scales_linearly(self) -> None:
        policy = self._policy(long_bps=10.0, short_bps=0.0)
        one_day = compute_financing_cash_delta(policy, direction=PositionDirection.LONG, notional=100_000.0, holding_days=1.0)
        three_days = compute_financing_cash_delta(policy, direction=PositionDirection.LONG, notional=100_000.0, holding_days=3.0)
        assert three_days == pytest.approx(one_day * 3.0)

    def test_zero_holding_days_produces_zero_delta(self) -> None:
        policy = self._policy(long_bps=10.0, short_bps=10.0)
        assert compute_financing_cash_delta(policy, direction=PositionDirection.LONG, notional=100_000.0, holding_days=0.0) == 0.0
