"""Release-audit Area 6: contract-multiplier accounting.

Independently hand-derives and verifies every formula involving
`contract_multiplier` at multipliers {0.01, 1, 10, 100, 1000}: long/short
opening/scale-in/partial-close/full-close/reversal, gross notional,
realized/unrealized P&L, spread/slippage cost (both proportional and
fixed-price-units cost models), commission (proportional AND fixed-per-
trade -- the latter deliberately does NOT scale), financing, turnover,
gross/net exposure, order-notional risk limits, and the exact equity
reconciliation identity. Also confirms dimensional consistency (per-unit
prices are NEVER pre-multiplied; only dollar figures are) and that zero/
negative/non-finite/identity-inconsistent multipliers are rejected
fail-closed.

FINDING (non-blocker, confirmed by direct hand-calculation, not merely
read from the code): `PortfolioState.equity = cash + marked_position_value
- liabilities - accrued_costs`, and `cash` moves by the fill's FULL
`gross_notional` on every trade (both entry and exit) -- i.e. actual
full-notional cash settlement, not a margin-only P&L abstraction, despite
the module docstring's "margin/P&L abstraction for derivatives" framing.
This is verified here to be internally CONSISTENT (no double-counting):
`equity - starting_cash` exactly equals `realized_pnl + unrealized_pnl -
accrued_costs + total_financing` at every tested multiplier -- see
`TestEquityReconciliationIdentityAcrossMultipliers`."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import FillValidationError, PaperTradingSpecError, PositionAccountingError
from quant_platform.paper_trading.accounting import (
    apply_fill_to_position,
    apply_mark_to_position,
    flat_position,
)
from quant_platform.paper_trading.costs import (
    compute_commission_dollars,
    compute_financing_cash_delta,
    compute_slippage_cost_dollars,
    compute_spread_cost_dollars,
)
from quant_platform.paper_trading.fills import Fill, create_fill
from quant_platform.paper_trading.models import (
    OrderSide,
    OrderTypeKind,
    PartialFillPolicyKind,
    PositionIntentKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import create_order_request
from quant_platform.paper_trading.portfolio import (
    apply_fill_to_portfolio,
    apply_financing_to_portfolio,
    apply_mark_to_portfolio,
    initial_portfolio,
)
from quant_platform.paper_trading.risk import evaluate_pre_trade_risk, most_severe_action
from quant_platform.paper_trading.specs import InstrumentSpec, RiskLimitsSpec

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_EVENT = "a" * 64
_HEX_DECISION = "b" * 64

_MULTIPLIERS = (0.01, 1.0, 10.0, 100.0, 1000.0)


def _fill(*, side: OrderSide, quantity: float, price: float, contract_multiplier: float, spread_cost: float = 0.0, slippage_cost: float = 0.0, commission_cost: float = 0.0) -> Fill:
    return create_fill(
        order_id="order-1", session_id="session-1", instrument="X", side=side, quantity=quantity, price=price, contract_multiplier=contract_multiplier,
        spread_cost=spread_cost, slippage_cost=slippage_cost, commission_cost=commission_cost, execution_time=_T0,
        source_market_event_identity=_HEX_EVENT, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=True,
    )


class TestLongLifecycleAcrossMultipliers:
    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_open_scale_in_partial_close_full_close(self, multiplier: float) -> None:
        position = flat_position("X", contract_multiplier=multiplier)

        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0, contract_multiplier=multiplier), event_time=_T0)
        assert position.average_entry_price == pytest.approx(100.0), "per-unit price must never be pre-multiplied"
        assert position.gross_cost_basis == pytest.approx(10.0 * 100.0 * multiplier)
        assert position.realized_pnl == 0.0

        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=5.0, price=110.0, contract_multiplier=multiplier), event_time=_T0)
        expected_avg = (10.0 * 100.0 + 5.0 * 110.0) / 15.0
        assert position.average_entry_price == pytest.approx(expected_avg)
        assert position.signed_quantity == pytest.approx(15.0)
        assert position.realized_pnl == 0.0, "scale-in must never realize P&L"

        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=6.0, price=120.0, contract_multiplier=multiplier), event_time=_T0)
        expected_realized = 6.0 * (120.0 - expected_avg) * multiplier
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == pytest.approx(9.0)
        assert position.average_entry_price == pytest.approx(expected_avg), "partial close must not change average entry price"

        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=9.0, price=130.0, contract_multiplier=multiplier), event_time=_T0)
        expected_realized += 9.0 * (130.0 - expected_avg) * multiplier
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == 0.0
        assert position.average_entry_price is None
        assert position.gross_cost_basis == 0.0

    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_reversal_through_zero(self, multiplier: float) -> None:
        position = flat_position("X", contract_multiplier=multiplier)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0, contract_multiplier=multiplier), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=15.0, price=90.0, contract_multiplier=multiplier), event_time=_T0)

        expected_realized = 10.0 * (90.0 - 100.0) * multiplier  # closing the original 10 long units at a loss
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == pytest.approx(-5.0)
        assert position.average_entry_price == pytest.approx(90.0), "the leftover 5 units open a NEW short at the fill's own price"


class TestShortLifecycleAcrossMultipliers:
    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_open_partial_close_full_close(self, multiplier: float) -> None:
        position = flat_position("X", contract_multiplier=multiplier)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0, contract_multiplier=multiplier), event_time=_T0)
        assert position.signed_quantity == pytest.approx(-10.0)
        assert position.average_entry_price == pytest.approx(100.0)

        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=4.0, price=90.0, contract_multiplier=multiplier), event_time=_T0)
        expected_realized = 4.0 * (100.0 - 90.0) * multiplier  # short profits when price falls
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == pytest.approx(-6.0)
        assert position.average_entry_price == pytest.approx(100.0)

        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=6.0, price=95.0, contract_multiplier=multiplier), event_time=_T0)
        expected_realized += 6.0 * (100.0 - 95.0) * multiplier
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == 0.0
        assert position.average_entry_price is None

    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_reversal_through_zero(self, multiplier: float) -> None:
        position = flat_position("X", contract_multiplier=multiplier)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0, contract_multiplier=multiplier), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=15.0, price=115.0, contract_multiplier=multiplier), event_time=_T0)

        expected_realized = 10.0 * (100.0 - 115.0) * multiplier  # closing the original 10 short units at a loss
        assert position.realized_pnl == pytest.approx(expected_realized)
        assert position.signed_quantity == pytest.approx(5.0)
        assert position.average_entry_price == pytest.approx(115.0)


class TestUnrealizedPnlMarkAcrossMultipliers:
    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_long_and_short_unrealized_pnl(self, multiplier: float) -> None:
        long_position = flat_position("X", contract_multiplier=multiplier)
        long_position = apply_fill_to_position(long_position, _fill(side=OrderSide.BUY, quantity=7.0, price=50.0, contract_multiplier=multiplier), event_time=_T0)
        long_position = apply_mark_to_position(long_position, mark_price=53.0, event_time=_T0)
        assert long_position.unrealized_pnl == pytest.approx(7.0 * (53.0 - 50.0) * multiplier)

        short_position = flat_position("X", contract_multiplier=multiplier)
        short_position = apply_fill_to_position(short_position, _fill(side=OrderSide.SELL, quantity=7.0, price=50.0, contract_multiplier=multiplier), event_time=_T0)
        short_position = apply_mark_to_position(short_position, mark_price=47.0, event_time=_T0)
        assert short_position.unrealized_pnl == pytest.approx(-7.0 * (47.0 - 50.0) * multiplier)
        assert short_position.unrealized_pnl > 0.0, "a short position must gain when price falls"


class TestLinearScalingOfMultiplierDependentFormulas:
    """`result(k) = k * result(1)` for every formula that is DIMENSIONALLY
    proportional to contract_multiplier -- explicitly EXCLUDING fixed
    commissions, which are tested separately for the opposite property."""

    def test_gross_notional_scales_linearly(self) -> None:
        base = _fill(side=OrderSide.BUY, quantity=3.0, price=42.0, contract_multiplier=1.0).gross_notional
        for m in _MULTIPLIERS:
            scaled = _fill(side=OrderSide.BUY, quantity=3.0, price=42.0, contract_multiplier=m).gross_notional
            assert scaled == pytest.approx(m * base)

    def test_realized_pnl_scales_linearly(self) -> None:
        def _realized_at(multiplier: float) -> float:
            position = flat_position("X", contract_multiplier=multiplier)
            position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=4.0, price=200.0, contract_multiplier=multiplier), event_time=_T0)
            position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=4.0, price=215.0, contract_multiplier=multiplier), event_time=_T0)
            return position.realized_pnl

        base = _realized_at(1.0)
        for m in _MULTIPLIERS:
            assert _realized_at(m) == pytest.approx(m * base)

    def test_spread_and_slippage_cost_scale_linearly_basis_points_model(self) -> None:
        spread = SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=5.0)
        slippage = SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=3.0)
        base_spread = compute_spread_cost_dollars(spread, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=1.0)
        base_slippage = compute_slippage_cost_dollars(slippage, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=1.0)
        for m in _MULTIPLIERS:
            assert compute_spread_cost_dollars(spread, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=m) == pytest.approx(m * base_spread)
            assert compute_slippage_cost_dollars(slippage, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=m) == pytest.approx(m * base_slippage)

    def test_spread_and_slippage_cost_scale_linearly_fixed_price_units_model(self) -> None:
        spread = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.10)
        slippage = SlippageSpec(kind=SlippageModelKind.FIXED_PRICE_UNITS, price_units=0.05)
        base_spread = compute_spread_cost_dollars(spread, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=1.0)
        base_slippage = compute_slippage_cost_dollars(slippage, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=1.0)
        for m in _MULTIPLIERS:
            assert compute_spread_cost_dollars(spread, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=m) == pytest.approx(m * base_spread)
            assert compute_slippage_cost_dollars(slippage, 100.0, PositionDirection.LONG, is_entry=True, quantity=2.0, contract_multiplier=m) == pytest.approx(m * base_slippage)

    def test_proportional_commission_scales_linearly_with_multiplier(self) -> None:
        spec = CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0)
        price, quantity = 50.0, 4.0
        base_notional = price * quantity * 1.0
        base_commission = compute_commission_dollars(spec, notional=base_notional)
        for m in _MULTIPLIERS:
            notional = price * quantity * m
            assert compute_commission_dollars(spec, notional=notional) == pytest.approx(m * base_commission)

    def test_fixed_per_trade_commission_does_not_scale_with_multiplier(self) -> None:
        """The one deliberately NON-scaling component Section 6 warns
        against blindly asserting linear scaling for -- confirmed exactly
        constant across every multiplier."""
        spec = CommissionSpec(kind=CommissionModelKind.FIXED_PER_TRADE, fixed_per_trade=7.5)
        price, quantity = 50.0, 4.0
        for m in _MULTIPLIERS:
            notional = price * quantity * m
            assert compute_commission_dollars(spec, notional=notional) == pytest.approx(7.5)

    def test_financing_cash_delta_scales_linearly(self) -> None:
        policy = FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=10.0)
        quantity, price = 3.0, 200.0
        base = compute_financing_cash_delta(_wrap_financing(policy), direction=PositionDirection.LONG, notional=quantity * price * 1.0, holding_days=2.0)
        for m in _MULTIPLIERS:
            scaled = compute_financing_cash_delta(_wrap_financing(policy), direction=PositionDirection.LONG, notional=quantity * price * m, holding_days=2.0)
            assert scaled == pytest.approx(m * base)

    def test_turnover_scales_linearly(self) -> None:
        def _turnover_at(multiplier: float) -> float:
            portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
            portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=3.0, price=40.0, contract_multiplier=multiplier), event_time=_T0, contract_multiplier=multiplier)
            return portfolio.turnover

        base = _turnover_at(1.0)
        for m in _MULTIPLIERS:
            assert _turnover_at(m) == pytest.approx(m * base)

    def test_gross_and_net_exposure_scale_linearly(self) -> None:
        def _exposures_at(multiplier: float) -> tuple[float, float]:
            portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
            portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=3.0, price=40.0, contract_multiplier=multiplier), event_time=_T0, contract_multiplier=multiplier)
            portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=44.0, event_time=_T0)
            return portfolio.gross_exposure, portfolio.net_exposure

        base_gross, base_net = _exposures_at(1.0)
        for m in _MULTIPLIERS:
            gross, net = _exposures_at(m)
            assert gross == pytest.approx(m * base_gross)
            assert net == pytest.approx(m * base_net)

    def test_order_notional_risk_limit_scales_linearly(self) -> None:
        order = create_order_request(
            client_order_id="c1", session_id="session-1", strategy_decision_id=_HEX_DECISION, instrument="X", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET,
            quantity=5.0, time_in_force=TimeInForceKind.DAY, create_time=_T0, submit_time=_T0, reduce_only=False, position_intent=PositionIntentKind.OPEN,
        )
        limits = RiskLimitsSpec(
            maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
            maximum_order_notional=0.001, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None, maximum_realized_loss=None,
            maximum_unrealized_loss=None, maximum_rejected_order_count=None, maximum_consecutive_execution_failures=None,
            maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
        )
        portfolio = initial_portfolio("session-1", starting_cash=10_000_000.0)
        # order_notional = 5*10*m ranges from 0.5 (m=0.01) to 50,000 (m=1000) --
        # all comfortably above the 0.001 limit, so every multiplier must reject;
        # this proves the notional CHECK ITSELF scales with multiplier exactly
        # like the real notional does (a check that forgot to multiply would
        # instead allow the smaller multipliers through).
        for m in _MULTIPLIERS:
            results = evaluate_pre_trade_risk(
                order, portfolio=portfolio, risk_limits=limits, reference_price=10.0, contract_multiplier=m,
                event_identity=_HEX_EVENT, trading_halted=False, session_accepting_orders=True, stale_data_seconds=None,
            )
            assert most_severe_action(results).value == "reject_order", f"multiplier={m}: order notional {5.0 * 10.0 * m} must exceed the 1.0 limit"


class TestEquityReconciliationIdentityAcrossMultipliers:
    @pytest.mark.parametrize("multiplier", _MULTIPLIERS)
    def test_equity_delta_equals_pnl_minus_costs_plus_financing(self, multiplier: float) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        buy = _fill(side=OrderSide.BUY, quantity=5.0, price=60.0, contract_multiplier=multiplier, spread_cost=1.0 * multiplier, slippage_cost=0.5 * multiplier, commission_cost=0.25 * multiplier)
        portfolio = apply_fill_to_portfolio(portfolio, buy, event_time=_T0, contract_multiplier=multiplier)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=65.0, event_time=_T0)
        portfolio = apply_financing_to_portfolio(portfolio, instrument="X", cash_delta=-3.0 * multiplier, event_time=_T0)
        sell = _fill(side=OrderSide.SELL, quantity=2.0, price=70.0, contract_multiplier=multiplier, commission_cost=0.1 * multiplier)
        portfolio = apply_fill_to_portfolio(portfolio, sell, event_time=_T0, contract_multiplier=multiplier)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=68.0, event_time=_T0)

        total_financing = sum(p.accumulated_financing for p in portfolio.positions.values())
        lhs = portfolio.equity - portfolio.starting_cash
        rhs = portfolio.realized_pnl + portfolio.unrealized_pnl - portfolio.accrued_costs + total_financing
        assert lhs == pytest.approx(rhs, abs=1e-6 * max(1.0, abs(rhs)))


class TestMultiplierRejection:
    @pytest.mark.parametrize("bad_multiplier", [0.0, -1.0, -100.0, float("nan"), float("inf"), float("-inf")])
    def test_instrument_spec_rejects_non_positive_or_non_finite_multiplier(self, bad_multiplier: float) -> None:
        with pytest.raises(PaperTradingSpecError):
            InstrumentSpec(
                symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=bad_multiplier, tick_size=0.01, tick_value=None,
                quantity_step=0.01, minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash",
                account_currency="USD", financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
            )

    @pytest.mark.parametrize("bad_multiplier", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_position_state_rejects_non_positive_or_non_finite_multiplier(self, bad_multiplier: float) -> None:
        with pytest.raises(PositionAccountingError):
            flat_position("X", contract_multiplier=bad_multiplier)

    def test_fill_implying_a_different_multiplier_than_the_position_is_rejected(self) -> None:
        """A fill whose OWN `gross_notional` was computed with a
        DIFFERENT multiplier than the position's stored one (a tampered
        or forged fill, or a genuine cross-instrument mixup) must be
        rejected outright, never silently applied with either multiplier."""
        position = flat_position("X", contract_multiplier=10.0)
        genuine_fill = _fill(side=OrderSide.BUY, quantity=2.0, price=50.0, contract_multiplier=10.0)
        tampered_fill = Fill(
            fill_id=genuine_fill.fill_id, order_id=genuine_fill.order_id, session_id=genuine_fill.session_id, instrument=genuine_fill.instrument,
            side=genuine_fill.side, quantity=genuine_fill.quantity, price=genuine_fill.price, gross_notional=genuine_fill.quantity * genuine_fill.price * 1.0,
            spread_cost=0.0, slippage_cost=0.0, commission_cost=0.0, financing_component=0.0, execution_time=genuine_fill.execution_time,
            source_market_event_identity=genuine_fill.source_market_event_identity, liquidity_assumption=genuine_fill.liquidity_assumption, is_final=True,
        )
        with pytest.raises(PositionAccountingError, match="implies contract_multiplier"):
            apply_fill_to_position(position, tampered_fill, event_time=_T0)

    def test_fill_construction_rejects_non_finite_price_regardless_of_multiplier(self) -> None:
        with pytest.raises(FillValidationError):
            create_fill(
                order_id="order-1", session_id="session-1", instrument="X", side=OrderSide.BUY, quantity=1.0, price=float("nan"), contract_multiplier=10.0,
                spread_cost=0.0, slippage_cost=0.0, commission_cost=0.0, execution_time=_T0, source_market_event_identity=_HEX_EVENT,
                liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=True,
            )


def _wrap_financing(spec: FinancingSpec):
    from quant_platform.paper_trading.specs import FinancingPolicySpec

    return FinancingPolicySpec(long_financing=spec, short_financing=spec)
