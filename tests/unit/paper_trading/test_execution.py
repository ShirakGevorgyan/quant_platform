"""Milestone 7, Section 10: paper execution engine. Covers market/limit/
stop order fill semantics in both QUOTE and BAR modes, spread/slippage/
commission cost application, gap behavior, partial fills (deterministic
and fail-closed), and IOC/FOK time-in-force handling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.backtesting.models import CommissionModelKind, SlippageModelKind, SpreadModelKind
from quant_platform.backtesting.specs import CommissionSpec, SlippageSpec, SpreadSpec
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.events import create_bar_event, create_quote_event
from quant_platform.paper_trading.execution import evaluate_order_against_event, process_order_against_event
from quant_platform.paper_trading.models import (
    OrderSide,
    OrderState,
    OrderTypeKind,
    PartialFillPolicyKind,
    PositionIntentKind,
    RejectReasonKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import create_order_request
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    FillPolicySpec,
    InstrumentSpec,
    LiquidityPolicySpec,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_DECISION = "a" * 64

_ZERO_SPREAD = SpreadSpec(kind=SpreadModelKind.ZERO)
_ZERO_SLIPPAGE = SlippageSpec(kind=SlippageModelKind.ZERO)
_ZERO_COMMISSION = CommissionSpec(kind=CommissionModelKind.ZERO)
_FIXED_SPREAD = SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.4)
_FIXED_SLIPPAGE = SlippageSpec(kind=SlippageModelKind.FIXED_PRICE_UNITS, price_units=0.1)
_FIXED_COMMISSION = CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=10.0)
_FULL_FILL_ONLY = FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY)
_DETERMINISTIC_PARTIAL = FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.DETERMINISTIC_PARTIAL)
_NO_TRUST_LIQUIDITY = LiquidityPolicySpec(trust_disclosed_size=False)
_TRUST_LIQUIDITY = LiquidityPolicySpec(trust_disclosed_size=True)


def _instrument(contract_multiplier: float = 1.0) -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=contract_multiplier, tick_size=0.01, tick_value=None,
        quantity_step=0.01, minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash",
        account_currency="USD", financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _order(
    *, side: OrderSide, order_type: OrderTypeKind, quantity: float = 1.0, limit_price: float | None = None, stop_price: float | None = None,
    time_in_force: TimeInForceKind = TimeInForceKind.DAY, position_intent: PositionIntentKind = PositionIntentKind.OPEN,
):
    return create_order_request(
        client_order_id="c1", session_id="session-1", strategy_decision_id=_HEX_DECISION, instrument="X", side=side, order_type=order_type,
        quantity=quantity, limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force, create_time=_T0, submit_time=_T0,
        reduce_only=False, position_intent=position_intent,
    )


def _quote(*, bid: float, ask: float, bid_size: float | None = None, ask_size: float | None = None, sequence: int = 1):
    return create_quote_event(instrument="X", event_time=_T0, sequence=sequence, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size, source="s")


def _bar(*, open: float, high: float, low: float, close: float, sequence: int = 1):
    return create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=open, high=high, low=low, close=close, sequence=sequence, source="s")


def _evaluate(order, event, *, remaining_quantity: float = 1.0, spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE, liquidity_policy=_NO_TRUST_LIQUIDITY, fill_policy=_FULL_FILL_ONLY):
    return evaluate_order_against_event(
        order, event, remaining_quantity=remaining_quantity, execution_policy=DEFAULT_EXECUTION_POLICY, spread_policy=spread_policy,
        slippage_policy=slippage_policy, liquidity_policy=liquidity_policy, fill_policy=fill_policy,
    )


class TestMarketOrderQuoteMode:
    def test_buy_fills_at_ask(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1))
        assert candidate is not None
        assert candidate.price == pytest.approx(100.1)
        assert candidate.quantity == pytest.approx(1.0)

    def test_sell_fills_at_bid(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.MARKET)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1))
        assert candidate is not None
        assert candidate.price == pytest.approx(99.9)

    def test_slippage_applied_on_top(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1), slippage_policy=_FIXED_SLIPPAGE)
        assert candidate is not None
        assert candidate.price == pytest.approx(100.2)


class TestMarketOrderBarMode:
    def test_buy_fills_at_close_plus_half_spread(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET)
        candidate = _evaluate(order, _bar(open=99.0, high=102.0, low=98.0, close=100.0), spread_policy=_FIXED_SPREAD)
        assert candidate is not None
        assert candidate.price == pytest.approx(100.2)

    def test_sell_fills_at_close_minus_half_spread(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.MARKET)
        candidate = _evaluate(order, _bar(open=99.0, high=102.0, low=98.0, close=100.0), spread_policy=_FIXED_SPREAD)
        assert candidate is not None
        assert candidate.price == pytest.approx(99.8)


class TestLimitOrderQuoteMode:
    def test_buy_limit_fills_with_price_improvement(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=101.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1))
        assert candidate is not None
        assert candidate.price == pytest.approx(100.1)

    def test_buy_limit_does_not_fill_when_ask_above_limit(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=99.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1))
        assert candidate is None

    def test_sell_limit_fills_when_bid_at_or_above_limit(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.LIMIT, limit_price=99.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1))
        assert candidate is not None
        assert candidate.price == pytest.approx(99.9)


class TestLimitOrderBarMode:
    def test_buy_limit_fills_at_limit_price_when_bar_touches_it(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=99.0)
        candidate = _evaluate(order, _bar(open=100.0, high=101.0, low=98.5, close=100.5))
        assert candidate is not None
        assert candidate.price == pytest.approx(99.0)

    def test_buy_limit_does_not_fill_when_bar_low_above_limit(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=95.0)
        candidate = _evaluate(order, _bar(open=100.0, high=101.0, low=98.5, close=100.5))
        assert candidate is None

    def test_buy_limit_gap_improvement_when_open_below_limit(self) -> None:
        """The bar opened BELOW the limit price -- a real, disclosed
        gap-down -- so the fill is at the (better) open price, never at
        the limit price the strategy would have settled for."""
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=99.0)
        candidate = _evaluate(order, _bar(open=97.0, high=99.5, low=96.5, close=98.0))
        assert candidate is not None
        assert candidate.price == pytest.approx(97.0)

    def test_sell_limit_fills_at_limit_price_when_bar_touches_it(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.LIMIT, limit_price=101.0)
        candidate = _evaluate(order, _bar(open=100.0, high=101.5, low=99.5, close=100.5))
        assert candidate is not None
        assert candidate.price == pytest.approx(101.0)


class TestStopOrderQuoteMode:
    def test_buy_stop_triggers_and_fills_at_ask(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=100.0)
        candidate = _evaluate(order, _quote(bid=100.4, ask=100.6))
        assert candidate is not None
        assert candidate.price == pytest.approx(100.6)

    def test_buy_stop_does_not_trigger_below_stop(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=105.0)
        candidate = _evaluate(order, _quote(bid=100.4, ask=100.6))
        assert candidate is None

    def test_buy_stop_gap_fill_worse_than_stop_price(self) -> None:
        """A fast-moving quote stream can already be past the stop price
        the instant it triggers -- QUOTE-mode gap behavior is automatic:
        the fill is at the real (worse) ask, never the stop price."""
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=100.0)
        candidate = _evaluate(order, _quote(bid=104.8, ask=105.0))
        assert candidate is not None
        assert candidate.price == pytest.approx(105.0)


class TestStopOrderBarMode:
    def test_buy_stop_fills_at_stop_price_when_bar_touches_it(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=101.0)
        candidate = _evaluate(order, _bar(open=100.0, high=101.5, low=99.5, close=100.8))
        assert candidate is not None
        assert candidate.price == pytest.approx(101.0)

    def test_buy_stop_gap_fill_at_open_when_bar_gaps_through(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=101.0)
        candidate = _evaluate(order, _bar(open=103.0, high=104.0, low=102.5, close=103.5))
        assert candidate is not None
        assert candidate.price == pytest.approx(103.0)

    def test_sell_stop_fills_at_stop_price_when_bar_touches_it(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.STOP, stop_price=99.0)
        candidate = _evaluate(order, _bar(open=100.0, high=100.5, low=98.5, close=99.5))
        assert candidate is not None
        assert candidate.price == pytest.approx(99.0)

    def test_sell_stop_does_not_trigger_above_stop(self) -> None:
        order = _order(side=OrderSide.SELL, order_type=OrderTypeKind.STOP, stop_price=95.0)
        candidate = _evaluate(order, _bar(open=100.0, high=100.5, low=98.5, close=99.5))
        assert candidate is None


class TestPartialFills:
    def test_full_fill_only_ignores_disclosed_size(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1, ask_size=3.0), remaining_quantity=10.0, liquidity_policy=_TRUST_LIQUIDITY, fill_policy=_FULL_FILL_ONLY)
        assert candidate is not None
        assert candidate.quantity == pytest.approx(10.0)

    def test_deterministic_partial_honors_disclosed_size(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1, ask_size=3.0), remaining_quantity=10.0, liquidity_policy=_TRUST_LIQUIDITY, fill_policy=_DETERMINISTIC_PARTIAL)
        assert candidate is not None
        assert candidate.quantity == pytest.approx(3.0)
        assert candidate.liquidity_assumption is PartialFillPolicyKind.DETERMINISTIC_PARTIAL

    def test_deterministic_partial_without_trust_falls_back_to_full(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1, ask_size=3.0), remaining_quantity=10.0, liquidity_policy=_NO_TRUST_LIQUIDITY, fill_policy=_DETERMINISTIC_PARTIAL)
        assert candidate is not None
        assert candidate.quantity == pytest.approx(10.0)

    def test_deterministic_partial_falls_back_to_full_when_no_size_disclosed(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        candidate = _evaluate(order, _quote(bid=99.9, ask=100.1), remaining_quantity=10.0, liquidity_policy=_TRUST_LIQUIDITY, fill_policy=_DETERMINISTIC_PARTIAL)
        assert candidate is not None
        assert candidate.quantity == pytest.approx(10.0)
        assert candidate.liquidity_assumption is PartialFillPolicyKind.FULL_FILL_ONLY

    def test_bar_mode_never_partial_fills_even_with_deterministic_partial_configured(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        candidate = _evaluate(order, _bar(open=99.0, high=101.0, low=98.0, close=100.0), remaining_quantity=10.0, liquidity_policy=_TRUST_LIQUIDITY, fill_policy=_DETERMINISTIC_PARTIAL)
        assert candidate is not None
        assert candidate.quantity == pytest.approx(10.0)


class TestProcessOrderAgainstEvent:
    def _process(self, order, event, *, current_state=OrderState.WORKING, filled_quantity_so_far: float = 0.0, sequence: int = 1, contract_multiplier: float = 1.0, commission_policy=_ZERO_COMMISSION):
        return process_order_against_event(
            order, current_state=current_state, filled_quantity_so_far=filled_quantity_so_far, event=event, event_time=_T0, sequence=sequence,
            instrument=_instrument(contract_multiplier), execution_policy=DEFAULT_EXECUTION_POLICY, spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE,
            commission_policy=commission_policy, fill_policy=_FULL_FILL_ONLY, liquidity_policy=_NO_TRUST_LIQUIDITY,
        )

    def test_non_working_state_is_a_no_op(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET)
        outcome = self._process(order, _quote(bid=99.9, ask=100.1), current_state=OrderState.CREATED)
        assert outcome.fills == ()
        assert outcome.order_state_events == ()

    def test_market_order_produces_one_fill_and_filled_transition(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=5.0)
        outcome = self._process(order, _quote(bid=99.9, ask=100.1))
        assert len(outcome.fills) == 1
        assert outcome.fills[0].is_final is True
        assert len(outcome.order_state_events) == 1
        assert outcome.order_state_events[0].to_state is OrderState.FILLED

    def test_commission_computed_from_fill_notional(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        outcome = self._process(order, _quote(bid=99.9, ask=100.0), commission_policy=_FIXED_COMMISSION)
        assert len(outcome.fills) == 1
        expected_notional = 100.0 * 10.0
        assert outcome.fills[0].commission_cost == pytest.approx(expected_notional * 10.0 / 10_000.0)

    def test_limit_order_no_trigger_produces_no_change_for_day_tif(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=90.0, time_in_force=TimeInForceKind.DAY)
        outcome = self._process(order, _quote(bid=99.9, ask=100.1))
        assert outcome.fills == ()
        assert outcome.order_state_events == ()

    def test_ioc_order_no_trigger_is_cancelled(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=90.0, time_in_force=TimeInForceKind.IOC)
        outcome = self._process(order, _quote(bid=99.9, ask=100.1))
        assert outcome.fills == ()
        assert len(outcome.order_state_events) == 1
        assert outcome.order_state_events[0].to_state is OrderState.CANCELLED
        assert outcome.order_state_events[0].reason_code is RejectReasonKind.IOC_NOT_IMMEDIATELY_FILLABLE

    def test_fok_order_that_cannot_fully_fill_is_cancelled_with_no_fill(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0, time_in_force=TimeInForceKind.FOK)
        outcome = process_order_against_event(
            order, current_state=OrderState.WORKING, filled_quantity_so_far=0.0, event=_quote(bid=99.9, ask=100.1, ask_size=3.0), event_time=_T0,
            sequence=1, instrument=_instrument(), execution_policy=DEFAULT_EXECUTION_POLICY, spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE,
            commission_policy=_ZERO_COMMISSION, fill_policy=_DETERMINISTIC_PARTIAL, liquidity_policy=_TRUST_LIQUIDITY,
        )
        assert outcome.fills == ()
        assert len(outcome.order_state_events) == 1
        assert outcome.order_state_events[0].to_state is OrderState.CANCELLED
        assert outcome.order_state_events[0].reason_code is RejectReasonKind.FOK_NOT_FULLY_FILLABLE

    def test_fok_order_that_can_fully_fill_fills_completely(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=3.0, time_in_force=TimeInForceKind.FOK)
        outcome = process_order_against_event(
            order, current_state=OrderState.WORKING, filled_quantity_so_far=0.0, event=_quote(bid=99.9, ask=100.1, ask_size=10.0), event_time=_T0,
            sequence=1, instrument=_instrument(), execution_policy=DEFAULT_EXECUTION_POLICY, spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE,
            commission_policy=_ZERO_COMMISSION, fill_policy=_DETERMINISTIC_PARTIAL, liquidity_policy=_TRUST_LIQUIDITY,
        )
        assert len(outcome.fills) == 1
        assert outcome.fills[0].quantity == pytest.approx(3.0)
        assert outcome.fills[0].is_final is True

    def test_ioc_partial_fill_produces_fill_and_cancels_remainder(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0, time_in_force=TimeInForceKind.IOC)
        outcome = process_order_against_event(
            order, current_state=OrderState.WORKING, filled_quantity_so_far=0.0, event=_quote(bid=99.9, ask=100.1, ask_size=4.0), event_time=_T0,
            sequence=1, instrument=_instrument(), execution_policy=DEFAULT_EXECUTION_POLICY, spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE,
            commission_policy=_ZERO_COMMISSION, fill_policy=_DETERMINISTIC_PARTIAL, liquidity_policy=_TRUST_LIQUIDITY,
        )
        assert len(outcome.fills) == 1
        assert outcome.fills[0].quantity == pytest.approx(4.0)
        assert outcome.fills[0].is_final is False
        assert len(outcome.order_state_events) == 2
        assert outcome.order_state_events[0].to_state is OrderState.PARTIALLY_FILLED
        assert outcome.order_state_events[1].to_state is OrderState.CANCELLED
        assert outcome.order_state_events[1].reason_code is RejectReasonKind.IOC_NOT_IMMEDIATELY_FILLABLE

    def test_second_partial_fill_transitions_from_partially_filled_to_filled(self) -> None:
        order = _order(side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0)
        outcome = self._process(order, _quote(bid=99.9, ask=100.1), current_state=OrderState.PARTIALLY_FILLED, filled_quantity_so_far=6.0)
        assert len(outcome.fills) == 1
        assert outcome.fills[0].quantity == pytest.approx(4.0)
        assert outcome.fills[0].is_final is True
        assert outcome.order_state_events[0].from_state is OrderState.PARTIALLY_FILLED
        assert outcome.order_state_events[0].to_state is OrderState.FILLED
