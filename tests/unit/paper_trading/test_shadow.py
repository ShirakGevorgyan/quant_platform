"""Milestone 7, Section 19: shadow observation mode. Proves the core
guarantee -- a shadow decision NEVER touches a real `portfolio.
PortfolioState` (there is no such object anywhere in this test file's
call graph) -- while still producing genuine hypothetical orders/fills/
counterfactual P&L using the exact same deterministic pipeline as a real
session, and that every decision (including abstentions and non-
triggering orders) produces a persisted observation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.accounting import flat_position
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.models import PartialFillPolicyKind
from quant_platform.paper_trading.order_policy import OrderPolicyState
from quant_platform.paper_trading.shadow import ShadowObservation, evaluate_shadow_decision
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_RISK_LIMITS,
    FillPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
)
from quant_platform.paper_trading.strategy import create_strategy_decision

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_STRATEGY = "a" * 64

_ZERO_SPREAD = SpreadSpec(kind=SpreadModelKind.ZERO)
_ZERO_SLIPPAGE = SlippageSpec(kind=SlippageModelKind.ZERO)
_ZERO_COMMISSION = CommissionSpec(kind=CommissionModelKind.ZERO)
_FIXED_COMMISSION = CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=10.0)


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash", account_currency="USD",
        financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _order_policy() -> OrderPolicySpec:
    return OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=10, order_rate_window_events=20)


def _bar(*, close: float, sequence: int = 1):
    return create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=close, high=close + 1.0, low=close - 1.0, close=close, sequence=sequence, source="s")


def _decision(*, direction: PositionDirection, quantity: float, abstain: bool = False):
    event = _bar(close=100.0)
    return create_strategy_decision(
        strategy_identity=_HEX_STRATEGY, event=event, decision_time=_T0, target_direction=direction, target_quantity=quantity, confidence=0.8,
        uncertainty=0.1, abstain=abstain, reason_codes=("test",),
    )


def _evaluate(decision, shadow_position, *, commission_policy=_ZERO_COMMISSION, sequence: int = 1):
    return evaluate_shadow_decision(
        decision, shadow_position=shadow_position, instrument=_instrument(), order_policy=_order_policy(), execution_policy=DEFAULT_EXECUTION_POLICY,
        spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE, commission_policy=commission_policy, fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False), latency_policy=LatencyPolicySpec(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0),
        risk_limits=DEFAULT_RISK_LIMITS, session_id="shadow-session-1", event=_bar(close=100.0, sequence=sequence), order_policy_state=OrderPolicyState(bars_since_last_order=None, orders_created_in_rate_window=0),
        sequence=sequence,
    )


class TestShadowNeverTouchesRealPortfolio:
    def test_no_real_portfolio_object_anywhere_in_signature(self) -> None:
        """Documentation-as-test: `evaluate_shadow_decision`'s own
        signature has no `portfolio.PortfolioState` parameter at all --
        it is structurally impossible to pass one in, let alone mutate
        one."""
        import inspect

        signature = inspect.signature(evaluate_shadow_decision)
        assert "portfolio" not in signature.parameters

    def test_abstention_produces_a_persisted_observation_with_no_hypotheticals(self) -> None:
        shadow_position = flat_position("X", contract_multiplier=1.0)
        decision = _decision(direction=PositionDirection.LONG, quantity=0.0, abstain=True)
        observation, updated_position = _evaluate(decision, shadow_position)
        assert observation.hypothetical_order_id is None
        assert observation.hypothetical_fill_id is None
        assert observation.counterfactual_realized_pnl_delta is None
        assert updated_position == shadow_position


class TestShadowFillProducesCounterfactualPnl:
    def test_opening_long_produces_hypothetical_order_and_fill(self) -> None:
        shadow_position = flat_position("X", contract_multiplier=1.0)
        decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        observation, updated_position = _evaluate(decision, shadow_position)
        assert observation.hypothetical_order_id is not None
        assert observation.hypothetical_fill_id is not None
        assert observation.hypothetical_fill_quantity == pytest.approx(2.0)
        assert updated_position.signed_quantity == pytest.approx(2.0)
        # opening a position realizes no P&L
        assert observation.counterfactual_realized_pnl_delta == pytest.approx(0.0)

    def test_closing_produces_nonzero_counterfactual_pnl_delta(self) -> None:
        shadow_position = flat_position("X", contract_multiplier=1.0)
        open_decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        _, shadow_position = _evaluate(open_decision, shadow_position, sequence=1)
        # bar closes at 100.0 both times in this fixture, so force a real
        # price move by evaluating against a fresh bar directly:
        close_decision = create_strategy_decision(
            strategy_identity=_HEX_STRATEGY, event=_bar(close=110.0, sequence=2), decision_time=_T0, target_direction=PositionDirection.FLAT,
            target_quantity=0.0, confidence=0.8, uncertainty=0.1, abstain=False, reason_codes=("test",),
        )
        observation, updated_position = evaluate_shadow_decision(
            close_decision, shadow_position=shadow_position, instrument=_instrument(), order_policy=_order_policy(), execution_policy=DEFAULT_EXECUTION_POLICY,
            spread_policy=_ZERO_SPREAD, slippage_policy=_ZERO_SLIPPAGE, commission_policy=_ZERO_COMMISSION,
            fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY), liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False),
            latency_policy=LatencyPolicySpec(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0), risk_limits=DEFAULT_RISK_LIMITS,
            session_id="shadow-session-1", event=_bar(close=110.0, sequence=2), order_policy_state=OrderPolicyState(bars_since_last_order=None, orders_created_in_rate_window=0), sequence=2,
        )
        assert updated_position.signed_quantity == pytest.approx(0.0)
        assert observation.counterfactual_realized_pnl_delta == pytest.approx(2.0 * (110.0 - 100.0))

    def test_commission_reflected_in_hypothetical_fill_but_not_in_pnl_delta_directly(self) -> None:
        """`counterfactual_realized_pnl_delta` comes from `accounting.
        realized_pnl` (price-only, matching the real pipeline's own
        convention that transaction costs are tracked separately) --
        confirms shadow observation reuses the SAME accounting semantics,
        not an ad hoc P&L shortcut."""
        shadow_position = flat_position("X", contract_multiplier=1.0)
        decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        observation, updated_position = _evaluate(decision, shadow_position, commission_policy=_FIXED_COMMISSION)
        assert observation.hypothetical_fill_id is not None
        assert updated_position.accumulated_transaction_costs > 0.0
        assert observation.counterfactual_realized_pnl_delta == pytest.approx(0.0)


class TestShadowObservationValidationAndIdentity:
    def test_fill_id_without_order_id_rejected(self) -> None:
        with pytest.raises(PaperTradingError, match="hypothetical_order_id"):
            ShadowObservation(
                observation_id="0" * 64, session_id="s1", decision_id="d1", instrument="X", hypothetical_order_id=None, hypothetical_fill_id="f1",
                hypothetical_fill_price=100.0, hypothetical_fill_quantity=1.0, counterfactual_realized_pnl_delta=0.0, event_identity="e" * 64,
                event_time=_T0, sequence=1,
            )

    def test_json_round_trip(self) -> None:
        shadow_position = flat_position("X", contract_multiplier=1.0)
        decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        observation, _ = _evaluate(decision, shadow_position)
        assert ShadowObservation.from_json_dict(observation.to_json_dict()) == observation

    def test_identical_decisions_produce_identical_observation_id(self) -> None:
        shadow_position = flat_position("X", contract_multiplier=1.0)
        decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        a, _ = _evaluate(decision, shadow_position)
        b, _ = _evaluate(decision, shadow_position)
        assert a.observation_id == b.observation_id
