"""Milestone 7, Section 9: order policy hand fixtures. All 10 required
named fixtures (flat-to-long; flat-to-short; long-increase; long-reduce;
long-to-flat; long-to-short; short-to-long; rounded-to-zero-quantity;
duplicate-decision; exposure-clamped-decision) plus supporting cases
(abstention, cooldown, atomic-target-delta reversal mode, max-orders-per-
event truncation)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import OrderValidationError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.models import OrderSide, PositionIntentKind
from quant_platform.paper_trading.order_policy import OrderPolicyState, apply_order_policy
from quant_platform.paper_trading.specs import (
    DEFAULT_RISK_LIMITS,
    InstrumentSpec,
    LatencyPolicySpec,
    OrderPolicySpec,
    RiskLimitsSpec,
)
from quant_platform.paper_trading.strategy import PortfolioSnapshot, create_strategy_decision

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_STRATEGY = "a" * 64


def _instrument(**overrides: object) -> InstrumentSpec:
    defaults: dict[str, object] = {
        "symbol": "X", "base_currency": None, "quote_currency": "USD", "contract_multiplier": 1.0, "tick_size": 0.01, "tick_value": None,
        "quantity_step": 0.1, "minimum_quantity": 0.1, "maximum_quantity": None, "price_precision": 2, "quantity_precision": 2,
        "margin_mode": "cash", "account_currency": "USD", "financing_convention": "none", "trading_timezone": "UTC",
        "session_calendar_identity": "always_open",
    }
    defaults.update(overrides)
    return InstrumentSpec(**defaults)  # type: ignore[arg-type]


def _portfolio(signed_quantity: float, average_entry_price: float | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(instrument="X", signed_quantity=signed_quantity, average_entry_price=average_entry_price, cash=100_000.0, equity=100_000.0, unrealized_pnl=0.0, realized_pnl=0.0)


def _order_policy(**overrides: object) -> OrderPolicySpec:
    defaults: dict[str, object] = {"close_before_reverse": True, "cooldown_bars": 0, "maximum_orders_per_event": 5, "maximum_order_rate_per_window": 10, "order_rate_window_events": 20}
    defaults.update(overrides)
    return OrderPolicySpec(**defaults)  # type: ignore[arg-type]


def _latency() -> LatencyPolicySpec:
    return LatencyPolicySpec(decision_to_submit_ms=10, submit_to_accept_ms=10, accept_to_fill_eligible_ms=10)


def _decision(*, direction: PositionDirection, quantity: float, abstain: bool = False):
    event = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s")
    return create_strategy_decision(
        strategy_identity=_HEX_STRATEGY, event=event, decision_time=_T0, target_direction=direction, target_quantity=quantity, confidence=0.8,
        uncertainty=0.1, abstain=abstain, reason_codes=("test",),
    )


def _state(*, bars_since_last_order: int | None = None, orders_created_in_rate_window: int = 0) -> OrderPolicyState:
    return OrderPolicyState(bars_since_last_order=bars_since_last_order, orders_created_in_rate_window=orders_created_in_rate_window)


def _apply(decision, portfolio, *, instrument=None, policy=None, risk_limits=None, state=None):
    return apply_order_policy(
        decision, portfolio=portfolio, instrument=instrument or _instrument(), policy=policy or _order_policy(), risk_limits=risk_limits or DEFAULT_RISK_LIMITS,
        latency_policy=_latency(), session_id="session-1", create_time=_T0, state=state or _state(),
    )


class TestTenRequiredHandFixtures:
    def test_flat_to_long(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(0.0))
        assert len(orders) == 1
        assert orders[0].side is OrderSide.BUY
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[0].position_intent is PositionIntentKind.OPEN

    def test_flat_to_short(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.SHORT, quantity=2.0), _portfolio(0.0))
        assert len(orders) == 1
        assert orders[0].side is OrderSide.SELL
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[0].position_intent is PositionIntentKind.OPEN

    def test_long_increase(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=3.0), _portfolio(1.0, 100.0))
        assert len(orders) == 1
        assert orders[0].side is OrderSide.BUY
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[0].position_intent is PositionIntentKind.INCREASE
        assert orders[0].reduce_only is False

    def test_long_reduce(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=1.0), _portfolio(3.0, 100.0))
        assert len(orders) == 1
        assert orders[0].side is OrderSide.SELL
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[0].position_intent is PositionIntentKind.REDUCE
        assert orders[0].reduce_only is True

    def test_long_to_flat(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.FLAT, quantity=0.0), _portfolio(2.0, 100.0))
        assert len(orders) == 1
        assert orders[0].side is OrderSide.SELL
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[0].position_intent is PositionIntentKind.CLOSE
        assert orders[0].reduce_only is True

    def test_long_to_short_close_before_reverse(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.SHORT, quantity=1.5), _portfolio(2.0, 100.0), policy=_order_policy(close_before_reverse=True))
        assert len(orders) == 2
        assert orders[0].position_intent is PositionIntentKind.CLOSE
        assert orders[0].side is OrderSide.SELL
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[1].position_intent is PositionIntentKind.OPEN
        assert orders[1].side is OrderSide.SELL
        assert orders[1].quantity == pytest.approx(1.5)

    def test_short_to_long_close_before_reverse(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=1.0), _portfolio(-2.0, 100.0), policy=_order_policy(close_before_reverse=True))
        assert len(orders) == 2
        assert orders[0].position_intent is PositionIntentKind.CLOSE
        assert orders[0].side is OrderSide.BUY
        assert orders[0].quantity == pytest.approx(2.0)
        assert orders[1].position_intent is PositionIntentKind.OPEN
        assert orders[1].side is OrderSide.BUY
        assert orders[1].quantity == pytest.approx(1.0)

    def test_rounded_to_zero_quantity_produces_no_order(self) -> None:
        """A tiny requested delta that rounds down below `minimum_quantity`
        must produce NO order, not an order for a fabricated size."""
        instrument = _instrument(quantity_step=1.0, minimum_quantity=1.0)
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=0.3), _portfolio(0.0), instrument=instrument)
        assert orders == ()

    def test_duplicate_decision_is_idempotent(self) -> None:
        """Calling the policy twice with the IDENTICAL decision produces
        byte-identical orders (same client_order_id/order_id) -- the
        deterministic basis for the runner's own duplicate-event rejection
        at the ledger layer."""
        decision = _decision(direction=PositionDirection.LONG, quantity=2.0)
        first = _apply(decision, _portfolio(0.0))
        second = _apply(decision, _portfolio(0.0))
        assert first == second
        assert first[0].client_order_id == second[0].client_order_id

    def test_exposure_clamped_decision(self) -> None:
        risk_limits = RiskLimitsSpec(
            maximum_signed_position=None, maximum_absolute_position=1.5, maximum_gross_exposure=None, maximum_order_quantity=None,
            maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None,
            maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
            maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
        )
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=5.0), _portfolio(0.0), risk_limits=risk_limits)
        assert len(orders) == 1
        assert orders[0].quantity == pytest.approx(1.5)


class TestOrderPolicySupportingBehavior:
    def test_abstention_produces_no_orders(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0, abstain=True), _portfolio(0.0))
        assert orders == ()

    def test_no_op_at_target_produces_no_orders(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(2.0, 100.0))
        assert orders == ()

    def test_cooldown_blocks_new_order(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(0.0), policy=_order_policy(cooldown_bars=5), state=_state(bars_since_last_order=2))
        assert orders == ()

    def test_cooldown_elapsed_allows_order(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(0.0), policy=_order_policy(cooldown_bars=5), state=_state(bars_since_last_order=5))
        assert len(orders) == 1

    def test_no_prior_order_ignores_cooldown(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(0.0), policy=_order_policy(cooldown_bars=5), state=_state(bars_since_last_order=None))
        assert len(orders) == 1

    def test_atomic_target_delta_reversal_produces_one_order(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.SHORT, quantity=1.0), _portfolio(2.0, 100.0), policy=_order_policy(close_before_reverse=False))
        assert len(orders) == 1
        assert orders[0].position_intent is PositionIntentKind.REVERSE
        assert orders[0].side is OrderSide.SELL
        assert orders[0].quantity == pytest.approx(3.0)

    def test_maximum_orders_per_event_truncates(self) -> None:
        orders = _apply(
            _decision(direction=PositionDirection.SHORT, quantity=1.0), _portfolio(2.0, 100.0),
            policy=_order_policy(close_before_reverse=True, maximum_orders_per_event=1),
        )
        assert len(orders) == 1

    def test_rate_limit_budget_exhausted_produces_no_orders(self) -> None:
        orders = _apply(_decision(direction=PositionDirection.LONG, quantity=2.0), _portfolio(0.0), state=_state(orders_created_in_rate_window=10))
        assert orders == ()

    def test_mismatched_portfolio_instrument_rejected(self) -> None:
        mismatched = PortfolioSnapshot(instrument="OTHER", signed_quantity=0.0, average_entry_price=None, cash=0.0, equity=0.0, unrealized_pnl=0.0, realized_pnl=0.0)
        with pytest.raises(OrderValidationError, match="instrument"):
            _apply(_decision(direction=PositionDirection.LONG, quantity=1.0), mismatched)
