"""Order policy: deterministic conversion of one `StrategyDecision` into
zero-or-more `OrderRequest`s (Milestone 7, Section 9). A pure function of
its inputs -- no wall-clock, no random numbers, and it is NEVER handed
anything beyond the current decision/portfolio/instrument/policy/state, so
it structurally cannot "inspect future prices" (Section 9's own
instruction) -- there is no price field on any of its parameters at all;
sizing comes entirely from `StrategyDecision.target_quantity`, already
fixed by the strategy at decision time.

`OrderPolicyState` is the small piece of caller-tracked history (cooldown/
rate-limit counters) this function needs but must not own itself --
`runner.py` maintains and threads it through, exactly like `Clock` is
injected rather than read globally."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import OrderValidationError
from quant_platform.paper_trading.clock import submission_time_for
from quant_platform.paper_trading.models import OrderSide, OrderTypeKind, PositionIntentKind, TimeInForceKind
from quant_platform.paper_trading.orders import OrderRequest, create_order_request
from quant_platform.paper_trading.specs import (
    InstrumentSpec,
    LatencyPolicySpec,
    OrderPolicySpec,
    RiskLimitsSpec,
)
from quant_platform.paper_trading.strategy import PortfolioSnapshot, StrategyDecision


@dataclass(frozen=True, slots=True)
class OrderPolicyState:
    """Caller-tracked history -- owned and updated by `runner.py`, never
    by `apply_order_policy` itself."""

    bars_since_last_order: int | None
    orders_created_in_rate_window: int


def _signed_target_quantity(decision: StrategyDecision) -> float:
    if decision.target_direction is PositionDirection.LONG:
        return decision.target_quantity
    if decision.target_direction is PositionDirection.SHORT:
        return -decision.target_quantity
    return 0.0


def _clamp_to_exposure_limit(signed_target: float, *, risk_limits: RiskLimitsSpec) -> float:
    limit = risk_limits.maximum_absolute_position
    if limit is None or abs(signed_target) <= limit:
        return signed_target
    return math.copysign(limit, signed_target) if signed_target != 0.0 else 0.0


def _round_to_step(quantity: float, *, instrument: InstrumentSpec) -> float:
    """Floors toward zero to the instrument's `quantity_step` -- never
    rounds UP past what was actually requested/clamped."""
    steps = math.floor(round(quantity / instrument.quantity_step, 9))
    return round(steps * instrument.quantity_step, instrument.quantity_precision)


def _position_intent_for(*, current: float, target: float) -> PositionIntentKind:
    if current == 0.0:
        return PositionIntentKind.OPEN
    if target == 0.0:
        return PositionIntentKind.CLOSE
    same_sign = (current > 0) == (target > 0)
    if not same_sign:
        return PositionIntentKind.REVERSE
    return PositionIntentKind.INCREASE if abs(target) > abs(current) else PositionIntentKind.REDUCE


def apply_order_policy(
    decision: StrategyDecision, *, portfolio: PortfolioSnapshot, instrument: InstrumentSpec, policy: OrderPolicySpec, risk_limits: RiskLimitsSpec,
    latency_policy: LatencyPolicySpec, session_id: str, create_time: datetime, state: OrderPolicyState,
) -> tuple[OrderRequest, ...]:
    if portfolio.instrument != instrument.symbol:
        raise OrderValidationError(f"apply_order_policy: portfolio.instrument={portfolio.instrument!r} does not match instrument.symbol={instrument.symbol!r}")

    if decision.abstain:
        return ()

    if state.bars_since_last_order is not None and state.bars_since_last_order < policy.cooldown_bars:
        return ()

    current_signed = portfolio.signed_quantity
    raw_target_signed = _signed_target_quantity(decision)
    target_signed = _clamp_to_exposure_limit(raw_target_signed, risk_limits=risk_limits)

    delta = target_signed - current_signed
    if delta == 0.0:
        return ()

    submit_time = submission_time_for(create_time, latency_policy)

    def _make_order(*, index: int, side: OrderSide, quantity: float, position_intent: PositionIntentKind) -> OrderRequest | None:
        rounded_quantity = _round_to_step(quantity, instrument=instrument)
        if rounded_quantity < instrument.minimum_quantity:
            return None
        return create_order_request(
            client_order_id=f"{decision.decision_id}:{index}", session_id=session_id, strategy_decision_id=decision.decision_id,
            instrument=instrument.symbol, side=side, order_type=OrderTypeKind.MARKET, quantity=rounded_quantity, time_in_force=TimeInForceKind.DAY,
            create_time=create_time, submit_time=submit_time, reduce_only=(position_intent in (PositionIntentKind.CLOSE, PositionIntentKind.REDUCE)),
            position_intent=position_intent,
        )

    orders: list[OrderRequest] = []
    is_reversal = current_signed != 0.0 and target_signed != 0.0 and (current_signed > 0) != (target_signed > 0)

    if is_reversal and policy.close_before_reverse:
        close_order = _make_order(index=0, side=(OrderSide.SELL if current_signed > 0 else OrderSide.BUY), quantity=abs(current_signed), position_intent=PositionIntentKind.CLOSE)
        if close_order is not None:
            orders.append(close_order)
        open_order = _make_order(index=1, side=(OrderSide.BUY if target_signed > 0 else OrderSide.SELL), quantity=abs(target_signed), position_intent=PositionIntentKind.OPEN)
        if open_order is not None:
            orders.append(open_order)
    else:
        intent = PositionIntentKind.REVERSE if is_reversal else _position_intent_for(current=current_signed, target=target_signed)
        order = _make_order(index=0, side=(OrderSide.BUY if delta > 0 else OrderSide.SELL), quantity=abs(delta), position_intent=intent)
        if order is not None:
            orders.append(order)

    remaining_rate_budget = max(0, policy.maximum_order_rate_per_window - state.orders_created_in_rate_window)
    max_orders = min(policy.maximum_orders_per_event, remaining_rate_budget)
    return tuple(orders[:max_orders])


__all__ = ["OrderPolicyState", "apply_order_policy"]
