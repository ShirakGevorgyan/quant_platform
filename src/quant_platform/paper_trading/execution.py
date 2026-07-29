"""Paper execution engine (Milestone 7, Section 10). Pure functions that
decide whether/how a single `OrderRequest` fills against a single
`MarketEvent`, and translate that decision into the state-machine
transitions and `Fill` records the rest of the pipeline needs -- no
mutable engine object, no wall-clock, no hidden state, matching every
other module in this package.

PRICE SEMANTICS, EXACTLY (Section 10.1-10.3):
- MARKET: QUOTE mode fills at the real ask (buy) / bid (sell) directly
  from the `QuoteEvent`, plus slippage on top. BAR mode has no real
  ask/bid, so it fills at the bar's configured mark field (`ExecutionPolicySpec.
  mark_field`, currently always CLOSE) with `costs.bar_mode_spread_
  adjusted_price` approximating the spread, plus slippage on top.
- LIMIT: QUOTE mode gets genuine price improvement (a buy limit fills at
  the real ask when ask <= limit_price, never worse than the limit).
  BAR mode has no intrabar path, so Section 10's "no impossible intrabar
  information" rule applies: it fills EXACTLY at the limit price when the
  bar's [low, high] range reaches it, UNLESS the bar's own OPEN already
  gapped through the limit (a real, disclosed price, not a guess) -- then
  it fills at that gap-improved open price, never a better undisclosed
  price.
- STOP: same bar-mode gap logic, symmetric: fills at the stop price
  unless the bar's open already gapped through it, in which case it fills
  at that (realistically worse) open price. QUOTE mode fills at the real
  ask/bid once the trigger condition is met, which may already be worse
  than the stop price on a fast-moving quote stream -- gap behavior is
  therefore automatic in QUOTE mode (no special-casing needed).

Slippage (Section 10.6, deterministic-only per `backtesting.specs.
SlippageSpec`'s own documented limitation) is applied identically on top
of whatever base price was determined, for every order type and mode.

Partial fills (Section 10.7-10.8): `DETERMINISTIC_PARTIAL` is only ever
honored in QUOTE mode against an event that actually discloses a size for
the relevant side (`liquidity_policy.trust_disclosed_size` AND `bid_size`/
`ask_size` present) -- every other case (BAR mode, missing size, or
`FULL_FILL_ONLY` configured) fails closed to a full-or-nothing fill,
never inventing a liquidity assumption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.backtesting.models import PositionDirection
from quant_platform.backtesting.specs import CommissionSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import StrategyRuntimeError
from quant_platform.paper_trading.costs import (
    bar_mode_spread_adjusted_price,
    compute_commission_dollars,
    compute_slippage_cost_dollars,
    compute_spread_cost_dollars,
    quote_mode_spread_cost_dollars,
    slippage_adjusted_price,
)
from quant_platform.paper_trading.events import BarEvent, MarketEvent, QuoteEvent, market_event_id
from quant_platform.paper_trading.fills import Fill, create_fill
from quant_platform.paper_trading.models import (
    MarkFieldKind,
    OrderSide,
    OrderState,
    OrderTypeKind,
    PartialFillPolicyKind,
    PositionIntentKind,
    RejectReasonKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import OrderRequest, OrderStateEvent, create_order_state_event
from quant_platform.paper_trading.specs import (
    ExecutionPolicySpec,
    FillPolicySpec,
    InstrumentSpec,
    LiquidityPolicySpec,
)

_QUANTITY_TOLERANCE = 1e-9


def _direction_for_side(side: OrderSide) -> PositionDirection:
    return PositionDirection.LONG if side is OrderSide.BUY else PositionDirection.SHORT


def _is_entry_fill(position_intent: PositionIntentKind) -> bool:
    """OPEN/INCREASE/REVERSE are entry-side cost semantics; REDUCE/CLOSE
    are exit-side. A REVERSE order is treated as entry (see `costs.py`
    module docstring precedent in `order_policy.py`: the dominant economic
    effect of an atomic reversal is opening new exposure in the new
    direction)."""
    return position_intent in (PositionIntentKind.OPEN, PositionIntentKind.INCREASE, PositionIntentKind.REVERSE)


@dataclass(frozen=True, slots=True)
class FillCandidate:
    """What ONE evaluation of (order, event) produces, before cost/Fill
    construction -- `price` is the base execution price BEFORE slippage,
    matching `costs.py`'s `is_entry`-aware adjustment functions."""

    price: float
    quantity: float
    liquidity_assumption: PartialFillPolicyKind


def _bar_mark_field_price(bar: BarEvent, mark_field: MarkFieldKind) -> float:
    """Selects the bar field `ExecutionPolicySpec.mark_field` names --
    currently always `CLOSE` (the enum's only member), but routed through
    the spec rather than hardcoded so a future additional mark-field
    option does not require touching this function's callers."""
    if mark_field is MarkFieldKind.CLOSE:
        return bar.close
    raise StrategyRuntimeError(f"_bar_mark_field_price: unsupported mark_field {mark_field!r}")  # pragma: no cover - exhaustive enum


def _market_order_fill_candidate(order: OrderRequest, event: MarketEvent, *, execution_policy: ExecutionPolicySpec, spread_policy: SpreadSpec, is_entry: bool) -> FillCandidate | None:
    if isinstance(event, QuoteEvent):
        price = event.ask if order.side is OrderSide.BUY else event.bid
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    if isinstance(event, BarEvent):
        direction = _direction_for_side(order.side)
        reference_price = _bar_mark_field_price(event, execution_policy.mark_field)
        price = bar_mode_spread_adjusted_price(spread_policy, reference_price, direction, is_entry=is_entry)
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    return None


def _limit_order_fill_candidate(order: OrderRequest, event: MarketEvent) -> FillCandidate | None:
    assert order.limit_price is not None
    if isinstance(event, QuoteEvent):
        if order.side is OrderSide.BUY:
            if event.ask > order.limit_price:
                return None
            price = event.ask  # genuine price improvement -- real quoted information
        else:
            if event.bid < order.limit_price:
                return None
            price = event.bid
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    if isinstance(event, BarEvent):
        if order.side is OrderSide.BUY:
            if event.low > order.limit_price:
                return None
            # Gap-improvement only from a DISCLOSED price (the bar's own
            # open) -- never an undisclosed intrabar guess.
            price = min(order.limit_price, event.open) if event.open <= order.limit_price else order.limit_price
        else:
            if event.high < order.limit_price:
                return None
            price = max(order.limit_price, event.open) if event.open >= order.limit_price else order.limit_price
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    return None


def _stop_order_fill_candidate(order: OrderRequest, event: MarketEvent) -> FillCandidate | None:
    assert order.stop_price is not None
    if isinstance(event, QuoteEvent):
        if order.side is OrderSide.BUY:
            if event.ask < order.stop_price:
                return None
            price = event.ask  # may already be worse than stop_price -- real gap behavior
        else:
            if event.bid > order.stop_price:
                return None
            price = event.bid
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    if isinstance(event, BarEvent):
        if order.side is OrderSide.BUY:
            if event.high < order.stop_price:
                return None
            price = max(order.stop_price, event.open) if event.open >= order.stop_price else order.stop_price
        else:
            if event.low > order.stop_price:
                return None
            price = min(order.stop_price, event.open) if event.open <= order.stop_price else order.stop_price
        return FillCandidate(price=price, quantity=order.quantity, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY)
    return None


def _apply_liquidity_limit(candidate: FillCandidate, order: OrderRequest, event: MarketEvent, *, remaining_quantity: float, liquidity_policy: LiquidityPolicySpec, fill_policy: FillPolicySpec) -> FillCandidate:
    """Clamps `candidate.quantity` to `remaining_quantity`, and -- only
    when genuinely licensed to (Section 10.8) -- further to a disclosed
    QUOTE size. Never invents a liquidity number."""
    quantity = min(candidate.quantity, remaining_quantity)
    liquidity_assumption = PartialFillPolicyKind.FULL_FILL_ONLY
    if fill_policy.partial_fill_policy is PartialFillPolicyKind.DETERMINISTIC_PARTIAL and liquidity_policy.trust_disclosed_size and isinstance(event, QuoteEvent):
        disclosed_size = event.ask_size if order.side is OrderSide.BUY else event.bid_size
        if disclosed_size is not None:
            quantity = min(quantity, disclosed_size)
            liquidity_assumption = PartialFillPolicyKind.DETERMINISTIC_PARTIAL
    return FillCandidate(price=candidate.price, quantity=quantity, liquidity_assumption=liquidity_assumption)


def evaluate_order_against_event(
    order: OrderRequest, event: MarketEvent, *, remaining_quantity: float, execution_policy: ExecutionPolicySpec, spread_policy: SpreadSpec,
    slippage_policy: SlippageSpec, liquidity_policy: LiquidityPolicySpec, fill_policy: FillPolicySpec,
) -> FillCandidate | None:
    """The single entry point deciding whether `order` fills against
    `event` at all, and if so at what price/quantity (before cost
    computation). Returns `None` if the order's trigger condition is not
    met by this event -- the order remains WORKING.

    NOTE (scope): this evaluates ONE order in isolation. `execution_
    policy.bar_ambiguity_policy` (resolving a same-bar race between a
    working STOP and a working LIMIT on the SAME position) is not
    consulted here -- `order_policy.py`'s order policy never creates such
    a simultaneous bracket, so this function is never actually asked to
    resolve one in this milestone's own pipeline. Documented as a known
    limitation for a future strategy that submits bracket orders
    directly."""
    if remaining_quantity <= _QUANTITY_TOLERANCE:
        return None
    is_entry = _is_entry_fill(order.position_intent)

    if order.order_type is OrderTypeKind.MARKET:
        base = _market_order_fill_candidate(order, event, execution_policy=execution_policy, spread_policy=spread_policy, is_entry=is_entry)
    elif order.order_type is OrderTypeKind.LIMIT:
        base = _limit_order_fill_candidate(order, event)
    elif order.order_type is OrderTypeKind.STOP:
        base = _stop_order_fill_candidate(order, event)
    else:
        raise StrategyRuntimeError(f"evaluate_order_against_event: unsupported order_type {order.order_type!r}")  # pragma: no cover - exhaustive enum

    if base is None:
        return None

    direction = _direction_for_side(order.side)
    slipped_price = slippage_adjusted_price(slippage_policy, base.price, direction, is_entry=is_entry)
    candidate = FillCandidate(price=slipped_price, quantity=base.quantity, liquidity_assumption=base.liquidity_assumption)
    return _apply_liquidity_limit(candidate, order, event, remaining_quantity=remaining_quantity, liquidity_policy=liquidity_policy, fill_policy=fill_policy)


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    order_state_events: tuple[OrderStateEvent, ...]
    fills: tuple[Fill, ...]


def _quote_spread_cost(event: MarketEvent, order: OrderRequest, *, quantity: float, contract_multiplier: float) -> float:
    if isinstance(event, QuoteEvent):
        return quote_mode_spread_cost_dollars(bid=event.bid, ask=event.ask, side=order.side, quantity=quantity, contract_multiplier=contract_multiplier)
    return 0.0


def process_order_against_event(
    order: OrderRequest, *, current_state: OrderState, filled_quantity_so_far: float, event: MarketEvent, event_time: datetime, sequence: int,
    instrument: InstrumentSpec, execution_policy: ExecutionPolicySpec, spread_policy: SpreadSpec, slippage_policy: SlippageSpec,
    commission_policy: CommissionSpec, fill_policy: FillPolicySpec, liquidity_policy: LiquidityPolicySpec,
) -> ExecutionOutcome:
    """Orchestrates one order's reaction to one market event: decides
    whether/how it fills (`evaluate_order_against_event`), and produces
    the resulting `OrderStateEvent`(s) and `Fill`(s). A no-op (empty
    outcome) if `current_state` is not `WORKING`/`PARTIALLY_FILLED`, or if
    the order simply does not trigger against this event and its
    time-in-force allows it to keep waiting."""
    if current_state not in (OrderState.WORKING, OrderState.PARTIALLY_FILLED):
        return ExecutionOutcome(order_state_events=(), fills=())

    remaining_quantity = order.quantity - filled_quantity_so_far
    candidate = evaluate_order_against_event(
        order, event, remaining_quantity=remaining_quantity, execution_policy=execution_policy, spread_policy=spread_policy,
        slippage_policy=slippage_policy, liquidity_policy=liquidity_policy, fill_policy=fill_policy,
    )

    if candidate is None:
        if order.time_in_force in (TimeInForceKind.IOC, TimeInForceKind.FOK):
            reason = RejectReasonKind.FOK_NOT_FULLY_FILLABLE if order.time_in_force is TimeInForceKind.FOK else RejectReasonKind.IOC_NOT_IMMEDIATELY_FILLABLE
            cancel_event = create_order_state_event(
                order_id=order.order_id, session_id=order.session_id, from_state=current_state, to_state=OrderState.CANCELLED, event_time=event_time,
                sequence=sequence, reason_code=reason, source_market_event_identity=market_event_id(event),
            )
            return ExecutionOutcome(order_state_events=(cancel_event,), fills=())
        return ExecutionOutcome(order_state_events=(), fills=())

    if order.time_in_force is TimeInForceKind.FOK and candidate.quantity + _QUANTITY_TOLERANCE < remaining_quantity:
        cancel_event = create_order_state_event(
            order_id=order.order_id, session_id=order.session_id, from_state=current_state, to_state=OrderState.CANCELLED, event_time=event_time,
            sequence=sequence, reason_code=RejectReasonKind.FOK_NOT_FULLY_FILLABLE, source_market_event_identity=market_event_id(event),
        )
        return ExecutionOutcome(order_state_events=(cancel_event,), fills=())

    is_entry = _is_entry_fill(order.position_intent)
    direction = _direction_for_side(order.side)
    is_final = candidate.quantity + _QUANTITY_TOLERANCE >= remaining_quantity

    if isinstance(event, QuoteEvent):
        spread_cost = _quote_spread_cost(event, order, quantity=candidate.quantity, contract_multiplier=instrument.contract_multiplier)
    else:
        spread_cost = compute_spread_cost_dollars(spread_policy, candidate.price, direction, is_entry=is_entry, quantity=candidate.quantity, contract_multiplier=instrument.contract_multiplier)
    slippage_cost = compute_slippage_cost_dollars(slippage_policy, candidate.price, direction, is_entry=is_entry, quantity=candidate.quantity, contract_multiplier=instrument.contract_multiplier)
    notional = candidate.price * candidate.quantity * instrument.contract_multiplier
    commission_cost = compute_commission_dollars(commission_policy, notional=notional)

    fill = create_fill(
        order_id=order.order_id, session_id=order.session_id, instrument=order.instrument, side=order.side, quantity=candidate.quantity,
        price=candidate.price, contract_multiplier=instrument.contract_multiplier, spread_cost=spread_cost, slippage_cost=slippage_cost,
        commission_cost=commission_cost, execution_time=event_time, source_market_event_identity=market_event_id(event),
        liquidity_assumption=candidate.liquidity_assumption, is_final=is_final,
    )

    target_state = OrderState.FILLED if is_final else OrderState.PARTIALLY_FILLED
    state_event = create_order_state_event(
        order_id=order.order_id, session_id=order.session_id, from_state=current_state, to_state=target_state, event_time=event_time,
        sequence=sequence, source_market_event_identity=market_event_id(event),
    )

    events: list[OrderStateEvent] = [state_event]
    if not is_final and order.time_in_force in (TimeInForceKind.IOC, TimeInForceKind.FOK):
        # IOC never leaves a remainder working; the FOK all-or-nothing
        # branch above already guarantees this partial-fill path is
        # unreachable for FOK, so only IOC can reach here.
        cancel_remainder = create_order_state_event(
            order_id=order.order_id, session_id=order.session_id, from_state=target_state, to_state=OrderState.CANCELLED, event_time=event_time,
            sequence=sequence + 1, reason_code=RejectReasonKind.IOC_NOT_IMMEDIATELY_FILLABLE, source_market_event_identity=market_event_id(event),
        )
        events.append(cancel_remainder)

    return ExecutionOutcome(order_state_events=tuple(events), fills=(fill,))


__all__ = ["ExecutionOutcome", "FillCandidate", "evaluate_order_against_event", "process_order_against_event"]
