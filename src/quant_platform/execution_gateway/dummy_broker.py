"""Deterministic in-process dummy broker (Milestone 8, Section 12) -- the
ONLY adapter implementation this milestone ships. A real, stateful
order-book simulator, not a trivial mock: it tracks working orders,
triggers LIMIT/STOP conditions against supplied market events, applies a
scenario's partial-fill schedule, and maintains its own independent
position/cash bookkeeping (so `reconciliation.py` has a genuinely
separate view to check the gateway's own reconstruction against).

NO WALL-CLOCK DEPENDENCY: every result-affecting behavior here is a pure
function of `DummyBrokerScenarioSpec`, the commands it receives, the
market events `advance_market_event` is fed, and its own internal event
count -- `datetime.now()`/`time.time()` never appear in this file.

ARCHITECTURE: `submit_order`/`cancel_order`/`replace_order` are
SYNCHRONOUS acknowledgements only (Section 11's `AdapterCallResult` --
"did the call itself definitely reach the broker"), never the order's
actual lifecycle. `advance_market_event` (called once per consumed
market event by `runner.py`, exactly like `paper_trading.execution`'s
own event-driven fill engine) is where ORDER_ACKNOWLEDGED/fill/expiry
events are actually GENERATED into this broker's authoritative internal
event log. `poll_events` is where scenario-configured duplicate/delayed/
out-of-order DELIVERY is applied on top of that already-generated,
already-ordered log -- generation order and delivery order are
deliberately different concerns.

RAW PRICES ONLY, NO COST MODELING: fills are reported at the raw
ask/bid/close reference price the market event discloses, with no
spread/slippage/commission applied here -- `dispatcher.py` computes
those cost components using this platform's EXISTING cost formulas
(`backtesting.costs`/`paper_trading.costs`) when it builds the
`ExecutionFill` record, so this file never duplicates that math. No
margin is modeled (Section 25: "if margin is not modeled, document that
explicitly") -- this is a fully cash-settled, single-instrument-friendly
account view, exactly like `paper_trading.portfolio`'s own explicit
choice."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import BrokerSnapshotError
from quant_platform.execution_gateway.adapter import (
    DETERMINISTIC_DUMMY_CAPABILITIES,
    AdapterCallResult,
    AdapterCapabilities,
    BrokerAccountSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
)
from quant_platform.execution_gateway.commands import (
    CancelOrderCommand,
    ReplaceOrderCommand,
    SubmitOrderCommand,
)
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.health import AdapterHealthSnapshot
from quant_platform.execution_gateway.heartbeat import HeartbeatOutcome
from quant_platform.execution_gateway.models import (
    AdapterHealthStatus,
    BrokerEventType,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.normalization import DummyBrokerRawEvent, normalize_dummy_broker_event
from quant_platform.execution_gateway.specs import DummyBrokerScenarioSpec, RejectionRuleSpec
from quant_platform.paper_trading.events import BarEvent, MarketEvent, QuoteEvent

_AUTOMATIC_CANCEL_REASON_CODES = frozenset({"fok_not_fully_fillable", "ioc_not_immediately_fillable"})


def _instrument_of(event: MarketEvent) -> str | None:
    return getattr(event, "instrument", None)


def _reference_prices(event: MarketEvent) -> tuple[Decimal, Decimal] | None:
    """`(bid_or_close, ask_or_close)` -- `None` for a market event with no
    price content (session open/close, halt/resume, financing, end of
    stream)."""
    if isinstance(event, QuoteEvent):
        return (Decimal(str(event.bid)), Decimal(str(event.ask)))
    if isinstance(event, BarEvent):
        close = Decimal(str(event.close))
        return (close, close)
    return None


@dataclass
class _DummyPosition:
    signed_quantity: Decimal = Decimal(0)
    average_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)


@dataclass
class _DummyOrder:
    client_order_id: str
    broker_order_id: str
    instrument_id: str
    side: OrderSide
    order_type: OrderTypeKind
    quantity: Decimal
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForceKind
    reduce_only: bool
    command_sequence: int
    state: ExecutionOrderState = ExecutionOrderState.DISPATCHED
    filled_quantity: Decimal = Decimal(0)
    stop_triggered: bool = False
    ack_delay_remaining: int = 0
    fill_delay_remaining_this_fill: int = 0
    fok_ioc_evaluated: bool = False
    next_fill_index: int = 0
    next_fill_sequence: int = 0

    @property
    def is_terminal(self) -> bool:
        from quant_platform.execution_gateway.models import TERMINAL_EXECUTION_ORDER_STATES

        return self.state in TERMINAL_EXECUTION_ORDER_STATES


@dataclass
class DeterministicDummyBrokerAdapter:
    """The one and only adapter this milestone ships. Construct one per
    execution session; `initialize()` binds it to that session's id."""

    adapter_id: str
    scenario: DummyBrokerScenarioSpec
    starting_cash: Decimal = Decimal(0)

    _execution_session_id: str | None = field(default=None, init=False, repr=False)
    _orders: dict[str, _DummyOrder] = field(default_factory=dict, init=False, repr=False)
    _broker_order_id_to_client: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _all_events: list[DummyBrokerRawEvent] = field(default_factory=list, init=False, repr=False)
    _next_broker_sequence: int = field(default=1, init=False, repr=False)
    _operation_counter: int = field(default=0, init=False, repr=False)
    _delivered_once: set[int] = field(default_factory=set, init=False, repr=False)
    _positions: dict[str, _DummyPosition] = field(default_factory=dict, init=False, repr=False)
    _cash: Decimal = field(default=Decimal(0), init=False, repr=False)
    _last_reference_price: dict[str, Decimal] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cash = self.starting_cash

    # ------------------------------------------------------------------
    # ExecutionAdapter protocol
    # ------------------------------------------------------------------
    def capabilities(self) -> AdapterCapabilities:
        """Derives `guarantees_idempotent_*` from THIS adapter's own
        `scenario` rather than returning the static all-True constant --
        a scenario configured with `supports_idempotent_submit=False`
        (Section 12) must make `recovery.recover_unknown_orders` actually
        see a non-idempotent adapter, or the `remains_unknown`/never-
        blind-retry safety path (Section 16/23) can never be exercised."""
        return dataclasses.replace(
            DETERMINISTIC_DUMMY_CAPABILITIES, guarantees_idempotent_submit=self.scenario.supports_idempotent_submit,
            guarantees_idempotent_cancel=self.scenario.supports_idempotent_cancel, guarantees_idempotent_replace=self.scenario.supports_idempotent_replace,
        )

    def initialize(self, *, execution_session_id: str, event_time: datetime) -> None:  # noqa: ARG002 -- Protocol-required parameter, unused by this adapter
        self._execution_session_id = execution_session_id

    def _session_id(self) -> str:
        if self._execution_session_id is None:
            raise BrokerSnapshotError("DeterministicDummyBrokerAdapter.initialize() must be called before use")
        return self._execution_session_id

    def _is_connected(self) -> bool:
        if self.scenario.disconnect_at_sequence is None:
            return True
        reconnect = self.scenario.reconnect_at_sequence if self.scenario.reconnect_at_sequence is not None else 2**62
        return not (self.scenario.disconnect_at_sequence <= self._operation_counter < reconnect)

    def _new_call_id(self) -> str:
        return f"dummy-call-{self._operation_counter}"

    def _rule_matches(
        self, rule: RejectionRuleSpec, *, instrument_id: str | None, quantity: Decimal | None, command_sequence: int, client_order_id: str,
        order_type: OrderTypeKind | None, connected: bool,
    ) -> bool:
        if rule.reject_instrument_id is not None and instrument_id is not None and rule.reject_instrument_id == instrument_id:
            return True
        if rule.reject_quantity_above is not None and quantity is not None and quantity > rule.reject_quantity_above:
            return True
        if rule.reject_command_sequence is not None and rule.reject_command_sequence == command_sequence:
            return True
        if rule.reject_client_order_id is not None and rule.reject_client_order_id == client_order_id:
            return True
        if rule.reject_unsupported_order_type and order_type is not None:
            return True
        return bool(rule.reject_when_disconnected and not connected)

    def submit_order(self, command: SubmitOrderCommand, *, event_time: datetime) -> AdapterCallResult:
        self._operation_counter += 1
        connected = self._is_connected()
        for rule in self.scenario.rejection_rules:
            if self._rule_matches(
                rule, instrument_id=command.instrument_id, quantity=command.quantity, command_sequence=command.command_sequence,
                client_order_id=command.client_order_id, order_type=command.order_type, connected=connected,
            ):
                return AdapterCallResult(accepted_for_processing=False, rejection_reason=f"rejection_rule_{rule.rule_index}", adapter_call_id=self._new_call_id())
        if not connected:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="adapter_disconnected", adapter_call_id=self._new_call_id())

        existing = self._orders.get(command.client_order_id)
        if existing is not None:
            # Idempotent resubmission of the SAME economic operation
            # (Section 16) -- never mints a second order for one
            # client_order_id.
            return AdapterCallResult(accepted_for_processing=True, rejection_reason=None, adapter_call_id=self._new_call_id())

        broker_order_id = f"dummy-order-{command.client_order_id[:20]}"
        order = _DummyOrder(
            client_order_id=command.client_order_id, broker_order_id=broker_order_id, instrument_id=command.instrument_id, side=command.side,
            order_type=command.order_type, quantity=command.quantity, limit_price=command.limit_price, stop_price=command.stop_price,
            time_in_force=command.time_in_force, reduce_only=command.reduce_only, command_sequence=command.command_sequence,
            ack_delay_remaining=self.scenario.acknowledgement_delay_events,
        )
        self._orders[command.client_order_id] = order
        self._broker_order_id_to_client[broker_order_id] = command.client_order_id
        self._emit(
            event_type=BrokerEventType.ORDER_RECEIVED, broker_timestamp=event_time, broker_order_id=broker_order_id, client_order_id=command.client_order_id,
            original_quantity=command.quantity, source_command_id=command.command_id,
        )
        if order.ack_delay_remaining <= 0:
            self._acknowledge(order, event_time=event_time)
        return AdapterCallResult(accepted_for_processing=True, rejection_reason=None, adapter_call_id=self._new_call_id())

    def cancel_order(self, command: CancelOrderCommand, *, event_time: datetime) -> AdapterCallResult:
        self._operation_counter += 1
        connected = self._is_connected()
        order = self._orders.get(command.client_order_id)
        if order is None:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="unknown_order", adapter_call_id=self._new_call_id())
        for rule in self.scenario.rejection_rules:
            if self._rule_matches(
                rule, instrument_id=None, quantity=None, command_sequence=command.command_sequence, client_order_id=command.client_order_id,
                order_type=None, connected=connected,
            ):
                return AdapterCallResult(accepted_for_processing=False, rejection_reason=f"rejection_rule_{rule.rule_index}", adapter_call_id=self._new_call_id())
        if not connected:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="adapter_disconnected", adapter_call_id=self._new_call_id())
        if order.is_terminal:
            # Idempotent: cancelling an already-terminal order is a safe
            # no-op (Section 16), not an error.
            return AdapterCallResult(accepted_for_processing=True, rejection_reason=None, adapter_call_id=self._new_call_id())
        order.state = ExecutionOrderState.CANCELLED
        self._emit(event_type=BrokerEventType.CANCEL_ACKNOWLEDGED, broker_timestamp=event_time, broker_order_id=order.broker_order_id, client_order_id=order.client_order_id, source_command_id=command.command_id)
        return AdapterCallResult(accepted_for_processing=True, rejection_reason=None, adapter_call_id=self._new_call_id())

    def replace_order(self, command: ReplaceOrderCommand, *, event_time: datetime) -> AdapterCallResult:
        self._operation_counter += 1
        connected = self._is_connected()
        order = self._orders.get(command.client_order_id)
        if order is None:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="unknown_order", adapter_call_id=self._new_call_id())
        for rule in self.scenario.rejection_rules:
            if self._rule_matches(
                rule, instrument_id=None, quantity=command.replacement_quantity, command_sequence=command.command_sequence,
                client_order_id=command.client_order_id, order_type=None, connected=connected,
            ):
                return AdapterCallResult(accepted_for_processing=False, rejection_reason=f"rejection_rule_{rule.rule_index}", adapter_call_id=self._new_call_id())
        if not connected:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="adapter_disconnected", adapter_call_id=self._new_call_id())
        if order.is_terminal:
            return AdapterCallResult(accepted_for_processing=False, rejection_reason="order_already_terminal", adapter_call_id=self._new_call_id())
        if command.replacement_quantity is not None:
            if command.replacement_quantity <= order.filled_quantity:
                return AdapterCallResult(accepted_for_processing=False, rejection_reason="replacement_quantity_below_filled_quantity", adapter_call_id=self._new_call_id())
            order.quantity = command.replacement_quantity
        if command.replacement_limit_price is not None:
            order.limit_price = command.replacement_limit_price
        if command.replacement_stop_price is not None:
            order.stop_price = command.replacement_stop_price
        if command.replacement_time_in_force is not None:
            order.time_in_force = command.replacement_time_in_force
        self._emit(
            event_type=BrokerEventType.REPLACE_ACKNOWLEDGED, broker_timestamp=event_time, broker_order_id=order.broker_order_id,
            client_order_id=order.client_order_id, original_quantity=order.quantity, source_command_id=command.command_id,
        )
        return AdapterCallResult(accepted_for_processing=True, rejection_reason=None, adapter_call_id=self._new_call_id())

    # ------------------------------------------------------------------
    # Market-event-driven simulation (dummy-broker-specific, not part of
    # the generic ExecutionAdapter Protocol -- `runner.py` calls this once
    # per consumed market event, exactly like `paper_trading.execution`'s
    # own fill engine).
    # ------------------------------------------------------------------
    def advance_market_event(self, event: MarketEvent, *, event_time: datetime) -> None:
        instrument = _instrument_of(event)
        prices = _reference_prices(event)
        if instrument is not None and prices is not None:
            self._last_reference_price[instrument] = (prices[0] + prices[1]) / 2

        for order in list(self._orders.values()):
            if instrument is not None and order.instrument_id != instrument:
                continue
            if order.is_terminal:
                continue
            if order.ack_delay_remaining > 0:
                order.ack_delay_remaining -= 1
                if order.ack_delay_remaining <= 0:
                    self._acknowledge(order, event_time=event_time)
                continue
            if prices is None:
                continue
            reference = prices[1] if order.side is OrderSide.BUY else prices[0]
            fillable, fill_price = self._evaluate_fill_condition(order, reference)
            if not fillable:
                if order.time_in_force in (TimeInForceKind.FOK, TimeInForceKind.IOC) and not order.fok_ioc_evaluated:
                    order.fok_ioc_evaluated = True
                    reason = "fok_not_fully_fillable" if order.time_in_force is TimeInForceKind.FOK else "ioc_not_immediately_fillable"
                    self._auto_cancel(order, event_time=event_time, reason_code=reason)
                continue
            assert fill_price is not None
            if self.scenario.fill_delay_events > 0 and order.next_fill_index == 0 and order.fill_delay_remaining_this_fill < self.scenario.fill_delay_events:
                order.fill_delay_remaining_this_fill += 1
                continue
            self._apply_fill(order, fill_price=fill_price, event_time=event_time)

    def expire_day_orders(self, *, event_time: datetime) -> None:
        """Deterministic DAY-order expiry at session close (Section 13) --
        `runner.py` calls this when it consumes a `SessionCloseEvent`."""
        for order in self._orders.values():
            if order.time_in_force is TimeInForceKind.DAY and not order.is_terminal:
                order.state = ExecutionOrderState.EXPIRED
                self._emit(event_type=BrokerEventType.ORDER_EXPIRED, broker_timestamp=event_time, broker_order_id=order.broker_order_id, client_order_id=order.client_order_id)

    def _evaluate_fill_condition(self, order: _DummyOrder, reference_price: Decimal) -> tuple[bool, Decimal | None]:
        if order.order_type is OrderTypeKind.MARKET:
            return True, reference_price
        if order.order_type is OrderTypeKind.LIMIT:
            assert order.limit_price is not None
            if order.side is OrderSide.BUY and reference_price <= order.limit_price:
                return True, order.limit_price
            if order.side is OrderSide.SELL and reference_price >= order.limit_price:
                return True, order.limit_price
            return False, None
        # STOP: triggers once, then behaves market-like for its remaining
        # (and any subsequent) evaluation (Section 13).
        assert order.stop_price is not None
        if not order.stop_triggered:
            if (order.side is OrderSide.BUY and reference_price >= order.stop_price) or (order.side is OrderSide.SELL and reference_price <= order.stop_price):
                order.stop_triggered = True
            else:
                return False, None
        return True, reference_price

    def _acknowledge(self, order: _DummyOrder, *, event_time: datetime) -> None:
        order.state = ExecutionOrderState.ACKNOWLEDGED
        self._emit(event_type=BrokerEventType.ORDER_ACKNOWLEDGED, broker_timestamp=event_time, broker_order_id=order.broker_order_id, client_order_id=order.client_order_id, original_quantity=order.quantity)

    def _auto_cancel(self, order: _DummyOrder, *, event_time: datetime, reason_code: str) -> None:
        assert reason_code in _AUTOMATIC_CANCEL_REASON_CODES
        order.state = ExecutionOrderState.CANCELLED
        self._emit(event_type=BrokerEventType.CANCEL_ACKNOWLEDGED, broker_timestamp=event_time, broker_order_id=order.broker_order_id, client_order_id=order.client_order_id)

    def _apply_fill(self, order: _DummyOrder, *, fill_price: Decimal, event_time: datetime) -> None:
        remaining = order.quantity - order.filled_quantity
        schedule = self.scenario.partial_fill_schedule
        if schedule and order.next_fill_index < len(schedule):
            fraction = schedule[order.next_fill_index]
            fill_qty = min(remaining, order.quantity * fraction)
            order.next_fill_index += 1
        else:
            fill_qty = remaining

        if order.time_in_force is TimeInForceKind.FOK and fill_qty < remaining:
            # FOK must never partially fill (Section 13) -- if the
            # schedule would only partially satisfy it, the order is
            # cancelled outright instead.
            self._auto_cancel(order, event_time=event_time, reason_code="fok_not_fully_fillable")
            return

        order.filled_quantity += fill_qty
        order.fill_delay_remaining_this_fill = 0
        broker_fill_id = f"dummy-fill-{order.broker_order_id}-{order.next_fill_sequence}"
        order.next_fill_sequence += 1
        cumulative = order.filled_quantity
        remaining_after = order.quantity - cumulative
        is_full = remaining_after <= 0
        order.state = ExecutionOrderState.FILLED if is_full else ExecutionOrderState.PARTIALLY_FILLED
        self._emit(
            event_type=(BrokerEventType.ORDER_FILLED if is_full else BrokerEventType.ORDER_PARTIALLY_FILLED), broker_timestamp=event_time,
            broker_order_id=order.broker_order_id, broker_fill_id=broker_fill_id, client_order_id=order.client_order_id, original_quantity=order.quantity,
            filled_quantity=fill_qty, cumulative_filled_quantity=cumulative, remaining_quantity=remaining_after, fill_price=fill_price,
        )
        self._update_position_and_cash(order, fill_qty=fill_qty, fill_price=fill_price)
        if order.time_in_force is TimeInForceKind.IOC and not is_full:
            self._auto_cancel(order, event_time=event_time, reason_code="ioc_not_immediately_fillable")

    def _update_position_and_cash(self, order: _DummyOrder, *, fill_qty: Decimal, fill_price: Decimal) -> None:
        position = self._positions.setdefault(order.instrument_id, _DummyPosition())
        signed_delta = fill_qty if order.side is OrderSide.BUY else -fill_qty
        if position.signed_quantity == 0 or (position.signed_quantity > 0) == (signed_delta > 0):
            new_quantity = position.signed_quantity + signed_delta
            if new_quantity != 0:
                position.average_price = (position.average_price * abs(position.signed_quantity) + fill_price * abs(signed_delta)) / abs(new_quantity)
            position.signed_quantity = new_quantity
        else:
            closing_quantity = min(abs(signed_delta), abs(position.signed_quantity))
            direction = Decimal(1) if position.signed_quantity > 0 else Decimal(-1)
            position.realized_pnl += closing_quantity * (fill_price - position.average_price) * direction
            position.signed_quantity += signed_delta
            if abs(signed_delta) > closing_quantity:
                position.average_price = fill_price
            elif position.signed_quantity == 0:
                position.average_price = Decimal(0)
        self._cash += (-signed_delta) * fill_price

    def _emit(
        self, *, event_type: BrokerEventType, broker_timestamp: datetime, broker_order_id: str | None = None, broker_fill_id: str | None = None,
        client_order_id: str | None = None, original_quantity: Decimal | None = None, filled_quantity: Decimal | None = None,
        cumulative_filled_quantity: Decimal | None = None, remaining_quantity: Decimal | None = None, fill_price: Decimal | None = None,
        reject_code: str | None = None, reject_message: str | None = None, source_command_id: str | None = None,
    ) -> DummyBrokerRawEvent:
        raw = DummyBrokerRawEvent(
            broker_sequence=self._next_broker_sequence, event_type=event_type, broker_timestamp=broker_timestamp, broker_order_id=broker_order_id,
            broker_fill_id=broker_fill_id, client_order_id=client_order_id, original_quantity=original_quantity, filled_quantity=filled_quantity,
            cumulative_filled_quantity=cumulative_filled_quantity, remaining_quantity=remaining_quantity, fill_price=fill_price, reject_code=reject_code,
            reject_message=reject_message, source_command_id=source_command_id,
        )
        self._next_broker_sequence += 1
        self._all_events.append(raw)
        return raw

    # ------------------------------------------------------------------
    # poll_events -- scenario-configured duplicate/delayed/out-of-order
    # DELIVERY applied on top of the already-generated, already-ordered
    # authoritative event log (Section 12).
    # ------------------------------------------------------------------
    def _apply_out_of_order(self, events: list[DummyBrokerRawEvent]) -> list[DummyBrokerRawEvent]:
        result = list(events)
        position_by_index: dict[int, int] = {e.broker_sequence - 1: pos for pos, e in enumerate(result)}
        for group in self.scenario.out_of_order_event_groups:
            if all(i in position_by_index for i in group):
                positions = sorted(position_by_index[i] for i in group)
                originals = [result[p] for p in positions]
                for position, replacement in zip(positions, reversed(originals), strict=True):
                    result[position] = replacement
        return result

    def poll_events(self, *, after_sequence: int, max_events: int, event_time: datetime) -> tuple[BrokerEvent, ...]:
        self._operation_counter += 1
        candidates = [e for e in self._all_events if e.broker_sequence > after_sequence]
        delayed_indices = set(self.scenario.delayed_event_indices)
        deliverable: list[DummyBrokerRawEvent] = []
        for raw in candidates:
            idx = raw.broker_sequence - 1
            if idx in delayed_indices and idx not in self._delivered_once:
                self._delivered_once.add(idx)
                continue
            deliverable.append(raw)

        duplicate_indices = set(self.scenario.duplicate_event_indices)
        with_duplicates: list[DummyBrokerRawEvent] = []
        for raw in deliverable:
            with_duplicates.append(raw)
            if (raw.broker_sequence - 1) in duplicate_indices:
                with_duplicates.append(raw)

        reordered = self._apply_out_of_order(with_duplicates)
        truncated = reordered[:max_events]
        session_id = self._session_id()
        return tuple(normalize_dummy_broker_event(raw, execution_session_id=session_id, adapter_id=self.adapter_id, received_event_time=event_time) for raw in truncated)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _snapshot_of(self, order: _DummyOrder) -> BrokerOrderSnapshot:
        remaining = order.quantity - order.filled_quantity
        average_fill_price = self._last_reference_price.get(order.instrument_id) if order.filled_quantity > 0 else None
        return BrokerOrderSnapshot(
            broker_order_id=order.broker_order_id, client_order_id=order.client_order_id, instrument_id=order.instrument_id, side=order.side,
            order_type=order.order_type, state=order.state, original_quantity=order.quantity, current_quantity=order.quantity,
            filled_quantity=order.filled_quantity, remaining_quantity=remaining, average_fill_price=average_fill_price, limit_price=order.limit_price,
            stop_price=order.stop_price, time_in_force=order.time_in_force, last_broker_sequence=max(self._next_broker_sequence - 1, 1),
        )

    def query_order(self, *, broker_order_id: str | None, client_order_id: str | None, event_time: datetime) -> BrokerOrderSnapshot | None:  # noqa: ARG002
        self._operation_counter += 1
        if self._operation_counter in self.scenario.order_query_failure_sequences:
            raise BrokerSnapshotError(f"dummy broker: simulated order-query failure at operation {self._operation_counter}")
        cid = client_order_id
        if cid is None and broker_order_id is not None:
            cid = self._broker_order_id_to_client.get(broker_order_id)
        if cid is None:
            return None
        order = self._orders.get(cid)
        return None if order is None else self._snapshot_of(order)

    def query_open_orders(self, *, event_time: datetime) -> tuple[BrokerOrderSnapshot, ...]:  # noqa: ARG002
        self._operation_counter += 1
        if self._operation_counter in self.scenario.order_query_failure_sequences:
            raise BrokerSnapshotError(f"dummy broker: simulated open-orders-query failure at operation {self._operation_counter}")
        return tuple(self._snapshot_of(o) for o in self._orders.values() if not o.is_terminal)

    def query_positions(self, *, event_time: datetime) -> tuple[BrokerPositionSnapshot, ...]:  # noqa: ARG002
        self._operation_counter += 1
        if self._operation_counter in self.scenario.account_query_failure_sequences:
            raise BrokerSnapshotError(f"dummy broker: simulated positions-query failure at operation {self._operation_counter}")
        snapshots = []
        for instrument_id, position in self._positions.items():
            if position.signed_quantity == 0:
                continue
            mark = self._last_reference_price.get(instrument_id, position.average_price)
            unrealized = position.signed_quantity * (mark - position.average_price)
            snapshots.append(BrokerPositionSnapshot(
                instrument_id=instrument_id, signed_quantity=position.signed_quantity, average_price=position.average_price, contract_multiplier=Decimal(1),
                realized_pnl=position.realized_pnl, unrealized_pnl=unrealized,
            ))
        return tuple(snapshots)

    def query_account(self, *, event_time: datetime) -> BrokerAccountSnapshot:
        self._operation_counter += 1
        if self._operation_counter in self.scenario.account_query_failure_sequences:
            raise BrokerSnapshotError(f"dummy broker: simulated account-query failure at operation {self._operation_counter}")
        realized = sum((p.realized_pnl for p in self._positions.values()), Decimal(0))
        unrealized = sum(
            (p.signed_quantity * (self._last_reference_price.get(iid, p.average_price) - p.average_price) for iid, p in self._positions.items()), Decimal(0)
        )
        return BrokerAccountSnapshot(
            cash=self._cash, equity=self._cash + unrealized, realized_pnl=realized, unrealized_pnl=unrealized, accrued_costs=Decimal(0),
            position_count=sum(1 for p in self._positions.values() if p.signed_quantity != 0), snapshot_sequence=self._operation_counter,
            snapshot_event_time=event_time,
        )

    def health(self, *, event_time: datetime) -> AdapterHealthSnapshot:
        connected = self._is_connected()
        last_event_time = self._all_events[-1].broker_timestamp if self._all_events else None
        return AdapterHealthSnapshot(
            adapter_id=self.adapter_id, status=(AdapterHealthStatus.HEALTHY if connected else AdapterHealthStatus.UNAVAILABLE),
            last_successful_contact_event_time=(event_time if connected else None), last_event_received_event_time=last_event_time,
            last_heartbeat_event_time=event_time, consecutive_failures=(0 if connected else 1), event_lag=0, heartbeat_lag=0, can_submit=connected,
            can_cancel=connected, can_replace=connected, can_query=True, reason_codes=(() if connected else ("adapter_disconnected",)),
        )

    def heartbeat(self, *, event_time: datetime) -> HeartbeatOutcome:
        self._operation_counter += 1
        success = self._operation_counter not in self.scenario.heartbeat_failure_sequences and self._is_connected()
        return HeartbeatOutcome(adapter_id=self.adapter_id, success=success, event_time=event_time)

    def close(self) -> None:
        self._closed = True


__all__ = ["DeterministicDummyBrokerAdapter"]
