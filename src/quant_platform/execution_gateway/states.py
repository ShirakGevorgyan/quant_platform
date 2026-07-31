"""Reconstructed execution-order aggregate and fill model (Milestone 8,
Section 9/10). `ExecutionOrder` is NEVER persisted as the authoritative
mutable state of an order -- `persistence.py`'s append-only ledger is the
source of truth; `reconstruct_execution_order` REBUILDS an `ExecutionOrder`
snapshot purely by replaying a `SubmitOrderCommand`, its
`ExecutionOrderStateEvent` history, its `BrokerEvent` history, and its
`ExecutionFill` history -- exactly like `paper_trading.reconciliation`
rebuilds a `PositionState` purely from the ledger, never from cached
bookkeeping.

`remaining_quantity` is ALWAYS a DERIVED field (`current_quantity -
filled_quantity`), never independently stored -- this is what makes
Section 9's invariant `filled_quantity + remaining_quantity ==
current_quantity` hold BY CONSTRUCTION rather than needing a runtime
reconciliation check. `current_quantity` itself already reflects the
latest ACCEPTED replace (if any) by the time it reaches
`reconstruct_execution_order` -- the caller (`dispatcher.py`/
`reconciliation.py`, which walk the full ledger) is responsible for
resolving "which replace won" and passing the resulting quantity in;
this module only enforces internal consistency of the resulting snapshot.

CROSS-ORDER invariants ("one broker order ID cannot belong to two
unrelated client order IDs", "one client order ID cannot map to two
unrelated economic submits", "broker order ID cannot silently change")
are SESSION-WIDE checks across every order, not a single order's own
construction -- see `reconciliation.py`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import ExecutionFillError, ExecutionOrderStateError
from quant_platform.execution_gateway.commands import SubmitOrderCommand
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.execution_gateway.models import (
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    resolve_execution_order_state,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp

EXECUTION_ORDER_ID_KIND = "execution_order_id"
EXECUTION_FILL_KIND = "execution_fill"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionFillError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise ExecutionFillError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ExecutionFillError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise ExecutionFillError(f"{field_name}: {exc}") from exc


def compute_execution_order_id(submit_command: SubmitOrderCommand) -> str:
    """One `SubmitOrderCommand` -> exactly one `execution_order_id`,
    forever -- deterministic, so `dispatcher.py` recomputes the SAME
    order identity on a safe retry of the same submit rather than
    minting a second, orphaned order."""
    return compute_content_id(EXECUTION_ORDER_ID_KIND, {"command_id": submit_command.command_id})


# --------------------------------------------------------------------------
# ExecutionFill (Section 10)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionFill:
    execution_fill_id: str
    execution_session_id: str
    execution_order_id: str
    execution_intent_id: str

    broker_event_id: str
    broker_fill_id: str
    broker_order_id: str
    client_order_id: str

    instrument_id: str
    side: OrderSide

    quantity: Decimal
    price: Decimal
    contract_multiplier: Decimal
    gross_notional: Decimal

    commission: Decimal
    spread_component: Decimal
    slippage_component: Decimal

    broker_timestamp: datetime
    received_event_time: datetime
    fill_sequence: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("execution_session_id", self.execution_session_id), ("execution_order_id", self.execution_order_id),
            ("execution_intent_id", self.execution_intent_id), ("broker_event_id", self.broker_event_id),
        ):
            if not is_valid_sha256_hex(value):
                raise ExecutionFillError(f"ExecutionFill.{field_name} must be a valid sha256 hex digest, got {value!r}")
        if not self.broker_fill_id or not self.broker_order_id or not self.client_order_id or not self.instrument_id:
            raise ExecutionFillError("ExecutionFill.broker_fill_id/broker_order_id/client_order_id/instrument_id must not be empty")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise ExecutionFillError(f"ExecutionFill.quantity must be finite and > 0, got {self.quantity!r}")
        if not self.price.is_finite() or self.price <= 0:
            raise ExecutionFillError(f"ExecutionFill.price must be finite and > 0, got {self.price!r}")
        if not self.contract_multiplier.is_finite() or self.contract_multiplier <= 0:
            raise ExecutionFillError(f"ExecutionFill.contract_multiplier must be finite and > 0, got {self.contract_multiplier!r}")
        expected_notional = self.quantity * self.price * self.contract_multiplier
        if self.gross_notional != expected_notional:
            raise ExecutionFillError(f"ExecutionFill.gross_notional ({self.gross_notional}) does not equal quantity*price*contract_multiplier ({expected_notional})")
        for cost_field_name, cost_value in (("commission", self.commission), ("spread_component", self.spread_component), ("slippage_component", self.slippage_component)):
            if not cost_value.is_finite() or cost_value < 0:
                raise ExecutionFillError(f"ExecutionFill.{cost_field_name} must be finite and >= 0, got {cost_value!r}")
        _require_tz_aware(self.broker_timestamp, field_name="ExecutionFill.broker_timestamp")
        _require_tz_aware(self.received_event_time, field_name="ExecutionFill.received_event_time")
        if self.fill_sequence < 0:
            raise ExecutionFillError(f"ExecutionFill.fill_sequence must be >= 0, got {self.fill_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "execution_fill_id": self.execution_fill_id, "execution_session_id": self.execution_session_id, "execution_order_id": self.execution_order_id,
            "execution_intent_id": self.execution_intent_id, "broker_event_id": self.broker_event_id, "broker_fill_id": self.broker_fill_id,
            "broker_order_id": self.broker_order_id, "client_order_id": self.client_order_id, "instrument_id": self.instrument_id, "side": self.side.value,
            "quantity": decimal_to_json(self.quantity), "price": decimal_to_json(self.price), "contract_multiplier": decimal_to_json(self.contract_multiplier),
            "gross_notional": decimal_to_json(self.gross_notional), "commission": decimal_to_json(self.commission),
            "spread_component": decimal_to_json(self.spread_component), "slippage_component": decimal_to_json(self.slippage_component),
            "broker_timestamp": _serialize_timestamp(self.broker_timestamp, field_name="broker_timestamp"),
            "received_event_time": _serialize_timestamp(self.received_event_time, field_name="received_event_time"), "fill_sequence": self.fill_sequence,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionFill:
        return cls(
            execution_fill_id=str(raw["execution_fill_id"]), execution_session_id=str(raw["execution_session_id"]),
            execution_order_id=str(raw["execution_order_id"]), execution_intent_id=str(raw["execution_intent_id"]), broker_event_id=str(raw["broker_event_id"]),
            broker_fill_id=str(raw["broker_fill_id"]), broker_order_id=str(raw["broker_order_id"]), client_order_id=str(raw["client_order_id"]),
            instrument_id=str(raw["instrument_id"]), side=OrderSide(raw["side"]), quantity=parse_decimal(raw["quantity"], field_name="quantity"),
            price=parse_decimal(raw["price"], field_name="price"), contract_multiplier=parse_decimal(raw["contract_multiplier"], field_name="contract_multiplier"),
            gross_notional=parse_decimal(raw["gross_notional"], field_name="gross_notional"), commission=parse_decimal(raw["commission"], field_name="commission"),
            spread_component=parse_decimal(raw["spread_component"], field_name="spread_component"),
            slippage_component=parse_decimal(raw["slippage_component"], field_name="slippage_component"),
            broker_timestamp=_deserialize_timestamp(raw["broker_timestamp"], field_name="broker_timestamp"),
            received_event_time=_deserialize_timestamp(raw["received_event_time"], field_name="received_event_time"), fill_sequence=int(str(raw["fill_sequence"])),
        )


def create_execution_fill(
    *, execution_session_id: str, execution_order_id: str, execution_intent_id: str, broker_event: BrokerEvent, client_order_id: str, instrument_id: str,
    side: OrderSide, quantity: Decimal, price: Decimal, contract_multiplier: Decimal, commission: Decimal, spread_component: Decimal,
    slippage_component: Decimal, fill_sequence: int,
) -> ExecutionFill:
    """Stable fill identity (Section 10): keyed off the AUTHORITATIVE
    `broker_fill_id` combined with adapter/session ownership
    (`execution_session_id`, `execution_order_id`) -- two deliveries of
    the SAME underlying broker fill (e.g. a duplicated `BrokerEvent`)
    always recompute the SAME `execution_fill_id` and are therefore
    idempotently absorbed by `persistence.py`'s ledger append, never
    double-counted. Two DISTINCT legitimate partial fills always differ
    in at least `broker_fill_id` (the dummy broker never reuses one) or
    `fill_sequence`, so they can never collide."""
    if broker_event.broker_fill_id is None:
        raise ExecutionFillError(f"BrokerEvent {broker_event.broker_event_id!r} has no broker_fill_id -- cannot construct an ExecutionFill from it")
    gross_notional = quantity * price * contract_multiplier
    identity_payload = {
        "execution_session_id": execution_session_id, "execution_order_id": execution_order_id, "broker_fill_id": broker_event.broker_fill_id,
        "fill_sequence": fill_sequence,
    }
    execution_fill_id = compute_content_id(EXECUTION_FILL_KIND, identity_payload)
    return ExecutionFill(
        execution_fill_id=execution_fill_id, execution_session_id=execution_session_id, execution_order_id=execution_order_id,
        execution_intent_id=execution_intent_id, broker_event_id=broker_event.broker_event_id, broker_fill_id=broker_event.broker_fill_id,
        broker_order_id=(broker_event.broker_order_id or ""), client_order_id=client_order_id, instrument_id=instrument_id, side=side, quantity=quantity,
        price=price, contract_multiplier=contract_multiplier, gross_notional=gross_notional, commission=commission, spread_component=spread_component,
        slippage_component=slippage_component, broker_timestamp=broker_event.broker_timestamp, received_event_time=broker_event.received_event_time,
        fill_sequence=fill_sequence,
    )


# --------------------------------------------------------------------------
# ExecutionOrder aggregate (Section 9)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionOrder:
    execution_order_id: str
    execution_session_id: str
    execution_intent_id: str

    client_order_id: str
    broker_order_id: str | None

    instrument_id: str
    side: OrderSide
    order_type: OrderTypeKind
    time_in_force: TimeInForceKind

    original_quantity: Decimal
    current_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal

    limit_price: Decimal | None
    stop_price: Decimal | None
    average_fill_price: Decimal | None

    reduce_only: bool
    current_state: ExecutionOrderState

    last_broker_sequence: int | None
    last_broker_event_id: str | None

    created_event_time: datetime
    last_updated_event_time: datetime

    def __post_init__(self) -> None:
        for field_name, value in (
            ("execution_order_id", self.execution_order_id), ("execution_session_id", self.execution_session_id),
            ("execution_intent_id", self.execution_intent_id),
        ):
            if not is_valid_sha256_hex(value):
                raise ExecutionOrderStateError(f"ExecutionOrder.{field_name} must be a valid sha256 hex digest, got {value!r}")
        if self.original_quantity <= 0 or self.current_quantity <= 0:
            raise ExecutionOrderStateError("ExecutionOrder.original_quantity/current_quantity must be > 0")
        if self.filled_quantity < 0:
            raise ExecutionOrderStateError(f"ExecutionOrder.filled_quantity must be >= 0, got {self.filled_quantity!r}")
        if self.remaining_quantity < 0:
            raise ExecutionOrderStateError(f"ExecutionOrder.remaining_quantity must be >= 0 (filled_quantity {self.filled_quantity!r} exceeds current_quantity {self.current_quantity!r})")
        if self.filled_quantity + self.remaining_quantity != self.current_quantity:
            raise ExecutionOrderStateError(
                f"ExecutionOrder: filled_quantity ({self.filled_quantity}) + remaining_quantity ({self.remaining_quantity}) != current_quantity ({self.current_quantity})"
            )
        if self.time_in_force is TimeInForceKind.FOK and self.current_state is ExecutionOrderState.PARTIALLY_FILLED:
            raise ExecutionOrderStateError("ExecutionOrder: a FOK order must never be observed in PARTIALLY_FILLED state")
        _require_tz_aware(self.created_event_time, field_name="ExecutionOrder.created_event_time")
        _require_tz_aware(self.last_updated_event_time, field_name="ExecutionOrder.last_updated_event_time")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "execution_order_id": self.execution_order_id, "execution_session_id": self.execution_session_id, "execution_intent_id": self.execution_intent_id,
            "client_order_id": self.client_order_id, "broker_order_id": self.broker_order_id, "instrument_id": self.instrument_id, "side": self.side.value,
            "order_type": self.order_type.value, "time_in_force": self.time_in_force.value, "original_quantity": decimal_to_json(self.original_quantity),
            "current_quantity": decimal_to_json(self.current_quantity), "filled_quantity": decimal_to_json(self.filled_quantity),
            "remaining_quantity": decimal_to_json(self.remaining_quantity), "limit_price": (None if self.limit_price is None else decimal_to_json(self.limit_price)),
            "stop_price": (None if self.stop_price is None else decimal_to_json(self.stop_price)),
            "average_fill_price": (None if self.average_fill_price is None else decimal_to_json(self.average_fill_price)), "reduce_only": self.reduce_only,
            "current_state": self.current_state.value, "last_broker_sequence": self.last_broker_sequence, "last_broker_event_id": self.last_broker_event_id,
            "created_event_time": _serialize_timestamp(self.created_event_time, field_name="created_event_time"),
            "last_updated_event_time": _serialize_timestamp(self.last_updated_event_time, field_name="last_updated_event_time"),
        }


def reconstruct_execution_order(
    *, submit_command: SubmitOrderCommand, state_events: list[ExecutionOrderStateEvent], broker_events: list[BrokerEvent], fills: list[ExecutionFill],
    current_quantity: Decimal | None = None,
) -> ExecutionOrder:
    """Rebuilds one `ExecutionOrder` snapshot from durable evidence only.
    `state_events`/`broker_events`/`fills` must already be filtered to
    this order and sorted by sequence; `current_quantity` should be
    supplied by the caller as the quantity implied by the latest ACCEPTED
    replace (if any), else defaults to `submit_command.quantity`."""
    execution_order_id = compute_execution_order_id(submit_command)
    current_state = resolve_execution_order_state(execution_order_id, state_events)
    resolved_current_quantity = submit_command.quantity if current_quantity is None else current_quantity

    broker_order_id: str | None = None
    last_broker_sequence: int | None = None
    last_broker_event_id: str | None = None
    for be in broker_events:
        if be.broker_order_id is not None:
            broker_order_id = be.broker_order_id
        last_broker_sequence = be.broker_sequence
        last_broker_event_id = be.broker_event_id

    filled_quantity = sum((f.quantity for f in fills), Decimal(0))
    remaining_quantity = resolved_current_quantity - filled_quantity
    average_fill_price: Decimal | None = None
    if fills:
        notional = sum((f.price * f.quantity for f in fills), Decimal(0))
        average_fill_price = notional / filled_quantity

    last_updated = submit_command.event_time
    for ts in (*(e.event_time for e in state_events), *(be.received_event_time for be in broker_events), *(f.received_event_time for f in fills)):
        if ts > last_updated:
            last_updated = ts

    return ExecutionOrder(
        execution_order_id=execution_order_id, execution_session_id=submit_command.execution_session_id, execution_intent_id=submit_command.execution_intent_id,
        client_order_id=submit_command.client_order_id, broker_order_id=broker_order_id, instrument_id=submit_command.instrument_id, side=submit_command.side,
        order_type=submit_command.order_type, time_in_force=submit_command.time_in_force, original_quantity=submit_command.quantity,
        current_quantity=resolved_current_quantity, filled_quantity=filled_quantity, remaining_quantity=remaining_quantity, limit_price=submit_command.limit_price,
        stop_price=submit_command.stop_price, average_fill_price=average_fill_price, reduce_only=submit_command.reduce_only, current_state=current_state,
        last_broker_sequence=last_broker_sequence, last_broker_event_id=last_broker_event_id, created_event_time=submit_command.event_time,
        last_updated_event_time=last_updated,
    )


__all__ = [
    "EXECUTION_FILL_KIND",
    "EXECUTION_ORDER_ID_KIND",
    "ExecutionFill",
    "ExecutionOrder",
    "compute_execution_order_id",
    "create_execution_fill",
    "reconstruct_execution_order",
]
