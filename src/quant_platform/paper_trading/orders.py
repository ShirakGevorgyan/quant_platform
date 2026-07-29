"""Order model and state machine (Milestone 7, Section 8). `OrderRequest`
is immutable and content-addressed exactly like a `MarketEvent`/
`StrategyDecision` -- its quantity/price/side/type never change after
creation. State PROGRESS (CREATED -> VALIDATED -> ... -> FILLED) is
event-sourced: each transition is its own immutable, content-addressed
`OrderStateEvent`, and an order's CURRENT state is always DERIVED by
replaying its events via `resolve_order_state`, never stored as a mutable
field anywhere -- this is what lets `verification.py` (Section 26)
reconstruct order state purely from the ledger, independent of whatever
the forward run's own in-memory bookkeeping claimed.

Quantization against `InstrumentSpec.quantity_step`/exposure limits/
duplicate-`client_order_id` detection/session-acceptance-state checks all
need context `OrderRequest` itself does not carry (the instrument spec, the
session's other orders) -- those are `orders.py`'s validation helpers
(usable by `order_policy.py`, not `OrderRequest.__post_init__`), never
silently skipped, just layered where the necessary context actually
lives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import OrderStateError, OrderValidationError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import (
    TERMINAL_ORDER_STATES,
    OrderSide,
    OrderState,
    OrderTypeKind,
    PositionIntentKind,
    RejectReasonKind,
    TimeInForceKind,
)

ORDER_REQUEST_KIND = "order_request"
ORDER_STATE_EVENT_KIND = "order_state_event"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise OrderValidationError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise OrderValidationError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise OrderValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise OrderValidationError(f"{field_name}: {exc}") from exc


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise OrderValidationError(f"{field_name} must not be empty")


# --------------------------------------------------------------------------
# Order state machine (Section 8)
# --------------------------------------------------------------------------
_LEGAL_ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.VALIDATED, OrderState.REJECTED}),
    OrderState.VALIDATED: frozenset({OrderState.ACCEPTED, OrderState.REJECTED}),
    OrderState.ACCEPTED: frozenset({OrderState.WORKING, OrderState.REJECTED}),
    OrderState.WORKING: frozenset({
        OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_REQUESTED, OrderState.CANCELLED, OrderState.EXPIRED,
    }),
    OrderState.PARTIALLY_FILLED: frozenset({
        OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_REQUESTED, OrderState.CANCELLED, OrderState.EXPIRED,
    }),
    # A cancel request may lose a race against an in-flight fill -- both
    # FILLED and PARTIALLY_FILLED remain legal targets from CANCEL_REQUESTED.
    OrderState.CANCEL_REQUESTED: frozenset({OrderState.CANCELLED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.EXPIRED}),
    OrderState.REJECTED: frozenset(),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


def is_legal_order_transition(current: OrderState, target: OrderState) -> bool:
    return target in _LEGAL_ORDER_TRANSITIONS[current]


# --------------------------------------------------------------------------
# OrderRequest
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OrderRequest:
    order_id: str
    client_order_id: str
    session_id: str
    strategy_decision_id: str
    instrument: str
    side: OrderSide
    order_type: OrderTypeKind
    quantity: float
    limit_price: float | None
    stop_price: float | None
    time_in_force: TimeInForceKind
    create_time: datetime
    submit_time: datetime
    reduce_only: bool
    position_intent: PositionIntentKind

    def __post_init__(self) -> None:
        for field_name, value in (("client_order_id", self.client_order_id), ("session_id", self.session_id), ("instrument", self.instrument)):
            _require_non_empty(value, field_name=f"OrderRequest.{field_name}")
        if not is_valid_sha256_hex(self.strategy_decision_id):
            raise OrderValidationError(f"OrderRequest.strategy_decision_id must be a valid sha256 hex digest, got {self.strategy_decision_id!r}")
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise OrderValidationError(f"OrderRequest.quantity must be finite and > 0, got {self.quantity!r}")
        _require_tz_aware(self.create_time, field_name="OrderRequest.create_time")
        _require_tz_aware(self.submit_time, field_name="OrderRequest.submit_time")
        if self.submit_time < self.create_time:
            raise OrderValidationError(f"OrderRequest.submit_time ({self.submit_time}) must be >= create_time ({self.create_time})")

        if self.order_type is OrderTypeKind.MARKET:
            if self.limit_price is not None or self.stop_price is not None:
                raise OrderValidationError("OrderRequest: limit_price/stop_price must be None for a MARKET order")
        elif self.order_type is OrderTypeKind.LIMIT:
            if self.limit_price is None:
                raise OrderValidationError("OrderRequest.limit_price is required for a LIMIT order")
            if not math.isfinite(self.limit_price) or self.limit_price <= 0.0:
                raise OrderValidationError(f"OrderRequest.limit_price must be finite and > 0, got {self.limit_price!r}")
            if self.stop_price is not None:
                raise OrderValidationError("OrderRequest.stop_price must be None for a LIMIT order")
        elif self.order_type is OrderTypeKind.STOP:
            if self.stop_price is None:
                raise OrderValidationError("OrderRequest.stop_price is required for a STOP order")
            if not math.isfinite(self.stop_price) or self.stop_price <= 0.0:
                raise OrderValidationError(f"OrderRequest.stop_price must be finite and > 0, got {self.stop_price!r}")
            if self.limit_price is not None:
                raise OrderValidationError("OrderRequest.limit_price must be None for a STOP order")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id, "client_order_id": self.client_order_id, "session_id": self.session_id,
            "strategy_decision_id": self.strategy_decision_id, "instrument": self.instrument, "side": self.side.value,
            "order_type": self.order_type.value, "quantity": self.quantity, "limit_price": self.limit_price, "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value, "create_time": _serialize_timestamp(self.create_time, field_name="create_time"),
            "submit_time": _serialize_timestamp(self.submit_time, field_name="submit_time"), "reduce_only": self.reduce_only,
            "position_intent": self.position_intent.value,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["order_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OrderRequest:
        return cls(
            order_id=str(raw["order_id"]), client_order_id=str(raw["client_order_id"]), session_id=str(raw["session_id"]),
            strategy_decision_id=str(raw["strategy_decision_id"]), instrument=str(raw["instrument"]), side=OrderSide(raw["side"]),
            order_type=OrderTypeKind(raw["order_type"]), quantity=float(str(raw["quantity"])),
            limit_price=(None if raw.get("limit_price") is None else float(str(raw["limit_price"]))),
            stop_price=(None if raw.get("stop_price") is None else float(str(raw["stop_price"]))), time_in_force=TimeInForceKind(raw["time_in_force"]),
            create_time=_deserialize_timestamp(raw["create_time"], field_name="create_time"),
            submit_time=_deserialize_timestamp(raw["submit_time"], field_name="submit_time"), reduce_only=bool(raw["reduce_only"]),
            position_intent=PositionIntentKind(raw["position_intent"]),
        )


def create_order_request(
    *, client_order_id: str, session_id: str, strategy_decision_id: str, instrument: str, side: OrderSide, order_type: OrderTypeKind,
    quantity: float, time_in_force: TimeInForceKind, create_time: datetime, submit_time: datetime, reduce_only: bool,
    position_intent: PositionIntentKind, limit_price: float | None = None, stop_price: float | None = None,
) -> OrderRequest:
    """The only supported way to mint a fresh `OrderRequest` -- computes
    its deterministic `order_id` from every other field first."""
    provisional = OrderRequest(
        order_id="0" * 64, client_order_id=client_order_id, session_id=session_id, strategy_decision_id=strategy_decision_id,
        instrument=instrument, side=side, order_type=order_type, quantity=quantity, limit_price=limit_price, stop_price=stop_price,
        time_in_force=time_in_force, create_time=create_time, submit_time=submit_time, reduce_only=reduce_only, position_intent=position_intent,
    )
    order_id = compute_content_id(ORDER_REQUEST_KIND, provisional.to_identity_payload())
    return OrderRequest(
        order_id=order_id, client_order_id=client_order_id, session_id=session_id, strategy_decision_id=strategy_decision_id,
        instrument=instrument, side=side, order_type=order_type, quantity=quantity, limit_price=limit_price, stop_price=stop_price,
        time_in_force=time_in_force, create_time=create_time, submit_time=submit_time, reduce_only=reduce_only, position_intent=position_intent,
    )


# --------------------------------------------------------------------------
# OrderStateEvent -- one event-sourced ledger entry per transition
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OrderStateEvent:
    event_id: str
    order_id: str
    session_id: str
    from_state: OrderState
    to_state: OrderState
    event_time: datetime
    sequence: int
    reason_code: RejectReasonKind | None
    source_market_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.order_id, field_name="OrderStateEvent.order_id")
        _require_non_empty(self.session_id, field_name="OrderStateEvent.session_id")
        _require_tz_aware(self.event_time, field_name="OrderStateEvent.event_time")
        if self.sequence < 0:
            raise OrderStateError(f"OrderStateEvent.sequence must be >= 0, got {self.sequence}")
        if not is_legal_order_transition(self.from_state, self.to_state):
            raise OrderStateError(f"Illegal order state transition {self.from_state.value!r} -> {self.to_state.value!r} for order {self.order_id!r}")
        if self.to_state in (OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED) and self.reason_code is None:
            raise OrderStateError(f"OrderStateEvent.reason_code is required when transitioning to {self.to_state.value!r}")
        if self.to_state not in (OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED) and self.reason_code is not None:
            raise OrderStateError(f"OrderStateEvent.reason_code must be None when transitioning to {self.to_state.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id, "order_id": self.order_id, "session_id": self.session_id, "from_state": self.from_state.value,
            "to_state": self.to_state.value, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "reason_code": (None if self.reason_code is None else self.reason_code.value), "source_market_event_identity": self.source_market_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OrderStateEvent:
        return cls(
            event_id=str(raw["event_id"]), order_id=str(raw["order_id"]), session_id=str(raw["session_id"]),
            from_state=OrderState(raw["from_state"]), to_state=OrderState(raw["to_state"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            reason_code=(None if raw.get("reason_code") is None else RejectReasonKind(raw["reason_code"])),
            source_market_event_identity=(None if raw.get("source_market_event_identity") is None else str(raw["source_market_event_identity"])),
        )


def create_order_state_event(
    *, order_id: str, session_id: str, from_state: OrderState, to_state: OrderState, event_time: datetime, sequence: int,
    reason_code: RejectReasonKind | None = None, source_market_event_identity: str | None = None,
) -> OrderStateEvent:
    provisional = OrderStateEvent(
        event_id="0" * 64, order_id=order_id, session_id=session_id, from_state=from_state, to_state=to_state, event_time=event_time,
        sequence=sequence, reason_code=reason_code, source_market_event_identity=source_market_event_identity,
    )
    event_id = compute_content_id(ORDER_STATE_EVENT_KIND, provisional.to_identity_payload())
    return OrderStateEvent(
        event_id=event_id, order_id=order_id, session_id=session_id, from_state=from_state, to_state=to_state, event_time=event_time,
        sequence=sequence, reason_code=reason_code, source_market_event_identity=source_market_event_identity,
    )


def resolve_order_state(order_id: str, events: list[OrderStateEvent]) -> OrderState:
    """Pure event-sourced state derivation -- replays `events` (assumed
    already in ledger/sequence order) from the implicit initial `CREATED`
    state and returns the final state. Never trusts a persisted "current
    state" field anywhere; this IS the only correct way to know an order's
    state, used identically by the forward runner's own bookkeeping and by
    `verification.py`'s independent reconstruction."""
    current = OrderState.CREATED
    for event in events:
        if event.order_id != order_id:
            raise OrderStateError(f"resolve_order_state({order_id!r}): event {event.event_id!r} belongs to a different order {event.order_id!r}")
        if event.from_state is not current:
            raise OrderStateError(f"resolve_order_state({order_id!r}): event {event.event_id!r} expects from_state={event.from_state.value!r} but current state is {current.value!r}")
        if not is_legal_order_transition(current, event.to_state):
            raise OrderStateError(f"resolve_order_state({order_id!r}): illegal transition {current.value!r} -> {event.to_state.value!r}")
        current = event.to_state
    return current


def is_order_in_terminal_state(order_id: str, events: list[OrderStateEvent]) -> bool:
    return resolve_order_state(order_id, events) in TERMINAL_ORDER_STATES


__all__ = [
    "ORDER_REQUEST_KIND",
    "ORDER_STATE_EVENT_KIND",
    "OrderRequest",
    "OrderStateEvent",
    "create_order_request",
    "create_order_state_event",
    "is_legal_order_transition",
    "is_order_in_terminal_state",
    "resolve_order_state",
]
