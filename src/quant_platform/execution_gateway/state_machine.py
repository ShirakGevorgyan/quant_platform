"""Execution-order state-transition events (Milestone 8, Section 8).
Mirrors `paper_trading.orders`'s `OrderStateEvent`/`resolve_order_state`
pattern exactly: an execution order's CURRENT state is never a mutable
field anywhere -- it is always DERIVED by replaying its
`ExecutionOrderStateEvent` sequence via `resolve_execution_order_state`.
This is what lets `verification.py` (Section 28) reconstruct order state
purely from the append-only ledger, independent of whatever the forward
dispatcher's own in-memory bookkeeping claimed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import ExecutionOrderStateError
from quant_platform.execution_gateway.identity import compute_content_id, is_valid_sha256_hex
from quant_platform.execution_gateway.models import (
    TERMINAL_EXECUTION_ORDER_STATES,
    ExecutionOrderState,
    is_blocking_execution_order_state,
    is_legal_execution_order_transition,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp

EXECUTION_ORDER_STATE_EVENT_KIND = "execution_order_state_event"

_REASON_REQUIRED_STATES: frozenset[ExecutionOrderState] = frozenset({
    ExecutionOrderState.REJECTED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED, ExecutionOrderState.FAILED, ExecutionOrderState.UNKNOWN,
})


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionOrderStateError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise ExecutionOrderStateError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ExecutionOrderStateError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise ExecutionOrderStateError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExecutionOrderStateEvent:
    event_id: str
    execution_order_id: str
    execution_session_id: str
    from_state: ExecutionOrderState
    to_state: ExecutionOrderState
    event_time: datetime
    sequence: int
    reason_code: str | None
    source_broker_event_id: str | None

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.execution_order_id):
            raise ExecutionOrderStateError(f"ExecutionOrderStateEvent.execution_order_id must be a valid sha256 hex digest, got {self.execution_order_id!r}")
        if not is_valid_sha256_hex(self.execution_session_id):
            raise ExecutionOrderStateError(f"ExecutionOrderStateEvent.execution_session_id must be a valid sha256 hex digest, got {self.execution_session_id!r}")
        _require_tz_aware(self.event_time, field_name="ExecutionOrderStateEvent.event_time")
        if self.sequence < 0:
            raise ExecutionOrderStateError(f"ExecutionOrderStateEvent.sequence must be >= 0, got {self.sequence}")
        if not is_legal_execution_order_transition(self.from_state, self.to_state):
            raise ExecutionOrderStateError(f"Illegal execution order state transition {self.from_state.value!r} -> {self.to_state.value!r} for order {self.execution_order_id!r}")
        if self.to_state in _REASON_REQUIRED_STATES and not self.reason_code:
            raise ExecutionOrderStateError(f"ExecutionOrderStateEvent.reason_code is required when transitioning to {self.to_state.value!r}")
        if self.to_state not in _REASON_REQUIRED_STATES and self.reason_code is not None:
            raise ExecutionOrderStateError(f"ExecutionOrderStateEvent.reason_code must be None when transitioning to {self.to_state.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id, "execution_order_id": self.execution_order_id, "execution_session_id": self.execution_session_id,
            "from_state": self.from_state.value, "to_state": self.to_state.value, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
            "sequence": self.sequence, "reason_code": self.reason_code, "source_broker_event_id": self.source_broker_event_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionOrderStateEvent:
        return cls(
            event_id=str(raw["event_id"]), execution_order_id=str(raw["execution_order_id"]), execution_session_id=str(raw["execution_session_id"]),
            from_state=ExecutionOrderState(raw["from_state"]), to_state=ExecutionOrderState(raw["to_state"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            reason_code=(None if raw.get("reason_code") is None else str(raw["reason_code"])),
            source_broker_event_id=(None if raw.get("source_broker_event_id") is None else str(raw["source_broker_event_id"])),
        )


def create_execution_order_state_event(
    *, execution_order_id: str, execution_session_id: str, from_state: ExecutionOrderState, to_state: ExecutionOrderState, event_time: datetime,
    sequence: int, reason_code: str | None = None, source_broker_event_id: str | None = None,
) -> ExecutionOrderStateEvent:
    provisional = ExecutionOrderStateEvent(
        event_id="0" * 64, execution_order_id=execution_order_id, execution_session_id=execution_session_id, from_state=from_state, to_state=to_state,
        event_time=event_time, sequence=sequence, reason_code=reason_code, source_broker_event_id=source_broker_event_id,
    )
    event_id = compute_content_id(EXECUTION_ORDER_STATE_EVENT_KIND, provisional.to_identity_payload())
    return ExecutionOrderStateEvent(
        event_id=event_id, execution_order_id=execution_order_id, execution_session_id=execution_session_id, from_state=from_state, to_state=to_state,
        event_time=event_time, sequence=sequence, reason_code=reason_code, source_broker_event_id=source_broker_event_id,
    )


def resolve_execution_order_state(execution_order_id: str, events: list[ExecutionOrderStateEvent]) -> ExecutionOrderState:
    """Pure event-sourced state derivation -- replays `events` (assumed
    already in ledger/sequence order) from the implicit initial `CREATED`
    state. Used identically by the forward dispatcher's own bookkeeping
    and by `verification.py`'s independent reconstruction."""
    current = ExecutionOrderState.CREATED
    for event in events:
        if event.execution_order_id != execution_order_id:
            raise ExecutionOrderStateError(f"resolve_execution_order_state({execution_order_id!r}): event {event.event_id!r} belongs to a different order {event.execution_order_id!r}")
        if event.from_state is not current:
            raise ExecutionOrderStateError(f"resolve_execution_order_state({execution_order_id!r}): event {event.event_id!r} expects from_state={event.from_state.value!r} but current state is {current.value!r}")
        if not is_legal_execution_order_transition(current, event.to_state):
            raise ExecutionOrderStateError(f"resolve_execution_order_state({execution_order_id!r}): illegal transition {current.value!r} -> {event.to_state.value!r}")
        current = event.to_state
    return current


def is_execution_order_in_terminal_state(execution_order_id: str, events: list[ExecutionOrderStateEvent]) -> bool:
    return resolve_execution_order_state(execution_order_id, events) in TERMINAL_EXECUTION_ORDER_STATES


def is_execution_order_blocking(execution_order_id: str, events: list[ExecutionOrderStateEvent]) -> bool:
    return is_blocking_execution_order_state(resolve_execution_order_state(execution_order_id, events))


__all__ = [
    "EXECUTION_ORDER_STATE_EVENT_KIND",
    "ExecutionOrderStateEvent",
    "create_execution_order_state_event",
    "is_execution_order_blocking",
    "is_execution_order_in_terminal_state",
    "resolve_execution_order_state",
]
