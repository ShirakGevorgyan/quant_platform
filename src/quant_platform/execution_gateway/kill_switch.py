"""Execution kill switch and centralized dispatch gate (Milestone 8,
Section 22). Event-sourced exactly like `paper_trading.risk`'s kill
switch and `state_machine.py`'s own order-state pattern:
`resolve_execution_kill_switch_state` replays a
`ExecutionKillSwitchTransitionEvent` sequence from the implicit initial
`ACTIVE` state.

`authorize_dispatch` is THE centralized dispatch gate (Section 22:
"Every dispatch path must pass through one centralized dispatch gate. No
command may bypass the kill switch.") -- `runner.py` calls this
immediately before every `dispatcher.dispatch_command` call, never
after. A safety cancel or an authorized reduce-only submit is NEVER
blocked by the same new-exposure rule it exists to mitigate (Section 22's
own explicit instruction) -- `models.REDUCE_ONLY_PERMITTING_KILL_SWITCH_
STATES` is deliberately more permissive than `NEW_EXPOSURE_PERMITTING_
KILL_SWITCH_STATES`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import ExecutionHaltError
from quant_platform.execution_gateway.commands import (
    CancelOrderCommand,
    HeartbeatCommand,
    QueryAccountCommand,
    QueryOpenOrdersCommand,
    QueryOrderCommand,
    QueryPositionsCommand,
    ReplaceOrderCommand,
    SubmitOrderCommand,
)
from quant_platform.execution_gateway.identity import compute_content_id, is_valid_sha256_hex
from quant_platform.execution_gateway.models import (
    NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES,
    QUERY_PERMITTING_KILL_SWITCH_STATES,
    REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES,
    ExecutionKillSwitchState,
    is_legal_execution_kill_switch_transition,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp

EXECUTION_KILL_SWITCH_TRANSITION_EVENT_KIND = "execution_kill_switch_transition_event"


class ExecutionKillSwitchTriggerKind(Enum):
    OPERATOR_REQUEST = "operator_request"
    UNRESOLVED_UNKNOWN_OPERATIONS = "unresolved_unknown_operations"
    BROKER_SEQUENCE_CONFLICT = "broker_sequence_conflict"
    BLOCKING_RECONCILIATION_MISMATCH = "blocking_reconciliation_mismatch"
    ADAPTER_STALE = "adapter_stale"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    REPEATED_ADAPTER_FAILURE = "repeated_adapter_failure"
    DUPLICATE_FILL_PAYLOAD_CONFLICT = "duplicate_fill_payload_conflict"
    SESSION_OWNERSHIP_MISMATCH = "session_ownership_mismatch"
    SOURCE_ELIGIBILITY_FAILURE_ON_RESUME = "source_eligibility_failure_on_resume"
    ILLEGAL_STATE_TRANSITION = "illegal_state_transition"
    LEDGER_SEMANTIC_CORRUPTION = "ledger_semantic_corruption"
    RECOVERED = "recovered"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionHaltError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise ExecutionHaltError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ExecutionHaltError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise ExecutionHaltError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExecutionKillSwitchTransitionEvent:
    event_id: str
    execution_session_id: str
    from_state: ExecutionKillSwitchState
    to_state: ExecutionKillSwitchState
    trigger: ExecutionKillSwitchTriggerKind
    event_time: datetime
    sequence: int
    detail: str

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.execution_session_id):
            raise ExecutionHaltError(f"ExecutionKillSwitchTransitionEvent.execution_session_id must be a valid sha256 hex digest, got {self.execution_session_id!r}")
        _require_tz_aware(self.event_time, field_name="ExecutionKillSwitchTransitionEvent.event_time")
        if self.sequence < 0:
            raise ExecutionHaltError(f"ExecutionKillSwitchTransitionEvent.sequence must be >= 0, got {self.sequence}")
        if not is_legal_execution_kill_switch_transition(self.from_state, self.to_state):
            raise ExecutionHaltError(f"Illegal execution kill switch transition {self.from_state.value!r} -> {self.to_state.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id, "execution_session_id": self.execution_session_id, "from_state": self.from_state.value, "to_state": self.to_state.value,
            "trigger": self.trigger.value, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "detail": self.detail,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionKillSwitchTransitionEvent:
        return cls(
            event_id=str(raw["event_id"]), execution_session_id=str(raw["execution_session_id"]), from_state=ExecutionKillSwitchState(raw["from_state"]),
            to_state=ExecutionKillSwitchState(raw["to_state"]), trigger=ExecutionKillSwitchTriggerKind(raw["trigger"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])), detail=str(raw["detail"]),
        )


def create_execution_kill_switch_transition_event(
    *, execution_session_id: str, from_state: ExecutionKillSwitchState, to_state: ExecutionKillSwitchState, trigger: ExecutionKillSwitchTriggerKind,
    event_time: datetime, sequence: int, detail: str,
) -> ExecutionKillSwitchTransitionEvent:
    provisional = ExecutionKillSwitchTransitionEvent(
        event_id="0" * 64, execution_session_id=execution_session_id, from_state=from_state, to_state=to_state, trigger=trigger, event_time=event_time,
        sequence=sequence, detail=detail,
    )
    event_id = compute_content_id(EXECUTION_KILL_SWITCH_TRANSITION_EVENT_KIND, provisional.to_identity_payload())
    return ExecutionKillSwitchTransitionEvent(
        event_id=event_id, execution_session_id=execution_session_id, from_state=from_state, to_state=to_state, trigger=trigger, event_time=event_time,
        sequence=sequence, detail=detail,
    )


def resolve_execution_kill_switch_state(events: list[ExecutionKillSwitchTransitionEvent]) -> ExecutionKillSwitchState:
    current = ExecutionKillSwitchState.ACTIVE
    for event in events:
        if event.from_state is not current:
            raise ExecutionHaltError(f"resolve_execution_kill_switch_state: event {event.event_id!r} expects from_state={event.from_state.value!r} but current state is {current.value!r}")
        if not is_legal_execution_kill_switch_transition(current, event.to_state):
            raise ExecutionHaltError(f"resolve_execution_kill_switch_state: illegal transition {current.value!r} -> {event.to_state.value!r}")
        current = event.to_state
    return current


# --------------------------------------------------------------------------
# Centralized dispatch gate (Section 22)
# --------------------------------------------------------------------------
def authorize_dispatch(kill_switch_state: ExecutionKillSwitchState, command: SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand | QueryOrderCommand | QueryOpenOrdersCommand | QueryPositionsCommand | QueryAccountCommand | HeartbeatCommand) -> None:
    """Raises `ExecutionHaltError` if `command` is not permitted while
    the kill switch is at `kill_switch_state`. EVERY command in this
    package must pass through this function before
    `dispatcher.dispatch_command` -- there is no other path to the
    adapter."""
    if isinstance(command, (QueryOrderCommand, QueryOpenOrdersCommand, QueryPositionsCommand, QueryAccountCommand, HeartbeatCommand)):
        if kill_switch_state not in QUERY_PERMITTING_KILL_SWITCH_STATES:
            raise ExecutionHaltError(f"query/heartbeat commands are not permitted while the kill switch is {kill_switch_state.value!r}")
        return
    if isinstance(command, CancelOrderCommand):
        if kill_switch_state not in REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES:
            raise ExecutionHaltError(f"cancel is not permitted while the kill switch is {kill_switch_state.value!r} -- even a safety cancel requires at least DEGRADED/HALTING/ACTIVE")
        return
    if isinstance(command, SubmitOrderCommand):
        is_risk_reducing = command.reduce_only
        allowed_states = REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES if is_risk_reducing else NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES
        if kill_switch_state not in allowed_states:
            kind = "reduce_only" if is_risk_reducing else "new-exposure"
            raise ExecutionHaltError(f"{kind} submit is not permitted while the kill switch is {kill_switch_state.value!r}")
        return
    if isinstance(command, ReplaceOrderCommand):
        # A replace can both increase and decrease exposure depending on
        # its terms -- fail closed and require the strict ACTIVE-only
        # allow-list, exactly like a new-exposure submit, unless the
        # caller has independently established the replacement is
        # risk-reducing (that determination belongs to `runner.py`,
        # which has the order's current quantity to compare against).
        if kill_switch_state not in NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES:
            raise ExecutionHaltError(f"replace is not permitted while the kill switch is {kill_switch_state.value!r}")
        return
    raise ExecutionHaltError(f"authorize_dispatch: unsupported command type {type(command).__name__}")


__all__ = [
    "EXECUTION_KILL_SWITCH_TRANSITION_EVENT_KIND",
    "ExecutionKillSwitchTransitionEvent",
    "ExecutionKillSwitchTriggerKind",
    "authorize_dispatch",
    "create_execution_kill_switch_transition_event",
    "resolve_execution_kill_switch_state",
]
