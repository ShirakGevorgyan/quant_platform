"""Shared vocabulary for `quant_platform.execution_gateway` (Milestone 8)
-- stage machines, enums, and small cross-cutting value types every other
module in this package imports. Mirrors `paper_trading.models`'s
identical role one layer down.

`OrderSide`/`OrderTypeKind`/`TimeInForceKind` are deliberately NOT
redefined here -- Section 2's own instruction ("may reuse ... existing
paper-trading models") and Section 5's required side/order-type/TIF
vocabulary (`BUY`/`SELL`, `MARKET`/`LIMIT`/`STOP`,
`DAY`/`GTC`/`IOC`/`FOK`) are byte-for-byte identical to
`paper_trading.models`'s own definitions; importing them directly avoids
a second, parallel closed vocabulary for the exact same concepts.

EXECUTION_MODE AND ADAPTER_KIND ARE SINGLE-MEMBER ENUMS ON PURPOSE:
`ExecutionMode` has exactly one member (`TEST_ONLY`) and `AdapterKind`
has exactly one member (`DETERMINISTIC_DUMMY`) -- there is no `LIVE`,
`MT5`, `FXPRO`, or `REAL_BROKER` value anywhere in this enum, so no such
value can ever be constructed, stored, or compared against, let alone
reached at runtime. A future MT5 adapter milestone would ADD a member to
`AdapterKind` (and update `is_supported_adapter_kind`'s allow-list) --
this milestone does not, and self-tests confirm it does not."""

from __future__ import annotations

from enum import Enum

# Re-exported for `execution_gateway` callers so they never need to reach
# into `paper_trading.models` directly for these three closed vocabularies.
from quant_platform.paper_trading.models import (
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)

__all__ = [
    "OrderSide",
    "OrderTypeKind",
    "TimeInForceKind",
]


# --------------------------------------------------------------------------
# Execution mode / adapter kind -- Section 4: "No enum value for LIVE, MT5,
# FXPRO, REAL_BROKER, or similar may exist."
# --------------------------------------------------------------------------
class ExecutionMode(Enum):
    TEST_ONLY = "test_only"


class AdapterKind(Enum):
    DETERMINISTIC_DUMMY = "deterministic_dummy"


class AuthorizationMode(Enum):
    TEST_ONLY_DUMMY_EXECUTION = "test_only_dummy_execution"


EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Command model (Section 7)
# --------------------------------------------------------------------------
class CommandType(Enum):
    SUBMIT_ORDER = "submit_order"
    CANCEL_ORDER = "cancel_order"
    REPLACE_ORDER = "replace_order"
    QUERY_ORDER = "query_order"
    QUERY_OPEN_ORDERS = "query_open_orders"
    QUERY_POSITIONS = "query_positions"
    QUERY_ACCOUNT = "query_account"
    HEARTBEAT = "heartbeat"


MUTATING_COMMAND_TYPES: frozenset[CommandType] = frozenset({CommandType.SUBMIT_ORDER, CommandType.CANCEL_ORDER, CommandType.REPLACE_ORDER})
"""Commands that can change broker-side order state -- as opposed to a
pure query/heartbeat, which never does. Used by `kill_switch.py`'s
dispatch gate to distinguish "new exposure or a mutation" from
"observation only", and by `commands.py` to decide which commands
require an `execution_intent_id`."""


# --------------------------------------------------------------------------
# Execution-order state machine (Section 8)
# --------------------------------------------------------------------------
class ExecutionOrderState(Enum):
    CREATED = "created"
    VALIDATED = "validated"
    DISPATCH_PENDING = "dispatch_pending"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    CANCEL_PENDING = "cancel_pending"
    REPLACE_PENDING = "replace_pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"
    FAILED = "failed"


TERMINAL_EXECUTION_ORDER_STATES: frozenset[ExecutionOrderState] = frozenset({
    ExecutionOrderState.FILLED, ExecutionOrderState.CANCELLED, ExecutionOrderState.REJECTED, ExecutionOrderState.EXPIRED, ExecutionOrderState.FAILED,
})
"""`UNKNOWN` is deliberately NOT terminal -- Section 8: "UNKNOWN is
non-terminal but blocking." It always has a legal path forward (see
`_LEGAL_EXECUTION_ORDER_TRANSITIONS[ExecutionOrderState.UNKNOWN]`) once
proof arrives via query/reconciliation, but a session may never reach
`COMPLETED` while any order remains in it (`manifests.py`'s completion
gate)."""

BLOCKING_EXECUTION_ORDER_STATES: frozenset[ExecutionOrderState] = frozenset({ExecutionOrderState.UNKNOWN})

_LEGAL_EXECUTION_ORDER_TRANSITIONS: dict[ExecutionOrderState, frozenset[ExecutionOrderState]] = {
    ExecutionOrderState.CREATED: frozenset({ExecutionOrderState.VALIDATED, ExecutionOrderState.REJECTED}),
    ExecutionOrderState.VALIDATED: frozenset({ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.REJECTED}),
    # DISPATCH_PENDING -> FAILED is reserved for a dispatch attempt that
    # fails with PROOF nothing reached the adapter (e.g. a capability
    # check raised synchronously before any adapter call was made) --
    # never used for "the adapter call itself may or may not have
    # landed", which is UNKNOWN's job.
    ExecutionOrderState.DISPATCH_PENDING: frozenset({ExecutionOrderState.DISPATCHED, ExecutionOrderState.UNKNOWN, ExecutionOrderState.FAILED}),
    ExecutionOrderState.DISPATCHED: frozenset({ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.REJECTED, ExecutionOrderState.UNKNOWN}),
    ExecutionOrderState.ACKNOWLEDGED: frozenset({
        ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED, ExecutionOrderState.CANCEL_PENDING, ExecutionOrderState.REPLACE_PENDING,
        ExecutionOrderState.EXPIRED, ExecutionOrderState.SUSPENDED,
    }),
    ExecutionOrderState.PARTIALLY_FILLED: frozenset({
        ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED, ExecutionOrderState.CANCEL_PENDING, ExecutionOrderState.EXPIRED,
        ExecutionOrderState.SUSPENDED,
    }),
    # A cancel may lose a race against an in-flight fill, or be rejected
    # outright (order reverts to its prior working state) -- mirrors
    # `paper_trading.orders`'s identical CANCEL_REQUESTED allowance.
    ExecutionOrderState.CANCEL_PENDING: frozenset({
        ExecutionOrderState.CANCELLED, ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED,
        ExecutionOrderState.EXPIRED, ExecutionOrderState.UNKNOWN,
    }),
    # A replace resolves back to ACKNOWLEDGED or PARTIALLY_FILLED --
    # whichever the order actually was immediately before the replace was
    # dispatched -- whether accepted (new terms) or rejected (prior terms
    # restored). The REPLACE_REJECTED fact itself is a broker-event/
    # ledger classification, not a distinct order state (Section 14).
    ExecutionOrderState.REPLACE_PENDING: frozenset({ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.UNKNOWN}),
    ExecutionOrderState.SUSPENDED: frozenset({
        ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED,
        ExecutionOrderState.FILLED,
    }),
    # UNKNOWN resolves once query/reconciliation supplies proof -- to
    # ANY outcome a legitimate broker response could have produced, or to
    # FAILED if reconciliation proves the ambiguity can never be resolved
    # (e.g. the broker itself is permanently gone). Never back to a
    # pre-dispatch state.
    ExecutionOrderState.UNKNOWN: frozenset({
        ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED, ExecutionOrderState.CANCELLED,
        ExecutionOrderState.REJECTED, ExecutionOrderState.EXPIRED, ExecutionOrderState.FAILED,
    }),
    ExecutionOrderState.FILLED: frozenset(),
    ExecutionOrderState.CANCELLED: frozenset(),
    ExecutionOrderState.REJECTED: frozenset(),
    ExecutionOrderState.EXPIRED: frozenset(),
    ExecutionOrderState.FAILED: frozenset(),
}


def is_legal_execution_order_transition(current: ExecutionOrderState, target: ExecutionOrderState) -> bool:
    return target in _LEGAL_EXECUTION_ORDER_TRANSITIONS[current]


def is_terminal_execution_order_state(state: ExecutionOrderState) -> bool:
    return state in TERMINAL_EXECUTION_ORDER_STATES


def is_blocking_execution_order_state(state: ExecutionOrderState) -> bool:
    return state in BLOCKING_EXECUTION_ORDER_STATES


# --------------------------------------------------------------------------
# Broker-event sequencing (Section 15)
# --------------------------------------------------------------------------
class SequencingPolicyKind(Enum):
    STRICT_SEQUENCE = "strict_sequence"
    TIMESTAMP_AND_ID = "timestamp_and_id"
    ARRIVAL_ORDER_ONLY = "arrival_order_only"
    """Lower-assurance generic policy. Deliberately not the dummy-broker
    default (Section 15: "must not be the dummy-broker default"); any
    verification report generated under this policy explicitly flags the
    reduced assurance (`verification.VerificationIndependence` /
    `SequencingPolicyKind.assurance_is_reduced`)."""

    @property
    def assurance_is_reduced(self) -> bool:
        return self is SequencingPolicyKind.ARRIVAL_ORDER_ONLY


# --------------------------------------------------------------------------
# Broker event model (Section 14)
# --------------------------------------------------------------------------
class BrokerEventType(Enum):
    ORDER_RECEIVED = "order_received"
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_REJECTED = "order_rejected"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"

    CANCEL_ACKNOWLEDGED = "cancel_acknowledged"
    CANCEL_REJECTED = "cancel_rejected"

    REPLACE_ACKNOWLEDGED = "replace_acknowledged"
    REPLACE_REJECTED = "replace_rejected"

    ORDER_EXPIRED = "order_expired"
    ORDER_SUSPENDED = "order_suspended"
    ORDER_STATUS_SNAPSHOT = "order_status_snapshot"

    BROKER_DISCONNECTED = "broker_disconnected"
    BROKER_RECONNECTED = "broker_reconnected"

    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_MISSED = "heartbeat_missed"


# --------------------------------------------------------------------------
# Dispatch transaction (Section 17)
# --------------------------------------------------------------------------
class DispatchStage(Enum):
    COMMAND_CREATED = "command_created"
    COMMAND_VALIDATED = "command_validated"
    COMMAND_DISPATCH_INTENT_RECORDED = "command_dispatch_intent_recorded"
    ADAPTER_CALL_STARTED = "adapter_call_started"
    COMMAND_DISPATCH_SUCCEEDED = "command_dispatch_succeeded"
    COMMAND_DISPATCH_REJECTED = "command_dispatch_rejected"
    COMMAND_DISPATCH_FAILED = "command_dispatch_failed"
    COMMAND_MARKED_UNKNOWN = "command_marked_unknown"


# --------------------------------------------------------------------------
# Append-only execution ledger (Section 18)
# --------------------------------------------------------------------------
class ExecutionLedgerEntryKind(Enum):
    EXECUTION_SESSION_CREATED = "execution_session_created"
    EXECUTION_SESSION_STARTED = "execution_session_started"

    SOURCE_ELIGIBILITY_VERIFIED = "source_eligibility_verified"

    EXECUTION_INTENT_ACCEPTED = "execution_intent_accepted"
    EXECUTION_INTENT_REJECTED = "execution_intent_rejected"

    COMMAND_CREATED = "command_created"
    COMMAND_VALIDATED = "command_validated"
    COMMAND_REJECTED = "command_rejected"
    COMMAND_DISPATCH_INTENT = "command_dispatch_intent"
    COMMAND_DISPATCH_SUCCEEDED = "command_dispatch_succeeded"
    COMMAND_DISPATCH_REJECTED = "command_dispatch_rejected"
    COMMAND_DISPATCH_FAILED = "command_dispatch_failed"
    COMMAND_MARKED_UNKNOWN = "command_marked_unknown"

    BROKER_EVENT_RECEIVED = "broker_event_received"
    BROKER_EVENT_DUPLICATE = "broker_event_duplicate"
    BROKER_EVENT_OUT_OF_ORDER = "broker_event_out_of_order"
    BROKER_EVENT_CONFLICT = "broker_event_conflict"

    ORDER_STATE_TRANSITION = "order_state_transition"
    EXECUTION_FILL_RECORDED = "execution_fill_recorded"

    RECONCILIATION_STARTED = "reconciliation_started"
    RECONCILIATION_COMPLETED = "reconciliation_completed"
    RECONCILIATION_FAILED = "reconciliation_failed"

    ADAPTER_HEALTH_CHANGED = "adapter_health_changed"
    HEARTBEAT_MISSED = "heartbeat_missed"
    HEARTBEAT_RESTORED = "heartbeat_restored"

    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    EXECUTION_SESSION_PAUSED = "execution_session_paused"
    EXECUTION_SESSION_RESUMED = "execution_session_resumed"
    EXECUTION_SESSION_HALTED = "execution_session_halted"
    EXECUTION_SESSION_COMPLETED = "execution_session_completed"
    EXECUTION_SESSION_FAILED = "execution_session_failed"
    EXECUTION_SESSION_TERMINATED = "execution_session_terminated"


# --------------------------------------------------------------------------
# Session manifest stage machine (Section 19)
# --------------------------------------------------------------------------
class ExecutionSessionStage(Enum):
    CREATED = "created"
    SPEC_VERIFIED = "spec_verified"
    SOURCE_ELIGIBILITY_VERIFIED = "source_eligibility_verified"
    ADAPTER_INITIALIZED = "adapter_initialized"
    RECOVERY_CHECKED = "recovery_checked"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RECONCILING = "reconciling"
    HALTING = "halting"
    HALTED = "halted"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


TERMINAL_EXECUTION_SESSION_STAGES: frozenset[ExecutionSessionStage] = frozenset({
    ExecutionSessionStage.COMPLETED, ExecutionSessionStage.FAILED, ExecutionSessionStage.TERMINATED,
})

_EXECUTION_SESSION_STAGE_ORDER: tuple[ExecutionSessionStage, ...] = (
    ExecutionSessionStage.CREATED, ExecutionSessionStage.SPEC_VERIFIED, ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED,
    ExecutionSessionStage.ADAPTER_INITIALIZED, ExecutionSessionStage.RECOVERY_CHECKED, ExecutionSessionStage.RUNNING,
    ExecutionSessionStage.RECONCILING, ExecutionSessionStage.VERIFYING, ExecutionSessionStage.COMPLETED,
)
_EXECUTION_SESSION_STAGE_INDEX: dict[ExecutionSessionStage, int] = {stage: i for i, stage in enumerate(_EXECUTION_SESSION_STAGE_ORDER)}

_LEGAL_EXECUTION_SESSION_TRANSITIONS: dict[ExecutionSessionStage, frozenset[ExecutionSessionStage]] = {
    ExecutionSessionStage.CREATED: frozenset({ExecutionSessionStage.SPEC_VERIFIED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.SPEC_VERIFIED: frozenset({ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED: frozenset({ExecutionSessionStage.ADAPTER_INITIALIZED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.ADAPTER_INITIALIZED: frozenset({ExecutionSessionStage.RECOVERY_CHECKED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.RECOVERY_CHECKED: frozenset({ExecutionSessionStage.RUNNING, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.RUNNING: frozenset({
        ExecutionSessionStage.RUNNING, ExecutionSessionStage.PAUSING, ExecutionSessionStage.HALTING, ExecutionSessionStage.RECONCILING,
        ExecutionSessionStage.FAILED,
    }),
    ExecutionSessionStage.PAUSING: frozenset({ExecutionSessionStage.PAUSED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.PAUSED: frozenset({
        ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED, ExecutionSessionStage.HALTING, ExecutionSessionStage.FAILED,
    }),
    ExecutionSessionStage.HALTING: frozenset({ExecutionSessionStage.HALTED, ExecutionSessionStage.FAILED}),
    # A HALTED session may still be reconciled/verified/reported (evidence
    # preserved) before being administratively TERMINATED -- it may NEVER
    # transition back to RUNNING (no silent auto-resume), mirroring
    # `paper_trading.models`'s identical HALTED allowance.
    ExecutionSessionStage.HALTED: frozenset({ExecutionSessionStage.TERMINATED, ExecutionSessionStage.RECONCILING, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.RECONCILING: frozenset({ExecutionSessionStage.VERIFYING, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.VERIFYING: frozenset({ExecutionSessionStage.COMPLETED, ExecutionSessionStage.FAILED}),
    ExecutionSessionStage.COMPLETED: frozenset(),
    ExecutionSessionStage.FAILED: frozenset(),
    ExecutionSessionStage.TERMINATED: frozenset(),
}


def is_legal_execution_session_transition(current: ExecutionSessionStage, target: ExecutionSessionStage) -> bool:
    if target in _LEGAL_EXECUTION_SESSION_TRANSITIONS[current]:
        return True
    # Same "rewind on detected corruption" resume allowance as
    # `paper_trading.models.is_legal_paper_session_transition`: resume
    # must be able to demote a manifest to the last independently
    # verified spine stage.
    if current in TERMINAL_EXECUTION_SESSION_STAGES or target in TERMINAL_EXECUTION_SESSION_STAGES:
        return False
    if current not in _EXECUTION_SESSION_STAGE_INDEX or target not in _EXECUTION_SESSION_STAGE_INDEX:
        return False
    return _EXECUTION_SESSION_STAGE_INDEX[target] < _EXECUTION_SESSION_STAGE_INDEX[current]


def is_terminal_execution_session_stage(stage: ExecutionSessionStage) -> bool:
    return stage in TERMINAL_EXECUTION_SESSION_STAGES


# --------------------------------------------------------------------------
# Health (Section 21)
# --------------------------------------------------------------------------
class AdapterHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    RECOVERING = "recovering"


# --------------------------------------------------------------------------
# Kill switch (Section 22)
# --------------------------------------------------------------------------
class ExecutionKillSwitchState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    HALTING = "halting"
    HALTED = "halted"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


_LEGAL_EXECUTION_KILL_SWITCH_TRANSITIONS: dict[ExecutionKillSwitchState, frozenset[ExecutionKillSwitchState]] = {
    ExecutionKillSwitchState.ACTIVE: frozenset({ExecutionKillSwitchState.DEGRADED, ExecutionKillSwitchState.HALTING}),
    ExecutionKillSwitchState.DEGRADED: frozenset({
        ExecutionKillSwitchState.ACTIVE, ExecutionKillSwitchState.HALTING, ExecutionKillSwitchState.RECOVERING,
    }),
    ExecutionKillSwitchState.HALTING: frozenset({ExecutionKillSwitchState.HALTED}),
    ExecutionKillSwitchState.HALTED: frozenset({ExecutionKillSwitchState.RECOVERING, ExecutionKillSwitchState.TERMINATED}),
    # RECOVERING may return to ACTIVE -- unlike Milestone 7's paper-trading
    # kill switch (which never resumes a genuine safety HALT), Section 22
    # explicitly defines a RECOVERING state whose entire purpose is to
    # restore ACTIVE once broker query + reconciliation succeed again
    # (e.g. after a transient DEGRADED/health-driven episode, never after
    # a HALTED safety stop -- HALTED only ever leads to RECOVERING via an
    # explicit administrative action, and RECOVERING itself may re-detect
    # the same problem and return to HALTING rather than ACTIVE).
    ExecutionKillSwitchState.RECOVERING: frozenset({ExecutionKillSwitchState.ACTIVE, ExecutionKillSwitchState.HALTING}),
    ExecutionKillSwitchState.TERMINATED: frozenset(),
}


def is_legal_execution_kill_switch_transition(current: ExecutionKillSwitchState, target: ExecutionKillSwitchState) -> bool:
    return target in _LEGAL_EXECUTION_KILL_SWITCH_TRANSITIONS[current]


NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES: frozenset[ExecutionKillSwitchState] = frozenset({ExecutionKillSwitchState.ACTIVE})
"""Only `ACTIVE` permits a brand-new SUBMIT that opens/increases
exposure. `DEGRADED` still permits cancel/reduce-only (Section 22)."""

REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES: frozenset[ExecutionKillSwitchState] = frozenset({
    ExecutionKillSwitchState.ACTIVE, ExecutionKillSwitchState.DEGRADED, ExecutionKillSwitchState.HALTING,
})
"""Section 22: "Safety cancel/reduce-only actions must not be blocked by
the same new-exposure rule they are intended to mitigate." A CANCEL, or a
SUBMIT that is both `reduce_only=True` and provably risk-reducing, is
permitted through `DEGRADED` and `HALTING` (to let a halt actually flatten
the position it halted over) but never through `HALTED`/`RECOVERING`/
`TERMINATED`, where only inspection/query/reconciliation are allowed."""

QUERY_PERMITTING_KILL_SWITCH_STATES: frozenset[ExecutionKillSwitchState] = frozenset({
    ExecutionKillSwitchState.ACTIVE, ExecutionKillSwitchState.DEGRADED, ExecutionKillSwitchState.HALTING, ExecutionKillSwitchState.HALTED,
    ExecutionKillSwitchState.RECOVERING,
})
"""Every non-`TERMINATED` state permits query/reconciliation -- this is
precisely what lets HALTED/RECOVERING resolve ambiguity and reconcile
without ever taking a new position."""


__all__ = [
    "BLOCKING_EXECUTION_ORDER_STATES",
    "EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION",
    "MUTATING_COMMAND_TYPES",
    "NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES",
    "QUERY_PERMITTING_KILL_SWITCH_STATES",
    "REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES",
    "TERMINAL_EXECUTION_ORDER_STATES",
    "TERMINAL_EXECUTION_SESSION_STAGES",
    "AdapterHealthStatus",
    "AdapterKind",
    "AuthorizationMode",
    "BrokerEventType",
    "CommandType",
    "DispatchStage",
    "ExecutionKillSwitchState",
    "ExecutionLedgerEntryKind",
    "ExecutionMode",
    "ExecutionOrderState",
    "ExecutionSessionStage",
    "OrderSide",
    "OrderTypeKind",
    "SequencingPolicyKind",
    "TimeInForceKind",
    "is_blocking_execution_order_state",
    "is_legal_execution_kill_switch_transition",
    "is_legal_execution_order_transition",
    "is_legal_execution_session_transition",
    "is_terminal_execution_order_state",
    "is_terminal_execution_session_stage",
]
