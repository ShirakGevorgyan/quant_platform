"""Risk engine and kill switch (Milestone 7, Sections 17-18). Every check
here is a PURE function returning a `RiskCheckResult` -- persisted
verbatim (check identity, measured value, limit, comparison operator,
result, action, reason code, event identity), never silently skipped: a
check whose limit is `None` (not configured) legitimately returns `None`
(there is nothing to check), but every CONFIGURED limit and every
mandatory boolean check (trading-halt, session-acceptance) is ALWAYS
evaluated and reported, pass or fail.

NO RISK ACTION SENDS A REAL ORDER (Section 17's own instruction) -- every
`RiskActionKind` member (`ALLOW`/`REJECT_ORDER`/`CANCEL_OPEN_ORDERS`/
`HALT_NEW_ORDERS`/`FLATTEN_SIMULATED_POSITIONS`/`TERMINATE_SESSION`) is a
purely LOCAL state transition the runner applies to its own simulated
book; none of them, even at their most severe, constructs a network
request or touches a broker.

`RiskCheckResult.reason_code` is a union: `RejectReasonKind` for a
pre-trade, ORDER-specific rejection, `RiskTriggerKind` for a continuous,
ACCOUNT-level trigger -- they are deliberately different closed
vocabularies (Section 8 vs Section 18), never conflated into one.

Kill-switch state is event-sourced exactly like `OrderState`
(`orders.resolve_order_state`'s identical pattern):
`resolve_kill_switch_state` replays a `KillSwitchTransitionEvent` sequence
from the implicit initial `ACTIVE` state -- and, per `models.
_LEGAL_KILL_SWITCH_TRANSITIONS`'s own construction, there is NO transition
back to `ACTIVE` from anywhere else in the graph, so "never silently
auto-resume after a safety halt" is a structural property, not a runtime
check."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import RiskHaltError, RiskLimitError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import (
    ComparisonOperatorKind,
    KillSwitchState,
    OrderSide,
    RejectReasonKind,
    RiskActionKind,
    RiskTriggerKind,
    is_legal_kill_switch_transition,
)
from quant_platform.paper_trading.orders import OrderRequest
from quant_platform.paper_trading.portfolio import PortfolioState
from quant_platform.paper_trading.specs import RiskLimitsSpec

KILL_SWITCH_TRANSITION_EVENT_KIND = "kill_switch_transition_event"

_ACTION_SEVERITY: dict[RiskActionKind, int] = {
    RiskActionKind.ALLOW: 0,
    RiskActionKind.REJECT_ORDER: 1,
    RiskActionKind.CANCEL_OPEN_ORDERS: 2,
    RiskActionKind.HALT_NEW_ORDERS: 3,
    RiskActionKind.FLATTEN_SIMULATED_POSITIONS: 4,
    RiskActionKind.TERMINATE_SESSION: 5,
}


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise RiskLimitError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise RiskLimitError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise RiskLimitError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise RiskLimitError(f"{field_name}: {exc}") from exc


def _compare(measured: float, limit: float, operator: ComparisonOperatorKind) -> bool:
    if operator is ComparisonOperatorKind.LESS_THAN_OR_EQUAL:
        return measured <= limit
    if operator is ComparisonOperatorKind.GREATER_THAN_OR_EQUAL:
        return measured >= limit
    if operator is ComparisonOperatorKind.LESS_THAN:
        return measured < limit
    return measured > limit  # GREATER_THAN


# --------------------------------------------------------------------------
# RiskCheckResult
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    check_identity: str
    measured_value: float
    limit: float
    comparison_operator: ComparisonOperatorKind
    passed: bool
    action: RiskActionKind
    reason_code: RejectReasonKind | RiskTriggerKind | None
    event_identity: str

    def __post_init__(self) -> None:
        if not self.check_identity:
            raise RiskLimitError("RiskCheckResult.check_identity must not be empty")
        if not math.isfinite(self.measured_value):
            raise RiskLimitError(f"RiskCheckResult.measured_value must be finite, got {self.measured_value!r}")
        if not math.isfinite(self.limit):
            raise RiskLimitError(f"RiskCheckResult.limit must be finite, got {self.limit!r}")
        if self.passed and self.action is not RiskActionKind.ALLOW:
            raise RiskLimitError("RiskCheckResult.action must be ALLOW when passed=True")
        if not self.passed and self.action is RiskActionKind.ALLOW:
            raise RiskLimitError("RiskCheckResult.action must not be ALLOW when passed=False")
        if self.passed and self.reason_code is not None:
            raise RiskLimitError("RiskCheckResult.reason_code must be None when passed=True")
        if not self.passed and self.reason_code is None:
            raise RiskLimitError("RiskCheckResult.reason_code is required when passed=False")

    def to_json_dict(self) -> dict[str, object]:
        reason_value: str | None = None if self.reason_code is None else self.reason_code.value
        return {
            "check_identity": self.check_identity, "measured_value": self.measured_value, "limit": self.limit,
            "comparison_operator": self.comparison_operator.value, "passed": self.passed, "action": self.action.value,
            "reason_code": reason_value, "reason_code_kind": (None if self.reason_code is None else type(self.reason_code).__name__),
            "event_identity": self.event_identity,
        }


def _numeric_check(
    check_identity: str, *, measured_value: float, limit: float | None, comparison_operator: ComparisonOperatorKind, action_if_failed: RiskActionKind,
    reason_code_if_failed: RejectReasonKind | RiskTriggerKind, event_identity: str,
) -> RiskCheckResult | None:
    if limit is None:
        return None
    passed = _compare(measured_value, limit, comparison_operator)
    return RiskCheckResult(
        check_identity=check_identity, measured_value=measured_value, limit=limit, comparison_operator=comparison_operator, passed=passed,
        action=(RiskActionKind.ALLOW if passed else action_if_failed), reason_code=(None if passed else reason_code_if_failed), event_identity=event_identity,
    )


def _mandatory_boolean_check(
    check_identity: str, *, triggered: bool, action_if_triggered: RiskActionKind, reason_code_if_triggered: RejectReasonKind | RiskTriggerKind,
    event_identity: str,
) -> RiskCheckResult:
    passed = not triggered
    return RiskCheckResult(
        check_identity=check_identity, measured_value=(1.0 if triggered else 0.0), limit=0.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL,
        passed=passed, action=(RiskActionKind.ALLOW if passed else action_if_triggered), reason_code=(None if passed else reason_code_if_triggered),
        event_identity=event_identity,
    )


def most_severe_action(results: tuple[RiskCheckResult, ...]) -> RiskActionKind:
    if not results:
        return RiskActionKind.ALLOW
    return max((r.action for r in results), key=lambda a: _ACTION_SEVERITY[a])


# --------------------------------------------------------------------------
# Pre-trade checks (Section 17, first list)
# --------------------------------------------------------------------------
def evaluate_pre_trade_risk(
    order: OrderRequest, *, portfolio: PortfolioState, risk_limits: RiskLimitsSpec, reference_price: float, contract_multiplier: float,
    event_identity: str, trading_halted: bool, session_accepting_orders: bool, stale_data_seconds: float | None,
) -> tuple[RiskCheckResult, ...]:
    """Evaluates EVERY applicable pre-trade check (mandatory booleans
    always run; numeric checks run whenever their limit is configured)
    and returns every result -- callers decide the overall action via
    `most_severe_action`, never by inspecting only the first failure."""
    current_position = portfolio.position_for(order.instrument)
    current_signed = 0.0 if current_position is None else current_position.signed_quantity
    delta = order.quantity if order.side is OrderSide.BUY else -order.quantity
    projected_signed = current_signed + delta
    projected_gross_exposure = abs(projected_signed) * reference_price * contract_multiplier
    order_notional = order.quantity * reference_price * contract_multiplier

    results: list[RiskCheckResult] = [
        _mandatory_boolean_check("trading_halted", triggered=trading_halted, action_if_triggered=RiskActionKind.REJECT_ORDER, reason_code_if_triggered=RejectReasonKind.TRADING_HALTED, event_identity=event_identity),
        _mandatory_boolean_check("session_accepting_orders", triggered=not session_accepting_orders, action_if_triggered=RiskActionKind.REJECT_ORDER, reason_code_if_triggered=RejectReasonKind.SESSION_NOT_ACCEPTING_ORDERS, event_identity=event_identity),
    ]

    if risk_limits.maximum_stale_data_seconds is not None and stale_data_seconds is not None:
        check = _numeric_check(
            "stale_data", measured_value=stale_data_seconds, limit=risk_limits.maximum_stale_data_seconds, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL,
            action_if_failed=RiskActionKind.REJECT_ORDER, reason_code_if_failed=RejectReasonKind.STALE_MARKET_DATA, event_identity=event_identity,
        )
        if check is not None:
            results.append(check)

    for check_identity, measured, limit, action, reason in (
        ("max_order_quantity", order.quantity, risk_limits.maximum_order_quantity, RiskActionKind.REJECT_ORDER, RejectReasonKind.ORDER_QUANTITY_LIMIT_EXCEEDED),
        ("max_order_notional", order_notional, risk_limits.maximum_order_notional, RiskActionKind.REJECT_ORDER, RejectReasonKind.ORDER_NOTIONAL_LIMIT_EXCEEDED),
        ("max_absolute_position", abs(projected_signed), risk_limits.maximum_absolute_position, RiskActionKind.REJECT_ORDER, RejectReasonKind.EXPOSURE_LIMIT_EXCEEDED),
        ("max_signed_position", abs(projected_signed), risk_limits.maximum_signed_position, RiskActionKind.REJECT_ORDER, RejectReasonKind.EXPOSURE_LIMIT_EXCEEDED),
        ("max_gross_exposure", projected_gross_exposure, risk_limits.maximum_gross_exposure, RiskActionKind.REJECT_ORDER, RejectReasonKind.EXPOSURE_LIMIT_EXCEEDED),
    ):
        check = _numeric_check(check_identity, measured_value=measured, limit=limit, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, action_if_failed=action, reason_code_if_failed=reason, event_identity=event_identity)
        if check is not None:
            results.append(check)

    return tuple(results)


# --------------------------------------------------------------------------
# Continuous / post-fill checks (Section 17, second list)
# --------------------------------------------------------------------------
def evaluate_continuous_risk(
    portfolio: PortfolioState, *, risk_limits: RiskLimitsSpec, event_identity: str, rejected_order_count: int, consecutive_execution_failures: int,
    stale_data_seconds: float | None, reconciliation_discrepancy: float | None,
) -> tuple[RiskCheckResult, ...]:
    results: list[RiskCheckResult] = []
    total_pnl = portfolio.realized_pnl + portfolio.unrealized_pnl
    for check_identity, measured, limit, action, reason in (
        ("max_daily_loss", -total_pnl, risk_limits.maximum_daily_loss, RiskActionKind.HALT_NEW_ORDERS, RiskTriggerKind.LOSS_LIMIT),
        ("max_drawdown", portfolio.drawdown_fraction, risk_limits.maximum_drawdown_fraction, RiskActionKind.FLATTEN_SIMULATED_POSITIONS, RiskTriggerKind.DRAWDOWN_LIMIT),
        ("max_realized_loss", -portfolio.realized_pnl, risk_limits.maximum_realized_loss, RiskActionKind.HALT_NEW_ORDERS, RiskTriggerKind.LOSS_LIMIT),
        ("max_unrealized_loss", -portfolio.unrealized_pnl, risk_limits.maximum_unrealized_loss, RiskActionKind.FLATTEN_SIMULATED_POSITIONS, RiskTriggerKind.LOSS_LIMIT),
        ("max_rejected_order_count", float(rejected_order_count), (None if risk_limits.maximum_rejected_order_count is None else float(risk_limits.maximum_rejected_order_count)), RiskActionKind.HALT_NEW_ORDERS, RiskTriggerKind.REPEATED_EXECUTION_ERRORS),
        ("max_consecutive_execution_failures", float(consecutive_execution_failures), (None if risk_limits.maximum_consecutive_execution_failures is None else float(risk_limits.maximum_consecutive_execution_failures)), RiskActionKind.TERMINATE_SESSION, RiskTriggerKind.REPEATED_EXECUTION_ERRORS),
    ):
        check = _numeric_check(check_identity, measured_value=measured, limit=limit, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, action_if_failed=action, reason_code_if_failed=reason, event_identity=event_identity)
        if check is not None:
            results.append(check)

    if risk_limits.maximum_stale_data_seconds is not None and stale_data_seconds is not None:
        check = _numeric_check(
            "max_stale_data_duration", measured_value=stale_data_seconds, limit=risk_limits.maximum_stale_data_seconds, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL,
            action_if_failed=RiskActionKind.HALT_NEW_ORDERS, reason_code_if_failed=RiskTriggerKind.STALE_DATA, event_identity=event_identity,
        )
        if check is not None:
            results.append(check)

    if reconciliation_discrepancy is not None:
        check = _numeric_check(
            "max_reconciliation_discrepancy", measured_value=abs(reconciliation_discrepancy), limit=risk_limits.maximum_reconciliation_discrepancy,
            comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, action_if_failed=RiskActionKind.TERMINATE_SESSION,
            reason_code_if_failed=RiskTriggerKind.RECONCILIATION_FAILURE, event_identity=event_identity,
        )
        if check is not None:
            results.append(check)

    return tuple(results)


# --------------------------------------------------------------------------
# Kill switch (Section 18) -- event-sourced exactly like OrderState.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KillSwitchTransitionEvent:
    event_id: str
    session_id: str
    from_state: KillSwitchState
    to_state: KillSwitchState
    trigger: RiskTriggerKind
    event_time: datetime
    sequence: int
    detail: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise RiskHaltError("KillSwitchTransitionEvent.session_id must not be empty")
        _require_tz_aware(self.event_time, field_name="KillSwitchTransitionEvent.event_time")
        if self.sequence < 0:
            raise RiskHaltError(f"KillSwitchTransitionEvent.sequence must be >= 0, got {self.sequence}")
        if not is_legal_kill_switch_transition(self.from_state, self.to_state):
            raise RiskHaltError(f"Illegal kill switch transition {self.from_state.value!r} -> {self.to_state.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id, "session_id": self.session_id, "from_state": self.from_state.value, "to_state": self.to_state.value,
            "trigger": self.trigger.value, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "detail": self.detail,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> KillSwitchTransitionEvent:
        return cls(
            event_id=str(raw["event_id"]), session_id=str(raw["session_id"]), from_state=KillSwitchState(raw["from_state"]),
            to_state=KillSwitchState(raw["to_state"]), trigger=RiskTriggerKind(raw["trigger"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])), detail=str(raw["detail"]),
        )


def create_kill_switch_transition_event(
    *, session_id: str, from_state: KillSwitchState, to_state: KillSwitchState, trigger: RiskTriggerKind, event_time: datetime, sequence: int, detail: str,
) -> KillSwitchTransitionEvent:
    provisional = KillSwitchTransitionEvent(
        event_id="0" * 64, session_id=session_id, from_state=from_state, to_state=to_state, trigger=trigger, event_time=event_time, sequence=sequence,
        detail=detail,
    )
    event_id = compute_content_id(KILL_SWITCH_TRANSITION_EVENT_KIND, provisional.to_identity_payload())
    return KillSwitchTransitionEvent(
        event_id=event_id, session_id=session_id, from_state=from_state, to_state=to_state, trigger=trigger, event_time=event_time, sequence=sequence,
        detail=detail,
    )


def resolve_kill_switch_state(events: list[KillSwitchTransitionEvent]) -> KillSwitchState:
    """Event-sourced derivation, mirroring `orders.resolve_order_state`.
    Since `models._LEGAL_KILL_SWITCH_TRANSITIONS` has no edge back to
    `ACTIVE` from anywhere, no sequence of legal events can ever return
    here to `ACTIVE` once it has left -- "never silently auto-resume
    after a safety halt" holds by construction, not by a check in this
    function."""
    current = KillSwitchState.ACTIVE
    for event in events:
        if event.from_state is not current:
            raise RiskHaltError(f"resolve_kill_switch_state: event {event.event_id!r} expects from_state={event.from_state.value!r} but current state is {current.value!r}")
        if not is_legal_kill_switch_transition(current, event.to_state):
            raise RiskHaltError(f"resolve_kill_switch_state: illegal transition {current.value!r} -> {event.to_state.value!r}")
        current = event.to_state
    return current


__all__ = [
    "KILL_SWITCH_TRANSITION_EVENT_KIND",
    "KillSwitchTransitionEvent",
    "RiskCheckResult",
    "create_kill_switch_transition_event",
    "evaluate_continuous_risk",
    "evaluate_pre_trade_risk",
    "most_severe_action",
    "resolve_kill_switch_state",
]
