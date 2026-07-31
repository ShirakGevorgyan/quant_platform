"""Independent reconciliation (Milestone 8, Section 24) between the
gateway's own from-ledger reconstruction and the dummy broker's
independently-maintained snapshots. `reconcile_execution_session` NEVER
raises for an ordinary "state does not match" outcome -- every mismatch
becomes a structured `ReconciliationIssue`, even when the underlying
state is badly corrupted (Section 24: "must not crash with an unhandled
domain exception"). Genuine structural failures (the adapter itself
cannot be queried) are ALSO captured as a `CRITICAL` issue, never an
uncaught exception."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.execution_gateway.adapter import ExecutionAdapter
from quant_platform.execution_gateway.commands import SubmitOrderCommand
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.models import (
    TERMINAL_EXECUTION_ORDER_STATES,
    ExecutionLedgerEntryKind,
    ExecutionOrderState,
)
from quant_platform.execution_gateway.persistence import ExecutionLedgerEntry
from quant_platform.execution_gateway.specs import ReconciliationPolicySpec
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
)
from quant_platform.execution_gateway.states import (
    ExecutionFill,
    compute_execution_order_id,
    reconstruct_execution_order,
)


class ReconciliationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"
    CRITICAL = "critical"


_BLOCKING_SEVERITIES: frozenset[ReconciliationSeverity] = frozenset({ReconciliationSeverity.BLOCKING, ReconciliationSeverity.CRITICAL})


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    issue_code: str
    severity: ReconciliationSeverity
    entity_type: str
    entity_id: str
    expected: str
    actual: str
    description: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "issue_code": self.issue_code, "severity": self.severity.value, "entity_type": self.entity_type, "entity_id": self.entity_id,
            "expected": self.expected, "actual": self.actual, "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReconciliationReport:
    issues: tuple[ReconciliationIssue, ...]

    @property
    def is_reconciled(self) -> bool:
        return not any(i.severity in _BLOCKING_SEVERITIES for i in self.issues)

    @property
    def blocking_issues(self) -> tuple[ReconciliationIssue, ...]:
        return tuple(i for i in self.issues if i.severity in _BLOCKING_SEVERITIES)

    def to_json_dict(self) -> dict[str, object]:
        return {"issues": [i.to_json_dict() for i in self.issues], "is_reconciled": self.is_reconciled}


def reconstruct_all_orders_from_ledger(ledger: list[ExecutionLedgerEntry]) -> dict[str, tuple[SubmitOrderCommand, list[ExecutionOrderStateEvent], list[BrokerEvent], list[ExecutionFill]]]:
    """Shared evidence-reconstruction helper -- also used by
    `recovery.py`, which needs the same per-order `(command, state_events,
    broker_events, fills)` tuples to resolve `UNKNOWN` ambiguity."""
    submits: dict[str, SubmitOrderCommand] = {}
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and e.payload.get("command_type") == "submit_order":
            command = SubmitOrderCommand.from_json_dict(e.payload)
            submits[compute_execution_order_id(command)] = command

    state_events_by_order: dict[str, list[ExecutionOrderStateEvent]] = {oid: [] for oid in submits}
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION:
            event = ExecutionOrderStateEvent.from_json_dict(e.payload)
            if event.execution_order_id in state_events_by_order:
                state_events_by_order[event.execution_order_id].append(event)

    broker_events_by_client_order: dict[str, list[BrokerEvent]] = {}
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED:
            be = BrokerEvent.from_json_dict(e.payload)
            if be.client_order_id is not None:
                broker_events_by_client_order.setdefault(be.client_order_id, []).append(be)

    fills_by_order: dict[str, list[ExecutionFill]] = {oid: [] for oid in submits}
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED:
            fill = ExecutionFill.from_json_dict(e.payload)
            if fill.execution_order_id in fills_by_order:
                fills_by_order[fill.execution_order_id].append(fill)

    result = {}
    for order_id, command in submits.items():
        broker_events = broker_events_by_client_order.get(command.client_order_id, [])
        result[order_id] = (command, state_events_by_order[order_id], broker_events, fills_by_order[order_id])
    return result


def _broker_sequence_gaps(ledger: list[ExecutionLedgerEntry]) -> list[int]:
    sequences = sorted({BrokerEvent.from_json_dict(e.payload).broker_sequence for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED})
    if not sequences:
        return []
    return [s for s in range(sequences[0], sequences[-1] + 1) if s not in set(sequences)]


def reconcile_execution_session(
    *, execution_session_id: str, ledger: list[ExecutionLedgerEntry], adapter: ExecutionAdapter, event_time: datetime, policy: ReconciliationPolicySpec,
) -> ExecutionReconciliationReport:
    issues: list[ReconciliationIssue] = []

    try:
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
    except Exception as exc:
        return ExecutionReconciliationReport(issues=(ReconciliationIssue(issue_code="internal_reconstruction_failed", severity=ReconciliationSeverity.CRITICAL, entity_type="session", entity_id=execution_session_id, expected="reconstructable ledger", actual=str(exc), description="internal order reconstruction raised"),))

    internal_orders = {}
    for order_id, (command, state_events, broker_events, fills) in reconstructed.items():
        try:
            internal_orders[order_id] = reconstruct_execution_order(submit_command=command, state_events=state_events, broker_events=broker_events, fills=fills)
        except Exception as exc:
            issues.append(ReconciliationIssue(issue_code="order_state_transition_illegal", severity=ReconciliationSeverity.CRITICAL, entity_type="order", entity_id=order_id, expected="legal state history", actual=str(exc), description="order-state replay raised during reconciliation"))

    # 22: unresolved UNKNOWN commands/orders.
    for order_id, order in internal_orders.items():
        if order.current_state is ExecutionOrderState.UNKNOWN:
            issues.append(ReconciliationIssue(issue_code="unresolved_unknown_order", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected="resolved state", actual="unknown", description="order remains UNKNOWN"))

    # Orphan fills: an EXECUTION_FILL_RECORDED entry whose execution_order_id
    # matches no submitted order in this ledger is invisible to `reconstructed`
    # (which groups fills under KNOWN orders only) -- scanned directly against
    # the raw ledger here so it is never silently dropped (mirrors verify_
    # execution_session's own equivalent check).
    known_order_ids = set(reconstructed)
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED:
            fill_order_id = e.payload.get("execution_order_id")
            if fill_order_id not in known_order_ids:
                issues.append(ReconciliationIssue(issue_code="orphan_fill_no_matching_order", severity=ReconciliationSeverity.CRITICAL, entity_type="fill", entity_id=str(e.entry_id), expected="execution_order_id referencing a submitted order", actual=str(fill_order_id), description="fill entry references an execution_order_id with no corresponding submitted order"))

    # 24: broker sequence gaps.
    for gap_sequence in _broker_sequence_gaps(ledger):
        issues.append(ReconciliationIssue(issue_code="broker_sequence_gap", severity=ReconciliationSeverity.BLOCKING, entity_type="session", entity_id=execution_session_id, expected="contiguous broker_sequence", actual=str(gap_sequence), description=f"broker_sequence {gap_sequence} was never recorded"))

    # 20: session ownership mismatch -- every ledger entry belongs to this session.
    for e in ledger:
        if e.execution_session_id != execution_session_id:
            issues.append(ReconciliationIssue(issue_code="session_ownership_mismatch", severity=ReconciliationSeverity.CRITICAL, entity_type="ledger_entry", entity_id=e.entry_id, expected=execution_session_id, actual=e.execution_session_id, description="ledger entry belongs to a different execution session"))

    # 1/2/3/4/5/6/8/9/10/11/12: open orders vs broker open orders.
    try:
        broker_open_orders = {o.client_order_id: o for o in adapter.query_open_orders(event_time=event_time)}
        broker_query_ok = True
    except Exception as exc:
        issues.append(ReconciliationIssue(issue_code="broker_open_orders_query_failed", severity=ReconciliationSeverity.CRITICAL, entity_type="session", entity_id=execution_session_id, expected="successful query", actual=str(exc), description="adapter.query_open_orders raised"))
        broker_open_orders = {}
        broker_query_ok = False

    if broker_query_ok:
        for order_id, order in internal_orders.items():
            broker_order = broker_open_orders.get(order.client_order_id)
            if order.current_state not in TERMINAL_EXECUTION_ORDER_STATES:
                if broker_order is None:
                    issues.append(ReconciliationIssue(issue_code="missing_broker_order", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected="present in broker open orders", actual="absent", description="internally non-terminal order is not open at the broker"))
                    continue
                if broker_order.broker_order_id != (order.broker_order_id or broker_order.broker_order_id):
                    issues.append(ReconciliationIssue(issue_code="broker_order_id_mismatch", severity=ReconciliationSeverity.CRITICAL, entity_type="order", entity_id=order_id, expected=str(order.broker_order_id), actual=broker_order.broker_order_id, description="broker_order_id disagreement"))
                if broker_order.side is not order.side:
                    issues.append(ReconciliationIssue(issue_code="side_mismatch", severity=ReconciliationSeverity.CRITICAL, entity_type="order", entity_id=order_id, expected=order.side.value, actual=broker_order.side.value, description="side disagreement"))
                if abs(broker_order.current_quantity - order.current_quantity) > policy.quantity_tolerance:
                    issues.append(ReconciliationIssue(issue_code="current_quantity_mismatch", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected=str(order.current_quantity), actual=str(broker_order.current_quantity), description="current_quantity disagreement"))
                if abs(broker_order.filled_quantity - order.filled_quantity) > policy.quantity_tolerance:
                    issues.append(ReconciliationIssue(issue_code="cumulative_filled_quantity_mismatch", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected=str(order.filled_quantity), actual=str(broker_order.filled_quantity), description="filled_quantity disagreement"))
                if abs(broker_order.remaining_quantity - order.remaining_quantity) > policy.quantity_tolerance:
                    issues.append(ReconciliationIssue(issue_code="remaining_quantity_mismatch", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected=str(order.remaining_quantity), actual=str(broker_order.remaining_quantity), description="remaining_quantity disagreement"))
                if broker_order.state is not order.current_state:
                    issues.append(ReconciliationIssue(issue_code="order_status_mismatch", severity=ReconciliationSeverity.BLOCKING, entity_type="order", entity_id=order_id, expected=order.current_state.value, actual=broker_order.state.value, description="order status disagreement"))

        internal_client_order_ids = {o.client_order_id for o in internal_orders.values()}
        for client_order_id in broker_open_orders:
            if client_order_id not in internal_client_order_ids:
                issues.append(ReconciliationIssue(issue_code="unknown_broker_order", severity=ReconciliationSeverity.CRITICAL, entity_type="order", entity_id=client_order_id, expected="known internally", actual="unknown internally", description="broker reports an open order this session never created"))

    # 13/14/15/16: positions vs broker.
    internal_positions: dict[str, Decimal] = {}
    for order in internal_orders.values():
        internal_positions[order.instrument_id] = internal_positions.get(order.instrument_id, Decimal(0)) + (order.filled_quantity if order.side.value == "buy" else -order.filled_quantity)
    try:
        broker_positions = {p.instrument_id: p for p in adapter.query_positions(event_time=event_time)}
    except Exception as exc:
        issues.append(ReconciliationIssue(issue_code="broker_positions_query_failed", severity=ReconciliationSeverity.CRITICAL, entity_type="session", entity_id=execution_session_id, expected="successful query", actual=str(exc), description="adapter.query_positions raised"))
        broker_positions = {}
    for instrument_id, internal_qty in internal_positions.items():
        broker_position = broker_positions.get(instrument_id)
        broker_qty = broker_position.signed_quantity if broker_position is not None else Decimal(0)
        if abs(internal_qty - broker_qty) > policy.quantity_tolerance:
            issues.append(ReconciliationIssue(issue_code="position_quantity_mismatch", severity=ReconciliationSeverity.BLOCKING, entity_type="position", entity_id=instrument_id, expected=str(internal_qty), actual=str(broker_qty), description="signed_quantity disagreement"))

    # 17/18: account balance/equity consistency (sanity: equity must be finite and cash must be finite -- deep P&L cross-checking against paper_trading formulas is reports.py's/verification.py's job).
    try:
        account = adapter.query_account(event_time=event_time)
        if not account.cash.is_finite() or not account.equity.is_finite():
            issues.append(ReconciliationIssue(issue_code="account_balance_not_finite", severity=ReconciliationSeverity.CRITICAL, entity_type="account", entity_id=execution_session_id, expected="finite cash/equity", actual=f"cash={account.cash!r} equity={account.equity!r}", description="broker account snapshot is not finite"))
    except Exception as exc:
        issues.append(ReconciliationIssue(issue_code="broker_account_query_failed", severity=ReconciliationSeverity.CRITICAL, entity_type="account", entity_id=execution_session_id, expected="successful query", actual=str(exc), description="adapter.query_account raised"))

    # 21: contract multiplier consistency across intent/order/fill.
    for _order_id, (command, _state_events, _broker_events, fills) in reconstructed.items():
        for fill in fills:
            if fill.contract_multiplier != command.contract_multiplier:
                issues.append(ReconciliationIssue(issue_code="contract_multiplier_mismatch", severity=ReconciliationSeverity.CRITICAL, entity_type="fill", entity_id=fill.execution_fill_id, expected=str(command.contract_multiplier), actual=str(fill.contract_multiplier), description="fill contract_multiplier does not match its order's own"))

    # 10: duplicate fills (same broker_fill_id claimed by two distinct execution_fill_id values for the same order).
    seen_broker_fill_ids: dict[str, str] = {}
    for order_id, (_c, _s, _b, fills) in reconstructed.items():
        for fill in fills:
            key = f"{order_id}:{fill.broker_fill_id}"
            existing = seen_broker_fill_ids.get(key)
            if existing is not None and existing != fill.execution_fill_id:
                issues.append(ReconciliationIssue(issue_code="duplicate_fill", severity=ReconciliationSeverity.CRITICAL, entity_type="fill", entity_id=fill.execution_fill_id, expected=existing, actual=fill.execution_fill_id, description="the same broker_fill_id produced two different execution_fill_id values"))
            seen_broker_fill_ids[key] = fill.execution_fill_id

    return ExecutionReconciliationReport(issues=tuple(issues))


__all__ = ["ExecutionReconciliationReport", "ReconciliationIssue", "ReconciliationSeverity", "reconcile_execution_session", "reconstruct_all_orders_from_ledger"]
