"""Execution reports (Milestone 8, Section 27), recomputed purely from
the ledger and (where supplied) fresh broker snapshots -- never trusting
a precomputed summary field. `generate_execution_session_report` returns
one JSON-safe, Decimal-free (converted to `str`) nested dict covering
every one of Section 27's 15 named summaries; no NaN/Infinity can appear
since every underlying value is either a `Decimal` (already rejects
non-finite values at construction) or a plain count."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_platform.execution_gateway.commands import SubmitOrderCommand
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.models import (
    BrokerEventType,
    ExecutionLedgerEntryKind,
)
from quant_platform.execution_gateway.persistence import (
    ExecutionLedgerEntry,
    compute_execution_ledger_semantic_digest,
)
from quant_platform.execution_gateway.reconciliation import (
    ExecutionReconciliationReport,
    ReconciliationSeverity,
)
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    resolve_execution_order_state,
)
from quant_platform.execution_gateway.states import ExecutionFill, compute_execution_order_id


def _d(value: Decimal) -> str:
    return str(value)


@dataclass(frozen=True, slots=True)
class ExecutionSessionReport:
    execution_session_id: str
    sections: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return {"execution_session_id": self.execution_session_id, "sections": self.sections}


def generate_execution_session_report(
    *, execution_session_id: str, ledger: list[ExecutionLedgerEntry], reconciliation_report: ExecutionReconciliationReport | None = None,
) -> ExecutionSessionReport:
    intents = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED]
    intents_rejected = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_INTENT_REJECTED]

    commands = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED]
    commands_by_type: dict[str, int] = {}
    for c in commands:
        kind = str(c.payload.get("command_type"))
        commands_by_type[kind] = commands_by_type.get(kind, 0) + 1
    dispatch_succeeded = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED]
    dispatch_rejected = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED]
    dispatch_unknown = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_MARKED_UNKNOWN]

    submits = {compute_execution_order_id(SubmitOrderCommand.from_json_dict(e.payload)): SubmitOrderCommand.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and e.payload.get("command_type") == "submit_order"}
    state_events_by_order: dict[str, list[ExecutionOrderStateEvent]] = {oid: [] for oid in submits}
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION:
            event = ExecutionOrderStateEvent.from_json_dict(e.payload)
            if event.execution_order_id in state_events_by_order:
                state_events_by_order[event.execution_order_id].append(event)
    final_states = {oid: resolve_execution_order_state(oid, events) for oid, events in state_events_by_order.items()}
    order_state_counts: dict[str, int] = {}
    for state in final_states.values():
        order_state_counts[state.value] = order_state_counts.get(state.value, 0) + 1

    fills = [ExecutionFill.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED]
    total_fill_quantity = sum((f.quantity for f in fills), Decimal(0))
    total_gross_notional = sum((f.gross_notional for f in fills), Decimal(0))
    total_commission = sum((f.commission for f in fills), Decimal(0))
    total_spread = sum((f.spread_component for f in fills), Decimal(0))
    total_slippage = sum((f.slippage_component for f in fills), Decimal(0))
    average_fill_price = (sum((f.price * f.quantity for f in fills), Decimal(0)) / total_fill_quantity) if fills else None

    broker_events = [BrokerEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED]
    duplicate_events = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE]
    out_of_order_events = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_OUT_OF_ORDER]
    conflict_events = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT]
    partial_fill_events = [b for b in broker_events if b.event_type is BrokerEventType.ORDER_PARTIALLY_FILLED]
    full_fill_events = [b for b in broker_events if b.event_type is BrokerEventType.ORDER_FILLED]
    cancel_events = [b for b in broker_events if b.event_type is BrokerEventType.CANCEL_ACKNOWLEDGED]
    replace_events = [b for b in broker_events if b.event_type is BrokerEventType.REPLACE_ACKNOWLEDGED]
    disconnect_events = [b for b in broker_events if b.event_type is BrokerEventType.BROKER_DISCONNECTED]
    heartbeat_missed_events = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.HEARTBEAT_MISSED]

    kill_switch_events = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.KILL_SWITCH_TRIGGERED]

    recon_by_severity: dict[str, int] = {s.value: 0 for s in ReconciliationSeverity}
    if reconciliation_report is not None:
        for issue in reconciliation_report.issues:
            recon_by_severity[issue.severity.value] += 1

    completion_entries = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_SESSION_COMPLETED]
    semantic_digest = compute_execution_ledger_semantic_digest(ledger)

    sections: dict[str, object] = {
        "ExecutionSessionSummary": {"execution_session_id": execution_session_id, "total_ledger_entries": len(ledger), "final_semantic_digest": semantic_digest},
        "ExecutionIntentSummary": {"accepted_count": len(intents), "rejected_count": len(intents_rejected)},
        "CommandSummary": {"total_commands": len(commands), "by_type": commands_by_type},
        "OrderSummary": {"total_orders": len(submits), "final_state_counts": order_state_counts},
        "FillSummary": {
            "fill_count": len(fills), "total_fill_quantity": _d(total_fill_quantity), "average_fill_price": (None if average_fill_price is None else _d(average_fill_price)),
            "gross_notional": _d(total_gross_notional), "total_commission": _d(total_commission), "spread_cost": _d(total_spread), "slippage_cost": _d(total_slippage),
        },
        "BrokerEventSummary": {
            "received_count": len(broker_events), "duplicate_count": len(duplicate_events), "out_of_order_count": len(out_of_order_events),
            "conflict_count": len(conflict_events), "partial_fill_count": len(partial_fill_events), "full_fill_count": len(full_fill_events),
            "cancel_acknowledged_count": len(cancel_events), "replace_acknowledged_count": len(replace_events), "disconnect_count": len(disconnect_events),
        },
        "DispatchQualitySummary": {"succeeded": len(dispatch_succeeded), "rejected": len(dispatch_rejected), "marked_unknown": len(dispatch_unknown)},
        "HealthSummary": {"adapter_health_changes": len([e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ADAPTER_HEALTH_CHANGED])},
        "HeartbeatSummary": {"missed_count": len(heartbeat_missed_events), "restored_count": len([e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.HEARTBEAT_RESTORED])},
        "ReconciliationSummary": {"issues_by_severity": recon_by_severity, "is_reconciled": (None if reconciliation_report is None else reconciliation_report.is_reconciled)},
        "KillSwitchSummary": {"transition_count": len(kill_switch_events)},
        "RecoverySummary": {"resume_count_events": len([e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_SESSION_RESUMED])},
        "DeterminismSummary": {"semantic_digest": semantic_digest},
        "VerificationSummary": {"completed": bool(completion_entries)},
        "PaperComparisonSummary": {"note": "diagnostic comparison against the source paper session is out of this report's scope -- see verification.py"},
    }
    return ExecutionSessionReport(execution_session_id=execution_session_id, sections=sections)


__all__ = ["ExecutionSessionReport", "generate_execution_session_report"]
