"""Deterministic crash recovery (Milestone 8, Section 23). Recovery
NEVER trusts cached order state or a final report -- it reconstructs
purely from the spec, manifest, append-only ledger, and (for `UNKNOWN`
ambiguity specifically) a fresh adapter query. Every one of Section 16's
recovery rules is implemented exactly:

- command created but no dispatch intent recorded: safe to dispatch
  (`dispatcher.dispatch_command` on the next `runner.py` pass will do
  this correctly on its own, since no dispatch intent means the command
  was never sent -- recovery does not need to intervene).
- dispatch intent persisted, no dispatch result: AMBIGUOUS -- resolved
  here by querying the adapter for the order's authoritative state.
- broker confirms the order: reconstruct the acknowledgement/fill/
  rejection state from the query response.
- broker does not confirm and the adapter GUARANTEES idempotent submit:
  safe to retry using the SAME `client_order_id` (never a new one).
- broker does not confirm and the adapter does NOT guarantee
  idempotency: marked/left `UNKNOWN`, new exposure halted -- NEVER a
  blind retry of a potentially-already-accepted non-idempotent submit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.execution_gateway.adapter import (
    AdapterCapabilities,
    BrokerOrderSnapshot,
    ExecutionAdapter,
)
from quant_platform.execution_gateway.commands import SubmitOrderCommand
from quant_platform.execution_gateway.dispatcher import (
    _append,
    _append_order_transition,
)
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind, ExecutionOrderState
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.reconciliation import reconstruct_all_orders_from_ledger
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    resolve_execution_order_state,
)

_QUERY_STATE_TO_TARGET: dict[ExecutionOrderState, ExecutionOrderState] = {
    ExecutionOrderState.ACKNOWLEDGED: ExecutionOrderState.ACKNOWLEDGED,
    ExecutionOrderState.PARTIALLY_FILLED: ExecutionOrderState.PARTIALLY_FILLED,
    ExecutionOrderState.FILLED: ExecutionOrderState.FILLED,
    ExecutionOrderState.CANCELLED: ExecutionOrderState.CANCELLED,
    ExecutionOrderState.REJECTED: ExecutionOrderState.REJECTED,
    ExecutionOrderState.EXPIRED: ExecutionOrderState.EXPIRED,
}


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    execution_order_id: str
    client_order_id: str
    action: str
    """One of `resolved_by_query`, `safe_retry_authorized`, `remains_unknown`."""
    detail: str


def recover_unknown_orders(
    *, execution_session_id: str, event_store: ExecutionSessionEventStore, adapter: ExecutionAdapter, capabilities: AdapterCapabilities, event_time: datetime,
) -> list[RecoveryAction]:
    """Resolves every order currently in `UNKNOWN` state (Section 23).
    Called by `runner.py` immediately after `ADAPTER_INITIALIZED` on
    every resume, BEFORE any new command is ever dispatched."""
    ledger = event_store.read_events(execution_session_id)
    reconstructed = reconstruct_all_orders_from_ledger(ledger)
    actions: list[RecoveryAction] = []

    for order_id, (command, state_events, _broker_events, _fills) in reconstructed.items():
        current = resolve_execution_order_state(order_id, state_events)
        if current is not ExecutionOrderState.UNKNOWN:
            continue
        actions.append(_recover_one_order(execution_session_id=execution_session_id, event_store=event_store, adapter=adapter, capabilities=capabilities, order_id=order_id, command=command, order_state_events=state_events, event_time=event_time))
    return actions


def _recover_one_order(
    *, execution_session_id: str, event_store: ExecutionSessionEventStore, adapter: ExecutionAdapter, capabilities: AdapterCapabilities, order_id: str,
    command: SubmitOrderCommand, order_state_events: list[ExecutionOrderStateEvent], event_time: datetime,
) -> RecoveryAction:
    if not capabilities.supports_order_query:
        return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="remains_unknown", detail="adapter does not support order query -- cannot safely resolve ambiguity")

    try:
        snapshot: BrokerOrderSnapshot | None = adapter.query_order(broker_order_id=None, client_order_id=command.client_order_id, event_time=event_time)
    except Exception as exc:
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.RECONCILIATION_FAILED, payload={"execution_order_id": order_id, "reason": str(exc)}, event_time=event_time)
        return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="remains_unknown", detail=f"order query raised: {exc}")

    if snapshot is not None:
        target = _QUERY_STATE_TO_TARGET.get(snapshot.state)
        if target is not None:
            _append_order_transition(event_store, execution_session_id, execution_order_id=order_id, existing_events=order_state_events, from_state=ExecutionOrderState.UNKNOWN, to_state=target, event_time=event_time, reason_code=(None if target not in (ExecutionOrderState.REJECTED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED) else "resolved_by_recovery_query"))
            return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="resolved_by_query", detail=f"broker confirms state={snapshot.state.value!r}")
        return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="remains_unknown", detail=f"broker returned an unrecognized state {snapshot.state.value!r}")

    # Broker has no record of this order.
    if capabilities.guarantees_idempotent_submit:
        return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="safe_retry_authorized", detail="broker has no record and the adapter guarantees idempotent submit -- safe to redispatch using the same client_order_id")
    return RecoveryAction(execution_order_id=order_id, client_order_id=command.client_order_id, action="remains_unknown", detail="broker has no record and the adapter does NOT guarantee idempotent submit -- refusing to blindly retry a possibly-already-accepted order")


__all__ = ["RecoveryAction", "recover_unknown_orders"]
