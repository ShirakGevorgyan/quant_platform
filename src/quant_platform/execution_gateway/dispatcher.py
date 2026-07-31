"""Dispatch transaction (Milestone 8, Section 17) and broker-event
processing. `dispatch_command` is the ONE centralized place every
command in this package is ever sent to the adapter through -- no
command bypasses it (Section 22: "Every dispatch path must pass through
one centralized dispatch gate"). Kill-switch/health gating happens in
`runner.py` BEFORE `dispatch_command` is ever called; this module's own
job is the DISPATCH TRANSACTION itself: durable stages, in order,
persisted as they happen, with the dispatch INTENT durably recorded
BEFORE the adapter is ever called (Section 17's core requirement) --
never mapping every exception to a single "FAILED" bucket, always
distinguishing definitely-not-dispatched, definitely-dispatched,
rejected-before-dispatch, adapter-rejected, and unknown.

`process_broker_events` is the corresponding INBOUND half: polls the
adapter, classifies each event via `sequencing.classify_broker_event`,
and applies its consequences (order-state transition, fill recording)
to the ledger -- duplicates are recorded and idempotently absorbed,
never reprocessed; conflicts are recorded and raised, never silently
resolved."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.execution_gateway.adapter import AdapterCallResult, ExecutionAdapter, require_capability
from quant_platform.execution_gateway.commands import (
    CancelOrderCommand,
    ReplaceOrderCommand,
    SubmitOrderCommand,
)
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.idempotency import is_command_already_recorded
from quant_platform.execution_gateway.models import (
    BrokerEventType,
    ExecutionLedgerEntryKind,
    ExecutionOrderState,
    SequencingPolicyKind,
    is_legal_execution_order_transition,
)
from quant_platform.execution_gateway.persistence import (
    ExecutionLedgerEntry,
    ExecutionSessionEventStore,
    create_execution_ledger_entry,
)
from quant_platform.execution_gateway.sequencing import SequenceClassification, classify_broker_event
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    create_execution_order_state_event,
    resolve_execution_order_state,
)
from quant_platform.execution_gateway.states import compute_execution_order_id, create_execution_fill

_RESOLVED_DISPATCH_KINDS: frozenset[ExecutionLedgerEntryKind] = frozenset({
    ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED, ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED, ExecutionLedgerEntryKind.COMMAND_MARKED_UNKNOWN,
    ExecutionLedgerEntryKind.COMMAND_REJECTED,
})

_FILL_EVENT_TYPES: frozenset[BrokerEventType] = frozenset({BrokerEventType.ORDER_PARTIALLY_FILLED, BrokerEventType.ORDER_FILLED})


def _append(event_store: ExecutionSessionEventStore, execution_session_id: str, *, kind: ExecutionLedgerEntryKind, payload: dict[str, object], event_time: datetime) -> ExecutionLedgerEntry:
    sequence = event_store.next_sequence(execution_session_id)
    previous_hash = event_store.last_entry_hash(execution_session_id)
    entry = create_execution_ledger_entry(execution_session_id=execution_session_id, entry_sequence=sequence, entry_kind=kind, payload=payload, event_time=event_time, previous_entry_hash=previous_hash)
    return event_store.append(execution_session_id, entry)


def _append_order_transition(
    event_store: ExecutionSessionEventStore, execution_session_id: str, *, execution_order_id: str, existing_events: list[ExecutionOrderStateEvent], from_state: ExecutionOrderState,
    to_state: ExecutionOrderState, event_time: datetime, reason_code: str | None = None, source_broker_event_id: str | None = None,
) -> None:
    event = create_execution_order_state_event(
        execution_order_id=execution_order_id, execution_session_id=execution_session_id, from_state=from_state, to_state=to_state, event_time=event_time,
        sequence=len(existing_events), reason_code=reason_code, source_broker_event_id=source_broker_event_id,
    )
    existing_events.append(event)
    _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION, payload=event.to_json_dict(), event_time=event_time)


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    resolution: ExecutionLedgerEntry | None
    """`None` means the command was already durably recorded but never
    reached a resolved dispatch outcome (a genuine mid-dispatch crash) --
    `recovery.py`'s job, never silently retried here."""
    was_already_resolved: bool


def dispatch_command(
    *, execution_session_id: str, event_store: ExecutionSessionEventStore, adapter: ExecutionAdapter, command: SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand,
    event_time: datetime,
) -> DispatchOutcome:
    ledger = event_store.read_events(execution_session_id)
    command_payload = command.to_json_dict()
    already_recorded = is_command_already_recorded(ledger, command_id=command.command_id, payload=command_payload)
    if already_recorded:
        resolved = [e for e in ledger if e.entry_kind in _RESOLVED_DISPATCH_KINDS and e.payload.get("command_id") == command.command_id]
        return DispatchOutcome(resolution=(resolved[-1] if resolved else None), was_already_resolved=bool(resolved))

    _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_CREATED, payload=command_payload, event_time=event_time)

    if isinstance(command, SubmitOrderCommand):
        execution_order_id = compute_execution_order_id(command)
    else:
        execution_order_id = command.execution_order_id
    order_state_events = [e for e in _order_state_events_from_ledger(ledger) if e.execution_order_id == execution_order_id]

    if isinstance(command, SubmitOrderCommand):
        pre_dispatch_state = ExecutionOrderState.CREATED
        validated_state = ExecutionOrderState.VALIDATED
        pending_state = ExecutionOrderState.DISPATCH_PENDING
        settled_success_state: ExecutionOrderState | None = ExecutionOrderState.DISPATCHED
    else:
        pre_dispatch_state = resolve_execution_order_state(execution_order_id, order_state_events)
        validated_state = pre_dispatch_state
        pending_state = ExecutionOrderState.CANCEL_PENDING if isinstance(command, CancelOrderCommand) else ExecutionOrderState.REPLACE_PENDING
        # A successful dispatch of a cancel/replace only proves the
        # REQUEST reached the adapter -- the order remains PENDING,
        # awaiting the asynchronous CANCEL_ACKNOWLEDGED/REPLACE_
        # ACKNOWLEDGED (or _REJECTED) broker event to actually resolve it.
        settled_success_state = None

    if validated_state is not pre_dispatch_state:
        _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=pre_dispatch_state, to_state=validated_state, event_time=event_time)

    try:
        require_capability(adapter.capabilities(), command)
    except Exception as exc:
        if isinstance(command, SubmitOrderCommand):
            _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=validated_state, to_state=ExecutionOrderState.REJECTED, event_time=event_time, reason_code="capability_violation")
        entry = _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_REJECTED, payload={"command_id": command.command_id, "reason": str(exc)}, event_time=event_time)
        return DispatchOutcome(resolution=entry, was_already_resolved=False)

    _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=validated_state, to_state=pending_state, event_time=event_time)
    _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT, payload={"command_id": command.command_id}, event_time=event_time)

    try:
        result = _call_adapter(adapter, command, event_time=event_time)
    except Exception as exc:
        _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=pending_state, to_state=ExecutionOrderState.UNKNOWN, event_time=event_time, reason_code="ambiguous_dispatch")
        entry = _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_MARKED_UNKNOWN, payload={"command_id": command.command_id, "reason": str(exc)}, event_time=event_time)
        return DispatchOutcome(resolution=entry, was_already_resolved=False)

    if result.accepted_for_processing:
        if settled_success_state is not None:
            _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=pending_state, to_state=settled_success_state, event_time=event_time)
        entry = _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED, payload={"command_id": command.command_id, "client_order_id": command.client_order_id, "adapter_call_id": result.adapter_call_id}, event_time=event_time)
        return DispatchOutcome(resolution=entry, was_already_resolved=False)

    # Adapter synchronously refused the request -- the order reverts to
    # its pre-dispatch state for cancel/replace (Section 8: CANCEL_PENDING/
    # REPLACE_PENDING may legally return to ACKNOWLEDGED/PARTIALLY_FILLED),
    # or is REJECTED for a fresh submit. REJECTED is only a legal target
    # from DISPATCHED (Section 8's required transition list), never
    # directly from DISPATCH_PENDING, so a submit rejection passes
    # through DISPATCHED first -- the adapter DID synchronously respond,
    # it just responded with a refusal.
    if isinstance(command, SubmitOrderCommand):
        _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=pending_state, to_state=ExecutionOrderState.DISPATCHED, event_time=event_time)
        _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=ExecutionOrderState.DISPATCHED, to_state=ExecutionOrderState.REJECTED, event_time=event_time, reason_code="adapter_rejected")
    else:
        _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=pending_state, to_state=pre_dispatch_state, event_time=event_time)
    entry = _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED, payload={"command_id": command.command_id, "reason": result.rejection_reason}, event_time=event_time)
    return DispatchOutcome(resolution=entry, was_already_resolved=False)


def _call_adapter(adapter: ExecutionAdapter, command: SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand, *, event_time: datetime) -> AdapterCallResult:
    if isinstance(command, SubmitOrderCommand):
        return adapter.submit_order(command, event_time=event_time)
    if isinstance(command, CancelOrderCommand):
        return adapter.cancel_order(command, event_time=event_time)
    return adapter.replace_order(command, event_time=event_time)


def _order_state_events_from_ledger(ledger: list[ExecutionLedgerEntry]) -> list[ExecutionOrderStateEvent]:
    return [
        ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION
    ]


# --------------------------------------------------------------------------
# Inbound: poll + classify + apply broker events (Section 15/18).
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BrokerEventProcessingResult:
    new_events: tuple[BrokerEvent, ...]
    critical_conflict: bool


def process_broker_events(
    *, execution_session_id: str, event_store: ExecutionSessionEventStore, adapter: ExecutionAdapter, max_events: int, event_time: datetime,
) -> BrokerEventProcessingResult:
    """Polls `adapter` for new events since the last durably-recorded
    `broker_sequence`, classifies each via `sequencing.py`, and applies
    its consequences. NEVER silently skips a sequence gap or drops a
    contradictory event -- every classification is persisted as its own
    ledger entry kind."""
    ledger = event_store.read_events(execution_session_id)
    events_by_sequence: dict[int, BrokerEvent] = {}
    max_seen_sequence = 0
    for e in ledger:
        if e.entry_kind in (ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE):
            broker_event = BrokerEvent.from_json_dict(e.payload)
            events_by_sequence[broker_event.broker_sequence] = broker_event
            max_seen_sequence = max(max_seen_sequence, broker_event.broker_sequence)

    polled = adapter.poll_events(after_sequence=max_seen_sequence, max_events=max_events, event_time=event_time)
    new_events: list[BrokerEvent] = []
    critical_conflict = False
    order_state_cache: dict[str, list[ExecutionOrderStateEvent]] = {}

    for broker_event in polled:
        classification = classify_broker_event(broker_event, policy=SequencingPolicyKind.STRICT_SEQUENCE, events_by_sequence=events_by_sequence, max_seen_sequence=max_seen_sequence)
        if classification.classification is SequenceClassification.DUPLICATE:
            _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE, payload=broker_event.to_json_dict(), event_time=event_time)
            continue
        if classification.classification is SequenceClassification.SEQUENCE_CONFLICT:
            _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT, payload=broker_event.to_json_dict(), event_time=event_time)
            critical_conflict = True
            continue
        if classification.classification is SequenceClassification.OUT_OF_ORDER:
            _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_OUT_OF_ORDER, payload=broker_event.to_json_dict(), event_time=event_time)
            events_by_sequence[broker_event.broker_sequence] = broker_event
            max_seen_sequence = max(max_seen_sequence, broker_event.broker_sequence)
            new_events.append(broker_event)
            _apply_broker_event(event_store, execution_session_id, broker_event, event_time=event_time, order_state_cache=order_state_cache, ledger_hint=ledger)
            continue

        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload=broker_event.to_json_dict(), event_time=event_time)
        events_by_sequence[broker_event.broker_sequence] = broker_event
        max_seen_sequence = max(max_seen_sequence, broker_event.broker_sequence)
        new_events.append(broker_event)
        _apply_broker_event(event_store, execution_session_id, broker_event, event_time=event_time, order_state_cache=order_state_cache, ledger_hint=ledger)

    return BrokerEventProcessingResult(new_events=tuple(new_events), critical_conflict=critical_conflict)


_EVENT_TYPE_TARGET_STATE: dict[BrokerEventType, ExecutionOrderState] = {
    BrokerEventType.ORDER_ACKNOWLEDGED: ExecutionOrderState.ACKNOWLEDGED,
    BrokerEventType.ORDER_REJECTED: ExecutionOrderState.REJECTED,
    BrokerEventType.ORDER_EXPIRED: ExecutionOrderState.EXPIRED,
    BrokerEventType.ORDER_SUSPENDED: ExecutionOrderState.SUSPENDED,
    BrokerEventType.CANCEL_ACKNOWLEDGED: ExecutionOrderState.CANCELLED,
}


def _find_execution_order_id_for_client_order(ledger: list[ExecutionLedgerEntry], *, client_order_id: str) -> str | None:
    for entry in ledger:
        if entry.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and entry.payload.get("client_order_id") == client_order_id and entry.payload.get("command_type") == "submit_order":
            from quant_platform.execution_gateway.commands import SubmitOrderCommand as _Submit

            command = _Submit.from_json_dict(entry.payload)
            return compute_execution_order_id(command)
    return None


def _apply_broker_event(
    event_store: ExecutionSessionEventStore, execution_session_id: str, broker_event: BrokerEvent, *, event_time: datetime,
    order_state_cache: dict[str, list[ExecutionOrderStateEvent]], ledger_hint: list[ExecutionLedgerEntry],
) -> None:
    if broker_event.client_order_id is None:
        return
    execution_order_id = _find_execution_order_id_for_client_order(ledger_hint, client_order_id=broker_event.client_order_id)
    if execution_order_id is None:
        return
    if execution_order_id not in order_state_cache:
        order_state_cache[execution_order_id] = [e for e in _order_state_events_from_ledger(ledger_hint) if e.execution_order_id == execution_order_id]
    order_state_events = order_state_cache[execution_order_id]
    current = resolve_execution_order_state(execution_order_id, order_state_events)

    if broker_event.event_type in _FILL_EVENT_TYPES:
        _apply_fill_event(event_store, execution_session_id, broker_event, execution_order_id=execution_order_id, current=current, order_state_events=order_state_events, event_time=event_time)
        return

    if broker_event.event_type is BrokerEventType.REPLACE_ACKNOWLEDGED:
        _apply_replace_acknowledged_event(event_store, execution_session_id, broker_event, execution_order_id=execution_order_id, current=current, order_state_events=order_state_events, event_time=event_time)
        return

    target = _EVENT_TYPE_TARGET_STATE.get(broker_event.event_type)
    if target is None:
        return  # informational event type (ORDER_RECEIVED, ORDER_STATUS_SNAPSHOT, ...) -- no state transition.

    if not is_legal_execution_order_transition(current, target):
        # Contradictory event -- persisted as a conflict, never silently dropped, never silently applied.
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT, payload=broker_event.to_json_dict(), event_time=event_time)
        return
    reason = broker_event.reject_code if target in (ExecutionOrderState.REJECTED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED) else None
    if target in (ExecutionOrderState.REJECTED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED) and reason is None:
        reason = broker_event.event_type.value
    _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=current, to_state=target, event_time=event_time, reason_code=reason, source_broker_event_id=broker_event.broker_event_id)


def _apply_fill_event(
    event_store: ExecutionSessionEventStore, execution_session_id: str, broker_event: BrokerEvent, *, execution_order_id: str, current: ExecutionOrderState,
    order_state_events: list[ExecutionOrderStateEvent], event_time: datetime,
) -> None:
    is_full = broker_event.event_type is BrokerEventType.ORDER_FILLED
    target = ExecutionOrderState.FILLED if is_full else ExecutionOrderState.PARTIALLY_FILLED
    if not is_legal_execution_order_transition(current, target):
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT, payload=broker_event.to_json_dict(), event_time=event_time)
        return

    ledger = event_store.read_events(execution_session_id)
    submit_entry = next((e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and e.payload.get("command_type") == "submit_order" and _compute_id_matches(e, execution_order_id)), None)
    if submit_entry is None:
        return
    submit_command = SubmitOrderCommand.from_json_dict(submit_entry.payload)

    assert broker_event.filled_quantity is not None and broker_event.fill_price is not None
    fill = create_execution_fill(
        execution_session_id=execution_session_id, execution_order_id=execution_order_id, execution_intent_id=submit_command.execution_intent_id,
        broker_event=broker_event, client_order_id=submit_command.client_order_id, instrument_id=submit_command.instrument_id, side=submit_command.side,
        quantity=broker_event.filled_quantity, price=broker_event.fill_price, contract_multiplier=submit_command.contract_multiplier, commission=Decimal(0),
        spread_component=Decimal(0), slippage_component=Decimal(0), fill_sequence=_next_fill_sequence(ledger, execution_order_id=execution_order_id),
    )
    duplicate = any(e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED and e.payload.get("execution_fill_id") == fill.execution_fill_id for e in ledger)
    if not duplicate:
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED, payload=fill.to_json_dict(), event_time=event_time)
    _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=current, to_state=target, event_time=event_time, source_broker_event_id=broker_event.broker_event_id)


def _apply_replace_acknowledged_event(
    event_store: ExecutionSessionEventStore, execution_session_id: str, broker_event: BrokerEvent, *, execution_order_id: str, current: ExecutionOrderState,
    order_state_events: list[ExecutionOrderStateEvent], event_time: datetime,
) -> None:
    """CONFIRMED DEFECT, FIXED (found during this milestone's own
    acceptance testing): `REPLACE_ACKNOWLEDGED` was absent from
    `_EVENT_TYPE_TARGET_STATE`, so `_apply_broker_event`'s generic
    lookup silently treated it as an informational, no-op event type --
    an order left `REPLACE_PENDING` by a successful replace NEVER
    transitioned back to a live state, permanently stuck (and therefore
    unable to ever fill again), regardless of how many further broker
    events arrived. Unlike every other event type, REPLACE_ACKNOWLEDGED's
    target is not a single static state -- it must resolve back to
    whatever the order's state was IMMEDIATELY BEFORE the replace was
    dispatched (ACKNOWLEDGED or PARTIALLY_FILLED, Section 8's own
    REPLACE_PENDING legal-target set already reflects exactly this) --
    recovered here from the LAST recorded transition's own `from_state`,
    which is precisely that pre-replace state."""
    if current is not ExecutionOrderState.REPLACE_PENDING:
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT, payload=broker_event.to_json_dict(), event_time=event_time)
        return
    pre_replace_state = order_state_events[-1].from_state if order_state_events else ExecutionOrderState.ACKNOWLEDGED
    if not is_legal_execution_order_transition(current, pre_replace_state):
        _append(event_store, execution_session_id, kind=ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT, payload=broker_event.to_json_dict(), event_time=event_time)
        return
    _append_order_transition(event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=order_state_events, from_state=current, to_state=pre_replace_state, event_time=event_time, source_broker_event_id=broker_event.broker_event_id)


def _compute_id_matches(entry: ExecutionLedgerEntry, execution_order_id: str) -> bool:
    command = SubmitOrderCommand.from_json_dict(entry.payload)
    return compute_execution_order_id(command) == execution_order_id


def _next_fill_sequence(ledger: list[ExecutionLedgerEntry], *, execution_order_id: str) -> int:
    count = 0
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED and e.payload.get("execution_order_id") == execution_order_id:
            count += 1
    return count


__all__ = ["BrokerEventProcessingResult", "DispatchOutcome", "dispatch_command", "process_broker_events"]
