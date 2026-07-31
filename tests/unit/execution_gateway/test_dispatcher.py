"""Unit tests for `execution_gateway.dispatcher` (Milestone 8, Section
17/18): the full dispatch transaction end-to-end against the real
deterministic dummy broker, idempotent redispatch, and broker-event
processing (including duplicate absorption and fill reconstruction)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_platform.execution_gateway.commands import create_cancel_order_command, create_submit_order_command
from quant_platform.execution_gateway.dispatcher import dispatch_command, process_broker_events
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.models import (
    ExecutionLedgerEntryKind,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.specs import DEFAULT_DUMMY_BROKER_SCENARIO
from quant_platform.execution_gateway.state_machine import resolve_execution_order_state
from quant_platform.execution_gateway.states import compute_execution_order_id, reconstruct_execution_order
from quant_platform.paper_trading.events import create_quote_event

_SHA_SESSION = "a" * 64
_SHA_INTENT = "b" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _setup(tmp_path):
    event_store = ExecutionSessionEventStore(tmp_path)
    adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
    adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
    return event_store, adapter


def _submit_command(*, side=OrderSide.BUY, quantity=Decimal("10"), sequence=0, intent_id=_SHA_INTENT):
    return create_submit_order_command(
        execution_session_id=_SHA_SESSION, execution_intent_id=intent_id, command_sequence=sequence, event_time=_NOW, instrument_id="EURUSD", side=side,
        quantity=quantity, order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=False, contract_multiplier=Decimal("1"),
    )


class TestDispatchCommandSubmit:
    def test_successful_submit_records_full_transaction_stages(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        command = _submit_command()
        outcome = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        assert outcome.resolution is not None
        assert outcome.resolution.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED
        ledger = event_store.read_events(_SHA_SESSION)
        kinds = [e.entry_kind for e in ledger]
        assert ExecutionLedgerEntryKind.COMMAND_CREATED in kinds
        assert ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT in kinds
        assert kinds.index(ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT) < kinds.index(ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED)

    def test_order_state_reaches_dispatched(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        ledger = event_store.read_events(_SHA_SESSION)
        order_id = compute_execution_order_id(command)
        from quant_platform.execution_gateway.state_machine import ExecutionOrderStateEvent

        events = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION]
        assert resolve_execution_order_state(order_id, events) == ExecutionOrderState.DISPATCHED

    def test_redispatching_identical_command_is_idempotent_no_op(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        command = _submit_command()
        first = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        ledger_after_first = event_store.read_events(_SHA_SESSION)
        second = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        ledger_after_second = event_store.read_events(_SHA_SESSION)
        assert second.was_already_resolved
        assert first.resolution is not None and second.resolution is not None
        assert first.resolution.entry_id == second.resolution.entry_id
        assert len(ledger_after_first) == len(ledger_after_second)  # no new ledger entries from the redispatch

    def test_synchronous_rejection_rule_produces_dispatch_rejected(self, tmp_path) -> None:
        from dataclasses import replace

        from quant_platform.execution_gateway.specs import RejectionRuleSpec

        rule = RejectionRuleSpec(rule_index=0, reject_instrument_id="EURUSD", reject_quantity_above=None, reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, rejection_rules=(rule,))
        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=scenario, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        outcome = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        assert outcome.resolution is not None
        assert outcome.resolution.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED


class TestBrokerEventProcessingAndFillReconstruction:
    def test_market_order_fills_and_reconstructs_correctly(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        command = _submit_command(quantity=Decimal("10"))
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)

        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        adapter.advance_market_event(tick, event_time=_NOW)
        result = process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
        assert not result.critical_conflict
        assert len(result.new_events) > 0

        ledger = event_store.read_events(_SHA_SESSION)
        order_id = compute_execution_order_id(command)
        from quant_platform.execution_gateway.events import BrokerEvent
        from quant_platform.execution_gateway.state_machine import ExecutionOrderStateEvent
        from quant_platform.execution_gateway.states import ExecutionFill

        state_events = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload["execution_order_id"] == order_id]
        broker_events = [BrokerEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED]
        fills = [ExecutionFill.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED]

        order = reconstruct_execution_order(submit_command=command, state_events=state_events, broker_events=broker_events, fills=fills)
        assert order.current_state is ExecutionOrderState.FILLED
        assert order.filled_quantity == Decimal("10")
        assert order.remaining_quantity == Decimal("0")
        assert len(fills) == 1
        assert fills[0].gross_notional == Decimal("10") * Decimal("1.1005")

    def test_duplicate_broker_event_is_absorbed_not_double_counted(self, tmp_path) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, duplicate_event_indices=(0,))
        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=scenario, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
        ledger = event_store.read_events(_SHA_SESSION)
        duplicate_entries = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE]
        received_entries = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED]
        assert len(duplicate_entries) == 1
        # The duplicated event's underlying broker_event_id must match a genuine RECEIVED entry -- proof it's recognized as the SAME event, not a new one.
        assert duplicate_entries[0].payload["broker_event_id"] in {e.payload["broker_event_id"] for e in received_entries}

    def test_processing_broker_events_is_idempotent_across_repeated_calls(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
        first_len = len(event_store.read_events(_SHA_SESSION))
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
        second_len = len(event_store.read_events(_SHA_SESSION))
        assert first_len == second_len  # nothing new to process -- no new ledger entries.


class TestCancelDispatch:
    def test_cancel_after_submit_transitions_order_to_cancelled(self, tmp_path) -> None:
        event_store, adapter = _setup(tmp_path)
        submit = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=submit, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)  # order reaches ACKNOWLEDGED

        order_id = compute_execution_order_id(submit)
        cancel = create_cancel_order_command(execution_session_id=_SHA_SESSION, execution_order_id=order_id, client_order_id=submit.client_order_id, cancellation_reason="operator_request", command_sequence=1, event_time=_NOW + timedelta(seconds=1))
        outcome = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=cancel, event_time=_NOW + timedelta(seconds=1))
        assert outcome.resolution is not None and outcome.resolution.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW + timedelta(seconds=1))

        ledger = event_store.read_events(_SHA_SESSION)
        from quant_platform.execution_gateway.state_machine import ExecutionOrderStateEvent

        state_events = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload["execution_order_id"] == order_id]
        assert resolve_execution_order_state(order_id, state_events) == ExecutionOrderState.CANCELLED


class TestReplaceDispatch:
    """Regression coverage for a real, confirmed defect found during
    Milestone 8's own acceptance testing: `REPLACE_ACKNOWLEDGED` was
    absent from `dispatcher._EVENT_TYPE_TARGET_STATE`, so a successfully
    replaced order was left PERMANENTLY stuck at `REPLACE_PENDING` --
    never resolving back to a live (fillable) state at all."""

    def test_replace_after_submit_returns_to_acknowledged_not_stuck_at_replace_pending(self, tmp_path) -> None:
        from quant_platform.execution_gateway.commands import create_replace_order_command
        from quant_platform.execution_gateway.state_machine import ExecutionOrderStateEvent

        event_store, adapter = _setup(tmp_path)
        submit = create_submit_order_command(
            execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, command_sequence=0, event_time=_NOW, instrument_id="EURUSD",
            side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderTypeKind.LIMIT, time_in_force=TimeInForceKind.DAY, reduce_only=False,
            contract_multiplier=Decimal("1"), limit_price=Decimal("50"),
        )
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=submit, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)  # order reaches ACKNOWLEDGED

        order_id = compute_execution_order_id(submit)
        replace = create_replace_order_command(
            execution_session_id=_SHA_SESSION, execution_order_id=order_id, client_order_id=submit.client_order_id, command_sequence=1,
            event_time=_NOW + timedelta(seconds=1), replacement_limit_price=Decimal("99"),
        )
        outcome = dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=replace, event_time=_NOW + timedelta(seconds=1))
        assert outcome.resolution is not None and outcome.resolution.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED

        ledger_before_ack = event_store.read_events(_SHA_SESSION)
        state_events_before = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger_before_ack if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload["execution_order_id"] == order_id]
        assert resolve_execution_order_state(order_id, state_events_before) == ExecutionOrderState.REPLACE_PENDING

        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW + timedelta(seconds=1))
        ledger_after_ack = event_store.read_events(_SHA_SESSION)
        state_events_after = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger_after_ack if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload["execution_order_id"] == order_id]
        assert resolve_execution_order_state(order_id, state_events_after) == ExecutionOrderState.ACKNOWLEDGED, "must resolve back to ACKNOWLEDGED, never remain stuck at REPLACE_PENDING"

        # The order must still be genuinely fillable after the replace -- at the NEW (99), not the original (50), limit price.
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=98.99, ask=99.0, source="test")
        adapter.advance_market_event(tick, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
        final_ledger = event_store.read_events(_SHA_SESSION)
        state_events_final = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in final_ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload["execution_order_id"] == order_id]
        assert resolve_execution_order_state(order_id, state_events_final) == ExecutionOrderState.FILLED
