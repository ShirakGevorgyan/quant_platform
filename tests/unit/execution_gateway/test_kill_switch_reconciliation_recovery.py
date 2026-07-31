"""Unit tests for `execution_gateway.kill_switch` (Section 22),
`execution_gateway.reconciliation` (Section 24), and
`execution_gateway.recovery` (Section 23)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExecutionHaltError
from quant_platform.execution_gateway.commands import (
    create_cancel_order_command,
    create_heartbeat_command,
    create_submit_order_command,
)
from quant_platform.execution_gateway.dispatcher import dispatch_command, process_broker_events
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.kill_switch import (
    ExecutionKillSwitchTriggerKind,
    authorize_dispatch,
    create_execution_kill_switch_transition_event,
    resolve_execution_kill_switch_state,
)
from quant_platform.execution_gateway.models import (
    ExecutionKillSwitchState,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.reconciliation import (
    ReconciliationSeverity,
    reconcile_execution_session,
)
from quant_platform.execution_gateway.recovery import recover_unknown_orders
from quant_platform.execution_gateway.specs import DEFAULT_DUMMY_BROKER_SCENARIO, ReconciliationPolicySpec
from quant_platform.execution_gateway.state_machine import resolve_execution_order_state
from quant_platform.execution_gateway.states import compute_execution_order_id
from quant_platform.paper_trading.events import create_quote_event

_SHA_SESSION = "a" * 64
_SHA_INTENT = "b" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_POLICY = ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True)


def _submit_command(sequence=0):
    return create_submit_order_command(
        execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, command_sequence=sequence, event_time=_NOW, instrument_id="EURUSD",
        side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=False,
        contract_multiplier=Decimal("1"),
    )


class TestKillSwitchTransitions:
    def test_active_to_halting_to_halted(self) -> None:
        e1 = create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.ACTIVE, to_state=ExecutionKillSwitchState.HALTING, trigger=ExecutionKillSwitchTriggerKind.OPERATOR_REQUEST, event_time=_NOW, sequence=0, detail="test")
        e2 = create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.HALTING, to_state=ExecutionKillSwitchState.HALTED, trigger=ExecutionKillSwitchTriggerKind.OPERATOR_REQUEST, event_time=_NOW, sequence=1, detail="test")
        assert resolve_execution_kill_switch_state([e1, e2]) == ExecutionKillSwitchState.HALTED

    def test_halted_never_returns_directly_to_active(self) -> None:
        with pytest.raises(ExecutionHaltError):
            create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.HALTED, to_state=ExecutionKillSwitchState.ACTIVE, trigger=ExecutionKillSwitchTriggerKind.RECOVERED, event_time=_NOW, sequence=0, detail="illegal")

    def test_recovering_can_return_to_active(self) -> None:
        e1 = create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.ACTIVE, to_state=ExecutionKillSwitchState.DEGRADED, trigger=ExecutionKillSwitchTriggerKind.ADAPTER_STALE, event_time=_NOW, sequence=0, detail="x")
        e2 = create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.DEGRADED, to_state=ExecutionKillSwitchState.RECOVERING, trigger=ExecutionKillSwitchTriggerKind.RECOVERED, event_time=_NOW, sequence=1, detail="x")
        e3 = create_execution_kill_switch_transition_event(execution_session_id=_SHA_SESSION, from_state=ExecutionKillSwitchState.RECOVERING, to_state=ExecutionKillSwitchState.ACTIVE, trigger=ExecutionKillSwitchTriggerKind.RECOVERED, event_time=_NOW, sequence=2, detail="x")
        assert resolve_execution_kill_switch_state([e1, e2, e3]) == ExecutionKillSwitchState.ACTIVE


class TestAuthorizeDispatch:
    def test_new_exposure_submit_allowed_when_active(self) -> None:
        authorize_dispatch(ExecutionKillSwitchState.ACTIVE, _submit_command())

    def test_new_exposure_submit_blocked_when_degraded(self) -> None:
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(ExecutionKillSwitchState.DEGRADED, _submit_command())

    def test_new_exposure_submit_blocked_when_halted(self) -> None:
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(ExecutionKillSwitchState.HALTED, _submit_command())

    def test_reduce_only_submit_allowed_when_halting(self) -> None:
        command = create_submit_order_command(execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, command_sequence=0, event_time=_NOW, instrument_id="EURUSD", side=OrderSide.SELL, quantity=Decimal("5"), order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=True, contract_multiplier=Decimal("1"))
        authorize_dispatch(ExecutionKillSwitchState.HALTING, command)  # must not raise -- safety exception

    def test_reduce_only_submit_blocked_when_halted(self) -> None:
        command = create_submit_order_command(execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, command_sequence=0, event_time=_NOW, instrument_id="EURUSD", side=OrderSide.SELL, quantity=Decimal("5"), order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=True, contract_multiplier=Decimal("1"))
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(ExecutionKillSwitchState.HALTED, command)

    def test_cancel_allowed_when_halting(self) -> None:
        cancel = create_cancel_order_command(execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, client_order_id="cid", cancellation_reason="risk_halt", command_sequence=0, event_time=_NOW)
        authorize_dispatch(ExecutionKillSwitchState.HALTING, cancel)  # must not raise

    def test_cancel_blocked_when_halted(self) -> None:
        cancel = create_cancel_order_command(execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, client_order_id="cid", cancellation_reason="risk_halt", command_sequence=0, event_time=_NOW)
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(ExecutionKillSwitchState.HALTED, cancel)

    def test_query_and_heartbeat_allowed_even_when_halted(self) -> None:
        heartbeat = create_heartbeat_command(execution_session_id=_SHA_SESSION, command_sequence=0, event_time=_NOW)
        authorize_dispatch(ExecutionKillSwitchState.HALTED, heartbeat)  # must not raise

    def test_query_blocked_when_terminated(self) -> None:
        heartbeat = create_heartbeat_command(execution_session_id=_SHA_SESSION, command_sequence=0, event_time=_NOW)
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(ExecutionKillSwitchState.TERMINATED, heartbeat)


class TestReconciliationCleanSession:
    def test_clean_filled_session_reconciles(self, tmp_path) -> None:
        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        adapter.advance_market_event(tick, event_time=_NOW)
        process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)

        ledger = event_store.read_events(_SHA_SESSION)
        report = reconcile_execution_session(execution_session_id=_SHA_SESSION, ledger=ledger, adapter=adapter, event_time=_NOW, policy=_POLICY)
        assert report.is_reconciled, [i.to_json_dict() for i in report.blocking_issues]

    def test_reconciliation_never_raises_on_corrupted_ledger(self, tmp_path) -> None:
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        # Empty ledger -- degenerate but must not raise.
        report = reconcile_execution_session(execution_session_id=_SHA_SESSION, ledger=[], adapter=adapter, event_time=_NOW, policy=_POLICY)
        assert isinstance(report.is_reconciled, bool)

    def test_session_ownership_mismatch_detected(self, tmp_path) -> None:
        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)
        ledger = event_store.read_events(_SHA_SESSION)
        report = reconcile_execution_session(execution_session_id="f" * 64, ledger=ledger, adapter=adapter, event_time=_NOW, policy=_POLICY)
        codes = {i.issue_code for i in report.issues}
        assert "session_ownership_mismatch" in codes
        assert any(i.severity is ReconciliationSeverity.CRITICAL for i in report.issues if i.issue_code == "session_ownership_mismatch")


class TestRecoveryResolvesUnknown:
    def test_recovery_resolves_unknown_order_via_query_when_broker_confirms(self, tmp_path) -> None:
        from quant_platform.execution_gateway.dispatcher import (
            _append_order_transition,
            _order_state_events_from_ledger,
        )

        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command, event_time=_NOW)  # order reaches DISPATCHED, and the dummy broker synchronously acknowledges it internally too

        # Simulate a genuine mid-dispatch crash-and-resume ambiguity: force
        # the LEDGER's own view to UNKNOWN even though the broker already
        # has the order ACKNOWLEDGED -- exactly the "ambiguous, but broker
        # confirms" recovery scenario.
        order_id = compute_execution_order_id(command)
        ledger = event_store.read_events(_SHA_SESSION)
        events = _order_state_events_from_ledger(ledger)
        events = [e for e in events if e.execution_order_id == order_id]
        _append_order_transition(event_store, _SHA_SESSION, execution_order_id=order_id, existing_events=events, from_state=ExecutionOrderState.DISPATCHED, to_state=ExecutionOrderState.UNKNOWN, event_time=_NOW, reason_code="ambiguous_dispatch")

        actions = recover_unknown_orders(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=_NOW)
        assert len(actions) == 1
        assert actions[0].action == "resolved_by_query"

        final_ledger = event_store.read_events(_SHA_SESSION)
        final_events = _order_state_events_from_ledger(final_ledger)
        final_events = [e for e in final_events if e.execution_order_id == order_id]
        assert resolve_execution_order_state(order_id, final_events) == ExecutionOrderState.ACKNOWLEDGED

    def test_recovery_authorizes_safe_retry_when_broker_has_no_record_and_idempotent(self, tmp_path) -> None:
        from quant_platform.execution_gateway.dispatcher import (
            _append_order_transition,
        )

        event_store = ExecutionSessionEventStore(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        command = _submit_command()
        # Record the COMMAND_CREATED/VALIDATED/DISPATCH_PENDING/UNKNOWN
        # chain WITHOUT ever actually calling adapter.submit_order --
        # simulating a crash strictly between "dispatch intent recorded"
        # and "adapter call made", so the broker genuinely has no record.
        from quant_platform.execution_gateway.dispatcher import _append
        from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind

        _append(event_store, _SHA_SESSION, kind=ExecutionLedgerEntryKind.COMMAND_CREATED, payload=command.to_json_dict(), event_time=_NOW)
        order_id = compute_execution_order_id(command)
        events: list = []
        _append_order_transition(event_store, _SHA_SESSION, execution_order_id=order_id, existing_events=events, from_state=ExecutionOrderState.CREATED, to_state=ExecutionOrderState.VALIDATED, event_time=_NOW)
        _append_order_transition(event_store, _SHA_SESSION, execution_order_id=order_id, existing_events=events, from_state=ExecutionOrderState.VALIDATED, to_state=ExecutionOrderState.DISPATCH_PENDING, event_time=_NOW)
        _append_order_transition(event_store, _SHA_SESSION, execution_order_id=order_id, existing_events=events, from_state=ExecutionOrderState.DISPATCH_PENDING, to_state=ExecutionOrderState.UNKNOWN, event_time=_NOW, reason_code="ambiguous_dispatch")

        actions = recover_unknown_orders(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=_NOW)
        assert len(actions) == 1
        assert actions[0].action == "safe_retry_authorized"
