"""Unit tests for `execution_gateway.models`'s execution-order state
machine and `execution_gateway.state_machine`'s event-sourced derivation
(Milestone 8, Section 8): every required legal transition, every
explicitly forbidden transition, and `resolve_execution_order_state`'s
replay correctness."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.core.exceptions import ExecutionOrderStateError
from quant_platform.execution_gateway.models import (
    TERMINAL_EXECUTION_ORDER_STATES,
    ExecutionOrderState,
    is_blocking_execution_order_state,
    is_legal_execution_order_transition,
    is_terminal_execution_order_state,
)
from quant_platform.execution_gateway.state_machine import (
    create_execution_order_state_event,
    resolve_execution_order_state,
)

_SHA_ORDER = "a" * 64
_SHA_SESSION = "b" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_REQUIRED_LEGAL_TRANSITIONS = [
    (ExecutionOrderState.CREATED, ExecutionOrderState.VALIDATED),
    (ExecutionOrderState.VALIDATED, ExecutionOrderState.DISPATCH_PENDING),
    (ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.DISPATCHED),
    (ExecutionOrderState.DISPATCHED, ExecutionOrderState.ACKNOWLEDGED),
    (ExecutionOrderState.DISPATCHED, ExecutionOrderState.REJECTED),
    (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED),
    (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.FILLED),
    (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.CANCEL_PENDING),
    (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.REPLACE_PENDING),
    (ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.PARTIALLY_FILLED),
    (ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED),
    (ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.CANCEL_PENDING),
    (ExecutionOrderState.CANCEL_PENDING, ExecutionOrderState.CANCELLED),
    (ExecutionOrderState.REPLACE_PENDING, ExecutionOrderState.ACKNOWLEDGED),
    (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.EXPIRED),
    (ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.EXPIRED),
]

_REQUIRED_AMBIGUITY_TRANSITIONS = [
    (ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.UNKNOWN),
    (ExecutionOrderState.DISPATCHED, ExecutionOrderState.UNKNOWN),
    (ExecutionOrderState.CANCEL_PENDING, ExecutionOrderState.UNKNOWN),
    (ExecutionOrderState.REPLACE_PENDING, ExecutionOrderState.UNKNOWN),
]

_FORBIDDEN_TRANSITIONS = [
    (ExecutionOrderState.FILLED, ExecutionOrderState.ACKNOWLEDGED),
    (ExecutionOrderState.CANCELLED, ExecutionOrderState.PARTIALLY_FILLED),
    (ExecutionOrderState.REJECTED, ExecutionOrderState.FILLED),
    (ExecutionOrderState.EXPIRED, ExecutionOrderState.ACKNOWLEDGED),
    (ExecutionOrderState.FAILED, ExecutionOrderState.DISPATCHED),
    (ExecutionOrderState.FILLED, ExecutionOrderState.CANCEL_PENDING),
]


class TestRequiredLegalTransitions:
    @pytest.mark.parametrize("current,target", _REQUIRED_LEGAL_TRANSITIONS)
    def test_required_transition_is_legal(self, current: ExecutionOrderState, target: ExecutionOrderState) -> None:
        assert is_legal_execution_order_transition(current, target)

    @pytest.mark.parametrize("current,target", _REQUIRED_AMBIGUITY_TRANSITIONS)
    def test_required_ambiguity_transition_is_legal(self, current: ExecutionOrderState, target: ExecutionOrderState) -> None:
        assert is_legal_execution_order_transition(current, target)


class TestForbiddenTransitionsNeverSilentlyPass:
    @pytest.mark.parametrize("current,target", _FORBIDDEN_TRANSITIONS)
    def test_forbidden_transition_is_illegal(self, current: ExecutionOrderState, target: ExecutionOrderState) -> None:
        assert not is_legal_execution_order_transition(current, target)

    @pytest.mark.parametrize("current,target", _FORBIDDEN_TRANSITIONS)
    def test_forbidden_transition_raises_when_constructed(self, current: ExecutionOrderState, target: ExecutionOrderState) -> None:
        with pytest.raises(ExecutionOrderStateError):
            create_execution_order_state_event(
                execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=current, to_state=target, event_time=_NOW, sequence=0,
                reason_code=("x" if target in {ExecutionOrderState.REJECTED, ExecutionOrderState.CANCELLED, ExecutionOrderState.EXPIRED, ExecutionOrderState.FAILED, ExecutionOrderState.UNKNOWN} else None),
            )


class TestTerminalAndBlockingClassification:
    def test_terminal_states_match_spec(self) -> None:
        assert {
            ExecutionOrderState.FILLED, ExecutionOrderState.CANCELLED, ExecutionOrderState.REJECTED, ExecutionOrderState.EXPIRED, ExecutionOrderState.FAILED,
        } == TERMINAL_EXECUTION_ORDER_STATES

    def test_unknown_is_not_terminal(self) -> None:
        assert not is_terminal_execution_order_state(ExecutionOrderState.UNKNOWN)

    def test_unknown_is_blocking(self) -> None:
        assert is_blocking_execution_order_state(ExecutionOrderState.UNKNOWN)

    def test_terminal_states_have_no_outgoing_legal_transitions(self) -> None:
        for state in TERMINAL_EXECUTION_ORDER_STATES:
            for target in ExecutionOrderState:
                assert not is_legal_execution_order_transition(state, target), f"{state} -> {target} should be illegal (terminal)"

    def test_unknown_always_has_a_path_forward(self) -> None:
        assert any(is_legal_execution_order_transition(ExecutionOrderState.UNKNOWN, target) for target in ExecutionOrderState)


class TestResolveExecutionOrderState:
    def test_empty_history_resolves_to_created(self) -> None:
        assert resolve_execution_order_state(_SHA_ORDER, []) == ExecutionOrderState.CREATED

    def test_replays_full_happy_path(self) -> None:
        events = []
        chain = [
            (ExecutionOrderState.CREATED, ExecutionOrderState.VALIDATED), (ExecutionOrderState.VALIDATED, ExecutionOrderState.DISPATCH_PENDING),
            (ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.DISPATCHED), (ExecutionOrderState.DISPATCHED, ExecutionOrderState.ACKNOWLEDGED),
            (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.FILLED),
        ]
        for i, (frm, to) in enumerate(chain):
            events.append(create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=frm, to_state=to, event_time=_NOW, sequence=i))
        assert resolve_execution_order_state(_SHA_ORDER, events) == ExecutionOrderState.FILLED

    def test_gapped_sequence_raises(self) -> None:
        # Two individually-legal events whose second from_state does not
        # match the first's to_state -- resolve must reject the replay,
        # not silently skip the gap.
        e1 = create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.CREATED, to_state=ExecutionOrderState.VALIDATED, event_time=_NOW, sequence=0)
        e2 = create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.DISPATCH_PENDING, to_state=ExecutionOrderState.DISPATCHED, event_time=_NOW, sequence=1)
        with pytest.raises(ExecutionOrderStateError):
            resolve_execution_order_state(_SHA_ORDER, [e1, e2])

    def test_event_from_another_order_is_rejected(self) -> None:
        e1 = create_execution_order_state_event(execution_order_id="c" * 64, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.CREATED, to_state=ExecutionOrderState.VALIDATED, event_time=_NOW, sequence=0)
        with pytest.raises(ExecutionOrderStateError):
            resolve_execution_order_state(_SHA_ORDER, [e1])

    def test_reason_code_required_for_rejected(self) -> None:
        with pytest.raises(ExecutionOrderStateError):
            create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.DISPATCHED, to_state=ExecutionOrderState.REJECTED, event_time=_NOW, sequence=0)

    def test_reason_code_forbidden_for_acknowledged(self) -> None:
        with pytest.raises(ExecutionOrderStateError):
            create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.DISPATCHED, to_state=ExecutionOrderState.ACKNOWLEDGED, event_time=_NOW, sequence=0, reason_code="unexpected")

    def test_unknown_resolves_to_acknowledged_after_query_confirms(self) -> None:
        chain = [
            (ExecutionOrderState.CREATED, ExecutionOrderState.VALIDATED), (ExecutionOrderState.VALIDATED, ExecutionOrderState.DISPATCH_PENDING),
            (ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.DISPATCHED),
        ]
        events = [
            create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=frm, to_state=to, event_time=_NOW, sequence=i)
            for i, (frm, to) in enumerate(chain)
        ]
        events.append(create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.DISPATCHED, to_state=ExecutionOrderState.UNKNOWN, event_time=_NOW, sequence=len(events), reason_code="ambiguous_dispatch"))
        events.append(create_execution_order_state_event(execution_order_id=_SHA_ORDER, execution_session_id=_SHA_SESSION, from_state=ExecutionOrderState.UNKNOWN, to_state=ExecutionOrderState.ACKNOWLEDGED, event_time=_NOW, sequence=len(events)))
        assert resolve_execution_order_state(_SHA_ORDER, events) == ExecutionOrderState.ACKNOWLEDGED
