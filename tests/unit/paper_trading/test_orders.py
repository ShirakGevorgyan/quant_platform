"""Milestone 7, Section 8: `OrderRequest` construction/validation and the
`OrderState` machine's event-sourced derivation (`resolve_order_state`).
Every legal/illegal transition in `_LEGAL_ORDER_TRANSITIONS` is exercised
at least once, plus the two documented terminal-state properties (no
transition out of a terminal state; a REJECTED/CANCELLED/EXPIRED event
always carries a reason code, no other event ever does)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import OrderStateError, OrderValidationError
from quant_platform.paper_trading.models import (
    TERMINAL_ORDER_STATES,
    OrderSide,
    OrderState,
    OrderTypeKind,
    PositionIntentKind,
    RejectReasonKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import (
    OrderRequest,
    OrderStateEvent,
    create_order_request,
    create_order_state_event,
    is_legal_order_transition,
    is_order_in_terminal_state,
    resolve_order_state,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_DECISION = "a" * 64


def _order(**overrides: object) -> OrderRequest:
    defaults: dict[str, object] = {
        "client_order_id": "client-1", "session_id": "session-1", "strategy_decision_id": _HEX_DECISION, "instrument": "X",
        "side": OrderSide.BUY, "order_type": OrderTypeKind.MARKET, "quantity": 1.0, "time_in_force": TimeInForceKind.DAY,
        "create_time": _T0, "submit_time": _T0, "reduce_only": False, "position_intent": PositionIntentKind.OPEN,
    }
    defaults.update(overrides)
    return create_order_request(**defaults)  # type: ignore[arg-type]


class TestOrderRequestValidation:
    def test_valid_market_order(self) -> None:
        order = _order()
        assert order.order_type is OrderTypeKind.MARKET

    def test_market_order_with_limit_price_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="MARKET"):
            _order(order_type=OrderTypeKind.MARKET, limit_price=100.0)

    def test_limit_order_requires_limit_price(self) -> None:
        with pytest.raises(OrderValidationError, match="limit_price"):
            _order(order_type=OrderTypeKind.LIMIT)

    def test_limit_order_with_stop_price_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="stop_price"):
            _order(order_type=OrderTypeKind.LIMIT, limit_price=100.0, stop_price=99.0)

    def test_valid_limit_order(self) -> None:
        order = _order(order_type=OrderTypeKind.LIMIT, limit_price=100.0)
        assert order.limit_price == 100.0

    def test_stop_order_requires_stop_price(self) -> None:
        with pytest.raises(OrderValidationError, match="stop_price"):
            _order(order_type=OrderTypeKind.STOP)

    def test_valid_stop_order(self) -> None:
        order = _order(order_type=OrderTypeKind.STOP, stop_price=95.0)
        assert order.stop_price == 95.0

    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="quantity"):
            _order(quantity=0.0)

    def test_invalid_strategy_decision_id_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="strategy_decision_id"):
            _order(strategy_decision_id="not-a-hash")

    def test_empty_client_order_id_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="client_order_id"):
            _order(client_order_id="")

    def test_submit_time_before_create_time_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="submit_time"):
            _order(create_time=_T0, submit_time=_T0 - timedelta(seconds=1))

    def test_naive_create_time_rejected(self) -> None:
        with pytest.raises(OrderValidationError, match="timezone-aware"):
            _order(create_time=datetime(2026, 1, 5, 10, 0, 0), submit_time=datetime(2026, 1, 5, 10, 0, 0))

    def test_json_round_trip(self) -> None:
        order = _order(order_type=OrderTypeKind.LIMIT, limit_price=101.5)
        assert OrderRequest.from_json_dict(order.to_json_dict()) == order


class TestOrderRequestIdentity:
    def test_identical_arguments_produce_identical_order_id(self) -> None:
        assert _order().order_id == _order().order_id

    def test_different_quantity_changes_order_id(self) -> None:
        assert _order(quantity=1.0).order_id != _order(quantity=2.0).order_id

    def test_different_client_order_id_changes_order_id(self) -> None:
        assert _order(client_order_id="a").order_id != _order(client_order_id="b").order_id


class TestOrderStateTransitionTable:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (OrderState.CREATED, OrderState.VALIDATED), (OrderState.CREATED, OrderState.REJECTED),
            (OrderState.VALIDATED, OrderState.ACCEPTED), (OrderState.VALIDATED, OrderState.REJECTED),
            (OrderState.ACCEPTED, OrderState.WORKING), (OrderState.ACCEPTED, OrderState.REJECTED),
            (OrderState.WORKING, OrderState.PARTIALLY_FILLED), (OrderState.WORKING, OrderState.FILLED),
            (OrderState.WORKING, OrderState.CANCEL_REQUESTED), (OrderState.WORKING, OrderState.CANCELLED), (OrderState.WORKING, OrderState.EXPIRED),
            (OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED), (OrderState.PARTIALLY_FILLED, OrderState.FILLED),
            (OrderState.PARTIALLY_FILLED, OrderState.CANCEL_REQUESTED), (OrderState.PARTIALLY_FILLED, OrderState.CANCELLED),
            (OrderState.PARTIALLY_FILLED, OrderState.EXPIRED), (OrderState.CANCEL_REQUESTED, OrderState.CANCELLED),
            (OrderState.CANCEL_REQUESTED, OrderState.PARTIALLY_FILLED), (OrderState.CANCEL_REQUESTED, OrderState.FILLED),
            (OrderState.CANCEL_REQUESTED, OrderState.EXPIRED),
        ],
    )
    def test_legal_transitions(self, current: OrderState, target: OrderState) -> None:
        assert is_legal_order_transition(current, target)

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_ORDER_STATES, key=lambda s: s.value))
    def test_no_transition_out_of_terminal_states(self, terminal: OrderState) -> None:
        for target in OrderState:
            assert not is_legal_order_transition(terminal, target)

    def test_illegal_skip_from_created_to_working_rejected(self) -> None:
        assert not is_legal_order_transition(OrderState.CREATED, OrderState.WORKING)

    def test_illegal_backward_transition_rejected(self) -> None:
        assert not is_legal_order_transition(OrderState.FILLED, OrderState.WORKING)


class TestOrderStateEventValidation:
    def test_valid_transition_event(self) -> None:
        event = create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1)
        assert event.to_state is OrderState.VALIDATED

    def test_illegal_transition_rejected(self) -> None:
        with pytest.raises(OrderStateError, match="Illegal"):
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.FILLED, event_time=_T0, sequence=1)

    def test_reject_transition_requires_reason_code(self) -> None:
        with pytest.raises(OrderStateError, match="reason_code"):
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.REJECTED, event_time=_T0, sequence=1)

    def test_reject_transition_with_reason_code_succeeds(self) -> None:
        event = create_order_state_event(
            order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.REJECTED, event_time=_T0, sequence=1,
            reason_code=RejectReasonKind.NON_POSITIVE_QUANTITY,
        )
        assert event.reason_code is RejectReasonKind.NON_POSITIVE_QUANTITY

    def test_non_reject_transition_with_reason_code_rejected(self) -> None:
        with pytest.raises(OrderStateError, match="reason_code"):
            create_order_state_event(
                order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1,
                reason_code=RejectReasonKind.NON_POSITIVE_QUANTITY,
            )

    def test_negative_sequence_rejected(self) -> None:
        with pytest.raises(OrderStateError, match="sequence"):
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=-1)

    def test_json_round_trip(self) -> None:
        event = create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1)
        assert OrderStateEvent.from_json_dict(event.to_json_dict()) == event


class TestResolveOrderState:
    def test_no_events_resolves_to_created(self) -> None:
        assert resolve_order_state("o1", []) == OrderState.CREATED

    def test_full_lifecycle_to_filled(self) -> None:
        events = [
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=_T0, sequence=2),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=_T0, sequence=3),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.WORKING, to_state=OrderState.FILLED, event_time=_T0, sequence=4),
        ]
        assert resolve_order_state("o1", events) == OrderState.FILLED
        assert is_order_in_terminal_state("o1", events)

    def test_partial_fill_sequence(self) -> None:
        events = [
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=_T0, sequence=2),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=_T0, sequence=3),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.WORKING, to_state=OrderState.PARTIALLY_FILLED, event_time=_T0, sequence=4),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.PARTIALLY_FILLED, to_state=OrderState.PARTIALLY_FILLED, event_time=_T0, sequence=5),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.PARTIALLY_FILLED, to_state=OrderState.FILLED, event_time=_T0, sequence=6),
        ]
        assert resolve_order_state("o1", events) == OrderState.FILLED

    def test_mismatched_order_id_in_event_sequence_rejected(self) -> None:
        events = [create_order_state_event(order_id="o2", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1)]
        with pytest.raises(OrderStateError, match="different order"):
            resolve_order_state("o1", events)

    def test_out_of_order_from_state_rejected(self) -> None:
        """An event claiming `from_state=WORKING` when the order is still
        actually at `CREATED` (e.g. a corrupted/reordered ledger) must be
        rejected, not silently accepted."""
        events = [create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.WORKING, to_state=OrderState.FILLED, event_time=_T0, sequence=1)]
        with pytest.raises(OrderStateError, match="from_state"):
            resolve_order_state("o1", events)

    def test_event_out_of_terminal_state_cannot_even_be_constructed(self) -> None:
        """A ledger can never contain an event appended after a terminal
        state was reached at all -- `OrderStateEvent.__post_init__`
        already rejects the illegal transition at construction time
        (defense in depth), so `resolve_order_state` can never actually
        observe one; `TestOrderStateEventValidation.test_illegal_
        transition_rejected` covers the construction-time rejection."""
        with pytest.raises(OrderStateError, match="Illegal"):
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.REJECTED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=2)

    def test_not_in_terminal_state_while_working(self) -> None:
        events = [
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=_T0, sequence=1),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=_T0, sequence=2),
            create_order_state_event(order_id="o1", session_id="s1", from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=_T0, sequence=3),
        ]
        assert not is_order_in_terminal_state("o1", events)
