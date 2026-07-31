"""Unit tests for `execution_gateway.events` (Section 14) and
`execution_gateway.states` (Section 9/10): `BrokerEvent` validation and
identity-based deduplication, `ExecutionFill` financial-identity and
dedup guarantees, and `reconstruct_execution_order`'s aggregate
invariants."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExecutionFillError, ExecutionOrderStateError
from quant_platform.execution_gateway.commands import create_submit_order_command
from quant_platform.execution_gateway.events import create_broker_event
from quant_platform.execution_gateway.models import (
    BrokerEventType,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.state_machine import (
    create_execution_order_state_event,
    resolve_execution_order_state,
)
from quant_platform.execution_gateway.states import (
    compute_execution_order_id,
    create_execution_fill,
    reconstruct_execution_order,
)

_SHA_SESSION = "a" * 64
_SHA_INTENT = "b" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _broker_event(**overrides: object) -> object:
    base: dict[str, object] = {
        "execution_session_id": _SHA_SESSION, "adapter_id": "dummy-1", "broker_sequence": 1, "event_type": BrokerEventType.ORDER_ACKNOWLEDGED,
        "broker_timestamp": _NOW, "received_event_time": _NOW, "broker_order_id": "bo-1", "client_order_id": "co-1", "original_quantity": Decimal("1"),
    }
    base.update(overrides)
    return create_broker_event(**base)  # type: ignore[arg-type]


class TestBrokerEventValidation:
    def test_valid_acknowledged_event(self) -> None:
        event = _broker_event()
        assert event.event_type is BrokerEventType.ORDER_ACKNOWLEDGED

    def test_rejected_event_requires_reject_code(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _broker_event(event_type=BrokerEventType.ORDER_REJECTED, original_quantity=None)

    def test_reject_code_forbidden_outside_reject_events(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _broker_event(reject_code="unsupported_order_type")

    def test_fill_event_requires_fill_fields(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _broker_event(event_type=BrokerEventType.ORDER_FILLED, original_quantity=None)

    def test_valid_fill_event(self) -> None:
        event = _broker_event(
            event_type=BrokerEventType.ORDER_FILLED, broker_fill_id="bf-1", fill_price=Decimal("1.1"), filled_quantity=Decimal("1"),
            cumulative_filled_quantity=Decimal("1"), remaining_quantity=Decimal("0"),
        )
        assert event.event_type is BrokerEventType.ORDER_FILLED  # type: ignore[attr-defined]

    def test_rejects_broker_sequence_below_one(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _broker_event(broker_sequence=0)

    def test_rejects_cumulative_remaining_original_mismatch(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _broker_event(original_quantity=Decimal("10"), cumulative_filled_quantity=Decimal("3"), remaining_quantity=Decimal("5"))

    def test_deduplication_identity_ignores_received_event_time(self) -> None:
        a = _broker_event(received_event_time=_NOW)
        from datetime import timedelta

        b = _broker_event(received_event_time=_NOW + timedelta(seconds=30))
        assert a.broker_event_id == b.broker_event_id  # type: ignore[attr-defined]

    def test_different_broker_sequence_gives_different_identity(self) -> None:
        a = _broker_event(broker_sequence=1)
        b = _broker_event(broker_sequence=2)
        assert a.broker_event_id != b.broker_event_id  # type: ignore[attr-defined]

    def test_identity_ignores_adapter_id(self) -> None:
        """Regression test for a real, confirmed defect found during
        Milestone 8's own acceptance testing: `adapter_id` -- a purely
        operational label for which adapter instance produced this event,
        with zero economic consequence -- used to participate in
        `broker_event_id` identity. Two economically-identical events
        reported by differently-NAMED adapter instances must produce the
        SAME `broker_event_id`, exactly like `received_event_time` is
        already excluded above."""
        a = _broker_event(adapter_id="adapter-one")
        b = _broker_event(adapter_id="adapter-two")
        assert a.broker_event_id == b.broker_event_id  # type: ignore[attr-defined]


class TestExecutionFillIdentityAndDedup:
    def _fill_inputs(self) -> dict[str, object]:
        broker_event = _broker_event(
            event_type=BrokerEventType.ORDER_FILLED, broker_fill_id="bf-1", fill_price=Decimal("1.1"), filled_quantity=Decimal("1"),
            cumulative_filled_quantity=Decimal("1"), remaining_quantity=Decimal("0"),
        )
        return {
            "execution_session_id": _SHA_SESSION, "execution_order_id": "c" * 64, "execution_intent_id": _SHA_INTENT, "broker_event": broker_event,
            "client_order_id": "co-1", "instrument_id": "EURUSD", "side": OrderSide.BUY, "quantity": Decimal("1"), "price": Decimal("1.1"),
            "contract_multiplier": Decimal("1"), "commission": Decimal("0.01"), "spread_component": Decimal("0.001"), "slippage_component": Decimal("0"),
            "fill_sequence": 0,
        }

    def test_gross_notional_is_exact(self) -> None:
        fill = create_execution_fill(**self._fill_inputs())  # type: ignore[arg-type]
        assert fill.gross_notional == Decimal("1") * Decimal("1.1") * Decimal("1")

    def test_duplicate_broker_event_produces_same_fill_id(self) -> None:
        inputs = self._fill_inputs()
        a = create_execution_fill(**inputs)  # type: ignore[arg-type]
        b = create_execution_fill(**inputs)  # type: ignore[arg-type]
        assert a.execution_fill_id == b.execution_fill_id

    def test_distinct_partial_fills_never_collide(self) -> None:
        inputs_a = self._fill_inputs()
        inputs_b = self._fill_inputs()
        inputs_b["fill_sequence"] = 1
        a = create_execution_fill(**inputs_a)  # type: ignore[arg-type]
        b = create_execution_fill(**inputs_b)  # type: ignore[arg-type]
        assert a.execution_fill_id != b.execution_fill_id

    def test_rejects_non_positive_quantity(self) -> None:
        inputs = self._fill_inputs()
        inputs["quantity"] = Decimal("0")
        with pytest.raises(ExecutionFillError):
            create_execution_fill(**inputs)  # type: ignore[arg-type]

    def test_rejects_wrong_gross_notional_via_direct_construction(self) -> None:
        from quant_platform.execution_gateway.states import ExecutionFill

        with pytest.raises(ExecutionFillError):
            ExecutionFill(
                execution_fill_id="0" * 64, execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, execution_intent_id=_SHA_INTENT,
                broker_event_id="d" * 64, broker_fill_id="bf-1", broker_order_id="bo-1", client_order_id="co-1", instrument_id="EURUSD", side=OrderSide.BUY,
                quantity=Decimal("1"), price=Decimal("1.1"), contract_multiplier=Decimal("1"), gross_notional=Decimal("999"), commission=Decimal("0"),
                spread_component=Decimal("0"), slippage_component=Decimal("0"), broker_timestamp=_NOW, received_event_time=_NOW, fill_sequence=0,
            )


class TestReconstructExecutionOrder:
    def _submit(self) -> object:
        return create_submit_order_command(
            execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, command_sequence=0, event_time=_NOW, instrument_id="EURUSD",
            side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=False,
            contract_multiplier=Decimal("1"),
        )

    def test_freshly_created_order_has_zero_fills(self) -> None:
        submit = self._submit()
        order = reconstruct_execution_order(submit_command=submit, state_events=[], broker_events=[], fills=[])  # type: ignore[arg-type]
        assert order.filled_quantity == Decimal("0")
        assert order.remaining_quantity == Decimal("10")
        assert order.current_state is ExecutionOrderState.CREATED

    def test_partial_then_full_fill_reconciles_exactly(self) -> None:
        submit = self._submit()
        order_id = compute_execution_order_id(submit)  # type: ignore[arg-type]
        chain = [
            (ExecutionOrderState.CREATED, ExecutionOrderState.VALIDATED), (ExecutionOrderState.VALIDATED, ExecutionOrderState.DISPATCH_PENDING),
            (ExecutionOrderState.DISPATCH_PENDING, ExecutionOrderState.DISPATCHED), (ExecutionOrderState.DISPATCHED, ExecutionOrderState.ACKNOWLEDGED),
            (ExecutionOrderState.ACKNOWLEDGED, ExecutionOrderState.PARTIALLY_FILLED), (ExecutionOrderState.PARTIALLY_FILLED, ExecutionOrderState.FILLED),
        ]
        state_events = [
            create_execution_order_state_event(execution_order_id=order_id, execution_session_id=_SHA_SESSION, from_state=frm, to_state=to, event_time=_NOW, sequence=i)
            for i, (frm, to) in enumerate(chain)
        ]
        broker_event_1 = create_broker_event(
            execution_session_id=_SHA_SESSION, adapter_id="dummy-1", broker_sequence=1, event_type=BrokerEventType.ORDER_PARTIALLY_FILLED, broker_timestamp=_NOW,
            received_event_time=_NOW, broker_order_id="bo-1", broker_fill_id="bf-1", client_order_id=submit.client_order_id, fill_price=Decimal("1.1"),  # type: ignore[attr-defined]
            filled_quantity=Decimal("4"), cumulative_filled_quantity=Decimal("4"), remaining_quantity=Decimal("6"),
        )
        broker_event_2 = create_broker_event(
            execution_session_id=_SHA_SESSION, adapter_id="dummy-1", broker_sequence=2, event_type=BrokerEventType.ORDER_FILLED, broker_timestamp=_NOW,
            received_event_time=_NOW, broker_order_id="bo-1", broker_fill_id="bf-2", client_order_id=submit.client_order_id, fill_price=Decimal("1.2"),  # type: ignore[attr-defined]
            filled_quantity=Decimal("6"), cumulative_filled_quantity=Decimal("10"), remaining_quantity=Decimal("0"),
        )
        fill_1 = create_execution_fill(
            execution_session_id=_SHA_SESSION, execution_order_id=order_id, execution_intent_id=_SHA_INTENT, broker_event=broker_event_1,
            client_order_id=submit.client_order_id, instrument_id="EURUSD", side=OrderSide.BUY, quantity=Decimal("4"), price=Decimal("1.1"),  # type: ignore[attr-defined]
            contract_multiplier=Decimal("1"), commission=Decimal("0"), spread_component=Decimal("0"), slippage_component=Decimal("0"), fill_sequence=0,
        )
        fill_2 = create_execution_fill(
            execution_session_id=_SHA_SESSION, execution_order_id=order_id, execution_intent_id=_SHA_INTENT, broker_event=broker_event_2,
            client_order_id=submit.client_order_id, instrument_id="EURUSD", side=OrderSide.BUY, quantity=Decimal("6"), price=Decimal("1.2"),  # type: ignore[attr-defined]
            contract_multiplier=Decimal("1"), commission=Decimal("0"), spread_component=Decimal("0"), slippage_component=Decimal("0"), fill_sequence=1,
        )
        order = reconstruct_execution_order(submit_command=submit, state_events=state_events, broker_events=[broker_event_1, broker_event_2], fills=[fill_1, fill_2])  # type: ignore[arg-type]
        assert order.current_state is ExecutionOrderState.FILLED
        assert order.filled_quantity == Decimal("10")
        assert order.remaining_quantity == Decimal("0")
        assert order.filled_quantity + order.remaining_quantity == order.current_quantity
        assert order.average_fill_price == (Decimal("4") * Decimal("1.1") + Decimal("6") * Decimal("1.2")) / Decimal("10")
        assert order.broker_order_id == "bo-1"
        assert resolve_execution_order_state(order_id, state_events) == ExecutionOrderState.FILLED

    def test_cumulative_fill_cannot_exceed_accepted_quantity(self) -> None:
        from quant_platform.execution_gateway.states import ExecutionOrder

        with pytest.raises(ExecutionOrderStateError):
            ExecutionOrder(
                execution_order_id="c" * 64, execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, client_order_id="co-1",
                broker_order_id="bo-1", instrument_id="EURUSD", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY,
                original_quantity=Decimal("10"), current_quantity=Decimal("10"), filled_quantity=Decimal("11"), remaining_quantity=Decimal("-1"),
                limit_price=None, stop_price=None, average_fill_price=Decimal("1"), reduce_only=False, current_state=ExecutionOrderState.FILLED,
                last_broker_sequence=1, last_broker_event_id="d" * 64, created_event_time=_NOW, last_updated_event_time=_NOW,
            )

    def test_fok_order_can_never_be_observed_partially_filled(self) -> None:
        from quant_platform.execution_gateway.states import ExecutionOrder

        with pytest.raises(ExecutionOrderStateError):
            ExecutionOrder(
                execution_order_id="c" * 64, execution_session_id=_SHA_SESSION, execution_intent_id=_SHA_INTENT, client_order_id="co-1",
                broker_order_id="bo-1", instrument_id="EURUSD", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.FOK,
                original_quantity=Decimal("10"), current_quantity=Decimal("10"), filled_quantity=Decimal("4"), remaining_quantity=Decimal("6"),
                limit_price=None, stop_price=None, average_fill_price=Decimal("1"), reduce_only=False, current_state=ExecutionOrderState.PARTIALLY_FILLED,
                last_broker_sequence=1, last_broker_event_id="d" * 64, created_event_time=_NOW, last_updated_event_time=_NOW,
            )
