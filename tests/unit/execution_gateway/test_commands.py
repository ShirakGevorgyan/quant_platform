"""Unit tests for `execution_gateway.commands` (Milestone 8, Section 7):
command validation, deterministic command identity, and client-order-id
stability across safe retries."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExecutionCommandValidationError
from quant_platform.execution_gateway.commands import (
    create_cancel_order_command,
    create_heartbeat_command,
    create_query_account_command,
    create_query_open_orders_command,
    create_query_order_command,
    create_query_positions_command,
    create_replace_order_command,
    create_submit_order_command,
    derive_client_order_id,
)
from quant_platform.execution_gateway.models import OrderSide, OrderTypeKind, TimeInForceKind

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _submit(**overrides: object) -> object:
    base: dict[str, object] = {
        "execution_session_id": _SHA_A, "execution_intent_id": _SHA_B, "command_sequence": 0, "event_time": _NOW, "instrument_id": "EURUSD",
        "side": OrderSide.BUY, "quantity": Decimal("1"), "order_type": OrderTypeKind.MARKET, "time_in_force": TimeInForceKind.DAY, "reduce_only": False,
        "contract_multiplier": Decimal("1"),
    }
    base.update(overrides)
    return create_submit_order_command(**base)  # type: ignore[arg-type]


class TestClientOrderIdDerivation:
    def test_deterministic(self) -> None:
        assert derive_client_order_id(_SHA_A) == derive_client_order_id(_SHA_A)

    def test_differs_by_intent(self) -> None:
        assert derive_client_order_id(_SHA_A) != derive_client_order_id(_SHA_B)

    def test_looks_like_sha256(self) -> None:
        cid = derive_client_order_id(_SHA_A)
        assert len(cid) == 64
        int(cid, 16)  # does not raise


class TestSubmitOrderCommandValidation:
    def test_valid_market_submit(self) -> None:
        cmd = _submit()
        assert cmd.client_order_id == derive_client_order_id(_SHA_B)  # type: ignore[attr-defined]

    def test_market_rejects_limit_price(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            _submit(limit_price=Decimal("1.1"))

    def test_limit_requires_limit_price(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            _submit(order_type=OrderTypeKind.LIMIT)

    def test_stop_requires_stop_price(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            _submit(order_type=OrderTypeKind.STOP)

    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            _submit(quantity=Decimal("0"))

    def test_rejects_non_positive_contract_multiplier(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            _submit(contract_multiplier=Decimal("-1"))

    def test_round_trips_through_json(self) -> None:
        cmd = _submit()
        from quant_platform.execution_gateway.commands import SubmitOrderCommand

        restored = SubmitOrderCommand.from_json_dict(cmd.to_json_dict())  # type: ignore[attr-defined]
        assert restored.to_json_dict() == cmd.to_json_dict()  # type: ignore[attr-defined]


class TestCommandIdentityStableAcrossSafeRetry:
    def test_submit_retry_with_identical_payload_reuses_command_id(self) -> None:
        first = _submit(command_sequence=0)
        second = _submit(command_sequence=5)  # different operational sequence, same economics
        assert first.command_id == second.command_id  # type: ignore[attr-defined]
        assert first.client_order_id == second.client_order_id  # type: ignore[attr-defined]

    def test_submit_with_different_quantity_gets_different_command_id(self) -> None:
        first = _submit(quantity=Decimal("1"))
        second = _submit(quantity=Decimal("2"))
        assert first.command_id != second.command_id  # type: ignore[attr-defined]

    def test_submit_with_different_intent_gets_different_command_id_and_client_order_id(self) -> None:
        first = _submit(execution_intent_id=_SHA_B)
        second = _submit(execution_intent_id=_SHA_C)
        assert first.command_id != second.command_id  # type: ignore[attr-defined]
        assert first.client_order_id != second.client_order_id  # type: ignore[attr-defined]

    def test_cancel_retry_with_identical_payload_reuses_command_id(self) -> None:
        first = create_cancel_order_command(
            execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", cancellation_reason="risk_halt", command_sequence=0, event_time=_NOW,
        )
        second = create_cancel_order_command(
            execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", cancellation_reason="risk_halt", command_sequence=99, event_time=_NOW,
        )
        assert first.command_id == second.command_id

    def test_cancel_with_different_order_gets_different_command_id(self) -> None:
        a = create_cancel_order_command(execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", cancellation_reason="x", command_sequence=0, event_time=_NOW)
        b = create_cancel_order_command(execution_session_id=_SHA_A, execution_order_id=_SHA_C, client_order_id="cid", cancellation_reason="x", command_sequence=0, event_time=_NOW)
        assert a.command_id != b.command_id

    def test_replace_retry_with_identical_payload_reuses_command_id(self) -> None:
        first = create_replace_order_command(execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", command_sequence=0, event_time=_NOW, replacement_quantity=Decimal("2"))
        second = create_replace_order_command(execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", command_sequence=7, event_time=_NOW, replacement_quantity=Decimal("2"))
        assert first.command_id == second.command_id

    def test_replace_requires_at_least_one_change(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            create_replace_order_command(execution_session_id=_SHA_A, execution_order_id=_SHA_B, client_order_id="cid", command_sequence=0, event_time=_NOW)

    def test_query_order_requires_an_identifier(self) -> None:
        with pytest.raises(ExecutionCommandValidationError):
            create_query_order_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW)

    def test_query_commands_are_sequence_scoped(self) -> None:
        a = create_query_open_orders_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW)
        b = create_query_open_orders_command(execution_session_id=_SHA_A, command_sequence=1, event_time=_NOW)
        assert a.command_id != b.command_id
        c = create_query_open_orders_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW)
        assert a.command_id == c.command_id

    def test_query_positions_query_account_heartbeat_construct(self) -> None:
        assert create_query_positions_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW).command_id
        assert create_query_account_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW).command_id
        assert create_heartbeat_command(execution_session_id=_SHA_A, command_sequence=0, event_time=_NOW).command_id
