"""Immutable execution commands (Milestone 8, Section 7). A `Command` is
the ONLY thing this package ever hands to an adapter (`adapter.py`) --
`ExecutionIntent` (Section 5, `paper_bridge.py`) expresses WHAT should
happen economically; a `Command` is the concrete, adapter-facing
operation that carries it out (or queries/heartbeats, which carry no
economic intent at all).

COMMAND IDENTITY (Section 7's own requirements) is computed from each
command's OWN ECONOMIC PAYLOAD ONLY -- `command_sequence`/`event_time`
are operational metadata, deliberately excluded, exactly like a ledger
entry's own `entry_id`/`checksum`/`event_time` are excluded from
`paper_trading.persistence.compute_ledger_semantic_digest`. This is what
makes command identity STABLE ACROSS A SAFE RETRY: dispatching the SAME
economic submit/cancel/replace operation a second time (because the
first attempt's outcome was ambiguous) recomputes the IDENTICAL
`command_id`, so `persistence.py`'s idempotent ledger append recognizes
it as the same operation rather than a new one. A genuinely DIFFERENT
economic operation (different intent, different order, different
replacement terms) always produces a DIFFERENT id.

`client_order_id` is derived deterministically from `execution_intent_id`
alone (`derive_client_order_id`) -- Section 7: "one economic submit
operation maps to one stable client_order_id; safe retries reuse that
client_order_id." It is NEVER a random UUID."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import ExecutionCommandValidationError
from quant_platform.execution_gateway.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.execution_gateway.models import CommandType, OrderSide, OrderTypeKind, TimeInForceKind
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp

CLIENT_ORDER_ID_KIND = "execution_client_order_id"
COMMAND_SUBMIT_KIND = "execution_command_submit"
COMMAND_CANCEL_KIND = "execution_command_cancel"
COMMAND_REPLACE_KIND = "execution_command_replace"
COMMAND_QUERY_ORDER_KIND = "execution_command_query_order"
COMMAND_QUERY_OPEN_ORDERS_KIND = "execution_command_query_open_orders"
COMMAND_QUERY_POSITIONS_KIND = "execution_command_query_positions"
COMMAND_QUERY_ACCOUNT_KIND = "execution_command_query_account"
COMMAND_HEARTBEAT_KIND = "execution_command_heartbeat"

IDENTITY_VERSION = 1


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise ExecutionCommandValidationError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionCommandValidationError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise ExecutionCommandValidationError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ExecutionCommandValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise ExecutionCommandValidationError(f"{field_name}: {exc}") from exc


def _opt_decimal_json(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_json(value)


def derive_client_order_id(execution_intent_id: str) -> str:
    """The ONE stable `client_order_id` for the economic submit operation
    that originates from `execution_intent_id`. Deterministic, never a
    random UUID -- calling this twice for the same intent always returns
    the same value, which is exactly what makes a safe submit retry reuse
    the original `client_order_id` rather than mint a new one."""
    _require_sha256(execution_intent_id, field_name="execution_intent_id")
    return compute_content_id(CLIENT_ORDER_ID_KIND, {"execution_intent_id": execution_intent_id})


# --------------------------------------------------------------------------
# SubmitOrderCommand
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SubmitOrderCommand:
    command_id: str
    execution_session_id: str
    execution_intent_id: str
    client_order_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    instrument_id: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderTypeKind
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForceKind
    reduce_only: bool
    contract_multiplier: Decimal

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.SUBMIT_ORDER:
            raise ExecutionCommandValidationError(f"SubmitOrderCommand.command_type must be submit_order, got {self.command_type!r}")
        for field_name, value in (("execution_session_id", self.execution_session_id), ("execution_intent_id", self.execution_intent_id)):
            _require_sha256(value, field_name=f"SubmitOrderCommand.{field_name}")
        if self.client_order_id != derive_client_order_id(self.execution_intent_id):
            raise ExecutionCommandValidationError("SubmitOrderCommand.client_order_id must be derive_client_order_id(execution_intent_id)")
        _require_tz_aware(self.event_time, field_name="SubmitOrderCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"SubmitOrderCommand.command_sequence must be >= 0, got {self.command_sequence}")
        if not self.instrument_id:
            raise ExecutionCommandValidationError("SubmitOrderCommand.instrument_id must not be empty")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise ExecutionCommandValidationError(f"SubmitOrderCommand.quantity must be finite and > 0, got {self.quantity!r}")
        if not self.contract_multiplier.is_finite() or self.contract_multiplier <= 0:
            raise ExecutionCommandValidationError(f"SubmitOrderCommand.contract_multiplier must be finite and > 0, got {self.contract_multiplier!r}")
        if self.order_type is OrderTypeKind.MARKET:
            if self.limit_price is not None or self.stop_price is not None:
                raise ExecutionCommandValidationError("SubmitOrderCommand: limit_price/stop_price must be None for a MARKET order")
        elif self.order_type is OrderTypeKind.LIMIT:
            if self.limit_price is None or not self.limit_price.is_finite() or self.limit_price <= 0:
                raise ExecutionCommandValidationError("SubmitOrderCommand.limit_price must be finite and > 0 for a LIMIT order")
            if self.stop_price is not None:
                raise ExecutionCommandValidationError("SubmitOrderCommand.stop_price must be None for a LIMIT order")
        elif self.order_type is OrderTypeKind.STOP:
            if self.stop_price is None or not self.stop_price.is_finite() or self.stop_price <= 0:
                raise ExecutionCommandValidationError("SubmitOrderCommand.stop_price must be finite and > 0 for a STOP order")
            if self.limit_price is not None:
                raise ExecutionCommandValidationError("SubmitOrderCommand.limit_price must be None for a STOP order")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id, "execution_session_id": self.execution_session_id, "execution_intent_id": self.execution_intent_id,
            "client_order_id": self.client_order_id, "command_type": self.command_type.value, "command_sequence": self.command_sequence,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "identity_version": self.identity_version,
            "instrument_id": self.instrument_id, "side": self.side.value, "quantity": decimal_to_json(self.quantity), "order_type": self.order_type.value,
            "limit_price": _opt_decimal_json(self.limit_price), "stop_price": _opt_decimal_json(self.stop_price), "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only, "contract_multiplier": decimal_to_json(self.contract_multiplier),
        }

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "execution_intent_id": self.execution_intent_id, "client_order_id": self.client_order_id, "instrument_id": self.instrument_id,
            "side": self.side.value, "quantity": decimal_to_json(self.quantity), "order_type": self.order_type.value,
            "limit_price": _opt_decimal_json(self.limit_price), "stop_price": _opt_decimal_json(self.stop_price), "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only, "contract_multiplier": decimal_to_json(self.contract_multiplier),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SubmitOrderCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), execution_intent_id=str(raw["execution_intent_id"]),
            client_order_id=str(raw["client_order_id"]), command_type=CommandType(raw["command_type"]), command_sequence=int(str(raw["command_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), identity_version=int(str(raw["identity_version"])),
            instrument_id=str(raw["instrument_id"]), side=OrderSide(raw["side"]), quantity=parse_decimal(raw["quantity"], field_name="quantity"),
            order_type=OrderTypeKind(raw["order_type"]), limit_price=(None if raw.get("limit_price") is None else parse_decimal(raw["limit_price"], field_name="limit_price")),
            stop_price=(None if raw.get("stop_price") is None else parse_decimal(raw["stop_price"], field_name="stop_price")),
            time_in_force=TimeInForceKind(raw["time_in_force"]), reduce_only=bool(raw["reduce_only"]),
            contract_multiplier=parse_decimal(raw["contract_multiplier"], field_name="contract_multiplier"),
        )


def create_submit_order_command(
    *, execution_session_id: str, execution_intent_id: str, command_sequence: int, event_time: datetime, instrument_id: str, side: OrderSide,
    quantity: Decimal, order_type: OrderTypeKind, time_in_force: TimeInForceKind, reduce_only: bool, contract_multiplier: Decimal,
    limit_price: Decimal | None = None, stop_price: Decimal | None = None,
) -> SubmitOrderCommand:
    client_order_id = derive_client_order_id(execution_intent_id)
    kwargs = {
        "execution_session_id": execution_session_id, "execution_intent_id": execution_intent_id, "client_order_id": client_order_id,
        "command_type": CommandType.SUBMIT_ORDER, "command_sequence": command_sequence, "event_time": event_time, "identity_version": IDENTITY_VERSION,
        "instrument_id": instrument_id, "side": side, "quantity": quantity, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price,
        "time_in_force": time_in_force, "reduce_only": reduce_only, "contract_multiplier": contract_multiplier,
    }
    provisional = SubmitOrderCommand(command_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    command_id = compute_content_id(COMMAND_SUBMIT_KIND, provisional.to_identity_payload())
    return SubmitOrderCommand(command_id=command_id, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# CancelOrderCommand
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CancelOrderCommand:
    command_id: str
    execution_session_id: str
    execution_intent_id: str | None
    client_order_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    execution_order_id: str
    broker_order_id: str | None
    cancellation_reason: str

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.CANCEL_ORDER:
            raise ExecutionCommandValidationError(f"CancelOrderCommand.command_type must be cancel_order, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="CancelOrderCommand.execution_session_id")
        if self.execution_intent_id is not None:
            _require_sha256(self.execution_intent_id, field_name="CancelOrderCommand.execution_intent_id")
        _require_sha256(self.execution_order_id, field_name="CancelOrderCommand.execution_order_id")
        if not self.client_order_id:
            raise ExecutionCommandValidationError("CancelOrderCommand.client_order_id must not be empty")
        if not self.cancellation_reason:
            raise ExecutionCommandValidationError("CancelOrderCommand.cancellation_reason must not be empty")
        _require_tz_aware(self.event_time, field_name="CancelOrderCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"CancelOrderCommand.command_sequence must be >= 0, got {self.command_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id, "execution_session_id": self.execution_session_id, "execution_intent_id": self.execution_intent_id,
            "client_order_id": self.client_order_id, "command_type": self.command_type.value, "command_sequence": self.command_sequence,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "identity_version": self.identity_version,
            "execution_order_id": self.execution_order_id, "broker_order_id": self.broker_order_id, "cancellation_reason": self.cancellation_reason,
        }

    def to_identity_payload(self) -> dict[str, object]:
        return {"execution_order_id": self.execution_order_id, "cancellation_reason": self.cancellation_reason}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CancelOrderCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]),
            execution_intent_id=(None if raw.get("execution_intent_id") is None else str(raw["execution_intent_id"])), client_order_id=str(raw["client_order_id"]),
            command_type=CommandType(raw["command_type"]), command_sequence=int(str(raw["command_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), identity_version=int(str(raw["identity_version"])),
            execution_order_id=str(raw["execution_order_id"]), broker_order_id=(None if raw.get("broker_order_id") is None else str(raw["broker_order_id"])),
            cancellation_reason=str(raw["cancellation_reason"]),
        )


def create_cancel_order_command(
    *, execution_session_id: str, execution_order_id: str, client_order_id: str, cancellation_reason: str, command_sequence: int, event_time: datetime,
    execution_intent_id: str | None = None, broker_order_id: str | None = None,
) -> CancelOrderCommand:
    kwargs = {
        "execution_session_id": execution_session_id, "execution_intent_id": execution_intent_id, "client_order_id": client_order_id,
        "command_type": CommandType.CANCEL_ORDER, "command_sequence": command_sequence, "event_time": event_time, "identity_version": IDENTITY_VERSION,
        "execution_order_id": execution_order_id, "broker_order_id": broker_order_id, "cancellation_reason": cancellation_reason,
    }
    provisional = CancelOrderCommand(command_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    command_id = compute_content_id(COMMAND_CANCEL_KIND, provisional.to_identity_payload())
    return CancelOrderCommand(command_id=command_id, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# ReplaceOrderCommand
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReplaceOrderCommand:
    command_id: str
    execution_session_id: str
    execution_intent_id: str | None
    client_order_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    execution_order_id: str
    broker_order_id: str | None
    replacement_quantity: Decimal | None
    replacement_limit_price: Decimal | None
    replacement_stop_price: Decimal | None
    replacement_time_in_force: TimeInForceKind | None

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.REPLACE_ORDER:
            raise ExecutionCommandValidationError(f"ReplaceOrderCommand.command_type must be replace_order, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="ReplaceOrderCommand.execution_session_id")
        if self.execution_intent_id is not None:
            _require_sha256(self.execution_intent_id, field_name="ReplaceOrderCommand.execution_intent_id")
        _require_sha256(self.execution_order_id, field_name="ReplaceOrderCommand.execution_order_id")
        if not self.client_order_id:
            raise ExecutionCommandValidationError("ReplaceOrderCommand.client_order_id must not be empty")
        _require_tz_aware(self.event_time, field_name="ReplaceOrderCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"ReplaceOrderCommand.command_sequence must be >= 0, got {self.command_sequence}")
        if self.replacement_quantity is not None and (not self.replacement_quantity.is_finite() or self.replacement_quantity <= 0):
            raise ExecutionCommandValidationError(f"ReplaceOrderCommand.replacement_quantity must be finite and > 0 when set, got {self.replacement_quantity!r}")
        if self.replacement_limit_price is not None and (not self.replacement_limit_price.is_finite() or self.replacement_limit_price <= 0):
            raise ExecutionCommandValidationError("ReplaceOrderCommand.replacement_limit_price must be finite and > 0 when set")
        if self.replacement_stop_price is not None and (not self.replacement_stop_price.is_finite() or self.replacement_stop_price <= 0):
            raise ExecutionCommandValidationError("ReplaceOrderCommand.replacement_stop_price must be finite and > 0 when set")
        if all(v is None for v in (self.replacement_quantity, self.replacement_limit_price, self.replacement_stop_price, self.replacement_time_in_force)):
            raise ExecutionCommandValidationError("ReplaceOrderCommand must change at least one of quantity/limit_price/stop_price/time_in_force")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id, "execution_session_id": self.execution_session_id, "execution_intent_id": self.execution_intent_id,
            "client_order_id": self.client_order_id, "command_type": self.command_type.value, "command_sequence": self.command_sequence,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "identity_version": self.identity_version,
            "execution_order_id": self.execution_order_id, "broker_order_id": self.broker_order_id,
            "replacement_quantity": _opt_decimal_json(self.replacement_quantity), "replacement_limit_price": _opt_decimal_json(self.replacement_limit_price),
            "replacement_stop_price": _opt_decimal_json(self.replacement_stop_price),
            "replacement_time_in_force": (None if self.replacement_time_in_force is None else self.replacement_time_in_force.value),
        }

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "execution_order_id": self.execution_order_id, "replacement_quantity": _opt_decimal_json(self.replacement_quantity),
            "replacement_limit_price": _opt_decimal_json(self.replacement_limit_price), "replacement_stop_price": _opt_decimal_json(self.replacement_stop_price),
            "replacement_time_in_force": (None if self.replacement_time_in_force is None else self.replacement_time_in_force.value),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReplaceOrderCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]),
            execution_intent_id=(None if raw.get("execution_intent_id") is None else str(raw["execution_intent_id"])), client_order_id=str(raw["client_order_id"]),
            command_type=CommandType(raw["command_type"]), command_sequence=int(str(raw["command_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), identity_version=int(str(raw["identity_version"])),
            execution_order_id=str(raw["execution_order_id"]), broker_order_id=(None if raw.get("broker_order_id") is None else str(raw["broker_order_id"])),
            replacement_quantity=(None if raw.get("replacement_quantity") is None else parse_decimal(raw["replacement_quantity"], field_name="replacement_quantity")),
            replacement_limit_price=(None if raw.get("replacement_limit_price") is None else parse_decimal(raw["replacement_limit_price"], field_name="replacement_limit_price")),
            replacement_stop_price=(None if raw.get("replacement_stop_price") is None else parse_decimal(raw["replacement_stop_price"], field_name="replacement_stop_price")),
            replacement_time_in_force=(None if raw.get("replacement_time_in_force") is None else TimeInForceKind(raw["replacement_time_in_force"])),
        )


def create_replace_order_command(
    *, execution_session_id: str, execution_order_id: str, client_order_id: str, command_sequence: int, event_time: datetime,
    execution_intent_id: str | None = None, broker_order_id: str | None = None, replacement_quantity: Decimal | None = None,
    replacement_limit_price: Decimal | None = None, replacement_stop_price: Decimal | None = None, replacement_time_in_force: TimeInForceKind | None = None,
) -> ReplaceOrderCommand:
    kwargs = {
        "execution_session_id": execution_session_id, "execution_intent_id": execution_intent_id, "client_order_id": client_order_id,
        "command_type": CommandType.REPLACE_ORDER, "command_sequence": command_sequence, "event_time": event_time, "identity_version": IDENTITY_VERSION,
        "execution_order_id": execution_order_id, "broker_order_id": broker_order_id, "replacement_quantity": replacement_quantity,
        "replacement_limit_price": replacement_limit_price, "replacement_stop_price": replacement_stop_price, "replacement_time_in_force": replacement_time_in_force,
    }
    provisional = ReplaceOrderCommand(command_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    command_id = compute_content_id(COMMAND_REPLACE_KIND, provisional.to_identity_payload())
    return ReplaceOrderCommand(command_id=command_id, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Query / heartbeat commands -- no economic identity, sequence-scoped.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QueryOrderCommand:
    command_id: str
    execution_session_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int
    execution_order_id: str | None
    client_order_id: str | None
    broker_order_id: str | None

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.QUERY_ORDER:
            raise ExecutionCommandValidationError(f"QueryOrderCommand.command_type must be query_order, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="QueryOrderCommand.execution_session_id")
        _require_tz_aware(self.event_time, field_name="QueryOrderCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"QueryOrderCommand.command_sequence must be >= 0, got {self.command_sequence}")
        if not any((self.execution_order_id, self.client_order_id, self.broker_order_id)):
            raise ExecutionCommandValidationError("QueryOrderCommand must identify the target order via execution_order_id/client_order_id/broker_order_id")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id, "execution_session_id": self.execution_session_id, "command_type": self.command_type.value,
            "command_sequence": self.command_sequence, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
            "identity_version": self.identity_version, "execution_order_id": self.execution_order_id, "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QueryOrderCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), command_type=CommandType(raw["command_type"]),
            command_sequence=int(str(raw["command_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            identity_version=int(str(raw["identity_version"])), execution_order_id=(None if raw.get("execution_order_id") is None else str(raw["execution_order_id"])),
            client_order_id=(None if raw.get("client_order_id") is None else str(raw["client_order_id"])),
            broker_order_id=(None if raw.get("broker_order_id") is None else str(raw["broker_order_id"])),
        )


def create_query_order_command(
    *, execution_session_id: str, command_sequence: int, event_time: datetime, execution_order_id: str | None = None, client_order_id: str | None = None,
    broker_order_id: str | None = None,
) -> QueryOrderCommand:
    command_id = compute_content_id(COMMAND_QUERY_ORDER_KIND, {"command_sequence": command_sequence})
    return QueryOrderCommand(
        command_id=command_id, execution_session_id=execution_session_id, command_type=CommandType.QUERY_ORDER, command_sequence=command_sequence,
        event_time=event_time, identity_version=IDENTITY_VERSION, execution_order_id=execution_order_id, client_order_id=client_order_id,
        broker_order_id=broker_order_id,
    )


def _simple_query_to_json(command_id: str, execution_session_id: str, command_type: CommandType, command_sequence: int, event_time: datetime, identity_version: int) -> dict[str, object]:
    return {
        "command_id": command_id, "execution_session_id": execution_session_id, "command_type": command_type.value, "command_sequence": command_sequence,
        "event_time": _serialize_timestamp(event_time, field_name="event_time"), "identity_version": identity_version,
    }


@dataclass(frozen=True, slots=True)
class QueryOpenOrdersCommand:
    command_id: str
    execution_session_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.QUERY_OPEN_ORDERS:
            raise ExecutionCommandValidationError(f"QueryOpenOrdersCommand.command_type must be query_open_orders, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="QueryOpenOrdersCommand.execution_session_id")
        _require_tz_aware(self.event_time, field_name="QueryOpenOrdersCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"QueryOpenOrdersCommand.command_sequence must be >= 0, got {self.command_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return _simple_query_to_json(self.command_id, self.execution_session_id, self.command_type, self.command_sequence, self.event_time, self.identity_version)

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QueryOpenOrdersCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), command_type=CommandType(raw["command_type"]),
            command_sequence=int(str(raw["command_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            identity_version=int(str(raw["identity_version"])),
        )


def create_query_open_orders_command(*, execution_session_id: str, command_sequence: int, event_time: datetime) -> QueryOpenOrdersCommand:
    command_id = compute_content_id(COMMAND_QUERY_OPEN_ORDERS_KIND, {"command_sequence": command_sequence})
    return QueryOpenOrdersCommand(
        command_id=command_id, execution_session_id=execution_session_id, command_type=CommandType.QUERY_OPEN_ORDERS, command_sequence=command_sequence,
        event_time=event_time, identity_version=IDENTITY_VERSION,
    )


@dataclass(frozen=True, slots=True)
class QueryPositionsCommand:
    command_id: str
    execution_session_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.QUERY_POSITIONS:
            raise ExecutionCommandValidationError(f"QueryPositionsCommand.command_type must be query_positions, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="QueryPositionsCommand.execution_session_id")
        _require_tz_aware(self.event_time, field_name="QueryPositionsCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"QueryPositionsCommand.command_sequence must be >= 0, got {self.command_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return _simple_query_to_json(self.command_id, self.execution_session_id, self.command_type, self.command_sequence, self.event_time, self.identity_version)

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QueryPositionsCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), command_type=CommandType(raw["command_type"]),
            command_sequence=int(str(raw["command_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            identity_version=int(str(raw["identity_version"])),
        )


def create_query_positions_command(*, execution_session_id: str, command_sequence: int, event_time: datetime) -> QueryPositionsCommand:
    command_id = compute_content_id(COMMAND_QUERY_POSITIONS_KIND, {"command_sequence": command_sequence})
    return QueryPositionsCommand(
        command_id=command_id, execution_session_id=execution_session_id, command_type=CommandType.QUERY_POSITIONS, command_sequence=command_sequence,
        event_time=event_time, identity_version=IDENTITY_VERSION,
    )


@dataclass(frozen=True, slots=True)
class QueryAccountCommand:
    command_id: str
    execution_session_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.QUERY_ACCOUNT:
            raise ExecutionCommandValidationError(f"QueryAccountCommand.command_type must be query_account, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="QueryAccountCommand.execution_session_id")
        _require_tz_aware(self.event_time, field_name="QueryAccountCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"QueryAccountCommand.command_sequence must be >= 0, got {self.command_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return _simple_query_to_json(self.command_id, self.execution_session_id, self.command_type, self.command_sequence, self.event_time, self.identity_version)

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QueryAccountCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), command_type=CommandType(raw["command_type"]),
            command_sequence=int(str(raw["command_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            identity_version=int(str(raw["identity_version"])),
        )


def create_query_account_command(*, execution_session_id: str, command_sequence: int, event_time: datetime) -> QueryAccountCommand:
    command_id = compute_content_id(COMMAND_QUERY_ACCOUNT_KIND, {"command_sequence": command_sequence})
    return QueryAccountCommand(
        command_id=command_id, execution_session_id=execution_session_id, command_type=CommandType.QUERY_ACCOUNT, command_sequence=command_sequence,
        event_time=event_time, identity_version=IDENTITY_VERSION,
    )


@dataclass(frozen=True, slots=True)
class HeartbeatCommand:
    command_id: str
    execution_session_id: str
    command_type: CommandType
    command_sequence: int
    event_time: datetime
    identity_version: int

    def __post_init__(self) -> None:
        if self.command_type is not CommandType.HEARTBEAT:
            raise ExecutionCommandValidationError(f"HeartbeatCommand.command_type must be heartbeat, got {self.command_type!r}")
        _require_sha256(self.execution_session_id, field_name="HeartbeatCommand.execution_session_id")
        _require_tz_aware(self.event_time, field_name="HeartbeatCommand.event_time")
        if self.command_sequence < 0:
            raise ExecutionCommandValidationError(f"HeartbeatCommand.command_sequence must be >= 0, got {self.command_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return _simple_query_to_json(self.command_id, self.execution_session_id, self.command_type, self.command_sequence, self.event_time, self.identity_version)

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HeartbeatCommand:
        return cls(
            command_id=str(raw["command_id"]), execution_session_id=str(raw["execution_session_id"]), command_type=CommandType(raw["command_type"]),
            command_sequence=int(str(raw["command_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            identity_version=int(str(raw["identity_version"])),
        )


def create_heartbeat_command(*, execution_session_id: str, command_sequence: int, event_time: datetime) -> HeartbeatCommand:
    command_id = compute_content_id(COMMAND_HEARTBEAT_KIND, {"command_sequence": command_sequence})
    return HeartbeatCommand(
        command_id=command_id, execution_session_id=execution_session_id, command_type=CommandType.HEARTBEAT, command_sequence=command_sequence,
        event_time=event_time, identity_version=IDENTITY_VERSION,
    )


ExecutionCommand = (
    SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand | QueryOrderCommand | QueryOpenOrdersCommand | QueryPositionsCommand | QueryAccountCommand
    | HeartbeatCommand
)

__all__ = [
    "CLIENT_ORDER_ID_KIND",
    "COMMAND_CANCEL_KIND",
    "COMMAND_HEARTBEAT_KIND",
    "COMMAND_QUERY_ACCOUNT_KIND",
    "COMMAND_QUERY_OPEN_ORDERS_KIND",
    "COMMAND_QUERY_ORDER_KIND",
    "COMMAND_QUERY_POSITIONS_KIND",
    "COMMAND_REPLACE_KIND",
    "COMMAND_SUBMIT_KIND",
    "IDENTITY_VERSION",
    "CancelOrderCommand",
    "ExecutionCommand",
    "HeartbeatCommand",
    "QueryAccountCommand",
    "QueryOpenOrdersCommand",
    "QueryOrderCommand",
    "QueryPositionsCommand",
    "ReplaceOrderCommand",
    "SubmitOrderCommand",
    "create_cancel_order_command",
    "create_heartbeat_command",
    "create_query_account_command",
    "create_query_open_orders_command",
    "create_query_order_command",
    "create_query_positions_command",
    "create_replace_order_command",
    "create_submit_order_command",
    "derive_client_order_id",
]
