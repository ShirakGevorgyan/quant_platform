"""Normalized broker event model (Milestone 8, Section 14). Every raw
response `dummy_broker.py` produces is translated into exactly one
`BrokerEvent` through `normalization.py` BEFORE anything else in this
package ever sees it -- `states.py`/`state_machine.py`/`reconciliation.py`
/`verification.py` only ever consume `BrokerEvent`, never a dummy-broker-
specific type, which is precisely what lets a future MT5 adapter reuse
every one of those modules unchanged (Section 1's "architecture must
allow a future MT5 adapter to implement the same adapter protocol
without changing ... execution domain logic")."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import BrokerEventError
from quant_platform.execution_gateway.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.execution_gateway.models import BrokerEventType
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp

BROKER_EVENT_KIND = "broker_event"
IDENTITY_VERSION = 1

_REJECT_EVENT_TYPES: frozenset[BrokerEventType] = frozenset({BrokerEventType.ORDER_REJECTED, BrokerEventType.CANCEL_REJECTED, BrokerEventType.REPLACE_REJECTED})
_FILL_EVENT_TYPES: frozenset[BrokerEventType] = frozenset({BrokerEventType.ORDER_PARTIALLY_FILLED, BrokerEventType.ORDER_FILLED})


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise BrokerEventError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise BrokerEventError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise BrokerEventError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise BrokerEventError(f"{field_name}: {exc}") from exc


def _opt_decimal_json(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_json(value)


def _opt_decimal(raw: object, *, field_name: str) -> Decimal | None:
    return None if raw is None else parse_decimal(raw, field_name=field_name)


@dataclass(frozen=True, slots=True)
class BrokerEvent:
    broker_event_id: str
    execution_session_id: str
    adapter_id: str

    broker_sequence: int
    broker_order_id: str | None
    broker_fill_id: str | None
    client_order_id: str | None

    event_type: BrokerEventType

    original_quantity: Decimal | None
    filled_quantity: Decimal | None
    cumulative_filled_quantity: Decimal | None
    remaining_quantity: Decimal | None
    fill_price: Decimal | None

    broker_timestamp: datetime
    received_event_time: datetime

    reject_code: str | None
    reject_message: str | None

    source_command_id: str | None
    identity_version: int

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.execution_session_id):
            raise BrokerEventError(f"BrokerEvent.execution_session_id must be a valid sha256 hex digest, got {self.execution_session_id!r}")
        if not self.adapter_id:
            raise BrokerEventError("BrokerEvent.adapter_id must not be empty")
        if self.broker_sequence < 1:
            raise BrokerEventError(f"BrokerEvent.broker_sequence must be >= 1, got {self.broker_sequence}")
        _require_tz_aware(self.broker_timestamp, field_name="BrokerEvent.broker_timestamp")
        _require_tz_aware(self.received_event_time, field_name="BrokerEvent.received_event_time")
        if self.received_event_time < self.broker_timestamp:
            raise BrokerEventError("BrokerEvent.received_event_time must be >= broker_timestamp")

        for value, field_name in (
            (self.original_quantity, "original_quantity"), (self.filled_quantity, "filled_quantity"),
            (self.cumulative_filled_quantity, "cumulative_filled_quantity"), (self.remaining_quantity, "remaining_quantity"),
        ):
            if value is not None and (not value.is_finite() or value < 0):
                raise BrokerEventError(f"BrokerEvent.{field_name} must be finite and >= 0 when set, got {value!r}")
        if self.fill_price is not None and (not self.fill_price.is_finite() or self.fill_price <= 0):
            raise BrokerEventError(f"BrokerEvent.fill_price must be finite and > 0 when set, got {self.fill_price!r}")

        if self.event_type in _REJECT_EVENT_TYPES:
            if not self.reject_code:
                raise BrokerEventError(f"BrokerEvent.reject_code is required for event_type={self.event_type.value!r}")
        elif self.reject_code is not None or self.reject_message is not None:
            raise BrokerEventError(f"BrokerEvent.reject_code/reject_message must be None for event_type={self.event_type.value!r}")

        if self.event_type in _FILL_EVENT_TYPES:
            if self.fill_price is None or self.filled_quantity is None or self.broker_fill_id is None:
                raise BrokerEventError(f"BrokerEvent.fill_price/filled_quantity/broker_fill_id are required for event_type={self.event_type.value!r}")
            if self.filled_quantity <= 0:
                raise BrokerEventError("BrokerEvent.filled_quantity must be > 0 for a fill event")

        # Section 9's core invariant, checked as early as the event itself
        # when all three fields are present: filled + remaining == current
        # (original) quantity.
        if (
            self.original_quantity is not None and self.cumulative_filled_quantity is not None and self.remaining_quantity is not None
            and self.cumulative_filled_quantity + self.remaining_quantity != self.original_quantity
        ):
            raise BrokerEventError(
                f"BrokerEvent: cumulative_filled_quantity ({self.cumulative_filled_quantity}) + remaining_quantity ({self.remaining_quantity}) "
                f"!= original_quantity ({self.original_quantity})"
            )
        if self.filled_quantity is not None and self.cumulative_filled_quantity is not None and self.filled_quantity > self.cumulative_filled_quantity:
            raise BrokerEventError(
                f"BrokerEvent.filled_quantity ({self.filled_quantity}) must be <= cumulative_filled_quantity ({self.cumulative_filled_quantity})"
            )
        if self.identity_version < 1:
            raise BrokerEventError(f"BrokerEvent.identity_version must be >= 1, got {self.identity_version}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "broker_event_id": self.broker_event_id, "execution_session_id": self.execution_session_id, "adapter_id": self.adapter_id,
            "broker_sequence": self.broker_sequence, "broker_order_id": self.broker_order_id, "broker_fill_id": self.broker_fill_id,
            "client_order_id": self.client_order_id, "event_type": self.event_type.value, "original_quantity": _opt_decimal_json(self.original_quantity),
            "filled_quantity": _opt_decimal_json(self.filled_quantity), "cumulative_filled_quantity": _opt_decimal_json(self.cumulative_filled_quantity),
            "remaining_quantity": _opt_decimal_json(self.remaining_quantity), "fill_price": _opt_decimal_json(self.fill_price),
            "broker_timestamp": _serialize_timestamp(self.broker_timestamp, field_name="broker_timestamp"),
            "received_event_time": _serialize_timestamp(self.received_event_time, field_name="received_event_time"), "reject_code": self.reject_code,
            "reject_message": self.reject_message, "source_command_id": self.source_command_id, "identity_version": self.identity_version,
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Every field EXCEPT `broker_event_id` itself, `received_event_
        time` (this package's own wall-clock-ish receipt timestamp, not
        something the broker asserts), and `adapter_id` participates in
        identity -- two deliveries of the genuinely same broker event
        (Section 16: "duplicate broker event with exact payload is
        idempotent") must produce the SAME `broker_event_id` regardless of
        when this process happened to receive/process them.

        CONFIRMED DEFECT, FIXED (found during this milestone's own
        acceptance testing): `adapter_id` -- a purely OPERATIONAL label
        for which adapter instance produced this event, with zero
        economic consequence -- used to participate in this identity
        computation. Two genuinely economically-identical sessions
        constructed with different `adapter_id` strings (an operator is
        free to name their adapter instance differently between runs)
        produced DIFFERENT `broker_event_id` values for the SAME
        underlying broker fact, which cascaded into different
        `execution_fill_id`/`ExecutionOrderStateEvent.event_id` values and
        therefore a different session-level semantic digest -- exactly
        the defect class Section 40 names for PYTHONHASHSEED/temp-path
        leaking into what must depend on economics alone."""
        payload = dict(self.to_json_dict())
        del payload["broker_event_id"]
        del payload["received_event_time"]
        del payload["adapter_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BrokerEvent:
        return cls(
            broker_event_id=str(raw["broker_event_id"]), execution_session_id=str(raw["execution_session_id"]), adapter_id=str(raw["adapter_id"]),
            broker_sequence=int(str(raw["broker_sequence"])), broker_order_id=(None if raw.get("broker_order_id") is None else str(raw["broker_order_id"])),
            broker_fill_id=(None if raw.get("broker_fill_id") is None else str(raw["broker_fill_id"])),
            client_order_id=(None if raw.get("client_order_id") is None else str(raw["client_order_id"])), event_type=BrokerEventType(raw["event_type"]),
            original_quantity=_opt_decimal(raw.get("original_quantity"), field_name="original_quantity"),
            filled_quantity=_opt_decimal(raw.get("filled_quantity"), field_name="filled_quantity"),
            cumulative_filled_quantity=_opt_decimal(raw.get("cumulative_filled_quantity"), field_name="cumulative_filled_quantity"),
            remaining_quantity=_opt_decimal(raw.get("remaining_quantity"), field_name="remaining_quantity"),
            fill_price=_opt_decimal(raw.get("fill_price"), field_name="fill_price"),
            broker_timestamp=_deserialize_timestamp(raw["broker_timestamp"], field_name="broker_timestamp"),
            received_event_time=_deserialize_timestamp(raw["received_event_time"], field_name="received_event_time"),
            reject_code=(None if raw.get("reject_code") is None else str(raw["reject_code"])),
            reject_message=(None if raw.get("reject_message") is None else str(raw["reject_message"])),
            source_command_id=(None if raw.get("source_command_id") is None else str(raw["source_command_id"])), identity_version=int(str(raw["identity_version"])),
        )


def create_broker_event(
    *, execution_session_id: str, adapter_id: str, broker_sequence: int, event_type: BrokerEventType, broker_timestamp: datetime,
    received_event_time: datetime, broker_order_id: str | None = None, broker_fill_id: str | None = None, client_order_id: str | None = None,
    original_quantity: Decimal | None = None, filled_quantity: Decimal | None = None, cumulative_filled_quantity: Decimal | None = None,
    remaining_quantity: Decimal | None = None, fill_price: Decimal | None = None, reject_code: str | None = None, reject_message: str | None = None,
    source_command_id: str | None = None,
) -> BrokerEvent:
    kwargs = {
        "execution_session_id": execution_session_id, "adapter_id": adapter_id, "broker_sequence": broker_sequence, "broker_order_id": broker_order_id,
        "broker_fill_id": broker_fill_id, "client_order_id": client_order_id, "event_type": event_type, "original_quantity": original_quantity,
        "filled_quantity": filled_quantity, "cumulative_filled_quantity": cumulative_filled_quantity, "remaining_quantity": remaining_quantity,
        "fill_price": fill_price, "broker_timestamp": broker_timestamp, "received_event_time": received_event_time, "reject_code": reject_code,
        "reject_message": reject_message, "source_command_id": source_command_id, "identity_version": IDENTITY_VERSION,
    }
    provisional = BrokerEvent(broker_event_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    broker_event_id = compute_content_id(BROKER_EVENT_KIND, provisional.to_identity_payload())
    return BrokerEvent(broker_event_id=broker_event_id, **kwargs)  # type: ignore[arg-type]


__all__ = ["BROKER_EVENT_KIND", "IDENTITY_VERSION", "BrokerEvent", "create_broker_event"]
