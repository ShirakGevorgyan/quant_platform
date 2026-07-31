"""Normalization boundary (Milestone 8, Section 14). Translates ONE
adapter's raw, adapter-specific response shape into this package's
single normalized `BrokerEvent` -- the ONE place adapter-specific
knowledge is allowed to exist. `dummy_broker.py` calls
`normalize_dummy_broker_event`; a future MT5 adapter would implement its
own `normalize_mt5_event` against `DummyBrokerRawEvent`'s sibling shape,
but every module downstream of normalization (`states.py`,
`sequencing.py`, `reconciliation.py`, `verification.py`, ...) only ever
sees `BrokerEvent` -- dummy-specific data never leaks past this file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.execution_gateway.events import BrokerEvent, create_broker_event
from quant_platform.execution_gateway.models import BrokerEventType


@dataclass(frozen=True, slots=True)
class DummyBrokerRawEvent:
    """The dummy broker's own internal, adapter-specific event shape --
    never consumed by any module other than this one."""

    broker_sequence: int
    event_type: BrokerEventType
    broker_timestamp: datetime
    broker_order_id: str | None = None
    broker_fill_id: str | None = None
    client_order_id: str | None = None
    original_quantity: Decimal | None = None
    filled_quantity: Decimal | None = None
    cumulative_filled_quantity: Decimal | None = None
    remaining_quantity: Decimal | None = None
    fill_price: Decimal | None = None
    reject_code: str | None = None
    reject_message: str | None = None
    source_command_id: str | None = None


def normalize_dummy_broker_event(raw: DummyBrokerRawEvent, *, execution_session_id: str, adapter_id: str, received_event_time: datetime) -> BrokerEvent:
    return create_broker_event(
        execution_session_id=execution_session_id, adapter_id=adapter_id, broker_sequence=raw.broker_sequence, event_type=raw.event_type,
        broker_timestamp=raw.broker_timestamp, received_event_time=received_event_time, broker_order_id=raw.broker_order_id,
        broker_fill_id=raw.broker_fill_id, client_order_id=raw.client_order_id, original_quantity=raw.original_quantity,
        filled_quantity=raw.filled_quantity, cumulative_filled_quantity=raw.cumulative_filled_quantity, remaining_quantity=raw.remaining_quantity,
        fill_price=raw.fill_price, reject_code=raw.reject_code, reject_message=raw.reject_message, source_command_id=raw.source_command_id,
    )


__all__ = ["DummyBrokerRawEvent", "normalize_dummy_broker_event"]
