"""Broker-neutral adapter protocol (Milestone 8, Section 11). Every
module downstream of this one (`dispatcher.py`, `recovery.py`,
`reconciliation.py`, `runner.py`) talks to `ExecutionAdapter` ONLY --
never to `dummy_broker.DeterministicDummyBrokerAdapter` directly by name
-- which is precisely what lets a future MT5 adapter implement this same
Protocol and be dropped in without changing a single line of strategy,
risk, ledger, reconciliation, or execution domain logic (Section 1).

CAPABILITY CHECKS FAIL CLOSED (Section 11's own instruction): `require_
capability` is called by `dispatcher.py` BEFORE every adapter call, never
after -- a command whose requirements the active adapter does not
declare support for is rejected before dispatch, never silently
downgraded or attempted anyway."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from quant_platform.core.exceptions import BrokerCapabilityError, BrokerSnapshotError
from quant_platform.execution_gateway.commands import (
    CancelOrderCommand,
    ReplaceOrderCommand,
    SubmitOrderCommand,
)
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.health import AdapterHealthSnapshot
from quant_platform.execution_gateway.heartbeat import HeartbeatOutcome
from quant_platform.execution_gateway.identity import decimal_to_json
from quant_platform.execution_gateway.models import (
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)


# --------------------------------------------------------------------------
# Capabilities (Section 11)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    supports_market_orders: bool
    supports_limit_orders: bool
    supports_stop_orders: bool
    supports_cancel: bool
    supports_replace: bool
    supports_partial_fills: bool
    supports_client_order_id: bool
    supports_broker_sequence: bool
    supports_reduce_only: bool
    supports_open_order_query: bool
    supports_order_query: bool
    supports_position_query: bool
    supports_account_query: bool
    guarantees_idempotent_submit: bool
    guarantees_idempotent_cancel: bool
    guarantees_idempotent_replace: bool
    guarantees_event_ordering: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "supports_market_orders": self.supports_market_orders, "supports_limit_orders": self.supports_limit_orders,
            "supports_stop_orders": self.supports_stop_orders, "supports_cancel": self.supports_cancel, "supports_replace": self.supports_replace,
            "supports_partial_fills": self.supports_partial_fills, "supports_client_order_id": self.supports_client_order_id,
            "supports_broker_sequence": self.supports_broker_sequence, "supports_reduce_only": self.supports_reduce_only,
            "supports_open_order_query": self.supports_open_order_query, "supports_order_query": self.supports_order_query,
            "supports_position_query": self.supports_position_query, "supports_account_query": self.supports_account_query,
            "guarantees_idempotent_submit": self.guarantees_idempotent_submit, "guarantees_idempotent_cancel": self.guarantees_idempotent_cancel,
            "guarantees_idempotent_replace": self.guarantees_idempotent_replace, "guarantees_event_ordering": self.guarantees_event_ordering,
        }


DETERMINISTIC_DUMMY_CAPABILITIES = AdapterCapabilities(
    supports_market_orders=True, supports_limit_orders=True, supports_stop_orders=True, supports_cancel=True, supports_replace=True,
    supports_partial_fills=True, supports_client_order_id=True, supports_broker_sequence=True, supports_reduce_only=True, supports_open_order_query=True,
    supports_order_query=True, supports_position_query=True, supports_account_query=True, guarantees_idempotent_submit=True,
    guarantees_idempotent_cancel=True, guarantees_idempotent_replace=True, guarantees_event_ordering=True,
)


def require_capability(capabilities: AdapterCapabilities, command: SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand) -> None:
    """Fail-closed pre-dispatch capability gate (Section 11). Raises
    `BrokerCapabilityError` -- never silently downgrades or ignores a
    requested semantic."""
    if isinstance(command, SubmitOrderCommand):
        if command.order_type is OrderTypeKind.MARKET and not capabilities.supports_market_orders:
            raise BrokerCapabilityError("adapter does not support MARKET orders")
        if command.order_type is OrderTypeKind.LIMIT and not capabilities.supports_limit_orders:
            raise BrokerCapabilityError("adapter does not support LIMIT orders")
        if command.order_type is OrderTypeKind.STOP and not capabilities.supports_stop_orders:
            raise BrokerCapabilityError("adapter does not support STOP orders")
        if command.reduce_only and not capabilities.supports_reduce_only:
            raise BrokerCapabilityError("adapter does not support reduce_only orders")
        if not capabilities.supports_client_order_id:
            raise BrokerCapabilityError("adapter does not support client_order_id, but every submit in this package requires one")
    elif isinstance(command, CancelOrderCommand):
        if not capabilities.supports_cancel:
            raise BrokerCapabilityError("adapter does not support cancel")
    elif isinstance(command, ReplaceOrderCommand):
        if not capabilities.supports_replace:
            raise BrokerCapabilityError("adapter does not support replace")


def require_query_capability(capabilities: AdapterCapabilities, *, query_kind: str) -> None:
    mapping = {
        "order": capabilities.supports_order_query, "open_orders": capabilities.supports_open_order_query, "positions": capabilities.supports_position_query,
        "account": capabilities.supports_account_query,
    }
    if query_kind not in mapping:
        raise BrokerCapabilityError(f"unknown query_kind {query_kind!r}")
    if not mapping[query_kind]:
        raise BrokerCapabilityError(f"adapter does not support {query_kind} query -- recovery/reconciliation that depends on it cannot proceed safely")


# --------------------------------------------------------------------------
# Synchronous call outcome
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AdapterCallResult:
    """The adapter call's own SYNCHRONOUS outcome only -- proof the call
    definitely reached (`accepted_for_processing=True`) or definitely did
    NOT reach (`accepted_for_processing=False`, with a reason) the broker.
    The actual order lifecycle (acknowledged/filled/rejected-by-broker/
    etc.) is never returned here -- it arrives later, asynchronously,
    through `poll_events` as normalized `BrokerEvent`s. This split is what
    lets the dummy broker (and a future real adapter) model delayed,
    duplicated, and out-of-order event delivery without conflating it
    with "did the call itself go through.\""""

    accepted_for_processing: bool
    rejection_reason: str | None
    adapter_call_id: str

    def __post_init__(self) -> None:
        if self.accepted_for_processing and self.rejection_reason is not None:
            raise BrokerSnapshotError("AdapterCallResult.rejection_reason must be None when accepted_for_processing=True")
        if not self.accepted_for_processing and not self.rejection_reason:
            raise BrokerSnapshotError("AdapterCallResult.rejection_reason is required when accepted_for_processing=False")
        if not self.adapter_call_id:
            raise BrokerSnapshotError("AdapterCallResult.adapter_call_id must not be empty")


# --------------------------------------------------------------------------
# Broker snapshots (Section 25)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    client_order_id: str
    instrument_id: str
    side: OrderSide
    order_type: OrderTypeKind
    state: ExecutionOrderState
    original_quantity: Decimal
    current_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Decimal | None
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForceKind
    last_broker_sequence: int

    def __post_init__(self) -> None:
        if not self.broker_order_id or not self.client_order_id or not self.instrument_id:
            raise BrokerSnapshotError("BrokerOrderSnapshot.broker_order_id/client_order_id/instrument_id must not be empty")
        if self.filled_quantity + self.remaining_quantity != self.current_quantity:
            raise BrokerSnapshotError("BrokerOrderSnapshot: filled_quantity + remaining_quantity must equal current_quantity")
        if self.last_broker_sequence < 1:
            raise BrokerSnapshotError(f"BrokerOrderSnapshot.last_broker_sequence must be >= 1, got {self.last_broker_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "broker_order_id": self.broker_order_id, "client_order_id": self.client_order_id, "instrument_id": self.instrument_id, "side": self.side.value,
            "order_type": self.order_type.value, "state": self.state.value, "original_quantity": decimal_to_json(self.original_quantity),
            "current_quantity": decimal_to_json(self.current_quantity), "filled_quantity": decimal_to_json(self.filled_quantity),
            "remaining_quantity": decimal_to_json(self.remaining_quantity),
            "average_fill_price": (None if self.average_fill_price is None else decimal_to_json(self.average_fill_price)),
            "limit_price": (None if self.limit_price is None else decimal_to_json(self.limit_price)),
            "stop_price": (None if self.stop_price is None else decimal_to_json(self.stop_price)), "time_in_force": self.time_in_force.value,
            "last_broker_sequence": self.last_broker_sequence,
        }


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    instrument_id: str
    signed_quantity: Decimal
    average_price: Decimal
    contract_multiplier: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise BrokerSnapshotError("BrokerPositionSnapshot.instrument_id must not be empty")
        if self.contract_multiplier <= 0:
            raise BrokerSnapshotError(f"BrokerPositionSnapshot.contract_multiplier must be > 0, got {self.contract_multiplier!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id, "signed_quantity": decimal_to_json(self.signed_quantity), "average_price": decimal_to_json(self.average_price),
            "contract_multiplier": decimal_to_json(self.contract_multiplier), "realized_pnl": decimal_to_json(self.realized_pnl),
            "unrealized_pnl": decimal_to_json(self.unrealized_pnl),
        }


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    cash: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    accrued_costs: Decimal
    position_count: int
    snapshot_sequence: int
    snapshot_event_time: datetime

    def __post_init__(self) -> None:
        if self.accrued_costs < 0:
            raise BrokerSnapshotError(f"BrokerAccountSnapshot.accrued_costs must be >= 0, got {self.accrued_costs!r}")
        if self.position_count < 0:
            raise BrokerSnapshotError(f"BrokerAccountSnapshot.position_count must be >= 0, got {self.position_count}")
        if self.snapshot_sequence < 0:
            raise BrokerSnapshotError(f"BrokerAccountSnapshot.snapshot_sequence must be >= 0, got {self.snapshot_sequence}")
        if self.snapshot_event_time.tzinfo is None:
            raise BrokerSnapshotError("BrokerAccountSnapshot.snapshot_event_time must be timezone-aware")

    def to_json_dict(self) -> dict[str, object]:
        import pandas as pd

        from quant_platform.ml.persistence import format_utc_timestamp

        return {
            "cash": decimal_to_json(self.cash), "equity": decimal_to_json(self.equity), "realized_pnl": decimal_to_json(self.realized_pnl),
            "unrealized_pnl": decimal_to_json(self.unrealized_pnl), "accrued_costs": decimal_to_json(self.accrued_costs),
            "position_count": self.position_count, "snapshot_sequence": self.snapshot_sequence,
            "snapshot_event_time": format_utc_timestamp(pd.Timestamp(self.snapshot_event_time)),
        }


# --------------------------------------------------------------------------
# The adapter protocol itself (Section 11)
# --------------------------------------------------------------------------
class ExecutionAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def capabilities(self) -> AdapterCapabilities: ...

    def initialize(self, *, execution_session_id: str, event_time: datetime) -> None: ...

    def submit_order(self, command: SubmitOrderCommand, *, event_time: datetime) -> AdapterCallResult: ...

    def cancel_order(self, command: CancelOrderCommand, *, event_time: datetime) -> AdapterCallResult: ...

    def replace_order(self, command: ReplaceOrderCommand, *, event_time: datetime) -> AdapterCallResult: ...

    def poll_events(self, *, after_sequence: int, max_events: int, event_time: datetime) -> tuple[BrokerEvent, ...]: ...

    def query_order(self, *, broker_order_id: str | None, client_order_id: str | None, event_time: datetime) -> BrokerOrderSnapshot | None: ...

    def query_open_orders(self, *, event_time: datetime) -> tuple[BrokerOrderSnapshot, ...]: ...

    def query_positions(self, *, event_time: datetime) -> tuple[BrokerPositionSnapshot, ...]: ...

    def query_account(self, *, event_time: datetime) -> BrokerAccountSnapshot: ...

    def health(self, *, event_time: datetime) -> AdapterHealthSnapshot: ...

    def heartbeat(self, *, event_time: datetime) -> HeartbeatOutcome: ...

    def close(self) -> None: ...


__all__ = [
    "DETERMINISTIC_DUMMY_CAPABILITIES",
    "AdapterCallResult",
    "AdapterCapabilities",
    "BrokerAccountSnapshot",
    "BrokerOrderSnapshot",
    "BrokerPositionSnapshot",
    "ExecutionAdapter",
    "require_capability",
    "require_query_capability",
]
