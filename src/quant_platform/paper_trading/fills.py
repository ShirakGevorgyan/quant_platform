"""Immutable `Fill` records and cross-fill order-sequence validation
(Milestone 7, Section 11). A `Fill` is content-addressed exactly like an
`OrderRequest`/`MarketEvent` -- once persisted it never mutates (there are
no setter methods on a frozen dataclass at all); "duplicate fills must be
idempotently rejected or recognized as identical" is automatically true
for two fills with identical content, since they compute the SAME
`fill_id` -- true deduplication (not re-applying an already-seen fill to
the account) is the event ledger's job (Section 21), not this module's.

`financing_component` is always `0.0` in this implementation -- Section
16's own recognition-timing rule is that financing is recognized only
when the runner processes a `FinancingEvent` (a session-boundary trigger),
never at fill time. The field exists to match Section 11's required field
list, is validated same as any other cost component, and is reserved for
a future per-fill financing accrual model; today it is always zero and a
mutation test pins that."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import FillValidationError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import OrderSide, PartialFillPolicyKind
from quant_platform.paper_trading.orders import OrderRequest

FILL_KIND = "fill"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise FillValidationError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise FillValidationError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise FillValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise FillValidationError(f"{field_name}: {exc}") from exc


def _require_non_negative_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise FillValidationError(f"{field_name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    session_id: str
    instrument: str
    side: OrderSide
    quantity: float
    price: float
    gross_notional: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    financing_component: float
    execution_time: datetime
    source_market_event_identity: str
    liquidity_assumption: PartialFillPolicyKind
    is_final: bool

    def __post_init__(self) -> None:
        if not self.order_id:
            raise FillValidationError("Fill.order_id must not be empty")
        if not self.session_id:
            raise FillValidationError("Fill.session_id must not be empty")
        if not self.instrument:
            raise FillValidationError("Fill.instrument must not be empty")
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise FillValidationError(f"Fill.quantity must be finite and > 0, got {self.quantity!r}")
        if not math.isfinite(self.price) or self.price <= 0.0:
            raise FillValidationError(f"Fill.price must be finite and > 0, got {self.price!r}")
        _require_non_negative_finite(self.gross_notional, field_name="Fill.gross_notional")
        _require_non_negative_finite(self.spread_cost, field_name="Fill.spread_cost")
        _require_non_negative_finite(self.slippage_cost, field_name="Fill.slippage_cost")
        _require_non_negative_finite(self.commission_cost, field_name="Fill.commission_cost")
        if not math.isfinite(self.financing_component):
            raise FillValidationError(f"Fill.financing_component must be finite, got {self.financing_component!r}")
        _require_tz_aware(self.execution_time, field_name="Fill.execution_time")
        if not is_valid_sha256_hex(self.source_market_event_identity):
            raise FillValidationError(f"Fill.source_market_event_identity must be a valid sha256 hex digest, got {self.source_market_event_identity!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fill_id": self.fill_id, "order_id": self.order_id, "session_id": self.session_id, "instrument": self.instrument,
            "side": self.side.value, "quantity": self.quantity, "price": self.price, "gross_notional": self.gross_notional,
            "spread_cost": self.spread_cost, "slippage_cost": self.slippage_cost, "commission_cost": self.commission_cost,
            "financing_component": self.financing_component, "execution_time": _serialize_timestamp(self.execution_time, field_name="execution_time"),
            "source_market_event_identity": self.source_market_event_identity, "liquidity_assumption": self.liquidity_assumption.value,
            "is_final": self.is_final,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["fill_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Fill:
        return cls(
            fill_id=str(raw["fill_id"]), order_id=str(raw["order_id"]), session_id=str(raw["session_id"]), instrument=str(raw["instrument"]),
            side=OrderSide(raw["side"]), quantity=float(str(raw["quantity"])), price=float(str(raw["price"])),
            gross_notional=float(str(raw["gross_notional"])), spread_cost=float(str(raw["spread_cost"])), slippage_cost=float(str(raw["slippage_cost"])),
            commission_cost=float(str(raw["commission_cost"])), financing_component=float(str(raw["financing_component"])),
            execution_time=_deserialize_timestamp(raw["execution_time"], field_name="execution_time"),
            source_market_event_identity=str(raw["source_market_event_identity"]), liquidity_assumption=PartialFillPolicyKind(raw["liquidity_assumption"]),
            is_final=bool(raw["is_final"]),
        )


def create_fill(
    *, order_id: str, session_id: str, instrument: str, side: OrderSide, quantity: float, price: float, contract_multiplier: float,
    spread_cost: float, slippage_cost: float, commission_cost: float, execution_time: datetime, source_market_event_identity: str,
    liquidity_assumption: PartialFillPolicyKind, is_final: bool,
) -> Fill:
    """The only supported way to mint a fresh `Fill` -- computes `gross_
    notional = price * quantity * contract_multiplier` (enforced by
    construction, not merely validated) and the deterministic `fill_id`
    from every other field. `financing_component` is always `0.0` (see
    module docstring)."""
    gross_notional = price * quantity * contract_multiplier
    provisional = Fill(
        fill_id="0" * 64, order_id=order_id, session_id=session_id, instrument=instrument, side=side, quantity=quantity, price=price,
        gross_notional=gross_notional, spread_cost=spread_cost, slippage_cost=slippage_cost, commission_cost=commission_cost,
        financing_component=0.0, execution_time=execution_time, source_market_event_identity=source_market_event_identity,
        liquidity_assumption=liquidity_assumption, is_final=is_final,
    )
    fill_id = compute_content_id(FILL_KIND, provisional.to_identity_payload())
    return Fill(
        fill_id=fill_id, order_id=order_id, session_id=session_id, instrument=instrument, side=side, quantity=quantity, price=price,
        gross_notional=gross_notional, spread_cost=spread_cost, slippage_cost=slippage_cost, commission_cost=commission_cost,
        financing_component=0.0, execution_time=execution_time, source_market_event_identity=source_market_event_identity,
        liquidity_assumption=liquidity_assumption, is_final=is_final,
    )


def validate_fill_sequence_for_order(order: OrderRequest, fills: list[Fill]) -> None:
    """Cross-fill invariants that no single `Fill` can validate alone --
    every fill belongs to `order`, matches its side, and the sequence's
    cumulative quantity never exceeds `order.quantity`; at most one fill
    is marked `is_final`, and only when cumulative quantity reaches
    `order.quantity` exactly."""
    cumulative = 0.0
    final_seen = False
    for fill in fills:
        if fill.order_id != order.order_id:
            raise FillValidationError(f"Fill {fill.fill_id!r} belongs to order {fill.order_id!r}, not {order.order_id!r}")
        if fill.side is not order.side:
            raise FillValidationError(f"Fill {fill.fill_id!r} side {fill.side.value!r} does not match order side {order.side.value!r}")
        if final_seen:
            raise FillValidationError(f"Fill {fill.fill_id!r} follows an already-final fill for order {order.order_id!r}")
        cumulative += fill.quantity
        if cumulative - order.quantity > 1e-9:
            raise FillValidationError(f"Cumulative filled quantity {cumulative!r} exceeds order quantity {order.quantity!r} for order {order.order_id!r}")
        if fill.is_final:
            final_seen = True
            if abs(cumulative - order.quantity) > 1e-9:
                raise FillValidationError(f"Fill {fill.fill_id!r} marked is_final but cumulative quantity {cumulative!r} != order quantity {order.quantity!r}")


__all__ = ["FILL_KIND", "Fill", "create_fill", "validate_fill_sequence_for_order"]
