"""Position accounting (Milestone 7, Section 12). `PositionState` is a
plain float-based, weighted-average-cost per-instrument record --
following the repository's own established exact-financial-arithmetic
convention (float + tight invariant validation, exactly as `backtesting`
uses throughout) rather than introducing `Decimal` a second convention;
Section 12 itself permits either. This module tracks a SINGLE weighted
average entry price per position (not individual FIFO/LIFO lots) -- the
standard, simplest-correct convention for a single-instrument account,
documented explicitly as the chosen model.

Every function here is PURE: it takes a `PositionState` and returns a NEW
one, never mutates in place (there is nothing TO mutate -- the dataclass
is frozen). `apply_fill_to_position`/`apply_financing_to_position`/
`apply_mark_to_position` are the only three ways a `PositionState`
changes, matching the three kinds of ledger events that can affect a
position (a fill, a financing event, a mark-to-market event).

CONTRACT MULTIPLIER (Section 13: "Do not assume every instrument has a
unit contract multiplier of 1"): `average_entry_price`/`last_mark` are
always PURE PER-UNIT PRICES (what you'd see quoted), never pre-multiplied
-- `contract_multiplier` is applied exactly once, at the point a price
delta is converted into a dollar P&L/cost-basis figure
(`gross_cost_basis`, `realized_pnl`, `unrealized_pnl`). `PositionState`
stores its own `contract_multiplier`, fixed once a position is opened
from flat and re-validated (never silently re-derived or overwritten) on
every subsequent fill, DERIVED from that fill's own `gross_notional /
(price * quantity)` rather than trusted from a separately-threaded
parameter -- this makes it structurally impossible for `apply_fill_to_
position` to apply a multiplier inconsistent with whatever `fills.
create_fill` actually used to build the fill in the first place.

LONG AND SHORT UNIFIED VIA SIGN (documented explicitly, per Section 12's
own instruction to "define formulas for both LONG and SHORT explicitly"):
`signed_quantity` is positive for long, negative for short. Unrealized
P&L is ALWAYS `signed_quantity * (mark_price - average_entry_price) *
contract_multiplier` -- this single formula is correct for both
directions without a branch: a long position (positive signed_quantity)
gains when price rises; a short position (negative signed_quantity) gains
when price FALLS, and multiplying a negative quantity by a negative price
delta already produces the correct positive gain. The formulas below
spell out the LONG/SHORT-specific reasoning in comments precisely so this
isn't a "clever trick" no one can audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import PositionAccountingError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.fills import Fill
from quant_platform.paper_trading.models import OrderSide

_ZERO_QUANTITY_TOLERANCE = 1e-9
_MULTIPLIER_CONSISTENCY_TOLERANCE = 1e-6


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PositionAccountingError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PositionAccountingError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PositionAccountingError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PositionAccountingError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PositionState:
    instrument: str
    contract_multiplier: float
    signed_quantity: float
    average_entry_price: float | None
    gross_cost_basis: float
    realized_pnl: float
    unrealized_pnl: float
    accumulated_transaction_costs: float
    accumulated_financing: float
    last_mark: float | None
    last_event_time: datetime | None
    position_version: int

    def __post_init__(self) -> None:
        if not self.instrument:
            raise PositionAccountingError("PositionState.instrument must not be empty")
        if not math.isfinite(self.contract_multiplier) or self.contract_multiplier <= 0.0:
            raise PositionAccountingError(f"PositionState.contract_multiplier must be finite and > 0, got {self.contract_multiplier!r}")
        for field_name, value in (
            ("signed_quantity", self.signed_quantity), ("gross_cost_basis", self.gross_cost_basis), ("realized_pnl", self.realized_pnl),
            ("unrealized_pnl", self.unrealized_pnl), ("accumulated_transaction_costs", self.accumulated_transaction_costs),
            ("accumulated_financing", self.accumulated_financing),
        ):
            if not math.isfinite(value):
                raise PositionAccountingError(f"PositionState.{field_name} must be finite, got {value!r}")
        if self.accumulated_transaction_costs < 0.0:
            raise PositionAccountingError(f"PositionState.accumulated_transaction_costs must be >= 0, got {self.accumulated_transaction_costs!r}")
        if self.gross_cost_basis < 0.0:
            raise PositionAccountingError(f"PositionState.gross_cost_basis must be >= 0, got {self.gross_cost_basis!r}")
        is_flat = abs(self.signed_quantity) <= _ZERO_QUANTITY_TOLERANCE
        if is_flat:
            if self.average_entry_price is not None:
                raise PositionAccountingError("PositionState.average_entry_price must be None when flat")
        elif self.average_entry_price is None or not math.isfinite(self.average_entry_price) or self.average_entry_price <= 0.0:
            raise PositionAccountingError(f"PositionState.average_entry_price must be finite and > 0 when not flat, got {self.average_entry_price!r}")
        if self.last_mark is not None and (not math.isfinite(self.last_mark) or self.last_mark <= 0.0):
            raise PositionAccountingError(f"PositionState.last_mark must be finite and > 0 when present, got {self.last_mark!r}")
        if self.position_version < 0:
            raise PositionAccountingError(f"PositionState.position_version must be >= 0, got {self.position_version}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument, "contract_multiplier": self.contract_multiplier, "signed_quantity": self.signed_quantity,
            "average_entry_price": self.average_entry_price, "gross_cost_basis": self.gross_cost_basis, "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl, "accumulated_transaction_costs": self.accumulated_transaction_costs,
            "accumulated_financing": self.accumulated_financing, "last_mark": self.last_mark,
            "last_event_time": (None if self.last_event_time is None else _serialize_timestamp(self.last_event_time, field_name="last_event_time")),
            "position_version": self.position_version,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PositionState:
        last_event_time_raw = raw.get("last_event_time")
        return cls(
            instrument=str(raw["instrument"]), contract_multiplier=float(str(raw["contract_multiplier"])),
            signed_quantity=float(str(raw["signed_quantity"])),
            average_entry_price=(None if raw.get("average_entry_price") is None else float(str(raw["average_entry_price"]))),
            gross_cost_basis=float(str(raw["gross_cost_basis"])), realized_pnl=float(str(raw["realized_pnl"])),
            unrealized_pnl=float(str(raw["unrealized_pnl"])), accumulated_transaction_costs=float(str(raw["accumulated_transaction_costs"])),
            accumulated_financing=float(str(raw["accumulated_financing"])), last_mark=(None if raw.get("last_mark") is None else float(str(raw["last_mark"]))),
            last_event_time=(None if last_event_time_raw is None else _deserialize_timestamp(last_event_time_raw, field_name="last_event_time")),
            position_version=int(str(raw["position_version"])),
        )


def flat_position(instrument: str, *, contract_multiplier: float) -> PositionState:
    return PositionState(
        instrument=instrument, contract_multiplier=contract_multiplier, signed_quantity=0.0, average_entry_price=None, gross_cost_basis=0.0,
        realized_pnl=0.0, unrealized_pnl=0.0, accumulated_transaction_costs=0.0, accumulated_financing=0.0, last_mark=None, last_event_time=None,
        position_version=0,
    )


def _implied_contract_multiplier(fill: Fill) -> float:
    return fill.gross_notional / (fill.price * fill.quantity)


def apply_fill_to_position(position: PositionState, fill: Fill, *, event_time: datetime) -> PositionState:
    """The one function that implements every one of Section 12's
    required scenarios (open/scale-in/partial-close/full-close/reversal)
    -- which branch applies falls directly out of the fill's signed
    quantity relative to the position's current signed quantity, never a
    separately-selected "mode"."""
    _require_tz_aware(event_time, field_name="event_time")
    if fill.instrument != position.instrument:
        raise PositionAccountingError(f"Fill instrument {fill.instrument!r} does not match position instrument {position.instrument!r}")
    implied_multiplier = _implied_contract_multiplier(fill)
    if abs(implied_multiplier - position.contract_multiplier) > _MULTIPLIER_CONSISTENCY_TOLERANCE * max(1.0, position.contract_multiplier):
        raise PositionAccountingError(
            f"Fill {fill.fill_id!r} implies contract_multiplier={implied_multiplier!r}, inconsistent with "
            f"position.contract_multiplier={position.contract_multiplier!r} for instrument {position.instrument!r}"
        )
    multiplier = position.contract_multiplier

    signed_fill_quantity = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
    current_quantity = position.signed_quantity
    new_quantity = current_quantity + signed_fill_quantity

    realized_pnl_delta = 0.0
    new_average_entry_price = position.average_entry_price

    same_or_flat = current_quantity == 0.0 or (current_quantity > 0) == (signed_fill_quantity > 0)

    if same_or_flat:
        # Opening from flat, or scale-in (adding to an existing position in
        # the SAME direction): the average entry price is the quantity-
        # weighted blend of the old position and this fill's PER-UNIT
        # price -- no P&L is realized, since nothing was closed.
        prior_notional = abs(current_quantity) * (position.average_entry_price or 0.0)
        fill_notional = fill.quantity * fill.price
        new_average_entry_price = (prior_notional + fill_notional) / abs(new_quantity)
    else:
        # The fill is in the OPPOSITE direction of the current position --
        # closing (partially or fully), or a full reversal through zero.
        assert position.average_entry_price is not None
        closing_quantity = min(fill.quantity, abs(current_quantity))
        if current_quantity > 0:
            # LONG being reduced: realized P&L is positive when the exit
            # price is ABOVE the average entry price.
            realized_pnl_delta = closing_quantity * (fill.price - position.average_entry_price) * multiplier
        else:
            # SHORT being reduced: realized P&L is positive when the exit
            # (buy-to-cover) price is BELOW the average entry price.
            realized_pnl_delta = closing_quantity * (position.average_entry_price - fill.price) * multiplier

        remaining_fill_quantity = fill.quantity - closing_quantity
        if abs(new_quantity) <= _ZERO_QUANTITY_TOLERANCE:
            new_quantity = 0.0
            new_average_entry_price = None
        elif remaining_fill_quantity > _ZERO_QUANTITY_TOLERANCE:
            # Reversal: the closing portion realizes P&L against the OLD
            # average price; the leftover fill quantity opens a brand new
            # position in the opposite direction at the fill's own price.
            new_average_entry_price = fill.price
        # else: a partial close leaves the average entry price UNCHANGED --
        # new_average_entry_price already holds position.average_entry_price.

    new_gross_cost_basis = 0.0 if new_average_entry_price is None else abs(new_quantity) * new_average_entry_price * multiplier

    # `unrealized_pnl` must NEVER be carried over unchanged -- the
    # quantity and/or average_entry_price this fill just changed makes
    # the PRE-fill unrealized_pnl stale (most visibly: a position that
    # just closed to flat must report exactly 0.0, never a leftover
    # nonzero value from its last mark). Recompute fresh against
    # `last_mark` (unaffected by a fill) and the NEW position shape --
    # the same formula `apply_mark_to_position` uses.
    if new_average_entry_price is None:
        new_unrealized_pnl = 0.0
    elif position.last_mark is not None:
        new_unrealized_pnl = new_quantity * (position.last_mark - new_average_entry_price) * multiplier
    else:
        new_unrealized_pnl = 0.0

    return PositionState(
        instrument=position.instrument, contract_multiplier=position.contract_multiplier, signed_quantity=new_quantity,
        average_entry_price=new_average_entry_price, gross_cost_basis=new_gross_cost_basis, realized_pnl=position.realized_pnl + realized_pnl_delta,
        unrealized_pnl=new_unrealized_pnl,
        accumulated_transaction_costs=position.accumulated_transaction_costs + fill.spread_cost + fill.slippage_cost + fill.commission_cost,
        accumulated_financing=position.accumulated_financing, last_mark=position.last_mark, last_event_time=event_time,
        position_version=position.position_version + 1,
    )


def apply_financing_to_position(position: PositionState, *, cash_delta: float, event_time: datetime) -> PositionState:
    _require_tz_aware(event_time, field_name="event_time")
    if not math.isfinite(cash_delta):
        raise PositionAccountingError(f"cash_delta must be finite, got {cash_delta!r}")
    return PositionState(
        instrument=position.instrument, contract_multiplier=position.contract_multiplier, signed_quantity=position.signed_quantity,
        average_entry_price=position.average_entry_price, gross_cost_basis=position.gross_cost_basis, realized_pnl=position.realized_pnl,
        unrealized_pnl=position.unrealized_pnl, accumulated_transaction_costs=position.accumulated_transaction_costs,
        accumulated_financing=position.accumulated_financing + cash_delta, last_mark=position.last_mark, last_event_time=event_time,
        position_version=position.position_version + 1,
    )


def apply_mark_to_position(position: PositionState, *, mark_price: float, event_time: datetime) -> PositionState:
    _require_tz_aware(event_time, field_name="event_time")
    if not math.isfinite(mark_price) or mark_price <= 0.0:
        raise PositionAccountingError(f"mark_price must be finite and > 0, got {mark_price!r}")
    if position.average_entry_price is None:
        new_unrealized_pnl = 0.0
    else:
        # Uniform for LONG and SHORT -- see module docstring.
        new_unrealized_pnl = position.signed_quantity * (mark_price - position.average_entry_price) * position.contract_multiplier
    return PositionState(
        instrument=position.instrument, contract_multiplier=position.contract_multiplier, signed_quantity=position.signed_quantity,
        average_entry_price=position.average_entry_price, gross_cost_basis=position.gross_cost_basis, realized_pnl=position.realized_pnl,
        unrealized_pnl=new_unrealized_pnl, accumulated_transaction_costs=position.accumulated_transaction_costs,
        accumulated_financing=position.accumulated_financing, last_mark=mark_price, last_event_time=event_time,
        position_version=position.position_version + 1,
    )


__all__ = ["PositionState", "apply_fill_to_position", "apply_financing_to_position", "apply_mark_to_position", "flat_position"]
