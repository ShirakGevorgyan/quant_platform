"""Transaction-cost framework (Milestone 5, Sections 12/13). Every cost
component is computed as an explicit, separately-named PRICE-SPACE
adjustment (spread, slippage) or RETURN-FRACTION charge (commission,
financing) -- never one unexplained "cost" field.

THE ADVERSE-PRICE SIGN CONVENTION (Section 13)
--------------------------------------------------------------------------
Long entry price increases; long exit price decreases; short entry price
decreases; short exit price increases -- i.e. every adjustment always
moves the realized price AGAINST the position, on both legs. This mirrors
`quant_platform.costs.models.FixedSpreadCostModel`'s already-established,
already-used sign convention exactly (`entry = reference + side.sign *
adverse`, `exit = reference - position_side.sign * adverse`) -- the
MATH is reused verbatim (a spread/slippage adjustment is a spread/
slippage adjustment regardless of which package computes it), only
re-expressed against this package's own `PositionDirection` enum instead
of the older engine's `OrderSide`, since this package does not depend on
that ABC-based, mutable-state engine (see `backtesting.models`'s module
docstring for why).

WHY SPREAD/SLIPPAGE ARE PRICE-SPACE AND COMMISSION/FINANCING ARE
RETURN-FRACTION
--------------------------------------------------------------------------
Spread and slippage are genuine PRICE adjustments (Section 10 requires
persisting "observed market price" separately from "final effective fill
price"). Commission and financing are FEES, not price movements -- a
per-side basis-point commission is already a return fraction by
definition, and a fixed-per-trade commission is converted to a return
fraction via the position's notional (never mixed into the fill price
itself, which would double-count it against `initial_notional`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import CostModelError

_BPS_DIVISOR = 10_000.0


def _direction_sign(direction: PositionDirection) -> int:
    if direction is PositionDirection.LONG:
        return 1
    if direction is PositionDirection.SHORT:
        return -1
    raise CostModelError(f"_direction_sign: PositionDirection.FLAT has no sign -- got {direction!r}")


def _require_finite_positive_price(price: float, *, field_name: str) -> None:
    if not math.isfinite(price) or price <= 0.0:
        raise CostModelError(f"{field_name} must be a finite, positive price, got {price!r}")


def entry_spread_price_adjustment(spec: SpreadSpec, reference_price: float, direction: PositionDirection) -> float:
    """Signed price-space adjustment: ADD this to `reference_price` to
    obtain the spread-adjusted entry price. Always adverse (see module
    docstring)."""
    _require_finite_positive_price(reference_price, field_name="reference_price")
    half_spread = _half_spread_price(spec, reference_price)
    return _direction_sign(direction) * half_spread


def exit_spread_price_adjustment(spec: SpreadSpec, reference_price: float, direction: PositionDirection) -> float:
    """Signed price-space adjustment: ADD this to `reference_price` to
    obtain the spread-adjusted exit price (closing a position originally
    opened in `direction`)."""
    _require_finite_positive_price(reference_price, field_name="reference_price")
    half_spread = _half_spread_price(spec, reference_price)
    return -_direction_sign(direction) * half_spread


def _half_spread_price(spec: SpreadSpec, reference_price: float) -> float:
    if spec.kind is SpreadModelKind.ZERO:
        return 0.0
    if spec.kind is SpreadModelKind.FIXED_PRICE_UNITS:
        assert spec.price_units is not None
        return spec.price_units / 2.0
    if spec.kind is SpreadModelKind.FIXED_BASIS_POINTS:
        assert spec.basis_points is not None
        return (spec.basis_points / _BPS_DIVISOR) * reference_price / 2.0
    if spec.kind is SpreadModelKind.BID_ASK_OBSERVED:
        raise CostModelError(
            "_half_spread_price: kind=bid_ask_observed has no fixed half-spread -- the bid/ask-aware fill "
            "path (backtesting.fills) must read the bar's own bid/ask columns directly, never this function"
        )
    raise CostModelError(f"Unsupported SpreadModelKind: {spec.kind!r}")  # pragma: no cover - exhaustive enum


def entry_slippage_price_adjustment(
    spec: SlippageSpec, reference_price: float, direction: PositionDirection,
) -> float:
    _require_finite_positive_price(reference_price, field_name="reference_price")
    magnitude = _slippage_price_magnitude(spec, reference_price)
    return _direction_sign(direction) * magnitude


def exit_slippage_price_adjustment(
    spec: SlippageSpec, reference_price: float, direction: PositionDirection,
) -> float:
    _require_finite_positive_price(reference_price, field_name="reference_price")
    magnitude = _slippage_price_magnitude(spec, reference_price)
    return -_direction_sign(direction) * magnitude


def _slippage_price_magnitude(spec: SlippageSpec, reference_price: float) -> float:
    if spec.kind is SlippageModelKind.ZERO:
        return 0.0
    if spec.kind is SlippageModelKind.FIXED_PRICE_UNITS:
        assert spec.price_units is not None
        return spec.price_units
    if spec.kind is SlippageModelKind.FIXED_BASIS_POINTS:
        assert spec.basis_points is not None
        return (spec.basis_points / _BPS_DIVISOR) * reference_price
    raise CostModelError(f"Unsupported SlippageModelKind: {spec.kind!r}")  # pragma: no cover - exhaustive enum


def per_side_commission_fraction(spec: CommissionSpec, *, notional: float) -> float:
    """Return-fraction commission for ONE side (charged once on entry,
    once on exit -- never combined into a single per-trade figure at this
    layer)."""
    if spec.kind is CommissionModelKind.ZERO:
        return 0.0
    if spec.kind is CommissionModelKind.PER_SIDE_BASIS_POINTS:
        assert spec.per_side_basis_points is not None
        return spec.per_side_basis_points / _BPS_DIVISOR
    if spec.kind is CommissionModelKind.FIXED_PER_TRADE:
        assert spec.fixed_per_trade is not None
        if notional <= 0.0:
            raise CostModelError(f"per_side_commission_fraction: notional must be > 0 to convert a fixed commission to a fraction, got {notional!r}")
        return spec.fixed_per_trade / notional
    raise CostModelError(f"Unsupported CommissionModelKind: {spec.kind!r}")  # pragma: no cover - exhaustive enum


def financing_fraction(spec: FinancingSpec, *, holding_days: float) -> float:
    if spec.kind is FinancingModelKind.NONE:
        return 0.0
    if spec.kind is FinancingModelKind.FIXED_DAILY_BASIS_POINTS:
        assert spec.daily_basis_points is not None
        if holding_days < 0.0:
            raise CostModelError(f"financing_fraction: holding_days must be >= 0, got {holding_days!r}")
        return (spec.daily_basis_points / _BPS_DIVISOR) * holding_days
    raise CostModelError(f"Unsupported FinancingModelKind: {spec.kind!r}")  # pragma: no cover - exhaustive enum


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Section 12: every cost component kept separate, summing exactly
    (by construction) to `total_cost` -- all in RETURN-FRACTION units,
    each measured relative to the SAME `entry_observed_price` denominator
    (see `backtesting.returns`'s module docstring for why this makes
    `net_return = gross_return - total_cost` hold EXACTLY, not merely
    approximately)."""

    entry_spread_cost: float
    exit_spread_cost: float
    entry_commission: float
    exit_commission: float
    entry_slippage: float
    exit_slippage: float
    financing_cost: float

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_spread_cost", self.entry_spread_cost), ("exit_spread_cost", self.exit_spread_cost),
            ("entry_commission", self.entry_commission), ("exit_commission", self.exit_commission),
            ("entry_slippage", self.entry_slippage), ("exit_slippage", self.exit_slippage),
            ("financing_cost", self.financing_cost),
        ):
            if not math.isfinite(value):
                raise CostModelError(f"CostBreakdown.{name} must be finite, got {value!r}")
            if value < 0.0:
                raise CostModelError(f"CostBreakdown.{name} must be >= 0 (this milestone's reference cost models support no negative-cost rebate policy), got {value!r}")

    @property
    def total_cost(self) -> float:
        return (
            self.entry_spread_cost + self.exit_spread_cost + self.entry_commission + self.exit_commission
            + self.entry_slippage + self.exit_slippage + self.financing_cost
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entry_spread_cost": self.entry_spread_cost, "exit_spread_cost": self.exit_spread_cost,
            "entry_commission": self.entry_commission, "exit_commission": self.exit_commission,
            "entry_slippage": self.entry_slippage, "exit_slippage": self.exit_slippage,
            "financing_cost": self.financing_cost, "total_cost": self.total_cost,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostBreakdown:
        breakdown = cls(
            entry_spread_cost=float(str(raw["entry_spread_cost"])), exit_spread_cost=float(str(raw["exit_spread_cost"])),
            entry_commission=float(str(raw["entry_commission"])), exit_commission=float(str(raw["exit_commission"])),
            entry_slippage=float(str(raw["entry_slippage"])), exit_slippage=float(str(raw["exit_slippage"])),
            financing_cost=float(str(raw["financing_cost"])),
        )
        persisted_total = float(str(raw["total_cost"]))
        if abs(breakdown.total_cost - persisted_total) > 1e-9:
            raise CostModelError(
                f"CostBreakdown: persisted total_cost={persisted_total!r} does not match the sum of its own "
                f"itemized components ({breakdown.total_cost!r}) -- tampered or corrupted cost breakdown"
            )
        return breakdown


__all__ = [
    "CostBreakdown",
    "entry_slippage_price_adjustment",
    "entry_spread_price_adjustment",
    "exit_slippage_price_adjustment",
    "exit_spread_price_adjustment",
    "financing_fraction",
    "per_side_commission_fraction",
]
