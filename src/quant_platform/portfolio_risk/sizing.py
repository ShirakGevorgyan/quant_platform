"""Deterministic position-sizing helpers for `quant_platform.portfolio_risk`
(Milestone 9, Phase 2). Every function is PURE and Decimal-native. This
module implements ONLY policy-limit-based sizing (the maximum quantity
each configured limit independently permits, combined via `min`) --
volatility/Kelly/ATR/model-based capital allocation sizing is explicitly
out of scope for this phase (per the governing instructions) and does
not exist anywhere in this module.

Every `max_quantity_by_*` function returns `None` when its corresponding
policy limit is not configured (unconstrained by that specific limit),
or a `Decimal` otherwise (which may be `<= 0`, meaning that limit alone
already forbids any additional quantity). `compute_maximum_allowed_
quantity` combines every applicable constraint via `min`, together with
the caller's own `requested_quantity` -- the final result can therefore
NEVER exceed what was requested, and is always quantized DOWN
(conservative -- never up) to the nearest multiple of an explicit
caller-supplied `quantity_step`."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

from quant_platform.core.exceptions import PositionSizingError
from quant_platform.portfolio_risk.models import OrderSide

__all__ = [
    "compute_maximum_allowed_quantity",
    "max_quantity_by_cash_buffer",
    "max_quantity_by_instrument_exposure",
    "max_quantity_by_leverage",
    "max_quantity_by_order_notional",
    "max_quantity_by_portfolio_gross_exposure",
    "max_quantity_by_position_notional",
    "quantize_quantity_down",
]


def _require_positive(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise PositionSizingError(f"{field_name} must be finite and > 0, got {value!r}")


def _max_quantity_from_remaining_capacity(*, current_measure: Decimal, limit_value: Decimal | None, reference_price: Decimal, contract_multiplier: Decimal) -> Decimal | None:
    if limit_value is None:
        return None
    _require_positive(reference_price, field_name="reference_price")
    _require_positive(contract_multiplier, field_name="contract_multiplier")
    remaining = limit_value - current_measure
    if remaining <= 0:
        return Decimal(0)
    return remaining / (reference_price * contract_multiplier)


def max_quantity_by_order_notional(*, reference_price: Decimal, contract_multiplier: Decimal, limit_value: Decimal | None) -> Decimal | None:
    if limit_value is None:
        return None
    _require_positive(reference_price, field_name="reference_price")
    _require_positive(contract_multiplier, field_name="contract_multiplier")
    return limit_value / (reference_price * contract_multiplier)


def max_quantity_by_position_notional(*, current_position_notional: Decimal, reference_price: Decimal, contract_multiplier: Decimal, limit_value: Decimal | None) -> Decimal | None:
    return _max_quantity_from_remaining_capacity(current_measure=current_position_notional, limit_value=limit_value, reference_price=reference_price, contract_multiplier=contract_multiplier)


def max_quantity_by_instrument_exposure(*, current_instrument_gross_exposure: Decimal, reference_price: Decimal, contract_multiplier: Decimal, limit_value: Decimal | None) -> Decimal | None:
    return _max_quantity_from_remaining_capacity(current_measure=current_instrument_gross_exposure, limit_value=limit_value, reference_price=reference_price, contract_multiplier=contract_multiplier)


def max_quantity_by_portfolio_gross_exposure(*, current_portfolio_gross_exposure: Decimal, reference_price: Decimal, contract_multiplier: Decimal, limit_value: Decimal | None) -> Decimal | None:
    return _max_quantity_from_remaining_capacity(current_measure=current_portfolio_gross_exposure, limit_value=limit_value, reference_price=reference_price, contract_multiplier=contract_multiplier)


def max_quantity_by_leverage(*, current_portfolio_gross_exposure: Decimal, equity: Decimal, reference_price: Decimal, contract_multiplier: Decimal, limit_value: Decimal | None) -> Decimal | None:
    if limit_value is None:
        return None
    _require_positive(reference_price, field_name="reference_price")
    _require_positive(contract_multiplier, field_name="contract_multiplier")
    if equity <= 0:
        return Decimal(0)  # fail closed -- no safe quantity when equity is non-positive.
    max_gross_exposure = limit_value * equity
    remaining = max_gross_exposure - current_portfolio_gross_exposure
    if remaining <= 0:
        return Decimal(0)
    return remaining / (reference_price * contract_multiplier)


def max_quantity_by_cash_buffer(*, cash: Decimal, minimum_cash_buffer: Decimal | None, reference_price: Decimal, contract_multiplier: Decimal, side: OrderSide) -> Decimal | None:
    """A cash-buffer floor only constrains a BUY (which consumes cash
    under this phase's gross-price-only convention -- see `valuation.py`'s
    module docstring); a SELL always increases cash and is never
    buffer-constrained."""
    if side is OrderSide.SELL:
        return None
    if minimum_cash_buffer is None:
        return None
    _require_positive(reference_price, field_name="reference_price")
    _require_positive(contract_multiplier, field_name="contract_multiplier")
    available = cash - minimum_cash_buffer
    if available <= 0:
        return Decimal(0)
    return available / (reference_price * contract_multiplier)


def quantize_quantity_down(quantity: Decimal, *, step: Decimal) -> Decimal:
    """Rounds `quantity` DOWN (never up -- conservative) to the nearest
    multiple of `step`. Rejects a non-positive `step`."""
    if not step.is_finite() or step <= 0:
        raise PositionSizingError(f"quantity_step must be finite and > 0, got {step!r}")
    if quantity <= 0:
        return Decimal(0)
    units = (quantity / step).to_integral_value(rounding=ROUND_FLOOR)
    return units * step


def compute_maximum_allowed_quantity(*, requested_quantity: Decimal, quantity_step: Decimal, constraints: tuple[Decimal | None, ...]) -> Decimal:
    """Combines `requested_quantity` with every applicable (non-`None`)
    constraint via `min`, then quantizes the result DOWN to `quantity_step`.
    The result can never exceed `requested_quantity` -- `requested_quantity`
    itself participates in the `min` as the first candidate."""
    _require_positive(requested_quantity, field_name="requested_quantity")
    if not quantity_step.is_finite() or quantity_step <= 0:
        raise PositionSizingError(f"quantity_step must be finite and > 0, got {quantity_step!r}")
    effective = requested_quantity
    for candidate in constraints:
        if candidate is not None:
            effective = min(effective, max(Decimal(0), candidate))
    if effective <= 0:
        return Decimal(0)
    return quantize_quantity_down(effective, step=quantity_step)
