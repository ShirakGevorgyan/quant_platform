"""Cost/financing computation (Milestone 7, Section 16). Reuses
`backtesting.costs`'s already-validated per-side formulas directly --
`entry_spread_price_adjustment`/`exit_spread_price_adjustment`/
`entry_slippage_price_adjustment`/`exit_slippage_price_adjustment`/
`per_side_commission_fraction`/`financing_fraction` -- nothing here
reimplements a cost formula a second time.

QUOTE-mode spread is NEVER approximated via `SpreadSpec` -- a `QuoteEvent`
carries the REAL bid/ask, and Section 10.5's own rule ("Buy fills must use
ask-side logic. Sell fills must use bid-side logic") means the raw quoted
price already IS the spread-adjusted price. Applying `SpreadSpec` on top
of a real quote would double-count the spread (Section 16: "Avoid
double-counting costs already embedded in bid/ask prices"). `SpreadSpec`
(via `bar_mode_spread_adjusted_price`) is used ONLY to approximate spread
from a single reference price in BAR mode, where there is no real bid/ask.
`quote_mode_spread_cost_dollars` is diagnostic-only reporting attribution
for QUOTE-mode fills, never an additional price adjustment.

Recognition timing (Section 16): spread and slippage apply exactly once,
priced directly into the fill's own execution price (`fills.py`).
Commission is recognized once per fill, from that fill's own notional.
Financing is recognized only when the runner processes a `FinancingEvent`
(a session-boundary trigger), applied to whatever position exists AT THAT
MOMENT for the elapsed holding interval -- never pre-accrued, never
applied twice for the same interval."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.costs import (
    entry_slippage_price_adjustment,
    entry_spread_price_adjustment,
    exit_slippage_price_adjustment,
    exit_spread_price_adjustment,
    financing_fraction,
    per_side_commission_fraction,
)
from quant_platform.backtesting.models import PositionDirection
from quant_platform.backtesting.specs import CommissionSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingError
from quant_platform.paper_trading.models import OrderSide
from quant_platform.paper_trading.specs import FinancingPolicySpec


def _require_non_negative_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise PaperTradingError(f"{field_name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class FillCostComponents:
    """Every cost line item for ONE fill, in absolute price/dollar units
    (unlike `backtesting.costs.CostBreakdown`'s return-fraction units --
    a paper-trading fill has no fixed "entry_observed_price" denominator
    to normalize against)."""

    spread_cost: float
    slippage_cost: float
    commission_cost: float

    def __post_init__(self) -> None:
        _require_non_negative_finite(self.spread_cost, field_name="FillCostComponents.spread_cost")
        _require_non_negative_finite(self.slippage_cost, field_name="FillCostComponents.slippage_cost")
        _require_non_negative_finite(self.commission_cost, field_name="FillCostComponents.commission_cost")

    @property
    def total_cost(self) -> float:
        return self.spread_cost + self.slippage_cost + self.commission_cost


def bar_mode_spread_adjusted_price(spec: SpreadSpec, reference_price: float, direction: PositionDirection, *, is_entry: bool) -> float:
    """BAR mode ONLY -- approximates the spread-crossed execution price
    from a single reference price (e.g. the bar's close). Never called
    for a QUOTE-mode fill, which already has a real ask/bid to use
    directly."""
    adjustment = entry_spread_price_adjustment(spec, reference_price, direction) if is_entry else exit_spread_price_adjustment(spec, reference_price, direction)
    return reference_price + adjustment


def slippage_adjusted_price(spec: SlippageSpec, reference_price: float, direction: PositionDirection, *, is_entry: bool) -> float:
    """Applied in BOTH market-event modes, on top of whatever price
    spread handling already produced (the real ask/bid in QUOTE mode, or
    `bar_mode_spread_adjusted_price`'s output in BAR mode)."""
    adjustment = entry_slippage_price_adjustment(spec, reference_price, direction) if is_entry else exit_slippage_price_adjustment(spec, reference_price, direction)
    return reference_price + adjustment


def compute_spread_cost_dollars(spec: SpreadSpec, reference_price: float, direction: PositionDirection, *, is_entry: bool, quantity: float, contract_multiplier: float) -> float:
    """BAR mode ONLY -- dollar cost attribution for reporting, matching
    `bar_mode_spread_adjusted_price`'s own adjustment magnitude."""
    adjustment = entry_spread_price_adjustment(spec, reference_price, direction) if is_entry else exit_spread_price_adjustment(spec, reference_price, direction)
    return abs(adjustment) * quantity * contract_multiplier


def compute_slippage_cost_dollars(spec: SlippageSpec, reference_price: float, direction: PositionDirection, *, is_entry: bool, quantity: float, contract_multiplier: float) -> float:
    adjustment = entry_slippage_price_adjustment(spec, reference_price, direction) if is_entry else exit_slippage_price_adjustment(spec, reference_price, direction)
    return abs(adjustment) * quantity * contract_multiplier


def quote_mode_spread_cost_dollars(*, bid: float, ask: float, side: OrderSide, quantity: float, contract_multiplier: float) -> float:
    """QUOTE mode ONLY, diagnostic attribution -- the spread cost already
    embedded in executing at ask (buy) or bid (sell) instead of the mid
    price. Never added a second time to the execution price itself."""
    mid = (bid + ask) / 2.0
    execution_price = ask if side is OrderSide.BUY else bid
    return abs(execution_price - mid) * quantity * contract_multiplier


def compute_commission_dollars(spec: CommissionSpec, *, notional: float) -> float:
    return per_side_commission_fraction(spec, notional=notional) * notional


def compute_financing_cash_delta(policy: FinancingPolicySpec, *, direction: PositionDirection, notional: float, holding_days: float) -> float:
    """Signed cash delta to APPLY DIRECTLY to account cash -- negative
    means cash decreases (a cost to the position holder), positive means
    cash increases (a credit). Which side's `FinancingSpec` applies is
    determined entirely by `direction`, giving genuinely asymmetric
    long/short financing (Section 16) from two independently configured,
    already-validated per-side specs."""
    if direction is PositionDirection.FLAT:
        return 0.0
    spec = policy.long_financing if direction is PositionDirection.LONG else policy.short_financing
    fraction = financing_fraction(spec, holding_days=holding_days)
    return -notional * fraction


__all__ = [
    "FillCostComponents",
    "bar_mode_spread_adjusted_price",
    "compute_commission_dollars",
    "compute_financing_cash_delta",
    "compute_slippage_cost_dollars",
    "compute_spread_cost_dollars",
    "quote_mode_spread_cost_dollars",
    "slippage_adjusted_price",
]
