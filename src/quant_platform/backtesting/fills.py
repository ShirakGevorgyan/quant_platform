"""Fill price calculation (Milestone 5, Section 10) -- `compute_fill_price`
is the ONE authoritative function producing an executable price, used
identically for both entry and exit (via `is_entry`), so spread/slippage
can never be applied twice or inconsistently between the two legs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from quant_platform.backtesting.costs import (
    entry_slippage_price_adjustment,
    entry_spread_price_adjustment,
    exit_slippage_price_adjustment,
    exit_spread_price_adjustment,
)
from quant_platform.backtesting.models import EntryPolicyKind, PositionDirection, PriceBasisKind
from quant_platform.backtesting.specs import EntrySpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import FillCalculationError

_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class FillPriceResult:
    observed_price: float
    spread_adjustment: float
    slippage_adjustment: float
    effective_price: float

    def __post_init__(self) -> None:
        for name, value in (
            ("observed_price", self.observed_price), ("spread_adjustment", self.spread_adjustment),
            ("slippage_adjustment", self.slippage_adjustment), ("effective_price", self.effective_price),
        ):
            if not math.isfinite(value):
                raise FillCalculationError(f"FillPriceResult.{name} must be finite, got {value!r}")
        if self.observed_price <= 0.0 or self.effective_price <= 0.0:
            raise FillCalculationError(
                f"FillPriceResult: observed_price ({self.observed_price}) and effective_price "
                f"({self.effective_price}) must both be positive"
            )
        expected = self.observed_price + self.spread_adjustment + self.slippage_adjustment
        if abs(self.effective_price - expected) > _TOLERANCE:
            raise FillCalculationError(
                f"FillPriceResult: effective_price ({self.effective_price}) does not equal observed_price + "
                f"spread_adjustment + slippage_adjustment ({expected}) -- spread/slippage applied inconsistently "
                "or applied more than once"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "observed_price": self.observed_price, "spread_adjustment": self.spread_adjustment,
            "slippage_adjustment": self.slippage_adjustment, "effective_price": self.effective_price,
        }


def compute_fill_price(
    *, reference_price: float, direction: PositionDirection, is_entry: bool, spread_spec: SpreadSpec, slippage_spec: SlippageSpec,
) -> FillPriceResult:
    """THE one authoritative fill-price function (Section 10). `is_entry`
    selects which of `backtesting.costs`'s entry-/exit- adjustment
    functions apply -- both funnel through the exact same
    `observed + spread_adjustment + slippage_adjustment` composition
    here, so a caller cannot accidentally apply either component twice."""
    if direction is PositionDirection.FLAT:
        raise FillCalculationError("compute_fill_price: direction must be LONG or SHORT, never FLAT")
    if reference_price <= 0.0 or not math.isfinite(reference_price):
        raise FillCalculationError(f"compute_fill_price: reference_price must be finite and positive, got {reference_price!r}")

    if is_entry:
        spread_adjustment = entry_spread_price_adjustment(spread_spec, reference_price, direction)
        slippage_adjustment = entry_slippage_price_adjustment(slippage_spec, reference_price, direction)
    else:
        spread_adjustment = exit_spread_price_adjustment(spread_spec, reference_price, direction)
        slippage_adjustment = exit_slippage_price_adjustment(slippage_spec, reference_price, direction)

    effective_price = reference_price + spread_adjustment + slippage_adjustment
    return FillPriceResult(
        observed_price=reference_price, spread_adjustment=spread_adjustment, slippage_adjustment=slippage_adjustment,
        effective_price=effective_price,
    )


def _is_buy_side(direction: PositionDirection, *, is_entry: bool) -> bool:
    return (direction is PositionDirection.LONG) == is_entry


def select_basis_reference_price(bar: pd.Series, *, price_basis: PriceBasisKind, direction: PositionDirection, is_entry: bool) -> float:
    """Section 6/10: which raw, UNADJUSTED market price a fill's spread/
    slippage adjustment is computed relative to, per the backtest's
    GENERAL `price_basis` declaration -- used for every EXIT (Section 11's
    exit policies are purely about WHEN, never WHICH price column) and as
    the entry fallback for `EntrySpec.kind=DELAYED_BAR` (which likewise
    declares no price-column preference of its own). `price_basis=BID_ASK`
    requires the bar to carry real `bid`/`ask` columns (validated by
    `backtesting.models.validate_market_bar_frame` upstream) -- buying
    occurs at ask, selling at bid, exactly Section 10's explicit
    requirement; this function itself does not add or remove spread, it
    only SELECTS which raw column the caller's `compute_fill_price` call
    should treat as `reference_price`."""
    if price_basis is PriceBasisKind.CLOSE:
        return float(bar["close"])
    if price_basis is PriceBasisKind.MID:
        if "bid" in bar.index and "ask" in bar.index and pd.notna(bar["bid"]) and pd.notna(bar["ask"]):
            return (float(bar["bid"]) + float(bar["ask"])) / 2.0
        return float(bar["close"])
    if price_basis is PriceBasisKind.BID_ASK:
        if "bid" not in bar.index or "ask" not in bar.index or pd.isna(bar["bid"]) or pd.isna(bar["ask"]):
            raise FillCalculationError("select_basis_reference_price: price_basis=bid_ask requires a bar with real bid/ask values")
        return float(bar["ask"]) if _is_buy_side(direction, is_entry=is_entry) else float(bar["bid"])
    raise FillCalculationError(f"select_basis_reference_price: unsupported price_basis {price_basis!r}")  # pragma: no cover - exhaustive enum


def select_entry_reference_price(bar: pd.Series, *, entry_spec: EntrySpec, price_basis: PriceBasisKind, direction: PositionDirection) -> float:
    """Section 10 A-D: `EntrySpec.kind` -- not the general `price_basis`
    -- determines WHICH price column an entry fill is computed relative
    to (this was a real defect found and fixed during development:
    `NEXT_BAR_OPEN` must use the bar's OWN `open` column, never silently
    fall back to `price_basis`'s close/mid/bid-ask choice)."""
    if entry_spec.kind is EntryPolicyKind.NEXT_BAR_OPEN:
        return float(bar["open"])
    if entry_spec.kind is EntryPolicyKind.NEXT_BAR_MID:
        if "bid" in bar.index and "ask" in bar.index and pd.notna(bar["bid"]) and pd.notna(bar["ask"]):
            return (float(bar["bid"]) + float(bar["ask"])) / 2.0
        return float(bar["open"])
    if entry_spec.kind is EntryPolicyKind.NEXT_BAR_SIDE_AWARE:
        if "bid" in bar.index and "ask" in bar.index and pd.notna(bar["bid"]) and pd.notna(bar["ask"]):
            return float(bar["ask"]) if _is_buy_side(direction, is_entry=True) else float(bar["bid"])
        return float(bar["open"])  # no bid/ask available -- fall back to the bar's own open, spread modeled separately
    if entry_spec.kind is EntryPolicyKind.DELAYED_BAR:
        return select_basis_reference_price(bar, price_basis=price_basis, direction=direction, is_entry=True)
    raise FillCalculationError(f"select_entry_reference_price: unsupported EntryPolicyKind {entry_spec.kind!r}")  # pragma: no cover - exhaustive enum


def select_exit_reference_price(bar: pd.Series, *, price_basis: PriceBasisKind, direction: PositionDirection) -> float:
    return select_basis_reference_price(bar, price_basis=price_basis, direction=direction, is_entry=False)


__all__ = [
    "FillPriceResult",
    "compute_fill_price",
    "select_basis_reference_price",
    "select_entry_reference_price",
    "select_exit_reference_price",
]
