"""Deterministic post-trade projection for `quant_platform.portfolio_risk`
(Milestone 9, Phase 2). Every function here is PURE: it takes an
immutable `PortfolioSnapshot`/`PositionSnapshot` and returns a NEW one,
never mutating the original (trivially guaranteed by both being frozen
dataclasses -- there is nothing TO mutate).

GROSS-PRICE-ONLY, NO COST MODELING (explicitly documented, not silently
omitted): this module applies no commission, spread, or slippage cost --
a projected fill happens at exactly `price.ask` (BUY) or `price.bid`
(SELL), mirroring `execution_gateway.dummy_broker`'s own MARKET-order
fill convention. `PortfolioSnapshot.cash` moves by exactly
`quantity * fill_price * contract_multiplier` (decreasing for a BUY,
increasing for a SELL), the same GROSS trade-cash-flow convention
`paper_trading.portfolio.apply_fill_to_portfolio` already uses. This is a
documented Phase 2 simplification (see `docs/portfolio_risk_architecture.md`'s
Known Limitations), the same "no cost is modeled" limitation Milestones 7
and 8 already carry for their own dummy-broker/paper-trading fills.

POSITION PROJECTION MIRRORS `paper_trading.accounting.
apply_fill_to_position`'s SAME weighted-average-cost / realize-on-reduce
logic, Decimal-native and adapted to `PositionSnapshot`'s lighter shape
(a magnitude `quantity` + `side`, not a stored `signed_quantity`; no
`accumulated_transaction_costs`/`accumulated_financing`/`position_version`
fields, since Phase 2 has no cost model and no event history)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.core.exceptions import ExposureCalculationError
from quant_platform.portfolio_risk.models import OrderSide
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    PositionSnapshot,
    PriceSnapshot,
    create_portfolio_snapshot,
)

__all__ = [
    "PortfolioProjection",
    "PositionProjection",
    "TradeRiskClassification",
    "classify_trade_risk",
    "project_fill_price",
    "project_portfolio",
    "project_position",
]


def project_fill_price(price: PriceSnapshot, *, side: OrderSide) -> Decimal:
    """A BUY is assumed filled at `ask`; a SELL at `bid` -- the same
    convention `execution_gateway.dummy_broker` already uses for a
    MARKET order. `price.reference_price` is never used as a fill price
    (only as the post-trade MARK for the resulting position -- see
    `project_position`), since a fill price must reflect the side of the
    book actually crossed."""
    return price.ask if side is OrderSide.BUY else price.bid


class TradeRiskClassification(Enum):
    INCREASING = "increasing"
    """Opens a new position from flat, adds to an existing position in
    the SAME direction, or crosses through zero into a NEW directional
    exposure (even if the resulting magnitude is smaller than the
    original -- a direction change is always treated as risk-increasing,
    never risk-reducing, because it opens fresh directional exposure)."""
    REDUCING = "reducing"
    """Strictly decreases the position's absolute magnitude WITHOUT
    changing its direction, including fully closing it to flat."""
    NEUTRAL = "neutral"
    """Degenerate: the position's signed quantity is unchanged. Given
    `RiskEvaluationRequest.quantity` must be `> 0` (Phase 1's own
    invariant), this can only occur for a flat-to-flat non-trade, which
    should not reach this function in practice; included for
    defensiveness and completeness, never assumed unreachable."""


def classify_trade_risk(*, current_signed_quantity: Decimal, projected_signed_quantity: Decimal) -> TradeRiskClassification:
    if current_signed_quantity == projected_signed_quantity:
        return TradeRiskClassification.NEUTRAL
    if projected_signed_quantity == 0:
        return TradeRiskClassification.REDUCING
    if current_signed_quantity == 0:
        return TradeRiskClassification.INCREASING
    if (current_signed_quantity > 0) != (projected_signed_quantity > 0):
        return TradeRiskClassification.INCREASING
    if abs(projected_signed_quantity) > abs(current_signed_quantity):
        return TradeRiskClassification.INCREASING
    return TradeRiskClassification.REDUCING


@dataclass(frozen=True, slots=True)
class PositionProjection:
    """The result of projecting one trade against one (possibly absent)
    existing position. `new_position` is `None` when the trade fully
    closes the position to flat (Phase 1's own "flat positions are not
    represented" convention -- see `snapshots.py`'s module docstring).
    `realized_pnl_delta` is always reported explicitly, even when
    `new_position is None`, because a fully-closed position's realized
    pnl must still be added to the PORTFOLIO's own running total even
    though the position itself disappears from `PortfolioSnapshot.
    positions`."""

    new_position: PositionSnapshot | None
    realized_pnl_delta: Decimal
    classification: TradeRiskClassification


def project_position(
    current: PositionSnapshot | None, *, instrument_id: str, strategy_id: str, side: OrderSide, quantity: Decimal, fill_price: Decimal,
    mark_price: Decimal, contract_multiplier: Decimal,
) -> PositionProjection:
    if current is not None and (current.instrument_id != instrument_id or current.strategy_id != strategy_id):
        raise ExposureCalculationError(
            f"project_position: current position ({current.instrument_id!r}/{current.strategy_id!r}) does not match the requested "
            f"({instrument_id!r}/{strategy_id!r})"
        )
    current_signed = current.signed_quantity if current is not None else Decimal(0)
    delta_signed = quantity if side is OrderSide.BUY else -quantity
    projected_signed = current_signed + delta_signed
    classification = classify_trade_risk(current_signed_quantity=current_signed, projected_signed_quantity=projected_signed)

    if projected_signed == 0:
        realized_pnl_delta = Decimal(0)
        if current is not None:
            closing_quantity = min(quantity, abs(current_signed))
            realized_pnl_delta = _realized_pnl_delta(current_signed, closing_quantity, current.average_entry_price, fill_price, contract_multiplier)
        return PositionProjection(new_position=None, realized_pnl_delta=realized_pnl_delta, classification=classification)

    same_direction = current is None or current_signed == 0 or (current_signed > 0) == (delta_signed > 0)
    new_side = OrderSide.BUY if projected_signed > 0 else OrderSide.SELL
    new_quantity = abs(projected_signed)

    if same_direction:
        realized_pnl_delta = Decimal(0)
        if current is None or current_signed == 0:
            new_average_entry_price = fill_price
        else:
            prior_notional = abs(current_signed) * current.average_entry_price
            fill_notional = quantity * fill_price
            new_average_entry_price = (prior_notional + fill_notional) / new_quantity
        new_realized_pnl = current.realized_pnl if current is not None else Decimal(0)
    else:
        assert current is not None
        closing_quantity = min(quantity, abs(current_signed))
        realized_pnl_delta = _realized_pnl_delta(current_signed, closing_quantity, current.average_entry_price, fill_price, contract_multiplier)
        remaining = quantity - closing_quantity
        # Reversal: leftover quantity beyond what closed the old position
        # opens a brand-new position, in the OPPOSITE direction, at the
        # fill price -- never at the old average entry price.
        new_average_entry_price = fill_price if remaining > 0 else current.average_entry_price
        new_realized_pnl = current.realized_pnl + realized_pnl_delta

    new_unrealized_pnl = projected_signed * (mark_price - new_average_entry_price) * contract_multiplier
    new_position = PositionSnapshot(
        instrument_id=instrument_id, strategy_id=strategy_id, side=new_side, quantity=new_quantity, average_entry_price=new_average_entry_price,
        mark_price=mark_price, unrealized_pnl=new_unrealized_pnl, realized_pnl=new_realized_pnl, contract_multiplier=contract_multiplier,
    )
    return PositionProjection(new_position=new_position, realized_pnl_delta=realized_pnl_delta, classification=classification)


def _realized_pnl_delta(current_signed: Decimal, closing_quantity: Decimal, average_entry_price: Decimal, fill_price: Decimal, contract_multiplier: Decimal) -> Decimal:
    if current_signed > 0:
        # LONG being reduced: profit when exit price is ABOVE entry.
        return closing_quantity * (fill_price - average_entry_price) * contract_multiplier
    # SHORT being reduced (buy-to-cover): profit when exit price is BELOW entry.
    return closing_quantity * (average_entry_price - fill_price) * contract_multiplier


@dataclass(frozen=True, slots=True)
class PortfolioProjection:
    portfolio: PortfolioSnapshot
    classification: TradeRiskClassification


def project_portfolio(
    portfolio: PortfolioSnapshot, *, instrument_id: str, strategy_id: str, side: OrderSide, quantity: Decimal, price: PriceSnapshot,
    contract_multiplier: Decimal, evaluation_time: datetime,
) -> PortfolioProjection:
    """Projects the WHOLE portfolio forward by one hypothetical trade,
    never mutating `portfolio`. `evaluation_time` is the caller-supplied
    `datetime` the resulting snapshot's own `event_time` is stamped
    with -- never an internal wall-clock read."""
    current = portfolio.position_for(instrument_id=instrument_id, strategy_id=strategy_id)
    fill_price = project_fill_price(price, side=side)
    projection = project_position(
        current, instrument_id=instrument_id, strategy_id=strategy_id, side=side, quantity=quantity, fill_price=fill_price,
        mark_price=price.reference_price, contract_multiplier=contract_multiplier,
    )

    cash_delta = -(quantity * fill_price * contract_multiplier) if side is OrderSide.BUY else (quantity * fill_price * contract_multiplier)
    new_cash = portfolio.cash + cash_delta
    new_realized_pnl = portfolio.realized_pnl + projection.realized_pnl_delta

    new_positions = tuple(p for p in portfolio.positions if not (p.instrument_id == instrument_id and p.strategy_id == strategy_id))
    if projection.new_position is not None:
        new_positions = (*new_positions, projection.new_position)
    new_unrealized_pnl = sum((p.unrealized_pnl for p in new_positions), start=Decimal(0))
    marked_position_value = sum((p.market_value for p in new_positions), start=Decimal(0))
    new_equity = new_cash + marked_position_value
    new_peak_equity = max(portfolio.peak_equity, new_equity)

    projected_portfolio = create_portfolio_snapshot(
        portfolio_id=portfolio.portfolio_id, event_time=evaluation_time, cash=new_cash, equity=new_equity, realized_pnl=new_realized_pnl,
        unrealized_pnl=new_unrealized_pnl, peak_equity=new_peak_equity, daily_start_equity=portfolio.daily_start_equity, positions=new_positions,
        source_execution_session_id=portfolio.source_execution_session_id,
    )
    return PortfolioProjection(portfolio=projected_portfolio, classification=projection.classification)
