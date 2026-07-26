"""Simulates order fills and intrabar exit (SL/TP/time-stop) resolution.

Cost application (spread/slippage/commission) and exit-trigger detection
are deliberately separate concerns here: `resolve_intrabar_exit` only
determines *whether* an exit condition was met this bar and at *what raw
level* (the stop/target/close price), never touching cost. The caller
always routes that raw level back through `fill_exit` to get a realistic,
cost-adjusted fill -- so a stop-loss exit pays the spread exactly the same
way a strategy-driven exit does, and cost logic lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from quant_platform.core.types import Bar, Fill, OrderSide
from quant_platform.costs.models import CostModel

SL_REASON = "SL"
TP_REASON = "TP"
TIME_STOP_REASON = "TIME_STOP"


@dataclass(frozen=True, slots=True)
class IntrabarExitResult:
    exit_price: float
    exit_reason: str


class BrokerSimulator:
    """Turns reference prices into realistic fills, and detects intrabar
    stop-loss/take-profit/time-stop triggers against a single closed bar.
    """

    def __init__(self, cost_model: CostModel) -> None:
        self._cost_model = cost_model

    def fill_entry(
        self,
        timestamp: datetime,
        side: OrderSide,
        quantity: float,
        reference_price: float,
        *,
        current_volatility: float | None = None,
    ) -> Fill:
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        fill_price = self._cost_model.entry_fill_price(
            reference_price, side, current_volatility=current_volatility
        )
        commission = self._cost_model.commission(quantity, fill_price)
        return Fill(
            timestamp=timestamp, side=side, quantity=quantity, price=fill_price, commission=commission
        )

    def fill_exit(
        self,
        timestamp: datetime,
        position_side: OrderSide,
        quantity: float,
        reference_price: float,
        *,
        current_volatility: float | None = None,
    ) -> Fill:
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        fill_price = self._cost_model.exit_fill_price(
            reference_price, position_side, current_volatility=current_volatility
        )
        commission = self._cost_model.commission(quantity, fill_price)
        exit_side = OrderSide.SELL if position_side is OrderSide.BUY else OrderSide.BUY
        return Fill(
            timestamp=timestamp, side=exit_side, quantity=quantity, price=fill_price, commission=commission
        )

    def resolve_intrabar_exit(
        self,
        *,
        position_side: OrderSide,
        bar: Bar,
        entry_time: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        max_hold: timedelta | None = None,
    ) -> IntrabarExitResult | None:
        """Check whether `bar` (a single, already-closed bar strictly after
        the entry bar) triggers SL, TP, or a time-stop. If a bar's range
        touches both SL and TP, SL is treated as having been hit first --
        the conservative assumption, since which one actually happened
        first within the bar is not observable from OHLC alone.

        Stop-loss gap-through: a stop order becomes a market order once
        triggered, so if the bar's OPEN is already beyond the stop level
        (a genuine gap, not just an intrabar touch), the realistic fill is
        `bar.open` -- the first price actually traded that bar -- not the
        nominal stop level, which the market never traded at. Filling at
        the nominal level in a gap would silently overstate performance
        (standard convention, matching e.g. Backtrader/Zipline/QuantConnect).

        Take-profit gap-through is deliberately NOT given the same
        treatment: many retail/CFD brokers cap a limit-style take-profit
        fill at the requested level even when price gaps favorably past
        it (no price improvement), which is both the more conservative
        assumption and common real-world execution behavior. This is a
        documented asymmetry, not an oversight.

        Returns None if the position should remain open.
        """
        if position_side is OrderSide.BUY:
            if stop_loss is not None and bar.low <= stop_loss:
                fill_price = min(bar.open, stop_loss)
                return IntrabarExitResult(exit_price=fill_price, exit_reason=SL_REASON)
            if take_profit is not None and bar.high >= take_profit:
                return IntrabarExitResult(exit_price=take_profit, exit_reason=TP_REASON)
        else:
            if stop_loss is not None and bar.high >= stop_loss:
                fill_price = max(bar.open, stop_loss)
                return IntrabarExitResult(exit_price=fill_price, exit_reason=SL_REASON)
            if take_profit is not None and bar.low <= take_profit:
                return IntrabarExitResult(exit_price=take_profit, exit_reason=TP_REASON)

        if max_hold is not None and (bar.close_time - entry_time) >= max_hold:
            return IntrabarExitResult(exit_price=bar.close, exit_reason=TIME_STOP_REASON)

        return None
