"""Single-symbol, single-position portfolio state.

Milestone 1 deliberately supports one open position at a time in one
symbol -- matching the position-management constraint of the reference
strategy this engine's design was validated against, and avoiding
premature multi-asset portfolio accounting (margining, cross-symbol
correlation, netting) that a later milestone should own deliberately
rather than bolt on here as an afterthought.
"""

from __future__ import annotations

from datetime import datetime

from quant_platform.core.exceptions import EngineError
from quant_platform.core.types import EquityPoint, OrderSide, Position, Trade


class Portfolio:
    def __init__(self, initial_capital: float, point_value: float = 1.0) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if point_value <= 0:
            raise ValueError(f"point_value must be positive, got {point_value}")

        self._initial_capital = initial_capital
        self._point_value = point_value
        self._cash = initial_capital
        self._position: Position | None = None
        self._closed_trades: list[Trade] = []
        self._equity_curve: list[EquityPoint] = []
        self._peak_equity = initial_capital

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> Position | None:
        return self._position

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed_trades)

    @property
    def equity_curve(self) -> list[EquityPoint]:
        return list(self._equity_curve)

    def has_open_position(self) -> bool:
        return self._position is not None and not self._position.is_flat

    def equity(self, current_price: float | None = None) -> float:
        """Total account value: cash plus unrealized P&L on any open
        position, marked to `current_price`. Falls back to cash-only if
        there is no open position or no price is supplied."""
        if not self.has_open_position() or current_price is None:
            return self._cash
        assert self._position is not None  # for type-checkers; guaranteed by has_open_position()
        return self._cash + self._position.unrealized_pnl(current_price, self._point_value)

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        fill_price: float,
        commission: float,
        timestamp: datetime,
    ) -> None:
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if self.has_open_position():
            raise EngineError(
                "Cannot open a new position while one is already open",
                context={"symbol": symbol, "existing_symbol": self._position.symbol if self._position else None},
            )

        self._cash -= commission
        self._position = Position(
            symbol=symbol,
            quantity=quantity * side.sign,
            average_entry_price=fill_price,
            entry_commission=commission,
            opened_at=timestamp,
        )

    def close_position(
        self, exit_price: float, commission: float, timestamp: datetime, exit_reason: str
    ) -> Trade:
        """`commission` here is the EXIT-side commission only; the trade's
        `total_cost` also folds in the entry commission recorded when the
        position was opened, so `trade.net_pnl` reconciles exactly with the
        change in `cash` this round trip produced."""
        if not self.has_open_position():
            raise EngineError("No open position to close")
        position = self._position
        assert position is not None
        assert position.opened_at is not None, "invariant: an open position always has opened_at set"

        gross_pnl = (exit_price - position.average_entry_price) * position.quantity * self._point_value
        self._cash += gross_pnl - commission

        trade = Trade(
            entry_time=position.opened_at,
            exit_time=timestamp,
            side=OrderSide.BUY if position.quantity > 0 else OrderSide.SELL,
            entry_price=position.average_entry_price,
            exit_price=exit_price,
            quantity=abs(position.quantity),
            gross_pnl=gross_pnl,
            total_cost=position.entry_commission + commission,
            exit_reason=exit_reason,
        )
        self._closed_trades.append(trade)
        self._position = None
        return trade

    def record_equity_point(self, timestamp: datetime, current_price: float | None = None) -> EquityPoint:
        equity = self.equity(current_price)
        self._peak_equity = max(self._peak_equity, equity)
        drawdown_pct = (
            (self._peak_equity - equity) / self._peak_equity * 100.0 if self._peak_equity > 0 else 0.0
        )
        point = EquityPoint(timestamp=timestamp, cash=self._cash, equity=equity, drawdown_pct=drawdown_pct)
        self._equity_curve.append(point)
        return point
