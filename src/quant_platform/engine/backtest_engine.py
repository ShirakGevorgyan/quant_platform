"""The backtest engine orchestrator.

Ties `TimeframeCursor` (point-in-time, leak-free data access), `Strategy`
(directional signal), `PositionSizer` (risk-managed quantity),
`BrokerSimulator` (realistic fills and intrabar exit detection), and
`Portfolio` (accounting) into a single sequential, event-driven simulation.

The base timeframe drives the simulation clock one bar at a time; every
other tracked timeframe is exposed only through its own `TimeframeCursor`,
so a strategy can never see a higher-timeframe bar before it has actually
closed relative to the base clock's current instant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType

import pandas as pd

from quant_platform.core.exceptions import ConfigurationError, InsufficientDataError
from quant_platform.core.types import (
    Bar,
    EquityPoint,
    OrderSide,
    Position,
    Signal,
    SignalAction,
    Timeframe,
    Trade,
)
from quant_platform.costs.models import CostModel
from quant_platform.engine.broker_simulator import BrokerSimulator
from quant_platform.engine.portfolio import Portfolio
from quant_platform.multiframe.cursor import TimeframeCursor
from quant_platform.risk.position_sizing import PositionSizer
from quant_platform.strategy.interfaces import Strategy, StrategyContext

logger = logging.getLogger(__name__)

_REASON_SIGNAL = "SIGNAL"
_REASON_REVERSAL = "REVERSAL"
_REASON_END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    initial_capital: float
    final_equity: float

    @property
    def net_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100.0


class BacktestEngine:
    def __init__(
        self,
        *,
        data: dict[Timeframe, pd.DataFrame],
        base_timeframe: Timeframe,
        strategy: Strategy,
        cost_model: CostModel,
        position_sizer: PositionSizer,
        initial_capital: float,
        point_value: float = 1.0,
        symbol: str = "SYMBOL",
        max_window_bars: int = 500,
    ) -> None:
        if base_timeframe not in data:
            raise ConfigurationError(
                f"base_timeframe {base_timeframe.value} has no corresponding entry in `data`"
            )
        if max_window_bars <= 0:
            raise ValueError(f"max_window_bars must be positive, got {max_window_bars}")

        self._base_timeframe = base_timeframe
        # TimeframeCursor's own constructor validates schema/monotonicity for
        # every timeframe we're given, base included -- no duplicate checks needed here.
        self._base_cursor = TimeframeCursor(data[base_timeframe], base_timeframe)
        self._base_open_times = data[base_timeframe]["open_time"].reset_index(drop=True)
        self._higher_cursors: dict[Timeframe, TimeframeCursor] = {
            tf: TimeframeCursor(df, tf) for tf, df in data.items() if tf is not base_timeframe
        }

        self._strategy = strategy
        self._broker = BrokerSimulator(cost_model)
        self._position_sizer = position_sizer
        self._symbol = symbol
        self._point_value = point_value
        self._max_window_bars = max_window_bars

        self._portfolio = Portfolio(initial_capital=initial_capital, point_value=point_value)

        self._entry_bar_index: int | None = None
        self._active_stop_loss: float | None = None
        self._active_take_profit: float | None = None
        self._active_max_hold: timedelta | None = None

    def run(self) -> BacktestResult:
        n = len(self._base_cursor)
        if n == 0:
            raise InsufficientDataError(
                "Base timeframe data is empty; cannot run backtest",
                context={"base_timeframe": self._base_timeframe.value},
            )

        logger.info(
            "Starting backtest: symbol=%s base_timeframe=%s bars=%d initial_capital=%.2f",
            self._symbol, self._base_timeframe.value, n, self._portfolio.initial_capital,
        )

        bar: Bar | None = None
        for i in range(n):
            as_of = self._base_open_times.iloc[i] + self._base_timeframe.duration
            self._base_cursor.advance_to(as_of)
            for cursor in self._higher_cursors.values():
                cursor.advance_to(as_of)

            bar = self._base_cursor.current_bar
            assert bar is not None  # guaranteed: we just advanced past index i

            self._maybe_resolve_open_position(bar, current_index=i)

            windows = {self._base_timeframe: self._base_cursor.window(self._max_window_bars)}
            for tf, cursor in self._higher_cursors.items():
                windows[tf] = cursor.window(self._max_window_bars)

            position = self._portfolio.position or Position(symbol=self._symbol)
            context = StrategyContext(
                timestamp=bar.close_time,
                windows=MappingProxyType(windows),
                position=position,
                account_equity=self._portfolio.equity(bar.close),
            )
            signal = self._strategy.on_bar(context)
            self._handle_signal(signal, bar, current_index=i)

            if i == n - 1 and self._portfolio.has_open_position():
                # Force-close a still-open position on the final bar BEFORE
                # recording that bar's equity point, so there is exactly one
                # point per bar, always reflecting final (realized, if
                # applicable) state. Recording a mark-to-market point here
                # and then a second, separately-timestamped "realized" point
                # after the fact (the previous approach) created a spurious
                # extra equity-curve entry worth about half the spread, which
                # `compute_performance_report`'s return series would have
                # treated as a real period-over-period return.
                self._close_position(bar, exit_reason=_REASON_END_OF_DATA)

            current_price = bar.close if self._portfolio.has_open_position() else None
            self._portfolio.record_equity_point(bar.close_time, current_price)

        final_equity = self._portfolio.equity()
        logger.info(
            "Backtest complete: trades=%d final_equity=%.2f net_return=%.2f%%",
            len(self._portfolio.closed_trades), final_equity,
            (final_equity - self._portfolio.initial_capital) / self._portfolio.initial_capital * 100.0,
        )

        return BacktestResult(
            trades=self._portfolio.closed_trades,
            equity_curve=self._portfolio.equity_curve,
            initial_capital=self._portfolio.initial_capital,
            final_equity=final_equity,
        )

    # ------------------------------------------------------------------
    def _maybe_resolve_open_position(self, bar: Bar, *, current_index: int) -> None:
        if not self._portfolio.has_open_position():
            return
        if self._entry_bar_index is not None and current_index <= self._entry_bar_index:
            return  # never check exits on the same bar a position was opened

        position = self._portfolio.position
        assert position is not None
        side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
        assert position.opened_at is not None

        result = self._broker.resolve_intrabar_exit(
            position_side=side,
            bar=bar,
            entry_time=position.opened_at,
            stop_loss=self._active_stop_loss,
            take_profit=self._active_take_profit,
            max_hold=self._active_max_hold,
        )
        if result is not None:
            quantity = abs(position.quantity)
            fill = self._broker.fill_exit(bar.close_time, side, quantity, result.exit_price)
            trade = self._portfolio.close_position(fill.price, fill.commission, bar.close_time, result.exit_reason)
            self._reset_active_bracket()
            logger.info(
                "Position closed (%s): entry=%.5f exit=%.5f net_pnl=%.2f",
                result.exit_reason, trade.entry_price, trade.exit_price, trade.net_pnl,
            )

    def _handle_signal(self, signal: Signal, bar: Bar, *, current_index: int) -> None:
        has_position = self._portfolio.has_open_position()

        if signal.action is SignalAction.HOLD:
            return

        if signal.action is SignalAction.FLAT:
            if has_position:
                self._close_position(bar, exit_reason=_REASON_SIGNAL)
            return

        desired_side = OrderSide.BUY if signal.action is SignalAction.LONG else OrderSide.SELL

        if has_position:
            position = self._portfolio.position
            assert position is not None
            current_side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
            if current_side is desired_side:
                return  # already positioned this way
            self._close_position(bar, exit_reason=_REASON_REVERSAL)

        self._open_position(signal, bar, desired_side, current_index=current_index)

    def _open_position(self, signal: Signal, bar: Bar, side: OrderSide, *, current_index: int) -> None:
        reference_price = bar.close
        quantity = self._position_sizer.size(
            account_equity=self._portfolio.equity(bar.close),
            entry_price=reference_price,
            point_value=self._point_value,
            stop_loss_price=signal.stop_loss,
            current_volatility=signal.current_volatility,
        )
        if quantity <= 0:
            logger.warning("Signal %s at %s sized to zero quantity; skipping.", signal.action, bar.close_time)
            return

        fill = self._broker.fill_entry(
            bar.close_time, side, quantity, reference_price, current_volatility=signal.current_volatility
        )
        self._portfolio.open_position(self._symbol, side, quantity, fill.price, fill.commission, bar.close_time)
        self._entry_bar_index = current_index

        # `signal.stop_loss`/`take_profit` are absolute MID-price levels (per
        # `Signal`'s own docstring: derived from the strategy's view of market
        # structure, e.g. "stop below the recent swing low") and are used
        # AS-IS here -- NOT re-anchored to `fill.price`.
        #
        # An earlier version re-anchored them to preserve the *risk distance*
        # from the actual (cost-adjusted) fill price, mirroring how a prior
        # project handled broker requotes. That reasoning does not apply
        # here: this engine's fill price differs from `reference_price` by a
        # single, deterministic cost-model computation, not a stochastic
        # requote. Re-anchoring to it double-counts that same cost: once via
        # the shift in the stored trigger level, and again when
        # `resolve_intrabar_exit`'s result is routed back through
        # `fill_exit`, which applies the exit-side spread a second time.
        # Confirmed via adversarial audit with a concrete numeric trace: with
        # a 2-point spread and an intended 10-point mid-price stop, the
        # re-anchored trigger fired at a 9-point mid-price move instead of
        # 10 (stops tightened by half the spread) and, symmetrically, take-
        # profits required an 11-point move instead of 10 (widened by half
        # the spread) -- silently deviating from the strategy's stated risk
        # levels on every trade that set stop_loss/take_profit, in both
        # directions, on every cost model. Every existing unit test used a
        # zero-cost model for its SL/TP checks, which makes fill.price ==
        # reference_price and hides this exact bug -- see
        # tests/unit/test_backtest_engine.py::TestStopLossTakeProfitCostInteraction
        # for the regression coverage that would have caught it.
        self._active_stop_loss = signal.stop_loss
        self._active_take_profit = signal.take_profit
        self._active_max_hold = signal.max_hold

        logger.info(
            "Position opened: side=%s quantity=%.6f fill=%.5f sl=%s tp=%s",
            side.value, quantity, fill.price, self._active_stop_loss, self._active_take_profit,
        )

    def _close_position(self, bar: Bar, *, exit_reason: str) -> None:
        position = self._portfolio.position
        assert position is not None
        side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
        quantity = abs(position.quantity)
        fill = self._broker.fill_exit(bar.close_time, side, quantity, bar.close)
        trade = self._portfolio.close_position(fill.price, fill.commission, bar.close_time, exit_reason)
        self._reset_active_bracket()
        logger.info(
            "Position closed (%s): entry=%.5f exit=%.5f net_pnl=%.2f",
            exit_reason, trade.entry_price, trade.exit_price, trade.net_pnl,
        )

    def _reset_active_bracket(self) -> None:
        self._entry_bar_index = None
        self._active_stop_loss = None
        self._active_take_profit = None
        self._active_max_hold = None
