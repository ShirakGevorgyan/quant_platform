"""Reference strategy: long-only SMA crossover.

Deliberately simple -- it exists to exercise and validate the `Strategy`
interface and the engine end to end (see
`tests/integration/test_full_backtest_sma.py`), not as a strategy
recommended for real capital.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.types import Signal, SignalAction, Timeframe
from quant_platform.strategy.interfaces import Strategy, StrategyContext


@dataclass(slots=True)
class SmaCrossoverStrategy(Strategy):
    """Long when the fast SMA crosses above the slow SMA; flat (exit, no
    shorting) when it crosses back below."""

    timeframe: Timeframe
    fast_period: int = 10
    slow_period: int = 30

    def __post_init__(self) -> None:
        if self.fast_period <= 0:
            raise ValueError(f"fast_period must be positive, got {self.fast_period}")
        if self.slow_period <= self.fast_period:
            raise ValueError(
                f"slow_period ({self.slow_period}) must exceed fast_period ({self.fast_period})"
            )

    def required_warmup(self, timeframe: Timeframe) -> int:
        if timeframe is not self.timeframe:
            return 0
        return self.slow_period + 1  # +1 so a "previous" SMA pair can be computed too

    def on_bar(self, context: StrategyContext) -> Signal:
        window = context.window(self.timeframe)
        min_bars = self.required_warmup(self.timeframe)

        if len(window) < min_bars:
            return Signal(timestamp=context.timestamp, action=SignalAction.HOLD)

        closes = window["close"]
        fast_now = closes.iloc[-self.fast_period :].mean()
        slow_now = closes.iloc[-self.slow_period :].mean()
        fast_prev = closes.iloc[-self.fast_period - 1 : -1].mean()
        slow_prev = closes.iloc[-self.slow_period - 1 : -1].mean()

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up:
            return Signal(
                timestamp=context.timestamp,
                action=SignalAction.LONG,
                metadata={"fast_sma": fast_now, "slow_sma": slow_now},
            )
        if crossed_down:
            return Signal(
                timestamp=context.timestamp,
                action=SignalAction.FLAT,
                metadata={"fast_sma": fast_now, "slow_sma": slow_now},
            )
        return Signal(timestamp=context.timestamp, action=SignalAction.HOLD)
