"""Strategy interface: point-in-time context in, a `Signal` out.

A `Strategy` never sees costs, position sizing, or raw broker state -- it
receives a `StrategyContext` built fresh by the engine each step from
`TimeframeCursor` windows, and returns a directional `Signal`. This
decoupling means a strategy can be unit tested with a hand-built context
and no engine, cost model, or broker simulator involved at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.types import Position, Signal, Timeframe


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to see at one point in time.

    `windows` maps each timeframe the engine is tracking to the bars
    revealed so far on that timeframe (oldest first, via
    `TimeframeCursor.window`) -- by construction there is no way to reach
    data whose close time is after `timestamp`.
    """

    timestamp: datetime
    windows: Mapping[Timeframe, pd.DataFrame]
    position: Position
    account_equity: float

    def window(self, timeframe: Timeframe) -> pd.DataFrame:
        """The revealed-bars window for `timeframe`, or an empty frame if
        the engine is not tracking that timeframe for this run."""
        return self.windows.get(timeframe, pd.DataFrame())


class Strategy(ABC):
    """Base class for all trading strategies."""

    __slots__ = ()  # so concrete `@dataclass(slots=True)` strategies stay slotted

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> Signal:
        """Produce a `Signal` for the current point in time."""
        raise NotImplementedError

    def required_warmup(self, timeframe: Timeframe) -> int:  # noqa: ARG002 - overridable hook, base is a no-op
        """Minimum number of revealed bars on `timeframe` this strategy
        needs before it can produce a meaningful (non-HOLD) signal.
        Defaults to 0 (no requirement); override for strategies that need
        a specific lookback window, so the engine can validate sufficient
        history exists before running."""
        return 0
