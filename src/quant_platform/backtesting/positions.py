"""Position-intent and open-position value types (Milestone 5, Section 9)
-- deliberately distinct from `Signal` (prediction-derived intent, no
market data) and `TradeRecord` (the terminal, closed/discarded record).
`OpenPosition` is the transient carrier `backtesting.execution`'s
chronological walk uses between an entry fill and its eventual exit --
never persisted on its own (only the final `TradeRecord` is an artifact),
and never mutated: closing a position means constructing a NEW
`TradeRecord` from an `OpenPosition` plus an exit fill, never editing the
`OpenPosition` in place."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.models import PositionDirection, SignalReasonCode
from quant_platform.core.exceptions import ExecutionSimulationError


@dataclass(frozen=True, slots=True)
class OpenPosition:
    direction: PositionDirection
    signal_sample_position: int
    signal_timestamp: str
    decision_timestamp: str
    entry_bar_position: int
    entry_timestamp: str
    entry_observed_price: float
    entry_effective_price: float
    entry_spread_cost: float
    entry_commission: float
    entry_slippage: float
    confidence: float
    uncertainty: float
    calibrated_probability: float
    entry_reason: SignalReasonCode

    def __post_init__(self) -> None:
        if self.direction is PositionDirection.FLAT:
            raise ExecutionSimulationError("OpenPosition.direction must be LONG or SHORT, never FLAT")
        if self.entry_bar_position < 0:
            raise ExecutionSimulationError(f"OpenPosition.entry_bar_position must be >= 0, got {self.entry_bar_position}")
        for name, value in (
            ("entry_observed_price", self.entry_observed_price), ("entry_effective_price", self.entry_effective_price),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ExecutionSimulationError(f"OpenPosition.{name} must be finite and positive, got {value!r}")


__all__ = ["OpenPosition"]
