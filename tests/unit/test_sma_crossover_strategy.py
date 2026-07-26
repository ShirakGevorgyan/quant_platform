"""Tests for the reference SmaCrossoverStrategy, using hand-verified
crossover fixtures so expected signals can be checked by arithmetic, not
just "it ran without crashing"."""

from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType

import pandas as pd
import pytest

from quant_platform.core.types import Position, SignalAction, Timeframe
from quant_platform.strategy.examples.sma_crossover import SmaCrossoverStrategy
from quant_platform.strategy.interfaces import StrategyContext

UTC = timezone.utc


def _frame_from_closes(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    open_times = pd.date_range(start=datetime(2024, 1, 1, tzinfo=UTC), periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def _context(closes: list[float], timeframe: Timeframe = Timeframe.M15) -> StrategyContext:
    frame = _frame_from_closes(closes)
    return StrategyContext(
        timestamp=frame["open_time"].iloc[-1] + timeframe.duration,
        windows=MappingProxyType({timeframe: frame}),
        position=Position(symbol="TEST"),
        account_equity=10_000.0,
    )


class TestConstructionValidation:
    def test_rejects_non_positive_fast_period(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=0, slow_period=10)

    def test_rejects_slow_period_not_exceeding_fast(self) -> None:
        with pytest.raises(ValueError, match="slow_period"):
            SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=10, slow_period=10)


class TestRequiredWarmup:
    def test_warmup_is_slow_period_plus_one_on_own_timeframe(self) -> None:
        strategy = SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=2, slow_period=4)
        assert strategy.required_warmup(Timeframe.M15) == 5

    def test_warmup_is_zero_on_other_timeframes(self) -> None:
        strategy = SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=2, slow_period=4)
        assert strategy.required_warmup(Timeframe.H1) == 0


class TestSignalGeneration:
    """fast=2, slow=4 -> requires 5 bars: fast/slow SMA at the last bar vs
    the bar before it. Fixtures are hand-computed so expected behavior is
    verifiable by arithmetic, not just 'the code says so'."""

    STRATEGY = SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=2, slow_period=4)

    def test_insufficient_bars_holds(self) -> None:
        context = _context([10.0, 10.0, 10.0])  # only 3 bars, need 5
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.HOLD

    def test_flat_series_holds(self) -> None:
        context = _context([10.0] * 6)
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.HOLD

    def test_bullish_crossover_emits_long(self) -> None:
        # fast_prev=mean(10,10)=10, slow_prev=mean(10,10,10,10)=10 (equal, not yet crossed)
        # fast_now=mean(10,20)=15, slow_now=mean(10,10,10,20)=12.5 (now above)
        context = _context([10.0, 10.0, 10.0, 10.0, 10.0, 20.0])
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.LONG
        assert signal.metadata["fast_sma"] == pytest.approx(15.0)
        assert signal.metadata["slow_sma"] == pytest.approx(12.5)

    def test_bearish_crossover_emits_flat(self) -> None:
        # Mirror image of the bullish case: a downward jump on the last bar.
        context = _context([10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.FLAT

    def test_no_crossover_on_smooth_uptrend_holds(self) -> None:
        # A gentle, already-established uptrend where fast was already
        # above slow on the prior bar too -- no fresh crossover.
        context = _context([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.HOLD

    def test_missing_timeframe_in_context_is_treated_as_insufficient_data(self) -> None:
        context = StrategyContext(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            windows=MappingProxyType({}),
            position=Position(symbol="TEST"),
            account_equity=10_000.0,
        )
        signal = self.STRATEGY.on_bar(context)
        assert signal.action is SignalAction.HOLD
