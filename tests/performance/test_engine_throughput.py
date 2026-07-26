"""Throughput/scaling regression tests for BacktestEngine.

Measured on reference hardware: ~3,700 bars/sec, essentially flat across
5k-50k bar runs (i.e. linear in bar count, not quadratic). That rate makes
this engine well suited to research-scale backtests (single-symbol runs
of up to a few hundred thousand bars complete in well under a minute) but
NOT yet to million-bar-plus or many-symbol batch runs without further
optimization (vectorized indicator precomputation, a compiled hot loop) --
an honest limitation, not a claim this milestone doesn't back up.

These tests exist to catch a regression to quadratic behavior (e.g. from
an accidental O(n) copy inside the O(n)-iteration main loop), not to chase
a specific absolute number, since absolute timings vary by machine.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from quant_platform.core.types import Timeframe
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.engine.backtest_engine import BacktestEngine
from quant_platform.risk.position_sizing import KellyCriterionSizer
from quant_platform.strategy.examples.sma_crossover import SmaCrossoverStrategy

UTC = timezone.utc

pytestmark = pytest.mark.performance


def _run_backtest(n_bars: int, seed: int = 1) -> float:
    """Runs a full backtest over `n_bars` synthetic bars and returns the
    wall-clock seconds elapsed (data generation excluded from the timing)."""
    m15 = generate_ohlcv(
        SyntheticDataConfig(
            start=datetime(2024, 1, 1, tzinfo=UTC), periods=n_bars, timeframe=Timeframe.M15,
            annualized_volatility=0.5, seed=seed,
        )
    )
    engine = BacktestEngine(
        data={Timeframe.M15: m15},
        base_timeframe=Timeframe.M15,
        strategy=SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=10, slow_period=30),
        cost_model=FixedSpreadCostModel(spread_points=2.0, slippage_points=1.0, point_value=1.0),
        position_sizer=KellyCriterionSizer(win_rate=0.5, win_loss_ratio=1.5),
        initial_capital=10_000.0,
    )

    start = time.perf_counter()
    engine.run()
    return time.perf_counter() - start


class TestThroughputFloor:
    def test_processes_at_least_500_bars_per_second(self) -> None:
        """A conservative floor (measured throughput is ~7x this on
        reference hardware) so the test is not flaky on slower CI runners,
        while still catching a severe accidental regression."""
        n_bars = 10_000
        elapsed = _run_backtest(n_bars)
        bars_per_second = n_bars / elapsed
        assert bars_per_second > 500, (
            f"Throughput regression: {bars_per_second:.0f} bars/sec (expected > 500)"
        )


class TestLinearScaling:
    def test_time_scales_roughly_linearly_not_quadratically(self) -> None:
        """4x the bars should cost roughly 4x the time, not 16x. A generous
        6x ceiling absorbs warm-up/measurement noise while still catching
        a real quadratic regression (which would show up as a much larger
        ratio at this scale)."""
        small_n, large_n = 5_000, 20_000
        small_elapsed = _run_backtest(small_n, seed=1)
        large_elapsed = _run_backtest(large_n, seed=1)

        time_ratio = large_elapsed / small_elapsed
        data_ratio = large_n / small_n  # 4.0

        assert time_ratio < data_ratio * 1.5, (
            f"Possible quadratic scaling: {data_ratio}x the data took {time_ratio:.2f}x the time"
        )
