"""Integration test: the full stack (synthetic data -> BacktestEngine with
the real SmaCrossoverStrategy -> Portfolio accounting -> performance
analytics) exercised end to end on realistic, multi-timeframe data.

Unlike the unit tests, this deliberately does NOT hand-verify exact trade
prices -- its job is to prove the pieces integrate correctly (multiple
timeframes wired together, realistic volatility producing real trades,
the trade log and equity curve reconciling with each other, and the
analytics module consuming engine output without error), not to re-verify
arithmetic already covered at the unit level.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.analytics.performance import compute_performance_report
from quant_platform.core.types import Timeframe
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.engine.backtest_engine import BacktestEngine
from quant_platform.risk.position_sizing import KellyCriterionSizer
from quant_platform.strategy.examples.sma_crossover import SmaCrossoverStrategy

UTC = timezone.utc


def _build_engine(seed: int) -> BacktestEngine:
    m15 = generate_ohlcv(
        SyntheticDataConfig(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            periods=3_000,
            timeframe=Timeframe.M15,
            annualized_volatility=0.8,  # deliberately high, so crossovers occur reliably
            seed=seed,
        )
    )
    # An extra, unused-by-the-strategy higher timeframe, tracked purely to
    # prove the engine wires up multiple timeframes correctly in a
    # realistic run (not just the hand-crafted unit-level wiring test).
    h1 = generate_ohlcv(
        SyntheticDataConfig(
            start=datetime(2024, 1, 1, tzinfo=UTC), periods=800, timeframe=Timeframe.H1,
            annualized_volatility=0.8, seed=seed + 1,
        )
    )

    strategy = SmaCrossoverStrategy(timeframe=Timeframe.M15, fast_period=10, slow_period=30)
    cost_model = FixedSpreadCostModel(
        spread_points=2.0, slippage_points=1.0, point_value=1.0, commission_per_unit=0.01
    )
    # SmaCrossoverStrategy exits via the opposite crossover rather than a
    # stop-loss/volatility target, so it never sets Signal.stop_loss or
    # Signal.current_volatility -- Kelly sizing needs neither.
    sizer = KellyCriterionSizer(win_rate=0.5, win_loss_ratio=1.5, kelly_fraction=0.3, max_position_fraction=0.5)

    return BacktestEngine(
        data={Timeframe.M15: m15, Timeframe.H1: h1},
        base_timeframe=Timeframe.M15,
        strategy=strategy,
        cost_model=cost_model,
        position_sizer=sizer,
        initial_capital=10_000.0,
        point_value=1.0,
        symbol="SYNTH",
    )


class TestFullBacktestIntegration:
    def test_runs_to_completion_and_produces_trades(self) -> None:
        engine = _build_engine(seed=1)
        result = engine.run()

        assert len(result.trades) > 0, "expected at least one trade on a volatile 3000-bar synthetic series"
        assert result.final_equity > 0
        # Exactly one equity point per bar, always -- including when a still-
        # open position is force-closed on the final bar (regression guard
        # for a fixed bug where that produced a spurious extra point).
        assert len(result.equity_curve) == 3_000

    def test_trade_log_reconciles_exactly_with_final_equity(self) -> None:
        engine = _build_engine(seed=2)
        result = engine.run()

        reconciled_equity = result.initial_capital + sum(t.net_pnl for t in result.trades)
        assert result.final_equity == pytest.approx(reconciled_equity, abs=1e-6)

    def test_every_trade_has_sane_price_and_time_ordering(self) -> None:
        engine = _build_engine(seed=3)
        result = engine.run()

        for trade in result.trades:
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.quantity > 0
            assert trade.exit_time >= trade.entry_time
            assert trade.exit_reason in {"SIGNAL", "REVERSAL", "END_OF_DATA", "SL", "TP", "TIME_STOP"}

    def test_only_the_last_trade_may_be_end_of_data(self) -> None:
        engine = _build_engine(seed=4)
        result = engine.run()
        end_of_data_positions = [i for i, t in enumerate(result.trades) if t.exit_reason == "END_OF_DATA"]
        assert end_of_data_positions in ([], [len(result.trades) - 1])

    def test_equity_curve_is_chronologically_ordered(self) -> None:
        engine = _build_engine(seed=5)
        result = engine.run()
        timestamps = [point.timestamp for point in result.equity_curve]
        assert timestamps == sorted(timestamps)

    def test_analytics_module_consumes_engine_output_without_error(self) -> None:
        engine = _build_engine(seed=6)
        result = engine.run()

        report = compute_performance_report(result.equity_curve, result.trades, periods_per_year=252 * 96)
        # 96 M15 bars/day * 252 trading days/year, for a plausible annualization factor.

        assert report.total_trades == len(result.trades)
        assert 0.0 <= report.win_rate_pct <= 100.0
        assert report.max_drawdown_pct >= 0.0
        assert report.profit_factor >= 0.0

    def test_deterministic_given_the_same_seed(self) -> None:
        result1 = _build_engine(seed=7).run()
        result2 = _build_engine(seed=7).run()

        assert len(result1.trades) == len(result2.trades)
        assert result1.final_equity == pytest.approx(result2.final_equity)
        for t1, t2 in zip(result1.trades, result2.trades, strict=True):
            assert t1.entry_price == pytest.approx(t2.entry_price)
            assert t1.exit_price == pytest.approx(t2.exit_price)
