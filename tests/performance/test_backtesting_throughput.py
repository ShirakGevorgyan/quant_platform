"""Performance benchmarks for the Milestone 5 leakage-safe financial
evaluation / backtesting framework. Same philosophy as
`test_calibration_throughput.py`: conservative floors (roughly 10x-100x
below measured numbers on reference hardware) to catch a severe
accidental regression without being flaky on a slower CI runner -- these
are NOT production throughput guarantees, and no safety check (market-
bar fold-boundary enforcement, cost-breakdown self-consistency, financial-
metrics recomputation) is ever skipped to make a number look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%):
  - `simulate_outer_fold_trades` (2,000 bars, ~400 accepted signals,
    fixed-horizon exit), 50 iterations: ~23ms/iter median, ~43 iter/sec.
  - `compute_financial_metrics` (post-simulation, ~200 closed trades),
    100 iterations: ~0.6ms/iter median, ~1,600 iter/sec.
  - `BacktestRunner.run` (full 2-outer-fold pipeline, distinct backtest
    every iteration, `constant_test_model`), 5 iterations: ~252ms/iter
    median, ~4/sec.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.execution import simulate_outer_fold_trades
from quant_platform.backtesting.metrics import compute_financial_metrics
from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    Decision,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
    VerifiedPredictionSet,
)
from quant_platform.backtesting.signals import generate_signals
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.backtesting.timeline import bar_return_timeline_to_equity_curve, build_bar_return_timeline
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.types import Timeframe

pytestmark = pytest.mark.performance


def _timed_iterations(fn, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> float:
    median = statistics.median(timings)
    rate = 1.0 / median if median > 0 else float("inf")
    print(f"\n{label}: n={len(timings)} median={median * 1000:.3f}ms p95={sorted(timings)[int(len(timings) * 0.95)] * 1000:.3f}ms rate={rate:,.0f}/sec")
    return median


def _spec() -> BacktestSpec:
    return BacktestSpec(
        schema_version=1, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
        dataset_content_id="c" * 64, split_plan_fingerprint="d" * 64, instrument_identity="XAUUSD", market_timezone="UTC",
        bar_interval=Timeframe.H1, decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT,
        entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=3, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        overlap_policy=OverlapPolicyKind.IGNORE, price_basis=PriceBasisKind.CLOSE,
        spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=5.0),
        commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
        slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
        return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.NON_COMPOUNDED,
        initial_notional=10_000.0, determinism_policy=DeterminismPolicy.STRICT,
    )


def _bars(n: int) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    prices = 2000.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    return pd.DataFrame({"open_time": timestamps, "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices + 0.1})


def _predictions(n: int, *, seed: int) -> VerifiedPredictionSet:
    rng = np.random.default_rng(seed)
    calibrated = rng.uniform(0.0, 1.0, size=n)
    timestamps = tuple(ts.isoformat() for ts in pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    return VerifiedPredictionSet(
        schema_version=1, outer_fold_index=0, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
        base_model_definition_identity="m:1", sample_positions=tuple(range(n)), timestamps=timestamps,
        raw_probabilities=tuple(float(v) for v in calibrated), calibrated_probabilities=tuple(float(v) for v in calibrated), threshold=0.5,
        decisions=tuple(Decision.POSITIVE.value if v >= 0.5 else Decision.NEGATIVE.value for v in calibrated),
        abstention_reason_codes=("none",) * n, confidence_scores=tuple(abs(v - 0.5) * 2 for v in calibrated),
        confidence_categories=("medium",) * n, uncertainty_scores=tuple(1.0 - abs(v - 0.5) * 2 for v in calibrated),
    )


class TestExecutionSimulationThroughput:
    def test_simulate_outer_fold_trades(self) -> None:
        """2,000 bars, ~400 accepted signals (half the 800 predicted
        positions, fixed-horizon exit)."""
        n_bars = 2000
        bars = _bars(n_bars)
        predictions = _predictions(800, seed=1)
        signals = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        spec = _spec()

        median = _report("simulate_outer_fold_trades (2000 bars, ~400 signals)", _timed_iterations(lambda: simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=n_bars - 1), 50))
        assert median < 1.0, "simulating ~400 signals over 2000 bars should not take >1s (a generous floor)"


class TestBarReturnTimelineThroughput:
    def test_build_bar_return_timeline(self) -> None:
        """Milestone 5.1: the bar-level timeline walks every bar in the
        fold (2,000 here), not just trade-exit events -- benchmarked
        separately since it is now on the critical path of every fold."""
        n_bars = 2000
        bars = _bars(n_bars)
        predictions = _predictions(800, seed=2)
        signals = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        spec = _spec()
        trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=n_bars - 1)
        assert len(trade_set.closed_trades) > 50, "expected a meaningful number of closed trades to benchmark the timeline over"

        median = _report(
            f"build_bar_return_timeline ({n_bars} bars, {len(trade_set.closed_trades)} closed trades)",
            _timed_iterations(lambda: build_bar_return_timeline(trades=trade_set.trades, bars=bars, fold_start_position=0, fold_end_position=n_bars - 1, outer_fold_index=0, return_calculation_policy=spec.return_calculation_policy, exposure_cap=spec.exposure_cap, compounded=False), 20),
        )
        assert median < 2.0, "walking 2000 bars to build a mark-to-market timeline should not take >2s (a generous floor)"


class TestFinancialMetricsThroughput:
    def test_compute_financial_metrics(self) -> None:
        n_bars = 2000
        bars = _bars(n_bars)
        predictions = _predictions(800, seed=2)
        signals = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        spec = _spec()
        trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=n_bars - 1)
        closed = trade_set.closed_trades
        assert len(closed) > 50, "expected a meaningful number of closed trades to benchmark metrics over"

        bar_timeline = build_bar_return_timeline(trades=trade_set.trades, bars=bars, fold_start_position=0, fold_end_position=n_bars - 1, outer_fold_index=0, return_calculation_policy=spec.return_calculation_policy, exposure_cap=spec.exposure_cap, compounded=False)
        equity_curve = bar_return_timeline_to_equity_curve(bar_timeline)
        gross_drawdown = compute_drawdown_report(equity_curve, equity_basis="gross")
        net_drawdown = compute_drawdown_report(equity_curve, equity_basis="net")

        median = _report(
            f"compute_financial_metrics ({len(closed)} closed trades, {n_bars} bar-level observations)",
            _timed_iterations(lambda: compute_financial_metrics(trades=trade_set, equity_curve=equity_curve, gross_drawdown=gross_drawdown, net_drawdown=net_drawdown, signals=signals, bar_timeline=bar_timeline, bar_interval=Timeframe.H1, annual_risk_free_rate=0.0, initial_notional=10_000.0, exposure_cap=spec.exposure_cap), 100),
        )
        assert median < 0.5, "computing ~30 financial metrics over a bar-level timeline should not take >500ms (a generous floor)"


class TestBacktestRunnerThroughput:
    def test_run_distinct_backtests(self, tmp_path: Path) -> None:
        """A full 2-outer-fold `BacktestRunner.run()` against a fresh,
        DISTINCT backtest every iteration (never an idempotent no-op),
        reusing ONE already-completed calibration+execution (built once,
        outside the timed loop) with a fresh `BacktestSpec` (distinct
        `seed`) per iteration."""
        from tests.integration.test_backtesting_engine import _build_ready_setup

        spec_template, runner, _ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        counter = {"i": 0}

        def run() -> None:
            counter["i"] += 1
            from dataclasses import replace

            distinct_spec = replace(spec_template, seed=counter["i"])
            runner.run(distinct_spec)

        median = _report("BacktestRunner.run (distinct backtests, 2 outer folds, constant_test_model)", _timed_iterations(run, 5))
        assert median < 10.0, "a full small 2-outer-fold backtest run should not take >10s (~4x the measured ~2.6s floor)"
