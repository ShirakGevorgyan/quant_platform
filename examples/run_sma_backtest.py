"""End-to-end example: configure, run, and report on a backtest.

Demonstrates the full milestone-1 pipeline using the same
`BacktestConfig`/`CostModelConfig`/`RiskConfig` schemas a production
caller would use, deterministic synthetic data (no external data source
or network access required to run this), and the analytics module for
reporting. Run with:

    python examples/run_sma_backtest.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from quant_platform.analytics.performance import compute_performance_report
from quant_platform.config.schemas import BacktestConfig, CostModelConfig, RiskConfig
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.data.validation import validate_ohlcv
from quant_platform.engine.backtest_engine import BacktestEngine
from quant_platform.strategy.examples.sma_crossover import SmaCrossoverStrategy

UTC = timezone.utc
BARS_PER_YEAR_M15 = 96 * 252  # 96 fifteen-minute bars/trading day * 252 trading days/year


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = BacktestConfig(
        symbol="DEMO",
        base_timeframe=Timeframe.M15,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 7, 1, tzinfo=UTC),
        initial_capital=10_000.0,
        point_value=1.0,
        cost=CostModelConfig(spread_points=2.0, slippage_points=1.0, point_value=1.0, commission_per_unit=0.01),
        risk=RiskConfig(sizer_type="kelly", win_rate=0.5, win_loss_ratio=1.5, kelly_fraction=0.3, max_position_fraction=0.5),
    )

    periods = int((config.end - config.start).total_seconds() / config.base_timeframe.duration.total_seconds())
    data = generate_ohlcv(
        SyntheticDataConfig(
            start=config.start, periods=periods, timeframe=config.base_timeframe,
            annualized_volatility=0.5, seed=42,
        )
    )

    quality_report = validate_ohlcv(data, symbol=config.symbol, timeframe=config.base_timeframe)
    print(quality_report.summary())
    quality_report.raise_if_invalid()

    strategy = SmaCrossoverStrategy(timeframe=config.base_timeframe, fast_period=10, slow_period=30)

    engine = BacktestEngine(
        data={config.base_timeframe: data},
        base_timeframe=config.base_timeframe,
        strategy=strategy,
        cost_model=config.cost.build(),
        position_sizer=config.risk.build(),
        initial_capital=config.initial_capital,
        point_value=config.point_value,
        symbol=config.symbol,
    )
    result = engine.run()

    report = compute_performance_report(result.equity_curve, result.trades, periods_per_year=BARS_PER_YEAR_M15)

    print()
    print("=" * 60)
    print(f"BACKTEST REPORT: {config.symbol} ({config.base_timeframe.value})")
    print("=" * 60)
    print(f"Period:              {config.start.date()} -> {config.end.date()}")
    print(f"Starting capital:    ${result.initial_capital:,.2f}")
    print(f"Ending equity:       ${result.final_equity:,.2f}")
    print(f"Net return:          {result.net_return_pct:+.2f}%")
    print(f"Total trades:        {report.total_trades}")
    print(f"Win rate:            {report.win_rate_pct:.1f}%")
    print(f"Profit factor:       {report.profit_factor:.2f}")
    print(f"Average trade P&L:   ${report.average_trade_pnl:,.2f}")
    print(f"Max drawdown:        {report.max_drawdown_pct:.2f}%")
    print(f"CAGR:                {report.cagr_pct:.2f}%")
    print(f"Sharpe ratio:        {report.sharpe_ratio:.2f}" if report.sharpe_ratio is not None else "Sharpe ratio:        N/A")
    print(f"Sortino ratio:       {report.sortino_ratio:.2f}" if report.sortino_ratio is not None else "Sortino ratio:       N/A")
    print(f"Calmar ratio:        {report.calmar_ratio:.2f}" if report.calmar_ratio is not None else "Calmar ratio:        N/A")
    print(f"VaR (95%):           {report.value_at_risk_95:.4f}")
    print(f"CVaR (95%):          {report.conditional_value_at_risk_95:.4f}")
    print()

    if result.trades:
        print("First 5 trades:")
        for trade in result.trades[:5]:
            print(
                f"  {trade.entry_time} | {trade.side.value:4s} | entry={trade.entry_price:.4f} "
                f"exit={trade.exit_price:.4f} | net_pnl=${trade.net_pnl:+.2f} ({trade.exit_reason})"
            )


if __name__ == "__main__":
    main()
