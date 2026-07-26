"""Tests for performance analytics functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quant_platform.analytics.performance import (
    average_trade_pnl,
    cagr,
    calmar_ratio,
    compute_performance_report,
    conditional_value_at_risk,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    win_rate,
)
from quant_platform.core.types import EquityPoint, OrderSide, Trade

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _trade(net_pnl: float) -> Trade:
    # gross_pnl == net_pnl (zero cost) keeps trade fixtures simple for these tests.
    return Trade(
        entry_time=T0,
        exit_time=T0 + timedelta(minutes=15),
        side=OrderSide.BUY,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        quantity=1.0,
        gross_pnl=net_pnl,
        total_cost=0.0,
        exit_reason="TP" if net_pnl > 0 else "SL",
    )


class TestMaxDrawdown:
    def test_known_series(self) -> None:
        equity = pd.Series([100.0, 110.0, 105.0, 90.0, 95.0, 120.0])
        dd_pct, peak_pos, trough_pos = max_drawdown(equity)
        assert dd_pct == pytest.approx((110.0 - 90.0) / 110.0 * 100.0)
        assert peak_pos == 1
        assert trough_pos == 3

    def test_flat_series_has_zero_drawdown(self) -> None:
        equity = pd.Series([100.0, 100.0, 100.0])
        dd_pct, peak_pos, trough_pos = max_drawdown(equity)
        assert dd_pct == 0.0
        assert (peak_pos, trough_pos) == (0, 0)

    def test_monotonically_increasing_has_zero_drawdown(self) -> None:
        equity = pd.Series([100.0, 110.0, 120.0])
        dd_pct, _, _ = max_drawdown(equity)
        assert dd_pct == 0.0

    def test_single_point_returns_zero(self) -> None:
        assert max_drawdown(pd.Series([100.0])) == (0.0, 0, 0)

    def test_empty_returns_zero(self) -> None:
        assert max_drawdown(pd.Series([], dtype=float)) == (0.0, 0, 0)


class TestCagr:
    def test_doubling_over_one_year(self) -> None:
        equity = pd.Series([100.0] + [150.0] * 251 + [200.0])  # 253 points = 252 periods
        result = cagr(equity, periods_per_year=252)
        assert result == pytest.approx(100.0)

    def test_total_loss_returns_negative_100(self) -> None:
        equity = pd.Series([100.0, 50.0, 0.0])
        assert cagr(equity, periods_per_year=252) == -100.0

    def test_too_few_points_returns_zero(self) -> None:
        assert cagr(pd.Series([100.0]), periods_per_year=252) == 0.0

    def test_non_positive_starting_equity_returns_zero(self) -> None:
        assert cagr(pd.Series([0.0, 100.0]), periods_per_year=252) == 0.0


class TestSharpeRatio:
    def test_matches_direct_computation(self) -> None:
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
        assert sharpe_ratio(returns, periods_per_year=252) == pytest.approx(expected)

    def test_zero_variance_returns_none(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01])
        assert sharpe_ratio(returns, periods_per_year=252) is None

    def test_too_few_points_returns_none(self) -> None:
        assert sharpe_ratio(pd.Series([0.01]), periods_per_year=252) is None

    def test_higher_risk_free_rate_lowers_sharpe(self) -> None:
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        low_rf = sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        high_rf = sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.5)
        assert high_rf < low_rf


class TestSortinoRatio:
    def test_matches_direct_computation(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        downside = np.minimum(returns.to_numpy(), 0.0)
        downside_dev = np.sqrt(np.mean(downside**2))
        expected = (returns.mean() / downside_dev) * np.sqrt(252)
        assert sortino_ratio(returns, periods_per_year=252) == pytest.approx(expected)

    def test_no_downside_returns_none(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.005])
        assert sortino_ratio(returns, periods_per_year=252) is None

    def test_too_few_points_returns_none(self) -> None:
        assert sortino_ratio(pd.Series([0.01]), periods_per_year=252) is None


class TestCalmarRatio:
    def test_basic_ratio(self) -> None:
        assert calmar_ratio(cagr_pct=20.0, max_drawdown_pct=10.0) == pytest.approx(2.0)

    def test_zero_drawdown_returns_none(self) -> None:
        assert calmar_ratio(cagr_pct=20.0, max_drawdown_pct=0.0) is None

    def test_negative_drawdown_returns_none(self) -> None:
        assert calmar_ratio(cagr_pct=20.0, max_drawdown_pct=-5.0) is None


class TestProfitFactor:
    def test_mixed_trades(self) -> None:
        trades = [_trade(100.0), _trade(-40.0), _trade(50.0), _trade(-30.0)]
        assert profit_factor(trades) == pytest.approx(150.0 / 70.0)

    def test_all_winners_is_infinite(self) -> None:
        assert profit_factor([_trade(10.0), _trade(20.0)]) == float("inf")

    def test_all_losers_is_zero(self) -> None:
        assert profit_factor([_trade(-10.0), _trade(-20.0)]) == 0.0

    def test_no_trades_is_zero(self) -> None:
        assert profit_factor([]) == 0.0


class TestWinRate:
    def test_mixed_trades(self) -> None:
        trades = [_trade(10.0), _trade(-5.0), _trade(20.0), _trade(-1.0)]
        assert win_rate(trades) == pytest.approx(50.0)

    def test_no_trades_is_zero(self) -> None:
        assert win_rate([]) == 0.0


class TestAverageTradePnl:
    def test_mixed_trades(self) -> None:
        trades = [_trade(10.0), _trade(-5.0), _trade(20.0)]
        assert average_trade_pnl(trades) == pytest.approx((10.0 - 5.0 + 20.0) / 3.0)

    def test_no_trades_is_zero(self) -> None:
        assert average_trade_pnl([]) == 0.0


class TestValueAtRisk:
    def test_matches_numpy_quantile(self) -> None:
        returns = pd.Series([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        expected = -np.quantile(returns.to_numpy(), 0.05)
        assert value_at_risk(returns, confidence=0.95) == pytest.approx(expected)

    def test_empty_returns_zero(self) -> None:
        assert value_at_risk(pd.Series([], dtype=float)) == 0.0

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_confidence(self, confidence: float) -> None:
        with pytest.raises(ValueError):
            value_at_risk(pd.Series([0.01, 0.02]), confidence=confidence)


class TestConditionalValueAtRisk:
    def test_is_at_least_as_large_as_var(self) -> None:
        returns = pd.Series(np.linspace(-0.10, 0.10, 50))
        var = value_at_risk(returns, confidence=0.95)
        cvar = conditional_value_at_risk(returns, confidence=0.95)
        assert cvar >= var

    def test_empty_returns_zero(self) -> None:
        assert conditional_value_at_risk(pd.Series([], dtype=float)) == 0.0


class TestComputePerformanceReport:
    def test_aggregates_all_metrics_consistently(self) -> None:
        equity_curve = [
            EquityPoint(timestamp=T0 + timedelta(days=i), cash=v, equity=v, drawdown_pct=0.0)
            for i, v in enumerate([100.0, 105.0, 102.0, 108.0, 112.0])
        ]
        trades = [_trade(10.0), _trade(-4.0), _trade(6.0)]

        report = compute_performance_report(equity_curve, trades, periods_per_year=252)

        assert report.total_trades == 3
        assert report.win_rate_pct == pytest.approx(win_rate(trades))
        assert report.profit_factor == pytest.approx(profit_factor(trades))
        assert report.total_net_pnl == pytest.approx(12.0)

    def test_handles_empty_trades_and_short_equity_curve_gracefully(self) -> None:
        equity_curve = [EquityPoint(timestamp=T0, cash=100.0, equity=100.0, drawdown_pct=0.0)]
        report = compute_performance_report(equity_curve, [], periods_per_year=252)

        assert report.total_trades == 0
        assert report.win_rate_pct == 0.0
        assert report.profit_factor == 0.0
        assert report.sharpe_ratio is None
        assert report.sortino_ratio is None
        assert report.calmar_ratio is None
