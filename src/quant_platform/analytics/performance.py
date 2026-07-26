"""Performance analytics: pure functions over an equity curve and/or trade
log, plus a `PerformanceReport` aggregator.

Every function here is a pure, side-effect-free transformation of its
inputs and degrades gracefully (returning `None` or a documented sentinel)
rather than raising on statistically undefined inputs (e.g. Sharpe ratio
with zero variance, Calmar ratio with zero drawdown) -- a backtest with
too few trades to compute a ratio is a legitimate, common outcome, not a
bug that should crash reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.types import EquityPoint, Trade


def max_drawdown(equity: pd.Series) -> tuple[float, int, int]:
    """Returns (max_drawdown_pct, peak_position, trough_position) as
    positional (iloc-style) indices into `equity`. (0.0, 0, 0) if `equity`
    has fewer than 2 points or never has a positive running peak."""
    if len(equity) < 2:
        return 0.0, 0, 0

    values = equity.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_max > 0, (running_max - values) / running_max, 0.0)

    trough_pos = int(np.argmax(drawdown))
    if drawdown[trough_pos] <= 0:
        return 0.0, 0, 0
    peak_pos = int(np.argmax(values[: trough_pos + 1]))
    return float(drawdown[trough_pos] * 100.0), peak_pos, trough_pos


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    """Compound annual growth rate, in percent. 0.0 if fewer than 2 points,
    the starting equity is non-positive, or the elapsed time is zero.
    -100.0 (total loss) if ending equity is non-positive."""
    if len(equity) < 2 or equity.iloc[0] <= 0 or periods_per_year <= 0:
        return 0.0

    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    total_return = float(equity.iloc[-1]) / float(equity.iloc[0])
    if total_return <= 0:
        return -100.0

    return float((total_return ** (1.0 / years) - 1.0) * 100.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0) -> float | None:
    """Annualized Sharpe ratio from a period-return series. None if fewer
    than 2 observations or the return series has zero variance."""
    if len(returns) < 2:
        return None
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=1)
    if std == 0 or pd.isna(std):
        return None
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0) -> float | None:
    """Annualized Sortino ratio (downside-deviation-adjusted). None if
    fewer than 2 observations or there is no downside variance to measure."""
    if len(returns) < 2:
        return None
    excess = returns - (risk_free_rate / periods_per_year)
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2))
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return None
    return float((excess.mean() / downside_deviation) * np.sqrt(periods_per_year))


def calmar_ratio(cagr_pct: float, max_drawdown_pct: float) -> float | None:
    """CAGR / max drawdown. None if there was no drawdown to normalize by."""
    if max_drawdown_pct <= 0:
        return None
    return cagr_pct / max_drawdown_pct


def profit_factor(trades: Sequence[Trade]) -> float:
    """Gross profit / gross loss. `inf` if there were profits and no
    losses; 0.0 if there were neither (including no trades at all)."""
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = -sum(t.net_pnl for t in trades if t.net_pnl <= 0)
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return 0.0


def win_rate(trades: Sequence[Trade]) -> float:
    """Percentage of trades with positive net P&L. 0.0 if there are no trades."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.is_win)
    return wins / len(trades) * 100.0


def average_trade_pnl(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(t.net_pnl for t in trades) / len(trades)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at `confidence`, reported as a positive loss
    magnitude (i.e. "expect to lose at least this much, (1-confidence)
    of the time"). 0.0 if `returns` is empty."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(returns) == 0:
        return 0.0
    quantile_level = 1.0 - confidence
    return float(-np.quantile(returns.to_numpy(dtype=float), quantile_level))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall beyond the VaR threshold at `confidence`, as a
    positive loss magnitude. Falls back to the VaR value itself if the
    tail beyond the threshold is empty (can happen with small/discrete
    samples). 0.0 if `returns` is empty."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(returns) == 0:
        return 0.0
    quantile_level = 1.0 - confidence
    values = returns.to_numpy(dtype=float)
    threshold = np.quantile(values, quantile_level)
    tail = values[values <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    average_trade_pnl: float
    total_net_pnl: float
    max_drawdown_pct: float
    cagr_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    value_at_risk_95: float
    conditional_value_at_risk_95: float


def compute_performance_report(
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[Trade],
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Aggregate every metric above into a single report. Safe to call
    with an empty trade list or a degenerate (0-1 point) equity curve --
    every field degrades to its documented "no data" value rather than
    raising."""
    equity_series = pd.Series([point.equity for point in equity_curve])
    returns = equity_series.pct_change().dropna() if len(equity_series) > 1 else pd.Series(dtype=float)

    dd_pct, _, _ = max_drawdown(equity_series)
    cagr_pct = cagr(equity_series, periods_per_year)

    return PerformanceReport(
        total_trades=len(trades),
        win_rate_pct=win_rate(trades),
        profit_factor=profit_factor(trades),
        average_trade_pnl=average_trade_pnl(trades),
        total_net_pnl=sum(t.net_pnl for t in trades),
        max_drawdown_pct=dd_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=sharpe_ratio(returns, periods_per_year, risk_free_rate),
        sortino_ratio=sortino_ratio(returns, periods_per_year, risk_free_rate),
        calmar_ratio=calmar_ratio(cagr_pct, dd_pct),
        value_at_risk_95=value_at_risk(returns, confidence=0.95) if len(returns) else 0.0,
        conditional_value_at_risk_95=(
            conditional_value_at_risk(returns, confidence=0.95) if len(returns) else 0.0
        ),
    )
