"""Financial metrics (Milestone 5, Sections 17/18; corrected in Milestone
5.1 Section 2) -- every formula documented at its own computation site;
an undefined metric is OMITTED from `values` and explained in `skipped`,
reusing `ml.metrics.MetricComputationReport`'s exact "skip with reason,
never NaN" shape.

MILESTONE 5.1 SECTION 2: METRIC NAMING AUDIT
--------------------------------------------------------------------------
`bar_return_sharpe`/`annualized_volatility`/`sortino_ratio`/
`drawdown_duration_bars` are now computed from `timeline.
BarReturnTimeline` (via its `EquityCurve` projection) -- a genuinely
bar-sampled series, one point per outer-test market bar, never
compressed to trade-exit events (see `backtesting.timeline`'s module
docstring for the defect this corrects and why it mattered: annualizing
an EVENT-sampled series at BAR frequency inflated Sharpe/Sortino by
roughly `sqrt(bars_per_fold / events_per_fold)`).

Renamed/redefined for accuracy (the old name is not reused for a
different meaning anywhere in this codebase):
  - `turnover` (previously "2x closed-trade count", ambiguous) is now
    `transaction_count` (the same 2x count, honestly named) PLUS
    `total_transacted_notional` (currency units) and `turnover_notional_
    ratio` (dimensionless, transacted notional / capital base). MILESTONE
    5.2 SECTION 4 CORRECTION: `turnover_notional_ratio` was previously
    computed as `transaction_count * exposure_cap`, an identity that only
    holds if every trade's entry AND exit leg transact the exact SAME
    dollar notional -- true for the ENTRY leg under this platform's
    fixed-notional sizing, but FALSE for the EXIT leg whenever price
    moved between entry and exit (the exit leg's dollar notional is
    `entry_notional * exit_effective_price / entry_effective_price`, not
    `entry_notional` again). It is now computed directly from each closed
    trade's own persisted `entry_effective_price`/`exit_effective_price`
    (real fill prices, never merely assumed constant) -- see the
    computation site's own comment for the direction-agnostic derivation.
    `annualized_turnover` is added, reported only when the fold's own
    duration is long enough to extrapolate meaningfully (same guard as
    `annualized_return`) -- a within-fold TOTAL (`turnover_notional_
    ratio`) is always reported regardless, since it needs no
    extrapolation.
  - `exposure` (previously the mean of a PER-EXIT-EVENT indicator field)
    is now `time_in_market_fraction` (fraction of BARS with at least one
    open position) and `mean_total_absolute_exposure` (mean, across
    every bar, of `BarReturnPoint.total_absolute_exposure`) -- two
    genuinely bar-sampled, distinctly-named quantities.
  - `cost_drag` (a MEAN per-trade cost, previously risked being read as
    a total) is renamed `mean_trade_cost`; a genuine total is added
    separately as `total_cost_fraction` (sum, not mean) and
    `total_cost_notional` (the same total, in notional currency units).
    A report table must never label `mean_trade_cost` "Total cost" --
    see `docs/financial_evaluation_backtesting.md`'s explicit warning.

MILESTONE 5.2, SECTION 7: "ROUND-TRIPS PER BAR" VS. "TRANSACTION SIDES
PER BAR" ARE NOT THE SAME NUMBER
--------------------------------------------------------------------------
A prior delivery report described `628 transactions / 350 bars = 1.79`
as "round-trips per bar" -- WRONG: `transaction_count` counts SIDES
(one entry + one exit per closed trade = 2 per round trip), not round
trips themselves. The correctly-named value for that computation is
`transaction_sides_per_bar`. The GENUINE round-trips-per-bar figure is
`trades_per_bar` (`trade_count / n_points`, e.g. `314 trades / 350 bars =
0.90`) -- a materially different number, now computed and persisted
separately, never conflated. `reversal_rate` (fraction of chronologically
adjacent closed-trade pairs whose direction flips) is a third, distinct
new metric -- about the SEQUENCE of trade directions, not a per-bar rate
at all. `average_holding_period_bars` (Section 17, unchanged) was already
correctly named and needed no correction.

`bar_return_sharpe` vs. `trade_return_sharpe` remain distinct exactly as
before (Section 18): the former samples the bar-level timeline, the
latter the per-TRADE net-return series. Both persist their own sample
count and share the same annualization factor/risk-free convention. No
third "exit_event_return_sharpe" is added -- it would be redundant with
`trade_return_sharpe` (both are event/trade-sampled; the old
`bar_return_sharpe` was the ONLY one that was mislabeled)."""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass

from quant_platform.backtesting.drawdown import DrawdownReport
from quant_platform.backtesting.returns import EquityCurve
from quant_platform.backtesting.signals import SignalSet
from quant_platform.backtesting.timeline import BarReturnTimeline
from quant_platform.backtesting.trades import TradeSet
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.core.types import Timeframe
from quant_platform.ml.metrics import MetricComputationReport

_CALENDAR_MINUTES_PER_YEAR = 365.25 * 24 * 60


def bars_per_year(bar_interval: Timeframe) -> float:
    return _CALENDAR_MINUTES_PER_YEAR / bar_interval.minutes


@dataclass(frozen=True, slots=True)
class SharpeAssumptions:
    """Section 18: persisted alongside every Sharpe/Sortino value."""

    annualization_factor: float
    annual_risk_free_rate: float
    per_period_risk_free_rate: float
    sample_count: int
    stdev_convention: str = "population"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "annualization_factor": self.annualization_factor, "annual_risk_free_rate": self.annual_risk_free_rate,
            "per_period_risk_free_rate": self.per_period_risk_free_rate, "sample_count": self.sample_count,
            "stdev_convention": self.stdev_convention,
        }


def population_stdev(values: list[float]) -> float | None:
    """Public (Milestone 5.1, Section 3): reused by `stitching.py` to
    compute Sharpe/Sortino over the pooled, stitched bar-return series --
    the same population-stdev convention as every per-fold Sharpe/Sortino
    computed here."""
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def sharpe_like(
    returns: list[float], *, annual_risk_free_rate: float, periods_per_year: float, downside_only: bool,
) -> tuple[float | None, str | None, SharpeAssumptions]:
    """Public (Milestone 5.1, Section 3): the one Sharpe/Sortino-shaped
    ratio computation, reused unchanged by `stitching.compute_stitched_
    financial_metrics` so stitched and per-fold ratios share an identical
    formula/annualization/undefined-handling convention."""
    per_period_rf = annual_risk_free_rate / periods_per_year
    assumptions = SharpeAssumptions(
        annualization_factor=math.sqrt(periods_per_year), annual_risk_free_rate=annual_risk_free_rate,
        per_period_risk_free_rate=per_period_rf, sample_count=len(returns),
    )
    if len(returns) < 2:
        return None, "fewer than 2 return observations -- standard deviation is undefined", assumptions
    excess = [r - per_period_rf for r in returns]
    if downside_only:
        downside = [min(e, 0.0) for e in excess]
        stdev = population_stdev(downside)
        if stdev is None or stdev == 0.0:
            return None, "zero (or undefined) downside deviation -- Sortino ratio is undefined", assumptions
    else:
        stdev = population_stdev(excess)
        if stdev is None or stdev == 0.0:
            return None, "zero (or undefined) return variance -- Sharpe ratio is undefined", assumptions
    mean_excess = statistics.fmean(excess)
    ratio = (mean_excess / stdev) * math.sqrt(periods_per_year)
    return ratio, None, assumptions


def compute_financial_metrics(
    *, trades: TradeSet, equity_curve: EquityCurve, gross_drawdown: DrawdownReport, net_drawdown: DrawdownReport,
    signals: SignalSet, bar_timeline: BarReturnTimeline, bar_interval: Timeframe, annual_risk_free_rate: float,
    initial_notional: float, exposure_cap: float,
) -> MetricComputationReport:
    """Section 17: every metric in one pass, `skip`-not-`NaN` for
    anything undefined for this particular fold's data. `equity_curve`/
    `gross_drawdown`/`net_drawdown` must be derived from `bar_timeline`
    (via `timeline.bar_return_timeline_to_equity_curve` and
    `drawdown.compute_drawdown_report`) -- passed separately only because
    `compute_drawdown_report` is a distinct, independently-reusable
    function, not because they carry independent information."""
    values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    periods_per_year = bars_per_year(bar_interval)

    closed = [t for t in trades.trades if t.status.value in ("closed", "incomplete_force_closed")]
    net_returns = [t.net_return for t in closed]
    gross_returns_list = [t.gross_return for t in closed]

    values["trade_count"] = float(len(closed))
    values["long_trade_count"] = float(sum(1 for t in closed if t.direction.value == "long"))
    values["short_trade_count"] = float(sum(1 for t in closed if t.direction.value == "short"))

    accepted = sum(1 for s in signals.signals if s.accepted)
    abstained = sum(1 for s in signals.signals if s.reason_code.value == "abstained_by_calibration_policy")
    values["accepted_signal_rate"] = accepted / len(signals.signals)
    values["abstention_rate"] = abstained / len(signals.signals)

    last_point = equity_curve.points[-1]
    values["total_gross_return"] = last_point.cumulative_gross_equity - 1.0
    values["total_net_return"] = last_point.cumulative_net_equity - 1.0

    n_points = len(equity_curve.points)
    # Milestone 5.2, Section 7: "round-trips per bar" -- the CORRECTLY
    # named, bar-normalized round-trip rate (a prior delivery report
    # mislabeled `transaction_sides_per_bar`'s VALUE, computed below
    # inside the `if closed:` block, with this name; the two are
    # genuinely different quantities -- see that computation site's own
    # comment for the exact numeric example this milestone corrects).
    # Well-defined (0.0) even with zero closed trades, so reported
    # unconditionally, alongside `trade_count` itself.
    if n_points > 0:
        values["trades_per_bar"] = values["trade_count"] / n_points
    else:
        skipped["trades_per_bar"] = "empty bar timeline"

    duration_years = n_points / periods_per_year if periods_per_year > 0 else 0.0
    if duration_years > 0 and last_point.cumulative_net_equity > 0:
        if duration_years < 1e-9:
            values["annualized_return"] = values["total_net_return"]
        else:
            try:
                # Log-space compounding (`exp(log1p(r) / years) - 1`, not
                # `(1+r) ** (1/years)`) -- mathematically identical but
                # avoids raising a possibly-large base to a possibly-huge
                # power directly; still guarded below because a fold
                # spanning a tiny fraction of a year extrapolated to a
                # full year can EXCEED double precision's ~1.8e308 range
                # regardless of formulation -- that overflow means "not
                # meaningfully annualizable from this little data", not a
                # bug to paper over with inf/NaN.
                values["annualized_return"] = math.exp(math.log1p(values["total_net_return"]) / duration_years) - 1.0
            except OverflowError:
                skipped["annualized_return"] = (
                    "extrapolating this fold's return to a full year overflowed double precision -- the fold "
                    "duration is too short relative to a year for annualization to be numerically meaningful"
                )
    else:
        skipped["annualized_return"] = "insufficient equity-curve duration to annualize"

    # Section 1/2: genuinely bar-sampled -- one observation per outer-test
    # market bar (via `equity_curve`, itself a projection of `bar_timeline`),
    # never compressed to trade-exit events.
    period_net_returns = [p.period_net_return for p in equity_curve.points]
    bar_stdev = population_stdev(period_net_returns)
    if bar_stdev is None:
        skipped["volatility"] = "fewer than 2 period returns -- volatility is undefined"
        skipped["annualized_volatility"] = "fewer than 2 period returns -- volatility is undefined"
    else:
        values["volatility"] = bar_stdev
        values["annualized_volatility"] = bar_stdev * math.sqrt(periods_per_year)

    bar_sharpe, bar_sharpe_reason, bar_assumptions = sharpe_like(period_net_returns, annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year, downside_only=False)
    if bar_sharpe is None:
        skipped["bar_return_sharpe"] = bar_sharpe_reason or "undefined"
    else:
        values["bar_return_sharpe"] = bar_sharpe

    trade_sharpe, trade_sharpe_reason, trade_assumptions = sharpe_like(net_returns, annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year, downside_only=False)
    if trade_sharpe is None:
        skipped["trade_return_sharpe"] = trade_sharpe_reason or "undefined"
    else:
        values["trade_return_sharpe"] = trade_sharpe

    # Sortino is computed on the BAR-level series (Section 1's explicit
    # "Recompute ... Sortino" instruction) -- downside deviation of a
    # trade-exit-only series would suffer the identical sampling-mismatch
    # defect `bar_return_sharpe` had.
    sortino, sortino_reason, _sortino_assumptions = sharpe_like(period_net_returns, annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year, downside_only=True)
    if sortino is None:
        skipped["sortino_ratio"] = sortino_reason or "undefined"
    else:
        values["sortino_ratio"] = sortino

    # Section 18: Sharpe/Sortino assumptions are PERSISTED, not just used
    # internally -- annualization factor, risk-free assumption, return
    # frequency, sample count. `bar_return_sharpe`/`sortino_ratio` and
    # `trade_return_sharpe` share the same annualization factor and
    # risk-free rate (both derived from `bar_interval`/`annual_risk_free_
    # rate` alone) but differ in sample count, so both sample counts are
    # recorded distinctly.
    values["sharpe_annualization_factor"] = bar_assumptions.annualization_factor
    values["sharpe_annual_risk_free_rate"] = bar_assumptions.annual_risk_free_rate
    values["sharpe_per_period_risk_free_rate"] = bar_assumptions.per_period_risk_free_rate
    values["bar_return_sharpe_sample_count"] = float(bar_assumptions.sample_count)
    values["trade_return_sharpe_sample_count"] = float(trade_assumptions.sample_count)

    values["maximum_drawdown"] = net_drawdown.maximum_drawdown
    values["gross_maximum_drawdown"] = gross_drawdown.maximum_drawdown
    values["drawdown_duration_bars"] = float(net_drawdown.longest_drawdown_duration_bars)
    if net_drawdown.maximum_drawdown > 0.0 and "annualized_return" in values:
        values["calmar_ratio"] = values["annualized_return"] / net_drawdown.maximum_drawdown
    else:
        skipped["calmar_ratio"] = "maximum_drawdown is zero, or annualized_return is unavailable -- Calmar ratio is undefined"

    # Section 1: genuinely bar-sampled exposure -- `time_in_market_fraction`
    # is the fraction of BARS with at least one open position;
    # `mean_total_absolute_exposure` is the mean, across every bar, of
    # `BarReturnPoint.total_absolute_exposure` (can exceed 1.0 under
    # `overlap_policy=independent_overlapping`, honestly reported rather
    # than silently capped).
    if bar_timeline.points:
        values["time_in_market_fraction"] = statistics.fmean(1.0 if p.open_trade_count > 0 else 0.0 for p in bar_timeline.points)
        values["mean_total_absolute_exposure"] = statistics.fmean(p.total_absolute_exposure for p in bar_timeline.points)
    else:
        skipped["time_in_market_fraction"] = "empty bar timeline"
        skipped["mean_total_absolute_exposure"] = "empty bar timeline"

    if closed:
        wins = [r for r in net_returns if r > 0.0]
        losses = [r for r in net_returns if r < 0.0]
        values["hit_rate"] = len(wins) / len(closed)
        values["average_trade_return"] = statistics.fmean(net_returns)
        values["median_trade_return"] = statistics.median(net_returns)
        values["best_trade"] = max(net_returns)
        values["worst_trade"] = min(net_returns)
        values["average_holding_period_bars"] = statistics.fmean(t.holding_bars for t in closed)

        # Milestone 5.2, Section 7: fraction of chronologically ADJACENT
        # closed-trade pairs (ordered by entry_bar_position) whose
        # direction FLIPS (LONG->SHORT or SHORT->LONG) -- a genuinely
        # distinct quantity from `trades_per_bar`/`transaction_sides_per_
        # bar` below (this is about the SEQUENCE of directions, not a
        # rate against bar count). Undefined with fewer than 2 closed
        # trades (nothing to compare).
        if len(closed) >= 2:
            ordered_by_entry = sorted(closed, key=lambda t: t.entry_bar_position)
            reversals = sum(1 for a, b in itertools.pairwise(ordered_by_entry) if a.direction != b.direction)
            values["reversal_rate"] = reversals / (len(ordered_by_entry) - 1)
        else:
            skipped["reversal_rate"] = "fewer than 2 closed trades -- reversal rate is undefined"

        # Milestone 5.2, Section 4: `turnover` (ambiguous) split into an
        # honestly-named raw transaction-SIDE count and an ACTUAL-
        # transacted-notional ratio, computed from each trade's own
        # persisted EFFECTIVE fill prices -- never merely assumed
        # constant across entry/exit. Under this platform's fixed-notional
        # position sizing, `quantity = entry_notional / entry_effective_
        # price` shares (or short-sold units) are transacted at entry, and
        # the SAME `quantity` reverses at exit -- so exit-leg notional is
        # `quantity * exit_effective_price = entry_notional *
        # (exit_effective_price / entry_effective_price)`. This is
        # DIRECTION-AGNOSTIC (a SHORT's entry/exit legs transact the same
        # dollar notional a LONG's would at the same prices, even though
        # the two directions' P&L/cash accounting differs -- see
        # `backtesting.timeline`'s `M`/`V` formulas, which are NOT reused
        # here because those model VALUE/liability, not transacted
        # notional). No time-normalization is applied to the total-notional
        # figures (a WITHIN-FOLD total has no natural per-year rate
        # without an arbitrary period choice); `annualized_turnover` is
        # the one exception, guarded identically to `annualized_return`
        # above -- reported only when this fold's own duration is long
        # enough to extrapolate meaningfully.
        entry_notional = exposure_cap * initial_notional
        transacted_notional_per_trade = [entry_notional * (1.0 + t.exit_effective_price / t.entry_effective_price) for t in closed]
        values["transaction_count"] = float(len(closed)) * 2.0
        # Milestone 5.2, Section 7: a prior delivery report described
        # `628 transactions / 350 bars = 1.79` as "round-trips per bar" --
        # WRONG, since `transaction_count` counts SIDES (entry + exit),
        # not round trips. The correctly-named value for that exact
        # computation is `transaction_sides_per_bar`; `trades_per_bar`
        # (computed unconditionally above, alongside `trade_count`) is
        # the genuine round-trips-per-bar figure (e.g. `314 trades / 350
        # bars = 0.90`) -- the two must never be conflated or reported
        # under each other's name.
        if n_points > 0:
            values["transaction_sides_per_bar"] = values["transaction_count"] / n_points
        else:
            skipped["transaction_sides_per_bar"] = "empty bar timeline"
        values["total_transacted_notional"] = sum(transacted_notional_per_trade)
        values["turnover_notional_ratio"] = values["total_transacted_notional"] / initial_notional
        if duration_years >= 1e-9:
            values["annualized_turnover"] = values["turnover_notional_ratio"] / duration_years
        else:
            skipped["annualized_turnover"] = "insufficient fold duration to annualize turnover meaningfully"

        if wins:
            values["average_win"] = statistics.fmean(wins)
        else:
            skipped["average_win"] = "no winning trades"
        if losses:
            values["average_loss"] = statistics.fmean(losses)
        else:
            skipped["average_loss"] = "no losing trades"

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss > 0.0:
            values["profit_factor"] = gross_profit / gross_loss
        else:
            skipped["profit_factor"] = "zero gross loss -- profit factor is undefined (infinite)"
        if wins and losses:
            values["payoff_ratio"] = statistics.fmean(wins) / abs(statistics.fmean(losses))
        else:
            skipped["payoff_ratio"] = "requires at least one win and one loss"

        win_rate = len(wins) / len(closed)
        if "average_win" in values and "average_loss" in values:
            values["expectancy"] = win_rate * values["average_win"] + (1.0 - win_rate) * values["average_loss"]
        else:
            skipped["expectancy"] = "requires both average_win and average_loss to be defined"

        values["consecutive_wins"] = float(_max_streak(net_returns, positive=True))
        values["consecutive_losses"] = float(_max_streak(net_returns, positive=False))

        mean_gross = statistics.fmean(gross_returns_list)
        mean_net = statistics.fmean(net_returns)
        # Section 2: renamed from `cost_drag` -- this is a MEAN per-trade
        # cost fraction, never to be labeled "total cost" in any report.
        values["mean_trade_cost"] = mean_gross - mean_net
        if mean_gross != 0.0:
            values["gross_to_net_degradation"] = (mean_gross - mean_net) / abs(mean_gross)
        else:
            skipped["gross_to_net_degradation"] = "mean gross return is exactly zero -- relative degradation is undefined"

        # Section 2: the genuine TOTAL (sum, not mean) cost across every
        # closed trade -- distinct from `mean_trade_cost` above.
        total_cost_fraction = sum(t.cost_breakdown.total_cost for t in closed)
        values["total_cost_fraction"] = total_cost_fraction
        values["total_cost_notional"] = total_cost_fraction * initial_notional * exposure_cap
    else:
        for name in (
            "hit_rate", "average_trade_return", "median_trade_return", "best_trade", "worst_trade", "average_holding_period_bars",
            "reversal_rate", "transaction_count", "transaction_sides_per_bar", "total_transacted_notional", "turnover_notional_ratio",
            "annualized_turnover", "average_win", "average_loss", "profit_factor", "payoff_ratio", "expectancy", "consecutive_wins",
            "consecutive_losses", "mean_trade_cost", "gross_to_net_degradation", "total_cost_fraction", "total_cost_notional",
        ):
            skipped[name] = "no closed trades in this fold"

    for name, value in values.items():
        if not math.isfinite(value):
            raise FinancialMetricError(f"compute_financial_metrics: computed metric {name!r} is non-finite ({value!r}) -- must be skipped with a reason instead, never persisted")

    return MetricComputationReport(values=values, skipped=skipped)


def _max_streak(returns: list[float], *, positive: bool) -> int:
    best = 0
    current = 0
    for r in returns:
        is_match = (r > 0.0) if positive else (r < 0.0)
        if is_match:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


__all__ = ["SharpeAssumptions", "bars_per_year", "compute_financial_metrics", "population_stdev", "sharpe_like"]
