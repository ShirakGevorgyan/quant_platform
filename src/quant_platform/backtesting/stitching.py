"""Stitched walk-forward equity (Milestone 5.1, Section 3) -- the single
chronological equity path spanning EVERY outer test fold of one backtest,
built by chaining each fold's own bar-level `timeline.BarReturnTimeline`
(Section 1) end to end.

FOUR DISTINCT NOTIONS THIS MODULE DISAMBIGUATES (Section 3's own wording)
--------------------------------------------------------------------------
- PER-FOLD NORMALIZED EQUITY: `BarReturnTimeline`'s own cumulative_gross/
  net_equity, each fold independently restarting at its own ~1.0 baseline
  -- exactly what `OuterFoldBacktestResult.equity_curve_reference` (and
  `bar_return_timeline_reference`) already hold. NOT this module's output.
- FOLD-WISE SCALAR AGGREGATION: a single number PER FOLD (e.g. each
  fold's own `total_net_return`) summarized/averaged across folds -- that
  is `reporting.build_backtest_report_json`'s job, computed from the
  per-fold `OuterFoldBacktestResult`s directly. NOT this module's output.
- POOLED TRADE STATISTICS: trade-level metrics (hit_rate, profit_factor,
  ...) computed over the UNION of every fold's closed trades, ignoring
  fold/bar structure entirely -- a distinct, trade-sampled concept. NOT
  computed here (this module is bar-sampled throughout).
- STITCHED CHRONOLOGICAL EQUITY (what THIS module builds): ONE continuous,
  bar-sampled equity path across the whole backtest, where fold i+1's
  path CONTINUES from fold i's ending normalized equity rather than
  restarting at 1.0 -- what a single trader running this strategy
  continuously through every outer-test window, one after another, would
  have actually experienced.

HOW CONTINUATION WORKS (deterministic, carry-forward, no averaging)
--------------------------------------------------------------------------
Let `carry_gross_i`/`carry_net_i` be the stitched equity value at the
INSTANT fold `i` begins (`carry_gross_0 = carry_net_0 = 1.0`: the
backtest's own starting point). For every bar `j` inside fold `i`, with
`local_gross`/`local_net` its OWN `BarReturnTimeline.cumulative_*_equity`
value (fold-locally normalized, per Section 1):
    COMPOUNDED   (multiplicative): stitched = carry_i * local
    NON_COMPOUNDED (additive):     stitched = carry_i + (local - 1.0)
After the fold's last bar, `carry_{i+1}` is set to that fold's OWN final
stitched value -- i.e. `carry_{i+1} = carry_i * local_final` (compounded)
or `carry_i + (local_final - 1.0)` (non-compounded). This is exactly
"every fold starts from the previous fold's ending normalized equity"
(Section 3), expressed uniformly for both compounding conventions. No
equity curve is ever averaged; each stitched point is a pure, order-
dependent function of every earlier fold's own already-computed totals
and the current fold's own local per-bar path -- never of a LATER fold's
data (folds are processed strictly in `outer_fold_index` order, each
depending only on the running `carry` computed from folds already
consumed).

POSITIONS NEVER CROSS A FOLD BOUNDARY (already true; not re-derived here)
--------------------------------------------------------------------------
`execution.simulate_outer_fold_trades` simulates trades independently per
outer fold (each fold's `TradeRecord`s reference only that fold's own bar
positions) -- no position spans two folds by construction, upstream of
this module. `FoldBoundaryMarker.stitched_point_start_index`/`_end_index`
below record, per fold, exactly which stitched points came from it, so a
reader can confirm the boundary independently.

WHAT "STITCHED SHARPE/SORTINO/EXPOSURE/TURNOVER" MEANS HERE
--------------------------------------------------------------------------
`compute_stitched_financial_metrics` pools every fold's own per-bar
`gross_return`/`net_return` observations (LOCAL, fold-relative values --
unaffected by the `carry` rescaling, which only affects the CUMULATIVE
level) into one chronological series and applies the exact same
`metrics.sharpe_like`/`metrics.population_stdev` formulas the per-fold
`bar_return_sharpe` uses, annualized at the same `bars_per_year`. Exposure
(`stitched_time_in_market_fraction`/`stitched_mean_total_absolute_
exposure`) and turnover (`stitched_transaction_count`/`stitched_turnover_
notional_ratio`) are likewise bar-sampled aggregates over the whole
stitched point series, distinct from (never mixed with) the pooled TRADE
statistics this module deliberately does not compute."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from quant_platform.backtesting.drawdown import DrawdownReport
from quant_platform.backtesting.metrics import bars_per_year, population_stdev, sharpe_like
from quant_platform.backtesting.returns import EQUITY_POINT_SCHEMA_VERSION, EquityCurve, EquityPoint
from quant_platform.backtesting.timeline import BarReturnTimeline
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.core.types import Timeframe
from quant_platform.ml.metrics import MetricComputationReport
from quant_platform.ml.persistence import (
    as_json_list,
    parse_utc_timestamp,
    require_schema_version,
)

STITCHED_WALK_FORWARD_EQUITY_SCHEMA_VERSION = 1
STITCHED_EQUITY_OUTER_FOLD_INDEX = -1
"""Sentinel `outer_fold_index` used when projecting a stitched series to
`EquityCurve` (for reuse by `drawdown.compute_drawdown_report`) -- never a
valid single outer-fold index (those are always `>= 0`), so a reader can
tell at a glance that a `DrawdownReport`/`EquityCurve` with this index
spans the WHOLE backtest, not one fold."""


@dataclass(frozen=True, slots=True)
class StitchedFoldBoundary:
    """One persisted marker per outer fold contributing to the stitched
    series (Section 3: "fold-boundary markers persisted")."""

    outer_fold_index: int
    stitched_point_start_index: int
    stitched_point_end_index: int
    fold_start_position: int
    fold_end_position: int
    carry_in_gross_equity: float
    carry_in_net_equity: float

    def __post_init__(self) -> None:
        if self.outer_fold_index < 0:
            raise FinancialMetricError(f"StitchedFoldBoundary.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if self.stitched_point_start_index < 0 or self.stitched_point_end_index < self.stitched_point_start_index:
            raise FinancialMetricError("StitchedFoldBoundary: stitched_point_start_index/_end_index must describe a non-empty, non-negative range")
        if self.fold_end_position < self.fold_start_position:
            raise FinancialMetricError("StitchedFoldBoundary: fold_end_position must be >= fold_start_position")
        if not (math.isfinite(self.carry_in_gross_equity) and math.isfinite(self.carry_in_net_equity)):
            raise FinancialMetricError("StitchedFoldBoundary: carry_in_gross_equity/carry_in_net_equity must be finite")
        if self.carry_in_gross_equity <= 0.0 or self.carry_in_net_equity <= 0.0:
            raise FinancialMetricError("StitchedFoldBoundary: carry_in_gross_equity/carry_in_net_equity must remain positive")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "outer_fold_index": self.outer_fold_index, "stitched_point_start_index": self.stitched_point_start_index,
            "stitched_point_end_index": self.stitched_point_end_index, "fold_start_position": self.fold_start_position,
            "fold_end_position": self.fold_end_position, "carry_in_gross_equity": self.carry_in_gross_equity,
            "carry_in_net_equity": self.carry_in_net_equity,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StitchedFoldBoundary:
        return cls(
            outer_fold_index=int(str(raw["outer_fold_index"])), stitched_point_start_index=int(str(raw["stitched_point_start_index"])),
            stitched_point_end_index=int(str(raw["stitched_point_end_index"])), fold_start_position=int(str(raw["fold_start_position"])),
            fold_end_position=int(str(raw["fold_end_position"])), carry_in_gross_equity=float(str(raw["carry_in_gross_equity"])),
            carry_in_net_equity=float(str(raw["carry_in_net_equity"])),
        )


@dataclass(frozen=True, slots=True)
class StitchedEquityPoint:
    schema_version: int
    outer_fold_index: int
    bar_position: int
    """Fold-LOCAL bar position (the same value the source `BarReturnPoint.
    bar_position` carries) -- kept for traceability back to its origin
    fold; NOT a position in a single continuous stitched bar index (folds
    may have gaps/overlap-free but non-contiguous positions between
    them)."""
    timestamp: str
    bar_gross_return: float
    bar_net_return: float
    """The fold-LOCAL per-bar return (identical to the source
    `BarReturnPoint.gross_return`/`net_return`) -- unaffected by stitching
    (the `carry` rescaling only ever changes the CUMULATIVE level, never a
    per-bar relative return)."""
    stitched_gross_equity: float
    stitched_net_equity: float
    total_absolute_exposure: float
    net_exposure: float
    open_trade_count: int
    entries_count: int
    exits_count: int

    def __post_init__(self) -> None:
        parse_utc_timestamp(self.timestamp)
        if self.outer_fold_index < 0 or self.bar_position < 0:
            raise FinancialMetricError("StitchedEquityPoint.outer_fold_index/bar_position must be >= 0")
        for name, value in (
            ("bar_gross_return", self.bar_gross_return), ("bar_net_return", self.bar_net_return),
            ("stitched_gross_equity", self.stitched_gross_equity), ("stitched_net_equity", self.stitched_net_equity),
            ("total_absolute_exposure", self.total_absolute_exposure), ("net_exposure", self.net_exposure),
        ):
            if not math.isfinite(value):
                raise FinancialMetricError(f"StitchedEquityPoint.{name} must be finite, got {value!r}")
        if self.stitched_gross_equity <= 0.0 or self.stitched_net_equity <= 0.0:
            raise FinancialMetricError("StitchedEquityPoint: stitched cumulative equity must remain positive")
        if self.total_absolute_exposure < 0.0:
            raise FinancialMetricError("StitchedEquityPoint.total_absolute_exposure must be >= 0")
        if self.open_trade_count < 0 or self.entries_count < 0 or self.exits_count < 0:
            raise FinancialMetricError("StitchedEquityPoint: open_trade_count/entries_count/exits_count must be >= 0")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "bar_position": self.bar_position,
            "timestamp": self.timestamp, "bar_gross_return": self.bar_gross_return, "bar_net_return": self.bar_net_return,
            "stitched_gross_equity": self.stitched_gross_equity, "stitched_net_equity": self.stitched_net_equity,
            "total_absolute_exposure": self.total_absolute_exposure, "net_exposure": self.net_exposure,
            "open_trade_count": self.open_trade_count, "entries_count": self.entries_count, "exits_count": self.exits_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StitchedEquityPoint:
        return cls(
            schema_version=int(str(raw["schema_version"])), outer_fold_index=int(str(raw["outer_fold_index"])),
            bar_position=int(str(raw["bar_position"])), timestamp=str(raw["timestamp"]),
            bar_gross_return=float(str(raw["bar_gross_return"])), bar_net_return=float(str(raw["bar_net_return"])),
            stitched_gross_equity=float(str(raw["stitched_gross_equity"])), stitched_net_equity=float(str(raw["stitched_net_equity"])),
            total_absolute_exposure=float(str(raw["total_absolute_exposure"])), net_exposure=float(str(raw["net_exposure"])),
            open_trade_count=int(str(raw["open_trade_count"])), entries_count=int(str(raw["entries_count"])), exits_count=int(str(raw["exits_count"])),
        )


@dataclass(frozen=True, slots=True)
class StitchedWalkForwardEquity:
    schema_version: int
    backtest_id: str
    compounded: bool
    fold_boundaries: tuple[StitchedFoldBoundary, ...]
    points: tuple[StitchedEquityPoint, ...]

    def __post_init__(self) -> None:
        if not self.fold_boundaries or not self.points:
            raise FinancialMetricError("StitchedWalkForwardEquity: fold_boundaries and points must both be non-empty")
        expected_start = 0
        previous_fold_index = -1
        for boundary in self.fold_boundaries:
            if boundary.outer_fold_index <= previous_fold_index:
                raise FinancialMetricError("StitchedWalkForwardEquity.fold_boundaries must be strictly increasing by outer_fold_index")
            if boundary.stitched_point_start_index != expected_start:
                raise FinancialMetricError(
                    f"StitchedWalkForwardEquity.fold_boundaries must contiguously cover points: expected fold "
                    f"{boundary.outer_fold_index} to start at point index {expected_start}, got {boundary.stitched_point_start_index}"
                )
            previous_fold_index = boundary.outer_fold_index
            expected_start = boundary.stitched_point_end_index + 1
        if expected_start != len(self.points):
            raise FinancialMetricError(
                f"StitchedWalkForwardEquity.fold_boundaries must contiguously cover every point: covered "
                f"{expected_start}, but points has {len(self.points)}"
            )
        timestamps = [p.timestamp for p in self.points]
        if timestamps != sorted(timestamps):
            raise FinancialMetricError("StitchedWalkForwardEquity.points must be in non-decreasing chronological order")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "backtest_id": self.backtest_id, "compounded": self.compounded,
            "fold_boundaries": [b.to_json_dict() for b in self.fold_boundaries], "points": [p.to_json_dict() for p in self.points],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StitchedWalkForwardEquity:
        require_schema_version(raw, supported=STITCHED_WALK_FORWARD_EQUITY_SCHEMA_VERSION, context="StitchedWalkForwardEquity")
        return cls(
            schema_version=STITCHED_WALK_FORWARD_EQUITY_SCHEMA_VERSION, backtest_id=str(raw["backtest_id"]), compounded=bool(raw["compounded"]),
            fold_boundaries=tuple(StitchedFoldBoundary.from_json_dict(b) for b in as_json_list(raw["fold_boundaries"], field_name="fold_boundaries")),
            points=tuple(StitchedEquityPoint.from_json_dict(p) for p in as_json_list(raw["points"], field_name="points")),
        )


def build_stitched_walk_forward_equity(*, backtest_id: str, timelines: list[BarReturnTimeline]) -> StitchedWalkForwardEquity:
    """Section 3: chain `timelines` (one `BarReturnTimeline` per outer
    fold) into a single stitched series. `timelines` must already be in
    strictly increasing `outer_fold_index` order (the caller's natural
    iteration order over `BacktestManifest.outer_fold_result_references`)
    -- this function additionally VALIDATES, rather than merely assumes,
    that the folds are also strictly chronologically non-overlapping by
    TIMESTAMP (Section 3: "folds ordered by test timestamps"), failing
    closed if a caller ever passes folds out of order or a caller's outer
    fold plan somehow produced overlapping test windows."""
    if not timelines:
        raise FinancialMetricError("build_stitched_walk_forward_equity: timelines must be non-empty")
    compounded = timelines[0].compounded
    if any(t.compounded != compounded for t in timelines):
        raise FinancialMetricError("build_stitched_walk_forward_equity: every fold's BarReturnTimeline must share the same compounding mode")

    points: list[StitchedEquityPoint] = []
    boundaries: list[StitchedFoldBoundary] = []
    carry_gross = 1.0
    carry_net = 1.0
    previous_fold_index = -1
    previous_last_timestamp: str | None = None

    for timeline in timelines:
        if timeline.outer_fold_index <= previous_fold_index:
            raise FinancialMetricError(
                f"build_stitched_walk_forward_equity: outer folds must be strictly increasing -- got "
                f"outer_fold_index={timeline.outer_fold_index} after {previous_fold_index}"
            )
        if not timeline.points:
            raise FinancialMetricError(f"build_stitched_walk_forward_equity: outer fold {timeline.outer_fold_index} has an empty BarReturnTimeline")
        first_timestamp = timeline.points[0].timestamp
        if previous_last_timestamp is not None and first_timestamp <= previous_last_timestamp:
            raise FinancialMetricError(
                f"build_stitched_walk_forward_equity: outer fold {timeline.outer_fold_index} begins at "
                f"{first_timestamp!r}, which does not come strictly after the previous fold's last bar "
                f"({previous_last_timestamp!r}) -- outer-test windows must be chronologically ordered and non-overlapping"
            )

        start_index = len(points)
        for p in timeline.points:
            if compounded:
                stitched_gross = carry_gross * p.cumulative_gross_equity
                stitched_net = carry_net * p.cumulative_net_equity
            else:
                stitched_gross = carry_gross + (p.cumulative_gross_equity - 1.0)
                stitched_net = carry_net + (p.cumulative_net_equity - 1.0)
            points.append(StitchedEquityPoint(
                schema_version=1, outer_fold_index=timeline.outer_fold_index, bar_position=p.bar_position, timestamp=p.timestamp,
                bar_gross_return=p.gross_return, bar_net_return=p.net_return, stitched_gross_equity=stitched_gross,
                stitched_net_equity=stitched_net, total_absolute_exposure=p.total_absolute_exposure, net_exposure=p.net_exposure,
                open_trade_count=p.open_trade_count, entries_count=p.entries_count, exits_count=p.exits_count,
            ))

        boundaries.append(StitchedFoldBoundary(
            outer_fold_index=timeline.outer_fold_index, stitched_point_start_index=start_index, stitched_point_end_index=len(points) - 1,
            fold_start_position=timeline.fold_start_position, fold_end_position=timeline.fold_end_position,
            carry_in_gross_equity=carry_gross, carry_in_net_equity=carry_net,
        ))

        last = timeline.points[-1]
        carry_gross, carry_net = points[-1].stitched_gross_equity, points[-1].stitched_net_equity
        previous_fold_index = timeline.outer_fold_index
        previous_last_timestamp = last.timestamp

    return StitchedWalkForwardEquity(
        schema_version=STITCHED_WALK_FORWARD_EQUITY_SCHEMA_VERSION, backtest_id=backtest_id, compounded=compounded,
        fold_boundaries=tuple(boundaries), points=tuple(points),
    )


def stitched_walk_forward_equity_to_equity_curve(stitched: StitchedWalkForwardEquity) -> EquityCurve:
    """Projects the stitched series to `returns.EquityCurve`'s existing
    shape for reuse by `drawdown.compute_drawdown_report` UNCHANGED --
    exactly the same projection trick `timeline.bar_return_timeline_to_
    equity_curve` uses per fold. `outer_fold_index` is set to the
    `STITCHED_EQUITY_OUTER_FOLD_INDEX` sentinel."""
    trade_count_to_date = 0
    points: list[EquityPoint] = []
    for p in stitched.points:
        trade_count_to_date += p.exits_count
        points.append(EquityPoint(
            timestamp=p.timestamp, period_gross_return=p.bar_gross_return, period_net_return=p.bar_net_return,
            cumulative_gross_equity=p.stitched_gross_equity, cumulative_net_equity=p.stitched_net_equity,
            exposure=p.total_absolute_exposure, trade_count_to_date=trade_count_to_date,
        ))
    return EquityCurve(schema_version=EQUITY_POINT_SCHEMA_VERSION, outer_fold_index=STITCHED_EQUITY_OUTER_FOLD_INDEX, compounded=stitched.compounded, points=tuple(points))


def compute_stitched_financial_metrics(
    *, stitched: StitchedWalkForwardEquity, gross_drawdown: DrawdownReport, net_drawdown: DrawdownReport,
    bar_interval: Timeframe, annual_risk_free_rate: float, initial_notional: float, total_transacted_notional: float,
) -> MetricComputationReport:
    """Section 3: stitched gross/net total return, Sharpe/Sortino (pooled
    over every fold's own bar-level returns, chronologically), exposure,
    turnover -- deliberately excludes pooled TRADE-level statistics (hit
    rate, profit factor, ...), which are a distinct concept (see module
    docstring) computed elsewhere from the UNION of folds' `TradeSet`s,
    never from this bar-sampled series.

    MILESTONE 5.2, SECTION 4: `total_transacted_notional` is NOT computed
    here (this function only sees bar-sampled `StitchedEquityPoint`s, not
    individual trades' fill prices) -- it is the CALLER's responsibility
    to sum each fold's own exact, fill-price-derived `total_transacted_
    notional` (`metrics.compute_financial_metrics`'s corrected Section 4
    formula) and pass the pooled total in here, exactly as `reporting.
    _stitched_equity_section` does. This mirrors `stitched_transaction_
    count` (a pure count, safe to sum) but for notional -- the previous
    `stitched_transaction_count * exposure_cap` formula had the identical
    "every leg transacts the same notional" defect the per-fold formula
    had, and is not reused here."""
    values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    periods_per_year = bars_per_year(bar_interval)

    last = stitched.points[-1]
    values["stitched_total_gross_return"] = last.stitched_gross_equity - 1.0
    values["stitched_total_net_return"] = last.stitched_net_equity - 1.0

    n_points = len(stitched.points)
    duration_years = n_points / periods_per_year if periods_per_year > 0 else 0.0
    if duration_years > 0 and last.stitched_net_equity > 0:
        if duration_years < 1e-9:
            values["stitched_annualized_return"] = values["stitched_total_net_return"]
        else:
            try:
                values["stitched_annualized_return"] = math.exp(math.log1p(values["stitched_total_net_return"]) / duration_years) - 1.0
            except OverflowError:
                skipped["stitched_annualized_return"] = (
                    "extrapolating the stitched return to a full year overflowed double precision -- the stitched "
                    "series' duration is too short relative to a year for annualization to be numerically meaningful"
                )
    else:
        skipped["stitched_annualized_return"] = "insufficient stitched-equity duration to annualize"

    bar_net_returns = [p.bar_net_return for p in stitched.points]
    stdev = population_stdev(bar_net_returns)
    if stdev is None:
        skipped["stitched_volatility"] = "fewer than 2 stitched bar-return observations -- volatility is undefined"
        skipped["stitched_annualized_volatility"] = skipped["stitched_volatility"]
    else:
        values["stitched_volatility"] = stdev
        values["stitched_annualized_volatility"] = stdev * math.sqrt(periods_per_year)

    sharpe, sharpe_reason, assumptions = sharpe_like(bar_net_returns, annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year, downside_only=False)
    if sharpe is None:
        skipped["stitched_bar_return_sharpe"] = sharpe_reason or "undefined"
    else:
        values["stitched_bar_return_sharpe"] = sharpe

    sortino, sortino_reason, _assumptions = sharpe_like(bar_net_returns, annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year, downside_only=True)
    if sortino is None:
        skipped["stitched_sortino_ratio"] = sortino_reason or "undefined"
    else:
        values["stitched_sortino_ratio"] = sortino

    values["stitched_sharpe_annualization_factor"] = assumptions.annualization_factor
    values["stitched_sharpe_annual_risk_free_rate"] = assumptions.annual_risk_free_rate
    values["stitched_bar_return_sharpe_sample_count"] = float(assumptions.sample_count)

    values["stitched_maximum_drawdown"] = net_drawdown.maximum_drawdown
    values["stitched_gross_maximum_drawdown"] = gross_drawdown.maximum_drawdown
    values["stitched_drawdown_duration_bars"] = float(net_drawdown.longest_drawdown_duration_bars)
    if net_drawdown.maximum_drawdown > 0.0 and "stitched_annualized_return" in values:
        values["stitched_calmar_ratio"] = values["stitched_annualized_return"] / net_drawdown.maximum_drawdown
    else:
        skipped["stitched_calmar_ratio"] = "stitched_maximum_drawdown is zero, or stitched_annualized_return is unavailable"

    values["stitched_time_in_market_fraction"] = statistics.fmean(1.0 if p.open_trade_count > 0 else 0.0 for p in stitched.points)
    values["stitched_mean_total_absolute_exposure"] = statistics.fmean(p.total_absolute_exposure for p in stitched.points)

    total_entries = sum(p.entries_count for p in stitched.points)
    total_exits = sum(p.exits_count for p in stitched.points)
    values["stitched_transaction_count"] = float(total_entries + total_exits)
    # Milestone 5.2, Section 4 correction: computed from the pooled,
    # exact, fill-price-derived per-fold totals (see this function's own
    # docstring) -- never `stitched_transaction_count * exposure_cap`.
    values["stitched_total_transacted_notional"] = total_transacted_notional
    values["stitched_turnover_notional_ratio"] = total_transacted_notional / initial_notional
    if duration_years >= 1e-9:
        values["stitched_annualized_turnover"] = values["stitched_turnover_notional_ratio"] / duration_years
    else:
        skipped["stitched_annualized_turnover"] = "insufficient stitched-equity duration to annualize turnover meaningfully"

    for name, value in values.items():
        if not math.isfinite(value):
            raise FinancialMetricError(f"compute_stitched_financial_metrics: computed metric {name!r} is non-finite ({value!r})")

    return MetricComputationReport(values=values, skipped=skipped)


__all__ = [
    "STITCHED_EQUITY_OUTER_FOLD_INDEX",
    "STITCHED_WALK_FORWARD_EQUITY_SCHEMA_VERSION",
    "StitchedEquityPoint",
    "StitchedFoldBoundary",
    "StitchedWalkForwardEquity",
    "build_stitched_walk_forward_equity",
    "compute_stitched_financial_metrics",
    "stitched_walk_forward_equity_to_equity_curve",
]
