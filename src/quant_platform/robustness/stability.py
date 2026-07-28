"""Fold stability and concentration-risk analysis (Milestone 6, Section
8). Measures how EVENLY a backtest's apparent performance is spread
across its own outer folds, trades, calendar days, directions, and
confidence buckets -- a backtest whose entire apparent edge comes from
one fold, one trade, one day, one direction, or one confidence bucket is
NOT robust evidence of a repeatable edge, even if its aggregate metrics
look good."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from quant_platform.backtesting.runner import BenchmarkReport, OuterFoldBacktestResult
from quant_platform.backtesting.trades import TradeRecord
from quant_platform.core.exceptions import StabilityAnalysisError
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    parse_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.specs import StabilityThresholds

FOLD_STABILITY_REPORT_SCHEMA_VERSION = 1


def _tercile_bucket(value: float) -> str:
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"


def _concentration_ratio(contributions: dict[str, float]) -> float | None:
    """`max positive contribution / total positive contribution`, or
    `None` if there is no positive total to divide by (no profitable
    unit at all -- concentration is undefined, not zero)."""
    positive = {k: v for k, v in contributions.items() if v > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return None
    return max(positive.values()) / total


@dataclass(frozen=True, slots=True)
class ConcentrationReport:
    schema_version: int
    single_fold_profit_concentration: float | None
    single_trade_profit_concentration: float | None
    single_day_profit_concentration: float | None
    single_direction_profit_concentration: float | None
    single_confidence_bucket_profit_concentration: float | None
    warning_codes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "single_fold_profit_concentration": self.single_fold_profit_concentration,
            "single_trade_profit_concentration": self.single_trade_profit_concentration, "single_day_profit_concentration": self.single_day_profit_concentration,
            "single_direction_profit_concentration": self.single_direction_profit_concentration,
            "single_confidence_bucket_profit_concentration": self.single_confidence_bucket_profit_concentration,
            "warning_codes": list(self.warning_codes),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ConcentrationReport:
        def _opt(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            schema_version=1, single_fold_profit_concentration=_opt("single_fold_profit_concentration"),
            single_trade_profit_concentration=_opt("single_trade_profit_concentration"), single_day_profit_concentration=_opt("single_day_profit_concentration"),
            single_direction_profit_concentration=_opt("single_direction_profit_concentration"),
            single_confidence_bucket_profit_concentration=_opt("single_confidence_bucket_profit_concentration"),
            warning_codes=tuple(str(c) for c in as_json_list(raw.get("warning_codes") or [], field_name="warning_codes")),
        )


def compute_concentration_report(fold_results: tuple[OuterFoldBacktestResult, ...], *, all_closed_trades: tuple[TradeRecord, ...], thresholds: StabilityThresholds) -> ConcentrationReport:
    fold_contributions = {str(r.outer_fold_index): float(r.financial_metrics.get("total_net_return", 0.0) or 0.0) for r in fold_results}
    trade_contributions = {t.trade_id: t.net_return for t in all_closed_trades}
    day_contributions: dict[str, float] = {}
    direction_contributions: dict[str, float] = {}
    bucket_contributions: dict[str, float] = {}
    for t in all_closed_trades:
        day_key = parse_utc_timestamp(t.exit_timestamp).date().isoformat()
        day_contributions[day_key] = day_contributions.get(day_key, 0.0) + t.net_return
        direction_contributions[t.direction.value] = direction_contributions.get(t.direction.value, 0.0) + t.net_return
        bucket_key = _tercile_bucket(t.confidence)
        bucket_contributions[bucket_key] = bucket_contributions.get(bucket_key, 0.0) + t.net_return

    fold_c = _concentration_ratio(fold_contributions)
    trade_c = _concentration_ratio(trade_contributions)
    day_c = _concentration_ratio(day_contributions)
    direction_c = _concentration_ratio(direction_contributions)
    bucket_c = _concentration_ratio(bucket_contributions)

    warnings: list[str] = []
    if fold_c is not None and fold_c > thresholds.maximum_single_fold_profit_concentration:
        warnings.append("single_fold_profit_concentration_exceeded")
    if trade_c is not None and trade_c > thresholds.maximum_single_trade_profit_concentration:
        warnings.append("single_trade_profit_concentration_exceeded")
    if direction_c is not None and direction_c > thresholds.maximum_single_direction_profit_concentration:
        warnings.append("single_direction_profit_concentration_exceeded")

    return ConcentrationReport(
        schema_version=1, single_fold_profit_concentration=fold_c, single_trade_profit_concentration=trade_c,
        single_day_profit_concentration=day_c, single_direction_profit_concentration=direction_c,
        single_confidence_bucket_profit_concentration=bucket_c, warning_codes=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class FoldStabilityReport:
    schema_version: int
    fold_count: int
    profitable_fold_fraction: float
    positive_sharpe_fold_fraction: float | None
    median_fold_return: float
    worst_fold_return: float
    fold_return_stdev: float
    fold_sharpe_dispersion: float | None
    maximum_fold_drawdown: float
    worst_fold_cost_drag: float | None
    fold_trade_count_dispersion: float
    fold_exposure_dispersion: float | None
    direction_consistency: float | None
    benchmark_outperformance_fraction: float | None
    concentration: ConcentrationReport
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "fold_count": self.fold_count, "profitable_fold_fraction": self.profitable_fold_fraction,
            "positive_sharpe_fold_fraction": self.positive_sharpe_fold_fraction, "median_fold_return": self.median_fold_return,
            "worst_fold_return": self.worst_fold_return, "fold_return_stdev": self.fold_return_stdev,
            "fold_sharpe_dispersion": self.fold_sharpe_dispersion, "maximum_fold_drawdown": self.maximum_fold_drawdown,
            "worst_fold_cost_drag": self.worst_fold_cost_drag, "fold_trade_count_dispersion": self.fold_trade_count_dispersion,
            "fold_exposure_dispersion": self.fold_exposure_dispersion, "direction_consistency": self.direction_consistency,
            "benchmark_outperformance_fraction": self.benchmark_outperformance_fraction, "concentration": self.concentration.to_json_dict(),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FoldStabilityReport:
        require_schema_version(raw, supported=FOLD_STABILITY_REPORT_SCHEMA_VERSION, context="FoldStabilityReport")

        def _opt(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            schema_version=FOLD_STABILITY_REPORT_SCHEMA_VERSION, fold_count=int(str(raw["fold_count"])),
            profitable_fold_fraction=float(str(raw["profitable_fold_fraction"])), positive_sharpe_fold_fraction=_opt("positive_sharpe_fold_fraction"),
            median_fold_return=float(str(raw["median_fold_return"])), worst_fold_return=float(str(raw["worst_fold_return"])),
            fold_return_stdev=float(str(raw["fold_return_stdev"])), fold_sharpe_dispersion=_opt("fold_sharpe_dispersion"),
            maximum_fold_drawdown=float(str(raw["maximum_fold_drawdown"])), worst_fold_cost_drag=_opt("worst_fold_cost_drag"),
            fold_trade_count_dispersion=float(str(raw["fold_trade_count_dispersion"])), fold_exposure_dispersion=_opt("fold_exposure_dispersion"),
            direction_consistency=_opt("direction_consistency"), benchmark_outperformance_fraction=_opt("benchmark_outperformance_fraction"),
            concentration=ConcentrationReport.from_json_dict(as_json_dict(raw["concentration"], field_name="concentration")),
            generated_at=str(raw["generated_at"]),
        )


def _numeric(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compute_fold_stability_report(
    fold_results: tuple[OuterFoldBacktestResult, ...], *, all_closed_trades: tuple[TradeRecord, ...], thresholds: StabilityThresholds,
    benchmark_reports: tuple[BenchmarkReport, ...] = (), benchmark_name: str = "always_flat_net_cost",
) -> FoldStabilityReport:
    if not fold_results:
        raise StabilityAnalysisError("compute_fold_stability_report: fold_results must not be empty")
    fold_results = tuple(sorted(fold_results, key=lambda r: r.outer_fold_index))

    net_returns = [_numeric(r.financial_metrics.get("total_net_return")) for r in fold_results]
    net_returns_defined = [v for v in net_returns if v is not None]
    if not net_returns_defined:
        raise StabilityAnalysisError("compute_fold_stability_report: no fold reports a defined total_net_return")

    sharpes = [_numeric(r.financial_metrics.get("bar_return_sharpe")) for r in fold_results]
    sharpes_defined = [v for v in sharpes if v is not None]

    drawdowns = [_numeric(r.financial_metrics.get("maximum_drawdown")) for r in fold_results]
    drawdowns_defined = [v for v in drawdowns if v is not None]

    cost_drags = [_numeric(r.financial_metrics.get("mean_trade_cost")) for r in fold_results]
    cost_drags_defined = [v for v in cost_drags if v is not None]

    exposures = [_numeric(r.financial_metrics.get("time_in_market_fraction")) for r in fold_results]
    exposures_defined = [v for v in exposures if v is not None]

    trade_counts = [float(r.closed_trade_count) for r in fold_results]

    profitable_fraction = sum(1 for v in net_returns_defined if v > 0.0) / len(net_returns_defined)
    positive_sharpe_fraction = (sum(1 for v in sharpes_defined if v > 0.0) / len(sharpes_defined)) if sharpes_defined else None
    fold_sharpe_dispersion = statistics.pstdev(sharpes_defined) if len(sharpes_defined) >= 2 else None
    worst_cost_drag = max(cost_drags_defined) if cost_drags_defined else None
    fold_exposure_dispersion = statistics.pstdev(exposures_defined) if len(exposures_defined) >= 2 else None

    # Direction consistency is computed from EACH fold's own trades -- but
    # `all_closed_trades` is a POOLED, cross-fold list; group by the
    # trade's own `outer_fold_index` (persisted on every TradeRecord).
    per_fold_direction: dict[int, str] = {}
    trades_by_fold: dict[int, list[TradeRecord]] = {}
    for t in all_closed_trades:
        trades_by_fold.setdefault(t.outer_fold_index, []).append(t)
    for fold_index, trades in trades_by_fold.items():
        longs = sum(1 for t in trades if t.direction.value == "long")
        shorts = sum(1 for t in trades if t.direction.value == "short")
        if longs != shorts:
            per_fold_direction[fold_index] = "long" if longs > shorts else "short"
    direction_consistency: float | None = None
    if per_fold_direction:
        overall_long = sum(1 for t in all_closed_trades if t.direction.value == "long")
        overall_short = sum(1 for t in all_closed_trades if t.direction.value == "short")
        if overall_long != overall_short:
            overall_dominant = "long" if overall_long > overall_short else "short"
            direction_consistency = sum(1 for d in per_fold_direction.values() if d == overall_dominant) / len(per_fold_direction)

    benchmark_outperformance: float | None = None
    if benchmark_reports:
        by_fold = {b.outer_fold_index: b for b in benchmark_reports}
        wins = 0
        total = 0
        for r in fold_results:
            report = by_fold.get(r.outer_fold_index)
            fold_return = _numeric(r.financial_metrics.get("total_net_return"))
            if report is None or fold_return is None:
                continue
            matching = [b for b in report.benchmarks if b.name == benchmark_name]
            if not matching:
                continue
            total += 1
            if fold_return > matching[0].net_return:
                wins += 1
        benchmark_outperformance = (wins / total) if total > 0 else None

    concentration = compute_concentration_report(fold_results, all_closed_trades=all_closed_trades, thresholds=thresholds)

    return FoldStabilityReport(
        schema_version=FOLD_STABILITY_REPORT_SCHEMA_VERSION, fold_count=len(fold_results), profitable_fold_fraction=profitable_fraction,
        positive_sharpe_fold_fraction=positive_sharpe_fraction, median_fold_return=statistics.median(net_returns_defined),
        worst_fold_return=min(net_returns_defined), fold_return_stdev=(statistics.pstdev(net_returns_defined) if len(net_returns_defined) >= 2 else 0.0),
        fold_sharpe_dispersion=fold_sharpe_dispersion, maximum_fold_drawdown=(max(drawdowns_defined) if drawdowns_defined else 0.0),
        worst_fold_cost_drag=worst_cost_drag, fold_trade_count_dispersion=(statistics.pstdev(trade_counts) if len(trade_counts) >= 2 else 0.0),
        fold_exposure_dispersion=fold_exposure_dispersion, direction_consistency=direction_consistency,
        benchmark_outperformance_fraction=benchmark_outperformance, concentration=concentration, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "FOLD_STABILITY_REPORT_SCHEMA_VERSION",
    "ConcentrationReport",
    "FoldStabilityReport",
    "compute_concentration_report",
    "compute_fold_stability_report",
]
