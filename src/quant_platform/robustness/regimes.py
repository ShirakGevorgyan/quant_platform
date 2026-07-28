"""Leakage-safe regime robustness analysis (Milestone 6, Section 11).
Every regime label is computed from information available strictly
AT-OR-BEFORE the bar being classified:

- Calendar dimensions (`SESSION`, `DAY_OF_WEEK`, `HOUR_OF_DAY`) use only
  that bar's OWN already-known timestamp.
- Trailing dimensions (`TREND_DIRECTION`, `PRICE_REGIME`,
  `VOLATILITY_QUANTILE`, `LIQUIDITY_QUANTILE`) use a fixed backward-
  looking window of raw OHLCV bars ending at (and including) the
  classified bar -- never a bar past it. Quantile dimensions additionally
  rank each bar's trailing metric against the EXPANDING set of trailing
  metrics observed at or before that same bar -- again, never against a
  value that depends on a later bar.
- `SPREAD_QUANTILE` is reported UNAVAILABLE at this layer: this
  platform's backtesting-layer OHLCV schema (`core.types.OHLCV_COLUMNS`
  -- open_time/open/high/low/close/volume) carries no spread column, so
  there is nothing to classify. Per Section 11's own instruction, this is
  reported explicitly rather than fabricated from a cost-model constant.

BENCHMARK COMPARISON: each bucket is compared against a simple, zero-cost,
always-long buy-and-hold per-bar return (`close.pct_change()`) computed
directly from the same OHLCV bars -- NOT the fold-level `BenchmarkReport`
artifact used elsewhere in this platform (that artifact has no per-bar
granularity to align with scattered regime buckets).

PER-REGIME DRAWDOWN: a regime bucket's bars are, in general, NOT
contiguous in time. `RegimeBucketResult.maximum_drawdown` is computed on
a SYNTHETIC equity curve built by compounding only that bucket's own bar
net returns in their original chronological order -- an illustrative "if
I had only lived through this regime's bars back to back" figure, never a
real, continuously-held drawdown experience. This is documented, not
hidden.

MINIMUM-SAMPLE ENFORCEMENT: a bucket with fewer observations than its
`RegimeDefinitionSpec.minimum_regime_samples` is still reported (never
silently dropped), with `skipped=True` and every aggregate field `None`."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd

from quant_platform.backtesting.metrics import bars_per_year, sharpe_like
from quant_platform.backtesting.timeline import BarReturnTimeline
from quant_platform.backtesting.trades import TradeRecord
from quant_platform.core.exceptions import RegimeAnalysisError
from quant_platform.core.types import Timeframe
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import RegimeDimensionKind
from quant_platform.robustness.specs import RegimeDefinitionSpec, RobustnessSpec

REGIME_REPORT_SCHEMA_VERSION = 1

_CALENDAR_DIMENSIONS: frozenset[RegimeDimensionKind] = frozenset({RegimeDimensionKind.SESSION, RegimeDimensionKind.DAY_OF_WEEK, RegimeDimensionKind.HOUR_OF_DAY})
_PRICE_REGIME_BAND = 0.005
"""A fixed, documented, instrument-agnostic threshold for `PRICE_REGIME`
(0.5% trailing return) -- deliberately not calibrated per-instrument, to
avoid fabricating a data-derived threshold this module cannot honestly
support at this layer."""
_SESSION_BOUNDARIES_UTC_HOUR: tuple[tuple[int, int, str], ...] = (
    (0, 7, "asia"), (7, 13, "london"), (13, 16, "london_ny_overlap"), (16, 21, "new_york"), (21, 24, "other"),
)
"""Coarse, illustrative UTC-hour session boundaries -- not exact real
trading-desk session definitions."""


def _session_label(hour: int) -> str:
    for lo, hi, label in _SESSION_BOUNDARIES_UTC_HOUR:
        if lo <= hour < hi:
            return label
    raise RegimeAnalysisError(f"_session_label: hour={hour!r} did not match any declared session boundary")


@dataclass(frozen=True, slots=True)
class _BarRecord:
    net_return: float
    gross_return: float
    transaction_costs: float
    total_absolute_exposure: float
    benchmark_bar_return: float


def _build_bar_records(bar_timelines: tuple[BarReturnTimeline, ...], *, bars: pd.DataFrame) -> dict[int, _BarRecord]:
    benchmark_returns = bars["close"].pct_change().fillna(0.0)
    records: dict[int, _BarRecord] = {}
    for timeline in bar_timelines:
        for point in timeline.points:
            if point.bar_position in records:
                continue
            records[point.bar_position] = _BarRecord(
                net_return=point.net_return, gross_return=point.gross_return, transaction_costs=point.transaction_costs,
                total_absolute_exposure=point.total_absolute_exposure, benchmark_bar_return=float(benchmark_returns.iloc[point.bar_position]),
            )
    return records


# --------------------------------------------------------------------------
# Per-dimension bar classification: bar_position -> regime label. A bar
# absent from the returned mapping had insufficient trailing history to
# classify under this dimension (reported in `excluded_insufficient_
# history_count`, never silently merged into a bucket).
# --------------------------------------------------------------------------
def _classify_calendar(dimension: RegimeDimensionKind, bar_positions: list[int], *, bars: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    for pos in bar_positions:
        ts = bars.iloc[pos]["open_time"]
        if dimension is RegimeDimensionKind.DAY_OF_WEEK:
            labels[pos] = str(ts.day_name())
        elif dimension is RegimeDimensionKind.HOUR_OF_DAY:
            labels[pos] = f"{int(ts.hour):02d}"
        else:
            labels[pos] = _session_label(int(ts.hour))
    return labels


def _classify_trend_direction(bar_positions: list[int], *, bars: pd.DataFrame, window: int) -> dict[int, str]:
    close = bars["close"]
    labels: dict[int, str] = {}
    for pos in bar_positions:
        if pos < window:
            continue
        trailing_return = float(close.iloc[pos]) / float(close.iloc[pos - window]) - 1.0
        labels[pos] = "up" if trailing_return > 0.0 else ("down" if trailing_return < 0.0 else "flat")
    return labels


def _classify_price_regime(bar_positions: list[int], *, bars: pd.DataFrame, window: int) -> dict[int, str]:
    close = bars["close"]
    labels: dict[int, str] = {}
    for pos in bar_positions:
        if pos < window:
            continue
        trailing_return = float(close.iloc[pos]) / float(close.iloc[pos - window]) - 1.0
        if trailing_return > _PRICE_REGIME_BAND:
            labels[pos] = "bull"
        elif trailing_return < -_PRICE_REGIME_BAND:
            labels[pos] = "bear"
        else:
            labels[pos] = "sideways"
    return labels


def _classify_quantile_dimension(bar_positions: list[int], *, metric: pd.Series, n_quantiles: int) -> dict[int, str]:
    """Ranks each bar's trailing metric against the EXPANDING set of
    trailing-metric values observed at or before that same bar --
    leakage-safe by construction. O(n^2) in the number of defined metric
    observations; acceptable at walk-forward-fold scale, not intended for
    multi-million-bar analysis without further optimization (see
    Milestone 6 delivery report's own limitations section)."""
    defined_positions = [p for p in bar_positions if pd.notna(metric.iloc[p])]
    values_so_far: list[float] = []
    labels: dict[int, str] = {}
    for pos in sorted(defined_positions):
        value = float(metric.iloc[pos])
        values_so_far.append(value)
        rank = sum(1 for v in values_so_far if v <= value) / len(values_so_far)
        bucket_index = min(n_quantiles - 1, int(rank * n_quantiles))
        labels[pos] = f"q{bucket_index + 1}_of_{n_quantiles}"
    return labels


def _classify_dimension(regime_def: RegimeDefinitionSpec, *, bar_positions: list[int], bars: pd.DataFrame) -> tuple[dict[int, str] | None, str | None]:
    dimension = regime_def.dimension
    if dimension in _CALENDAR_DIMENSIONS:
        return _classify_calendar(dimension, bar_positions, bars=bars), None
    if dimension is RegimeDimensionKind.SPREAD_QUANTILE:
        return None, (
            "not available: this platform's backtesting-layer OHLCV bar schema (open_time, open, high, low, close, "
            "volume) has no spread column -- spread-quantile regime classification requires tick-level bid/ask data "
            "not present at this layer"
        )
    assert regime_def.trailing_window_bars is not None
    window = regime_def.trailing_window_bars
    if dimension is RegimeDimensionKind.TREND_DIRECTION:
        return _classify_trend_direction(bar_positions, bars=bars, window=window), None
    if dimension is RegimeDimensionKind.PRICE_REGIME:
        return _classify_price_regime(bar_positions, bars=bars, window=window), None
    if dimension is RegimeDimensionKind.VOLATILITY_QUANTILE:
        assert regime_def.n_quantiles is not None
        volatility = bars["close"].pct_change().rolling(window=window).std()
        return _classify_quantile_dimension(bar_positions, metric=volatility, n_quantiles=regime_def.n_quantiles), None
    if dimension is RegimeDimensionKind.LIQUIDITY_QUANTILE:
        assert regime_def.n_quantiles is not None
        liquidity = bars["volume"].rolling(window=window).mean()
        return _classify_quantile_dimension(bar_positions, metric=liquidity, n_quantiles=regime_def.n_quantiles), None
    raise RegimeAnalysisError(f"_classify_dimension: unhandled RegimeDimensionKind={dimension!r}")


# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RegimeBucketResult:
    label: str
    observation_count: int
    trade_count: int
    skipped: bool
    skip_reason: str | None
    total_gross_return: float | None
    total_net_return: float | None
    sharpe: float | None
    maximum_drawdown: float | None
    hit_rate: float | None
    mean_exposure: float | None
    total_transaction_costs: float | None
    benchmark_total_net_return: float | None
    outperforms_benchmark: bool | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "label": self.label, "observation_count": self.observation_count, "trade_count": self.trade_count, "skipped": self.skipped,
            "skip_reason": self.skip_reason, "total_gross_return": self.total_gross_return, "total_net_return": self.total_net_return,
            "sharpe": self.sharpe, "maximum_drawdown": self.maximum_drawdown, "hit_rate": self.hit_rate, "mean_exposure": self.mean_exposure,
            "total_transaction_costs": self.total_transaction_costs, "benchmark_total_net_return": self.benchmark_total_net_return,
            "outperforms_benchmark": self.outperforms_benchmark,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RegimeBucketResult:
        def _opt_float(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            label=str(raw["label"]), observation_count=int(str(raw["observation_count"])), trade_count=int(str(raw["trade_count"])),
            skipped=bool(raw["skipped"]), skip_reason=(None if raw.get("skip_reason") is None else str(raw["skip_reason"])),
            total_gross_return=_opt_float("total_gross_return"), total_net_return=_opt_float("total_net_return"), sharpe=_opt_float("sharpe"),
            maximum_drawdown=_opt_float("maximum_drawdown"), hit_rate=_opt_float("hit_rate"), mean_exposure=_opt_float("mean_exposure"),
            total_transaction_costs=_opt_float("total_transaction_costs"), benchmark_total_net_return=_opt_float("benchmark_total_net_return"),
            outperforms_benchmark=(None if raw.get("outperforms_benchmark") is None else bool(raw["outperforms_benchmark"])),
        )


@dataclass(frozen=True, slots=True)
class RegimeDimensionResult:
    dimension: RegimeDimensionKind
    unavailable: bool
    unavailable_reason: str | None
    excluded_insufficient_history_count: int
    buckets: tuple[RegimeBucketResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension.value, "unavailable": self.unavailable, "unavailable_reason": self.unavailable_reason,
            "excluded_insufficient_history_count": self.excluded_insufficient_history_count, "buckets": [b.to_json_dict() for b in self.buckets],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RegimeDimensionResult:
        return cls(
            dimension=RegimeDimensionKind(raw["dimension"]), unavailable=bool(raw["unavailable"]),
            unavailable_reason=(None if raw.get("unavailable_reason") is None else str(raw["unavailable_reason"])),
            excluded_insufficient_history_count=int(str(raw["excluded_insufficient_history_count"])),
            buckets=tuple(RegimeBucketResult.from_json_dict(as_json_dict(b, field_name="buckets[]")) for b in as_json_list(raw.get("buckets") or [], field_name="buckets")),
        )


@dataclass(frozen=True, slots=True)
class RegimeReport:
    schema_version: int
    source_backtest_id: str
    total_bar_observations: int
    dimension_results: tuple[RegimeDimensionResult, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_backtest_id": self.source_backtest_id, "total_bar_observations": self.total_bar_observations,
            "dimension_results": [d.to_json_dict() for d in self.dimension_results], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RegimeReport:
        require_schema_version(raw, supported=REGIME_REPORT_SCHEMA_VERSION, context="RegimeReport")
        return cls(
            schema_version=REGIME_REPORT_SCHEMA_VERSION, source_backtest_id=str(raw["source_backtest_id"]),
            total_bar_observations=int(str(raw["total_bar_observations"])),
            dimension_results=tuple(
                RegimeDimensionResult.from_json_dict(as_json_dict(d, field_name="dimension_results[]")) for d in as_json_list(raw.get("dimension_results") or [], field_name="dimension_results")
            ),
            generated_at=str(raw["generated_at"]),
        )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _compute_bucket(
    label: str, bar_positions: list[int], *, bar_records: dict[int, _BarRecord], trades_by_entry_position: dict[int, list[TradeRecord]],
    periods_per_year: float, minimum_regime_samples: int,
) -> RegimeBucketResult:
    observation_count = len(bar_positions)
    trades = [t for p in bar_positions for t in trades_by_entry_position.get(p, [])]
    if observation_count < minimum_regime_samples:
        return RegimeBucketResult(
            label=label, observation_count=observation_count, trade_count=len(trades), skipped=True,
            skip_reason=f"insufficient samples: {observation_count} observation(s) < minimum_regime_samples={minimum_regime_samples}",
            total_gross_return=None, total_net_return=None, sharpe=None, maximum_drawdown=None, hit_rate=None, mean_exposure=None,
            total_transaction_costs=None, benchmark_total_net_return=None, outperforms_benchmark=None,
        )

    ordered = sorted(bar_positions)
    net_returns = [bar_records[p].net_return for p in ordered]
    gross_returns = [bar_records[p].gross_return for p in ordered]
    sharpe, _reason, _assumptions = sharpe_like(net_returns, annual_risk_free_rate=0.0, periods_per_year=periods_per_year, downside_only=False)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for r in net_returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak > 0.0 else 0.0)

    hit_rate = (sum(1 for t in trades if t.net_return > 0.0) / len(trades)) if trades else None
    total_net = sum(net_returns)
    benchmark_total = sum(bar_records[p].benchmark_bar_return for p in ordered)
    return RegimeBucketResult(
        label=label, observation_count=observation_count, trade_count=len(trades), skipped=False, skip_reason=None,
        total_gross_return=sum(gross_returns), total_net_return=total_net, sharpe=sharpe, maximum_drawdown=max_drawdown, hit_rate=hit_rate,
        mean_exposure=statistics.fmean(bar_records[p].total_absolute_exposure for p in ordered),
        total_transaction_costs=sum(bar_records[p].transaction_costs for p in ordered), benchmark_total_net_return=benchmark_total,
        outperforms_benchmark=total_net > benchmark_total,
    )


def _compute_dimension_result(
    regime_def: RegimeDefinitionSpec, *, bar_positions: list[int], bars: pd.DataFrame, bar_records: dict[int, _BarRecord],
    trades_by_entry_position: dict[int, list[TradeRecord]], periods_per_year: float,
) -> RegimeDimensionResult:
    label_by_position, unavailable_reason = _classify_dimension(regime_def, bar_positions=bar_positions, bars=bars)
    if label_by_position is None:
        return RegimeDimensionResult(dimension=regime_def.dimension, unavailable=True, unavailable_reason=unavailable_reason, excluded_insufficient_history_count=0, buckets=())

    excluded_count = len(bar_positions) - len(label_by_position)
    positions_by_label: dict[str, list[int]] = {}
    for pos, label in label_by_position.items():
        positions_by_label.setdefault(label, []).append(pos)

    buckets = tuple(
        _compute_bucket(
            label, positions, bar_records=bar_records, trades_by_entry_position=trades_by_entry_position,
            periods_per_year=periods_per_year, minimum_regime_samples=regime_def.minimum_regime_samples,
        )
        for label, positions in sorted(positions_by_label.items())
    )
    return RegimeDimensionResult(dimension=regime_def.dimension, unavailable=False, unavailable_reason=None, excluded_insufficient_history_count=excluded_count, buckets=buckets)


def compute_regime_report(
    *, spec: RobustnessSpec, bar_interval: Timeframe, bar_timelines: tuple[BarReturnTimeline, ...], bars: pd.DataFrame, all_closed_trades: tuple[TradeRecord, ...],
) -> RegimeReport:
    """Section 11's entry point. `bar_timelines` are every outer fold's
    already-persisted `BarReturnTimeline` (pooled here, deduplicated by
    `bar_position` -- overlapping outer test folds are not expected in
    this platform's walk-forward split plans, but the first occurrence
    wins if one is ever encountered rather than silently double-counting).
    `bars` is the FULL underlying OHLCV series `outer_folds` are positions
    into -- trailing windows and expanding quantile ranks may legitimately
    reach back before a given fold's own test start, since that is still
    strictly past information relative to the bar being classified."""
    if not bar_timelines:
        raise RegimeAnalysisError("compute_regime_report: bar_timelines must not be empty", context={"source_backtest_id": spec.source_backtest_id})
    bar_records = _build_bar_records(bar_timelines, bars=bars)
    if not bar_records:
        raise RegimeAnalysisError("compute_regime_report: no bar observations found across bar_timelines", context={"source_backtest_id": spec.source_backtest_id})

    trades_by_entry_position: dict[int, list[TradeRecord]] = {}
    for trade in all_closed_trades:
        trades_by_entry_position.setdefault(trade.entry_bar_position, []).append(trade)

    bar_positions = list(bar_records.keys())
    periods_per_year = bars_per_year(bar_interval)
    dimension_results = tuple(
        _compute_dimension_result(
            regime_def, bar_positions=bar_positions, bars=bars, bar_records=bar_records, trades_by_entry_position=trades_by_entry_position,
            periods_per_year=periods_per_year,
        )
        for regime_def in spec.regime_definitions
    )
    return RegimeReport(
        schema_version=REGIME_REPORT_SCHEMA_VERSION, source_backtest_id=spec.source_backtest_id, total_bar_observations=len(bar_records),
        dimension_results=dimension_results, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "REGIME_REPORT_SCHEMA_VERSION",
    "RegimeBucketResult",
    "RegimeDimensionResult",
    "RegimeReport",
    "compute_regime_report",
]
