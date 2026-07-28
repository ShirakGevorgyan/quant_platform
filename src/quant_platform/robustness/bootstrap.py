"""Dependence-aware bootstrap (Milestone 6, Section 5) and downside/
loss-probability analysis (Section 6).

FOUR METHODS, ONE HONEST CAVEAT PER METHOD
--------------------------------------------------------------------------
- IID: resamples individual observations with replacement, independently.
  EXPLICITLY WEAK for autocorrelated financial time series (it destroys
  any serial dependence) -- implemented for comparison/completeness only,
  never this package's recommended default.
- MOVING_BLOCK (Kunsch 1989): resamples fixed-length, OVERLAPPING blocks
  with replacement and concatenates them -- preserves local dependence
  structure up to `block_length`, at the cost of a somewhat "blocky"
  resample (block boundaries are seams).
- STATIONARY (Politis & Romano 1994): like moving-block, but each block's
  length is itself random (geometric, mean `block_length`), which makes
  the resampled series itself stationary (removes the block-boundary
  seam artifact moving-block has) at the cost of a slightly noisier
  effective block length.
- FOLD_LEVEL: resamples WHOLE outer folds with replacement, preserving
  each selected fold's own internal bar/trade order intact -- the
  coarsest, most conservative dependence structure (appropriate ONLY
  when `ReturnSeriesBundle.fold_boundaries` is populated, i.e. the series
  actually has fold structure; single continuous stitched series still
  qualify since fold boundaries are carried through from `series.py`).

UNDEFINED REPETITIONS ARE RECORDED, NEVER SILENTLY NaN'D
--------------------------------------------------------------------------
A resample can legitimately fail to define a statistic (e.g. Sharpe on a
zero-variance resample, profit_factor on a resample with zero losses).
Such repetitions are counted and reasoned about in `BootstrapEstimate.
skipped_repetitions`/`failure_reasons`, and EXCLUDED from the point
estimate/CI computation -- never coerced to `0.0`/`NaN`/`inf`. If EVERY
repetition fails, `BootstrapError` is raised (no CI can be reported)."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from quant_platform.core.exceptions import BootstrapError
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import BootstrapMethodKind
from quant_platform.robustness.series import FoldBoundary, ReturnSeriesBundle
from quant_platform.robustness.specs import BootstrapSpec

BOOTSTRAP_REPORT_SCHEMA_VERSION = 1

StatisticFn = Callable[[Sequence[float]], float | None]


# --------------------------------------------------------------------------
# Resampling (deterministic given `seed`)
# --------------------------------------------------------------------------
def _iid_resample(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    return values[rng.integers(0, n, size=n)]


def _moving_block_resample(values: np.ndarray, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    block_length = min(block_length, n)
    n_blocks_available = n - block_length + 1
    out: list[float] = []
    while len(out) < n:
        start = int(rng.integers(0, n_blocks_available))
        out.extend(values[start : start + block_length].tolist())
    return np.asarray(out[:n])


def _stationary_resample(values: np.ndarray, *, mean_block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994): EACH block starts at a FRESH random
    position (the defect this fixed: an earlier version only randomized
    the very first block's start and then let `pos` drift forward
    contiguously forever, degenerating into a single circular rotation of
    the whole series -- order-INDEPENDENT statistics like `total_return`
    are then numerically identical for every "resample", producing a
    collapsed, near-zero-width CI that looks plausible but is silently
    wrong. Caught via a direct smoke test comparing this method's CI
    width against `moving_block`'s on the same series, not by reasoning
    alone.)"""
    n = len(values)
    p = 1.0 / max(1, mean_block_length)
    out: list[float] = []
    while len(out) < n:
        pos = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        for _ in range(length):
            out.append(float(values[pos % n]))
            pos += 1
            if len(out) >= n:
                break
    return np.asarray(out[:n])


def _fold_level_resample(values: np.ndarray, *, fold_boundaries: tuple[FoldBoundary, ...], rng: np.random.Generator) -> np.ndarray:
    if not fold_boundaries:
        raise BootstrapError("fold_level bootstrap requires a non-empty ReturnSeriesBundle.fold_boundaries")
    n_folds = len(fold_boundaries)
    chosen = rng.integers(0, n_folds, size=n_folds)
    out: list[float] = []
    for idx in chosen:
        b = fold_boundaries[int(idx)]
        out.extend(values[b.start_index : b.end_index + 1].tolist())
    return np.asarray(out)


def _resample_once(values: np.ndarray, *, spec: BootstrapSpec, fold_boundaries: tuple[FoldBoundary, ...], rng: np.random.Generator) -> np.ndarray:
    if spec.method is BootstrapMethodKind.IID:
        return _iid_resample(values, rng=rng)
    if spec.method is BootstrapMethodKind.MOVING_BLOCK:
        assert spec.block_length is not None
        return _moving_block_resample(values, block_length=spec.block_length, rng=rng)
    if spec.method is BootstrapMethodKind.STATIONARY:
        assert spec.block_length is not None
        return _stationary_resample(values, mean_block_length=spec.block_length, rng=rng)
    if spec.method is BootstrapMethodKind.FOLD_LEVEL:
        return _fold_level_resample(values, fold_boundaries=fold_boundaries, rng=rng)
    raise BootstrapError(f"Unsupported BootstrapMethodKind: {spec.method!r}")  # pragma: no cover - exhaustive enum


# --------------------------------------------------------------------------
# Named statistic functions -- each takes a (resampled) array of PERIOD
# returns and returns a scalar, or None if undefined for that array.
# `periods_per_year=None` disables annualization (Sharpe/Sortino are then
# reported in native per-period units).
# --------------------------------------------------------------------------
def make_total_return_statistic() -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        product = 1.0
        for v in values:
            product *= 1.0 + v
        return product - 1.0
    return _fn


def make_mean_return_statistic() -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        return statistics.fmean(values) if values else None
    return _fn


def make_sharpe_statistic(*, periods_per_year: float | None, annual_risk_free_rate: float = 0.0) -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        if len(values) < 2:
            return None
        stdev = statistics.pstdev(values)
        if stdev == 0.0:
            return None
        per_period_rf = 0.0
        if periods_per_year and periods_per_year > 0:
            per_period_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        mean_excess = statistics.fmean(v - per_period_rf for v in values)
        ratio = mean_excess / stdev
        if periods_per_year and periods_per_year > 0:
            ratio *= math.sqrt(periods_per_year)
        return ratio if math.isfinite(ratio) else None
    return _fn


def make_sortino_statistic(*, periods_per_year: float | None, annual_risk_free_rate: float = 0.0) -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        if len(values) < 2:
            return None
        per_period_rf = 0.0
        if periods_per_year and periods_per_year > 0:
            per_period_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        excess = [v - per_period_rf for v in values]
        downside = [min(0.0, e) for e in excess]
        downside_dev = math.sqrt(statistics.fmean(d * d for d in downside))
        if downside_dev == 0.0:
            return None
        ratio = statistics.fmean(excess) / downside_dev
        if periods_per_year and periods_per_year > 0:
            ratio *= math.sqrt(periods_per_year)
        return ratio if math.isfinite(ratio) else None
    return _fn


def make_maximum_drawdown_statistic() -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        if not values:
            return None
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for v in values:
            equity *= 1.0 + v
            if equity <= 0.0:
                return None
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        return max_dd
    return _fn


def make_hit_rate_statistic() -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        return (sum(1 for v in values if v > 0.0) / len(values)) if values else None
    return _fn


def make_profit_factor_statistic() -> StatisticFn:
    def _fn(values: Sequence[float]) -> float | None:
        gross_profit = sum(v for v in values if v > 0.0)
        gross_loss = abs(sum(v for v in values if v < 0.0))
        return (gross_profit / gross_loss) if gross_loss > 0.0 else None
    return _fn


def make_expectancy_statistic() -> StatisticFn:
    return make_mean_return_statistic()


STANDARD_STATISTICS: dict[str, StatisticFn] = {
    "total_return": make_total_return_statistic(),
    "mean_return": make_mean_return_statistic(),
    "maximum_drawdown": make_maximum_drawdown_statistic(),
    "hit_rate": make_hit_rate_statistic(),
    "profit_factor": make_profit_factor_statistic(),
    "expectancy": make_expectancy_statistic(),
}


# --------------------------------------------------------------------------
# Persisted result
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    statistic_name: str
    point_estimate: float | None
    lower_bound: float | None
    upper_bound: float | None
    valid_repetitions: int
    skipped_repetitions: int
    failure_reasons: dict[str, int]

    def __post_init__(self) -> None:
        if self.valid_repetitions < 0 or self.skipped_repetitions < 0:
            raise BootstrapError(f"BootstrapEstimate[{self.statistic_name}]: repetition counts must be >= 0")
        for name, value in (("point_estimate", self.point_estimate), ("lower_bound", self.lower_bound), ("upper_bound", self.upper_bound)):
            if value is not None and not math.isfinite(value):
                raise BootstrapError(f"BootstrapEstimate[{self.statistic_name}].{name} must be finite or None, got {value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "statistic_name": self.statistic_name, "point_estimate": self.point_estimate, "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound, "valid_repetitions": self.valid_repetitions, "skipped_repetitions": self.skipped_repetitions,
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BootstrapEstimate:
        return cls(
            statistic_name=str(raw["statistic_name"]), point_estimate=(None if raw.get("point_estimate") is None else float(str(raw["point_estimate"]))),
            lower_bound=(None if raw.get("lower_bound") is None else float(str(raw["lower_bound"]))),
            upper_bound=(None if raw.get("upper_bound") is None else float(str(raw["upper_bound"]))),
            valid_repetitions=int(str(raw["valid_repetitions"])), skipped_repetitions=int(str(raw["skipped_repetitions"])),
            failure_reasons={str(k): int(str(v)) for k, v in as_json_dict(raw.get("failure_reasons") or {}, field_name="failure_reasons").items()},
        )


def bootstrap_statistic(
    bundle: ReturnSeriesBundle, *, statistic_name: str, statistic_fn: StatisticFn, spec: BootstrapSpec, seed: int,
) -> BootstrapEstimate:
    """Section 5's core procedure for ONE statistic on ONE already-built
    return series: resample `spec.repetitions` times (deterministically,
    from `seed`), evaluate `statistic_fn` on each resample, and report the
    point estimate (on the ORIGINAL, unresampled series) plus the
    `spec.confidence_level` percentile interval of the valid resample
    results. Fails closed (`BootstrapError`) if every repetition is
    undefined for this statistic."""
    values = np.asarray(bundle.values, dtype=float)
    point_estimate = statistic_fn(bundle.values)
    rng = np.random.default_rng(seed)
    results: list[float] = []
    failure_reasons: Counter[str] = Counter()
    for _ in range(spec.repetitions):
        resample = _resample_once(values, spec=spec, fold_boundaries=bundle.fold_boundaries, rng=rng)
        try:
            value = statistic_fn(resample.tolist())
        except (ArithmeticError, ValueError) as exc:
            failure_reasons[type(exc).__name__] += 1
            continue
        if value is None or not math.isfinite(value):
            failure_reasons["undefined_for_resample"] += 1
            continue
        results.append(value)

    if not results:
        raise BootstrapError(
            f"bootstrap_statistic[{statistic_name}]: every one of {spec.repetitions} repetitions failed to define "
            f"this statistic -- no confidence interval can be reported (reasons: {dict(failure_reasons)})"
        )
    alpha = 1.0 - spec.confidence_level
    lower = float(np.percentile(results, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(results, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapEstimate(
        statistic_name=statistic_name, point_estimate=point_estimate, lower_bound=lower, upper_bound=upper,
        valid_repetitions=len(results), skipped_repetitions=spec.repetitions - len(results), failure_reasons=dict(failure_reasons),
    )


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    schema_version: int
    series_kind: str
    method: str
    repetitions: int
    confidence_level: float
    seed: int
    estimates: tuple[BootstrapEstimate, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "series_kind": self.series_kind, "method": self.method,
            "repetitions": self.repetitions, "confidence_level": self.confidence_level, "seed": self.seed,
            "estimates": [e.to_json_dict() for e in self.estimates], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BootstrapReport:
        require_schema_version(raw, supported=BOOTSTRAP_REPORT_SCHEMA_VERSION, context="BootstrapReport")
        return cls(
            schema_version=BOOTSTRAP_REPORT_SCHEMA_VERSION, series_kind=str(raw["series_kind"]), method=str(raw["method"]),
            repetitions=int(str(raw["repetitions"])), confidence_level=float(str(raw["confidence_level"])), seed=int(str(raw["seed"])),
            estimates=tuple(BootstrapEstimate.from_json_dict(as_json_dict(e, field_name="estimates[]")) for e in as_json_list(raw["estimates"], field_name="estimates")),
            generated_at=str(raw["generated_at"]),
        )


def compute_bootstrap_report(
    bundle: ReturnSeriesBundle, *, spec: BootstrapSpec, seed: int, statistics_by_name: dict[str, StatisticFn] | None = None,
) -> BootstrapReport:
    """Bootstraps every requested named statistic (default: `STANDARD_
    STATISTICS`) for ONE return series. A statistic that fails closed for
    every repetition aborts the WHOLE report (`BootstrapError`) -- never
    silently omitted, since that would misrepresent what was actually
    attempted."""
    fns = statistics_by_name or STANDARD_STATISTICS
    estimates = tuple(
        bootstrap_statistic(bundle, statistic_name=name, statistic_fn=fn, spec=spec, seed=seed + i)
        for i, (name, fn) in enumerate(sorted(fns.items()))
    )
    return BootstrapReport(
        schema_version=BOOTSTRAP_REPORT_SCHEMA_VERSION, series_kind=bundle.kind.value, method=spec.method.value,
        repetitions=spec.repetitions, confidence_level=spec.confidence_level, seed=seed, estimates=estimates,
        generated_at=format_utc_timestamp(utc_now()),
    )


# --------------------------------------------------------------------------
# Section 6: downside / loss-probability analysis
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DownsideAnalysisReport:
    """Section 6: every probability here is a RESAMPLING-BASED ESTIMATE
    under the observed sample -- never described or persisted as a
    guaranteed future probability."""

    schema_version: int
    probability_total_net_return_non_positive: float
    probability_mean_return_non_positive: float
    probability_sharpe_non_positive: float
    probability_underperforms_always_flat: float | None
    probability_underperforms_always_long: float | None
    probability_cost_stressed_unprofitable: float | None
    probability_maximum_drawdown_exceeds_limit: float | None
    drawdown_limit: float | None
    repetitions: int
    seed: int
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "probability_total_net_return_non_positive": self.probability_total_net_return_non_positive,
            "probability_mean_return_non_positive": self.probability_mean_return_non_positive,
            "probability_sharpe_non_positive": self.probability_sharpe_non_positive,
            "probability_underperforms_always_flat": self.probability_underperforms_always_flat,
            "probability_underperforms_always_long": self.probability_underperforms_always_long,
            "probability_cost_stressed_unprofitable": self.probability_cost_stressed_unprofitable,
            "probability_maximum_drawdown_exceeds_limit": self.probability_maximum_drawdown_exceeds_limit,
            "drawdown_limit": self.drawdown_limit, "repetitions": self.repetitions, "seed": self.seed, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DownsideAnalysisReport:
        require_schema_version(raw, supported=1, context="DownsideAnalysisReport")

        def _opt(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            schema_version=1, probability_total_net_return_non_positive=float(str(raw["probability_total_net_return_non_positive"])),
            probability_mean_return_non_positive=float(str(raw["probability_mean_return_non_positive"])),
            probability_sharpe_non_positive=float(str(raw["probability_sharpe_non_positive"])),
            probability_underperforms_always_flat=_opt("probability_underperforms_always_flat"),
            probability_underperforms_always_long=_opt("probability_underperforms_always_long"),
            probability_cost_stressed_unprofitable=_opt("probability_cost_stressed_unprofitable"),
            probability_maximum_drawdown_exceeds_limit=_opt("probability_maximum_drawdown_exceeds_limit"),
            drawdown_limit=_opt("drawdown_limit"), repetitions=int(str(raw["repetitions"])), seed=int(str(raw["seed"])), generated_at=str(raw["generated_at"]),
        )


def _resampled_statistic_distribution(bundle: ReturnSeriesBundle, *, statistic_fn: StatisticFn, spec: BootstrapSpec, seed: int) -> list[float]:
    values = np.asarray(bundle.values, dtype=float)
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(spec.repetitions):
        resample = _resample_once(values, spec=spec, fold_boundaries=bundle.fold_boundaries, rng=rng)
        v = statistic_fn(resample.tolist())
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


def compute_downside_analysis(
    bundle: ReturnSeriesBundle, *, spec: BootstrapSpec, seed: int, benchmark_relative_to_flat: ReturnSeriesBundle | None = None,
    benchmark_relative_to_long: ReturnSeriesBundle | None = None, cost_stressed_bundle: ReturnSeriesBundle | None = None,
    drawdown_limit: float | None = None,
) -> DownsideAnalysisReport:
    """Section 6. Every "probability underperforms X" figure requires the
    CALLER to have already built the corresponding `BENCHMARK_RELATIVE`
    (or cost-stressed) series -- this function never fabricates one; the
    corresponding probability is reported as `None` (not `0.0`, not
    `NaN`) when the input was not supplied."""
    total_return_fn = make_total_return_statistic()
    mean_return_fn = make_mean_return_statistic()
    sharpe_fn = make_sharpe_statistic(periods_per_year=None)
    max_dd_fn = make_maximum_drawdown_statistic()

    total_returns = _resampled_statistic_distribution(bundle, statistic_fn=total_return_fn, spec=spec, seed=seed)
    mean_returns = _resampled_statistic_distribution(bundle, statistic_fn=mean_return_fn, spec=spec, seed=seed + 1)
    sharpes = _resampled_statistic_distribution(bundle, statistic_fn=sharpe_fn, spec=spec, seed=seed + 2)
    if not total_returns or not mean_returns or not sharpes:
        raise BootstrapError("compute_downside_analysis: bootstrap resampling produced no valid repetitions for one or more base statistics")

    prob_flat = None
    if benchmark_relative_to_flat is not None:
        rel = _resampled_statistic_distribution(benchmark_relative_to_flat, statistic_fn=total_return_fn, spec=spec, seed=seed + 3)
        if rel:
            prob_flat = sum(1 for v in rel if v <= 0.0) / len(rel)
    prob_long = None
    if benchmark_relative_to_long is not None:
        rel = _resampled_statistic_distribution(benchmark_relative_to_long, statistic_fn=total_return_fn, spec=spec, seed=seed + 4)
        if rel:
            prob_long = sum(1 for v in rel if v <= 0.0) / len(rel)
    prob_cost_stressed = None
    if cost_stressed_bundle is not None:
        stressed = _resampled_statistic_distribution(cost_stressed_bundle, statistic_fn=total_return_fn, spec=spec, seed=seed + 5)
        if stressed:
            prob_cost_stressed = sum(1 for v in stressed if v <= 0.0) / len(stressed)
    prob_dd_exceeds = None
    if drawdown_limit is not None:
        dds = _resampled_statistic_distribution(bundle, statistic_fn=max_dd_fn, spec=spec, seed=seed + 6)
        if dds:
            prob_dd_exceeds = sum(1 for v in dds if v > drawdown_limit) / len(dds)

    return DownsideAnalysisReport(
        schema_version=1, probability_total_net_return_non_positive=sum(1 for v in total_returns if v <= 0.0) / len(total_returns),
        probability_mean_return_non_positive=sum(1 for v in mean_returns if v <= 0.0) / len(mean_returns),
        probability_sharpe_non_positive=sum(1 for v in sharpes if v <= 0.0) / len(sharpes),
        probability_underperforms_always_flat=prob_flat, probability_underperforms_always_long=prob_long,
        probability_cost_stressed_unprofitable=prob_cost_stressed, probability_maximum_drawdown_exceeds_limit=prob_dd_exceeds,
        drawdown_limit=drawdown_limit, repetitions=spec.repetitions, seed=seed, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "BOOTSTRAP_REPORT_SCHEMA_VERSION",
    "STANDARD_STATISTICS",
    "BootstrapEstimate",
    "BootstrapReport",
    "DownsideAnalysisReport",
    "StatisticFn",
    "bootstrap_statistic",
    "compute_bootstrap_report",
    "compute_downside_analysis",
    "make_expectancy_statistic",
    "make_hit_rate_statistic",
    "make_maximum_drawdown_statistic",
    "make_mean_return_statistic",
    "make_profit_factor_statistic",
    "make_sharpe_statistic",
    "make_sortino_statistic",
    "make_total_return_statistic",
]
