"""Return-series contract (Milestone 6, Section 4). Every downstream
statistical procedure (bootstrap, stability, sensitivity, stress, regime
analysis) consumes a `ReturnSeriesBundle` -- never a raw list of floats
picked up ad hoc from wherever looked convenient. This is the ONE place
bar returns, trade returns, fold summaries, and event-sampled returns are
told apart; nothing downstream may silently mix them.

EFFECTIVE SAMPLE SIZE
--------------------------------------------------------------------------
Bar-level return series are typically autocorrelated (a position held
across many bars repeats a large share of the SAME underlying price move
across consecutive observations), so the RAW observation count overstates
how much independent information the series actually carries. This
module reports an `effective_sample_count` via the standard AR(1)
approximation `n_eff = n * (1 - rho_1) / (1 + rho_1)`, where `rho_1` is
the series' own lag-1 sample autocorrelation, clamped to `[1, n]`. This
is a coarse, well-known, and honestly-documented approximation -- real
market dependence is not exactly AR(1) -- never a claim of statistical
precision beyond that. Trade-level and fold-level series (already
one-observation-per-independent-event, by construction never
overlapping) report `effective_sample_count == observation_count`
unadjusted.

`BAR_NET` VERSUS `PER_FOLD_BAR_NET` (closure-audit finding, Section 4)
--------------------------------------------------------------------------
These two kinds are BOTH built by `_build_per_fold_bar_series` and
currently produce BYTE-IDENTICAL `values`/`fold_boundaries`/
`effective_sample_count` for the same source -- confirmed intentional,
not a bug: `fold_boundaries` is already carried on every bar-sampled
kind (including plain `BAR_NET`), so there is no additional information
`PER_FOLD_BAR_NET` currently conveys that `BAR_NET` lacks. The two
labels exist so a caller/report can SIGNAL fold-partition-aware intent
(e.g. "this analysis cares about fold structure") via `ReturnSeriesBundle
.kind` without requiring separate construction logic. No downstream
consumer currently branches on this distinction. See
`tests/unit/robustness/test_series.py::TestPerFoldBarNetIsIntentionally
IdenticalToBarNet` for the tripwire test that must be deliberately
updated if a future feature needs them to diverge."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from quant_platform.backtesting.runner import BenchmarkReport, OuterFoldBacktestResult
from quant_platform.backtesting.stitching import StitchedWalkForwardEquity
from quant_platform.backtesting.timeline import BarReturnTimeline
from quant_platform.backtesting.trades import TradeSet
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ReturnSeriesError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.persistence import (
    as_json_list,
    format_utc_timestamp,
    parse_json_strict,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import ReturnSeriesKind
from quant_platform.robustness.source import VerifiedBacktestSource

RETURN_SERIES_BUNDLE_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_NAME = "always_flat_net_cost"

_UNVERIFIABLE_SERIES_ERRORS: tuple[type[Exception], ...] = (ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError)


@dataclass(frozen=True, slots=True)
class FoldBoundary:
    outer_fold_index: int
    start_index: int
    end_index: int
    """Inclusive `[start_index, end_index]` positions within `values`."""

    def to_json_dict(self) -> dict[str, object]:
        return {"outer_fold_index": self.outer_fold_index, "start_index": self.start_index, "end_index": self.end_index}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FoldBoundary:
        return cls(outer_fold_index=int(str(raw["outer_fold_index"])), start_index=int(str(raw["start_index"])), end_index=int(str(raw["end_index"])))


def _lag1_autocorrelation(values: tuple[float, ...]) -> float | None:
    if len(values) < 3:
        return None
    x, y = values[:-1], values[1:]
    if statistics.pstdev(x) == 0.0 or statistics.pstdev(y) == 0.0:
        return 0.0
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / len(x)
    denom = statistics.pstdev(x) * statistics.pstdev(y)
    return cov / denom if denom > 0 else 0.0


def _effective_sample_count(values: tuple[float, ...], *, adjust_for_autocorrelation: bool) -> int:
    n = len(values)
    if n == 0:
        return 0
    if not adjust_for_autocorrelation:
        return n
    rho1 = _lag1_autocorrelation(values)
    if rho1 is None or not math.isfinite(rho1) or rho1 <= -0.999 or rho1 >= 0.999:
        return n
    n_eff = n * (1.0 - rho1) / (1.0 + rho1)
    return max(1, min(n, round(n_eff)))


@dataclass(frozen=True, slots=True)
class ReturnSeriesBundle:
    schema_version: int
    kind: ReturnSeriesKind
    sampling_frequency: str
    observation_count: int
    effective_sample_count: int
    values: tuple[float, ...]
    fold_boundaries: tuple[FoldBoundary, ...]
    source_artifact_content_hashes: tuple[str, ...]
    time_range_start: str | None
    time_range_end: str | None
    built_at: str

    def __post_init__(self) -> None:
        if self.observation_count != len(self.values):
            raise ReturnSeriesError(f"ReturnSeriesBundle.observation_count ({self.observation_count}) must equal len(values) ({len(self.values)})")
        if not (1 <= self.effective_sample_count <= max(1, self.observation_count)):
            raise ReturnSeriesError(f"ReturnSeriesBundle.effective_sample_count ({self.effective_sample_count}) must be in [1, observation_count] for a non-empty series")
        for v in self.values:
            if not math.isfinite(v):
                raise ReturnSeriesError(f"ReturnSeriesBundle.values must all be finite, got {v!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "kind": self.kind.value, "sampling_frequency": self.sampling_frequency,
            "observation_count": self.observation_count, "effective_sample_count": self.effective_sample_count,
            "values": list(self.values), "fold_boundaries": [b.to_json_dict() for b in self.fold_boundaries],
            "source_artifact_content_hashes": list(self.source_artifact_content_hashes),
            "time_range_start": self.time_range_start, "time_range_end": self.time_range_end, "built_at": self.built_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReturnSeriesBundle:
        require_schema_version(raw, supported=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, context="ReturnSeriesBundle")
        return cls(
            schema_version=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, kind=ReturnSeriesKind(raw["kind"]), sampling_frequency=str(raw["sampling_frequency"]),
            observation_count=int(str(raw["observation_count"])), effective_sample_count=int(str(raw["effective_sample_count"])),
            values=tuple(float(str(v)) for v in as_json_list(raw["values"], field_name="values")),
            fold_boundaries=tuple(FoldBoundary.from_json_dict(b) for b in as_json_list(raw.get("fold_boundaries") or [], field_name="fold_boundaries")),
            source_artifact_content_hashes=tuple(str(h) for h in as_json_list(raw.get("source_artifact_content_hashes") or [], field_name="source_artifact_content_hashes")),
            time_range_start=(None if raw.get("time_range_start") is None else str(raw["time_range_start"])),
            time_range_end=(None if raw.get("time_range_end") is None else str(raw["time_range_end"])),
            built_at=str(raw["built_at"]),
        )


def _decode_fold_result(manifest_ref_content_hash: str, *, artifact_store: MLArtifactStore) -> OuterFoldBacktestResult:
    raw = artifact_store.read_artifact(manifest_ref_content_hash)
    return OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))


def build_return_series(
    kind: ReturnSeriesKind, *, source: VerifiedBacktestSource, artifact_store: MLArtifactStore, benchmark_name: str = DEFAULT_BENCHMARK_NAME,
) -> ReturnSeriesBundle:
    """Section 4's dispatcher: builds exactly ONE declared series kind
    from the already-verified source backtest's own persisted artifacts.
    Never falls back to a different kind, never blends two kinds
    together."""
    try:
        if kind in (ReturnSeriesKind.BAR_NET, ReturnSeriesKind.BAR_GROSS, ReturnSeriesKind.PER_FOLD_BAR_NET):
            return _build_per_fold_bar_series(kind, source=source, artifact_store=artifact_store)
        if kind in (ReturnSeriesKind.TRADE_NET, ReturnSeriesKind.TRADE_GROSS):
            return _build_trade_series(kind, source=source, artifact_store=artifact_store)
        if kind is ReturnSeriesKind.STITCHED_BAR_NET:
            return _build_stitched_series(source=source, artifact_store=artifact_store)
        if kind is ReturnSeriesKind.BENCHMARK_RELATIVE:
            return _build_benchmark_relative_series(source=source, artifact_store=artifact_store, benchmark_name=benchmark_name)
        raise ReturnSeriesError(f"Unsupported ReturnSeriesKind: {kind!r}")  # pragma: no cover - exhaustive enum
    except _UNVERIFIABLE_SERIES_ERRORS as exc:
        raise ReturnSeriesError(f"build_return_series: could not build kind={kind.value!r} from source_backtest_id={source.manifest.backtest_id!r}: {exc}") from exc


def _build_per_fold_bar_series(kind: ReturnSeriesKind, *, source: VerifiedBacktestSource, artifact_store: MLArtifactStore) -> ReturnSeriesBundle:
    values: list[float] = []
    boundaries: list[FoldBoundary] = []
    hashes: list[str] = []
    timestamps: list[str] = []
    for outer_fold_index in sorted(source.manifest.outer_fold_result_references):
        ref = source.manifest.outer_fold_result_references[outer_fold_index]
        result = _decode_fold_result(ref.content_hash, artifact_store=artifact_store)
        raw = artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash)
        timeline = BarReturnTimeline.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        start = len(values)
        for p in timeline.points:
            values.append(p.gross_return if kind is ReturnSeriesKind.BAR_GROSS else p.net_return)
            timestamps.append(p.timestamp)
        boundaries.append(FoldBoundary(outer_fold_index=outer_fold_index, start_index=start, end_index=len(values) - 1))
        hashes.append(result.bar_return_timeline_reference.content_hash)
    if not values:
        raise ReturnSeriesError("build_return_series: source backtest has no bar-return observations across any outer fold")
    return ReturnSeriesBundle(
        schema_version=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, kind=kind, sampling_frequency="bar", observation_count=len(values),
        effective_sample_count=_effective_sample_count(tuple(values), adjust_for_autocorrelation=True), values=tuple(values),
        fold_boundaries=tuple(boundaries), source_artifact_content_hashes=tuple(hashes),
        time_range_start=timestamps[0], time_range_end=timestamps[-1], built_at=format_utc_timestamp(utc_now()),
    )


def _build_trade_series(kind: ReturnSeriesKind, *, source: VerifiedBacktestSource, artifact_store: MLArtifactStore) -> ReturnSeriesBundle:
    values: list[float] = []
    boundaries: list[FoldBoundary] = []
    hashes: list[str] = []
    timestamps: list[str] = []
    for outer_fold_index in sorted(source.manifest.outer_fold_result_references):
        ref = source.manifest.outer_fold_result_references[outer_fold_index]
        result = _decode_fold_result(ref.content_hash, artifact_store=artifact_store)
        raw = artifact_store.read_artifact(result.trade_set_reference.content_hash)
        trade_set = TradeSet.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        closed = sorted(trade_set.closed_trades, key=lambda t: (t.entry_bar_position, t.exit_bar_position))
        start = len(values)
        for t in closed:
            values.append(t.gross_return if kind is ReturnSeriesKind.TRADE_GROSS else t.net_return)
            timestamps.append(t.exit_timestamp)
        boundaries.append(FoldBoundary(outer_fold_index=outer_fold_index, start_index=start, end_index=len(values) - 1))
        hashes.append(result.trade_set_reference.content_hash)
    if not values:
        raise ReturnSeriesError("build_return_series: source backtest has no closed trades across any outer fold")
    return ReturnSeriesBundle(
        schema_version=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, kind=kind, sampling_frequency="trade", observation_count=len(values),
        effective_sample_count=_effective_sample_count(tuple(values), adjust_for_autocorrelation=False), values=tuple(values),
        fold_boundaries=tuple(boundaries), source_artifact_content_hashes=tuple(hashes),
        time_range_start=min(timestamps), time_range_end=max(timestamps), built_at=format_utc_timestamp(utc_now()),
    )


def _build_stitched_series(*, source: VerifiedBacktestSource, artifact_store: MLArtifactStore) -> ReturnSeriesBundle:
    if source.manifest.stitched_equity_reference is None:
        raise ReturnSeriesError(f"build_return_series: source backtest {source.manifest.backtest_id!r} has no stitched_equity_reference")
    raw = artifact_store.read_artifact(source.manifest.stitched_equity_reference.content_hash)
    stitched = StitchedWalkForwardEquity.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    values = tuple(p.bar_net_return for p in stitched.points)
    boundaries = tuple(
        FoldBoundary(outer_fold_index=b.outer_fold_index, start_index=b.stitched_point_start_index, end_index=b.stitched_point_end_index)
        for b in stitched.fold_boundaries
    )
    return ReturnSeriesBundle(
        schema_version=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, kind=ReturnSeriesKind.STITCHED_BAR_NET, sampling_frequency="bar",
        observation_count=len(values), effective_sample_count=_effective_sample_count(values, adjust_for_autocorrelation=True), values=values,
        fold_boundaries=boundaries, source_artifact_content_hashes=(source.manifest.stitched_equity_reference.content_hash,),
        time_range_start=stitched.points[0].timestamp, time_range_end=stitched.points[-1].timestamp, built_at=format_utc_timestamp(utc_now()),
    )


def _build_benchmark_relative_series(*, source: VerifiedBacktestSource, artifact_store: MLArtifactStore, benchmark_name: str) -> ReturnSeriesBundle:
    """Benchmarks are only persisted at FOLD granularity (Section 26 of
    the backtesting framework), so `BENCHMARK_RELATIVE` is necessarily a
    fold-level series: one observation per outer fold, `fold's own total
    net return - the declared benchmark's net return for that same
    fold`."""
    values: list[float] = []
    boundaries: list[FoldBoundary] = []
    hashes: list[str] = []
    for outer_fold_index in sorted(source.manifest.outer_fold_result_references):
        ref = source.manifest.outer_fold_result_references[outer_fold_index]
        result = _decode_fold_result(ref.content_hash, artifact_store=artifact_store)
        raw = artifact_store.read_artifact(result.benchmark_report_reference.content_hash)
        benchmark_report = BenchmarkReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        matching = [b for b in benchmark_report.benchmarks if b.name == benchmark_name]
        if not matching:
            raise ReturnSeriesError(
                f"build_return_series: benchmark_name={benchmark_name!r} not found in outer fold {outer_fold_index}'s "
                f"BenchmarkReport (available: {sorted(b.name for b in benchmark_report.benchmarks)})"
            )
        raw_total_net_return = result.financial_metrics.get("total_net_return")
        if raw_total_net_return is None or isinstance(raw_total_net_return, bool) or not isinstance(raw_total_net_return, (int, float)):
            raise ReturnSeriesError(f"build_return_series: outer fold {outer_fold_index} has no finite total_net_return to compare against the benchmark")
        fold_net_return = float(raw_total_net_return)
        if not math.isfinite(fold_net_return):
            raise ReturnSeriesError(f"build_return_series: outer fold {outer_fold_index} has no finite total_net_return to compare against the benchmark")
        values.append(fold_net_return - matching[0].net_return)
        boundaries.append(FoldBoundary(outer_fold_index=outer_fold_index, start_index=len(values) - 1, end_index=len(values) - 1))
        hashes.append(result.benchmark_report_reference.content_hash)
    if not values:
        raise ReturnSeriesError("build_return_series: source backtest has no outer folds to compare against a benchmark")
    return ReturnSeriesBundle(
        schema_version=RETURN_SERIES_BUNDLE_SCHEMA_VERSION, kind=ReturnSeriesKind.BENCHMARK_RELATIVE, sampling_frequency="fold",
        observation_count=len(values), effective_sample_count=_effective_sample_count(tuple(values), adjust_for_autocorrelation=False),
        values=tuple(values), fold_boundaries=tuple(boundaries), source_artifact_content_hashes=tuple(hashes),
        time_range_start=None, time_range_end=None, built_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "DEFAULT_BENCHMARK_NAME",
    "RETURN_SERIES_BUNDLE_SCHEMA_VERSION",
    "FoldBoundary",
    "ReturnSeriesBundle",
    "build_return_series",
]
