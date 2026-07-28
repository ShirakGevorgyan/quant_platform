"""`BacktestRunner`: the top-level orchestrator for one leakage-safe
financial evaluation / backtest run (Milestone 5, Section 23). Mirrors
`calibration.runner.CalibrationRunner`'s pipeline shape -- staged
`BacktestStage` transitions, lock-guarded run/resume entry points,
idempotent no-op on an already-`COMPLETED` backtest, resume via re-
verified artifacts rather than trusting the manifest.

WHY THIS RUNNER NEVER REFITS A MODEL
--------------------------------------------------------------------------
Unlike `calibration.runner`/`optimization.runner` (which both retrain on
the outer-train partition), this runner consumes ALREADY-COMPUTED,
already-verified outer-fold predictions from a completed calibration run
(`calibration.runner.OuterFoldCalibrationResult`) -- there is no model
refit anywhere in this package. `resolve_backtest_inputs` therefore needs
none of `optimization`/`calibration`'s `ModelFactory`/`ModelSerializer`/
`ModelRegistry` machinery; it resolves exactly three things: the outer
fold boundaries (reused, unchanged, from `execution.splitters` -- the SAME
boundaries calibration used), the calibration run's persisted per-fold
prediction artifacts (independently re-verified, never trusted by
filename/hash alone -- see `verify_and_load_predictions`), and raw market
bars (Section 6's separate market-data contract, positionally aligned to
the SAME timeline row-index space the outer folds are defined over).

THE INDEPENDENT-RE-VERIFICATION CONTRACT (Section 5)
--------------------------------------------------------------------------
`verify_and_load_predictions` does not just read a calibration's
persisted `OuterFoldCalibrationResult` and trust it -- it re-derives the
frozen calibrator from its own persisted parameters and re-applies it to
the persisted raw probabilities, asserting the result reproduces the
persisted calibrated probabilities bit-for-bit (the exact recomputation
check `calibration.verification._verify_calibrated_probabilities_
reproduce` performs, repeated here independently rather than imported,
since this package must not simply trust that calibration's own
verification was ever run)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.backtesting.costs import (
    entry_slippage_price_adjustment,
    entry_spread_price_adjustment,
    exit_slippage_price_adjustment,
    exit_spread_price_adjustment,
)
from quant_platform.backtesting.drawdown import DrawdownReport, compute_drawdown_report
from quant_platform.backtesting.execution import simulate_outer_fold_trades
from quant_platform.backtesting.manifests import (
    BACKTEST_MANIFEST_SCHEMA_VERSION,
    BacktestEventStore,
    BacktestEventType,
    BacktestManifest,
    BacktestManifestStore,
)
from quant_platform.backtesting.metrics import compute_financial_metrics
from quant_platform.backtesting.models import (
    BacktestStage,
    DecisionTimestampPolicyKind,
    PositionDirection,
    PositionMode,
    SignalReasonCode,
    VerifiedPredictionSet,
    prediction_availability_timestamp,
    validate_market_bar_frame,
)
from quant_platform.backtesting.returns import (
    EquityCurve,
    TradeReturnResult,
    compute_gross_return,
)
from quant_platform.backtesting.signals import Signal, SignalSet, generate_signals
from quant_platform.backtesting.specs import BacktestSpec, compute_backtest_identity
from quant_platform.backtesting.stitching import build_stitched_walk_forward_equity
from quant_platform.backtesting.timeline import (
    BarReturnTimeline,
    bar_return_timeline_to_equity_curve,
    build_bar_return_timeline,
)
from quant_platform.backtesting.trades import TradeRecord, TradeSet
from quant_platform.calibration.fitting import FrozenDecisionPolicy
from quant_platform.calibration.manifests import CalibrationManifest, CalibrationManifestStore
from quant_platform.calibration.models import CalibrationStage
from quant_platform.calibration.runner import OuterFoldCalibrationResult
from quant_platform.calibration.specs import CalibrationSpec, compute_calibration_identity
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    BacktestResumeError,
    BacktestValidationError,
    ExperimentLockError,
    MarketDataBindingError,
    QuantPlatformError,
    SchemaVersionError,
)
from quant_platform.core.types import Timeframe
from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.execution.runner import (
    assert_preprocessing_is_safe_for_execution,
    extract_label_horizon_bars,
)
from quant_platform.execution.splitters import (
    Fold,
    build_folds_from_split_binding,
    reconstruct_dataset_timeline,
)
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.locking import DEFAULT_STALE_AFTER, DatasetLock, LockInfo
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.metrics import MetricComputationReport
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    EnvironmentSnapshot,
    JsonPrimitive,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    require_schema_version,
    utc_now,
)

logger = logging.getLogger(__name__)

_TIMESTAMP_COLUMN = "open_time"
_BACKTEST_RUN_LOCK_FILE_NAME = ".backtest_run.lock"
"""Held for the entire run's duration -- distinct from `BacktestManifestStore`'s
own brief per-transition `.backtest.lock`, exactly `calibration.runner`'s
identical `_CALIBRATION_RUN_LOCK_FILE_NAME` reentrancy argument."""
_RECOMPUTATION_TOLERANCE = 1e-9
_MIN_BUCKET_SAMPLES = 5
OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION = 1
_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, BacktestValidationError, KeyError, ValueError, TypeError,
)


# --------------------------------------------------------------------------
# Section 5: independently re-verified source predictions
# --------------------------------------------------------------------------
def verify_and_load_predictions(
    *, outer_fold_index: int, calibration_manifest: CalibrationManifest, calibration_spec: CalibrationSpec, timeline: pd.DataFrame,
    timestamp_column: str, bar_interval: Timeframe, decision_timestamp_policy: DecisionTimestampPolicyKind, artifact_store: MLArtifactStore,
) -> VerifiedPredictionSet:
    """Section 5: load ONE outer fold's calibrated predictions from a
    completed calibration run and independently re-verify them -- never
    trust `OuterFoldCalibrationResult` by filename/hash alone. Re-derives
    the frozen calibrator from its own persisted parameters and re-applies
    it to the persisted raw probabilities, asserting the result reproduces
    the persisted calibrated probabilities (mirrors `calibration.
    verification._verify_calibrated_probabilities_reproduce`'s exact
    recomputation check, repeated independently here).

    Milestone 5.1, Section 5: `VerifiedPredictionSet.timestamps` (which
    becomes every downstream `Signal.decision_timestamp`) is each sample's
    bar's `prediction_availability_timestamp` -- its CLOSE, not its open
    -- since a prediction derived from bar N's own data is only knowable
    once bar N has finished. `decision_timestamp_policy` is passed through
    (not merely assumed) so `prediction_availability_timestamp` itself
    enforces, at the one place this computation happens, that only a
    supported policy (`BacktestSpec.__post_init__` already guarantees
    AFTER_BAR_CLOSE is the only one that can reach here) is ever used."""
    reference = calibration_manifest.outer_fold_result_references.get(outer_fold_index)
    if reference is None:
        raise BacktestValidationError(
            f"verify_and_load_predictions: source calibration {calibration_manifest.calibration_id!r} has no "
            f"recorded result for outer fold {outer_fold_index}"
        )
    if reference.category is not ArtifactCategory.OUTER_FOLD_CALIBRATION_RESULT:
        raise BacktestValidationError(f"verify_and_load_predictions: outer fold {outer_fold_index} reference has the wrong ArtifactCategory")
    try:
        raw = artifact_store.read_artifact(reference.content_hash)
        decoded = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        raise BacktestValidationError(f"verify_and_load_predictions: outer fold {outer_fold_index} result could not be read and decoded: {exc}") from exc
    if decoded.outer_fold_index != outer_fold_index or decoded.calibration_id != calibration_manifest.calibration_id:
        raise BacktestValidationError(f"verify_and_load_predictions: outer fold {outer_fold_index} result does not match its own filing key")

    try:
        policy_raw = artifact_store.read_artifact(decoded.decision_policy_reference.content_hash)
        policy = FrozenDecisionPolicy.from_json_dict(parse_json_strict(policy_raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        raise BacktestValidationError(f"verify_and_load_predictions: outer fold {outer_fold_index} FrozenDecisionPolicy could not be read and decoded: {exc}") from exc

    recomputed = policy.selected_calibrator().transform(np.asarray(decoded.raw_probabilities))
    persisted = np.asarray(decoded.calibrated_probabilities)
    max_diff = float(np.max(np.abs(recomputed - persisted))) if len(persisted) else 0.0
    if max_diff > _RECOMPUTATION_TOLERANCE:
        raise BacktestValidationError(
            f"verify_and_load_predictions: outer fold {outer_fold_index}: re-applying the persisted calibrator to "
            f"the persisted raw probabilities does not reproduce the persisted calibrated probabilities "
            f"(max abs diff={max_diff:.3g}) -- the source calibration artifact is hash-consistent but SEMANTICALLY WRONG"
        )

    timestamps = tuple(
        format_utc_timestamp(prediction_availability_timestamp(timeline.iloc[p][timestamp_column], bar_interval=bar_interval, policy=decision_timestamp_policy))
        for p in decoded.sample_positions
    )
    return VerifiedPredictionSet(
        schema_version=1, outer_fold_index=outer_fold_index, source_calibration_id=calibration_manifest.calibration_id,
        source_experiment_id=calibration_manifest.source_experiment_id, source_execution_id=calibration_manifest.source_experiment_id,
        base_model_definition_identity=calibration_spec.base_model_definition_identity, sample_positions=decoded.sample_positions, timestamps=timestamps,
        raw_probabilities=decoded.raw_probabilities, calibrated_probabilities=decoded.calibrated_probabilities,
        threshold=policy.threshold_report.selected_threshold, decisions=decoded.decisions,
        abstention_reason_codes=decoded.abstention_reason_codes, confidence_scores=decoded.confidence_scores,
        confidence_categories=decoded.confidence_categories, uncertainty_scores=decoded.uncertainty_scores,
    )


def resolve_market_bars_for_timeline(
    dataset_loader: DatasetLoader, *, symbol: str, base_timeframe: str, timeline: pd.DataFrame, timestamp_column: str,
) -> pd.DataFrame:
    """Section 6: raw OHLC(V) market bars, positionally aligned to
    `timeline`'s own row-index space (`bars.iloc[i]` corresponds to the
    SAME bar as `timeline.iloc[i]`) -- this is what lets `backtesting.
    execution`'s bar-position walk use `signal.sample_position` (an index
    INTO `timeline`) directly as a `bars` row position. No forward-fill,
    no silent gap-bridging (Section 6): any timeline bar with no matching
    raw market bar fails closed."""
    tf = Timeframe(base_timeframe)
    timestamps = timeline[timestamp_column]
    start = timestamps.iloc[0]
    end = timestamps.iloc[-1] + tf.duration
    request = LoadRequest(symbol=symbol, timeframe=tf, start=start, end=end)
    raw_bars = dataset_loader.load(request)

    indexed = raw_bars.set_index("open_time")
    aligned = indexed.reindex(pd.Index(timestamps.to_numpy(), name="open_time"))
    missing = aligned["open"].isna()
    if missing.any():
        raise MarketDataBindingError(
            f"resolve_market_bars_for_timeline: {int(missing.sum())} timeline bar(s) have no matching raw market "
            f"bar for symbol={symbol!r} timeframe={tf.value!r} -- refusing to forward-fill or silently bridge the gap",
            context={"symbol": symbol, "timeframe": tf.value, "missing_count": int(missing.sum())},
        )
    aligned = aligned.reset_index()
    validate_market_bar_frame(aligned, context="resolve_market_bars_for_timeline")
    return aligned


# --------------------------------------------------------------------------
# Benchmarks (Section 26)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    description: str
    gross_return: float
    net_return: float

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "description": self.description, "gross_return": self.gross_return, "net_return": self.net_return}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BenchmarkResult:
        return cls(name=str(raw["name"]), description=str(raw["description"]), gross_return=float(str(raw["gross_return"])), net_return=float(str(raw["net_return"])))


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    schema_version: int
    outer_fold_index: int
    benchmarks: tuple[BenchmarkResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "benchmarks": [b.to_json_dict() for b in self.benchmarks]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BenchmarkReport:
        require_schema_version(raw, supported=1, context="BenchmarkReport")
        return cls(schema_version=1, outer_fold_index=int(str(raw["outer_fold_index"])), benchmarks=tuple(BenchmarkResult.from_json_dict(b) for b in as_json_list(raw["benchmarks"], field_name="benchmarks")))


def _buy_and_hold_benchmark_pair(*, name_prefix: str, description: str, direction: PositionDirection, bars: pd.DataFrame, spec: BacktestSpec, fold_start_position: int, fold_end_position: int) -> tuple[BenchmarkResult, BenchmarkResult]:
    """The simple, direct-price reference (Section 26's original
    `always_long_zero_cost`/`always_long_net_cost`): one round-trip
    (`fold_start_position` close -> `fold_end_position` close) using
    `close` prices directly, deliberately NOT routed through the full
    entry/exit fill machinery -- a genuinely different (simpler) construction
    from `_signal_variant_benchmark_pair`'s full-pipeline strategies below."""
    entry_price = float(bars.iloc[fold_start_position]["close"])
    exit_price = float(bars.iloc[fold_end_position]["close"])
    gross = compute_gross_return(direction=direction, entry_observed_price=entry_price, exit_observed_price=exit_price, return_calculation_policy=spec.return_calculation_policy)
    zero_cost = BenchmarkResult(name=f"{name_prefix}_zero_cost", description=f"{description} (transaction costs ignored).", gross_return=gross, net_return=gross)

    entry_spread = entry_spread_price_adjustment(spec.spread_spec, entry_price, direction)
    exit_spread = exit_spread_price_adjustment(spec.spread_spec, exit_price, direction)
    entry_slippage = entry_slippage_price_adjustment(spec.slippage_spec, entry_price, direction)
    exit_slippage = exit_slippage_price_adjustment(spec.slippage_spec, exit_price, direction)
    holding_days = max((bars.iloc[fold_end_position][_TIMESTAMP_COLUMN] - bars.iloc[fold_start_position][_TIMESTAMP_COLUMN]).total_seconds() / 86400.0, 0.0)
    net_result = _trade_return_result_for_benchmark(
        direction=direction, entry_price=entry_price, exit_price=exit_price, entry_spread=entry_spread, exit_spread=exit_spread,
        entry_slippage=entry_slippage, exit_slippage=exit_slippage, spec=spec, holding_days=holding_days,
    )
    net_cost = BenchmarkResult(name=f"{name_prefix}_net_cost", description=f"{description} (this backtest's own declared cost model applied).", gross_return=net_result.gross_return, net_return=net_result.net_return)
    return zero_cost, net_cost


def _trade_return_result_for_benchmark(*, direction: PositionDirection, entry_price: float, exit_price: float, entry_spread: float, exit_spread: float, entry_slippage: float, exit_slippage: float, spec: BacktestSpec, holding_days: float) -> TradeReturnResult:
    from quant_platform.backtesting.returns import compute_trade_return_result

    return compute_trade_return_result(
        direction=direction, entry_observed_price=entry_price, exit_observed_price=exit_price,
        entry_spread_adjustment=entry_spread, exit_spread_adjustment=exit_spread, entry_slippage_adjustment=entry_slippage,
        exit_slippage_adjustment=exit_slippage, commission_spec=spec.commission_spec, financing_spec=spec.financing_spec,
        holding_days=holding_days, notional=spec.initial_notional * spec.exposure_cap, return_calculation_policy=spec.return_calculation_policy,
    )


def _constant_direction_signal_set(predictions: VerifiedPredictionSet, *, direction: PositionDirection) -> SignalSet:
    """A synthetic `SignalSet` that always signals `direction` -- the
    source for `always_long`/`always_short` strategy benchmarks (Milestone
    5.1, Section 4). Every field this module never actually READS
    (confidence/uncertainty/calibrated_probability) is set to an
    uninformative, valid placeholder (never fabricated to look
    meaningful) -- `strength=1.0` (maximal, matching "always" commitment)."""
    reason = SignalReasonCode.ACCEPTED_POSITIVE if direction is PositionDirection.LONG else SignalReasonCode.ACCEPTED_NEGATIVE
    signals = tuple(
        Signal(
            sample_position=predictions.sample_positions[i], decision_timestamp=predictions.timestamps[i], direction=direction,
            strength=1.0, accepted=True, reason_code=reason, confidence=1.0, uncertainty=0.0, threshold=predictions.threshold,
            calibrated_probability=predictions.calibrated_probabilities[i], source_calibration_id=predictions.source_calibration_id,
            source_experiment_id=predictions.source_experiment_id, outer_fold_index=predictions.outer_fold_index,
        )
        for i in range(predictions.n_samples)
    )
    return SignalSet(schema_version=1, outer_fold_index=predictions.outer_fold_index, signals=signals)


def _raw_uncalibrated_threshold_signal_set(predictions: VerifiedPredictionSet, *, position_mode: PositionMode) -> SignalSet:
    """Section 4's "raw uncalibrated threshold signal" benchmark: decides
    direction from `predictions.raw_probabilities` alone (bypassing
    calibration entirely) against a FIXED neutral cutpoint of 0.5 -- raw,
    uncalibrated probabilities have no fitted decision threshold of their
    own (that is precisely what calibration/threshold selection produces),
    so 0.5 is the only principled, non-data-dependent choice. No
    abstention concept applies here (abstention is a CALIBRATION-policy
    concept -- Section 8); every sample is accepted."""
    short_direction = PositionDirection.SHORT if position_mode is PositionMode.LONG_SHORT else PositionDirection.FLAT
    signals = []
    for i in range(predictions.n_samples):
        raw = predictions.raw_probabilities[i]
        is_positive = raw >= 0.5
        direction = PositionDirection.LONG if is_positive else short_direction
        reason = SignalReasonCode.ACCEPTED_POSITIVE if is_positive else SignalReasonCode.ACCEPTED_NEGATIVE
        strength = min(abs(raw - 0.5) * 2.0, 1.0)
        signals.append(Signal(
            sample_position=predictions.sample_positions[i], decision_timestamp=predictions.timestamps[i], direction=direction,
            strength=strength, accepted=True, reason_code=reason, confidence=predictions.confidence_scores[i],
            uncertainty=predictions.uncertainty_scores[i], threshold=0.5, calibrated_probability=raw,
            source_calibration_id=predictions.source_calibration_id, source_experiment_id=predictions.source_experiment_id,
            outer_fold_index=predictions.outer_fold_index,
        ))
    return SignalSet(schema_version=1, outer_fold_index=predictions.outer_fold_index, signals=tuple(signals))


def _signal_variant_benchmark_pair(
    *, name_prefix: str, description: str, signals: SignalSet, bars: pd.DataFrame, spec: BacktestSpec,
    fold_start_position: int, fold_end_position: int, outer_fold_index: int,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Runs `signals` through the SAME full execution pipeline the actual
    strategy uses (`simulate_outer_fold_trades` -> `build_bar_return_
    timeline`, sharing this backtest's own outer-fold timestamps,
    execution timing, fold-boundary rules, price basis, capital
    normalization, and return timeline -- Section 4's explicit
    requirement), then reports the resulting total return as a
    `_zero_cost`/`_net_cost` pair from that ONE simulation's gross vs net
    cumulative equity. Because both rows are read off the SAME trade set,
    `_zero_cost.gross_return == _net_cost.gross_return` always (gross
    return is cost-independent by construction) -- Section 4's explicit
    "zero-cost and net-cost gross returns match" property, proven in
    `test_backtesting_benchmark_matrix.py`."""
    trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=fold_end_position, cost_scenario=None)
    timeline = build_bar_return_timeline(
        trades=trade_set.trades, bars=bars, fold_start_position=fold_start_position, fold_end_position=fold_end_position,
        outer_fold_index=outer_fold_index, return_calculation_policy=spec.return_calculation_policy, exposure_cap=spec.exposure_cap,
        compounded=spec.compounding_policy.value == "compounded",
    )
    last = timeline.points[-1]
    gross_total = last.cumulative_gross_equity - 1.0
    net_total = last.cumulative_net_equity - 1.0
    zero_cost = BenchmarkResult(name=f"{name_prefix}_zero_cost", description=f"{description} (transaction costs ignored).", gross_return=gross_total, net_return=gross_total)
    net_cost = BenchmarkResult(name=f"{name_prefix}_net_cost", description=f"{description} (this backtest's own declared cost model applied).", gross_return=gross_total, net_return=net_total)
    return zero_cost, net_cost


def compute_benchmark_report(
    *, predictions: VerifiedPredictionSet, bars: pd.DataFrame, spec: BacktestSpec, fold_start_position: int, fold_end_position: int, outer_fold_index: int,
) -> BenchmarkReport:
    """Section 26, completed per Milestone 5.1 Section 4. Every benchmark
    below shares this fold's own outer-test timestamps, execution timing
    (`spec.entry_spec`/`exit_spec`/`overlap_policy`/`decision_timestamp_
    policy`), price basis, capital normalization (`spec.initial_notional`/
    `exposure_cap`), and return-calculation policy -- either directly
    (`_buy_and_hold_benchmark_pair`'s simple price-ratio construction) or
    by literally running through `simulate_outer_fold_trades` +
    `build_bar_return_timeline`, the SAME functions/policy objects the
    real strategy uses (`_signal_variant_benchmark_pair`).

    MATRIX (Section 4's exact list, `always_short*`/`*_short_strategy`
    omitted under `PositionMode.LONG_FLAT` -- "where supported"):
      - `always_flat`: the trivial zero-return reference.
      - `always_long_zero_cost`/`always_long_net_cost`: buy-and-hold
        (simple direct close-to-close price ratio, Section 26 original).
      - `always_short_zero_cost`/`always_short_net_cost`: sell-and-hold,
        the SAME simple construction, direction=SHORT (LONG_SHORT only).
      - `always_long_strategy_zero_cost`/`_net_cost`: a constant LONG
        signal on EVERY sample, run through the full fill/cost pipeline
        (distinct from buy-and-hold: respects entry-delay/exit timing).
      - `always_short_strategy_zero_cost`/`_net_cost`: the same, SHORT
        (LONG_SHORT only).
      - `raw_uncalibrated_threshold_zero_cost`/`_net_cost`: direction from
        `raw_probabilities >= 0.5` (bypassing calibration), full pipeline.
      - `calibrated_no_abstention_zero_cost`/`_net_cost`: `spec.signal_
        mapping`'s own decision rule with abstention filtering forced OFF.
      - `calibrated_with_abstention_zero_cost`/`_net_cost`: the same
        signal-mapping rule with abstention filtering forced ON --
        independently recomputed here (not merely re-reporting the
        primary strategy result) so this row exists regardless of
        `spec.respect_calibration_abstention`'s own setting."""
    always_flat = BenchmarkResult(name="always_flat", description="Never enter a position -- the trivial zero-return reference.", gross_return=0.0, net_return=0.0)
    benchmarks: list[BenchmarkResult] = [always_flat]

    long_zero, long_net = _buy_and_hold_benchmark_pair(
        name_prefix="always_long", description="Buy and hold from the fold's first test bar close to its last", direction=PositionDirection.LONG,
        bars=bars, spec=spec, fold_start_position=fold_start_position, fold_end_position=fold_end_position,
    )
    benchmarks += [long_zero, long_net]

    if spec.position_mode is PositionMode.LONG_SHORT:
        short_zero, short_net = _buy_and_hold_benchmark_pair(
            name_prefix="always_short", description="Sell and hold from the fold's first test bar close to its last", direction=PositionDirection.SHORT,
            bars=bars, spec=spec, fold_start_position=fold_start_position, fold_end_position=fold_end_position,
        )
        benchmarks += [short_zero, short_net]

    variants: list[tuple[str, str, SignalSet]] = [
        ("always_long_strategy", "A constant LONG signal on every sample, executed through the full entry/exit/cost pipeline", _constant_direction_signal_set(predictions, direction=PositionDirection.LONG)),
        ("raw_uncalibrated_threshold", "Direction from the raw (uncalibrated) probability against a fixed 0.5 cutpoint, executed through the full pipeline", _raw_uncalibrated_threshold_signal_set(predictions, position_mode=spec.position_mode)),
        ("calibrated_no_abstention", "This fold's own calibrated signal-mapping rule with calibration abstention filtering forced OFF", generate_signals(predictions, spec=spec.signal_mapping, position_mode=spec.position_mode, respect_calibration_abstention=False)),
        ("calibrated_with_abstention", "This fold's own calibrated signal-mapping rule with calibration abstention filtering forced ON", generate_signals(predictions, spec=spec.signal_mapping, position_mode=spec.position_mode, respect_calibration_abstention=True)),
    ]
    if spec.position_mode is PositionMode.LONG_SHORT:
        variants.insert(1, ("always_short_strategy", "A constant SHORT signal on every sample, executed through the full entry/exit/cost pipeline", _constant_direction_signal_set(predictions, direction=PositionDirection.SHORT)))

    for name_prefix, description, variant_signals in variants:
        zero_cost, net_cost = _signal_variant_benchmark_pair(
            name_prefix=name_prefix, description=description, signals=variant_signals, bars=bars, spec=spec,
            fold_start_position=fold_start_position, fold_end_position=fold_end_position, outer_fold_index=outer_fold_index,
        )
        benchmarks += [zero_cost, net_cost]

    return BenchmarkReport(schema_version=1, outer_fold_index=outer_fold_index, benchmarks=tuple(benchmarks))


# --------------------------------------------------------------------------
# Cost sensitivity (Section 20)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CostSensitivityResult:
    scenario_name: str
    trade_count: int
    total_net_return: float

    def to_json_dict(self) -> dict[str, object]:
        return {"scenario_name": self.scenario_name, "trade_count": self.trade_count, "total_net_return": self.total_net_return}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostSensitivityResult:
        return cls(scenario_name=str(raw["scenario_name"]), trade_count=int(str(raw["trade_count"])), total_net_return=float(str(raw["total_net_return"])))


@dataclass(frozen=True, slots=True)
class CostSensitivityReport:
    schema_version: int
    outer_fold_index: int
    results: tuple[CostSensitivityResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "results": [r.to_json_dict() for r in self.results]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostSensitivityReport:
        require_schema_version(raw, supported=1, context="CostSensitivityReport")
        return cls(schema_version=1, outer_fold_index=int(str(raw["outer_fold_index"])), results=tuple(CostSensitivityResult.from_json_dict(r) for r in as_json_list(raw["results"], field_name="results")))


def compute_cost_sensitivity_report(
    *, signals: SignalSet, bars: pd.DataFrame, spec: BacktestSpec, fold_end_position: int, outer_fold_index: int,
) -> CostSensitivityReport:
    """Section 20: every scenario in `spec.cost_sensitivity_scenarios` is
    a bounded, PRE-DECLARED multiplier on the same cost model -- never a
    post-hoc "best scenario" search. `base_cost` is the primary trade set
    already computed by the caller; re-simulated here too (bounded, small
    scenario count) purely for a self-contained, uniformly-computed table."""
    results: list[CostSensitivityResult] = []
    for scenario in spec.cost_sensitivity_scenarios:
        scenario_trades = simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=fold_end_position, cost_scenario=scenario)
        closed = scenario_trades.closed_trades
        results.append(CostSensitivityResult(scenario_name=scenario.name, trade_count=len(closed), total_net_return=sum(t.net_return for t in closed)))
    return CostSensitivityReport(schema_version=1, outer_fold_index=outer_fold_index, results=tuple(results))


# --------------------------------------------------------------------------
# Confidence / uncertainty bucket analysis (Section 21)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BucketResult:
    dimension: str
    """`"confidence"` or `"uncertainty"`."""
    bucket: str
    """`"low"`, `"medium"`, or `"high"` -- fixed tercile boundaries
    ([0, 1/3), [1/3, 2/3), [2/3, 1]), never data-dependent percentiles
    (Section 21: bucket boundaries must not be chosen post-hoc from the
    very outer-test outcomes being analyzed)."""
    sample_count: int
    insufficient_sample: bool
    average_net_return: float | None
    hit_rate: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension, "bucket": self.bucket, "sample_count": self.sample_count,
            "insufficient_sample": self.insufficient_sample, "average_net_return": self.average_net_return, "hit_rate": self.hit_rate,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BucketResult:
        return cls(
            dimension=str(raw["dimension"]), bucket=str(raw["bucket"]), sample_count=int(str(raw["sample_count"])),
            insufficient_sample=bool(raw["insufficient_sample"]),
            average_net_return=(None if raw.get("average_net_return") is None else float(str(raw["average_net_return"]))),
            hit_rate=(None if raw.get("hit_rate") is None else float(str(raw["hit_rate"]))),
        )


@dataclass(frozen=True, slots=True)
class BucketAnalysisReport:
    schema_version: int
    outer_fold_index: int
    minimum_bucket_samples: int
    buckets: tuple[BucketResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "minimum_bucket_samples": self.minimum_bucket_samples, "buckets": [b.to_json_dict() for b in self.buckets]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BucketAnalysisReport:
        require_schema_version(raw, supported=1, context="BucketAnalysisReport")
        return cls(
            schema_version=1, outer_fold_index=int(str(raw["outer_fold_index"])), minimum_bucket_samples=int(str(raw["minimum_bucket_samples"])),
            buckets=tuple(BucketResult.from_json_dict(b) for b in as_json_list(raw["buckets"], field_name="buckets")),
        )


def _tercile_bucket(value: float) -> str:
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"


def compute_bucket_analysis_report(*, trades: Sequence[TradeRecord], outer_fold_index: int, minimum_bucket_samples: int = _MIN_BUCKET_SAMPLES) -> BucketAnalysisReport:
    """Section 21: bucket CLOSED trades by their originating signal's
    confidence/uncertainty into fixed terciles, reporting average net
    return and hit rate per bucket -- a bucket with fewer than
    `minimum_bucket_samples` trades is flagged `insufficient_sample`
    rather than reporting a statistically meaningless average
    (`ml.metrics.MetricComputationReport`'s "skip, never fabricate"
    convention applied per-bucket)."""
    def _confidence(t: TradeRecord) -> float:
        return t.confidence

    def _uncertainty(t: TradeRecord) -> float:
        return t.uncertainty

    buckets: list[BucketResult] = []
    dimensions: tuple[tuple[str, Callable[[TradeRecord], float]], ...] = (("confidence", _confidence), ("uncertainty", _uncertainty))
    for dimension, extractor in dimensions:
        grouped: dict[str, list[TradeRecord]] = defaultdict(list)
        for t in trades:
            grouped[_tercile_bucket(extractor(t))].append(t)
        for bucket_name in ("low", "medium", "high"):
            members = grouped.get(bucket_name, [])
            insufficient = len(members) < minimum_bucket_samples
            if members and not insufficient:
                avg_return = sum(t.net_return for t in members) / len(members)
                hit_rate = sum(1 for t in members if t.net_return > 0.0) / len(members)
            else:
                avg_return, hit_rate = None, None
            buckets.append(BucketResult(dimension=dimension, bucket=bucket_name, sample_count=len(members), insufficient_sample=insufficient, average_net_return=avg_return, hit_rate=hit_rate))
    return BucketAnalysisReport(schema_version=1, outer_fold_index=outer_fold_index, minimum_bucket_samples=minimum_bucket_samples, buckets=tuple(buckets))


# --------------------------------------------------------------------------
# One outer fold's complete backtest result (Section 19/23)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OuterFoldBacktestResult:
    schema_version: int
    backtest_id: str
    outer_fold_index: int
    signal_set_reference: ArtifactReference
    trade_set_reference: ArtifactReference
    bar_return_timeline_reference: ArtifactReference
    equity_curve_reference: ArtifactReference
    gross_drawdown_reference: ArtifactReference
    net_drawdown_reference: ArtifactReference
    benchmark_report_reference: ArtifactReference
    cost_sensitivity_report_reference: ArtifactReference
    bucket_analysis_report_reference: ArtifactReference
    outer_test_row_count: int
    closed_trade_count: int
    meets_minimum_trade_threshold: bool
    financial_metrics: Mapping[str, JsonPrimitive]
    skipped_metrics: Mapping[str, str]
    evaluated_at: str

    def __post_init__(self) -> None:
        if self.outer_fold_index < 0:
            raise BacktestValidationError(f"OuterFoldBacktestResult.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        validate_json_primitive_mapping(self.financial_metrics, field_name="OuterFoldBacktestResult.financial_metrics")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "backtest_id": self.backtest_id, "outer_fold_index": self.outer_fold_index,
            "signal_set_reference": self.signal_set_reference.to_json_dict(), "trade_set_reference": self.trade_set_reference.to_json_dict(),
            "bar_return_timeline_reference": self.bar_return_timeline_reference.to_json_dict(),
            "equity_curve_reference": self.equity_curve_reference.to_json_dict(), "gross_drawdown_reference": self.gross_drawdown_reference.to_json_dict(),
            "net_drawdown_reference": self.net_drawdown_reference.to_json_dict(), "benchmark_report_reference": self.benchmark_report_reference.to_json_dict(),
            "cost_sensitivity_report_reference": self.cost_sensitivity_report_reference.to_json_dict(),
            "bucket_analysis_report_reference": self.bucket_analysis_report_reference.to_json_dict(),
            "outer_test_row_count": self.outer_test_row_count, "closed_trade_count": self.closed_trade_count,
            "meets_minimum_trade_threshold": self.meets_minimum_trade_threshold,
            "financial_metrics": dict(sorted(self.financial_metrics.items())), "skipped_metrics": dict(sorted(self.skipped_metrics.items())),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OuterFoldBacktestResult:
        require_schema_version(raw, supported=OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION, context="OuterFoldBacktestResult")

        def _ref(key: str) -> ArtifactReference:
            return ArtifactReference.from_json_dict(as_json_dict(raw[key], field_name=key))

        return cls(
            schema_version=OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION, backtest_id=str(raw["backtest_id"]), outer_fold_index=int(str(raw["outer_fold_index"])),
            signal_set_reference=_ref("signal_set_reference"), trade_set_reference=_ref("trade_set_reference"),
            bar_return_timeline_reference=_ref("bar_return_timeline_reference"), equity_curve_reference=_ref("equity_curve_reference"),
            gross_drawdown_reference=_ref("gross_drawdown_reference"), net_drawdown_reference=_ref("net_drawdown_reference"),
            benchmark_report_reference=_ref("benchmark_report_reference"), cost_sensitivity_report_reference=_ref("cost_sensitivity_report_reference"),
            bucket_analysis_report_reference=_ref("bucket_analysis_report_reference"), outer_test_row_count=int(str(raw["outer_test_row_count"])),
            closed_trade_count=int(str(raw["closed_trade_count"])), meets_minimum_trade_threshold=bool(raw["meets_minimum_trade_threshold"]),
            financial_metrics=dict(as_json_dict(raw.get("financial_metrics") or {}, field_name="financial_metrics")),
            skipped_metrics={k: str(v) for k, v in as_json_dict(raw.get("skipped_metrics") or {}, field_name="skipped_metrics").items()},
            evaluated_at=str(raw["evaluated_at"]),
        )


@dataclass(frozen=True, slots=True)
class RecomputedOuterFoldArtifacts:
    """Milestone 5.2, Section 3: every in-memory object one outer fold's
    backtest derives, computed WITHOUT writing anything to `artifact_
    store` -- the pure-computation half of `run_outer_fold_backtest`,
    split out so `backtesting.verification` can call the EXACT SAME
    production code path (never a re-implemented, drift-prone parallel
    computation) to independently rebuild a fold's results from raw
    sources (predictions, market bars, the declared `BacktestSpec`)
    during verification, without mutating the artifact store as a side
    effect of what must remain a read-only audit."""

    signals: SignalSet
    trade_set: TradeSet
    bar_timeline: BarReturnTimeline
    equity_curve: EquityCurve
    gross_drawdown: DrawdownReport
    net_drawdown: DrawdownReport
    metrics_report: MetricComputationReport
    benchmark_report: BenchmarkReport
    cost_sensitivity_report: CostSensitivityReport
    bucket_report: BucketAnalysisReport


def recompute_outer_fold_backtest_artifacts(
    *, spec: BacktestSpec, outer_fold: Fold, timeline: pd.DataFrame, bars: pd.DataFrame,
    calibration_manifest: CalibrationManifest, calibration_spec: CalibrationSpec, artifact_store: MLArtifactStore,
) -> RecomputedOuterFoldArtifacts:
    """Section 23's steps for ONE outer fold: verify predictions -> map
    signals -> simulate fills/trades -> compute returns/equity/drawdown ->
    compute metrics -> compute benchmarks/cost-sensitivity/bucket
    analysis. `artifact_store` is used READ-ONLY here (`verify_and_load_
    predictions` reads the source calibration's own persisted artifacts
    through it) -- nothing is written; see `run_outer_fold_backtest` for
    the thin persistence wrapper around this function."""
    predictions = verify_and_load_predictions(
        outer_fold_index=outer_fold.fold_index, calibration_manifest=calibration_manifest, calibration_spec=calibration_spec, timeline=timeline,
        timestamp_column=spec.timestamp_column, bar_interval=spec.bar_interval, decision_timestamp_policy=spec.decision_timestamp_policy,
        artifact_store=artifact_store,
    )
    fold_end_position = int(outer_fold.test_indices.max())
    fold_start_position = int(outer_fold.test_indices.min())

    signals = generate_signals(predictions, spec=spec.signal_mapping, position_mode=spec.position_mode, respect_calibration_abstention=spec.respect_calibration_abstention)
    trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=spec, fold_end_position=fold_end_position, cost_scenario=None)
    closed = trade_set.closed_trades

    # Section 1 (Milestone 5.1): a TRUE bar-level mark-to-market timeline
    # -- one point per outer-test market bar, never compressed to
    # trade-exit events (see `backtesting.timeline`'s module docstring for
    # the defect this corrects). `equity_curve`/`gross_drawdown`/
    # `net_drawdown` are all DERIVED from this single source of truth.
    bar_timeline: BarReturnTimeline = build_bar_return_timeline(
        trades=trade_set.trades, bars=bars, fold_start_position=fold_start_position, fold_end_position=fold_end_position,
        outer_fold_index=outer_fold.fold_index, return_calculation_policy=spec.return_calculation_policy, exposure_cap=spec.exposure_cap,
        compounded=spec.compounding_policy.value == "compounded",
    )
    equity_curve = bar_return_timeline_to_equity_curve(bar_timeline)
    gross_drawdown: DrawdownReport = compute_drawdown_report(equity_curve, equity_basis="gross")
    net_drawdown: DrawdownReport = compute_drawdown_report(equity_curve, equity_basis="net")
    metrics_report = compute_financial_metrics(
        trades=trade_set, equity_curve=equity_curve, gross_drawdown=gross_drawdown, net_drawdown=net_drawdown,
        signals=signals, bar_timeline=bar_timeline, bar_interval=spec.bar_interval, annual_risk_free_rate=spec.annual_risk_free_rate,
        initial_notional=spec.initial_notional, exposure_cap=spec.exposure_cap,
    )
    benchmark_report = compute_benchmark_report(predictions=predictions, bars=bars, spec=spec, fold_start_position=fold_start_position, fold_end_position=fold_end_position, outer_fold_index=outer_fold.fold_index)
    cost_sensitivity_report = compute_cost_sensitivity_report(signals=signals, bars=bars, spec=spec, fold_end_position=fold_end_position, outer_fold_index=outer_fold.fold_index)
    bucket_report = compute_bucket_analysis_report(trades=closed, outer_fold_index=outer_fold.fold_index)

    return RecomputedOuterFoldArtifacts(
        signals=signals, trade_set=trade_set, bar_timeline=bar_timeline, equity_curve=equity_curve, gross_drawdown=gross_drawdown,
        net_drawdown=net_drawdown, metrics_report=metrics_report, benchmark_report=benchmark_report,
        cost_sensitivity_report=cost_sensitivity_report, bucket_report=bucket_report,
    )


def run_outer_fold_backtest(
    *, spec: BacktestSpec, backtest_id: str, outer_fold: Fold, timeline: pd.DataFrame, bars: pd.DataFrame,
    calibration_manifest: CalibrationManifest, calibration_spec: CalibrationSpec, artifact_store: MLArtifactStore,
) -> tuple[OuterFoldBacktestResult, list[ArtifactReference]]:
    """Thin persistence wrapper around `recompute_outer_fold_backtest_
    artifacts`: writes every recomputed object to `artifact_store` and
    assembles the `OuterFoldBacktestResult` referencing them. Returns the
    result plus every artifact reference written along the way, so the
    caller can fold them into the manifest's `artifact_references` audit
    trail."""
    recomputed = recompute_outer_fold_backtest_artifacts(
        spec=spec, outer_fold=outer_fold, timeline=timeline, bars=bars,
        calibration_manifest=calibration_manifest, calibration_spec=calibration_spec, artifact_store=artifact_store,
    )
    signal_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.signals.to_json_dict()), category=ArtifactCategory.SIGNAL_SET)
    trade_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.trade_set.to_json_dict()), category=ArtifactCategory.TRADE_SET)
    bar_timeline_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.bar_timeline.to_json_dict()), category=ArtifactCategory.BAR_RETURN_TIMELINE)
    equity_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.equity_curve.to_json_dict()), category=ArtifactCategory.RETURN_SERIES)
    gross_dd_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.gross_drawdown.to_json_dict()), category=ArtifactCategory.DRAWDOWN_REPORT)
    net_dd_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.net_drawdown.to_json_dict()), category=ArtifactCategory.DRAWDOWN_REPORT)
    benchmark_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.benchmark_report.to_json_dict()), category=ArtifactCategory.BENCHMARK_REPORT)
    cost_sensitivity_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.cost_sensitivity_report.to_json_dict()), category=ArtifactCategory.COST_SENSITIVITY_REPORT)
    bucket_ref = artifact_store.write_artifact(canonical_json_bytes(recomputed.bucket_report.to_json_dict()), category=ArtifactCategory.BUCKET_ANALYSIS_REPORT)

    closed = recomputed.trade_set.closed_trades
    result = OuterFoldBacktestResult(
        schema_version=OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION, backtest_id=backtest_id, outer_fold_index=outer_fold.fold_index,
        signal_set_reference=signal_ref, trade_set_reference=trade_ref, bar_return_timeline_reference=bar_timeline_ref,
        equity_curve_reference=equity_ref, gross_drawdown_reference=gross_dd_ref,
        net_drawdown_reference=net_dd_ref, benchmark_report_reference=benchmark_ref, cost_sensitivity_report_reference=cost_sensitivity_ref,
        bucket_analysis_report_reference=bucket_ref, outer_test_row_count=len(outer_fold.test_indices), closed_trade_count=len(closed),
        meets_minimum_trade_threshold=len(closed) >= spec.minimum_trades_for_valid_fold, financial_metrics=dict(recomputed.metrics_report.values),
        skipped_metrics=dict(recomputed.metrics_report.skipped), evaluated_at=format_utc_timestamp(utc_now()),
    )
    written = [signal_ref, trade_ref, bar_timeline_ref, equity_ref, gross_dd_ref, net_dd_ref, benchmark_ref, cost_sensitivity_ref, bucket_ref]
    return result, written


# --------------------------------------------------------------------------
# Source resolution: BacktestSpec -> concrete, per-outer-fold inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ResolvedBacktestInputs:
    timeline: pd.DataFrame
    bars: pd.DataFrame
    outer_folds: tuple[Fold, ...]
    calibration_manifest: CalibrationManifest
    calibration_spec: CalibrationSpec


def resolve_backtest_inputs(
    spec: BacktestSpec, *, calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
    execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore, research_dataset_store: ResearchDatasetStore,
    dataset_loader: DatasetLoader, artifact_store: MLArtifactStore,
) -> ResolvedBacktestInputs:
    """Resolves `BacktestSpec.source_calibration_id` (and transitively
    `.source_experiment_id`/`.source_execution_id`) into concrete data.
    Every identity-bearing `BacktestSpec` field is cross-checked against
    what was actually loaded -- an inconsistency fails closed
    (`BacktestValidationError`), mirroring `calibration.runner.
    resolve_calibration_inputs`'s identical discipline. Additionally
    verifies `ExecutionManifest.stage is ExecutionStage.COMPLETED` -- a
    genuinely NEW integrity check beyond what calibration itself requires
    (see `backtesting.specs.BacktestSpec.source_execution_id`'s own
    docstring for why this check belongs here)."""
    calibration_manifest = calibration_manifest_store.load(spec.source_calibration_id)
    if calibration_manifest.stage is not CalibrationStage.COMPLETED:
        raise BacktestValidationError(
            f"BacktestSpec.source_calibration_id={spec.source_calibration_id!r} has not reached "
            f"CalibrationStage.COMPLETED (currently {calibration_manifest.stage.value!r}) -- refusing to "
            "backtest against an in-progress or failed calibration"
        )
    if calibration_manifest.source_experiment_id != spec.source_experiment_id:
        raise BacktestValidationError(
            f"BacktestSpec.source_experiment_id ({spec.source_experiment_id!r}) does not match the source "
            f"calibration's recorded source_experiment_id ({calibration_manifest.source_experiment_id!r})"
        )
    if calibration_manifest.spec_reference is None:
        raise BacktestValidationError(f"Source calibration {spec.source_calibration_id!r} has no recorded CalibrationSpec artifact")
    calibration_spec_raw = artifact_store.read_artifact(calibration_manifest.spec_reference.content_hash)
    calibration_spec = CalibrationSpec.from_json_dict(parse_json_strict(calibration_spec_raw.decode("utf-8")))
    if compute_calibration_identity(calibration_spec).calibration_id != spec.source_calibration_id:
        raise BacktestValidationError("Source calibration's persisted CalibrationSpec does not reproduce its own recorded calibration_id")

    execution_manifest = execution_manifest_store.load(spec.source_execution_id)
    if execution_manifest.stage is not ExecutionStage.COMPLETED:
        raise BacktestValidationError(
            f"BacktestSpec.source_execution_id={spec.source_execution_id!r} has not reached "
            f"ExecutionStage.COMPLETED (currently {execution_manifest.stage.value!r}) -- refusing to backtest "
            "against an in-progress or failed execution run"
        )

    experiment_manifest = experiment_manifest_store.load(spec.source_experiment_id)
    experiment_spec = experiment_manifest.spec
    if experiment_spec.dataset_binding.content_id != spec.dataset_content_id:
        raise BacktestValidationError(
            f"BacktestSpec.dataset_content_id ({spec.dataset_content_id!r}) does not match the source "
            f"experiment's bound dataset content id ({experiment_spec.dataset_binding.content_id!r})"
        )
    if experiment_spec.dataset_binding.symbol != spec.instrument_identity:
        raise BacktestValidationError(
            f"BacktestSpec.instrument_identity ({spec.instrument_identity!r}) does not match the source "
            f"experiment's bound dataset symbol ({experiment_spec.dataset_binding.symbol!r})"
        )
    if experiment_spec.dataset_binding.base_timeframe != spec.bar_interval.value:
        raise BacktestValidationError(
            f"BacktestSpec.bar_interval ({spec.bar_interval.value!r}) does not match the source experiment's "
            f"bound dataset base_timeframe ({experiment_spec.dataset_binding.base_timeframe!r})"
        )

    dataset_manifest = research_manifest_store.load(experiment_spec.dataset_binding.dataset_id, experiment_spec.dataset_binding.manifest_version)
    assert_preprocessing_is_safe_for_execution(dataset_manifest)
    label_horizon_bars = extract_label_horizon_bars(dataset_manifest)
    timeline = reconstruct_dataset_timeline(
        research_dataset_store, dataset_id=experiment_spec.dataset_binding.dataset_id, content_id=experiment_spec.dataset_binding.content_id,
        timestamp_column=_TIMESTAMP_COLUMN,
    )

    outer_plan = build_folds_from_split_binding(experiment_spec.split_binding, timeline[_TIMESTAMP_COLUMN], label_horizon_bars=label_horizon_bars)
    plan_validation = validate_fold_plan(outer_plan, timeline=timeline, timestamp_column=_TIMESTAMP_COLUMN)
    if not plan_validation.is_ready:
        blocking = [*plan_validation.criticals, *plan_validation.errors]
        summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
        raise BacktestValidationError(f"BacktestRunner: outer fold plan validation failed: {summary}")
    plan_fingerprint = fingerprint_json(experiment_spec.split_binding.to_json_dict())
    if plan_fingerprint != spec.split_plan_fingerprint:
        raise BacktestValidationError(
            f"BacktestSpec.split_plan_fingerprint ({spec.split_plan_fingerprint!r}) does not match the "
            f"recomputed split binding fingerprint ({plan_fingerprint!r})"
        )

    bars = resolve_market_bars_for_timeline(
        dataset_loader, symbol=experiment_spec.dataset_binding.symbol, base_timeframe=experiment_spec.dataset_binding.base_timeframe,
        timeline=timeline, timestamp_column=spec.timestamp_column,
    )

    return ResolvedBacktestInputs(timeline=timeline, bars=bars, outer_folds=outer_plan.folds, calibration_manifest=calibration_manifest, calibration_spec=calibration_spec)


# --------------------------------------------------------------------------
# Milestone 5.2, Section 6: hard-kill lock recovery
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BacktestLockDiagnostics:
    """Everything `recover-backtest-lock` inspects and shows an operator
    BEFORE any recovery action is taken -- lock age, owner PID/host,
    manifest state, run identity, and a lightweight artifact-consistency
    check. `owner_pid_alive` is `None` whenever liveness could not be
    confidently determined (the common case on Windows, per this
    section's own explicit caveat -- see `_best_effort_pid_alive`) and is
    NEVER treated as "confirmed dead" by the recovery decision below."""

    backtest_id: str
    lock_path: str
    lock_exists: bool
    lock_age_seconds: float | None
    stale_after_seconds: float
    is_stale_by_age: bool
    owner_pid: int | None
    owner_hostname: str | None
    owner_pid_alive: bool | None
    manifest_exists: bool
    manifest_backtest_id_matches: bool | None
    manifest_stage: str | None
    spec_artifact_readable: bool | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backtest_id": self.backtest_id, "lock_path": self.lock_path, "lock_exists": self.lock_exists,
            "lock_age_seconds": self.lock_age_seconds, "stale_after_seconds": self.stale_after_seconds,
            "is_stale_by_age": self.is_stale_by_age, "owner_pid": self.owner_pid, "owner_hostname": self.owner_hostname,
            "owner_pid_alive": self.owner_pid_alive, "manifest_exists": self.manifest_exists,
            "manifest_backtest_id_matches": self.manifest_backtest_id_matches, "manifest_stage": self.manifest_stage,
            "spec_artifact_readable": self.spec_artifact_readable,
        }


def _best_effort_pid_alive(pid: int) -> bool | None:
    """Milestone 5.2, Section 6: NEVER the sole basis for a recovery
    decision -- "if Windows PID liveness is not reliable, use the
    platform's documented age-based policy [instead]". `os.kill(pid, 0)`
    reliably distinguishes alive/dead on POSIX (`ProcessLookupError` /
    silent success); on Windows it has been observed (empirically, during
    this milestone's own implementation) to raise a generic `OSError`
    (WinError 87, "the parameter is incorrect") rather than a specific,
    trustworthy signal for an unowned/nonexistent PID, so any `OSError`
    beyond the two POSIX-specific cases below is reported as `None`
    (undetermined) -- never misreported as a confident "dead"."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    else:
        return True


def _backtest_run_lock_path(ml_artifacts_root: Path, backtest_id: str) -> Path:
    return Path(ml_artifacts_root).resolve() / "backtests" / backtest_id / _BACKTEST_RUN_LOCK_FILE_NAME


def inspect_backtest_lock(backtest_id: str, *, ml_artifacts_root: Path) -> BacktestLockDiagnostics:
    """Milestone 5.2, Section 6: READ-ONLY -- never acquires, contests,
    or removes the lock, and is therefore always safe to call, including
    while a genuinely live run holds it. Inspects: lock age; owner PID;
    owner host; manifest state; run identity (does the manifest's own
    `backtest_id` match the one this lock is filed under); and a
    lightweight artifact-consistency check (can the manifest's own
    recorded `BACKTEST_SPEC` artifact still be read)."""
    root = Path(ml_artifacts_root).resolve()
    lock_path = _backtest_run_lock_path(root, backtest_id)
    lock = DatasetLock(lock_path)
    info: LockInfo | None = lock.peek()

    lock_age_seconds: float | None = None
    is_stale_by_age = False
    owner_pid: int | None = None
    owner_hostname: str | None = None
    owner_pid_alive: bool | None = None
    if info is not None:
        age = pd.Timestamp.now(tz="UTC") - info.acquired_at
        lock_age_seconds = age.total_seconds()
        is_stale_by_age = age > DEFAULT_STALE_AFTER
        owner_pid = info.pid
        owner_hostname = info.hostname
        owner_pid_alive = _best_effort_pid_alive(info.pid)

    manifest = BacktestManifestStore(root).load_if_exists(backtest_id)
    manifest_backtest_id_matches = None if manifest is None else manifest.backtest_id == backtest_id

    spec_artifact_readable: bool | None = None
    if manifest is not None and manifest.spec_reference is not None:
        try:
            MLArtifactStore(root).read_artifact(manifest.spec_reference.content_hash)
            spec_artifact_readable = True
        except _UNVERIFIABLE_ARTIFACT_ERRORS:
            spec_artifact_readable = False

    return BacktestLockDiagnostics(
        backtest_id=backtest_id, lock_path=str(lock_path), lock_exists=lock_path.is_file(), lock_age_seconds=lock_age_seconds,
        stale_after_seconds=DEFAULT_STALE_AFTER.total_seconds(), is_stale_by_age=is_stale_by_age, owner_pid=owner_pid,
        owner_hostname=owner_hostname, owner_pid_alive=owner_pid_alive, manifest_exists=manifest is not None,
        manifest_backtest_id_matches=manifest_backtest_id_matches, manifest_stage=(None if manifest is None else manifest.stage.value),
        spec_artifact_readable=spec_artifact_readable,
    )


def recover_backtest_lock(backtest_id: str, *, ml_artifacts_root: Path, force: bool = False) -> BacktestLockDiagnostics:
    """Milestone 5.2, Section 6: the explicit, diagnosed, auditable
    operator recovery path -- replaces undocumented manual deletion of
    `.backtest_run.lock`. MUST NOT STEAL A LIVE LOCK: refuses (raising
    `BacktestResumeError`, with the same diagnostics this function always
    computes attached as context) unless the lock has gone stale BY AGE
    (the platform's documented, already-implemented `DatasetLock`
    policy -- Windows PID liveness being unreliable is exactly why age,
    not liveness, is the load-bearing signal) or the caller explicitly
    passes `force=True`. `force=True` is meant to be used only AFTER a
    human has reviewed `inspect_backtest_lock`'s diagnostics (e.g. via
    `inspect-backtest-lock`) and independently confirmed the owning
    process is dead -- this function does not and cannot verify that
    claim itself, which is exactly why it requires an explicit flag
    rather than inferring it. Idempotent: recovering an already-absent
    lock is a safe no-op. After recovery, the NEXT `BacktestRunner.run`/
    `.resume` call acquires the lock normally, through the ordinary
    `experiment_lock` path -- this function does not itself acquire
    anything, it only removes an abandoned lock file."""
    root = Path(ml_artifacts_root).resolve()
    diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=root)
    if not diagnostics.lock_exists:
        return diagnostics
    if not diagnostics.is_stale_by_age and not force:
        age_description = (
            "unknown (lock metadata is unreadable/corrupted)" if diagnostics.lock_age_seconds is None else f"only {diagnostics.lock_age_seconds:.0f}s"
        )
        raise BacktestResumeError(
            f"recover_backtest_lock: lock at {diagnostics.lock_path} is {age_description} old "
            f"(stale_after={diagnostics.stale_after_seconds:.0f}s); owner_pid={diagnostics.owner_pid} on "
            f"host={diagnostics.owner_hostname!r} has best-effort liveness={diagnostics.owner_pid_alive!r} -- "
            "refusing to steal a lock that has not gone stale. If the owning process is confirmed dead, "
            "retry with force=True.",
            context={
                "backtest_id": backtest_id, "lock_path": diagnostics.lock_path, "lock_age_seconds": diagnostics.lock_age_seconds,
                "owner_pid": diagnostics.owner_pid, "owner_pid_alive": diagnostics.owner_pid_alive,
            },
        )
    lock_path = _backtest_run_lock_path(root, backtest_id)
    lock_path.unlink(missing_ok=True)
    logger.warning(
        "Backtest run lock recovered: backtest_id=%s lock_path=%s owner_pid=%s stale_by_age=%s forced=%s",
        backtest_id[:12], lock_path, diagnostics.owner_pid, diagnostics.is_stale_by_age, force,
    )
    return diagnostics


# --------------------------------------------------------------------------
# Top-level manifest-driven orchestration
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    manifest: BacktestManifest
    was_idempotent_no_op: bool


class BacktestRunner:
    def __init__(
        self, *, ml_artifacts_root: Path | str, calibration_manifest_store: CalibrationManifestStore,
        experiment_manifest_store: ExperimentManifestStore, execution_manifest_store: ExecutionManifestStore,
        research_manifest_store: ResearchManifestStore, research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
    ) -> None:
        self._root = Path(ml_artifacts_root).resolve()
        self._artifact_store = MLArtifactStore(self._root)
        self._manifest_store = BacktestManifestStore(self._root)
        self._event_store = BacktestEventStore(self._root)
        self._calibration_manifest_store = calibration_manifest_store
        self._experiment_manifest_store = experiment_manifest_store
        self._execution_manifest_store = execution_manifest_store
        self._research_manifest_store = research_manifest_store
        self._research_dataset_store = research_dataset_store
        self._dataset_loader = dataset_loader

    def _run_lock_path(self, backtest_id: str) -> Path:
        return self._root / "backtests" / backtest_id / _BACKTEST_RUN_LOCK_FILE_NAME

    def inspect_lock(self, backtest_id: str) -> BacktestLockDiagnostics:
        """Milestone 5.2, Section 6: convenience wrapper over the
        module-level `inspect_backtest_lock`, using this runner's own
        `ml_artifacts_root` -- read-only, safe to call at any time."""
        return inspect_backtest_lock(backtest_id, ml_artifacts_root=self._root)

    def recover_lock(self, backtest_id: str, *, force: bool = False) -> BacktestLockDiagnostics:
        """Milestone 5.2, Section 6: convenience wrapper over the
        module-level `recover_backtest_lock`, using this runner's own
        `ml_artifacts_root`. See that function's docstring for the full
        "must not steal a live lock" policy."""
        return recover_backtest_lock(backtest_id, ml_artifacts_root=self._root, force=force)

    def run(self, spec: BacktestSpec) -> BacktestOutcome:
        identity = compute_backtest_identity(spec)
        with experiment_lock(self._run_lock_path(identity.backtest_id)):
            return self._run_locked(spec, identity.backtest_id, require_existing=False)

    def resume(self, backtest_id: str, *, spec: BacktestSpec | None = None) -> BacktestOutcome:
        from quant_platform.backtesting.resume import require_backtest_resumable

        existing = self._manifest_store.load_if_exists(backtest_id)
        if existing is not None and existing.stage is BacktestStage.COMPLETED:
            return BacktestOutcome(manifest=existing, was_idempotent_no_op=True)
        require_backtest_resumable(existing, backtest_id=backtest_id)
        assert existing is not None
        resolved_spec = spec if spec is not None else self._load_spec_artifact(existing)
        # Milestone 5.1, Section 6: "source identity changes invalidate
        # resume" -- an explicitly-passed `spec` must reproduce the SAME
        # backtest_id being resumed. Without this check, a caller passing
        # a spec that diverges from what this manifest actually recorded
        # (e.g. a different cost model) would silently resume UNDER a
        # foreign identity, never caught until a later, separate
        # `verify_backtest` call -- rejected here, at resume time, instead.
        resolved_identity = compute_backtest_identity(resolved_spec)
        if resolved_identity.backtest_id != backtest_id:
            raise BacktestResumeError(
                f"resume: the provided BacktestSpec reproduces backtest_id={resolved_identity.backtest_id!r}, "
                f"which does not match the backtest_id being resumed ({backtest_id!r}) -- refusing to resume "
                "under a mismatched source identity",
                context={"backtest_id": backtest_id, "resolved_backtest_id": resolved_identity.backtest_id},
            )
        self._require_compatible_environment(existing)
        with experiment_lock(self._run_lock_path(backtest_id)):
            return self._run_locked(resolved_spec, backtest_id, require_existing=True)

    def _require_compatible_environment(self, manifest: BacktestManifest) -> None:
        """This runner performs no model fitting/refitting -- the
        financial-evaluation computations here (signal mapping, fills,
        cost arithmetic, metrics) are pure Python/NumPy with no
        library-version-sensitive numerics comparable to scikit-learn's
        calibrator `.fit()`, so unlike `CalibrationRunner`'s identically-
        named method, there is no library version this needs to gate
        resumption on. Kept as an explicit no-op (not simply omitted) so
        the parallel to `calibration.runner.CalibrationRunner._require_
        compatible_environment` is visible, and so a future genuinely
        version-sensitive dependency has an obvious place to be added."""
        _ = manifest

    def _load_spec_artifact(self, manifest: BacktestManifest) -> BacktestSpec:
        if manifest.spec_reference is None:
            raise BacktestResumeError(f"Backtest {manifest.backtest_id!r} has no recorded BACKTEST_SPEC artifact to resume from")
        raw = self._artifact_store.read_artifact(manifest.spec_reference.content_hash)
        return BacktestSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))

    def _run_locked(self, spec: BacktestSpec, backtest_id: str, *, require_existing: bool) -> BacktestOutcome:
        manifest = self._manifest_store.load_if_exists(backtest_id)
        if require_existing and manifest is None:
            raise BacktestResumeError(f"No backtest manifest exists for backtest_id={backtest_id!r}")  # pragma: no cover - guarded identically before locking

        if manifest is not None and manifest.stage in (BacktestStage.COMPLETED, BacktestStage.FAILED):
            if manifest.stage is BacktestStage.COMPLETED:
                return BacktestOutcome(manifest=manifest, was_idempotent_no_op=True)
            raise BacktestResumeError(
                f"Backtest {backtest_id!r} already reached terminal stage {manifest.stage.value!r}",
                context={"backtest_id": backtest_id, "stage": manifest.stage.value},
            )

        if manifest is None:
            now = format_utc_timestamp(utc_now())
            spec_ref = self._artifact_store.write_artifact(canonical_json_bytes(spec.to_json_dict()), category=ArtifactCategory.BACKTEST_SPEC)
            env_snapshot: EnvironmentSnapshot = capture_environment_snapshot()
            env_ref = self._artifact_store.write_artifact(canonical_json_bytes(env_snapshot.to_json_dict()), category=ArtifactCategory.ENVIRONMENT_SNAPSHOT)
            manifest = BacktestManifest(
                schema_version=BACKTEST_MANIFEST_SCHEMA_VERSION, backtest_id=backtest_id, source_calibration_id=spec.source_calibration_id,
                stage=BacktestStage.CREATED, created_at=now, updated_at=now, spec_reference=spec_ref, environment_snapshot_reference=env_ref,
                artifact_references=(spec_ref, env_ref),
            )
            self._manifest_store.create(manifest)
            self._event_store.append(backtest_id, BacktestEventType.BACKTEST_CREATED)
            self._event_store.append(backtest_id, BacktestEventType.RUN_STARTED)
        else:
            manifest = self._manifest_store.bump_resume_count(backtest_id)
            self._event_store.append(backtest_id, BacktestEventType.BACKTEST_RESUMED, details={"resume_count": manifest.resume_count})

        try:
            manifest = self._execute_pipeline(spec, manifest)
        except QuantPlatformError as exc:
            # Mirrors `CalibrationRunner._run_locked`'s exact reasoning
            # (see that method's own comment): every domain exception
            # raised while a backtest is in progress must leave an
            # accurate, diagnosable FAILED record with a persisted reason
            # -- this is the lesson carried forward from the Milestone 4E
            # audit, where `CalibrationRunner._fail` was found DEFINED but
            # never actually CALLED (making FAILED unreachable). Wired in
            # from the initial implementation here, not discovered later.
            # `ExperimentLockError` is excluded for the identical reason
            # `CalibrationRunner._run_locked` excludes it: it signals lock
            # contention / an aborted process, never a genuine backtest-
            # domain failure about this backtest's own data/config.
            if not isinstance(exc, ExperimentLockError):
                self._fail(backtest_id, str(exc))
            raise
        return BacktestOutcome(manifest=manifest, was_idempotent_no_op=False)

    def _fail(self, backtest_id: str, failure_summary: str) -> None:
        now = format_utc_timestamp(utc_now())
        self._manifest_store.transition(backtest_id, new_stage=BacktestStage.FAILED, updated_at=now, completed_at=now, failure_summary=failure_summary)
        self._event_store.append(backtest_id, BacktestEventType.RUN_FAILED, details={"reason": failure_summary[:200]})

    def _execute_pipeline(self, spec: BacktestSpec, manifest: BacktestManifest) -> BacktestManifest:
        backtest_id = manifest.backtest_id
        inputs = resolve_backtest_inputs(
            spec, calibration_manifest_store=self._calibration_manifest_store, experiment_manifest_store=self._experiment_manifest_store,
            execution_manifest_store=self._execution_manifest_store, research_manifest_store=self._research_manifest_store,
            research_dataset_store=self._research_dataset_store, dataset_loader=self._dataset_loader, artifact_store=self._artifact_store,
        )

        if manifest.stage is BacktestStage.CREATED:
            manifest = self._manifest_store.transition(
                backtest_id, new_stage=BacktestStage.SOURCES_VERIFIED, updated_at=format_utc_timestamp(utc_now()),
                total_outer_folds=len(inputs.outer_folds), current_outer_fold_index=inputs.outer_folds[0].fold_index if inputs.outer_folds else None,
            )

        from quant_platform.backtesting.resume import verify_completed_backtest_outer_folds

        completed = set(manifest.completed_outer_fold_indices)
        verified_complete, needs_rerun = verify_completed_backtest_outer_folds(manifest, artifact_store=self._artifact_store)
        if needs_rerun:
            logger.warning(
                "Backtest %s: outer fold(s) %s were recorded as completed but their artifacts could not be "
                "verified -- re-running them", backtest_id[:12], sorted(needs_rerun),
            )
            completed -= needs_rerun

        results: list[OuterFoldBacktestResult] = []
        for fold in inputs.outer_folds:
            if fold.fold_index in completed and fold.fold_index in verified_complete:
                reference = manifest.outer_fold_result_references[fold.fold_index]
                raw = self._artifact_store.read_artifact(reference.content_hash)
                results.append(OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8"))))
                continue

            manifest = self._manifest_store.transition(
                backtest_id, new_stage=BacktestStage.SOURCES_VERIFIED, updated_at=format_utc_timestamp(utc_now()), current_outer_fold_index=fold.fold_index,
            ) if manifest.stage is not BacktestStage.SOURCES_VERIFIED else manifest
            self._event_store.append(backtest_id, BacktestEventType.OUTER_FOLD_STARTED, details={"outer_fold_index": fold.fold_index})
            self._event_store.append(backtest_id, BacktestEventType.SOURCES_VERIFIED, details={"outer_fold_index": fold.fold_index})

            # `run_outer_fold_backtest` computes signals through bucket
            # analysis as one atomic, pure function of already-fixed
            # inputs -- exactly `calibration.runner`'s own precedent for
            # why redoing the whole fold from scratch after a crash is
            # always safe. A crash during this call simply leaves the
            # manifest at `SOURCES_VERIFIED`; resume redoes the WHOLE
            # fold, bit-for-bit identically.
            result, written_refs = run_outer_fold_backtest(
                spec=spec, backtest_id=backtest_id, outer_fold=fold, timeline=inputs.timeline, bars=inputs.bars,
                calibration_manifest=inputs.calibration_manifest, calibration_spec=inputs.calibration_spec, artifact_store=self._artifact_store,
            )
            for stage, event_type in (
                (BacktestStage.SIGNALS_READY, BacktestEventType.SIGNALS_GENERATED),
                (BacktestStage.FILLS_READY, BacktestEventType.FILLS_COMPUTED),
                (BacktestStage.TRADES_READY, BacktestEventType.TRADES_CONSTRUCTED),
                (BacktestStage.RETURNS_READY, BacktestEventType.RETURNS_COMPUTED),
                (BacktestStage.METRICS_READY, BacktestEventType.METRICS_COMPUTED),
            ):
                manifest = self._manifest_store.transition(backtest_id, new_stage=stage, updated_at=format_utc_timestamp(utc_now()))
                self._event_store.append(backtest_id, event_type, details={"outer_fold_index": fold.fold_index})

            result_ref = self._artifact_store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.OUTER_FOLD_BACKTEST_RESULT)
            completed.add(fold.fold_index)
            manifest = self._manifest_store.transition(
                backtest_id, new_stage=BacktestStage.REPORTS_READY, updated_at=format_utc_timestamp(utc_now()),
                completed_outer_fold_indices=tuple(sorted(completed)), outer_fold_result_references={**manifest.outer_fold_result_references, fold.fold_index: result_ref},
                artifact_references=tuple({*manifest.artifact_references, *written_refs, result_ref}),
            )
            self._event_store.append(backtest_id, BacktestEventType.OUTER_FOLD_REPORT_READY, details={"outer_fold_index": fold.fold_index})
            self._event_store.append(backtest_id, BacktestEventType.OUTER_FOLD_COMPLETED, details={"outer_fold_index": fold.fold_index})
            results.append(result)

            is_last = fold.fold_index == inputs.outer_folds[-1].fold_index
            if not is_last:
                manifest = self._manifest_store.transition(backtest_id, new_stage=BacktestStage.SOURCES_VERIFIED, updated_at=format_utc_timestamp(utc_now()))

        if manifest.stage is BacktestStage.REPORTS_READY:
            manifest = self._manifest_store.transition(backtest_id, new_stage=BacktestStage.VERIFIED, updated_at=format_utc_timestamp(utc_now()))
        self._event_store.append(backtest_id, BacktestEventType.RUN_VERIFIED)

        # Milestone 5.1, Section 3: the stitched, whole-backtest
        # chronological equity curve -- built (and re-built on every
        # resume/re-run, deterministically, never partially trusted from a
        # prior attempt) from every outer fold's own persisted
        # `BarReturnTimeline`, strictly in `outer_fold_index` order.
        ordered_results = sorted(results, key=lambda r: r.outer_fold_index)
        bar_timelines: list[BarReturnTimeline] = []
        for r in ordered_results:
            raw_timeline = self._artifact_store.read_artifact(r.bar_return_timeline_reference.content_hash)
            bar_timelines.append(BarReturnTimeline.from_json_dict(parse_json_strict(raw_timeline.decode("utf-8"))))
        stitched = build_stitched_walk_forward_equity(backtest_id=backtest_id, timelines=bar_timelines)
        stitched_ref = self._artifact_store.write_artifact(canonical_json_bytes(stitched.to_json_dict()), category=ArtifactCategory.STITCHED_WALK_FORWARD_EQUITY)

        from quant_platform.backtesting.reporting import build_backtest_report_json

        report_json = build_backtest_report_json(
            manifest, spec=spec, outer_fold_results=results, stitched=stitched, stitched_equity_reference=stitched_ref,
        )
        report_ref = self._artifact_store.write_artifact(canonical_json_bytes(report_json), category=ArtifactCategory.BACKTEST_REPORT)

        completed_at = format_utc_timestamp(utc_now())
        manifest = self._manifest_store.transition(
            backtest_id, new_stage=BacktestStage.COMPLETED, updated_at=completed_at, completed_at=completed_at,
            stitched_equity_reference=stitched_ref, aggregate_report_reference=report_ref,
        )
        self._event_store.append(backtest_id, BacktestEventType.RUN_COMPLETED)
        return manifest


__all__ = [
    "OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION",
    "BacktestLockDiagnostics",
    "BacktestOutcome",
    "BacktestRunner",
    "BenchmarkReport",
    "BenchmarkResult",
    "BucketAnalysisReport",
    "BucketResult",
    "CostSensitivityReport",
    "CostSensitivityResult",
    "OuterFoldBacktestResult",
    "RecomputedOuterFoldArtifacts",
    "ResolvedBacktestInputs",
    "compute_benchmark_report",
    "compute_bucket_analysis_report",
    "compute_cost_sensitivity_report",
    "inspect_backtest_lock",
    "recompute_outer_fold_backtest_artifacts",
    "recover_backtest_lock",
    "resolve_backtest_inputs",
    "resolve_market_bars_for_timeline",
    "run_outer_fold_backtest",
    "verify_and_load_predictions",
]
