"""Shared enums, the market-bar contract, the verified-prediction
contract, and the `BacktestStage` state machine for the leakage-safe
financial evaluation / backtesting framework (Milestone 5).

WHY THIS PACKAGE DOES NOT REUSE `quant_platform.engine`/`costs`/`risk`/
`strategy`
--------------------------------------------------------------------------
This platform already has a bar-by-bar, cursor-driven backtest engine
(`engine.backtest_engine.BacktestEngine`, `multiframe.cursor.TimeframeCursor`,
`costs.models.CostModel` as a mutable ABC, `risk.position_sizing`,
`strategy.interfaces.Strategy`) from the platform's original milestone --
built for a RULE-BASED strategy that receives bars one at a time and
maintains mutable `Position`/`Portfolio` state. That design is a poor fit
here: this milestone evaluates an ALREADY-COMPUTED, already-verified
outer-fold prediction SERIES (from `calibration.runner.
OuterFoldCalibrationResult`) against historical bars, deterministically,
with content-addressed immutable specs, resumable/verifiable artifacts,
and no mutable global state -- exactly the shape `calibration`/
`optimization`/`execution` (Milestones 4A-4E) already established, not the
shape the original engine established. Reusing the mutable ABC-based
`CostModel`/`Portfolio` machinery here would force an awkward, leaky
adapter between two incompatible design philosophies for no real benefit;
instead this package reuses the platform primitives that ARE genuinely
shape-compatible: `core.types.Timeframe`, `historical.loader.DatasetLoader`
(for raw OHLCV bars), `execution.splitters.Fold`/
`build_folds_from_split_binding` (for the SAME outer-fold boundaries
calibration used), `calibration.models.DeterminismPolicy`/`Decision`
(reused directly, never redefined), and every `ml.*` persistence/artifact/
locking/fingerprint primitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.calibration.models import Decision
from quant_platform.core.exceptions import MarketDataBindingError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import require_utc
from quant_platform.ml.persistence import as_json_list, parse_utc_timestamp, require_schema_version

VERIFIED_PREDICTION_SET_SCHEMA_VERSION = 1
"""`Decision` (POSITIVE/NEGATIVE/ABSTAIN) is re-exported, unchanged, from
`calibration.models` -- this package never redefines the abstain-aware
decision vocabulary calibration already established."""


# --------------------------------------------------------------------------
# Position / signal-mapping / overlap / entry / exit / cost enums
# --------------------------------------------------------------------------
class PositionDirection(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class PositionMode(Enum):
    """Section 9: which non-flat directions this backtest may ever hold."""

    LONG_FLAT = "long_flat"
    LONG_SHORT = "long_short"


class SignalMappingPolicyKind(Enum):
    """Section 8."""

    DIRECTIONAL_LONG_FLAT = "directional_long_flat"
    """A: predicted positive -> long, negative -> flat."""
    DIRECTIONAL_LONG_SHORT = "directional_long_short"
    """B: predicted positive -> long, negative -> short."""
    PROBABILITY_BANDS = "probability_bands"
    """C: calibrated-probability bands map to long/flat/short."""
    ABSTENTION_AWARE = "abstention_aware"
    """D: calibration's own abstain decision maps directly to flat/reject."""
    CONFIDENCE_FLOOR = "confidence_floor"
    """E: reject (flat) below a declared confidence floor."""
    UNCERTAINTY_CEILING = "uncertainty_ceiling"
    """F: reject (flat) above a declared uncertainty ceiling."""
    COMBINED_CONFIDENCE_UNCERTAINTY = "combined_confidence_uncertainty"
    """G: both floors/ceilings applied together."""


class OverlapPolicyKind(Enum):
    """Section 9: behavior when a new opposing (or same-direction) signal
    arrives while a position is already open."""

    IGNORE = "ignore"
    CLOSE_AND_REVERSE = "close_and_reverse"
    CLOSE_ONLY = "close_only"
    QUEUE = "queue"
    INDEPENDENT_OVERLAPPING = "independent_overlapping"


class EntryPolicyKind(Enum):
    """Section 10."""

    NEXT_BAR_OPEN = "next_bar_open"
    NEXT_BAR_MID = "next_bar_mid"
    NEXT_BAR_SIDE_AWARE = "next_bar_side_aware"
    DELAYED_BAR = "delayed_bar"


class ExitPolicyKind(Enum):
    """Section 11."""

    FIXED_HORIZON = "fixed_horizon"
    NEXT_BAR_CLOSE = "next_bar_close"
    END_OF_FOLD = "end_of_fold"
    OPPOSITE_SIGNAL = "opposite_signal"


class FinalTradePolicyKind(Enum):
    """Section 11: behavior for a trade still open when the outer-test
    partition ends."""

    DISCARD_INCOMPLETE = "discard_incomplete"
    FORCE_CLOSE_AT_FINAL_PRICE = "force_close_at_final_price"
    MARK_INCOMPLETE_EXCLUDE = "mark_incomplete_exclude"


class DecisionTimestampPolicyKind(Enum):
    """Section 7. Milestone 5.1, Section 5: exactly ONE of these three is
    actually IMPLEMENTABLE and enforced from this platform's current
    source artifacts -- `BacktestSpec.__post_init__` rejects the other two
    at construction time (fail-closed, not silently ignored) rather than
    accepting them as decorative configuration. See `SUPPORTED_DECISION_
    TIMESTAMP_POLICIES` and `prediction_availability_timestamp`."""

    AFTER_BAR_CLOSE = "after_bar_close"
    """SUPPORTED. A prediction/signal derived from bar N's data is only
    considered "knowable" at bar N's CLOSE (`prediction_availability_
    timestamp` below) -- never before. Enforced two ways: (1) `Verified
    PredictionSet.timestamps`/`Signal.decision_timestamp` are computed as
    bar N's CLOSE, not its open (`runner.verify_and_load_predictions`);
    (2) `BacktestSpec.__post_init__` requires `entry_spec.delay_bars >= 1`
    under this policy -- the only value for which EVERY `EntryPolicyKind`
    convention (which all price off the TARGET bar's own open/mid/side-
    aware value) necessarily lands at or after the signal bar's close,
    given this platform's gapless, contiguous bar series."""
    BEFORE_NEXT_BAR_OPEN = "before_next_bar_open"
    """UNSUPPORTED -- rejected at `BacktestSpec` construction. Would
    require knowing WHEN, strictly between bar N's open and its close, a
    prediction became available (sub-bar timestamp granularity) -- this
    platform's market-bar contract (Section 6) carries only one timestamp
    per bar (`open_time`; OHLC aggregates have no recorded sub-bar
    instants), so this policy cannot be distinguished from, or verified
    against, `AFTER_BAR_CLOSE` from any artifact this platform actually
    produces. Silently treating it as equivalent to `AFTER_BAR_CLOSE`
    would be exactly the "decorative policy" defect Section 5 corrects."""
    EXTERNALLY_TIMESTAMPED = "externally_timestamped"
    """UNSUPPORTED -- rejected at `BacktestSpec` construction. Implies an
    externally-supplied decision timestamp independent of the source
    bar's own timing (e.g. real-world signal-generation/publication
    latency) -- neither `VerifiedPredictionSet` nor `Signal` carries any
    such external timestamp field distinct from the bar's own timestamp;
    nothing in this platform's current source artifacts provides it, so
    honoring this policy would fabricate unearned timing precision."""


SUPPORTED_DECISION_TIMESTAMP_POLICIES = frozenset({DecisionTimestampPolicyKind.AFTER_BAR_CLOSE})


def prediction_availability_timestamp(bar_open_time: pd.Timestamp, *, bar_interval: Timeframe, policy: DecisionTimestampPolicyKind) -> pd.Timestamp:
    """Milestone 5.1, Section 5: THE single, named place "when does a
    prediction derived from this bar become knowable" is computed --
    never inlined ad hoc at each call site. Only `AFTER_BAR_CLOSE` is
    supported (guaranteed by `BacktestSpec.__post_init__`'s own
    construction-time rejection of the other two policies); called with
    any other policy is an internal-consistency violation (a policy that
    should have been rejected far earlier reached this function), so it
    raises rather than silently guessing a timestamp."""
    if policy is not DecisionTimestampPolicyKind.AFTER_BAR_CLOSE:
        raise MarketDataBindingError(
            f"prediction_availability_timestamp: {policy.value!r} is not a supported DecisionTimestampPolicyKind "
            "-- this should have been rejected at BacktestSpec construction"
        )
    return bar_open_time + bar_interval.duration


class PriceBasisKind(Enum):
    CLOSE = "close"
    MID = "mid"
    BID_ASK = "bid_ask"


class SpreadModelKind(Enum):
    """Section 12."""

    ZERO = "zero"
    FIXED_PRICE_UNITS = "fixed_price_units"
    FIXED_BASIS_POINTS = "fixed_basis_points"
    BID_ASK_OBSERVED = "bid_ask_observed"


class CommissionModelKind(Enum):
    ZERO = "zero"
    PER_SIDE_BASIS_POINTS = "per_side_basis_points"
    FIXED_PER_TRADE = "fixed_per_trade"


class SlippageModelKind(Enum):
    ZERO = "zero"
    FIXED_BASIS_POINTS = "fixed_basis_points"
    FIXED_PRICE_UNITS = "fixed_price_units"


class FinancingModelKind(Enum):
    NONE = "none"
    FIXED_DAILY_BASIS_POINTS = "fixed_daily_basis_points"


class CompoundingPolicyKind(Enum):
    NON_COMPOUNDED = "non_compounded"
    COMPOUNDED = "compounded"


class ReturnCalculationPolicyKind(Enum):
    SIMPLE = "simple"
    LOG = "log"


class TradeStatus(Enum):
    CLOSED = "closed"
    INCOMPLETE_DISCARDED = "incomplete_discarded"
    INCOMPLETE_FORCE_CLOSED = "incomplete_force_closed"
    INCOMPLETE_EXCLUDED = "incomplete_excluded"


class SignalReasonCode(Enum):
    """Section 8's exact, closed reason-code list."""

    ACCEPTED_POSITIVE = "accepted_positive"
    ACCEPTED_NEGATIVE = "accepted_negative"
    ABSTAINED_BY_CALIBRATION_POLICY = "abstained_by_calibration_policy"
    BELOW_CONFIDENCE_FLOOR = "below_confidence_floor"
    ABOVE_UNCERTAINTY_CEILING = "above_uncertainty_ceiling"
    UNSUPPORTED_CLASS = "unsupported_class"
    MISSING_MARKET_BAR = "missing_market_bar"
    INVALID_PREDICTION = "invalid_prediction"
    OVERLAP_POLICY_REJECTION = "overlap_policy_rejection"


class ExitReasonCode(Enum):
    FIXED_HORIZON_REACHED = "fixed_horizon_reached"
    NEXT_BAR_CLOSE = "next_bar_close"
    END_OF_FOLD_FORCED_CLOSE = "end_of_fold_forced_close"
    OPPOSITE_SIGNAL = "opposite_signal"
    DISCARDED_INCOMPLETE = "discarded_incomplete"


# --------------------------------------------------------------------------
# BacktestStage state machine (Section 28) -- mirrors
# `calibration.models.CalibrationStage` exactly: one atomic per-outer-fold
# pipeline (compute signals -> fills -> trades -> returns -> metrics ->
# reports as one deterministic, pure function of already-fixed inputs), so
# every mid-fold stage has a legal restart edge back to
# `SOURCES_VERIFIED` rather than a dedicated `RECOVERABLE_FAILURE` stage.
# --------------------------------------------------------------------------
class BacktestStage(Enum):
    CREATED = "created"
    SOURCES_VERIFIED = "sources_verified"
    SIGNALS_READY = "signals_ready"
    FILLS_READY = "fills_ready"
    TRADES_READY = "trades_ready"
    RETURNS_READY = "returns_ready"
    METRICS_READY = "metrics_ready"
    REPORTS_READY = "reports_ready"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_BACKTEST_STAGES: frozenset[BacktestStage] = frozenset({BacktestStage.COMPLETED, BacktestStage.FAILED})
_MID_FOLD_BACKTEST_STAGES: frozenset[BacktestStage] = frozenset({
    BacktestStage.SIGNALS_READY, BacktestStage.FILLS_READY, BacktestStage.TRADES_READY,
    BacktestStage.RETURNS_READY, BacktestStage.METRICS_READY, BacktestStage.REPORTS_READY,
})
"""Every stage strictly between `SOURCES_VERIFIED` and the fold-loop-back
point -- each can legally restart straight back to `SOURCES_VERIFIED`,
exactly `CalibrationStage._MID_FOLD_CALIBRATION_STAGES`'s identical
reasoning."""

_LEGAL_BACKTEST_TRANSITIONS: dict[BacktestStage, frozenset[BacktestStage]] = {
    BacktestStage.CREATED: frozenset({BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.SOURCES_VERIFIED: frozenset({BacktestStage.SIGNALS_READY, BacktestStage.FAILED}),
    BacktestStage.SIGNALS_READY: frozenset({BacktestStage.FILLS_READY, BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.FILLS_READY: frozenset({BacktestStage.TRADES_READY, BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.TRADES_READY: frozenset({BacktestStage.RETURNS_READY, BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.RETURNS_READY: frozenset({BacktestStage.METRICS_READY, BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.METRICS_READY: frozenset({BacktestStage.REPORTS_READY, BacktestStage.SOURCES_VERIFIED, BacktestStage.FAILED}),
    BacktestStage.REPORTS_READY: frozenset({
        # Loop back for the NEXT outer fold, or move on to whole-run verification.
        BacktestStage.SOURCES_VERIFIED, BacktestStage.VERIFIED, BacktestStage.FAILED,
    }),
    BacktestStage.VERIFIED: frozenset({BacktestStage.COMPLETED, BacktestStage.FAILED}),
    BacktestStage.COMPLETED: frozenset(),
    BacktestStage.FAILED: frozenset(),
}


def is_legal_backtest_transition(current: BacktestStage, target: BacktestStage) -> bool:
    return target in _LEGAL_BACKTEST_TRANSITIONS[current]


def is_terminal_backtest_stage(stage: BacktestStage) -> bool:
    return stage in TERMINAL_BACKTEST_STAGES


# --------------------------------------------------------------------------
# Market-bar contract (Section 6)
# --------------------------------------------------------------------------
MARKET_BAR_REQUIRED_COLUMNS: tuple[str, ...] = ("open_time", "open", "high", "low", "close")
MARKET_BAR_OPTIONAL_COLUMNS: tuple[str, ...] = ("bid", "ask", "volume", "spread")


def validate_market_bar_frame(df: pd.DataFrame, *, context: str, allow_duplicate_timestamps: bool = False) -> None:
    """Independent validation of a market-bar `DataFrame`'s RELATIONAL
    invariants (Section 6) -- deliberately separate from, and IN ADDITION
    TO, `historical.models.validate_historical_schema` (which checks
    column presence/dtype/timezone but not price relationships). Raises
    `MarketDataBindingError`, never silently repairs or forward-fills."""
    missing = set(MARKET_BAR_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise MarketDataBindingError(f"{context}: market bar frame is missing required column(s): {sorted(missing)}")
    if len(df) == 0:
        raise MarketDataBindingError(f"{context}: market bar frame must not be empty")

    require_utc(df["open_time"], context=context)
    if not df["open_time"].is_monotonic_increasing:
        raise MarketDataBindingError(f"{context}: market bar frame is not in strict chronological order")
    if not allow_duplicate_timestamps and df["open_time"].duplicated().any():
        raise MarketDataBindingError(f"{context}: market bar frame contains duplicate open_time value(s)")

    price_columns = ["open", "high", "low", "close"]
    prices = df[price_columns].to_numpy(dtype="float64")
    if not np.all(np.isfinite(prices)):
        raise MarketDataBindingError(f"{context}: market bar frame contains non-finite price value(s)")
    if np.any(prices <= 0.0):
        raise MarketDataBindingError(f"{context}: market bar frame contains non-positive price value(s)")

    open_, high, low, close = (df[c].to_numpy(dtype="float64") for c in price_columns)
    if np.any(high < np.maximum.reduce([open_, close, low])):
        raise MarketDataBindingError(f"{context}: market bar frame violates high >= max(open, close, low) for at least one bar")
    if np.any(low > np.minimum.reduce([open_, close, high])):
        raise MarketDataBindingError(f"{context}: market bar frame violates low <= min(open, close, high) for at least one bar")

    if "bid" in df.columns and "ask" in df.columns:
        bid = df["bid"].to_numpy(dtype="float64")
        ask = df["ask"].to_numpy(dtype="float64")
        if not (np.all(np.isfinite(bid)) and np.all(np.isfinite(ask))):
            raise MarketDataBindingError(f"{context}: bid/ask contains non-finite value(s)")
        if np.any(ask < bid):
            raise MarketDataBindingError(f"{context}: market bar frame violates ask >= bid for at least one bar")
    if "spread" in df.columns:
        spread = df["spread"].to_numpy(dtype="float64")
        if not np.all(np.isfinite(spread)):
            raise MarketDataBindingError(f"{context}: spread column contains non-finite value(s)")
        if np.any(spread < 0.0):
            raise MarketDataBindingError(f"{context}: spread column contains negative value(s)")


def validate_bar_interval_consistency(df: pd.DataFrame, *, expected_interval: pd.Timedelta, context: str) -> None:
    """Section 6: "no missing bars unless explicitly permitted." Checked
    as an OPT-IN call (not part of `validate_market_bar_frame`, which must
    also validate legitimately-gappy real-world sources like FX weekend
    closures) -- callers that require strict, gap-free bars call this
    separately."""
    if len(df) < 2:
        return
    diffs = df["open_time"].diff().dropna()
    bad = diffs[diffs != expected_interval]
    if not bad.empty:
        raise MarketDataBindingError(
            f"{context}: {len(bad)} bar interval gap(s)/overlap(s) inconsistent with the expected "
            f"interval {expected_interval} -- first at position {bad.index[0]}",
            context={"expected_interval": str(expected_interval), "violation_count": len(bad)},
        )


# --------------------------------------------------------------------------
# Verified prediction contract (Section 5)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VerifiedPredictionSet:
    """One outer fold's independently RE-verified calibrated predictions
    -- the sole input `backtesting.signals` is permitted to read. Built by
    `backtesting.runner.verify_and_load_predictions` from a calibration
    run's persisted `OuterFoldCalibrationResult` (never trusted from a
    filename/hash alone -- every field here is independently re-checked
    against the loaded calibration artifacts, see that function's own
    docstring for exactly what is re-verified).

    `threshold` and `positive_class_label` are single scalars (not
    per-row arrays): calibration freezes exactly ONE decision threshold
    per outer fold (see `calibration.fitting.FrozenDecisionPolicy`), and
    this platform's positive-class convention is fixed platform-wide."""

    schema_version: int
    outer_fold_index: int
    source_calibration_id: str
    source_experiment_id: str
    source_execution_id: str
    base_model_definition_identity: str
    sample_positions: tuple[int, ...]
    timestamps: tuple[str, ...]
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...]
    threshold: float
    decisions: tuple[str, ...]
    abstention_reason_codes: tuple[str, ...]
    confidence_scores: tuple[float, ...]
    confidence_categories: tuple[str, ...]
    uncertainty_scores: tuple[float, ...]
    positive_class_label: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.sample_positions)
        if n == 0:
            raise MarketDataBindingError("VerifiedPredictionSet must contain at least one prediction")
        for name, arr in (
            ("timestamps", self.timestamps), ("raw_probabilities", self.raw_probabilities),
            ("calibrated_probabilities", self.calibrated_probabilities), ("decisions", self.decisions),
            ("abstention_reason_codes", self.abstention_reason_codes), ("confidence_scores", self.confidence_scores),
            ("confidence_categories", self.confidence_categories), ("uncertainty_scores", self.uncertainty_scores),
        ):
            if len(arr) != n:
                raise MarketDataBindingError(f"VerifiedPredictionSet.{name} has length {len(arr)}, expected {n}")
        if self.outer_fold_index < 0:
            raise MarketDataBindingError(f"VerifiedPredictionSet.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if len(set(self.sample_positions)) != n:
            raise MarketDataBindingError("VerifiedPredictionSet.sample_positions must not contain duplicate sample identities")
        if list(self.sample_positions) != sorted(self.sample_positions):
            raise MarketDataBindingError("VerifiedPredictionSet.sample_positions must be strictly ascending")
        if list(self.timestamps) != sorted(self.timestamps):
            raise MarketDataBindingError("VerifiedPredictionSet.timestamps must be non-decreasing in sample_positions order")
        for ts in self.timestamps:
            parse_utc_timestamp(ts)
        if not math.isfinite(self.threshold) or not (0.0 <= self.threshold <= 1.0):
            raise MarketDataBindingError(f"VerifiedPredictionSet.threshold must be a finite value in [0, 1], got {self.threshold!r}")
        for name, arr, lo, hi in (
            ("raw_probabilities", self.raw_probabilities, 0.0, 1.0), ("calibrated_probabilities", self.calibrated_probabilities, 0.0, 1.0),
            ("confidence_scores", self.confidence_scores, 0.0, 1.0), ("uncertainty_scores", self.uncertainty_scores, 0.0, 1.0),
        ):
            for v in arr:
                if not math.isfinite(v) or not (lo <= v <= hi):
                    raise MarketDataBindingError(f"VerifiedPredictionSet.{name}[] must be finite and in [{lo}, {hi}], got {v!r}")
        valid_decisions = frozenset(d.value for d in Decision)
        for d in self.decisions:
            if d not in valid_decisions:
                raise MarketDataBindingError(f"VerifiedPredictionSet.decisions[] contains an invalid decision {d!r}")

    @property
    def n_samples(self) -> int:
        return len(self.sample_positions)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index,
            "source_calibration_id": self.source_calibration_id, "source_experiment_id": self.source_experiment_id,
            "source_execution_id": self.source_execution_id, "base_model_definition_identity": self.base_model_definition_identity,
            "sample_positions": list(self.sample_positions), "timestamps": list(self.timestamps),
            "raw_probabilities": list(self.raw_probabilities), "calibrated_probabilities": list(self.calibrated_probabilities),
            "threshold": self.threshold, "decisions": list(self.decisions),
            "abstention_reason_codes": list(self.abstention_reason_codes), "confidence_scores": list(self.confidence_scores),
            "confidence_categories": list(self.confidence_categories), "uncertainty_scores": list(self.uncertainty_scores),
            "positive_class_label": self.positive_class_label,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> VerifiedPredictionSet:
        require_schema_version(raw, supported=VERIFIED_PREDICTION_SET_SCHEMA_VERSION, context="VerifiedPredictionSet")
        return cls(
            schema_version=VERIFIED_PREDICTION_SET_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            source_calibration_id=str(raw["source_calibration_id"]), source_experiment_id=str(raw["source_experiment_id"]),
            source_execution_id=str(raw["source_execution_id"]), base_model_definition_identity=str(raw["base_model_definition_identity"]),
            sample_positions=tuple(int(v) for v in as_json_list(raw["sample_positions"], field_name="sample_positions")),
            timestamps=tuple(str(v) for v in as_json_list(raw["timestamps"], field_name="timestamps")),
            raw_probabilities=tuple(float(v) for v in as_json_list(raw["raw_probabilities"], field_name="raw_probabilities")),
            calibrated_probabilities=tuple(float(v) for v in as_json_list(raw["calibrated_probabilities"], field_name="calibrated_probabilities")),
            threshold=float(str(raw["threshold"])), decisions=tuple(str(v) for v in as_json_list(raw["decisions"], field_name="decisions")),
            abstention_reason_codes=tuple(str(v) for v in as_json_list(raw["abstention_reason_codes"], field_name="abstention_reason_codes")),
            confidence_scores=tuple(float(v) for v in as_json_list(raw["confidence_scores"], field_name="confidence_scores")),
            confidence_categories=tuple(str(v) for v in as_json_list(raw["confidence_categories"], field_name="confidence_categories")),
            uncertainty_scores=tuple(float(v) for v in as_json_list(raw["uncertainty_scores"], field_name="uncertainty_scores")),
            positive_class_label=float(str(raw.get("positive_class_label", 1.0))),
        )


__all__ = [
    "MARKET_BAR_OPTIONAL_COLUMNS",
    "MARKET_BAR_REQUIRED_COLUMNS",
    "SUPPORTED_DECISION_TIMESTAMP_POLICIES",
    "TERMINAL_BACKTEST_STAGES",
    "VERIFIED_PREDICTION_SET_SCHEMA_VERSION",
    "BacktestStage",
    "CommissionModelKind",
    "CompoundingPolicyKind",
    "Decision",
    "DecisionTimestampPolicyKind",
    "EntryPolicyKind",
    "ExitPolicyKind",
    "ExitReasonCode",
    "FinalTradePolicyKind",
    "FinancingModelKind",
    "OverlapPolicyKind",
    "PositionDirection",
    "PositionMode",
    "PriceBasisKind",
    "ReturnCalculationPolicyKind",
    "SignalMappingPolicyKind",
    "SignalReasonCode",
    "SlippageModelKind",
    "SpreadModelKind",
    "TradeStatus",
    "VerifiedPredictionSet",
    "is_legal_backtest_transition",
    "is_terminal_backtest_stage",
    "prediction_availability_timestamp",
    "validate_bar_interval_consistency",
    "validate_market_bar_frame",
]
