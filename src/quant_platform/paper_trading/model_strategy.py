"""A concrete `strategy.StrategyRuntime` implementation wrapping a REAL,
already-fitted ML model plus point-in-time feature computation (Section
7 defines the `StrategyRuntime` Protocol; this is the only concrete
non-test implementation of it in this repository -- used by the CLI's
`run-paper-session`/`run-shadow-session` commands and by the real
acceptance workflow, Section 33).

`runner.py` always hands `decide()` an EMPTY `StrategyContext`
(`feature_snapshot={}`, `model_output=0.0`) -- see its own module
docstring: "whatever feature computation or model inference produced
`context.model_output`/`context.feature_snapshot` happened entirely
INSIDE the caller-supplied `StrategyRuntime` implementation." This module
is that implementation: it buffers incoming `BarEvent`s itself and
computes both the feature snapshot and the model prediction on every
call, entirely independent of the runner.

FEATURE COMPUTATION REUSES `features.technical.price`'s PURE INDICATOR
FUNCTIONS DIRECTLY, NOT THE FULL `FeatureRegistry`/`FeatureComputationContext`
ORCHESTRATION LAYER: that layer exists for POINT-IN-TIME JOINS across a
base timeframe, higher-timeframe data, cross-asset data, and macro data
during BATCH historical-dataset construction (`features.dataset_builder`)
-- machinery a single-instrument, single-timeframe, incrementally-arriving
bar stream does not need. `_FEATURE_COMPUTERS` calls the SAME validated
pure functions (`log_return`, `candle_body_ratio`, `moving_average_
distance`, `rolling_zscore`) against a small rolling buffer this module
owns -- reusing the formulas, never re-deriving them, while skipping
orchestration this context has no use for. Only a small, explicit,
hand-picked set of feature names is supported (see `_FEATURE_COMPUTERS`)
-- a model trained against a feature outside this set cannot be used
here, a deliberate, documented scope boundary rather than a silently
wrong computation.

NO LOOKAHEAD: every feature value used for a decision at bar N is
computed from bars `[0..N]` only (`trailing_rolling`'s own `center=False`
guarantee, inherited unchanged) -- this module never buffers or reads a
bar past the one currently being decided on. A decision only fires once
the rolling buffer holds enough bars for every selected feature's own
warm-up window; before that, and whenever a computed feature is non-
finite (still warming up), the runtime abstains explicitly rather than
feed a fabricated value into the model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import StrategyRuntimeError
from quant_platform.features.technical.price import (
    candle_body_ratio,
    candle_lower_wick_ratio,
    candle_upper_wick_ratio,
    log_return,
    moving_average_distance,
    rolling_high_low_distance,
    rolling_zscore,
)
from quant_platform.ml.interfaces import FittedModel, ProbabilisticPredictor
from quant_platform.paper_trading.events import BarEvent
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.strategy import StrategyContext, StrategyDecision, create_strategy_decision

_POSITIVE_CLASS_LABEL = 1.0

# feature_name -> (compute_fn(df) -> pd.Series, minimum bars required for a non-NaN last value)
_FEATURE_COMPUTERS: dict[str, tuple[Callable[[pd.DataFrame], pd.Series], int]] = {
    "return_log_1": (lambda df: log_return(df["close"], 1), 2),
    "return_log_5": (lambda df: log_return(df["close"], 5), 6),
    "return_log_10": (lambda df: log_return(df["close"], 10), 11),
    "ma_distance_10": (lambda df: moving_average_distance(df["close"], 10), 10),
    "ma_distance_20": (lambda df: moving_average_distance(df["close"], 20), 20),
    "rolling_zscore_close_10": (lambda df: rolling_zscore(df["close"], 10), 10),
    "rolling_zscore_close_20": (lambda df: rolling_zscore(df["close"], 20), 20),
    "high_low_distance_10": (lambda df: rolling_high_low_distance(df["close"], df["high"], df["low"], 10), 10),
    "high_low_distance_20": (lambda df: rolling_high_low_distance(df["close"], df["high"], df["low"], 20), 20),
    "candle_body_ratio": (lambda df: candle_body_ratio(df["open"], df["high"], df["low"], df["close"]), 1),
    "candle_upper_wick_ratio": (lambda df: candle_upper_wick_ratio(df["open"], df["high"], df["low"], df["close"]), 1),
    "candle_lower_wick_ratio": (lambda df: candle_lower_wick_ratio(df["open"], df["high"], df["low"], df["close"]), 1),
}


def _bars_to_dataframe(bars: list[BarEvent]) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [b.open for b in bars], "high": [b.high for b in bars], "low": [b.low for b in bars], "close": [b.close for b in bars],
    })


def _compute_feature_snapshot(bars: list[BarEvent], feature_names: tuple[str, ...]) -> dict[str, float] | None:
    """Returns `None` (never a fabricated value) if the buffer is too
    short yet, or if any requested feature's freshly-computed value is
    non-finite (still inside its own rolling warm-up window)."""
    required = max(_FEATURE_COMPUTERS[name][1] for name in feature_names)
    if len(bars) < required:
        return None
    df = _bars_to_dataframe(bars)
    snapshot: dict[str, float] = {}
    for name in feature_names:
        compute_fn, _ = _FEATURE_COMPUTERS[name]
        value = float(compute_fn(df).iloc[-1])
        if not pd.notna(value) or not float("-inf") < value < float("inf"):
            return None
        snapshot[name] = value
    return snapshot


@dataclass
class ModelStrategyRuntime:
    """Stateful (buffers bars across calls) -- unlike every immutable
    domain object elsewhere in `paper_trading`, a `StrategyRuntime`
    implementation is explicitly permitted to hold state (Section 7 only
    constrains the DECISION it produces to be immutable, never the
    runtime object itself)."""

    strategy_identity: str
    fitted_model: FittedModel
    feature_names: tuple[str, ...]
    long_threshold: float
    short_threshold: float
    target_quantity: float
    confidence_scaled_sizing: bool = False
    """When `True`, a non-abstain decision's `target_quantity` scales with
    model confidence: `target_quantity * (1.0 + confidence)`, ranging from
    `target_quantity` (confidence at the decision threshold) up to `2 *
    target_quantity` (confidence approaching 1.0) -- a genuine, model-
    driven sizing signal (higher conviction, larger size), not a fabricated
    one. `False` (the default) preserves the original fixed-size behavior."""
    _bar_buffer: list[BarEvent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fitted_model, ProbabilisticPredictor):
            raise StrategyRuntimeError(f"ModelStrategyRuntime requires a model that supports predict_proba/class_labels; {type(self.fitted_model).__name__!r} does not")
        unsupported = [name for name in self.feature_names if name not in _FEATURE_COMPUTERS]
        if unsupported:
            raise StrategyRuntimeError(f"ModelStrategyRuntime: unsupported feature name(s) {unsupported} -- see model_strategy._FEATURE_COMPUTERS for the supported set")
        model_feature_names = set(self.fitted_model.metadata.feature_schema.feature_names)
        if not model_feature_names.issubset(set(self.feature_names)):
            raise StrategyRuntimeError(
                f"ModelStrategyRuntime.feature_names {self.feature_names} does not cover the fitted model's own declared "
                f"feature_schema {tuple(self.fitted_model.metadata.feature_schema.feature_names)}"
            )
        if not (0.5 < self.long_threshold <= 1.0):
            raise StrategyRuntimeError(f"long_threshold must be in (0.5, 1.0], got {self.long_threshold!r}")
        if not (0.0 <= self.short_threshold < 0.5):
            raise StrategyRuntimeError(f"short_threshold must be in [0.0, 0.5), got {self.short_threshold!r}")
        if not (self.target_quantity > 0.0):
            raise StrategyRuntimeError(f"target_quantity must be > 0, got {self.target_quantity!r}")

    def decide(self, context: StrategyContext) -> StrategyDecision:
        event = context.event
        if not isinstance(event, BarEvent):
            return create_strategy_decision(
                strategy_identity=self.strategy_identity, event=event, decision_time=context.decision_time, target_direction=PositionDirection.FLAT,
                target_quantity=0.0, confidence=0.0, uncertainty=1.0, abstain=True, reason_codes=("non_bar_event",),
            )
        self._bar_buffer.append(event)

        feature_snapshot = _compute_feature_snapshot(self._bar_buffer, self.feature_names)
        if feature_snapshot is None:
            return create_strategy_decision(
                strategy_identity=self.strategy_identity, event=event, decision_time=context.decision_time, target_direction=PositionDirection.FLAT,
                target_quantity=0.0, confidence=0.0, uncertainty=1.0, abstain=True, reason_codes=("insufficient_history_or_warmup",),
            )
        feature_snapshot_identity = compute_content_id("paper_trading_feature_snapshot", {k: feature_snapshot[k] for k in sorted(feature_snapshot)})

        assert isinstance(self.fitted_model, ProbabilisticPredictor)  # narrows for type-checking; already validated in __post_init__
        model_feature_names = tuple(self.fitted_model.metadata.feature_schema.feature_names)
        row = pd.DataFrame([feature_snapshot])[list(model_feature_names)]
        probabilities = self.fitted_model.predict_proba(row)[0]
        class_labels = list(self.fitted_model.class_labels)
        positive_index = class_labels.index(_POSITIVE_CLASS_LABEL)
        p_positive = float(probabilities[positive_index])
        model_output_identity = compute_content_id("paper_trading_model_output", {"p_positive": p_positive, "event_id": event.event_id})

        confidence = min(1.0, abs(p_positive - 0.5) * 2.0)
        uncertainty = 1.0 - confidence
        sized_quantity = self.target_quantity * (1.0 + confidence) if self.confidence_scaled_sizing else self.target_quantity
        if p_positive >= self.long_threshold:
            direction, quantity, abstain, reason = PositionDirection.LONG, sized_quantity, False, "model_signal_long"
        elif p_positive <= self.short_threshold:
            direction, quantity, abstain, reason = PositionDirection.SHORT, sized_quantity, False, "model_signal_short"
        else:
            direction, quantity, abstain, reason = PositionDirection.FLAT, 0.0, True, "model_signal_neutral"

        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=event, decision_time=context.decision_time, target_direction=direction,
            target_quantity=quantity, confidence=confidence, uncertainty=uncertainty, abstain=abstain, reason_codes=(reason,),
            model_output_identity=model_output_identity, feature_snapshot_identity=feature_snapshot_identity,
            diagnostics={"p_positive": p_positive},
        )


__all__ = ["ModelStrategyRuntime"]
