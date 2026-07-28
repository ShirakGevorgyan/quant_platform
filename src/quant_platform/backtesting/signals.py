"""Deterministic signal mapping (Milestone 5, Section 8) -- prediction
output to position intent, and ONLY that. `generate_signals` is a pure
function of a `VerifiedPredictionSet` and a `SignalMappingSpec` alone: no
market-bar parameter exists anywhere in this module's call surface, so
there is no Python expression through which a future price or an
outer-test financial result could reach signal construction (see
`backtesting.models.VerifiedPredictionSet`'s own docstring for the
independent-re-verification guarantee upstream of this module).

`MISSING_MARKET_BAR`/`OVERLAP_POLICY_REJECTION` (two of Section 8's nine
reason codes) are deliberately NOT assignable here -- both require market
data or already-open-position state this module structurally never sees;
they are assigned downstream, in `backtesting.execution`, when an
ACCEPTED signal is attempted against real bars. A `Signal` object, once
constructed, is never mutated afterward -- `backtesting.execution`
produces a separate `TradeAttempt` outcome referencing the original
signal rather than editing it (Section 9: "clearly distinguish signal;
position intent; order intent; fill")."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.models import (
    Decision,
    PositionDirection,
    PositionMode,
    SignalMappingPolicyKind,
    SignalReasonCode,
    VerifiedPredictionSet,
)
from quant_platform.backtesting.specs import SignalMappingSpec
from quant_platform.core.exceptions import SignalGenerationError
from quant_platform.ml.persistence import as_json_list, require_schema_version

SIGNAL_SET_SCHEMA_VERSION = 1

_ACCEPTED_REASONS = frozenset({SignalReasonCode.ACCEPTED_POSITIVE, SignalReasonCode.ACCEPTED_NEGATIVE})


@dataclass(frozen=True, slots=True)
class Signal:
    """Section 8's exact per-signal field list."""

    sample_position: int
    decision_timestamp: str
    direction: PositionDirection
    strength: float
    accepted: bool
    reason_code: SignalReasonCode
    confidence: float
    uncertainty: float
    threshold: float
    calibrated_probability: float
    source_calibration_id: str
    source_experiment_id: str
    outer_fold_index: int

    def __post_init__(self) -> None:
        if self.sample_position < 0:
            raise SignalGenerationError(f"Signal.sample_position must be >= 0, got {self.sample_position}")
        if not math.isfinite(self.strength) or not (0.0 <= self.strength <= 1.0):
            raise SignalGenerationError(f"Signal.strength must be a finite value in [0, 1], got {self.strength!r}")
        for name, value in (("confidence", self.confidence), ("uncertainty", self.uncertainty), ("threshold", self.threshold), ("calibrated_probability", self.calibrated_probability)):
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise SignalGenerationError(f"Signal.{name} must be a finite value in [0, 1], got {value!r}")
        if self.accepted != (self.reason_code in _ACCEPTED_REASONS):
            raise SignalGenerationError(
                f"Signal.accepted={self.accepted!r} is inconsistent with reason_code={self.reason_code.value!r} "
                f"(accepted must be True iff reason_code is accepted_positive/accepted_negative)"
            )
        if not self.accepted and self.direction is not PositionDirection.FLAT:
            raise SignalGenerationError("Signal: a rejected signal must always have direction=FLAT")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "sample_position": self.sample_position, "decision_timestamp": self.decision_timestamp,
            "direction": self.direction.value, "strength": self.strength, "accepted": self.accepted,
            "reason_code": self.reason_code.value, "confidence": self.confidence, "uncertainty": self.uncertainty,
            "threshold": self.threshold, "calibrated_probability": self.calibrated_probability,
            "source_calibration_id": self.source_calibration_id, "source_experiment_id": self.source_experiment_id,
            "outer_fold_index": self.outer_fold_index,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Signal:
        return cls(
            sample_position=int(str(raw["sample_position"])), decision_timestamp=str(raw["decision_timestamp"]),
            direction=PositionDirection(raw["direction"]), strength=float(str(raw["strength"])), accepted=bool(raw["accepted"]),
            reason_code=SignalReasonCode(raw["reason_code"]), confidence=float(str(raw["confidence"])),
            uncertainty=float(str(raw["uncertainty"])), threshold=float(str(raw["threshold"])),
            calibrated_probability=float(str(raw["calibrated_probability"])), source_calibration_id=str(raw["source_calibration_id"]),
            source_experiment_id=str(raw["source_experiment_id"]), outer_fold_index=int(str(raw["outer_fold_index"])),
        )


@dataclass(frozen=True, slots=True)
class SignalSet:
    schema_version: int
    outer_fold_index: int
    signals: tuple[Signal, ...]

    def __post_init__(self) -> None:
        if not self.signals:
            raise SignalGenerationError("SignalSet must contain at least one signal")
        positions = [s.sample_position for s in self.signals]
        if positions != sorted(positions):
            raise SignalGenerationError("SignalSet.signals must be strictly ascending by sample_position")
        if len(set(positions)) != len(positions):
            raise SignalGenerationError("SignalSet.signals must not contain duplicate sample_position values")
        for s in self.signals:
            if s.outer_fold_index != self.outer_fold_index:
                raise SignalGenerationError(
                    f"SignalSet.signals contains a signal with outer_fold_index={s.outer_fold_index}, "
                    f"expected {self.outer_fold_index}"
                )

    @property
    def accepted_signals(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.accepted)

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "signals": [s.to_json_dict() for s in self.signals]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SignalSet:
        require_schema_version(raw, supported=SIGNAL_SET_SCHEMA_VERSION, context="SignalSet")
        return cls(
            schema_version=SIGNAL_SET_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            signals=tuple(Signal.from_json_dict(s) for s in as_json_list(raw["signals"], field_name="signals")),
        )


def _strength(calibrated_probability: float, threshold: float) -> float:
    """`0.0` exactly at the threshold, `1.0` at the probability furthest
    from it -- structurally identical to `calibration.confidence.
    distance_from_threshold_component`, deliberately NOT imported from
    there (this module must not depend on calibration's confidence
    machinery for its own, independent notion of signal strength)."""
    normalizer = max(threshold, 1.0 - threshold)
    if normalizer == 0.0:
        return 0.0
    return min(abs(calibrated_probability - threshold) / normalizer, 1.0)


def generate_signals(predictions: VerifiedPredictionSet, *, spec: SignalMappingSpec, position_mode: PositionMode, respect_calibration_abstention: bool) -> SignalSet:
    """Section 8: pure, deterministic mapping from `predictions` alone.
    No market data, no outer-test financial result, anywhere in scope."""
    signals: list[Signal] = []
    short_direction = PositionDirection.SHORT if position_mode is PositionMode.LONG_SHORT else PositionDirection.FLAT

    for i in range(predictions.n_samples):
        calibrated_probability = predictions.calibrated_probabilities[i]
        confidence = predictions.confidence_scores[i]
        uncertainty = predictions.uncertainty_scores[i]
        threshold = predictions.threshold
        decision = predictions.decisions[i]
        strength = _strength(calibrated_probability, threshold)

        is_positive = calibrated_probability >= threshold
        provisional_direction = PositionDirection.LONG if is_positive else short_direction
        provisional_reason = SignalReasonCode.ACCEPTED_POSITIVE if is_positive else SignalReasonCode.ACCEPTED_NEGATIVE

        direction = provisional_direction
        reason = provisional_reason
        accepted = True

        if respect_calibration_abstention and decision == Decision.ABSTAIN.value:
            direction, reason, accepted = PositionDirection.FLAT, SignalReasonCode.ABSTAINED_BY_CALIBRATION_POLICY, False
        elif spec.kind in (SignalMappingPolicyKind.CONFIDENCE_FLOOR, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY) and spec.confidence_floor is not None and confidence < spec.confidence_floor:
            direction, reason, accepted = PositionDirection.FLAT, SignalReasonCode.BELOW_CONFIDENCE_FLOOR, False
        elif spec.kind in (SignalMappingPolicyKind.UNCERTAINTY_CEILING, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY) and spec.uncertainty_ceiling is not None and uncertainty > spec.uncertainty_ceiling:
            direction, reason, accepted = PositionDirection.FLAT, SignalReasonCode.ABOVE_UNCERTAINTY_CEILING, False
        elif spec.kind is SignalMappingPolicyKind.PROBABILITY_BANDS:
            assert spec.probability_band_long_min is not None and spec.probability_band_short_max is not None
            if calibrated_probability >= spec.probability_band_long_min:
                direction, reason = PositionDirection.LONG, SignalReasonCode.ACCEPTED_POSITIVE
            elif calibrated_probability <= spec.probability_band_short_max:
                direction, reason = short_direction, SignalReasonCode.ACCEPTED_NEGATIVE
            else:
                direction, reason = PositionDirection.FLAT, SignalReasonCode.ACCEPTED_NEGATIVE

        if position_mode is PositionMode.LONG_FLAT and direction is PositionDirection.SHORT:
            # Structurally unreachable given __post_init__'s mode/mapping
            # compatibility check on the owning BacktestSpec, kept as a
            # defense-in-depth fail-closed guard rather than trusted
            # silently.
            direction, reason, accepted = PositionDirection.FLAT, SignalReasonCode.UNSUPPORTED_CLASS, False

        signals.append(Signal(
            sample_position=predictions.sample_positions[i], decision_timestamp=predictions.timestamps[i],
            direction=direction, strength=strength, accepted=accepted, reason_code=reason, confidence=confidence,
            uncertainty=uncertainty, threshold=threshold, calibrated_probability=calibrated_probability,
            source_calibration_id=predictions.source_calibration_id, source_experiment_id=predictions.source_experiment_id,
            outer_fold_index=predictions.outer_fold_index,
        ))

    return SignalSet(schema_version=SIGNAL_SET_SCHEMA_VERSION, outer_fold_index=predictions.outer_fold_index, signals=tuple(signals))


__all__ = ["SIGNAL_SET_SCHEMA_VERSION", "Signal", "SignalSet", "generate_signals"]
