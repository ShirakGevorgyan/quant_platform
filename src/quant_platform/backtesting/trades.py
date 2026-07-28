"""The trade record model (Milestone 5, Section 16) -- the terminal,
immutable record of one round-trip (or discarded/force-closed/excluded
incomplete) position. `trade_id` is a pure, deterministic function of
every identity-relevant field (Section 16: "Equivalent inputs must
produce identical trade IDs"), computed the same way every other content
identity in this platform is (`ml.fingerprints.fingerprint_json`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.models import ExitReasonCode, PositionDirection, SignalReasonCode, TradeStatus
from quant_platform.core.exceptions import TradeConstructionError
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    parse_utc_timestamp,
    require_schema_version,
)

TRADE_RECORD_SCHEMA_VERSION = 1


def compute_trade_id(
    *, source_calibration_id: str, outer_fold_index: int, signal_sample_position: int, direction: PositionDirection,
    entry_timestamp: str, exit_timestamp: str,
) -> str:
    payload = {
        "source_calibration_id": source_calibration_id, "outer_fold_index": outer_fold_index,
        "signal_sample_position": signal_sample_position, "direction": direction.value,
        "entry_timestamp": entry_timestamp, "exit_timestamp": exit_timestamp,
    }
    return fingerprint_json(payload)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    schema_version: int
    trade_id: str
    signal_sample_position: int
    outer_fold_index: int
    direction: PositionDirection
    signal_timestamp: str
    decision_timestamp: str
    entry_timestamp: str
    entry_bar_position: int
    entry_observed_price: float
    entry_effective_price: float
    exit_timestamp: str
    exit_bar_position: int
    exit_observed_price: float
    exit_effective_price: float
    holding_bars: int
    gross_return: float
    net_return: float
    cost_breakdown: CostBreakdown
    confidence: float
    uncertainty: float
    calibrated_probability: float
    entry_reason: SignalReasonCode
    exit_reason: ExitReasonCode
    status: TradeStatus
    source_calibration_id: str
    source_experiment_id: str

    def __post_init__(self) -> None:
        if self.direction is PositionDirection.FLAT:
            raise TradeConstructionError("TradeRecord.direction must be LONG or SHORT, never FLAT")
        if self.entry_bar_position < 0 or self.exit_bar_position < 0:
            raise TradeConstructionError("TradeRecord bar positions must be >= 0")
        if self.exit_bar_position < self.entry_bar_position:
            raise TradeConstructionError(
                f"TradeRecord: exit_bar_position ({self.exit_bar_position}) must be >= entry_bar_position "
                f"({self.entry_bar_position}) -- exit before entry"
            )
        if self.holding_bars != self.exit_bar_position - self.entry_bar_position:
            raise TradeConstructionError(
                f"TradeRecord.holding_bars ({self.holding_bars}) must equal exit_bar_position - "
                f"entry_bar_position ({self.exit_bar_position - self.entry_bar_position})"
            )
        for name, value in (
            ("entry_observed_price", self.entry_observed_price), ("entry_effective_price", self.entry_effective_price),
            ("exit_observed_price", self.exit_observed_price), ("exit_effective_price", self.exit_effective_price),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise TradeConstructionError(f"TradeRecord.{name} must be a finite, positive price, got {value!r}")
        for name, value in (("gross_return", self.gross_return), ("net_return", self.net_return)):
            if not math.isfinite(value):
                raise TradeConstructionError(f"TradeRecord.{name} must be finite, got {value!r}")
        for ts in (self.signal_timestamp, self.decision_timestamp, self.entry_timestamp, self.exit_timestamp):
            parse_utc_timestamp(ts)
        # Milestone 5.1, Section 5: an IMPOSSIBLE fill -- a fill timestamped
        # before the decision that produced it could even be known -- is
        # rejected here, at the one place every TradeRecord (freshly
        # simulated OR decoded from a persisted artifact) passes through.
        if parse_utc_timestamp(self.entry_timestamp) < parse_utc_timestamp(self.decision_timestamp):
            raise TradeConstructionError(
                f"TradeRecord: entry_timestamp ({self.entry_timestamp!r}) is before decision_timestamp "
                f"({self.decision_timestamp!r}) -- an IMPOSSIBLE fill (this trade would have been entered "
                "using information not yet knowable at that instant)"
            )
        for name, value in (("confidence", self.confidence), ("uncertainty", self.uncertainty), ("calibrated_probability", self.calibrated_probability)):
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise TradeConstructionError(f"TradeRecord.{name} must be a finite value in [0, 1], got {value!r}")

        expected_id = compute_trade_id(
            source_calibration_id=self.source_calibration_id, outer_fold_index=self.outer_fold_index,
            signal_sample_position=self.signal_sample_position, direction=self.direction,
            entry_timestamp=self.entry_timestamp, exit_timestamp=self.exit_timestamp,
        )
        if self.trade_id != expected_id:
            raise TradeConstructionError(
                f"TradeRecord.trade_id={self.trade_id!r} does not match the recomputed deterministic id "
                f"{expected_id!r} for its own identity-relevant fields -- tampered or hand-constructed inconsistently"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "trade_id": self.trade_id, "signal_sample_position": self.signal_sample_position,
            "outer_fold_index": self.outer_fold_index, "direction": self.direction.value, "signal_timestamp": self.signal_timestamp,
            "decision_timestamp": self.decision_timestamp, "entry_timestamp": self.entry_timestamp, "entry_bar_position": self.entry_bar_position,
            "entry_observed_price": self.entry_observed_price, "entry_effective_price": self.entry_effective_price,
            "exit_timestamp": self.exit_timestamp, "exit_bar_position": self.exit_bar_position, "exit_observed_price": self.exit_observed_price,
            "exit_effective_price": self.exit_effective_price, "holding_bars": self.holding_bars, "gross_return": self.gross_return,
            "net_return": self.net_return, "cost_breakdown": self.cost_breakdown.to_json_dict(), "confidence": self.confidence,
            "uncertainty": self.uncertainty, "calibrated_probability": self.calibrated_probability, "entry_reason": self.entry_reason.value,
            "exit_reason": self.exit_reason.value, "status": self.status.value, "source_calibration_id": self.source_calibration_id,
            "source_experiment_id": self.source_experiment_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TradeRecord:
        require_schema_version(raw, supported=TRADE_RECORD_SCHEMA_VERSION, context="TradeRecord")
        return cls(
            schema_version=TRADE_RECORD_SCHEMA_VERSION, trade_id=str(raw["trade_id"]), signal_sample_position=int(str(raw["signal_sample_position"])),
            outer_fold_index=int(str(raw["outer_fold_index"])), direction=PositionDirection(raw["direction"]), signal_timestamp=str(raw["signal_timestamp"]),
            decision_timestamp=str(raw["decision_timestamp"]), entry_timestamp=str(raw["entry_timestamp"]), entry_bar_position=int(str(raw["entry_bar_position"])),
            entry_observed_price=float(str(raw["entry_observed_price"])), entry_effective_price=float(str(raw["entry_effective_price"])),
            exit_timestamp=str(raw["exit_timestamp"]), exit_bar_position=int(str(raw["exit_bar_position"])), exit_observed_price=float(str(raw["exit_observed_price"])),
            exit_effective_price=float(str(raw["exit_effective_price"])), holding_bars=int(str(raw["holding_bars"])), gross_return=float(str(raw["gross_return"])),
            net_return=float(str(raw["net_return"])), cost_breakdown=CostBreakdown.from_json_dict(as_json_dict(raw["cost_breakdown"], field_name="cost_breakdown")),
            confidence=float(str(raw["confidence"])), uncertainty=float(str(raw["uncertainty"])), calibrated_probability=float(str(raw["calibrated_probability"])),
            entry_reason=SignalReasonCode(raw["entry_reason"]), exit_reason=ExitReasonCode(raw["exit_reason"]), status=TradeStatus(raw["status"]),
            source_calibration_id=str(raw["source_calibration_id"]), source_experiment_id=str(raw["source_experiment_id"]),
        )


@dataclass(frozen=True, slots=True)
class TradeSet:
    schema_version: int
    outer_fold_index: int
    trades: tuple[TradeRecord, ...]

    def __post_init__(self) -> None:
        for t in self.trades:
            if t.outer_fold_index != self.outer_fold_index:
                raise TradeConstructionError(f"TradeSet: trade {t.trade_id!r} has outer_fold_index={t.outer_fold_index}, expected {self.outer_fold_index}")
        ids = [t.trade_id for t in self.trades]
        if len(set(ids)) != len(ids):
            raise TradeConstructionError("TradeSet.trades contains duplicate trade_id values")

    @property
    def closed_trades(self) -> tuple[TradeRecord, ...]:
        return tuple(t for t in self.trades if t.status is TradeStatus.CLOSED or t.status is TradeStatus.INCOMPLETE_FORCE_CLOSED)

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "trades": [t.to_json_dict() for t in self.trades]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TradeSet:
        require_schema_version(raw, supported=1, context="TradeSet")
        return cls(
            schema_version=1, outer_fold_index=int(str(raw["outer_fold_index"])),
            trades=tuple(TradeRecord.from_json_dict(t) for t in as_json_list(raw.get("trades") or [], field_name="trades")),
        )


__all__ = ["TRADE_RECORD_SCHEMA_VERSION", "TradeRecord", "TradeSet", "compute_trade_id"]
