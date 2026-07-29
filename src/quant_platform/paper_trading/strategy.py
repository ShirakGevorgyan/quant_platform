"""Broker-neutral strategy runtime interface and `StrategyDecision`
(Milestone 7, Section 7). A `StrategyDecision` is NOT an order -- turning
one into zero-or-more `OrderRequest`s is `orders.py`'s order-policy
layer's job exclusively; nothing here ever constructs an order.

`PortfolioSnapshot`/`RiskState`/`SessionState` below are the deliberately
NARROW, read-only views a strategy is allowed to observe -- NOT the full
internal mutable engine state (`portfolio.PortfolioState`/the risk
engine's own bookkeeping/`manifests.PaperSessionManifest`), which carry
cost-basis internals, version counters, and other information a strategy
has no business inspecting. `portfolio.py`/`risk.py`/`manifests.py`
(built later) each grow a `to_strategy_snapshot()`-style method that
narrows their own richer state down to these types -- the types
themselves live here because the STRATEGY RUNTIME'S OWN INTERFACE
CONTRACT is what defines "the exact shape of what a strategy may see,"
which is Section 7's concern, not those modules'.

No strategy code may call `datetime.now()`/`time.time()` -- `StrategyContext.
decision_time` is always supplied by the injected `clock.Clock`, never
read directly (Section 6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import StrategyRuntimeError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import JsonPrimitive
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    parse_utc_timestamp,
)
from quant_platform.paper_trading.events import MarketEvent, market_event_id, market_event_time
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import KillSwitchState, PaperSessionStage

STRATEGY_DECISION_KIND = "strategy_decision"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise StrategyRuntimeError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise StrategyRuntimeError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise StrategyRuntimeError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise StrategyRuntimeError(f"{field_name}: {exc}") from exc


def _require_unit_interval(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise StrategyRuntimeError(f"{field_name} must be finite and in [0, 1], got {value!r}")


def _require_non_negative_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise StrategyRuntimeError(f"{field_name} must be finite and >= 0, got {value!r}")


def _require_valid_sha256_or_none(value: str | None, *, field_name: str) -> None:
    if value is not None and not is_valid_sha256_hex(value):
        raise StrategyRuntimeError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest or None, got {value!r}")


def _require_json_safe_diagnostics(diagnostics: dict[str, JsonPrimitive], *, field_name: str = "diagnostics") -> None:
    for key, value in diagnostics.items():
        if not isinstance(key, str):
            raise StrategyRuntimeError(f"{field_name} keys must be strings, got {type(key).__name__}")
        if not isinstance(value, str | int | float | bool | type(None)):
            raise StrategyRuntimeError(f"{field_name}[{key!r}] must be a JSON-safe primitive (str/int/float/bool/None), got {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise StrategyRuntimeError(f"{field_name}[{key!r}] must be finite, got {value!r}")


# --------------------------------------------------------------------------
# Narrow, strategy-visible snapshots (Section 7's own input contract)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    instrument: str
    signed_quantity: float
    average_entry_price: float | None
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float

    def __post_init__(self) -> None:
        if not self.instrument:
            raise StrategyRuntimeError("PortfolioSnapshot.instrument must not be empty")
        for field_name, value in (("signed_quantity", self.signed_quantity), ("cash", self.cash), ("equity", self.equity), ("unrealized_pnl", self.unrealized_pnl), ("realized_pnl", self.realized_pnl)):
            if not math.isfinite(value):
                raise StrategyRuntimeError(f"PortfolioSnapshot.{field_name} must be finite, got {value!r}")
        if self.average_entry_price is not None and (not math.isfinite(self.average_entry_price) or self.average_entry_price <= 0.0):
            raise StrategyRuntimeError(f"PortfolioSnapshot.average_entry_price must be finite and > 0 when present, got {self.average_entry_price!r}")
        if self.signed_quantity == 0.0 and self.average_entry_price is not None:
            raise StrategyRuntimeError("PortfolioSnapshot.average_entry_price must be None when flat (signed_quantity == 0)")


@dataclass(frozen=True, slots=True)
class RiskState:
    trading_halted: bool
    kill_switch_state: KillSwitchState


@dataclass(frozen=True, slots=True)
class SessionState:
    paper_session_id: str
    stage: PaperSessionStage

    def __post_init__(self) -> None:
        if not self.paper_session_id:
            raise StrategyRuntimeError("SessionState.paper_session_id must not be empty")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a `StrategyRuntime` receives to produce ONE decision, in
    response to ONE market event. `feature_snapshot` is a flat name ->
    value mapping (already point-in-time correct -- computing it without
    look-ahead is `features.py`'s job, not this package's); this context
    only carries its already-computed VALUES plus a content identity for
    the decision's own provenance chain."""

    event: MarketEvent
    feature_snapshot: dict[str, float]
    feature_snapshot_identity: str | None
    model_output: float
    model_output_identity: str | None
    calibrated_probability: float | None
    confidence: float
    uncertainty: float
    portfolio: PortfolioSnapshot
    risk: RiskState
    session: SessionState
    decision_time: datetime

    def __post_init__(self) -> None:
        _require_tz_aware(self.decision_time, field_name="StrategyContext.decision_time")
        if not math.isfinite(self.model_output):
            raise StrategyRuntimeError(f"StrategyContext.model_output must be finite, got {self.model_output!r}")
        if self.calibrated_probability is not None:
            _require_unit_interval(self.calibrated_probability, field_name="StrategyContext.calibrated_probability")
        _require_unit_interval(self.confidence, field_name="StrategyContext.confidence")
        _require_non_negative_finite(self.uncertainty, field_name="StrategyContext.uncertainty")
        _require_valid_sha256_or_none(self.feature_snapshot_identity, field_name="StrategyContext.feature_snapshot_identity")
        _require_valid_sha256_or_none(self.model_output_identity, field_name="StrategyContext.model_output_identity")


# --------------------------------------------------------------------------
# StrategyDecision
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StopTargetIntent:
    """A strategy's OWN view of where it would place a protective stop/
    target -- a hint the order-policy layer MAY use to construct
    additional STOP/LIMIT orders; never binding by itself."""

    stop_loss_price: float | None
    take_profit_price: float | None

    def __post_init__(self) -> None:
        if self.stop_loss_price is not None and (not math.isfinite(self.stop_loss_price) or self.stop_loss_price <= 0.0):
            raise StrategyRuntimeError(f"StopTargetIntent.stop_loss_price must be finite and > 0 when present, got {self.stop_loss_price!r}")
        if self.take_profit_price is not None and (not math.isfinite(self.take_profit_price) or self.take_profit_price <= 0.0):
            raise StrategyRuntimeError(f"StopTargetIntent.take_profit_price must be finite and > 0 when present, got {self.take_profit_price!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {"stop_loss_price": self.stop_loss_price, "take_profit_price": self.take_profit_price}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StopTargetIntent:
        return cls(
            stop_loss_price=(None if raw.get("stop_loss_price") is None else float(str(raw["stop_loss_price"]))),
            take_profit_price=(None if raw.get("take_profit_price") is None else float(str(raw["take_profit_price"]))),
        )


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    decision_id: str
    strategy_identity: str
    event_identity: str
    decision_time: datetime
    target_direction: PositionDirection
    target_quantity: float
    confidence: float
    uncertainty: float
    abstain: bool
    reason_codes: tuple[str, ...]
    model_output_identity: str | None
    feature_snapshot_identity: str | None
    stop_target_intent: StopTargetIntent | None
    diagnostics: dict[str, JsonPrimitive]

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.strategy_identity):
            raise StrategyRuntimeError(f"StrategyDecision.strategy_identity must be a valid sha256 hex digest, got {self.strategy_identity!r}")
        if not is_valid_sha256_hex(self.event_identity):
            raise StrategyRuntimeError(f"StrategyDecision.event_identity must be a valid sha256 hex digest, got {self.event_identity!r}")
        _require_tz_aware(self.decision_time, field_name="StrategyDecision.decision_time")
        _require_non_negative_finite(self.target_quantity, field_name="StrategyDecision.target_quantity")
        if self.target_direction is PositionDirection.FLAT and self.target_quantity != 0.0:
            raise StrategyRuntimeError("StrategyDecision.target_quantity must be 0 when target_direction is FLAT")
        _require_unit_interval(self.confidence, field_name="StrategyDecision.confidence")
        _require_non_negative_finite(self.uncertainty, field_name="StrategyDecision.uncertainty")
        if not self.reason_codes:
            raise StrategyRuntimeError("StrategyDecision.reason_codes must not be empty -- every decision must explain itself")
        if any(not code for code in self.reason_codes):
            raise StrategyRuntimeError("StrategyDecision.reason_codes must not contain an empty string")
        _require_valid_sha256_or_none(self.model_output_identity, field_name="StrategyDecision.model_output_identity")
        _require_valid_sha256_or_none(self.feature_snapshot_identity, field_name="StrategyDecision.feature_snapshot_identity")
        _require_json_safe_diagnostics(self.diagnostics)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id, "strategy_identity": self.strategy_identity, "event_identity": self.event_identity,
            "decision_time": _serialize_timestamp(self.decision_time, field_name="decision_time"), "target_direction": self.target_direction.value, "target_quantity": self.target_quantity,
            "confidence": self.confidence, "uncertainty": self.uncertainty, "abstain": self.abstain, "reason_codes": list(self.reason_codes),
            "model_output_identity": self.model_output_identity, "feature_snapshot_identity": self.feature_snapshot_identity,
            "stop_target_intent": (None if self.stop_target_intent is None else self.stop_target_intent.to_json_dict()),
            "diagnostics": dict(self.diagnostics),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["decision_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StrategyDecision:
        stop_target_raw = raw.get("stop_target_intent")
        return cls(
            decision_id=str(raw["decision_id"]), strategy_identity=str(raw["strategy_identity"]), event_identity=str(raw["event_identity"]),
            decision_time=_deserialize_timestamp(raw["decision_time"], field_name="decision_time"), target_direction=PositionDirection(raw["target_direction"]),
            target_quantity=float(str(raw["target_quantity"])), confidence=float(str(raw["confidence"])), uncertainty=float(str(raw["uncertainty"])),
            abstain=bool(raw["abstain"]), reason_codes=tuple(str(c) for c in as_json_list(raw.get("reason_codes") or [], field_name="reason_codes")),
            model_output_identity=(None if raw.get("model_output_identity") is None else str(raw["model_output_identity"])),
            feature_snapshot_identity=(None if raw.get("feature_snapshot_identity") is None else str(raw["feature_snapshot_identity"])),
            stop_target_intent=(None if stop_target_raw is None else StopTargetIntent.from_json_dict(as_json_dict(stop_target_raw, field_name="stop_target_intent"))),
            diagnostics=dict(as_json_dict(raw.get("diagnostics") or {}, field_name="diagnostics")),
        )


def create_strategy_decision(
    *, strategy_identity: str, event: MarketEvent, decision_time: datetime, target_direction: PositionDirection, target_quantity: float,
    confidence: float, uncertainty: float, abstain: bool, reason_codes: tuple[str, ...], model_output_identity: str | None = None,
    feature_snapshot_identity: str | None = None, stop_target_intent: StopTargetIntent | None = None, diagnostics: dict[str, JsonPrimitive] | None = None,
) -> StrategyDecision:
    """The only supported way to mint a fresh `StrategyDecision` -- computes
    its deterministic `decision_id` from every other field (including
    `event_identity`, `feature_snapshot_identity`, and `model_output_
    identity`) before constructing it. Two calls with identical arguments
    always produce a byte-identical decision; changing ANY input --
    including feeding in a feature snapshot or model output computed from
    future information -- changes `decision_id`, which is exactly what lets
    `verification.py` catch a leaked decision: an independent recomputation
    using only correctly-scoped, at-or-before-decision-time information will
    never reproduce a leaked decision's identity."""
    event_identity = market_event_id(event)
    provisional = StrategyDecision(
        decision_id="0" * 64, strategy_identity=strategy_identity, event_identity=event_identity, decision_time=decision_time,
        target_direction=target_direction, target_quantity=target_quantity, confidence=confidence, uncertainty=uncertainty, abstain=abstain,
        reason_codes=reason_codes, model_output_identity=model_output_identity, feature_snapshot_identity=feature_snapshot_identity,
        stop_target_intent=stop_target_intent, diagnostics=diagnostics or {},
    )
    decision_id = compute_content_id(STRATEGY_DECISION_KIND, provisional.to_identity_payload())
    return StrategyDecision(
        decision_id=decision_id, strategy_identity=strategy_identity, event_identity=event_identity, decision_time=decision_time,
        target_direction=target_direction, target_quantity=target_quantity, confidence=confidence, uncertainty=uncertainty, abstain=abstain,
        reason_codes=reason_codes, model_output_identity=model_output_identity, feature_snapshot_identity=feature_snapshot_identity,
        stop_target_intent=stop_target_intent, diagnostics=diagnostics or {},
    )


class StrategyRuntime(Protocol):
    """Broker-neutral strategy interface -- ANY concrete strategy
    implementation (a fitted model wrapper, a rule-based baseline, a test
    double) satisfies this by implementing `decide`. Nothing about this
    Protocol references MT5, a broker, or any live-execution concept."""

    @property
    def strategy_identity(self) -> str: ...

    def decide(self, context: StrategyContext) -> StrategyDecision: ...


def event_time_of(context: StrategyContext) -> datetime:
    """The market event time that triggered `context` -- convenience
    accessor mirroring `events.market_event_time`, so callers never need
    to import `events` separately just to log/compare against it."""
    return market_event_time(context.event)


__all__ = [
    "STRATEGY_DECISION_KIND",
    "PortfolioSnapshot",
    "RiskState",
    "SessionState",
    "StopTargetIntent",
    "StrategyContext",
    "StrategyDecision",
    "StrategyRuntime",
    "create_strategy_decision",
    "event_time_of",
]
