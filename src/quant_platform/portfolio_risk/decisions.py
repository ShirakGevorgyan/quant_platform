"""`RiskCheckResult`, `RiskEvaluationRequest`, and `RiskDecision` for
`quant_platform.portfolio_risk` (Milestone 9, Phase 1). No evaluator
exists in this phase -- nothing in this package yet DECIDES what a
`RiskDecision` should contain given a portfolio/price snapshot and a
policy. This module only defines the shapes and validates their INTERNAL
coherence: a `RiskDecision` that claims `APPROVED` while also carrying a
DENY/HALT-severity check result is rejected at construction, exactly the
same fail-closed spirit as every other model in this package, even
though nothing yet constructs a `RiskDecision` from real portfolio
state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import RiskEvaluationError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.portfolio_risk.models import (
    OrderSide,
    RiskCheckSeverity,
    RiskDecisionKind,
    RiskDenialReason,
)

RISK_EVALUATION_REQUEST_KIND = "risk_evaluation_request"
RISK_DECISION_KIND = "risk_decision"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise RiskEvaluationError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    _require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise RiskEvaluationError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise RiskEvaluationError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise RiskEvaluationError(f"{field_name}: {exc}") from exc


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise RiskEvaluationError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise RiskEvaluationError(f"{field_name} must be finite and > 0, got {value!r}")


def _finite_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite():
        raise RiskEvaluationError(f"{field_name} must be finite, got {value!r}")


# --------------------------------------------------------------------------
# RiskCheckResult -- nested value object, no independent content id
# (identified implicitly by its position within a RiskDecision's own
# check_results tuple, exactly like PositionSnapshot within
# PortfolioSnapshot.positions).
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    check_identity: str
    measured_value: Decimal
    limit_value: Decimal
    passed: bool
    severity: RiskCheckSeverity
    denial_reason: RiskDenialReason | None

    def __post_init__(self) -> None:
        if not self.check_identity:
            raise RiskEvaluationError("RiskCheckResult.check_identity must not be empty")
        _finite_decimal(self.measured_value, field_name="RiskCheckResult.measured_value")
        _finite_decimal(self.limit_value, field_name="RiskCheckResult.limit_value")
        if self.passed:
            if self.severity is not RiskCheckSeverity.INFO:
                raise RiskEvaluationError(f"RiskCheckResult.severity must be INFO when passed=True, got {self.severity!r}")
            if self.denial_reason is not None:
                raise RiskEvaluationError("RiskCheckResult.denial_reason must be None when passed=True")
        else:
            if self.severity is RiskCheckSeverity.INFO:
                raise RiskEvaluationError("RiskCheckResult.severity must not be INFO when passed=False")
            if self.denial_reason is None:
                raise RiskEvaluationError("RiskCheckResult.denial_reason is required when passed=False")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "check_identity": self.check_identity, "measured_value": decimal_to_json(self.measured_value),
            "limit_value": decimal_to_json(self.limit_value), "passed": self.passed, "severity": self.severity.value,
            "denial_reason": (None if self.denial_reason is None else self.denial_reason.value),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskCheckResult:
        return cls(
            check_identity=str(raw["check_identity"]), measured_value=parse_decimal(raw["measured_value"], field_name="measured_value"),
            limit_value=parse_decimal(raw["limit_value"], field_name="limit_value"), passed=bool(raw["passed"]),
            severity=RiskCheckSeverity(raw["severity"]), denial_reason=(None if raw.get("denial_reason") is None else RiskDenialReason(raw["denial_reason"])),
        )


# --------------------------------------------------------------------------
# RiskEvaluationRequest
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskEvaluationRequest:
    risk_evaluation_request_id: str
    execution_intent_id: str
    execution_session_id: str
    portfolio_id: str
    strategy_id: str
    instrument_id: str
    side: OrderSide
    quantity: Decimal
    portfolio_snapshot_id: str
    price_snapshot_id: str
    risk_policy_id: str
    reduce_only: bool
    requested_sequence: int
    event_time: datetime

    def __post_init__(self) -> None:
        for field_name, value in (
            ("execution_intent_id", self.execution_intent_id), ("execution_session_id", self.execution_session_id),
            ("portfolio_snapshot_id", self.portfolio_snapshot_id), ("price_snapshot_id", self.price_snapshot_id),
            ("risk_policy_id", self.risk_policy_id),
        ):
            _require_sha256(value, field_name=f"RiskEvaluationRequest.{field_name}")
        for field_name, value in (("portfolio_id", self.portfolio_id), ("strategy_id", self.strategy_id), ("instrument_id", self.instrument_id)):
            if not value:
                raise RiskEvaluationError(f"RiskEvaluationRequest.{field_name} must not be empty")
        _positive_decimal(self.quantity, field_name="RiskEvaluationRequest.quantity")
        if self.requested_sequence < 0:
            raise RiskEvaluationError(f"RiskEvaluationRequest.requested_sequence must be >= 0, got {self.requested_sequence}")
        _require_tz_aware(self.event_time, field_name="RiskEvaluationRequest.event_time")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "risk_evaluation_request_id": self.risk_evaluation_request_id, "execution_intent_id": self.execution_intent_id,
            "execution_session_id": self.execution_session_id, "portfolio_id": self.portfolio_id, "strategy_id": self.strategy_id,
            "instrument_id": self.instrument_id, "side": self.side.value, "quantity": decimal_to_json(self.quantity),
            "portfolio_snapshot_id": self.portfolio_snapshot_id, "price_snapshot_id": self.price_snapshot_id,
            "risk_policy_id": self.risk_policy_id, "reduce_only": self.reduce_only, "requested_sequence": self.requested_sequence,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["risk_evaluation_request_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskEvaluationRequest:
        return cls(
            risk_evaluation_request_id=str(raw["risk_evaluation_request_id"]), execution_intent_id=str(raw["execution_intent_id"]),
            execution_session_id=str(raw["execution_session_id"]), portfolio_id=str(raw["portfolio_id"]), strategy_id=str(raw["strategy_id"]),
            instrument_id=str(raw["instrument_id"]), side=OrderSide(raw["side"]), quantity=parse_decimal(raw["quantity"], field_name="quantity"),
            portfolio_snapshot_id=str(raw["portfolio_snapshot_id"]), price_snapshot_id=str(raw["price_snapshot_id"]),
            risk_policy_id=str(raw["risk_policy_id"]), reduce_only=bool(raw["reduce_only"]), requested_sequence=int(str(raw["requested_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_risk_evaluation_request(**kwargs: object) -> RiskEvaluationRequest:
    provisional = RiskEvaluationRequest(risk_evaluation_request_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    risk_evaluation_request_id = compute_content_id(RISK_EVALUATION_REQUEST_KIND, provisional.to_identity_payload())
    return RiskEvaluationRequest(risk_evaluation_request_id=risk_evaluation_request_id, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# RiskDecision
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskDecision:
    risk_decision_id: str
    risk_evaluation_request_id: str
    kind: RiskDecisionKind
    denial_reasons: tuple[RiskDenialReason, ...]
    check_results: tuple[RiskCheckResult, ...]
    evaluated_quantity: Decimal
    evaluated_price: Decimal
    portfolio_snapshot_id: str
    price_snapshot_id: str
    risk_policy_id: str
    decision_sequence: int
    event_time: datetime

    def __post_init__(self) -> None:
        for field_name, value in (
            ("risk_evaluation_request_id", self.risk_evaluation_request_id), ("portfolio_snapshot_id", self.portfolio_snapshot_id),
            ("price_snapshot_id", self.price_snapshot_id), ("risk_policy_id", self.risk_policy_id),
        ):
            _require_sha256(value, field_name=f"RiskDecision.{field_name}")
        _positive_decimal(self.evaluated_quantity, field_name="RiskDecision.evaluated_quantity")
        _positive_decimal(self.evaluated_price, field_name="RiskDecision.evaluated_price")
        if self.decision_sequence < 0:
            raise RiskEvaluationError(f"RiskDecision.decision_sequence must be >= 0, got {self.decision_sequence}")
        _require_tz_aware(self.event_time, field_name="RiskDecision.event_time")

        has_deny_check = any(c.severity is RiskCheckSeverity.DENY for c in self.check_results)
        has_halt_check = any(c.severity is RiskCheckSeverity.HALT for c in self.check_results)
        triggering_reasons = {
            c.denial_reason for c in self.check_results if c.severity in (RiskCheckSeverity.DENY, RiskCheckSeverity.HALT) and c.denial_reason is not None
        }

        if self.kind is RiskDecisionKind.APPROVED:
            if self.denial_reasons:
                raise RiskEvaluationError("RiskDecision.denial_reasons must be empty when kind is APPROVED")
            if has_deny_check or has_halt_check:
                raise RiskEvaluationError("RiskDecision.kind is APPROVED but a check_result has DENY/HALT severity")
        else:
            if not self.denial_reasons:
                raise RiskEvaluationError(f"RiskDecision.denial_reasons must not be empty when kind is {self.kind.value!r}")
            if not triggering_reasons.issubset(set(self.denial_reasons)):
                missing = sorted(r.value for r in (triggering_reasons - set(self.denial_reasons)))
                raise RiskEvaluationError(f"RiskDecision.denial_reasons is missing reason(s) present in check_results: {missing!r}")
            if self.kind is RiskDecisionKind.HALTED and not has_halt_check:
                raise RiskEvaluationError("RiskDecision.kind is HALTED but no check_result has HALT severity")
            if self.kind is RiskDecisionKind.DENIED and has_halt_check:
                raise RiskEvaluationError("RiskDecision.kind is DENIED but a check_result has HALT severity (should be HALTED)")

    @property
    def is_approved(self) -> bool:
        return self.kind is RiskDecisionKind.APPROVED

    def to_json_dict(self) -> dict[str, object]:
        return {
            "risk_decision_id": self.risk_decision_id, "risk_evaluation_request_id": self.risk_evaluation_request_id, "kind": self.kind.value,
            "denial_reasons": [r.value for r in self.denial_reasons], "check_results": [c.to_json_dict() for c in self.check_results],
            "evaluated_quantity": decimal_to_json(self.evaluated_quantity), "evaluated_price": decimal_to_json(self.evaluated_price),
            "portfolio_snapshot_id": self.portfolio_snapshot_id, "price_snapshot_id": self.price_snapshot_id, "risk_policy_id": self.risk_policy_id,
            "decision_sequence": self.decision_sequence, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["risk_decision_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskDecision:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        return cls(
            risk_decision_id=str(raw["risk_decision_id"]), risk_evaluation_request_id=str(raw["risk_evaluation_request_id"]),
            kind=RiskDecisionKind(raw["kind"]), denial_reasons=tuple(RiskDenialReason(r) for r in as_json_list(raw.get("denial_reasons") or [], field_name="denial_reasons")),
            check_results=tuple(RiskCheckResult.from_json_dict(as_json_dict(c, field_name="check_results[]")) for c in as_json_list(raw.get("check_results") or [], field_name="check_results")),
            evaluated_quantity=parse_decimal(raw["evaluated_quantity"], field_name="evaluated_quantity"),
            evaluated_price=parse_decimal(raw["evaluated_price"], field_name="evaluated_price"), portfolio_snapshot_id=str(raw["portfolio_snapshot_id"]),
            price_snapshot_id=str(raw["price_snapshot_id"]), risk_policy_id=str(raw["risk_policy_id"]),
            decision_sequence=int(str(raw["decision_sequence"])), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_risk_decision(**kwargs: object) -> RiskDecision:
    provisional = RiskDecision(risk_decision_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    risk_decision_id = compute_content_id(RISK_DECISION_KIND, provisional.to_identity_payload())
    return RiskDecision(risk_decision_id=risk_decision_id, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "RISK_DECISION_KIND",
    "RISK_EVALUATION_REQUEST_KIND",
    "RiskCheckResult",
    "RiskDecision",
    "RiskEvaluationRequest",
    "create_risk_decision",
    "create_risk_evaluation_request",
]
