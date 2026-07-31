"""`RiskAuthorization` -- the immutable, content-addressed proof produced
by a portfolio risk evaluation for `quant_platform.portfolio_risk`
(Milestone 9, Phase 1).

THIS PHASE DOES NOT ENFORCE ANYTHING: no evaluator constructs a
`RiskAuthorization` from real portfolio state yet, and
`execution_gateway`'s dispatch path does not check for one yet (Phase 1's
own explicit scope boundary -- see `docs/portfolio_risk_architecture.md`).
This module defines the shape, its identity rule, and the verification
function a LATER phase's dispatch gate will call.

BINDING IS STRUCTURAL, NOT A SEPARATE CHECK: `RiskAuthorization.
risk_authorization_id` is a pure function of every field this class binds
to (`execution_intent_id`, `execution_session_id`, `portfolio_id`,
`portfolio_snapshot_id`, `price_snapshot_id`, `risk_policy_id`,
`risk_decision_id`, `decision_kind`, `evaluated_quantity`,
`evaluated_price`, `authorization_sequence`, `event_time`). Changing ANY
one of those fields changes the id -- so an authorization can never be
"reused" for a different intent, session, portfolio snapshot, price
snapshot, policy, quantity, or price by construction, not by a separate
runtime check. `verify_risk_authorization_binding` makes this
recompute-and-compare pattern explicit for a future dispatch gate to
call, exactly mirroring `execution_gateway.specs.
verify_execution_gateway_spec_identity`'s identical shape.

`event_time` is ALWAYS caller-supplied here -- never an internal
`utc_now()` read. This directly avoids `execution_gateway` Milestone 8's
own defect #12 (a wall-clock-captured `source_event_time` silently
cascading into `ExecutionIntent` identity, breaking deterministic
replay): because this package's caller supplies the SAME `event_time` on
every legitimate deterministic replay of the same scenario, including it
in identity is safe and does not reintroduce that defect class."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import RiskAuthorizationIdentityError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.portfolio_risk.models import RiskDecisionKind

RISK_AUTHORIZATION_KIND = "risk_authorization"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise RiskAuthorizationIdentityError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    _require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise RiskAuthorizationIdentityError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise RiskAuthorizationIdentityError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise RiskAuthorizationIdentityError(f"{field_name}: {exc}") from exc


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise RiskAuthorizationIdentityError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise RiskAuthorizationIdentityError(f"{field_name} must be finite and > 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class RiskAuthorization:
    risk_authorization_id: str

    execution_intent_id: str
    execution_session_id: str
    portfolio_id: str
    portfolio_snapshot_id: str
    price_snapshot_id: str
    risk_policy_id: str
    """Binds to `specs.compute_portfolio_risk_spec_id(...)
    .portfolio_risk_spec_id` -- the content-addressed id of the
    `PortfolioRiskSpec` (policy) in effect at evaluation time."""

    risk_decision_id: str
    decision_kind: RiskDecisionKind
    """Denormalized from the referenced `RiskDecision` for convenience --
    a future phase's recovery/verification pass is responsible for
    cross-checking this against the actual stored `RiskDecision`; Phase 1
    has no such store, so no cross-check is performed here."""

    evaluated_quantity: Decimal
    evaluated_price: Decimal

    authorization_sequence: int
    """Caller-supplied, deterministic, monotonically-assigned-by-caller
    disambiguator (never `uuid4`/random) -- lets two authorizations that
    would otherwise be economically identical (the same intent
    re-evaluated against the same snapshot) remain distinguishable."""

    event_time: datetime

    def __post_init__(self) -> None:
        for field_name, value in (
            ("execution_intent_id", self.execution_intent_id), ("execution_session_id", self.execution_session_id),
            ("portfolio_snapshot_id", self.portfolio_snapshot_id), ("price_snapshot_id", self.price_snapshot_id),
            ("risk_policy_id", self.risk_policy_id), ("risk_decision_id", self.risk_decision_id),
        ):
            _require_sha256(value, field_name=f"RiskAuthorization.{field_name}")
        if not self.portfolio_id:
            raise RiskAuthorizationIdentityError("RiskAuthorization.portfolio_id must not be empty")
        _positive_decimal(self.evaluated_quantity, field_name="RiskAuthorization.evaluated_quantity")
        _positive_decimal(self.evaluated_price, field_name="RiskAuthorization.evaluated_price")
        if self.authorization_sequence < 0:
            raise RiskAuthorizationIdentityError(f"RiskAuthorization.authorization_sequence must be >= 0, got {self.authorization_sequence}")
        _require_tz_aware(self.event_time, field_name="RiskAuthorization.event_time")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "risk_authorization_id": self.risk_authorization_id, "execution_intent_id": self.execution_intent_id,
            "execution_session_id": self.execution_session_id, "portfolio_id": self.portfolio_id,
            "portfolio_snapshot_id": self.portfolio_snapshot_id, "price_snapshot_id": self.price_snapshot_id,
            "risk_policy_id": self.risk_policy_id, "risk_decision_id": self.risk_decision_id, "decision_kind": self.decision_kind.value,
            "evaluated_quantity": decimal_to_json(self.evaluated_quantity), "evaluated_price": decimal_to_json(self.evaluated_price),
            "authorization_sequence": self.authorization_sequence, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["risk_authorization_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskAuthorization:
        return cls(
            risk_authorization_id=str(raw["risk_authorization_id"]), execution_intent_id=str(raw["execution_intent_id"]),
            execution_session_id=str(raw["execution_session_id"]), portfolio_id=str(raw["portfolio_id"]),
            portfolio_snapshot_id=str(raw["portfolio_snapshot_id"]), price_snapshot_id=str(raw["price_snapshot_id"]),
            risk_policy_id=str(raw["risk_policy_id"]), risk_decision_id=str(raw["risk_decision_id"]),
            decision_kind=RiskDecisionKind(raw["decision_kind"]), evaluated_quantity=parse_decimal(raw["evaluated_quantity"], field_name="evaluated_quantity"),
            evaluated_price=parse_decimal(raw["evaluated_price"], field_name="evaluated_price"),
            authorization_sequence=int(str(raw["authorization_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_risk_authorization(**kwargs: object) -> RiskAuthorization:
    provisional = RiskAuthorization(risk_authorization_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    risk_authorization_id = compute_content_id(RISK_AUTHORIZATION_KIND, provisional.to_identity_payload())
    return RiskAuthorization(risk_authorization_id=risk_authorization_id, **kwargs)  # type: ignore[arg-type]


def verify_risk_authorization_binding(
    authorization: RiskAuthorization, *, execution_intent_id: str, execution_session_id: str, portfolio_id: str, portfolio_snapshot_id: str,
    price_snapshot_id: str, risk_policy_id: str, risk_decision_id: str, decision_kind: RiskDecisionKind, evaluated_quantity: Decimal,
    evaluated_price: Decimal,
) -> bool:
    """Recomputes the EXACT id a `RiskAuthorization` issued for this
    precise tuple would have, and compares against `authorization`'s own
    id -- never trusts `authorization`'s claimed fields directly. A
    mismatch on ANY single bound field (a different intent, session,
    portfolio, portfolio snapshot, price snapshot, policy, decision, or
    price/quantity) produces a different expected id and is therefore
    caught uniformly by this one check, rather than by a separate
    per-field comparison that could be forgotten for a newly-added
    field."""
    candidate = RiskAuthorization(
        risk_authorization_id="0" * 64, execution_intent_id=execution_intent_id, execution_session_id=execution_session_id,
        portfolio_id=portfolio_id, portfolio_snapshot_id=portfolio_snapshot_id, price_snapshot_id=price_snapshot_id, risk_policy_id=risk_policy_id,
        risk_decision_id=risk_decision_id, decision_kind=decision_kind, evaluated_quantity=evaluated_quantity, evaluated_price=evaluated_price,
        authorization_sequence=authorization.authorization_sequence, event_time=authorization.event_time,
    )
    expected_id = compute_content_id(RISK_AUTHORIZATION_KIND, candidate.to_identity_payload())
    return expected_id == authorization.risk_authorization_id


__all__ = [
    "RISK_AUTHORIZATION_KIND",
    "RiskAuthorization",
    "create_risk_authorization",
    "verify_risk_authorization_binding",
]
