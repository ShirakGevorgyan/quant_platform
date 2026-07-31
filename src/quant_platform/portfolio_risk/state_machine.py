"""Event-sourced `RiskAuthorizationStatus` reconstruction for
`quant_platform.portfolio_risk` (Milestone 9, Phase 3) -- mirrors
`execution_gateway.state_machine.ExecutionOrderStateEvent`/
`resolve_execution_order_state` exactly: an implicit initial state
(`ISSUED`), one immutable, content-addressed event per transition, and a
pure replay function that raises on ANY discontinuity or illegal
transition rather than silently tolerating one.

`RiskAuthorizationStatusEvent.__post_init__` and `resolve_risk_
authorization_status` BOTH independently re-check transition legality
against `models.is_legal_risk_authorization_status_transition` -- the
same intentional defense-in-depth double-check
`execution_gateway.state_machine.ExecutionOrderStateEvent` already uses,
so a single missed guard anywhere in this module cannot let an illegal
transition through."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import PortfolioRiskRecoveryError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import compute_content_id
from quant_platform.portfolio_risk.models import (
    RiskAuthorizationStatus,
    is_legal_risk_authorization_status_transition,
)

RISK_AUTHORIZATION_STATUS_EVENT_KIND = "risk_authorization_status_event"

_CONSUMPTION_IDENTITY_REQUIRED_STATES = frozenset({RiskAuthorizationStatus.RESERVED, RiskAuthorizationStatus.CONSUMED})
_REASON_CODE_REQUIRED_STATES = frozenset({
    RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED, RiskAuthorizationStatus.REVOKED,
})


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PortfolioRiskRecoveryError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    _require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PortfolioRiskRecoveryError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PortfolioRiskRecoveryError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PortfolioRiskRecoveryError(f"{field_name}: {exc}") from exc


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise PortfolioRiskRecoveryError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


@dataclass(frozen=True, slots=True)
class RiskAuthorizationStatusEvent:
    event_id: str
    authorization_id: str
    portfolio_id: str
    from_state: RiskAuthorizationStatus
    to_state: RiskAuthorizationStatus
    event_time: datetime
    sequence: int
    consumption_identity: str | None
    """A caller-supplied, deterministic identity for the ATTEMPTED
    economic use (e.g. a future execution-gateway command id) --
    required exactly when `to_state` is `RESERVED` or `CONSUMED`, so a
    safe, exact retry (same `consumption_identity`) can be distinguished
    from a genuine conflict (a different one) -- see `lifecycle.py`."""
    reason_code: str | None
    detail: str

    def __post_init__(self) -> None:
        _require_sha256(self.authorization_id, field_name="RiskAuthorizationStatusEvent.authorization_id")
        if not self.portfolio_id:
            raise PortfolioRiskRecoveryError("RiskAuthorizationStatusEvent.portfolio_id must not be empty")
        _require_tz_aware(self.event_time, field_name="RiskAuthorizationStatusEvent.event_time")
        if self.sequence < 0:
            raise PortfolioRiskRecoveryError(f"RiskAuthorizationStatusEvent.sequence must be >= 0, got {self.sequence}")
        if not is_legal_risk_authorization_status_transition(self.from_state, self.to_state):
            raise PortfolioRiskRecoveryError(f"Illegal risk authorization status transition {self.from_state.value!r} -> {self.to_state.value!r}")
        if self.to_state in _CONSUMPTION_IDENTITY_REQUIRED_STATES:
            if not self.consumption_identity:
                raise PortfolioRiskRecoveryError(f"RiskAuthorizationStatusEvent.consumption_identity is required when to_state is {self.to_state.value!r}")
        elif self.consumption_identity is not None:
            raise PortfolioRiskRecoveryError(f"RiskAuthorizationStatusEvent.consumption_identity must be None when to_state is {self.to_state.value!r}")
        if self.to_state in _REASON_CODE_REQUIRED_STATES:
            if not self.reason_code:
                raise PortfolioRiskRecoveryError(f"RiskAuthorizationStatusEvent.reason_code is required when to_state is {self.to_state.value!r}")
        elif self.reason_code is not None:
            raise PortfolioRiskRecoveryError(f"RiskAuthorizationStatusEvent.reason_code must be None when to_state is {self.to_state.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id, "authorization_id": self.authorization_id, "portfolio_id": self.portfolio_id,
            "from_state": self.from_state.value, "to_state": self.to_state.value,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "consumption_identity": self.consumption_identity, "reason_code": self.reason_code, "detail": self.detail,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskAuthorizationStatusEvent:
        return cls(
            event_id=str(raw["event_id"]), authorization_id=str(raw["authorization_id"]), portfolio_id=str(raw["portfolio_id"]),
            from_state=RiskAuthorizationStatus(raw["from_state"]), to_state=RiskAuthorizationStatus(raw["to_state"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            consumption_identity=(None if raw.get("consumption_identity") is None else str(raw["consumption_identity"])),
            reason_code=(None if raw.get("reason_code") is None else str(raw["reason_code"])), detail=str(raw["detail"]),
        )


def create_risk_authorization_status_event(
    *, authorization_id: str, portfolio_id: str, from_state: RiskAuthorizationStatus, to_state: RiskAuthorizationStatus, event_time: datetime,
    sequence: int, consumption_identity: str | None, reason_code: str | None, detail: str,
) -> RiskAuthorizationStatusEvent:
    provisional = RiskAuthorizationStatusEvent(
        event_id="0" * 64, authorization_id=authorization_id, portfolio_id=portfolio_id, from_state=from_state, to_state=to_state,
        event_time=event_time, sequence=sequence, consumption_identity=consumption_identity, reason_code=reason_code, detail=detail,
    )
    event_id = compute_content_id(RISK_AUTHORIZATION_STATUS_EVENT_KIND, provisional.to_identity_payload())
    return RiskAuthorizationStatusEvent(
        event_id=event_id, authorization_id=authorization_id, portfolio_id=portfolio_id, from_state=from_state, to_state=to_state,
        event_time=event_time, sequence=sequence, consumption_identity=consumption_identity, reason_code=reason_code, detail=detail,
    )


def resolve_risk_authorization_status(authorization_id: str, events: list[RiskAuthorizationStatusEvent]) -> RiskAuthorizationStatus:
    current = RiskAuthorizationStatus.ISSUED
    bound_identity: str | None = None
    for event in events:
        if event.authorization_id != authorization_id:
            raise PortfolioRiskRecoveryError(
                f"resolve_risk_authorization_status: event {event.event_id!r} belongs to authorization {event.authorization_id!r}, expected {authorization_id!r}"
            )
        if event.from_state is not current:
            raise PortfolioRiskRecoveryError(
                f"resolve_risk_authorization_status: event {event.event_id!r} expects from_state={event.from_state.value!r} but current state is {current.value!r}"
            )
        if not is_legal_risk_authorization_status_transition(current, event.to_state):
            raise PortfolioRiskRecoveryError(f"resolve_risk_authorization_status: illegal transition {current.value!r} -> {event.to_state.value!r}")
        if event.to_state is RiskAuthorizationStatus.RESERVED:
            # RESERVED is reachable only from ISSUED (the only legal edge into it), so this is
            # always the FIRST and ONLY reservation -- the economic identity binds here.
            bound_identity = event.consumption_identity
        elif event.to_state is RiskAuthorizationStatus.CONSUMED and event.consumption_identity != bound_identity:
            # CONSUMED is reachable only from RESERVED. A CONSUMED event whose consumption_identity
            # does not match the identity bound at RESERVED cannot have been produced by
            # `lifecycle.py`'s own write-time gate (which always re-binds the SAME identity for an
            # exact retry and rejects a conflicting one before ever appending) -- it can only exist
            # via direct ledger tampering, e.g. a coherently re-chained forged entry. Fail closed
            # rather than silently reconstructing a state that implies an unauthorized economic use.
            raise PortfolioRiskRecoveryError(
                f"resolve_risk_authorization_status: event {event.event_id!r} consumes authorization {authorization_id!r} "
                f"with consumption_identity={event.consumption_identity!r}, but it was reserved with {bound_identity!r}"
            )
        current = event.to_state
    return current


def consumption_identity_for(authorization_id: str, events: list[RiskAuthorizationStatusEvent]) -> str | None:
    """The `consumption_identity` bound to `authorization_id` by its most
    recent `RESERVED`/`CONSUMED` transition, or `None` if never
    reserved/consumed. Used by `lifecycle.py` to distinguish a safe exact
    retry from a genuine conflict."""
    bound: str | None = None
    for event in events:
        if event.authorization_id == authorization_id and event.consumption_identity is not None:
            bound = event.consumption_identity
    return bound


__all__ = [
    "RISK_AUTHORIZATION_STATUS_EVENT_KIND",
    "RiskAuthorizationStatusEvent",
    "consumption_identity_for",
    "create_risk_authorization_status_event",
    "resolve_risk_authorization_status",
]
