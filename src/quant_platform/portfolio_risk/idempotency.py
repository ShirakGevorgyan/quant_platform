"""Durable idempotency indexes for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3) -- every index here is rebuilt BY SCANNING THE
LEDGER (never an in-memory-only set), mirroring `execution_gateway.
idempotency`'s identical pure, ledger-scanning, conflict-raising
convention exactly."""

from __future__ import annotations

from quant_platform.core.exceptions import PortfolioRiskPersistenceError
from quant_platform.portfolio_risk.ledger import RiskLedgerEntry, RiskLedgerEntryKind
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus
from quant_platform.portfolio_risk.state_machine import (
    RiskAuthorizationStatusEvent,
    consumption_identity_for,
    resolve_risk_authorization_status,
)

_STATUS_ENTRY_KINDS: dict[RiskLedgerEntryKind, RiskAuthorizationStatus] = {
    RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED: RiskAuthorizationStatus.RESERVED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED: RiskAuthorizationStatus.CONSUMED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_EXPIRED: RiskAuthorizationStatus.EXPIRED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_INVALIDATED: RiskAuthorizationStatus.INVALIDATED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_REVOKED: RiskAuthorizationStatus.REVOKED,
}

__all__ = [
    "build_authorization_consumption_index",
    "build_authorization_payload_index",
    "build_authorization_status_index",
    "build_consumption_identity_index",
    "build_decision_to_authorization_index",
    "build_execution_intent_index",
    "build_status_events_index",
]


def build_decision_to_authorization_index(entries: list[RiskLedgerEntry]) -> dict[str, str]:
    """`risk_decision_id -> authorization_id`. Rejects one decision
    mapped to two different authorization ids."""
    index: dict[str, str] = {}
    for entry in entries:
        if entry.entry_kind is not RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED:
            continue
        decision_id = str(entry.payload["risk_decision_id"])
        authorization_id = str(entry.payload["risk_authorization_id"])
        existing = index.get(decision_id)
        if existing is not None and existing != authorization_id:
            raise PortfolioRiskPersistenceError(
                f"risk_decision_id {decision_id!r} is bound to two different authorization ids: {existing!r} and {authorization_id!r}"
            )
        index[decision_id] = authorization_id
    return index


def build_authorization_payload_index(entries: list[RiskLedgerEntry]) -> dict[str, dict[str, object]]:
    """`authorization_id -> bound economic payload` (the `RISK_
    AUTHORIZATION_ISSUED` entry's own payload). Rejects the same id
    recorded twice with a different payload -- structurally impossible
    for a genuinely content-addressed id, so a violation here always
    indicates ledger corruption or tampering."""
    index: dict[str, dict[str, object]] = {}
    for entry in entries:
        if entry.entry_kind is not RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED:
            continue
        authorization_id = str(entry.payload["risk_authorization_id"])
        existing = index.get(authorization_id)
        if existing is not None and existing != entry.payload:
            raise PortfolioRiskPersistenceError(f"authorization {authorization_id!r} was recorded twice with a different payload")
        index[authorization_id] = entry.payload
    return index


def build_status_events_index(entries: list[RiskLedgerEntry]) -> dict[str, list[RiskAuthorizationStatusEvent]]:
    """`authorization_id -> ordered list of RiskAuthorizationStatusEvent`,
    in ledger (append) order."""
    index: dict[str, list[RiskAuthorizationStatusEvent]] = {}
    for entry in entries:
        if entry.entry_kind not in _STATUS_ENTRY_KINDS:
            continue
        event = RiskAuthorizationStatusEvent.from_json_dict(entry.payload)
        index.setdefault(event.authorization_id, []).append(event)
    return index


def build_authorization_status_index(entries: list[RiskLedgerEntry]) -> dict[str, RiskAuthorizationStatus]:
    """`authorization_id -> current lifecycle state`, derived purely by
    replaying each authorization's own status events (`state_machine.
    resolve_risk_authorization_status`, which itself raises on any
    discontinuity or illegal transition -- corruption never resolves to
    a silently-accepted state).

    The authorization-id universe is seeded from `build_authorization_
    payload_index` (every `RISK_AUTHORIZATION_ISSUED` entry), NOT from
    `build_status_events_index` alone -- an authorization that was issued
    but has never had ANY subsequent transition recorded has an EMPTY
    event list, and `resolve_risk_authorization_status` correctly
    resolves an empty list to the implicit initial state (`ISSUED`).
    Seeding from the events index alone would silently omit every such
    authorization from this index entirely (not merely misclassify it --
    it would never appear as a key at all), which is exactly the failure
    mode a recovery/reconciliation pass must never have."""
    events_index = build_status_events_index(entries)
    issued_ids = build_authorization_payload_index(entries).keys()
    return {authorization_id: resolve_risk_authorization_status(authorization_id, events_index.get(authorization_id, [])) for authorization_id in issued_ids}


def build_authorization_consumption_index(entries: list[RiskLedgerEntry]) -> dict[str, str | None]:
    """`authorization_id -> its bound consumption_identity` (the identity
    of the economic use it was reserved/consumed for), or `None` if never
    reserved. Seeded from `build_authorization_payload_index` -- see
    `build_authorization_status_index`'s identical rationale -- so every
    issued authorization is a key, never silently absent."""
    events_index = build_status_events_index(entries)
    issued_ids = build_authorization_payload_index(entries).keys()
    return {authorization_id: consumption_identity_for(authorization_id, events_index.get(authorization_id, [])) for authorization_id in issued_ids}


def build_execution_intent_index(entries: list[RiskLedgerEntry]) -> dict[str, str]:
    """`execution_intent_id -> authorization_id`. Rejects one execution
    intent mapped to two different (conflicting) authorizations."""
    index: dict[str, str] = {}
    for entry in entries:
        if entry.entry_kind is not RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED:
            continue
        intent_id = str(entry.payload["execution_intent_id"])
        authorization_id = str(entry.payload["risk_authorization_id"])
        existing = index.get(intent_id)
        if existing is not None and existing != authorization_id:
            raise PortfolioRiskPersistenceError(
                f"execution_intent_id {intent_id!r} is bound to two different authorization ids: {existing!r} and {authorization_id!r}"
            )
        index[intent_id] = authorization_id
    return index


def build_consumption_identity_index(entries: list[RiskLedgerEntry]) -> dict[str, str]:
    """`economic submit identity (consumption_identity) -> authorization_id`.
    Rejects one economic submit identity bound to two different
    authorizations (one authorization consumed by two different
    economic submits is instead caught by `validate_authorization_use`'s
    own `CONFLICTING_CONSUMPTION` rejection at record time -- this index
    is the ledger-wide, after-the-fact defense-in-depth confirmation that
    no such conflict slipped through)."""
    index: dict[str, str] = {}
    events_index = build_status_events_index(entries)
    for authorization_id, events in events_index.items():
        for event in events:
            if event.consumption_identity is None:
                continue
            existing = index.get(event.consumption_identity)
            if existing is not None and existing != authorization_id:
                raise PortfolioRiskPersistenceError(
                    f"consumption_identity {event.consumption_identity!r} is bound to two different authorizations: {existing!r} and {authorization_id!r}"
                )
            index[event.consumption_identity] = authorization_id
    return index
