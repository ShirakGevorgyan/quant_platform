"""Lifecycle transactions for `quant_platform.portfolio_risk` (Milestone
9, Phase 3) -- the only functions in this package that APPEND to a risk
ledger. Every transition is validated (`validation.
validate_authorization_use`) BEFORE anything is written; an exact retry
of an already-recorded transition is idempotently absorbed (no new
ledger entry, the prior event is returned unchanged); a genuine conflict
is durably recorded as a `RISK_AUTHORIZATION_USE_REJECTED` entry and then
raised, so a rejected attempt is never silently invisible in the ledger's
own audit trail.

The ledger is ALWAYS partitioned by `authorization.portfolio_id` (the
authorization's own TRUE, content-addressed owner) -- never by a
caller-claimed portfolio id, which may differ from the true one in a
cross-portfolio attack attempt (`validation.py`'s `BINDING_MISMATCH`
rejection is itself recorded in the TRUE owner's own ledger, exactly
where an attempted misuse against that portfolio belongs)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import RiskAuthorizationReuseError
from quant_platform.portfolio_risk.authorization import RiskAuthorization
from quant_platform.portfolio_risk.decisions import RiskDecision, RiskEvaluationRequest
from quant_platform.portfolio_risk.idempotency import build_status_events_index
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntry,
    RiskLedgerEntryKind,
    append_ledger_entry,
)
from quant_platform.portfolio_risk.models import (
    RiskAuthorizationStatus,
    is_legal_risk_authorization_status_transition,
)
from quant_platform.portfolio_risk.state_machine import (
    RiskAuthorizationStatusEvent,
    consumption_identity_for,
    create_risk_authorization_status_event,
    resolve_risk_authorization_status,
)
from quant_platform.portfolio_risk.validation import validate_authorization_use

__all__ = [
    "consume_authorization",
    "expire_authorization",
    "invalidate_authorization",
    "record_authorization_issuance",
    "record_risk_decision",
    "record_risk_evaluation_request",
    "reserve_authorization",
    "revoke_authorization",
]


def record_risk_evaluation_request(store: PortfolioRiskLedgerStore, request: RiskEvaluationRequest, *, event_time: datetime) -> RiskLedgerEntry:
    return append_ledger_entry(store, portfolio_id=request.portfolio_id, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload=request.to_json_dict(), event_time=event_time)


def record_risk_decision(store: PortfolioRiskLedgerStore, decision: RiskDecision, *, portfolio_id: str, event_time: datetime) -> RiskLedgerEntry:
    return append_ledger_entry(store, portfolio_id=portfolio_id, entry_kind=RiskLedgerEntryKind.RISK_DECISION_RECORDED, payload=decision.to_json_dict(), event_time=event_time)


def record_authorization_issuance(store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, event_time: datetime) -> RiskLedgerEntry:
    return append_ledger_entry(
        store, portfolio_id=authorization.portfolio_id, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED,
        payload=authorization.to_json_dict(), event_time=event_time,
    )


def _own_events(store: PortfolioRiskLedgerStore, authorization: RiskAuthorization) -> list[RiskAuthorizationStatusEvent]:
    entries = store.read_events(authorization.portfolio_id)
    return build_status_events_index(entries).get(authorization.risk_authorization_id, [])


_MAX_APPEND_RACE_RETRIES = 20
"""Bound on `_record_economic_use_transition`/`_record_administrative_
transition`'s optimistic-concurrency retry loop -- see their own
docstrings for why this exists. 20 is far beyond what any real thread
contention could exhaust (each retry only fires on an actual lost race at
the exact same ledger sequence slot); it exists purely so a pathological
scenario fails loudly instead of looping forever."""


def _record_economic_use_transition(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, target_status: RiskAuthorizationStatus, ledger_entry_kind: RiskLedgerEntryKind,
    execution_intent_id: str, execution_session_id: str, portfolio_id: str, portfolio_snapshot_id: str, price_snapshot_id: str, risk_policy_id: str,
    quantity: Decimal, price: Decimal, consumption_identity: str, detail: str, evaluation_time: datetime, expiry_time: datetime | None,
) -> RiskAuthorizationStatusEvent:
    """Retries its own read-validate-append cycle (bounded by `_MAX_
    APPEND_RACE_RETRIES`) on a losing race at the storage layer: two
    threads can both read state BEFORE either has written (both seeing
    the authorization as not-yet-reserved), both pass `validate_
    authorization_use` against that identical stale view, and then race
    `append_ledger_entry`'s own sequence-slot check -- the loser gets a
    raw `RiskAuthorizationReuseError` from `PortfolioRiskLedgerStore.
    append` with NO ledger entry recorded for its own attempt at all,
    which would silently violate this module's own invariant that a
    rejected attempt is never invisible in the ledger's own audit trail
    (a real defect found and fixed during this phase's own adversarial
    concurrency testing). Re-reading fresh state after losing such a race
    and re-validating against it resolves the retry to either an
    idempotent exact-retry absorption (the winner used the SAME
    `consumption_identity`) or a properly AUDITED `RISK_AUTHORIZATION_
    USE_REJECTED` entry (the winner used a different one) -- never a bare,
    unrecorded exception."""
    for _ in range(_MAX_APPEND_RACE_RETRIES):
        own_events = _own_events(store, authorization)
        current_status = resolve_risk_authorization_status(authorization.risk_authorization_id, own_events)
        bound_identity = consumption_identity_for(authorization.risk_authorization_id, own_events)

        outcome = validate_authorization_use(
            authorization=authorization, current_status=current_status, bound_consumption_identity=bound_identity, target_status=target_status,
            execution_intent_id=execution_intent_id, execution_session_id=execution_session_id, portfolio_id=portfolio_id,
            portfolio_snapshot_id=portfolio_snapshot_id, price_snapshot_id=price_snapshot_id, risk_policy_id=risk_policy_id, quantity=quantity,
            price=price, consumption_identity=consumption_identity, expiry_time=expiry_time, evaluation_time=evaluation_time,
        )

        if not outcome.approved:
            assert outcome.rejection_reason is not None
            append_ledger_entry(
                store, portfolio_id=authorization.portfolio_id, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED,
                payload={
                    "authorization_id": authorization.risk_authorization_id, "target_status": target_status.value,
                    "rejection_reason": outcome.rejection_reason.value, "detail": outcome.detail, "consumption_identity": consumption_identity,
                    "claimed_execution_intent_id": execution_intent_id, "claimed_portfolio_id": portfolio_id,
                },
                event_time=evaluation_time,
            )
            raise RiskAuthorizationReuseError(
                f"{target_status.value} rejected for authorization {authorization.risk_authorization_id!r}: {outcome.rejection_reason.value} -- {outcome.detail}"
            )

        if outcome.is_exact_retry:
            return own_events[-1]

        status_event = create_risk_authorization_status_event(
            authorization_id=authorization.risk_authorization_id, portfolio_id=authorization.portfolio_id, from_state=current_status,
            to_state=target_status, event_time=evaluation_time, sequence=len(own_events), consumption_identity=consumption_identity, reason_code=None,
            detail=detail,
        )
        try:
            append_ledger_entry(store, portfolio_id=authorization.portfolio_id, entry_kind=ledger_entry_kind, payload=status_event.to_json_dict(), event_time=evaluation_time)
        except RiskAuthorizationReuseError:
            continue  # lost a race for this exact sequence slot -- re-read fresh state and re-validate
        return status_event
    raise RiskAuthorizationReuseError(
        f"Could not resolve a ledger append race for authorization {authorization.risk_authorization_id!r} after {_MAX_APPEND_RACE_RETRIES} attempts"
    )


def reserve_authorization(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, execution_intent_id: str, execution_session_id: str, portfolio_id: str,
    portfolio_snapshot_id: str, price_snapshot_id: str, risk_policy_id: str, quantity: Decimal, price: Decimal, consumption_identity: str,
    evaluation_time: datetime, expiry_time: datetime | None = None,
) -> RiskAuthorizationStatusEvent:
    return _record_economic_use_transition(
        store, authorization, target_status=RiskAuthorizationStatus.RESERVED, ledger_entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED,
        execution_intent_id=execution_intent_id, execution_session_id=execution_session_id, portfolio_id=portfolio_id,
        portfolio_snapshot_id=portfolio_snapshot_id, price_snapshot_id=price_snapshot_id, risk_policy_id=risk_policy_id, quantity=quantity,
        price=price, consumption_identity=consumption_identity, detail="reserved for economic use", evaluation_time=evaluation_time,
        expiry_time=expiry_time,
    )


def consume_authorization(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, execution_intent_id: str, execution_session_id: str, portfolio_id: str,
    portfolio_snapshot_id: str, price_snapshot_id: str, risk_policy_id: str, quantity: Decimal, price: Decimal, consumption_identity: str,
    evaluation_time: datetime, expiry_time: datetime | None = None,
) -> RiskAuthorizationStatusEvent:
    return _record_economic_use_transition(
        store, authorization, target_status=RiskAuthorizationStatus.CONSUMED, ledger_entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
        execution_intent_id=execution_intent_id, execution_session_id=execution_session_id, portfolio_id=portfolio_id,
        portfolio_snapshot_id=portfolio_snapshot_id, price_snapshot_id=price_snapshot_id, risk_policy_id=risk_policy_id, quantity=quantity,
        price=price, consumption_identity=consumption_identity, detail="consumed", evaluation_time=evaluation_time, expiry_time=expiry_time,
    )


def _record_administrative_transition(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, target_status: RiskAuthorizationStatus, ledger_entry_kind: RiskLedgerEntryKind,
    reason_code: str, detail: str, evaluation_time: datetime,
) -> RiskAuthorizationStatusEvent:
    """Same optimistic-concurrency retry rationale as `_record_economic_
    use_transition` -- see its docstring."""
    for _ in range(_MAX_APPEND_RACE_RETRIES):
        own_events = _own_events(store, authorization)
        current_status = resolve_risk_authorization_status(authorization.risk_authorization_id, own_events)

        if current_status is target_status:
            return own_events[-1]  # idempotent -- exact same administrative transition already recorded

        if not is_legal_risk_authorization_status_transition(current_status, target_status):
            append_ledger_entry(
                store, portfolio_id=authorization.portfolio_id, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED,
                payload={
                    "authorization_id": authorization.risk_authorization_id, "target_status": target_status.value,
                    "rejection_reason": "status_does_not_permit_use", "detail": f"cannot transition from {current_status.value!r} to {target_status.value!r}",
                    "consumption_identity": None, "claimed_execution_intent_id": None, "claimed_portfolio_id": authorization.portfolio_id,
                },
                event_time=evaluation_time,
            )
            raise RiskAuthorizationReuseError(
                f"{target_status.value} rejected for authorization {authorization.risk_authorization_id!r}: cannot transition from "
                f"{current_status.value!r} to {target_status.value!r}"
            )

        status_event = create_risk_authorization_status_event(
            authorization_id=authorization.risk_authorization_id, portfolio_id=authorization.portfolio_id, from_state=current_status,
            to_state=target_status, event_time=evaluation_time, sequence=len(own_events), consumption_identity=None, reason_code=reason_code, detail=detail,
        )
        try:
            append_ledger_entry(store, portfolio_id=authorization.portfolio_id, entry_kind=ledger_entry_kind, payload=status_event.to_json_dict(), event_time=evaluation_time)
        except RiskAuthorizationReuseError:
            continue  # lost a race for this exact sequence slot -- re-read fresh state and re-validate
        return status_event
    raise RiskAuthorizationReuseError(
        f"Could not resolve a ledger append race for authorization {authorization.risk_authorization_id!r} after {_MAX_APPEND_RACE_RETRIES} attempts"
    )


def expire_authorization(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, reason_code: str, detail: str, evaluation_time: datetime,
) -> RiskAuthorizationStatusEvent:
    return _record_administrative_transition(
        store, authorization, target_status=RiskAuthorizationStatus.EXPIRED, ledger_entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_EXPIRED,
        reason_code=reason_code, detail=detail, evaluation_time=evaluation_time,
    )


def invalidate_authorization(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, reason_code: str, detail: str, evaluation_time: datetime,
) -> RiskAuthorizationStatusEvent:
    return _record_administrative_transition(
        store, authorization, target_status=RiskAuthorizationStatus.INVALIDATED, ledger_entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_INVALIDATED,
        reason_code=reason_code, detail=detail, evaluation_time=evaluation_time,
    )


def revoke_authorization(
    store: PortfolioRiskLedgerStore, authorization: RiskAuthorization, *, reason_code: str, detail: str, evaluation_time: datetime,
) -> RiskAuthorizationStatusEvent:
    return _record_administrative_transition(
        store, authorization, target_status=RiskAuthorizationStatus.REVOKED, ledger_entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_REVOKED,
        reason_code=reason_code, detail=detail, evaluation_time=evaluation_time,
    )
