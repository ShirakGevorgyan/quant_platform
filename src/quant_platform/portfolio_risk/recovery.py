"""Deterministic crash recovery for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3). `recover_portfolio_risk_session` reconstructs
every authorization's lifecycle SOLELY from durable ledger evidence
(`idempotency.py`'s index builders, which themselves replay `state_
machine.resolve_risk_authorization_status`) -- it never trusts an
in-memory idempotency set, and it NEVER itself calls `lifecycle.
reserve_authorization`/`consume_authorization`/etc. Recovery only
CLASSIFIES and REPORTS; it cannot authorize a blind reuse because it
never mutates authorization status at all (only its own `RECOVERY_
STARTED`/`RECOVERY_COMPLETED` audit entries are appended).

WHY A `RESERVED`-WITH-NO-CONSUMPTION-EVIDENCE AUTHORIZATION STAYS
BLOCKED: this phase has no `execution_gateway` integration (explicitly
out of scope), so there is no external system to QUERY about whether the
downstream dispatch this reservation was made for actually happened.
Unlike `execution_gateway.recovery`'s own `recover_unknown_orders`
(which CAN query the adapter), this recovery function has nothing to
query -- it therefore classifies such a case as durably, honestly
`RESERVED_UNRESOLVED_BLOCKED` rather than pretending to resolve an
ambiguity it has no evidence to resolve. This is a genuine, deliberate
scope limitation (see `docs/portfolio_risk_architecture.md`'s Known
Limitations), not a defect. An EXACT retry of the SAME
`consumption_identity` remains safely, idempotently usable regardless
(structurally, via `validate_authorization_use`'s own exact-retry rule)
-- recovery does not need to, and does not, special-case that."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.portfolio_risk.idempotency import (
    build_authorization_consumption_index,
    build_authorization_status_index,
)
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntryKind,
    append_ledger_entry,
)
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus

__all__ = ["RecoveryAction", "recover_portfolio_risk_session"]

_TERMINAL_ACTION_BY_STATUS: dict[RiskAuthorizationStatus, str] = {
    RiskAuthorizationStatus.CONSUMED: "terminal_consumed",
    RiskAuthorizationStatus.EXPIRED: "terminal_expired",
    RiskAuthorizationStatus.INVALIDATED: "terminal_invalidated",
    RiskAuthorizationStatus.REVOKED: "terminal_revoked",
}


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    authorization_id: str
    status: RiskAuthorizationStatus
    action: str
    """One of: `"issued_only"`, `"reserved_unresolved_blocked"`,
    `"terminal_consumed"`, `"terminal_expired"`, `"terminal_invalidated"`,
    `"terminal_revoked"`."""
    detail: str

    def to_json_dict(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, "status": self.status.value, "action": self.action, "detail": self.detail}


def _classify(authorization_id: str, status: RiskAuthorizationStatus, consumption_identity: str | None) -> RecoveryAction:
    if status is RiskAuthorizationStatus.ISSUED:
        return RecoveryAction(authorization_id, status, "issued_only", "never reserved -- no ambiguity, remains normally usable")
    if status is RiskAuthorizationStatus.RESERVED:
        return RecoveryAction(
            authorization_id, status, "reserved_unresolved_blocked",
            f"reserved under consumption_identity={consumption_identity!r} with no consumption evidence and no execution-gateway integration "
            "to query external state -- durably blocked from any NEW consumption_identity; an exact retry of the same consumption_identity "
            "remains safely idempotent",
        )
    return RecoveryAction(authorization_id, status, _TERMINAL_ACTION_BY_STATUS[status], f"already {status.value}, terminal -- no action needed")


def recover_portfolio_risk_session(*, portfolio_id: str, store: PortfolioRiskLedgerStore, recovery_time: datetime) -> list[RecoveryAction]:
    append_ledger_entry(store, portfolio_id=portfolio_id, entry_kind=RiskLedgerEntryKind.RECOVERY_STARTED, payload={"portfolio_id": portfolio_id}, event_time=recovery_time)

    entries = store.read_events(portfolio_id)
    status_index = build_authorization_status_index(entries)
    consumption_index = build_authorization_consumption_index(entries)

    actions = [_classify(authorization_id, status, consumption_index.get(authorization_id)) for authorization_id, status in sorted(status_index.items())]

    append_ledger_entry(
        store, portfolio_id=portfolio_id, entry_kind=RiskLedgerEntryKind.RECOVERY_COMPLETED,
        payload={"portfolio_id": portfolio_id, "actions": [a.to_json_dict() for a in actions]}, event_time=recovery_time,
    )
    return actions
