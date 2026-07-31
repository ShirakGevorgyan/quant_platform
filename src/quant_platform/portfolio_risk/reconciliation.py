"""Portfolio-risk-domain reconciliation for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3) -- mirrors `execution_gateway.reconciliation`'s
identical shape: ordinary mismatches become structured, non-raising
`RiskReconciliationIssue`s; only genuine structural corruption (the
ledger itself cannot be reconstructed at all) raises `PortfolioRisk
ReconciliationError`.

COHERENT-RE-CHAINING DEFENSE: a ledger with an entry REMOVED and its
physical hash chain re-computed to still validate (`verify_risk_ledger_
chain_integrity` alone would not catch this) is still caught here,
because `idempotency.build_authorization_status_index` replays each
authorization's OWN `RiskAuthorizationStatusEvent.from_state` chain via
`state_machine.resolve_risk_authorization_status` -- removing a `RESERVED`
event but leaving a `CONSUMED` event whose own `from_state=reserved`
payload field still claims one existed produces a `from_state` that no
longer matches the replayed `current` state, raising
`PortfolioRiskRecoveryError`, caught here and reported as `CRITICAL`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.core.exceptions import PortfolioRiskPersistenceError, PortfolioRiskRecoveryError
from quant_platform.portfolio_risk.idempotency import (
    build_authorization_payload_index,
    build_authorization_status_index,
    build_consumption_identity_index,
    build_decision_to_authorization_index,
    build_execution_intent_index,
)
from quant_platform.portfolio_risk.ledger import RiskLedgerEntry, RiskLedgerEntryKind

__all__ = ["PortfolioRiskReconciliationReport", "RiskReconciliationIssue", "RiskReconciliationSeverity", "reconcile_portfolio_risk_session"]

_STATUS_ENTRY_KINDS = frozenset({
    RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED, RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_EXPIRED, RiskLedgerEntryKind.RISK_AUTHORIZATION_INVALIDATED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_REVOKED,
})


class RiskReconciliationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"
    CRITICAL = "critical"


_BLOCKING_SEVERITIES = frozenset({RiskReconciliationSeverity.BLOCKING, RiskReconciliationSeverity.CRITICAL})


@dataclass(frozen=True, slots=True)
class RiskReconciliationIssue:
    issue_code: str
    severity: RiskReconciliationSeverity
    entity_type: str
    entity_id: str
    expected: str
    actual: str
    description: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "issue_code": self.issue_code, "severity": self.severity.value, "entity_type": self.entity_type, "entity_id": self.entity_id,
            "expected": self.expected, "actual": self.actual, "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class PortfolioRiskReconciliationReport:
    portfolio_id: str
    issues: tuple[RiskReconciliationIssue, ...]

    @property
    def is_reconciled(self) -> bool:
        return not any(issue.severity in _BLOCKING_SEVERITIES for issue in self.issues)

    @property
    def blocking_issues(self) -> tuple[RiskReconciliationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity in _BLOCKING_SEVERITIES)

    def to_json_dict(self) -> dict[str, object]:
        return {"portfolio_id": self.portfolio_id, "issues": [issue.to_json_dict() for issue in self.issues], "is_reconciled": self.is_reconciled}


def reconcile_portfolio_risk_session(*, portfolio_id: str, ledger: list[RiskLedgerEntry]) -> PortfolioRiskReconciliationReport:
    issues: list[RiskReconciliationIssue] = []

    try:
        decision_index = build_decision_to_authorization_index(ledger)
        payload_index = build_authorization_payload_index(ledger)
        status_index = build_authorization_status_index(ledger)
        build_execution_intent_index(ledger)
        build_consumption_identity_index(ledger)
    except (PortfolioRiskPersistenceError, PortfolioRiskRecoveryError) as exc:
        issues.append(RiskReconciliationIssue(
            issue_code="internal_reconstruction_failed", severity=RiskReconciliationSeverity.CRITICAL, entity_type="portfolio",
            entity_id=portfolio_id, expected="ledger reconstructs without conflict or illegal transition", actual=str(exc),
            description="The ledger could not be reconstructed at all -- structural corruption, a coherently re-chained tamper, or a genuine "
            "idempotency-index conflict.",
        ))
        return PortfolioRiskReconciliationReport(portfolio_id=portfolio_id, issues=tuple(issues))

    # Required invariant #1/#2: only an APPROVED decision may have produced a usable authorization.
    for authorization_id, payload in payload_index.items():
        if payload.get("decision_kind") != "approved":
            issues.append(RiskReconciliationIssue(
                issue_code="authorization_from_non_approved_decision", severity=RiskReconciliationSeverity.CRITICAL, entity_type="authorization",
                entity_id=authorization_id, expected="decision_kind == 'approved'", actual=str(payload.get("decision_kind")),
                description="A RiskAuthorization exists whose bound RiskDecision was not APPROVED -- structurally forbidden.",
            ))

    # Orphan lifecycle events: a status-transition entry referencing an authorization_id that was never issued.
    issued_ids = set(payload_index.keys())
    for entry in ledger:
        if entry.entry_kind not in _STATUS_ENTRY_KINDS:
            continue
        referenced_id = str(entry.payload.get("authorization_id"))
        if referenced_id not in issued_ids:
            issues.append(RiskReconciliationIssue(
                issue_code="orphan_authorization_lifecycle_event", severity=RiskReconciliationSeverity.CRITICAL, entity_type="authorization",
                entity_id=referenced_id, expected="a RISK_AUTHORIZATION_ISSUED entry exists for this authorization_id", actual="none found",
                description=f"A {entry.entry_kind.value} entry references an authorization that was never issued.",
            ))

    # Cross-portfolio contamination: an authorization's own declared portfolio_id does not match the ledger it was recorded in.
    for authorization_id, payload in payload_index.items():
        declared_portfolio_id = payload.get("portfolio_id")
        if declared_portfolio_id != portfolio_id:
            issues.append(RiskReconciliationIssue(
                issue_code="cross_portfolio_contamination", severity=RiskReconciliationSeverity.CRITICAL, entity_type="authorization",
                entity_id=authorization_id, expected=portfolio_id, actual=str(declared_portfolio_id),
                description="An authorization recorded in this portfolio's own ledger declares a different portfolio_id.",
            ))

    # Every RISK_DECISION_RECORDED entry with an APPROVED kind should have a corresponding issued authorization.
    for entry in ledger:
        if entry.entry_kind is not RiskLedgerEntryKind.RISK_DECISION_RECORDED:
            continue
        if entry.payload.get("kind") != "approved":
            continue
        decision_id = str(entry.payload.get("risk_decision_id"))
        if decision_id not in decision_index:
            issues.append(RiskReconciliationIssue(
                issue_code="approved_decision_never_issued", severity=RiskReconciliationSeverity.WARNING, entity_type="decision",
                entity_id=decision_id, expected="a RISK_AUTHORIZATION_ISSUED entry referencing this decision", actual="none found",
                description="An APPROVED decision was recorded but no authorization was ever issued from it (may be a legitimate caller choice).",
            ))

    # Unresolved RESERVED authorizations with no further evidence -- informational, not blocking (recovery.py's own job to classify further).
    for authorization_id, status in status_index.items():
        if status.value == "reserved":
            issues.append(RiskReconciliationIssue(
                issue_code="unresolved_reservation", severity=RiskReconciliationSeverity.INFO, entity_type="authorization",
                entity_id=authorization_id, expected="a terminal status (consumed/expired/invalidated/revoked)", actual="reserved",
                description="An authorization remains reserved with no terminal outcome recorded yet -- not necessarily a problem, but worth surfacing.",
            ))

    return PortfolioRiskReconciliationReport(portfolio_id=portfolio_id, issues=tuple(issues))
