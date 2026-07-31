"""Deterministic reporting for `quant_platform.portfolio_risk` (Milestone
9, Phase 3) -- every section is recomputed FRESH from the ledger's own
raw entries each time (never a cached/stale derived value), mirroring
`execution_gateway.reports.generate_execution_session_report`'s
identical convention exactly. Calling this function TWICE against the
same ledger state independently re-runs reconciliation/verification both
times -- it never reuses a report from a prior call."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.portfolio_risk.idempotency import (
    build_authorization_consumption_index,
    build_authorization_payload_index,
    build_authorization_status_index,
    build_status_events_index,
)
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntryKind,
    compute_risk_ledger_physical_digest,
    compute_risk_ledger_semantic_digest,
)
from quant_platform.portfolio_risk.reconciliation import reconcile_portfolio_risk_session
from quant_platform.portfolio_risk.verification import verify_portfolio_risk_session

__all__ = ["PortfolioRiskSessionReport", "generate_portfolio_risk_session_report"]


@dataclass(frozen=True, slots=True)
class PortfolioRiskSessionReport:
    portfolio_id: str
    sections: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return {"portfolio_id": self.portfolio_id, "sections": self.sections}


def generate_portfolio_risk_session_report(*, portfolio_id: str, store: PortfolioRiskLedgerStore, report_time: datetime) -> PortfolioRiskSessionReport:
    ledger = store.read_events(portfolio_id)

    request_entries = [e for e in ledger if e.entry_kind is RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED]
    decision_entries = [e for e in ledger if e.entry_kind is RiskLedgerEntryKind.RISK_DECISION_RECORDED]
    decision_kind_counts: dict[str, int] = {}
    for entry in decision_entries:
        kind = str(entry.payload.get("kind"))
        decision_kind_counts[kind] = decision_kind_counts.get(kind, 0) + 1

    payload_index = build_authorization_payload_index(ledger)
    status_index = build_authorization_status_index(ledger)
    consumption_index = build_authorization_consumption_index(ledger)
    events_index = build_status_events_index(ledger)

    status_counts: dict[str, int] = {}
    for status in status_index.values():
        status_counts[status.value] = status_counts.get(status.value, 0) + 1

    lifecycle_history = [
        event.to_json_dict()
        for authorization_id in sorted(events_index.keys())
        for event in events_index[authorization_id]
    ]

    consumption_evidence = {authorization_id: identity for authorization_id, identity in consumption_index.items() if identity is not None}

    rejected_entries = [e for e in ledger if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED]

    reconciliation_report = reconcile_portfolio_risk_session(portfolio_id=portfolio_id, ledger=ledger)
    verification_report = verify_portfolio_risk_session(portfolio_id=portfolio_id, store=store, verification_time=report_time)

    # Re-read: verification just durably appended its own VERIFICATION_COMPLETED entry.
    final_ledger = store.read_events(portfolio_id)

    sections: dict[str, object] = {
        "PortfolioRiskSessionSummary": {"portfolio_id": portfolio_id, "total_ledger_entries": len(final_ledger)},
        "RequestSummary": {"total_requests": len(request_entries)},
        "DecisionSummary": {"total_decisions": len(decision_entries), "by_kind": decision_kind_counts},
        "AuthorizationSummary": {"total_authorizations": len(payload_index), "status_counts": status_counts},
        "AuthorizationBinding": dict(sorted(payload_index.items())),
        "LifecycleHistory": lifecycle_history,
        "ConsumptionEvidence": consumption_evidence,
        "RejectedUseAttempts": [e.payload for e in rejected_entries],
        "ReconciliationSummary": reconciliation_report.to_json_dict(),
        "VerificationSummary": {
            "critical_count": len(verification_report.criticals), "total_issue_count": len(verification_report.issues),
            "generated_at": verification_report.generated_at,
        },
        "PhysicalLedgerDigest": compute_risk_ledger_physical_digest(final_ledger),
        "SemanticDigest": compute_risk_ledger_semantic_digest(final_ledger),
    }
    return PortfolioRiskSessionReport(portfolio_id=portfolio_id, sections=sections)
