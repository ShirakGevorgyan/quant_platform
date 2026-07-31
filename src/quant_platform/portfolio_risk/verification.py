"""Independent verification for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3). `verify_portfolio_risk_session` reuses `quant_
platform.ml.models.ValidationIssue`/`ValidationReport`/
`ValidationSeverity` directly (no new report shape), mirroring
`execution_gateway.verification`'s identical choice.

HONESTY CLASSIFICATION (documented explicitly, per the governing
instruction -- mirrors `execution_gateway.verification`'s own 3-tier
taxonomy):

- **STRUCTURALLY INDEPENDENT** (this module's entire scope): ledger
  physical chain integrity, portfolio ownership, idempotency-index
  reconstruction (which itself replays every authorization's lifecycle
  via `state_machine.resolve_risk_authorization_status`, catching a
  coherently re-chained tamper), APPROVED-only issuance, forged-identity
  detection (recomputing each authorization's own content id from its
  ledger-recorded payload), single-use-identity coherence, and orphan-
  event detection. None of these require trusting a cached report, a
  persisted status, an in-memory set, or a caller assertion -- every one
  is a pure recomputation from the ledger's own raw entries.
- **NOT INDEPENDENTLY RE-VERIFIED** (an honest, explicit limitation, not
  an oversight): this module does NOT re-run Phase 2's evaluator
  (`evaluator.evaluate_risk`) against the original `PortfolioSnapshot`/
  `PriceSnapshot`/`PortfolioRiskPolicy` to confirm a recorded
  `RiskDecision`'s own 18 checks were computed correctly in the first
  place -- the ledger only durably stores the DECISION's own already-
  serialized JSON (`RiskDecision.to_json_dict()`), not the full snapshot/
  policy inputs it was computed from. Re-deriving THAT would require this
  module to also durably store (or have access to) the original
  snapshots/policy, which is out of Phase 3's own scope (no snapshot/
  policy artifact store exists in this milestone). This module verifies
  the decision's OWN INTERNAL coherence (already guaranteed at
  construction by Phase 1/2's own `RiskDecision.__post_init__`) and its
  BINDING into an authorization -- never whether the decision's 18 checks
  were the economically correct ones for the real portfolio state."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import PortfolioRiskPersistenceError, PortfolioRiskRecoveryError
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp
from quant_platform.portfolio_risk.authorization import RiskAuthorization, verify_risk_authorization_binding
from quant_platform.portfolio_risk.idempotency import (
    build_authorization_payload_index,
    build_authorization_status_index,
    build_consumption_identity_index,
    build_decision_to_authorization_index,
    build_execution_intent_index,
    build_status_events_index,
)
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntry,
    RiskLedgerEntryKind,
    append_ledger_entry,
    compute_risk_ledger_semantic_digest,
    verify_risk_ledger_chain_integrity,
)
from quant_platform.portfolio_risk.models import RiskDecisionKind

__all__ = ["VERIFICATION_REPORT_SCHEMA_VERSION", "compute_verification_semantic_digest", "verify_portfolio_risk_session"]

VERIFICATION_REPORT_SCHEMA_VERSION = 1
_STATUS_ENTRY_KINDS = frozenset({
    RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED, RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_EXPIRED, RiskLedgerEntryKind.RISK_AUTHORIZATION_INVALIDATED,
    RiskLedgerEntryKind.RISK_AUTHORIZATION_REVOKED,
})


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def compute_verification_semantic_digest(ledger: list[RiskLedgerEntry]) -> str:
    return compute_risk_ledger_semantic_digest(ledger)


def verify_portfolio_risk_session(
    *, portfolio_id: str, store: PortfolioRiskLedgerStore, verification_time: datetime, record: bool = True,
) -> ValidationReport:
    """`record=True` (the default) durably appends a `VERIFICATION_
    COMPLETED` audit entry -- every independently-run verification is a
    real, auditable event worth recording, mirroring `recovery.py`'s own
    unconditional `RECOVERY_STARTED`/`RECOVERY_COMPLETED` bookkeeping.
    `record=False` performs the IDENTICAL checks with NO ledger mutation
    -- for a read-only comparison/measurement caller (`replay.py`'s own
    `compute_replay_result`) where appending an entry as a side effect of
    merely COMPARING two scenarios would make the comparison itself
    non-idempotent (calling it twice would change its own answer the
    second time)."""
    ledger = store.read_events(portfolio_id)
    issues: list[ValidationIssue] = []

    if not verify_risk_ledger_chain_integrity(ledger):
        issues.append(_issue(ValidationSeverity.CRITICAL, "ledger_chain_broken", "The risk ledger's physical hash chain does not validate."))

    for entry in ledger:
        if entry.portfolio_id != portfolio_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "ledger_portfolio_ownership_mismatch",
                f"Entry {entry.entry_id!r} declares portfolio_id={entry.portfolio_id!r}, expected {portfolio_id!r}.",
            ))

    try:
        decision_index = build_decision_to_authorization_index(ledger)
        payload_index = build_authorization_payload_index(ledger)
        status_index = build_authorization_status_index(ledger)
        events_index = build_status_events_index(ledger)
        build_execution_intent_index(ledger)
        build_consumption_identity_index(ledger)
    except (PortfolioRiskPersistenceError, PortfolioRiskRecoveryError) as exc:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "idempotency_index_reconstruction_failed",
            f"The ledger could not be independently reconstructed: {exc}",
        ))
        report = ValidationReport(
            schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues),
            generated_at=format_utc_timestamp(pd.Timestamp(verification_time)),
        )
        if record:
            _record_completion(store, portfolio_id, report, verification_time)
        return report

    for authorization_id, payload in payload_index.items():
        if payload.get("decision_kind") != RiskDecisionKind.APPROVED.value:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "authorization_from_non_approved_decision",
                f"Authorization {authorization_id!r} is bound to a decision with kind={payload.get('decision_kind')!r}, not 'approved'.",
            ))
        authorization = RiskAuthorization.from_json_dict(payload)
        if not verify_risk_authorization_binding(
            authorization, execution_intent_id=authorization.execution_intent_id, execution_session_id=authorization.execution_session_id,
            portfolio_id=authorization.portfolio_id, portfolio_snapshot_id=authorization.portfolio_snapshot_id,
            price_snapshot_id=authorization.price_snapshot_id, risk_policy_id=authorization.risk_policy_id,
            risk_decision_id=authorization.risk_decision_id, decision_kind=authorization.decision_kind,
            evaluated_quantity=authorization.evaluated_quantity, evaluated_price=authorization.evaluated_price,
        ):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "forged_authorization_identity",
                f"Authorization {authorization_id!r}'s own recorded fields do not reproduce its own id -- forged or tampered.",
            ))
        if decision_index.get(str(payload.get("risk_decision_id"))) != authorization_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "decision_authorization_binding_mismatch",
                f"Authorization {authorization_id!r}'s own risk_decision_id does not resolve back to it via the decision index.",
            ))

    issued_ids = set(payload_index.keys())
    for entry in ledger:
        if entry.entry_kind not in _STATUS_ENTRY_KINDS:
            continue
        referenced_id = str(entry.payload.get("authorization_id"))
        if referenced_id not in issued_ids:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "orphan_authorization_lifecycle_event",
                f"A {entry.entry_kind.value} entry references authorization {referenced_id!r}, which was never issued.",
            ))

    for authorization_id, events in events_index.items():
        identities = {event.consumption_identity for event in events if event.consumption_identity is not None}
        if len(identities) > 1:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "authorization_bound_to_multiple_consumption_identities",
                f"Authorization {authorization_id!r} is bound to {len(identities)} distinct consumption identities: {sorted(identities)!r}.",
            ))

    if status_index.keys() - payload_index.keys():
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "status_index_references_unissued_authorization",
            "The status index contains an authorization id absent from the payload index.",
        ))

    report = ValidationReport(
        schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(verification_time)),
    )
    if record:
        _record_completion(store, portfolio_id, report, verification_time)
    return report


def _record_completion(store: PortfolioRiskLedgerStore, portfolio_id: str, report: ValidationReport, verification_time: datetime) -> None:
    ledger = store.read_events(portfolio_id)
    payload = {
        "portfolio_id": portfolio_id, "critical_count": len(report.criticals), "issue_count": len(report.issues),
        "semantic_digest": compute_verification_semantic_digest(ledger),
    }
    append_ledger_entry(store, portfolio_id=portfolio_id, entry_kind=RiskLedgerEntryKind.VERIFICATION_COMPLETED, payload=payload, event_time=verification_time)
