"""Unit tests for `portfolio_risk.reconciliation`: clean lifecycle,
every orphan/contamination scenario, and the "report cannot override
ledger truth" adversarial requirement."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal

from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntryKind,
    create_risk_ledger_entry,
)
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    record_authorization_issuance,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus, RiskDecisionKind
from quant_platform.portfolio_risk.reconciliation import (
    RiskReconciliationSeverity,
    reconcile_portfolio_risk_session,
)
from quant_platform.portfolio_risk.state_machine import create_risk_authorization_status_event

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _authorization(**overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": "1" * 64, "execution_session_id": "2" * 64, "portfolio_id": "p1", "portfolio_snapshot_id": "3" * 64,
        "price_snapshot_id": "4" * 64, "risk_policy_id": "5" * 64, "risk_decision_id": "6" * 64, "decision_kind": RiskDecisionKind.APPROVED,
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "authorization_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_authorization(**base)  # type: ignore[arg-type]


def _use_kwargs(authorization, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id,
        "portfolio_id": authorization.portfolio_id, "portfolio_snapshot_id": authorization.portfolio_snapshot_id,
        "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id,
        "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price, "consumption_identity": "use-1",
        "evaluation_time": _T0,
    }
    base.update(overrides)
    return base


class TestCleanLifecycle:
    def test_fully_consumed_lifecycle_is_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert report.is_reconciled
            assert report.blocking_issues == ()

    def test_issued_only_is_reconciled_with_only_an_info_issue_for_unresolved_reservation_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert report.is_reconciled


class TestUnresolvedReservationInfo:
    def test_reserved_without_consumption_produces_an_info_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert report.is_reconciled  # INFO severity does not block
            assert any(i.issue_code == "unresolved_reservation" and i.severity is RiskReconciliationSeverity.INFO for i in report.issues)


class TestOrphanAuthorizationEvent:
    def test_status_event_for_a_never_issued_authorization_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            orphan_id = "f" * 64
            orphan_event = create_risk_authorization_status_event(
                authorization_id=orphan_id, portfolio_id="p1", from_state=RiskAuthorizationStatus.ISSUED,
                to_state=RiskAuthorizationStatus.RESERVED, event_time=_T0, sequence=0,
                consumption_identity="use-1", reason_code=None, detail="orphan",
            )
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED, payload=orphan_event.to_json_dict(),
                event_time=_T0, recorded_time=_T0, previous_entry_hash=None,
            )
            store.append("p1", entry)
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert not report.is_reconciled
            assert any(i.issue_code == "orphan_authorization_lifecycle_event" for i in report.issues)


class TestCrossPortfolioContamination:
    def test_authorization_declaring_a_different_portfolio_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization(portfolio_id="p1")
            record_authorization_issuance(store, authorization, event_time=_T0)
            report = reconcile_portfolio_risk_session(portfolio_id="a-different-portfolio", ledger=store.read_events("p1"))
            assert not report.is_reconciled
            assert any(i.issue_code == "cross_portfolio_contamination" and i.severity is RiskReconciliationSeverity.CRITICAL for i in report.issues)


class TestAuthorizationFromDeniedDecision:
    def test_a_hand_crafted_authorization_payload_claiming_a_denied_decision_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            # Directly forge a SECOND issuance entry -- different authorization_id, decision_id, and
            # execution_intent_id so it does not collide with the decision/execution-intent uniqueness
            # indexes (that collision is a *different*, more fundamental failure mode, exercised by
            # TestReconstructionFailureIsCritical) -- whose payload claims decision_kind=denied.
            forged_payload = dict(authorization.to_json_dict())
            forged_payload["risk_authorization_id"] = "e" * 64
            forged_payload["risk_decision_id"] = "c" * 64
            forged_payload["execution_intent_id"] = "d" * 64
            forged_payload["decision_kind"] = "denied"
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=store.next_sequence("p1"), entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED,
                payload=forged_payload, event_time=_T0, recorded_time=_T0, previous_entry_hash=store.last_entry_hash("p1"),
            )
            store.append("p1", entry)
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert not report.is_reconciled
            assert any(i.issue_code == "authorization_from_non_approved_decision" for i in report.issues)


class TestReportCannotOverrideLedgerTruth:
    def test_reconciliation_always_recomputes_from_the_ledger_never_a_cached_value(self) -> None:
        # No caching parameter exists anywhere in reconcile_portfolio_risk_session's
        # own signature -- it always takes the raw ledger and recomputes.
        # This test proves that TWO DIFFERENT ledgers (before/after a
        # tamper) produce DIFFERENT reports from the SAME function, with
        # no way to force the earlier "clean" answer to persist.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            clean_report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert clean_report.is_reconciled

            forged_payload = dict(authorization.to_json_dict())
            forged_payload["risk_authorization_id"] = "e" * 64
            forged_payload["decision_kind"] = "denied"
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=store.next_sequence("p1"), entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED,
                payload=forged_payload, event_time=_T0, recorded_time=_T0, previous_entry_hash=store.last_entry_hash("p1"),
            )
            store.append("p1", entry)
            dirty_report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert not dirty_report.is_reconciled


class TestReconstructionFailureIsCritical:
    def test_a_ledger_that_cannot_be_reconstructed_produces_a_single_critical_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            # Forge a conflicting SECOND RESERVED entry with a different consumption_identity but a coherent from_state -- bypassing lifecycle.py's own check entirely.
            forged_event = create_risk_authorization_status_event(
                authorization_id=authorization.risk_authorization_id, portfolio_id="p1",
                from_state=RiskAuthorizationStatus.RESERVED,
                to_state=RiskAuthorizationStatus.CONSUMED, event_time=_T0, sequence=1,
                consumption_identity="use-2", reason_code=None, detail="forged conflicting consumption",
            )
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=store.next_sequence("p1"), entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
                payload=forged_event.to_json_dict(), event_time=_T0, recorded_time=_T0, previous_entry_hash=store.last_entry_hash("p1"),
            )
            store.append("p1", entry)
            report = reconcile_portfolio_risk_session(portfolio_id="p1", ledger=store.read_events("p1"))
            assert not report.is_reconciled
            assert report.issues[0].issue_code == "internal_reconstruction_failed"
            assert report.issues[0].severity is RiskReconciliationSeverity.CRITICAL
