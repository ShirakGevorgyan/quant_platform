"""Unit tests for `portfolio_risk.reports`: report structure/content
correctness and the "no cached/stale derived values" requirement (every
section is recomputed fresh from the ledger on every call)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskAuthorizationReuseError
from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.decisions import create_risk_decision, create_risk_evaluation_request
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntryKind,
    create_risk_ledger_entry,
)
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    record_authorization_issuance,
    record_risk_decision,
    record_risk_evaluation_request,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import OrderSide, RiskDecisionKind
from quant_platform.portfolio_risk.reports import generate_portfolio_risk_session_report

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


def _full_scenario(store: PortfolioRiskLedgerStore) -> None:
    request = create_risk_evaluation_request(
        execution_intent_id="1" * 64, execution_session_id="2" * 64, portfolio_id="p1", strategy_id="strat-1", instrument_id="EURUSD",
        side=OrderSide.BUY, quantity=Decimal("1000"), portfolio_snapshot_id="3" * 64, price_snapshot_id="4" * 64, risk_policy_id="5" * 64,
        reduce_only=False, requested_sequence=0, event_time=_T0,
    )
    record_risk_evaluation_request(store, request, event_time=_T0)
    decision = create_risk_decision(
        risk_evaluation_request_id=request.risk_evaluation_request_id, kind=RiskDecisionKind.APPROVED, denial_reasons=(), check_results=(),
        evaluated_quantity=Decimal("1000"), evaluated_price=Decimal("1.10"), portfolio_snapshot_id="3" * 64, price_snapshot_id="4" * 64,
        risk_policy_id="5" * 64, decision_sequence=0, event_time=_T0,
    )
    record_risk_decision(store, decision, portfolio_id="p1", event_time=_T0)
    authorization = _authorization(risk_decision_id=decision.risk_decision_id)
    record_authorization_issuance(store, authorization, event_time=_T0)
    reserve_authorization(store, authorization, **_use_kwargs(authorization))
    consume_authorization(store, authorization, **_use_kwargs(authorization))


class TestReportStructureAndContent:
    def test_full_scenario_report_sections_reflect_recorded_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            _full_scenario(store)
            report = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)

            assert report.portfolio_id == "p1"
            sections = report.sections
            assert sections["RequestSummary"] == {"total_requests": 1}
            assert sections["DecisionSummary"]["total_decisions"] == 1
            assert sections["DecisionSummary"]["by_kind"] == {"approved": 1}
            assert sections["AuthorizationSummary"]["total_authorizations"] == 1
            assert sections["AuthorizationSummary"]["status_counts"] == {"consumed": 1}
            assert len(sections["AuthorizationBinding"]) == 1
            assert len(sections["LifecycleHistory"]) == 2  # RESERVED, CONSUMED
            assert len(sections["ConsumptionEvidence"]) == 1
            assert sections["RejectedUseAttempts"] == []
            assert sections["ReconciliationSummary"]["is_reconciled"] is True
            assert sections["VerificationSummary"]["critical_count"] == 0
            assert isinstance(sections["PhysicalLedgerDigest"], str) and len(sections["PhysicalLedgerDigest"]) == 64
            assert isinstance(sections["SemanticDigest"], str) and len(sections["SemanticDigest"]) == 64

    def test_empty_portfolio_report_has_zeroed_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            report = generate_portfolio_risk_session_report(portfolio_id="empty", store=store, report_time=_T0)
            sections = report.sections
            assert sections["RequestSummary"] == {"total_requests": 0}
            assert sections["AuthorizationSummary"]["total_authorizations"] == 0
            assert sections["LifecycleHistory"] == []


class TestRejectedUseAttemptsAppearInReport:
    def test_a_rejected_reuse_attempt_is_surfaced_in_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))

            report = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)
            assert len(report.sections["RejectedUseAttempts"]) == 1
            assert report.sections["RejectedUseAttempts"][0]["rejection_reason"] == "conflicting_consumption"


class TestNoCachedOrStaleDerivedValues:
    def test_generating_the_report_twice_reflects_the_ledger_growing_between_calls(self) -> None:
        # `generate_portfolio_risk_session_report` calls `verify_portfolio_
        # risk_session` with its default `record=True` -- every report
        # generation is itself a durable, auditable act that appends a NEW
        # VERIFICATION_COMPLETED entry. A second call must see the ledger
        # AFTER the first call's own append, never a value cached from the
        # first call.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)

            first = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)
            second = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)

            first_total = first.sections["PortfolioRiskSessionSummary"]["total_ledger_entries"]
            second_total = second.sections["PortfolioRiskSessionSummary"]["total_ledger_entries"]
            assert second_total == first_total + 1
            assert first.sections["SemanticDigest"] != second.sections["SemanticDigest"]

    def test_a_tamper_between_two_report_calls_is_reflected_in_the_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            clean = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)
            assert clean.sections["ReconciliationSummary"]["is_reconciled"] is True

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

            dirty = generate_portfolio_risk_session_report(portfolio_id="p1", store=store, report_time=_T0)
            assert dirty.sections["ReconciliationSummary"]["is_reconciled"] is False
            assert dirty.sections["VerificationSummary"]["critical_count"] > 0
