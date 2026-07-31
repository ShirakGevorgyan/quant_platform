"""Unit tests for `portfolio_risk.issuance`: `issue_risk_authorization`
only ever produces an authorization from an APPROVED decision, and
recomputes identity rather than trusting caller-supplied ids."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskAuthorizationIdentityError, RiskDenialError
from quant_platform.portfolio_risk.decisions import create_risk_decision, create_risk_evaluation_request
from quant_platform.portfolio_risk.issuance import issue_risk_authorization
from quant_platform.portfolio_risk.models import (
    OrderSide,
    RiskCheckSeverity,
    RiskDecisionKind,
    RiskDenialReason,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SHA = {name: c * 64 for name, c in zip(("intent", "session", "portfolio_snap", "price_snap", "policy"), "123456789abcdef", strict=False)}


def _request(**overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": "1" * 64, "execution_session_id": "2" * 64, "portfolio_id": "p1", "strategy_id": "s1", "instrument_id": "EURUSD",
        "side": OrderSide.BUY, "quantity": Decimal("1000"), "portfolio_snapshot_id": "3" * 64, "price_snapshot_id": "4" * 64,
        "risk_policy_id": "5" * 64, "reduce_only": False, "requested_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_evaluation_request(**base)  # type: ignore[arg-type]


def _approved_decision(request, **overrides: object):
    base: dict[str, object] = {
        "risk_evaluation_request_id": request.risk_evaluation_request_id, "kind": RiskDecisionKind.APPROVED, "denial_reasons": (), "check_results": (),
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "portfolio_snapshot_id": request.portfolio_snapshot_id,
        "price_snapshot_id": request.price_snapshot_id, "risk_policy_id": request.risk_policy_id, "decision_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_decision(**base)  # type: ignore[arg-type]


def _denied_decision(request):
    from quant_platform.portfolio_risk.decisions import RiskCheckResult

    check = RiskCheckResult(
        check_identity="order_notional_limit", measured_value=Decimal("999999"), limit_value=Decimal("1"), passed=False,
        severity=RiskCheckSeverity.DENY, denial_reason=RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,
    )
    return _approved_decision(request, kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,), check_results=(check,))


def _halted_decision(request):
    from quant_platform.portfolio_risk.decisions import RiskCheckResult

    check = RiskCheckResult(
        check_identity="portfolio_halted", measured_value=Decimal("1"), limit_value=Decimal("0"), passed=False, severity=RiskCheckSeverity.HALT,
        denial_reason=RiskDenialReason.PORTFOLIO_HALTED,
    )
    return _approved_decision(request, kind=RiskDecisionKind.HALTED, denial_reasons=(RiskDenialReason.PORTFOLIO_HALTED,), check_results=(check,))


class TestApprovedDecisionIssuesAuthorization:
    def test_issues_successfully(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        authorization = issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)
        assert authorization.execution_intent_id == request.execution_intent_id
        assert authorization.risk_decision_id == decision.risk_decision_id
        assert authorization.evaluated_quantity == decision.evaluated_quantity
        assert authorization.evaluated_price == decision.evaluated_price

    def test_deterministic_issuance_identity(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        a = issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)
        b = issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)
        assert a.risk_authorization_id == b.risk_authorization_id

    def test_different_authorization_sequence_changes_identity(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        a = issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)
        b = issue_risk_authorization(request=request, decision=decision, authorization_sequence=1, event_time=_T0)
        assert a.risk_authorization_id != b.risk_authorization_id


class TestDeniedCannotIssue:
    def test_denied_raises(self) -> None:
        request = _request()
        decision = _denied_decision(request)
        with pytest.raises(RiskDenialError):
            issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)


class TestHaltedCannotIssue:
    def test_halted_raises(self) -> None:
        request = _request()
        decision = _halted_decision(request)
        with pytest.raises(RiskDenialError):
            issue_risk_authorization(request=request, decision=decision, authorization_sequence=0, event_time=_T0)


class TestForgedIdentityRejected:
    def test_forged_decision_identity_rejected(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        tampered_raw = decision.to_json_dict()
        tampered_raw["evaluated_quantity"] = "999999"
        tampered = decision.__class__.from_json_dict(tampered_raw)
        with pytest.raises(RiskAuthorizationIdentityError):
            issue_risk_authorization(request=request, decision=tampered, authorization_sequence=0, event_time=_T0)

    def test_forged_request_identity_rejected(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        tampered_raw = request.to_json_dict()
        tampered_raw["quantity"] = "9999999"
        tampered_request = request.__class__.from_json_dict(tampered_raw)
        with pytest.raises(RiskAuthorizationIdentityError):
            issue_risk_authorization(request=tampered_request, decision=decision, authorization_sequence=0, event_time=_T0)


class TestCrossRequestMismatchRejected:
    def test_decision_bound_to_a_different_request_is_rejected(self) -> None:
        request_a = _request(quantity=Decimal("1000"))
        request_b = _request(quantity=Decimal("2000"))
        decision_for_b = _approved_decision(request_b)
        with pytest.raises(RiskAuthorizationIdentityError):
            issue_risk_authorization(request=request_a, decision=decision_for_b, authorization_sequence=0, event_time=_T0)

    def test_decision_snapshot_mismatch_rejected(self) -> None:
        request = _request()
        decision = _approved_decision(request)
        mismatched_decision = replace(decision, portfolio_snapshot_id="f" * 64)
        # replace() bypasses the create_* factory, so the id won't self-verify either --
        # issuance must catch the cross-field mismatch regardless of which check fires first.
        with pytest.raises(RiskAuthorizationIdentityError):
            issue_risk_authorization(request=request, decision=mismatched_decision, authorization_sequence=0, event_time=_T0)
