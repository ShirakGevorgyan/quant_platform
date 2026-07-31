"""Unit tests for `portfolio_risk.decisions`: `RiskCheckResult`,
`RiskEvaluationRequest`, and `RiskDecision`."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskEvaluationError
from quant_platform.portfolio_risk.decisions import (
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
    create_risk_decision,
    create_risk_evaluation_request,
)
from quant_platform.portfolio_risk.models import (
    OrderSide,
    RiskCheckSeverity,
    RiskDecisionKind,
    RiskDenialReason,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _request(**overrides: object) -> RiskEvaluationRequest:
    base: dict[str, object] = {
        "execution_intent_id": _SHA_A, "execution_session_id": _SHA_B, "portfolio_id": "portfolio-1", "strategy_id": "strategy-a",
        "instrument_id": "EURUSD", "side": OrderSide.BUY, "quantity": Decimal("1000"), "portfolio_snapshot_id": _SHA_C,
        "price_snapshot_id": _SHA_D, "risk_policy_id": _SHA_E, "reduce_only": False, "requested_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_evaluation_request(**base)  # type: ignore[arg-type]


def _passing_check(check_identity: str = "max_order_notional") -> RiskCheckResult:
    return RiskCheckResult(
        check_identity=check_identity, measured_value=Decimal("1000"), limit_value=Decimal("100000"), passed=True, severity=RiskCheckSeverity.INFO,
        denial_reason=None,
    )


def _denying_check(*, severity: RiskCheckSeverity = RiskCheckSeverity.DENY, reason: RiskDenialReason = RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED) -> RiskCheckResult:
    return RiskCheckResult(
        check_identity="max_order_notional", measured_value=Decimal("200000"), limit_value=Decimal("100000"), passed=False, severity=severity,
        denial_reason=reason,
    )


def _decision(**overrides: object) -> RiskDecision:
    base: dict[str, object] = {
        "risk_evaluation_request_id": _SHA_A, "kind": RiskDecisionKind.APPROVED, "denial_reasons": (), "check_results": (_passing_check(),),
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "portfolio_snapshot_id": _SHA_C, "price_snapshot_id": _SHA_D,
        "risk_policy_id": _SHA_E, "decision_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_decision(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# RiskCheckResult
# --------------------------------------------------------------------------
class TestRiskCheckResultInvariants:
    def test_passing_check_constructs(self) -> None:
        check = _passing_check()
        assert check.passed is True

    def test_passing_check_with_non_info_severity_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(
                check_identity="x", measured_value=Decimal("1"), limit_value=Decimal("2"), passed=True, severity=RiskCheckSeverity.WARNING,
                denial_reason=None,
            )

    def test_passing_check_with_denial_reason_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(
                check_identity="x", measured_value=Decimal("1"), limit_value=Decimal("2"), passed=True, severity=RiskCheckSeverity.INFO,
                denial_reason=RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,
            )

    def test_failing_check_with_info_severity_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(
                check_identity="x", measured_value=Decimal("1"), limit_value=Decimal("2"), passed=False, severity=RiskCheckSeverity.INFO,
                denial_reason=RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,
            )

    def test_failing_check_without_denial_reason_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(
                check_identity="x", measured_value=Decimal("1"), limit_value=Decimal("2"), passed=False, severity=RiskCheckSeverity.DENY,
                denial_reason=None,
            )

    def test_empty_check_identity_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(check_identity="", measured_value=Decimal("1"), limit_value=Decimal("2"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)

    def test_round_trips_through_json(self) -> None:
        check = _denying_check()
        restored = RiskCheckResult.from_json_dict(check.to_json_dict())
        assert restored.to_json_dict() == check.to_json_dict()


# --------------------------------------------------------------------------
# RiskEvaluationRequest
# --------------------------------------------------------------------------
class TestRiskEvaluationRequestInvariants:
    def test_default_constructs(self) -> None:
        request = _request()
        assert len(request.risk_evaluation_request_id) == 64

    @pytest.mark.parametrize("field_name", ["execution_intent_id", "execution_session_id", "portfolio_snapshot_id", "price_snapshot_id", "risk_policy_id"])
    def test_non_sha256_reference_rejected(self, field_name: str) -> None:
        with pytest.raises(RiskEvaluationError):
            _request(**{field_name: "not-a-hash"})

    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _request(quantity=Decimal("0"))

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _request(event_time=datetime(2026, 1, 1))

    def test_negative_requested_sequence_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _request(requested_sequence=-1)


class TestRiskEvaluationRequestIdentity:
    def test_deterministic(self) -> None:
        assert _request().risk_evaluation_request_id == _request().risk_evaluation_request_id

    def test_different_execution_intent_id_changes_identity(self) -> None:
        a = _request(execution_intent_id=_SHA_A).risk_evaluation_request_id
        b = _request(execution_intent_id=_SHA_B).risk_evaluation_request_id
        assert a != b

    def test_round_trips_through_json(self) -> None:
        request = _request()
        restored = RiskEvaluationRequest.from_json_dict(request.to_json_dict())
        assert restored.to_json_dict() == request.to_json_dict()


# --------------------------------------------------------------------------
# RiskDecision
# --------------------------------------------------------------------------
class TestRiskDecisionApprovedCoherence:
    def test_approved_with_only_passing_checks_constructs(self) -> None:
        decision = _decision(kind=RiskDecisionKind.APPROVED, denial_reasons=(), check_results=(_passing_check(),))
        assert decision.is_approved

    def test_approved_with_denial_reasons_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(kind=RiskDecisionKind.APPROVED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,), check_results=(_passing_check(),))

    def test_approved_with_a_deny_severity_check_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(kind=RiskDecisionKind.APPROVED, denial_reasons=(), check_results=(_passing_check(), _denying_check(severity=RiskCheckSeverity.DENY)))

    def test_approved_with_a_halt_severity_check_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(kind=RiskDecisionKind.APPROVED, denial_reasons=(), check_results=(_denying_check(severity=RiskCheckSeverity.HALT),))


class TestRiskDecisionDeniedCoherence:
    def test_denied_requires_at_least_one_denial_reason(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(kind=RiskDecisionKind.DENIED, denial_reasons=(), check_results=(_denying_check(severity=RiskCheckSeverity.DENY),))

    def test_denied_with_matching_reason_constructs(self) -> None:
        decision = _decision(
            kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,),
            check_results=(_denying_check(severity=RiskCheckSeverity.DENY),),
        )
        assert not decision.is_approved

    def test_denied_missing_a_reason_present_in_check_results_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(
                kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.LEVERAGE_LIMIT_EXCEEDED,),
                check_results=(_denying_check(severity=RiskCheckSeverity.DENY, reason=RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED),),
            )

    def test_denied_with_a_halt_severity_check_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(
                kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,),
                check_results=(_denying_check(severity=RiskCheckSeverity.HALT),),
            )


class TestRiskDecisionHaltedCoherence:
    def test_halted_requires_a_halt_severity_check(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(
                kind=RiskDecisionKind.HALTED, denial_reasons=(RiskDenialReason.PORTFOLIO_HALTED,),
                check_results=(_denying_check(severity=RiskCheckSeverity.DENY, reason=RiskDenialReason.PORTFOLIO_HALTED),),
            )

    def test_halted_with_halt_severity_check_constructs(self) -> None:
        decision = _decision(
            kind=RiskDecisionKind.HALTED, denial_reasons=(RiskDenialReason.PORTFOLIO_HALTED,),
            check_results=(_denying_check(severity=RiskCheckSeverity.HALT, reason=RiskDenialReason.PORTFOLIO_HALTED),),
        )
        assert decision.kind is RiskDecisionKind.HALTED


class TestRiskDecisionBasicFieldValidation:
    def test_non_positive_evaluated_quantity_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(evaluated_quantity=Decimal("0"))

    def test_non_positive_evaluated_price_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(evaluated_price=Decimal("0"))

    def test_negative_decision_sequence_rejected(self) -> None:
        with pytest.raises(RiskEvaluationError):
            _decision(decision_sequence=-1)


class TestRiskDecisionIdentity:
    def test_deterministic(self) -> None:
        assert _decision().risk_decision_id == _decision().risk_decision_id

    def test_different_kind_changes_identity_when_still_coherent(self) -> None:
        approved = _decision(kind=RiskDecisionKind.APPROVED, denial_reasons=(), check_results=(_passing_check(),)).risk_decision_id
        denied = _decision(
            kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,),
            check_results=(_denying_check(severity=RiskCheckSeverity.DENY),),
        ).risk_decision_id
        assert approved != denied

    def test_round_trips_through_json(self) -> None:
        decision = _decision(
            kind=RiskDecisionKind.DENIED, denial_reasons=(RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED,),
            check_results=(_denying_check(severity=RiskCheckSeverity.DENY),),
        )
        restored = RiskDecision.from_json_dict(decision.to_json_dict())
        assert restored.to_json_dict() == decision.to_json_dict()
