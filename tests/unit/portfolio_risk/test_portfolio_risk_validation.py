"""Unit tests for `portfolio_risk.validation`: `validate_authorization_use`
across every binding/expiry/status/conflict scenario, and
`AuthorizationUseValidation`'s own construction invariants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus, RiskDecisionKind
from quant_platform.portfolio_risk.validation import (
    AuthorizationRejectionReason,
    AuthorizationUseValidation,
    validate_authorization_use,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _authorization(**overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": "1" * 64, "execution_session_id": "2" * 64, "portfolio_id": "p1", "portfolio_snapshot_id": "3" * 64,
        "price_snapshot_id": "4" * 64, "risk_policy_id": "5" * 64, "risk_decision_id": "6" * 64, "decision_kind": RiskDecisionKind.APPROVED,
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "authorization_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_authorization(**base)  # type: ignore[arg-type]


def _validate(authorization, **overrides: object) -> AuthorizationUseValidation:
    base: dict[str, object] = {
        "authorization": authorization, "current_status": RiskAuthorizationStatus.ISSUED, "bound_consumption_identity": None,
        "target_status": RiskAuthorizationStatus.RESERVED, "execution_intent_id": authorization.execution_intent_id,
        "execution_session_id": authorization.execution_session_id, "portfolio_id": authorization.portfolio_id,
        "portfolio_snapshot_id": authorization.portfolio_snapshot_id, "price_snapshot_id": authorization.price_snapshot_id,
        "risk_policy_id": authorization.risk_policy_id, "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price,
        "consumption_identity": "use-1", "expiry_time": None, "evaluation_time": _T0,
    }
    base.update(overrides)
    return validate_authorization_use(**base)  # type: ignore[arg-type]


class TestAuthorizationUseValidationConstruction:
    def test_approved_forbids_rejection_reason(self) -> None:
        with pytest.raises(ValueError):
            AuthorizationUseValidation(approved=True, rejection_reason=AuthorizationRejectionReason.EXPIRED, detail="x", is_exact_retry=False)

    def test_rejected_requires_rejection_reason(self) -> None:
        with pytest.raises(ValueError):
            AuthorizationUseValidation(approved=False, rejection_reason=None, detail="x", is_exact_retry=False)

    def test_exact_retry_requires_approved(self) -> None:
        with pytest.raises(ValueError):
            AuthorizationUseValidation(approved=False, rejection_reason=AuthorizationRejectionReason.EXPIRED, detail="x", is_exact_retry=True)


class TestNewTransitionApproved:
    def test_issued_to_reserved_is_a_new_approved_transition(self) -> None:
        authorization = _authorization()
        result = _validate(authorization)
        assert result.approved and not result.is_exact_retry

    def test_reserved_to_consumed_is_a_new_approved_transition(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, current_status=RiskAuthorizationStatus.RESERVED, bound_consumption_identity="use-1", target_status=RiskAuthorizationStatus.CONSUMED)
        assert result.approved and not result.is_exact_retry


class TestExactRetryIdempotent:
    def test_same_target_and_identity_is_an_exact_retry(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, current_status=RiskAuthorizationStatus.RESERVED, bound_consumption_identity="use-1", target_status=RiskAuthorizationStatus.RESERVED)
        assert result.approved and result.is_exact_retry


class TestConflictingUseRejected:
    def test_same_target_different_identity_is_a_conflict(self) -> None:
        authorization = _authorization()
        result = _validate(
            authorization, current_status=RiskAuthorizationStatus.RESERVED, bound_consumption_identity="use-1", target_status=RiskAuthorizationStatus.RESERVED,
            consumption_identity="use-2",
        )
        assert not result.approved
        assert result.rejection_reason is AuthorizationRejectionReason.CONFLICTING_CONSUMPTION
        assert not result.is_exact_retry

    def test_a_new_transition_into_consumed_under_a_different_identity_than_it_was_reserved_is_a_conflict(self) -> None:
        # Regression test for a real, confirmed defect found during this
        # phase's own adversarial concurrency testing: RESERVED -> CONSUMED
        # is a NEW transition (current_status is NOT target_status), so it
        # used to fall straight through to unconditional approval without
        # ever comparing `consumption_identity` against `bound_consumption_
        # identity` -- silently letting a first-time CONSUME use a
        # completely different economic identity than the one the
        # authorization was reserved under.
        authorization = _authorization()
        result = _validate(
            authorization, current_status=RiskAuthorizationStatus.RESERVED, bound_consumption_identity="use-1",
            target_status=RiskAuthorizationStatus.CONSUMED, consumption_identity="use-2",
        )
        assert not result.approved
        assert result.rejection_reason is AuthorizationRejectionReason.CONFLICTING_CONSUMPTION
        assert not result.is_exact_retry

    def test_a_new_transition_into_consumed_under_the_same_reserved_identity_is_approved(self) -> None:
        authorization = _authorization()
        result = _validate(
            authorization, current_status=RiskAuthorizationStatus.RESERVED, bound_consumption_identity="use-1",
            target_status=RiskAuthorizationStatus.CONSUMED, consumption_identity="use-1",
        )
        assert result.approved
        assert not result.is_exact_retry

    def test_the_first_reservation_with_no_prior_bound_identity_is_approved(self) -> None:
        authorization = _authorization()
        result = _validate(
            authorization, current_status=RiskAuthorizationStatus.ISSUED, bound_consumption_identity=None,
            target_status=RiskAuthorizationStatus.RESERVED, consumption_identity="use-1",
        )
        assert result.approved
        assert not result.is_exact_retry


class TestBindingMismatchRejected:
    @pytest.mark.parametrize(
        "field_name,bad_value",
        [
            ("execution_intent_id", "f" * 64), ("execution_session_id", "f" * 64), ("portfolio_id", "wrong-portfolio"),
            ("portfolio_snapshot_id", "f" * 64), ("price_snapshot_id", "f" * 64), ("risk_policy_id", "f" * 64),
        ],
    )
    def test_each_bound_field_mismatch_is_rejected(self, field_name: str, bad_value: str) -> None:
        authorization = _authorization()
        result = _validate(authorization, **{field_name: bad_value})
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.BINDING_MISMATCH

    def test_quantity_mismatch_rejected(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, quantity=Decimal("9999"))
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.BINDING_MISMATCH

    def test_price_mismatch_rejected(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, price=Decimal("9.99"))
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.BINDING_MISMATCH


class TestExpiry:
    def test_before_expiry_is_fine(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, expiry_time=_T0 + timedelta(hours=1), evaluation_time=_T0)
        assert result.approved

    def test_exactly_at_expiry_is_fine(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, expiry_time=_T0, evaluation_time=_T0)
        assert result.approved

    def test_after_expiry_is_rejected(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, expiry_time=_T0, evaluation_time=_T0 + timedelta(seconds=1))
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.EXPIRED


class TestStatusDoesNotPermitUse:
    def test_consume_before_reserve_rejected(self) -> None:
        authorization = _authorization()
        result = _validate(authorization, current_status=RiskAuthorizationStatus.ISSUED, target_status=RiskAuthorizationStatus.CONSUMED)
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.STATUS_DOES_NOT_PERMIT_USE

    @pytest.mark.parametrize(
        "terminal", [RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED, RiskAuthorizationStatus.REVOKED]
    )
    def test_reserve_after_any_terminal_status_rejected(self, terminal: RiskAuthorizationStatus) -> None:
        authorization = _authorization()
        result = _validate(authorization, current_status=terminal, target_status=RiskAuthorizationStatus.RESERVED)
        assert not result.approved and result.rejection_reason is AuthorizationRejectionReason.STATUS_DOES_NOT_PERMIT_USE
