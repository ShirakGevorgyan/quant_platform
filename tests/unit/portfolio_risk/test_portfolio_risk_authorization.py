"""Unit tests for `portfolio_risk.authorization`: `RiskAuthorization`
identity and cross-binding mismatch detection."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskAuthorizationIdentityError
from quant_platform.portfolio_risk.authorization import (
    RiskAuthorization,
    create_risk_authorization,
    verify_risk_authorization_binding,
)
from quant_platform.portfolio_risk.models import RiskDecisionKind

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SHA_INTENT = "1" * 64
_SHA_SESSION = "2" * 64
_SHA_PORTFOLIO_SNAPSHOT = "3" * 64
_SHA_PRICE_SNAPSHOT = "4" * 64
_SHA_POLICY = "5" * 64
_SHA_DECISION = "6" * 64

_SHA_OTHER_INTENT = "7" * 64
_SHA_OTHER_SESSION = "8" * 64
_SHA_OTHER_PORTFOLIO_SNAPSHOT = "9" * 64
_SHA_OTHER_PRICE_SNAPSHOT = "a" * 64
_SHA_OTHER_POLICY = "b" * 64
_SHA_OTHER_DECISION = "c" * 64


def _authorization(**overrides: object) -> RiskAuthorization:
    base: dict[str, object] = {
        "execution_intent_id": _SHA_INTENT, "execution_session_id": _SHA_SESSION, "portfolio_id": "portfolio-1",
        "portfolio_snapshot_id": _SHA_PORTFOLIO_SNAPSHOT, "price_snapshot_id": _SHA_PRICE_SNAPSHOT, "risk_policy_id": _SHA_POLICY,
        "risk_decision_id": _SHA_DECISION, "decision_kind": RiskDecisionKind.APPROVED, "evaluated_quantity": Decimal("1000"),
        "evaluated_price": Decimal("1.10"), "authorization_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_authorization(**base)  # type: ignore[arg-type]


def _binding_kwargs_for(authorization: RiskAuthorization) -> dict[str, object]:
    return {
        "execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id,
        "portfolio_id": authorization.portfolio_id, "portfolio_snapshot_id": authorization.portfolio_snapshot_id,
        "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id,
        "risk_decision_id": authorization.risk_decision_id, "decision_kind": authorization.decision_kind,
        "evaluated_quantity": authorization.evaluated_quantity, "evaluated_price": authorization.evaluated_price,
    }


class TestRiskAuthorizationConstruction:
    def test_default_constructs(self) -> None:
        authorization = _authorization()
        assert len(authorization.risk_authorization_id) == 64

    @pytest.mark.parametrize(
        "field_name",
        ["execution_intent_id", "execution_session_id", "portfolio_snapshot_id", "price_snapshot_id", "risk_policy_id", "risk_decision_id"],
    )
    def test_non_sha256_reference_rejected(self, field_name: str) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(**{field_name: "not-a-hash"})

    def test_empty_portfolio_id_rejected(self) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(portfolio_id="")

    def test_non_positive_evaluated_quantity_rejected(self) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(evaluated_quantity=Decimal("0"))

    def test_non_positive_evaluated_price_rejected(self) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(evaluated_price=Decimal("0"))

    def test_negative_authorization_sequence_rejected(self) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(authorization_sequence=-1)

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(RiskAuthorizationIdentityError):
            _authorization(event_time=datetime(2026, 1, 1))


class TestRiskAuthorizationRoundTrip:
    def test_round_trips_through_json(self) -> None:
        authorization = _authorization()
        restored = RiskAuthorization.from_json_dict(authorization.to_json_dict())
        assert restored.to_json_dict() == authorization.to_json_dict()


class TestRiskAuthorizationIdentity:
    def test_deterministic(self) -> None:
        assert _authorization().risk_authorization_id == _authorization().risk_authorization_id

    def test_authorization_sequence_disambiguates_otherwise_identical_authorizations(self) -> None:
        a = _authorization(authorization_sequence=0).risk_authorization_id
        b = _authorization(authorization_sequence=1).risk_authorization_id
        assert a != b

    def test_event_time_participates_in_identity(self) -> None:
        from datetime import timedelta

        a = _authorization(event_time=_T0).risk_authorization_id
        b = _authorization(event_time=_T0 + timedelta(seconds=1)).risk_authorization_id
        assert a != b

    def test_evaluated_quantity_participates_in_identity(self) -> None:
        a = _authorization(evaluated_quantity=Decimal("1000")).risk_authorization_id
        b = _authorization(evaluated_quantity=Decimal("2000")).risk_authorization_id
        assert a != b

    def test_evaluated_price_participates_in_identity(self) -> None:
        a = _authorization(evaluated_price=Decimal("1.10")).risk_authorization_id
        b = _authorization(evaluated_price=Decimal("1.11")).risk_authorization_id
        assert a != b


class TestRiskAuthorizationBindingVerification:
    """`verify_risk_authorization_binding` is what a future dispatch gate
    calls -- these tests prove it correctly ACCEPTS the exact tuple an
    authorization was issued for and REJECTS every single-field
    deviation, satisfying the "must not be reusable for a different
    intent/quantity/price/policy/portfolio snapshot/execution session"
    requirement structurally."""

    def test_exact_matching_tuple_accepted(self) -> None:
        authorization = _authorization()
        assert verify_risk_authorization_binding(authorization, **_binding_kwargs_for(authorization))

    def test_cross_intent_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["execution_intent_id"] = _SHA_OTHER_INTENT
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_session_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["execution_session_id"] = _SHA_OTHER_SESSION
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_portfolio_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["portfolio_id"] = "a-different-portfolio"
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_portfolio_snapshot_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["portfolio_snapshot_id"] = _SHA_OTHER_PORTFOLIO_SNAPSHOT
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_price_snapshot_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["price_snapshot_id"] = _SHA_OTHER_PRICE_SNAPSHOT
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_policy_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["risk_policy_id"] = _SHA_OTHER_POLICY
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_cross_decision_mismatch_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["risk_decision_id"] = _SHA_OTHER_DECISION
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_different_quantity_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["evaluated_quantity"] = Decimal("9999")
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_different_price_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["evaluated_price"] = Decimal("9.99")
        assert not verify_risk_authorization_binding(authorization, **kwargs)

    def test_different_decision_kind_rejected(self) -> None:
        authorization = _authorization()
        kwargs = _binding_kwargs_for(authorization)
        kwargs["decision_kind"] = RiskDecisionKind.DENIED
        assert not verify_risk_authorization_binding(authorization, **kwargs)
