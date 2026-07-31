"""Unit tests for `portfolio_risk.state_machine`:
`RiskAuthorizationStatusEvent` construction invariants and
`resolve_risk_authorization_status`'s replay semantics."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import PortfolioRiskRecoveryError
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus
from quant_platform.portfolio_risk.state_machine import (
    consumption_identity_for,
    create_risk_authorization_status_event,
    resolve_risk_authorization_status,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_AUTH_ID = "a" * 64


def _event(**overrides: object):
    base: dict[str, object] = {
        "authorization_id": _AUTH_ID, "portfolio_id": "p1", "from_state": RiskAuthorizationStatus.ISSUED,
        "to_state": RiskAuthorizationStatus.RESERVED, "event_time": _T0, "sequence": 0, "consumption_identity": "use-1", "reason_code": None,
        "detail": "reserved",
    }
    base.update(overrides)
    return create_risk_authorization_status_event(**base)  # type: ignore[arg-type]


class TestConstructionInvariants:
    def test_valid_reservation_constructs(self) -> None:
        event = _event()
        assert event.to_state is RiskAuthorizationStatus.RESERVED

    def test_illegal_transition_rejected(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(from_state=RiskAuthorizationStatus.ISSUED, to_state=RiskAuthorizationStatus.CONSUMED)

    def test_consumption_identity_required_for_reserved(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(to_state=RiskAuthorizationStatus.RESERVED, consumption_identity=None)

    def test_consumption_identity_required_for_consumed(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(from_state=RiskAuthorizationStatus.RESERVED, to_state=RiskAuthorizationStatus.CONSUMED, consumption_identity=None)

    def test_consumption_identity_forbidden_for_expired(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(to_state=RiskAuthorizationStatus.EXPIRED, consumption_identity="use-1", reason_code="timed out")

    def test_reason_code_required_for_expired(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(to_state=RiskAuthorizationStatus.EXPIRED, consumption_identity=None, reason_code=None)

    def test_reason_code_forbidden_for_reserved(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(to_state=RiskAuthorizationStatus.RESERVED, reason_code="should not be here")

    def test_non_sha256_authorization_id_rejected(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(authorization_id="not-a-hash")

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(event_time=datetime(2026, 1, 1))

    def test_negative_sequence_rejected(self) -> None:
        with pytest.raises(PortfolioRiskRecoveryError):
            _event(sequence=-1)


class TestRoundTrip:
    def test_round_trips_through_json(self) -> None:
        from quant_platform.portfolio_risk.state_machine import RiskAuthorizationStatusEvent

        event = _event()
        restored = RiskAuthorizationStatusEvent.from_json_dict(event.to_json_dict())
        assert restored.to_json_dict() == event.to_json_dict()


class TestResolveRiskAuthorizationStatus:
    def test_empty_events_resolves_to_issued(self) -> None:
        assert resolve_risk_authorization_status(_AUTH_ID, []) is RiskAuthorizationStatus.ISSUED

    def test_single_reservation_resolves_to_reserved(self) -> None:
        event = _event()
        assert resolve_risk_authorization_status(_AUTH_ID, [event]) is RiskAuthorizationStatus.RESERVED

    def test_reserve_then_consume_resolves_to_consumed(self) -> None:
        e0 = _event()
        e1 = _event(from_state=RiskAuthorizationStatus.RESERVED, to_state=RiskAuthorizationStatus.CONSUMED, sequence=1, detail="consumed")
        assert resolve_risk_authorization_status(_AUTH_ID, [e0, e1]) is RiskAuthorizationStatus.CONSUMED

    def test_mismatched_authorization_id_in_event_list_raises(self) -> None:
        event = _event(authorization_id="b" * 64)
        with pytest.raises(PortfolioRiskRecoveryError):
            resolve_risk_authorization_status(_AUTH_ID, [event])

    def test_discontinuous_from_state_raises(self) -> None:
        # from_state=RESERVED but current is actually ISSUED (no reservation preceded it)
        event = _event(from_state=RiskAuthorizationStatus.RESERVED, to_state=RiskAuthorizationStatus.CONSUMED)
        with pytest.raises(PortfolioRiskRecoveryError):
            resolve_risk_authorization_status(_AUTH_ID, [event])


class TestConsumptionIdentityFor:
    def test_none_when_never_reserved(self) -> None:
        assert consumption_identity_for(_AUTH_ID, []) is None

    def test_returns_the_bound_identity(self) -> None:
        event = _event(consumption_identity="use-42")
        assert consumption_identity_for(_AUTH_ID, [event]) == "use-42"

    def test_returns_the_latest_binding(self) -> None:
        e0 = _event(consumption_identity="use-1")
        e1 = _event(from_state=RiskAuthorizationStatus.RESERVED, to_state=RiskAuthorizationStatus.CONSUMED, sequence=1, consumption_identity="use-1", detail="consumed")
        assert consumption_identity_for(_AUTH_ID, [e0, e1]) == "use-1"
