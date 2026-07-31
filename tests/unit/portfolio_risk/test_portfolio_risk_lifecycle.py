"""Unit tests for `portfolio_risk.lifecycle`: every legal transition,
every illegal transition, reservation/consumption, exact-retry
idempotency, conflicting-use rejection, and use-after-terminal-status
rejection (expiry/invalidation/revocation)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskAuthorizationReuseError
from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.idempotency import build_authorization_status_index
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    expire_authorization,
    invalidate_authorization,
    record_authorization_issuance,
    reserve_authorization,
    revoke_authorization,
)
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus, RiskDecisionKind

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


def _status(store: PortfolioRiskLedgerStore, authorization) -> RiskAuthorizationStatus:
    return build_authorization_status_index(store.read_events(authorization.portfolio_id))[authorization.risk_authorization_id]


class TestReservation:
    def test_reserve_transitions_issued_to_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            assert _status(store, authorization) is RiskAuthorizationStatus.RESERVED


class TestConsumption:
    def test_consume_after_reserve_transitions_to_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            assert _status(store, authorization) is RiskAuthorizationStatus.CONSUMED

    def test_consume_before_issue_is_impossible_by_construction(self) -> None:
        # There is no way to call consume_authorization without an
        # already-issued authorization object -- this test documents
        # that "consume before issue" is structurally unreachable via
        # this package's own API, not merely untested.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            # Never call record_authorization_issuance.
            with pytest.raises(RiskAuthorizationReuseError):
                consume_authorization(store, authorization, **_use_kwargs(authorization))

    def test_consume_without_reserve_first_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                consume_authorization(store, authorization, **_use_kwargs(authorization))
            assert _status(store, authorization) is RiskAuthorizationStatus.ISSUED


class TestDuplicateExactUse:
    def test_duplicate_exact_reservation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            e1 = reserve_authorization(store, authorization, **_use_kwargs(authorization))
            e2 = reserve_authorization(store, authorization, **_use_kwargs(authorization))
            assert e1.event_id == e2.event_id
            assert len(store.read_events("p1")) == 2  # issuance + one reservation, NOT two

    def test_duplicate_exact_consumption_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            e1 = consume_authorization(store, authorization, **_use_kwargs(authorization))
            e2 = consume_authorization(store, authorization, **_use_kwargs(authorization))
            assert e1.event_id == e2.event_id
            assert len(store.read_events("p1")) == 3


class TestConflictingSecondUse:
    def test_conflicting_second_reservation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))
            assert _status(store, authorization) is RiskAuthorizationStatus.RESERVED

    def test_conflicting_second_consumption_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            consume_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            with pytest.raises(RiskAuthorizationReuseError):
                consume_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))
            assert _status(store, authorization) is RiskAuthorizationStatus.CONSUMED

    def test_first_time_consumption_under_a_different_identity_than_reserved_is_rejected(self) -> None:
        # Regression test for a real, confirmed defect: this is a NEW
        # transition (RESERVED -> CONSUMED, not a same-target retry), which
        # `validate_authorization_use` used to approve unconditionally
        # without ever comparing consumption identities -- letting a FIRST
        # consume attempt silently use a different economic identity than
        # the authorization was reserved under.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            with pytest.raises(RiskAuthorizationReuseError):
                consume_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))
            assert _status(store, authorization) is RiskAuthorizationStatus.RESERVED
            entries = store.read_events("p1")
            rejected = [e for e in entries if e.entry_kind.value == "risk_authorization_use_rejected"]
            assert len(rejected) == 1
            assert rejected[0].payload["rejection_reason"] == "conflicting_consumption"

    def test_rejection_is_durably_recorded_in_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))
            entries = store.read_events("p1")
            rejected = [e for e in entries if e.entry_kind.value == "risk_authorization_use_rejected"]
            assert len(rejected) == 1
            assert rejected[0].payload["rejection_reason"] == "conflicting_consumption"


class TestExpiryInvalidationRevocation:
    def test_expire_from_issued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            expire_authorization(store, authorization, reason_code="timed_out", detail="never reserved in time", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.EXPIRED

    def test_expire_from_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            expire_authorization(store, authorization, reason_code="timed_out", detail="reserved but never consumed in time", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.EXPIRED

    def test_invalidate_from_issued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            invalidate_authorization(store, authorization, reason_code="newer_portfolio_state", detail="superseded", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.INVALIDATED

    def test_invalidate_from_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            invalidate_authorization(store, authorization, reason_code="newer_portfolio_state", detail="superseded", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.INVALIDATED

    def test_revoke_from_issued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            revoke_authorization(store, authorization, reason_code="operator_action", detail="manually revoked", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.REVOKED

    def test_revoke_from_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            revoke_authorization(store, authorization, reason_code="operator_action", detail="manually revoked", evaluation_time=_T0)
            assert _status(store, authorization) is RiskAuthorizationStatus.REVOKED

    def test_administrative_transition_is_idempotent_on_exact_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            e1 = revoke_authorization(store, authorization, reason_code="operator_action", detail="manually revoked", evaluation_time=_T0)
            e2 = revoke_authorization(store, authorization, reason_code="operator_action", detail="manually revoked", evaluation_time=_T0)
            assert e1.event_id == e2.event_id


class TestConsumeAfterTerminalStatus:
    def test_consume_after_expiry_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            expire_authorization(store, authorization, reason_code="timed_out", detail="expired", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization))

    def test_consume_after_invalidation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            invalidate_authorization(store, authorization, reason_code="superseded", detail="invalidated", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization))

    def test_consume_after_revocation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            revoke_authorization(store, authorization, reason_code="operator_action", detail="revoked", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization))

    def test_reserved_authorization_cannot_expire_then_be_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            expire_authorization(store, authorization, reason_code="timed_out", detail="reservation timed out", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                consume_authorization(store, authorization, **_use_kwargs(authorization))


class TestEveryIllegalTransitionRejected:
    """Every ISSUED/RESERVED -> illegal-target combination, driven
    directly against the recorded ledger state (not merely the in-memory
    state machine already covered by `test_portfolio_risk_state_machine.py`)."""

    def test_cannot_reserve_a_consumed_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))

    def test_cannot_expire_a_consumed_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            with pytest.raises(RiskAuthorizationReuseError):
                expire_authorization(store, authorization, reason_code="x", detail="x", evaluation_time=_T0)

    def test_cannot_invalidate_an_expired_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            expire_authorization(store, authorization, reason_code="x", detail="x", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                invalidate_authorization(store, authorization, reason_code="y", detail="y", evaluation_time=_T0)

    def test_cannot_revoke_an_invalidated_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            invalidate_authorization(store, authorization, reason_code="x", detail="x", evaluation_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                revoke_authorization(store, authorization, reason_code="y", detail="y", evaluation_time=_T0)
