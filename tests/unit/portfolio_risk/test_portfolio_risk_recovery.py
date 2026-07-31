"""Unit tests for `portfolio_risk.recovery`: classification of every
required lifecycle-depth scenario, and recovery's own determinism."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal

from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    record_authorization_issuance,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus, RiskDecisionKind
from quant_platform.portfolio_risk.recovery import recover_portfolio_risk_session

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


class TestIssuedOnly:
    def test_classified_as_issued_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            actions = recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            assert len(actions) == 1
            assert actions[0].action == "issued_only"
            assert actions[0].status is RiskAuthorizationStatus.ISSUED


class TestReservedWithoutConsumed:
    def test_classified_as_unresolved_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            actions = recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            assert len(actions) == 1
            assert actions[0].action == "reserved_unresolved_blocked"
            assert "use-1" in actions[0].detail


class TestExactSafeRetryClassification:
    def test_exact_retry_of_a_reserved_authorization_remains_idempotently_usable_after_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            # Recovery itself never mutates lifecycle state -- an exact
            # retry of the SAME consumption_identity remains safely
            # idempotent afterward.
            event = reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            assert event.to_state is RiskAuthorizationStatus.RESERVED


class TestConflictingRetryClassification:
    def test_a_new_consumption_identity_after_recovery_still_conflicts(self) -> None:
        import pytest

        from quant_platform.core.exceptions import RiskAuthorizationReuseError

        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            with pytest.raises(RiskAuthorizationReuseError):
                reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-2"))


class TestUnresolvedAmbiguityRemainsBlocked:
    def test_recovery_never_authorizes_a_blind_new_reservation(self) -> None:
        # Recovery classifies but never itself calls reserve/consume --
        # confirmed by checking that status is UNCHANGED by recovery.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            entries_before = len(store.read_events("p1"))
            recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            entries_after = len(store.read_events("p1"))
            # Only RECOVERY_STARTED + RECOVERY_COMPLETED were added -- no
            # lifecycle-transition entry.
            assert entries_after == entries_before + 2


class TestTerminalStatusesClassified:
    def test_consumed_classified_as_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            actions = recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            assert actions[0].action == "terminal_consumed"


class TestRecoveryIsDeterministic:
    def test_running_recovery_twice_yields_the_same_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            actions1 = recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            actions2 = recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            assert [a.to_json_dict() for a in actions1] == [a.to_json_dict() for a in actions2]

    def test_recovery_records_started_and_completed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            recover_portfolio_risk_session(portfolio_id="p1", store=store, recovery_time=_T0)
            kinds = [e.entry_kind.value for e in store.read_events("p1")]
            assert "recovery_started" in kinds
            assert "recovery_completed" in kinds

    def test_empty_portfolio_recovery_produces_no_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            actions = recover_portfolio_risk_session(portfolio_id="empty-portfolio", store=store, recovery_time=_T0)
            assert actions == []
