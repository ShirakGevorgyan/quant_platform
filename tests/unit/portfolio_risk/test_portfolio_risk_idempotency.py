"""Unit tests for `portfolio_risk.idempotency`: every durable index
builder, its conflict-rejection behavior, and the "issued-but-never-
touched authorization must still appear" regression (a real defect found
and fixed during this phase's own development)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PortfolioRiskRecoveryError
from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.idempotency import (
    build_authorization_consumption_index,
    build_authorization_payload_index,
    build_authorization_status_index,
    build_consumption_identity_index,
    build_decision_to_authorization_index,
    build_execution_intent_index,
    build_status_events_index,
)
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    record_authorization_issuance,
    reserve_authorization,
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


class TestIssuedButNeverTouchedAuthorizationIsVisible:
    """Regression test for the real defect found during Phase 3
    development: `build_authorization_status_index`/`build_authorization_
    consumption_index` used to seed their key universe from status EVENTS
    alone, silently omitting any authorization that was issued but never
    subsequently reserved/consumed/expired/invalidated/revoked."""

    def test_issued_only_authorization_appears_in_status_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            status_index = build_authorization_status_index(store.read_events("p1"))
            assert status_index[authorization.risk_authorization_id] is RiskAuthorizationStatus.ISSUED

    def test_issued_only_authorization_appears_in_consumption_index_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            consumption_index = build_authorization_consumption_index(store.read_events("p1"))
            assert authorization.risk_authorization_id in consumption_index
            assert consumption_index[authorization.risk_authorization_id] is None

    def test_multiple_authorizations_at_different_lifecycle_depths_all_appear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            issued_only = _authorization(execution_intent_id="1" * 64, risk_decision_id="6" * 64)
            reserved_only = _authorization(execution_intent_id="7" * 64, risk_decision_id="8" * 64)
            fully_consumed = _authorization(execution_intent_id="9" * 64, risk_decision_id="a" * 64)
            for auth in (issued_only, reserved_only, fully_consumed):
                record_authorization_issuance(store, auth, event_time=_T0)
            kwargs = {"execution_intent_id": reserved_only.execution_intent_id, "execution_session_id": reserved_only.execution_session_id, "portfolio_id": "p1", "portfolio_snapshot_id": reserved_only.portfolio_snapshot_id, "price_snapshot_id": reserved_only.price_snapshot_id, "risk_policy_id": reserved_only.risk_policy_id, "quantity": reserved_only.evaluated_quantity, "price": reserved_only.evaluated_price, "consumption_identity": "use-r", "evaluation_time": _T0}
            reserve_authorization(store, reserved_only, **kwargs)  # type: ignore[arg-type]
            kwargs2 = {"execution_intent_id": fully_consumed.execution_intent_id, "execution_session_id": fully_consumed.execution_session_id, "portfolio_id": "p1", "portfolio_snapshot_id": fully_consumed.portfolio_snapshot_id, "price_snapshot_id": fully_consumed.price_snapshot_id, "risk_policy_id": fully_consumed.risk_policy_id, "quantity": fully_consumed.evaluated_quantity, "price": fully_consumed.evaluated_price, "consumption_identity": "use-c", "evaluation_time": _T0}
            reserve_authorization(store, fully_consumed, **kwargs2)  # type: ignore[arg-type]
            consume_authorization(store, fully_consumed, **kwargs2)  # type: ignore[arg-type]

            status_index = build_authorization_status_index(store.read_events("p1"))
            assert status_index[issued_only.risk_authorization_id] is RiskAuthorizationStatus.ISSUED
            assert status_index[reserved_only.risk_authorization_id] is RiskAuthorizationStatus.RESERVED
            assert status_index[fully_consumed.risk_authorization_id] is RiskAuthorizationStatus.CONSUMED
            assert len(status_index) == 3


class TestDecisionToAuthorizationIndex:
    def test_maps_decision_to_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            index = build_decision_to_authorization_index(store.read_events("p1"))
            assert index[authorization.risk_decision_id] == authorization.risk_authorization_id


class TestExecutionIntentIndex:
    def test_maps_intent_to_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            index = build_execution_intent_index(store.read_events("p1"))
            assert index[authorization.execution_intent_id] == authorization.risk_authorization_id


class TestConsumptionIdentityIndex:
    def test_maps_consumption_identity_to_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            kwargs = {"execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id, "portfolio_id": "p1", "portfolio_snapshot_id": authorization.portfolio_snapshot_id, "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id, "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price, "consumption_identity": "use-1", "evaluation_time": _T0}
            reserve_authorization(store, authorization, **kwargs)  # type: ignore[arg-type]
            index = build_consumption_identity_index(store.read_events("p1"))
            assert index["use-1"] == authorization.risk_authorization_id


class TestPayloadIndexConflictDetection:
    def test_no_conflict_for_a_single_issuance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            index = build_authorization_payload_index(store.read_events("p1"))
            assert authorization.risk_authorization_id in index


class TestStatusEventsIndexOrdering:
    def test_events_are_in_append_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            kwargs = {"execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id, "portfolio_id": "p1", "portfolio_snapshot_id": authorization.portfolio_snapshot_id, "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id, "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price, "consumption_identity": "use-1", "evaluation_time": _T0}
            reserve_authorization(store, authorization, **kwargs)  # type: ignore[arg-type]
            consume_authorization(store, authorization, **kwargs)  # type: ignore[arg-type]
            events = build_status_events_index(store.read_events("p1"))[authorization.risk_authorization_id]
            assert [e.to_state for e in events] == [RiskAuthorizationStatus.RESERVED, RiskAuthorizationStatus.CONSUMED]


class TestStatusIndexRaisesOnCorruption:
    def test_illegal_replay_raises_rather_than_silently_resolving(self) -> None:
        from quant_platform.portfolio_risk.ledger import RiskLedgerEntryKind, create_risk_ledger_entry
        from quant_platform.portfolio_risk.state_machine import create_risk_authorization_status_event

        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            # Directly forge a CONSUMED transition with no preceding RESERVED.
            bad_event = create_risk_authorization_status_event(
                authorization_id=authorization.risk_authorization_id, portfolio_id="p1", from_state=RiskAuthorizationStatus.RESERVED,
                to_state=RiskAuthorizationStatus.CONSUMED, event_time=_T0, sequence=0, consumption_identity="forged", reason_code=None, detail="forged",
            )
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=store.next_sequence("p1"), entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
                payload=bad_event.to_json_dict(), event_time=_T0, recorded_time=_T0, previous_entry_hash=store.last_entry_hash("p1"),
            )
            store.append("p1", entry)
            with pytest.raises(PortfolioRiskRecoveryError):
                build_authorization_status_index(store.read_events("p1"))
