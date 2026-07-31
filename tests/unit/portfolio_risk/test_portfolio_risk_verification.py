"""Unit tests for `portfolio_risk.verification`: `verify_portfolio_risk_
session`'s independent-reconstruction checks (physical chain, portfolio
ownership, forged identity, non-approved-decision issuance, orphan
events), and its explicit `record=True`/`record=False` behavior."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.json import canonical_json_bytes
from quant_platform.ml.models import ValidationSeverity
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
from quant_platform.portfolio_risk.state_machine import create_risk_authorization_status_event
from quant_platform.portfolio_risk.verification import verify_portfolio_risk_session

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


def _events_path(root: str, portfolio_id: str) -> Path:
    return Path(root) / "portfolio_risk_ledgers" / portfolio_id / "events.jsonl"


class TestCleanLifecycleVerifies:
    def test_fully_consumed_lifecycle_verifies_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))
            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0)
            assert report.is_ready
            assert report.criticals == ()

    def test_empty_portfolio_verifies_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            report = verify_portfolio_risk_session(portfolio_id="empty", store=store, verification_time=_T0)
            assert report.is_ready


class TestRecordFlag:
    def test_record_true_appends_a_verification_completed_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            before = len(store.read_events("p1"))
            verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=True)
            after = len(store.read_events("p1"))
            assert after == before + 1
            assert store.read_events("p1")[-1].entry_kind is RiskLedgerEntryKind.VERIFICATION_COMPLETED

    def test_record_false_appends_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            before = len(store.read_events("p1"))
            verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            after = len(store.read_events("p1"))
            assert after == before

    def test_record_false_is_idempotent_across_repeated_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            report1 = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            report2 = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert report1.to_json_dict() == report2.to_json_dict()


class TestLedgerChainBroken:
    def test_a_removed_middle_entry_breaks_physical_chain_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization))
            consume_authorization(store, authorization, **_use_kwargs(authorization))

            path = _events_path(tmp, "p1")
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(lines) == 3
            path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")

            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            assert any(i.code == "ledger_chain_broken" and i.severity is ValidationSeverity.CRITICAL for i in report.issues)


class TestCrossPortfolioOwnership:
    def test_an_entry_declaring_a_foreign_portfolio_id_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)

            # `store.append` itself refuses a portfolio_id mismatch -- simulate
            # direct file tampering (a raw line written straight into p1's own
            # ledger file) to bypass that write-time guard entirely, which is
            # exactly the threat model independent verification defends against.
            foreign_entry = create_risk_ledger_entry(
                portfolio_id="a-foreign-portfolio", entry_sequence=1, entry_kind=RiskLedgerEntryKind.RECOVERY_STARTED,
                payload={"portfolio_id": "a-foreign-portfolio"}, event_time=_T0, recorded_time=_T0,
                previous_entry_hash=store.last_entry_hash("p1"),
            )
            path = _events_path(tmp, "p1")
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(foreign_entry.to_json_dict()))
                handle.write(b"\n")

            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            assert any(i.code == "ledger_portfolio_ownership_mismatch" for i in report.issues)


class TestAuthorizationFromDeniedDecision:
    def test_a_hand_crafted_authorization_payload_claiming_a_denied_decision_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
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
            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            assert any(i.code == "authorization_from_non_approved_decision" for i in report.issues)


class TestForgedAuthorizationIdentity:
    def test_a_payload_whose_own_id_does_not_reproduce_its_own_fields_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            genuine = _authorization()
            # A single, standalone forged issuance -- keeps the ORIGINAL (valid-looking)
            # risk_authorization_id but changes evaluated_quantity underneath it, so
            # the id no longer reproduces from the (now-different) bound fields. This
            # is the sole RISK_AUTHORIZATION_ISSUED entry for this id, so no earlier
            # idempotency-index conflict masks the forged-identity check itself.
            forged_payload = dict(genuine.to_json_dict())
            forged_payload["evaluated_quantity"] = "9999"
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED,
                payload=forged_payload, event_time=_T0, recorded_time=_T0, previous_entry_hash=None,
            )
            store.append("p1", entry)
            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            assert any(i.code == "forged_authorization_identity" for i in report.issues)


class TestOrphanAuthorizationLifecycleEvent:
    def test_status_event_for_a_never_issued_authorization_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            orphan_id = "f" * 64
            orphan_event = create_risk_authorization_status_event(
                authorization_id=orphan_id, portfolio_id="p1",
                from_state=RiskAuthorizationStatus.ISSUED,
                to_state=RiskAuthorizationStatus.RESERVED,
                event_time=_T0, sequence=0, consumption_identity="use-1", reason_code=None, detail="orphan",
            )
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED, payload=orphan_event.to_json_dict(),
                event_time=_T0, recorded_time=_T0, previous_entry_hash=None,
            )
            store.append("p1", entry)
            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            assert any(i.code == "orphan_authorization_lifecycle_event" for i in report.issues)


class TestCoherentReChainingCaughtByReplayNotChainIntegrity:
    def test_a_forged_conflicting_consumption_identity_fails_verification_via_reconstruction_not_chain_check(self) -> None:
        # A CONSUMED event whose consumption_identity does not match the
        # bound RESERVED identity is a coherently-chained forgery -- the
        # physical chain (sequence/previous_hash) is perfectly intact.
        # This must still be caught, but via the idempotency-index
        # reconstruction failing (state_machine.resolve_risk_authorization_
        # status), never via `ledger_chain_broken`.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_authorization(store, authorization, **_use_kwargs(authorization, consumption_identity="use-1"))
            forged_event = create_risk_authorization_status_event(
                authorization_id=authorization.risk_authorization_id, portfolio_id="p1", from_state=RiskAuthorizationStatus.RESERVED,
                to_state=RiskAuthorizationStatus.CONSUMED, event_time=_T0, sequence=1, consumption_identity="use-2", reason_code=None,
                detail="forged conflicting consumption",
            )
            entry = create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=store.next_sequence("p1"), entry_kind=RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED,
                payload=forged_event.to_json_dict(), event_time=_T0, recorded_time=_T0, previous_entry_hash=store.last_entry_hash("p1"),
            )
            store.append("p1", entry)
            report = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not report.is_ready
            codes = [i.code for i in report.issues]
            assert "ledger_chain_broken" not in codes
            assert "idempotency_index_reconstruction_failed" in codes


class TestVerificationCannotBeShortCircuitedByAnEarlierCleanResult:
    def test_verifying_twice_after_a_tamper_reflects_the_new_state_not_the_old_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            clean = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert clean.is_ready

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
            dirty = verify_portfolio_risk_session(portfolio_id="p1", store=store, verification_time=_T0, record=False)
            assert not dirty.is_ready
