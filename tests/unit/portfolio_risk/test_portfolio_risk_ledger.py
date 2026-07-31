"""Unit tests for `portfolio_risk.ledger`: entry hash self-validation,
previous-hash chaining, sequence-gap/conflict rejection, idempotent
identical append, and semantic digest stability."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import PortfolioRiskPersistenceError, RiskAuthorizationReuseError
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntry,
    RiskLedgerEntryKind,
    compute_risk_ledger_physical_digest,
    compute_risk_ledger_semantic_digest,
    create_risk_ledger_entry,
    verify_risk_ledger_chain_integrity,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _entry(sequence: int, *, previous_entry_hash: str | None, payload: dict[str, object] | None = None) -> RiskLedgerEntry:
    return create_risk_ledger_entry(
        portfolio_id="p1", entry_sequence=sequence, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload=(payload or {"x": sequence}),
        event_time=_T0, recorded_time=_T0, previous_entry_hash=previous_entry_hash,
    )


class TestEntryHashSelfValidation:
    def test_valid_entry_constructs(self) -> None:
        entry = _entry(0, previous_entry_hash=None)
        assert len(entry.entry_hash) == 64

    def test_tampered_payload_after_construction_via_from_json_dict_is_rejected(self) -> None:
        entry = _entry(0, previous_entry_hash=None)
        raw = entry.to_json_dict()
        raw["payload"] = {"x": 999}  # entry_hash now stale relative to this payload
        with pytest.raises(PortfolioRiskPersistenceError):
            RiskLedgerEntry.from_json_dict(raw)

    def test_entry_hash_depends_only_on_payload_not_sequence(self) -> None:
        a = create_risk_ledger_entry(portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload={"x": 1}, event_time=_T0, recorded_time=_T0, previous_entry_hash=None)
        b = create_risk_ledger_entry(portfolio_id="p2", entry_sequence=5, entry_kind=RiskLedgerEntryKind.RECOVERY_STARTED, payload={"x": 1}, event_time=_T0, recorded_time=_T0, previous_entry_hash="a" * 64)
        assert a.entry_hash == b.entry_hash

    def test_entry_id_depends_on_the_whole_envelope(self) -> None:
        a = _entry(0, previous_entry_hash=None, payload={"x": 1})
        b = _entry(0, previous_entry_hash=None, payload={"x": 2})
        assert a.entry_id != b.entry_id
        assert a.entry_hash != b.entry_hash


class TestPreviousHashChain:
    def test_sequence_zero_requires_no_previous_hash(self) -> None:
        with pytest.raises(PortfolioRiskPersistenceError):
            _entry(0, previous_entry_hash="a" * 64)

    def test_nonzero_sequence_requires_previous_hash(self) -> None:
        with pytest.raises(PortfolioRiskPersistenceError):
            _entry(1, previous_entry_hash=None)

    def test_chain_integrity_true_for_a_correctly_linked_sequence(self) -> None:
        e0 = _entry(0, previous_entry_hash=None)
        e1 = _entry(1, previous_entry_hash=e0.entry_id)
        e2 = _entry(2, previous_entry_hash=e1.entry_id)
        assert verify_risk_ledger_chain_integrity([e0, e1, e2])

    def test_chain_integrity_false_when_a_link_is_wrong(self) -> None:
        e0 = _entry(0, previous_entry_hash=None)
        e1 = _entry(1, previous_entry_hash="f" * 64)
        assert not verify_risk_ledger_chain_integrity([e0, e1])


class TestStoreAppendBehavior:
    def test_sequence_gap_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            # A structurally valid entry (its own __post_init__ requires
            # SOME previous_entry_hash for a nonzero sequence) that still
            # leaves a GAP relative to what the store has actually seen
            # (nothing yet) -- the store's own gap check must catch this
            # independently of entry-level self-validation.
            gap = _entry(5, previous_entry_hash="f" * 64)
            with pytest.raises(PortfolioRiskPersistenceError):
                store.append("p1", gap)

    def test_conflicting_same_sequence_append_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            store.append("p1", _entry(0, previous_entry_hash=None, payload={"x": 1}))
            with pytest.raises(RiskAuthorizationReuseError):
                store.append("p1", _entry(0, previous_entry_hash=None, payload={"x": 2}))

    def test_identical_append_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            entry = _entry(0, previous_entry_hash=None, payload={"x": 1})
            store.append("p1", entry)
            result = store.append("p1", entry)
            assert result.entry_id == entry.entry_id
            assert len(store.read_events("p1")) == 1

    def test_wrong_previous_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            store.append("p1", _entry(0, previous_entry_hash=None))
            bad = _entry(1, previous_entry_hash="f" * 64)
            with pytest.raises(PortfolioRiskPersistenceError):
                store.append("p1", bad)

    def test_entry_for_a_different_portfolio_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            entry = _entry(0, previous_entry_hash=None)
            with pytest.raises(PortfolioRiskPersistenceError):
                store.append("some-other-portfolio", entry)

    def test_next_sequence_and_last_entry_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            assert store.next_sequence("p1") == 0
            assert store.last_entry_hash("p1") is None
            e0 = _entry(0, previous_entry_hash=None)
            store.append("p1", e0)
            assert store.next_sequence("p1") == 1
            assert store.last_entry_hash("p1") == e0.entry_id

    def test_read_events_on_a_never_written_portfolio_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            assert store.read_events("never-touched") == []


class TestSemanticDigest:
    def test_deterministic_for_identical_entries(self) -> None:
        e0 = _entry(0, previous_entry_hash=None, payload={"x": 1})
        assert compute_risk_ledger_semantic_digest([e0]) == compute_risk_ledger_semantic_digest([e0])

    def test_excludes_recorded_time_operational_metadata(self) -> None:
        a = create_risk_ledger_entry(portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload={"x": 1}, event_time=_T0, recorded_time=_T0, previous_entry_hash=None)
        b = create_risk_ledger_entry(portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload={"x": 1}, event_time=_T0, recorded_time=datetime(2030, 6, 1, tzinfo=timezone.utc), previous_entry_hash=None)
        assert compute_risk_ledger_semantic_digest([a]) == compute_risk_ledger_semantic_digest([b])

    def test_economic_payload_mutation_changes_digest(self) -> None:
        a = _entry(0, previous_entry_hash=None, payload={"x": 1})
        b = _entry(0, previous_entry_hash=None, payload={"x": 2})
        assert compute_risk_ledger_semantic_digest([a]) != compute_risk_ledger_semantic_digest([b])

    def test_different_entry_kind_changes_digest_even_with_same_payload(self) -> None:
        a = create_risk_ledger_entry(portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload={"x": 1}, event_time=_T0, recorded_time=_T0, previous_entry_hash=None)
        b = create_risk_ledger_entry(portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RECOVERY_STARTED, payload={"x": 1}, event_time=_T0, recorded_time=_T0, previous_entry_hash=None)
        assert compute_risk_ledger_semantic_digest([a]) != compute_risk_ledger_semantic_digest([b])


class TestPhysicalDigest:
    def test_deterministic(self) -> None:
        e0 = _entry(0, previous_entry_hash=None)
        assert compute_risk_ledger_physical_digest([e0]) == compute_risk_ledger_physical_digest([e0])

    def test_differs_from_semantic_digest(self) -> None:
        e0 = _entry(0, previous_entry_hash=None)
        assert compute_risk_ledger_physical_digest([e0]) != compute_risk_ledger_semantic_digest([e0])


class TestPayloadValidation:
    def test_rejects_raw_decimal_in_payload(self) -> None:
        from decimal import Decimal

        with pytest.raises(PortfolioRiskPersistenceError):
            create_risk_ledger_entry(
                portfolio_id="p1", entry_sequence=0, entry_kind=RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED, payload={"x": Decimal("1.5")},
                event_time=_T0, recorded_time=_T0, previous_entry_hash=None,
            )
