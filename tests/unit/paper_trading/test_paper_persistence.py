"""Milestone 7, Section 21: the event-sourced ledger. Covers `LedgerEntry`
validation/identity, `verify_ledger_chain_integrity`'s hash-chain check,
and `PaperSessionEventStore`'s append/read/duplicate-detection/chain-
violation-rejection against a real temp directory."""

from __future__ import annotations

import dataclasses
import threading
from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import DuplicateEventError, PaperTradingArtifactError
from quant_platform.paper_trading.models import LedgerEntryKind
from quant_platform.paper_trading.persistence import (
    LedgerEntry,
    PaperSessionEventStore,
    create_ledger_entry,
    verify_ledger_chain_integrity,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_SESSION_ID = "a" * 64


def _entry(*, sequence: int, previous_entry_hash: str | None, payload: dict[str, object] | None = None) -> LedgerEntry:
    return create_ledger_entry(
        session_id=_HEX_SESSION_ID, sequence=sequence, kind=LedgerEntryKind.SESSION_TRANSITION, payload=(payload or {"stage": sequence}),
        event_time=_T0, previous_entry_hash=previous_entry_hash,
    )


class TestLedgerEntryValidation:
    def test_valid_first_entry(self) -> None:
        entry = _entry(sequence=0, previous_entry_hash=None)
        assert entry.sequence == 0

    def test_first_entry_with_previous_hash_rejected(self) -> None:
        with pytest.raises(PaperTradingArtifactError, match="previous_entry_hash"):
            LedgerEntry(
                entry_id="0" * 64, session_id=_HEX_SESSION_ID, sequence=0, kind=LedgerEntryKind.SESSION_TRANSITION, payload={},
                event_time=_T0, previous_entry_hash="b" * 64, checksum="c" * 64,
            )

    def test_non_first_entry_without_previous_hash_rejected(self) -> None:
        with pytest.raises(PaperTradingArtifactError, match="previous_entry_hash"):
            LedgerEntry(
                entry_id="0" * 64, session_id=_HEX_SESSION_ID, sequence=1, kind=LedgerEntryKind.SESSION_TRANSITION, payload={},
                event_time=_T0, previous_entry_hash=None, checksum="c" * 64,
            )

    def test_tampered_checksum_rejected(self) -> None:
        entry = _entry(sequence=0, previous_entry_hash=None)
        with pytest.raises(PaperTradingArtifactError, match="checksum"):
            dataclasses.replace(entry, payload={"stage": "tampered"})

    def test_json_round_trip(self) -> None:
        entry = _entry(sequence=0, previous_entry_hash=None)
        assert LedgerEntry.from_json_dict(entry.to_json_dict()) == entry


class TestLedgerEntryIdentity:
    def test_identical_arguments_produce_identical_entry_id(self) -> None:
        assert _entry(sequence=0, previous_entry_hash=None).entry_id == _entry(sequence=0, previous_entry_hash=None).entry_id

    def test_different_payload_changes_entry_id(self) -> None:
        a = _entry(sequence=0, previous_entry_hash=None, payload={"stage": "created"})
        b = _entry(sequence=0, previous_entry_hash=None, payload={"stage": "running"})
        assert a.entry_id != b.entry_id


class TestVerifyLedgerChainIntegrity:
    def test_valid_chain(self) -> None:
        first = _entry(sequence=0, previous_entry_hash=None)
        second = _entry(sequence=1, previous_entry_hash=first.entry_id)
        third = _entry(sequence=2, previous_entry_hash=second.entry_id)
        verify_ledger_chain_integrity([first, second, third])

    def test_empty_chain_is_valid(self) -> None:
        verify_ledger_chain_integrity([])

    def test_out_of_order_sequence_rejected(self) -> None:
        first = _entry(sequence=0, previous_entry_hash=None)
        second = _entry(sequence=2, previous_entry_hash=first.entry_id)
        with pytest.raises(PaperTradingArtifactError, match="sequence"):
            verify_ledger_chain_integrity([first, second])

    def test_substituted_entry_breaks_chain(self) -> None:
        """A different entry with the SAME sequence but a different
        payload (hence different entry_id) inserted in place of the real
        one breaks the hash-chain link to the next entry -- exactly the
        tamper-detection property the chain exists for."""
        first = _entry(sequence=0, previous_entry_hash=None)
        second = _entry(sequence=1, previous_entry_hash=first.entry_id)
        substituted_first = _entry(sequence=0, previous_entry_hash=None, payload={"stage": "tampered"})
        with pytest.raises(PaperTradingArtifactError, match="previous_entry_hash"):
            verify_ledger_chain_integrity([substituted_first, second])


class TestPaperSessionEventStore:
    def test_append_then_read(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        entry = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, entry)
        events = store.read_events(_HEX_SESSION_ID)
        assert events == [entry]

    def test_read_events_empty_when_no_file(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        assert store.read_events(_HEX_SESSION_ID) == []

    def test_append_multiple_entries_in_order(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        first = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, first)
        second = _entry(sequence=1, previous_entry_hash=first.entry_id)
        store.append(_HEX_SESSION_ID, second)
        events = store.read_events(_HEX_SESSION_ID)
        assert [e.sequence for e in events] == [0, 1]
        verify_ledger_chain_integrity(events)

    def test_appending_identical_entry_twice_is_idempotent(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        entry = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, entry)
        store.append(_HEX_SESSION_ID, entry)
        events = store.read_events(_HEX_SESSION_ID)
        assert len(events) == 1

    def test_appending_different_entry_at_occupied_sequence_rejected(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        first = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, first)
        conflicting = _entry(sequence=0, previous_entry_hash=None, payload={"stage": "different"})
        with pytest.raises(DuplicateEventError):
            store.append(_HEX_SESSION_ID, conflicting)

    def test_appending_with_wrong_previous_hash_rejected(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        first = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, first)
        wrong_link = _entry(sequence=1, previous_entry_hash="f" * 64)
        with pytest.raises(PaperTradingArtifactError, match="previous_entry_hash"):
            store.append(_HEX_SESSION_ID, wrong_link)

    def test_appending_ahead_of_ledger_length_rejected(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        skipping_ahead = _entry(sequence=5, previous_entry_hash="f" * 64)
        with pytest.raises(PaperTradingArtifactError, match="skips ahead"):
            store.append(_HEX_SESSION_ID, skipping_ahead)

    def test_next_sequence_and_last_entry_hash(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        assert store.next_sequence(_HEX_SESSION_ID) == 0
        assert store.last_entry_hash(_HEX_SESSION_ID) is None
        first = _entry(sequence=0, previous_entry_hash=None)
        store.append(_HEX_SESSION_ID, first)
        assert store.next_sequence(_HEX_SESSION_ID) == 1
        assert store.last_entry_hash(_HEX_SESSION_ID) == first.entry_id

    def test_different_sessions_are_independent(self, tmp_path) -> None:
        store = PaperSessionEventStore(tmp_path)
        other_session = "b" * 64
        store.append(_HEX_SESSION_ID, _entry(sequence=0, previous_entry_hash=None))
        assert store.read_events(other_session) == []


class TestConcurrentAppends:
    def test_concurrent_appends_produce_a_valid_unbroken_chain(self, tmp_path) -> None:
        """Multiple threads racing to append the NEXT entry: the shared
        lock serializes them so the resulting ledger is always a single,
        valid, unbroken chain -- never two entries claiming the same
        sequence with different content, never a gap."""
        store = PaperSessionEventStore(tmp_path)
        attempts = 12
        errors: list[Exception] = []
        lock = threading.Lock()

        def _append_next() -> None:
            try:
                with lock:
                    sequence = store.next_sequence(_HEX_SESSION_ID)
                    previous_hash = store.last_entry_hash(_HEX_SESSION_ID)
                    entry = _entry(sequence=sequence, previous_entry_hash=previous_hash, payload={"i": sequence})
                    store.append(_HEX_SESSION_ID, entry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_append_next) for _ in range(attempts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        events = store.read_events(_HEX_SESSION_ID)
        assert len(events) == attempts
        verify_ledger_chain_integrity(events)
