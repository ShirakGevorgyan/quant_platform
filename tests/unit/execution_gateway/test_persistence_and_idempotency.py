"""Unit tests for `execution_gateway.persistence` (Section 18) and
`execution_gateway.idempotency` (Section 16): append-only ledger
idempotency/conflict detection, hash-chain integrity, semantic digest
determinism, and durable idempotency-evidence reconstruction."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_platform.core.exceptions import ExecutionGatewayArtifactError, ExecutionIdempotencyError
from quant_platform.execution_gateway.idempotency import (
    build_broker_order_index,
    build_client_order_index,
    build_command_index,
    is_command_already_recorded,
)
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind
from quant_platform.execution_gateway.persistence import (
    ExecutionSessionEventStore,
    compute_execution_ledger_semantic_digest,
    create_execution_ledger_entry,
    verify_execution_ledger_chain_integrity,
)

_SHA_SESSION = "a" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(session_id: str, sequence: int, *, previous_hash: str | None, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_CREATED, payload=None):
    return create_execution_ledger_entry(
        execution_session_id=session_id, entry_sequence=sequence, entry_kind=kind, payload=(payload or {"n": sequence}), event_time=_NOW,
        previous_entry_hash=previous_hash, recorded_time=_NOW,
    )


class TestExecutionLedgerEntryValidation:
    def test_sequence_zero_requires_no_previous_hash(self) -> None:
        with pytest.raises(ExecutionGatewayArtifactError):
            create_execution_ledger_entry(execution_session_id=_SHA_SESSION, entry_sequence=0, entry_kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_CREATED, payload={}, event_time=_NOW, previous_entry_hash="x" * 64, recorded_time=_NOW)

    def test_nonzero_sequence_requires_previous_hash(self) -> None:
        with pytest.raises(ExecutionGatewayArtifactError):
            create_execution_ledger_entry(execution_session_id=_SHA_SESSION, entry_sequence=1, entry_kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_CREATED, payload={}, event_time=_NOW, previous_entry_hash=None, recorded_time=_NOW)

    def test_tampered_payload_after_construction_is_detected_on_reconstruction(self) -> None:
        entry = _entry(_SHA_SESSION, 0, previous_hash=None)
        tampered_dict = entry.to_json_dict()
        tampered_dict["payload"] = {"n": 999}
        with pytest.raises(ExecutionGatewayArtifactError):
            from quant_platform.execution_gateway.persistence import ExecutionLedgerEntry

            ExecutionLedgerEntry.from_json_dict(tampered_dict)


class TestExecutionSessionEventStoreAppend:
    def _store(self, tmp_path: Path) -> ExecutionSessionEventStore:
        return ExecutionSessionEventStore(tmp_path)

    def test_append_then_read_round_trips(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        store.append(_SHA_SESSION, e0)
        e1 = _entry(_SHA_SESSION, 1, previous_hash=e0.entry_id)
        store.append(_SHA_SESSION, e1)
        read_back = store.read_events(_SHA_SESSION)
        assert [e.entry_id for e in read_back] == [e0.entry_id, e1.entry_id]

    def test_reappending_identical_entry_is_idempotent(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        store.append(_SHA_SESSION, e0)
        result = store.append(_SHA_SESSION, e0)
        assert result.entry_id == e0.entry_id
        assert len(store.read_events(_SHA_SESSION)) == 1

    def test_same_sequence_different_entry_id_is_rejected(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        store.append(_SHA_SESSION, e0)
        conflicting = _entry(_SHA_SESSION, 0, previous_hash=None, payload={"n": 12345})
        with pytest.raises(Exception):  # noqa: B017 -- DuplicateEventError-equivalent conflict
            store.append(_SHA_SESSION, conflicting)

    def test_skipping_ahead_in_sequence_is_rejected(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        e5 = _entry(_SHA_SESSION, 5, previous_hash="x" * 64)
        with pytest.raises(ExecutionGatewayArtifactError):
            store.append(_SHA_SESSION, e5)

    def test_wrong_previous_hash_is_rejected(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        store.append(_SHA_SESSION, e0)
        wrong = _entry(_SHA_SESSION, 1, previous_hash="0" * 64)
        with pytest.raises(ExecutionGatewayArtifactError):
            store.append(_SHA_SESSION, wrong)


class TestChainIntegrity:
    def test_valid_chain_passes(self) -> None:
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        e1 = _entry(_SHA_SESSION, 1, previous_hash=e0.entry_id)
        verify_execution_ledger_chain_integrity([e0, e1])

    def test_deleted_middle_entry_is_detected(self) -> None:
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None)
        e1 = _entry(_SHA_SESSION, 1, previous_hash=e0.entry_id)
        e2 = _entry(_SHA_SESSION, 2, previous_hash=e1.entry_id)
        # Simulate deletion of e1: e2 now claims sequence 1, but its
        # previous_entry_hash still points at e1 -- must be detected even
        # though "recomputing all hashes" was never attempted (that's
        # covered by the semantic-tampering test group).
        tampered = [e0, replace(e2, entry_sequence=1)]
        with pytest.raises(ExecutionGatewayArtifactError):
            verify_execution_ledger_chain_integrity(tampered)


class TestSemanticDigest:
    def test_identical_economic_content_different_recorded_time_same_digest(self) -> None:
        from datetime import timedelta

        e0a = _entry(_SHA_SESSION, 0, previous_hash=None)
        e0b = create_execution_ledger_entry(execution_session_id=_SHA_SESSION, entry_sequence=0, entry_kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_CREATED, payload={"n": 0}, event_time=_NOW, previous_entry_hash=None, recorded_time=_NOW + timedelta(hours=3))
        assert compute_execution_ledger_semantic_digest([e0a]) == compute_execution_ledger_semantic_digest([e0b])

    def test_different_payload_changes_digest(self) -> None:
        e0 = _entry(_SHA_SESSION, 0, previous_hash=None, payload={"n": 1})
        e0_other = _entry(_SHA_SESSION, 0, previous_hash=None, payload={"n": 2})
        assert compute_execution_ledger_semantic_digest([e0]) != compute_execution_ledger_semantic_digest([e0_other])

    def test_different_adapter_id_in_a_broker_event_payload_does_not_change_digest(self) -> None:
        """Regression test for a real, confirmed defect found during
        Milestone 8's own acceptance testing: `BrokerEvent.adapter_id` --
        a purely operational label with zero economic consequence -- was
        included in `BROKER_EVENT_*` entry payloads and therefore fed
        into this digest. Two genuinely economically-identical replays
        constructed with different `adapter_id` strings (an operator is
        free to name their adapter instance differently between runs)
        produced DIFFERENT digests -- the same defect class Section 40
        names for PYTHONHASHSEED/temp-path leaking into what must be a
        purely economic fingerprint."""
        e0a = _entry(_SHA_SESSION, 0, previous_hash=None, kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"adapter_id": "adapter-one", "broker_sequence": 1, "fill_price": "1.1"})
        e0b = _entry(_SHA_SESSION, 0, previous_hash=None, kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"adapter_id": "adapter-two", "broker_sequence": 1, "fill_price": "1.1"})
        assert compute_execution_ledger_semantic_digest([e0a]) == compute_execution_ledger_semantic_digest([e0b])

    def test_a_genuinely_different_broker_sequence_still_changes_the_digest(self) -> None:
        """Sanity check alongside the adapter_id exclusion above: stripping
        `adapter_id` must not accidentally strip anything ECONOMICALLY
        meaningful -- a different `broker_sequence` (or any other real
        field) must still change the digest."""
        e0a = _entry(_SHA_SESSION, 0, previous_hash=None, kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"adapter_id": "adapter-one", "broker_sequence": 1, "fill_price": "1.1"})
        e0b = _entry(_SHA_SESSION, 0, previous_hash=None, kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"adapter_id": "adapter-one", "broker_sequence": 2, "fill_price": "1.1"})
        assert compute_execution_ledger_semantic_digest([e0a]) != compute_execution_ledger_semantic_digest([e0b])


class TestIdempotencyIndexes:
    def _command_created_entry(self, sequence: int, *, command_id: str, command_type: str = "submit_order", client_order_id: str = "cid-1", execution_intent_id: str = "b" * 64, previous_hash=None):
        return create_execution_ledger_entry(
            execution_session_id=_SHA_SESSION, entry_sequence=sequence, entry_kind=ExecutionLedgerEntryKind.COMMAND_CREATED,
            payload={"command_id": command_id, "command_type": command_type, "client_order_id": client_order_id, "execution_intent_id": execution_intent_id},
            event_time=_NOW, previous_entry_hash=previous_hash, recorded_time=_NOW,
        )

    def test_build_command_index_finds_entry(self) -> None:
        e = self._command_created_entry(0, command_id="cmd-1")
        index = build_command_index([e])
        assert "cmd-1" in index

    def test_build_command_index_raises_on_conflicting_duplicate(self) -> None:
        e0 = self._command_created_entry(0, command_id="cmd-1", client_order_id="cid-1")
        e1 = self._command_created_entry(1, command_id="cmd-1", client_order_id="cid-DIFFERENT", previous_hash=e0.entry_id)
        with pytest.raises(ExecutionIdempotencyError):
            build_command_index([e0, e1])

    def test_client_order_index_raises_when_reused_for_different_intent(self) -> None:
        e0 = self._command_created_entry(0, command_id="cmd-1", client_order_id="cid-1", execution_intent_id="b" * 64)
        e1 = self._command_created_entry(1, command_id="cmd-2", client_order_id="cid-1", execution_intent_id="c" * 64, previous_hash=e0.entry_id)
        with pytest.raises(ExecutionIdempotencyError):
            build_client_order_index([e0, e1])

    def test_broker_order_index_raises_when_broker_order_id_reused_for_different_client_order(self) -> None:
        e0 = create_execution_ledger_entry(execution_session_id=_SHA_SESSION, entry_sequence=0, entry_kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"broker_order_id": "bo-1", "client_order_id": "cid-1"}, event_time=_NOW, previous_entry_hash=None, recorded_time=_NOW)
        e1 = create_execution_ledger_entry(execution_session_id=_SHA_SESSION, entry_sequence=1, entry_kind=ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, payload={"broker_order_id": "bo-1", "client_order_id": "cid-DIFFERENT"}, event_time=_NOW, previous_entry_hash=e0.entry_id, recorded_time=_NOW)
        with pytest.raises(ExecutionIdempotencyError):
            build_broker_order_index([e0, e1])

    def test_is_command_already_recorded_true_for_identical_payload(self) -> None:
        e0 = self._command_created_entry(0, command_id="cmd-1")
        assert is_command_already_recorded([e0], command_id="cmd-1", payload=e0.payload)

    def test_is_command_already_recorded_raises_for_conflicting_payload(self) -> None:
        e0 = self._command_created_entry(0, command_id="cmd-1", client_order_id="cid-1")
        with pytest.raises(ExecutionIdempotencyError):
            is_command_already_recorded([e0], command_id="cmd-1", payload={**e0.payload, "client_order_id": "cid-DIFFERENT"})

    def test_is_command_already_recorded_false_when_absent(self) -> None:
        assert not is_command_already_recorded([], command_id="cmd-1", payload={})
