"""Idempotency evidence, reconstructed from the durable ledger (Milestone
8, Section 16). Every index here is a PURE function of the append-only
`ExecutionLedgerEntry` sequence -- Section 16's own instruction ("Create
durable idempotency evidence. Do not rely only on in-memory sets.") is
satisfied by the ledger itself being the durable evidence; this module's
job is to correctly RECONSTRUCT dedup/collision-detection views from it,
exactly like `paper_trading.reconciliation` reconstructs position state
purely from the ledger rather than trusting cached bookkeeping.

Building an index is itself a genuine independent check: Section 9's
cross-order invariants ("one client_order_id maps to only one economic
submit", "one broker_order_id cannot belong to two unrelated
client_order_ids", "broker_order_id cannot silently change") are enforced
HERE by raising `ExecutionIdempotencyError` the moment a scan finds a
violation, not merely assumed to hold because `persistence.py`'s
append-time check already prevented it -- this is the SAME "never trust
that it can't have happened, independently re-check" discipline the
Milestone 7 release audit required throughout."""

from __future__ import annotations

from quant_platform.core.exceptions import ExecutionIdempotencyError
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind
from quant_platform.execution_gateway.persistence import ExecutionLedgerEntry


def build_command_index(entries: list[ExecutionLedgerEntry]) -> dict[str, ExecutionLedgerEntry]:
    """`command_id` -> its `COMMAND_CREATED` ledger entry. Raises if the
    SAME `command_id` appears twice with a DIFFERENT payload (Section 16
    invariant 4: "same ID with different payload is rejected") -- this
    should already be structurally impossible given `persistence.py`'s
    own append-time conflict check, but this is the independent,
    ledger-only re-verification of that fact."""
    index: dict[str, ExecutionLedgerEntry] = {}
    for entry in entries:
        if entry.entry_kind is not ExecutionLedgerEntryKind.COMMAND_CREATED:
            continue
        command_id = str(entry.payload.get("command_id"))
        existing = index.get(command_id)
        if existing is not None and existing.payload != entry.payload:
            raise ExecutionIdempotencyError(f"command_id {command_id!r} appears in the ledger twice with different payloads -- durable idempotency evidence is corrupted")
        index[command_id] = entry
    return index


def build_client_order_index(entries: list[ExecutionLedgerEntry]) -> dict[str, str]:
    """`client_order_id` -> the ONE `execution_intent_id` it was derived
    from. Raises if a `client_order_id` is ever seen mapped to two
    different intents (Section 9: "one client order ID cannot map to two
    unrelated economic submits")."""
    index: dict[str, str] = {}
    for entry in entries:
        if entry.entry_kind is not ExecutionLedgerEntryKind.COMMAND_CREATED or entry.payload.get("command_type") != "submit_order":
            continue
        client_order_id = str(entry.payload["client_order_id"])
        intent_id = str(entry.payload["execution_intent_id"])
        existing = index.get(client_order_id)
        if existing is not None and existing != intent_id:
            raise ExecutionIdempotencyError(f"client_order_id {client_order_id!r} maps to two different execution_intent_id values: {existing!r} and {intent_id!r}")
        index[client_order_id] = intent_id
    return index


def build_broker_order_index(entries: list[ExecutionLedgerEntry]) -> dict[str, str]:
    """`broker_order_id` -> the ONE `client_order_id` it belongs to.
    Raises if a `broker_order_id` is ever associated with two different
    `client_order_id` values (Section 9: "one broker order ID cannot
    belong to two unrelated client order IDs" / "broker order ID cannot
    silently change")."""
    index: dict[str, str] = {}
    for entry in entries:
        if entry.entry_kind is not ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED:
            continue
        broker_order_id = entry.payload.get("broker_order_id")
        client_order_id = entry.payload.get("client_order_id")
        if broker_order_id is None or client_order_id is None:
            continue
        existing = index.get(str(broker_order_id))
        if existing is not None and existing != str(client_order_id):
            raise ExecutionIdempotencyError(f"broker_order_id {broker_order_id!r} is associated with two different client_order_id values: {existing!r} and {client_order_id!r}")
        index[str(broker_order_id)] = str(client_order_id)
    return index


def build_broker_event_index(entries: list[ExecutionLedgerEntry]) -> dict[str, ExecutionLedgerEntry]:
    """`broker_event_id` -> its `BROKER_EVENT_RECEIVED` ledger entry --
    the durable evidence `dispatcher.py` consults to recognize a
    redelivered broker event as an idempotent duplicate (Section 16
    invariant 5) rather than re-processing it."""
    index: dict[str, ExecutionLedgerEntry] = {}
    for entry in entries:
        if entry.entry_kind is not ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED:
            continue
        broker_event_id = str(entry.payload.get("broker_event_id"))
        existing = index.get(broker_event_id)
        if existing is not None and existing.payload != entry.payload:
            raise ExecutionIdempotencyError(f"broker_event_id {broker_event_id!r} appears in the ledger twice with different payloads")
        index[broker_event_id] = entry
    return index


def is_command_already_recorded(entries: list[ExecutionLedgerEntry], *, command_id: str, payload: dict[str, object]) -> bool:
    """`True` if `command_id` is already durably recorded with an
    IDENTICAL payload (safe to treat this dispatch attempt as already
    done). Raises `ExecutionIdempotencyError` if `command_id` is already
    recorded with a DIFFERENT payload (Section 16 invariant 4) -- never
    silently proceeds with a conflicting economic operation."""
    index = build_command_index(entries)
    existing = index.get(command_id)
    if existing is None:
        return False
    if existing.payload != payload:
        raise ExecutionIdempotencyError(f"command_id {command_id!r} is already recorded with a different payload -- refusing to treat this as the same operation")
    return True


__all__ = [
    "build_broker_event_index",
    "build_broker_order_index",
    "build_client_order_index",
    "build_command_index",
    "is_command_already_recorded",
]
