"""Append-only, hash-chained execution ledger (Milestone 8, Section 18).
Mirrors `paper_trading.persistence.PaperSessionEventStore`'s identical
architecture exactly: every domain fact this package produces is wrapped
in one `ExecutionLedgerEntry` and appended here -- never mutated in
place, never overwritten. `verification.py` reconstructs every piece of
execution-session state by replaying this ledger from scratch.

CHAIN INTEGRITY: each `ExecutionLedgerEntry.previous_entry_hash` is the
prior entry's own `entry_id` (`None` only for sequence 0) --
`verify_execution_ledger_chain_integrity` walks this like a hash chain,
catching a deleted/reordered/substituted entry.

DUPLICATE DETECTION: `append` treats appending an entry whose `entry_id`
already exists at the same sequence as an IDEMPOTENT no-op -- this is
what makes crash recovery safe: replaying an already-applied step
produces the identical entry and is silently absorbed, never duplicated.
An entry with the same `entry_id` at a DIFFERENT sequence, or a
different `entry_id` reused at an already-occupied sequence, is a
genuine conflict, never silently absorbed (Section 18: "same entry ID
with different payload is rejected")."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import ExecutionGatewayArtifactError, ExecutionIdempotencyError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.execution_gateway.identity import compute_content_id
from quant_platform.execution_gateway.manifests import execution_session_lock
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind
from quant_platform.ml.persistence import as_json_dict, format_utc_timestamp, parse_utc_timestamp, utc_now

EXECUTION_LEDGER_ENTRY_KIND = "execution_ledger_entry"
_EVENTS_FILE_NAME = "events.jsonl"
_EVENTS_LOCK_FILE_NAME = ".events.lock"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionGatewayArtifactError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise ExecutionGatewayArtifactError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ExecutionGatewayArtifactError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise ExecutionGatewayArtifactError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExecutionLedgerEntry:
    entry_id: str
    execution_session_id: str
    entry_sequence: int
    entry_kind: ExecutionLedgerEntryKind
    payload: dict[str, object]
    event_time: datetime
    recorded_time: datetime
    previous_entry_hash: str | None
    entry_hash: str

    def __post_init__(self) -> None:
        if not self.execution_session_id:
            raise ExecutionGatewayArtifactError("ExecutionLedgerEntry.execution_session_id must not be empty")
        if self.entry_sequence < 0:
            raise ExecutionGatewayArtifactError(f"ExecutionLedgerEntry.entry_sequence must be >= 0, got {self.entry_sequence}")
        if self.entry_sequence == 0 and self.previous_entry_hash is not None:
            raise ExecutionGatewayArtifactError("ExecutionLedgerEntry.previous_entry_hash must be None for entry_sequence 0")
        if self.entry_sequence > 0 and self.previous_entry_hash is None:
            raise ExecutionGatewayArtifactError(f"ExecutionLedgerEntry.previous_entry_hash is required for entry_sequence {self.entry_sequence}")
        _require_tz_aware(self.event_time, field_name="ExecutionLedgerEntry.event_time")
        _require_tz_aware(self.recorded_time, field_name="ExecutionLedgerEntry.recorded_time")
        expected_hash = compute_content_id("execution_ledger_entry_payload", self.payload)
        if self.entry_hash != expected_hash:
            raise ExecutionGatewayArtifactError(f"ExecutionLedgerEntry.entry_hash does not match its own payload -- expected {expected_hash}, got {self.entry_hash}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id, "execution_session_id": self.execution_session_id, "entry_sequence": self.entry_sequence,
            "entry_kind": self.entry_kind.value, "payload": self.payload, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
            "recorded_time": _serialize_timestamp(self.recorded_time, field_name="recorded_time"), "previous_entry_hash": self.previous_entry_hash,
            "entry_hash": self.entry_hash,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["entry_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionLedgerEntry:
        return cls(
            entry_id=str(raw["entry_id"]), execution_session_id=str(raw["execution_session_id"]), entry_sequence=int(str(raw["entry_sequence"])),
            entry_kind=ExecutionLedgerEntryKind(raw["entry_kind"]), payload=as_json_dict(raw["payload"], field_name="payload"),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            recorded_time=_deserialize_timestamp(raw["recorded_time"], field_name="recorded_time"),
            previous_entry_hash=(None if raw.get("previous_entry_hash") is None else str(raw["previous_entry_hash"])), entry_hash=str(raw["entry_hash"]),
        )


def create_execution_ledger_entry(
    *, execution_session_id: str, entry_sequence: int, entry_kind: ExecutionLedgerEntryKind, payload: dict[str, object], event_time: datetime,
    previous_entry_hash: str | None, recorded_time: datetime | None = None,
) -> ExecutionLedgerEntry:
    entry_hash = compute_content_id("execution_ledger_entry_payload", payload)
    resolved_recorded_time = recorded_time if recorded_time is not None else utc_now().to_pydatetime()
    provisional = ExecutionLedgerEntry(
        entry_id="0" * 64, execution_session_id=execution_session_id, entry_sequence=entry_sequence, entry_kind=entry_kind, payload=payload,
        event_time=event_time, recorded_time=resolved_recorded_time, previous_entry_hash=previous_entry_hash, entry_hash=entry_hash,
    )
    entry_id = compute_content_id(EXECUTION_LEDGER_ENTRY_KIND, provisional.to_identity_payload())
    return ExecutionLedgerEntry(
        entry_id=entry_id, execution_session_id=execution_session_id, entry_sequence=entry_sequence, entry_kind=entry_kind, payload=payload,
        event_time=event_time, recorded_time=resolved_recorded_time, previous_entry_hash=previous_entry_hash, entry_hash=entry_hash,
    )


def verify_execution_ledger_chain_integrity(entries: list[ExecutionLedgerEntry]) -> None:
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        if entry.entry_sequence != index:
            raise ExecutionGatewayArtifactError(f"Ledger chain break: expected entry_sequence {index}, found {entry.entry_sequence} (entry_id={entry.entry_id!r})")
        if entry.previous_entry_hash != previous_hash:
            raise ExecutionGatewayArtifactError(f"Ledger chain break at entry_sequence {entry.entry_sequence}: expected previous_entry_hash={previous_hash!r}, found {entry.previous_entry_hash!r}")
        previous_hash = entry.entry_id


def compute_execution_ledger_semantic_digest(entries: list[ExecutionLedgerEntry]) -> str:
    """Section 18's canonical semantic digest -- excludes operational
    fields (`entry_id`/`entry_hash`/`previous_entry_hash`/`recorded_time`/
    `event_time`) that legitimately vary across two economically-identical
    runs (wall-clock recording time, hash-chain linkage artifacts), while
    including every economic/state-affecting fact. Excluding `event_time`
    too (not just `recorded_time`) mirrors `paper_trading.persistence.
    compute_ledger_semantic_digest`'s own reasoning: a `SESSION_*`
    transition entry has no market event to anchor its timestamp to and is
    genuinely wall-clock-derived, so this digest applies the same
    exclusion uniformly to every entry kind rather than special-casing.

    CONFIRMED DEFECT, FIXED (found during this milestone's own acceptance
    testing): `BrokerEvent.adapter_id` -- a purely OPERATIONAL label for
    which adapter instance produced/normalized an event, with zero
    economic consequence -- was included in `BROKER_EVENT_*` entry
    payloads and therefore fed into this digest (it was ALSO, more
    fundamentally, part of `BrokerEvent.to_identity_payload()` itself --
    see that fix in `events.py`, which is the actual root cause: two
    economically-identical events produced different `broker_event_id`
    values whenever `adapter_id` differed, cascading into different fill/
    state-event ids throughout the ledger). Stripping `adapter_id` here
    too, defense-in-depth, mirrors how `recorded_time`/`event_time` are
    already excluded above -- this digest must depend on economics alone,
    exactly the same defect class Section 40 names for PYTHONHASHSEED/
    temp-path leaking into a digest."""
    canonical = []
    for e in entries:
        payload = dict(e.payload)
        payload.pop("adapter_id", None)
        canonical.append({"entry_sequence": e.entry_sequence, "entry_kind": e.entry_kind.value, "payload": payload})
    return compute_content_id("execution_ledger_semantic_digest", {"entries": canonical})


class ExecutionSessionEventStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _session_dir(self, execution_session_id: str) -> Path:
        return self._root / "execution_sessions" / execution_session_id

    def _events_path(self, execution_session_id: str) -> Path:
        return self._session_dir(execution_session_id) / _EVENTS_FILE_NAME

    def _lock_path(self, execution_session_id: str) -> Path:
        return self._session_dir(execution_session_id) / _EVENTS_LOCK_FILE_NAME

    def read_events(self, execution_session_id: str) -> list[ExecutionLedgerEntry]:
        path = self._events_path(execution_session_id)
        if not path.is_file():
            return []
        entries: list[ExecutionLedgerEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = parse_json_strict(stripped)
                except ValueError as exc:
                    raise ExecutionGatewayArtifactError(f"Malformed ledger line {line_number} in {path}: {exc}", context={"path": str(path), "line": line_number}) from exc
                entries.append(ExecutionLedgerEntry.from_json_dict(as_json_dict(raw, field_name=f"events.jsonl line {line_number}")))
        return entries

    def append(self, execution_session_id: str, entry: ExecutionLedgerEntry) -> ExecutionLedgerEntry:
        if entry.execution_session_id != execution_session_id:
            raise ExecutionGatewayArtifactError(f"ExecutionLedgerEntry.execution_session_id {entry.execution_session_id!r} does not match target {execution_session_id!r}")
        with execution_session_lock(self._lock_path(execution_session_id)):
            existing_entries = self.read_events(execution_session_id)
            if entry.entry_sequence < len(existing_entries):
                existing = existing_entries[entry.entry_sequence]
                if existing.entry_id == entry.entry_id:
                    return existing
                raise ExecutionIdempotencyError(
                    f"ExecutionLedgerEntry at entry_sequence {entry.entry_sequence} already exists with a DIFFERENT entry_id "
                    f"(existing={existing.entry_id!r}, new={entry.entry_id!r}) for session {execution_session_id!r}"
                )
            if entry.entry_sequence > len(existing_entries):
                raise ExecutionGatewayArtifactError(f"ExecutionLedgerEntry.entry_sequence {entry.entry_sequence} skips ahead of the ledger's current length {len(existing_entries)} for session {execution_session_id!r}")
            expected_previous_hash = existing_entries[-1].entry_id if existing_entries else None
            if entry.previous_entry_hash != expected_previous_hash:
                raise ExecutionGatewayArtifactError(
                    f"ExecutionLedgerEntry.previous_entry_hash {entry.previous_entry_hash!r} does not match the ledger's current tail "
                    f"{expected_previous_hash!r} for session {execution_session_id!r}"
                )
            path = self._events_path(execution_session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(entry.to_json_dict())
            with path.open("ab") as handle:
                handle.write(line)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return entry

    def next_sequence(self, execution_session_id: str) -> int:
        return len(self.read_events(execution_session_id))

    def last_entry_hash(self, execution_session_id: str) -> str | None:
        entries = self.read_events(execution_session_id)
        return entries[-1].entry_id if entries else None


__all__ = [
    "EXECUTION_LEDGER_ENTRY_KIND",
    "ExecutionLedgerEntry",
    "ExecutionSessionEventStore",
    "compute_execution_ledger_semantic_digest",
    "create_execution_ledger_entry",
    "verify_execution_ledger_chain_integrity",
]
