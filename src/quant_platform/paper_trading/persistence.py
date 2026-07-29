"""Event-sourced persistence: the append-only ledger that is the SOLE
source of truth for a paper session (Milestone 7, Section 21). Every
domain object this package produces (a `MarketEvent`, a `StrategyDecision`,
an `OrderStateEvent`, a `Fill`, a mark, a financing application, a
`RiskCheckResult` batch, a `KillSwitchTransitionEvent`, a `PortfolioState`
snapshot, a reconciliation result, a session-stage transition, a
`ShadowObservation`) is wrapped in exactly one `LedgerEntry` and appended
here -- never mutated in place, never overwritten. `verify_paper_session`
(built later, Section 26) reconstructs every piece of session state by
replaying this ledger from scratch; a persisted `PortfolioState` snapshot
inside the ledger is a CACHE for fast resume, never a trusted shortcut
verification is allowed to use instead of the replay.

CHAIN INTEGRITY: each `LedgerEntry.previous_entry_hash` is the prior
entry's own `entry_id` (`None` only for the session's very first entry) --
`verify_ledger_chain_integrity` walks this exactly like a hash chain,
catching a deleted/reordered/substituted entry that a bare sequence-number
check alone would miss (sequence numbers could be forged consistently;
the hash chain cannot, without also finding a second preimage for
sha256).

DUPLICATE DETECTION: `PaperSessionEventStore.append` treats appending an
entry whose `entry_id` already exists in the ledger as an IDEMPOTENT no-op
(returns the existing entry, appends nothing) rather than an error -- this
is what makes `runner.py`'s resume-after-interruption path safe: replaying
an already-applied step produces the identical entry and is silently
absorbed, never duplicated."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import DuplicateEventError, PaperTradingArtifactError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.ml.persistence import as_json_dict, format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.manifests import session_lock
from quant_platform.paper_trading.models import LedgerEntryKind

LEDGER_ENTRY_KIND = "ledger_entry"
_EVENTS_FILE_NAME = "events.jsonl"
_EVENTS_LOCK_FILE_NAME = ".events.lock"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PaperTradingArtifactError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PaperTradingArtifactError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PaperTradingArtifactError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PaperTradingArtifactError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    entry_id: str
    session_id: str
    sequence: int
    kind: LedgerEntryKind
    payload: dict[str, object]
    event_time: datetime
    previous_entry_hash: str | None
    checksum: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise PaperTradingArtifactError("LedgerEntry.session_id must not be empty")
        if self.sequence < 0:
            raise PaperTradingArtifactError(f"LedgerEntry.sequence must be >= 0, got {self.sequence}")
        if self.sequence == 0 and self.previous_entry_hash is not None:
            raise PaperTradingArtifactError("LedgerEntry.previous_entry_hash must be None for sequence 0 (the session's first entry)")
        if self.sequence > 0 and self.previous_entry_hash is None:
            raise PaperTradingArtifactError(f"LedgerEntry.previous_entry_hash is required for sequence {self.sequence} (non-first entry)")
        _require_tz_aware(self.event_time, field_name="LedgerEntry.event_time")
        expected_checksum = compute_content_id("ledger_entry_payload", self.payload)
        if self.checksum != expected_checksum:
            raise PaperTradingArtifactError(f"LedgerEntry.checksum does not match its own payload -- expected {expected_checksum}, got {self.checksum}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id, "session_id": self.session_id, "sequence": self.sequence, "kind": self.kind.value, "payload": self.payload,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "previous_entry_hash": self.previous_entry_hash,
            "checksum": self.checksum,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["entry_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LedgerEntry:
        return cls(
            entry_id=str(raw["entry_id"]), session_id=str(raw["session_id"]), sequence=int(str(raw["sequence"])), kind=LedgerEntryKind(raw["kind"]),
            payload=as_json_dict(raw["payload"], field_name="payload"), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            previous_entry_hash=(None if raw.get("previous_entry_hash") is None else str(raw["previous_entry_hash"])), checksum=str(raw["checksum"]),
        )


def create_ledger_entry(
    *, session_id: str, sequence: int, kind: LedgerEntryKind, payload: dict[str, object], event_time: datetime, previous_entry_hash: str | None,
) -> LedgerEntry:
    checksum = compute_content_id("ledger_entry_payload", payload)
    provisional = LedgerEntry(
        entry_id="0" * 64, session_id=session_id, sequence=sequence, kind=kind, payload=payload, event_time=event_time,
        previous_entry_hash=previous_entry_hash, checksum=checksum,
    )
    entry_id = compute_content_id(LEDGER_ENTRY_KIND, provisional.to_identity_payload())
    return LedgerEntry(
        entry_id=entry_id, session_id=session_id, sequence=sequence, kind=kind, payload=payload, event_time=event_time,
        previous_entry_hash=previous_entry_hash, checksum=checksum,
    )


def verify_ledger_chain_integrity(entries: list[LedgerEntry]) -> None:
    """Walks `entries` (assumed already sorted by `sequence`) confirming
    strictly increasing sequence numbers starting at 0 and an unbroken
    `previous_entry_hash` chain. Raises `PaperTradingArtifactError` on the
    first break -- a deleted, reordered, or substituted entry."""
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        if entry.sequence != index:
            raise PaperTradingArtifactError(f"Ledger chain break: expected sequence {index}, found {entry.sequence} (entry_id={entry.entry_id!r})")
        if entry.previous_entry_hash != previous_hash:
            raise PaperTradingArtifactError(f"Ledger chain break at sequence {entry.sequence}: expected previous_entry_hash={previous_hash!r}, found {entry.previous_entry_hash!r}")
        previous_hash = entry.entry_id


def compute_ledger_semantic_digest(entries: list[LedgerEntry]) -> str:
    """Release-audit addition (Section 10): a single, reusable, content-
    addressed digest of what a ledger actually MEANS, deliberately
    excluding fields that are legitimately variable OPERATIONAL metadata
    rather than economic/decision content -- `entry_id`/`checksum`/
    `previous_entry_hash` (a pure hash-chain-linkage artifact: even one
    entry with a wall-clock-derived field cascades a DIFFERENT hash
    through every subsequent entry despite nothing meaningful changing)
    and each entry's own `event_time` (stable/reproducible for any entry
    anchored to a market event's own deterministic timestamp, but
    genuinely wall-clock-derived for a `SESSION_TRANSITION` entry, which
    has no market event to anchor to -- excluded uniformly here rather
    than special-cased per kind, for a single, simple, honestly-labeled
    definition).

    Two runs of the SAME deterministic spec/event stream must produce
    the SAME digest; the exact same discipline the runner's own resume
    logic requires (`events` must be the SAME deterministic source every
    call) applies here to what a caller diffs against. Never claims
    byte-for-byte artifact identity -- only that every economically/
    decision-meaningful field replays identically."""
    canonical = [{"sequence": e.sequence, "kind": e.kind.value, "payload": e.payload} for e in entries]
    return compute_content_id("ledger_semantic_digest", {"entries": canonical})


class PaperSessionEventStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _session_dir(self, paper_session_id: str) -> Path:
        return self._root / "paper_sessions" / paper_session_id

    def _events_path(self, paper_session_id: str) -> Path:
        return self._session_dir(paper_session_id) / _EVENTS_FILE_NAME

    def _lock_path(self, paper_session_id: str) -> Path:
        return self._session_dir(paper_session_id) / _EVENTS_LOCK_FILE_NAME

    def read_events(self, paper_session_id: str) -> list[LedgerEntry]:
        path = self._events_path(paper_session_id)
        if not path.is_file():
            return []
        entries: list[LedgerEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = parse_json_strict(stripped)
                except ValueError as exc:
                    raise PaperTradingArtifactError(f"Malformed ledger line {line_number} in {path}: {exc}", context={"path": str(path), "line": line_number}) from exc
                entries.append(LedgerEntry.from_json_dict(as_json_dict(raw, field_name=f"events.jsonl line {line_number}")))
        return entries

    def append(self, paper_session_id: str, entry: LedgerEntry) -> LedgerEntry:
        """Idempotent: if an entry with the SAME `entry_id` already exists
        at the SAME `sequence` position, this is a silent no-op (returns
        the existing entry) -- the exact property `runner.py`'s resume
        path relies on. An entry with the same `entry_id` at a DIFFERENT
        sequence, or a different entry_id reused at an already-occupied
        sequence, is a genuine `DuplicateEventError`/chain violation,
        never silently absorbed."""
        if entry.session_id != paper_session_id:
            raise PaperTradingArtifactError(f"LedgerEntry.session_id {entry.session_id!r} does not match target paper_session_id {paper_session_id!r}")
        with session_lock(self._lock_path(paper_session_id)):
            existing_entries = self.read_events(paper_session_id)
            if entry.sequence < len(existing_entries):
                existing = existing_entries[entry.sequence]
                if existing.entry_id == entry.entry_id:
                    return existing
                raise DuplicateEventError(
                    f"LedgerEntry at sequence {entry.sequence} already exists with a DIFFERENT entry_id "
                    f"(existing={existing.entry_id!r}, new={entry.entry_id!r}) for session {paper_session_id!r}"
                )
            if entry.sequence > len(existing_entries):
                raise PaperTradingArtifactError(f"LedgerEntry.sequence {entry.sequence} skips ahead of the ledger's current length {len(existing_entries)} for session {paper_session_id!r}")
            expected_previous_hash = existing_entries[-1].entry_id if existing_entries else None
            if entry.previous_entry_hash != expected_previous_hash:
                raise PaperTradingArtifactError(
                    f"LedgerEntry.previous_entry_hash {entry.previous_entry_hash!r} does not match the ledger's current tail "
                    f"{expected_previous_hash!r} for session {paper_session_id!r}"
                )
            path = self._events_path(paper_session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(entry.to_json_dict())
            with path.open("ab") as handle:
                handle.write(line)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return entry

    def next_sequence(self, paper_session_id: str) -> int:
        return len(self.read_events(paper_session_id))

    def last_entry_hash(self, paper_session_id: str) -> str | None:
        entries = self.read_events(paper_session_id)
        return entries[-1].entry_id if entries else None


__all__ = ["LEDGER_ENTRY_KIND", "LedgerEntry", "PaperSessionEventStore", "compute_ledger_semantic_digest", "create_ledger_entry", "verify_ledger_chain_integrity"]
