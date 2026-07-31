"""Append-only, hash-chained risk ledger for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3) -- mirrors `execution_gateway.persistence`'s
identical `ExecutionLedgerEntry`/`ExecutionSessionEventStore` shape
exactly, adapted to this package's own domain and partitioned by
`portfolio_id` rather than `execution_session_id` (a single portfolio
persists across many execution sessions over its lifetime, and this
package's own domain concept is fundamentally per-portfolio risk
management -- see `docs/portfolio_risk_architecture.md`'s "Ledger
partitioning" section for the full rationale).

PHYSICAL VS SEMANTIC INTEGRITY (mirrors Milestone 8's identical split):
`verify_risk_ledger_chain_integrity` proves the ledger's own PHYSICAL
storage integrity (nothing removed, reordered, or hash-broken) --
independent of whether the ECONOMIC content inside each entry's payload
is itself coherent (that is `verification.py`'s job).
`compute_risk_ledger_semantic_digest` computes a digest of the ledger's
ECONOMIC content alone: `entry_sequence`, `entry_kind`, and `payload` --
excluding `entry_id`/`entry_hash`/`previous_entry_hash`/`recorded_time`/
`event_time` at the ENTRY level (purely operational/physical
bookkeeping; any economically meaningful timestamp already lives INSIDE
a domain object's own JSON, nested in `payload`, and participates via
that object's own already-established identity rules -- see Phase 1/2's
own "caller-supplied timestamps participate in identity" convention)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import (
    ExperimentLockError,
    PortfolioRiskLockError,
    PortfolioRiskPersistenceError,
    RiskAuthorizationReuseError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import format_utc_timestamp, parse_json_strict, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import compute_content_id

RISK_LEDGER_ENTRY_KIND = "risk_ledger_entry"
RISK_LEDGER_ENTRY_PAYLOAD_KIND = "risk_ledger_entry_payload"
RISK_LEDGER_SEMANTIC_DIGEST_KIND = "risk_ledger_semantic_digest"


class RiskLedgerEntryKind(Enum):
    RISK_EVALUATION_REQUESTED = "risk_evaluation_requested"
    RISK_DECISION_RECORDED = "risk_decision_recorded"
    RISK_AUTHORIZATION_ISSUED = "risk_authorization_issued"
    RISK_AUTHORIZATION_RESERVED = "risk_authorization_reserved"
    RISK_AUTHORIZATION_CONSUMED = "risk_authorization_consumed"
    RISK_AUTHORIZATION_EXPIRED = "risk_authorization_expired"
    RISK_AUTHORIZATION_INVALIDATED = "risk_authorization_invalidated"
    RISK_AUTHORIZATION_REVOKED = "risk_authorization_revoked"
    RISK_AUTHORIZATION_USE_REJECTED = "risk_authorization_use_rejected"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    VERIFICATION_COMPLETED = "verification_completed"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PortfolioRiskPersistenceError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    _require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PortfolioRiskPersistenceError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PortfolioRiskPersistenceError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PortfolioRiskPersistenceError(f"{field_name}: {exc}") from exc


def _canonicalize_payload_value(value: object) -> object:
    """`payload` fields are plain JSON-shaped values already (every
    caller builds them from a domain object's own `to_json_dict()`), so
    this only needs to guard against a `Decimal` slipping in unconverted
    -- `canonical_json_bytes` cannot encode one."""
    if isinstance(value, Decimal):
        raise PortfolioRiskPersistenceError(f"RiskLedgerEntry.payload must not contain a raw Decimal ({value!r}) -- serialize via decimal_to_json first")
    return value


def _validate_payload_shape(payload: dict[str, object]) -> None:
    """Shared by `RiskLedgerEntry.__post_init__` AND `create_risk_ledger_
    entry` (called there BEFORE hashing) -- hashing an unvalidated
    payload containing a raw `Decimal` would otherwise let `canonical_
    json_bytes` raise a bare `TypeError` instead of this package's own
    domain exception."""
    for key, value in payload.items():
        _canonicalize_payload_value(value)
        if not isinstance(key, str):
            raise PortfolioRiskPersistenceError(f"RiskLedgerEntry.payload keys must be strings, got {type(key).__name__}")


@dataclass(frozen=True, slots=True)
class RiskLedgerEntry:
    entry_id: str
    portfolio_id: str
    entry_sequence: int
    entry_kind: RiskLedgerEntryKind
    payload: dict[str, object]
    event_time: datetime
    recorded_time: datetime
    previous_entry_hash: str | None
    entry_hash: str
    """SELF-VALIDATING hash of `payload` ALONE (`compute_content_id(
    RISK_LEDGER_ENTRY_PAYLOAD_KIND, payload)`), checked in `__post_init__`
    -- distinct from `entry_id` (a hash of the WHOLE entry, used for
    chaining). This is what makes a tampered payload fail to even
    CONSTRUCT (via `from_json_dict`, immediately at load time) rather
    than only being caught later, opportunistically, by whichever
    domain-level check happens to exist for that specific entry kind --
    an entry kind with no such check (e.g. `RECOVERY_STARTED`) would
    otherwise let a tampered payload through completely undetected."""

    def __post_init__(self) -> None:
        if not self.portfolio_id:
            raise PortfolioRiskPersistenceError("RiskLedgerEntry.portfolio_id must not be empty")
        if self.entry_sequence < 0:
            raise PortfolioRiskPersistenceError(f"RiskLedgerEntry.entry_sequence must be >= 0, got {self.entry_sequence}")
        if self.entry_sequence == 0 and self.previous_entry_hash is not None:
            raise PortfolioRiskPersistenceError("RiskLedgerEntry.previous_entry_hash must be None when entry_sequence == 0")
        if self.entry_sequence > 0 and self.previous_entry_hash is None:
            raise PortfolioRiskPersistenceError("RiskLedgerEntry.previous_entry_hash is required when entry_sequence > 0")
        _require_tz_aware(self.event_time, field_name="RiskLedgerEntry.event_time")
        _require_tz_aware(self.recorded_time, field_name="RiskLedgerEntry.recorded_time")
        _validate_payload_shape(self.payload)
        expected_entry_hash = compute_content_id(RISK_LEDGER_ENTRY_PAYLOAD_KIND, self.payload)
        if self.entry_hash != expected_entry_hash:
            raise PortfolioRiskPersistenceError(
                f"RiskLedgerEntry.entry_hash {self.entry_hash!r} does not match the hash of its own payload ({expected_entry_hash!r}) -- "
                "tampered or corrupted entry"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id, "portfolio_id": self.portfolio_id, "entry_sequence": self.entry_sequence,
            "entry_kind": self.entry_kind.value, "payload": self.payload,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
            "recorded_time": _serialize_timestamp(self.recorded_time, field_name="recorded_time"),
            "previous_entry_hash": self.previous_entry_hash, "entry_hash": self.entry_hash,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["entry_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskLedgerEntry:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            entry_id=str(raw["entry_id"]), portfolio_id=str(raw["portfolio_id"]), entry_sequence=int(str(raw["entry_sequence"])),
            entry_kind=RiskLedgerEntryKind(raw["entry_kind"]), payload=as_json_dict(raw["payload"], field_name="payload"),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            recorded_time=_deserialize_timestamp(raw["recorded_time"], field_name="recorded_time"),
            previous_entry_hash=(None if raw.get("previous_entry_hash") is None else str(raw["previous_entry_hash"])),
            entry_hash=str(raw["entry_hash"]),
        )


def create_risk_ledger_entry(
    *, portfolio_id: str, entry_sequence: int, entry_kind: RiskLedgerEntryKind, payload: dict[str, object], event_time: datetime,
    recorded_time: datetime, previous_entry_hash: str | None,
) -> RiskLedgerEntry:
    _validate_payload_shape(payload)
    entry_hash = compute_content_id(RISK_LEDGER_ENTRY_PAYLOAD_KIND, payload)
    provisional = RiskLedgerEntry(
        entry_id="0" * 64, portfolio_id=portfolio_id, entry_sequence=entry_sequence, entry_kind=entry_kind, payload=payload,
        event_time=event_time, recorded_time=recorded_time, previous_entry_hash=previous_entry_hash, entry_hash=entry_hash,
    )
    entry_id = compute_content_id(RISK_LEDGER_ENTRY_KIND, provisional.to_identity_payload())
    return RiskLedgerEntry(
        entry_id=entry_id, portfolio_id=portfolio_id, entry_sequence=entry_sequence, entry_kind=entry_kind, payload=payload,
        event_time=event_time, recorded_time=recorded_time, previous_entry_hash=previous_entry_hash, entry_hash=entry_hash,
    )


def verify_risk_ledger_chain_integrity(entries: list[RiskLedgerEntry]) -> bool:
    """Physical integrity only -- `previous_entry_hash` is literally the
    prior entry's own `entry_id`, exactly mirroring `execution_gateway.
    persistence.verify_execution_ledger_chain_integrity`."""
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        if entry.entry_sequence != index:
            return False
        if entry.previous_entry_hash != previous_hash:
            return False
        previous_hash = entry.entry_id
    return True


def compute_risk_ledger_physical_digest(entries: list[RiskLedgerEntry]) -> str:
    """A hash of the entry-id CHAIN alone (ordering/physical integrity) --
    distinct from `compute_risk_ledger_semantic_digest`, which hashes the
    ECONOMIC content instead. Two ledgers with identical economic content
    but different incidental entry ordering (which cannot legitimately
    happen given `entry_sequence`/`previous_entry_hash` enforcement, but
    this digest exists as an independent, orthogonal integrity signal a
    report can surface alongside the semantic one)."""
    return compute_content_id("risk_ledger_physical_digest", {"entry_ids": [e.entry_id for e in entries]})


def compute_risk_ledger_semantic_digest(entries: list[RiskLedgerEntry]) -> str:
    canonical = [{"entry_sequence": e.entry_sequence, "entry_kind": e.entry_kind.value, "payload": e.payload} for e in entries]
    return compute_content_id(RISK_LEDGER_SEMANTIC_DIGEST_KIND, {"entries": canonical})


@contextmanager
def portfolio_risk_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise PortfolioRiskLockError(f"Could not acquire portfolio risk ledger lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        # `experiment_lock`'s own `finally: lock.release()` calls `historical.
        # locking.DatasetLock.release`'s bare `Path.unlink(missing_ok=True)`
        # UNPROTECTED -- on Windows, two threads racing to acquire/release
        # the SAME lock file in quick succession can hit a genuine sharing-
        # violation `PermissionError` (a WinError 32 subclass of `OSError`)
        # at release time, propagating uncaught out of `experiment_lock`
        # entirely (this is the release-side twin of the acquire-side
        # "documented Windows stale-lock-reclaim limitation" already
        # mentioned in `ml.concurrency.experiment_lock`'s own acquire-side
        # translation -- confirmed reproducible under this phase's own
        # concurrency tests). Translating it into the SAME `PortfolioRisk
        # LockError` callers already retry on is the correct, narrowly-
        # scoped fix here: `historical.locking` is shared infrastructure
        # well outside this package's own scope to modify.
        raise PortfolioRiskLockError(f"Portfolio risk ledger lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class PortfolioRiskLedgerStore:
    """Storage layout: `{storage_root}/portfolio_risk_ledgers/{portfolio_id}/
    events.jsonl` -- a sibling namespace to `execution_sessions/`/
    `paper_sessions/`, never colliding with either."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _portfolio_dir(self, portfolio_id: str) -> Path:
        return self._root / "portfolio_risk_ledgers" / portfolio_id

    def _events_path(self, portfolio_id: str) -> Path:
        return self._portfolio_dir(portfolio_id) / "events.jsonl"

    def _lock_path(self, portfolio_id: str) -> Path:
        return self._portfolio_dir(portfolio_id) / ".events.lock"

    def read_events(self, portfolio_id: str) -> list[RiskLedgerEntry]:
        path = self._events_path(portfolio_id)
        if not path.is_file():
            return []
        entries: list[RiskLedgerEntry] = []
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise PortfolioRiskPersistenceError(f"Corrupted risk ledger line for portfolio {portfolio_id!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise PortfolioRiskPersistenceError(f"Corrupted risk ledger line for portfolio {portfolio_id!r}: expected a JSON object")
            entries.append(RiskLedgerEntry.from_json_dict(raw))
        return entries

    def append(self, portfolio_id: str, entry: RiskLedgerEntry) -> RiskLedgerEntry:
        if entry.portfolio_id != portfolio_id:
            raise PortfolioRiskPersistenceError(f"RiskLedgerEntry.portfolio_id {entry.portfolio_id!r} does not match target portfolio {portfolio_id!r}")
        lock_path = self._lock_path(portfolio_id)
        self._portfolio_dir(portfolio_id).mkdir(parents=True, exist_ok=True)
        with portfolio_risk_lock(lock_path):
            existing_entries = self.read_events(portfolio_id)
            if entry.entry_sequence < len(existing_entries):
                existing = existing_entries[entry.entry_sequence]
                if existing.entry_id == entry.entry_id:
                    return existing  # idempotent no-op: identical append
                raise RiskAuthorizationReuseError(
                    f"Conflicting append at sequence {entry.entry_sequence} for portfolio {portfolio_id!r}: existing entry_id "
                    f"{existing.entry_id!r} != new entry_id {entry.entry_id!r}"
                )
            if entry.entry_sequence > len(existing_entries):
                raise PortfolioRiskPersistenceError(
                    f"RiskLedgerEntry.entry_sequence {entry.entry_sequence} leaves a gap for portfolio {portfolio_id!r} "
                    f"(expected {len(existing_entries)})"
                )
            expected_previous_hash = existing_entries[-1].entry_id if existing_entries else None
            if entry.previous_entry_hash != expected_previous_hash:
                raise PortfolioRiskPersistenceError(
                    f"RiskLedgerEntry.previous_entry_hash {entry.previous_entry_hash!r} does not match expected "
                    f"{expected_previous_hash!r} for portfolio {portfolio_id!r}"
                )
            path = self._events_path(portfolio_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(entry.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def next_sequence(self, portfolio_id: str) -> int:
        return len(self.read_events(portfolio_id))

    def last_entry_hash(self, portfolio_id: str) -> str | None:
        entries = self.read_events(portfolio_id)
        return entries[-1].entry_id if entries else None


def append_ledger_entry(
    store: PortfolioRiskLedgerStore, *, portfolio_id: str, entry_kind: RiskLedgerEntryKind, payload: dict[str, object], event_time: datetime,
) -> RiskLedgerEntry:
    """Convenience wrapper every writer in this package shares
    (`lifecycle.py`, `recovery.py`, `verification.py`) -- computes the
    next sequence and previous-hash automatically from the store's own
    current state, so no caller ever has to (or could incorrectly)
    compute them by hand."""
    sequence = store.next_sequence(portfolio_id)
    previous_hash = store.last_entry_hash(portfolio_id)
    entry = create_risk_ledger_entry(
        portfolio_id=portfolio_id, entry_sequence=sequence, entry_kind=entry_kind, payload=payload, event_time=event_time,
        recorded_time=event_time, previous_entry_hash=previous_hash,
    )
    return store.append(portfolio_id, entry)


__all__ = [
    "RISK_LEDGER_ENTRY_KIND",
    "RISK_LEDGER_ENTRY_PAYLOAD_KIND",
    "RISK_LEDGER_SEMANTIC_DIGEST_KIND",
    "PortfolioRiskLedgerStore",
    "RiskLedgerEntry",
    "RiskLedgerEntryKind",
    "append_ledger_entry",
    "compute_risk_ledger_physical_digest",
    "compute_risk_ledger_semantic_digest",
    "create_risk_ledger_entry",
    "portfolio_risk_lock",
    "verify_risk_ledger_chain_integrity",
]
