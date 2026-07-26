"""Lightweight, local, append-only experiment event tracking -- no
dashboard, no server, just a durable history of what happened to one
experiment and when. Distinct from `ml/manifests.py`'s
`ExperimentManifest` (the CURRENT state) the way a database's
write-ahead log is distinct from its current row: `ExperimentManifest`
answers "what is this experiment's status right now"; `ExperimentEventStore`
answers "what happened, in what order, and when" -- operational history
that is NEVER part of scientific identity (see `experiment_identity.py`'s
module docstring; nothing here is ever hashed into an `experiment_id`).

STORAGE FORMAT: JSON LINES, NOT ONE JSON DOCUMENT
--------------------------------------------------------------------------
`<storage_root>/experiments/<experiment_id>/events.jsonl` -- one
canonical-JSON `EventRecord` per line. This is deliberate, not
incidental: a single JSON array/object would need to be rewritten in
full on every append (an O(n) operation, and a bigger window for a
crash to corrupt EVERYTHING rather than just the newest entry). JSON
Lines makes each entry independently parseable, so a crash mid-write can
only ever damage the trailing, in-flight line -- every prior line
remains intact and readable regardless.

CRASH SAFETY: REPAIR-ON-ACCESS, NOT TRUE ATOMIC APPEND
--------------------------------------------------------------------------
Unlike this package's content-addressed writes (`artifacts.py`,
`persistence.write_json_atomic`), an append to a growing file is not a
single atomic rename -- there is no portable way to atomically append
an arbitrary-length line to an existing file on both POSIX and Windows.
Instead, every access (`append` and `read_events`) first calls
`_load_repaired`, which tolerates and silently repairs (truncates away)
an unparseable FINAL line -- the signature of a process that crashed
mid-write -- while treating an unparseable line ANYWHERE ELSE as
unrecoverable corruption (`ArtifactCorruptionError`): prior, fully-
written events are never in question; only the newest, in-flight one
ever is. This is a documented, deliberately weaker guarantee than this
package's content-addressed stores, appropriate for an operational
event log rather than scientific-identity-bearing content.

ORDERING AND CONCURRENCY
--------------------------------------------------------------------------
Every event carries an explicit, gapless, strictly increasing
`sequence` number starting at 1 -- never relative wall-clock ordering
alone (clock resolution/skew is not a substitute for an explicit
counter). Appends are serialized per-experiment via a dedicated
`.events.lock` (through `ml.concurrency.experiment_lock`, a thin adapter
over `historical.locking.DatasetLock` -- never a second hand-rolled lock
implementation) -- concurrent appenders to the SAME experiment_id are
safely serialized; different experiment_ids never contend.
`experiment_lock` translates a contested or racing lock ACQUISITION into
`ExperimentLockError`, never a raw `DatasetLockError`/`PermissionError`
across the ML public boundary -- see `ml/concurrency.py`.

NO SECRETS
--------------------------------------------------------------------------
`details` is restricted to JSON-primitive values via `models.
validate_json_primitive_mapping` -- the same restriction `ml/models.py`
already applies to `ModelHyperparameters`/`LabelBinding.params`/etc. --
so an event can carry small structured context (e.g. an artifact's
content hash and category) but never an arbitrary object, environment
variable dump, or credential.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import ArtifactCorruptionError
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import JsonPrimitive, validate_json_primitive_mapping
from quant_platform.ml.persistence import (
    assert_within_root,
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    parse_utc_timestamp,
    require_schema_version,
    utc_now,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_EVENTS_FILE_NAME = "events.jsonl"
_LOCK_FILE_NAME = ".events.lock"


class EventType(Enum):
    EXPERIMENT_CREATED = "experiment_created"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    RUN_STARTED = "run_started"
    ARTIFACT_WRITTEN = "artifact_written"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    EXPERIMENT_CANCELLED = "experiment_cancelled"


@dataclass(frozen=True, slots=True)
class EventRecord:
    schema_version: int
    sequence: int
    experiment_id: str
    event_type: EventType
    occurred_at: str
    details: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"EventRecord.sequence must be >= 1, got {self.sequence}")
        if not is_valid_sha256_hex(self.experiment_id):
            raise ValueError(f"EventRecord.experiment_id must be a valid sha256 hex id, got {self.experiment_id!r}")
        parse_utc_timestamp(self.occurred_at)
        validate_json_primitive_mapping(self.details, field_name="EventRecord.details")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "experiment_id": self.experiment_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "details": dict(sorted(self.details.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EventRecord:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="EventRecord")
        details_raw = raw.get("details") or {}
        if not isinstance(details_raw, dict):
            raise ValueError("EventRecord.details must be a JSON object")
        return cls(
            schema_version=_SCHEMA_VERSION,
            sequence=int(str(raw["sequence"])),
            experiment_id=str(raw["experiment_id"]),
            event_type=EventType(raw["event_type"]),
            occurred_at=str(raw["occurred_at"]),
            details=dict(details_raw),
        )


class ExperimentEventStore:
    """Append-only per-experiment event log. See module docstring for the
    JSON-Lines format, repair-on-access crash safety, and locking design."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not is_valid_sha256_hex(experiment_id):
            raise ValueError(f"Invalid experiment_id {experiment_id!r}: must be a 64-character lowercase hex SHA-256 digest")
        path = self._root / "experiments" / experiment_id
        self._assert_within_root(path)
        return path

    def _events_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _EVENTS_FILE_NAME

    def _lock_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _LOCK_FILE_NAME

    def _assert_within_root(self, path: Path) -> None:
        assert_within_root(path, root=self._root)

    def _load_repaired(self, events_path: Path) -> list[EventRecord]:
        """Parse every event line, tolerating (and repairing away) an
        unparseable FINAL line as an interrupted write. An unparseable
        line anywhere else is unrecoverable corruption."""
        if not events_path.is_file():
            return []
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return []

        records: list[EventRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                raw = parse_json_strict(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                record = EventRecord.from_json_dict(raw)
            except (ValueError, KeyError) as exc:
                if not is_last:
                    raise ArtifactCorruptionError(
                        f"Event log {events_path} is corrupted: unreadable event at line {index + 1} "
                        f"(not the final line, so this cannot be an interrupted write): {exc}",
                        context={"path": str(events_path), "line": index + 1},
                    ) from exc
                logger.warning(
                    "Event log %s: final line is unreadable (interrupted write) -- repairing by truncating it: %s",
                    events_path, exc,
                )
                self._rewrite(events_path, records)
                break
            if record.sequence != len(records) + 1:
                raise ArtifactCorruptionError(
                    f"Event log {events_path} has a non-sequential sequence number at line {index + 1}: "
                    f"expected {len(records) + 1}, got {record.sequence}",
                    context={"path": str(events_path), "line": index + 1, "expected": len(records) + 1, "actual": record.sequence},
                )
            records.append(record)
        return records

    def _rewrite(self, events_path: Path, records: list[EventRecord]) -> None:
        tmp_path = events_path.parent / f".{events_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            text = "".join(canonical_json_bytes(r.to_json_dict()).decode("utf-8") + "\n" for r in records)
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(events_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def append(
        self, experiment_id: str, event_type: EventType, *, details: Mapping[str, JsonPrimitive] | None = None
    ) -> EventRecord:
        events_path = self._events_path(experiment_id)
        with experiment_lock(self._lock_path(experiment_id)):
            existing = self._load_repaired(events_path)
            record = EventRecord(
                schema_version=_SCHEMA_VERSION,
                sequence=len(existing) + 1,
                experiment_id=experiment_id,
                event_type=event_type,
                occurred_at=format_utc_timestamp(utc_now()),
                details=dict(details or {}),
            )
            events_path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(record.to_json_dict()).decode("utf-8") + "\n"
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        logger.info(
            "Experiment event appended: experiment_id=%s sequence=%d event_type=%s",
            experiment_id[:12], record.sequence, event_type.value,
        )
        return record

    def read_events(self, experiment_id: str) -> tuple[EventRecord, ...]:
        with experiment_lock(self._lock_path(experiment_id)):
            return tuple(self._load_repaired(self._events_path(experiment_id)))


__all__ = ["EventRecord", "EventType", "ExperimentEventStore"]
