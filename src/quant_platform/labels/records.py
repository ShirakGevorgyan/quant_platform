"""`LabelRecord` and `LabelRecordLedger` (Milestone 11, Phase 3, Part B):
the per-ROW identity/provenance record the governing specification's
"LABEL GENERATION" section requires (`label_id`, `label_specification_id`,
`dataset_id`, `row_identity`, `event_time`, `availability_time`,
`generation_version`, `content_hash`), materialized FROM an
already-built `builder.LabelBundle` -- never a second generation pass.

`row_identity` is the row's `event_time` rendered as an ISO-8601 UTC
string -- content-based and portable (never a positional DataFrame
index, which says nothing about WHICH row it is once rows are
re-ordered, filtered, or reloaded from a different source). `label_id`
identifies WHICH label slot this is (`label_specification_id` +
`row_identity`) independent of its value; `content_hash` additionally
covers the value itself, so a value change (without an identity change)
is still detectable.

`LabelRecordLedger` is the APPEND-ONLY commit tracker "Recovery" needs:
`commit` refuses to silently overwrite an already-committed
`row_identity` for a given specification (`LabelRecordConflictError`);
`recover` returns only the NOT-yet-committed subset of a candidate
batch, so resuming an interrupted generation run never regenerates (or
double-commits) work that already landed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

from quant_platform.core.exceptions import LabelRecordConflictError, LabelRequestError
from quant_platform.core.types import Timeframe
from quant_platform.labels.builder import LabelBundle
from quant_platform.ml.persistence import require_schema_version

__all__ = [
    "LABEL_RECORD_SCHEMA_VERSION",
    "LabelRecord",
    "LabelRecordLedger",
    "compute_content_hash",
    "compute_label_id",
    "materialize_label_records",
]

LABEL_RECORD_SCHEMA_VERSION = 1


def compute_label_id(label_specification_id: str, row_identity: str) -> str:
    return hashlib.sha256(f"{label_specification_id}|{row_identity}".encode()).hexdigest()


def compute_content_hash(label_id: str, value: float | None) -> str:
    payload = json.dumps({"label_id": label_id, "value": value}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelRecord:
    schema_version: int
    label_id: str
    label_specification_id: str
    dataset_id: str
    row_identity: str
    event_time: str
    availability_time: str
    generation_version: str
    content_hash: str
    value: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_id": self.label_id, "label_specification_id": self.label_specification_id,
            "dataset_id": self.dataset_id, "row_identity": self.row_identity, "event_time": self.event_time,
            "availability_time": self.availability_time, "generation_version": self.generation_version,
            "content_hash": self.content_hash, "value": self.value,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelRecord:
        require_schema_version(raw, supported=LABEL_RECORD_SCHEMA_VERSION, context="LabelRecord")
        raw_value = raw.get("value")
        return cls(
            schema_version=LABEL_RECORD_SCHEMA_VERSION, label_id=str(raw["label_id"]), label_specification_id=str(raw["label_specification_id"]),
            dataset_id=str(raw["dataset_id"]), row_identity=str(raw["row_identity"]), event_time=str(raw["event_time"]),
            availability_time=str(raw["availability_time"]), generation_version=str(raw["generation_version"]),
            content_hash=str(raw["content_hash"]), value=(None if raw_value is None else float(str(raw_value))),
        )

    def verify_self_consistency(self) -> tuple[bool, tuple[str, ...]]:
        issues: list[str] = []
        if self.row_identity != self.event_time:
            issues.append(f"row_identity: claims {self.row_identity!r}, does not match event_time {self.event_time!r} it must equal by construction")
        recomputed_label_id = compute_label_id(self.label_specification_id, self.row_identity)
        if recomputed_label_id != self.label_id:
            issues.append(f"label_id: claims {self.label_id!r}, recomputed is {recomputed_label_id!r}")
        recomputed_content_hash = compute_content_hash(self.label_id, self.value)
        if recomputed_content_hash != self.content_hash:
            issues.append(f"content_hash: claims {self.content_hash!r}, recomputed is {recomputed_content_hash!r}")
        return (not issues, tuple(issues))


def materialize_label_records(
    bundle: LabelBundle, source_data: pd.DataFrame, *, dataset_id: str, timeframe: Timeframe, horizon_bars: int,
) -> tuple[LabelRecord, ...]:
    """`event_time = open_time + timeframe.duration` (the bar's CLOSE
    time -- when this row's own data first became knowable, mirroring
    `features.interfaces.FeatureContext`'s identical convention).
    `availability_time = event_time + timeframe.duration * horizon_bars`
    -- the WORST-CASE instant this label is guaranteed knowable; a label
    that resolves earlier (e.g. an early triple-barrier touch) is still
    conservatively marked available only this late, never earlier than
    it could possibly be known and never dependent on the wall clock."""
    if dataset_id != bundle.specification.created_from_dataset:
        raise LabelRequestError(
            f"dataset_id={dataset_id!r} does not match bundle.specification.created_from_dataset={bundle.specification.created_from_dataset!r} "
            "-- a label's records must always be materialized against the dataset its specification was built for",
            context={"dataset_id": dataset_id, "created_from_dataset": bundle.specification.created_from_dataset},
        )
    if "open_time" not in source_data.columns:
        raise LabelRequestError("source_data is missing required column 'open_time'", context={"dataset_id": dataset_id})
    event_times = source_data["open_time"] + timeframe.duration
    availability_times = event_times + timeframe.duration * horizon_bars

    records = []
    for i in range(bundle.row_count):
        event_time = event_times.iloc[i]
        row_identity = event_time.isoformat()
        raw_value = bundle.values.iloc[i]
        value = None if pd.isna(raw_value) else float(raw_value)
        label_id = compute_label_id(bundle.specification.label_specification_id, row_identity)
        content_hash = compute_content_hash(label_id, value)
        records.append(LabelRecord(
            schema_version=LABEL_RECORD_SCHEMA_VERSION, label_id=label_id, label_specification_id=bundle.specification.label_specification_id,
            dataset_id=dataset_id, row_identity=row_identity, event_time=event_time.isoformat(), availability_time=availability_times.iloc[i].isoformat(),
            generation_version=bundle.specification.generation_version, content_hash=content_hash, value=value,
        ))
    return tuple(records)


@dataclass(slots=True)
class LabelRecordLedger:
    """In-memory, append-only commit tracker -- matches this package's
    established "no persistence store yet" scope (see `docs/
    labels_architecture.md`'s Known Limitations); the append-only
    CONTRACT itself, not durable storage, is what "Recovery" requires."""

    _committed: dict[tuple[str, str], LabelRecord] = field(default_factory=dict)

    def is_committed(self, label_specification_id: str, row_identity: str) -> bool:
        return (label_specification_id, row_identity) in self._committed

    def commit(self, records: tuple[LabelRecord, ...]) -> None:
        for record in records:
            key = (record.label_specification_id, record.row_identity)
            if key in self._committed:
                raise LabelRecordConflictError(
                    f"row_identity={record.row_identity!r} is already committed for label_specification_id={record.label_specification_id!r} "
                    "-- the ledger is append-only and never overwrites",
                    context={"label_specification_id": record.label_specification_id, "row_identity": record.row_identity},
                )
        for record in records:
            self._committed[(record.label_specification_id, record.row_identity)] = record

    def recover(self, candidates: tuple[LabelRecord, ...]) -> tuple[LabelRecord, ...]:
        """Returns only the subset of `candidates` NOT already
        committed -- "never regenerate partially committed labels":
        resuming after an interruption re-derives candidates for the
        WHOLE requested range, but only the genuinely missing ones are
        returned for a caller to act on."""
        return tuple(r for r in candidates if not self.is_committed(r.label_specification_id, r.row_identity))

    def committed_count(self, label_specification_id: str) -> int:
        return sum(1 for spec_id, _ in self._committed if spec_id == label_specification_id)
