"""Per-fold time-bound `Timeline` (Milestone 4B, Sections 12/14) -- a
human- and machine-readable summary of WHEN each fold's train/validation/
test windows fall, distinct from `ml.tracking.ExperimentEventStore`'s
append-only log of WHAT SYSTEM EVENTS occurred and in what order. Both
are reused/kept, deliberately not merged: the event log answers
"what happened, and when, operationally"; the timeline answers
"what data, chronologically, did fold K actually train/validate/test on".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quant_platform.execution.results import FoldResult
from quant_platform.execution.splitters import FoldPlan
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    validation_start: str | None = None
    validation_end: str | None = None
    status: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start, "train_end": self.train_end,
            "validation_start": self.validation_start, "validation_end": self.validation_end,
            "test_start": self.test_start, "test_end": self.test_end,
            "status": self.status,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TimelineEntry:
        return cls(
            fold_index=int(str(raw["fold_index"])),
            train_start=str(raw["train_start"]), train_end=str(raw["train_end"]),
            validation_start=(None if raw.get("validation_start") is None else str(raw["validation_start"])),
            validation_end=(None if raw.get("validation_end") is None else str(raw["validation_end"])),
            test_start=str(raw["test_start"]), test_end=str(raw["test_end"]),
            status=(None if raw.get("status") is None else str(raw["status"])),
        )


@dataclass(frozen=True, slots=True)
class Timeline:
    schema_version: int
    experiment_id: str
    entries: tuple[TimelineEntry, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "entries": [e.to_json_dict() for e in self.entries],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Timeline:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="Timeline")
        return cls(
            schema_version=_SCHEMA_VERSION,
            experiment_id=str(raw["experiment_id"]),
            entries=tuple(
                TimelineEntry.from_json_dict(as_json_dict(e, field_name="entries[]"))
                for e in as_json_list(raw.get("entries") or [], field_name="entries")
            ),
        )

    @classmethod
    def from_fold_plan(cls, experiment_id: str, plan: FoldPlan) -> Timeline:
        entries = tuple(
            TimelineEntry(
                fold_index=f.fold_index,
                train_start=f.train_start.isoformat(), train_end=f.train_end.isoformat(),
                validation_start=(None if f.validation_start is None else f.validation_start.isoformat()),
                validation_end=(None if f.validation_end is None else f.validation_end.isoformat()),
                test_start=f.test_start.isoformat(), test_end=f.test_end.isoformat(),
                status=None,
            )
            for f in plan.folds
        )
        return cls(schema_version=_SCHEMA_VERSION, experiment_id=experiment_id, entries=entries)

    @classmethod
    def from_fold_results(cls, experiment_id: str, results: Sequence[FoldResult]) -> Timeline:
        entries = tuple(
            TimelineEntry(
                fold_index=r.fold_index,
                train_start=r.train_start, train_end=r.train_end,
                validation_start=r.validation_start, validation_end=r.validation_end,
                test_start=r.test_start, test_end=r.test_end,
                status=r.status.value,
            )
            for r in sorted(results, key=lambda r: r.fold_index)
        )
        return cls(schema_version=_SCHEMA_VERSION, experiment_id=experiment_id, entries=entries)


def render_timeline_markdown(timeline: Timeline) -> str:
    lines = [f"# Timeline: {timeline.experiment_id}", ""]
    lines.append("| Fold | Train | Validation | Test | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for e in timeline.entries:
        validation_cell = "-" if e.validation_start is None else f"{e.validation_start} .. {e.validation_end}"
        status_cell = e.status or "-"
        lines.append(f"| {e.fold_index} | {e.train_start} .. {e.train_end} | {validation_cell} | {e.test_start} .. {e.test_end} | {status_cell} |")
    return "\n".join(lines) + "\n"


__all__ = ["Timeline", "TimelineEntry", "render_timeline_markdown"]
