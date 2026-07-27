"""`OptimizationManifest`/`OptimizationManifestStore` and
`OptimizationEventStore` (Milestone 4D) -- mirror `execution.manifests.
ExecutionManifest`/`ExecutionManifestStore` and `ml.tracking.
ExperimentEventStore` exactly: a mutable CURRENT-state manifest (this
file) plus an append-only JSON-Lines event history (`OptimizationEventStore`),
both rooted at `<ml_artifacts_root>/optimizations/<optimization_id>/` --
a SIBLING tree of `experiments/<experiment_id>/`, never nested inside it.
One parent experiment can be the subject of MANY optimizations (different
models, different search spaces, different feature-selection strategies),
each with its own independent `optimization_id` and directory; nesting
under the experiment's own directory would force a choice about whether
deleting/archiving an experiment should cascade to its optimizations that
this milestone does not need to make.

WHY `OptimizationEventStore` IS A NEW CLASS, NOT A REUSE OF
`ml.tracking.ExperimentEventStore`
--------------------------------------------------------------------------
`ExperimentEventStore._experiment_dir` hardcodes `"experiments"` as the
storage-root subdirectory and validates its key as an experiment_id-shaped
sha256 hex string -- there is no configurable prefix to reuse for a
different root (`"optimizations"`) or a different `EventType` enum. Rather
than generalizing that class with a caller-supplied prefix (an
unnecessary abstraction nothing else would use), this module mirrors its
exact design (JSON Lines, gapless sequence numbers, repair-on-access for
a truncated final line, `.events.lock`-guarded appends via `ml.concurrency.
experiment_lock` -- the SAME corrected `DatasetLock` this whole platform
already uses) under its own name and root.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    OptimizationStateError,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ArtifactReference, JsonPrimitive, validate_json_primitive_mapping
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    assert_within_root,
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    parse_utc_timestamp,
    read_json_file,
    require_schema_version,
    utc_now,
    write_json_atomic,
)
from quant_platform.optimization.models import OptimizationStage, is_legal_optimization_transition

logger = logging.getLogger(__name__)

OPTIMIZATION_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_FILE_NAME = "optimization_manifest.json"
_MANIFEST_LOCK_FILE_NAME = ".optimization.lock"
_EVENTS_FILE_NAME = "events.jsonl"
_EVENTS_LOCK_FILE_NAME = ".events.lock"
_ROOT_SUBDIR = "optimizations"


@dataclass(frozen=True, slots=True)
class OptimizationManifest:
    schema_version: int
    optimization_id: str
    parent_experiment_id: str
    stage: OptimizationStage
    created_at: str
    updated_at: str
    total_outer_folds: int | None = None
    current_outer_fold_index: int | None = None
    completed_outer_fold_indices: tuple[int, ...] = ()
    total_trials_completed: int = 0
    total_trials_failed: int = 0
    total_trials_invalid: int = 0
    total_trials_pruned: int = 0
    winning_trial_by_outer_fold: Mapping[int, int] = field(default_factory=dict)
    trial_result_references: Mapping[str, ArtifactReference] = field(default_factory=dict)
    """`trial_reference_key(outer_fold_index, trial_number) -> ArtifactReference`
    for EVERY trial this optimization has ever completed/failed/marked-
    invalid/pruned, across every outer fold -- the manifest-level record
    `optimization.resume`/`optimization.verification` re-verify against
    the artifact store rather than trusting blindly. Flat (string-keyed)
    rather than nested, purely so this dataclass's JSON round-trip stays
    a single dict-of-primitives, like every other mapping field in this
    codebase's manifests."""
    outer_fold_result_references: Mapping[int, ArtifactReference] = field(default_factory=dict)
    summary_references: tuple[ArtifactReference, ...] = ()
    completed_at: str | None = None
    failure_summary: str | None = None
    resume_count: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.optimization_id):
            raise ValueError(f"OptimizationManifest.optimization_id must be a valid sha256 hex id, got {self.optimization_id!r}")
        if not is_valid_sha256_hex(self.parent_experiment_id):
            raise ValueError(f"OptimizationManifest.parent_experiment_id must be a valid sha256 hex id, got {self.parent_experiment_id!r}")
        parse_utc_timestamp(self.created_at)
        parse_utc_timestamp(self.updated_at)
        if self.completed_at is not None:
            parse_utc_timestamp(self.completed_at)
        if len(set(self.completed_outer_fold_indices)) != len(self.completed_outer_fold_indices):
            raise ValueError("OptimizationManifest.completed_outer_fold_indices must not contain duplicates")
        for name, value in (
            ("total_trials_completed", self.total_trials_completed), ("total_trials_failed", self.total_trials_failed),
            ("total_trials_invalid", self.total_trials_invalid), ("total_trials_pruned", self.total_trials_pruned),
            ("resume_count", self.resume_count),
        ):
            if value < 0:
                raise ValueError(f"OptimizationManifest.{name} must be >= 0, got {value}")
        if self.stage is OptimizationStage.FAILED and not self.failure_summary:
            raise ValueError("OptimizationManifest.failure_summary is required when stage=FAILED")
        if self.stage is not OptimizationStage.FAILED and self.failure_summary is not None:
            raise ValueError("OptimizationManifest.failure_summary must be None unless stage=FAILED")
        missing_refs = set(self.completed_outer_fold_indices) - set(self.outer_fold_result_references)
        if missing_refs:
            raise ValueError(
                f"OptimizationManifest.completed_outer_fold_indices references outer fold(s) {sorted(missing_refs)} "
                "with no corresponding outer_fold_result_references entry"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "parent_experiment_id": self.parent_experiment_id, "stage": self.stage.value,
            "created_at": self.created_at, "updated_at": self.updated_at, "total_outer_folds": self.total_outer_folds,
            "current_outer_fold_index": self.current_outer_fold_index,
            "completed_outer_fold_indices": list(self.completed_outer_fold_indices),
            "total_trials_completed": self.total_trials_completed, "total_trials_failed": self.total_trials_failed,
            "total_trials_invalid": self.total_trials_invalid, "total_trials_pruned": self.total_trials_pruned,
            "winning_trial_by_outer_fold": {str(k): v for k, v in sorted(self.winning_trial_by_outer_fold.items())},
            "trial_result_references": {k: v.to_json_dict() for k, v in sorted(self.trial_result_references.items())},
            "outer_fold_result_references": {str(k): v.to_json_dict() for k, v in sorted(self.outer_fold_result_references.items())},
            "summary_references": [a.to_json_dict() for a in self.summary_references],
            "completed_at": self.completed_at, "failure_summary": self.failure_summary, "resume_count": self.resume_count,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OptimizationManifest:
        require_schema_version(raw, supported=OPTIMIZATION_MANIFEST_SCHEMA_VERSION, context="OptimizationManifest")
        return cls(
            schema_version=OPTIMIZATION_MANIFEST_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            parent_experiment_id=str(raw["parent_experiment_id"]), stage=OptimizationStage(raw["stage"]),
            created_at=str(raw["created_at"]), updated_at=str(raw["updated_at"]),
            total_outer_folds=(None if raw.get("total_outer_folds") is None else int(str(raw["total_outer_folds"]))),
            current_outer_fold_index=(None if raw.get("current_outer_fold_index") is None else int(str(raw["current_outer_fold_index"]))),
            completed_outer_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("completed_outer_fold_indices") or [], field_name="completed_outer_fold_indices")
            ),
            total_trials_completed=int(str(raw.get("total_trials_completed", 0))), total_trials_failed=int(str(raw.get("total_trials_failed", 0))),
            total_trials_invalid=int(str(raw.get("total_trials_invalid", 0))), total_trials_pruned=int(str(raw.get("total_trials_pruned", 0))),
            winning_trial_by_outer_fold={
                int(k): int(v) for k, v in as_json_dict(raw.get("winning_trial_by_outer_fold") or {}, field_name="winning_trial_by_outer_fold").items()
            },
            trial_result_references={
                str(k): ArtifactReference.from_json_dict(as_json_dict(v, field_name="trial_result_references[]"))
                for k, v in as_json_dict(raw.get("trial_result_references") or {}, field_name="trial_result_references").items()
            },
            outer_fold_result_references={
                int(k): ArtifactReference.from_json_dict(as_json_dict(v, field_name="outer_fold_result_references[]"))
                for k, v in as_json_dict(raw.get("outer_fold_result_references") or {}, field_name="outer_fold_result_references").items()
            },
            summary_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="summary_references[]"))
                for a in as_json_list(raw.get("summary_references") or [], field_name="summary_references")
            ),
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_summary=(None if raw.get("failure_summary") is None else str(raw["failure_summary"])),
            resume_count=int(str(raw.get("resume_count", 0))),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
        )


def trial_reference_key(outer_fold_index: int, trial_number: int) -> str:
    return f"{outer_fold_index}:{trial_number}"


def trial_references_for_outer_fold(manifest: OptimizationManifest, outer_fold_index: int) -> dict[int, ArtifactReference]:
    prefix = f"{outer_fold_index}:"
    return {
        int(key[len(prefix):]): reference
        for key, reference in manifest.trial_result_references.items()
        if key.startswith(prefix)
    }


def _optimization_dir(root: Path, optimization_id: str) -> Path:
    if not is_valid_sha256_hex(optimization_id):
        raise ValueError(f"Invalid optimization_id {optimization_id!r}: must be a 64-character lowercase hex SHA-256 digest")
    path = root / _ROOT_SUBDIR / optimization_id
    assert_within_root(path, root=root)
    return path


class OptimizationManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _manifest_path(self, optimization_id: str) -> Path:
        return _optimization_dir(self._root, optimization_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, optimization_id: str) -> Path:
        return _optimization_dir(self._root, optimization_id) / _MANIFEST_LOCK_FILE_NAME

    def exists(self, optimization_id: str) -> bool:
        return self._manifest_path(optimization_id).is_file()

    def create(self, manifest: OptimizationManifest) -> None:
        if manifest.stage is not OptimizationStage.INITIALIZING:
            raise OptimizationStateError(
                f"OptimizationManifestStore.create requires stage=INITIALIZING, got {manifest.stage.value!r}",
                context={"stage": manifest.stage.value},
            )
        with experiment_lock(self._lock_path(manifest.optimization_id)):
            manifest_path = self._manifest_path(manifest.optimization_id)
            if manifest_path.is_file():
                raise OptimizationStateError(
                    f"An optimization manifest already exists for optimization_id={manifest.optimization_id!r}; refusing to overwrite",
                    context={"optimization_id": manifest.optimization_id, "path": str(manifest_path)},
                )
            write_json_atomic(manifest_path, manifest.to_json_dict())
        logger.info("Optimization manifest created: optimization_id=%s", manifest.optimization_id[:12])

    def load(self, optimization_id: str) -> OptimizationManifest:
        manifest_path = self._manifest_path(optimization_id)
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(
                f"No optimization manifest found for optimization_id={optimization_id!r}", context={"optimization_id": optimization_id}
            )
        raw = read_json_file(manifest_path)
        return OptimizationManifest.from_json_dict(as_json_dict(raw, field_name="optimization_manifest"))

    def load_if_exists(self, optimization_id: str) -> OptimizationManifest | None:
        if not self.exists(optimization_id):
            return None
        return self.load(optimization_id)

    def transition(
        self, optimization_id: str, *, new_stage: OptimizationStage, updated_at: str,
        total_outer_folds: int | None = None, current_outer_fold_index: int | None = -1,
        completed_outer_fold_indices: tuple[int, ...] | None = None,
        total_trials_completed: int | None = None, total_trials_failed: int | None = None,
        total_trials_invalid: int | None = None, total_trials_pruned: int | None = None,
        winning_trial_by_outer_fold: Mapping[int, int] | None = None,
        trial_result_references: Mapping[str, ArtifactReference] | None = None,
        outer_fold_result_references: Mapping[int, ArtifactReference] | None = None,
        summary_references: tuple[ArtifactReference, ...] | None = None,
        completed_at: str | None = None, failure_summary: str | None = None,
        artifact_references: tuple[ArtifactReference, ...] | None = None,
    ) -> OptimizationManifest:
        """`current_outer_fold_index`'s sentinel default (`-1`, never a
        valid outer-fold index) distinguishes "leave unchanged" (omitted)
        from "explicitly clear to None" (passed as `None`) -- exactly
        `execution.manifests.ExecutionManifestStore.transition`'s own
        convention for the identical ambiguity."""
        with experiment_lock(self._lock_path(optimization_id)):
            current = self.load(optimization_id)
            if not is_legal_optimization_transition(current.stage, new_stage):
                raise OptimizationStateError(
                    f"Illegal optimization stage transition for {optimization_id!r}: {current.stage.value!r} -> {new_stage.value!r}",
                    context={"optimization_id": optimization_id, "from": current.stage.value, "to": new_stage.value},
                )
            updated = replace(
                current, stage=new_stage, updated_at=updated_at,
                total_outer_folds=(current.total_outer_folds if total_outer_folds is None else total_outer_folds),
                current_outer_fold_index=(current.current_outer_fold_index if current_outer_fold_index == -1 else current_outer_fold_index),
                completed_outer_fold_indices=(current.completed_outer_fold_indices if completed_outer_fold_indices is None else completed_outer_fold_indices),
                total_trials_completed=(current.total_trials_completed if total_trials_completed is None else total_trials_completed),
                total_trials_failed=(current.total_trials_failed if total_trials_failed is None else total_trials_failed),
                total_trials_invalid=(current.total_trials_invalid if total_trials_invalid is None else total_trials_invalid),
                total_trials_pruned=(current.total_trials_pruned if total_trials_pruned is None else total_trials_pruned),
                winning_trial_by_outer_fold=(current.winning_trial_by_outer_fold if winning_trial_by_outer_fold is None else winning_trial_by_outer_fold),
                trial_result_references=(current.trial_result_references if trial_result_references is None else trial_result_references),
                outer_fold_result_references=(current.outer_fold_result_references if outer_fold_result_references is None else outer_fold_result_references),
                summary_references=(current.summary_references if summary_references is None else summary_references),
                completed_at=completed_at, failure_summary=failure_summary,
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
            )
            write_json_atomic(self._manifest_path(optimization_id), updated.to_json_dict())
        logger.info("Optimization stage transition: optimization_id=%s %s -> %s", optimization_id[:12], current.stage.value, new_stage.value)
        return updated

    def bump_resume_count(self, optimization_id: str) -> OptimizationManifest:
        with experiment_lock(self._lock_path(optimization_id)):
            current = self.load(optimization_id)
            updated = replace(current, updated_at=format_utc_timestamp(utc_now()), resume_count=current.resume_count + 1)
            write_json_atomic(self._manifest_path(optimization_id), updated.to_json_dict())
        logger.info("Optimization resume recorded: optimization_id=%s resume_count=%d", optimization_id[:12], updated.resume_count)
        return updated


class OptimizationEventType(Enum):
    OPTIMIZATION_CREATED = "optimization_created"
    RUN_STARTED = "run_started"
    OPTIMIZATION_RESUMED = "optimization_resumed"
    OUTER_FOLD_STARTED = "outer_fold_started"
    TRIAL_STARTED = "trial_started"
    TRIAL_COMPLETED = "trial_completed"
    TRIAL_FAILED = "trial_failed"
    TRIAL_INVALID = "trial_invalid"
    TRIAL_PRUNED = "trial_pruned"
    CANDIDATE_SELECTED = "candidate_selected"
    OUTER_FOLD_FINALIZED = "outer_fold_finalized"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    OPTIMIZATION_CANCELLED = "optimization_cancelled"


@dataclass(frozen=True, slots=True)
class OptimizationEventRecord:
    schema_version: int
    sequence: int
    optimization_id: str
    event_type: OptimizationEventType
    occurred_at: str
    details: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"OptimizationEventRecord.sequence must be >= 1, got {self.sequence}")
        if not is_valid_sha256_hex(self.optimization_id):
            raise ValueError(f"OptimizationEventRecord.optimization_id must be a valid sha256 hex id, got {self.optimization_id!r}")
        parse_utc_timestamp(self.occurred_at)
        validate_json_primitive_mapping(self.details, field_name="OptimizationEventRecord.details")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "sequence": self.sequence, "optimization_id": self.optimization_id,
            "event_type": self.event_type.value, "occurred_at": self.occurred_at, "details": dict(sorted(self.details.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OptimizationEventRecord:
        require_schema_version(raw, supported=1, context="OptimizationEventRecord")
        details_raw = raw.get("details") or {}
        if not isinstance(details_raw, dict):
            raise ValueError("OptimizationEventRecord.details must be a JSON object")
        return cls(
            schema_version=1, sequence=int(str(raw["sequence"])), optimization_id=str(raw["optimization_id"]),
            event_type=OptimizationEventType(raw["event_type"]), occurred_at=str(raw["occurred_at"]), details=dict(details_raw),
        )


class OptimizationEventStore:
    """Append-only per-optimization event log -- see module docstring for
    why this is a new class rather than a reuse of `ml.tracking.
    ExperimentEventStore`, and that class's own docstring for the
    JSON-Lines format/crash-safety/locking design this mirrors exactly."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _events_path(self, optimization_id: str) -> Path:
        return _optimization_dir(self._root, optimization_id) / _EVENTS_FILE_NAME

    def _lock_path(self, optimization_id: str) -> Path:
        return _optimization_dir(self._root, optimization_id) / _EVENTS_LOCK_FILE_NAME

    def _load_repaired(self, events_path: Path) -> list[OptimizationEventRecord]:
        if not events_path.is_file():
            return []
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return []
        records: list[OptimizationEventRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                raw = parse_json_strict(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                record = OptimizationEventRecord.from_json_dict(raw)
            except (ValueError, KeyError) as exc:
                if not is_last:
                    raise ArtifactCorruptionError(
                        f"Optimization event log {events_path} is corrupted: unreadable event at line {index + 1} "
                        f"(not the final line, so this cannot be an interrupted write): {exc}",
                        context={"path": str(events_path), "line": index + 1},
                    ) from exc
                logger.warning(
                    "Optimization event log %s: final line is unreadable (interrupted write) -- repairing by truncating it: %s",
                    events_path, exc,
                )
                self._rewrite(events_path, records)
                break
            if record.sequence != len(records) + 1:
                raise ArtifactCorruptionError(
                    f"Optimization event log {events_path} has a non-sequential sequence number at line {index + 1}: "
                    f"expected {len(records) + 1}, got {record.sequence}",
                    context={"path": str(events_path), "line": index + 1, "expected": len(records) + 1, "actual": record.sequence},
                )
            records.append(record)
        return records

    def _rewrite(self, events_path: Path, records: list[OptimizationEventRecord]) -> None:
        tmp_path = events_path.parent / f".{events_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            text = "".join(canonical_json_bytes(r.to_json_dict()).decode("utf-8") + "\n" for r in records)
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(events_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def append(
        self, optimization_id: str, event_type: OptimizationEventType, *, details: Mapping[str, JsonPrimitive] | None = None,
    ) -> OptimizationEventRecord:
        events_path = self._events_path(optimization_id)
        with experiment_lock(self._lock_path(optimization_id)):
            existing = self._load_repaired(events_path)
            record = OptimizationEventRecord(
                schema_version=1, sequence=len(existing) + 1, optimization_id=optimization_id, event_type=event_type,
                occurred_at=format_utc_timestamp(utc_now()), details=dict(details or {}),
            )
            events_path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(record.to_json_dict()).decode("utf-8") + "\n"
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        logger.info(
            "Optimization event appended: optimization_id=%s sequence=%d event_type=%s",
            optimization_id[:12], record.sequence, event_type.value,
        )
        return record

    def read_events(self, optimization_id: str) -> tuple[OptimizationEventRecord, ...]:
        with experiment_lock(self._lock_path(optimization_id)):
            return tuple(self._load_repaired(self._events_path(optimization_id)))


__all__ = [
    "OPTIMIZATION_MANIFEST_SCHEMA_VERSION",
    "OptimizationEventRecord",
    "OptimizationEventStore",
    "OptimizationEventType",
    "OptimizationManifest",
    "OptimizationManifestStore",
    "trial_reference_key",
    "trial_references_for_outer_fold",
]
