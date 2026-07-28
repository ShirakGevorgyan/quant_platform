"""`BacktestManifest`/`BacktestManifestStore` and `BacktestEventStore`
(Milestone 5) -- mirrors `calibration.manifests.CalibrationManifest`/
`CalibrationManifestStore`/`CalibrationEventStore` exactly: a mutable
CURRENT-state manifest (this file) plus an append-only JSON-Lines event
history, both rooted at `<ml_artifacts_root>/backtests/<backtest_id>/` --
a SIBLING tree of `calibrations/<calibration_id>/`, never nested inside
it. One parent calibration can be the subject of MANY backtest runs, each
with its own independent `backtest_id` and directory."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from quant_platform.backtesting.models import BacktestStage, is_legal_backtest_transition
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    BacktestStateError,
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

logger = logging.getLogger(__name__)

BACKTEST_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_FILE_NAME = "backtest_manifest.json"
_MANIFEST_LOCK_FILE_NAME = ".backtest.lock"
_EVENTS_FILE_NAME = "events.jsonl"
_EVENTS_LOCK_FILE_NAME = ".events.lock"
_ROOT_SUBDIR = "backtests"


@dataclass(frozen=True, slots=True)
class BacktestManifest:
    schema_version: int
    backtest_id: str
    source_calibration_id: str
    stage: BacktestStage
    created_at: str
    updated_at: str
    spec_reference: ArtifactReference | None = None
    total_outer_folds: int | None = None
    current_outer_fold_index: int | None = None
    completed_outer_fold_indices: tuple[int, ...] = ()
    outer_fold_result_references: Mapping[int, ArtifactReference] = field(default_factory=dict)
    stitched_equity_reference: ArtifactReference | None = None
    """Milestone 5.1, Section 3: the run-level `stitching.
    StitchedWalkForwardEquity` artifact -- written once, after every outer
    fold has completed, alongside `aggregate_report_reference` (never
    per-fold, since it spans all of them)."""
    aggregate_report_reference: ArtifactReference | None = None
    verification_report_reference: ArtifactReference | None = None
    environment_snapshot_reference: ArtifactReference | None = None
    completed_at: str | None = None
    failure_summary: str | None = None
    resume_count: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.backtest_id):
            raise ValueError(f"BacktestManifest.backtest_id must be a valid sha256 hex id, got {self.backtest_id!r}")
        if not is_valid_sha256_hex(self.source_calibration_id):
            raise ValueError(f"BacktestManifest.source_calibration_id must be a valid sha256 hex id, got {self.source_calibration_id!r}")
        parse_utc_timestamp(self.created_at)
        parse_utc_timestamp(self.updated_at)
        if self.completed_at is not None:
            parse_utc_timestamp(self.completed_at)
        if len(set(self.completed_outer_fold_indices)) != len(self.completed_outer_fold_indices):
            raise ValueError("BacktestManifest.completed_outer_fold_indices must not contain duplicates")
        if self.resume_count < 0:
            raise ValueError(f"BacktestManifest.resume_count must be >= 0, got {self.resume_count}")
        if self.stage is BacktestStage.FAILED and not self.failure_summary:
            raise ValueError("BacktestManifest.failure_summary is required when stage=FAILED")
        if self.stage is not BacktestStage.FAILED and self.failure_summary is not None:
            raise ValueError("BacktestManifest.failure_summary must be None unless stage=FAILED")
        missing_refs = set(self.completed_outer_fold_indices) - set(self.outer_fold_result_references)
        if missing_refs:
            raise ValueError(
                f"BacktestManifest.completed_outer_fold_indices references outer fold(s) {sorted(missing_refs)} "
                "with no corresponding outer_fold_result_references entry"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "backtest_id": self.backtest_id, "source_calibration_id": self.source_calibration_id,
            "stage": self.stage.value, "created_at": self.created_at, "updated_at": self.updated_at,
            "spec_reference": (None if self.spec_reference is None else self.spec_reference.to_json_dict()),
            "total_outer_folds": self.total_outer_folds, "current_outer_fold_index": self.current_outer_fold_index,
            "completed_outer_fold_indices": list(self.completed_outer_fold_indices),
            "outer_fold_result_references": {str(k): v.to_json_dict() for k, v in sorted(self.outer_fold_result_references.items())},
            "stitched_equity_reference": (None if self.stitched_equity_reference is None else self.stitched_equity_reference.to_json_dict()),
            "aggregate_report_reference": (None if self.aggregate_report_reference is None else self.aggregate_report_reference.to_json_dict()),
            "verification_report_reference": (None if self.verification_report_reference is None else self.verification_report_reference.to_json_dict()),
            "environment_snapshot_reference": (None if self.environment_snapshot_reference is None else self.environment_snapshot_reference.to_json_dict()),
            "completed_at": self.completed_at, "failure_summary": self.failure_summary, "resume_count": self.resume_count,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestManifest:
        require_schema_version(raw, supported=BACKTEST_MANIFEST_SCHEMA_VERSION, context="BacktestManifest")

        def _opt_ref(key: str) -> ArtifactReference | None:
            value = raw.get(key)
            return None if value is None else ArtifactReference.from_json_dict(as_json_dict(value, field_name=key))

        return cls(
            schema_version=BACKTEST_MANIFEST_SCHEMA_VERSION, backtest_id=str(raw["backtest_id"]),
            source_calibration_id=str(raw["source_calibration_id"]), stage=BacktestStage(raw["stage"]),
            created_at=str(raw["created_at"]), updated_at=str(raw["updated_at"]), spec_reference=_opt_ref("spec_reference"),
            total_outer_folds=(None if raw.get("total_outer_folds") is None else int(str(raw["total_outer_folds"]))),
            current_outer_fold_index=(None if raw.get("current_outer_fold_index") is None else int(str(raw["current_outer_fold_index"]))),
            completed_outer_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("completed_outer_fold_indices") or [], field_name="completed_outer_fold_indices")
            ),
            outer_fold_result_references={
                int(k): ArtifactReference.from_json_dict(as_json_dict(v, field_name="outer_fold_result_references[]"))
                for k, v in as_json_dict(raw.get("outer_fold_result_references") or {}, field_name="outer_fold_result_references").items()
            },
            stitched_equity_reference=_opt_ref("stitched_equity_reference"),
            aggregate_report_reference=_opt_ref("aggregate_report_reference"), verification_report_reference=_opt_ref("verification_report_reference"),
            environment_snapshot_reference=_opt_ref("environment_snapshot_reference"),
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_summary=(None if raw.get("failure_summary") is None else str(raw["failure_summary"])),
            resume_count=int(str(raw.get("resume_count", 0))),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
        )


def _backtest_dir(root: Path, backtest_id: str) -> Path:
    if not is_valid_sha256_hex(backtest_id):
        raise ValueError(f"Invalid backtest_id {backtest_id!r}: must be a 64-character lowercase hex SHA-256 digest")
    path = root / _ROOT_SUBDIR / backtest_id
    assert_within_root(path, root=root)
    return path


class BacktestManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _manifest_path(self, backtest_id: str) -> Path:
        return _backtest_dir(self._root, backtest_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, backtest_id: str) -> Path:
        return _backtest_dir(self._root, backtest_id) / _MANIFEST_LOCK_FILE_NAME

    def exists(self, backtest_id: str) -> bool:
        return self._manifest_path(backtest_id).is_file()

    def create(self, manifest: BacktestManifest) -> None:
        if manifest.stage is not BacktestStage.CREATED:
            raise BacktestStateError(
                f"BacktestManifestStore.create requires stage=CREATED, got {manifest.stage.value!r}",
                context={"stage": manifest.stage.value},
            )
        with experiment_lock(self._lock_path(manifest.backtest_id)):
            manifest_path = self._manifest_path(manifest.backtest_id)
            if manifest_path.is_file():
                raise BacktestStateError(
                    f"A backtest manifest already exists for backtest_id={manifest.backtest_id!r}; refusing to overwrite",
                    context={"backtest_id": manifest.backtest_id, "path": str(manifest_path)},
                )
            write_json_atomic(manifest_path, manifest.to_json_dict())
        logger.info("Backtest manifest created: backtest_id=%s", manifest.backtest_id[:12])

    def load(self, backtest_id: str) -> BacktestManifest:
        manifest_path = self._manifest_path(backtest_id)
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(f"No backtest manifest found for backtest_id={backtest_id!r}", context={"backtest_id": backtest_id})
        raw = read_json_file(manifest_path)
        return BacktestManifest.from_json_dict(as_json_dict(raw, field_name="backtest_manifest"))

    def load_if_exists(self, backtest_id: str) -> BacktestManifest | None:
        if not self.exists(backtest_id):
            return None
        return self.load(backtest_id)

    def transition(
        self, backtest_id: str, *, new_stage: BacktestStage, updated_at: str, spec_reference: ArtifactReference | None = None,
        total_outer_folds: int | None = None, current_outer_fold_index: int | None = -1,
        completed_outer_fold_indices: tuple[int, ...] | None = None,
        outer_fold_result_references: Mapping[int, ArtifactReference] | None = None,
        stitched_equity_reference: ArtifactReference | None = None,
        aggregate_report_reference: ArtifactReference | None = None, verification_report_reference: ArtifactReference | None = None,
        environment_snapshot_reference: ArtifactReference | None = None, completed_at: str | None = None,
        failure_summary: str | None = None, artifact_references: tuple[ArtifactReference, ...] | None = None,
    ) -> BacktestManifest:
        """`current_outer_fold_index`'s sentinel default (`-1`, never a
        valid outer-fold index) distinguishes "leave unchanged" (omitted)
        from "explicitly clear to None" (passed as `None`) -- exactly
        `calibration.manifests.CalibrationManifestStore.transition`'s own
        convention for the identical ambiguity."""
        with experiment_lock(self._lock_path(backtest_id)):
            current = self.load(backtest_id)
            if not is_legal_backtest_transition(current.stage, new_stage):
                raise BacktestStateError(
                    f"Illegal backtest stage transition for {backtest_id!r}: {current.stage.value!r} -> {new_stage.value!r}",
                    context={"backtest_id": backtest_id, "from": current.stage.value, "to": new_stage.value},
                )
            updated = replace(
                current, stage=new_stage, updated_at=updated_at,
                spec_reference=(current.spec_reference if spec_reference is None else spec_reference),
                total_outer_folds=(current.total_outer_folds if total_outer_folds is None else total_outer_folds),
                current_outer_fold_index=(current.current_outer_fold_index if current_outer_fold_index == -1 else current_outer_fold_index),
                completed_outer_fold_indices=(current.completed_outer_fold_indices if completed_outer_fold_indices is None else completed_outer_fold_indices),
                outer_fold_result_references=(current.outer_fold_result_references if outer_fold_result_references is None else outer_fold_result_references),
                stitched_equity_reference=(current.stitched_equity_reference if stitched_equity_reference is None else stitched_equity_reference),
                aggregate_report_reference=(current.aggregate_report_reference if aggregate_report_reference is None else aggregate_report_reference),
                verification_report_reference=(current.verification_report_reference if verification_report_reference is None else verification_report_reference),
                environment_snapshot_reference=(current.environment_snapshot_reference if environment_snapshot_reference is None else environment_snapshot_reference),
                completed_at=completed_at, failure_summary=failure_summary,
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
            )
            write_json_atomic(self._manifest_path(backtest_id), updated.to_json_dict())
        logger.info("Backtest stage transition: backtest_id=%s %s -> %s", backtest_id[:12], current.stage.value, new_stage.value)
        return updated

    def bump_resume_count(self, backtest_id: str) -> BacktestManifest:
        with experiment_lock(self._lock_path(backtest_id)):
            current = self.load(backtest_id)
            updated = replace(current, updated_at=format_utc_timestamp(utc_now()), resume_count=current.resume_count + 1)
            write_json_atomic(self._manifest_path(backtest_id), updated.to_json_dict())
        logger.info("Backtest resume recorded: backtest_id=%s resume_count=%d", backtest_id[:12], updated.resume_count)
        return updated


class BacktestEventType(Enum):
    BACKTEST_CREATED = "backtest_created"
    RUN_STARTED = "run_started"
    BACKTEST_RESUMED = "backtest_resumed"
    OUTER_FOLD_STARTED = "outer_fold_started"
    SOURCES_VERIFIED = "sources_verified"
    SIGNALS_GENERATED = "signals_generated"
    FILLS_COMPUTED = "fills_computed"
    TRADES_CONSTRUCTED = "trades_constructed"
    RETURNS_COMPUTED = "returns_computed"
    METRICS_COMPUTED = "metrics_computed"
    OUTER_FOLD_REPORT_READY = "outer_fold_report_ready"
    OUTER_FOLD_COMPLETED = "outer_fold_completed"
    RUN_VERIFIED = "run_verified"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    BACKTEST_CANCELLED = "backtest_cancelled"


@dataclass(frozen=True, slots=True)
class BacktestEventRecord:
    schema_version: int
    sequence: int
    backtest_id: str
    event_type: BacktestEventType
    occurred_at: str
    details: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"BacktestEventRecord.sequence must be >= 1, got {self.sequence}")
        if not is_valid_sha256_hex(self.backtest_id):
            raise ValueError(f"BacktestEventRecord.backtest_id must be a valid sha256 hex id, got {self.backtest_id!r}")
        parse_utc_timestamp(self.occurred_at)
        validate_json_primitive_mapping(self.details, field_name="BacktestEventRecord.details")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "sequence": self.sequence, "backtest_id": self.backtest_id,
            "event_type": self.event_type.value, "occurred_at": self.occurred_at, "details": dict(sorted(self.details.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestEventRecord:
        require_schema_version(raw, supported=1, context="BacktestEventRecord")
        details_raw = raw.get("details") or {}
        if not isinstance(details_raw, dict):
            raise ValueError("BacktestEventRecord.details must be a JSON object")
        return cls(
            schema_version=1, sequence=int(str(raw["sequence"])), backtest_id=str(raw["backtest_id"]),
            event_type=BacktestEventType(raw["event_type"]), occurred_at=str(raw["occurred_at"]), details=dict(details_raw),
        )


class BacktestEventStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _events_path(self, backtest_id: str) -> Path:
        return _backtest_dir(self._root, backtest_id) / _EVENTS_FILE_NAME

    def _lock_path(self, backtest_id: str) -> Path:
        return _backtest_dir(self._root, backtest_id) / _EVENTS_LOCK_FILE_NAME

    def _load_repaired(self, events_path: Path) -> list[BacktestEventRecord]:
        if not events_path.is_file():
            return []
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return []
        records: list[BacktestEventRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                raw = parse_json_strict(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                record = BacktestEventRecord.from_json_dict(raw)
            except (ValueError, KeyError) as exc:
                if not is_last:
                    raise ArtifactCorruptionError(
                        f"Backtest event log {events_path} is corrupted: unreadable event at line {index + 1} "
                        f"(not the final line, so this cannot be an interrupted write): {exc}",
                        context={"path": str(events_path), "line": index + 1},
                    ) from exc
                logger.warning(
                    "Backtest event log %s: final line is unreadable (interrupted write) -- repairing by truncating it: %s",
                    events_path, exc,
                )
                self._rewrite(events_path, records)
                break
            if record.sequence != len(records) + 1:
                raise ArtifactCorruptionError(
                    f"Backtest event log {events_path} has a non-sequential sequence number at line {index + 1}: "
                    f"expected {len(records) + 1}, got {record.sequence}",
                    context={"path": str(events_path), "line": index + 1, "expected": len(records) + 1, "actual": record.sequence},
                )
            records.append(record)
        return records

    def _rewrite(self, events_path: Path, records: list[BacktestEventRecord]) -> None:
        tmp_path = events_path.parent / f".{events_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            text = "".join(canonical_json_bytes(r.to_json_dict()).decode("utf-8") + "\n" for r in records)
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(events_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def append(self, backtest_id: str, event_type: BacktestEventType, *, details: Mapping[str, JsonPrimitive] | None = None) -> BacktestEventRecord:
        events_path = self._events_path(backtest_id)
        with experiment_lock(self._lock_path(backtest_id)):
            existing = self._load_repaired(events_path)
            record = BacktestEventRecord(
                schema_version=1, sequence=len(existing) + 1, backtest_id=backtest_id, event_type=event_type,
                occurred_at=format_utc_timestamp(utc_now()), details=dict(details or {}),
            )
            events_path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(record.to_json_dict()).decode("utf-8") + "\n"
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        logger.info("Backtest event appended: backtest_id=%s sequence=%d event_type=%s", backtest_id[:12], record.sequence, event_type.value)
        return record

    def read_events(self, backtest_id: str) -> tuple[BacktestEventRecord, ...]:
        with experiment_lock(self._lock_path(backtest_id)):
            return tuple(self._load_repaired(self._events_path(backtest_id)))


__all__ = [
    "BACKTEST_MANIFEST_SCHEMA_VERSION",
    "BacktestEventRecord",
    "BacktestEventStore",
    "BacktestEventType",
    "BacktestManifest",
    "BacktestManifestStore",
]
