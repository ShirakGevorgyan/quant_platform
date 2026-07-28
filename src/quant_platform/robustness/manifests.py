"""`RobustnessManifest`/`RobustnessManifestStore` and
`RobustnessEventStore` (Milestone 6, Section 14) -- mirrors
`backtesting.manifests.BacktestManifest`/`BacktestManifestStore`/
`BacktestEventStore` exactly: a mutable CURRENT-state manifest (this
file) plus an append-only JSON-Lines event history, both rooted at
`<ml_artifacts_root>/robustness/<robustness_id>/` -- a SIBLING tree of
`backtests/<backtest_id>/`, never nested inside it.

Unlike `BacktestManifest` (which loops over outer folds), one
`RobustnessSpec` run is a SINGLE LINEAR pipeline (see `robustness.models.
RobustnessStage`'s own docstring) -- there is no per-fold looping state
to track. `stage_artifacts` is a single `RobustnessStage -> ArtifactReference`
mapping, one entry added per completed stage, instead of `BacktestManifest`'s
several stage-specific named fields -- a deliberate, documented
simplification this package's linear pipeline shape allows."""

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
    RobustnessStateError,
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
from quant_platform.robustness.models import RobustnessStage, is_legal_robustness_transition

logger = logging.getLogger(__name__)

ROBUSTNESS_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_FILE_NAME = "robustness_manifest.json"
_MANIFEST_LOCK_FILE_NAME = ".robustness.lock"
_EVENTS_FILE_NAME = "events.jsonl"
_EVENTS_LOCK_FILE_NAME = ".events.lock"
_ROOT_SUBDIR = "robustness"

ARTIFACT_KINDS: tuple[str, ...] = (
    "source_verification_report", "return_series_bundle", "bootstrap_report", "downside_analysis_report",
    "fold_stability_report", "sensitivity_report", "stress_report", "regime_report", "selection_report",
    "promotion_decision", "verification_report", "robustness_report",
)
"""The fixed, closed set of artifact-kind keys `RobustnessManifest.
named_artifacts` may use. Deliberately keyed by ARTIFACT KIND, not by
`RobustnessStage` -- `BOOTSTRAP_COMPLETED` alone produces two distinct
artifacts (`bootstrap_report` and `downside_analysis_report`, Sections
5/6), which a stage-keyed mapping could not represent 1:1."""


@dataclass(frozen=True, slots=True)
class RobustnessManifest:
    schema_version: int
    robustness_id: str
    source_backtest_id: str
    stage: RobustnessStage
    created_at: str
    updated_at: str
    spec_reference: ArtifactReference | None = None
    named_artifacts: Mapping[str, ArtifactReference] = field(default_factory=dict)
    completed_at: str | None = None
    failure_summary: str | None = None
    resume_count: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.robustness_id):
            raise ValueError(f"RobustnessManifest.robustness_id must be a valid sha256 hex id, got {self.robustness_id!r}")
        if not is_valid_sha256_hex(self.source_backtest_id):
            raise ValueError(f"RobustnessManifest.source_backtest_id must be a valid sha256 hex id, got {self.source_backtest_id!r}")
        parse_utc_timestamp(self.created_at)
        parse_utc_timestamp(self.updated_at)
        if self.completed_at is not None:
            parse_utc_timestamp(self.completed_at)
        if self.resume_count < 0:
            raise ValueError(f"RobustnessManifest.resume_count must be >= 0, got {self.resume_count}")
        if self.stage is RobustnessStage.FAILED and not self.failure_summary:
            raise ValueError("RobustnessManifest.failure_summary is required when stage=FAILED")
        if self.stage is not RobustnessStage.FAILED and self.failure_summary is not None:
            raise ValueError("RobustnessManifest.failure_summary must be None unless stage=FAILED")
        for key in self.named_artifacts:
            if key not in ARTIFACT_KINDS:
                raise ValueError(f"RobustnessManifest.named_artifacts has an unknown artifact kind key {key!r} -- must be one of {ARTIFACT_KINDS}")

    def artifact(self, kind: str) -> ArtifactReference | None:
        return self.named_artifacts.get(kind)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "robustness_id": self.robustness_id, "source_backtest_id": self.source_backtest_id,
            "stage": self.stage.value, "created_at": self.created_at, "updated_at": self.updated_at,
            "spec_reference": (None if self.spec_reference is None else self.spec_reference.to_json_dict()),
            "named_artifacts": {k: v.to_json_dict() for k, v in sorted(self.named_artifacts.items())},
            "completed_at": self.completed_at, "failure_summary": self.failure_summary, "resume_count": self.resume_count,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RobustnessManifest:
        require_schema_version(raw, supported=ROBUSTNESS_MANIFEST_SCHEMA_VERSION, context="RobustnessManifest")

        def _opt_ref(key: str) -> ArtifactReference | None:
            value = raw.get(key)
            return None if value is None else ArtifactReference.from_json_dict(as_json_dict(value, field_name=key))

        return cls(
            schema_version=ROBUSTNESS_MANIFEST_SCHEMA_VERSION, robustness_id=str(raw["robustness_id"]),
            source_backtest_id=str(raw["source_backtest_id"]), stage=RobustnessStage(raw["stage"]),
            created_at=str(raw["created_at"]), updated_at=str(raw["updated_at"]), spec_reference=_opt_ref("spec_reference"),
            named_artifacts={
                str(k): ArtifactReference.from_json_dict(as_json_dict(v, field_name="named_artifacts[]"))
                for k, v in as_json_dict(raw.get("named_artifacts") or {}, field_name="named_artifacts").items()
            },
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_summary=(None if raw.get("failure_summary") is None else str(raw["failure_summary"])),
            resume_count=int(str(raw.get("resume_count", 0))),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
        )


def _robustness_dir(root: Path, robustness_id: str) -> Path:
    if not is_valid_sha256_hex(robustness_id):
        raise ValueError(f"Invalid robustness_id {robustness_id!r}: must be a 64-character lowercase hex SHA-256 digest")
    path = root / _ROOT_SUBDIR / robustness_id
    assert_within_root(path, root=root)
    return path


class RobustnessManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _manifest_path(self, robustness_id: str) -> Path:
        return _robustness_dir(self._root, robustness_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, robustness_id: str) -> Path:
        return _robustness_dir(self._root, robustness_id) / _MANIFEST_LOCK_FILE_NAME

    def exists(self, robustness_id: str) -> bool:
        return self._manifest_path(robustness_id).is_file()

    def create(self, manifest: RobustnessManifest) -> None:
        if manifest.stage is not RobustnessStage.CREATED:
            raise RobustnessStateError(
                f"RobustnessManifestStore.create requires stage=CREATED, got {manifest.stage.value!r}",
                context={"stage": manifest.stage.value},
            )
        with experiment_lock(self._lock_path(manifest.robustness_id)):
            manifest_path = self._manifest_path(manifest.robustness_id)
            if manifest_path.is_file():
                raise RobustnessStateError(
                    f"A robustness manifest already exists for robustness_id={manifest.robustness_id!r}; refusing to overwrite",
                    context={"robustness_id": manifest.robustness_id, "path": str(manifest_path)},
                )
            write_json_atomic(manifest_path, manifest.to_json_dict())
        logger.info("Robustness manifest created: robustness_id=%s", manifest.robustness_id[:12])

    def load(self, robustness_id: str) -> RobustnessManifest:
        manifest_path = self._manifest_path(robustness_id)
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(f"No robustness manifest found for robustness_id={robustness_id!r}", context={"robustness_id": robustness_id})
        raw = read_json_file(manifest_path)
        return RobustnessManifest.from_json_dict(as_json_dict(raw, field_name="robustness_manifest"))

    def load_if_exists(self, robustness_id: str) -> RobustnessManifest | None:
        if not self.exists(robustness_id):
            return None
        return self.load(robustness_id)

    def transition(
        self, robustness_id: str, *, new_stage: RobustnessStage, updated_at: str,
        new_named_artifacts: tuple[tuple[str, ArtifactReference], ...] = (),
        completed_at: str | None = None, failure_summary: str | None = None,
        artifact_references: tuple[ArtifactReference, ...] | None = None,
    ) -> RobustnessManifest:
        """`new_named_artifacts` accepts more than one (kind, reference)
        pair in a single transition, since `BOOTSTRAP_COMPLETED` alone
        produces two distinct artifacts (`bootstrap_report` and
        `downside_analysis_report`, Sections 5/6) -- see `ARTIFACT_KINDS`."""
        with experiment_lock(self._lock_path(robustness_id)):
            current = self.load(robustness_id)
            if not is_legal_robustness_transition(current.stage, new_stage):
                raise RobustnessStateError(
                    f"Illegal robustness stage transition for {robustness_id!r}: {current.stage.value!r} -> {new_stage.value!r}",
                    context={"robustness_id": robustness_id, "from": current.stage.value, "to": new_stage.value},
                )
            merged_named_artifacts = dict(current.named_artifacts)
            for kind, reference in new_named_artifacts:
                merged_named_artifacts[kind] = reference
            updated = replace(
                current, stage=new_stage, updated_at=updated_at, named_artifacts=merged_named_artifacts,
                completed_at=completed_at, failure_summary=failure_summary,
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
            )
            write_json_atomic(self._manifest_path(robustness_id), updated.to_json_dict())
        logger.info("Robustness stage transition: robustness_id=%s %s -> %s", robustness_id[:12], current.stage.value, new_stage.value)
        return updated

    def bump_resume_count(self, robustness_id: str) -> RobustnessManifest:
        with experiment_lock(self._lock_path(robustness_id)):
            current = self.load(robustness_id)
            updated = replace(current, updated_at=format_utc_timestamp(utc_now()), resume_count=current.resume_count + 1)
            write_json_atomic(self._manifest_path(robustness_id), updated.to_json_dict())
        logger.info("Robustness resume recorded: robustness_id=%s resume_count=%d", robustness_id[:12], updated.resume_count)
        return updated


class RobustnessEventType(Enum):
    ROBUSTNESS_CREATED = "robustness_created"
    RUN_STARTED = "run_started"
    ROBUSTNESS_RESUMED = "robustness_resumed"
    SOURCE_VERIFIED = "source_verified"
    SERIES_BUILT = "series_built"
    BOOTSTRAP_COMPLETED = "bootstrap_completed"
    STABILITY_COMPLETED = "stability_completed"
    STRESS_COMPLETED = "stress_completed"
    REGIMES_COMPLETED = "regimes_completed"
    SELECTION_COMPLETED = "selection_completed"
    PROMOTION_EVALUATED = "promotion_evaluated"
    RUN_VERIFIED = "run_verified"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    ROBUSTNESS_CANCELLED = "robustness_cancelled"


@dataclass(frozen=True, slots=True)
class RobustnessEventRecord:
    schema_version: int
    sequence: int
    robustness_id: str
    event_type: RobustnessEventType
    occurred_at: str
    details: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"RobustnessEventRecord.sequence must be >= 1, got {self.sequence}")
        if not is_valid_sha256_hex(self.robustness_id):
            raise ValueError(f"RobustnessEventRecord.robustness_id must be a valid sha256 hex id, got {self.robustness_id!r}")
        parse_utc_timestamp(self.occurred_at)
        validate_json_primitive_mapping(self.details, field_name="RobustnessEventRecord.details")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "sequence": self.sequence, "robustness_id": self.robustness_id,
            "event_type": self.event_type.value, "occurred_at": self.occurred_at, "details": dict(sorted(self.details.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RobustnessEventRecord:
        require_schema_version(raw, supported=1, context="RobustnessEventRecord")
        details_raw = raw.get("details") or {}
        if not isinstance(details_raw, dict):
            raise ValueError("RobustnessEventRecord.details must be a JSON object")
        return cls(
            schema_version=1, sequence=int(str(raw["sequence"])), robustness_id=str(raw["robustness_id"]),
            event_type=RobustnessEventType(raw["event_type"]), occurred_at=str(raw["occurred_at"]), details=dict(details_raw),
        )


class RobustnessEventStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _events_path(self, robustness_id: str) -> Path:
        return _robustness_dir(self._root, robustness_id) / _EVENTS_FILE_NAME

    def _lock_path(self, robustness_id: str) -> Path:
        return _robustness_dir(self._root, robustness_id) / _EVENTS_LOCK_FILE_NAME

    def _load_repaired(self, events_path: Path) -> list[RobustnessEventRecord]:
        if not events_path.is_file():
            return []
        text = events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return []
        records: list[RobustnessEventRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                raw = parse_json_strict(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
                record = RobustnessEventRecord.from_json_dict(raw)
            except (ValueError, KeyError) as exc:
                if not is_last:
                    raise ArtifactCorruptionError(
                        f"Robustness event log {events_path} is corrupted: unreadable event at line {index + 1} "
                        f"(not the final line, so this cannot be an interrupted write): {exc}",
                        context={"path": str(events_path), "line": index + 1},
                    ) from exc
                logger.warning(
                    "Robustness event log %s: final line is unreadable (interrupted write) -- repairing by truncating it: %s",
                    events_path, exc,
                )
                self._rewrite(events_path, records)
                break
            if record.sequence != len(records) + 1:
                raise ArtifactCorruptionError(
                    f"Robustness event log {events_path} has a non-sequential sequence number at line {index + 1}: "
                    f"expected {len(records) + 1}, got {record.sequence}",
                    context={"path": str(events_path), "line": index + 1, "expected": len(records) + 1, "actual": record.sequence},
                )
            records.append(record)
        return records

    def _rewrite(self, events_path: Path, records: list[RobustnessEventRecord]) -> None:
        tmp_path = events_path.parent / f".{events_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            text = "".join(canonical_json_bytes(r.to_json_dict()).decode("utf-8") + "\n" for r in records)
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(events_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def append(self, robustness_id: str, event_type: RobustnessEventType, *, details: Mapping[str, JsonPrimitive] | None = None) -> RobustnessEventRecord:
        events_path = self._events_path(robustness_id)
        with experiment_lock(self._lock_path(robustness_id)):
            existing = self._load_repaired(events_path)
            record = RobustnessEventRecord(
                schema_version=1, sequence=len(existing) + 1, robustness_id=robustness_id, event_type=event_type,
                occurred_at=format_utc_timestamp(utc_now()), details=dict(details or {}),
            )
            events_path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(record.to_json_dict()).decode("utf-8") + "\n"
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        logger.info("Robustness event appended: robustness_id=%s sequence=%d event_type=%s", robustness_id[:12], record.sequence, event_type.value)
        return record

    def read_events(self, robustness_id: str) -> tuple[RobustnessEventRecord, ...]:
        with experiment_lock(self._lock_path(robustness_id)):
            return tuple(self._load_repaired(self._events_path(robustness_id)))


__all__ = [
    "ARTIFACT_KINDS",
    "ROBUSTNESS_MANIFEST_SCHEMA_VERSION",
    "RobustnessEventRecord",
    "RobustnessEventStore",
    "RobustnessEventType",
    "RobustnessManifest",
    "RobustnessManifestStore",
]
