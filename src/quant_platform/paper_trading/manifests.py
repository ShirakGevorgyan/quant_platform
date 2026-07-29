"""Paper session manifest and its durable store (Milestone 7, Section
20). `PaperSessionManifest` is the mutable "current state" record for one
paper-trading (or shadow-observation) session -- mirrors `robustness.
manifests.RobustnessManifest`/`backtesting.manifests.BacktestManifest`'s
identical role and shape one layer up, including their exact locking
convention (`ml.concurrency.experiment_lock`, reused unchanged rather than
reimplemented -- `session_lock` below is a thin translation adapter over
it, exactly like `ml.concurrency.experiment_lock` itself is a thin
adapter over `historical.locking.DatasetLock`).

FAILED must record a typed failure category, the exact stage, an event
identity, recoverability, and a safe resume stage (Section 20) --
`PaperSessionManifest.failure_*` fields capture exactly these, all
optional (populated only when `stage is FAILED`).

"Operational metadata must not affect session identity" (Section 20):
`created_at`/`updated_at`/`resume_count`/lock state never participate in
`paper_session_spec_id` (computed once, purely from the immutable spec,
in `specs.py`) -- this manifest carries that id as a foreign key, never
recomputes or influences it."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from quant_platform.core.exceptions import ExperimentLockError, PaperTradingManifestError, SessionLockError
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ArtifactReference
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    read_json_file,
    require_schema_version,
    utc_now,
    write_json_atomic,
)
from quant_platform.paper_trading.models import (
    PaperSessionStage,
    SessionMode,
    is_legal_paper_session_transition,
)

PAPER_SESSION_MANIFEST_SCHEMA_VERSION = 1

PAPER_SESSION_ARTIFACT_KINDS: tuple[str, ...] = (
    "eligibility_verification_report", "reconciliation_report", "paper_session_report", "execution_quality_report",
    "shadow_observation_report", "paper_verification_report", "backtest_paper_comparison_report",
)

_MANIFEST_FILE_NAME = "manifest.json"
_MANIFEST_LOCK_FILE_NAME = ".paper_session.lock"


@contextmanager
def session_lock(lock_path: Path) -> Iterator[None]:
    """Thin translation adapter over `ml.concurrency.experiment_lock`
    (itself a thin adapter over `historical.locking.DatasetLock`) --
    reuses the SAME file-locking mechanism every other manifest store in
    this platform uses, translating a contested acquisition into this
    package's own `SessionLockError` rather than leaking `ExperimentLockError`."""
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise SessionLockError(f"Could not acquire paper session lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc


@dataclass(frozen=True, slots=True)
class PaperSessionManifest:
    schema_version: int
    paper_session_id: str
    session_mode: SessionMode
    stage: PaperSessionStage
    created_at: str
    updated_at: str
    spec_reference: ArtifactReference | None = None
    named_artifacts: Mapping[str, ArtifactReference] = field(default_factory=dict)
    completed_at: str | None = None
    failure_category: str | None = None
    failure_stage: str | None = None
    failure_event_identity: str | None = None
    failure_recoverable: bool | None = None
    failure_safe_resume_stage: str | None = None
    resume_count: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.paper_session_id):
            raise PaperTradingManifestError(f"PaperSessionManifest.paper_session_id must be a valid sha256 hex digest, got {self.paper_session_id!r}")
        if self.resume_count < 0:
            raise PaperTradingManifestError(f"PaperSessionManifest.resume_count must be >= 0, got {self.resume_count}")
        if self.stage is PaperSessionStage.FAILED:
            if self.failure_category is None or self.failure_stage is None:
                raise PaperTradingManifestError("PaperSessionManifest: failure_category and failure_stage are required when stage is FAILED")
        elif any(v is not None for v in (self.failure_category, self.failure_stage, self.failure_event_identity, self.failure_recoverable, self.failure_safe_resume_stage)):
            raise PaperTradingManifestError("PaperSessionManifest: failure_* fields must be None when stage is not FAILED")
        if self.stage is PaperSessionStage.COMPLETED and self.completed_at is None:
            raise PaperTradingManifestError("PaperSessionManifest.completed_at is required when stage is COMPLETED")

    def artifact(self, kind: str) -> ArtifactReference | None:
        return self.named_artifacts.get(kind)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "paper_session_id": self.paper_session_id, "session_mode": self.session_mode.value,
            "stage": self.stage.value, "created_at": self.created_at, "updated_at": self.updated_at,
            "spec_reference": (None if self.spec_reference is None else self.spec_reference.to_json_dict()),
            "named_artifacts": {k: v.to_json_dict() for k, v in sorted(self.named_artifacts.items())},
            "completed_at": self.completed_at, "failure_category": self.failure_category, "failure_stage": self.failure_stage,
            "failure_event_identity": self.failure_event_identity, "failure_recoverable": self.failure_recoverable,
            "failure_safe_resume_stage": self.failure_safe_resume_stage, "resume_count": self.resume_count,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PaperSessionManifest:
        require_schema_version(raw, supported=PAPER_SESSION_MANIFEST_SCHEMA_VERSION, context="PaperSessionManifest")
        spec_reference_raw = raw.get("spec_reference")
        named_artifacts_raw = as_json_dict(raw.get("named_artifacts") or {}, field_name="named_artifacts")
        return cls(
            schema_version=PAPER_SESSION_MANIFEST_SCHEMA_VERSION, paper_session_id=str(raw["paper_session_id"]), session_mode=SessionMode(raw["session_mode"]),
            stage=PaperSessionStage(raw["stage"]), created_at=str(raw["created_at"]), updated_at=str(raw["updated_at"]),
            spec_reference=(None if spec_reference_raw is None else ArtifactReference.from_json_dict(as_json_dict(spec_reference_raw, field_name="spec_reference"))),
            named_artifacts={k: ArtifactReference.from_json_dict(as_json_dict(v, field_name=f"named_artifacts[{k!r}]")) for k, v in named_artifacts_raw.items()},
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_category=(None if raw.get("failure_category") is None else str(raw["failure_category"])),
            failure_stage=(None if raw.get("failure_stage") is None else str(raw["failure_stage"])),
            failure_event_identity=(None if raw.get("failure_event_identity") is None else str(raw["failure_event_identity"])),
            failure_recoverable=(None if raw.get("failure_recoverable") is None else bool(raw["failure_recoverable"])),
            failure_safe_resume_stage=(None if raw.get("failure_safe_resume_stage") is None else str(raw["failure_safe_resume_stage"])),
            resume_count=int(str(raw.get("resume_count", 0))),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
        )


class PaperSessionManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _session_dir(self, paper_session_id: str) -> Path:
        if not is_valid_sha256_hex(paper_session_id):
            raise PaperTradingManifestError(f"Invalid paper_session_id {paper_session_id!r}: must be a 64-character lowercase hex SHA-256 digest")
        return self._root / "paper_sessions" / paper_session_id

    def _manifest_path(self, paper_session_id: str) -> Path:
        return self._session_dir(paper_session_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, paper_session_id: str) -> Path:
        return self._session_dir(paper_session_id) / _MANIFEST_LOCK_FILE_NAME

    def exists(self, paper_session_id: str) -> bool:
        return self._manifest_path(paper_session_id).is_file()

    def _write(self, manifest: PaperSessionManifest) -> None:
        write_json_atomic(self._manifest_path(manifest.paper_session_id), manifest.to_json_dict())

    def create(self, *, paper_session_id: str, session_mode: SessionMode, spec_reference: ArtifactReference | None) -> PaperSessionManifest:
        with session_lock(self._lock_path(paper_session_id)):
            if self.exists(paper_session_id):
                raise PaperTradingManifestError(f"A paper session manifest already exists for paper_session_id={paper_session_id!r}")
            now = format_utc_timestamp(utc_now())
            manifest = PaperSessionManifest(
                schema_version=PAPER_SESSION_MANIFEST_SCHEMA_VERSION, paper_session_id=paper_session_id, session_mode=session_mode,
                stage=PaperSessionStage.CREATED, created_at=now, updated_at=now, spec_reference=spec_reference,
            )
            self._write(manifest)
            return manifest

    def load(self, paper_session_id: str) -> PaperSessionManifest:
        path = self._manifest_path(paper_session_id)
        if not path.is_file():
            raise PaperTradingManifestError(f"No paper session manifest found for paper_session_id={paper_session_id!r}", context={"paper_session_id": paper_session_id})
        raw = read_json_file(path)
        return PaperSessionManifest.from_json_dict(as_json_dict(raw, field_name="paper_session_manifest"))

    def load_if_exists(self, paper_session_id: str) -> PaperSessionManifest | None:
        if not self.exists(paper_session_id):
            return None
        return self.load(paper_session_id)

    def transition(
        self, paper_session_id: str, *, target_stage: PaperSessionStage, named_artifacts: Mapping[str, ArtifactReference] | None = None,
        artifact_references: tuple[ArtifactReference, ...] | None = None, completed_at: str | None = None, failure_category: str | None = None,
        failure_stage: str | None = None, failure_event_identity: str | None = None, failure_recoverable: bool | None = None,
        failure_safe_resume_stage: str | None = None,
    ) -> PaperSessionManifest:
        with session_lock(self._lock_path(paper_session_id)):
            current = self.load(paper_session_id)
            if not is_legal_paper_session_transition(current.stage, target_stage):
                raise PaperTradingManifestError(f"Illegal paper session transition {current.stage.value!r} -> {target_stage.value!r} for {paper_session_id!r}")
            updated = replace(
                current, stage=target_stage, updated_at=format_utc_timestamp(utc_now()),
                named_artifacts=(current.named_artifacts if named_artifacts is None else named_artifacts),
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
                completed_at=(current.completed_at if completed_at is None else completed_at), failure_category=failure_category,
                failure_stage=failure_stage, failure_event_identity=failure_event_identity, failure_recoverable=failure_recoverable,
                failure_safe_resume_stage=failure_safe_resume_stage,
            )
            self._write(updated)
            return updated

    def bump_resume_count(self, paper_session_id: str) -> PaperSessionManifest:
        with session_lock(self._lock_path(paper_session_id)):
            current = self.load(paper_session_id)
            updated = replace(current, resume_count=current.resume_count + 1, updated_at=format_utc_timestamp(utc_now()))
            self._write(updated)
            return updated


__all__ = [
    "PAPER_SESSION_ARTIFACT_KINDS",
    "PAPER_SESSION_MANIFEST_SCHEMA_VERSION",
    "PaperSessionManifest",
    "PaperSessionManifestStore",
    "session_lock",
]
