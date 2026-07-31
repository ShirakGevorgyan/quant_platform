"""Execution session manifest and its durable store (Milestone 8,
Section 19). Mirrors `paper_trading.manifests.PaperSessionManifest`'s
identical role and shape one layer up, including its exact locking
convention -- `execution_session_lock` is a thin translation adapter
over `ml.concurrency.experiment_lock` (reused unchanged, never
reimplemented, per Section 20's own instruction), translating a
contested acquisition into this package's own `ExecutionSessionLockError`.

FAILED must record a typed failure category, the exact stage, an event
identity, recoverability, and a safe resume stage (Section 19) --
`ExecutionSessionManifest.failure_*` fields capture exactly these, all
optional (populated only when `stage is FAILED`)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from quant_platform.core.exceptions import (
    ExecutionGatewayManifestError,
    ExecutionSessionLockError,
    ExperimentLockError,
)
from quant_platform.execution_gateway.models import (
    ExecutionMode,
    ExecutionSessionStage,
    is_legal_execution_session_transition,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ArtifactReference
from quant_platform.ml.persistence import (
    as_json_dict,
    format_utc_timestamp,
    read_json_file,
    require_schema_version,
    utc_now,
    write_json_atomic,
)

EXECUTION_SESSION_MANIFEST_SCHEMA_VERSION = 1

_MANIFEST_FILE_NAME = "manifest.json"
_MANIFEST_LOCK_FILE_NAME = ".execution_session.lock"


@contextmanager
def execution_session_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise ExecutionSessionLockError(f"Could not acquire execution session lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc


@dataclass(frozen=True, slots=True)
class ExecutionSessionManifest:
    schema_version: int
    execution_session_id: str
    execution_gateway_spec_id: str
    paper_session_id: str
    adapter_id: str
    execution_mode: ExecutionMode
    current_stage: ExecutionSessionStage
    created_event_time: str
    last_transition_event_time: str
    spec_reference: ArtifactReference | None = None
    named_artifacts: Mapping[str, ArtifactReference] = field(default_factory=dict)
    failure_category: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    recoverable: bool | None = None
    safe_resume_stage: str | None = None
    semantic_digest: str | None = None
    final_verification_id: str | None = None
    resume_count: int = 0

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.execution_session_id):
            raise ExecutionGatewayManifestError(f"ExecutionSessionManifest.execution_session_id must be a valid sha256 hex digest, got {self.execution_session_id!r}")
        if self.resume_count < 0:
            raise ExecutionGatewayManifestError(f"ExecutionSessionManifest.resume_count must be >= 0, got {self.resume_count}")
        if self.current_stage is ExecutionSessionStage.FAILED:
            if self.failure_category is None or self.failure_stage is None:
                raise ExecutionGatewayManifestError("ExecutionSessionManifest: failure_category and failure_stage are required when current_stage is FAILED")
        elif any(v is not None for v in (self.failure_category, self.failure_stage, self.failure_reason, self.recoverable, self.safe_resume_stage)):
            raise ExecutionGatewayManifestError("ExecutionSessionManifest: failure_* fields must be None when current_stage is not FAILED")
        if self.current_stage is ExecutionSessionStage.COMPLETED and self.semantic_digest is None:
            raise ExecutionGatewayManifestError("ExecutionSessionManifest.semantic_digest is required when current_stage is COMPLETED")

    def artifact(self, kind: str) -> ArtifactReference | None:
        return self.named_artifacts.get(kind)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "execution_session_id": self.execution_session_id,
            "execution_gateway_spec_id": self.execution_gateway_spec_id, "paper_session_id": self.paper_session_id, "adapter_id": self.adapter_id,
            "execution_mode": self.execution_mode.value, "current_stage": self.current_stage.value, "created_event_time": self.created_event_time,
            "last_transition_event_time": self.last_transition_event_time,
            "spec_reference": (None if self.spec_reference is None else self.spec_reference.to_json_dict()),
            "named_artifacts": {k: v.to_json_dict() for k, v in sorted(self.named_artifacts.items())}, "failure_category": self.failure_category,
            "failure_stage": self.failure_stage, "failure_reason": self.failure_reason, "recoverable": self.recoverable,
            "safe_resume_stage": self.safe_resume_stage, "semantic_digest": self.semantic_digest, "final_verification_id": self.final_verification_id,
            "resume_count": self.resume_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionSessionManifest:
        require_schema_version(raw, supported=EXECUTION_SESSION_MANIFEST_SCHEMA_VERSION, context="ExecutionSessionManifest")
        spec_reference_raw = raw.get("spec_reference")
        named_artifacts_raw = as_json_dict(raw.get("named_artifacts") or {}, field_name="named_artifacts")
        return cls(
            schema_version=EXECUTION_SESSION_MANIFEST_SCHEMA_VERSION, execution_session_id=str(raw["execution_session_id"]),
            execution_gateway_spec_id=str(raw["execution_gateway_spec_id"]), paper_session_id=str(raw["paper_session_id"]), adapter_id=str(raw["adapter_id"]),
            execution_mode=ExecutionMode(raw["execution_mode"]), current_stage=ExecutionSessionStage(raw["current_stage"]),
            created_event_time=str(raw["created_event_time"]), last_transition_event_time=str(raw["last_transition_event_time"]),
            spec_reference=(None if spec_reference_raw is None else ArtifactReference.from_json_dict(as_json_dict(spec_reference_raw, field_name="spec_reference"))),
            named_artifacts={k: ArtifactReference.from_json_dict(as_json_dict(v, field_name=f"named_artifacts[{k!r}]")) for k, v in named_artifacts_raw.items()},
            failure_category=(None if raw.get("failure_category") is None else str(raw["failure_category"])),
            failure_stage=(None if raw.get("failure_stage") is None else str(raw["failure_stage"])),
            failure_reason=(None if raw.get("failure_reason") is None else str(raw["failure_reason"])),
            recoverable=(None if raw.get("recoverable") is None else bool(raw["recoverable"])),
            safe_resume_stage=(None if raw.get("safe_resume_stage") is None else str(raw["safe_resume_stage"])),
            semantic_digest=(None if raw.get("semantic_digest") is None else str(raw["semantic_digest"])),
            final_verification_id=(None if raw.get("final_verification_id") is None else str(raw["final_verification_id"])),
            resume_count=int(str(raw.get("resume_count", 0))),
        )


class ExecutionSessionManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _session_dir(self, execution_session_id: str) -> Path:
        if not is_valid_sha256_hex(execution_session_id):
            raise ExecutionGatewayManifestError(f"Invalid execution_session_id {execution_session_id!r}: must be a 64-character lowercase hex SHA-256 digest")
        return self._root / "execution_sessions" / execution_session_id

    def _manifest_path(self, execution_session_id: str) -> Path:
        return self._session_dir(execution_session_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, execution_session_id: str) -> Path:
        return self._session_dir(execution_session_id) / _MANIFEST_LOCK_FILE_NAME

    def exists(self, execution_session_id: str) -> bool:
        return self._manifest_path(execution_session_id).is_file()

    def _write(self, manifest: ExecutionSessionManifest) -> None:
        write_json_atomic(self._manifest_path(manifest.execution_session_id), manifest.to_json_dict())

    def create(
        self, *, execution_session_id: str, execution_gateway_spec_id: str, paper_session_id: str, adapter_id: str, execution_mode: ExecutionMode,
        spec_reference: ArtifactReference | None = None,
    ) -> ExecutionSessionManifest:
        with execution_session_lock(self._lock_path(execution_session_id)):
            if self.exists(execution_session_id):
                raise ExecutionGatewayManifestError(f"An execution session manifest already exists for execution_session_id={execution_session_id!r}")
            now = format_utc_timestamp(utc_now())
            manifest = ExecutionSessionManifest(
                schema_version=EXECUTION_SESSION_MANIFEST_SCHEMA_VERSION, execution_session_id=execution_session_id,
                execution_gateway_spec_id=execution_gateway_spec_id, paper_session_id=paper_session_id, adapter_id=adapter_id, execution_mode=execution_mode,
                current_stage=ExecutionSessionStage.CREATED, created_event_time=now, last_transition_event_time=now, spec_reference=spec_reference,
            )
            self._write(manifest)
            return manifest

    def load(self, execution_session_id: str) -> ExecutionSessionManifest:
        path = self._manifest_path(execution_session_id)
        if not path.is_file():
            raise ExecutionGatewayManifestError(f"No execution session manifest found for execution_session_id={execution_session_id!r}", context={"execution_session_id": execution_session_id})
        raw = read_json_file(path)
        return ExecutionSessionManifest.from_json_dict(as_json_dict(raw, field_name="execution_session_manifest"))

    def load_if_exists(self, execution_session_id: str) -> ExecutionSessionManifest | None:
        if not self.exists(execution_session_id):
            return None
        return self.load(execution_session_id)

    def transition(
        self, execution_session_id: str, *, target_stage: ExecutionSessionStage, named_artifacts: Mapping[str, ArtifactReference] | None = None,
        semantic_digest: str | None = None, final_verification_id: str | None = None, failure_category: str | None = None, failure_stage: str | None = None,
        failure_reason: str | None = None, recoverable: bool | None = None, safe_resume_stage: str | None = None,
    ) -> ExecutionSessionManifest:
        with execution_session_lock(self._lock_path(execution_session_id)):
            current = self.load(execution_session_id)
            if not is_legal_execution_session_transition(current.current_stage, target_stage):
                raise ExecutionGatewayManifestError(f"Illegal execution session transition {current.current_stage.value!r} -> {target_stage.value!r} for {execution_session_id!r}")
            updated = replace(
                current, current_stage=target_stage, last_transition_event_time=format_utc_timestamp(utc_now()),
                named_artifacts=(current.named_artifacts if named_artifacts is None else named_artifacts),
                semantic_digest=(current.semantic_digest if semantic_digest is None else semantic_digest),
                final_verification_id=(current.final_verification_id if final_verification_id is None else final_verification_id), failure_category=failure_category,
                failure_stage=failure_stage, failure_reason=failure_reason, recoverable=recoverable, safe_resume_stage=safe_resume_stage,
            )
            self._write(updated)
            return updated

    def bump_resume_count(self, execution_session_id: str) -> ExecutionSessionManifest:
        with execution_session_lock(self._lock_path(execution_session_id)):
            current = self.load(execution_session_id)
            updated = replace(current, resume_count=current.resume_count + 1, last_transition_event_time=format_utc_timestamp(utc_now()))
            self._write(updated)
            return updated


__all__ = [
    "EXECUTION_SESSION_MANIFEST_SCHEMA_VERSION",
    "ExecutionSessionManifest",
    "ExecutionSessionManifestStore",
    "execution_session_lock",
]
