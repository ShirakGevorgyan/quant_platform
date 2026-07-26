"""`ExperimentManifest`: the complete, durable record of one experiment's
preparation -- and `ExperimentManifestStore`, which persists it under
`<ml_artifacts_root>/experiments/<experiment_id>/manifest.json` (a sibling
of `ml/artifacts.py`'s content-addressed store over the same root).

WHY THE MANIFEST EMBEDS THE FULL `ExperimentSpec` RATHER THAN DUPLICATING
EACH BINDING AS ITS OWN TOP-LEVEL FIELD
--------------------------------------------------------------------------
`ExperimentSpec` already IS the canonical, immutable record of every
dataset/feature/label/split/preprocessing/model/hyperparameter/seed/code-
revision binding. Re-declaring each of those as a second, separate
manifest-level field would create two sources of truth that could drift
apart. `ExperimentManifest.spec` is that single source; callers read
`manifest.spec.dataset_binding`, `manifest.spec.feature_binding`, etc.
directly. `model_definition_fingerprint` is the one binding that
deliberately lives OUTSIDE `ExperimentSpec`: it records the exact
`ModelRegistry` definition content (capabilities, serializer id) at
preparation time for provenance/validation, without making internal
registry bookkeeping changes (that don't touch name/version/objective/
hyperparameters) affect experiment identity.

STATUS LIFECYCLE -- WHY A MUTABLE "CURRENT MANIFEST" FILE, NOT VERSIONED
REVISIONS
--------------------------------------------------------------------------
Section 10 of this milestone's spec allows either "immutable revisions"
or "append-only events" to preserve status-transition history -- this
implementation chooses the LATTER, deliberately: `ml/tracking.py`
already provides an append-only event log (`experiment_created`,
`validation_started`, `run_completed`, ...) that records every
transition's full history. Duplicating that history a second time as
numbered manifest revisions would be redundant. Instead, ONE manifest
file per experiment holds the CURRENT status, and `ExperimentManifestStore.
transition()` overwrites it in place (atomically -- temp-file-then-
rename, so a reader never observes a half-written file) ONLY when
`models.is_legal_transition(current_status, new_status)` allows it.
Because every terminal status (`COMPLETED`, `FAILED`, `CANCELLED`) maps
to an EMPTY legal-transitions set, `transition()` can structurally never
modify a manifest again once it reaches one -- "immutable completed
manifests must not be edited in place" is a consequence of the
transition table, not a separate check bolted on top.

CONCURRENCY
--------------------------------------------------------------------------
`create()` and `transition()` both wrap their file operations in
`ml.concurrency.experiment_lock` (a thin adapter over
`historical.locking.DatasetLock`, reused as-is per this milestone's "do
not duplicate locking logic" instruction) keyed to that specific
experiment's lock file -- concurrent operations on the SAME experiment_id
are serialized; operations on DIFFERENT experiment_ids never contend.
This is local-filesystem advisory locking, not distributed consensus --
see `historical.locking`'s own module docstring for that limitation.
`experiment_lock` translates a contested or racing lock ACQUISITION into
`ExperimentLockError` (never a raw `DatasetLockError` or `PermissionError`
leaking across the ML public boundary); failures from the protected body
itself (e.g. a genuine disk error while writing the manifest) are never
touched -- see `ml/concurrency.py`'s module docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from quant_platform.core.exceptions import (
    ArtifactNotFoundError,
    ExperimentIdentityError,
    ExperimentStateError,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.experiment_identity import ExperimentIdentity, compute_experiment_identity
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import (
    ArtifactReference,
    EnvironmentSnapshot,
    ExperimentStatus,
    is_legal_transition,
)
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    assert_within_root,
    parse_utc_timestamp,
    read_json_file,
    require_schema_version,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
_MANIFEST_FILE_NAME = "manifest.json"
_LOCK_FILE_NAME = ".lock"


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    schema_version: int
    identity: ExperimentIdentity
    spec: ExperimentSpec
    model_definition_fingerprint: str
    status: ExperimentStatus
    environment_snapshot: EnvironmentSnapshot
    artifact_references: tuple[ArtifactReference, ...]
    validation_report_reference: ArtifactReference | None
    created_at: str
    completed_at: str | None = None
    failure_summary: str | None = None
    parent_experiment_id: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_definition_fingerprint:
            raise ValueError("ExperimentManifest.model_definition_fingerprint must not be empty")
        parse_utc_timestamp(self.created_at)
        if self.completed_at is not None:
            parse_utc_timestamp(self.completed_at)
        if self.parent_experiment_id is not None and not is_valid_sha256_hex(self.parent_experiment_id):
            raise ValueError(
                f"ExperimentManifest.parent_experiment_id must be a valid sha256 hex experiment id, "
                f"got {self.parent_experiment_id!r}"
            )
        if self.status not in _TERMINAL_STATUSES:
            if self.completed_at is not None:
                raise ValueError(f"ExperimentManifest.completed_at must be None while status={self.status.value!r}")
            if self.failure_summary is not None:
                raise ValueError(f"ExperimentManifest.failure_summary must be None while status={self.status.value!r}")
        if self.status is ExperimentStatus.FAILED and not self.failure_summary:
            raise ValueError("ExperimentManifest.failure_summary is required when status=FAILED")
        if self.status is not ExperimentStatus.FAILED and self.failure_summary is not None:
            raise ValueError("ExperimentManifest.failure_summary must be None unless status=FAILED")

        computed_identity = compute_experiment_identity(self.spec)
        if computed_identity != self.identity:
            raise ExperimentIdentityError(
                f"ExperimentManifest.identity {self.identity.experiment_id!r} does not match the "
                f"identity computed from its own embedded spec ({computed_identity.experiment_id!r}) -- "
                "a manifest's recorded identity must always agree with its own content",
                context={"recorded": self.identity.experiment_id, "computed": computed_identity.experiment_id},
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_json_dict(),
            "spec": self.spec.to_json_dict(),
            "model_definition_fingerprint": self.model_definition_fingerprint,
            "status": self.status.value,
            "environment_snapshot": self.environment_snapshot.to_json_dict(),
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
            "validation_report_reference": (
                None if self.validation_report_reference is None else self.validation_report_reference.to_json_dict()
            ),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "failure_summary": self.failure_summary,
            "parent_experiment_id": self.parent_experiment_id,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExperimentManifest:
        require_schema_version(raw, supported=MANIFEST_SCHEMA_VERSION, context="ExperimentManifest")
        validation_report_raw = raw.get("validation_report_reference")
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            identity=ExperimentIdentity.from_json_dict(as_json_dict(raw["identity"], field_name="identity")),
            spec=ExperimentSpec.from_json_dict(as_json_dict(raw["spec"], field_name="spec")),
            model_definition_fingerprint=str(raw["model_definition_fingerprint"]),
            status=ExperimentStatus(raw["status"]),
            environment_snapshot=EnvironmentSnapshot.from_json_dict(
                as_json_dict(raw["environment_snapshot"], field_name="environment_snapshot")
            ),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
            validation_report_reference=(
                None if validation_report_raw is None
                else ArtifactReference.from_json_dict(as_json_dict(validation_report_raw, field_name="validation_report_reference"))
            ),
            created_at=str(raw["created_at"]),
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_summary=(None if raw.get("failure_summary") is None else str(raw["failure_summary"])),
            parent_experiment_id=(None if raw.get("parent_experiment_id") is None else str(raw["parent_experiment_id"])),
            limitations=tuple(str(s) for s in as_json_list(raw.get("limitations") or [], field_name="limitations")),
        )


class ExperimentManifestStore:
    """Persists one `ExperimentManifest` per `experiment_id` under
    `<storage_root>/experiments/<experiment_id>/manifest.json`."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not is_valid_sha256_hex(experiment_id):
            raise ExperimentIdentityError(
                f"Invalid experiment_id {experiment_id!r}: must be a 64-character lowercase hex SHA-256 digest",
                context={"experiment_id": experiment_id},
            )
        path = self._root / "experiments" / experiment_id
        self._assert_within_root(path)
        return path

    def _manifest_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _LOCK_FILE_NAME

    def _assert_within_root(self, path: Path) -> None:
        assert_within_root(path, root=self._root)

    def exists(self, experiment_id: str) -> bool:
        return self._manifest_path(experiment_id).is_file()

    def create(self, manifest: ExperimentManifest) -> None:
        """Write a brand-new manifest. Raises `ExperimentStateError` if a
        manifest already exists for this `experiment_id` -- this store
        never silently overwrites; idempotent "prepare the same experiment
        twice" handling belongs to the caller (`experiment_manager.py`),
        which loads and compares before ever calling `create`."""
        if manifest.status is not ExperimentStatus.CREATED:
            raise ExperimentStateError(
                f"ExperimentManifestStore.create requires status=CREATED, got {manifest.status.value!r}",
                context={"status": manifest.status.value},
            )
        experiment_id = manifest.identity.experiment_id
        with experiment_lock(self._lock_path(experiment_id)):
            manifest_path = self._manifest_path(experiment_id)
            if manifest_path.is_file():
                raise ExperimentStateError(
                    f"A manifest already exists for experiment_id={experiment_id!r}; refusing to overwrite",
                    context={"experiment_id": experiment_id, "path": str(manifest_path)},
                )
            write_json_atomic(manifest_path, manifest.to_json_dict())
        logger.info("Experiment manifest created: experiment_id=%s", experiment_id[:12])

    def load(self, experiment_id: str) -> ExperimentManifest:
        manifest_path = self._manifest_path(experiment_id)
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(
                f"No experiment manifest found for experiment_id={experiment_id!r}",
                context={"experiment_id": experiment_id},
            )
        raw = read_json_file(manifest_path)
        return ExperimentManifest.from_json_dict(as_json_dict(raw, field_name="manifest"))

    def load_if_exists(self, experiment_id: str) -> ExperimentManifest | None:
        if not self.exists(experiment_id):
            return None
        return self.load(experiment_id)

    def transition(
        self,
        experiment_id: str,
        *,
        new_status: ExperimentStatus,
        completed_at: str | None = None,
        failure_summary: str | None = None,
        artifact_references: tuple[ArtifactReference, ...] | None = None,
        validation_report_reference: ArtifactReference | None = None,
    ) -> ExperimentManifest:
        """Legally transition an experiment's status, overwriting its
        manifest file in place (atomically). Raises `ExperimentStateError`
        if `new_status` is not a legal transition from the manifest's
        current status -- in particular, once a manifest reaches a
        terminal status, every call here raises, structurally preventing
        further edits."""
        with experiment_lock(self._lock_path(experiment_id)):
            current = self.load(experiment_id)
            if not is_legal_transition(current.status, new_status):
                raise ExperimentStateError(
                    f"Illegal experiment status transition for {experiment_id!r}: "
                    f"{current.status.value!r} -> {new_status.value!r}",
                    context={"experiment_id": experiment_id, "from": current.status.value, "to": new_status.value},
                )
            updated = replace(
                current,
                status=new_status,
                completed_at=completed_at,
                failure_summary=failure_summary,
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
                validation_report_reference=(
                    current.validation_report_reference if validation_report_reference is None else validation_report_reference
                ),
            )
            write_json_atomic(self._manifest_path(experiment_id), updated.to_json_dict())
        logger.info(
            "Experiment status transition: experiment_id=%s %s -> %s",
            experiment_id[:12], current.status.value, new_status.value,
        )
        return updated

    def list_experiment_ids(self) -> list[str]:
        experiments_dir = self._root / "experiments"
        if not experiments_dir.is_dir():
            return []
        return sorted(
            child.name for child in experiments_dir.iterdir()
            if child.is_dir() and (child / _MANIFEST_FILE_NAME).is_file()
        )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ExperimentManifest",
    "ExperimentManifestStore",
]
