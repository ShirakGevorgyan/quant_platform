"""`ExecutionManifest`: the durable, CURRENT-state record of one
experiment's execution -- the Milestone 4B analogue of `ml.manifests.
ExperimentManifest`, at a finer grain. Persisted at
`<ml_artifacts_root>/experiments/<experiment_id>/execution_manifest.json`,
a SIBLING of `ml.manifests.ExperimentManifestStore`'s own
`manifest.json` under the same experiment directory -- one experiment,
two manifests at two different levels of detail (coarse `ExperimentStatus`
vs. fine-grained `ExecutionStage`), exactly like `ml.manifests`/
`ml.tracking` already split "current state" from "append-only history"
into two focused files sharing one root.

WHY A SEPARATE LOCK FILE FROM `ExperimentManifestStore`
--------------------------------------------------------------------------
`ExperimentManifestStore`'s `.lock` guards brief, atomic read-modify-write
operations. An execution's lock (`.execution.lock`) is held for the
ENTIRE run -- potentially many folds, real wall-clock time -- specifically
to satisfy Section 11 ("prevent duplicate execution / parallel execution
of the same experiment"). Sharing one lock file between the two would
mean a long-running execution blocks even an unrelated `inspect-experiment`
read of the coarse manifest for its entire duration; using a distinct
file keeps those concerns independent, exactly as `ml.tracking.
ExperimentEventStore`'s `.events.lock` is already distinct from
`ExperimentManifestStore`'s `.lock` for the same reason.

STATUS LIFECYCLE -- MUTABLE CURRENT MANIFEST, NOT VERSIONED REVISIONS
--------------------------------------------------------------------------
Same choice `ml.manifests` made for `ExperimentManifest`, for the same
reason: `ml.tracking.ExperimentEventStore` (reused here too, via new
`FOLD_STARTED`/`FOLD_COMPLETED`/`FOLD_FAILED`/`EXECUTION_RESUMED` event
types) already preserves full history; this manifest holds only the
CURRENT stage, overwritten atomically in place on each legal transition.
Every terminal stage (`COMPLETED`, `FAILED`, `CANCELLED`) maps to an
EMPTY legal-transition set (`execution.state_machine`), so `transition()`
can structurally never modify a manifest again once it reaches one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from quant_platform.core.exceptions import ArtifactNotFoundError, ExecutionStateError
from quant_platform.execution.state_machine import ExecutionStage, is_legal_execution_transition
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ArtifactReference
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    assert_within_root,
    format_utc_timestamp,
    parse_utc_timestamp,
    read_json_file,
    require_schema_version,
    utc_now,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

EXECUTION_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_FILE_NAME = "execution_manifest.json"
_LOCK_FILE_NAME = ".execution.lock"

LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST = "research_dataset_manifest"
"""The only `label_horizon_source` value this milestone ever records --
named as a constant (rather than an inlined literal at the one call site)
so `ExecutionManifest.label_horizon_source`'s meaning is self-documenting
wherever it is read: the label horizon a fold plan was validated against
came from the bound `features.manifests.ResearchDatasetManifest`'s own
`label_definition`, NEVER from a CLI flag, `SplitBinding.params` entry, or
any other user-editable input (see `execution.runner.
extract_label_horizon_bars`)."""

SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE = "reject_if_declared_purge_below_required_label_horizon"
"""The only `split_policy` value this milestone implements: an execution
whose declared `purge_bars` is less than the dataset's required label-
information purge is REJECTED outright (see `execution_validation.
_validate_label_horizon_purge`), never silently widened to
`max(declared, required)`. Recorded on the manifest so a reader never has
to guess which of the two documented policy choices produced a given
`effective_purge_bars` value -- see `docs/execution_engine.md`'s
"Label-information purge" section for why the alternative (silent
widening) was rejected."""


@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    schema_version: int
    experiment_id: str
    stage: ExecutionStage
    created_at: str
    updated_at: str
    fold_plan_strategy: str | None = None
    total_folds: int | None = None
    declared_purge_bars: int | None = None
    required_label_purge_bars: int | None = None
    effective_purge_bars: int | None = None
    embargo_bars: int | None = None
    label_horizon_source: str | None = None
    split_policy: str | None = None
    """These six fields (`declared_purge_bars` through `split_policy`) are
    the label-information-leakage audit trail, all recorded together, once,
    at the SAME `BUILDING_SPLITS -> RUNNING_FOLD` transition that already
    records `fold_plan_strategy`/`total_folds` (see `execution.runner.
    ExecutionRunner._execute_pipeline`) -- never re-derived by a reader.
    `declared_purge_bars` is `FoldPlan.purge_bars` (from `SplitBinding.
    params`); `required_label_purge_bars` is `FoldPlan.
    required_label_purge_bars` (derived from the bound dataset's true
    label horizon, see `execution.splitters.required_label_purge_bars_for`);
    `effective_purge_bars` equals `declared_purge_bars` in EVERY execution
    that reaches this transition (the REJECTION policy this milestone
    implements refuses to build a `FoldPlan` at all otherwise -- see
    `execution_validation._validate_label_horizon_purge`) but is recorded
    as its own explicitly-named field rather than left for a reader to
    re-derive from the other two. No `ExperimentSpec` identity field
    changes because of any of this: `required_label_purge_bars` is a pure
    function of `dataset_binding.dataset_id`/`manifest_version`, both
    already identity-relevant, and research dataset manifests are
    immutable once saved (see `docs/execution_engine.md`)."""
    completed_fold_indices: tuple[int, ...] = ()
    failed_fold_indices: tuple[int, ...] = ()
    current_fold_index: int | None = None
    completed_at: str | None = None
    failure_summary: str | None = None
    resume_count: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()
    fold_result_references: Mapping[int, ArtifactReference] = field(default_factory=dict)
    """`fold_index -> ArtifactReference` for that fold's persisted
    `FOLD_RESULT` artifact -- lets `execution.resume` re-verify (never
    blindly trust) that a fold the manifest CLAIMS is completed still has
    genuinely intact, readable content in the artifact store."""

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.experiment_id):
            raise ValueError(f"ExecutionManifest.experiment_id must be a valid sha256 hex id, got {self.experiment_id!r}")
        parse_utc_timestamp(self.created_at)
        parse_utc_timestamp(self.updated_at)
        if self.completed_at is not None:
            parse_utc_timestamp(self.completed_at)
        if len(set(self.completed_fold_indices)) != len(self.completed_fold_indices):
            raise ValueError("ExecutionManifest.completed_fold_indices must not contain duplicates")
        if len(set(self.failed_fold_indices)) != len(self.failed_fold_indices):
            raise ValueError("ExecutionManifest.failed_fold_indices must not contain duplicates")
        if set(self.completed_fold_indices) & set(self.failed_fold_indices):
            raise ValueError("A fold index cannot be both completed and failed")
        if self.resume_count < 0:
            raise ValueError("ExecutionManifest.resume_count must be >= 0")
        if self.stage is ExecutionStage.FAILED and not self.failure_summary:
            raise ValueError("ExecutionManifest.failure_summary is required when stage=FAILED")
        if self.stage is not ExecutionStage.FAILED and self.failure_summary is not None:
            raise ValueError("ExecutionManifest.failure_summary must be None unless stage=FAILED")
        missing_refs = set(self.completed_fold_indices) - set(self.fold_result_references)
        if missing_refs:
            raise ValueError(
                f"ExecutionManifest.completed_fold_indices references fold(s) {sorted(missing_refs)} "
                "with no corresponding fold_result_references entry"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fold_plan_strategy": self.fold_plan_strategy,
            "total_folds": self.total_folds,
            "declared_purge_bars": self.declared_purge_bars,
            "required_label_purge_bars": self.required_label_purge_bars,
            "effective_purge_bars": self.effective_purge_bars,
            "embargo_bars": self.embargo_bars,
            "label_horizon_source": self.label_horizon_source,
            "split_policy": self.split_policy,
            "completed_fold_indices": list(self.completed_fold_indices),
            "fold_result_references": {
                str(k): v.to_json_dict() for k, v in sorted(self.fold_result_references.items())
            },
            "failed_fold_indices": list(self.failed_fold_indices),
            "current_fold_index": self.current_fold_index,
            "completed_at": self.completed_at,
            "failure_summary": self.failure_summary,
            "resume_count": self.resume_count,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionManifest:
        require_schema_version(raw, supported=EXECUTION_MANIFEST_SCHEMA_VERSION, context="ExecutionManifest")
        return cls(
            schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION,
            experiment_id=str(raw["experiment_id"]),
            stage=ExecutionStage(raw["stage"]),
            created_at=str(raw["created_at"]), updated_at=str(raw["updated_at"]),
            fold_plan_strategy=(None if raw.get("fold_plan_strategy") is None else str(raw["fold_plan_strategy"])),
            total_folds=(None if raw.get("total_folds") is None else int(str(raw["total_folds"]))),
            declared_purge_bars=(None if raw.get("declared_purge_bars") is None else int(str(raw["declared_purge_bars"]))),
            required_label_purge_bars=(
                None if raw.get("required_label_purge_bars") is None else int(str(raw["required_label_purge_bars"]))
            ),
            effective_purge_bars=(
                None if raw.get("effective_purge_bars") is None else int(str(raw["effective_purge_bars"]))
            ),
            embargo_bars=(None if raw.get("embargo_bars") is None else int(str(raw["embargo_bars"]))),
            label_horizon_source=(None if raw.get("label_horizon_source") is None else str(raw["label_horizon_source"])),
            split_policy=(None if raw.get("split_policy") is None else str(raw["split_policy"])),
            completed_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("completed_fold_indices") or [], field_name="completed_fold_indices")
            ),
            failed_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("failed_fold_indices") or [], field_name="failed_fold_indices")
            ),
            current_fold_index=(None if raw.get("current_fold_index") is None else int(str(raw["current_fold_index"]))),
            completed_at=(None if raw.get("completed_at") is None else str(raw["completed_at"])),
            failure_summary=(None if raw.get("failure_summary") is None else str(raw["failure_summary"])),
            resume_count=int(str(raw.get("resume_count", 0))),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
            fold_result_references={
                int(k): ArtifactReference.from_json_dict(as_json_dict(v, field_name="fold_result_references[]"))
                for k, v in as_json_dict(raw.get("fold_result_references") or {}, field_name="fold_result_references").items()
            },
        )


class ExecutionManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not is_valid_sha256_hex(experiment_id):
            raise ValueError(f"Invalid experiment_id {experiment_id!r}: must be a 64-character lowercase hex SHA-256 digest")
        path = self._root / "experiments" / experiment_id
        assert_within_root(path, root=self._root)
        return path

    def _manifest_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _MANIFEST_FILE_NAME

    def _lock_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / _LOCK_FILE_NAME

    def exists(self, experiment_id: str) -> bool:
        return self._manifest_path(experiment_id).is_file()

    def create(self, manifest: ExecutionManifest) -> None:
        if manifest.stage is not ExecutionStage.INITIALIZING:
            raise ExecutionStateError(
                f"ExecutionManifestStore.create requires stage=INITIALIZING, got {manifest.stage.value!r}",
                context={"stage": manifest.stage.value},
            )
        experiment_id = manifest.experiment_id
        with experiment_lock(self._lock_path(experiment_id)):
            manifest_path = self._manifest_path(experiment_id)
            if manifest_path.is_file():
                raise ExecutionStateError(
                    f"An execution manifest already exists for experiment_id={experiment_id!r}; refusing to overwrite",
                    context={"experiment_id": experiment_id, "path": str(manifest_path)},
                )
            write_json_atomic(manifest_path, manifest.to_json_dict())
        logger.info("Execution manifest created: experiment_id=%s", experiment_id[:12])

    def load(self, experiment_id: str) -> ExecutionManifest:
        manifest_path = self._manifest_path(experiment_id)
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(
                f"No execution manifest found for experiment_id={experiment_id!r}", context={"experiment_id": experiment_id}
            )
        raw = read_json_file(manifest_path)
        return ExecutionManifest.from_json_dict(as_json_dict(raw, field_name="execution_manifest"))

    def load_if_exists(self, experiment_id: str) -> ExecutionManifest | None:
        if not self.exists(experiment_id):
            return None
        return self.load(experiment_id)

    def transition(
        self,
        experiment_id: str,
        *,
        new_stage: ExecutionStage,
        updated_at: str,
        fold_plan_strategy: str | None = None,
        total_folds: int | None = None,
        declared_purge_bars: int | None = None,
        required_label_purge_bars: int | None = None,
        effective_purge_bars: int | None = None,
        embargo_bars: int | None = None,
        label_horizon_source: str | None = None,
        split_policy: str | None = None,
        completed_fold_indices: tuple[int, ...] | None = None,
        failed_fold_indices: tuple[int, ...] | None = None,
        current_fold_index: int | None = -1,
        completed_at: str | None = None,
        failure_summary: str | None = None,
        resume_count: int | None = None,
        artifact_references: tuple[ArtifactReference, ...] | None = None,
        fold_result_references: Mapping[int, ArtifactReference] | None = None,
    ) -> ExecutionManifest:
        """Legally transition an execution's stage, overwriting its
        manifest file in place (atomically). `current_fold_index`'s
        sentinel default (`-1`, never a valid fold index) distinguishes
        "leave unchanged" (omitted) from "explicitly clear to None"
        (passed as `None`) -- every OTHER optional field's absence (the
        ordinary `None` default) already means "leave unchanged", the
        same convention `ml.manifests.ExperimentManifestStore.transition`
        uses for `artifact_references`."""
        with experiment_lock(self._lock_path(experiment_id)):
            current = self.load(experiment_id)
            if not is_legal_execution_transition(current.stage, new_stage):
                raise ExecutionStateError(
                    f"Illegal execution stage transition for {experiment_id!r}: "
                    f"{current.stage.value!r} -> {new_stage.value!r}",
                    context={"experiment_id": experiment_id, "from": current.stage.value, "to": new_stage.value},
                )
            updated = replace(
                current,
                stage=new_stage,
                updated_at=updated_at,
                fold_plan_strategy=(current.fold_plan_strategy if fold_plan_strategy is None else fold_plan_strategy),
                total_folds=(current.total_folds if total_folds is None else total_folds),
                declared_purge_bars=(current.declared_purge_bars if declared_purge_bars is None else declared_purge_bars),
                required_label_purge_bars=(
                    current.required_label_purge_bars if required_label_purge_bars is None else required_label_purge_bars
                ),
                effective_purge_bars=(
                    current.effective_purge_bars if effective_purge_bars is None else effective_purge_bars
                ),
                embargo_bars=(current.embargo_bars if embargo_bars is None else embargo_bars),
                label_horizon_source=(
                    current.label_horizon_source if label_horizon_source is None else label_horizon_source
                ),
                split_policy=(current.split_policy if split_policy is None else split_policy),
                completed_fold_indices=(current.completed_fold_indices if completed_fold_indices is None else completed_fold_indices),
                failed_fold_indices=(current.failed_fold_indices if failed_fold_indices is None else failed_fold_indices),
                current_fold_index=(current.current_fold_index if current_fold_index == -1 else current_fold_index),
                completed_at=completed_at,
                failure_summary=failure_summary,
                resume_count=(current.resume_count if resume_count is None else resume_count),
                artifact_references=(current.artifact_references if artifact_references is None else artifact_references),
                fold_result_references=(
                    current.fold_result_references if fold_result_references is None else fold_result_references
                ),
            )
            write_json_atomic(self._manifest_path(experiment_id), updated.to_json_dict())
        logger.info(
            "Execution stage transition: experiment_id=%s %s -> %s",
            experiment_id[:12], current.stage.value, new_stage.value,
        )
        return updated

    def bump_resume_count(self, experiment_id: str) -> ExecutionManifest:
        """Records a resume ATTEMPT without changing `stage` -- NOT a
        stage transition (a resume of a `RECOVERABLE_FAILURE` or other
        non-terminal stage does not, by itself, advance the stage; the
        pipeline resuming from it does that via ordinary `transition()`
        calls next). Deliberately bypasses `is_legal_execution_transition`
        entirely rather than requiring meaningless self-loop entries
        (`X -> X`) in the legal-transition table for every stage."""
        with experiment_lock(self._lock_path(experiment_id)):
            current = self.load(experiment_id)
            updated = replace(
                current, updated_at=format_utc_timestamp(utc_now()), resume_count=current.resume_count + 1,
            )
            write_json_atomic(self._manifest_path(experiment_id), updated.to_json_dict())
        logger.info("Execution resume recorded: experiment_id=%s resume_count=%d", experiment_id[:12], updated.resume_count)
        return updated


__all__ = [
    "EXECUTION_MANIFEST_SCHEMA_VERSION",
    "LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST",
    "SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE",
    "ExecutionManifest",
    "ExecutionManifestStore",
]
