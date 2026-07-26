"""`ExperimentPreparer`: the one service this milestone provides for
turning an `ExperimentSpec` into a validated, `ready`-or-`failed`
`ExperimentManifest` -- CREATION AND PREPARATION ONLY. It never fits a
model, never produces a prediction, never writes a metrics artifact.
Section 14 of this milestone's spec is explicit that this class must not
train anything; the `TrainableModel.fit(...)` call that would turn a
`ready` manifest into a `running`/`completed` one belongs to a future
milestone entirely, and is deliberately absent here -- there is no
method on this class that resolves a model FACTORY and calls `.create()`
on it, only the registry lookup needed to validate that the declared
model/objective combination exists and is supported.

THE PREPARATION PIPELINE
--------------------------------------------------------------------------
    1. Compute the experiment's deterministic identity from `spec`.
    2. If a manifest already exists for that identity, return it AS-IS
       (idempotent -- see "IDEMPOTENCY AND DESCRIPTIVE METADATA" below).
       `ExperimentManifestStore.load` itself raises `ExperimentIdentityError`
       if that manifest's declared identity is ever inconsistent with its
       own embedded spec -- corruption is detected and fails loudly here
       automatically, with no separate check needed in this class.
    3. Resolve the model definition from the registry (propagates
       `UnknownModelDefinitionError` directly -- no manifest exists yet,
       so this failure is side-effect-free).
    4. Resolve the research dataset manifest referenced by
       `spec.dataset_binding` (propagates `ResearchDatasetError`/
       `ManifestError` directly, same reasoning).
    5. Capture an `EnvironmentSnapshot` (informational only).
    6. Create the immutable initial manifest at status `CREATED` and
       append an `experiment_created` event.
    7. Transition to `VALIDATING`, append `validation_started`.
    8. Run `validation.validate_experiment_spec` and write the resulting
       `ValidationReport` as a content-addressed `REPORT` artifact.
    9. Transition to `READY` (if the report has no `ERROR`/`CRITICAL`
       issues) or `FAILED` (recording a `failure_summary`) -- either way,
       recording the validation report's `ArtifactReference` on the
       manifest -- THEN append `validation_passed`/`validation_failed`.

ORDERING INVARIANT: MANIFEST WRITE ALWAYS HAPPENS-BEFORE ITS EVENT
--------------------------------------------------------------------------
Every step above writes/transitions the manifest FIRST and appends the
event describing that write SECOND (steps 6-7-9 all follow this order).
This is deliberate: `ExperimentManifest` is the authoritative current-
state record, `ExperimentEventStore` a supplementary append-only audit
trail (see `tracking.py`'s module docstring) -- so the event log must
never claim a transition happened before the manifest itself durably
reflects it. The residual gap this does NOT close: a crash strictly
BETWEEN a manifest write and its describing event-append still leaves
that one event missing from the log (an acknowledged, recoverable
incompleteness -- the manifest itself, the authoritative record, is
never wrong), since there is no cross-file atomic transaction spanning
both stores. A SEPARATE, also-documented limitation: a crash between
`create()`/`transition(VALIDATING)` and the final `READY`/`FAILED`
transition leaves that experiment_id permanently at its last-recorded
non-terminal status -- `prepare()`'s idempotency check (step 2) treats
ANY existing manifest, terminal or not, as "already prepared" and never
resumes/retries it. Resuming interrupted preparation would require
redesigning this class's idempotency semantics and is out of scope for
this milestone.

IDEMPOTENCY AND DESCRIPTIVE METADATA
--------------------------------------------------------------------------
Two calls to `prepare()` with specs that are scientifically identical
(same `to_identity_payload()`) but textually different `notes`/`tags`/
`primary_metric` produce the SAME `experiment_id`. The FIRST call's
descriptive metadata is what gets durably recorded; a later call for the
same identity returns the existing manifest UNCHANGED -- its notes/tags/
primary_metric are never silently rewritten. This is a deliberate
consequence of "never overwrite" (Section 14): there is no mechanism in
this milestone for updating an experiment's descriptive metadata after
its first preparation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.experiment_identity import compute_experiment_identity
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import MANIFEST_SCHEMA_VERSION, ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, ExperimentStatus
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.tracking import EventType, ExperimentEventStore
from quant_platform.ml.validation import validate_experiment_spec

logger = logging.getLogger(__name__)


class ExperimentPreparer:
    """See module docstring for the full pipeline. Construct one per
    `(ml_artifacts_root, model_registry, research_manifest_store)`
    combination -- it holds no experiment-specific state itself."""

    def __init__(
        self,
        *,
        ml_artifacts_root: Path | str,
        model_registry: ModelRegistry,
        research_manifest_store: ResearchManifestStore,
    ) -> None:
        self._artifacts_root = Path(ml_artifacts_root).resolve()
        self._model_registry = model_registry
        self._research_manifest_store = research_manifest_store
        self._manifest_store = ExperimentManifestStore(self._artifacts_root)
        self._artifact_store = MLArtifactStore(self._artifacts_root)
        self._event_store = ExperimentEventStore(self._artifacts_root)

    @property
    def manifest_store(self) -> ExperimentManifestStore:
        return self._manifest_store

    @property
    def event_store(self) -> ExperimentEventStore:
        return self._event_store

    @property
    def artifact_store(self) -> MLArtifactStore:
        return self._artifact_store

    def prepare(self, spec: ExperimentSpec) -> ExperimentManifest:
        identity = compute_experiment_identity(spec)
        experiment_id = identity.experiment_id

        existing = self._manifest_store.load_if_exists(experiment_id)
        if existing is not None:
            logger.info(
                "ExperimentPreparer.prepare: identical experiment already prepared (idempotent no-op): "
                "experiment_id=%s status=%s", experiment_id[:12], existing.status.value,
            )
            return existing

        model_definition = self._model_registry.get(spec.model_name, spec.model_version)
        dataset_manifest = self._research_manifest_store.load(
            spec.dataset_binding.dataset_id, spec.dataset_binding.manifest_version
        )
        environment_snapshot = capture_environment_snapshot()

        manifest = ExperimentManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            identity=identity,
            spec=spec,
            model_definition_fingerprint=model_definition.fingerprint(),
            status=ExperimentStatus.CREATED,
            environment_snapshot=environment_snapshot,
            artifact_references=(),
            validation_report_reference=None,
            created_at=format_utc_timestamp(utc_now()),
        )
        self._manifest_store.create(manifest)
        self._event_store.append(
            experiment_id, EventType.EXPERIMENT_CREATED,
            details={
                "model_name": spec.model_name, "model_version": spec.model_version,
                "dataset_id": spec.dataset_binding.dataset_id, "objective": spec.objective.value,
            },
        )
        logger.info("Experiment created: experiment_id=%s model=%s objective=%s", experiment_id[:12], model_definition.qualified_name, spec.objective.value)

        self._manifest_store.transition(experiment_id, new_status=ExperimentStatus.VALIDATING)
        self._event_store.append(experiment_id, EventType.VALIDATION_STARTED)

        report = validate_experiment_spec(
            spec, model_registry=self._model_registry, dataset_manifest=dataset_manifest,
            ml_artifacts_root=self._artifacts_root, expected_identity=identity,
        )
        report_bytes = canonical_json_bytes(report.to_json_dict())
        report_ref = self._artifact_store.write_artifact(report_bytes, category=ArtifactCategory.REPORT)

        if report.is_ready:
            updated = self._manifest_store.transition(
                experiment_id, new_status=ExperimentStatus.READY,
                artifact_references=(report_ref,), validation_report_reference=report_ref,
            )
            self._event_store.append(
                experiment_id, EventType.VALIDATION_PASSED,
                details={"warning_count": len(report.warnings), "info_count": len(report.infos)},
            )
            logger.info("Experiment ready: experiment_id=%s", experiment_id[:12])
        else:
            blocking = [*report.criticals, *report.errors]
            failure_summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            updated = self._manifest_store.transition(
                experiment_id, new_status=ExperimentStatus.FAILED, completed_at=format_utc_timestamp(utc_now()),
                failure_summary=failure_summary, artifact_references=(report_ref,), validation_report_reference=report_ref,
            )
            self._event_store.append(
                experiment_id, EventType.VALIDATION_FAILED,
                details={"critical_count": len(report.criticals), "error_count": len(report.errors)},
            )
            logger.warning("Experiment failed validation: experiment_id=%s reasons=%s", experiment_id[:12], failure_summary)

        return updated


__all__ = ["ExperimentPreparer"]
