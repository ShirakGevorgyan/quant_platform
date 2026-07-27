"""`ExecutionRunner`: the ONE orchestrator this milestone ships for
actually RUNNING an already-`ready` `ExperimentSpec` (Milestone 4A) over
a walk-forward fold plan -- creation/preparation was 4A's job
(`ml.experiment_manager.ExperimentPreparer`); running is this one's.

THE PIPELINE (Section 5), AND HOW IT MAPS TO `ExecutionStage`
--------------------------------------------------------------------------
    load dataset        -> INITIALIZING -> LOADING_DATASET
    generate fold        -> LOADING_DATASET -> BUILDING_SPLITS
    validate fold        -> (still BUILDING_SPLITS; FAILED if invalid)
    run fold             -> BUILDING_SPLITS -> RUNNING_FOLD (repeats)
    collect outputs       -> (within RUNNING_FOLD, per fold)
    store fold artifacts  -> RUNNING_FOLD -> STORING_RESULTS (repeats)
    aggregate            -> STORING_RESULTS -> COMPLETED (final)

TWO INDEPENDENT MANIFESTS, TWO INDEPENDENT TRANSITIONS
--------------------------------------------------------------------------
`ml.manifests.ExperimentManifestStore` (Milestone 4A) sees exactly TWO
transitions from this class: `READY -> RUNNING` (once, at the very
start) and `RUNNING -> COMPLETED`/`RUNNING -> FAILED` (once, at the very
end). Everything else is tracked in `execution.manifests.
ExecutionManifestStore`. A `RECOVERABLE_FAILURE` NEVER touches the
`ExperimentManifest` at all -- it stays `RUNNING`, correctly describing
"still in progress, resumable", until a later call reaches a REAL
terminal outcome.

FOLD-LEVEL FAILURE VS. EXECUTION-LEVEL RECOVERABLE FAILURE
--------------------------------------------------------------------------
An exception raised while running ONE fold (e.g. the model/data for that
specific fold is somehow malformed) is recorded as THAT fold's
`FoldResult(status=FAILED)` -- the loop CONTINUES to the remaining folds
(other folds' data may be fine), and the execution's OVERALL stage ends
`FAILED` (not `COMPLETED`) once every fold has been attempted, never
silently reported as a success. An `ExperimentLockError` raised while
WRITING an artifact (this milestone's one clearly-recognizable
"transient, worth retrying" condition -- see `ml.concurrency.
experiment_lock`) is treated differently: it stops the fold loop
IMMEDIATELY and transitions the EXECUTION to `RECOVERABLE_FAILURE`,
re-raising the original error so the caller sees it -- a later
`resume()` call picks up exactly where this left off.

`STORING_RESULTS` IS ALWAYS REACHED BEFORE A TERMINAL TRANSITION
--------------------------------------------------------------------------
`execution.state_machine`'s legal-transition table only allows the final
`-> COMPLETED`/`-> FAILED` transition FROM `STORING_RESULTS`, never
directly from `RUNNING_FOLD` -- so even when a resumed run finds NOTHING
left to do (every fold already verified-complete), this class explicitly
performs one `RUNNING_FOLD -> STORING_RESULTS` "consolidation" transition
before aggregating, rather than special-casing the empty-loop path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ExecutionResumeError,
    ExperimentLockError,
    FeatureError,
    FoldValidationError,
    SchemaVersionError,
    UnknownModelDefinitionError,
)
from quant_platform.execution.context import FoldExecutionContext
from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.executor import DeterministicFoldExecutor, FoldData, FoldExecutor
from quant_platform.execution.manifests import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST,
    SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE,
    ExecutionManifest,
    ExecutionManifestStore,
)
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult, FoldStatus
from quant_platform.execution.resume import build_resume_plan
from quant_platform.execution.splitters import (
    Fold,
    build_folds_from_split_binding,
    reconstruct_dataset_timeline,
)
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.timeline import Timeline
from quant_platform.features.labels import LabelDefinition
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.interfaces import FeatureSchema, ModelDeserializer, ModelFactory, ModelSerializer
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference, ExperimentStatus
from quant_platform.ml.persistence import (
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    utc_now,
)
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedDomain
from quant_platform.ml.testing import ConstantTestModelDeserializer, ConstantTestModelSerializer
from quant_platform.ml.tracking import EventType, ExperimentEventStore

logger = logging.getLogger(__name__)

_LABEL_COLUMN = "label"
_TIMESTAMP_COLUMN = "open_time"
_EXECUTION_RUN_LOCK_FILE_NAME = ".execution_run.lock"
"""Deliberately distinct from `execution.manifests`' own `.execution.lock`
-- this is the OUTER lock, held for an entire run's duration (Section 11:
prevent duplicate/parallel execution); `ExecutionManifestStore`'s lock
guards its own brief, per-transition read-modify-write and is acquired
and released many times per run, by the SAME process, WHILE this outer
lock is held. Sharing one file between the two would self-deadlock,
since `historical.locking.DatasetLock` is not reentrant."""

_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError,
)
"""Every failure mode reading+decoding a previously-written durable
artifact can legitimately hit (missing/corrupted content, an unreadable
schema, malformed or non-finite JSON via `parse_json_strict` -- a
`ValueError` subclass --, or a `from_json_dict`/`__post_init__` field
problem). Mirrors `execution.verification`'s and `execution.resume`'s
identically-named constants: a completed execution's own previously-
verified-or-just-written artifacts must never let a raw decode exception
escape a public `ExecutionRunner` method -- see `_load_existing_aggregate`
and `_load_all_fold_results`."""

_SERIALIZER_REGISTRY: dict[str, tuple[ModelSerializer, ModelDeserializer]] = {
    # `ConstantTestModelSerializer.serialize`'s parameter is typed as the
    # concrete `FittedConstantTestModel`, narrower than the `ModelSerializer`
    # protocol's `FittedModel` -- sound in practice (this registry is only
    # ever consulted for a model whose OWN factory produced that exact
    # fitted type), but not something a structural type checker can prove;
    # narrowing here (once) is clearer than loosening this dict's value
    # type for every future entry.
    "constant_test_model_json_v1": (ConstantTestModelSerializer(), ConstantTestModelDeserializer()),  # type: ignore[dict-item]
}


def resolve_serializer(
    serializer_id: str, *, registry: Mapping[str, tuple[ModelSerializer, ModelDeserializer]] | None = None,
) -> ModelSerializer:
    """`ModelDefinition.serializer_id` is deliberately "resolved by a
    caller-provided lookup, never instantiated by the registry itself"
    (`ml.registry`'s own docstring) -- this is that lookup, the ML
    execution engine's one place that knows which serializer
    implementation goes with which id.

    `registry` defaults to the built-in `_SERIALIZER_REGISTRY` (only the
    test-only model) when omitted -- exactly the pre-Milestone-4C
    behavior, unchanged. A caller wanting real models resolved (e.g.
    `ExecutionRunner`, constructed with `additional_serializers`) passes
    its OWN merged mapping instead. Deliberately NOT a hardcoded, ever-
    growing module-level dict of every real model this platform will
    ever ship: `execution.runner` must stay unaware that `ml.model_zoo`
    (or any other real-model package) exists at all -- see
    `ExecutionRunner.__init__`'s `additional_serializers` parameter."""
    lookup = _SERIALIZER_REGISTRY if registry is None else registry
    entry = lookup.get(serializer_id)
    if entry is None:
        raise UnknownModelDefinitionError(
            f"No serializer registered for serializer_id={serializer_id!r}", context={"serializer_id": serializer_id}
        )
    return entry[0]


def extract_label_horizon_bars(dataset_manifest: ResearchDatasetManifest) -> int:
    """The dataset-manifest-derived FACT this milestone's label-information
    purge check is built on (see `execution.splitters.
    required_label_purge_bars_for` for the exact off-by-one proof, and
    `execution_validation._validate_label_horizon_purge` for the policy
    that consumes this value). Reuses Milestone 3's OWN typed
    `features.labels.LabelDefinition.from_json_dict` to parse the bound
    dataset's `label_definition` -- deliberately NOT a private/duplicated
    read of individual JSON keys, and NOT a new user-editable config
    field: the number returned here is a pure function of the already-
    loaded, immutable `ResearchDatasetManifest`, never a CLI parameter or
    a new `SplitBinding.params` entry.

    FAILS CLOSED: a `label_definition` that does not parse as a
    `LabelDefinition` (missing/malformed `name`/`kind`/`horizon_bars`, or
    an invalid `params` mapping) is a dataset this engine cannot safely
    reason about the leakage boundary for -- this raises one clear,
    actionable `FoldValidationError` rather than letting a raw
    `KeyError`/`ValueError`/`FeatureError` propagate to the caller."""
    try:
        label_definition = LabelDefinition.from_json_dict(dataset_manifest.label_definition)
    except (KeyError, ValueError, TypeError, FeatureError) as exc:
        raise FoldValidationError(
            f"Dataset {dataset_manifest.dataset_id!r} (version {dataset_manifest.version!r}) has a "
            f"label_definition that could not be parsed as a features.labels.LabelDefinition: {exc}. "
            "Refusing to execute: without a valid label horizon, the label-information purge required by "
            "this engine's leakage policy cannot be computed.",
            context={"dataset_id": dataset_manifest.dataset_id, "manifest_version": dataset_manifest.version},
        ) from exc
    return label_definition.horizon_bars


def assert_preprocessing_is_safe_for_execution(dataset_manifest: ResearchDatasetManifest) -> None:
    """Fail-closed preprocessing-leakage gate (Milestone 4B leakage audit).

    `execution.splitters.reconstruct_dataset_timeline` reassembles a
    dataset's FULL timeline, and this engine re-splits it using ITS OWN,
    independent fold configuration -- which does not, and cannot, align
    with whatever fold-group boundaries `features.dataset_builder.
    ResearchDatasetBuilder` used when it originally fit any
    `TransformPipeline` (scaler/imputer/etc). A `TransformPipeline` fit
    once on Milestone 3's OWN train indices is correct for MILESTONE 3'S
    OWN usage but says nothing about whether those same baked-in feature
    values stay safe under a DIFFERENT (this execution's) train/test
    boundary -- the fitted statistics may span rows this execution's own
    folds consider "future".

    This milestone does not implement a full preprocessing-refitting
    framework (out of scope, same as feature selection/hyperparameter
    optimization/calibration) -- instead it fails closed: an execution is
    refused outright whenever the bound dataset shows ANY sign of fitted
    preprocessing, checking BOTH manifest signals independently rather
    than trusting either alone:

      * `fitted_preprocessing_fingerprint is not None`
      * `preprocessing_definition` is non-empty

    (`features.dataset_builder.ResearchDatasetBuilder.build` sets
    `fitted_preprocessing_fingerprint` if, and only if, `request.
    preprocessing` was itself non-empty, so today the two signals always
    agree -- both are checked so this function does not silently stop
    detecting the unsafe state if a future Milestone 3 change ever lets
    them diverge.)

    The one dataset shape this engine CAN safely re-split is exactly the
    one every feature module registered as of this milestone actually
    produces: causal/raw features with `preprocessing_definition == {}`
    and `fitted_preprocessing_fingerprint is None` (see
    `docs/execution_engine.md`'s "Preprocessing safety" section for the
    code-level evidence this was verified against, and for the one
    documented residual gap this check cannot see:
    `MissingPolicyKind.TRAINING_STATISTIC_FILL`, a per-feature null-fill
    policy that is fold-group-fitted independently of `TransformPipeline`
    and currently unused by every registered feature module, but for
    which the manifest carries no typed signal at all)."""
    if dataset_manifest.fitted_preprocessing_fingerprint is not None or dataset_manifest.preprocessing_definition:
        raise FoldValidationError(
            f"Dataset {dataset_manifest.dataset_id!r} (version {dataset_manifest.version!r}) was built with "
            f"fitted preprocessing (preprocessing_definition={dict(dataset_manifest.preprocessing_definition)!r}, "
            f"fitted_preprocessing_fingerprint={dataset_manifest.fitted_preprocessing_fingerprint!r}). This "
            "engine re-splits this dataset's full timeline independently of whatever fold-group boundaries "
            "were used when that preprocessing was fit, so its baked-in feature values may depend on statistics "
            "spanning what this execution's own folds consider future data. Refusing to execute: only datasets "
            "with no fitted preprocessing (raw/causal features only) can be safely re-split by this engine -- "
            "see docs/execution_engine.md's 'Preprocessing safety' section.",
            context={"dataset_id": dataset_manifest.dataset_id, "manifest_version": dataset_manifest.version},
        )


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """What `ExecutionRunner.run`/`.resume` returns: the final aggregate
    plus whether this call actually did anything (a call against an
    already-`COMPLETED` execution is idempotent and returns the EXISTING
    aggregate without re-running a single fold)."""

    aggregate: AggregatedExecutionResult
    was_idempotent_no_op: bool


class ExecutionRunner:
    def __init__(
        self,
        *,
        ml_artifacts_root: Path | str,
        model_registry: ModelRegistry,
        research_manifest_store: ResearchManifestStore,
        research_dataset_store: ResearchDatasetStore,
        fold_executor: FoldExecutor | None = None,
        additional_serializers: Mapping[str, tuple[ModelSerializer, ModelDeserializer]] | None = None,
    ) -> None:
        self._artifacts_root = Path(ml_artifacts_root).resolve()
        self._model_registry = model_registry
        self._research_manifest_store = research_manifest_store
        self._research_dataset_store = research_dataset_store
        self._fold_executor: FoldExecutor = fold_executor if fold_executor is not None else DeterministicFoldExecutor()
        # Merged serializer lookup for THIS runner instance -- the built-in
        # test-only entry, plus whatever `additional_serializers` a caller
        # supplied (e.g. `ml.model_zoo.default_serializer_registry()` for
        # the real models). `execution.runner` itself still imports
        # nothing from `ml.model_zoo` (or any other real-model package) --
        # it only ever sees an externally-constructed mapping, preserving
        # "no model-specific execution logic inside ExecutionRunner".
        self._serializer_registry: dict[str, tuple[ModelSerializer, ModelDeserializer]] = {
            **_SERIALIZER_REGISTRY, **dict(additional_serializers or {}),
        }

        self._experiment_manifest_store = ExperimentManifestStore(self._artifacts_root)
        self._execution_manifest_store = ExecutionManifestStore(self._artifacts_root)
        self._artifact_store = MLArtifactStore(self._artifacts_root)
        self._event_store = ExperimentEventStore(self._artifacts_root)

    @property
    def execution_manifest_store(self) -> ExecutionManifestStore:
        return self._execution_manifest_store

    @property
    def artifact_store(self) -> MLArtifactStore:
        return self._artifact_store

    @property
    def event_store(self) -> ExperimentEventStore:
        return self._event_store

    def _lock_path(self, experiment_id: str) -> Path:
        return self._artifacts_root / "experiments" / experiment_id / _EXECUTION_RUN_LOCK_FILE_NAME

    def run(self, experiment_id: str, *, force_rerun_folds: frozenset[int] = frozenset()) -> ExecutionOutcome:
        """Starts a fresh execution, or transparently resumes one already
        in progress -- refuses only if the execution already reached a
        terminal stage (`COMPLETED`/`FAILED`/`CANCELLED`)."""
        with experiment_lock(self._lock_path(experiment_id)):
            return self._run_locked(experiment_id, force_rerun_folds=force_rerun_folds, require_existing=False)

    def resume(self, experiment_id: str, *, force_rerun_folds: frozenset[int] = frozenset()) -> ExecutionOutcome:
        """Same pipeline as `.run`, but REQUIRES a prior execution
        manifest to exist -- raises `ExecutionResumeError` (distinct from
        `.run`'s silent "start fresh") if there is nothing to resume, for
        callers that want to assert this is genuinely a resume."""
        if self._execution_manifest_store.load_if_exists(experiment_id) is None:
            raise ExecutionResumeError(
                f"No execution manifest exists for experiment_id={experiment_id!r} -- nothing to resume "
                "(use run() to start a fresh execution)",
                context={"experiment_id": experiment_id},
            )
        with experiment_lock(self._lock_path(experiment_id)):
            return self._run_locked(experiment_id, force_rerun_folds=force_rerun_folds, require_existing=True)

    def _run_locked(
        self, experiment_id: str, *, force_rerun_folds: frozenset[int], require_existing: bool,
    ) -> ExecutionOutcome:
        experiment_manifest = self._experiment_manifest_store.load(experiment_id)

        execution_manifest = self._execution_manifest_store.load_if_exists(experiment_id)
        if require_existing and execution_manifest is None:
            raise ExecutionResumeError(  # pragma: no cover - guarded identically in resume() before locking
                f"No execution manifest exists for experiment_id={experiment_id!r}", context={"experiment_id": experiment_id}
            )

        # The EXECUTION's own terminal state is checked BEFORE the
        # experiment's coarse status: once execution is COMPLETED/FAILED/
        # CANCELLED, `.run()` must behave idempotently (or refuse to
        # resume) regardless of what `ExperimentManifest.status` happens
        # to be -- checking the coarse status first would reject a
        # perfectly valid idempotent re-run of an already-COMPLETED
        # experiment (whose status is, by then, COMPLETED, not READY).
        if execution_manifest is not None and execution_manifest.stage in (
            ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED,
        ):
            if execution_manifest.stage is ExecutionStage.COMPLETED:
                return ExecutionOutcome(
                    aggregate=self._load_existing_aggregate(execution_manifest), was_idempotent_no_op=True,
                )
            raise ExecutionResumeError(
                f"Execution for experiment_id={experiment_id!r} already reached a terminal stage "
                f"{execution_manifest.stage.value!r} -- it cannot be resumed or restarted in place",
                context={"experiment_id": experiment_id, "stage": execution_manifest.stage.value},
            )

        if experiment_manifest.status not in (ExperimentStatus.READY, ExperimentStatus.RUNNING):
            raise ExecutionResumeError(
                f"Experiment {experiment_id!r} has status={experiment_manifest.status.value!r}; "
                "only a READY (or already RUNNING, i.e. previously started) experiment can be executed",
                context={"experiment_id": experiment_id, "status": experiment_manifest.status.value},
            )

        if experiment_manifest.status is ExperimentStatus.READY:
            self._experiment_manifest_store.transition(experiment_id, new_status=ExperimentStatus.RUNNING)

        if execution_manifest is None:
            now = format_utc_timestamp(utc_now())
            execution_manifest = ExecutionManifest(
                schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id=experiment_id,
                stage=ExecutionStage.INITIALIZING, created_at=now, updated_at=now,
            )
            self._execution_manifest_store.create(execution_manifest)
            self._event_store.append(experiment_id, EventType.RUN_STARTED)
        else:
            execution_manifest = self._execution_manifest_store.bump_resume_count(experiment_id)
            self._event_store.append(
                experiment_id, EventType.EXECUTION_RESUMED, details={"resume_count": execution_manifest.resume_count},
            )

        try:
            aggregate = self._execute_pipeline(
                experiment_id=experiment_id, execution_manifest=execution_manifest, force_rerun_folds=force_rerun_folds,
            )
        except ExperimentLockError:
            self._execution_manifest_store.transition(
                experiment_id, new_stage=ExecutionStage.RECOVERABLE_FAILURE, updated_at=format_utc_timestamp(utc_now()),
            )
            raise
        return ExecutionOutcome(aggregate=aggregate, was_idempotent_no_op=False)

    def _load_existing_aggregate(self, execution_manifest: ExecutionManifest) -> AggregatedExecutionResult:
        summary_ref = next(
            (r for r in execution_manifest.artifact_references if r.category is ArtifactCategory.EXECUTION_SUMMARY), None,
        )
        if summary_ref is None:
            raise ExecutionResumeError(  # pragma: no cover - defensive; a COMPLETED manifest always records one
                f"Execution manifest for {execution_manifest.experiment_id!r} is COMPLETED but has no recorded "
                "EXECUTION_SUMMARY artifact reference"
            )
        try:
            raw = self._artifact_store.read_artifact(summary_ref.content_hash)
            aggregate = AggregatedExecutionResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            raise ExecutionResumeError(
                f"Execution {execution_manifest.experiment_id!r} is recorded as COMPLETED, but its "
                f"EXECUTION_SUMMARY artifact could not be read and decoded: {exc}",
                context={"experiment_id": execution_manifest.experiment_id},
            ) from exc
        if aggregate.experiment_id != execution_manifest.experiment_id:
            raise ExecutionResumeError(
                f"Execution {execution_manifest.experiment_id!r}'s recorded EXECUTION_SUMMARY artifact decodes "
                f"to experiment_id={aggregate.experiment_id!r} -- a valid content hash proves the bytes are "
                "intact, not that this is genuinely this execution's own summary",
                context={"experiment_id": execution_manifest.experiment_id, "decoded_experiment_id": aggregate.experiment_id},
            )
        return aggregate

    def _execute_pipeline(
        self, *, experiment_id: str, execution_manifest: ExecutionManifest, force_rerun_folds: frozenset[int],
    ) -> AggregatedExecutionResult:
        experiment_manifest = self._experiment_manifest_store.load(experiment_id)
        spec = experiment_manifest.spec
        started_at = execution_manifest.created_at
        pipeline_started = time.perf_counter()

        stage = execution_manifest.stage
        if stage is ExecutionStage.INITIALIZING:
            stage = self._advance(experiment_id, ExecutionStage.LOADING_DATASET)

        dataset_manifest = self._research_manifest_store.load(
            spec.dataset_binding.dataset_id, spec.dataset_binding.manifest_version,
        )
        timeline = reconstruct_dataset_timeline(
            self._research_dataset_store, dataset_id=spec.dataset_binding.dataset_id,
            content_id=spec.dataset_binding.content_id, timestamp_column=_TIMESTAMP_COLUMN,
        )

        if stage is ExecutionStage.LOADING_DATASET:
            stage = self._advance(experiment_id, ExecutionStage.BUILDING_SPLITS)

        try:
            # Three independent failure sources -- an unsafe preprocessing
            # binding, a dataset manifest whose label_definition cannot be
            # parsed, or the fold plan itself failing to build (e.g.
            # `split_binding.params` requesting more folds than the dataset
            # has rows for, raised by `PurgedWalkForwardSplitter` before
            # `validate_fold_plan` ever runs) -- must all fail the execution
            # identically. Letting any of them propagate unhandled would
            # leave the manifest stuck at BUILDING_SPLITS forever: not
            # terminal, so a later resume() would retry and hit the
            # IDENTICAL deterministic error every time.
            assert_preprocessing_is_safe_for_execution(dataset_manifest)
            label_horizon_bars = extract_label_horizon_bars(dataset_manifest)
            fold_plan = build_folds_from_split_binding(
                spec.split_binding, timeline[_TIMESTAMP_COLUMN], label_horizon_bars=label_horizon_bars,
            )
        except Exception as exc:
            self._fail_fold_plan_stage(experiment_id, f"Fold plan could not be built: {exc}")
            raise ExecutionResumeError(
                f"Fold plan could not be built for experiment_id={experiment_id!r}: {exc}",
                context={"experiment_id": experiment_id},
            ) from exc

        validation_report = validate_fold_plan(fold_plan, timeline=timeline, timestamp_column=_TIMESTAMP_COLUMN)
        if not validation_report.is_ready:
            blocking = [*validation_report.criticals, *validation_report.errors]
            failure_summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            self._fail_fold_plan_stage(experiment_id, failure_summary)
            raise ExecutionResumeError(  # fold-plan validation is fatal and deterministic -- never resumable
                f"Fold plan validation failed for experiment_id={experiment_id!r}: {failure_summary}",
                context={"experiment_id": experiment_id},
            )

        if stage is ExecutionStage.BUILDING_SPLITS:
            execution_manifest = self._execution_manifest_store.transition(
                experiment_id, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=format_utc_timestamp(utc_now()),
                fold_plan_strategy=fold_plan.strategy, total_folds=len(fold_plan.folds),
                declared_purge_bars=fold_plan.purge_bars,
                required_label_purge_bars=fold_plan.required_label_purge_bars,
                effective_purge_bars=fold_plan.purge_bars, embargo_bars=fold_plan.embargo_bars,
                label_horizon_source=LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST,
                split_policy=SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE,
            )
            stage = ExecutionStage.RUNNING_FOLD
        elif stage is ExecutionStage.RECOVERABLE_FAILURE:
            # Resuming after a transient failure: the fold plan is
            # rebuilt (deterministically identical, from the same
            # `split_binding` and reconstructed timeline) but
            # `fold_plan_strategy`/`total_folds` were already recorded
            # before the failure -- only the stage itself needs to move
            # forward, back into the fold loop.
            execution_manifest = self._execution_manifest_store.transition(
                experiment_id, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=format_utc_timestamp(utc_now()),
            )
            stage = ExecutionStage.RUNNING_FOLD

        resume_plan = build_resume_plan(
            execution_manifest, fold_plan, artifact_store=self._artifact_store, force_rerun_folds=force_rerun_folds,
        )

        completed_indices = set(resume_plan.verified_complete)
        failed_indices = set(execution_manifest.failed_fold_indices) - force_rerun_folds
        fold_result_refs: dict[int, ArtifactReference] = {
            k: v for k, v in execution_manifest.fold_result_references.items() if k in completed_indices
        }

        model_definition = self._model_registry.get(spec.model_name, spec.model_version)
        serializer = resolve_serializer(model_definition.serializer_id, registry=self._serializer_registry)
        feature_schema = FeatureSchema(feature_names=spec.feature_binding.feature_names)

        for position, fold in enumerate(resume_plan.remaining_folds):
            fold_result = self._run_one_fold(
                fold, experiment_id=experiment_id, spec=spec, timeline=timeline, feature_schema=feature_schema,
                model_factory=model_definition.factory, serializer=serializer,
            )
            ref = self._artifact_store.write_artifact(
                canonical_json_bytes(fold_result.to_json_dict()), category=ArtifactCategory.FOLD_RESULT,
            )
            fold_result_refs[fold.fold_index] = ref
            if fold_result.status is FoldStatus.COMPLETED:
                completed_indices.add(fold.fold_index)
                failed_indices.discard(fold.fold_index)
                self._event_store.append(experiment_id, EventType.FOLD_COMPLETED, details={"fold_index": fold.fold_index})
            else:
                failed_indices.add(fold.fold_index)
                self._event_store.append(
                    experiment_id, EventType.FOLD_FAILED,
                    details={"fold_index": fold.fold_index, "reason": (fold_result.failure_reason or "")[:200]},
                )
            execution_manifest = self._execution_manifest_store.transition(
                experiment_id, new_stage=ExecutionStage.STORING_RESULTS, updated_at=format_utc_timestamp(utc_now()),
                completed_fold_indices=tuple(sorted(completed_indices)), failed_fold_indices=tuple(sorted(failed_indices)),
                fold_result_references=fold_result_refs, current_fold_index=None,
            )
            is_last = position == len(resume_plan.remaining_folds) - 1
            if not is_last:
                execution_manifest = self._execution_manifest_store.transition(
                    experiment_id, new_stage=ExecutionStage.RUNNING_FOLD, updated_at=format_utc_timestamp(utc_now()),
                    current_fold_index=resume_plan.remaining_folds[position + 1].fold_index,
                )

        if execution_manifest.stage is ExecutionStage.RUNNING_FOLD:
            # Nothing needed to run this call (every fold already verified
            # complete) -- still pass through STORING_RESULTS explicitly,
            # since the legal-transition table only allows the final
            # terminal transition FROM there (see module docstring).
            execution_manifest = self._execution_manifest_store.transition(
                experiment_id, new_stage=ExecutionStage.STORING_RESULTS, updated_at=format_utc_timestamp(utc_now()),
            )

        overall_status = ExecutionStage.FAILED if failed_indices else ExecutionStage.COMPLETED
        completed_at = format_utc_timestamp(utc_now())
        execution_duration = time.perf_counter() - pipeline_started

        all_fold_results = self._load_all_fold_results(fold_result_refs)
        timeline_obj = Timeline.from_fold_results(experiment_id, all_fold_results)
        timeline_ref = self._artifact_store.write_artifact(
            canonical_json_bytes(timeline_obj.to_json_dict()), category=ArtifactCategory.TIMELINE,
        )

        # `aggregate`'s own `artifact_references` deliberately holds ONLY
        # the timeline ref, never a reference to the EXECUTION_SUMMARY
        # artifact about to be written FROM this very object -- content
        # addressing means that artifact's hash cannot be known until
        # AFTER its bytes (this object's own serialization) are fixed, so
        # it cannot reference itself. The manifest below (not content-
        # addressed, mutable-in-place) is where BOTH refs are recorded.
        aggregate = AggregatedExecutionResult(
            schema_version=1, experiment_id=experiment_id, total_folds=len(fold_plan.folds),
            completed_fold_indices=tuple(sorted(completed_indices)), failed_fold_indices=tuple(sorted(failed_indices)),
            overall_status=overall_status, started_at=started_at, completed_at=completed_at,
            execution_duration_seconds=execution_duration, artifact_references=(timeline_ref,),
            resume_count=execution_manifest.resume_count,
        )
        summary_ref = self._artifact_store.write_artifact(
            canonical_json_bytes(aggregate.to_json_dict()), category=ArtifactCategory.EXECUTION_SUMMARY,
        )

        self._execution_manifest_store.transition(
            experiment_id, new_stage=overall_status, updated_at=completed_at, completed_at=completed_at,
            failure_summary=(f"{len(failed_indices)} fold(s) failed: {sorted(failed_indices)}" if failed_indices else None),
            artifact_references=(timeline_ref, summary_ref),
        )
        self._transition_experiment_to_terminal(
            experiment_id, ExperimentStatus.FAILED if failed_indices else ExperimentStatus.COMPLETED,
        )
        self._event_store.append(
            experiment_id, EventType.RUN_FAILED if failed_indices else EventType.RUN_COMPLETED,
            details={"completed_folds": len(completed_indices), "failed_folds": len(failed_indices)},
        )
        return aggregate

    def _advance(self, experiment_id: str, new_stage: ExecutionStage) -> ExecutionStage:
        self._execution_manifest_store.transition(experiment_id, new_stage=new_stage, updated_at=format_utc_timestamp(utc_now()))
        return new_stage

    def _fail_fold_plan_stage(self, experiment_id: str, failure_summary: str) -> None:
        """Shared terminal-FAILED handling for BOTH ways building/
        validating a fold plan can fail (a raised exception from
        generation, or `validate_fold_plan` reporting issues) -- legal
        from either `BUILDING_SPLITS` (first attempt) or
        `RECOVERABLE_FAILURE` (a resume that hits the identical,
        deterministic failure again)."""
        now = format_utc_timestamp(utc_now())
        self._execution_manifest_store.transition(
            experiment_id, new_stage=ExecutionStage.FAILED, updated_at=now, completed_at=now,
            failure_summary=failure_summary,
        )
        self._transition_experiment_to_terminal(experiment_id, ExperimentStatus.FAILED)
        self._event_store.append(experiment_id, EventType.RUN_FAILED, details={"reason": "fold_plan_stage_failed"})

    def _transition_experiment_to_terminal(self, experiment_id: str, status: ExperimentStatus) -> None:
        now = format_utc_timestamp(utc_now())
        current = self._experiment_manifest_store.load(experiment_id)
        if current.status is status:
            return  # pragma: no cover - defensive; the caller's own terminal-stage check already prevents this
        if status is ExperimentStatus.FAILED:
            self._experiment_manifest_store.transition(
                experiment_id, new_status=status, completed_at=now,
                failure_summary="Execution engine reported one or more failed folds -- see execution manifest",
            )
        else:
            self._experiment_manifest_store.transition(experiment_id, new_status=status, completed_at=now)

    def _run_one_fold(
        self, fold: Fold, *, experiment_id: str, spec: ExperimentSpec, timeline: pd.DataFrame,
        feature_schema: FeatureSchema, model_factory: ModelFactory, serializer: ModelSerializer,
    ) -> FoldResult:
        self._event_store.append(experiment_id, EventType.FOLD_STARTED, details={"fold_index": fold.fold_index})
        started = format_utc_timestamp(utc_now())
        seed = spec.seed_configuration.derive(f"{SeedDomain.CROSS_VALIDATION.value}:{fold.fold_index}")

        context = FoldExecutionContext(
            experiment_id=experiment_id, fold_index=fold.fold_index,
            split_id=f"fold:{fold.fold_index}", dataset_content_id=spec.dataset_binding.content_id,
            manifest=self._experiment_manifest_store.load(experiment_id), seed=seed,
            environment=capture_environment_snapshot(),
            artifact_store=self._artifact_store, event_store=self._event_store,
            artifacts_root=self._artifacts_root, started_at=started,
        )

        feature_names = list(spec.feature_binding.feature_names)
        train_df = timeline.iloc[fold.train_indices]
        test_df = timeline.iloc[fold.test_indices]
        validation_df = timeline.iloc[fold.validation_indices] if len(fold.validation_indices) else None

        data = FoldData(
            train_features=train_df[feature_names], train_labels=train_df[_LABEL_COLUMN],
            test_features=test_df[feature_names], test_labels=test_df[_LABEL_COLUMN],
            validation_features=(None if validation_df is None else validation_df[feature_names]),
            validation_labels=(None if validation_df is None else validation_df[_LABEL_COLUMN]),
        )

        train_start_iso, train_end_iso = fold.train_start.isoformat(), fold.train_end.isoformat()
        test_start_iso, test_end_iso = fold.test_start.isoformat(), fold.test_end.isoformat()
        validation_start_iso = None if fold.validation_start is None else fold.validation_start.isoformat()
        validation_end_iso = None if fold.validation_end is None else fold.validation_end.isoformat()

        try:
            outcome = self._fold_executor.execute(
                context, model_factory=model_factory, hyperparameters=spec.hyperparameters,
                feature_schema=feature_schema, objective=spec.objective, serializer=serializer, data=data,
            )
        except ExperimentLockError:
            raise
        except Exception as exc:
            return FoldResult(
                schema_version=1, fold_index=fold.fold_index,
                train_start=train_start_iso, train_end=train_end_iso, test_start=test_start_iso, test_end=test_end_iso,
                train_size=len(fold.train_indices), test_size=len(fold.test_indices),
                validation_size=len(fold.validation_indices),
                validation_start=validation_start_iso, validation_end=validation_end_iso,
                status=FoldStatus.FAILED, duration_seconds=0.0, failure_reason=str(exc)[:2000],
            )

        return FoldResult(
            schema_version=1, fold_index=fold.fold_index,
            train_start=train_start_iso, train_end=train_end_iso, test_start=test_start_iso, test_end=test_end_iso,
            train_size=len(fold.train_indices), test_size=len(fold.test_indices),
            validation_size=len(fold.validation_indices),
            validation_start=validation_start_iso, validation_end=validation_end_iso,
            status=FoldStatus.COMPLETED, duration_seconds=outcome.duration_seconds,
            artifact_references=outcome.artifact_references, metrics=outcome.metrics,
        )

    def _load_all_fold_results(self, fold_result_refs: dict[int, ArtifactReference]) -> list[FoldResult]:
        results = []
        for fold_index, ref in fold_result_refs.items():
            if ref.category is not ArtifactCategory.FOLD_RESULT:
                raise ExecutionResumeError(
                    f"Fold {fold_index}'s recorded artifact reference has category={ref.category.value!r}, "
                    f"expected {ArtifactCategory.FOLD_RESULT.value!r} -- refusing to finalize this execution "
                    "from a reference that was never recorded as a fold result",
                    context={"fold_index": fold_index, "category": ref.category.value},
                )
            try:
                raw = self._artifact_store.read_artifact(ref.content_hash)
                decoded = FoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
            except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                raise ExecutionResumeError(
                    f"Fold {fold_index}'s recorded FOLD_RESULT artifact could not be read and decoded while "
                    f"finalizing this execution: {exc}",
                    context={"fold_index": fold_index},
                ) from exc
            if decoded.fold_index != fold_index:
                raise ExecutionResumeError(
                    f"Fold {fold_index}'s recorded FOLD_RESULT artifact decodes to fold_index="
                    f"{decoded.fold_index} -- a valid content hash proves the bytes are intact, not that "
                    "they were filed under the correct key",
                    context={"fold_index": fold_index, "decoded_fold_index": decoded.fold_index},
                )
            results.append(decoded)
        return results


__all__ = [
    "ExecutionOutcome",
    "ExecutionRunner",
    "assert_preprocessing_is_safe_for_execution",
    "extract_label_horizon_bars",
    "resolve_serializer",
]
