"""`OptimizationRunner`: the top-level orchestrator for one leakage-safe
feature-selection-and-hyperparameter-optimization run (Milestone 4D).
Mirrors `execution.runner.ExecutionRunner`'s exact pipeline shape --
staged `OptimizationStage` transitions, lock-guarded run/resume entry
points, idempotent no-op on an already-`COMPLETED` optimization, resume
via re-verified artifacts rather than trusting the manifest -- one level
deeper (outer fold containing an inner trial search, rather than a flat
fold loop).

ONE MANIFEST, NOT TWO
--------------------------------------------------------------------------
`execution.runner` layers `ExecutionManifest` (fine-grained) on top of
`ml.manifests.ExperimentManifest` (coarse, pre-existing from Milestone
4A) because Milestone 4A's `ExperimentStatus` already existed and had to
be preserved unchanged. There is no pre-existing coarse "optimization
status" enum this milestone must preserve -- `OptimizationManifest`'s own
`OptimizationStage` already serves both roles, so this module uses
exactly one manifest, a deliberate simplification from the two-manifest
precedent, documented here and in the delivery report.

RESUME NORMALIZATION: WHY EVERY MID-OUTER-FOLD STAGE ROUTES THROUGH
`RECOVERABLE_FAILURE` FIRST
--------------------------------------------------------------------------
A crash can leave `OptimizationManifest.stage` at any of `RUNNING_TRIAL`,
`BUILDING_INNER_PLAN`, `SELECTING_CANDIDATE`, `REFITTING_WINNER`, or
`EVALUATING_OUTER_TEST` -- all "mid one outer fold's processing" stages.
`_run_outer_fold` always starts by (re-)entering `BUILDING_INNER_PLAN`,
which is only a legal transition FROM `RUNNING_OUTER_FOLD`. Rather than
special-casing every one of those five stages individually, `_normalize_
for_resume` routes any of them through one `RECOVERABLE_FAILURE ->
RUNNING_OUTER_FOLD` hop before the outer-fold loop begins -- legal from
every one of those stages (see `optimization.models`' legal-transition
table and its own docstring for why `RECOVERABLE_FAILURE` is a safe,
non-leaky target even from the late `SELECTING_CANDIDATE`/`REFITTING_
WINNER`/`EVALUATING_OUTER_TEST` stages: everything from candidate
selection through outer-test evaluation is a pure, deterministic
function of already-fixed inputs, so redoing it from scratch after a
crash reproduces the identical result). The outer-fold loop itself then
re-verifies (via `optimization.resume`) exactly how much of the
in-progress outer fold's own trial search can be trusted.

WHY A FAILED OUTER FOLD (NO VALID WINNER) FAILS THE WHOLE OPTIMIZATION
--------------------------------------------------------------------------
`execution.runner` lets one fold's failure coexist with other folds'
successes (each fold is an independent walk-forward evaluation). An outer
fold's nested search having NO valid (`COMPLETED`) trial to select as a
winner is different: nested cross-validation's scientific claim is "the
selected feature set/hyperparameters generalize across ALL outer folds
this dataset was evaluated over" -- silently proceeding to `COMPLETED`
with one outer fold missing its outer-test evaluation entirely would
misrepresent that claim. This runner treats "no valid trial to select"
as a hard failure of the whole optimization (`OptimizationStage.FAILED`),
not a partial success.
"""

from __future__ import annotations

import importlib.metadata
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ExperimentLockError,
    OptimizationResumeError,
    SchemaVersionError,
)
from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.runner import (
    _SERIALIZER_REGISTRY,
    assert_preprocessing_is_safe_for_execution,
    extract_label_horizon_bars,
    resolve_serializer,
)
from quant_platform.execution.splitters import (
    Fold,
    build_folds_from_split_binding,
    reconstruct_dataset_timeline,
)
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.interfaces import ModelDeserializer, ModelFactory, ModelSerializer
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, EnvironmentSnapshot
from quant_platform.ml.persistence import (
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    utc_now,
)
from quant_platform.ml.registry import ModelRegistry
from quant_platform.optimization.candidates import (
    RankingTable,
    TrialResult,
    TrialSpec,
    TrialStatus,
    rank_trials,
)
from quant_platform.optimization.feature_selection import FeatureSelectionResult, FeatureUniverse
from quant_platform.optimization.inner_splits import build_inner_fold_plan, validate_nested_plan
from quant_platform.optimization.manifests import (
    OPTIMIZATION_MANIFEST_SCHEMA_VERSION,
    OptimizationEventStore,
    OptimizationEventType,
    OptimizationManifest,
    OptimizationManifestStore,
    trial_reference_key,
    trial_references_for_outer_fold,
)
from quant_platform.optimization.models import (
    OptimizationSpec,
    OptimizationStage,
    compute_optimization_identity,
    trial_seed,
)
from quant_platform.optimization.outer_fold import OuterFoldResult, finalize_outer_fold
from quant_platform.optimization.reporting import build_optimization_report_json
from quant_platform.optimization.resume import (
    build_trial_resume_plan,
    load_historical_trial_records,
    require_resumable,
    verify_completed_outer_folds,
)
from quant_platform.optimization.stability import (
    summarize_feature_stability,
    summarize_hyperparameter_stability,
)
from quant_platform.optimization.study import (
    ask_next_trial,
    create_study,
    evaluate_median_stopping,
    rebuild_study_from_history,
    tell_trial_outcome,
)
from quant_platform.optimization.trial_executor import run_trial

logger = logging.getLogger(__name__)

_TIMESTAMP_COLUMN = "open_time"
_LABEL_COLUMN = "label"
_OPTIMIZATION_RUN_LOCK_FILE_NAME = ".optimization_run.lock"
"""Distinct from `OptimizationManifestStore`'s own `.optimization.lock`
(held briefly, per manifest transition) -- this outer lock is held for
the entire run's duration, preventing two processes from running the
SAME optimization concurrently. Never shared with the manifest's own
lock file, for the identical reentrancy reason `execution.runner`
documents for its own `_EXECUTION_RUN_LOCK_FILE_NAME`."""

_MID_OUTER_FOLD_STAGES = frozenset({
    OptimizationStage.RUNNING_TRIAL, OptimizationStage.BUILDING_INNER_PLAN, OptimizationStage.SELECTING_CANDIDATE,
    OptimizationStage.REFITTING_WINNER, OptimizationStage.EVALUATING_OUTER_TEST,
})
_TRIAL_EVENT_BY_STATUS: dict[TrialStatus, OptimizationEventType] = {
    TrialStatus.COMPLETED: OptimizationEventType.TRIAL_COMPLETED, TrialStatus.FAILED: OptimizationEventType.TRIAL_FAILED,
    TrialStatus.INVALID: OptimizationEventType.TRIAL_INVALID, TrialStatus.PRUNED: OptimizationEventType.TRIAL_PRUNED,
}


@dataclass(frozen=True, slots=True)
class OptimizationOutcome:
    manifest: OptimizationManifest
    was_idempotent_no_op: bool


class OptimizationRunner:
    def __init__(
        self, *, ml_artifacts_root: Path | str, model_registry: ModelRegistry,
        research_manifest_store: ResearchManifestStore, research_dataset_store: ResearchDatasetStore,
        experiment_manifest_store: ExperimentManifestStore,
        additional_serializers: Mapping[str, tuple[ModelSerializer, ModelDeserializer]] | None = None,
    ) -> None:
        self._artifacts_root = Path(ml_artifacts_root).resolve()
        self._model_registry = model_registry
        self._research_manifest_store = research_manifest_store
        self._research_dataset_store = research_dataset_store
        self._experiment_manifest_store = experiment_manifest_store
        # Merged serializer lookup for THIS runner instance -- the built-in
        # test-only entry, plus whatever `additional_serializers` a caller
        # supplied (e.g. `ml.model_zoo.default_serializer_registry()` for
        # the real models) -- mirrors `execution.runner.ExecutionRunner`'s
        # identical merge exactly (see that class's own comment for why:
        # `additional_serializers` alone would silently drop the built-in
        # test-only model the moment a caller supplies ANY real-model
        # registry, since `resolve_serializer` never merges on its own).
        self._serializer_registry: dict[str, tuple[ModelSerializer, ModelDeserializer]] = {
            **_SERIALIZER_REGISTRY, **dict(additional_serializers or {}),
        }

        self._manifest_store = OptimizationManifestStore(self._artifacts_root)
        self._event_store = OptimizationEventStore(self._artifacts_root)
        self._artifact_store = MLArtifactStore(self._artifacts_root)

    @property
    def manifest_store(self) -> OptimizationManifestStore:
        return self._manifest_store

    @property
    def event_store(self) -> OptimizationEventStore:
        return self._event_store

    @property
    def artifact_store(self) -> MLArtifactStore:
        return self._artifact_store

    def _run_lock_path(self, optimization_id: str) -> Path:
        return self._artifacts_root / "optimizations" / optimization_id / _OPTIMIZATION_RUN_LOCK_FILE_NAME

    def run(self, spec: OptimizationSpec) -> OptimizationOutcome:
        identity = compute_optimization_identity(spec)
        with experiment_lock(self._run_lock_path(identity.optimization_id)):
            return self._run_locked(spec, identity.optimization_id, require_existing=False)

    def resume(self, optimization_id: str, *, spec: OptimizationSpec | None = None) -> OptimizationOutcome:
        existing = self._manifest_store.load_if_exists(optimization_id)
        require_resumable(existing, optimization_id=optimization_id)
        assert existing is not None  # require_resumable already raised OptimizationResumeError otherwise
        self._require_compatible_optuna_version(existing)
        resolved_spec = spec if spec is not None else self._load_spec_artifact(existing)
        with experiment_lock(self._run_lock_path(optimization_id)):
            return self._run_locked(resolved_spec, optimization_id, require_existing=True)

    def _require_compatible_optuna_version(self, manifest: OptimizationManifest) -> None:
        """Deterministic Optuna sampler resume (`optimization.study`'s
        ask()/tell() replay mechanism) is EMPIRICALLY verified against one
        installed patch version, not guaranteed by Optuna's public API
        contract across this platform's whole declared dependency range
        (see `pyproject.toml`'s own comment on the `optuna` entry). Resuming
        under a DIFFERENT installed Optuna version than the one that
        created this optimization could silently replay a diverged sampler
        trajectory -- fail closed instead, before a single trial is
        replayed."""
        reference = next((r for r in manifest.artifact_references if r.category is ArtifactCategory.ENVIRONMENT_SNAPSHOT), None)
        if reference is None:
            return  # created before this check existed -- nothing recorded to compare against, so nothing to fail closed on
        try:
            raw = self._artifact_store.read_artifact(reference.content_hash)
            original_snapshot = EnvironmentSnapshot.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except (ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError) as exc:
            raise OptimizationResumeError(
                f"Optimization {manifest.optimization_id!r}: recorded ENVIRONMENT_SNAPSHOT artifact could not be "
                f"read and decoded, so Optuna version compatibility cannot be verified -- refusing to resume: {exc}",
                context={"optimization_id": manifest.optimization_id},
            ) from exc
        original_optuna_version = original_snapshot.package_versions.get("optuna")
        current_optuna_version = importlib.metadata.version("optuna")
        if original_optuna_version is not None and original_optuna_version != current_optuna_version:
            raise OptimizationResumeError(
                f"Optimization {manifest.optimization_id!r} was created under optuna=={original_optuna_version}, "
                f"but the currently installed version is optuna=={current_optuna_version} -- deterministic sampler "
                "resume is only empirically verified for a matching installed version (see optimization.study's "
                "module docstring); refusing to resume with a potentially diverged sampler. Re-install the "
                "original optuna version to resume this optimization.",
                context={
                    "optimization_id": manifest.optimization_id, "original_optuna_version": original_optuna_version,
                    "current_optuna_version": current_optuna_version,
                },
            )

    def _load_spec_artifact(self, manifest: OptimizationManifest) -> OptimizationSpec:
        reference = next((r for r in manifest.artifact_references if r.category is ArtifactCategory.OPTIMIZATION_SPEC), None)
        if reference is None:
            raise OptimizationResumeError(f"Optimization {manifest.optimization_id!r} has no recorded OPTIMIZATION_SPEC artifact to resume from")
        raw = self._artifact_store.read_artifact(reference.content_hash)
        return OptimizationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))

    def _run_locked(self, spec: OptimizationSpec, optimization_id: str, *, require_existing: bool) -> OptimizationOutcome:
        manifest = self._manifest_store.load_if_exists(optimization_id)
        if require_existing and manifest is None:
            raise OptimizationResumeError(f"No optimization manifest exists for optimization_id={optimization_id!r}")  # pragma: no cover - guarded identically before locking

        if manifest is not None and manifest.stage in (OptimizationStage.COMPLETED, OptimizationStage.FAILED, OptimizationStage.CANCELLED):
            if manifest.stage is OptimizationStage.COMPLETED:
                return OptimizationOutcome(manifest=manifest, was_idempotent_no_op=True)
            raise OptimizationResumeError(
                f"Optimization {optimization_id!r} already reached terminal stage {manifest.stage.value!r}",
                context={"optimization_id": optimization_id, "stage": manifest.stage.value},
            )

        if manifest is None:
            now = format_utc_timestamp(utc_now())
            # The OPTIMIZATION_SPEC artifact reference is embedded directly
            # in the manifest's INITIAL construction (never via a later
            # `transition()` call) -- INITIALIZING has no legal self-loop
            # (see `optimization.models`' state machine), so there is no
            # transition that could record it without changing stage.
            spec_ref = self._artifact_store.write_artifact(canonical_json_bytes(spec.to_json_dict()), category=ArtifactCategory.OPTIMIZATION_SPEC)
            env_ref = self._artifact_store.write_artifact(
                canonical_json_bytes(capture_environment_snapshot().to_json_dict()), category=ArtifactCategory.ENVIRONMENT_SNAPSHOT,
            )
            manifest = OptimizationManifest(
                schema_version=OPTIMIZATION_MANIFEST_SCHEMA_VERSION, optimization_id=optimization_id,
                parent_experiment_id=spec.parent_experiment_id, stage=OptimizationStage.INITIALIZING,
                created_at=now, updated_at=now, artifact_references=(spec_ref, env_ref),
            )
            self._manifest_store.create(manifest)
            self._event_store.append(optimization_id, OptimizationEventType.OPTIMIZATION_CREATED)
            self._event_store.append(optimization_id, OptimizationEventType.RUN_STARTED)
        else:
            manifest = self._manifest_store.bump_resume_count(optimization_id)
            self._event_store.append(optimization_id, OptimizationEventType.OPTIMIZATION_RESUMED, details={"resume_count": manifest.resume_count})

        try:
            manifest = self._execute_pipeline(spec, manifest)
        except ExperimentLockError:
            self._manifest_store.transition(optimization_id, new_stage=OptimizationStage.RECOVERABLE_FAILURE, updated_at=format_utc_timestamp(utc_now()))
            raise
        return OptimizationOutcome(manifest=manifest, was_idempotent_no_op=False)

    def _advance(self, optimization_id: str, new_stage: OptimizationStage) -> OptimizationManifest:
        return self._manifest_store.transition(optimization_id, new_stage=new_stage, updated_at=format_utc_timestamp(utc_now()))

    def _execute_pipeline(self, spec: OptimizationSpec, manifest: OptimizationManifest) -> OptimizationManifest:
        optimization_id = manifest.optimization_id
        if manifest.stage is OptimizationStage.INITIALIZING:
            manifest = self._advance(optimization_id, OptimizationStage.LOADING_EXPERIMENT)

        experiment_manifest = self._experiment_manifest_store.load(spec.parent_experiment_id)
        dataset_manifest = self._research_manifest_store.load(spec.dataset_binding.dataset_id, spec.dataset_binding.manifest_version)
        assert_preprocessing_is_safe_for_execution(dataset_manifest)
        label_horizon_bars = extract_label_horizon_bars(dataset_manifest)
        timeline = reconstruct_dataset_timeline(
            self._research_dataset_store, dataset_id=spec.dataset_binding.dataset_id,
            content_id=spec.dataset_binding.content_id, timestamp_column=_TIMESTAMP_COLUMN,
        )

        if manifest.stage is OptimizationStage.LOADING_EXPERIMENT:
            manifest = self._advance(optimization_id, OptimizationStage.BUILDING_OUTER_PLAN)

        outer_plan = build_folds_from_split_binding(spec.outer_split_binding, timeline[_TIMESTAMP_COLUMN], label_horizon_bars=label_horizon_bars)
        validation_report = validate_fold_plan(outer_plan, timeline=timeline, timestamp_column=_TIMESTAMP_COLUMN)
        if not validation_report.is_ready:
            blocking = [*validation_report.criticals, *validation_report.errors]
            summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            self._fail(optimization_id, f"Outer fold plan validation failed: {summary}")
            raise OptimizationResumeError(f"Outer fold plan validation failed for optimization_id={optimization_id!r}: {summary}")

        if manifest.stage is OptimizationStage.BUILDING_OUTER_PLAN:
            manifest = self._manifest_store.transition(
                optimization_id, new_stage=OptimizationStage.RUNNING_OUTER_FOLD, updated_at=format_utc_timestamp(utc_now()),
                total_outer_folds=len(outer_plan.folds),
            )
        elif manifest.stage is OptimizationStage.RECOVERABLE_FAILURE:
            manifest = self._advance(optimization_id, OptimizationStage.RUNNING_OUTER_FOLD)
        elif manifest.stage in _MID_OUTER_FOLD_STAGES:
            # See module docstring: normalize any mid-outer-fold crash
            # point back to RUNNING_OUTER_FOLD via RECOVERABLE_FAILURE
            # before re-entering the outer-fold loop below.
            manifest = self._advance(optimization_id, OptimizationStage.RECOVERABLE_FAILURE)
            manifest = self._advance(optimization_id, OptimizationStage.RUNNING_OUTER_FOLD)

        model_definition = self._model_registry.get(spec.model_name, spec.model_version)
        serializer = resolve_serializer(model_definition.serializer_id, registry=self._serializer_registry)
        model_definition_fingerprint = model_definition.fingerprint()
        feature_universe = FeatureUniverse.from_experiment_spec(experiment_manifest.spec)

        verified_outer, _needs_rerun_outer = verify_completed_outer_folds(manifest, artifact_store=self._artifact_store)
        outer_fold_results: list[OuterFoldResult] = []
        ranking_tables: dict[int, RankingTable] = {}

        for outer_fold in outer_plan.folds:
            if outer_fold.fold_index in verified_outer:
                raw = self._artifact_store.read_artifact(manifest.outer_fold_result_references[outer_fold.fold_index].content_hash)
                outer_fold_results.append(OuterFoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8"))))
                continue

            manifest = self._manifest_store.transition(
                optimization_id, new_stage=OptimizationStage.RUNNING_OUTER_FOLD, updated_at=format_utc_timestamp(utc_now()),
                current_outer_fold_index=outer_fold.fold_index,
            )
            self._event_store.append(optimization_id, OptimizationEventType.OUTER_FOLD_STARTED, details={"outer_fold_index": outer_fold.fold_index})

            manifest, outer_result, ranking_table = self._run_outer_fold(
                spec, manifest, outer_fold=outer_fold, timeline=timeline, feature_universe=feature_universe,
                model_factory=model_definition.factory, model_definition_fingerprint=model_definition_fingerprint,
                serializer=serializer, label_horizon_bars=label_horizon_bars,
            )
            outer_fold_results.append(outer_result)
            ranking_tables[outer_fold.fold_index] = ranking_table

        if manifest.stage is OptimizationStage.RUNNING_OUTER_FOLD:
            # Nothing needed to run this call (every outer fold was
            # already verified-complete) -- still pass through
            # STORING_RESULTS explicitly, mirroring execution.runner's
            # identical fallback for the empty-remaining-work case.
            manifest = self._advance(optimization_id, OptimizationStage.STORING_RESULTS)

        feature_selection_results = [
            FeatureSelectionResult.from_json_dict(parse_json_strict(self._artifact_store.read_artifact(r.feature_selection_result_reference.content_hash).decode("utf-8")))
            for r in outer_fold_results
        ]
        winning_feature_sets = [r.final_selected_features for r in outer_fold_results]
        feature_stability = summarize_feature_stability(
            optimization_id=optimization_id, feature_selection_results=feature_selection_results, winning_feature_sets=winning_feature_sets,
        ) if feature_selection_results else None

        winning_scores: list[float] = []
        for r in outer_fold_results:
            table = ranking_tables.get(r.outer_fold_index)
            winner = table.winner if table is not None else None
            if winner is not None and winner.primary_metric_aggregate is not None:
                winning_scores.append(winner.primary_metric_aggregate)
        hyperparameter_stability = summarize_hyperparameter_stability(
            optimization_id=optimization_id, search_space=spec.search_space,
            winning_hyperparameters=[r.final_hyperparameters for r in outer_fold_results],
            winning_primary_metric_values=winning_scores,
        ) if outer_fold_results else None

        summary_refs = []
        if feature_stability is not None:
            summary_refs.append(self._artifact_store.write_artifact(canonical_json_bytes(feature_stability.to_json_dict()), category=ArtifactCategory.FEATURE_STABILITY))
        if hyperparameter_stability is not None:
            summary_refs.append(self._artifact_store.write_artifact(canonical_json_bytes(hyperparameter_stability.to_json_dict()), category=ArtifactCategory.HYPERPARAMETER_STABILITY))

        report_json = build_optimization_report_json(
            manifest, spec=spec, outer_fold_results=outer_fold_results, ranking_tables=ranking_tables,
            feature_stability=feature_stability, hyperparameter_stability=hyperparameter_stability,
        )
        report_ref = self._artifact_store.write_artifact(canonical_json_bytes(report_json), category=ArtifactCategory.OPTIMIZATION_REPORT)
        summary_refs.append(report_ref)

        completed_at = format_utc_timestamp(utc_now())
        manifest = self._manifest_store.transition(
            optimization_id, new_stage=OptimizationStage.COMPLETED, updated_at=completed_at, completed_at=completed_at,
            summary_references=tuple(summary_refs),
        )
        self._event_store.append(optimization_id, OptimizationEventType.RUN_COMPLETED)
        return manifest

    def _fail(self, optimization_id: str, failure_summary: str) -> None:
        now = format_utc_timestamp(utc_now())
        self._manifest_store.transition(optimization_id, new_stage=OptimizationStage.FAILED, updated_at=now, completed_at=now, failure_summary=failure_summary)
        self._event_store.append(optimization_id, OptimizationEventType.RUN_FAILED, details={"reason": failure_summary[:200]})

    def _run_outer_fold(
        self, spec: OptimizationSpec, manifest: OptimizationManifest, *, outer_fold: Fold, timeline: pd.DataFrame,
        feature_universe: FeatureUniverse, model_factory: ModelFactory, model_definition_fingerprint: str,
        serializer: ModelSerializer, label_horizon_bars: int,
    ) -> tuple[OptimizationManifest, OuterFoldResult, RankingTable]:
        optimization_id = manifest.optimization_id
        manifest = self._advance(optimization_id, OptimizationStage.BUILDING_INNER_PLAN)

        inner_plan = build_inner_fold_plan(
            outer_fold, config=spec.inner_split_config, label_horizon_bars=label_horizon_bars, timeline=timeline, timestamp_column=_TIMESTAMP_COLUMN,
        )
        nested_validation = validate_nested_plan(outer_fold, inner_plan)
        if not nested_validation.is_ready:
            blocking = [*nested_validation.criticals, *nested_validation.errors]
            summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            self._fail(optimization_id, f"Outer fold {outer_fold.fold_index}: nested plan validation failed: {summary}")
            raise OptimizationResumeError(f"Nested plan validation failed for outer fold {outer_fold.fold_index}: {summary}")
        inner_plan_fingerprint = fingerprint_json(inner_plan.to_json_dict())

        manifest = self._advance(optimization_id, OptimizationStage.RUNNING_TRIAL)

        claimed_refs = trial_references_for_outer_fold(manifest, outer_fold.fold_index)
        resume_plan = build_trial_resume_plan(claimed_refs, optimization_id=optimization_id, outer_fold_index=outer_fold.fold_index, artifact_store=self._artifact_store)
        history = load_historical_trial_records(resume_plan, claimed_trial_result_references=claimed_refs, artifact_store=self._artifact_store)

        study = rebuild_study_from_history(spec, history) if history else create_study(spec)
        all_trial_results: list[TrialResult] = [
            TrialResult.from_json_dict(parse_json_strict(self._artifact_store.read_artifact(claimed_refs[record.trial_number].content_hash).decode("utf-8")))
            for record in history
        ]
        running_metrics = {r.trial_number: r.inner_fold_metrics for r in all_trial_results}

        outer_fold_started = time.perf_counter()
        failed_or_invalid_count = sum(1 for r in all_trial_results if r.status in (TrialStatus.FAILED, TrialStatus.INVALID))

        trial_number = resume_plan.next_trial_number
        while trial_number < spec.max_trials:
            if spec.timeout_seconds is not None and (time.perf_counter() - outer_fold_started) > spec.timeout_seconds:
                logger.info("Optimization %s outer fold %d: timeout reached, stopping trial loop at trial %d", optimization_id[:12], outer_fold.fold_index, trial_number)
                break
            if spec.max_failed_trials is not None and failed_or_invalid_count > spec.max_failed_trials:
                logger.info("Optimization %s outer fold %d: max_failed_trials exceeded, stopping trial loop at trial %d", optimization_id[:12], outer_fold.fold_index, trial_number)
                break

            trial, sampled_values = ask_next_trial(study, spec.search_space)
            this_trial_seed = trial_seed(spec.seed_configuration, outer_fold.fold_index, trial_number)
            trial_spec = TrialSpec(
                schema_version=1, optimization_id=optimization_id, outer_fold_index=outer_fold.fold_index, trial_number=trial_number,
                sampled_hyperparameters=sampled_values, feature_selection_spec=spec.feature_selection_spec, trial_seed=this_trial_seed,
                inner_split_plan_fingerprint=inner_plan_fingerprint, model_definition_fingerprint=model_definition_fingerprint,
                objective=spec.objective, primary_metric=spec.primary_metric,
            )
            self._event_store.append(optimization_id, OptimizationEventType.TRIAL_STARTED, details={"outer_fold_index": outer_fold.fold_index, "trial_number": trial_number})

            def _pruning_callback(metrics_so_far: tuple, _running_metrics: dict = running_metrics) -> bool:  # type: ignore[type-arg]
                return evaluate_median_stopping(
                    spec.pruning_config, primary_metric=spec.primary_metric,
                    other_trials_metrics=list(_running_metrics.values()), current_trial_metrics=metrics_so_far,
                )

            trial_result = run_trial(
                trial_spec, inner_fold_plan=inner_plan, timeline=timeline, feature_universe=feature_universe,
                model_name=spec.model_name, model_factory=model_factory, seed_configuration=spec.seed_configuration,
                min_successful_inner_folds=spec.min_successful_inner_folds, early_stopping_config=spec.early_stopping_config,
                artifact_store=self._artifact_store, label_column=_LABEL_COLUMN,
                pruning_callback=_pruning_callback,
            )

            trial_ref = self._artifact_store.write_artifact(canonical_json_bytes(trial_result.to_json_dict()), category=ArtifactCategory.TRIAL_RESULT)
            claimed_refs[trial_number] = trial_ref
            all_trial_results.append(trial_result)
            running_metrics[trial_number] = trial_result.inner_fold_metrics

            tell_trial_outcome(study, trial, status=trial_result.status, value=trial_result.primary_metric_aggregate if trial_result.status is TrialStatus.COMPLETED else None)

            total_completed, total_failed = manifest.total_trials_completed, manifest.total_trials_failed
            total_invalid, total_pruned = manifest.total_trials_invalid, manifest.total_trials_pruned
            if trial_result.status is TrialStatus.COMPLETED:
                total_completed += 1
            elif trial_result.status is TrialStatus.FAILED:
                total_failed += 1
                failed_or_invalid_count += 1
            elif trial_result.status is TrialStatus.INVALID:
                total_invalid += 1
                failed_or_invalid_count += 1
            else:
                total_pruned += 1

            updated_refs = dict(manifest.trial_result_references)
            updated_refs[trial_reference_key(outer_fold.fold_index, trial_number)] = trial_ref
            manifest = self._manifest_store.transition(
                optimization_id, new_stage=OptimizationStage.RUNNING_TRIAL, updated_at=format_utc_timestamp(utc_now()),
                trial_result_references=updated_refs, total_trials_completed=total_completed, total_trials_failed=total_failed,
                total_trials_invalid=total_invalid, total_trials_pruned=total_pruned,
            )
            self._event_store.append(
                optimization_id, _TRIAL_EVENT_BY_STATUS[trial_result.status],
                details={"outer_fold_index": outer_fold.fold_index, "trial_number": trial_number},
            )
            trial_number += 1

        manifest = self._advance(optimization_id, OptimizationStage.SELECTING_CANDIDATE)
        if not all_trial_results:
            self._fail(optimization_id, f"Outer fold {outer_fold.fold_index}: no trials were run")
            raise OptimizationResumeError(f"Outer fold {outer_fold.fold_index}: no trials were run")
        ranking_table = rank_trials(all_trial_results, primary_metric=spec.primary_metric)
        winner_entry = ranking_table.winner
        if winner_entry is None:
            self._fail(optimization_id, f"Outer fold {outer_fold.fold_index}: no valid (COMPLETED) trial to select as a winner")
            raise OptimizationResumeError(f"Outer fold {outer_fold.fold_index}: no valid trial to select as a winner")
        winning_trial = next(t for t in all_trial_results if t.trial_number == winner_entry.trial_number)
        search_summary_ref = self._artifact_store.write_artifact(canonical_json_bytes(ranking_table.to_json_dict()), category=ArtifactCategory.SEARCH_SUMMARY)
        self._event_store.append(optimization_id, OptimizationEventType.CANDIDATE_SELECTED, details={"outer_fold_index": outer_fold.fold_index, "trial_number": winning_trial.trial_number})

        manifest = self._advance(optimization_id, OptimizationStage.REFITTING_WINNER)
        manifest = self._advance(optimization_id, OptimizationStage.EVALUATING_OUTER_TEST)
        outer_result = finalize_outer_fold(
            optimization_spec=spec, outer_fold=outer_fold, winning_trial=winning_trial, timeline=timeline,
            feature_universe=feature_universe, model_factory=model_factory, serializer=serializer,
            artifact_store=self._artifact_store, label_column=_LABEL_COLUMN,
            search_summary_reference=search_summary_ref,
        )
        outer_result_ref = self._artifact_store.write_artifact(canonical_json_bytes(outer_result.to_json_dict()), category=ArtifactCategory.OUTER_FOLD_SELECTION)
        self._event_store.append(optimization_id, OptimizationEventType.OUTER_FOLD_FINALIZED, details={"outer_fold_index": outer_fold.fold_index})

        manifest = self._manifest_store.transition(
            optimization_id, new_stage=OptimizationStage.STORING_RESULTS, updated_at=format_utc_timestamp(utc_now()),
            completed_outer_fold_indices=tuple(sorted({*manifest.completed_outer_fold_indices, outer_fold.fold_index})),
            outer_fold_result_references={**manifest.outer_fold_result_references, outer_fold.fold_index: outer_result_ref},
            winning_trial_by_outer_fold={**manifest.winning_trial_by_outer_fold, outer_fold.fold_index: winning_trial.trial_number},
        )
        return manifest, outer_result, ranking_table


__all__ = ["OptimizationOutcome", "OptimizationRunner"]
