"""`CalibrationRunner`: the top-level orchestrator for one leakage-safe
calibration run (Milestone 4E). Mirrors `optimization.runner.
OptimizationRunner`'s pipeline shape -- staged `CalibrationStage`
transitions, lock-guarded run/resume entry points, idempotent no-op on an
already-`COMPLETED` calibration, resume via re-verified artifacts rather
than trusting the manifest -- one dimension simpler: calibration has no
inner TRIAL SEARCH loop (`fitting.select_calibrator` evaluates a small,
fixed candidate set in one deterministic pass), so there is no
`optimization`-style `RECOVERABLE_FAILURE`/pruning/study-replay machinery
here at all.

THE STRUCTURAL OUTER-TEST ISOLATION ARGUMENT
--------------------------------------------------------------------------
`run_outer_fold_calibration` is this package's ONE function that is
ALLOWED to read `outer_fold.test_indices`. Walk its body top to bottom:

  1. `generate_inner_oof_predictions` is called with `outer_fold` -- but
     that function (see `calibration.fitting`'s own docstring) has no
     code path that reads `.test_indices`/`.validation_indices` at all.
  2. `fit_decision_policy` is called with ONLY the `InnerOofPredictionSet`
     step 1 returned -- structurally, no outer-test bytes are in scope.
  3. The calibrator/threshold/reliability policy this produces is bound
     to a local `policy` variable and never mutated again ("frozen").
  4. ONLY NOW does this function read `outer_fold.train_indices` (to
     refit the final base model) and, separately, `outer_fold.
     test_indices` (to obtain raw features for prediction ONLY --
     `timeline.iloc[outer_fold.test_indices][feature_names]`, never the
     label column at this point).
  5. The refit model's raw predictions are transformed through `policy`
     (already frozen in step 3, before test features were ever touched).
  6. `timeline.iloc[outer_fold.test_indices][label_column]` -- the FIRST
     and ONLY read of outer-test LABELS in this entire function -- is
     read here, strictly after every calibration/threshold/confidence/
     uncertainty/abstention decision already exists, used exclusively to
     compute final evaluation metrics that influence nothing upstream.

This ordering is enforced by DATA FLOW, not a comment: `policy` (step 3)
has no reference to `outer_fold` at all (`fit_decision_policy`'s
signature takes an `InnerOofPredictionSet`, not a `Fold`), so there is no
Python expression through which information could flow backward from
step 6 into step 3 even by accident.
"""

from __future__ import annotations

import importlib.metadata
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.calibration.abstention import decide as decide_abstention
from quant_platform.calibration.abstention import evaluate_selective_prediction
from quant_platform.calibration.confidence import (
    ConfidenceResult,
    compute_confidence,
    distance_from_threshold_component,
    probability_extremity_component,
)
from quant_platform.calibration.diagnostics import ReliabilityBin, ReliabilityReport
from quant_platform.calibration.fitting import (
    FrozenDecisionPolicy,
    fit_decision_policy,
    generate_inner_oof_predictions,
)
from quant_platform.calibration.manifests import (
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    CalibrationEventStore,
    CalibrationEventType,
    CalibrationManifest,
    CalibrationManifestStore,
)
from quant_platform.calibration.models import CalibrationStage, Decision, DeterminismPolicy
from quant_platform.calibration.specs import CalibrationSpec, calibration_outer_refit_seed
from quant_platform.calibration.thresholds import apply_threshold
from quant_platform.calibration.uncertainty import (
    UncertaintyResult,
    bin_support_uncertainty_component,
    calibrator_disagreement_component,
    compute_uncertainty,
    entropy_component,
    margin_component,
)
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    CalibrationDataError,
    CalibrationResumeError,
    ExperimentLockError,
    QuantPlatformError,
    SchemaVersionError,
)
from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.runner import (
    _SERIALIZER_REGISTRY,
    assert_preprocessing_is_safe_for_execution,
    extract_label_horizon_bars,
    resolve_serializer,
)
from quant_platform.execution.runner import (
    _TIMESTAMP_COLUMN as _EXECUTION_TIMESTAMP_COLUMN,
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
from quant_platform.ml.interfaces import (
    FeatureSchema,
    ModelDeserializer,
    ModelFactory,
    ModelSerializer,
    ProbabilisticPredictor,
)
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.metrics import compute_metrics
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    EnvironmentSnapshot,
    JsonPrimitive,
    ModelHyperparameters,
    ObjectiveType,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    require_schema_version,
    utc_now,
)
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.manifests import OptimizationManifestStore
from quant_platform.optimization.models import OptimizationStage
from quant_platform.optimization.outer_fold import OuterFoldResult

logger = logging.getLogger(__name__)

_TIMESTAMP_COLUMN = _EXECUTION_TIMESTAMP_COLUMN
_LABEL_COLUMN = "label"
_CALIBRATION_RUN_LOCK_FILE_NAME = ".calibration_run.lock"
"""Distinct from `CalibrationManifestStore`'s own `.calibration.lock`
(held briefly, per manifest transition) -- this outer lock is held for
the entire run's duration, preventing two processes from running the
SAME calibration concurrently, exactly `optimization.runner`'s identical
`_OPTIMIZATION_RUN_LOCK_FILE_NAME` reentrancy argument."""
OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OuterFoldCalibrationResult:
    """One outer fold's complete, persisted calibration result (Section
    19). Row-level outer-test predictions are stored INLINE (plain JSON
    arrays, parallel to `sample_positions`) rather than as a separate
    artifact -- consistent with `RawPredictionSet`'s own precedent of
    storing probability arrays as plain JSON tuples, appropriate at this
    milestone's bounded/research data scale (see module docstring of
    `calibration.models` for the platform's existing position that a
    safe tabular format is used only where JSON already does not
    suffice)."""

    schema_version: int
    calibration_id: str
    outer_fold_index: int
    inner_oof_reference: ArtifactReference
    calibrator_selection_reference: ArtifactReference
    threshold_report_reference: ArtifactReference
    decision_policy_reference: ArtifactReference
    model_reference: ArtifactReference
    seed: int
    training_duration_seconds: float
    outer_train_row_count: int
    outer_test_row_count: int
    sample_positions: tuple[int, ...]
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...]
    decisions: tuple[str, ...]
    abstention_reason_codes: tuple[str, ...]
    confidence_scores: tuple[float, ...]
    confidence_categories: tuple[str, ...]
    uncertainty_scores: tuple[float, ...]
    classification_metrics: Mapping[str, JsonPrimitive]
    calibration_metrics_on_outer_test: Mapping[str, JsonPrimitive]
    selective_prediction_summary: Mapping[str, JsonPrimitive]
    evaluated_at: str

    def __post_init__(self) -> None:
        n = len(self.sample_positions)
        for name, arr in (
            ("raw_probabilities", self.raw_probabilities), ("calibrated_probabilities", self.calibrated_probabilities),
            ("decisions", self.decisions), ("abstention_reason_codes", self.abstention_reason_codes),
            ("confidence_scores", self.confidence_scores), ("confidence_categories", self.confidence_categories),
            ("uncertainty_scores", self.uncertainty_scores),
        ):
            if len(arr) != n:
                raise CalibrationDataError(f"OuterFoldCalibrationResult.{name} has length {len(arr)}, expected {n}")
        if self.outer_fold_index < 0:
            raise CalibrationDataError(f"OuterFoldCalibrationResult.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        for name, mapping in (
            ("classification_metrics", self.classification_metrics), ("calibration_metrics_on_outer_test", self.calibration_metrics_on_outer_test),
            ("selective_prediction_summary", self.selective_prediction_summary),
        ):
            try:
                validate_json_primitive_mapping(mapping, field_name=f"OuterFoldCalibrationResult.{name}")
            except ValueError as exc:
                raise CalibrationDataError(str(exc)) from exc

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "calibration_id": self.calibration_id, "outer_fold_index": self.outer_fold_index,
            "inner_oof_reference": self.inner_oof_reference.to_json_dict(),
            "calibrator_selection_reference": self.calibrator_selection_reference.to_json_dict(),
            "threshold_report_reference": self.threshold_report_reference.to_json_dict(),
            "decision_policy_reference": self.decision_policy_reference.to_json_dict(),
            "model_reference": self.model_reference.to_json_dict(), "seed": self.seed,
            "training_duration_seconds": self.training_duration_seconds, "outer_train_row_count": self.outer_train_row_count,
            "outer_test_row_count": self.outer_test_row_count, "sample_positions": list(self.sample_positions),
            "raw_probabilities": list(self.raw_probabilities), "calibrated_probabilities": list(self.calibrated_probabilities),
            "decisions": list(self.decisions), "abstention_reason_codes": list(self.abstention_reason_codes),
            "confidence_scores": list(self.confidence_scores), "confidence_categories": list(self.confidence_categories),
            "uncertainty_scores": list(self.uncertainty_scores),
            "classification_metrics": dict(sorted(self.classification_metrics.items())),
            "calibration_metrics_on_outer_test": dict(sorted(self.calibration_metrics_on_outer_test.items())),
            "selective_prediction_summary": dict(sorted(self.selective_prediction_summary.items())),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OuterFoldCalibrationResult:
        require_schema_version(raw, supported=OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION, context="OuterFoldCalibrationResult")

        def _ref(key: str) -> ArtifactReference:
            return ArtifactReference.from_json_dict(as_json_dict(raw[key], field_name=key))

        def _metrics(key: str) -> dict[str, JsonPrimitive]:
            return dict(as_json_dict(raw.get(key) or {}, field_name=key))

        return cls(
            schema_version=OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION, calibration_id=str(raw["calibration_id"]),
            outer_fold_index=int(str(raw["outer_fold_index"])), inner_oof_reference=_ref("inner_oof_reference"),
            calibrator_selection_reference=_ref("calibrator_selection_reference"), threshold_report_reference=_ref("threshold_report_reference"),
            decision_policy_reference=_ref("decision_policy_reference"), model_reference=_ref("model_reference"),
            seed=int(str(raw["seed"])), training_duration_seconds=float(str(raw["training_duration_seconds"])),
            outer_train_row_count=int(str(raw["outer_train_row_count"])), outer_test_row_count=int(str(raw["outer_test_row_count"])),
            sample_positions=tuple(int(v) for v in as_json_list(raw["sample_positions"], field_name="sample_positions")),
            raw_probabilities=tuple(float(v) for v in as_json_list(raw["raw_probabilities"], field_name="raw_probabilities")),
            calibrated_probabilities=tuple(float(v) for v in as_json_list(raw["calibrated_probabilities"], field_name="calibrated_probabilities")),
            decisions=tuple(str(v) for v in as_json_list(raw["decisions"], field_name="decisions")),
            abstention_reason_codes=tuple(str(v) for v in as_json_list(raw["abstention_reason_codes"], field_name="abstention_reason_codes")),
            confidence_scores=tuple(float(v) for v in as_json_list(raw["confidence_scores"], field_name="confidence_scores")),
            confidence_categories=tuple(str(v) for v in as_json_list(raw["confidence_categories"], field_name="confidence_categories")),
            uncertainty_scores=tuple(float(v) for v in as_json_list(raw["uncertainty_scores"], field_name="uncertainty_scores")),
            classification_metrics=_metrics("classification_metrics"), calibration_metrics_on_outer_test=_metrics("calibration_metrics_on_outer_test"),
            selective_prediction_summary=_metrics("selective_prediction_summary"), evaluated_at=str(raw["evaluated_at"]),
        )


def _find_reliability_bin(report: ReliabilityReport, probability: float) -> ReliabilityBin | None:
    for b in report.bins:
        upper_inclusive = b.bin_index == report.actual_n_bins - 1
        if b.lower_bound <= probability < b.upper_bound or (upper_inclusive and probability == b.upper_bound):
            return b
    return None  # pragma: no cover - defensive: bins partition [0, 1] by construction


def _confidence_and_uncertainty_for_row(
    probability: float, threshold: float, *, policy: FrozenDecisionPolicy, spec: CalibrationSpec,
) -> tuple[ConfidenceResult, UncertaintyResult]:
    minimum_support = spec.bin_support_minimum_samples
    reliability_bin = _find_reliability_bin(policy.reliability_reports[0], probability)
    bin_support = (
        None if reliability_bin is None
        else max(0.0, min(1.0, reliability_bin.sample_count / minimum_support))
    )
    bin_support_uncertainty = (
        None if reliability_bin is None
        else bin_support_uncertainty_component(reliability_bin.sample_count, minimum_support=minimum_support)
    )

    candidate_probabilities = [
        float(c.fitted.transform(np.asarray([probability]))[0])
        for c in policy.calibrator_selection.candidates if c.succeeded and c.fitted is not None
    ]
    disagreement = (
        calibrator_disagreement_component(candidate_probabilities) if len(candidate_probabilities) >= 2 else None
    )

    confidence_components: dict[str, float | None] = {
        "distance_from_threshold": distance_from_threshold_component(probability, threshold),
        "probability_extremity": probability_extremity_component(probability),
        "calibration_bin_support": bin_support,
    }
    confidence = compute_confidence(confidence_components, spec=spec.confidence_spec)

    uncertainty_components: dict[str, float | None] = {
        "entropy": entropy_component(probability), "margin": margin_component(probability, threshold),
        "model_disagreement": None,  # structurally unavailable: inner-fold models are transient, never persisted (see module docstring / delivery report)
        "calibrator_disagreement": disagreement, "bin_support": bin_support_uncertainty,
    }
    uncertainty = compute_uncertainty(
        {name: uncertainty_components[name] for name in spec.uncertainty_spec.components}, spec=spec.uncertainty_spec,
    )
    return confidence, uncertainty


def run_outer_fold_calibration(
    *, spec: CalibrationSpec, calibration_id: str, outer_fold: Fold, timeline: pd.DataFrame,
    feature_names: Sequence[str], label_column: str, label_horizon_bars: int,
    model_factory: ModelFactory, hyperparameters: ModelHyperparameters, objective: ObjectiveType,
    seed_configuration: SeedConfiguration, source_model_identity: str, source_experiment_id: str,
    artifact_store: MLArtifactStore, serializer: ModelSerializer,
) -> OuterFoldCalibrationResult:
    """Section 18's steps 2-15 for ONE outer fold. See module docstring
    for the structural (not conventional) outer-test isolation argument.
    Step 1 (load/verify source artifacts) and step 16 (independent
    verification) are the CALLER's responsibility (`CalibrationRunner`/
    `calibration.verification` respectively) -- exactly how
    `optimization.outer_fold.finalize_outer_fold` is scoped relative to
    `optimization.runner`."""
    oof = generate_inner_oof_predictions(
        outer_fold=outer_fold, timeline=timeline, feature_names=feature_names, label_column=label_column,
        label_horizon_bars=label_horizon_bars, model_factory=model_factory, hyperparameters=hyperparameters,
        objective=objective, seed_configuration=seed_configuration, spec=spec,
        source_model_identity=source_model_identity, source_experiment_id=source_experiment_id,
    )
    inner_oof_ref = artifact_store.write_artifact(canonical_json_bytes(oof.to_json_dict()), category=ArtifactCategory.INNER_OOF_PREDICTIONS)

    policy = fit_decision_policy(oof, spec=spec)
    calibrator_selection_ref = artifact_store.write_artifact(
        canonical_json_bytes(policy.calibrator_selection.to_json_dict()), category=ArtifactCategory.CALIBRATOR_CANDIDATE_REPORT,
    )
    threshold_report_ref = artifact_store.write_artifact(
        canonical_json_bytes(policy.threshold_report.to_json_dict()), category=ArtifactCategory.THRESHOLD_REPORT,
    )
    # "Freeze all post-processing" (Section 18 step 7): `policy` is bound
    # once here and never reassigned below -- everything from this point
    # forward reads FROM `policy`, nothing writes back into it.
    decision_policy_ref = artifact_store.write_artifact(canonical_json_bytes(policy.to_json_dict()), category=ArtifactCategory.DECISION_POLICY)

    refit_seed = calibration_outer_refit_seed(seed_configuration, outer_fold_index=outer_fold.fold_index)
    feature_schema = FeatureSchema(feature_names=tuple(feature_names))
    outer_train_df = timeline.iloc[outer_fold.train_indices]
    started = time.perf_counter()
    model = model_factory.create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
    fitted = model.fit(outer_train_df[list(feature_names)], outer_train_df[label_column], seeds=SeedConfiguration(master_seed=refit_seed))
    training_duration = time.perf_counter() - started
    model_ref = artifact_store.write_artifact(serializer.serialize(fitted), category=ArtifactCategory.MODEL)

    if not (fitted.metadata.capabilities.supports_predict_proba and isinstance(fitted, ProbabilisticPredictor)):
        raise CalibrationDataError(f"run_outer_fold_calibration: model {source_model_identity!r} does not support predict_proba")
    outer_test_features = timeline.iloc[outer_fold.test_indices][list(feature_names)]
    proba = fitted.predict_proba(outer_test_features)
    class_labels = list(fitted.class_labels)
    positive_index = class_labels.index(1)
    raw_probabilities = proba[:, positive_index]

    calibrated = policy.selected_calibrator().transform(raw_probabilities)
    threshold = policy.threshold_report.selected_threshold
    positive_mask = apply_threshold(calibrated, threshold)

    decisions: list[Decision] = []
    reason_codes: list[str] = []
    confidence_scores: list[float] = []
    confidence_categories: list[str] = []
    uncertainty_scores: list[float] = []
    for p in calibrated:
        confidence, uncertainty = _confidence_and_uncertainty_for_row(float(p), threshold, policy=policy, spec=spec)
        decision, reason = decide_abstention(
            float(p), threshold, spec=spec.abstention_spec, confidence=confidence.score, uncertainty=uncertainty.total_uncertainty,
        )
        decisions.append(decision)
        reason_codes.append(reason.value)
        confidence_scores.append(confidence.score)
        confidence_categories.append(confidence.category)
        uncertainty_scores.append(uncertainty.total_uncertainty)

    # THE FIRST AND ONLY READ OF OUTER-TEST LABELS IN THIS FUNCTION.
    # Every decision above (calibration, threshold, confidence,
    # uncertainty, abstention) is already final -- nothing beyond this
    # point can influence any of them.
    y_true = timeline.iloc[outer_fold.test_indices][label_column].to_numpy(dtype="float64")
    hard_predictions = positive_mask.astype("float64")
    classification_report = compute_metrics(objective, y_true, hard_predictions, calibrated)
    # `ProbabilityClippingPolicy.apply` is the ONE authoritative
    # scalar clipping transform (mirrors `apply_threshold`'s identical
    # single-authority role) -- applied element-wise here rather than
    # reimplementing the clip bounds inline.
    clipped = np.asarray([spec.probability_clipping.apply(float(p)) for p in calibrated], dtype="float64")
    from quant_platform.calibration.metrics import compute_calibration_metrics

    calibration_metrics_report = compute_calibration_metrics(clipped, y_true, binning_spec=spec.reliability_binning_specs[0])
    selective = evaluate_selective_prediction(decisions, [float(v) for v in y_true])

    result = OuterFoldCalibrationResult(
        schema_version=OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION, calibration_id=calibration_id, outer_fold_index=outer_fold.fold_index,
        inner_oof_reference=inner_oof_ref, calibrator_selection_reference=calibrator_selection_ref,
        threshold_report_reference=threshold_report_ref, decision_policy_reference=decision_policy_ref, model_reference=model_ref,
        seed=refit_seed, training_duration_seconds=training_duration, outer_train_row_count=len(outer_fold.train_indices),
        outer_test_row_count=len(outer_fold.test_indices), sample_positions=tuple(int(v) for v in outer_fold.test_indices.tolist()),
        raw_probabilities=tuple(float(v) for v in raw_probabilities), calibrated_probabilities=tuple(float(v) for v in calibrated),
        decisions=tuple(d.value for d in decisions), abstention_reason_codes=tuple(reason_codes),
        confidence_scores=tuple(confidence_scores), confidence_categories=tuple(confidence_categories),
        uncertainty_scores=tuple(uncertainty_scores), classification_metrics=dict(classification_report.values),
        calibration_metrics_on_outer_test=dict(calibration_metrics_report.values),
        selective_prediction_summary={
            "coverage": selective.coverage, "abstention_rate": selective.abstention_rate,
            "n_accepted": selective.n_accepted, "n_total": selective.n_total,
            "accuracy_on_accepted": selective.accuracy_on_accepted, "selective_risk": selective.selective_risk,
        },
        evaluated_at=format_utc_timestamp(utc_now()),
    )
    return result


# --------------------------------------------------------------------------
# Source resolution: CalibrationSpec -> concrete, per-outer-fold inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ResolvedCalibrationInputs:
    timeline: pd.DataFrame
    outer_folds: tuple[Fold, ...]
    label_horizon_bars: int
    objective: ObjectiveType
    model_factory: ModelFactory
    serializer: ModelSerializer
    source_model_identity: str
    per_outer_fold_hyperparameters: Mapping[int, ModelHyperparameters]
    per_outer_fold_features: Mapping[int, tuple[str, ...]]


def resolve_calibration_inputs(
    spec: CalibrationSpec, *, experiment_manifest_store: ExperimentManifestStore, research_manifest_store: ResearchManifestStore,
    research_dataset_store: ResearchDatasetStore, model_registry: ModelRegistry,
    optimization_manifest_store: OptimizationManifestStore | None, artifact_store: MLArtifactStore,
    serializer_registry: Mapping[str, tuple[ModelSerializer, ModelDeserializer]] | None = None,
) -> ResolvedCalibrationInputs:
    """Resolves `CalibrationSpec.source_experiment_id` (and, if bound,
    `.source_optimization_id`) into concrete data -- reusing the SAME
    experiment/dataset-loading primitives `optimization.runner` already
    uses (`reconstruct_dataset_timeline`, `assert_preprocessing_is_safe_
    for_execution`, `extract_label_horizon_bars`, `build_folds_from_
    split_binding`), never a second reimplementation of "how to load an
    experiment's bound dataset."

    Every identity-bearing `CalibrationSpec` field is cross-checked
    against what was actually loaded (dataset content id, split plan
    fingerprint, base model definition identity) -- an inconsistency
    fails closed (`CalibrationDataError`), never silently proceeds with
    whichever value happened to be loaded (Section 4)."""
    experiment_manifest = experiment_manifest_store.load(spec.source_experiment_id)
    experiment_spec = experiment_manifest.spec
    if experiment_spec.dataset_binding.content_id != spec.dataset_content_id:
        raise CalibrationDataError(
            f"CalibrationSpec.dataset_content_id ({spec.dataset_content_id!r}) does not match the source "
            f"experiment's bound dataset content id ({experiment_spec.dataset_binding.content_id!r})"
        )
    dataset_manifest = research_manifest_store.load(experiment_spec.dataset_binding.dataset_id, experiment_spec.dataset_binding.manifest_version)
    assert_preprocessing_is_safe_for_execution(dataset_manifest)
    label_horizon_bars = extract_label_horizon_bars(dataset_manifest)
    timeline = reconstruct_dataset_timeline(
        research_dataset_store, dataset_id=experiment_spec.dataset_binding.dataset_id,
        content_id=experiment_spec.dataset_binding.content_id, timestamp_column=_TIMESTAMP_COLUMN,
    )

    outer_plan = build_folds_from_split_binding(experiment_spec.split_binding, timeline[_TIMESTAMP_COLUMN], label_horizon_bars=label_horizon_bars)
    plan_validation = validate_fold_plan(outer_plan, timeline=timeline, timestamp_column=_TIMESTAMP_COLUMN)
    if not plan_validation.is_ready:
        blocking = [*plan_validation.criticals, *plan_validation.errors]
        summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
        raise CalibrationDataError(f"CalibrationRunner: outer fold plan validation failed: {summary}")
    # `Fold`/`FoldPlan` are deliberately execution-TRANSIENT (see
    # `execution.splitters`' own docstring) and have no JSON
    # representation -- the identity-relevant, re-derivable fact is the
    # DECLARATIVE `SplitBinding` that deterministically produces them,
    # exactly how `dataset_content_id` binds to a declarative dataset
    # reference rather than to re-serialized DataFrame bytes.
    plan_fingerprint = fingerprint_json(experiment_spec.split_binding.to_json_dict())
    if plan_fingerprint != spec.split_plan_fingerprint:
        raise CalibrationDataError(
            f"CalibrationSpec.split_plan_fingerprint ({spec.split_plan_fingerprint!r}) does not match the "
            f"recomputed split binding fingerprint ({plan_fingerprint!r})"
        )

    model_definition = model_registry.get(experiment_spec.model_name, experiment_spec.model_version)
    model_definition_fingerprint = model_definition.fingerprint()
    if model_definition_fingerprint != spec.base_model_definition_identity:
        raise CalibrationDataError(
            f"CalibrationSpec.base_model_definition_identity ({spec.base_model_definition_identity!r}) does not "
            f"match the resolved model definition's fingerprint ({model_definition_fingerprint!r})"
        )
    serializer = resolve_serializer(model_definition.serializer_id, registry=serializer_registry)
    source_model_identity = f"{model_definition.name}@{model_definition.version}#{model_definition_fingerprint[:12]}"

    per_fold_hyperparameters: dict[int, ModelHyperparameters] = {}
    per_fold_features: dict[int, tuple[str, ...]] = {}
    if spec.source_optimization_id is not None:
        if optimization_manifest_store is None:
            raise CalibrationDataError("CalibrationSpec.source_optimization_id is set but no OptimizationManifestStore was provided")
        optimization_manifest = optimization_manifest_store.load(spec.source_optimization_id)
        if optimization_manifest.stage is not OptimizationStage.COMPLETED:
            raise CalibrationDataError(
                f"CalibrationSpec.source_optimization_id={spec.source_optimization_id!r} has not reached "
                f"OptimizationStage.COMPLETED (currently {optimization_manifest.stage.value!r}) -- refusing to "
                "calibrate against an in-progress or failed optimization"
            )
        for fold in outer_plan.folds:
            reference = optimization_manifest.outer_fold_result_references.get(fold.fold_index)
            if reference is None:
                raise CalibrationDataError(f"Source optimization has no recorded result for outer fold {fold.fold_index}")
            raw = artifact_store.read_artifact(reference.content_hash)
            outer_result = OuterFoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
            per_fold_hyperparameters[fold.fold_index] = ModelHyperparameters(values=dict(outer_result.final_hyperparameters))
            per_fold_features[fold.fold_index] = outer_result.final_selected_features
    else:
        for fold in outer_plan.folds:
            per_fold_hyperparameters[fold.fold_index] = experiment_spec.hyperparameters
            per_fold_features[fold.fold_index] = experiment_spec.feature_binding.feature_names

    return ResolvedCalibrationInputs(
        timeline=timeline, outer_folds=outer_plan.folds, label_horizon_bars=label_horizon_bars, objective=experiment_spec.objective,
        model_factory=model_definition.factory, serializer=serializer, source_model_identity=source_model_identity,
        per_outer_fold_hyperparameters=per_fold_hyperparameters, per_outer_fold_features=per_fold_features,
    )


# --------------------------------------------------------------------------
# Top-level manifest-driven orchestration
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    manifest: CalibrationManifest
    was_idempotent_no_op: bool


class CalibrationRunner:
    def __init__(
        self, *, ml_artifacts_root: Path | str, model_registry: ModelRegistry, research_manifest_store: ResearchManifestStore,
        research_dataset_store: ResearchDatasetStore, experiment_manifest_store: ExperimentManifestStore,
        optimization_manifest_store: OptimizationManifestStore | None = None,
        additional_serializers: Mapping[str, tuple[ModelSerializer, ModelDeserializer]] | None = None,
    ) -> None:
        self._root = Path(ml_artifacts_root).resolve()
        self._artifact_store = MLArtifactStore(self._root)
        self._manifest_store = CalibrationManifestStore(self._root)
        self._event_store = CalibrationEventStore(self._root)
        self._model_registry = model_registry
        self._research_manifest_store = research_manifest_store
        self._research_dataset_store = research_dataset_store
        self._experiment_manifest_store = experiment_manifest_store
        self._optimization_manifest_store = optimization_manifest_store
        # Merge, never replace: `additional_serializers` alone would
        # silently drop the built-in test-only entry, exactly the pitfall
        # `optimization.runner.OptimizationRunner.__init__` already
        # documents and avoids for the identical reason.
        self._serializer_registry: dict[str, tuple[ModelSerializer, ModelDeserializer]] = {
            **_SERIALIZER_REGISTRY, **dict(additional_serializers or {}),
        }

    def _run_lock_path(self, calibration_id: str) -> Path:
        return self._root / "calibrations" / calibration_id / _CALIBRATION_RUN_LOCK_FILE_NAME

    def run(self, spec: CalibrationSpec) -> CalibrationOutcome:
        from quant_platform.calibration.specs import compute_calibration_identity

        identity = compute_calibration_identity(spec)
        with experiment_lock(self._run_lock_path(identity.calibration_id)):
            return self._run_locked(spec, identity.calibration_id, require_existing=False)

    def resume(self, calibration_id: str, *, spec: CalibrationSpec | None = None) -> CalibrationOutcome:
        # Local import: `calibration.resume` imports `OuterFoldCalibrationResult`
        # FROM this module, so a module-level import here would cycle.
        from quant_platform.calibration.resume import require_calibration_resumable

        existing = self._manifest_store.load_if_exists(calibration_id)
        if existing is not None and existing.stage is CalibrationStage.COMPLETED:
            return CalibrationOutcome(manifest=existing, was_idempotent_no_op=True)
        require_calibration_resumable(existing, calibration_id=calibration_id)
        assert existing is not None  # require_calibration_resumable already raised CalibrationResumeError otherwise
        resolved_spec = spec if spec is not None else self._load_spec_artifact(existing)
        self._require_compatible_environment(existing, resolved_spec)
        with experiment_lock(self._run_lock_path(calibration_id)):
            return self._run_locked(resolved_spec, calibration_id, require_existing=True)

    def _require_compatible_environment(self, manifest: CalibrationManifest, spec: CalibrationSpec) -> None:
        """Mirrors `optimization.runner.OptimizationRunner._require_
        compatible_optuna_version`'s fail-closed pattern exactly, scoped
        to `scikit-learn` -- the library backing every calibration
        method's `.fit()` (`LogisticRegression`/`IsotonicRegression`).
        Resuming under a DIFFERENT installed scikit-learn version than
        the one that created this calibration could fit the REMAINING
        outer folds' calibrators under different numerics than the
        already-completed folds used within the SAME calibration run --
        fail closed before that happens, unless `spec.determinism_policy`
        is the explicit, opt-in `WARN` relaxation (see that enum's own
        docstring)."""
        reference = manifest.environment_snapshot_reference
        if reference is None:
            return  # created before an environment snapshot was ever recorded -- nothing to compare against
        try:
            raw = self._artifact_store.read_artifact(reference.content_hash)
            original_snapshot = EnvironmentSnapshot.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except (ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError) as exc:
            raise CalibrationResumeError(
                f"Calibration {manifest.calibration_id!r}: recorded ENVIRONMENT_SNAPSHOT artifact could not be "
                f"read and decoded, so scikit-learn version compatibility cannot be verified -- refusing to "
                f"resume: {exc}",
                context={"calibration_id": manifest.calibration_id},
            ) from exc
        original_version = original_snapshot.package_versions.get("scikit-learn")
        current_version = importlib.metadata.version("scikit-learn")
        if original_version is None or original_version == current_version:
            return
        message = (
            f"Calibration {manifest.calibration_id!r} was created under scikit-learn=={original_version}, but the "
            f"currently installed version is scikit-learn=={current_version} -- resuming could fit the remaining "
            "outer folds' calibrators under a different library version than the already-completed folds used."
        )
        if spec.determinism_policy is DeterminismPolicy.WARN:
            logger.warning("%s Proceeding anyway (determinism_policy=WARN).", message)
            return
        raise CalibrationResumeError(
            f"{message} Re-install the original scikit-learn version to resume under DeterminismPolicy.STRICT "
            "(the default), or build this CalibrationSpec with determinism_policy=warn to proceed anyway.",
            context={
                "calibration_id": manifest.calibration_id, "original_scikit_learn_version": original_version,
                "current_scikit_learn_version": current_version,
            },
        )

    def _load_spec_artifact(self, manifest: CalibrationManifest) -> CalibrationSpec:
        if manifest.spec_reference is None:
            raise CalibrationResumeError(f"Calibration {manifest.calibration_id!r} has no recorded CALIBRATION_SPEC artifact to resume from")
        raw = self._artifact_store.read_artifact(manifest.spec_reference.content_hash)
        return CalibrationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))

    def _run_locked(self, spec: CalibrationSpec, calibration_id: str, *, require_existing: bool) -> CalibrationOutcome:
        manifest = self._manifest_store.load_if_exists(calibration_id)
        if require_existing and manifest is None:
            raise CalibrationResumeError(f"No calibration manifest exists for calibration_id={calibration_id!r}")  # pragma: no cover - guarded identically before locking

        if manifest is not None and manifest.stage in (CalibrationStage.COMPLETED, CalibrationStage.FAILED):
            if manifest.stage is CalibrationStage.COMPLETED:
                return CalibrationOutcome(manifest=manifest, was_idempotent_no_op=True)
            raise CalibrationResumeError(
                f"Calibration {calibration_id!r} already reached terminal stage {manifest.stage.value!r}",
                context={"calibration_id": calibration_id, "stage": manifest.stage.value},
            )

        if manifest is None:
            now = format_utc_timestamp(utc_now())
            spec_ref = self._artifact_store.write_artifact(canonical_json_bytes(spec.to_json_dict()), category=ArtifactCategory.CALIBRATION_SPEC)
            env_ref = self._artifact_store.write_artifact(
                canonical_json_bytes(capture_environment_snapshot().to_json_dict()), category=ArtifactCategory.ENVIRONMENT_SNAPSHOT,
            )
            manifest = CalibrationManifest(
                schema_version=CALIBRATION_MANIFEST_SCHEMA_VERSION, calibration_id=calibration_id,
                source_experiment_id=spec.source_experiment_id, source_optimization_id=spec.source_optimization_id,
                stage=CalibrationStage.CREATED, created_at=now, updated_at=now, spec_reference=spec_ref,
                environment_snapshot_reference=env_ref, artifact_references=(spec_ref, env_ref),
            )
            self._manifest_store.create(manifest)
            self._event_store.append(calibration_id, CalibrationEventType.CALIBRATION_CREATED)
            self._event_store.append(calibration_id, CalibrationEventType.RUN_STARTED)
        else:
            manifest = self._manifest_store.bump_resume_count(calibration_id)
            self._event_store.append(calibration_id, CalibrationEventType.CALIBRATION_RESUMED, details={"resume_count": manifest.resume_count})

        try:
            manifest = self._execute_pipeline(spec, manifest)
        except QuantPlatformError as exc:
            # Mirrors `OptimizationRunner`'s per-decision-point `_fail(...)`
            # calls, collapsed into one boundary here since calibration's
            # per-fold pipeline is a single atomic sequence rather than a
            # multi-decision search loop: ANY domain exception raised
            # while a calibration is in progress (a leakage-validation
            # failure, a missing upstream artifact, an unsupported model,
            # ...) must leave an accurate, diagnosable terminal record --
            # `stage=FAILED` with a persisted reason and a RUN_FAILED
            # event -- rather than silently stranding the manifest at
            # whatever intermediate stage it happened to reach. The
            # exception is always re-raised afterward, never swallowed;
            # `_fail` re-reads the manifest fresh from disk (never trusts
            # the possibly-stale in-memory `manifest` above), so it
            # correctly targets whatever stage was last durably persisted.
            #
            # `ExperimentLockError` is deliberately EXCLUDED: it signals
            # lock contention/an aborted process (this platform's own
            # crash-simulation tests raise it as a stand-in for an
            # uncontrolled process kill, which in reality never runs any
            # exception handler at all) -- never a genuine calibration-
            # domain failure about THIS calibration's own data/config, so
            # it must never be recorded as a terminal FAILED verdict.
            if not isinstance(exc, ExperimentLockError):
                self._fail(calibration_id, str(exc))
            raise
        return CalibrationOutcome(manifest=manifest, was_idempotent_no_op=False)

    def _fail(self, calibration_id: str, failure_summary: str) -> None:
        now = format_utc_timestamp(utc_now())
        self._manifest_store.transition(calibration_id, new_stage=CalibrationStage.FAILED, updated_at=now, completed_at=now, failure_summary=failure_summary)
        self._event_store.append(calibration_id, CalibrationEventType.RUN_FAILED, details={"reason": failure_summary[:200]})

    def _execute_pipeline(self, spec: CalibrationSpec, manifest: CalibrationManifest) -> CalibrationManifest:
        calibration_id = manifest.calibration_id
        inputs = resolve_calibration_inputs(
            spec, experiment_manifest_store=self._experiment_manifest_store, research_manifest_store=self._research_manifest_store,
            research_dataset_store=self._research_dataset_store, model_registry=self._model_registry,
            optimization_manifest_store=self._optimization_manifest_store, artifact_store=self._artifact_store,
            serializer_registry=self._serializer_registry,
        )

        if manifest.stage is CalibrationStage.CREATED:
            manifest = self._manifest_store.transition(
                calibration_id, new_stage=CalibrationStage.INNER_PREDICTIONS_READY, updated_at=format_utc_timestamp(utc_now()),
                total_outer_folds=len(inputs.outer_folds), current_outer_fold_index=inputs.outer_folds[0].fold_index if inputs.outer_folds else None,
            )

        # Local import: `calibration.resume` imports `OuterFoldCalibrationResult`
        # FROM this module, so a module-level import here would cycle.
        from quant_platform.calibration.resume import verify_completed_calibration_outer_folds

        completed = set(manifest.completed_outer_fold_indices)
        verified_complete, needs_rerun = verify_completed_calibration_outer_folds(manifest, artifact_store=self._artifact_store)
        if needs_rerun:
            # "Unverified completed work never trusted" (Section 24): a
            # missing/corrupted/schema-mismatched artifact for a fold the
            # manifest CLAIMS is completed is treated as NOT completed --
            # fall through and redo it.
            logger.warning(
                "Calibration %s: outer fold(s) %s were recorded as completed but their artifacts could not be "
                "verified -- re-running them", calibration_id[:12], sorted(needs_rerun),
            )
            completed -= needs_rerun

        results: list[OuterFoldCalibrationResult] = []
        for fold in inputs.outer_folds:
            if fold.fold_index in completed and fold.fold_index in verified_complete:
                reference = manifest.outer_fold_result_references[fold.fold_index]
                raw = self._artifact_store.read_artifact(reference.content_hash)
                results.append(OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8"))))
                continue

            manifest = self._manifest_store.transition(
                calibration_id, new_stage=CalibrationStage.INNER_PREDICTIONS_READY, updated_at=format_utc_timestamp(utc_now()),
                current_outer_fold_index=fold.fold_index,
            ) if manifest.stage is not CalibrationStage.INNER_PREDICTIONS_READY else manifest
            self._event_store.append(calibration_id, CalibrationEventType.OUTER_FOLD_STARTED, details={"outer_fold_index": fold.fold_index})

            # `run_outer_fold_calibration` computes ALL of Section 18's
            # steps 2-14 as one atomic, pure function of already-fixed
            # inputs -- exactly `optimization.runner`'s own precedent for
            # why redoing candidate-selection-through-evaluation from
            # scratch after a crash is always safe (see that module's
            # docstring). A crash during this call simply leaves the
            # manifest at `INNER_PREDICTIONS_READY`; resume redoes the
            # WHOLE fold, bit-for-bit identically. The stage machine is
            # still walked through in full immediately after, so the
            # persisted stage history/event log names every checkpoint
            # Section 23 requires, even though they land in one burst.
            result = run_outer_fold_calibration(
                spec=spec, calibration_id=calibration_id, outer_fold=fold, timeline=inputs.timeline,
                feature_names=inputs.per_outer_fold_features[fold.fold_index], label_column=_LABEL_COLUMN,
                label_horizon_bars=inputs.label_horizon_bars, model_factory=inputs.model_factory,
                hyperparameters=inputs.per_outer_fold_hyperparameters[fold.fold_index], objective=inputs.objective,
                seed_configuration=SeedConfiguration(master_seed=spec.seed), source_model_identity=inputs.source_model_identity,
                source_experiment_id=spec.source_experiment_id, artifact_store=self._artifact_store, serializer=inputs.serializer,
            )
            for stage, event_type in (
                (CalibrationStage.CALIBRATORS_EVALUATED, CalibrationEventType.CALIBRATOR_CANDIDATES_EVALUATED),
                (CalibrationStage.CALIBRATOR_SELECTED, CalibrationEventType.CALIBRATOR_SELECTED),
                (CalibrationStage.THRESHOLD_SELECTED, CalibrationEventType.THRESHOLD_SELECTED),
                (CalibrationStage.POLICIES_FROZEN, CalibrationEventType.POLICIES_FROZEN),
                (CalibrationStage.OUTER_PREDICTIONS_READY, CalibrationEventType.OUTER_FOLD_PREDICTED),
            ):
                manifest = self._manifest_store.transition(calibration_id, new_stage=stage, updated_at=format_utc_timestamp(utc_now()))
                self._event_store.append(calibration_id, event_type, details={"outer_fold_index": fold.fold_index})

            result_ref = self._artifact_store.write_artifact(canonical_json_bytes(result.to_json_dict()), category=ArtifactCategory.OUTER_FOLD_CALIBRATION_RESULT)
            completed.add(fold.fold_index)
            manifest = self._manifest_store.transition(
                calibration_id, new_stage=CalibrationStage.EVALUATED, updated_at=format_utc_timestamp(utc_now()),
                completed_outer_fold_indices=tuple(sorted(completed)),
                outer_fold_result_references={**manifest.outer_fold_result_references, fold.fold_index: result_ref},
            )
            self._event_store.append(calibration_id, CalibrationEventType.OUTER_FOLD_EVALUATED, details={"outer_fold_index": fold.fold_index})
            self._event_store.append(calibration_id, CalibrationEventType.OUTER_FOLD_COMPLETED, details={"outer_fold_index": fold.fold_index})
            results.append(result)

            is_last = fold.fold_index == inputs.outer_folds[-1].fold_index
            if not is_last:
                manifest = self._manifest_store.transition(calibration_id, new_stage=CalibrationStage.INNER_PREDICTIONS_READY, updated_at=format_utc_timestamp(utc_now()))

        if manifest.stage is CalibrationStage.EVALUATED:
            manifest = self._manifest_store.transition(calibration_id, new_stage=CalibrationStage.VERIFIED, updated_at=format_utc_timestamp(utc_now()))
        self._event_store.append(calibration_id, CalibrationEventType.RUN_VERIFIED)

        from quant_platform.calibration.reporting import build_calibration_report_json

        report_json = build_calibration_report_json(manifest, spec=spec, outer_fold_results=results)
        report_ref = self._artifact_store.write_artifact(canonical_json_bytes(report_json), category=ArtifactCategory.CALIBRATION_REPORT)

        completed_at = format_utc_timestamp(utc_now())
        manifest = self._manifest_store.transition(
            calibration_id, new_stage=CalibrationStage.COMPLETED, updated_at=completed_at, completed_at=completed_at,
            aggregate_report_reference=report_ref,
        )
        self._event_store.append(calibration_id, CalibrationEventType.RUN_COMPLETED)
        return manifest

    @staticmethod
    def _selected_kind(result: OuterFoldCalibrationResult) -> str:
        return result.decision_policy_reference.content_hash[:12]


__all__ = [
    "OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION",
    "CalibrationOutcome",
    "CalibrationRunner",
    "OuterFoldCalibrationResult",
    "ResolvedCalibrationInputs",
    "resolve_calibration_inputs",
    "run_outer_fold_calibration",
]
