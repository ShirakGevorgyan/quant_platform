"""Isolated, idempotent single-trial execution (Milestone 4D) -- the
nested walk-forward inner loop: for one outer fold's `InnerFoldPlan`,
fit feature selection on inner-train, fit the sampled candidate on
inner-train (selected features only), predict on inner-validation,
compute metrics -- once per inner fold, then aggregated into one
immutable `TrialResult`.

"One trial must never mutate another trial's state. Trial execution must
be isolated and idempotent." `run_trial` is a (near-)pure function of its
arguments: it reads `timeline` but never writes to it, constructs a FRESH
model/selector per inner fold (never reuses or mutates one across folds
or across trials), and every artifact it writes is content-addressed
(writing the same bytes twice is a safe no-op -- see `ml.artifacts.
MLArtifactStore`). Two calls to `run_trial` with the same `TrialSpec` and
the same underlying data always produce byte-identical `TrialResult`s.

WHY A BAD HYPERPARAMETER COMBINATION NEVER CRASHES THE WHOLE OPTIMIZATION
--------------------------------------------------------------------------
"Every sampled parameter combination must be validated by the model
wrapper before training. Invalid combinations should be rejected as
invalid trials with a stable reason, not crash with raw library
exceptions." Rather than trying to enumerate every way a third-party
library (LightGBM/XGBoost/CatBoost/scikit-learn) can reject a bad
combination up front, this module relies on one uniform mechanism: a
raised exception from feature selection OR model `fit` for one inner
fold demotes ONLY that inner fold to "did not produce a value" (never
propagates as a raw crash) -- see `_run_one_inner_fold`'s broad `except
Exception`, mirroring the identical, already-established pattern
`execution.runner._run_one_fold` uses for its own per-fold exception
handling. If a hyperparameter combination is fundamentally broken, it
fails identically on every inner fold, so `optimization.objectives.
aggregate_primary_metric`'s `min_successful_inner_folds` gate naturally
demotes the WHOLE trial to `TrialStatus.INVALID` with a clear,
accumulated reason -- the same outcome a bespoke up-front validator would
produce, without needing to anticipate every library's error shape.

WHAT IS -- AND IS NOT -- PERSISTED PER INNER FOLD
--------------------------------------------------------------------------
Only the `FeatureSelectionResult` is written as its own content-addressed
artifact per inner fold (needed for feature-stability analysis and
audit). Fitted models and raw predictions from INNER folds are
deliberately NOT persisted -- with `outer_folds x trials x inner_folds`
potentially numbering in the thousands, persisting a full fitted model
blob at every one of those would be an unbounded storage cost for
artifacts nothing ever reloads (inner-fold performance is already fully
captured, numerically, in `InnerFoldTrialMetrics`). The ONE model that is
ever persisted in full is the WINNING candidate's outer-train refit (see
`optimization.outer_fold`).

MODEL-CAPABILITY FAIL-CLOSED RE-CHECK
--------------------------------------------------------------------------
`run_trial` re-checks `ModelCapabilities.requires_scaled_numeric_features`
itself, unconditionally, before running a single inner fold -- the real,
always-active enforcement of this milestone's Option A preprocessing
policy (see `optimization.models`' module docstring). Whatever earlier,
friendlier check `build_optimization_spec` may have performed is not
trusted alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import TrainingDataValidationError, TrialExecutionError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory, ProbabilisticPredictor
from quant_platform.ml.metrics import compute_metrics
from quant_platform.ml.model_validation import validate_training_data
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    JsonPrimitive,
    ModelHyperparameters,
    ObjectiveType,
)
from quant_platform.ml.persistence import canonical_json_bytes
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.candidates import InnerFoldTrialMetrics, TrialResult, TrialSpec, TrialStatus
from quant_platform.optimization.feature_selection import FeatureUniverse, run_feature_selection
from quant_platform.optimization.inner_splits import InnerFold, InnerFoldPlan
from quant_platform.optimization.models import EarlyStoppingConfig, feature_selector_seed, model_fit_seed
from quant_platform.optimization.objectives import aggregate_primary_metric
from quant_platform.optimization.search_space import (
    CATBOOST_MODEL_NAME,
    LIGHTGBM_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)

GBM_MODEL_NAMES: tuple[str, ...] = (LIGHTGBM_MODEL_NAME, XGBOOST_MODEL_NAME, CATBOOST_MODEL_NAME)
"""Models whose declared, already-shipped hyperparameter contract
recognizes `early_stopping_rounds`/`validation_fraction` -- see
`ml.model_zoo.lightgbm_model`/`xgboost_model`/`catboost_model`'s own
`_split_hyperparameters`. Early stopping is only ever activated for these."""

_INVALID_FEATURE_COUNT_PLACEHOLDER = 1
"""`InnerFoldTrialMetrics.selected_feature_count` must be >= 1; used only
for the (unsuccessful) inner-fold record built when feature selection
itself failed before any feature count could be determined."""


def _inject_early_stopping(
    hyperparameters: dict[str, JsonPrimitive], *, model_name: str, early_stopping_config: EarlyStoppingConfig,
) -> dict[str, JsonPrimitive]:
    if model_name not in GBM_MODEL_NAMES or not early_stopping_config.enabled:
        return hyperparameters
    return {**hyperparameters, "early_stopping_rounds": early_stopping_config.patience, "validation_fraction": early_stopping_config.validation_fraction}


def _extract_best_iteration(fitted: object) -> int | None:
    """Generic, duck-typed accessor -- never a per-model `if`/`elif`
    branch. LightGBM's `FittedLightGBMModel.best_iteration` uses `0` as
    its own "early stopping was not used" sentinel (see that class's
    `_num_iteration`); normalized to `None` here so every model family's
    "not available" case looks identical to a caller."""
    value = getattr(fitted, "best_iteration", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _positive_class_probabilities(fitted: object, features: pd.DataFrame, objective: ObjectiveType) -> np.ndarray | None:
    if objective is ObjectiveType.REGRESSION:
        return None
    metadata = getattr(fitted, "metadata", None)
    if metadata is None or not metadata.capabilities.supports_predict_proba or not isinstance(fitted, ProbabilisticPredictor):
        return None
    proba = fitted.predict_proba(features)
    class_labels = list(fitted.class_labels)
    positive_index = class_labels.index(1)
    return np.asarray(proba[:, positive_index], dtype="float64")


def _run_one_inner_fold(
    inner_fold: InnerFold, *, trial_spec: TrialSpec, timeline: pd.DataFrame, feature_universe: FeatureUniverse,
    model_name: str, model_factory: ModelFactory, hyperparameters: ModelHyperparameters, seed_configuration: SeedConfiguration,
    artifact_store: MLArtifactStore, label_column: str,
) -> InnerFoldTrialMetrics:
    started = time.perf_counter()
    train_df = timeline.iloc[inner_fold.train_indices]
    validation_df = timeline.iloc[inner_fold.validation_indices]
    train_features = train_df[list(feature_universe.feature_names)]
    train_labels = train_df[label_column]

    selector_seed = feature_selector_seed(seed_configuration, trial_spec.outer_fold_index, trial_spec.trial_number, inner_fold.inner_fold_index)
    try:
        fs_result = run_feature_selection(
            trial_spec.feature_selection_spec, universe=feature_universe, features=train_features, labels=train_labels,
            row_positions=inner_fold.train_indices, seed=selector_seed, objective=trial_spec.objective,
            model_name=model_name, model_factory=model_factory, hyperparameters=hyperparameters,
        )
    except Exception:
        return InnerFoldTrialMetrics(
            inner_fold_index=inner_fold.inner_fold_index, primary_metric_value=None, secondary_metrics={},
            selected_feature_count=_INVALID_FEATURE_COUNT_PLACEHOLDER, feature_selection_result_reference=None,
            best_iteration=None, duration_seconds=time.perf_counter() - started,
        )

    fs_ref = artifact_store.write_artifact(canonical_json_bytes(fs_result.to_json_dict()), category=ArtifactCategory.FEATURE_SELECTION_RESULT)
    selected = list(fs_result.selected_features)

    model_seed = model_fit_seed(seed_configuration, trial_spec.outer_fold_index, trial_spec.trial_number, inner_fold.inner_fold_index)
    model_feature_schema = FeatureSchema(feature_names=tuple(selected))
    model = model_factory.create(hyperparameters=hyperparameters, feature_schema=model_feature_schema, objective=trial_spec.objective)

    try:
        validation_report = validate_training_data(metadata=model.metadata, features=train_features[selected], labels=train_labels)
        if not validation_report.is_ready:
            blocking = [*validation_report.criticals, *validation_report.errors]
            summary = "; ".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in blocking)
            raise TrainingDataValidationError(f"Inner fold {inner_fold.inner_fold_index}: {summary}")
        fitted = model.fit(train_features[selected], train_labels, seeds=SeedConfiguration(master_seed=model_seed))
    except Exception:
        return InnerFoldTrialMetrics(
            inner_fold_index=inner_fold.inner_fold_index, primary_metric_value=None, secondary_metrics={},
            selected_feature_count=len(selected), feature_selection_result_reference=fs_ref, best_iteration=None,
            duration_seconds=time.perf_counter() - started,
        )

    validation_features = validation_df[selected]
    predictions = fitted.predict(validation_features)
    y_true = validation_df[label_column].to_numpy(dtype="float64")
    y_proba_positive = _positive_class_probabilities(fitted, validation_features, trial_spec.objective)
    metric_report = compute_metrics(trial_spec.objective, y_true, predictions, y_proba_positive)

    primary_value = metric_report.values.get(trial_spec.primary_metric)
    secondary = {k: float(v) for k, v in metric_report.values.items() if k != trial_spec.primary_metric}
    return InnerFoldTrialMetrics(
        inner_fold_index=inner_fold.inner_fold_index,
        primary_metric_value=(None if primary_value is None else float(primary_value)),
        secondary_metrics=secondary, selected_feature_count=len(selected), feature_selection_result_reference=fs_ref,
        best_iteration=_extract_best_iteration(fitted), duration_seconds=time.perf_counter() - started,
    )


def run_trial(
    trial_spec: TrialSpec,
    *,
    inner_fold_plan: InnerFoldPlan,
    timeline: pd.DataFrame,
    feature_universe: FeatureUniverse,
    model_name: str,
    model_factory: ModelFactory,
    seed_configuration: SeedConfiguration,
    min_successful_inner_folds: int,
    early_stopping_config: EarlyStoppingConfig,
    artifact_store: MLArtifactStore,
    environment_snapshot_reference: ArtifactReference | None = None,
    label_column: str = "label",
    pruning_callback: Callable[[tuple[InnerFoldTrialMetrics, ...]], bool] | None = None,
) -> TrialResult:
    """Runs ONE trial's complete nested inner walk-forward loop. `timeline`
    is the full reconstructed dataset timeline; only rows named by
    `inner_fold_plan`'s own (already outer-train-confined) row positions
    are ever read -- this function has no access to, and never receives,
    the outer fold's test partition at all.

    `pruning_callback`, if given, is called after EVERY completed inner
    fold with the metrics accumulated SO FAR; returning `True` stops the
    trial early (`TrialStatus.PRUNED`). This function has no opinion on
    HOW that decision should be made (comparing against other trials'
    progress requires state this isolated function deliberately does not
    hold) -- see `optimization.study` for the orchestration that supplies
    a real pruning decision."""
    started = time.perf_counter()

    probe_schema = FeatureSchema(feature_names=feature_universe.feature_names)
    probe_model = model_factory.create(
        hyperparameters=ModelHyperparameters(values=dict(trial_spec.sampled_hyperparameters)),
        feature_schema=probe_schema, objective=trial_spec.objective,
    )
    if probe_model.metadata.capabilities.requires_scaled_numeric_features:
        raise TrialExecutionError(
            f"Model {model_name!r} requires scaled numeric features; this milestone's preprocessing policy "
            "(Option A) excludes scale-sensitive models from optimization entirely -- see optimization.models"
        )
    if not probe_model.metadata.capabilities.is_deterministic:
        raise TrialExecutionError(f"Model {model_name!r} does not declare is_deterministic=True -- not permitted in this optimization engine")

    hyperparameters = ModelHyperparameters(
        values=_inject_early_stopping(dict(trial_spec.sampled_hyperparameters), model_name=model_name, early_stopping_config=early_stopping_config)
    )

    per_inner_fold: list[InnerFoldTrialMetrics] = []
    artifact_refs: list[ArtifactReference] = []
    warnings: list[str] = []
    pruned = False
    for inner_fold in inner_fold_plan.inner_folds:
        metrics = _run_one_inner_fold(
            inner_fold, trial_spec=trial_spec, timeline=timeline, feature_universe=feature_universe, model_name=model_name,
            model_factory=model_factory, hyperparameters=hyperparameters, seed_configuration=seed_configuration,
            artifact_store=artifact_store, label_column=label_column,
        )
        per_inner_fold.append(metrics)
        if metrics.feature_selection_result_reference is not None:
            artifact_refs.append(metrics.feature_selection_result_reference)
        if metrics.primary_metric_value is None:
            warnings.append(f"inner fold {inner_fold.inner_fold_index}: did not produce primary metric {trial_spec.primary_metric!r}")

        if pruning_callback is not None and pruning_callback(tuple(per_inner_fold)):
            pruned = True
            break

    duration = time.perf_counter() - started
    values = [m.primary_metric_value for m in per_inner_fold]
    total_inner_folds = len(inner_fold_plan.inner_folds)

    if pruned:
        successful = [v for v in values if v is not None]
        partial_aggregate = (sum(successful) / len(successful)) if successful else None
        return TrialResult(
            schema_version=1, optimization_id=trial_spec.optimization_id, outer_fold_index=trial_spec.outer_fold_index,
            trial_number=trial_spec.trial_number, status=TrialStatus.PRUNED,
            sampled_hyperparameters=trial_spec.sampled_hyperparameters, inner_fold_metrics=tuple(per_inner_fold),
            primary_metric_aggregate=partial_aggregate, successful_inner_folds=len(successful),
            total_inner_folds=total_inner_folds, duration_seconds=duration, artifact_references=tuple(artifact_refs),
            environment_snapshot_reference=environment_snapshot_reference, failure_code="pruned",
            failure_reason=f"pruned after {len(per_inner_fold)}/{total_inner_folds} inner fold(s)", warnings=tuple(warnings),
        )

    outcome = aggregate_primary_metric(values, min_successful_inner_folds=min_successful_inner_folds)
    if not outcome.is_valid:
        return TrialResult(
            schema_version=1, optimization_id=trial_spec.optimization_id, outer_fold_index=trial_spec.outer_fold_index,
            trial_number=trial_spec.trial_number, status=TrialStatus.INVALID,
            sampled_hyperparameters=trial_spec.sampled_hyperparameters, inner_fold_metrics=tuple(per_inner_fold),
            primary_metric_aggregate=None, successful_inner_folds=outcome.successful_inner_folds,
            total_inner_folds=total_inner_folds, duration_seconds=duration, artifact_references=tuple(artifact_refs),
            environment_snapshot_reference=environment_snapshot_reference, failure_code="insufficient_successful_inner_folds",
            failure_reason=outcome.reason, warnings=tuple(warnings),
        )

    return TrialResult(
        schema_version=1, optimization_id=trial_spec.optimization_id, outer_fold_index=trial_spec.outer_fold_index,
        trial_number=trial_spec.trial_number, status=TrialStatus.COMPLETED,
        sampled_hyperparameters=trial_spec.sampled_hyperparameters, inner_fold_metrics=tuple(per_inner_fold),
        primary_metric_aggregate=outcome.aggregate_value, successful_inner_folds=outcome.successful_inner_folds,
        total_inner_folds=total_inner_folds, duration_seconds=duration, artifact_references=tuple(artifact_refs),
        environment_snapshot_reference=environment_snapshot_reference, warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class FinalRoundDecision:
    """The deterministic number of boosting rounds to use when refitting
    the winning candidate on the complete outer-train partition -- see
    `EarlyStoppingConfig.final_round_policy`'s own docstring."""

    rounds: int
    source: str

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"FinalRoundDecision.rounds must be >= 1, got {self.rounds} (source={self.source!r})")


def resolve_final_round_count(
    trial_result: TrialResult, *, sampled_rounds: int, policy: str,
) -> FinalRoundDecision:
    """Reject the winner outright (`TrialExecutionError`, never a silent
    0-or-negative round count reaching a real fit call) when the declared
    policy cannot produce a valid final round -- `sampled_rounds` itself
    is the one value BOTH policies can fall back to (`"fixed"` always;
    `"median_best_iteration"` when no inner fold reports a best
    iteration), so it is validated unconditionally, up front, rather than
    separately in each branch below."""
    if sampled_rounds < 1:
        raise TrialExecutionError(
            f"Cannot resolve a final round count for trial {trial_result.trial_number}: the winning trial's own "
            f"sampled/declared round count is {sampled_rounds}, which is not a valid positive boosting-round "
            "count -- refusing to refit with an invalid final round rather than silently producing an untrained "
            "or malformed model",
            context={"trial_number": trial_result.trial_number, "sampled_rounds": sampled_rounds, "policy": policy},
        )
    if policy == "fixed":
        return FinalRoundDecision(rounds=sampled_rounds, source="fixed_configured_rounds")
    best_iterations = [m.best_iteration for m in trial_result.inner_fold_metrics if m.best_iteration is not None]
    if not best_iterations:
        return FinalRoundDecision(rounds=sampled_rounds, source="fixed_configured_rounds_no_best_iteration_available")
    sorted_values = sorted(best_iterations)
    n = len(sorted_values)
    mid = n // 2
    median = sorted_values[mid] if n % 2 == 1 else round((sorted_values[mid - 1] + sorted_values[mid]) / 2)
    return FinalRoundDecision(rounds=max(1, median), source="median_best_iteration_across_inner_folds")


__all__ = [
    "GBM_MODEL_NAMES",
    "FinalRoundDecision",
    "resolve_final_round_count",
    "run_trial",
]
