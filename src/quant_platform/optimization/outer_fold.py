"""Outer-fold finalization (Milestone 4D) -- refits the winning candidate
on the COMPLETE outer-train partition and evaluates it EXACTLY ONCE on
the untouched outer-test partition.

THIS MODULE IS THE ACTUAL ENFORCEMENT POINT OF THE MILESTONE'S CENTRAL RULE
--------------------------------------------------------------------------
"Any implementation that observes the outer-test result before candidate
selection is complete is invalid." `finalize_outer_fold` is the ONLY
function in this entire package that ever reads `outer_fold.test_indices`
-- no other module (`trial_executor`, `feature_selection`, `study`,
`candidates`) accepts, references, or has any way to obtain those row
positions. `finalize_outer_fold` REQUIRES an already-selected, already-
`COMPLETED` `winning_trial` as an input parameter; there is no code path
here (or anywhere else in this package) that tries multiple candidates
against outer-test and picks the best one -- the winner was chosen by
`optimization.candidates.rank_trials` using ONLY inner-fold evidence,
strictly before this function is ever called.

FINAL SELECTED FEATURE SET: REFIT ONE MORE TIME, ON OUTER-TRAIN, NEVER A
VOTE ACROSS INNER FOLDS
--------------------------------------------------------------------------
A trial's feature selection was fit independently inside each inner fold
and may legitimately have selected a DIFFERENT feature set in each one.
Rather than inventing a new, separate "combine N inner selections"
heuristic (majority vote, intersection, ...), this module extends the
SAME methodology one level further: it reruns the winning trial's OWN
`FeatureSelectionSpec` ONE more time, now fit on the complete outer-train
partition (a fresh, larger "training partition" in exactly the same
sense every inner fold's inner-train was) -- deterministic, seeded from
its own dedicated branch (`optimization.models.
outer_train_feature_selector_seed`), and, like every other selector
invocation in this package, fit on train data only (outer-train), never
outer-test.

FINAL BOOSTING-ROUND POLICY: COMPUTED EXTERNALLY, NEVER DELEGATED BACK TO
THE MODEL'S OWN INTERNAL EARLY STOPPING
--------------------------------------------------------------------------
When early stopping is enabled, the final refit does NOT pass
`early_stopping_rounds`/`validation_fraction` through to the model
wrapper (which would carve its own internal pseudo-validation tail out
of outer-train and self-determine a best iteration, silently overriding
this milestone's own declared policy). Instead, `optimization.
trial_executor.resolve_final_round_count` computes the round count from
the WINNING TRIAL's own already-completed inner-fold `best_iteration`
values (median, or the fixed sampled count -- see that function's
docstring for the exact, documented policy), and that number is set
DIRECTLY as the boosting-round hyperparameter for a plain, non-early-
stopping fit. This guarantees the final round count is deterministic,
inner-fold-derived, and never influenced by outer-test in any way.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from quant_platform.execution.splitters import Fold
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory, ModelSerializer, ProbabilisticPredictor
from quant_platform.ml.metrics import compute_metrics
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
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
    require_schema_version,
    utc_now,
)
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.candidates import TrialResult
from quant_platform.optimization.feature_selection import FeatureUniverse, run_feature_selection
from quant_platform.optimization.models import (
    OptimizationSpec,
    outer_train_feature_selector_seed,
    outer_train_refit_seed,
)
from quant_platform.optimization.trial_executor import GBM_MODEL_NAMES, resolve_final_round_count

OUTER_FOLD_RESULT_SCHEMA_VERSION = 1
_ROUNDS_KEYS = ("num_boost_round", "iterations")
"""Matches `optimization.candidates`'s own recognized boosting-round key
names -- duplicated here (a trivial, closed, two-element tuple) rather
than imported, the same small-set-duplication precedent `ml.
model_validation._SCALING_TRANSFORM_KINDS` already establishes, to avoid
coupling this module to `candidates`'s private naming."""


@dataclass(frozen=True, slots=True)
class OuterFoldResult:
    schema_version: int
    optimization_id: str
    outer_fold_index: int
    winning_trial_number: int
    final_selected_features: tuple[str, ...]
    final_hyperparameters: Mapping[str, JsonPrimitive]
    final_round_source: str | None
    seed: int
    training_duration_seconds: float
    outer_train_row_count: int
    outer_test_row_count: int
    outer_test_metrics: Mapping[str, JsonPrimitive]
    feature_selection_result_reference: ArtifactReference
    model_reference: ArtifactReference
    predictions_reference: ArtifactReference
    evaluated_at: str
    probabilities_reference: ArtifactReference | None = None
    search_summary_reference: ArtifactReference | None = None

    def __post_init__(self) -> None:
        if not self.optimization_id:
            raise ValueError("OuterFoldResult.optimization_id must not be empty")
        if self.outer_fold_index < 0:
            raise ValueError(f"OuterFoldResult.outer_fold_index must be >= 0, got {self.outer_fold_index}")
        if self.winning_trial_number < 0:
            raise ValueError(f"OuterFoldResult.winning_trial_number must be >= 0, got {self.winning_trial_number}")
        if not self.final_selected_features:
            raise ValueError("OuterFoldResult.final_selected_features must not be empty")
        if len(set(self.final_selected_features)) != len(self.final_selected_features):
            raise ValueError("OuterFoldResult.final_selected_features must not contain duplicates")
        if self.seed < 0:
            raise ValueError(f"OuterFoldResult.seed must be >= 0, got {self.seed}")
        if not math.isfinite(self.training_duration_seconds) or self.training_duration_seconds < 0:
            raise ValueError(
                f"OuterFoldResult.training_duration_seconds must be a finite number >= 0, "
                f"got {self.training_duration_seconds}"
            )
        if self.outer_train_row_count < 1 or self.outer_test_row_count < 1:
            raise ValueError("OuterFoldResult.outer_train_row_count/outer_test_row_count must be >= 1")
        validate_json_primitive_mapping(self.final_hyperparameters, field_name="OuterFoldResult.final_hyperparameters")
        validate_json_primitive_mapping(self.outer_test_metrics, field_name="OuterFoldResult.outer_test_metrics")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "optimization_id": self.optimization_id,
            "outer_fold_index": self.outer_fold_index, "winning_trial_number": self.winning_trial_number,
            "final_selected_features": list(self.final_selected_features),
            "final_hyperparameters": dict(sorted(self.final_hyperparameters.items())),
            "final_round_source": self.final_round_source, "seed": self.seed,
            "training_duration_seconds": self.training_duration_seconds, "outer_train_row_count": self.outer_train_row_count,
            "outer_test_row_count": self.outer_test_row_count, "outer_test_metrics": dict(sorted(self.outer_test_metrics.items())),
            "feature_selection_result_reference": self.feature_selection_result_reference.to_json_dict(),
            "model_reference": self.model_reference.to_json_dict(), "predictions_reference": self.predictions_reference.to_json_dict(),
            "probabilities_reference": (None if self.probabilities_reference is None else self.probabilities_reference.to_json_dict()),
            "search_summary_reference": (None if self.search_summary_reference is None else self.search_summary_reference.to_json_dict()),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OuterFoldResult:
        require_schema_version(raw, supported=OUTER_FOLD_RESULT_SCHEMA_VERSION, context="OuterFoldResult")
        probabilities_raw = raw.get("probabilities_reference")
        search_summary_raw = raw.get("search_summary_reference")
        return cls(
            schema_version=OUTER_FOLD_RESULT_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]),
            outer_fold_index=int(str(raw["outer_fold_index"])), winning_trial_number=int(str(raw["winning_trial_number"])),
            final_selected_features=tuple(str(n) for n in as_json_list(raw["final_selected_features"], field_name="final_selected_features")),
            final_hyperparameters=as_json_dict(raw.get("final_hyperparameters") or {}, field_name="final_hyperparameters"),
            final_round_source=(None if raw.get("final_round_source") is None else str(raw["final_round_source"])),
            seed=int(str(raw["seed"])), training_duration_seconds=float(str(raw["training_duration_seconds"])),
            outer_train_row_count=int(str(raw["outer_train_row_count"])), outer_test_row_count=int(str(raw["outer_test_row_count"])),
            outer_test_metrics=as_json_dict(raw.get("outer_test_metrics") or {}, field_name="outer_test_metrics"),
            feature_selection_result_reference=ArtifactReference.from_json_dict(as_json_dict(raw["feature_selection_result_reference"], field_name="feature_selection_result_reference")),
            model_reference=ArtifactReference.from_json_dict(as_json_dict(raw["model_reference"], field_name="model_reference")),
            predictions_reference=ArtifactReference.from_json_dict(as_json_dict(raw["predictions_reference"], field_name="predictions_reference")),
            probabilities_reference=(None if probabilities_raw is None else ArtifactReference.from_json_dict(as_json_dict(probabilities_raw, field_name="probabilities_reference"))),
            search_summary_reference=(None if search_summary_raw is None else ArtifactReference.from_json_dict(as_json_dict(search_summary_raw, field_name="search_summary_reference"))),
            evaluated_at=str(raw["evaluated_at"]),
        )


def _strip_early_stopping_keys(values: dict[str, JsonPrimitive]) -> dict[str, JsonPrimitive]:
    values.pop("early_stopping_rounds", None)
    values.pop("validation_fraction", None)
    return values


def finalize_outer_fold(
    *, optimization_spec: OptimizationSpec, outer_fold: Fold, winning_trial: TrialResult, timeline: pd.DataFrame,
    feature_universe: FeatureUniverse, model_factory: ModelFactory, serializer: ModelSerializer, artifact_store: MLArtifactStore,
    label_column: str = "label", search_summary_reference: ArtifactReference | None = None,
) -> OuterFoldResult:
    """The one function in this package permitted to read `outer_fold.
    test_indices`. Raises `ValueError` if `winning_trial` is not a valid
    (`COMPLETED`) candidate -- there is no meaningful "finalize the best
    invalid trial" case."""
    if not winning_trial.is_valid_candidate:
        raise ValueError(
            f"finalize_outer_fold requires a valid COMPLETED winning trial, got status={winning_trial.status.value!r}"
        )
    if winning_trial.outer_fold_index != outer_fold.fold_index:
        raise ValueError(
            f"winning_trial.outer_fold_index ({winning_trial.outer_fold_index}) does not match outer_fold.fold_index "
            f"({outer_fold.fold_index})"
        )

    started = time.perf_counter()
    outer_train_df = timeline.iloc[outer_fold.train_indices]
    outer_test_df = timeline.iloc[outer_fold.test_indices]
    outer_train_features_full = outer_train_df[list(feature_universe.feature_names)]
    outer_train_labels = outer_train_df[label_column]

    winning_hyperparameters = ModelHyperparameters(values=dict(winning_trial.sampled_hyperparameters))
    selector_seed = outer_train_feature_selector_seed(optimization_spec.seed_configuration, outer_fold.fold_index)
    fs_result = run_feature_selection(
        optimization_spec.feature_selection_spec, universe=feature_universe, features=outer_train_features_full,
        labels=outer_train_labels, row_positions=outer_fold.train_indices, seed=selector_seed,
        objective=optimization_spec.objective, model_name=optimization_spec.model_name, model_factory=model_factory,
        hyperparameters=winning_hyperparameters,
    )
    fs_ref = artifact_store.write_artifact(canonical_json_bytes(fs_result.to_json_dict()), category=ArtifactCategory.FEATURE_SELECTION_RESULT)
    final_features = list(fs_result.selected_features)

    final_values: dict[str, JsonPrimitive] = dict(winning_trial.sampled_hyperparameters)
    round_source: str | None = None
    if optimization_spec.model_name in GBM_MODEL_NAMES:
        rounds_key = next((k for k in _ROUNDS_KEYS if k in final_values), None)
        if rounds_key is not None:
            sampled_rounds = int(str(final_values[rounds_key]))
            decision = resolve_final_round_count(
                winning_trial, sampled_rounds=sampled_rounds, policy=optimization_spec.early_stopping_config.final_round_policy,
            )
            final_values[rounds_key] = decision.rounds
            round_source = decision.source
        final_values = _strip_early_stopping_keys(final_values)
    final_hyperparameters = ModelHyperparameters(values=final_values)

    refit_seed = outer_train_refit_seed(optimization_spec.seed_configuration, outer_fold.fold_index)
    feature_schema = FeatureSchema(feature_names=tuple(final_features))
    model = model_factory.create(hyperparameters=final_hyperparameters, feature_schema=feature_schema, objective=optimization_spec.objective)
    fitted = model.fit(outer_train_df[final_features], outer_train_labels, seeds=SeedConfiguration(master_seed=refit_seed))
    training_duration = time.perf_counter() - started

    model_ref = artifact_store.write_artifact(serializer.serialize(fitted), category=ArtifactCategory.MODEL)

    outer_test_features = outer_test_df[final_features]
    predictions = fitted.predict(outer_test_features)
    predictions_ref = artifact_store.write_artifact(
        canonical_json_bytes({"schema_version": 1, "predictions": [float(p) for p in predictions]}), category=ArtifactCategory.PREDICTIONS,
    )

    y_true = outer_test_df[label_column].to_numpy(dtype="float64")
    y_proba_positive = None
    probabilities_ref = None
    can_predict_proba = (
        optimization_spec.objective is not ObjectiveType.REGRESSION
        and fitted.metadata.capabilities.supports_predict_proba
        and isinstance(fitted, ProbabilisticPredictor)
    )
    if can_predict_proba:
        assert isinstance(fitted, ProbabilisticPredictor)
        proba = fitted.predict_proba(outer_test_features)
        class_labels = list(fitted.class_labels)
        positive_index = class_labels.index(1)
        y_proba_positive = proba[:, positive_index]
        probabilities_ref = artifact_store.write_artifact(
            canonical_json_bytes({
                "schema_version": 1, "class_labels": [str(c) for c in class_labels],
                "probabilities": [[float(v) for v in row] for row in proba],
            }),
            category=ArtifactCategory.PROBABILITIES,
        )

    metric_report = compute_metrics(optimization_spec.objective, y_true, predictions, y_proba_positive)

    return OuterFoldResult(
        schema_version=OUTER_FOLD_RESULT_SCHEMA_VERSION, optimization_id=winning_trial.optimization_id,
        outer_fold_index=outer_fold.fold_index, winning_trial_number=winning_trial.trial_number,
        final_selected_features=tuple(final_features), final_hyperparameters=final_hyperparameters.values,
        final_round_source=round_source, seed=refit_seed, training_duration_seconds=training_duration,
        outer_train_row_count=len(outer_fold.train_indices), outer_test_row_count=len(outer_fold.test_indices),
        outer_test_metrics=metric_report.values, feature_selection_result_reference=fs_ref, model_reference=model_ref,
        predictions_reference=predictions_ref, probabilities_reference=probabilities_ref,
        search_summary_reference=search_summary_reference, evaluated_at=format_utc_timestamp(utc_now()),
    )


__all__ = ["OUTER_FOLD_RESULT_SCHEMA_VERSION", "OuterFoldResult", "finalize_outer_fold"]
