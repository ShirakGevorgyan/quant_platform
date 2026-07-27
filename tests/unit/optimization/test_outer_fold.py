"""Milestone 4D: outer-fold finalization -- `finalize_outer_fold` is the
ONLY function in this entire package permitted to read `Fold.test_indices`
(see that module's own docstring), so these tests focus on: rejecting an
invalid/mismatched winning trial, the final feature set being a fresh
refit on the COMPLETE outer-train partition (never a vote across inner
folds, and never touching test rows), the GBM final-round/early-stopping-
key-stripping policy, and `OuterFoldResult`'s own construction invariants."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tests.unit.optimization.conftest import make_optimization_spec

from quant_platform.execution.splitters import Fold
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.testing import ConstantTestModelFactory, ConstantTestModelSerializer
from quant_platform.optimization.candidates import InnerFoldTrialMetrics, TrialResult, TrialStatus
from quant_platform.optimization.feature_selection import (
    FeatureSelectionResult,
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
)
from quant_platform.optimization.models import EarlyStoppingConfig
from quant_platform.optimization.outer_fold import OuterFoldResult, finalize_outer_fold

_TS = pd.Timestamp("2024-01-01")


def _timeline(n_rows: int = 40, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows), "label": rng.normal(size=n_rows)})


def _feature_universe() -> FeatureUniverse:
    return FeatureUniverse(feature_names=("f1", "f2"), fingerprint="a" * 64)


def _fold(*, train: np.ndarray, test: np.ndarray, fold_index: int = 0) -> Fold:
    return Fold(fold_index=fold_index, train_indices=train, test_indices=test, train_start=_TS, train_end=_TS, test_start=_TS, test_end=_TS)


def _spec(**overrides: object):
    base: dict[str, object] = {
        "feature_selection_spec": FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE), "model_name": "constant_test_model",
    }
    base.update(overrides)
    return make_optimization_spec(**base)  # type: ignore[arg-type]


def _fold_metrics(best_iteration: int | None, *, index: int = 0) -> InnerFoldTrialMetrics:
    return InnerFoldTrialMetrics(
        inner_fold_index=index, primary_metric_value=0.5, secondary_metrics={}, selected_feature_count=2,
        feature_selection_result_reference=None, best_iteration=best_iteration, duration_seconds=0.1,
    )


def _winning_trial(*, outer_fold_index: int = 0, sampled_hyperparameters: dict[str, object] | None = None, inner_fold_metrics: tuple[InnerFoldTrialMetrics, ...] = (), status: TrialStatus = TrialStatus.COMPLETED, optimization_id: str = "a" * 64) -> TrialResult:
    kwargs: dict[str, object] = {
        "schema_version": 1, "optimization_id": optimization_id, "outer_fold_index": outer_fold_index, "trial_number": 3,
        "status": status, "sampled_hyperparameters": sampled_hyperparameters or {"alpha": 0.1},
        "inner_fold_metrics": inner_fold_metrics, "successful_inner_folds": max(len(inner_fold_metrics), 1),
        "total_inner_folds": max(len(inner_fold_metrics), 1), "duration_seconds": 1.0,
    }
    if status is TrialStatus.COMPLETED:
        kwargs["primary_metric_aggregate"] = 0.5
    else:
        kwargs["primary_metric_aggregate"] = None
        kwargs["failure_reason"] = "simulated"
    return TrialResult(**kwargs)  # type: ignore[arg-type]


def _finalize(tmp_path, *, spec=None, outer_fold=None, winning_trial=None) -> OuterFoldResult:
    return finalize_outer_fold(
        optimization_spec=spec or _spec(), outer_fold=outer_fold or _fold(train=np.arange(0, 30), test=np.arange(30, 40)),
        winning_trial=winning_trial or _winning_trial(), timeline=_timeline(), feature_universe=_feature_universe(),
        model_factory=ConstantTestModelFactory(), serializer=ConstantTestModelSerializer(), artifact_store=MLArtifactStore(tmp_path),
    )


class TestOuterFoldResultValidation:
    def _ref(self, category: ArtifactCategory = ArtifactCategory.MODEL) -> ArtifactReference:
        return ArtifactReference(category=category, content_hash="b" * 64, size_bytes=1, created_at="2024-01-01T00:00:00+00:00")

    def _base_kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "schema_version": 1, "optimization_id": "a" * 64, "outer_fold_index": 0, "winning_trial_number": 0,
            "final_selected_features": ("f1", "f2"), "final_hyperparameters": {}, "final_round_source": None, "seed": 1,
            "training_duration_seconds": 1.0, "outer_train_row_count": 10, "outer_test_row_count": 5,
            "outer_test_metrics": {"rmse": 0.1}, "feature_selection_result_reference": self._ref(ArtifactCategory.FEATURE_SELECTION_RESULT),
            "model_reference": self._ref(ArtifactCategory.MODEL), "predictions_reference": self._ref(ArtifactCategory.PREDICTIONS),
            "evaluated_at": "2024-01-01T00:00:00+00:00",
        }
        base.update(overrides)
        return base

    def test_empty_optimization_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="optimization_id"):
            OuterFoldResult(**self._base_kwargs(optimization_id=""))  # type: ignore[arg-type]

    def test_negative_outer_fold_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="outer_fold_index"):
            OuterFoldResult(**self._base_kwargs(outer_fold_index=-1))  # type: ignore[arg-type]

    def test_negative_winning_trial_number_rejected(self) -> None:
        with pytest.raises(ValueError, match="winning_trial_number"):
            OuterFoldResult(**self._base_kwargs(winning_trial_number=-1))  # type: ignore[arg-type]

    def test_empty_final_selected_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            OuterFoldResult(**self._base_kwargs(final_selected_features=()))  # type: ignore[arg-type]

    def test_duplicate_final_selected_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            OuterFoldResult(**self._base_kwargs(final_selected_features=("f1", "f1")))  # type: ignore[arg-type]

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            OuterFoldResult(**self._base_kwargs(seed=-1))  # type: ignore[arg-type]

    def test_negative_training_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="training_duration_seconds"):
            OuterFoldResult(**self._base_kwargs(training_duration_seconds=-1.0))  # type: ignore[arg-type]

    def test_zero_outer_train_row_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="outer_train_row_count"):
            OuterFoldResult(**self._base_kwargs(outer_train_row_count=0))  # type: ignore[arg-type]

    def test_zero_outer_test_row_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="outer_train_row_count"):
            OuterFoldResult(**self._base_kwargs(outer_test_row_count=0))  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        result = OuterFoldResult(**self._base_kwargs())  # type: ignore[arg-type]
        assert OuterFoldResult.from_json_dict(result.to_json_dict()) == result

    def test_round_trip_with_optional_references_present(self) -> None:
        result = OuterFoldResult(**self._base_kwargs(
            probabilities_reference=self._ref(ArtifactCategory.PROBABILITIES), search_summary_reference=self._ref(ArtifactCategory.SEARCH_SUMMARY),
        ))  # type: ignore[arg-type]
        assert OuterFoldResult.from_json_dict(result.to_json_dict()) == result


class TestFinalizeOuterFoldRejectsInvalidWinner:
    @pytest.mark.parametrize("status", [TrialStatus.FAILED, TrialStatus.INVALID, TrialStatus.PRUNED])
    def test_non_completed_winner_rejected(self, tmp_path, status: TrialStatus) -> None:
        with pytest.raises(ValueError, match="valid COMPLETED winning trial"):
            _finalize(tmp_path, winning_trial=_winning_trial(status=status))

    def test_outer_fold_index_mismatch_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="outer_fold_index"):
            _finalize(tmp_path, winning_trial=_winning_trial(outer_fold_index=1), outer_fold=_fold(train=np.arange(0, 30), test=np.arange(30, 40), fold_index=0))


class TestFinalizeOuterFoldHappyPath:
    def test_basic_result_shape(self, tmp_path) -> None:
        winning = _winning_trial()
        result = _finalize(tmp_path, winning_trial=winning)
        assert result.winning_trial_number == winning.trial_number
        assert result.optimization_id == winning.optimization_id
        assert result.outer_train_row_count == 30
        assert result.outer_test_row_count == 10
        assert result.outer_test_metrics  # real metrics were computed

    def test_none_strategy_selects_the_full_universe(self, tmp_path) -> None:
        result = _finalize(tmp_path, spec=_spec(feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE)))
        assert set(result.final_selected_features) == set(_feature_universe().feature_names)

    def test_model_and_predictions_artifacts_are_readable(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        result = _finalize(tmp_path)
        model_bytes = store.read_artifact(result.model_reference.content_hash)
        assert model_bytes
        predictions_raw = json.loads(store.read_artifact(result.predictions_reference.content_hash).decode("utf-8"))
        assert len(predictions_raw["predictions"]) == result.outer_test_row_count

    def test_regression_objective_never_writes_a_probabilities_artifact(self, tmp_path) -> None:
        result = _finalize(tmp_path)  # default make_optimization_spec() experiment is REGRESSION
        assert result.probabilities_reference is None

    def test_search_summary_reference_is_passed_through_when_given(self, tmp_path) -> None:
        summary_ref = ArtifactReference(category=ArtifactCategory.SEARCH_SUMMARY, content_hash="c" * 64, size_bytes=1, created_at="2024-01-01T00:00:00+00:00")
        result = finalize_outer_fold(
            optimization_spec=_spec(), outer_fold=_fold(train=np.arange(0, 30), test=np.arange(30, 40)), winning_trial=_winning_trial(),
            timeline=_timeline(), feature_universe=_feature_universe(), model_factory=ConstantTestModelFactory(),
            serializer=ConstantTestModelSerializer(), artifact_store=MLArtifactStore(tmp_path), search_summary_reference=summary_ref,
        )
        assert result.search_summary_reference == summary_ref


class TestFinalizeOuterFoldNeverTouchesTestRowsForFeatureSelection:
    """The concrete, evidence-based proof (not merely a documentation
    claim): the persisted `FeatureSelectionResult` artifact's own recorded
    training-row provenance must exactly match `outer_fold.train_indices`
    -- never extending into, or being computed from, `test_indices`."""

    def test_feature_selection_result_provenance_matches_outer_train_exactly(self, tmp_path) -> None:
        store = MLArtifactStore(tmp_path)
        train = np.arange(0, 30)
        test = np.arange(30, 40)
        result = _finalize(tmp_path, outer_fold=_fold(train=train, test=test))
        fs_result = FeatureSelectionResult.from_json_dict(json.loads(store.read_artifact(result.feature_selection_result_reference.content_hash).decode("utf-8")))
        assert fs_result.training_row_count == len(train)
        assert fs_result.training_row_first_position == int(train[0])
        assert fs_result.training_row_last_position == int(train[-1])


class TestFinalizeOuterFoldGbmRoundPolicy:
    def test_gbm_model_computes_median_best_iteration_and_strips_early_stopping_keys(self, tmp_path) -> None:
        winning = _winning_trial(
            sampled_hyperparameters={"num_boost_round": 500, "early_stopping_rounds": 20, "validation_fraction": 0.1},
            inner_fold_metrics=(_fold_metrics(40, index=0), _fold_metrics(60, index=1)),
        )
        spec = _spec(
            model_name="lightgbm", early_stopping_config=EarlyStoppingConfig(enabled=True, patience=20, validation_fraction=0.1, final_round_policy="median_best_iteration"),
        )
        result = _finalize(tmp_path, spec=spec, winning_trial=winning)
        assert result.final_hyperparameters["num_boost_round"] == 50  # median of [40, 60]
        assert "early_stopping_rounds" not in result.final_hyperparameters
        assert "validation_fraction" not in result.final_hyperparameters
        assert result.final_round_source == "median_best_iteration_across_inner_folds"

    def test_gbm_model_fixed_policy_ignores_best_iteration(self, tmp_path) -> None:
        winning = _winning_trial(sampled_hyperparameters={"num_boost_round": 500}, inner_fold_metrics=(_fold_metrics(40, index=0),))
        spec = _spec(model_name="lightgbm", early_stopping_config=EarlyStoppingConfig(enabled=True, patience=20, validation_fraction=0.1, final_round_policy="fixed"))
        result = _finalize(tmp_path, spec=spec, winning_trial=winning)
        assert result.final_hyperparameters["num_boost_round"] == 500
        assert result.final_round_source == "fixed_configured_rounds"

    def test_non_gbm_model_never_strips_or_computes_rounds(self, tmp_path) -> None:
        winning = _winning_trial(sampled_hyperparameters={"early_stopping_rounds": 20, "alpha": 0.1})
        result = _finalize(tmp_path, spec=_spec(model_name="constant_test_model"), winning_trial=winning)
        assert result.final_hyperparameters.get("early_stopping_rounds") == 20  # untouched -- not a GBM model name
        assert result.final_round_source is None
