"""Milestone 4D: isolated single-trial execution -- GBM early-stopping
injection, duck-typed best-iteration extraction, the REAL always-active
scale-sensitive/non-deterministic model gate (`run_trial` re-checks this
itself, unconditionally, regardless of whatever earlier check
`build_optimization_spec` already performed), per-inner-fold failure
isolation (a raised exception from one inner fold demotes only that fold,
never crashes the trial), pruning-callback early exit, and the
deterministic final-round-count policy used for GBM outer-train refits."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import TrialExecutionError
from quant_platform.execution.splitters import required_label_purge_bars_for
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.interfaces import FeatureSchema, ModelMetadata
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModel, ConstantTestModelFactory
from quant_platform.optimization.candidates import InnerFoldTrialMetrics, TrialResult, TrialSpec, TrialStatus
from quant_platform.optimization.feature_selection import (
    FeatureSelectionResult,
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
)
from quant_platform.optimization.inner_splits import INNER_SPLIT_SCHEMA_VERSION, InnerFold, InnerFoldPlan
from quant_platform.optimization.models import EarlyStoppingConfig
from quant_platform.optimization.search_space import LIGHTGBM_MODEL_NAME
from quant_platform.optimization.trial_executor import (
    GBM_MODEL_NAMES,
    FinalRoundDecision,
    _extract_best_iteration,
    _inject_early_stopping,
    _positive_class_probabilities,
    resolve_final_round_count,
    run_trial,
)

_TS = pd.Timestamp("2024-01-01")


def _timeline(n_rows: int = 40, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows), "label": rng.normal(size=n_rows),
    })


def _feature_universe() -> FeatureUniverse:
    return FeatureUniverse(feature_names=("f1", "f2"), fingerprint="a" * 64)


def _inner_fold(index: int, *, train: np.ndarray, validation: np.ndarray) -> InnerFold:
    return InnerFold(inner_fold_index=index, train_indices=train, validation_indices=validation, train_start=_TS, train_end=_TS, validation_start=_TS, validation_end=_TS)


def _inner_plan(*folds: InnerFold, outer_fold_index: int = 0, outer_train_row_count: int = 40) -> InnerFoldPlan:
    label_horizon_bars = 1
    purge = required_label_purge_bars_for(label_horizon_bars)
    return InnerFoldPlan(
        schema_version=INNER_SPLIT_SCHEMA_VERSION, outer_fold_index=outer_fold_index, strategy="expanding_walk_forward",
        inner_folds=tuple(folds), purge_bars=purge, embargo_bars=0, label_horizon_bars=label_horizon_bars,
        required_label_purge_bars=purge, outer_train_row_count=outer_train_row_count,
    )


def _two_fold_plan() -> InnerFoldPlan:
    return _inner_plan(
        _inner_fold(0, train=np.arange(0, 20), validation=np.arange(20, 25)),
        _inner_fold(1, train=np.arange(0, 30), validation=np.arange(30, 35)),
    )


def _trial_spec(*, trial_number: int = 0, feature_selection_spec: FeatureSelectionSpec | None = None, primary_metric: str = "rmse") -> TrialSpec:
    """`ConstantTestModel` always predicts the training label MEAN (see
    `ml.testing`'s own module docstring) -- a continuous value, never a
    thresholded 0/1 class label -- so these `run_trial`-level tests use a
    REGRESSION objective, exactly what that test double is designed for.
    Classification-specific behavior (`_positive_class_probabilities`) is
    covered directly, above, without going through `run_trial`/`compute_
    metrics` at all."""
    return TrialSpec(
        schema_version=1, optimization_id="a" * 64, outer_fold_index=0, trial_number=trial_number,
        sampled_hyperparameters={"alpha": 0.1}, feature_selection_spec=feature_selection_spec or FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE),
        trial_seed=1, inner_split_plan_fingerprint="b" * 64, model_definition_fingerprint="c" * 64,
        objective=ObjectiveType.REGRESSION, primary_metric=primary_metric,
    )


def _run(tmp_path, *, model_factory=None, inner_fold_plan=None, min_successful_inner_folds: int = 1, pruning_callback=None, trial_spec=None) -> TrialResult:
    return run_trial(
        trial_spec or _trial_spec(),
        inner_fold_plan=inner_fold_plan or _two_fold_plan(),
        timeline=_timeline(),
        feature_universe=_feature_universe(),
        model_name="constant_test_model",
        model_factory=model_factory or ConstantTestModelFactory(),
        seed_configuration=SeedConfiguration(master_seed=1),
        min_successful_inner_folds=min_successful_inner_folds,
        early_stopping_config=EarlyStoppingConfig(enabled=False),
        artifact_store=MLArtifactStore(tmp_path),
        pruning_callback=pruning_callback,
    )


class _CapabilityOverrideFactory:
    """Wraps `ConstantTestModelFactory` but overrides declared capabilities
    -- used to drive `run_trial`'s own always-active gate checks without
    needing a real scale-sensitive/non-deterministic model implementation."""

    def __init__(self, **capability_overrides: object) -> None:
        self._overrides = capability_overrides

    def create(self, *, hyperparameters, feature_schema, objective):
        base = ConstantTestModelFactory().create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
        capabilities = ModelCapabilities(supported_objectives=base.metadata.capabilities.supported_objectives, supports_predict_proba=True, **self._overrides)  # type: ignore[arg-type]
        metadata = ModelMetadata(
            name=base.metadata.name, version=base.metadata.version, objective=objective, feature_schema=feature_schema,
            capabilities=capabilities, hyperparameters=hyperparameters,
        )
        return ConstantTestModel(metadata=metadata)


@dataclass(frozen=True, slots=True)
class _AlwaysFailingModel:
    metadata: ModelMetadata

    def fit(self, features, labels, *, seeds):
        raise RuntimeError("simulated fit failure")


class _CountingModelFactory:
    """Counts every `create()` call (one per probe + one per attempted
    inner fold) so tests can prove pruning/failure-isolation stop the loop
    at exactly the expected point, never running extra inner folds."""

    def __init__(self, *, always_fail: bool = False) -> None:
        self.create_calls = 0
        self._always_fail = always_fail

    def create(self, *, hyperparameters, feature_schema, objective):
        self.create_calls += 1
        delegate = ConstantTestModelFactory().create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
        if self._always_fail:
            return _AlwaysFailingModel(metadata=delegate.metadata)
        return delegate


class TestInjectEarlyStopping:
    def test_gbm_model_enabled_injects_keys(self) -> None:
        config = EarlyStoppingConfig(enabled=True, patience=10, validation_fraction=0.2)
        result = _inject_early_stopping({"num_leaves": 5}, model_name=LIGHTGBM_MODEL_NAME, early_stopping_config=config)
        assert result == {"num_leaves": 5, "early_stopping_rounds": 10, "validation_fraction": 0.2}

    def test_gbm_model_disabled_unchanged(self) -> None:
        config = EarlyStoppingConfig(enabled=False)
        result = _inject_early_stopping({"num_leaves": 5}, model_name=LIGHTGBM_MODEL_NAME, early_stopping_config=config)
        assert result == {"num_leaves": 5}

    def test_non_gbm_model_enabled_unchanged(self) -> None:
        config = EarlyStoppingConfig(enabled=True, patience=10, validation_fraction=0.2)
        result = _inject_early_stopping({"alpha": 0.1}, model_name="constant_test_model", early_stopping_config=config)
        assert result == {"alpha": 0.1}

    def test_original_dict_never_mutated(self) -> None:
        original = {"num_leaves": 5}
        config = EarlyStoppingConfig(enabled=True, patience=10, validation_fraction=0.2)
        _inject_early_stopping(original, model_name=LIGHTGBM_MODEL_NAME, early_stopping_config=config)
        assert original == {"num_leaves": 5}

    def test_every_gbm_model_name_recognized(self) -> None:
        config = EarlyStoppingConfig(enabled=True, patience=1, validation_fraction=0.1)
        for name in GBM_MODEL_NAMES:
            result = _inject_early_stopping({}, model_name=name, early_stopping_config=config)
            assert "early_stopping_rounds" in result


class TestExtractBestIteration:
    def test_positive_value_extracted(self) -> None:
        assert _extract_best_iteration(type("F", (), {"best_iteration": 42})()) == 42

    def test_lightgbm_zero_sentinel_normalized_to_none(self) -> None:
        assert _extract_best_iteration(type("F", (), {"best_iteration": 0})()) is None

    def test_missing_attribute_is_none(self) -> None:
        assert _extract_best_iteration(object()) is None

    def test_negative_value_is_none(self) -> None:
        assert _extract_best_iteration(type("F", (), {"best_iteration": -5})()) is None

    def test_bool_value_is_none_despite_bool_being_an_int_subclass(self) -> None:
        assert _extract_best_iteration(type("F", (), {"best_iteration": True})()) is None


class TestPositiveClassProbabilities:
    def test_regression_objective_always_none(self) -> None:
        factory = ConstantTestModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={}), feature_schema=FeatureSchema(feature_names=("f1",)), objective=ObjectiveType.REGRESSION)
        fitted = model.fit(pd.DataFrame({"f1": [1.0, 2.0]}), pd.Series([1.0, 2.0]), seeds=SeedConfiguration(master_seed=1))
        assert _positive_class_probabilities(fitted, pd.DataFrame({"f1": [1.0]}), ObjectiveType.REGRESSION) is None

    def test_classification_with_probabilistic_model_returns_positive_column(self) -> None:
        factory = ConstantTestModelFactory()
        schema = FeatureSchema(feature_names=("f1",))
        model = factory.create(hyperparameters=ModelHyperparameters(values={}), feature_schema=schema, objective=ObjectiveType.BINARY_CLASSIFICATION)
        labels = pd.Series([0.0, 1.0, 1.0, 1.0])
        fitted = model.fit(pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}), labels, seeds=SeedConfiguration(master_seed=1))
        proba = _positive_class_probabilities(fitted, pd.DataFrame({"f1": [1.0]}), ObjectiveType.BINARY_CLASSIFICATION)
        assert proba is not None
        assert proba[0] == pytest.approx(labels.mean())

    def test_non_probabilistic_model_returns_none(self) -> None:
        schema = FeatureSchema(feature_names=("f1",))
        capabilities = ModelCapabilities(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), supports_predict_proba=False)
        metadata = ModelMetadata(name="np", version="1", objective=ObjectiveType.BINARY_CLASSIFICATION, feature_schema=schema, capabilities=capabilities, hyperparameters=ModelHyperparameters(values={}))
        model = ConstantTestModel(metadata=metadata)
        fitted = model.fit(pd.DataFrame({"f1": [1.0, 2.0]}), pd.Series([0.0, 1.0]), seeds=SeedConfiguration(master_seed=1))
        assert _positive_class_probabilities(fitted, pd.DataFrame({"f1": [1.0]}), ObjectiveType.BINARY_CLASSIFICATION) is None


def _fold_metrics(best_iteration: int | None, *, index: int = 0) -> InnerFoldTrialMetrics:
    return InnerFoldTrialMetrics(
        inner_fold_index=index, primary_metric_value=0.5, secondary_metrics={}, selected_feature_count=2,
        feature_selection_result_reference=None, best_iteration=best_iteration, duration_seconds=0.1,
    )


def _trial_result_with(metrics: tuple[InnerFoldTrialMetrics, ...]) -> TrialResult:
    return TrialResult(
        schema_version=1, optimization_id="a" * 64, outer_fold_index=0, trial_number=0, status=TrialStatus.COMPLETED,
        sampled_hyperparameters={"num_boost_round": 200}, inner_fold_metrics=metrics, primary_metric_aggregate=0.5,
        successful_inner_folds=len(metrics), total_inner_folds=len(metrics), duration_seconds=1.0,
    )


class TestResolveFinalRoundCount:
    def test_fixed_policy_always_uses_sampled_rounds(self) -> None:
        result = _trial_result_with((_fold_metrics(999, index=0),))
        decision = resolve_final_round_count(result, sampled_rounds=42, policy="fixed")
        assert decision.rounds == 42
        assert decision.source == "fixed_configured_rounds"

    def test_median_policy_falls_back_when_no_best_iteration_available(self) -> None:
        result = _trial_result_with((_fold_metrics(None, index=0), _fold_metrics(None, index=1)))
        decision = resolve_final_round_count(result, sampled_rounds=77, policy="median_best_iteration")
        assert decision.rounds == 77
        assert decision.source == "fixed_configured_rounds_no_best_iteration_available"

    def test_median_policy_odd_count_uses_exact_median(self) -> None:
        result = _trial_result_with((_fold_metrics(10, index=0), _fold_metrics(20, index=1), _fold_metrics(30, index=2)))
        decision = resolve_final_round_count(result, sampled_rounds=999, policy="median_best_iteration")
        assert decision.rounds == 20
        assert decision.source == "median_best_iteration_across_inner_folds"

    def test_median_policy_even_count_uses_rounded_average(self) -> None:
        result = _trial_result_with((_fold_metrics(10, index=0), _fold_metrics(21, index=1)))
        decision = resolve_final_round_count(result, sampled_rounds=999, policy="median_best_iteration")
        assert decision.rounds == round((10 + 21) / 2)

    def test_median_policy_ignores_folds_with_no_best_iteration(self) -> None:
        result = _trial_result_with((_fold_metrics(10, index=0), _fold_metrics(None, index=1), _fold_metrics(30, index=2)))
        decision = resolve_final_round_count(result, sampled_rounds=999, policy="median_best_iteration")
        assert decision.rounds == 20  # median of [10, 30], the None-iteration fold excluded

    def test_median_policy_floors_at_one_round(self) -> None:
        result = _trial_result_with((_fold_metrics(0, index=0),))
        decision = resolve_final_round_count(result, sampled_rounds=999, policy="median_best_iteration")
        assert decision.rounds == 1

    def test_rejects_the_winner_when_fixed_policy_cannot_produce_a_valid_final_round(self) -> None:
        """Adversarial audit, Section 3: 'Reject a winner when the declared
        policy cannot produce a valid final round.' `policy='fixed'` used
        `sampled_rounds` completely unvalidated -- a misconfigured search
        space (or a corrupted/tampered sampled_hyperparameters value) with
        `num_boost_round <= 0` would silently flow into a real model fit
        call as an invalid round count instead of being rejected here."""
        result = _trial_result_with((_fold_metrics(40, index=0),))
        with pytest.raises(TrialExecutionError, match="not a valid positive boosting-round count"):
            resolve_final_round_count(result, sampled_rounds=0, policy="fixed")
        with pytest.raises(TrialExecutionError, match="not a valid positive boosting-round count"):
            resolve_final_round_count(result, sampled_rounds=-5, policy="fixed")

    def test_rejects_the_winner_when_median_policy_falls_back_to_an_invalid_sampled_rounds(self) -> None:
        """The SAME invalid `sampled_rounds` rejection applies to
        `median_best_iteration`'s own fallback path (no inner fold
        reported a best_iteration) -- not merely the 'fixed' policy."""
        result = _trial_result_with((_fold_metrics(None, index=0),))
        with pytest.raises(TrialExecutionError, match="not a valid positive boosting-round count"):
            resolve_final_round_count(result, sampled_rounds=0, policy="median_best_iteration")

    def test_final_round_decision_itself_rejects_a_non_positive_round_count(self) -> None:
        """Defense in depth at the type level, matching every other
        durable result type in this package (e.g. `FeatureSelectionResult.
        per_feature_score`'s finiteness check) -- `FinalRoundDecision`
        cannot be constructed with an invalid round count even by a future
        caller that bypasses `resolve_final_round_count` entirely."""
        with pytest.raises(ValueError, match="rounds must be"):
            FinalRoundDecision(rounds=0, source="fixed_configured_rounds")
        with pytest.raises(ValueError, match="rounds must be"):
            FinalRoundDecision(rounds=-3, source="fixed_configured_rounds")


class TestRunTrialAlwaysActiveCapabilityGate:
    """The REAL, unconditional enforcement of Option A's preprocessing
    policy and the determinism requirement -- re-checked here regardless
    of whatever `build_optimization_spec`'s own, earlier, friendlier
    check already concluded (that check is opportunistic: it only fires
    when a `ModelRegistry` happens to be supplied at spec-construction
    time)."""

    def test_scale_sensitive_model_rejected_before_any_inner_fold_runs(self, tmp_path) -> None:
        factory = _CapabilityOverrideFactory(requires_scaled_numeric_features=True)
        with pytest.raises(TrialExecutionError, match="requires scaled numeric features"):
            _run(tmp_path, model_factory=factory)

    def test_non_deterministic_model_rejected_before_any_inner_fold_runs(self, tmp_path) -> None:
        factory = _CapabilityOverrideFactory(is_deterministic=False)
        with pytest.raises(TrialExecutionError, match="is_deterministic"):
            _run(tmp_path, model_factory=factory)

    def test_compliant_model_is_not_rejected(self, tmp_path) -> None:
        result = _run(tmp_path, model_factory=_CapabilityOverrideFactory())
        assert result.status in (TrialStatus.COMPLETED, TrialStatus.INVALID)  # reached inner-fold execution, not gate-rejected


class TestRunTrialPerInnerFoldFailureIsolation:
    def test_a_failing_inner_fold_never_crashes_the_trial(self, tmp_path) -> None:
        factory = _CountingModelFactory(always_fail=True)
        result = _run(tmp_path, model_factory=factory, min_successful_inner_folds=1)
        assert result.status is TrialStatus.INVALID
        assert result.failure_code == "insufficient_successful_inner_folds"
        assert all(m.primary_metric_value is None for m in result.inner_fold_metrics)
        assert len(result.inner_fold_metrics) == 2  # both inner folds were attempted, neither crashed the process
        assert factory.create_calls == 1 + 2  # 1 probe + 2 inner-fold attempts

    def test_warnings_accumulate_one_entry_per_failed_inner_fold(self, tmp_path) -> None:
        factory = _CountingModelFactory(always_fail=True)
        result = _run(tmp_path, model_factory=factory)
        assert len(result.warnings) == 2
        assert all("did not produce primary metric" in w for w in result.warnings)

    def test_status_completed_when_enough_inner_folds_succeed_despite_one_being_theoretically_possible_to_fail(self, tmp_path) -> None:
        result = _run(tmp_path, model_factory=ConstantTestModelFactory(), min_successful_inner_folds=1)
        assert result.status is TrialStatus.COMPLETED
        assert result.successful_inner_folds == 2
        assert result.primary_metric_aggregate is not None


class TestRunTrialPruning:
    def test_pruning_callback_stops_the_loop_and_marks_pruned(self, tmp_path) -> None:
        factory = _CountingModelFactory()
        result = _run(tmp_path, model_factory=factory, pruning_callback=lambda metrics_so_far: len(metrics_so_far) >= 1)
        assert result.status is TrialStatus.PRUNED
        assert result.failure_code == "pruned"
        assert len(result.inner_fold_metrics) == 1  # loop broke after the first inner fold
        assert factory.create_calls == 1 + 1  # 1 probe + only the first inner fold, never the second

    def test_pruning_callback_never_called_when_none(self, tmp_path) -> None:
        result = _run(tmp_path, model_factory=ConstantTestModelFactory(), pruning_callback=None)
        assert result.status is not TrialStatus.PRUNED

    def test_pruned_trial_aggregates_only_folds_that_actually_ran(self, tmp_path) -> None:
        factory = _CountingModelFactory()
        result = _run(tmp_path, model_factory=factory, pruning_callback=lambda metrics_so_far: len(metrics_so_far) >= 1)
        assert result.primary_metric_aggregate == pytest.approx(result.inner_fold_metrics[0].primary_metric_value)


class TestRunTrialDeterminism:
    def test_two_calls_with_identical_inputs_produce_identical_outcomes(self, tmp_path) -> None:
        result_a = _run(tmp_path, model_factory=ConstantTestModelFactory())
        result_b = _run(tmp_path, model_factory=ConstantTestModelFactory())
        assert result_a.status == result_b.status
        assert result_a.sampled_hyperparameters == result_b.sampled_hyperparameters
        assert result_a.primary_metric_aggregate == result_b.primary_metric_aggregate
        assert result_a.successful_inner_folds == result_b.successful_inner_folds
        for m_a, m_b in zip(result_a.inner_fold_metrics, result_b.inner_fold_metrics, strict=True):
            assert m_a.primary_metric_value == m_b.primary_metric_value
            assert m_a.selected_feature_count == m_b.selected_feature_count
            assert m_a.best_iteration == m_b.best_iteration

    def test_two_calls_select_identical_features(self, tmp_path) -> None:
        """The written `FeatureSelectionResult` ARTIFACTS themselves are
        not expected to be byte-identical across two separate calls --
        each embeds its own real `fitted_at` wall-clock timestamp (an
        audit field, by design) -- but the SELECTION they record must be,
        since selection is a pure function of the (identical) data/seed."""
        store = MLArtifactStore(tmp_path)
        result_a = _run(tmp_path, model_factory=ConstantTestModelFactory())
        result_b = _run(tmp_path, model_factory=ConstantTestModelFactory())

        def _selected_feature_sets(result: TrialResult) -> list[tuple[str, ...]]:
            selections = []
            for ref in result.artifact_references:
                decoded = FeatureSelectionResult.from_json_dict(json.loads(store.read_artifact(ref.content_hash).decode("utf-8")))
                selections.append(decoded.selected_features)
            return selections

        assert _selected_feature_sets(result_a) == _selected_feature_sets(result_b)
