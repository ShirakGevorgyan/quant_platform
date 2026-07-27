"""Milestone 4D: `OptimizationSpec` identity determinism/sensitivity,
`OptimizationStage` state machine legality, seed-derivation hierarchy
distinctness, and the Option A preprocessing-policy fail-closed gate."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest
from tests.unit.optimization.conftest import make_experiment_spec, make_optimization_spec

from quant_platform.ml.models import LabelBinding, LabelType, ModelCapabilities, ObjectiveType
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.feature_selection import FeatureSelectionSpec, FeatureSelectionStrategy
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    OptimizationStage,
    PreprocessingPolicy,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
    compute_optimization_identity,
    feature_selector_seed,
    inner_fold_seed,
    is_legal_optimization_transition,
    is_terminal_optimization_stage,
    model_fit_seed,
    outer_fold_seed,
    outer_train_feature_selector_seed,
    outer_train_refit_seed,
    sampler_seed,
    trial_seed,
    verify_optimization_identity,
)


class TestOptimizationIdentityDeterminism:
    def test_identical_specs_produce_identical_ids(self) -> None:
        spec_a = make_optimization_spec()
        spec_b = make_optimization_spec()
        assert compute_optimization_identity(spec_a) == compute_optimization_identity(spec_b)

    def test_identity_independent_of_tags_and_notes(self) -> None:
        spec_a = make_optimization_spec(tags=("a",), notes="hello")
        spec_b = make_optimization_spec(tags=("b", "c"), notes="goodbye")
        assert compute_optimization_identity(spec_a) == compute_optimization_identity(spec_b)

    def test_verify_optimization_identity_true_for_matching_spec(self) -> None:
        spec = make_optimization_spec()
        identity = compute_optimization_identity(spec)
        assert verify_optimization_identity(spec, identity)

    def test_verify_optimization_identity_false_for_tampered_identity(self) -> None:
        spec = make_optimization_spec()
        identity = compute_optimization_identity(spec)
        tampered = replace(identity, optimization_id="f" * 64)
        assert not verify_optimization_identity(spec, tampered)


class TestOptimizationIdentitySensitivity:
    """Every scientifically-relevant field change must change the id."""

    def test_max_trials_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(max_trials=5))
        changed = compute_optimization_identity(make_optimization_spec(max_trials=10))
        assert base != changed

    def test_sampler_kind_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(sampler_kind=SamplerKind.TPE))
        changed = compute_optimization_identity(make_optimization_spec(sampler_kind=SamplerKind.RANDOM))
        assert base != changed

    def test_feature_selection_strategy_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE)))
        changed = compute_optimization_identity(
            make_optimization_spec(feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}))
        )
        assert base != changed

    def test_seed_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(seed_configuration=SeedConfiguration(master_seed=1)))
        changed = compute_optimization_identity(make_optimization_spec(seed_configuration=SeedConfiguration(master_seed=2)))
        assert base != changed

    def test_pruning_config_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(pruning_config=PruningConfig(kind=PruningKind.NONE)))
        changed = compute_optimization_identity(make_optimization_spec(pruning_config=PruningConfig(kind=PruningKind.MEDIAN_STOPPING, min_completed_inner_folds=2)))
        assert base != changed

    def test_parent_experiment_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec())
        other_experiment = make_experiment_spec(feature_names=("f1", "f2", "f3", "f4", "f5", "f6"), notes="different experiment")
        changed_spec = make_optimization_spec(experiment=other_experiment, parent_experiment_id="e" * 64)
        assert base != compute_optimization_identity(changed_spec)

    def test_max_failed_trials_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(max_failed_trials=None))
        changed = compute_optimization_identity(make_optimization_spec(max_failed_trials=3))
        assert base != changed

    def test_timeout_seconds_change_changes_id(self) -> None:
        base = compute_optimization_identity(make_optimization_spec(timeout_seconds=None))
        changed = compute_optimization_identity(make_optimization_spec(timeout_seconds=60))
        assert base != changed


class TestMetricDirectionNeverTrustedFromCallerInput:
    def test_metric_direction_is_derived_not_hand_settable(self) -> None:
        spec = make_optimization_spec(primary_metric="rmse")
        assert spec.metric_direction == "minimize"
        spec2 = make_optimization_spec(primary_metric="accuracy")
        assert spec2.metric_direction == "maximize"

    def test_hand_constructing_a_contradictory_direction_is_rejected(self) -> None:
        spec = make_optimization_spec(primary_metric="accuracy")
        with pytest.raises(ValueError, match="metric_direction"):
            replace(spec, metric_direction="minimize")


class TestPreprocessingPolicyFailClosed:
    def test_only_exclude_scale_sensitive_policy_is_accepted(self) -> None:
        spec = make_optimization_spec()
        assert spec.preprocessing_policy is PreprocessingPolicy.EXCLUDE_SCALE_SENSITIVE

    def test_build_optimization_spec_rejects_scale_sensitive_model_when_registry_given(self) -> None:
        registry = ModelRegistry()
        registry.register(ModelDefinition(
            name="logistic_regression", version="1", description="scale-sensitive",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), requires_scaled_numeric_features=True),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        ))
        experiment = make_experiment_spec(
            feature_names=("f1", "f2"), objective=ObjectiveType.BINARY_CLASSIFICATION, primary_metric="accuracy",
            label_binding=LabelBinding(name="dir", kind="binary_direction", horizon_bars=5, label_type=LabelType.BINARY),
        )
        with pytest.raises(ValueError, match="requires scaled numeric features"):
            make_optimization_spec(experiment=experiment, model_name="logistic_regression", model_registry=registry)

    def test_build_optimization_spec_allows_non_scale_sensitive_model_with_registry(self) -> None:
        registry = ModelRegistry()
        registry.register(ModelDefinition(
            name="lightgbm", version="1", description="tree",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), requires_scaled_numeric_features=False),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        ))
        spec = make_optimization_spec(model_registry=registry)
        assert spec.model_name == "lightgbm"


class TestObjectiveCompatibilityEnforced:
    """Adversarial audit, Section 7: 'enforce model objective
    compatibility.' `build_optimization_spec` rejects a model whose
    declared `ModelCapabilities.supported_objectives` does not include the
    parent experiment's own objective -- checked AFTER the scale-
    sensitivity gate (so a model failing both reports the scale-
    sensitivity reason first, never masked)."""

    def test_regression_only_model_rejected_for_a_classification_experiment(self) -> None:
        registry = ModelRegistry()
        registry.register(ModelDefinition(
            name="regression_only_tree", version="1", description="regression-only",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), requires_scaled_numeric_features=False),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        ))
        experiment = make_experiment_spec(
            feature_names=("f1", "f2"), objective=ObjectiveType.BINARY_CLASSIFICATION, primary_metric="accuracy",
            label_binding=LabelBinding(name="dir", kind="binary_direction", horizon_bars=5, label_type=LabelType.BINARY),
        )
        with pytest.raises(ValueError, match="does not support objective"):
            make_optimization_spec(experiment=experiment, model_name="regression_only_tree", model_registry=registry, primary_metric="accuracy")

    def test_classification_only_model_rejected_for_a_regression_experiment(self) -> None:
        registry = ModelRegistry()
        registry.register(ModelDefinition(
            name="classification_only_tree", version="1", description="classification-only",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), requires_scaled_numeric_features=False),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        ))
        with pytest.raises(ValueError, match="does not support objective"):
            make_optimization_spec(model_name="classification_only_tree", model_registry=registry)  # default experiment is REGRESSION

    def test_matching_objective_is_accepted(self) -> None:
        registry = ModelRegistry()
        registry.register(ModelDefinition(
            name="both_objectives_tree", version="1", description="both",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION), requires_scaled_numeric_features=False),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        ))
        spec = make_optimization_spec(model_name="both_objectives_tree", model_registry=registry)
        assert spec.model_name == "both_objectives_tree"

    def test_no_registry_supplied_skips_the_check_entirely_documenting_the_opportunistic_layer(self) -> None:
        """Consistent with the scale-sensitivity gate's own documented
        design: this is the FRIENDLIER, opportunistic layer (fires only
        when a `ModelRegistry` happens to be supplied at spec-construction
        time) -- `trial_executor.run_trial`'s own probe-model check is the
        real, always-active gate for scale-sensitivity; objective
        compatibility has no equivalent always-active re-check inside
        `run_trial` (documented here so the asymmetry is a known,
        deliberate fact, not an oversight)."""
        spec = make_optimization_spec(model_name="anything_at_all", model_registry=None)
        assert spec.model_name == "anything_at_all"


class TestOptimizationStageStateMachine:
    def test_initializing_has_no_self_loop(self) -> None:
        assert not is_legal_optimization_transition(OptimizationStage.INITIALIZING, OptimizationStage.INITIALIZING)

    def test_running_outer_fold_self_loop_is_legal(self) -> None:
        assert is_legal_optimization_transition(OptimizationStage.RUNNING_OUTER_FOLD, OptimizationStage.RUNNING_OUTER_FOLD)

    def test_running_trial_self_loop_is_legal(self) -> None:
        assert is_legal_optimization_transition(OptimizationStage.RUNNING_TRIAL, OptimizationStage.RUNNING_TRIAL)

    @pytest.mark.parametrize("terminal", [OptimizationStage.COMPLETED, OptimizationStage.FAILED, OptimizationStage.CANCELLED])
    def test_terminal_stages_have_no_outgoing_transitions(self, terminal: OptimizationStage) -> None:
        assert is_terminal_optimization_stage(terminal)
        for target in OptimizationStage:
            assert not is_legal_optimization_transition(terminal, target)

    def test_mid_outer_fold_stages_can_recover(self) -> None:
        for stage in (
            OptimizationStage.RUNNING_TRIAL, OptimizationStage.BUILDING_INNER_PLAN, OptimizationStage.SELECTING_CANDIDATE,
            OptimizationStage.REFITTING_WINNER, OptimizationStage.EVALUATING_OUTER_TEST,
        ):
            assert is_legal_optimization_transition(stage, OptimizationStage.RECOVERABLE_FAILURE), stage

    def test_full_happy_path_sequence_is_legal(self) -> None:
        sequence = [
            OptimizationStage.INITIALIZING, OptimizationStage.LOADING_EXPERIMENT, OptimizationStage.BUILDING_OUTER_PLAN,
            OptimizationStage.RUNNING_OUTER_FOLD, OptimizationStage.BUILDING_INNER_PLAN, OptimizationStage.RUNNING_TRIAL,
            OptimizationStage.SELECTING_CANDIDATE, OptimizationStage.REFITTING_WINNER, OptimizationStage.EVALUATING_OUTER_TEST,
            OptimizationStage.STORING_RESULTS, OptimizationStage.COMPLETED,
        ]
        for current, target in itertools.pairwise(sequence):
            assert is_legal_optimization_transition(current, target), f"{current} -> {target}"


class TestSeedDerivationHierarchyDistinctness:
    """Every named seed-derivation function must produce values that are
    distinct from every OTHER function/index combination -- a collision
    would mean two supposedly-independent random processes share state."""

    def test_all_derived_seeds_for_one_configuration_are_pairwise_distinct(self) -> None:
        seeds = SeedConfiguration(master_seed=123)
        values = {
            "sampler": sampler_seed(seeds),
            "outer_fold_0": outer_fold_seed(seeds, 0),
            "outer_fold_1": outer_fold_seed(seeds, 1),
            "trial_0_0": trial_seed(seeds, 0, 0),
            "trial_0_1": trial_seed(seeds, 0, 1),
            "trial_1_0": trial_seed(seeds, 1, 0),
            "inner_fold_0_0_0": inner_fold_seed(seeds, 0, 0, 0),
            "inner_fold_0_0_1": inner_fold_seed(seeds, 0, 0, 1),
            "feature_selector_0_0_0": feature_selector_seed(seeds, 0, 0, 0),
            "model_fit_0_0_0": model_fit_seed(seeds, 0, 0, 0),
            "outer_train_refit_0": outer_train_refit_seed(seeds, 0),
            "outer_train_feature_selector_0": outer_train_feature_selector_seed(seeds, 0),
        }
        assert len(set(values.values())) == len(values), values

    def test_seed_derivation_is_deterministic_across_calls(self) -> None:
        seeds = SeedConfiguration(master_seed=999)
        assert trial_seed(seeds, 2, 3) == trial_seed(seeds, 2, 3)
        assert inner_fold_seed(seeds, 2, 3, 1) == inner_fold_seed(seeds, 2, 3, 1)

    def test_different_master_seeds_produce_different_derived_seeds(self) -> None:
        assert trial_seed(SeedConfiguration(master_seed=1), 0, 0) != trial_seed(SeedConfiguration(master_seed=2), 0, 0)

    def test_no_new_optimization_spec_touches_global_random_state(self) -> None:
        import random

        import numpy as np

        py_state_before = random.getstate()
        np_state_before = np.random.get_state()
        make_optimization_spec()
        assert random.getstate() == py_state_before
        np_state_after = np.random.get_state()
        assert np_state_after[1].tolist() == np_state_before[1].tolist()


class TestOptimizationSpecValidation:
    def test_max_trials_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_trials"):
            make_optimization_spec(max_trials=0)

    def test_min_successful_inner_folds_cannot_exceed_inner_n_splits(self) -> None:
        with pytest.raises(ValueError, match="min_successful_inner_folds"):
            make_optimization_spec(
                inner_split_config=make_optimization_spec().inner_split_config, min_successful_inner_folds=99,
            )

    def test_ranking_policy_version_must_match_current_code(self) -> None:
        spec = make_optimization_spec()
        with pytest.raises(ValueError, match="ranking_policy_version"):
            replace(spec, ranking_policy_version=999)

    def test_build_optimization_spec_rejects_incompatible_primary_metric_direction_is_never_reachable_by_hand(self) -> None:
        # build_optimization_spec always derives the correct direction --
        # this documents (never bypasses) that path is the only sanctioned one.
        spec = build_optimization_spec(
            experiment=make_experiment_spec(feature_names=("f1", "f2", "f3", "f4")),
            parent_experiment_id="a" * 64, model_name="lightgbm", model_version="1", primary_metric="rmse",
            inner_split_config=make_optimization_spec().inner_split_config,
            feature_selection_spec=FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE),
            search_space=make_optimization_spec().search_space, sampler_kind=SamplerKind.TPE,
            pruning_config=PruningConfig(kind=PruningKind.NONE), early_stopping_config=EarlyStoppingConfig(enabled=False),
            max_trials=5, min_successful_inner_folds=1, seed_configuration=SeedConfiguration(master_seed=1),
        )
        assert spec.metric_direction == "minimize"
