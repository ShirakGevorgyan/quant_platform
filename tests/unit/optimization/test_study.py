"""Milestone 4D: Optuna integration -- deterministic sampler construction,
typed-`SearchSpace`-to-suggestion translation, and the empirically critical
claim that replaying history via `ask()`/`tell()` reproduces byte-identical
subsequent suggestions to an uninterrupted run (see `optimization.study`'s
module docstring for why the naive `study.add_trial(...)` replay approach
does NOT work). Also covers the platform-owned median-stopping pruning
rule, which deliberately never touches Optuna's own `trial.report()`/
`optuna.pruners`."""

from __future__ import annotations

import random

import numpy as np
import optuna
import pytest
from tests.unit.optimization.conftest import (
    DEFAULT_FEATURE_NAMES,
    make_experiment_spec,
    make_optimization_spec,
)

from quant_platform.core.exceptions import OptimizationResumeError
from quant_platform.ml.models import LabelBinding, LabelType, ObjectiveType
from quant_platform.optimization.candidates import InnerFoldTrialMetrics, TrialStatus
from quant_platform.optimization.models import PruningConfig, PruningKind, SamplerKind
from quant_platform.optimization.search_space import (
    BooleanParameter,
    CategoricalParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    build_search_space,
)
from quant_platform.optimization.study import (
    HistoricalTrialRecord,
    ask_next_trial,
    build_sampler,
    create_study,
    evaluate_median_stopping,
    rebuild_study_from_history,
    replay_trial,
    suggest_hyperparameters,
    tell_trial_outcome,
)


def _mixed_space():
    return build_search_space([
        IntegerParameter(name="num_leaves", low=4, high=64),
        FloatParameter(name="learning_rate", low=1e-4, high=1.0, log=True),
        CategoricalParameter(name="boosting", choices=("gbdt", "dart", "goss")),
        BooleanParameter(name="use_bagging"),
        FixedParameter(name="verbosity", value=-1),
    ])


def _classification_optimization_spec(**overrides: object):
    experiment = make_experiment_spec(
        feature_names=DEFAULT_FEATURE_NAMES, objective=ObjectiveType.BINARY_CLASSIFICATION,
        label_binding=LabelBinding(name="dir", kind="binary_direction", horizon_bars=5, label_type=LabelType.BINARY),
    )
    base: dict[str, object] = {"experiment": experiment, "primary_metric": "accuracy"}
    base.update(overrides)
    return make_optimization_spec(**base)  # type: ignore[arg-type]


def _metrics(value: float | None, *, inner_fold_index: int = 0) -> InnerFoldTrialMetrics:
    return InnerFoldTrialMetrics(
        inner_fold_index=inner_fold_index, primary_metric_value=value, secondary_metrics={}, selected_feature_count=3,
        feature_selection_result_reference=None, best_iteration=None, duration_seconds=0.1,
    )


class TestBuildSampler:
    def test_tpe_kind(self) -> None:
        assert isinstance(build_sampler(SamplerKind.TPE.value, seed=1), optuna.samplers.TPESampler)

    def test_random_kind(self) -> None:
        assert isinstance(build_sampler(SamplerKind.RANDOM.value, seed=1), optuna.samplers.RandomSampler)

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown sampler kind"):
            build_sampler("not_a_real_sampler", seed=1)


class TestCreateStudy:
    def test_direction_matches_spec_regardless_of_which_direction(self) -> None:
        minimize_spec = make_optimization_spec()  # default experiment is REGRESSION -> primary_metric=rmse -> minimize
        assert minimize_spec.metric_direction == "minimize"
        assert create_study(minimize_spec).direction.name.lower() == "minimize"

        maximize_spec = _classification_optimization_spec()
        assert maximize_spec.metric_direction == "maximize"
        assert create_study(maximize_spec).direction.name.lower() == "maximize"

    def test_sampler_kind_is_honored(self) -> None:
        spec = make_optimization_spec(sampler_kind=SamplerKind.RANDOM)
        study = create_study(spec)
        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    def test_uses_no_persistent_storage(self) -> None:
        study = create_study(make_optimization_spec())
        # Optuna's default in-memory storage class -- never a file/RDB URL.
        assert type(study._storage).__name__ == "InMemoryStorage"


class TestSuggestHyperparameters:
    def test_returns_every_declared_parameter_in_declared_order(self) -> None:
        space = _mixed_space()
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        trial = study.ask()
        values = suggest_hyperparameters(trial, space)
        assert list(values.keys()) == ["num_leaves", "learning_rate", "boosting", "use_bagging", "verbosity"]

    def test_values_are_within_declared_bounds(self) -> None:
        space = _mixed_space()
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=3))
        for _ in range(20):
            trial = study.ask()
            values = suggest_hyperparameters(trial, space)
            assert 4 <= values["num_leaves"] <= 64
            assert 1e-4 <= values["learning_rate"] <= 1.0
            assert values["boosting"] in ("gbdt", "dart", "goss")
            assert isinstance(values["use_bagging"], bool)
            assert values["verbosity"] == -1
            study.tell(trial, 0.5)

    def test_fixed_parameter_never_consumes_sampler_randomness(self) -> None:
        """Two studies differing only in whether a FixedParameter is
        present must sample IDENTICAL values for every other (real)
        parameter, proving FixedParameter never touches the RNG."""
        space_without_fixed = build_search_space([IntegerParameter(name="n", low=1, high=100)])
        space_with_fixed = build_search_space([IntegerParameter(name="n", low=1, high=100), FixedParameter(name="extra", value="x")])
        study_a = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=99))
        study_b = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=99))
        for _ in range(5):
            trial_a = study_a.ask()
            values_a = suggest_hyperparameters(trial_a, space_without_fixed)
            study_a.tell(trial_a, 0.1)
            trial_b = study_b.ask()
            values_b = suggest_hyperparameters(trial_b, space_with_fixed)
            study_b.tell(trial_b, 0.1)
            assert values_a["n"] == values_b["n"]


class TestAskNextTrialAndTellTrialOutcome:
    def test_ask_next_trial_assigns_sequential_trial_numbers(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        space = build_search_space([IntegerParameter(name="n", low=1, high=10)])
        for expected_number in range(3):
            trial, _values = ask_next_trial(study, space)
            assert trial.number == expected_number
            tell_trial_outcome(study, trial, status=TrialStatus.COMPLETED, value=0.5)

    def test_tell_completed_without_value_rejected(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        trial = study.ask()
        with pytest.raises(ValueError, match="requires a non-None value"):
            tell_trial_outcome(study, trial, status=TrialStatus.COMPLETED, value=None)

    def test_tell_pruned_records_pruned_state(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        trial = study.ask()
        tell_trial_outcome(study, trial, status=TrialStatus.PRUNED, value=None)
        assert study.trials[0].state is optuna.trial.TrialState.PRUNED

    def test_tell_failed_records_fail_state(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        trial = study.ask()
        tell_trial_outcome(study, trial, status=TrialStatus.FAILED, value=None)
        assert study.trials[0].state is optuna.trial.TrialState.FAIL

    def test_tell_invalid_also_records_fail_state(self) -> None:
        """INVALID has no dedicated Optuna TrialState -- it must never be
        told with a fabricated numeric value, so it maps to FAIL exactly
        like FAILED."""
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        trial = study.ask()
        tell_trial_outcome(study, trial, status=TrialStatus.INVALID, value=None)
        assert study.trials[0].state is optuna.trial.TrialState.FAIL


def _run_uninterrupted(spec, space, *, n_trials: int, statuses: list[TrialStatus]) -> list[dict[str, object]]:
    study = create_study(spec)
    sampled: list[dict[str, object]] = []
    for i in range(n_trials):
        trial, values = ask_next_trial(study, space)
        sampled.append(dict(values))
        status = statuses[i]
        value = 0.5 + (i * 0.01) if status is TrialStatus.COMPLETED else None
        tell_trial_outcome(study, trial, status=status, value=value)
    return sampled


class TestDeterministicResumeAcrossSamplers:
    """THE empirically load-bearing claim this module exists to prove:
    replaying every historical trial via `ask()`/`tell()` (never
    `study.add_trial`) reproduces byte-identical subsequent suggestions to
    an uninterrupted run -- for both samplers this platform supports, and
    across COMPLETE/FAIL/PRUNED historical trial states."""

    @pytest.mark.parametrize("sampler_kind", [SamplerKind.TPE, SamplerKind.RANDOM])
    def test_resuming_after_every_possible_split_point_matches_uninterrupted_run(self, sampler_kind: SamplerKind) -> None:
        space = _mixed_space()
        spec = make_optimization_spec(sampler_kind=sampler_kind, search_space=space)
        total_trials = 10
        statuses = [TrialStatus.COMPLETED, TrialStatus.FAILED, TrialStatus.PRUNED, TrialStatus.COMPLETED, TrialStatus.INVALID] * 2
        uninterrupted = _run_uninterrupted(spec, space, n_trials=total_trials, statuses=statuses)

        for split_point in range(1, total_trials):
            history = [
                HistoricalTrialRecord(
                    trial_number=i, sampled_hyperparameters=uninterrupted[i], status=statuses[i],
                    primary_metric_aggregate=(0.5 + i * 0.01) if statuses[i] is TrialStatus.COMPLETED else None,
                )
                for i in range(split_point)
            ]
            resumed_study = rebuild_study_from_history(spec, history)
            for i in range(split_point, total_trials):
                trial, values = ask_next_trial(resumed_study, space)
                assert trial.number == i
                assert dict(values) == uninterrupted[i], (
                    f"sampler={sampler_kind.value} split_point={split_point}: trial {i} diverged after resume"
                )
                status = statuses[i]
                value = 0.5 + (i * 0.01) if status is TrialStatus.COMPLETED else None
                tell_trial_outcome(resumed_study, trial, status=status, value=value)

    def test_resume_is_insensitive_to_history_ordering_in_the_input_sequence(self) -> None:
        """`rebuild_study_from_history` sorts by `trial_number` itself --
        callers must not be required to pass history in order."""
        space = build_search_space([IntegerParameter(name="n", low=1, high=1000)])
        spec = make_optimization_spec(search_space=space)
        uninterrupted = _run_uninterrupted(spec, space, n_trials=4, statuses=[TrialStatus.COMPLETED] * 4)
        history = [
            HistoricalTrialRecord(trial_number=i, sampled_hyperparameters=uninterrupted[i], status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5 + i * 0.01)
            for i in (2, 0, 3, 1)  # deliberately shuffled
        ]
        resumed = rebuild_study_from_history(spec, history)
        trial, values = ask_next_trial(resumed, space)
        assert trial.number == 4
        # Sanity: still able to compute a valid next suggestion (main
        # assertion is that this doesn't raise or silently misorder).
        assert 1 <= values["n"] <= 1000


class TestReplayTrialMismatchDetection:
    def test_mismatched_sampled_hyperparameters_raises(self) -> None:
        space = build_search_space([IntegerParameter(name="n", low=1, high=1000)])
        spec = make_optimization_spec()
        study = create_study(spec)
        bogus_record = HistoricalTrialRecord(trial_number=0, sampled_hyperparameters={"n": -999}, status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5)
        with pytest.raises(OptimizationResumeError, match="Deterministic sampler resume failed"):
            replay_trial(study, space, bogus_record)

    def test_out_of_order_trial_number_raises(self) -> None:
        space = build_search_space([IntegerParameter(name="n", low=1, high=1000)])
        spec = make_optimization_spec()
        study = create_study(spec)
        bogus_record = HistoricalTrialRecord(trial_number=5, sampled_hyperparameters={"n": 1}, status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5)
        with pytest.raises(OptimizationResumeError, match="Replay trial-number mismatch"):
            replay_trial(study, space, bogus_record)


class TestRebuildStudyFromHistoryGapDetection:
    def test_non_contiguous_trial_numbers_rejected(self) -> None:
        spec = make_optimization_spec()
        history = [
            HistoricalTrialRecord(trial_number=0, sampled_hyperparameters={"n": 1}, status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5),
            HistoricalTrialRecord(trial_number=2, sampled_hyperparameters={"n": 1}, status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5),
        ]
        with pytest.raises(OptimizationResumeError, match="must be exactly"):
            rebuild_study_from_history(spec, history)

    def test_empty_history_produces_a_fresh_study(self) -> None:
        spec = make_optimization_spec()
        study = rebuild_study_from_history(spec, [])
        assert len(study.trials) == 0


class TestGlobalRngNeverTouched:
    def test_full_resume_cycle_does_not_perturb_pythons_global_random_or_numpy(self) -> None:
        space = _mixed_space()
        spec = make_optimization_spec(search_space=space)
        py_state_before = random.getstate()
        np_state_before = np.random.get_state()

        uninterrupted = _run_uninterrupted(spec, space, n_trials=6, statuses=[TrialStatus.COMPLETED] * 6)
        history = [
            HistoricalTrialRecord(trial_number=i, sampled_hyperparameters=uninterrupted[i], status=TrialStatus.COMPLETED, primary_metric_aggregate=0.5)
            for i in range(6)
        ]
        rebuild_study_from_history(spec, history)

        assert random.getstate() == py_state_before
        np_state_after = np.random.get_state()
        assert np_state_after[0] == np_state_before[0]
        assert (np_state_after[1] == np_state_before[1]).all()  # type: ignore[union-attr]


class TestEvaluateMedianStopping:
    def _config(self, kind: PruningKind = PruningKind.MEDIAN_STOPPING, *, min_completed_inner_folds: int = 1) -> PruningConfig:
        return PruningConfig(kind=kind, min_completed_inner_folds=min_completed_inner_folds)

    def test_none_kind_never_prunes(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(PruningKind.NONE), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.9),)], current_trial_metrics=(_metrics(0.1),),
        )
        assert pruned is False

    def test_below_minimum_completed_inner_folds_never_prunes(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(min_completed_inner_folds=2), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.9), _metrics(0.9))], current_trial_metrics=(_metrics(0.01),),
        )
        assert pruned is False

    def test_no_eligible_comparison_trials_never_prunes(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(), primary_metric="accuracy", other_trials_metrics=[], current_trial_metrics=(_metrics(0.5),),
        )
        assert pruned is False

    def test_worse_than_median_is_pruned_for_maximize_metric(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.9),), (_metrics(0.8),), (_metrics(0.7),)],
            current_trial_metrics=(_metrics(0.1),),
        )
        assert pruned is True

    def test_better_than_median_is_not_pruned_for_maximize_metric(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.1),), (_metrics(0.2),), (_metrics(0.3),)],
            current_trial_metrics=(_metrics(0.9),),
        )
        assert pruned is False

    def test_direction_is_flipped_for_minimize_metric(self) -> None:
        # For rmse (lower is better), a HIGH current value relative to the
        # median is what should get pruned.
        pruned = evaluate_median_stopping(
            self._config(), primary_metric="rmse",
            other_trials_metrics=[(_metrics(0.1),), (_metrics(0.2),), (_metrics(0.3),)],
            current_trial_metrics=(_metrics(0.9),),
        )
        assert pruned is True

    def test_none_valued_folds_are_excluded_never_treated_as_zero(self) -> None:
        # Current trial's own None entries must be skipped, not averaged in
        # as 0.0 (which would make it look catastrophically bad).
        pruned = evaluate_median_stopping(
            self._config(min_completed_inner_folds=1), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.5),)],
            current_trial_metrics=(_metrics(None), _metrics(0.6)),
        )
        assert pruned is False

    def test_comparison_trial_with_fewer_completed_folds_than_current_is_ineligible(self) -> None:
        pruned = evaluate_median_stopping(
            self._config(min_completed_inner_folds=1), primary_metric="accuracy",
            other_trials_metrics=[(_metrics(0.99),)],  # only 1 fold -- current has 2, so this comparison uses only the prefix
            current_trial_metrics=(_metrics(0.1), _metrics(0.1)),
        )
        # The comparison trial DOES qualify (it has >= n=2? No -- it has
        # only 1 < 2, so it must be excluded entirely) -- with no eligible
        # comparison trials left, never prune.
        assert pruned is False

    def test_deterministic_across_repeated_calls(self) -> None:
        config = self._config()
        args = {
            "primary_metric": "accuracy", "other_trials_metrics": [(_metrics(0.9),), (_metrics(0.2),)], "current_trial_metrics": (_metrics(0.5),),
        }
        first = evaluate_median_stopping(config, **args)  # type: ignore[arg-type]
        second = evaluate_median_stopping(config, **args)  # type: ignore[arg-type]
        assert first == second
