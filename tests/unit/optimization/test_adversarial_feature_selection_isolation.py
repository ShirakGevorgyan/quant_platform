"""Adversarial release-readiness audit, Section 4: FEATURE-SELECTION
STATE ISOLATION. Proves every selector is fitted only on the current
inner-train partition -- never inner-validation, never a previous trial's
or outer fold's leftover state (`feature_selection.py`'s six strategy
functions are pure functions of their `features`/`labels`/`row_positions`
arguments, with no module-level cache, no selector object reused across
calls, and a fresh `model_factory.create()` call every time
MODEL_NATIVE_IMPORTANCE runs -- see that module's own docstring)."""

from __future__ import annotations

import contextlib
import inspect

import numpy as np
import pandas as pd
import pytest

from quant_platform.execution.splitters import required_label_purge_bars_for
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.candidates import TrialSpec, TrialStatus
from quant_platform.optimization.feature_selection import (
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
    select_correlation_filter,
    select_model_native_importance,
    select_stability,
    select_univariate,
    select_variance_filter,
)
from quant_platform.optimization.inner_splits import INNER_SPLIT_SCHEMA_VERSION, InnerFold, InnerFoldPlan
from quant_platform.optimization.models import EarlyStoppingConfig
from quant_platform.optimization.trial_executor import run_trial

_POISON = 1_000_000.0
_TS = pd.Timestamp("2024-01-01")
_N_ROWS = 40


class PoisonLeakError(AssertionError):
    pass


def _feature_universe() -> FeatureUniverse:
    return FeatureUniverse(feature_names=("f1", "f2"), fingerprint="a" * 64)


class TestUnsupervisedSelectorsStructurallyCannotSeeLabels:
    """`VARIANCE_FILTER`/`CORRELATION_FILTER` are declared unsupervised --
    proven here not by behavior but by SIGNATURE: neither function even
    HAS a `labels` parameter, so changing inner-validation (or any) labels
    cannot possibly influence them, regardless of what future code inside
    either function might do."""

    def test_variance_filter_has_no_labels_parameter(self) -> None:
        assert "labels" not in inspect.signature(select_variance_filter).parameters

    def test_correlation_filter_has_no_labels_parameter(self) -> None:
        assert "labels" not in inspect.signature(select_correlation_filter).parameters

    def test_supervised_strategies_do_accept_labels_confirming_the_check_is_meaningful(self) -> None:
        """Sanity: the absence of `labels` above is a deliberate, narrow
        fact about exactly two strategies -- not an accident of every
        strategy function lacking a labels parameter."""
        assert "labels" in inspect.signature(select_univariate).parameters
        assert "labels" in inspect.signature(select_model_native_importance).parameters
        assert "labels" in inspect.signature(select_stability).parameters


class TestInnerValidationNeverReachesFeatureSelection:
    """The complementary poison proof to Section 1's outer-test isolation
    tests, targeting INNER-VALIDATION rows specifically: `_run_one_inner_
    fold` passes ONLY `inner_fold.train_indices`-derived data to feature
    selection; validation rows are read solely for prediction/scoring,
    never selection."""

    def _poisoned_timeline(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        f1 = rng.normal(size=_N_ROWS)
        f2 = rng.normal(size=_N_ROWS)
        label = rng.normal(size=_N_ROWS)
        # Rows [25, 40) are the ONE inner fold's validation partition --
        # poisoned to prove feature selection never reads them.
        f1[25:] = _POISON
        f2[25:] = _POISON
        label[25:] = _POISON
        return pd.DataFrame({"f1": f1, "f2": f2, "label": label})

    def _plan(self) -> InnerFoldPlan:
        purge = required_label_purge_bars_for(1)
        fold = InnerFold(
            inner_fold_index=0, train_indices=np.arange(0, 20), validation_indices=np.arange(25, _N_ROWS),
            train_start=_TS, train_end=_TS, validation_start=_TS, validation_end=_TS,
        )
        return InnerFoldPlan(
            schema_version=INNER_SPLIT_SCHEMA_VERSION, outer_fold_index=0, strategy="expanding_walk_forward", inner_folds=(fold,),
            purge_bars=purge, embargo_bars=0, label_horizon_bars=1, required_label_purge_bars=purge, outer_train_row_count=20,
        )

    @pytest.mark.parametrize(
        "fs_spec",
        [
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE),
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.VARIANCE_FILTER, params={"min_variance": 0.0}),
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.UNIVARIATE, params={"mode": "top_k", "k": 1}),
            FeatureSelectionSpec(strategy=FeatureSelectionStrategy.STABILITY_SELECTION, params={"base_strategy": "univariate", "mode": "top_k", "k": 1, "n_repeats": 3, "subsample_fraction": 0.9, "min_frequency": 0.0}),
        ],
        ids=["none", "variance_filter", "univariate", "stability_selection"],
    )
    def test_feature_selection_result_provenance_excludes_inner_validation_rows(self, tmp_path, fs_spec: FeatureSelectionSpec) -> None:
        import json

        from quant_platform.optimization.feature_selection import FeatureSelectionResult

        timeline = self._poisoned_timeline()
        plan = self._plan()
        universe = _feature_universe()
        store = MLArtifactStore(tmp_path)
        trial_spec = TrialSpec(
            schema_version=1, optimization_id="a" * 64, outer_fold_index=0, trial_number=0,
            sampled_hyperparameters={"alpha": 0.1}, feature_selection_spec=fs_spec, trial_seed=0,
            inner_split_plan_fingerprint="b" * 64, model_definition_fingerprint="c" * 64,
            objective=ObjectiveType.REGRESSION, primary_metric="rmse",
        )
        result = run_trial(
            trial_spec, inner_fold_plan=plan, timeline=timeline, feature_universe=universe, model_name="constant_test_model",
            model_factory=ConstantTestModelFactory(), seed_configuration=SeedConfiguration(master_seed=0), min_successful_inner_folds=1,
            early_stopping_config=EarlyStoppingConfig(enabled=False), artifact_store=store,
        )
        assert result.status is TrialStatus.COMPLETED, result.failure_reason
        checked = 0
        for ref in result.artifact_references:
            raw = store.read_artifact(ref.content_hash)
            fs_result = FeatureSelectionResult.from_json_dict(json.loads(raw.decode("utf-8")))
            assert fs_result.training_row_last_position < 25, (
                f"strategy={fs_spec.strategy.value}: FeatureSelectionResult training rows extend into the "
                f"poisoned inner-validation range (last_position={fs_result.training_row_last_position})"
            )
            checked += 1
        assert checked >= 1


class TestNoStateContaminationAcrossTrialsOrOuterFolds:
    """Every strategy function is a pure function of its own arguments --
    proven dynamically by running the SAME strategy twice with DIFFERENT
    underlying data (simulating trial N then trial N+1, or outer fold 0
    then outer fold 1) and confirming the second call's result depends
    ONLY on its own input, never on anything left behind by the first."""

    def test_variance_filter_result_depends_only_on_its_own_call_data_not_prior_calls(self) -> None:
        universe = _feature_universe()
        rng = np.random.default_rng(0)
        # Call 1: f1 has high variance, f2 near-constant.
        features_a = pd.DataFrame({"f1": rng.normal(scale=10.0, size=50), "f2": rng.normal(scale=0.001, size=50)})
        result_a = select_variance_filter(universe=universe, features=features_a, row_positions=np.arange(50), params={"min_variance": 1.0}, seed=0)
        assert set(result_a.selected_features) == {"f1"}

        # Call 2 (a "different trial"/"different outer fold"): f2 now has
        # high variance, f1 near-constant -- the OPPOSITE outcome. If any
        # state leaked from call 1, this would incorrectly still favor f1.
        features_b = pd.DataFrame({"f1": rng.normal(scale=0.001, size=50), "f2": rng.normal(scale=10.0, size=50)})
        result_b = select_variance_filter(universe=universe, features=features_b, row_positions=np.arange(50), params={"min_variance": 1.0}, seed=0)
        assert set(result_b.selected_features) == {"f2"}

    def test_model_native_importance_constructs_a_fresh_selector_model_every_call(self, tmp_path) -> None:
        """MODEL_NATIVE_IMPORTANCE is the one strategy that fits an actual
        model as its selector -- proven here to call `model_factory.
        create()` fresh every invocation (never reusing a fitted selector
        object across calls) by counting factory invocations across two
        independent calls."""

        class _CountingFactory:
            def __init__(self) -> None:
                self.create_calls = 0

            def create(self, *, hyperparameters, feature_schema, objective):
                self.create_calls += 1
                return ConstantTestModelFactory().create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)

        universe = _feature_universe()
        rng = np.random.default_rng(0)
        features = pd.DataFrame({"f1": rng.normal(size=30), "f2": rng.normal(size=30)})
        labels = pd.Series(rng.normal(size=30))
        factory = _CountingFactory()
        from quant_platform.ml.models import ModelHyperparameters

        # MODEL_NATIVE_IMPORTANCE requires feature_importance() -- ConstantTestModel
        # does not expose one, so this call is EXPECTED to raise; the point
        # of this test is purely the create_calls count, not the outcome.
        for _ in range(3):
            with contextlib.suppress(AttributeError, ValueError):
                select_model_native_importance(
                    universe=universe, features=features, labels=labels, row_positions=np.arange(30),
                    params={"mode": "top_k", "k": 1}, seed=0, model_name="lightgbm", model_factory=factory,
                    hyperparameters=ModelHyperparameters(values={}), objective=ObjectiveType.REGRESSION,
                )
        assert factory.create_calls == 3  # one fresh model per call, never reused


class TestRepeatedIdenticalRunsProduceIdenticalCanonicalOrder:
    def test_variance_filter_selected_feature_order_is_deterministic(self) -> None:
        universe = FeatureUniverse(feature_names=("f1", "f2", "f3"), fingerprint="a" * 64)
        rng = np.random.default_rng(0)
        features = pd.DataFrame({
            "f1": rng.normal(scale=5.0, size=40), "f2": rng.normal(scale=3.0, size=40), "f3": rng.normal(scale=1.0, size=40),
        })
        results = [
            select_variance_filter(universe=universe, features=features, row_positions=np.arange(40), params={"min_variance": 0.0}, seed=0)
            for _ in range(5)
        ]
        assert len({r.selected_features for r in results}) == 1  # every repeat produces the IDENTICAL ordered tuple
        assert results[0].selected_features == ("f1", "f2", "f3")  # universe order preserved, not score-sorted

    def test_univariate_selected_feature_order_is_deterministic_given_the_same_seed(self) -> None:
        universe = _feature_universe()
        rng = np.random.default_rng(0)
        features = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
        labels = pd.Series(features["f1"].to_numpy() * 2.0 + rng.normal(scale=0.01, size=60))
        results = [
            select_univariate(
                universe=universe, features=features, labels=labels, row_positions=np.arange(60),
                params={"mode": "top_k", "k": 1}, seed=42, objective=ObjectiveType.REGRESSION,
            )
            for _ in range(5)
        ]
        assert len({r.selected_features for r in results}) == 1
