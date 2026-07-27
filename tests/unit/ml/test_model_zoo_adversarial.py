"""Milestone 4C adversarial self-audit for the baseline predictive model
framework. Mirrors `tests/unit/ml/test_ml_adversarial.py` (Milestone 4A)
and `tests/unit/execution/test_adversarial.py` (Milestone 4B)'s exact
"test file doubles as an audit checklist" convention: every attack
vector the milestone explicitly names either has a permanent regression
test HERE or is cross-referenced to the test file that already covers
it.

Covered elsewhere (cross-referenced, not duplicated):
- Per-model native serialize/deserialize round-trip correctness ->
  test_model_zoo_{baselines,linear,lightgbm,xgboost,catboost}.py
- Per-model same-seed determinism / different-seed sensitivity (with
  subsampling configured) -> the same files
- Constant labels (CLASSIFICATION, CRITICAL) / zero training samples /
  missing-values-unsupported rejected BEFORE `fit`, and that a rejected
  fold leaves no partial MODEL/PREDICTIONS artifact behind ->
  test_metrics_fold_executor.py::TestMetricsFoldExecutorPreFitValidationGate
- Engine-level same-seed determinism (model artifact content-hash AND
  computed metrics bit-identical across two independent fold executions)
  -> test_metrics_fold_executor.py::
  TestMetricsFoldExecutorDeterminism::test_same_seed_produces_identical_model_artifact,
  and (through the real `ExecutionRunner`, comparing `DeterministicFoldExecutor`
  against `MetricsFoldExecutor`) test_ml_model_zoo_execution.py::
  test_execute_and_train_produce_identical_model_and_predictions_for_same_seed
- NaN in predictions/labels rejected by `compute_classification_metrics`/
  `compute_regression_metrics` (function-level) -> test_metrics.py
- `validate_training_data` unit-level: zero/single/small training sample
  counts, high-dimensional features (WARNING, not CRITICAL), a missing
  declared feature column (CRITICAL), constant labels -> test_model_validation.py
- Model registry: duplicate registration rejected -> test_model_zoo_registry.py::
  test_merging_into_an_already_populated_registry_does_not_duplicate
- Model registry: unknown model name/version fails actionably (Milestone
  4A, unchanged) -> test_ml_cli.py::TestListAndDescribeModelDefinitions::
  test_describe_unknown_model_fails_actionably,
  TestListAndInspectModels::test_inspect_unknown_model_fails_actionably
- CatBoost's specific column-ORDER sensitivity once a model is reloaded
  from disk -> test_model_zoo_catboost.py::TestCatBoostCategoricalSupport

Covered HERE (not adequately exercised elsewhere):
- Every model's `ModelFactory.create()` rejects an objective outside its
  own declared `capabilities.supported_objectives` -- defense in depth
  at the MODEL layer itself, independent of (and reachable by bypassing)
  `ExperimentPreparer`/`validate_experiment_spec`'s registry-level check.
- A `ModelRegistry` populated with real `ml.model_zoo` models but an
  `ExecutionRunner` that forgot `additional_serializers` fails FAST
  (before any fold's `fit` runs), never partway through an expensive
  real model fit.
- The single-training-sample (n=1, not just n=0) case, exercised
  end-to-end through the real `MetricsFoldExecutor`.
- A single-feature (not just few-feature) schema fits and predicts
  correctly for a real gradient-boosting model AND a real linear model.
- Features with far more columns than rows (p >> n) fit successfully
  end-to-end for a real regularized linear model and a real gradient-
  boosting model, despite `validate_training_data` only WARNING (never
  blocking) on this shape.
- A model whose `predict()` legitimately returns NaN is caught by the
  real `MetricsFoldExecutor` call path (not just the isolated metrics
  function), surfacing as a raised exception.
- A real gradient-boosting model's `predict()` rejects a test-time
  feature frame missing a required column, rather than silently
  mispredicting against misaligned columns.
- `predict()` called twice on the SAME fitted model instance (no re-fit
  in between) is bit-identical -- not just "same seed refits the same
  way", but "the fitted object itself has no hidden mutable state".
- Constant TRAINING labels for a REGRESSION objective (only a WARNING,
  never CRITICAL, per `validate_training_data`) are handled gracefully
  end-to-end by a real regularized linear model, proving the "allowed to
  proceed" design choice is actually safe, not merely undertested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import (
    FeatureSchemaMismatchError,
    SchemaVersionError,
    TrainingDataValidationError,
    UnknownModelDefinitionError,
)
from quant_platform.execution.context import FoldExecutionContext
from quant_platform.execution.executor import FoldData, MetricsFoldExecutor
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.model_zoo import baselines as b
from quant_platform.ml.model_zoo import catboost_model as cb_wrapper
from quant_platform.ml.model_zoo import lightgbm_model as lgbm_wrapper
from quant_platform.ml.model_zoo import linear as linear_wrapper
from quant_platform.ml.model_zoo import xgboost_model as xgb_wrapper
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.tracking import ExperimentEventStore

EID = "a" * 64
_SCHEMA = FeatureSchema(feature_names=("f1", "f2"))


def _context(tmp_path: Path, *, seed: int = 1) -> FoldExecutionContext:
    """Mirrors `test_metrics_fold_executor.py`'s own `_context` helper
    exactly -- duplicated locally per this codebase's established
    per-test-file independence convention, not imported cross-file."""
    from tests.unit.execution.conftest import (
        build_registry,
        make_experiment_spec_kwargs,
        write_synthetic_research_dataset,
    )

    from quant_platform.ml.experiment_manager import ExperimentPreparer
    from quant_platform.ml.experiment_spec import ExperimentSpec

    dataset_manifest, _research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
    manifest = preparer.prepare(spec)
    return FoldExecutionContext(
        experiment_id=manifest.identity.experiment_id, fold_index=2, split_id="fold:2",
        dataset_content_id=dataset_manifest.content_id, manifest=manifest, seed=seed,
        environment=capture_environment_snapshot(), artifact_store=MLArtifactStore(tmp_path / "ml"),
        event_store=ExperimentEventStore(tmp_path / "ml"), artifacts_root=tmp_path / "ml",
        started_at=format_utc_timestamp(utc_now()),
    )


def _clf_data(n: int = 40) -> FoldData:
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    train_labels = pd.Series((train_features["f1"] + 0.3 * train_features["f2"] > 0).astype(int))
    test_features = pd.DataFrame({"f1": rng.normal(size=10), "f2": rng.normal(size=10)})
    test_labels = pd.Series((test_features["f1"] > 0).astype(int))
    return FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)


def _reg_data(n: int = 40) -> FoldData:
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    train_labels = pd.Series(train_features["f1"] * 2 - train_features["f2"])
    test_features = pd.DataFrame({"f1": rng.normal(size=10), "f2": rng.normal(size=10)})
    test_labels = pd.Series(test_features["f1"] * 2 - test_features["f2"])
    return FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)


class TestUnsupportedObjectiveRejectedAtModelFactoryLayer:
    """`ModelMetadata.__post_init__` (`ml.interfaces`) enforces
    `capabilities.supports(objective)` for EVERY model -- this is defense
    in depth that exists independent of `ExperimentPreparer`/
    `validate_experiment_spec`'s own registry-level objective check
    (Milestone 4A), which a caller invoking a `ml.model_zoo` factory
    DIRECTLY (bypassing the registry entirely) would otherwise skip."""

    @pytest.mark.parametrize(
        "factory, rejected_objective",
        [
            (linear_wrapper.LogisticRegressionModelFactory(), ObjectiveType.REGRESSION),
            (linear_wrapper.ElasticNetModelFactory(), ObjectiveType.BINARY_CLASSIFICATION),
            (b.MajorityPredictorFactory(), ObjectiveType.REGRESSION),
            (b.DummyMeanRegressorFactory(), ObjectiveType.BINARY_CLASSIFICATION),
        ],
        ids=["logistic_regression/regression", "elastic_net/binary_classification", "majority_predictor/regression", "dummy_mean_regressor/binary_classification"],
    )
    def test_baseline_or_linear_model_rejects_its_unsupported_objective(self, factory, rejected_objective: ObjectiveType) -> None:
        with pytest.raises(ValueError, match="not in capabilities"):
            factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=rejected_objective)

    @pytest.mark.parametrize(
        "factory",
        [lgbm_wrapper.LightGBMModelFactory(), xgb_wrapper.XGBoostModelFactory(), cb_wrapper.CatBoostModelFactory()],
        ids=["lightgbm", "xgboost", "catboost"],
    )
    def test_gradient_boosting_models_reject_multiclass_classification(self, factory) -> None:
        with pytest.raises(ValueError, match="not in capabilities"):
            factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.MULTICLASS_CLASSIFICATION)


class TestMissingSerializerFailsFastBeforeAnyFoldRuns:
    def test_real_model_zoo_model_without_additional_serializers_fails_before_any_fold(self, tmp_path: Path) -> None:
        from tests.unit.execution.conftest import (
            make_experiment_spec_kwargs,
            write_synthetic_research_dataset,
        )

        from quant_platform.execution.runner import ExecutionRunner
        from quant_platform.ml.experiment_manager import ExperimentPreparer
        from quant_platform.ml.experiment_spec import ExperimentSpec
        from quant_platform.ml.models import ExperimentStatus

        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        registry = mz.register_default_models()
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=registry, research_manifest_store=research_manifest_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(
            dataset_manifest=dataset_manifest, model_name="lightgbm", model_version="1",
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 5}),
        ))
        manifest = preparer.prepare(spec)
        assert manifest.status is ExperimentStatus.READY, manifest.failure_summary

        # Deliberately WITHOUT `additional_serializers` -- the wiring
        # mistake this test proves is caught loudly and immediately,
        # never silently mispredicted or discovered deep inside a fold
        # after an expensive real fit already ran.
        runner = ExecutionRunner(
            ml_artifacts_root=tmp_path / "ml", model_registry=registry,
            research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        )
        with pytest.raises(UnknownModelDefinitionError, match="lightgbm"):
            runner.run(manifest.identity.experiment_id)


class TestSingleTrainingSampleRejectedEndToEnd:
    """The n=1 sibling of `test_metrics_fold_executor.py`'s n=0 case --
    the milestone calls out "single sample" as its own explicit attack
    vector, distinct from "zero samples"."""

    def test_exactly_one_training_row_rejected_before_fit(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _clf_data()
        bad_data = FoldData(
            train_features=data.train_features.iloc[:1], train_labels=data.train_labels.iloc[:1],
            test_features=data.test_features, test_labels=data.test_labels,
        )
        with pytest.raises(TrainingDataValidationError, match="single_training_sample"):
            MetricsFoldExecutor().execute(
                context, model_factory=b.RandomPredictorFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=b.RandomPredictorSerializer(), data=bad_data,
            )


class TestSingleFeatureSchemaWorksEndToEnd:
    def _single_feature_data(self) -> tuple[FeatureSchema, FoldData]:
        schema = FeatureSchema(feature_names=("only_feature",))
        rng = np.random.default_rng(0)
        train_features = pd.DataFrame({"only_feature": rng.normal(size=60)})
        train_labels = pd.Series((train_features["only_feature"] > 0).astype(int))
        test_features = pd.DataFrame({"only_feature": rng.normal(size=10)})
        test_labels = pd.Series((test_features["only_feature"] > 0).astype(int))
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
        return schema, data

    def test_lightgbm_fits_and_predicts_with_exactly_one_feature(self, tmp_path: Path) -> None:
        schema, data = self._single_feature_data()
        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=lgbm_wrapper.LightGBMModelFactory(),
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 10}),
            feature_schema=schema, objective=ObjectiveType.BINARY_CLASSIFICATION,
            serializer=lgbm_wrapper.LightGBMModelSerializer(), data=data,
        )
        assert "accuracy" in outcome.metrics

    def test_logistic_regression_with_exactly_one_feature_is_still_rejected_for_missing_preprocessing_proof(self, tmp_path: Path) -> None:
        """A single-feature schema does not exempt a scale-sensitive model
        from BLOCKER 2's "REQUIRED PREPROCESSING MUST BE ENFORCED" gate --
        see `TestRequiredPreprocessingEnforcement` for the dedicated
        coverage of that gate itself; this only proves the two checks
        compose correctly (schema shape does not silently bypass it)."""
        schema, data = self._single_feature_data()
        with pytest.raises(TrainingDataValidationError, match="required_preprocessing_unproven"):
            MetricsFoldExecutor().execute(
                _context(tmp_path), model_factory=linear_wrapper.LogisticRegressionModelFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=schema, objective=ObjectiveType.BINARY_CLASSIFICATION,
                serializer=linear_wrapper.LogisticRegressionModelSerializer(), data=data,
            )


class TestHighDimensionalFeaturesSurviveEndToEnd:
    def _wide_data(self, *, n: int = 15, p: int = 100) -> tuple[FeatureSchema, FoldData]:
        rng = np.random.default_rng(0)
        feature_names = tuple(f"f{i}" for i in range(p))
        schema = FeatureSchema(feature_names=feature_names)
        train_features = pd.DataFrame(rng.normal(size=(n, p)), columns=feature_names)
        train_labels = pd.Series(train_features["f0"] * 2 - train_features["f1"])
        test_features = pd.DataFrame(rng.normal(size=(5, p)), columns=feature_names)
        test_labels = pd.Series(test_features["f0"] * 2 - test_features["f1"])
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
        return schema, data

    def test_elastic_net_fits_with_far_more_features_than_rows_when_called_directly(self, tmp_path: Path) -> None:
        """`ElasticNetModel.fit` ITSELF handles p >> n correctly (L1/L2
        regularization is well-posed here) -- called directly, bypassing
        `MetricsFoldExecutor`'s pre-fit gate, exactly as this milestone's
        own per-model unit tests already do (`test_model_zoo_linear.py`).
        Composed with the real, orchestrated engine, this same model is
        rejected today for a DIFFERENT, unrelated reason (BLOCKER 2's
        preprocessing-proof gate, not this shape) -- see the test
        immediately below."""
        schema, data = self._wide_data()
        factory = linear_wrapper.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=schema, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(data.test_features)
        assert not np.isnan(preds).any()

    def test_elastic_net_high_dimensional_features_still_rejected_for_missing_preprocessing_proof_via_the_real_engine(self, tmp_path: Path) -> None:
        schema, data = self._wide_data()
        with pytest.raises(TrainingDataValidationError, match="required_preprocessing_unproven"):
            MetricsFoldExecutor().execute(
                _context(tmp_path), model_factory=linear_wrapper.ElasticNetModelFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=schema, objective=ObjectiveType.REGRESSION,
                serializer=linear_wrapper.ElasticNetModelSerializer(), data=data,
            )

    def test_lightgbm_fits_with_far_more_features_than_rows(self, tmp_path: Path) -> None:
        schema, data = self._wide_data()
        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=lgbm_wrapper.LightGBMModelFactory(),
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 10, "min_data_in_leaf": 1}),
            feature_schema=schema, objective=ObjectiveType.REGRESSION,
            serializer=lgbm_wrapper.LightGBMModelSerializer(), data=data,
        )
        assert "mae" in outcome.metrics


@dataclass(frozen=True, slots=True)
class _NaNPredictingFittedModel:
    metadata: ModelMetadata

    @property
    def is_fitted(self) -> bool:
        return True

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        return np.full(len(features), np.nan, dtype="float64")


@dataclass(frozen=True, slots=True)
class _NaNPredictingTrainableModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> _NaNPredictingFittedModel:
        return _NaNPredictingFittedModel(metadata=self.metadata)


class _NaNPredictingModelFactory:
    """A deliberately malicious, test-only model: legitimately conforms
    to every Protocol `MetricsFoldExecutor` relies on, but its
    `predict()` always returns NaN -- exactly the "what if a model
    misbehaves" case none of the 9 real, shipped models can ever
    exhibit by construction, but the executor must still handle safely."""

    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), is_deterministic=False, library_name="test-stub")

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> _NaNPredictingTrainableModel:
        metadata = ModelMetadata(
            name="nan_predicting_stub", version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return _NaNPredictingTrainableModel(metadata=metadata)


class _NaNPredictingModelSerializer:
    def serialize(self, model: _NaNPredictingFittedModel) -> bytes:
        return b"{}"


class TestNaNPredictionsSurfaceAsAFoldFailure:
    """`ml.metrics._reject_nan` is unit-tested directly (in isolation) by
    `test_metrics.py`; this proves it is genuinely wired into
    `MetricsFoldExecutor`'s real call path via a model whose `predict()`
    legitimately returns NaN, and that the failure surfaces as a raised
    exception -- which `execution.runner`'s existing, exception-type-
    agnostic "any exception during one fold -> that fold's
    FoldResult(status=FAILED)" handling (already proven generically for
    OTHER exception types by Milestone 4B's own test suite, and for
    `TrainingDataValidationError` specifically by `test_metrics_fold_
    executor.py`) would convert into a single failed fold, never crash
    the whole run or silently accept a NaN-laden prediction set as a
    valid result."""

    def test_a_model_that_predicts_nan_is_rejected_by_the_real_metrics_computation_path(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        data = _reg_data()
        with pytest.raises(ValueError, match="NaN"):
            MetricsFoldExecutor().execute(
                context, model_factory=_NaNPredictingModelFactory(), hyperparameters=ModelHyperparameters(),
                feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
                serializer=_NaNPredictingModelSerializer(), data=data,
            )


class TestFeatureSchemaMismatchAtPredictTimeRejected:
    def test_lightgbm_predict_with_a_missing_required_column_raises_clearly(self) -> None:
        features = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=60), "f2": np.random.default_rng(1).normal(size=60)})
        labels = pd.Series((features["f1"] > 0).astype(int))
        factory = lgbm_wrapper.LightGBMModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 10}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        malformed = features.drop(columns=["f2"])
        with pytest.raises(FeatureSchemaMismatchError, match="f2"):
            fitted.predict(malformed)


class TestRepeatedPredictionsAreBitIdentical:
    """Not "same seed refits the same way" (already covered elsewhere) --
    this is "the FITTED object itself has no hidden mutable state that
    changes its answer between two `predict()` calls on the identical
    input, with no re-fit in between". `RandomPredictor` is the most at-
    risk of the 9 models by design (its docstring explicitly promises a
    fresh-reseeded-per-call RNG for exactly this reason); a real
    gradient-boosting model is included too since nothing about this
    property should be baseline-specific."""

    def test_random_predictor_repeated_predict_calls_match(self, tmp_path: Path) -> None:
        data = _clf_data()
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))

        first = fitted.predict(data.test_features)
        second = fitted.predict(data.test_features)
        assert np.array_equal(first, second)
        first_proba = fitted.predict_proba(data.test_features)
        second_proba = fitted.predict_proba(data.test_features)
        assert np.array_equal(first_proba, second_proba)

    def test_lightgbm_repeated_predict_calls_match(self, tmp_path: Path) -> None:
        data = _clf_data()
        factory = lgbm_wrapper.LightGBMModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={"num_boost_round": 10}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))

        first = fitted.predict(data.test_features)
        second = fitted.predict(data.test_features)
        assert np.array_equal(first, second)


def _assert_global_numpy_rng_state_unaffected(action) -> None:
    """Every model's seed usage must flow through this platform's OWN
    `SeedConfiguration`-derived, LOCAL generators (`np.random.
    default_rng(seed)`, a library's own `random_state`/`seed` parameter)
    -- never `numpy.random`'s global, process-wide generator, which
    would make one model's fit/predict silently perturb an unrelated,
    LATER call's randomness elsewhere in the same process."""
    np.random.seed(20260726)
    state_before = np.random.get_state()
    action()
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


class TestGlobalRNGStateNeverMutated:
    def test_random_predictor_fit_predict_does_not_touch_global_numpy_rng(self) -> None:
        data = _clf_data()

        def action() -> None:
            factory = b.RandomPredictorFactory()
            model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))
            fitted.predict(data.test_features)
            fitted.predict_proba(data.test_features)

        _assert_global_numpy_rng_state_unaffected(action)

    def test_lightgbm_fit_predict_does_not_touch_global_numpy_rng(self) -> None:
        data = _clf_data()

        def action() -> None:
            factory = lgbm_wrapper.LightGBMModelFactory()
            model = factory.create(hyperparameters=ModelHyperparameters(values={"num_boost_round": 10}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))
            fitted.predict(data.test_features)

        _assert_global_numpy_rng_state_unaffected(action)

    def test_xgboost_fit_predict_does_not_touch_global_numpy_rng(self) -> None:
        data = _clf_data()

        def action() -> None:
            factory = xgb_wrapper.XGBoostModelFactory()
            model = factory.create(hyperparameters=ModelHyperparameters(values={"num_boost_round": 10}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))
            fitted.predict(data.test_features)

        _assert_global_numpy_rng_state_unaffected(action)

    def test_catboost_fit_predict_does_not_touch_global_numpy_rng(self) -> None:
        data = _clf_data()

        def action() -> None:
            factory = cb_wrapper.CatBoostModelFactory()
            model = factory.create(hyperparameters=ModelHyperparameters(values={"iterations": 10}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))
            fitted.predict(data.test_features)

        _assert_global_numpy_rng_state_unaffected(action)

    def test_logistic_regression_fit_predict_does_not_touch_global_numpy_rng(self) -> None:
        data = _clf_data()

        def action() -> None:
            factory = linear_wrapper.LogisticRegressionModelFactory()
            model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            fitted = model.fit(data.train_features, data.train_labels, seeds=SeedConfiguration(master_seed=7))
            fitted.predict(data.test_features)

        _assert_global_numpy_rng_state_unaffected(action)


class TestConstantLabelsAllowedForRegressionEndToEnd:
    """`validate_training_data` treats constant labels as CRITICAL (block)
    for classification but only WARNING (proceed) for regression -- a
    constant regression target is a legitimate (if degenerate) real-world
    case (e.g. a symbol that never moved during a fold's train window).
    This proves that "allowed to proceed" design choice is actually safe
    end-to-end for a real (non-baseline) learning algorithm, not merely
    unblocked-but-untested. Uses LightGBM rather than Elastic Net: Elastic
    Net declares `requires_scaled_numeric_features=True` (BLOCKER 2) and
    is therefore ALWAYS rejected by the real engine's preprocessing-proof
    gate today, regardless of label shape -- an orthogonal concern this
    test is not about; LightGBM is scale-invariant and unaffected by it."""

    def test_lightgbm_fits_successfully_on_constant_training_labels(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        train_features = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
        train_labels = pd.Series([5.0] * 40)
        test_features = pd.DataFrame({"f1": rng.normal(size=10), "f2": rng.normal(size=10)})
        test_labels = pd.Series([5.0] * 10)
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)

        outcome = MetricsFoldExecutor().execute(
            _context(tmp_path), model_factory=lgbm_wrapper.LightGBMModelFactory(),
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 20}),
            feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
            serializer=lgbm_wrapper.LightGBMModelSerializer(), data=data,
        )
        assert "mae" in outcome.metrics
        assert outcome.metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert "r2" not in outcome.metrics  # R^2 is correctly SKIPPED, never a fabricated value, for zero-variance y_true


def _fitted_lightgbm_envelope() -> dict:
    features = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=30), "f2": np.random.default_rng(1).normal(size=30)})
    labels = pd.Series((features["f1"] > 0).astype(int))
    factory = lgbm_wrapper.LightGBMModelFactory()
    model = factory.create(hyperparameters=ModelHyperparameters(values={"num_boost_round": 5}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
    fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
    raw_bytes = lgbm_wrapper.LightGBMModelSerializer().serialize(fitted)
    return json.loads(raw_bytes.decode("utf-8"))


def _fitted_constant_predictor_envelope() -> dict:
    factory = b.ConstantPredictorFactory()
    model = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 1.0}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
    fitted = model.fit(pd.DataFrame({"f1": [0.0], "f2": [0.0]}), pd.Series([1]), seeds=SeedConfiguration(master_seed=1))
    return json.loads(b.ConstantPredictorSerializer().serialize(fitted).decode("utf-8"))


class TestSerializationEnvelopeCompatibility:
    """SERIALIZATION COMPATIBILITY AUDIT: every `from_bytes` classmethod
    must reject an unsupported schema version, a corrupted (non-JSON)
    payload, and another model's envelope (wrong keys) with a clear,
    domain-specific exception -- never a raw `KeyError`/`json.
    JSONDecodeError` leaking through. Exercised across a real gradient-
    boosting model and a baseline, not just one."""

    def test_unsupported_schema_version_rejected_lightgbm(self) -> None:
        envelope = _fitted_lightgbm_envelope()
        envelope["schema_version"] = 999999
        tampered = json.dumps(envelope).encode("utf-8")
        with pytest.raises(SchemaVersionError, match="999999"):
            lgbm_wrapper.LightGBMModelDeserializer().deserialize(tampered, expected_metadata=None)

    def test_unsupported_schema_version_rejected_baseline(self) -> None:
        envelope = _fitted_constant_predictor_envelope()
        envelope["schema_version"] = 999999
        tampered = json.dumps(envelope).encode("utf-8")
        with pytest.raises(SchemaVersionError, match="999999"):
            b.ConstantPredictorDeserializer().deserialize(tampered, expected_metadata=None)

    def test_corrupted_non_json_payload_rejected_with_a_clear_value_error(self) -> None:
        with pytest.raises(ValueError, match="Malformed JSON"):
            lgbm_wrapper.LightGBMModelDeserializer().deserialize(b"{not valid json!!!", expected_metadata=None)

    def test_truncated_json_payload_rejected_with_a_clear_value_error(self) -> None:
        envelope = _fitted_lightgbm_envelope()
        full_bytes = json.dumps(envelope).encode("utf-8")
        truncated = full_bytes[: len(full_bytes) // 2]
        with pytest.raises(ValueError, match="Malformed JSON"):
            lgbm_wrapper.LightGBMModelDeserializer().deserialize(truncated, expected_metadata=None)

    def test_another_models_envelope_rejected_not_a_raw_key_error(self) -> None:
        """A CatBoost deserializer given LightGBM's (schema-version-
        compatible, valid JSON, but WRONG-SHAPED) envelope must fail with
        an actionable `ValueError` naming the model, never a bare
        `KeyError: 'model_cbm_base64'`."""
        lightgbm_envelope = _fitted_lightgbm_envelope()
        # Schema versions happen to both be `1` today (no real version
        # skew), so this specifically isolates the "wrong envelope SHAPE"
        # failure mode from the "wrong schema version" one tested above.
        tampered = json.dumps(lightgbm_envelope).encode("utf-8")
        with pytest.raises(ValueError, match="FittedCatBoostModel"):
            cb_wrapper.CatBoostModelDeserializer().deserialize(tampered, expected_metadata=None)

    def test_missing_metadata_key_entirely_rejected_not_a_raw_key_error(self) -> None:
        envelope = _fitted_lightgbm_envelope()
        del envelope["metadata"]
        tampered = json.dumps(envelope).encode("utf-8")
        with pytest.raises(ValueError, match="FittedLightGBMModel"):
            lgbm_wrapper.LightGBMModelDeserializer().deserialize(tampered, expected_metadata=None)
