from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml import model_zoo as mz
from quant_platform.ml.interfaces import DecisionFunctionPredictor, FeatureSchema, ProbabilisticPredictor
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration


class TestRegisterDefaultModels:
    def test_registers_exactly_nine_models(self) -> None:
        registry = mz.register_default_models()
        assert len(registry) == 9

    def test_returns_the_given_registry_when_provided(self) -> None:
        registry = ModelRegistry()
        result = mz.register_default_models(registry)
        assert result is registry

    def test_creates_a_fresh_registry_when_none_given(self) -> None:
        registry = mz.register_default_models()
        assert isinstance(registry, ModelRegistry)

    def test_every_model_name_present(self) -> None:
        registry = mz.register_default_models()
        names = {d.name for d in registry.list_definitions()}
        assert names == {
            "constant_predictor", "random_predictor", "majority_predictor", "dummy_mean_regressor",
            "logistic_regression", "elastic_net", "lightgbm", "xgboost", "catboost",
        }

    def test_every_baseline_name_matches_declared_constant(self) -> None:
        assert set(mz.BASELINE_MODEL_NAMES) == {
            mz.CONSTANT_PREDICTOR_NAME, mz.RANDOM_PREDICTOR_NAME, mz.MAJORITY_PREDICTOR_NAME, mz.DUMMY_MEAN_REGRESSOR_NAME,
        }

    def test_merging_into_an_already_populated_registry_does_not_duplicate(self) -> None:
        import pytest

        from quant_platform.core.exceptions import DuplicateModelDefinitionError

        registry = mz.register_default_models()
        with pytest.raises(DuplicateModelDefinitionError):
            mz.register_default_models(registry)

    def test_every_definition_has_a_stable_fingerprint(self) -> None:
        registry = mz.register_default_models()
        for definition in registry.list_definitions():
            fp1 = definition.fingerprint()
            fp2 = definition.fingerprint()
            assert fp1 == fp2
            assert len(fp1) == 64


class TestDefaultSerializerRegistry:
    def test_every_registered_model_has_a_resolvable_serializer(self) -> None:
        registry = mz.register_default_models()
        serializers = mz.default_serializer_registry()
        for definition in registry.list_definitions():
            assert definition.serializer_id in serializers, f"{definition.qualified_name} has no serializer entry"

    def test_every_entry_has_both_a_serializer_and_deserializer(self) -> None:
        serializers = mz.default_serializer_registry()
        for serializer_id, (serializer, deserializer) in serializers.items():
            assert serializer is not None, f"{serializer_id} has no serializer"
            assert deserializer is not None, f"{serializer_id} has no deserializer"
            assert hasattr(serializer, "serialize")
            assert hasattr(deserializer, "deserialize")

    def test_nine_serializer_entries(self) -> None:
        assert len(mz.default_serializer_registry()) == 9


_PREDICTION_CONTRACT_CASES: list[tuple[str, ObjectiveType, dict[str, object]]] = [
    ("constant_predictor", ObjectiveType.BINARY_CLASSIFICATION, {"constant": 1.0}),
    ("random_predictor", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("majority_predictor", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("logistic_regression", ObjectiveType.BINARY_CLASSIFICATION, {}),
    ("lightgbm", ObjectiveType.BINARY_CLASSIFICATION, {"num_boost_round": 10}),
    ("xgboost", ObjectiveType.BINARY_CLASSIFICATION, {"num_boost_round": 10}),
    ("catboost", ObjectiveType.BINARY_CLASSIFICATION, {"iterations": 10}),
    ("dummy_mean_regressor", ObjectiveType.REGRESSION, {}),
    ("elastic_net", ObjectiveType.REGRESSION, {}),
]
"""Every model in its natural, best-covered objective -- classification
wherever supported (proves predict_proba/class_labels too), regression
only for the two that are regression-only."""


class TestUnifiedPredictionContract:
    """PREDICTION CONTRACT: `predict()` always returns a 1-D float64
    array of length n; for BINARY_CLASSIFICATION, every value is in
    {0.0, 1.0} (hard labels, never a probability); `predict_proba()`
    (only on models declaring `supports_predict_proba`) always returns
    an (n, 2) float64 matrix whose rows sum to 1.0, with column order
    given by `class_labels`; `decision_function()` (only on models
    implementing `DecisionFunctionPredictor`) returns a 1-D float64
    array. Verified across all 9 registered models from ONE place,
    rather than scattered per-model assertions that could silently drift
    apart from each other. Calls `ModelFactory.create().fit(...)` DIRECTLY
    (not through `MetricsFoldExecutor`), exactly like this milestone's own
    per-model unit tests -- BLOCKER 2's preprocessing-proof gate is an
    orchestration-level concern, not a `TrainableModel.fit` constraint,
    so Logistic Regression/Elastic Net still fit here unaffected by it."""

    @pytest.mark.parametrize("model_name, objective, hyperparameters", _PREDICTION_CONTRACT_CASES, ids=[c[0] for c in _PREDICTION_CONTRACT_CASES])
    def test_predict_and_predict_proba_shapes_are_uniform(self, model_name: str, objective: ObjectiveType, hyperparameters: dict[str, object]) -> None:
        schema = FeatureSchema(feature_names=("f1", "f2"))
        rng = np.random.default_rng(0)
        n = 40
        features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        labels = pd.Series((features["f1"] > 0).astype(int)) if objective is ObjectiveType.BINARY_CLASSIFICATION else pd.Series(features["f1"] * 2 - features["f2"])
        test_features = features.iloc[:10]

        registry = mz.register_default_models()
        definition = registry.get(model_name, "1")
        model = definition.factory.create(hyperparameters=ModelHyperparameters(values=hyperparameters), feature_schema=schema, objective=objective)
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        predictions = fitted.predict(test_features)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float64
        assert predictions.shape == (len(test_features),)
        if objective is ObjectiveType.BINARY_CLASSIFICATION:
            assert set(np.unique(predictions)) <= {0.0, 1.0}

        if definition.capabilities.supports_predict_proba:
            assert isinstance(fitted, ProbabilisticPredictor)
            proba = fitted.predict_proba(test_features)
            assert isinstance(proba, np.ndarray)
            assert proba.dtype == np.float64
            assert proba.shape == (len(test_features), 2)
            assert np.allclose(proba.sum(axis=1), 1.0)
            assert len(fitted.class_labels) == 2

        if isinstance(fitted, DecisionFunctionPredictor):
            decision = fitted.decision_function(test_features)
            assert isinstance(decision, np.ndarray)
            assert decision.dtype == np.float64
            assert decision.shape == (len(test_features),)

    @pytest.mark.parametrize("model_name, objective, hyperparameters", _PREDICTION_CONTRACT_CASES, ids=[c[0] for c in _PREDICTION_CONTRACT_CASES])
    def test_decision_function_is_exclusive_to_logistic_regression(self, model_name: str, objective: ObjectiveType, hyperparameters: dict[str, object]) -> None:
        """Nothing else among the 9 models implements it -- an additive,
        optional capability, never assumed present."""
        schema = FeatureSchema(feature_names=("f1",))
        features = pd.DataFrame({"f1": [1.0, -1.0, 2.0, -2.0]})
        labels = pd.Series([1, 0, 1, 0]) if objective is ObjectiveType.BINARY_CLASSIFICATION else pd.Series([1.0, -1.0, 2.0, -2.0])

        registry = mz.register_default_models()
        definition = registry.get(model_name, "1")
        model = definition.factory.create(hyperparameters=ModelHyperparameters(values=hyperparameters), feature_schema=schema, objective=objective)
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        assert isinstance(fitted, DecisionFunctionPredictor) == (model_name == "logistic_regression")
