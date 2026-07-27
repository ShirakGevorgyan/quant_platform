from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.interfaces import DecisionFunctionPredictor, FeatureSchema
from quant_platform.ml.model_zoo import linear
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration

_SCHEMA = FeatureSchema(feature_names=("f1", "f2", "f3"))


def _features(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})


def _clf_labels(features: pd.DataFrame) -> pd.Series:
    return pd.Series((features["f1"] + 0.5 * features["f2"] > 0).astype(int))


def _reg_labels(features: pd.DataFrame) -> pd.Series:
    return pd.Series(features["f1"] * 2 - features["f2"] + 0.5)


class TestLogisticRegressionModel:
    def test_fit_predict_predict_proba_decision_function(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        features = _features()
        fitted = model.fit(features, _clf_labels(features), seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(features.iloc[:10])
        assert set(np.unique(preds)) <= {0.0, 1.0}
        proba = fitted.predict_proba(features.iloc[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert isinstance(fitted, DecisionFunctionPredictor)
        decision = fitted.decision_function(features.iloc[:10])
        assert decision.shape == (10,)

    def test_determinism_same_seed(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        features = _features()
        labels = _clf_labels(features)
        m1 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        m2 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        f1 = m1.fit(features, labels, seeds=SeedConfiguration(master_seed=42))
        f2 = m2.fit(features, labels, seeds=SeedConfiguration(master_seed=42))
        assert f1.coef == f2.coef
        assert np.array_equal(f1.predict(features.iloc[:20]), f2.predict(features.iloc[:20]))

    def test_reserved_hyperparameter_keys_rejected(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"random_state": 7}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        with pytest.raises(ValueError, match="reserved"):
            model.fit(_features(), _clf_labels(_features()), seeds=SeedConfiguration(master_seed=1))

    def test_feature_importance_keyed_by_declared_feature_names(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        features = _features()
        fitted = model.fit(features, _clf_labels(features), seeds=SeedConfiguration(master_seed=1))
        importance = fitted.feature_importance()
        assert set(importance) == set(_SCHEMA.feature_names)
        assert all(v >= 0 for v in importance.values())

    def test_serialization_round_trip_reproduces_predict_proba_and_decision_function(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        features = _features()
        fitted = model.fit(features, _clf_labels(features), seeds=SeedConfiguration(master_seed=1))
        data = linear.LogisticRegressionModelSerializer().serialize(fitted)
        restored = linear.LogisticRegressionModelDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert np.array_equal(restored.predict(features.iloc[:10]), fitted.predict(features.iloc[:10]))
        assert np.allclose(restored.predict_proba(features.iloc[:10]), fitted.predict_proba(features.iloc[:10]))
        assert np.allclose(restored.decision_function(features.iloc[:10]), fitted.decision_function(features.iloc[:10]))

    def test_missing_values_rejected_natively(self) -> None:
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        features = _features()
        features.loc[0, "f1"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            model.fit(features, _clf_labels(features), seeds=SeedConfiguration(master_seed=1))

    def test_constant_labels_raises(self) -> None:
        """Confirms sklearn's own native rejection -- the scenario `ml.
        model_validation`'s pre-fit gate is specifically designed to
        intercept before this is ever reached in the real pipeline."""
        factory = linear.LogisticRegressionModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        features = _features()
        labels = pd.Series([1] * len(features))
        with pytest.raises(ValueError, match="one class"):
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))


class TestElasticNetModel:
    def test_fit_predict(self) -> None:
        factory = linear.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        features = _features()
        fitted = model.fit(features, _reg_labels(features), seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(features.iloc[:10])
        assert preds.shape == (10,)
        assert not np.isnan(preds).any()

    def test_no_predict_proba(self) -> None:
        assert linear.ElasticNetModelFactory.capabilities().supports_predict_proba is False

    def test_determinism_cyclic_selection_default(self) -> None:
        factory = linear.ElasticNetModelFactory()
        features = _features()
        labels = _reg_labels(features)
        m1 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        m2 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        f1 = m1.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        f2 = m2.fit(features, labels, seeds=SeedConfiguration(master_seed=999))  # different seed, cyclic selection -> identical
        assert f1.coef == f2.coef

    def test_reserved_hyperparameter_keys_rejected(self) -> None:
        factory = linear.ElasticNetModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"seed": 3}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
        )
        with pytest.raises(ValueError, match="reserved"):
            model.fit(_features(), _reg_labels(_features()), seeds=SeedConfiguration(master_seed=1))

    def test_feature_importance(self) -> None:
        factory = linear.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={"alpha": 0.01}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        features = _features()
        fitted = model.fit(features, _reg_labels(features), seeds=SeedConfiguration(master_seed=1))
        importance = fitted.feature_importance()
        assert set(importance) == set(_SCHEMA.feature_names)

    def test_serialization_round_trip(self) -> None:
        factory = linear.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        features = _features()
        fitted = model.fit(features, _reg_labels(features), seeds=SeedConfiguration(master_seed=1))
        data = linear.ElasticNetModelSerializer().serialize(fitted)
        restored = linear.ElasticNetModelDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert np.allclose(restored.predict(features.iloc[:10]), fitted.predict(features.iloc[:10]))

    def test_missing_values_rejected_natively(self) -> None:
        factory = linear.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        features = _features()
        features.loc[0, "f1"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            model.fit(features, _reg_labels(features), seeds=SeedConfiguration(master_seed=1))

    def test_high_dimensional_p_greater_than_n(self) -> None:
        """Elastic Net is specifically designed to handle p > n --
        `ml.model_validation`'s high-dimensional check is a WARNING, not
        a rejection, for exactly this model."""
        rng = np.random.default_rng(0)
        n, p = 15, 40
        schema = FeatureSchema(feature_names=tuple(f"f{i}" for i in range(p)))
        features = pd.DataFrame(rng.normal(size=(n, p)), columns=schema.feature_names)
        labels = pd.Series(features["f0"] * 2 - features["f1"])
        factory = linear.ElasticNetModelFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=schema, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(features.iloc[:5])
        assert not np.isnan(preds).any()
