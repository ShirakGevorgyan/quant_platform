from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.model_zoo import catboost_model as cb_wrapper
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration

_SCHEMA = FeatureSchema(feature_names=("f1", "f2", "f3"))


def _features(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})


def _clf_labels(features: pd.DataFrame) -> pd.Series:
    return pd.Series((features["f1"] + 0.3 * features["f2"] > 0).astype(int))


def _reg_labels(features: pd.DataFrame) -> pd.Series:
    return pd.Series(features["f1"] * 2 - features["f2"] + 0.5)


def _fit(objective: ObjectiveType, features: pd.DataFrame, labels: pd.Series, *, seed: int, hyperparameters: dict | None = None, schema: FeatureSchema = _SCHEMA):
    factory = cb_wrapper.CatBoostModelFactory()
    model = factory.create(
        hyperparameters=ModelHyperparameters(values=hyperparameters or {"iterations": 20}),
        feature_schema=schema, objective=objective,
    )
    return model.fit(features, labels, seeds=SeedConfiguration(master_seed=seed))


class TestCatBoostFitPredict:
    def test_binary_classification_predict_and_proba(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, _clf_labels(features), seed=1)
        preds = fitted.predict(features.iloc[:10])
        assert set(np.unique(preds)) <= {0.0, 1.0}
        proba = fitted.predict_proba(features.iloc[:10])
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_regression(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.REGRESSION, features, _reg_labels(features), seed=1)
        preds = fitted.predict(features.iloc[:10])
        assert not np.isnan(preds).any()

    def test_determinism_same_seed(self) -> None:
        features = _features()
        labels = _clf_labels(features)
        f1 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=42)
        f2 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=42)
        assert np.array_equal(f1.predict(features.iloc[:30]), f2.predict(features.iloc[:30]))
        assert np.allclose(f1.predict_proba(features.iloc[:30]), f2.predict_proba(features.iloc[:30]))

    def test_different_seed_changes_output_with_random_strength(self) -> None:
        features = _features()
        labels = _clf_labels(features)
        hp = {"iterations": 50, "random_strength": 5.0}
        f1 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1, hyperparameters=hp)
        f2 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=2, hyperparameters=hp)
        assert not np.allclose(f1.predict_proba(features.iloc[:50]), f2.predict_proba(features.iloc[:50]))

    def test_reserved_hyperparameter_keys_rejected(self) -> None:
        factory = cb_wrapper.CatBoostModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"random_seed": 5}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
        )
        with pytest.raises(ValueError, match="reserved"):
            model.fit(_features(), _reg_labels(_features()), seeds=SeedConfiguration(master_seed=1))

    def test_feature_importance(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.REGRESSION, features, _reg_labels(features), seed=1)
        importance = fitted.feature_importance()
        assert set(importance) == set(_SCHEMA.feature_names)

    def test_serialization_round_trip(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, _clf_labels(features), seed=1)
        data = cb_wrapper.CatBoostModelSerializer().serialize(fitted)
        restored = cb_wrapper.CatBoostModelDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert np.array_equal(restored.predict(features.iloc[:10]), fitted.predict(features.iloc[:10]))
        assert np.allclose(restored.predict_proba(features.iloc[:10]), fitted.predict_proba(features.iloc[:10]))

    def test_missing_values_supported_natively(self) -> None:
        features = _features()
        features.loc[0:5, "f1"] = np.nan
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, _clf_labels(features), seed=1)
        preds = fitted.predict(features.iloc[:10])
        assert not np.isnan(preds).any()

    def test_early_stopping_produces_a_result_without_error(self) -> None:
        features = _features()
        labels = _clf_labels(features)
        fitted = _fit(
            ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1,
            hyperparameters={"iterations": 300, "early_stopping_rounds": 5, "validation_fraction": 0.2},
        )
        preds = fitted.predict(features.iloc[:10])
        assert not np.isnan(preds).any()


class TestCatBoostCategoricalSupport:
    def test_categorical_feature_handled_natively(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        features = pd.DataFrame({"num1": rng.normal(size=n), "cat1": rng.choice(["a", "b", "c"], size=n)})
        labels = pd.Series((features["num1"] > 0).astype(int))
        schema = FeatureSchema(feature_names=("num1", "cat1"))
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1, schema=schema)
        assert fitted.categorical_features == ("cat1",)
        preds = fitted.predict(features.iloc[:10])
        assert set(np.unique(preds)) <= {0.0, 1.0}

    def test_categorical_feature_survives_serialization_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        features = pd.DataFrame({"num1": rng.normal(size=n), "cat1": rng.choice(["a", "b", "c"], size=n)})
        labels = pd.Series((features["num1"] > 0).astype(int))
        schema = FeatureSchema(feature_names=("num1", "cat1"))
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1, schema=schema)
        data = cb_wrapper.CatBoostModelSerializer().serialize(fitted)
        restored = cb_wrapper.CatBoostModelDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert np.array_equal(restored.predict(features.iloc[:10]), fitted.predict(features.iloc[:10]))

    def test_purely_numeric_dataset_has_no_categorical_features(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.REGRESSION, features, _reg_labels(features), seed=1)
        assert fitted.categorical_features == ()
