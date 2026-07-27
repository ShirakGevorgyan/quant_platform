from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.model_zoo import lightgbm_model as lgb_wrapper
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


def _fit(objective: ObjectiveType, features: pd.DataFrame, labels: pd.Series, *, seed: int, hyperparameters: dict | None = None):
    factory = lgb_wrapper.LightGBMModelFactory()
    model = factory.create(
        hyperparameters=ModelHyperparameters(values=hyperparameters or {"num_boost_round": 20, "num_leaves": 7}),
        feature_schema=_SCHEMA, objective=objective,
    )
    return model.fit(features, labels, seeds=SeedConfiguration(master_seed=seed))


class TestLightGBMFitPredict:
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

    def test_different_seed_changes_output_with_subsampling(self) -> None:
        """With NO subsampling configured (the default), the seed has
        nothing to influence -- LightGBM's split-finding over the full
        data/feature set is itself deterministic. This is an honest
        documented limitation, not a bug; verified here with subsampling
        explicitly enabled, which DOES make the seed matter."""
        features = _features()
        labels = _clf_labels(features)
        hp = {"num_boost_round": 20, "num_leaves": 7, "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 1}
        f1 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1, hyperparameters=hp)
        f2 = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=2, hyperparameters=hp)
        assert not np.array_equal(f1.predict(features.iloc[:50]), f2.predict(features.iloc[:50]))

    def test_reserved_hyperparameter_keys_rejected(self) -> None:
        factory = lgb_wrapper.LightGBMModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"seed": 5}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
        )
        with pytest.raises(ValueError, match="reserved"):
            model.fit(_features(), _reg_labels(_features()), seeds=SeedConfiguration(master_seed=1))

    def test_feature_importance(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.REGRESSION, features, _reg_labels(features), seed=1)
        importance = fitted.feature_importance()
        assert set(importance) == set(_SCHEMA.feature_names)
        assert all(v >= 0 for v in importance.values())

    def test_serialization_round_trip(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, _clf_labels(features), seed=1)
        data = lgb_wrapper.LightGBMModelSerializer().serialize(fitted)
        restored = lgb_wrapper.LightGBMModelDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert np.array_equal(restored.predict(features.iloc[:10]), fitted.predict(features.iloc[:10]))
        assert np.allclose(restored.predict_proba(features.iloc[:10]), fitted.predict_proba(features.iloc[:10]))

    def test_missing_values_supported_natively(self) -> None:
        features = _features()
        features.loc[0:5, "f1"] = np.nan
        fitted = _fit(ObjectiveType.BINARY_CLASSIFICATION, features, _clf_labels(features), seed=1)
        preds = fitted.predict(features.iloc[:10])
        assert not np.isnan(preds).any()

    def test_early_stopping_produces_a_positive_best_iteration_le_num_boost_round(self) -> None:
        features = _features()
        labels = _clf_labels(features)
        fitted = _fit(
            ObjectiveType.BINARY_CLASSIFICATION, features, labels, seed=1,
            hyperparameters={"num_boost_round": 200, "num_leaves": 7, "early_stopping_rounds": 5, "validation_fraction": 0.2},
        )
        assert 0 < fitted.best_iteration <= 200
        preds = fitted.predict(features.iloc[:10])
        assert not np.isnan(preds).any()

    def test_without_early_stopping_best_iteration_is_zero_meaning_use_all(self) -> None:
        features = _features()
        fitted = _fit(ObjectiveType.REGRESSION, features, _reg_labels(features), seed=1)
        assert fitted.best_iteration == 0
