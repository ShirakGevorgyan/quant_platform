from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.model_zoo import baselines as b
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration

_SCHEMA = FeatureSchema(feature_names=("f1", "f2"))


def _features(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})


def _clf_labels(n: int = 20) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.integers(0, 2, size=n).astype(float))


def _reg_labels(n: int = 20) -> pd.Series:
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(size=n))


class TestConstantPredictor:
    def test_regression_predicts_declared_constant(self) -> None:
        factory = b.ConstantPredictorFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"constant": 3.5}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION,
        )
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(_features(5))
        assert np.array_equal(preds, np.full(5, 3.5))

    def test_never_looks_at_feature_values(self) -> None:
        """The SAME fitted model, given completely different feature
        VALUES (same shape), must predict identically -- proves this
        model genuinely ignores feature content."""
        factory = b.ConstantPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 1.0}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        p1 = fitted.predict(pd.DataFrame({"f1": [0.0, 0.0], "f2": [0.0, 0.0]}))
        p2 = fitted.predict(pd.DataFrame({"f1": [999.0, -999.0], "f2": [5.0, -5.0]}))
        assert np.array_equal(p1, p2)

    def test_classification_predict_proba_matches_constant(self) -> None:
        factory = b.ConstantPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 0.3}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted = model.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=1))
        proba = fitted.predict_proba(_features(4))
        assert np.allclose(proba[:, 1], 0.3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classification_hard_label_thresholds_at_half(self) -> None:
        factory = b.ConstantPredictorFactory()
        below = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 0.3}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        above = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 0.7}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted_below = below.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=1))
        fitted_above = above.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=1))
        assert np.array_equal(fitted_below.predict(_features(3)), np.zeros(3))
        assert np.array_equal(fitted_above.predict(_features(3)), np.ones(3))

    def test_default_constant_is_half(self) -> None:
        factory = b.ConstantPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        assert fitted.constant == 0.5

    def test_serialization_round_trip(self) -> None:
        factory = b.ConstantPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(values={"constant": 0.42}), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        data = b.ConstantPredictorSerializer().serialize(fitted)
        restored = b.ConstantPredictorDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert restored == fitted

    def test_deserialize_rejects_mismatched_expected_metadata(self) -> None:
        factory = b.ConstantPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        data = b.ConstantPredictorSerializer().serialize(fitted)
        other_schema = FeatureSchema(feature_names=("different",))
        other_model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=other_schema, objective=ObjectiveType.REGRESSION)
        other_fitted = other_model.fit(pd.DataFrame({"different": [1.0, 2.0]}), pd.Series([1.0, 2.0]), seeds=SeedConfiguration(master_seed=1))
        from quant_platform.core.exceptions import FeatureSchemaMismatchError

        with pytest.raises(FeatureSchemaMismatchError):
            b.ConstantPredictorDeserializer().deserialize(data, expected_metadata=other_fitted.metadata)


class TestRandomPredictor:
    def test_predict_is_pure_across_repeat_calls(self) -> None:
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted = model.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=7))
        p1 = fitted.predict(_features(10))
        p2 = fitted.predict(_features(10))
        assert np.array_equal(p1, p2)

    def test_different_seed_changes_draws(self) -> None:
        factory = b.RandomPredictorFactory()
        model1 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        model2 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted1 = model1.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=1))
        fitted2 = model2.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=2))
        p1 = fitted1.predict(_features(30))
        p2 = fitted2.predict(_features(30))
        assert not np.array_equal(p1, p2)

    def test_same_seed_refit_is_deterministic(self) -> None:
        factory = b.RandomPredictorFactory()
        model1 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        model2 = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted1 = model1.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=5))
        fitted2 = model2.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=5))
        assert np.array_equal(fitted1.predict(_features(10)), fitted2.predict(_features(10)))

    def test_regression_predictions_are_bootstrap_samples_of_training_labels(self) -> None:
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        labels = _reg_labels()
        fitted = model.fit(_features(), labels, seeds=SeedConfiguration(master_seed=1))
        preds = fitted.predict(_features(50))
        assert set(preds.tolist()) <= set(labels.tolist())

    def test_classification_predict_proba_is_constant_empirical_rate(self) -> None:
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        labels = _clf_labels()
        fitted = model.fit(_features(), labels, seeds=SeedConfiguration(master_seed=1))
        proba = fitted.predict_proba(_features(6))
        expected_rate = float(labels.mean())
        assert np.allclose(proba[:, 1], expected_rate)
        # constant across rows -- this is an ESTIMATE, not per-row noise
        assert len(set(proba[:, 1].tolist())) == 1

    def test_fit_requires_at_least_one_label(self) -> None:
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        with pytest.raises(ValueError, match="at least one label"):
            model.fit(pd.DataFrame({"f1": [], "f2": []}), pd.Series([], dtype="float64"), seeds=SeedConfiguration(master_seed=1))

    def test_serialization_round_trip(self) -> None:
        factory = b.RandomPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=3))
        data = b.RandomPredictorSerializer().serialize(fitted)
        restored = b.RandomPredictorDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert restored == fitted
        assert np.array_equal(restored.predict(_features(5)), fitted.predict(_features(5)))


class TestMajorityPredictor:
    def test_predicts_majority_class(self) -> None:
        factory = b.MajorityPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        labels = pd.Series([0, 0, 0, 1, 1])  # majority is 0
        fitted = model.fit(_features(5), labels, seeds=SeedConfiguration(master_seed=1))
        assert fitted.majority_class == 0.0
        assert np.array_equal(fitted.predict(_features(3)), np.zeros(3))

    def test_tie_breaks_toward_smaller_class_value_deterministically(self) -> None:
        factory = b.MajorityPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        labels = pd.Series([0, 0, 1, 1])  # exact tie
        fitted = model.fit(_features(4), labels, seeds=SeedConfiguration(master_seed=1))
        assert fitted.majority_class == 0.0

    def test_predict_proba_is_full_confidence_in_majority(self) -> None:
        factory = b.MajorityPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        labels = pd.Series([1, 1, 1, 0])
        fitted = model.fit(_features(4), labels, seeds=SeedConfiguration(master_seed=1))
        proba = fitted.predict_proba(_features(3))
        assert np.allclose(proba[:, 1], 1.0)

    def test_serialization_round_trip(self) -> None:
        factory = b.MajorityPredictorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
        fitted = model.fit(_features(), _clf_labels(), seeds=SeedConfiguration(master_seed=1))
        data = b.MajorityPredictorSerializer().serialize(fitted)
        restored = b.MajorityPredictorDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert restored == fitted


class TestDummyMeanRegressor:
    def test_predicts_training_mean(self) -> None:
        factory = b.DummyMeanRegressorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        labels = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fitted = model.fit(_features(5), labels, seeds=SeedConfiguration(master_seed=1))
        assert fitted.mean_label == 3.0
        assert np.array_equal(fitted.predict(_features(4)), np.full(4, 3.0))

    def test_has_no_predict_proba(self) -> None:
        factory = b.DummyMeanRegressorFactory()
        assert factory.capabilities().supports_predict_proba is False

    def test_serialization_round_trip(self) -> None:
        factory = b.DummyMeanRegressorFactory()
        model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
        fitted = model.fit(_features(), _reg_labels(), seeds=SeedConfiguration(master_seed=1))
        data = b.DummyMeanRegressorSerializer().serialize(fitted)
        restored = b.DummyMeanRegressorDeserializer().deserialize(data, expected_metadata=fitted.metadata)
        assert restored == fitted


class TestBaselineCapabilitiesDeclareEverythingHandled:
    """All four baselines never actually read feature VALUES -- they
    must therefore honestly declare support for missing values and
    categorical features (both irrelevant to something that never looks
    at feature content), never falsely under-declare."""

    @pytest.mark.parametrize(
        "factory_cls", [b.ConstantPredictorFactory, b.RandomPredictorFactory, b.MajorityPredictorFactory, b.DummyMeanRegressorFactory],
    )
    def test_declares_missing_value_and_categorical_support(self, factory_cls: type) -> None:
        caps = factory_cls.capabilities()
        assert caps.supports_missing_values is True
        assert caps.supports_categorical_features is True
        assert caps.is_deterministic is True
        assert caps.library_name == "quant-platform"
