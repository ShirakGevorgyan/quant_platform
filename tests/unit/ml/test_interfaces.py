from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.interfaces import (
    FeatureColumnPolicy,
    FeatureSchema,
    FittedModel,
    ModelFactory,
    ModelMetadata,
    Predictor,
    ProbabilisticPredictor,
    TrainableModel,
)
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory


class TestFeatureSchema:
    def test_round_trip(self) -> None:
        schema = FeatureSchema(feature_names=("a", "b", "c"))
        assert FeatureSchema.from_json_dict(schema.to_json_dict()) == schema

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            FeatureSchema(feature_names=())

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            FeatureSchema(feature_names=("a", "a"))

    def test_validate_frame_reorders_columns_to_schema_order(self) -> None:
        schema = FeatureSchema(feature_names=("a", "b"))
        frame = pd.DataFrame({"b": [1, 2], "a": [3, 4]})
        result = schema.validate_frame(frame)
        assert list(result.columns) == ["a", "b"]

    def test_validate_frame_missing_column_raises(self) -> None:
        schema = FeatureSchema(feature_names=("a", "b"))
        frame = pd.DataFrame({"a": [1]})
        with pytest.raises(FeatureSchemaMismatchError, match="Missing"):
            schema.validate_frame(frame)

    def test_validate_frame_extra_column_strict_raises(self) -> None:
        schema = FeatureSchema(feature_names=("a",))
        frame = pd.DataFrame({"a": [1], "extra": [2]})
        with pytest.raises(FeatureSchemaMismatchError, match="extra"):
            schema.validate_frame(frame, policy=FeatureColumnPolicy.STRICT)

    def test_validate_frame_extra_column_ignore_extra_accepted(self) -> None:
        schema = FeatureSchema(feature_names=("a",))
        frame = pd.DataFrame({"a": [1], "extra": [2]})
        result = schema.validate_frame(frame, policy=FeatureColumnPolicy.IGNORE_EXTRA)
        assert list(result.columns) == ["a"]

    def test_validate_frame_missing_still_raised_under_ignore_extra(self) -> None:
        schema = FeatureSchema(feature_names=("a", "b"))
        frame = pd.DataFrame({"a": [1]})
        with pytest.raises(FeatureSchemaMismatchError, match="Missing"):
            schema.validate_frame(frame, policy=FeatureColumnPolicy.IGNORE_EXTRA)


class TestModelMetadata:
    def _schema(self) -> FeatureSchema:
        return FeatureSchema(feature_names=("a", "b"))

    def test_round_trip(self) -> None:
        metadata = ModelMetadata(
            name="m", version="1", objective=ObjectiveType.REGRESSION, feature_schema=self._schema(),
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
            hyperparameters=ModelHyperparameters(values={"x": 1}),
        )
        assert ModelMetadata.from_json_dict(metadata.to_json_dict()) == metadata

    def test_objective_not_in_capabilities_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in capabilities"):
            ModelMetadata(
                name="m", version="1", objective=ObjectiveType.BINARY_CLASSIFICATION, feature_schema=self._schema(),
                capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
                hyperparameters=ModelHyperparameters(),
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ModelMetadata(
                name="", version="1", objective=ObjectiveType.REGRESSION, feature_schema=self._schema(),
                capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
                hyperparameters=ModelHyperparameters(),
            )

    def test_empty_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="version must not be empty"):
            ModelMetadata(
                name="m", version="", objective=ObjectiveType.REGRESSION, feature_schema=self._schema(),
                capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
                hyperparameters=ModelHyperparameters(),
            )


class TestStructuralProtocols:
    """The core design choice under test: `TrainableModel.fit(...)`
    returns a NEW `FittedModel` object -- an unfit model has no `predict`
    at all, making "predict before fit must fail" structural rather than
    a runtime flag check."""

    def _make_unfit(self, objective: ObjectiveType = ObjectiveType.REGRESSION) -> object:
        factory: ModelFactory = ConstantTestModelFactory()
        schema = FeatureSchema(feature_names=("a", "b"))
        return factory.create(hyperparameters=ModelHyperparameters(), feature_schema=schema, objective=objective)

    def test_unfit_model_is_trainable_not_fitted(self) -> None:
        unfit = self._make_unfit()
        assert isinstance(unfit, TrainableModel)
        assert not isinstance(unfit, FittedModel)
        assert not hasattr(unfit, "predict")

    def test_fit_returns_separate_fitted_object(self) -> None:
        unfit = self._make_unfit()
        features = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        labels = pd.Series([1.0, 2.0, 3.0])
        fitted = unfit.fit(features, labels, seeds=SeedConfiguration(master_seed=1))  # type: ignore[attr-defined]
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted, Predictor)
        assert fitted.is_fitted
        assert fitted is not unfit

    def test_predict_proba_on_regression_only_capability_raises(self) -> None:
        from quant_platform.core.exceptions import UnsupportedObjectiveError
        from quant_platform.ml.testing import ConstantTestModel

        schema = FeatureSchema(feature_names=("a",))
        capabilities = ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), supports_predict_proba=False)
        metadata = ModelMetadata(
            name="m", version="1", objective=ObjectiveType.REGRESSION, feature_schema=schema,
            capabilities=capabilities, hyperparameters=ModelHyperparameters(),
        )
        model = ConstantTestModel(metadata=metadata)
        fitted = model.fit(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1.0, 2.0]), seeds=SeedConfiguration(master_seed=1))
        # NOTE: `isinstance(fitted, ProbabilisticPredictor)` is still True here --
        # a `runtime_checkable` Protocol only checks method PRESENCE, not the
        # capability flag. The actual gate is `require_predict_proba` at call time.
        with pytest.raises(UnsupportedObjectiveError):
            fitted.predict_proba(pd.DataFrame({"a": [1.0]}))

    def test_classification_model_supports_predict_proba_with_class_labels(self) -> None:
        features = pd.DataFrame({"a": [1.0, 0.0, 1.0, 0.0]})
        labels = pd.Series([1.0, 0.0, 1.0, 0.0])
        # _make_unfit's feature_schema is ("a","b"); build a matching
        # single-column schema directly for this test instead.
        factory = ConstantTestModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(), feature_schema=FeatureSchema(feature_names=("a",)),
            objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        assert isinstance(fitted, ProbabilisticPredictor)
        proba = fitted.predict_proba(pd.DataFrame({"a": [1.0, 0.0]}))
        assert proba.shape == (2, 2)
        assert fitted.class_labels == (0, 1)

    def test_serialization_round_trip_preserves_predictions(self) -> None:
        from quant_platform.ml.testing import ConstantTestModelDeserializer, ConstantTestModelSerializer

        unfit = self._make_unfit()
        features = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        labels = pd.Series([10.0, 20.0])
        fitted = unfit.fit(features, labels, seeds=SeedConfiguration(master_seed=1))  # type: ignore[attr-defined]
        serialized = ConstantTestModelSerializer().serialize(fitted)
        restored = ConstantTestModelDeserializer().deserialize(serialized, expected_metadata=fitted.metadata)
        assert (restored.predict(features) == fitted.predict(features)).all()

    def test_deserialize_with_mismatched_expected_metadata_raises(self) -> None:
        from quant_platform.ml.testing import ConstantTestModelDeserializer, ConstantTestModelSerializer

        unfit = self._make_unfit()
        features = pd.DataFrame({"a": [1.0], "b": [2.0]})
        labels = pd.Series([1.0])
        fitted = unfit.fit(features, labels, seeds=SeedConfiguration(master_seed=1))  # type: ignore[attr-defined]
        serialized = ConstantTestModelSerializer().serialize(fitted)

        other_schema = FeatureSchema(feature_names=("different",))
        other_metadata = ModelMetadata(
            name="other", version="1", objective=ObjectiveType.REGRESSION, feature_schema=other_schema,
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
            hyperparameters=ModelHyperparameters(),
        )
        with pytest.raises(FeatureSchemaMismatchError):
            ConstantTestModelDeserializer().deserialize(serialized, expected_metadata=other_metadata)
