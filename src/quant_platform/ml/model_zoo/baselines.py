"""The four mandatory comparison baselines (Milestone 4C, "BASELINES"):
`ConstantPredictor`, `RandomPredictor`, `MajorityPredictor`,
`DummyMeanRegressor`. "Never declare a model successful unless it
statistically outperforms every baseline" (see `ml.comparison`) is only
meaningful if these four are genuinely simple, genuinely deterministic-
given-seed, and never secretly "cheat" by looking at test-time feature
values -- none of the four ever reads a single feature VALUE; only
`FeatureSchema` column presence/row count is checked, exactly the same
"ignore feature content, only shape matters" contract `ml.testing.
ConstantTestModel` already established for its own, non-registered,
test-only purpose.

WHY `predict()` IS A PURE FUNCTION OF `(fitted state, input)` FOR
`RandomPredictor` TOO
--------------------------------------------------------------------------
`RandomPredictor` resamples from the training label distribution, but
"random" must not mean "a different answer every time `predict()` is
called on the SAME fitted model with the SAME input" -- that would fail
this milestone's own "prediction determinism"/"seed reproducibility"
adversarial-audit requirement. Its RNG is re-seeded FRESH inside every
`predict()`/`predict_proba()` call (never a mutable generator advancing
across calls), from `(base_seed, row_count)` alone -- deliberately
INDEPENDENT of feature content, consistent with this model never reading
feature values at all. Calling `predict()` twice on the same fitted
model and the same row count always returns the identical array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.model_zoo.common import validate_and_order_features
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration, SeedDomain

CONSTANT_PREDICTOR_NAME = "constant_predictor"
RANDOM_PREDICTOR_NAME = "random_predictor"
MAJORITY_PREDICTOR_NAME = "majority_predictor"
DUMMY_MEAN_REGRESSOR_NAME = "dummy_mean_regressor"

_BASELINE_SCHEMA_VERSION = 1


def _class_labels_for(objective: ObjectiveType) -> tuple[object, ...]:
    return () if objective is ObjectiveType.REGRESSION else (0, 1)


def _hard_labels_for_constant(constant: float, n: int) -> np.ndarray:
    hard = 1.0 if constant >= 0.5 else 0.0
    return np.full(n, hard, dtype="float64")


def _proba_columns_for_constant(constant: float, n: int) -> np.ndarray:
    positive_rate = float(np.clip(constant, 0.0, 1.0))
    return np.column_stack([np.full(n, 1.0 - positive_rate), np.full(n, positive_rate)])


# --------------------------------------------------------------------------
# Constant Predictor
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedConstantPredictor:
    metadata: ModelMetadata
    constant: float
    fitted_class_labels: tuple[object, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self.fitted_class_labels

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        if self.metadata.objective is ObjectiveType.REGRESSION:
            return np.full(n, self.constant, dtype="float64")
        return _hard_labels_for_constant(self.constant, n)

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        return _proba_columns_for_constant(self.constant, n)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _BASELINE_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "constant": self.constant, "class_labels": list(self.fitted_class_labels),
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedConstantPredictor:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_BASELINE_SCHEMA_VERSION, context="FittedConstantPredictor")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(metadata=metadata, constant=float(raw["constant"]), fitted_class_labels=tuple(raw["class_labels"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedConstantPredictor: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class ConstantPredictor:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedConstantPredictor:
        validate_and_order_features(self.metadata.feature_schema, features)
        if len(features) != len(labels):
            raise ValueError(f"features ({len(features)} rows) and labels ({len(labels)} rows) must have matching length")
        _ = seeds.derive(SeedDomain.MODEL_INIT)  # unused: constant is declared, not fit -- derived anyway, see class docstring below
        constant = float(self.metadata.hyperparameters.values.get("constant", 0.5))  # type: ignore[arg-type]
        return FittedConstantPredictor(metadata=self.metadata, constant=constant, fitted_class_labels=_class_labels_for(self.metadata.objective))


class ConstantPredictorFactory:
    """`constant` (hyperparameter, default `0.5`) is a DECLARED value,
    never data-derived -- distinguishes this from `DummyMeanRegressor`
    (whose constant IS the training mean). `seeds.derive(MODEL_INIT)` is
    still called during `fit` (unused for anything) purely so this model
    exercises the same seed-propagation code path every other model
    does, rather than silently skipping it -- consistent with `ml.
    testing.ConstantTestModel`'s identical choice."""

    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
            supports_predict_proba=True, supports_missing_values=True, supports_categorical_features=True,
            is_deterministic=True, seed_usage="unused -- the predicted constant is a declared hyperparameter, never fit from data or randomness",
            required_preprocessing="none",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> ConstantPredictor:
        metadata = ModelMetadata(
            name=CONSTANT_PREDICTOR_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return ConstantPredictor(metadata=metadata)


class ConstantPredictorSerializer:
    def serialize(self, model: FittedConstantPredictor) -> bytes:
        return model.to_bytes()


class ConstantPredictorDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedConstantPredictor:
        return FittedConstantPredictor.from_bytes(data, expected_metadata=expected_metadata)


# --------------------------------------------------------------------------
# Random Predictor
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedRandomPredictor:
    metadata: ModelMetadata
    training_labels: tuple[float, ...]
    base_seed: int
    fitted_class_labels: tuple[object, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self.fitted_class_labels

    def _rng_for(self, n: int) -> np.random.Generator:
        # Re-seeded FRESH every call, from (base_seed, row_count) alone --
        # see module docstring for why this makes predict() pure.
        return np.random.default_rng([self.base_seed, n])

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        pool = np.asarray(self.training_labels, dtype="float64")
        rng = self._rng_for(n)
        draws = rng.choice(pool, size=n, replace=True)
        if self.metadata.objective is ObjectiveType.REGRESSION:
            return draws
        return (draws >= 0.5).astype("float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        """The empirical training positive rate, constant per row -- NOT a
        second independent random draw. A stochastic hard-label guess
        (`predict`) and a well-defined probability ESTIMATE
        (`predict_proba`) are different questions; this model answers
        both honestly rather than inventing a fake per-row probability
        for a decision that was actually made by a coin flip."""
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        positive_rate = float(np.mean(np.asarray(self.training_labels, dtype="float64")))
        return _proba_columns_for_constant(positive_rate, n)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _BASELINE_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "training_labels": list(self.training_labels), "base_seed": self.base_seed,
            "class_labels": list(self.fitted_class_labels),
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedRandomPredictor:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_BASELINE_SCHEMA_VERSION, context="FittedRandomPredictor")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(
                metadata=metadata, training_labels=tuple(float(v) for v in raw["training_labels"]),
                base_seed=int(raw["base_seed"]), fitted_class_labels=tuple(raw["class_labels"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedRandomPredictor: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class RandomPredictor:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedRandomPredictor:
        validate_and_order_features(self.metadata.feature_schema, features)
        if len(features) != len(labels):
            raise ValueError(f"features ({len(features)} rows) and labels ({len(labels)} rows) must have matching length")
        if len(labels) == 0:
            raise ValueError("RandomPredictor.fit requires at least one label to resample from")
        base_seed = seeds.derive(SeedDomain.MODEL_INIT)
        return FittedRandomPredictor(
            metadata=self.metadata, training_labels=tuple(float(v) for v in labels.to_numpy()), base_seed=base_seed,
            fitted_class_labels=_class_labels_for(self.metadata.objective),
        )


class RandomPredictorFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
            supports_predict_proba=True, supports_missing_values=True, supports_categorical_features=True,
            is_deterministic=True,
            seed_usage="seeds a fresh numpy Generator on every predict() call (derived from the fit-time "
            "seed plus the row count only) to bootstrap-resample training labels",
            required_preprocessing="none",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> RandomPredictor:
        metadata = ModelMetadata(
            name=RANDOM_PREDICTOR_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return RandomPredictor(metadata=metadata)


class RandomPredictorSerializer:
    def serialize(self, model: FittedRandomPredictor) -> bytes:
        return model.to_bytes()


class RandomPredictorDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedRandomPredictor:
        return FittedRandomPredictor.from_bytes(data, expected_metadata=expected_metadata)


# --------------------------------------------------------------------------
# Majority Predictor (classification only)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedMajorityPredictor:
    metadata: ModelMetadata
    majority_class: float
    fitted_class_labels: tuple[object, ...] = (0, 1)

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self.fitted_class_labels

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        return np.full(n, self.majority_class, dtype="float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        """Full confidence (100%) in the majority class -- `MajorityPredictor`
        is a HARD decision rule, not a calibrated probability estimate
        (that is `RandomPredictor.predict_proba`'s job)."""
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        return _proba_columns_for_constant(self.majority_class, n)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _BASELINE_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "majority_class": self.majority_class, "class_labels": list(self.fitted_class_labels),
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedMajorityPredictor:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_BASELINE_SCHEMA_VERSION, context="FittedMajorityPredictor")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(metadata=metadata, majority_class=float(raw["majority_class"]), fitted_class_labels=tuple(raw["class_labels"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedMajorityPredictor: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class MajorityPredictor:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedMajorityPredictor:
        validate_and_order_features(self.metadata.feature_schema, features)
        if len(features) != len(labels):
            raise ValueError(f"features ({len(features)} rows) and labels ({len(labels)} rows) must have matching length")
        if len(labels) == 0:
            raise ValueError("MajorityPredictor.fit requires at least one label")
        _ = seeds.derive(SeedDomain.MODEL_INIT)  # unused: the majority class is a deterministic function of the data alone
        counts = labels.value_counts()
        # Ties broken by the SMALLEST class value, deterministically --
        # never by whatever order pandas' value_counts happens to return.
        top_count = counts.max()
        # `v` is a pandas index label, typed only as `Hashable` -- always
        # genuinely numeric here (labels are always 0.0/1.0 for binary
        # classification), but not something a type checker can prove.
        majority_class = min(float(v) for v, c in counts.items() if c == top_count)  # type: ignore[arg-type]
        return FittedMajorityPredictor(metadata=self.metadata, majority_class=majority_class)


class MajorityPredictorFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), supports_predict_proba=True,
            supports_missing_values=True, supports_categorical_features=True, is_deterministic=True,
            seed_usage="unused -- the majority class is a deterministic function of the training labels alone",
            required_preprocessing="none",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> MajorityPredictor:
        metadata = ModelMetadata(
            name=MAJORITY_PREDICTOR_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return MajorityPredictor(metadata=metadata)


class MajorityPredictorSerializer:
    def serialize(self, model: FittedMajorityPredictor) -> bytes:
        return model.to_bytes()


class MajorityPredictorDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedMajorityPredictor:
        return FittedMajorityPredictor.from_bytes(data, expected_metadata=expected_metadata)


# --------------------------------------------------------------------------
# Dummy Mean Regressor (regression only)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedDummyMeanRegressor:
    metadata: ModelMetadata
    mean_label: float

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return ()

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        n = len(validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy))
        return np.full(n, self.mean_label, dtype="float64")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _BASELINE_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(), "mean_label": self.mean_label,
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedDummyMeanRegressor:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_BASELINE_SCHEMA_VERSION, context="FittedDummyMeanRegressor")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(metadata=metadata, mean_label=float(raw["mean_label"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedDummyMeanRegressor: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class DummyMeanRegressor:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedDummyMeanRegressor:
        validate_and_order_features(self.metadata.feature_schema, features)
        if len(features) != len(labels):
            raise ValueError(f"features ({len(features)} rows) and labels ({len(labels)} rows) must have matching length")
        if len(labels) == 0:
            raise ValueError("DummyMeanRegressor.fit requires at least one label")
        _ = seeds.derive(SeedDomain.MODEL_INIT)  # unused: the mean is a deterministic function of the data alone
        return FittedDummyMeanRegressor(metadata=self.metadata, mean_label=float(labels.mean()))


class DummyMeanRegressorFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION,), supports_predict_proba=False,
            supports_missing_values=True, supports_categorical_features=True, is_deterministic=True,
            seed_usage="unused -- the predicted mean is a deterministic function of the training labels alone",
            required_preprocessing="none",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> DummyMeanRegressor:
        metadata = ModelMetadata(
            name=DUMMY_MEAN_REGRESSOR_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return DummyMeanRegressor(metadata=metadata)


class DummyMeanRegressorSerializer:
    def serialize(self, model: FittedDummyMeanRegressor) -> bytes:
        return model.to_bytes()


class DummyMeanRegressorDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedDummyMeanRegressor:
        return FittedDummyMeanRegressor.from_bytes(data, expected_metadata=expected_metadata)


def _check_expected_metadata(metadata: ModelMetadata, expected_metadata: ModelMetadata | None) -> None:
    if expected_metadata is not None and metadata != expected_metadata:
        from quant_platform.core.exceptions import FeatureSchemaMismatchError

        raise FeatureSchemaMismatchError(
            "Deserialized model metadata does not match expected_metadata",
            context={"expected_name": expected_metadata.name, "found_name": metadata.name},
        )


__all__ = [
    "CONSTANT_PREDICTOR_NAME",
    "DUMMY_MEAN_REGRESSOR_NAME",
    "MAJORITY_PREDICTOR_NAME",
    "RANDOM_PREDICTOR_NAME",
    "ConstantPredictor",
    "ConstantPredictorDeserializer",
    "ConstantPredictorFactory",
    "ConstantPredictorSerializer",
    "DummyMeanRegressor",
    "DummyMeanRegressorDeserializer",
    "DummyMeanRegressorFactory",
    "DummyMeanRegressorSerializer",
    "FittedConstantPredictor",
    "FittedDummyMeanRegressor",
    "FittedMajorityPredictor",
    "FittedRandomPredictor",
    "MajorityPredictor",
    "MajorityPredictorDeserializer",
    "MajorityPredictorFactory",
    "MajorityPredictorSerializer",
    "RandomPredictor",
    "RandomPredictorDeserializer",
    "RandomPredictorFactory",
    "RandomPredictorSerializer",
]
