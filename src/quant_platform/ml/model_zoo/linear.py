"""scikit-learn `LogisticRegression` (binary classification) and
`ElasticNet` (regression) -- Milestone 4C's "LINEAR MODELS".

SERIALIZATION WITHOUT PICKLE: RECONSTRUCT A FRESH ESTIMATOR, DON'T PICKLE
THE FITTED ONE
--------------------------------------------------------------------------
Both estimators' entire fitted behavior (`predict`/`predict_proba`/
`decision_function`) is a pure function of a handful of learned NumPy
arrays (`coef_`, `intercept_`, `classes_` for `LogisticRegression`;
`coef_`, `intercept_` for `ElasticNet`) plus `n_iter_` (kept for
provenance/debugging, not read by prediction). Serializing means
extracting exactly those arrays into an explicit JSON envelope;
deserializing means constructing a FRESH, unfitted estimator instance and
assigning those same attributes directly (scikit-learn's own prediction
methods only ever read them, never anything set by `.fit()` alone) --
verified empirically (see this module's test suite) to reproduce
bit-identical `predict`/`predict_proba`/`decision_function` output to the
original fitted instance. No `pickle`/`joblib.dump` anywhere.

WHY THE SEED IS NEVER CALLER-OVERRIDABLE
--------------------------------------------------------------------------
`random_state` is ALWAYS set to this platform's own derived seed
(`model_zoo.common.derive_model_init_seed`), never left to a
hyperparameter value -- accepting a caller-supplied `random_state`/`seed`
hyperparameter (silently honoring it, or silently overriding it without
saying so) would either break this platform's single-source-of-truth
seed propagation or silently discard a value the caller thought was
taking effect. Both models' factories reject `random_state`/`seed` keys
in `ModelHyperparameters.values` outright, with a clear error explaining
why, rather than doing either silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from sklearn.linear_model import ElasticNet, LogisticRegression  # type: ignore[import-untyped]

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.model_zoo.common import (
    derive_model_init_seed,
    library_version,
    numeric_feature_matrix,
    validate_and_order_features,
)
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration

LOGISTIC_REGRESSION_MODEL_NAME = "logistic_regression"
ELASTIC_NET_MODEL_NAME = "elastic_net"

_LINEAR_SCHEMA_VERSION = 1
_RESERVED_HYPERPARAMETER_KEYS = ("random_state", "seed")


def _reject_reserved_hyperparameters(hyperparameters: ModelHyperparameters, *, model_name: str) -> dict[str, object]:
    values: dict[str, object] = dict(hyperparameters.values)
    reserved_present = [k for k in _RESERVED_HYPERPARAMETER_KEYS if k in values]
    if reserved_present:
        raise ValueError(
            f"{model_name}: hyperparameter key(s) {reserved_present} are reserved -- this platform's own "
            "derived seed always controls determinism; declare seeds via ExperimentSpec.seed_configuration, "
            "never via a model hyperparameter"
        )
    return values


def _library_version() -> str:
    return library_version(sklearn)


def _check_expected_metadata(metadata: ModelMetadata, expected_metadata: ModelMetadata | None) -> None:
    if expected_metadata is not None and metadata != expected_metadata:
        raise FeatureSchemaMismatchError(
            "Deserialized model metadata does not match expected_metadata",
            context={"expected_name": expected_metadata.name, "found_name": metadata.name},
        )


# --------------------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedLogisticRegressionModel:
    metadata: ModelMetadata
    coef: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    classes: tuple[float, ...]
    n_iter: tuple[int, ...]

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self.classes

    def _reconstruct(self) -> LogisticRegression:
        estimator = LogisticRegression()
        estimator.coef_ = np.asarray(self.coef, dtype="float64")
        estimator.intercept_ = np.asarray(self.intercept, dtype="float64")
        estimator.classes_ = np.asarray(self.classes, dtype="float64")
        estimator.n_iter_ = np.asarray(self.n_iter, dtype="int64")
        return estimator

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        return np.asarray(self._reconstruct().predict(matrix), dtype="float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        return np.asarray(self._reconstruct().predict_proba(matrix), dtype="float64")

    def decision_function(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        return np.asarray(self._reconstruct().decision_function(matrix), dtype="float64")

    def feature_importance(self) -> dict[str, float]:
        """Model-native "importance" for a linear model: the fitted
        coefficient magnitude per feature -- NOT SHAP, NOT permutation
        importance (both explicitly out of scope). Meaningful only when
        features share a comparable scale; this model declares
        `required_preprocessing` accordingly (see the factory below)."""
        coefs = np.asarray(self.coef, dtype="float64")[0]
        return dict(zip(self.metadata.feature_schema.feature_names, (float(abs(c)) for c in coefs), strict=True))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _LINEAR_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "coef": [list(row) for row in self.coef], "intercept": list(self.intercept),
            "classes": list(self.classes), "n_iter": list(self.n_iter),
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedLogisticRegressionModel:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_LINEAR_SCHEMA_VERSION, context="FittedLogisticRegressionModel")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(
                metadata=metadata, coef=tuple(tuple(float(v) for v in row) for row in raw["coef"]),
                intercept=tuple(float(v) for v in raw["intercept"]), classes=tuple(float(v) for v in raw["classes"]),
                n_iter=tuple(int(v) for v in raw["n_iter"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedLogisticRegressionModel: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class LogisticRegressionModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedLogisticRegressionModel:
        ordered = validate_and_order_features(self.metadata.feature_schema, features)
        if len(ordered) != len(labels):
            raise ValueError(f"features ({len(ordered)} rows) and labels ({len(labels)} rows) must have matching length")
        matrix = numeric_feature_matrix(ordered)
        seed = derive_model_init_seed(seeds)
        kwargs = _reject_reserved_hyperparameters(self.metadata.hyperparameters, model_name=LOGISTIC_REGRESSION_MODEL_NAME)
        estimator = LogisticRegression(random_state=seed, **kwargs)
        estimator.fit(matrix, labels.to_numpy())
        return FittedLogisticRegressionModel(
            metadata=self.metadata,
            coef=tuple(tuple(float(v) for v in row) for row in estimator.coef_),
            intercept=tuple(float(v) for v in estimator.intercept_),
            classes=tuple(float(v) for v in estimator.classes_),
            n_iter=tuple(int(v) for v in np.atleast_1d(estimator.n_iter_)),
        )


class LogisticRegressionModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,), supports_predict_proba=True,
            supports_feature_importance=True, supports_missing_values=False, supports_categorical_features=False,
            is_deterministic=True,
            seed_usage="random_state controls scikit-learn's internal solver initialization (lbfgs is "
            "otherwise a deterministic optimizer; the seed is set for consistency and forward-compatibility "
            "with stochastic solvers such as 'saga')",
            required_preprocessing="numeric features only, no missing values; coefficient-magnitude feature "
            "importance is only comparable across features when they share a comparable scale",
            requires_scaled_numeric_features=True,
            library_name="scikit-learn",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> LogisticRegressionModel:
        metadata = ModelMetadata(
            name=LOGISTIC_REGRESSION_MODEL_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return LogisticRegressionModel(metadata=metadata)


class LogisticRegressionModelSerializer:
    def serialize(self, model: FittedLogisticRegressionModel) -> bytes:
        return model.to_bytes()


class LogisticRegressionModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedLogisticRegressionModel:
        return FittedLogisticRegressionModel.from_bytes(data, expected_metadata=expected_metadata)


# --------------------------------------------------------------------------
# Elastic Net
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedElasticNetModel:
    metadata: ModelMetadata
    coef: tuple[float, ...]
    intercept: float
    n_iter: int

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return ()

    def _reconstruct(self) -> ElasticNet:
        estimator = ElasticNet()
        estimator.coef_ = np.asarray(self.coef, dtype="float64")
        estimator.intercept_ = float(self.intercept)
        estimator.n_iter_ = self.n_iter
        return estimator

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        return np.asarray(self._reconstruct().predict(matrix), dtype="float64")

    def feature_importance(self) -> dict[str, float]:
        return dict(zip(self.metadata.feature_schema.feature_names, (float(abs(c)) for c in self.coef), strict=True))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _LINEAR_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "coef": list(self.coef), "intercept": self.intercept, "n_iter": self.n_iter,
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedElasticNetModel:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_LINEAR_SCHEMA_VERSION, context="FittedElasticNetModel")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(metadata=metadata, coef=tuple(float(v) for v in raw["coef"]), intercept=float(raw["intercept"]), n_iter=int(raw["n_iter"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedElasticNetModel: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class ElasticNetModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedElasticNetModel:
        ordered = validate_and_order_features(self.metadata.feature_schema, features)
        if len(ordered) != len(labels):
            raise ValueError(f"features ({len(ordered)} rows) and labels ({len(labels)} rows) must have matching length")
        matrix = numeric_feature_matrix(ordered)
        seed = derive_model_init_seed(seeds)
        kwargs = _reject_reserved_hyperparameters(self.metadata.hyperparameters, model_name=ELASTIC_NET_MODEL_NAME)
        kwargs.setdefault("selection", "cyclic")  # deterministic path selection regardless of random_state
        estimator = ElasticNet(random_state=seed, **kwargs)
        estimator.fit(matrix, labels.to_numpy())
        n_iter_raw = np.atleast_1d(estimator.n_iter_)
        return FittedElasticNetModel(
            metadata=self.metadata, coef=tuple(float(v) for v in estimator.coef_),
            intercept=float(estimator.intercept_), n_iter=int(n_iter_raw[0]),
        )


class ElasticNetModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION,), supports_predict_proba=False,
            supports_feature_importance=True, supports_missing_values=False, supports_categorical_features=False,
            is_deterministic=True,
            seed_usage="random_state controls coordinate-descent path selection only when selection='random' "
            "is explicitly requested; this factory defaults selection='cyclic' (fully deterministic "
            "regardless of the seed) unless a caller overrides it",
            required_preprocessing="numeric features only, no missing values; L1/L2 penalty weighting is "
            "only meaningful when features share a comparable scale",
            requires_scaled_numeric_features=True,
            library_name="scikit-learn",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> ElasticNetModel:
        metadata = ModelMetadata(
            name=ELASTIC_NET_MODEL_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return ElasticNetModel(metadata=metadata)


class ElasticNetModelSerializer:
    def serialize(self, model: FittedElasticNetModel) -> bytes:
        return model.to_bytes()


class ElasticNetModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedElasticNetModel:
        return FittedElasticNetModel.from_bytes(data, expected_metadata=expected_metadata)


__all__ = [
    "ELASTIC_NET_MODEL_NAME",
    "LOGISTIC_REGRESSION_MODEL_NAME",
    "ElasticNetModel",
    "ElasticNetModelDeserializer",
    "ElasticNetModelFactory",
    "ElasticNetModelSerializer",
    "FittedElasticNetModel",
    "FittedLogisticRegressionModel",
    "LogisticRegressionModel",
    "LogisticRegressionModelDeserializer",
    "LogisticRegressionModelFactory",
    "LogisticRegressionModelSerializer",
]
