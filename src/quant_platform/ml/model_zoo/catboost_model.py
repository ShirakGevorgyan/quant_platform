"""CatBoost production wrapper (Milestone 4C, "CATBOOST") -- binary
classification and regression, with NATIVE categorical feature support
(the one model in this package that declares
`supports_categorical_features=True`).

WHY THIS WRAPPER USES `CatBoostClassifier`/`CatBoostRegressor` DIRECTLY
(UNLIKE `lightgbm_model.py`/`xgboost_model.py`, WHICH AVOID THE
SKLEARN-STYLE WRAPPER)
--------------------------------------------------------------------------
LightGBM's/XGBoost's scikit-learn-compatible wrapper classes can source
randomness from NumPy's global RNG state unless every seed-relevant
setting is passed explicitly -- both of those modules use the lower-
level `train`/`Booster` API instead, for full auditability. CatBoost's
`CatBoostClassifier`/`CatBoostRegressor` classes have no such hidden
global-state concern: `random_seed` is a direct, explicit constructor
parameter with no separate low-level API this platform would gain
anything from routing through instead, and CatBoost's OWN categorical
handling is only exposed through this class-based API (`cat_features`),
so this is both the simpler and the only complete choice.

CATEGORICAL FEATURE DETECTION AND ORDERING
--------------------------------------------------------------------------
A feature column is treated as categorical iff it is NOT numeric/boolean
dtype after `FeatureSchema.validate_frame`'s column selection --
determined FRESH at every `fit` call from the actual data, never a
caller-declared hyperparameter list that could silently go stale. Predict
time passes the SAME already-validated-and-ordered DataFrame (never
coerced to a NumPy array, unlike every other model in this package,
which would silently destroy categorical dtype information) -- verified
empirically that a reloaded model correctly remembers which columns were
categorical WITHOUT re-declaring `cat_features`, PROVIDED column order at
predict time exactly matches fit time (also verified: CatBoost matches
columns POSITIONALLY internally, not by name, once loaded -- this is
exactly why every call in this module routes through `FeatureSchema.
validate_frame` first, which always returns columns in one fixed,
declared order).
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import catboost  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.model_zoo.common import (
    derive_model_init_seed,
    library_version,
    validate_and_order_features,
)
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration

CATBOOST_MODEL_NAME = "catboost"

_CATBOOST_SCHEMA_VERSION = 1
_RESERVED_HYPERPARAMETER_KEYS = ("random_seed", "random_state", "seed", "thread_count", "allow_writing_files", "verbose")
_NON_CATBOOST_PARAM_KEYS = ("early_stopping_rounds", "validation_fraction")
_DEFAULT_ITERATIONS = 100
_DEFAULT_VALIDATION_FRACTION = 0.1


def _library_version() -> str:
    return library_version(catboost)


def _check_expected_metadata(metadata: ModelMetadata, expected_metadata: ModelMetadata | None) -> None:
    if expected_metadata is not None and metadata != expected_metadata:
        raise FeatureSchemaMismatchError(
            "Deserialized model metadata does not match expected_metadata",
            context={"expected_name": expected_metadata.name, "found_name": metadata.name},
        )


def _split_hyperparameters(hyperparameters: ModelHyperparameters) -> tuple[dict[str, object], int | None, float]:
    values: dict[str, object] = dict(hyperparameters.values)
    reserved_present = [k for k in _RESERVED_HYPERPARAMETER_KEYS if k in values]
    if reserved_present:
        raise ValueError(
            f"{CATBOOST_MODEL_NAME}: hyperparameter key(s) {reserved_present} are reserved -- this platform's "
            "own derived seed, thread-count, and file-writing settings always control these, never a model "
            "hyperparameter"
        )
    values.setdefault("iterations", _DEFAULT_ITERATIONS)
    early_stopping_rounds_raw = values.pop("early_stopping_rounds", None)
    early_stopping_rounds = None if early_stopping_rounds_raw is None else int(early_stopping_rounds_raw)  # type: ignore[call-overload]
    validation_fraction = float(values.pop("validation_fraction", _DEFAULT_VALIDATION_FRACTION))  # type: ignore[arg-type]
    return values, early_stopping_rounds, validation_fraction


def _categorical_columns(ordered_features: pd.DataFrame) -> list[str]:
    return [
        str(name) for name, dtype in ordered_features.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    ]


def _estimator_class(objective: ObjectiveType) -> type[CatBoostClassifier] | type[CatBoostRegressor]:
    if objective is ObjectiveType.BINARY_CLASSIFICATION:
        return CatBoostClassifier  # type: ignore[no-any-return]
    if objective is ObjectiveType.REGRESSION:
        return CatBoostRegressor  # type: ignore[no-any-return]
    raise ValueError(f"{CATBOOST_MODEL_NAME} does not support objective {objective.value!r}")  # pragma: no cover - guarded upstream by ModelCapabilities.supports


@dataclass(frozen=True, slots=True)
class FittedCatBoostModel:
    metadata: ModelMetadata
    model_cbm_base64: str
    categorical_features: tuple[str, ...]

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return () if self.metadata.objective is ObjectiveType.REGRESSION else (0, 1)

    def _estimator(self) -> CatBoostClassifier | CatBoostRegressor:
        estimator = _estimator_class(self.metadata.objective)()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.cbm"
            path.write_bytes(base64.b64decode(self.model_cbm_base64))
            estimator.load_model(str(path), format="cbm")
        return estimator

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        raw = self._estimator().predict(ordered)
        return np.asarray(raw, dtype="float64").reshape(-1)

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        return np.asarray(self._estimator().predict_proba(ordered), dtype="float64")

    def feature_importance(self) -> dict[str, float]:
        """Model-NATIVE importance only (`get_feature_importance`, CatBoost's
        PredictionValuesChange by default) -- never SHAP, never
        permutation importance."""
        values = self._estimator().get_feature_importance()
        return dict(zip(self.metadata.feature_schema.feature_names, (float(v) for v in values), strict=True))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _CATBOOST_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "model_cbm_base64": self.model_cbm_base64, "categorical_features": list(self.categorical_features),
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedCatBoostModel:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_CATBOOST_SCHEMA_VERSION, context="FittedCatBoostModel")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(
                metadata=metadata, model_cbm_base64=str(raw["model_cbm_base64"]),
                categorical_features=tuple(str(c) for c in raw.get("categorical_features") or []),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedCatBoostModel: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class CatBoostModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedCatBoostModel:
        ordered = validate_and_order_features(self.metadata.feature_schema, features)
        if len(ordered) != len(labels):
            raise ValueError(f"features ({len(ordered)} rows) and labels ({len(labels)} rows) must have matching length")
        seed = derive_model_init_seed(seeds)
        cat_kwargs, early_stopping_rounds, validation_fraction = _split_hyperparameters(self.metadata.hyperparameters)
        categorical_features = _categorical_columns(ordered)

        estimator_kwargs: dict[str, object] = {
            "random_seed": seed, "thread_count": 1, "allow_writing_files": False, "verbose": False,
            "cat_features": categorical_features or None, **cat_kwargs,
        }
        estimator = _estimator_class(self.metadata.objective)(**estimator_kwargs)

        if early_stopping_rounds is not None and 0 < validation_fraction < 1 and len(ordered) >= 4:
            split_at = max(1, int(len(ordered) * (1.0 - validation_fraction)))
            split_at = min(split_at, len(ordered) - 1)
            estimator.set_params(early_stopping_rounds=early_stopping_rounds)
            estimator.fit(
                ordered.iloc[:split_at], labels.iloc[:split_at],
                eval_set=(ordered.iloc[split_at:], labels.iloc[split_at:]),
            )
        else:
            estimator.fit(ordered, labels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.cbm"
            estimator.save_model(str(path), format="cbm")
            model_cbm_base64 = base64.b64encode(path.read_bytes()).decode("ascii")

        return FittedCatBoostModel(metadata=self.metadata, model_cbm_base64=model_cbm_base64, categorical_features=tuple(categorical_features))


class CatBoostModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION, ObjectiveType.REGRESSION),
            supports_predict_proba=True, supports_feature_importance=True, supports_missing_values=True,
            supports_categorical_features=True, is_deterministic=True,
            seed_usage="controls native RNG for the Bayesian bootstrap, row/column sampling, and "
            "random_strength split-score perturbation; has no visible effect if random_strength=0 and "
            "bagging_temperature=0 are both explicitly configured",
            required_preprocessing="none -- tree splits are scale-invariant; missing values and categorical "
            "features are both handled natively",
            library_name="catboost",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> CatBoostModel:
        metadata = ModelMetadata(
            name=CATBOOST_MODEL_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return CatBoostModel(metadata=metadata)


class CatBoostModelSerializer:
    def serialize(self, model: FittedCatBoostModel) -> bytes:
        return model.to_bytes()


class CatBoostModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedCatBoostModel:
        return FittedCatBoostModel.from_bytes(data, expected_metadata=expected_metadata)


__all__ = [
    "CATBOOST_MODEL_NAME",
    "CatBoostModel",
    "CatBoostModelDeserializer",
    "CatBoostModelFactory",
    "CatBoostModelSerializer",
    "FittedCatBoostModel",
]
