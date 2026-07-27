"""XGBoost production wrapper (Milestone 4C, "XGBOOST") -- binary
classification and regression, via the raw `xgboost.train`/`Booster`
API (never the `XGBClassifier`/`XGBRegressor` scikit-learn-compatible
wrapper, for the identical "every seed-relevant setting explicit in one
params dict" reason `lightgbm_model.py` documents).

DETERMINISM
--------------------------------------------------------------------------
`tree_method="hist"` with a fixed `seed` and `nthread=1` is deterministic
on the SAME hardware -- verified empirically (see this module's test
suite): identical seed/data reproduces bit-identical predictions;
different seeds produce different predictions whenever `subsample`/
`colsample_bytree` subsampling is configured (with the XGBoost defaults
of no subsampling, the seed has nothing to influence, exactly the same
honest limitation `lightgbm_model.py` documents for LightGBM).

SERIALIZATION: `save_raw`/`load_model`, BASE64-ENCODED FOR THE JSON
ENVELOPE
--------------------------------------------------------------------------
Unlike LightGBM's `model_to_string()` (plain text), XGBoost's native
`Booster.save_raw(raw_format="ubj")` produces a versioned BINARY blob --
still XGBoost's own format, never pickle, but binary bytes need base64
encoding to sit inside this platform's canonical JSON artifact envelope
(`ml.persistence.canonical_json_bytes` requires JSON-primitive values).
`Booster.load_model` accepts that exact blob back unchanged.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

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

XGBOOST_MODEL_NAME = "xgboost"

_XGBOOST_SCHEMA_VERSION = 1
_RESERVED_HYPERPARAMETER_KEYS = ("seed", "random_state", "nthread", "n_jobs", "objective")
_NON_XGBOOST_PARAM_KEYS = ("num_boost_round", "early_stopping_rounds", "validation_fraction")
_DEFAULT_NUM_BOOST_ROUND = 100
_DEFAULT_VALIDATION_FRACTION = 0.1


def _library_version() -> str:
    return library_version(xgb)


def _check_expected_metadata(metadata: ModelMetadata, expected_metadata: ModelMetadata | None) -> None:
    if expected_metadata is not None and metadata != expected_metadata:
        raise FeatureSchemaMismatchError(
            "Deserialized model metadata does not match expected_metadata",
            context={"expected_name": expected_metadata.name, "found_name": metadata.name},
        )


def _split_hyperparameters(hyperparameters: ModelHyperparameters) -> tuple[dict[str, object], int, int | None, float]:
    values: dict[str, object] = dict(hyperparameters.values)
    reserved_present = [k for k in _RESERVED_HYPERPARAMETER_KEYS if k in values]
    if reserved_present:
        raise ValueError(
            f"{XGBOOST_MODEL_NAME}: hyperparameter key(s) {reserved_present} are reserved -- this platform's "
            "own derived seed and thread-count settings always control these, never a model hyperparameter"
        )
    num_boost_round = int(values.pop("num_boost_round", _DEFAULT_NUM_BOOST_ROUND))  # type: ignore[call-overload]
    early_stopping_rounds_raw = values.pop("early_stopping_rounds", None)
    early_stopping_rounds = None if early_stopping_rounds_raw is None else int(early_stopping_rounds_raw)  # type: ignore[call-overload]
    validation_fraction = float(values.pop("validation_fraction", _DEFAULT_VALIDATION_FRACTION))  # type: ignore[arg-type]
    return values, num_boost_round, early_stopping_rounds, validation_fraction


def _objective_param(objective: ObjectiveType) -> str:
    if objective is ObjectiveType.BINARY_CLASSIFICATION:
        return "binary:logistic"
    if objective is ObjectiveType.REGRESSION:
        return "reg:squarederror"
    raise ValueError(f"{XGBOOST_MODEL_NAME} does not support objective {objective.value!r}")  # pragma: no cover - guarded upstream by ModelCapabilities.supports


@dataclass(frozen=True, slots=True)
class FittedXGBoostModel:
    metadata: ModelMetadata
    booster_raw_base64: str
    best_iteration: int | None

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return () if self.metadata.objective is ObjectiveType.REGRESSION else (0, 1)

    def _booster(self) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(bytearray(base64.b64decode(self.booster_raw_base64)))
        return booster

    def _iteration_range(self) -> tuple[int, int]:
        """`(0, 0)` is XGBoost's own documented sentinel for "use every
        boosted round" -- verified empirically to be identical to
        omitting `iteration_range` altogether; used here (rather than
        omitting the argument) so this method always returns one
        concrete value regardless of whether early stopping was used."""
        return (0, 0) if self.best_iteration is None else (0, self.best_iteration + 1)

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        dmatrix = xgb.DMatrix(matrix)
        raw = np.asarray(self._booster().predict(dmatrix, iteration_range=self._iteration_range()), dtype="float64")
        if self.metadata.objective is ObjectiveType.REGRESSION:
            return raw
        return (raw >= 0.5).astype("float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        dmatrix = xgb.DMatrix(matrix)
        positive = np.asarray(self._booster().predict(dmatrix, iteration_range=self._iteration_range()), dtype="float64")
        return np.column_stack([1.0 - positive, positive])

    def feature_importance(self, *, importance_type: str = "gain") -> dict[str, float]:
        """Model-NATIVE importance only (`Booster.get_score`) -- never
        SHAP, never permutation importance. `get_score` omits features
        with zero importance entirely; every declared feature is still
        given an explicit entry here (`0.0` when absent), matching
        `lightgbm_model.py`'s "one value per declared feature" contract."""
        scores = self._booster().get_score(importance_type=importance_type)
        return {
            # `Booster.get_score` is typed to allow `list[float]` (multi-
            # output boosters); this model only ever fits single-output
            # binary/regression objectives, so every value is genuinely a
            # plain `float` at runtime.
            name: float(scores.get(f"f{position}", 0.0))  # type: ignore[arg-type]
            for position, name in enumerate(self.metadata.feature_schema.feature_names)
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _XGBOOST_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "booster_raw_base64": self.booster_raw_base64, "best_iteration": self.best_iteration,
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedXGBoostModel:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_XGBOOST_SCHEMA_VERSION, context="FittedXGBoostModel")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            best_iteration_raw = raw.get("best_iteration")
            return cls(
                metadata=metadata, booster_raw_base64=str(raw["booster_raw_base64"]),
                best_iteration=(None if best_iteration_raw is None else int(best_iteration_raw)),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedXGBoostModel: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class XGBoostModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedXGBoostModel:
        ordered = validate_and_order_features(self.metadata.feature_schema, features)
        if len(ordered) != len(labels):
            raise ValueError(f"features ({len(ordered)} rows) and labels ({len(labels)} rows) must have matching length")
        matrix = numeric_feature_matrix(ordered)
        y = labels.to_numpy(dtype="float64")
        seed = derive_model_init_seed(seeds)
        xgb_params, num_boost_round, early_stopping_rounds, validation_fraction = _split_hyperparameters(self.metadata.hyperparameters)
        params: dict[str, object] = {
            "objective": _objective_param(self.metadata.objective), "seed": seed, "tree_method": "hist",
            "nthread": 1, **xgb_params,
        }

        best_iteration: int | None = None
        if early_stopping_rounds is not None and 0 < validation_fraction < 1 and len(matrix) >= 4:
            split_at = max(1, int(len(matrix) * (1.0 - validation_fraction)))
            split_at = min(split_at, len(matrix) - 1)
            dtrain = xgb.DMatrix(matrix[:split_at], label=y[:split_at])
            dvalid = xgb.DMatrix(matrix[split_at:], label=y[split_at:])
            booster = xgb.train(
                params, dtrain, num_boost_round=num_boost_round, evals=[(dvalid, "validation")],
                early_stopping_rounds=early_stopping_rounds, verbose_eval=False,
            )
            best_iteration = int(booster.best_iteration)
        else:
            dtrain = xgb.DMatrix(matrix, label=y)
            booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        raw_base64 = base64.b64encode(bytes(booster.save_raw(raw_format="ubj"))).decode("ascii")
        return FittedXGBoostModel(metadata=self.metadata, booster_raw_base64=raw_base64, best_iteration=best_iteration)


class XGBoostModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION, ObjectiveType.REGRESSION),
            supports_predict_proba=True, supports_feature_importance=True, supports_missing_values=True,
            supports_categorical_features=False, is_deterministic=True,
            seed_usage="controls native RNG for row/column subsampling (subsample/colsample_bytree) and "
            "split-gain tie-breaking; has no visible effect when subsampling is left at its 1.0 default",
            required_preprocessing="none -- tree splits are scale-invariant; missing values are handled "
            "natively",
            library_name="xgboost",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> XGBoostModel:
        metadata = ModelMetadata(
            name=XGBOOST_MODEL_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return XGBoostModel(metadata=metadata)


class XGBoostModelSerializer:
    def serialize(self, model: FittedXGBoostModel) -> bytes:
        return model.to_bytes()


class XGBoostModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedXGBoostModel:
        return FittedXGBoostModel.from_bytes(data, expected_metadata=expected_metadata)


__all__ = [
    "XGBOOST_MODEL_NAME",
    "FittedXGBoostModel",
    "XGBoostModel",
    "XGBoostModelDeserializer",
    "XGBoostModelFactory",
    "XGBoostModelSerializer",
]
