"""LightGBM production wrapper (Milestone 4C, "LIGHTGBM") -- binary
classification and regression, via the raw `lightgbm.train`/`Booster`
API (never the `LGBMClassifier`/`LGBMRegressor` scikit-learn-compatible
wrapper, which internally re-derives a random seed from numpy's global
state unless carefully configured -- the raw API's `params` dict makes
every seed-relevant setting explicit and auditable in one place).

DETERMINISM: WHAT `deterministic=True`/`force_row_wise=True` ACTUALLY
BUY, VERIFIED EMPIRICALLY
--------------------------------------------------------------------------
LightGBM's own documentation states that bit-for-bit reproducibility
across runs requires: a fixed `seed` (this wrapper derives it from
`ml.seeds`, never caller-overridable -- see `linear.py`'s identical
policy and this module's `_reject_reserved_hyperparameters`),
`deterministic=True` (forces a fixed floating-point reduction order),
`force_row_wise=True` (avoids a threading-dependent code path), and a
FIXED thread count -- this wrapper additionally pins `num_threads=1`
(never left to the LightGBM default of "all cores", which this
platform's own test suite has confirmed can otherwise vary reduction
order across machines). Verified directly (see this module's test
suite): refitting with an identical seed on identical data produces
BIT-IDENTICAL predictions; refitting with a different seed produces
DIFFERENT predictions whenever `feature_fraction`/`bagging_fraction`
subsampling is configured (with the LightGBM defaults of no subsampling,
the seed has nothing to influence at all -- an honest limitation, not a
bug: split-finding over the full data/feature set is itself
deterministic).

EARLY STOPPING, WITHOUT CHANGING `TrainableModel.fit`'S SIGNATURE
--------------------------------------------------------------------------
`ml.interfaces.TrainableModel.fit(features, labels, *, seeds)` takes only
ONE features/labels pair -- there is no separate validation-set
parameter, and adding one would be exactly the kind of core-interface
redesign this milestone must not do. Early stopping is instead an
entirely SELF-CONTAINED, opt-in hyperparameter: if `early_stopping_rounds`
is set, this wrapper carves the CHRONOLOGICALLY LAST `validation_fraction`
(default `0.1`) slice off the GIVEN training data as an internal
early-stopping validation set (never introducing new look-ahead: that
slice is still entirely within what the fold's own train partition
already contained) and trains with `lightgbm.early_stopping`. Without
`early_stopping_rounds` set, training simply runs for the declared
`num_boost_round` (default `100`), the simpler and default path.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

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

LIGHTGBM_MODEL_NAME = "lightgbm"

_LIGHTGBM_SCHEMA_VERSION = 1
_RESERVED_HYPERPARAMETER_KEYS = ("seed", "random_state", "deterministic", "force_row_wise", "num_threads", "objective")
_NON_LIGHTGBM_PARAM_KEYS = ("num_boost_round", "early_stopping_rounds", "validation_fraction")
_DEFAULT_NUM_BOOST_ROUND = 100
_DEFAULT_VALIDATION_FRACTION = 0.1


def _library_version() -> str:
    return library_version(lgb)


def _check_expected_metadata(metadata: ModelMetadata, expected_metadata: ModelMetadata | None) -> None:
    if expected_metadata is not None and metadata != expected_metadata:
        raise FeatureSchemaMismatchError(
            "Deserialized model metadata does not match expected_metadata",
            context={"expected_name": expected_metadata.name, "found_name": metadata.name},
        )


def _split_hyperparameters(hyperparameters: ModelHyperparameters) -> tuple[dict[str, object], int, int | None, float]:
    """Returns `(lightgbm_params, num_boost_round, early_stopping_rounds, validation_fraction)`."""
    values: dict[str, object] = dict(hyperparameters.values)
    reserved_present = [k for k in _RESERVED_HYPERPARAMETER_KEYS if k in values]
    if reserved_present:
        raise ValueError(
            f"{LIGHTGBM_MODEL_NAME}: hyperparameter key(s) {reserved_present} are reserved -- this platform's "
            "own derived seed and determinism settings always control these, never a model hyperparameter"
        )
    num_boost_round = int(values.pop("num_boost_round", _DEFAULT_NUM_BOOST_ROUND))  # type: ignore[call-overload]
    early_stopping_rounds_raw = values.pop("early_stopping_rounds", None)
    early_stopping_rounds = None if early_stopping_rounds_raw is None else int(early_stopping_rounds_raw)  # type: ignore[call-overload]
    validation_fraction = float(values.pop("validation_fraction", _DEFAULT_VALIDATION_FRACTION))  # type: ignore[arg-type]
    return values, num_boost_round, early_stopping_rounds, validation_fraction


def _objective_param(objective: ObjectiveType) -> str:
    if objective is ObjectiveType.BINARY_CLASSIFICATION:
        return "binary"
    if objective is ObjectiveType.REGRESSION:
        return "regression"
    raise ValueError(f"{LIGHTGBM_MODEL_NAME} does not support objective {objective.value!r}")  # pragma: no cover - guarded upstream by ModelCapabilities.supports


@dataclass(frozen=True, slots=True)
class FittedLightGBMModel:
    metadata: ModelMetadata
    booster_model_string: str
    best_iteration: int

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return () if self.metadata.objective is ObjectiveType.REGRESSION else (0, 1)

    def _booster(self) -> lgb.Booster:
        return lgb.Booster(model_str=self.booster_model_string)

    def _num_iteration(self) -> int | None:
        return self.best_iteration if self.best_iteration > 0 else None

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        raw = np.asarray(self._booster().predict(matrix, num_iteration=self._num_iteration()), dtype="float64")
        if self.metadata.objective is ObjectiveType.REGRESSION:
            return raw
        return (raw >= 0.5).astype("float64")

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        ordered = validate_and_order_features(self.metadata.feature_schema, features, column_policy=column_policy)
        matrix = numeric_feature_matrix(ordered)
        positive = np.asarray(self._booster().predict(matrix, num_iteration=self._num_iteration()), dtype="float64")
        return np.column_stack([1.0 - positive, positive])

    def feature_importance(self, *, importance_type: str = "gain") -> dict[str, float]:
        """Model-NATIVE importance only (`Booster.feature_importance`) --
        never SHAP, never permutation importance (both explicitly out of
        scope for this milestone)."""
        values = self._booster().feature_importance(importance_type=importance_type)
        return dict(zip(self.metadata.feature_schema.feature_names, (float(v) for v in values), strict=True))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_version": _LIGHTGBM_SCHEMA_VERSION, "metadata": self.metadata.to_json_dict(),
            "booster_model_string": self.booster_model_string, "best_iteration": self.best_iteration,
        })

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedLightGBMModel:
        raw = parse_json_strict(data.decode("utf-8"))
        require_schema_version(raw, supported=_LIGHTGBM_SCHEMA_VERSION, context="FittedLightGBMModel")
        try:
            metadata = ModelMetadata.from_json_dict(raw["metadata"])
            _check_expected_metadata(metadata, expected_metadata)
            return cls(metadata=metadata, booster_model_string=str(raw["booster_model_string"]), best_iteration=int(raw["best_iteration"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"FittedLightGBMModel: malformed or incompatible serialized model envelope ({exc})") from exc


@dataclass(frozen=True, slots=True)
class LightGBMModel:
    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedLightGBMModel:
        ordered = validate_and_order_features(self.metadata.feature_schema, features)
        if len(ordered) != len(labels):
            raise ValueError(f"features ({len(ordered)} rows) and labels ({len(labels)} rows) must have matching length")
        matrix = numeric_feature_matrix(ordered)
        y = labels.to_numpy(dtype="float64")
        seed = derive_model_init_seed(seeds)
        lgb_params, num_boost_round, early_stopping_rounds, validation_fraction = _split_hyperparameters(self.metadata.hyperparameters)
        params: dict[str, object] = {
            "objective": _objective_param(self.metadata.objective), "seed": seed, "deterministic": True,
            "force_row_wise": True, "num_threads": 1, "verbose": -1, **lgb_params,
        }

        best_iteration = 0
        if early_stopping_rounds is not None and 0 < validation_fraction < 1 and len(matrix) >= 4:
            split_at = max(1, int(len(matrix) * (1.0 - validation_fraction)))
            split_at = min(split_at, len(matrix) - 1)
            train_ds = lgb.Dataset(matrix[:split_at], label=y[:split_at])
            valid_ds = lgb.Dataset(matrix[split_at:], label=y[split_at:], reference=train_ds)
            booster = lgb.train(
                params, train_ds, num_boost_round=num_boost_round, valid_sets=[valid_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            )
            best_iteration = booster.best_iteration
        else:
            train_ds = lgb.Dataset(matrix, label=y)
            booster = lgb.train(params, train_ds, num_boost_round=num_boost_round)

        return FittedLightGBMModel(metadata=self.metadata, booster_model_string=booster.model_to_string(), best_iteration=best_iteration)


class LightGBMModelFactory:
    @staticmethod
    def capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION, ObjectiveType.REGRESSION),
            supports_predict_proba=True, supports_feature_importance=True, supports_missing_values=True,
            supports_categorical_features=False, is_deterministic=True,
            seed_usage="controls native RNG for row/column subsampling (feature_fraction/bagging_fraction) "
            "and split-gain tie-breaking; has no visible effect when subsampling is left at its 1.0 default",
            required_preprocessing="none -- tree splits are scale-invariant; missing values are handled "
            "natively",
            library_name="lightgbm",
        )

    def create(self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType) -> LightGBMModel:
        metadata = ModelMetadata(
            name=LIGHTGBM_MODEL_NAME, version="1", objective=objective, feature_schema=feature_schema,
            capabilities=self.capabilities(), hyperparameters=hyperparameters,
        )
        return LightGBMModel(metadata=metadata)


class LightGBMModelSerializer:
    def serialize(self, model: FittedLightGBMModel) -> bytes:
        return model.to_bytes()


class LightGBMModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedLightGBMModel:
        return FittedLightGBMModel.from_bytes(data, expected_metadata=expected_metadata)


__all__ = [
    "LIGHTGBM_MODEL_NAME",
    "FittedLightGBMModel",
    "LightGBMModel",
    "LightGBMModelDeserializer",
    "LightGBMModelFactory",
    "LightGBMModelSerializer",
]
