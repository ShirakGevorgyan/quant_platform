"""Shared, genuinely-reusable helpers for `ml.model_zoo`'s model
implementations -- kept deliberately small. Each model's actual fit/
predict/serialize logic is NOT unified into a base class here: LightGBM/
XGBoost/CatBoost/scikit-learn/the four baselines each have a different
native fitted-state shape (a text booster dump, a versioned binary blob,
a `.cbm` file's bytes, NumPy coefficient arrays, a handful of scalars),
so forcing a shared abstraction across them would cost more clarity than
it would save -- three similar `to_json_dict` methods are better than one
premature shared one that has to special-case every model anyway.

NO PICKLE, ANYWHERE IN THIS PACKAGE
--------------------------------------------------------------------------
Every serializer in `model_zoo` either calls the underlying library's OWN
native serialization (LightGBM/XGBoost/CatBoost) or extracts fitted
NumPy-array parameters into an explicit JSON envelope (scikit-learn
linear models, every baseline) -- `pickle`/`joblib.dump` never appears
anywhere in this package. See `ml.interfaces.ModelSerializer`'s own
docstring for why: a pickle-based serializer would need to be isolated
and documented as unsafe for untrusted input, and no model in this
milestone needs that risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema
from quant_platform.ml.seeds import SeedConfiguration, SeedDomain

ENVELOPE_SCHEMA_VERSION = 1
"""Shared starting schema version for every `model_zoo` serializer's JSON
envelope -- each model's own serializer still declares and checks its
OWN `_SCHEMA_VERSION` independently (never assumed to stay in lock-step
across models); this is only a common STARTING POINT, not a promise they
evolve together."""


def validate_and_order_features(
    feature_schema: FeatureSchema, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT,
) -> pd.DataFrame:
    """The one, shared "give me exactly the declared columns, in the
    declared order" step every model's `fit`/`predict`/`predict_proba`
    starts with -- never a bespoke reimplementation of `FeatureSchema.
    validate_frame` per model."""
    return feature_schema.validate_frame(features, policy=column_policy)


def numeric_feature_matrix(ordered_features: pd.DataFrame) -> np.ndarray:
    """Coerces an already schema-ordered frame to a `float64` NumPy array
    -- for the models in this package with NO native categorical support
    (everything except CatBoost). Deliberately does NOT fill/impute
    missing values -- a model declaring `supports_missing_values=False`
    must never silently receive imputed data it did not ask for; NaNs
    pass through unchanged into whatever the underlying library does
    with them (typically its own error), and `ml.model_validation`'s
    pre-fit gate is what is supposed to catch this BEFORE `fit` is ever
    called."""
    return ordered_features.to_numpy(dtype="float64")


def library_version(module: object) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        raise ValueError(f"Module {module!r} has no __version__ attribute -- cannot record a library version")
    return str(version)


def derive_model_init_seed(seeds: SeedConfiguration) -> int:
    """The ONE seed domain every model in this package derives its
    fit-time randomness from -- reuses `ml.seeds.SeedDomain.MODEL_INIT`
    exactly as `ml.testing.ConstantTestModel` already does, never a new,
    parallel seed domain invented for real models."""
    return seeds.derive(SeedDomain.MODEL_INIT)


def feature_schema_fingerprint(feature_schema: FeatureSchema) -> str:
    """A stable SHA-256 fingerprint of the declared, ORDERED feature
    schema -- what `ml.training_metadata.TrainingMetadata.
    feature_schema_fingerprint` records. Distinct from (and does not
    replace) `ml.models.FeatureBinding.feature_registry_fingerprint`
    (which fingerprints the whole FEATURE REGISTRY's definitions, an
    Milestone-3-level concern); this is a fingerprint of just the ordered
    NAME tuple a fitted model actually consumed."""
    return fingerprint_json(feature_schema.to_json_dict())


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "derive_model_init_seed",
    "feature_schema_fingerprint",
    "library_version",
    "numeric_feature_matrix",
    "validate_and_order_features",
]
