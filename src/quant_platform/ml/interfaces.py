"""Model contracts for future classification/regression models --
**no real algorithm lives here**. `LightGBM`/`XGBoost`/neural networks/etc.
are explicitly out of scope for Milestone 4A; this module defines the
SHAPE any future model must satisfy, validated end-to-end in this
milestone only by a minimal deterministic test-only model
(`ml.testing.ConstantTestModel`) that exists purely to exercise
orchestration and persistence, never to predict anything meaningful.

THE CORE DESIGN CHOICE: FIT RETURNS A NEW OBJECT
--------------------------------------------------------------------------
`TrainableModel.fit(...)` returns a NEW `FittedModel` instance rather than
mutating `self` in place. This is what makes "predict before fit must
fail" a STRUCTURAL property, not just a runtime flag check: a
`TrainableModel` simply has no `predict` method at all -- calling it is an
`AttributeError` at the object's actual shape, not a guarded runtime state
check that a careless implementation could forget. (A future model
implementation MAY still additionally track fitted-state internally and
raise `ModelNotFittedError` defensively -- both patterns satisfy these
Protocols; the test-only model in `ml.testing` uses the structural
"different object entirely" pattern.)

OTHER INVALID OPERATIONS THIS MODULE MAKES EXPLICIT
--------------------------------------------------------------------------
- `predict_proba` on a model whose `ModelCapabilities.supports_predict_proba`
  is `False` (e.g. a regression-only model) must raise
  `UnsupportedObjectiveError` -- `ModelCapabilities.require_predict_proba`
  is the shared guard every implementation should call first.
- Classification probability output must define class ordering explicitly:
  `ProbabilisticPredictor.class_labels` is the authoritative, ordered
  tuple describing what each column of `predict_proba`'s output means --
  never an implicit "alphabetical" or "insertion order" assumption.
- Input feature order/completeness is validated by `FeatureSchema.
  validate_frame`, under an explicit `FeatureColumnPolicy` (STRICT rejects
  both missing AND extra columns; IGNORE_EXTRA rejects only missing
  columns and silently drops anything else) -- never a silent reorder or
  a silent subset selection with no policy named at all.
- A deserialized model must retain its original `ModelMetadata`
  (`ModelDeserializer.deserialize`'s optional `expected_metadata` lets a
  caller assert this explicitly, raising `FeatureSchemaMismatchError` on
  mismatch) -- see `ml.testing`'s serializer for a concrete, safe (no
  pickle) example.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import as_json_list
from quant_platform.ml.seeds import SeedConfiguration


class FeatureColumnPolicy(Enum):
    STRICT = "strict"
    """Both missing AND extra columns are rejected."""
    IGNORE_EXTRA = "ignore_extra"
    """Missing columns are still rejected; extra columns are accepted and
    silently dropped (never silently reordered or coerced)."""


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """The ordered feature-column contract a model expects. Order is
    semantically meaningful (mirrors `features.models.FeatureBinding`) --
    `validate_frame` always returns columns in EXACTLY this order,
    regardless of the input frame's own column order."""

    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("FeatureSchema.feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureSchema.feature_names must not contain duplicates")

    def validate_frame(self, frame: pd.DataFrame, *, policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> pd.DataFrame:
        columns = set(frame.columns)
        required = set(self.feature_names)
        missing = sorted(required - columns)
        extra = sorted(columns - required)
        if missing:
            raise FeatureSchemaMismatchError(
                f"Missing required feature column(s): {missing}", context={"missing": missing}
            )
        if extra and policy is FeatureColumnPolicy.STRICT:
            raise FeatureSchemaMismatchError(
                f"Unexpected extra feature column(s): {extra} (policy=STRICT)", context={"extra": extra}
            )
        return frame[list(self.feature_names)]

    def to_json_dict(self) -> dict[str, object]:
        return {"feature_names": list(self.feature_names)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureSchema:
        return cls(feature_names=tuple(str(n) for n in as_json_list(raw["feature_names"], field_name="feature_names")))


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    name: str
    version: str
    objective: ObjectiveType
    feature_schema: FeatureSchema
    capabilities: ModelCapabilities
    hyperparameters: ModelHyperparameters

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelMetadata.name must not be empty")
        if not self.version:
            raise ValueError("ModelMetadata.version must not be empty")
        if not self.capabilities.supports(self.objective):
            raise ValueError(
                f"ModelMetadata: objective {self.objective.value!r} is not in capabilities."
                f"supported_objectives {[o.value for o in self.capabilities.supported_objectives]}"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "version": self.version, "objective": self.objective.value,
            "feature_schema": self.feature_schema.to_json_dict(),
            "capabilities": self.capabilities.to_json_dict(),
            "hyperparameters": self.hyperparameters.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ModelMetadata:
        return cls(
            name=str(raw["name"]), version=str(raw["version"]), objective=ObjectiveType(raw["objective"]),
            feature_schema=FeatureSchema.from_json_dict(raw["feature_schema"]),  # type: ignore[arg-type]
            capabilities=ModelCapabilities.from_json_dict(raw["capabilities"]),  # type: ignore[arg-type]
            hyperparameters=ModelHyperparameters.from_json_dict(raw["hyperparameters"]),  # type: ignore[arg-type]
        )


@runtime_checkable
class Predictor(Protocol):
    """The minimal "can produce a point prediction" contract."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def predict(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray: ...


@runtime_checkable
class ProbabilisticPredictor(Predictor, Protocol):
    """Only implemented by models whose `ModelCapabilities.
    supports_predict_proba` is `True`. `class_labels` makes probability
    COLUMN ordering explicit -- callers must never assume alphabetical or
    insertion order."""

    @property
    def class_labels(self) -> tuple[object, ...]: ...

    def predict_proba(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray: ...


@runtime_checkable
class DecisionFunctionPredictor(Predictor, Protocol):
    """Milestone 4C: an additional, OPTIONAL capability -- only a model
    with a natural raw margin/score BEFORE any probability transform
    (e.g. `LogisticRegression`'s linear decision boundary distance)
    implements this. Additive to the Protocol surface: it does not
    replace or narrow `Predictor`/`ProbabilisticPredictor`, and most
    models (tree ensembles, every baseline) simply do not implement it --
    there is no "not applicable" flag to set anywhere; Python's
    structural typing means a class either has the method or it doesn't."""

    def decision_function(self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT) -> np.ndarray: ...


@runtime_checkable
class FittedModel(Predictor, Protocol):
    """A model that has completed `fit`. Structurally distinct from
    `TrainableModel` -- see module docstring."""

    @property
    def is_fitted(self) -> bool: ...


@runtime_checkable
class TrainableModel(Protocol):
    """A model configured (hyperparameters, feature schema, objective) but
    NOT yet fit -- has no `predict`/`predict_proba` at all."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedModel: ...


@runtime_checkable
class ModelFactory(Protocol):
    """Constructs a fresh, unfit `TrainableModel` from declared
    configuration -- what `ModelRegistry` stores per `(name, version)`,
    never a fitted instance."""

    def create(
        self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType
    ) -> TrainableModel: ...


@runtime_checkable
class ModelSerializer(Protocol):
    """Serializes a `FittedModel` to bytes. **A serializer that uses pickle
    or any other code-executing format must document itself as unsafe for
    untrusted input** -- see `ml/artifacts.py`'s module docstring for the
    trust-boundary this protocol sits behind (the generic artifact layer
    never deserializes a model itself; only an explicitly-trusted
    `ModelDeserializer` may)."""

    def serialize(self, model: FittedModel) -> bytes: ...


@runtime_checkable
class ModelDeserializer(Protocol):
    """Reconstructs a `FittedModel` from bytes produced by a matching
    `ModelSerializer`. `expected_metadata`, when given, must be compared
    against the reconstructed model's own `metadata` and raise
    `FeatureSchemaMismatchError` on any mismatch -- deserialization must
    never silently return a model with a different feature schema than
    what the caller believes it is loading."""

    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedModel: ...


__all__ = [
    "DecisionFunctionPredictor",
    "FeatureColumnPolicy",
    "FeatureSchema",
    "FittedModel",
    "ModelDeserializer",
    "ModelFactory",
    "ModelMetadata",
    "ModelSerializer",
    "Predictor",
    "ProbabilisticPredictor",
    "TrainableModel",
]
