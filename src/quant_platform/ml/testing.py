"""A minimal, deterministic TEST-ONLY model satisfying `ml.interfaces`'
contracts -- **not a real predictive algorithm**. Always predicts the
training label mean (regression) or positive rate (binary classification)
regardless of input feature values. Exists specifically so this
milestone's tests can exercise fit -> predict -> serialize -> deserialize
without any genuine ML library dependency, mirroring `historical.
mt5_testing.FakeMt5Client`'s established "clearly-named testing double"
convention -- the module name and every docstring say so explicitly.

`ConstantTestModel` (unfit) has no `predict` method at all; `.fit(...)`
returns a SEPARATE `FittedConstantTestModel` object that does -- see
`ml.interfaces`'s module docstring for why this is the design choice that
makes "predict before fit must fail" structural rather than a runtime flag
check. A future real model MAY instead choose a mutable-self design with
an explicit `is_fitted` flag and `core.exceptions.ModelNotFittedError`;
both patterns satisfy the same Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.interfaces import FeatureColumnPolicy, FeatureSchema, ModelMetadata
from quant_platform.ml.models import ModelCapabilities, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict
from quant_platform.ml.seeds import SeedConfiguration, SeedDomain

TEST_MODEL_NAME = "constant_test_model"
TEST_MODEL_VERSION = "1"


@dataclass(frozen=True, slots=True)
class FittedConstantTestModel:
    metadata: ModelMetadata
    constant_value: float
    fitted_class_labels: tuple[object, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self.fitted_class_labels

    def predict(
        self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT
    ) -> np.ndarray:
        validated = self.metadata.feature_schema.validate_frame(features, policy=column_policy)
        return np.full(len(validated), self.constant_value, dtype="float64")

    def predict_proba(
        self, features: pd.DataFrame, *, column_policy: FeatureColumnPolicy = FeatureColumnPolicy.STRICT
    ) -> np.ndarray:
        self.metadata.capabilities.require_predict_proba(self.metadata.objective)
        validated = self.metadata.feature_schema.validate_frame(features, policy=column_policy)
        n = len(validated)
        positive_rate = float(np.clip(self.constant_value, 0.0, 1.0))
        return np.column_stack([np.full(n, 1.0 - positive_rate), np.full(n, positive_rate)])

    def to_bytes(self) -> bytes:
        payload = {
            "schema_version": 1,
            "metadata": self.metadata.to_json_dict(),
            "constant_value": self.constant_value,
            "class_labels": list(self.fitted_class_labels),
        }
        return canonical_json_bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedConstantTestModel:
        raw = parse_json_strict(data.decode("utf-8"))
        metadata = ModelMetadata.from_json_dict(raw["metadata"])
        if expected_metadata is not None and metadata != expected_metadata:
            raise FeatureSchemaMismatchError(
                "Deserialized model metadata does not match expected_metadata",
                context={"expected_name": expected_metadata.name, "found_name": metadata.name},
            )
        return cls(
            metadata=metadata, constant_value=float(raw["constant_value"]),
            fitted_class_labels=tuple(raw["class_labels"]),
        )


@dataclass(frozen=True, slots=True)
class ConstantTestModel:
    """`TrainableModel` -- not yet fit; has no `predict` at all."""

    metadata: ModelMetadata

    def fit(self, features: pd.DataFrame, labels: pd.Series, *, seeds: SeedConfiguration) -> FittedConstantTestModel:
        validated = self.metadata.feature_schema.validate_frame(features, policy=FeatureColumnPolicy.STRICT)
        if len(validated) != len(labels):
            raise ValueError(
                f"features ({len(validated)} rows) and labels ({len(labels)} rows) must have matching length"
            )
        # Deterministic seed usage: a real model would use this to seed
        # weight initialization; this constant model has nothing stochastic
        # to seed, but still derives (and thereby proves usable) the
        # MODEL_INIT domain seed rather than silently ignoring `seeds`.
        _ = seeds.derive(SeedDomain.MODEL_INIT)

        constant_value = float(labels.mean())
        class_labels: tuple[object, ...] = (
            () if self.metadata.objective is ObjectiveType.REGRESSION else (0, 1)
        )
        return FittedConstantTestModel(metadata=self.metadata, constant_value=constant_value, fitted_class_labels=class_labels)


class ConstantTestModelFactory:
    """`ModelFactory` for `ConstantTestModel` -- test/demo use only, never
    registered under a name suggesting real predictive capability."""

    def create(
        self, *, hyperparameters: ModelHyperparameters, feature_schema: FeatureSchema, objective: ObjectiveType
    ) -> ConstantTestModel:
        capabilities = ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
            supports_predict_proba=True,
        )
        metadata = ModelMetadata(
            name=TEST_MODEL_NAME, version=TEST_MODEL_VERSION, objective=objective, feature_schema=feature_schema,
            capabilities=capabilities, hyperparameters=hyperparameters,
        )
        return ConstantTestModel(metadata=metadata)


class ConstantTestModelSerializer:
    """`ModelSerializer` -- deterministic JSON, no pickle, safe for
    untrusted input (unlike a hypothetical future pickle-based serializer,
    which would need to be isolated and documented as unsafe)."""

    def serialize(self, model: FittedConstantTestModel) -> bytes:
        return model.to_bytes()


class ConstantTestModelDeserializer:
    def deserialize(self, data: bytes, *, expected_metadata: ModelMetadata | None = None) -> FittedConstantTestModel:
        return FittedConstantTestModel.from_bytes(data, expected_metadata=expected_metadata)


__all__ = [
    "TEST_MODEL_NAME",
    "TEST_MODEL_VERSION",
    "ConstantTestModel",
    "ConstantTestModelDeserializer",
    "ConstantTestModelFactory",
    "ConstantTestModelSerializer",
    "FittedConstantTestModel",
]
