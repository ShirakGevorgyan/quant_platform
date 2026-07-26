"""Shared fixtures/builders for the Milestone 4A ML infrastructure test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.ml.models import (
    CodeRevisionBinding,
    DatasetBinding,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory

CONTENT_ID = "a" * 64
FEATURE_REGISTRY_FINGERPRINT = "b" * 64
CODE_REVISION = "c" * 40


def make_dataset_manifest(**overrides: object) -> ResearchDatasetManifest:
    base: dict[str, object] = {
        "dataset_id": "xauusd_h1_v1", "version": "000001-aaaaaaaaaaaa", "source_historical_dataset_id": "hist1",
        "source_historical_manifest_version": "000001-abc", "symbol": "XAUUSD", "base_timeframe": Timeframe.H1,
        "utc_start": pd.Timestamp("2024-01-01", tz="UTC"), "utc_end": pd.Timestamp("2024-02-01", tz="UTC"),
        "feature_names": ("atr_14", "rsi_14"), "feature_versions": {"atr_14": "1", "rsi_14": "1"},
        "feature_registry_fingerprint": FEATURE_REGISTRY_FINGERPRINT,
        "label_definition": {"name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": 10, "params": {}},
        "split_definition": {"strategy": "time_ordered_holdout"},
        "preprocessing_definition": {}, "fitted_preprocessing_fingerprint": None, "code_revision": "content:abc",
        "input_content_hashes": {"h": "1"}, "output_content_hashes": {"train": "x"}, "row_counts": {"train": 10},
        "missing_data_summary": {}, "leakage_validation_result": {"is_valid": True},
        "created_at": pd.Timestamp.now(tz="UTC"), "content_id": CONTENT_ID,
    }
    base.update(overrides)
    return ResearchDatasetManifest(**base)  # type: ignore[arg-type]


def make_dataset_binding(**overrides: object) -> DatasetBinding:
    base: dict[str, object] = {
        "dataset_id": "xauusd_h1_v1", "manifest_version": "000001-aaaaaaaaaaaa", "content_id": CONTENT_ID,
        "symbol": "XAUUSD", "base_timeframe": "H1",
    }
    base.update(overrides)
    return DatasetBinding(**base)  # type: ignore[arg-type]


def make_feature_binding(**overrides: object) -> FeatureBinding:
    base: dict[str, object] = {
        "feature_names": ("atr_14", "rsi_14"), "feature_versions": {"atr_14": "1", "rsi_14": "1"},
        "feature_registry_fingerprint": FEATURE_REGISTRY_FINGERPRINT,
    }
    base.update(overrides)
    return FeatureBinding(**base)  # type: ignore[arg-type]


def make_label_binding(**overrides: object) -> LabelBinding:
    base: dict[str, object] = {
        "name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": 10, "label_type": LabelType.CONTINUOUS,
    }
    base.update(overrides)
    return LabelBinding(**base)  # type: ignore[arg-type]


def build_registry(
    *, supported_objectives: tuple[ObjectiveType, ...] = (ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION)
) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            name="constant_test_model", version="1", description="test",
            capabilities=ModelCapabilities(supported_objectives=supported_objectives, supports_predict_proba=True),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        )
    )
    return registry


def make_experiment_spec_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset_binding": make_dataset_binding(),
        "feature_binding": make_feature_binding(),
        "label_binding": make_label_binding(),
        "split_binding": SplitBinding(strategy="time_ordered_holdout"),
        "preprocessing_binding": PreprocessingBinding(),
        "model_name": "constant_test_model", "model_version": "1",
        "hyperparameters": ModelHyperparameters(values={"alpha": 0.1}),
        "objective": ObjectiveType.REGRESSION,
        "seed_configuration": SeedConfiguration(master_seed=42),
        "code_revision_binding": CodeRevisionBinding(revision=CODE_REVISION, source="git", is_dirty=True),
        "primary_metric": "rmse",
    }
    base.update(overrides)
    return base


@pytest.fixture
def dataset_manifest() -> ResearchDatasetManifest:
    return make_dataset_manifest()


@pytest.fixture
def model_registry() -> ModelRegistry:
    return build_registry()
