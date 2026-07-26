"""Shared fixtures/builders for the Milestone 4B execution engine test
suite. Mirrors `tests/unit/ml/conftest.py`'s conventions exactly, adding
one new builder: `write_synthetic_research_dataset`, which writes a
small, REAL (not mocked) research dataset directly via `ResearchDatasetStore`/
`ResearchManifestStore` -- bypassing `ResearchDatasetBuilder`'s full
historical-loader/feature-engine pipeline so unit tests here stay fast
and focused on the EXECUTION engine, not dataset construction (already
covered by Milestone 3's own test suite). Integration tests
(`tests/integration/test_execution_engine.py`) use the full, real
pipeline end-to-end instead."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
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

FEATURE_NAMES = ("f1", "f2")
LABEL_NAME = "fwd_ret_5"
CODE_REVISION = "d" * 40


def make_timeline(n: int = 1000, *, start: str = "2024-01-01", freq: str = "1min", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "open_time": timestamps,
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "label": rng.normal(size=n),
    })


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


def write_synthetic_research_dataset(
    tmp_path: Path, *, timeline: pd.DataFrame | None = None, dataset_id: str = "synthetic_dataset",
) -> tuple[ResearchDatasetManifest, ResearchDatasetStore, ResearchManifestStore]:
    """Writes ONE split ("train") containing every row of `timeline`
    directly through `ResearchDatasetStore`/`ResearchManifestStore` --
    the real, content-addressed Milestone 3 storage layer, just without
    routing through `ResearchDatasetBuilder`'s full historical/feature
    pipeline first."""
    timeline = timeline if timeline is not None else make_timeline()
    research_store = ResearchDatasetStore(tmp_path / "research")
    research_manifest_store = ResearchManifestStore(tmp_path / "research")

    content_id, output_hashes = research_store.write_artifacts(
        dataset_id, splits={"train": timeline}, preprocessing_json={},
    )
    manifest = ResearchDatasetManifest(
        dataset_id=dataset_id, version="", source_historical_dataset_id="hist1",
        source_historical_manifest_version="000001-abc", symbol="XAUUSD", base_timeframe=Timeframe.M1,
        utc_start=timeline["open_time"].iloc[0], utc_end=timeline["open_time"].iloc[-1],
        feature_names=FEATURE_NAMES, feature_versions={"f1": "1", "f2": "1"},
        feature_registry_fingerprint="b" * 64,
        label_definition={"name": LABEL_NAME, "kind": "future_return", "horizon_bars": 5, "params": {}},
        split_definition={"strategy": "single"}, preprocessing_definition={}, fitted_preprocessing_fingerprint=None,
        code_revision="content:abc", input_content_hashes={"h": "1"}, output_content_hashes=output_hashes,
        row_counts={"train": len(timeline)}, missing_data_summary={}, leakage_validation_result={"is_valid": True},
        created_at=pd.Timestamp.now(tz="UTC"), content_id=content_id,
    )
    version = research_manifest_store.save(manifest)
    return research_manifest_store.load(dataset_id, version), research_store, research_manifest_store


def make_experiment_spec_kwargs(
    *, dataset_manifest: ResearchDatasetManifest, split_strategy: str = "expanding_walk_forward",
    split_params: dict[str, object] | None = None, **overrides: object,
) -> dict[str, object]:
    default_params: dict[str, object] = {"n_splits": 3, "test_size": 100, "purge_bars": 5, "embargo_bars": 2}
    if split_params is not None:
        default_params.update(split_params)
    base: dict[str, object] = {
        "dataset_binding": DatasetBinding(
            dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
            content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
            base_timeframe=dataset_manifest.base_timeframe.value,
        ),
        "feature_binding": FeatureBinding(
            feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
            feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
        ),
        "label_binding": LabelBinding(name=LABEL_NAME, kind="future_return", horizon_bars=5, label_type=LabelType.CONTINUOUS),
        "split_binding": SplitBinding(strategy=split_strategy, params=default_params),  # type: ignore[arg-type]
        "preprocessing_binding": PreprocessingBinding(),
        "model_name": "constant_test_model", "model_version": "1",
        "hyperparameters": ModelHyperparameters(),
        "objective": ObjectiveType.REGRESSION,
        "seed_configuration": SeedConfiguration(master_seed=1),
        "code_revision_binding": CodeRevisionBinding(revision=CODE_REVISION, source="git", is_dirty=False),
        "primary_metric": "rmse",
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[ResearchDatasetManifest, ResearchDatasetStore, ResearchManifestStore]:
    return write_synthetic_research_dataset(tmp_path)
