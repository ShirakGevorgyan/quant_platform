"""End-to-end Milestone 4A integration test: real synthetic historical
data -> a real Milestone 3 research dataset (via `ResearchDatasetBuilder`)
-> a real Milestone 4A experiment preparation (via `ExperimentPreparer`),
proving the full stack works together, not just each module in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.exceptions import ArtifactCorruptionError
from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
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


def _build_real_research_dataset(tmp_path: Path):
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(2000, seed=3)
    seed_canonical_dataset(historical_root, df)

    canonical_store = CanonicalStore(historical_root)
    manifest_store = ManifestStore(historical_root)
    historical_loader = DatasetLoader(canonical_store, manifest_store)

    registry = FeatureRegistry()
    register_core_technical_features(
        registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1, 5), momentum_windows=(10,), atr_window=14)
    )

    research_store = ResearchDatasetStore(research_root)
    research_manifest_store = ResearchManifestStore(research_root)
    builder = ResearchDatasetBuilder(
        historical_loader=historical_loader, registry=registry, research_store=research_store,
        manifest_store=research_manifest_store,
    )
    feature_names = tuple(spec.name for spec in registry.list_features())
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1,
        start=start, end=end, feature_names=feature_names,
        label_definition=LabelDefinition(name="fut5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, registry


def _model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            name="constant_test_model", version="1", description="test",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION), supports_predict_proba=True),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        )
    )
    return registry


def _spec_for(dataset_manifest, **overrides: object) -> ExperimentSpec:
    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value,
        source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_binding = FeatureBinding(
        feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    base: dict[str, object] = {
        "dataset_binding": dataset_binding, "feature_binding": feature_binding,
        "label_binding": LabelBinding(name="fut5", kind="future_return", horizon_bars=5, label_type=LabelType.CONTINUOUS),
        "split_binding": SplitBinding(strategy="chronological"),
        "preprocessing_binding": preprocessing_binding,
        "model_name": "constant_test_model", "model_version": "1",
        "hyperparameters": ModelHyperparameters(),
        "objective": ObjectiveType.REGRESSION,
        "seed_configuration": SeedConfiguration(master_seed=1),
        "code_revision_binding": CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        "primary_metric": "rmse",
    }
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


def test_full_pipeline_reaches_ready_against_real_research_dataset(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(dataset_manifest)
    manifest = preparer.prepare(spec)

    assert manifest.status.value == "ready"
    assert manifest.validation_report_reference is not None

    events = preparer.event_store.read_events(manifest.identity.experiment_id)
    assert [e.event_type.value for e in events] == ["experiment_created", "validation_started", "validation_passed"]


def test_reconstruction_from_manifest_yields_equivalent_spec(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(dataset_manifest)
    manifest = preparer.prepare(spec)

    reloaded = preparer.manifest_store.load(manifest.identity.experiment_id)
    assert reloaded.spec == spec
    assert reloaded.spec.to_json_dict() == spec.to_json_dict()


def test_idempotent_repeated_preparation_against_real_dataset(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec = _spec_for(dataset_manifest)
    m1 = preparer.prepare(spec)
    m2 = preparer.prepare(spec)
    assert m1 == m2


def test_changed_hyperparameters_change_experiment_id(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec1 = _spec_for(dataset_manifest)
    spec2 = _spec_for(dataset_manifest, hyperparameters=ModelHyperparameters(values={"alpha": 0.5}))
    m1 = preparer.prepare(spec1)
    m2 = preparer.prepare(spec2)
    assert m1.identity.experiment_id != m2.identity.experiment_id
    assert m1.status.value == "ready"
    assert m2.status.value == "ready"


def test_changed_notes_does_not_change_experiment_id_but_is_not_persisted_twice(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    spec1 = _spec_for(dataset_manifest, notes="original")
    spec2 = _spec_for(dataset_manifest, notes="changed after the fact")
    m1 = preparer.prepare(spec1)
    m2 = preparer.prepare(spec2)
    assert m1.identity.experiment_id == m2.identity.experiment_id
    assert m2.spec.notes == "original"


def test_corrupted_validation_report_artifact_detected_on_read(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    manifest = preparer.prepare(_spec_for(dataset_manifest))
    assert manifest.validation_report_reference is not None

    content_path = preparer.artifact_store._content_path(manifest.validation_report_reference.content_hash)
    content_path.write_bytes(b"TAMPERED BYTES THAT DO NOT MATCH THE HASH")

    with pytest.raises(ArtifactCorruptionError):
        preparer.artifact_store.read_artifact(manifest.validation_report_reference.content_hash)


def test_label_binding_mismatch_against_real_dataset_blocks_ready(tmp_path: Path) -> None:
    """The real `ResearchDatasetBuilder` above builds a dataset whose
    label_definition is `LabelDefinition(name="fut5", kind=FUTURE_RETURN,
    horizon_bars=5)`. A spec that binds to the SAME dataset_id/content_id
    (so `_validate_dataset_binding` passes) but declares a DIFFERENT
    label horizon must still fail -- proving the dataset-identity match
    alone is not enough, exactly the gap this audit closes."""
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    mismatched_spec = _spec_for(
        dataset_manifest,
        label_binding=LabelBinding(name="fut5", kind="future_return", horizon_bars=999, label_type=LabelType.CONTINUOUS),
    )
    manifest = preparer.prepare(mismatched_spec)

    assert manifest.status.value == "failed"
    assert manifest.failure_summary is not None
    assert "label_horizon_mismatch" in manifest.failure_summary
    assert "research_dataset_label_binding_mismatch" in manifest.failure_summary

    report_bytes = preparer.artifact_store.read_artifact(manifest.validation_report_reference.content_hash)  # type: ignore[union-attr]
    import json

    report = json.loads(report_bytes)
    codes = {issue["code"] for issue in report["issues"]}
    assert "label_horizon_mismatch" in codes
    assert "research_dataset_label_binding_mismatch" in codes


def test_label_binding_mismatch_cannot_be_re_prepared_into_ready(tmp_path: Path) -> None:
    dataset_manifest, research_manifest_store, _ = _build_real_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=_model_registry(),
        research_manifest_store=research_manifest_store,
    )
    mismatched_spec = _spec_for(
        dataset_manifest,
        label_binding=LabelBinding(name="wrong_name", kind="future_return", horizon_bars=5, label_type=LabelType.CONTINUOUS),
    )
    m1 = preparer.prepare(mismatched_spec)
    m2 = preparer.prepare(mismatched_spec)  # idempotent: identical (still-mismatched) spec -> no re-attempt
    assert m1 == m2
    assert m2.status.value == "failed"
