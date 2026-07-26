from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)

from quant_platform.execution.context import FoldExecutionContext
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.tracking import ExperimentEventStore


def _prepared_manifest(tmp_path: Path):
    dataset_manifest, _research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
    return preparer.prepare(spec)


class TestFoldExecutionContext:
    def test_valid_context_builds(self, tmp_path: Path) -> None:
        manifest = _prepared_manifest(tmp_path)
        ctx = FoldExecutionContext(
            experiment_id=manifest.identity.experiment_id, fold_index=0, split_id="fold:0",
            dataset_content_id=manifest.spec.dataset_binding.content_id, manifest=manifest, seed=42,
            environment=capture_environment_snapshot(), artifact_store=MLArtifactStore(tmp_path / "ml"),
            event_store=ExperimentEventStore(tmp_path / "ml"), artifacts_root=tmp_path / "ml",
            started_at=format_utc_timestamp(utc_now()),
        )
        assert ctx.fold_index == 0
        assert ctx.seed == 42

    def test_negative_fold_index_rejected(self, tmp_path: Path) -> None:
        manifest = _prepared_manifest(tmp_path)
        with pytest.raises(ValueError, match="fold_index"):
            FoldExecutionContext(
                experiment_id=manifest.identity.experiment_id, fold_index=-1, split_id="fold:0",
                dataset_content_id="a" * 64, manifest=manifest, seed=1, environment=capture_environment_snapshot(),
                artifact_store=MLArtifactStore(tmp_path / "ml"), event_store=ExperimentEventStore(tmp_path / "ml"),
                artifacts_root=tmp_path / "ml", started_at=format_utc_timestamp(utc_now()),
            )

    def test_empty_split_id_rejected(self, tmp_path: Path) -> None:
        manifest = _prepared_manifest(tmp_path)
        with pytest.raises(ValueError, match="split_id"):
            FoldExecutionContext(
                experiment_id=manifest.identity.experiment_id, fold_index=0, split_id="",
                dataset_content_id="a" * 64, manifest=manifest, seed=1, environment=capture_environment_snapshot(),
                artifact_store=MLArtifactStore(tmp_path / "ml"), event_store=ExperimentEventStore(tmp_path / "ml"),
                artifacts_root=tmp_path / "ml", started_at=format_utc_timestamp(utc_now()),
            )

    def test_negative_seed_rejected(self, tmp_path: Path) -> None:
        manifest = _prepared_manifest(tmp_path)
        with pytest.raises(ValueError, match="seed"):
            FoldExecutionContext(
                experiment_id=manifest.identity.experiment_id, fold_index=0, split_id="fold:0",
                dataset_content_id="a" * 64, manifest=manifest, seed=-1, environment=capture_environment_snapshot(),
                artifact_store=MLArtifactStore(tmp_path / "ml"), event_store=ExperimentEventStore(tmp_path / "ml"),
                artifacts_root=tmp_path / "ml", started_at=format_utc_timestamp(utc_now()),
            )

    def test_non_utc_started_at_rejected(self, tmp_path: Path) -> None:
        manifest = _prepared_manifest(tmp_path)
        with pytest.raises(ValueError, match="not timezone-aware"):
            FoldExecutionContext(
                experiment_id=manifest.identity.experiment_id, fold_index=0, split_id="fold:0",
                dataset_content_id="a" * 64, manifest=manifest, seed=1, environment=capture_environment_snapshot(),
                artifact_store=MLArtifactStore(tmp_path / "ml"), event_store=ExperimentEventStore(tmp_path / "ml"),
                artifacts_root=tmp_path / "ml", started_at="2024-01-01T00:00:00",
            )
