from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.execution.context import FoldExecutionContext
from quant_platform.execution.executor import DeterministicFoldExecutor, FoldData, FoldExecutionOutcome
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.models import ArtifactCategory, ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.testing import (
    ConstantTestModelDeserializer,
    ConstantTestModelFactory,
    ConstantTestModelSerializer,
)
from quant_platform.ml.tracking import ExperimentEventStore

EID = "a" * 64


class TestFoldData:
    def test_valid_data_builds(self) -> None:
        features = pd.DataFrame({"f1": [1.0, 2.0]})
        labels = pd.Series([1.0, 2.0])
        data = FoldData(train_features=features, train_labels=labels, test_features=features, test_labels=labels)
        assert len(data.train_features) == 2

    def test_mismatched_train_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="train_features and train_labels"):
            FoldData(
                train_features=pd.DataFrame({"f1": [1.0, 2.0]}), train_labels=pd.Series([1.0]),
                test_features=pd.DataFrame({"f1": [1.0]}), test_labels=pd.Series([1.0]),
            )

    def test_mismatched_test_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="test_features and test_labels"):
            FoldData(
                train_features=pd.DataFrame({"f1": [1.0]}), train_labels=pd.Series([1.0]),
                test_features=pd.DataFrame({"f1": [1.0, 2.0]}), test_labels=pd.Series([1.0]),
            )

    def test_validation_features_and_labels_must_both_be_set_or_none(self) -> None:
        with pytest.raises(ValueError, match="validation_features and validation_labels"):
            FoldData(
                train_features=pd.DataFrame({"f1": [1.0]}), train_labels=pd.Series([1.0]),
                test_features=pd.DataFrame({"f1": [1.0]}), test_labels=pd.Series([1.0]),
                validation_features=pd.DataFrame({"f1": [1.0]}),
            )


class TestFoldExecutionOutcome:
    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds"):
            FoldExecutionOutcome(artifact_references=(), duration_seconds=-1.0)

    def test_non_primitive_metrics_rejected(self) -> None:
        with pytest.raises(ValueError):
            FoldExecutionOutcome(artifact_references=(), duration_seconds=1.0, metrics={"bad": [1]})  # type: ignore[dict-item]


class TestDeterministicFoldExecutor:
    def _context(self, tmp_path: Path, seed: int = 1) -> FoldExecutionContext:
        from tests.unit.execution.conftest import (
            build_registry,
            make_experiment_spec_kwargs,
            write_synthetic_research_dataset,
        )

        from quant_platform.ml.experiment_manager import ExperimentPreparer
        from quant_platform.ml.experiment_spec import ExperimentSpec

        dataset_manifest, _research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        )
        spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
        manifest = preparer.prepare(spec)
        return FoldExecutionContext(
            experiment_id=manifest.identity.experiment_id, fold_index=0, split_id="fold:0",
            dataset_content_id=dataset_manifest.content_id, manifest=manifest, seed=seed,
            environment=capture_environment_snapshot(), artifact_store=MLArtifactStore(tmp_path / "ml"),
            event_store=ExperimentEventStore(tmp_path / "ml"), artifacts_root=tmp_path / "ml",
            started_at=format_utc_timestamp(utc_now()),
        )

    def test_fit_predict_serialize_pipeline_executes_end_to_end(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        train_features = pd.DataFrame({"f1": np.arange(50.0), "f2": np.arange(50.0)})
        train_labels = pd.Series(np.arange(50.0))
        test_features = pd.DataFrame({"f1": np.arange(10.0), "f2": np.arange(10.0)})
        test_labels = pd.Series(np.arange(10.0))
        data = FoldData(train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)

        executor = DeterministicFoldExecutor()
        outcome = executor.execute(
            context, model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=FeatureSchema(feature_names=("f1", "f2")), objective=ObjectiveType.REGRESSION,
            serializer=ConstantTestModelSerializer(), data=data,
        )
        assert len(outcome.artifact_references) == 2
        categories = {r.category for r in outcome.artifact_references}
        assert categories == {ArtifactCategory.MODEL, ArtifactCategory.PREDICTIONS}
        assert outcome.metrics == {}
        assert outcome.duration_seconds >= 0

        model_ref = next(r for r in outcome.artifact_references if r.category is ArtifactCategory.MODEL)
        raw_model_bytes = context.artifact_store.read_artifact(model_ref.content_hash)
        fitted = ConstantTestModelDeserializer().deserialize(raw_model_bytes)
        assert fitted.constant_value == pytest.approx(float(train_labels.mean()))

    def test_deterministic_across_repeated_calls_with_same_seed(self, tmp_path: Path) -> None:
        """Same seed, same data -> byte-identical model artifact (the
        test model's `.fit()` derives, but does not need, the seed --
        determinism here comes from the label mean being a pure
        function of the data, proving the pipeline introduces no hidden
        nondeterminism of its own, e.g. from dict ordering or wall clock)."""
        context1 = self._context(tmp_path / "a", seed=7)
        context2 = self._context(tmp_path / "b", seed=7)
        features = pd.DataFrame({"f1": np.arange(20.0), "f2": np.arange(20.0)})
        labels = pd.Series(np.arange(20.0))
        data = FoldData(train_features=features, train_labels=labels, test_features=features, test_labels=labels)

        executor = DeterministicFoldExecutor()
        outcome1 = executor.execute(
            context1, model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=FeatureSchema(feature_names=("f1", "f2")), objective=ObjectiveType.REGRESSION,
            serializer=ConstantTestModelSerializer(), data=data,
        )
        outcome2 = executor.execute(
            context2, model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(),
            feature_schema=FeatureSchema(feature_names=("f1", "f2")), objective=ObjectiveType.REGRESSION,
            serializer=ConstantTestModelSerializer(), data=data,
        )
        hashes1 = sorted(r.content_hash for r in outcome1.artifact_references)
        hashes2 = sorted(r.content_hash for r in outcome2.artifact_references)
        assert hashes1 == hashes2
