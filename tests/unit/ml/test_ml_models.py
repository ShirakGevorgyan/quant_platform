from __future__ import annotations

import pytest
from tests.unit.ml.conftest import CONTENT_ID, FEATURE_REGISTRY_FINGERPRINT, make_dataset_manifest

from quant_platform.core.exceptions import UnsupportedObjectiveError
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    CodeRevisionBinding,
    DatasetBinding,
    EnvironmentSnapshot,
    ExperimentStatus,
    FeatureBinding,
    LabelBinding,
    LabelType,
    MetricsArtifactMetadata,
    ModelArtifactMetadata,
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PredictionArtifactMetadata,
    PreprocessingBinding,
    SplitBinding,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    is_legal_transition,
    objective_supports_label_type,
    validate_artifact_hash,
    validate_json_primitive_mapping,
)


class TestValidateJsonPrimitiveMapping:
    def test_accepts_primitives(self) -> None:
        validate_json_primitive_mapping({"a": 1, "b": "x", "c": 1.5, "d": True, "e": None}, field_name="f")

    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_json_primitive_mapping({"": 1}, field_name="f")

    def test_rejects_non_finite_float(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            validate_json_primitive_mapping({"a": float("nan")}, field_name="f")
        with pytest.raises(ValueError, match="finite"):
            validate_json_primitive_mapping({"a": float("inf")}, field_name="f")

    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="unsupported type"):
            validate_json_primitive_mapping({"a": [1, 2]}, field_name="f")
        with pytest.raises(ValueError, match="unsupported type"):
            validate_json_primitive_mapping({"a": {"nested": 1}}, field_name="f")


class TestObjectiveLabelCompatibility:
    @pytest.mark.parametrize(
        ("objective", "label_type", "expected"),
        [
            (ObjectiveType.BINARY_CLASSIFICATION, LabelType.BINARY, True),
            (ObjectiveType.BINARY_CLASSIFICATION, LabelType.CONTINUOUS, False),
            (ObjectiveType.MULTICLASS_CLASSIFICATION, LabelType.MULTICLASS, True),
            (ObjectiveType.REGRESSION, LabelType.CONTINUOUS, True),
            (ObjectiveType.REGRESSION, LabelType.BINARY, False),
        ],
    )
    def test_matrix(self, objective: ObjectiveType, label_type: LabelType, expected: bool) -> None:
        assert objective_supports_label_type(objective, label_type) is expected


class TestLegalTransitions:
    @pytest.mark.parametrize(
        ("current", "target", "legal"),
        [
            (ExperimentStatus.CREATED, ExperimentStatus.VALIDATING, True),
            (ExperimentStatus.CREATED, ExperimentStatus.RUNNING, False),
            (ExperimentStatus.VALIDATING, ExperimentStatus.READY, True),
            (ExperimentStatus.VALIDATING, ExperimentStatus.FAILED, True),
            (ExperimentStatus.READY, ExperimentStatus.RUNNING, True),
            (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED, True),
            (ExperimentStatus.RUNNING, ExperimentStatus.FAILED, True),
            (ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING, False),
            (ExperimentStatus.FAILED, ExperimentStatus.READY, False),
            (ExperimentStatus.CANCELLED, ExperimentStatus.CREATED, False),
        ],
    )
    def test_matrix(self, current: ExperimentStatus, target: ExperimentStatus, legal: bool) -> None:
        assert is_legal_transition(current, target) is legal

    def test_every_terminal_status_has_no_legal_transitions(self) -> None:
        for terminal in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
            for target in ExperimentStatus:
                assert not is_legal_transition(terminal, target)


class TestModelCapabilities:
    def test_round_trip(self) -> None:
        cap = ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
            supports_predict_proba=True, supports_feature_importance=True,
        )
        assert ModelCapabilities.from_json_dict(cap.to_json_dict()) == cap

    def test_empty_objectives_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ModelCapabilities(supported_objectives=())

    def test_duplicate_objectives_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.REGRESSION))

    def test_regression_only_cannot_support_predict_proba(self) -> None:
        with pytest.raises(ValueError, match="regression-only"):
            ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), supports_predict_proba=True)

    def test_require_predict_proba_raises_for_regression(self) -> None:
        cap = ModelCapabilities(
            supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION), supports_predict_proba=True
        )
        with pytest.raises(UnsupportedObjectiveError):
            cap.require_predict_proba(ObjectiveType.REGRESSION)
        cap.require_predict_proba(ObjectiveType.BINARY_CLASSIFICATION)  # must not raise

    def test_require_predict_proba_raises_when_unsupported(self) -> None:
        cap = ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,), supports_predict_proba=False)
        with pytest.raises(UnsupportedObjectiveError):
            cap.require_predict_proba(ObjectiveType.REGRESSION)


class TestModelHyperparameters:
    def test_round_trip(self) -> None:
        hp = ModelHyperparameters(values={"alpha": 0.1, "count": 3, "flag": True, "name": "x", "none": None})
        assert ModelHyperparameters.from_json_dict(hp.to_json_dict()) == hp

    def test_rejects_non_primitive_value(self) -> None:
        with pytest.raises(ValueError):
            ModelHyperparameters(values={"a": [1, 2]})  # type: ignore[dict-item]

    def test_dict_insertion_order_does_not_affect_equality_or_json(self) -> None:
        hp1 = ModelHyperparameters(values={"a": 1, "b": 2})
        hp2 = ModelHyperparameters(values={"b": 2, "a": 1})
        assert hp1.to_json_dict() == hp2.to_json_dict()


class TestDatasetBinding:
    def test_round_trip(self) -> None:
        binding = DatasetBinding(
            dataset_id="d", manifest_version="1", content_id=CONTENT_ID, symbol="XAUUSD", base_timeframe="H1",
            source_historical_dataset_id="hist1",
        )
        assert DatasetBinding.from_json_dict(binding.to_json_dict()) == binding

    @pytest.mark.parametrize("field_name", ["dataset_id", "manifest_version", "content_id", "symbol", "base_timeframe"])
    def test_empty_fields_rejected(self, field_name: str) -> None:
        kwargs = {
            "dataset_id": "d", "manifest_version": "1", "content_id": CONTENT_ID, "symbol": "XAUUSD",
            "base_timeframe": "H1",
        }
        kwargs[field_name] = ""
        with pytest.raises(ValueError, match="must not be empty"):
            DatasetBinding(**kwargs)  # type: ignore[arg-type]

    def test_malformed_content_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            DatasetBinding(dataset_id="d", manifest_version="1", content_id="not-a-hash", symbol="XAUUSD", base_timeframe="H1")


class TestFeatureBinding:
    def test_round_trip_preserves_order(self) -> None:
        binding = FeatureBinding(
            feature_names=("z", "a", "m"), feature_versions={"z": "1", "a": "1", "m": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        )
        restored = FeatureBinding.from_json_dict(binding.to_json_dict())
        assert restored.feature_names == ("z", "a", "m")
        assert restored == binding

    def test_empty_feature_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            FeatureBinding(feature_names=(), feature_versions={}, feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT)

    def test_duplicate_feature_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            FeatureBinding(
                feature_names=("a", "a"), feature_versions={"a": "1"}, feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT
            )

    def test_missing_version_for_feature_rejected(self) -> None:
        with pytest.raises(ValueError, match="no version recorded"):
            FeatureBinding(feature_names=("a", "b"), feature_versions={"a": "1"}, feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT)

    def test_malformed_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            FeatureBinding(feature_names=("a",), feature_versions={"a": "1"}, feature_registry_fingerprint="short")


class TestLabelBinding:
    def test_round_trip(self) -> None:
        binding = LabelBinding(name="n", kind="k", horizon_bars=5, label_type=LabelType.BINARY, params={"x": 1.0})
        assert LabelBinding.from_json_dict(binding.to_json_dict()) == binding

    def test_non_positive_horizon_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LabelBinding(name="n", kind="k", horizon_bars=0, label_type=LabelType.CONTINUOUS)

    def test_require_compatible_raises_on_mismatch(self) -> None:
        binding = LabelBinding(name="n", kind="k", horizon_bars=1, label_type=LabelType.BINARY)
        with pytest.raises(UnsupportedObjectiveError):
            binding.require_compatible(ObjectiveType.REGRESSION)
        binding.require_compatible(ObjectiveType.BINARY_CLASSIFICATION)  # must not raise


class TestSplitBinding:
    def test_round_trip(self) -> None:
        binding = SplitBinding(strategy="s", params={"k": 1.0})
        assert SplitBinding.from_json_dict(binding.to_json_dict()) == binding

    def test_empty_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SplitBinding(strategy="")


class TestPreprocessingBinding:
    def test_round_trip(self) -> None:
        binding = PreprocessingBinding(preprocessing_definition={"a": "b"}, fitted_preprocessing_fingerprint=CONTENT_ID)
        assert PreprocessingBinding.from_json_dict(binding.to_json_dict()) == binding

    def test_none_fingerprint_allowed(self) -> None:
        PreprocessingBinding()

    def test_malformed_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            PreprocessingBinding(fitted_preprocessing_fingerprint="nope")


class TestCodeRevisionBinding:
    def test_round_trip_git(self) -> None:
        binding = CodeRevisionBinding(revision="c" * 40, source="git", is_dirty=True)
        assert CodeRevisionBinding.from_json_dict(binding.to_json_dict()) == binding

    def test_round_trip_content(self) -> None:
        binding = CodeRevisionBinding(revision="content:" + "a" * 64, source="content", is_dirty=None)
        assert CodeRevisionBinding.from_json_dict(binding.to_json_dict()) == binding

    def test_empty_revision_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CodeRevisionBinding(revision="", source="git")

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"git.*content"):
            CodeRevisionBinding(revision="x", source="svn")  # type: ignore[arg-type]

    def test_content_source_with_is_dirty_rejected(self) -> None:
        with pytest.raises(ValueError, match="is_dirty must be None"):
            CodeRevisionBinding(revision="x", source="content", is_dirty=True)


class TestArtifactReference:
    def test_round_trip(self) -> None:
        ref = ArtifactReference(category=ArtifactCategory.MODEL, content_hash=CONTENT_ID, size_bytes=10, created_at="2024-01-01T00:00:00+00:00")
        assert ArtifactReference.from_json_dict(ref.to_json_dict()) == ref

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            ArtifactReference(category=ArtifactCategory.MODEL, content_hash=CONTENT_ID, size_bytes=-1, created_at="2024-01-01T00:00:00+00:00")

    def test_malformed_hash_rejected(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            ArtifactReference(category=ArtifactCategory.MODEL, content_hash="bad", size_bytes=1, created_at="2024-01-01T00:00:00+00:00")


class TestValidateArtifactHash:
    def test_valid_hash_ok(self) -> None:
        validate_artifact_hash(CONTENT_ID)

    def test_invalid_hash_raises_artifact_corruption_error(self) -> None:
        from quant_platform.core.exceptions import ArtifactCorruptionError

        with pytest.raises(ArtifactCorruptionError):
            validate_artifact_hash("not-a-hash")


class TestModelArtifactMetadata:
    def test_round_trip(self) -> None:
        meta = ModelArtifactMetadata(model_name="m", model_version="1", objective=ObjectiveType.REGRESSION, feature_names=("a", "b"))
        assert ModelArtifactMetadata.from_json_dict(meta.to_json_dict()) == meta


class TestPredictionArtifactMetadata:
    def test_round_trip(self) -> None:
        meta = PredictionArtifactMetadata(row_count=5, split_name="train")
        assert PredictionArtifactMetadata.from_json_dict(meta.to_json_dict()) == meta

    def test_negative_row_count_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            PredictionArtifactMetadata(row_count=-1, split_name="train")

    def test_empty_split_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PredictionArtifactMetadata(row_count=1, split_name="")


class TestMetricsArtifactMetadata:
    def test_round_trip(self) -> None:
        meta = MetricsArtifactMetadata(split_name="train", metric_names=("rmse", "mae"))
        assert MetricsArtifactMetadata.from_json_dict(meta.to_json_dict()) == meta

    def test_empty_metric_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            MetricsArtifactMetadata(split_name="train", metric_names=())


class TestValidationReport:
    def test_round_trip(self) -> None:
        issue = ValidationIssue(severity=ValidationSeverity.WARNING, code="c", message="m", context={"k": 1})
        report = ValidationReport(schema_version=1, issues=(issue,), generated_at="2024-01-01T00:00:00+00:00")
        assert ValidationReport.from_json_dict(report.to_json_dict()) == report

    def test_is_ready_true_with_only_warnings_and_infos(self) -> None:
        issues = (
            ValidationIssue(severity=ValidationSeverity.WARNING, code="w", message="m"),
            ValidationIssue(severity=ValidationSeverity.INFO, code="i", message="m"),
        )
        report = ValidationReport(schema_version=1, issues=issues, generated_at="2024-01-01T00:00:00+00:00")
        assert report.is_ready

    @pytest.mark.parametrize("severity", [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    def test_is_ready_false_with_error_or_critical(self, severity: ValidationSeverity) -> None:
        report = ValidationReport(
            schema_version=1, issues=(ValidationIssue(severity=severity, code="c", message="m"),),
            generated_at="2024-01-01T00:00:00+00:00",
        )
        assert not report.is_ready

    def test_severity_filters(self) -> None:
        issues = (
            ValidationIssue(severity=ValidationSeverity.CRITICAL, code="c1", message="m"),
            ValidationIssue(severity=ValidationSeverity.ERROR, code="c2", message="m"),
            ValidationIssue(severity=ValidationSeverity.WARNING, code="c3", message="m"),
            ValidationIssue(severity=ValidationSeverity.INFO, code="c4", message="m"),
        )
        report = ValidationReport(schema_version=1, issues=issues, generated_at="2024-01-01T00:00:00+00:00")
        assert len(report.criticals) == 1
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert len(report.infos) == 1


class TestEnvironmentSnapshotSchemaVersion:
    def test_round_trip(self) -> None:
        snap = EnvironmentSnapshot(
            schema_version=1, python_version="3.12.0", platform_system="Linux", platform_release="1",
            architecture="x86_64", package_versions={"numpy": "1.0", "missing": None}, cpu_count=8,
            captured_at="2024-01-01T00:00:00+00:00",
        )
        assert EnvironmentSnapshot.from_json_dict(snap.to_json_dict()) == snap


def test_research_dataset_manifest_fixture_builds_ok() -> None:
    manifest = make_dataset_manifest()
    assert manifest.dataset_id == "xauusd_h1_v1"
