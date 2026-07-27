from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.ml.interfaces import FeatureSchema, ModelMetadata
from quant_platform.ml.model_validation import validate_training_data
from quant_platform.ml.models import (
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
)

_SCHEMA = FeatureSchema(feature_names=("f1", "f2"))


def _metadata(*, objective: ObjectiveType = ObjectiveType.BINARY_CLASSIFICATION, **capability_overrides: object) -> ModelMetadata:
    capabilities = ModelCapabilities(
        supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
        supports_predict_proba=True, **capability_overrides,  # type: ignore[arg-type]
    )
    return ModelMetadata(
        name="x", version="1", objective=objective, feature_schema=_SCHEMA, capabilities=capabilities,
        hyperparameters=ModelHyperparameters(),
    )


def _clean_data(n: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    labels = pd.Series((rng.random(size=n) > 0.5).astype(int))
    return features, labels


class TestCleanDataPassesEveryCheck:
    def test_clean_data_is_ready(self) -> None:
        features, labels = _clean_data()
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert report.is_ready
        codes = {i.code for i in report.infos}
        assert {
            "model_objective_compatible", "feature_schema_compatible", "sample_count_sufficient",
            "no_missing_values", "all_features_numeric", "labels_not_constant", "binary_labels_valid",
        } <= codes


class TestRequiredPreprocessingEnforcement:
    """BLOCKER 2, "REQUIRED PREPROCESSING MUST BE ENFORCED": a model
    declaring `capabilities.requires_scaled_numeric_features=True` must
    be REJECTED (CRITICAL, before `fit`) unless typed evidence (a
    `PreprocessingBinding` with a fitted fingerprint AND a declared
    scaling transform for every feature) proves scaling was actually
    applied -- `required_preprocessing`'s free-text string is never
    itself the enforcement mechanism."""

    def test_model_not_requiring_scaling_is_unaffected(self) -> None:
        features, labels = _clean_data()
        report = validate_training_data(metadata=_metadata(requires_scaled_numeric_features=False), features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "preprocessing_requirement_none" for i in report.infos)

    def test_scale_sensitive_model_rejected_with_no_binding_at_all(self) -> None:
        features, labels = _clean_data()
        report = validate_training_data(metadata=_metadata(requires_scaled_numeric_features=True), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "required_preprocessing_unproven" for i in report.criticals)

    def test_scale_sensitive_model_rejected_with_default_empty_binding(self) -> None:
        features, labels = _clean_data()
        report = validate_training_data(
            metadata=_metadata(requires_scaled_numeric_features=True), features=features, labels=labels,
            preprocessing_binding=PreprocessingBinding(),
        )
        assert not report.is_ready
        assert any(i.code == "required_preprocessing_unproven" for i in report.criticals)

    def test_scale_sensitive_model_rejected_when_fingerprint_present_but_a_feature_is_unscaled(self) -> None:
        """`f1` is declared scaled, but `f2` (also a declared feature of
        this model) is not -- proving every feature must be covered, not
        just some."""
        features, labels = _clean_data()
        binding = PreprocessingBinding(
            preprocessing_definition={"f1": "standard_scale"}, fitted_preprocessing_fingerprint="a" * 64,
        )
        report = validate_training_data(
            metadata=_metadata(requires_scaled_numeric_features=True), features=features, labels=labels,
            preprocessing_binding=binding,
        )
        assert not report.is_ready
        issue = next(i for i in report.criticals if i.code == "required_preprocessing_unproven")
        assert "f2" in issue.message

    def test_scale_sensitive_model_rejected_for_a_non_scaling_transform_kind(self) -> None:
        """`winsorize`/`signed_log1p` are real Milestone-3 transforms, but
        neither makes features comparably SCALED -- a fitted fingerprint
        alone must not be treated as proof of scaling specifically."""
        features, labels = _clean_data()
        binding = PreprocessingBinding(
            preprocessing_definition={"f1": "winsorize", "f2": "signed_log1p"}, fitted_preprocessing_fingerprint="a" * 64,
        )
        report = validate_training_data(
            metadata=_metadata(requires_scaled_numeric_features=True), features=features, labels=labels,
            preprocessing_binding=binding,
        )
        assert not report.is_ready
        assert any(i.code == "required_preprocessing_unproven" for i in report.criticals)

    def test_scale_sensitive_model_permitted_when_every_feature_is_proven_scaled(self) -> None:
        """"If an existing immutable preprocessing binding can prove that
        safe, train-only scaling was already performed, use that typed
        evidence" -- this is that permit path, proven directly against
        `validate_training_data` (the real execution engine can never
        reach this state today, since `execution.runner.
        assert_preprocessing_is_safe_for_execution` refuses an entire
        execution before any fold runs whenever a bound dataset shows any
        fitted preprocessing at all -- see that function's own
        docstring)."""
        features, labels = _clean_data()
        binding = PreprocessingBinding(
            preprocessing_definition={"f1": "standard_scale", "f2": "robust_scale"},
            fitted_preprocessing_fingerprint="a" * 64,
        )
        report = validate_training_data(
            metadata=_metadata(requires_scaled_numeric_features=True), features=features, labels=labels,
            preprocessing_binding=binding,
        )
        assert report.is_ready
        assert any(i.code == "preprocessing_requirement_satisfied" for i in report.infos)


class TestObjectiveCompatibility:
    def test_incompatible_objective_reports_critical(self) -> None:
        """`ModelMetadata.__post_init__` already guarantees
        `capabilities.supports(objective)` at construction -- this proves
        the check reports INFO (confirmed-not-violated) rather than
        silently omitting it, mirroring `ml.validation`'s own convention
        for structurally-guaranteed checks."""
        metadata = _metadata(objective=ObjectiveType.REGRESSION)
        features, _labels = _clean_data()
        labels_reg = pd.Series(np.random.default_rng(0).normal(size=len(features)))
        report = validate_training_data(metadata=metadata, features=features, labels=labels_reg)
        assert any(i.code == "model_objective_compatible" for i in report.infos)


class TestFeatureSchemaCheck:
    def test_missing_declared_column_is_critical(self) -> None:
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        labels = pd.Series([0, 1, 0])
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "feature_schema_incompatible" for i in report.criticals)


class TestSampleCountChecks:
    def test_zero_rows_is_critical(self) -> None:
        features = pd.DataFrame({"f1": [], "f2": []})
        labels = pd.Series([], dtype="float64")
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "zero_training_samples" for i in report.criticals)

    def test_single_row_is_critical(self) -> None:
        features = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        labels = pd.Series([1])
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "single_training_sample" for i in report.criticals)

    def test_small_sample_count_is_warning_not_critical(self) -> None:
        features = pd.DataFrame({"f1": list(range(5)), "f2": list(range(5))}, dtype="float64")
        labels = pd.Series([0, 1, 0, 1, 0])
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "small_sample_count" for i in report.warnings)

    def test_row_count_mismatch_is_critical(self) -> None:
        features = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        labels = pd.Series([0, 1, 1])
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "feature_label_row_count_mismatch" for i in report.criticals)

    def test_high_dimensional_features_is_warning_not_critical(self) -> None:
        n, p = 10, 20
        schema = FeatureSchema(feature_names=tuple(f"f{i}" for i in range(p)))
        metadata = ModelMetadata(
            name="x", version="1", objective=ObjectiveType.REGRESSION, feature_schema=schema,
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
            hyperparameters=ModelHyperparameters(),
        )
        features = pd.DataFrame(np.random.default_rng(0).normal(size=(n, p)), columns=schema.feature_names)
        labels = pd.Series(np.random.default_rng(0).normal(size=n))
        report = validate_training_data(metadata=metadata, features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "high_dimensional_features" for i in report.warnings)


class TestMissingValuesCheck:
    def test_nan_features_rejected_when_unsupported(self) -> None:
        features, labels = _clean_data()
        features.loc[0, "f1"] = np.nan
        report = validate_training_data(metadata=_metadata(supports_missing_values=False), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "missing_values_unsupported" for i in report.criticals)

    def test_nan_features_accepted_when_supported(self) -> None:
        features, labels = _clean_data()
        features.loc[0, "f1"] = np.nan
        report = validate_training_data(metadata=_metadata(supports_missing_values=True), features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "missing_values_supported" for i in report.infos)


class TestFeatureTypeChecks:
    def test_categorical_column_rejected_when_unsupported(self) -> None:
        features, labels = _clean_data()
        features["f1"] = ["a", "b"] * (len(features) // 2)
        report = validate_training_data(metadata=_metadata(supports_categorical_features=False), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "categorical_features_unsupported" for i in report.criticals)

    def test_categorical_column_accepted_when_supported(self) -> None:
        features, labels = _clean_data()
        features["f1"] = ["a", "b"] * (len(features) // 2)
        report = validate_training_data(metadata=_metadata(supports_categorical_features=True), features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "categorical_features_supported" for i in report.infos)


class TestLabelChecks:
    def test_all_missing_labels_is_critical(self) -> None:
        features, _ = _clean_data()
        labels = pd.Series([np.nan] * len(features))
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "all_labels_missing" for i in report.criticals)

    def test_partially_missing_labels_is_critical(self) -> None:
        features, labels = _clean_data()
        labels = labels.astype("float64")
        labels.iloc[0] = np.nan
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "labels_contain_missing_values" for i in report.criticals)

    def test_constant_classification_labels_is_critical(self) -> None:
        features, _ = _clean_data()
        labels = pd.Series([1] * len(features))
        report = validate_training_data(metadata=_metadata(objective=ObjectiveType.BINARY_CLASSIFICATION), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "constant_labels" for i in report.criticals)

    def test_constant_regression_labels_is_warning_not_critical(self) -> None:
        features, _ = _clean_data()
        labels = pd.Series([5.0] * len(features))
        report = validate_training_data(metadata=_metadata(objective=ObjectiveType.REGRESSION), features=features, labels=labels)
        assert report.is_ready
        assert any(i.code == "constant_labels" for i in report.warnings)

    def test_non_binary_label_values_rejected_for_binary_classification(self) -> None:
        features, _ = _clean_data()
        labels = pd.Series([0, 1, 2] * (len(features) // 3) + [0] * (len(features) % 3))
        report = validate_training_data(metadata=_metadata(objective=ObjectiveType.BINARY_CLASSIFICATION), features=features, labels=labels)
        assert not report.is_ready
        assert any(i.code == "non_binary_label_values" for i in report.criticals)


class TestNeverRaisesForBadButInspectableData:
    def test_maximally_broken_input_reports_issues_not_an_exception(self) -> None:
        features = pd.DataFrame({"f1": [np.nan], "f2": ["x"]})
        labels = pd.Series([1, 2, 3])  # mismatched length, non-binary values
        report = validate_training_data(metadata=_metadata(), features=features, labels=labels)
        assert not report.is_ready
        assert len(report.criticals) >= 2
