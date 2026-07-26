"""Golden-hash and sensitivity tests for `experiment_identity.
compute_experiment_identity` -- the architectural centerpiece of
Milestone 4A. Every spec here is built EXPLICITLY inline (never via a
shared conftest builder that could silently drift) so the golden hash
below is reproducible from this file alone."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import ExperimentIdentityError
from quant_platform.ml.experiment_identity import (
    IDENTITY_SCHEMA_VERSION,
    ExperimentIdentity,
    compute_experiment_identity,
    verify_experiment_identity,
)
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import (
    CodeRevisionBinding,
    DatasetBinding,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.seeds import SeedConfiguration

GOLDEN_EXPERIMENT_ID = "e9e8f325653829d219e27050622c76b800176f8cd8be09b0abc3d4a41ed406b0"


def _golden_spec(**overrides: object) -> ExperimentSpec:
    base: dict[str, object] = {
        "dataset_binding": DatasetBinding(
            dataset_id="xauusd_h1_v1", manifest_version="1", content_id="a" * 64, symbol="XAUUSD", base_timeframe="H1"
        ),
        "feature_binding": FeatureBinding(
            feature_names=("atr_14", "rsi_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint="b" * 64,
        ),
        "label_binding": LabelBinding(name="fwd_ret_10", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS),
        "split_binding": SplitBinding(strategy="time_ordered_holdout"),
        "preprocessing_binding": PreprocessingBinding(),
        "model_name": "constant_test_model", "model_version": "1",
        "hyperparameters": ModelHyperparameters(values={"alpha": 0.1}),
        "objective": ObjectiveType.REGRESSION,
        "seed_configuration": SeedConfiguration(master_seed=42),
        "code_revision_binding": CodeRevisionBinding(revision="c" * 40, source="git", is_dirty=True),
        "primary_metric": "rmse",
    }
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


class TestGoldenHash:
    """Section 6's required "fixed canonical spec -> expected hash" test.
    If this ever fails after an intentional identity-scheme change, the
    fix is to update `GOLDEN_EXPERIMENT_ID` deliberately -- never to
    silence the test."""

    def test_golden_experiment_id(self) -> None:
        identity = compute_experiment_identity(_golden_spec())
        assert identity.experiment_id == GOLDEN_EXPERIMENT_ID
        assert identity.schema_version == IDENTITY_SCHEMA_VERSION

    def test_golden_hash_reproducible_across_independent_construction(self) -> None:
        spec_a = _golden_spec()
        spec_b = _golden_spec()
        assert compute_experiment_identity(spec_a) == compute_experiment_identity(spec_b)


class TestIdentityDeterminism:
    def test_dict_insertion_order_in_feature_versions_does_not_affect_identity(self) -> None:
        spec_a = _golden_spec()
        spec_b = _golden_spec(
            feature_binding=FeatureBinding(
                feature_names=("atr_14", "rsi_14"), feature_versions={"rsi_14": "1", "atr_14": "1"},
                feature_registry_fingerprint="b" * 64,
            )
        )
        assert compute_experiment_identity(spec_a) == compute_experiment_identity(spec_b)

    def test_dict_insertion_order_in_hyperparameters_does_not_affect_identity(self) -> None:
        spec_a = _golden_spec(hyperparameters=ModelHyperparameters(values={"alpha": 0.1, "beta": 2}))
        spec_b = _golden_spec(hyperparameters=ModelHyperparameters(values={"beta": 2, "alpha": 0.1}))
        assert compute_experiment_identity(spec_a) == compute_experiment_identity(spec_b)

    def test_dict_insertion_order_in_environment_requirements_does_not_affect_identity(self) -> None:
        spec_a = _golden_spec(environment_requirements={"numpy": ">=2.0", "pandas": ">=2.2"})
        spec_b = _golden_spec(environment_requirements={"pandas": ">=2.2", "numpy": ">=2.0"})
        assert compute_experiment_identity(spec_a) == compute_experiment_identity(spec_b)


class TestDescriptiveFieldsDoNotAffectIdentity:
    def test_notes_do_not_affect_identity(self) -> None:
        base_id = compute_experiment_identity(_golden_spec())
        changed_id = compute_experiment_identity(_golden_spec(notes="totally different notes"))
        assert base_id == changed_id

    def test_tags_do_not_affect_identity(self) -> None:
        base_id = compute_experiment_identity(_golden_spec())
        changed_id = compute_experiment_identity(_golden_spec(tags=("a", "b", "c")))
        assert base_id == changed_id

    def test_primary_metric_does_not_affect_identity(self) -> None:
        base_id = compute_experiment_identity(_golden_spec())
        changed_id = compute_experiment_identity(_golden_spec(primary_metric="mae"))
        assert base_id == changed_id

    def test_primary_metric_is_absent_from_identity_payload_entirely(self) -> None:
        """Belt-and-suspenders beyond the hash-equality test above: prove
        `primary_metric` is not merely coincidentally cancelled out by
        hashing, but genuinely never appears in the payload that gets
        hashed at all. Audited as part of the Milestone 4A correctness
        review -- see `ExperimentSpec`'s module docstring for the
        forward-looking condition under which this decision must change."""
        payload = _golden_spec().to_identity_payload()
        assert "primary_metric" not in payload

    def test_primary_metric_change_leaves_entire_identity_payload_byte_identical(self) -> None:
        base_payload = _golden_spec().to_identity_payload()
        changed_payload = _golden_spec(primary_metric="a_totally_different_metric_name").to_identity_payload()
        assert base_payload == changed_payload


class TestIdentitySensitiveToMaterialChanges:
    def test_dataset_content_id_change_changes_identity(self) -> None:
        changed = _golden_spec(dataset_binding=DatasetBinding(
            dataset_id="xauusd_h1_v1", manifest_version="1", content_id="f" * 64, symbol="XAUUSD", base_timeframe="H1"
        ))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_feature_version_change_changes_identity(self) -> None:
        changed = _golden_spec(feature_binding=FeatureBinding(
            feature_names=("atr_14", "rsi_14"), feature_versions={"atr_14": "2", "rsi_14": "1"},
            feature_registry_fingerprint="b" * 64,
        ))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_feature_order_change_changes_identity(self) -> None:
        changed = _golden_spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint="b" * 64,
        ))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_label_definition_change_changes_identity(self) -> None:
        changed = _golden_spec(label_binding=LabelBinding(name="different", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_split_definition_change_changes_identity(self) -> None:
        changed = _golden_spec(split_binding=SplitBinding(strategy="expanding_walk_forward"))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_preprocessing_fingerprint_change_changes_identity(self) -> None:
        changed = _golden_spec(preprocessing_binding=PreprocessingBinding(fitted_preprocessing_fingerprint="d" * 64))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_model_version_change_changes_identity(self) -> None:
        changed = _golden_spec(model_version="2")
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_hyperparameter_value_change_changes_identity(self) -> None:
        changed = _golden_spec(hyperparameters=ModelHyperparameters(values={"alpha": 0.2}))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_objective_change_changes_identity(self) -> None:
        # Regression-compatible label must change too to keep the spec constructible.
        binary_label = LabelBinding(name="fwd_ret_10", kind="binary_direction", horizon_bars=10, label_type=LabelType.BINARY)
        changed = _golden_spec(objective=ObjectiveType.BINARY_CLASSIFICATION, label_binding=binary_label)
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_master_seed_change_changes_identity(self) -> None:
        changed = _golden_spec(seed_configuration=SeedConfiguration(master_seed=43))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_code_revision_change_changes_identity(self) -> None:
        changed = _golden_spec(code_revision_binding=CodeRevisionBinding(revision="d" * 40, source="git", is_dirty=True))
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)

    def test_explicit_environment_requirement_changes_identity(self) -> None:
        changed = _golden_spec(environment_requirements={"numpy": ">=2.0"})
        assert compute_experiment_identity(_golden_spec()) != compute_experiment_identity(changed)


class TestNoMachineOrTimeDependence:
    def test_identity_has_no_wall_clock_timestamp_field(self) -> None:
        payload = _golden_spec().to_identity_payload()

        def _walk(obj: object) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    assert "created_at" not in str(key) and "timestamp" not in str(key).lower()
                    _walk(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item)

        _walk(payload)

    def test_identity_has_no_absolute_path_in_payload(self, tmp_path: object) -> None:
        payload = _golden_spec().to_identity_payload()
        assert str(tmp_path) not in repr(payload)
        assert "C:\\" not in repr(payload) and "/home/" not in repr(payload)


class TestExperimentIdentityModel:
    def test_round_trip(self) -> None:
        identity = ExperimentIdentity(schema_version=1, experiment_id="a" * 64)
        assert ExperimentIdentity.from_json_dict(identity.to_json_dict()) == identity

    def test_invalid_hash_format_rejected(self) -> None:
        with pytest.raises(ExperimentIdentityError):
            ExperimentIdentity(schema_version=1, experiment_id="not-a-hash")

    def test_verify_experiment_identity_true_for_matching(self) -> None:
        spec = _golden_spec()
        identity = compute_experiment_identity(spec)
        assert verify_experiment_identity(spec, identity)

    def test_verify_experiment_identity_false_for_tampered(self) -> None:
        spec = _golden_spec()
        tampered = ExperimentIdentity(schema_version=1, experiment_id="0" * 64)
        assert not verify_experiment_identity(spec, tampered)

    def test_verify_experiment_identity_false_when_spec_changed_but_identity_stale(self) -> None:
        spec = _golden_spec()
        identity = compute_experiment_identity(spec)
        changed_spec = replace(spec, model_version="2")
        assert not verify_experiment_identity(changed_spec, identity)
