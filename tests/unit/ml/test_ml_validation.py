from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.unit.ml.conftest import (
    FEATURE_REGISTRY_FINGERPRINT,
    build_registry,
    make_dataset_manifest,
    make_experiment_spec_kwargs,
)

from quant_platform.ml.experiment_identity import compute_experiment_identity
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import MANIFEST_SCHEMA_VERSION, ExperimentManifest
from quant_platform.ml.models import ExperimentStatus, FeatureBinding, ObjectiveType
from quant_platform.ml.validation import validate_experiment_spec


def _spec(**overrides: object) -> ExperimentSpec:
    return ExperimentSpec(**make_experiment_spec_kwargs(**overrides))


class TestHappyPath:
    def test_valid_spec_is_ready(self, tmp_path: Path) -> None:
        report = validate_experiment_spec(
            _spec(), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(),
            ml_artifacts_root=tmp_path,
        )
        assert report.is_ready
        assert not report.criticals
        assert not report.errors

    def test_all_expected_info_codes_present(self, tmp_path: Path) -> None:
        report = validate_experiment_spec(
            _spec(), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(),
            ml_artifacts_root=tmp_path,
        )
        codes = {i.code for i in report.issues}
        assert {
            "model_definition_resolved", "dataset_manifest_consistent", "feature_set_consistent",
            "feature_registry_fingerprint_consistent", "label_objective_compatible",
            "preprocessing_definition_consistent", "split_binding_present", "seed_configuration_valid",
            "code_revision_present", "feature_names_unique", "artifact_root_resolved", "identity_computed",
        } <= codes


class TestModelDefinitionChecks:
    def test_unknown_model_is_critical(self, tmp_path: Path) -> None:
        report = validate_experiment_spec(
            _spec(model_name="does_not_exist"), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(),
            ml_artifacts_root=tmp_path,
        )
        assert not report.is_ready
        assert any(i.code == "unknown_model_definition" for i in report.criticals)

    def test_model_not_supporting_objective_is_critical(self, tmp_path: Path) -> None:
        registry = build_registry(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,))
        report = validate_experiment_spec(
            _spec(), model_registry=registry, dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
        )
        assert not report.is_ready
        assert any(i.code == "model_does_not_support_objective" for i in report.criticals)


class TestDatasetBindingChecks:
    def test_content_id_mismatch_is_critical(self, tmp_path: Path) -> None:
        spec = _spec()
        bad_manifest = make_dataset_manifest(content_id="f" * 64)
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=bad_manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "dataset_manifest_mismatch" for i in report.criticals)

    def test_manifest_version_mismatch_is_critical(self, tmp_path: Path) -> None:
        spec = _spec()
        bad_manifest = make_dataset_manifest(version="different-version")
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=bad_manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "dataset_manifest_mismatch" for i in report.criticals)


class TestFeatureBindingChecks:
    def test_missing_feature_is_critical(self, tmp_path: Path) -> None:
        spec = _spec(feature_binding=FeatureBinding(
            feature_names=("atr_14", "missing_feature"), feature_versions={"atr_14": "1", "missing_feature": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "feature_set_mismatch" for i in report.criticals)

    def test_feature_order_mismatch_is_error(self, tmp_path: Path) -> None:
        spec = _spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "feature_order_mismatch" for i in report.errors)

    def test_feature_registry_fingerprint_mismatch_is_critical(self, tmp_path: Path) -> None:
        spec = _spec(feature_binding=FeatureBinding(
            feature_names=("atr_14", "rsi_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint="d" * 64,
        ))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "feature_registry_fingerprint_mismatch" for i in report.criticals)


class TestLabelBindingChecks:
    """`_validate_label_binding` cross-validates `LabelBinding` against the
    research dataset manifest's own recorded `label_definition`. Per that
    function's docstring: `name`/`kind`/`horizon_bars`/`params` are the
    fields actually present on both sides and so are compared directly;
    a "label version" and a "target column" concept do not exist anywhere
    in `ResearchDatasetManifest`'s schema, so there is nothing to compare
    for those (documented no-ops, not gaps); `LabelBinding.label_type` has
    no manifest analogue either and remains governed solely by
    `_validate_label_objective`. The default `make_dataset_manifest()`/
    `make_label_binding()` fixtures are constructed to agree exactly
    (name="fwd_ret_10", kind="forward_return", horizon_bars=10, params={})
    -- every test below starts from that consistent baseline and changes
    exactly one side to introduce a mismatch."""

    def test_exact_match_is_consistent_info(self, tmp_path: Path) -> None:
        report = validate_experiment_spec(
            _spec(), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
        )
        assert report.is_ready
        assert any(i.code == "research_dataset_label_binding_consistent" for i in report.infos)

    def test_name_mismatch_is_critical_and_blocks_ready(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(name="different_name", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_name_mismatch" for i in report.criticals)
        assert any(i.code == "research_dataset_label_binding_mismatch" for i in report.criticals)

    def test_kind_mismatch_is_critical_and_blocks_ready(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(name="fwd_ret_10", kind="binary_direction", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_kind_mismatch" for i in report.criticals)

    def test_horizon_mismatch_is_critical_and_blocks_ready(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(name="fwd_ret_10", kind="forward_return", horizon_bars=999, label_type=LabelType.CONTINUOUS))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_horizon_mismatch" for i in report.criticals)

    def test_params_mismatch_is_critical_and_blocks_ready(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(
            name="fwd_ret_10", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS,
            params={"threshold": 0.01},
        ))
        manifest = make_dataset_manifest(label_definition={
            "name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": 10, "params": {"threshold": 0.02},
        })
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_params_mismatch" for i in report.criticals)

    def test_params_missing_on_manifest_defaults_to_empty_dict_not_incomplete(self, tmp_path: Path) -> None:
        """A manifest's `label_definition` with no `"params"` key at all
        (e.g. a foreign/older schema) is treated as `{}`, matching
        `LabelDefinition`/`LabelBinding`'s own default -- NOT as missing
        metadata, since `params` is optional by design on both sides."""
        manifest = make_dataset_manifest(label_definition={"name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": 10})
        report = validate_experiment_spec(_spec(), model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert report.is_ready
        assert any(i.code == "research_dataset_label_binding_consistent" for i in report.infos)

    @pytest.mark.parametrize("missing_field", ["name", "kind", "horizon_bars"])
    def test_missing_required_label_metadata_is_critical_and_blocks_ready(self, tmp_path: Path, missing_field: str) -> None:
        full = {"name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": 10, "params": {}}
        del full[missing_field]
        manifest = make_dataset_manifest(label_definition=full)
        report = validate_experiment_spec(_spec(), model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "research_dataset_label_binding_incomplete" for i in report.criticals)
        # Never a raw exception, and never masked into a bare ValueError/KeyError.
        assert all(i.severity.value in ("critical", "error") or i.code != "research_dataset_label_binding_incomplete" for i in report.issues)

    def test_non_integer_horizon_in_manifest_is_incomplete_not_a_raised_exception(self, tmp_path: Path) -> None:
        manifest = make_dataset_manifest(label_definition={"name": "fwd_ret_10", "kind": "forward_return", "horizon_bars": "not-a-number", "params": {}})
        report = validate_experiment_spec(_spec(), model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "research_dataset_label_binding_incomplete" for i in report.criticals)

    def test_objective_compatible_but_dataset_inconsistent_label_still_fails(self, tmp_path: Path) -> None:
        """A `LabelBinding` whose `label_type` is perfectly compatible with
        the declared objective (so `_validate_label_objective` reports
        clean) must still fail overall if it disagrees with the dataset's
        actual recorded label -- these are two independent checks."""
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(name="fwd_ret_10", kind="forward_return", horizon_bars=20, label_type=LabelType.CONTINUOUS))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_objective_compatible" for i in report.infos)
        assert any(i.code == "label_horizon_mismatch" for i in report.criticals)

    def test_mismatch_never_silently_normalizes_equal(self, tmp_path: Path) -> None:
        """Different labels must never compare equal -- e.g. a name that
        differs only by case is still a genuine mismatch, never folded
        together by case-insensitive or whitespace-trimmed comparison."""
        from quant_platform.ml.models import LabelBinding, LabelType

        spec = _spec(label_binding=LabelBinding(name="FWD_RET_10", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "label_name_mismatch" for i in report.criticals)


class TestPreprocessingChecks:
    def test_preprocessing_definition_mismatch_is_error(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import PreprocessingBinding

        spec = _spec(preprocessing_binding=PreprocessingBinding(preprocessing_definition={"a": "standard_scale"}))
        report = validate_experiment_spec(
            spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(preprocessing_definition={}),
            ml_artifacts_root=tmp_path,
        )
        assert not report.is_ready
        assert any(i.code == "preprocessing_definition_mismatch" for i in report.errors)

    def test_fitted_fingerprint_mismatch_is_error(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import PreprocessingBinding

        spec = _spec(preprocessing_binding=PreprocessingBinding(fitted_preprocessing_fingerprint="d" * 64))
        manifest = make_dataset_manifest(fitted_preprocessing_fingerprint="e" * 64)
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert not report.is_ready
        assert any(i.code == "fitted_preprocessing_fingerprint_mismatch" for i in report.errors)

    def test_fitted_fingerprint_presence_mismatch_is_warning(self, tmp_path: Path) -> None:
        from quant_platform.ml.models import PreprocessingBinding

        spec = _spec(preprocessing_binding=PreprocessingBinding(fitted_preprocessing_fingerprint="d" * 64))
        manifest = make_dataset_manifest(fitted_preprocessing_fingerprint=None)
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert report.is_ready  # warning-only, does not block readiness
        assert any(i.code == "fitted_preprocessing_presence_mismatch" for i in report.warnings)


class TestSplitBindingChecks:
    def test_split_strategy_mismatch_is_warning_only(self, tmp_path: Path) -> None:
        spec = _spec()  # strategy="time_ordered_holdout"
        manifest = make_dataset_manifest(split_definition={"strategy": "chronological"})
        report = validate_experiment_spec(spec, model_registry=build_registry(), dataset_manifest=manifest, ml_artifacts_root=tmp_path)
        assert report.is_ready
        assert any(i.code == "split_strategy_mismatch" for i in report.warnings)


class TestArtifactRootChecks:
    def test_empty_artifact_root_is_critical(self) -> None:
        report = validate_experiment_spec(
            _spec(), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root="",
        )
        assert not report.is_ready
        assert any(i.code == "invalid_artifact_root" for i in report.criticals)


class TestIdentityChecks:
    def test_expected_identity_match_is_info(self, tmp_path: Path) -> None:
        spec = _spec()
        identity = compute_experiment_identity(spec)
        report = validate_experiment_spec(
            spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
            expected_identity=identity,
        )
        assert report.is_ready
        assert any(i.code == "identity_computed" for i in report.infos)

    def test_expected_identity_mismatch_is_critical(self, tmp_path: Path) -> None:
        spec = _spec()
        other_identity = compute_experiment_identity(_spec(model_version="99"))
        report = validate_experiment_spec(
            spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
            expected_identity=other_identity,
        )
        assert not report.is_ready
        assert any(i.code == "identity_mismatch" for i in report.criticals)


class TestNoAccidentalOverwriteChecks:
    def _manifest_for(self, spec: ExperimentSpec, status: ExperimentStatus, **overrides: object) -> ExperimentManifest:
        from quant_platform.ml.environment import capture_environment_snapshot
        from quant_platform.ml.persistence import format_utc_timestamp, utc_now

        base: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION, "identity": compute_experiment_identity(spec), "spec": spec,
            "model_definition_fingerprint": "f" * 64, "status": status,
            "environment_snapshot": capture_environment_snapshot(), "artifact_references": (),
            "validation_report_reference": None, "created_at": format_utc_timestamp(utc_now()),
        }
        base.update(overrides)
        return ExperimentManifest(**base)  # type: ignore[arg-type]

    def test_no_existing_manifest_produces_no_issue(self, tmp_path: Path) -> None:
        report = validate_experiment_spec(
            _spec(), model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
            existing_manifest=None,
        )
        assert not any(i.code.startswith("experiment_already") for i in report.issues)

    def test_existing_completed_manifest_is_info_not_blocking(self, tmp_path: Path) -> None:
        from quant_platform.ml.persistence import format_utc_timestamp, utc_now

        spec = _spec()
        existing = self._manifest_for(
            spec, ExperimentStatus.COMPLETED, completed_at=format_utc_timestamp(utc_now())
        )
        report = validate_experiment_spec(
            spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
            existing_manifest=existing,
        )
        assert report.is_ready
        assert any(i.code == "experiment_already_finalized" for i in report.infos)

    def test_existing_in_progress_manifest_is_info(self, tmp_path: Path) -> None:
        spec = _spec()
        existing = self._manifest_for(spec, ExperimentStatus.VALIDATING)
        report = validate_experiment_spec(
            spec, model_registry=build_registry(), dataset_manifest=make_dataset_manifest(), ml_artifacts_root=tmp_path,
            existing_manifest=existing,
        )
        assert any(i.code == "experiment_already_in_progress" for i in report.infos)


def test_validation_never_raises_for_bad_experiment(tmp_path: Path) -> None:
    """A maximally broken spec/registry/manifest combination should still
    produce a `ValidationReport`, never propagate an exception -- only
    genuinely unusable configuration (handled above) surfaces as a
    CRITICAL issue rather than a raised exception."""
    bad_registry = build_registry()
    bad_manifest = make_dataset_manifest(content_id="9" * 64, feature_names=("nothing_matches",), feature_versions={"nothing_matches": "1"})
    spec = replace(_spec(model_name="nope"))
    report = validate_experiment_spec(spec, model_registry=bad_registry, dataset_manifest=bad_manifest, ml_artifacts_root=tmp_path)
    assert not report.is_ready
    assert len(report.criticals) >= 2
