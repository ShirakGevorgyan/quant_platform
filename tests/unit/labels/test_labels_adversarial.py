"""Milestone 11, Phase 3, Part A: the adversarial audit, one test class
per attack, each run against the real infrastructure -- never a mock.
Some scenarios are also exercised (from a different angle) in
`test_labels_diagnostics.py`/`test_labels_verification.py`/
`test_labels_reconciliation.py`; this file exists for direct,
unambiguous traceability against a single named list, one class per
item."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_platform.core.exceptions import (
    DuplicateLabelSpecificationError,
    LabelGenerationContractError,
    LabelIdentityError,
    LabelMutableAliasError,
    LabelReconciliationError,
    LabelRecoveryError,
    LabelReplayError,
    SchemaVersionError,
    UnknownLabelSpecificationError,
)
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.diagnostics import compute_label_diagnostics
from quant_platform.labels.evidence import LabelDimensionKind
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.models import LabelSpecification
from quant_platform.labels.reconciliation import LabelReconciliation
from quant_platform.labels.recovery import LabelRecovery
from quant_platform.labels.registry import LabelRegistry
from quant_platform.labels.replay import LabelReplay
from quant_platform.labels.verification import verify_bundle_self_consistency


class Test01SpecificationIdTampering:
    def test_tampered_id_caught_by_self_consistency(self, specification: LabelSpecification) -> None:
        tampered = replace(specification, label_specification_id="0" * 64)
        consistent, issues = tampered.verify_self_consistency()
        assert consistent is False
        assert any("label_specification_id" in i for i in issues)

    def test_tampered_id_refused_at_registration(self, specification: LabelSpecification) -> None:
        tampered = replace(specification, label_specification_id="0" * 64)
        with pytest.raises(LabelIdentityError):
            LabelRegistry().register(tampered)


class Test02ParameterHashTampering:
    def test_tampered_parameter_hash_caught(self, specification: LabelSpecification) -> None:
        tampered = replace(specification, parameter_hash="0" * 64)
        consistent, issues = tampered.verify_self_consistency()
        assert consistent is False
        assert any("parameter_hash" in i for i in issues)

    def test_tampered_parameters_without_updating_hash_caught(self, specification: LabelSpecification) -> None:
        tampered = replace(specification, parameters={"horizon_bars": 999})
        consistent, _issues = tampered.verify_self_consistency()
        assert consistent is False


class Test03ManifestChecksumCorruption:
    def test_tampered_checksum_caught(self, manifest: LabelManifest) -> None:
        tampered = replace(manifest, manifest_checksum="deadbeef" * 8)
        consistent, _issues = tampered.verify_self_consistency()
        assert consistent is False

    def test_diagnostics_marks_it_blocking(self, bundle, manifest: LabelManifest) -> None:
        tampered = replace(manifest, manifest_checksum="deadbeef" * 8)
        diagnostics = compute_label_diagnostics(bundle, tampered)
        assert diagnostics.dimension_result(LabelDimensionKind.MANIFEST_INTEGRITY).is_blocking is True


class Test04DuplicateSpecification:
    def test_registering_the_same_id_twice_is_refused(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        with pytest.raises(DuplicateLabelSpecificationError):
            registry.register(specification)


class Test05UnknownSpecification:
    def test_lookup_of_never_registered_id_raises(self) -> None:
        with pytest.raises(UnknownLabelSpecificationError):
            LabelRegistry().lookup("ghost-specification-id")

    def test_freeze_of_never_registered_id_raises(self) -> None:
        with pytest.raises(UnknownLabelSpecificationError):
            LabelRegistry().freeze("ghost-specification-id")


class Test06MutableAliasInjection:
    def test_generator_returning_a_source_column_is_refused(self, specification, source_data, source_content_id, aliasing_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=aliasing_generator_fn)
        with pytest.raises(LabelMutableAliasError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)


class Test07GenerationContractViolations:
    def test_wrong_length_refused(self, specification, source_data, source_content_id, wrong_length_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=wrong_length_generator_fn)
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_non_series_refused(self, specification, source_data, source_content_id, non_series_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=non_series_generator_fn)  # type: ignore[arg-type]
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_non_numeric_refused(self, specification, source_data, source_content_id, non_numeric_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=non_numeric_generator_fn)
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)


class Test08CrossSpecificationReconciliationGuard:
    def test_reconciling_two_different_specifications_raises(self, bundle, manifest, other_family_specification) -> None:
        other_bundle = replace(bundle, specification=other_family_specification)
        with pytest.raises(LabelReconciliationError):
            LabelReconciliation().reconcile(bundle, other_bundle, baseline_manifest=manifest, candidate_manifest=manifest)


class Test09CrossSpecificationReplayGuard:
    def test_replaying_with_a_mismatched_definition_raises(self, bundle, source_data, source_content_id, other_family_specification, marker_generator_fn) -> None:
        wrong_definition = LabelDefinition(specification=other_family_specification, generate=marker_generator_fn)
        with pytest.raises(LabelReplayError):
            LabelReplay().replay(wrong_definition, source_data, source_content_id=source_content_id, original=bundle)


class Test10CrossSpecificationRecoveryGuard:
    def test_recovering_with_a_mismatched_definition_raises(self, specification, other_family_specification, source_data, source_content_id, marker_generator_fn) -> None:
        wrong_definition = LabelDefinition(specification=other_family_specification, generate=marker_generator_fn)
        with pytest.raises(LabelRecoveryError):
            LabelRecovery().recover(specification, wrong_definition, source_data, source_content_id=source_content_id)


class Test11NonTrailingNanPointInTimeShape:
    def test_a_nan_hole_followed_by_valid_data_is_flagged(self, specification, source_data, source_content_id, manifest, non_trailing_nan_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=non_trailing_nan_generator_fn)
        bad_bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        diagnostics = compute_label_diagnostics(bad_bundle, manifest)
        assert diagnostics.dimension_result(LabelDimensionKind.AVAILABILITY).score < 1.0


class Test12UnknownIdentityAlgorithm:
    def test_unrecognized_algorithm_flagged_non_blocking(self, bundle, manifest) -> None:
        tampered_spec = replace(bundle.specification, identity_algorithm="md5-v0")
        tampered_bundle = replace(bundle, specification=tampered_spec)
        diagnostics = compute_label_diagnostics(tampered_bundle, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.REPRODUCIBILITY)
        assert result.score < 1.0
        assert result.is_blocking is False


class Test13SchemaMismatch:
    def test_wrong_schema_version_rejected_for_specification(self) -> None:
        with pytest.raises(SchemaVersionError):
            LabelSpecification.from_json_dict({"schema_version": 999, "label_specification_id": "x"})

    def test_wrong_schema_version_rejected_for_manifest(self) -> None:
        with pytest.raises(SchemaVersionError):
            LabelManifest.from_json_dict({"schema_version": 999, "manifest_checksum": "x"})


class Test14RecoveryNeverReturnsAWrongBundle:
    def test_mismatched_expected_identity_fails_closed(self, specification, definition, source_data, source_content_id) -> None:
        bogus = compute_label_identity(specification.label_specification_id, pd.Series([0.0]), source_content_id="unrelated")
        result = LabelRecovery().recover(specification, definition, source_data, source_content_id=source_content_id, expected_identity=bogus)
        assert result.recoverable is False
        assert result.recovered_bundle is None


class Test15BundleSelfConsistencyManifestMismatch:
    def test_manifest_pointing_at_a_different_specification_caught(self, bundle, manifest) -> None:
        wrong_manifest = replace(manifest, label_specification_id="a-different-spec-entirely")
        consistent, issues = verify_bundle_self_consistency(bundle, wrong_manifest)
        assert consistent is False
        assert any("label_specification_id" in i for i in issues)
