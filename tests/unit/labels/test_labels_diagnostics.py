from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.diagnostics import LabelDiagnostics, compute_label_diagnostics
from quant_platform.labels.evidence import LABEL_DIMENSION_ORDER, LabelDimensionKind
from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.models import LabelSpecification
from quant_platform.ml.persistence import format_utc_timestamp, utc_now


class TestComputeLabelDiagnostics:
    def test_covers_all_seven_dimensions_in_fixed_order(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        assert tuple(r.dimension for r in diagnostics.dimension_results) == LABEL_DIMENSION_ORDER

    def test_clean_bundle_scores_full_on_identity(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        assert diagnostics.dimension_result(LabelDimensionKind.IDENTITY).score == 1.0
        assert diagnostics.is_blocking is False

    def test_overall_score_is_mean_of_dimension_scores(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        expected = sum(r.score for r in diagnostics.dimension_results) / len(diagnostics.dimension_results)
        assert diagnostics.overall_score == expected

    def test_json_round_trip(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        assert LabelDiagnostics.from_json_dict(diagnostics.to_json_dict()) == diagnostics


class TestIdentityDimensionCatchesTampering:
    def test_tampered_values_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_values = bundle.values.copy()
        tampered_values.iloc[0] = 999.0
        tampered = replace(bundle, values=tampered_values)
        diagnostics = compute_label_diagnostics(tampered, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.IDENTITY)
        assert result.score == 0.0
        assert result.is_blocking is True


class TestVersioningDimensionCatchesTampering:
    def test_tampered_specification_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_spec = replace(bundle.specification, parameter_hash="0" * 64)
        tampered = replace(bundle, specification=tampered_spec)
        diagnostics = compute_label_diagnostics(tampered, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.VERSIONING)
        assert result.score == 0.0
        assert result.is_blocking is True


class TestAvailabilityDimensionFlagsNonTrailingNan:
    def test_non_trailing_nan_pattern_flagged(
        self, specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, manifest: LabelManifest, non_trailing_nan_generator_fn,
    ) -> None:
        definition = LabelDefinition(specification=specification, generate=non_trailing_nan_generator_fn)
        bad_bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        diagnostics = compute_label_diagnostics(bad_bundle, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.AVAILABILITY)
        assert result.score < 1.0

    def test_well_formed_trailing_tail_scores_full(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.AVAILABILITY)
        assert result.score == 1.0


class TestManifestIntegrityDimensionCatchesTampering:
    def test_tampered_manifest_checksum_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_manifest = replace(manifest, manifest_checksum="0" * 64)
        diagnostics = compute_label_diagnostics(bundle, tampered_manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.MANIFEST_INTEGRITY)
        assert result.score == 0.0
        assert result.is_blocking is True

    def test_mismatched_specification_id_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        wrong_manifest = replace(manifest, label_specification_id="some-other-spec")
        diagnostics = compute_label_diagnostics(bundle, wrong_manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.MANIFEST_INTEGRITY)
        assert result.is_blocking is True


class TestDeterminismDimension:
    def test_clean_bundle_scores_full(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.DETERMINISM)
        assert result.score == 1.0
        assert result.is_blocking is False


class TestReproducibilityDimensionFlagsUnknownAlgorithm:
    def test_unknown_identity_algorithm_flagged(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_spec = replace(bundle.specification, identity_algorithm="md5-v0")
        tampered = replace(bundle, specification=tampered_spec)
        diagnostics = compute_label_diagnostics(tampered, manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.REPRODUCIBILITY)
        assert result.score < 1.0
        assert result.is_blocking is False  # a WARNING, not a hard block


class TestLineageDimensionFlagsMissingFields:
    def test_missing_dataset_identity_is_blocking(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        broken_manifest = replace(manifest, dataset_identity="")
        diagnostics = compute_label_diagnostics(bundle, broken_manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.LINEAGE)
        assert result.is_blocking is True

    def test_missing_optional_lineage_is_informational_only(
        self, specification: LabelSpecification, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        bare_manifest = build_label_manifest(specification, generation_timestamp=format_utc_timestamp(utc_now()))
        bare_bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        diagnostics = compute_label_diagnostics(bare_bundle, bare_manifest)
        result = diagnostics.dimension_result(LabelDimensionKind.LINEAGE)
        assert result.is_blocking is False
        assert result.score < 1.0
