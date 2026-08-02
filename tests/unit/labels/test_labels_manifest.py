from __future__ import annotations

from dataclasses import replace

from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.models import LabelSpecification
from quant_platform.ml.persistence import format_utc_timestamp, utc_now


class TestBuildLabelManifest:
    def test_self_consistent_by_construction(self, manifest: LabelManifest) -> None:
        consistent, issues = manifest.verify_self_consistency()
        assert consistent is True
        assert issues == ()

    def test_dependency_chain_references_lineage(self, manifest: LabelManifest) -> None:
        joined = "|".join(manifest.dependency_chain)
        assert manifest.dataset_identity in joined
        assert manifest.manifest_identity in joined
        assert (manifest.feature_identity or "") in joined
        assert (manifest.qualification_identity or "") in joined

    def test_missing_feature_and_qualification_identity_are_optional(self, specification: LabelSpecification) -> None:
        manifest = build_label_manifest(specification, generation_timestamp=format_utc_timestamp(utc_now()))
        assert manifest.feature_identity is None
        assert manifest.qualification_identity is None
        consistent, issues = manifest.verify_self_consistency()
        assert consistent is True
        assert issues == ()

    def test_generation_timestamp_excluded_from_checksum(self, specification: LabelSpecification) -> None:
        a = build_label_manifest(specification, generation_timestamp="2024-01-01T00:00:00+00:00")
        b = build_label_manifest(specification, generation_timestamp="2030-01-01T00:00:00+00:00")
        assert a.manifest_checksum == b.manifest_checksum


class TestTamperedManifestFailsSelfConsistency:
    def test_tampered_checksum_detected(self, manifest: LabelManifest) -> None:
        tampered = replace(manifest, manifest_checksum="0" * 64)
        consistent, issues = tampered.verify_self_consistency()
        assert consistent is False
        assert issues != ()

    def test_tampered_dependency_chain_detected(self, manifest: LabelManifest) -> None:
        tampered = replace(manifest, dependency_chain=("tampered:1", "tampered:2"))
        consistent, _issues = tampered.verify_self_consistency()
        assert consistent is False


class TestJsonRoundTrip:
    def test_round_trip(self, manifest: LabelManifest) -> None:
        restored = LabelManifest.from_json_dict(manifest.to_json_dict())
        assert restored == manifest
