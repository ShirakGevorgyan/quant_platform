from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.verification import (
    LabelVerificationResult,
    LabelVerifier,
    verify_bundle_self_consistency,
)


class TestVerifyBundleSelfConsistency:
    def test_clean_bundle_is_self_consistent(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        consistent, issues = verify_bundle_self_consistency(bundle, manifest)
        assert consistent is True
        assert issues == ()

    def test_tampered_values_caught(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_values = bundle.values.copy()
        tampered_values.iloc[0] = 12345.0
        tampered = replace(bundle, values=tampered_values)
        consistent, issues = verify_bundle_self_consistency(tampered, manifest)
        assert consistent is False
        assert any("content_id" in i for i in issues)

    def test_manifest_specification_mismatch_caught(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        wrong_manifest = replace(manifest, label_specification_id="different-spec")
        consistent, _issues = verify_bundle_self_consistency(bundle, wrong_manifest)
        assert consistent is False


class TestLabelVerifier:
    def test_clean_bundle_verifies(
        self, bundle: LabelBundle, manifest: LabelManifest, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        result = LabelVerifier().verify(bundle, manifest, definition, source_data, source_content_id=source_content_id)
        assert result.verified is True
        assert result.self_consistent is True
        assert result.reconciliation.reconciled is True

    def test_tampered_manifest_fails_verification(
        self, bundle: LabelBundle, manifest: LabelManifest, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        tampered_manifest = replace(manifest, manifest_checksum="0" * 64)
        result = LabelVerifier().verify(bundle, tampered_manifest, definition, source_data, source_content_id=source_content_id)
        assert result.verified is False
        assert result.self_consistent is False

    def test_json_round_trip(
        self, bundle: LabelBundle, manifest: LabelManifest, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        result = LabelVerifier().verify(bundle, manifest, definition, source_data, source_content_id=source_content_id)
        assert LabelVerificationResult.from_json_dict(result.to_json_dict()) == result
