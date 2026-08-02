from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import LabelReconciliationError
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.reconciliation import LabelReconciliation, LabelReconciliationResult


class TestLabelReconciliation:
    def test_self_reconciliation_is_clean(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        result = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=manifest)
        assert result.reconciled is True
        assert result.issues == ()

    def test_different_specification_ids_raise(self, bundle: LabelBundle, manifest: LabelManifest, other_family_specification) -> None:
        other_bundle = replace(bundle, specification=other_family_specification)
        with pytest.raises(LabelReconciliationError):
            LabelReconciliation().reconcile(bundle, other_bundle, baseline_manifest=manifest, candidate_manifest=manifest)

    def test_identity_drift_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_values = bundle.values.copy()
        tampered_values.iloc[0] = 555.0
        tampered_identity = compute_label_identity(bundle.specification.label_specification_id, tampered_values, source_content_id=bundle.identity.source_content_id)
        tampered_bundle = replace(bundle, values=tampered_values, identity=tampered_identity)
        result = LabelReconciliation().reconcile(bundle, tampered_bundle, baseline_manifest=manifest, candidate_manifest=manifest)
        assert result.reconciled is False
        assert any(i.kind == "identity_drift" for i in result.issues)

    def test_manifest_drift_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_manifest = replace(manifest, manifest_checksum="deadbeef" * 8)
        result = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=tampered_manifest)
        assert any(i.kind == "manifest_drift" for i in result.issues)

    def test_lineage_drift_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        drifted_manifest = replace(manifest, feature_identity="a-different-feature-fingerprint")
        result = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=drifted_manifest)
        assert any(i.kind == "lineage_drift" for i in result.issues)

    def test_specification_drift_detected(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        tampered_spec = replace(bundle.specification, generation_rule="a different rule text")
        tampered_bundle = replace(bundle, specification=tampered_spec)
        result = LabelReconciliation().reconcile(bundle, tampered_bundle, baseline_manifest=manifest, candidate_manifest=manifest)
        assert any(i.kind == "specification_drift" for i in result.issues)

    def test_json_round_trip(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        result = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=manifest)
        assert LabelReconciliationResult.from_json_dict(result.to_json_dict()) == result
