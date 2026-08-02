from __future__ import annotations

from dataclasses import replace

from quant_platform.feature_discovery.catalog import build_feature_infrastructure_bundle
from quant_platform.feature_discovery.infra_verification import (
    FeatureInfrastructureVerificationResult,
    FeatureInfrastructureVerifier,
    verify_bundle_self_consistency,
)


class TestVerifyBundleSelfConsistency:
    def test_unmodified_bundle_is_self_consistent(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        consistent, issues = verify_bundle_self_consistency(bundle, registry=discovered_registry)
        assert consistent is True
        assert issues == ()

    def test_tampered_manifest_id_is_caught(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered = replace(bundle, manifest=replace(bundle.manifest, manifest_id="deadbeef00000000"))
        consistent, issues = verify_bundle_self_consistency(tampered)
        assert consistent is False
        assert any("manifest_id" in i for i in issues)

    def test_tampered_deterministic_identity_is_caught(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(
            replace(m, deterministic_identity="corrupted") if m.feature_name == "trend" else m for m in bundle.snapshot.metadata
        )
        tampered = replace(bundle, snapshot=replace(bundle.snapshot, metadata=tampered_metadata))
        consistent, issues = verify_bundle_self_consistency(tampered, registry=discovered_registry)
        assert consistent is False
        assert any("identity" in i for i in issues)

    def test_no_registry_supplied_skips_identity_check(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(
            replace(m, deterministic_identity="corrupted") if m.feature_name == "trend" else m for m in bundle.snapshot.metadata
        )
        tampered = replace(bundle, snapshot=replace(bundle.snapshot, metadata=tampered_metadata))
        consistent, issues = verify_bundle_self_consistency(tampered)  # no registry
        assert consistent is True
        assert issues == ()


class TestFeatureInfrastructureVerifier:
    def test_clean_bundle_verifies(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureVerifier().verify(bundle, discovered_registry, discovered_manifest)
        assert result.verified is True
        assert result.self_consistent is True
        assert result.reconciliation.reconciled is True

    def test_tampered_bundle_fails_verification(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered = replace(bundle, manifest=replace(bundle.manifest, manifest_id="deadbeef00000000"))
        result = FeatureInfrastructureVerifier().verify(tampered, discovered_registry, discovered_manifest)
        assert result.verified is False
        assert result.self_consistent is False

    def test_dataset_id_mismatch_fails_gracefully(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        mismatched = replace(bundle, snapshot=replace(bundle.snapshot, dataset_id="f" * 16))
        result = FeatureInfrastructureVerifier().verify(mismatched, discovered_registry, discovered_manifest)
        assert result.verified is False
        assert any(i.kind == "dataset_id_mismatch" for i in result.reconciliation.issues)

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureVerifier().verify(bundle, discovered_registry, discovered_manifest)
        assert FeatureInfrastructureVerificationResult.from_json_dict(result.to_json_dict()) == result
