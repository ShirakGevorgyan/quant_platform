from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import FeatureDiscoveryReconciliationError
from quant_platform.feature_discovery.catalog import (
    FeatureInfrastructureBundle,
    build_feature_catalog,
    build_feature_infrastructure_bundle,
    build_feature_inventory,
    build_feature_manifest,
)
from quant_platform.feature_discovery.infra_reconciliation import (
    FeatureInfrastructureReconciliation,
    FeatureInfrastructureReconciliationResult,
)


def _rebuild_bundle_from_tampered_snapshot(bundle, tampered_snapshot):
    catalog = build_feature_catalog(tampered_snapshot)
    return FeatureInfrastructureBundle(
        snapshot=tampered_snapshot, graph=bundle.graph, catalog=catalog, inventory=build_feature_inventory(catalog),
        manifest=build_feature_manifest(tampered_snapshot),
    )


class TestFeatureInfrastructureReconciliation:
    def test_self_reconciliation_is_clean(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureReconciliation().reconcile(bundle, bundle)
        assert result.reconciled is True
        assert result.issues == ()

    def test_different_dataset_id_raises(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        other = replace(bundle, snapshot=replace(bundle.snapshot, dataset_id="f" * 16))
        with pytest.raises(FeatureDiscoveryReconciliationError):
            FeatureInfrastructureReconciliation().reconcile(bundle, other)

    def test_metadata_drift_detected(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(replace(m, warmup_requirement=999) if m.feature_name == "trend" else m for m in bundle.snapshot.metadata)
        tampered_bundle = _rebuild_bundle_from_tampered_snapshot(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered_bundle)
        assert result.reconciled is False
        assert any(i.kind == "metadata_drift" and i.feature_name == "trend" for i in result.issues)

    def test_manifest_drift_detected(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_bundle = replace(bundle, manifest=replace(bundle.manifest, manifest_id="deadbeef00000000"))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered_bundle)
        assert any(i.kind == "manifest_drift" for i in result.issues)

    def test_feature_drift_detected(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        reduced_metadata = tuple(m for m in bundle.snapshot.metadata if m.feature_name != "const")
        reduced_bundle = _rebuild_bundle_from_tampered_snapshot(bundle, replace(bundle.snapshot, metadata=reduced_metadata))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, reduced_bundle)
        assert any(i.kind == "feature_drift" for i in result.issues)

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_bundle = replace(bundle, manifest=replace(bundle.manifest, manifest_id="deadbeef00000000"))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered_bundle)
        assert FeatureInfrastructureReconciliationResult.from_json_dict(result.to_json_dict()) == result
