from __future__ import annotations

from quant_platform.feature_discovery.catalog import build_feature_infrastructure_bundle
from quant_platform.feature_discovery.diagnostics import compute_shared_discovery_facts
from quant_platform.feature_discovery.health import compute_feature_health
from quant_platform.feature_discovery.infra_reconciliation import FeatureInfrastructureReconciliation
from quant_platform.feature_discovery.infra_reports import (
    render_dependency_report,
    render_feature_catalog_report,
    render_feature_inventory_report,
    render_health_report,
    render_infrastructure_reconciliation_report,
    render_infrastructure_verification_report,
    render_metadata_report,
)
from quant_platform.feature_discovery.infra_verification import FeatureInfrastructureVerifier


class TestRenderFeatureCatalogReport:
    def test_includes_every_feature(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        text = render_feature_catalog_report(bundle.catalog)
        for entry in bundle.catalog.entries:
            assert entry.feature_name in text

    def test_deterministic_across_calls(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        assert render_feature_catalog_report(bundle.catalog) == render_feature_catalog_report(bundle.catalog)


class TestRenderFeatureInventoryReport:
    def test_includes_grouped_view(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        text = render_feature_inventory_report(bundle.inventory)
        assert "price" in text


class TestRenderDependencyReport:
    def test_includes_nodes_and_validity(self, graph_registry, graph_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(graph_registry, graph_manifest)
        text = render_dependency_report(bundle.graph)
        assert "trend" in text
        assert "is_valid: True" in text


class TestRenderHealthReport:
    def test_includes_every_feature_and_lineage_status(self, discovered_manifest, research_store) -> None:
        facts = compute_shared_discovery_facts(discovered_manifest, research_store)
        health_reports = tuple(compute_feature_health(name, facts) for name in discovered_manifest.feature_names)
        text = render_health_report(health_reports)
        for health in health_reports:
            assert health.feature_name in text


class TestRenderMetadataReport:
    def test_includes_every_field(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        text = render_metadata_report(bundle.snapshot.metadata)
        for entry in bundle.snapshot.metadata:
            assert entry.feature_id in text
            assert entry.deterministic_identity in text


class TestRenderInfrastructureVerificationReport:
    def test_includes_verified_status(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureVerifier().verify(bundle, discovered_registry, discovered_manifest)
        text = render_infrastructure_verification_report(result)
        assert f"verified: {result.verified}" in text


class TestRenderInfrastructureReconciliationReport:
    def test_includes_reconciled_status(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureReconciliation().reconcile(bundle, bundle)
        text = render_infrastructure_reconciliation_report(result)
        assert "reconciled=True" in text
