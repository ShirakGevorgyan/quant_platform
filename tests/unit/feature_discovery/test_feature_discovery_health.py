from __future__ import annotations

from dataclasses import replace

from quant_platform.feature_discovery.catalog import build_feature_catalog
from quant_platform.feature_discovery.diagnostics import compute_shared_discovery_facts
from quant_platform.feature_discovery.graph import build_feature_dependency_graph
from quant_platform.feature_discovery.health import (
    FeatureGroupReport,
    FeatureHealthReport,
    FeatureUsageReport,
    compute_feature_group_reports,
    compute_feature_health,
    compute_feature_usage,
)
from quant_platform.feature_discovery.registry_snapshot import capture_feature_registry_snapshot


class TestComputeFeatureHealth:
    def test_reports_lineage_present_for_a_clean_dataset(self, discovered_manifest, research_store) -> None:
        facts = compute_shared_discovery_facts(discovered_manifest, research_store)
        health = compute_feature_health("trend", facts)
        assert health.lineage_present is True
        assert health.missing_lineage_fields == ()

    def test_unhealthy_when_missing_lineage(self, discovered_manifest, research_store) -> None:
        tampered = replace(discovered_manifest, code_revision="")
        facts = compute_shared_discovery_facts(tampered, research_store)
        health = compute_feature_health("trend", facts)
        assert health.lineage_present is False
        assert health.is_healthy is False

    def test_json_round_trip(self, discovered_manifest, research_store) -> None:
        facts = compute_shared_discovery_facts(discovered_manifest, research_store)
        health = compute_feature_health("trend", facts)
        assert FeatureHealthReport.from_json_dict(health.to_json_dict()) == health


class TestComputeFeatureUsage:
    def test_root_and_leaf_classification(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        graph = build_feature_dependency_graph(snapshot)
        trend_usage = compute_feature_usage("trend", graph)
        double_usage = compute_feature_usage("trend_double", graph)
        assert trend_usage.is_root is True
        assert trend_usage.depended_on_by == ("trend_double",)
        assert double_usage.depends_on == ("trend",)
        assert double_usage.is_leaf is True

    def test_json_round_trip(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        graph = build_feature_dependency_graph(snapshot)
        usage = compute_feature_usage("trend", graph)
        assert FeatureUsageReport.from_json_dict(usage.to_json_dict()) == usage


class TestComputeFeatureGroupReports:
    def test_groups_by_feature_group_with_healthy_counts(self, discovered_registry, discovered_manifest, research_store) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        facts = compute_shared_discovery_facts(discovered_manifest, research_store)
        health_reports = tuple(compute_feature_health(name, facts) for name in discovered_manifest.feature_names)
        group_reports = compute_feature_group_reports(catalog, health_reports)
        assert len(group_reports) == 1
        assert group_reports[0].feature_group == "price"
        assert set(group_reports[0].feature_names) == {"trend", "const", "trend_copy"}
        assert group_reports[0].healthy_count + group_reports[0].unhealthy_count == 3

    def test_json_round_trip(self, discovered_registry, discovered_manifest, research_store) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        facts = compute_shared_discovery_facts(discovered_manifest, research_store)
        health_reports = tuple(compute_feature_health(name, facts) for name in discovered_manifest.feature_names)
        group_reports = compute_feature_group_reports(catalog, health_reports)
        assert FeatureGroupReport.from_json_dict(group_reports[0].to_json_dict()) == group_reports[0]
