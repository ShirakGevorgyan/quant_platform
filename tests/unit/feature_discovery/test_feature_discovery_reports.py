from __future__ import annotations

from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.reconciliation import FeatureDiscoveryReconciliation
from quant_platform.feature_discovery.reports import (
    render_blocking_findings_report,
    render_feature_discovery_reconciliation,
    render_feature_discovery_report,
    render_feature_discovery_verification,
    render_feature_signal_diagnostics,
    render_recommendations_report,
)
from quant_platform.feature_discovery.verification import FeatureDiscoveryVerifier


class TestRenderFeatureDiscoveryReport:
    def test_includes_dataset_id_and_every_feature(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        text = render_feature_discovery_report(report)
        assert report.dataset_id in text
        for diagnostics in report.per_feature_diagnostics:
            assert diagnostics.feature_name in text

    def test_deterministic_across_calls(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert render_feature_discovery_report(report) == render_feature_discovery_report(report)


class TestRenderFeatureSignalDiagnostics:
    def test_includes_every_dimension(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        text = render_feature_signal_diagnostics(report.feature_diagnostics("const"))
        assert "information_content" in text
        assert "leakage_safety" in text


class TestRenderBlockingFindingsReport:
    def test_reports_zero_findings_for_a_clean_dataset(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        text = render_blocking_findings_report(report)
        assert "0 finding(s)" in text


class TestRenderRecommendationsReport:
    def test_includes_every_recommendation(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        text = render_recommendations_report(report)
        for recommendation in report.recommendations:
            assert recommendation in text


class TestRenderFeatureDiscoveryReconciliation:
    def test_includes_every_issue(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        full = engine.discover(discovered_manifest, research_store)
        subset = engine.discover(discovered_manifest, research_store, feature_names=frozenset({"trend"}))
        result = FeatureDiscoveryReconciliation().reconcile(full, subset)
        text = render_feature_discovery_reconciliation(result)
        for issue in result.issues:
            assert issue.kind in text


class TestRenderFeatureDiscoveryVerification:
    def test_includes_verified_status(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        result = FeatureDiscoveryVerifier().verify(report, discovered_manifest, research_store)
        text = render_feature_discovery_verification(result)
        assert f"verified: {result.verified}" in text
