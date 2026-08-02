from __future__ import annotations

from dataclasses import replace

from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.verification import (
    FeatureDiscoveryVerificationResult,
    FeatureDiscoveryVerifier,
    verify_report_self_consistency,
)


class TestVerifyReportSelfConsistency:
    def test_unmodified_report_is_self_consistent(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        consistent, issues = verify_report_self_consistency(report)
        assert consistent is True
        assert issues == ()

    def test_tampered_dimension_score_is_caught(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        tampered = replace(report, dimension_scores={**report.dimension_scores, "coverage": 0.1234})
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("coverage" in i for i in issues)

    def test_tampered_summary_is_caught(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        tampered = replace(report, summary=replace(report.summary, blocked_count=99))
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("blocked_count" in i for i in issues)


class TestFeatureDiscoveryVerifier:
    def test_clean_report_verifies(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        result = FeatureDiscoveryVerifier().verify(report, discovered_manifest, research_store)
        assert result.verified is True
        assert result.self_consistent is True
        assert result.reconciliation.reconciled is True

    def test_tampered_report_fails_verification(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        tampered = replace(report, dimension_scores={**report.dimension_scores, "coverage": 0.0})
        result = FeatureDiscoveryVerifier().verify(tampered, discovered_manifest, research_store)
        assert result.verified is False
        assert result.self_consistent is False

    def test_report_for_a_different_dataset_id_fails_gracefully(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        mismatched = replace(report, dataset_id="f" * 16)
        result = FeatureDiscoveryVerifier().verify(mismatched, discovered_manifest, research_store)
        assert result.verified is False
        assert any(i.kind == "dataset_id_mismatch" for i in result.reconciliation.issues)

    def test_stale_feature_subset_fails_verification(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store, feature_names=frozenset({"trend"}))
        result = FeatureDiscoveryVerifier().verify(report, discovered_manifest, research_store, feature_names=None)
        assert result.verified is False

    def test_json_round_trip(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        result = FeatureDiscoveryVerifier().verify(report, discovered_manifest, research_store)
        assert FeatureDiscoveryVerificationResult.from_json_dict(result.to_json_dict()) == result
