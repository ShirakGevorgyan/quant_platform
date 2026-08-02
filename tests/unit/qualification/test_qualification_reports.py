from __future__ import annotations

from quant_platform.qualification.diagnostics import compute_diagnostics
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.reconciliation import QualificationReconciliation
from quant_platform.qualification.reports import (
    render_blocking_failure_report,
    render_dataset_qualification_report,
    render_evidence_report,
    render_independent_verification_report,
    render_qualification_diagnostics,
    render_qualification_reconciliation,
    render_recommendation_report,
)
from quant_platform.qualification.verification import QualificationIndependentVerifier


class TestRenderDatasetQualificationReport:
    def test_includes_decision_and_every_dimension(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        text = render_dataset_qualification_report(report)
        assert report.dataset_id in text
        assert "approved_for_research" in text
        for dimension_result in report.dimension_results:
            assert dimension_result.dimension.value in text

    def test_deterministic_across_calls(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert render_dataset_qualification_report(report) == render_dataset_qualification_report(report)


class TestRenderQualificationDiagnostics:
    def test_includes_every_split(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store)
        text = render_qualification_diagnostics(diagnostics)
        assert "train" in text
        assert "validation" in text
        assert "test" in text

    def test_includes_deep_evidence_sections(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        text = render_qualification_diagnostics(diagnostics)
        for label in ("Structural diagnostics", "Temporal diagnostics", "Statistical diagnostics", "Coverage diagnostics", "Stability diagnostics", "Safety diagnostics"):
            assert label in text


class TestRenderEvidenceReport:
    def test_includes_every_evidence_finding(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        text = render_evidence_report(diagnostics)
        assert str(len(diagnostics.all_evidence)) in text
        for evidence in diagnostics.all_evidence:
            assert evidence.finding in text


class TestRenderIndependentVerificationReport:
    def test_includes_verified_status(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        result = QualificationIndependentVerifier().verify(report, qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        text = render_independent_verification_report(result)
        assert f"verified: {result.verified}" in text
        assert result.dataset_id in text


class TestRenderBlockingFailureReport:
    def test_includes_blocking_failure_count(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        text = render_blocking_failure_report(report)
        assert "1 blocking failure" in text
        assert "required_feature_missing" in text


class TestRenderRecommendationReport:
    def test_includes_every_recommendation(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        text = render_recommendation_report(report)
        for dimension_result in report.dimension_results:
            for recommendation in dimension_result.recommendations:
                assert recommendation in text


class TestRenderQualificationReconciliation:
    def test_includes_every_issue_kind(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        baseline = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        text = render_qualification_reconciliation(result)
        for issue in result.issues:
            assert issue.kind in text
            assert issue.message in text
