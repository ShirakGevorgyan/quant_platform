from __future__ import annotations

from dataclasses import replace

from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.models import QualificationDecisionKind
from quant_platform.qualification.verification import (
    IndependentVerificationResult,
    QualificationIndependentVerifier,
    verify_report_self_consistency,
)


class TestVerifyReportSelfConsistency:
    def test_unmodified_report_is_self_consistent(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        consistent, issues = verify_report_self_consistency(report)
        assert consistent is True
        assert issues == ()

    def test_tampered_overall_score_is_caught(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        tampered = replace(report, decision=replace(report.decision, overall_score=0.1234))
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("overall_score" in i for i in issues)

    def test_tampered_decision_is_caught(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        tampered = replace(report, decision=replace(report.decision, decision=QualificationDecisionKind.REJECTED_FOR_RESEARCH))
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("decision" in i for i in issues)

    def test_tampered_blocking_failure_count_is_caught(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        tampered = replace(report, decision=replace(report.decision, blocking_failure_count=99))
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("blocking_failure_count" in i for i in issues)


class TestQualificationIndependentVerifier:
    def test_clean_unmodified_report_verifies(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        result = QualificationIndependentVerifier().verify(report, qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert result.verified is True
        assert result.self_consistent is True
        assert result.reconciliation.reconciled is True

    def test_tampered_report_fails_verification(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        tampered = replace(report, decision=replace(report.decision, overall_score=0.0))
        result = QualificationIndependentVerifier().verify(tampered, qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert result.verified is False
        assert result.self_consistent is False

    def test_report_for_a_different_dataset_id_fails_gracefully(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        mismatched = replace(report, dataset_id="f" * 16)
        result = QualificationIndependentVerifier().verify(mismatched, qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert result.verified is False
        assert any(issue.kind == "dataset_id_mismatch" for issue in result.reconciliation.issues)

    def test_stale_requirement_set_fails_verification(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        result = QualificationIndependentVerifier().verify(
            report, qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}),
        )
        assert result.verified is False

    def test_json_round_trip(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        result = QualificationIndependentVerifier().verify(report, qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        restored = IndependentVerificationResult.from_json_dict(result.to_json_dict())
        assert restored == result
