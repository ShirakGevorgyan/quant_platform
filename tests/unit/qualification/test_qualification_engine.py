from __future__ import annotations

from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.models import (
    QUALIFICATION_DIMENSION_ORDER,
    DatasetQualificationReport,
    QualificationDecisionKind,
)


class TestDatasetQualificationEngine:
    def test_approves_a_clean_dataset(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert report.decision.decision is QualificationDecisionKind.APPROVED_FOR_RESEARCH
        assert report.decision.blocking_failure_count == 0
        assert len(report.dimension_results) == len(QUALIFICATION_DIMENSION_ORDER)
        assert tuple(r.dimension for r in report.dimension_results) == QUALIFICATION_DIMENSION_ORDER

    def test_rejects_when_a_required_feature_is_missing(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(
            qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}),
        )
        assert report.decision.decision is QualificationDecisionKind.REJECTED_FOR_RESEARCH
        assert report.decision.blocking_failure_count == 1
        assert "required_feature_missing" in report.decision.decision_reason

    def test_report_json_round_trip(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        restored = DatasetQualificationReport.from_json_dict(report.to_json_dict())
        assert restored == report

    def test_qualify_is_deterministic_across_repeated_runs(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        report1 = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        report2 = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert report1.decision.decision is report2.decision.decision
        assert report1.decision.overall_score == report2.decision.overall_score
        assert [r.score for r in report1.dimension_results] == [r.score for r in report2.dimension_results]
