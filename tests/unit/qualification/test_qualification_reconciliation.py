from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import QualificationReconciliationError
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.models import DatasetQualificationReport, QualificationDimensionKind
from quant_platform.qualification.reconciliation import (
    QualificationReconciliation,
    QualificationReconciliationResult,
)


def _replace_dimension(report: DatasetQualificationReport, dimension: QualificationDimensionKind, **kwargs: object) -> DatasetQualificationReport:
    dimension_results = tuple(replace(r, **kwargs) if r.dimension is dimension else r for r in report.dimension_results)
    return replace(report, dimension_results=dimension_results)


class TestQualificationReconciliation:
    def test_identical_reruns_fully_reconcile(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        report_a = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        report_b = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        result = QualificationReconciliation().reconcile(report_a, report_b)
        assert result.reconciled is True
        assert result.issues == ()

    def test_diverged_candidate_surfaces_decision_and_score_issues(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        baseline = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        assert result.reconciled is False
        kinds = {issue.kind for issue in result.issues}
        assert "decision_mismatch" in kinds
        assert "dimension_score_drift" in kinds
        assert "blocking_failure_set_changed" in kinds

    def test_different_dataset_ids_raise(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        baseline = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        other = replace(baseline, dataset_id="f" * 16)
        with pytest.raises(QualificationReconciliationError):
            QualificationReconciliation().reconcile(baseline, other)

    def test_json_round_trip(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        baseline = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        restored = QualificationReconciliationResult.from_json_dict(result.to_json_dict())
        assert restored == result


class TestPart2DriftDetection:
    def test_finding_drift_detected_for_a_non_reproducibility_dimension(self, qualified_manifest, research_store) -> None:
        baseline = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = _replace_dimension(baseline, QualificationDimensionKind.SAFETY, findings=("a completely different finding",))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        assert result.reconciled is False
        assert any(i.kind == "finding_drift" and i.dimension == "safety" for i in result.issues)

    def test_warning_drift_detected(self, qualified_manifest, research_store) -> None:
        baseline = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = _replace_dimension(baseline, QualificationDimensionKind.SAFETY, warnings=("a new warning appeared",))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "warning_drift" and i.dimension == "safety" for i in result.issues)

    def test_recommendation_drift_detected(self, qualified_manifest, research_store) -> None:
        baseline = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = _replace_dimension(baseline, QualificationDimensionKind.SAFETY, recommendations=("do something about it",))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "recommendation_drift" and i.dimension == "safety" for i in result.issues)

    def test_reproducibility_finding_drift_is_reported_as_lineage_drift(self, qualified_manifest, research_store) -> None:
        baseline = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        candidate = _replace_dimension(baseline, QualificationDimensionKind.REPRODUCIBILITY, findings=("a different lineage finding",))
        result = QualificationReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "lineage_drift" and i.dimension == "reproducibility" for i in result.issues)
        assert not any(i.kind == "finding_drift" and i.dimension == "reproducibility" for i in result.issues)
