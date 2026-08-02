from __future__ import annotations

from dataclasses import replace

from quant_platform.qualification.diagnostics import QualificationDiagnostics, compute_diagnostics
from quant_platform.qualification.engine import DatasetQualificationEngine


class TestComputeDiagnostics:
    def test_produces_one_entry_per_split(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store)
        assert {s.split_name for s in diagnostics.split_diagnostics} == {"train", "validation", "test"}
        assert all(s.row_count > 0 for s in diagnostics.split_diagnostics)
        assert all(s.feature_null_fractions.get("trend") == 0.0 for s in diagnostics.split_diagnostics)

    def test_dimension_scores_match_the_source_report(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store)
        assert diagnostics.dimension_scores == {r.dimension.value: r.score for r in report.dimension_results}
        assert diagnostics.overall_score == report.decision.overall_score
        assert diagnostics.decision == report.decision.decision.value

    def test_json_round_trip(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store)
        restored = QualificationDiagnostics.from_json_dict(diagnostics.to_json_dict())
        assert restored == diagnostics

    def test_unreadable_artifacts_falls_back_to_empty_split_diagnostics(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        bogus_manifest = replace(qualified_manifest, dataset_id="0" * 16, content_id="0" * 64)
        diagnostics = compute_diagnostics(bogus_manifest, report, research_store)
        assert diagnostics.split_diagnostics == ()
