from __future__ import annotations

from dataclasses import replace

from quant_platform.qualification.diagnostics import QualificationDiagnostics, compute_diagnostics
from quant_platform.qualification.engine import DatasetQualificationEngine


class TestStructuralEvidence:
    def test_clean_dataset_has_no_critical_structural_evidence(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert len(diagnostics.structural_evidence) > 0
        assert all(not e.blocking for e in diagnostics.structural_evidence)

    def test_unreadable_artifacts_short_circuits_to_one_critical_record(self, qualified_manifest, research_store) -> None:
        bogus = replace(qualified_manifest, content_id="0" * 64)
        report = DatasetQualificationEngine().qualify(bogus, research_store, required_feature_names=frozenset())
        diagnostics = compute_diagnostics(bogus, report, research_store)
        assert len(diagnostics.structural_evidence) == 1
        assert diagnostics.structural_evidence[0].blocking is True


class TestTemporalEvidence:
    def test_no_market_data_lineage_reports_info(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert any("market_data_bridge" in e.finding for e in diagnostics.temporal_evidence)

    def test_future_visibility_reports_clean_when_no_leakage(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert any("future visibility" in e.finding and "no suspected" in e.finding for e in diagnostics.temporal_evidence)


class TestStatisticalEvidence:
    def test_constant_feature_is_flagged_as_zero_variance(self, two_feature_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(two_feature_manifest, research_store, required_feature_names=frozenset({"trend", "const"}))
        diagnostics = compute_diagnostics(two_feature_manifest, report, research_store, required_feature_names=frozenset({"trend", "const"}))
        assert any("zero variance" in e.finding and "const" in e.evidence[0] for e in diagnostics.statistical_evidence)

    def test_clean_dataset_has_no_statistical_evidence(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert diagnostics.statistical_evidence == ()


class TestCoverageEvidence:
    def test_reports_source_coverage_near_full(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        source_coverage = [e for e in diagnostics.coverage_evidence if "source coverage" in e.finding]
        assert len(source_coverage) == 1
        assert "99." in source_coverage[0].finding or "100" in source_coverage[0].finding


class TestStabilityEvidence:
    def test_detects_severe_psi_drift_for_the_monotonic_trend_feature(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        psi_records = [e for e in diagnostics.stability_evidence if "PSI" in e.finding]
        assert len(psi_records) >= 2
        assert any(e.severity.value == "WARNING" for e in psi_records)

    def test_regime_drift_record_present(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert any("regime drift" in e.finding for e in diagnostics.stability_evidence)


class TestSafetyEvidence:
    def test_clean_dataset_has_no_safety_evidence(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics = compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"}))
        assert diagnostics.safety_evidence == ()

    def test_label_contamination_detected_when_a_feature_equals_the_label(self, qualified_manifest, research_store) -> None:
        # Directly exercise the safety-evidence helper's label-contamination path against a hand-built,
        # deliberately contaminated split (a feature column set to an exact copy of the label column) --
        # verifies the CHECK works without needing a real builder run that reproduces contamination.
        from quant_platform.qualification.diagnostics import _safety_evidence
        from quant_platform.qualification.verifier import QualificationVerifier

        facts = QualificationVerifier().verify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        splits = dict(facts.artifacts.splits or {})
        contaminated = splits["train"].copy()
        contaminated["trend"] = contaminated["label"].to_numpy(dtype="float64")
        splits["train"] = contaminated
        evidence = _safety_evidence(qualified_manifest, splits, facts)
        assert any("label contamination" in e.finding for e in evidence)


class TestComputeDiagnosticsPart2JsonRoundTrip:
    def test_json_round_trip_with_deep_evidence(self, two_feature_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(two_feature_manifest, research_store, required_feature_names=frozenset({"trend", "const"}))
        diagnostics = compute_diagnostics(two_feature_manifest, report, research_store, required_feature_names=frozenset({"trend", "const"}))
        assert len(diagnostics.all_evidence) > 0
        restored = QualificationDiagnostics.from_json_dict(diagnostics.to_json_dict())
        assert restored == diagnostics
