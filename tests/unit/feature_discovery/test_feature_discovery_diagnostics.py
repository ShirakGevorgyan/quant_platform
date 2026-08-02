from __future__ import annotations

from dataclasses import replace

import numpy as np

from quant_platform.feature_discovery.diagnostics import (
    compute_feature_signal_diagnostics,
    compute_shared_discovery_facts,
    evaluate_availability,
    evaluate_coverage,
    evaluate_determinism,
    evaluate_drift_behaviour,
    evaluate_information_content,
    evaluate_leakage_safety,
    evaluate_redundancy,
    evaluate_regime_stability,
    evaluate_reproducibility,
    evaluate_temporal_stability,
)
from quant_platform.feature_discovery.evidence import BlockingFindingCode
from quant_platform.feature_discovery.statistics import compute_feature_statistics


def _facts(manifest, research_store):
    return compute_shared_discovery_facts(manifest, research_store)


class TestSharedDiscoveryFacts:
    def test_computed_correctly_for_a_clean_dataset(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        assert facts.artifacts_readable is True
        assert facts.identity_matches is True
        assert facts.lineage_present is True
        assert "const" in facts.redundancy_constant_features
        assert ("trend", "trend_copy") in facts.redundancy_exact_duplicate_pairs

    def test_unreadable_artifacts_produce_a_degraded_but_non_crashing_facts_bundle(self, discovered_manifest, research_store) -> None:
        bogus = replace(discovered_manifest, content_id="0" * 64)
        facts = _facts(bogus, research_store)
        assert facts.artifacts_readable is False
        assert facts.splits is None
        assert facts.validation_report is None


class TestInformationContent:
    def test_constant_feature_scores_lower_and_warns(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        stats = compute_feature_statistics(facts.splits["train"]["const"], feature_name="const")
        result = evaluate_information_content("const", stats)
        assert result.score < 1.0
        assert any("constant" in e.finding for e in result.evidence)
        assert result.is_blocking is False

    def test_healthy_feature_scores_perfectly(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        stats = compute_feature_statistics(facts.splits["train"]["trend"], feature_name="trend")
        result = evaluate_information_content("trend", stats)
        assert result.score == 1.0


class TestTemporalStability:
    def test_never_blocks(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_series = facts.splits["train"]["trend"]
        stats = compute_feature_statistics(train_series, feature_name="trend")
        result = evaluate_temporal_stability("trend", train_series, stats)
        assert result.is_blocking is False
        assert len(result.evidence) > 0


class TestRegimeStability:
    def test_never_blocks_and_documents_macro_scope_limitation(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_regime_stability("trend", facts.splits["train"]["trend"])
        assert result.is_blocking is False
        assert any("macro tightening/easing" in e.finding for e in result.evidence)


class TestDriftBehaviour:
    def test_detects_severe_drift_for_the_monotonic_trend_feature(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_drift_behaviour("trend", facts.splits)
        assert result.score < 1.0
        assert result.is_blocking is False
        assert any(e.severity.value == "WARNING" for e in result.evidence)

    def test_missing_feature_scores_zero_without_crashing(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_drift_behaviour("does_not_exist", facts.splits)
        assert result.score == 0.0


class TestRedundancy:
    def test_exact_duplicate_detected(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_redundancy("trend_copy", facts)
        assert any("identical" in e.finding for e in result.evidence)
        assert result.score < 1.0
        assert result.is_blocking is False

    def test_unrelated_feature_reports_no_redundancy(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_redundancy("const", facts)
        assert result.score == 1.0


class TestCoverage:
    def test_warmup_and_usable_rows_reported(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"]
        result = evaluate_coverage("trend", train_df["trend"], train_df["label"], compute_feature_statistics(train_df["trend"], feature_name="trend"))
        assert any("usable rows" in e.finding for e in result.evidence)
        assert result.is_blocking is False

    def test_missing_window_detected(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"]
        gapped = train_df["trend"].copy()
        gapped.iloc[100:120] = np.nan
        stats = compute_feature_statistics(gapped, feature_name="trend")
        result = evaluate_coverage("trend", gapped, train_df["label"], stats)
        assert any("missing windows" in e.finding for e in result.evidence)


class TestAvailability:
    def test_never_blocks(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_availability("trend", facts.splits["train"]["trend"], discovered_manifest)
        assert result.is_blocking is False

    def test_no_market_data_lineage_reports_info(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_availability("trend", facts.splits["train"]["trend"], discovered_manifest)
        assert any("market_data_bridge" in e.finding for e in result.evidence)


class TestLeakageSafety:
    def test_clean_feature_is_not_blocking(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_leakage_safety("trend", facts, facts.splits["train"]["trend"])
        assert result.is_blocking is False
        assert result.score == 1.0

    def test_label_identical_feature_is_blocking_with_leakage_code(self, discovered_manifest, research_store) -> None:
        from quant_platform.features.validation import validate_research_dataset

        facts = _facts(discovered_manifest, research_store)
        contaminated_train = facts.splits["train"].copy()
        contaminated_train["trend"] = contaminated_train["label"].to_numpy(dtype="float64")
        contaminated_splits = {**facts.splits, "train": contaminated_train}
        contaminated_validation_report = validate_research_dataset(
            contaminated_train[["trend", "const", "trend_copy"]], timestamps=contaminated_train["open_time"], labels=contaminated_train["label"],
        )
        contaminated_facts = replace(facts, splits=contaminated_splits, validation_report=contaminated_validation_report)
        result = evaluate_leakage_safety("trend", contaminated_facts, contaminated_train["trend"])
        assert result.is_blocking is True
        assert result.score == 0.0
        codes = {e.blocking_code for e in result.blocking_evidence}
        assert BlockingFindingCode.LEAKAGE in codes

    def test_availability_violation_detected_for_out_of_range_observation(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        # Shift one row's open_time to fall outside the manifest's declared range.
        train_df.loc[train_df.index[0], "open_time"] = discovered_manifest.utc_end + train_df["open_time"].diff().median()
        tampered_splits = {**facts.splits, "train": train_df}
        tampered_facts = replace(facts, splits=tampered_splits)
        result = evaluate_leakage_safety("trend", tampered_facts, train_df["trend"])
        assert result.is_blocking is True
        codes = {e.blocking_code for e in result.blocking_evidence}
        assert BlockingFindingCode.AVAILABILITY_VIOLATION in codes


class TestDeterminism:
    def test_clean_dataset_scores_one_with_no_blocking(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_determinism("trend", facts)
        assert result.score == 1.0
        assert result.is_blocking is False

    def test_checksum_mismatch_is_blocking(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        tampered_facts = replace(facts, metadata_checksums_match_manifest=False)
        result = evaluate_determinism("trend", tampered_facts)
        assert result.is_blocking is True
        codes = {e.blocking_code for e in result.blocking_evidence}
        assert BlockingFindingCode.NON_DETERMINISTIC_FEATURE in codes


class TestReproducibility:
    def test_clean_dataset_scores_one_with_no_blocking(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        result = evaluate_reproducibility("trend", facts)
        assert result.score == 1.0
        assert result.is_blocking is False

    def test_identity_mismatch_is_blocking(self, discovered_manifest, research_store) -> None:
        tampered = replace(discovered_manifest, dataset_id="0" * 16)
        facts = _facts(tampered, research_store)
        result = evaluate_reproducibility("trend", facts)
        assert result.is_blocking is True
        codes = {e.blocking_code for e in result.blocking_evidence}
        assert BlockingFindingCode.IDENTITY_MISMATCH in codes

    def test_unreadable_artifacts_is_a_manifest_mismatch(self, discovered_manifest, research_store) -> None:
        bogus = replace(discovered_manifest, content_id="0" * 64)
        facts = _facts(bogus, research_store)
        result = evaluate_reproducibility("trend", facts)
        assert result.is_blocking is True
        codes = {e.blocking_code for e in result.blocking_evidence}
        assert BlockingFindingCode.MANIFEST_MISMATCH in codes


class TestComputeFeatureSignalDiagnostics:
    def test_full_ten_dimension_bundle_for_a_clean_feature(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        diagnostics = compute_feature_signal_diagnostics("trend", facts)
        assert len(diagnostics.dimension_results) == 10
        assert diagnostics.is_blocking is False
        assert 0.0 <= diagnostics.overall_score <= 1.0
