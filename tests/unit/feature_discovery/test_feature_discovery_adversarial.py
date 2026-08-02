"""Milestone 11, Phase 2, Part 1: the 12 named adversarial test
scenarios, each run against the real `compute_shared_discovery_facts`/
`compute_feature_signal_diagnostics`/`FeatureDiscoveryEngine` pipeline
-- never a mock."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from quant_platform.feature_discovery.diagnostics import (
    compute_feature_signal_diagnostics,
    compute_shared_discovery_facts,
)
from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.evidence import BlockingFindingCode, FeatureDiscoveryDimensionKind
from quant_platform.feature_discovery.statistics import compute_feature_statistics
from quant_platform.features.validation import validate_research_dataset


def _facts(manifest, research_store):
    return compute_shared_discovery_facts(manifest, research_store)


class TestConstantFeatures:
    def test_constant_feature_is_flagged_in_information_content(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        diagnostics = compute_feature_signal_diagnostics("const", facts)
        info_content = diagnostics.dimension_result(diagnostics.dimension_results[0].dimension)
        assert any("constant" in e.finding for e in info_content.evidence)
        assert diagnostics.overall_score < 1.0
        assert diagnostics.is_blocking is False


class TestDuplicateFeatures:
    def test_exact_duplicate_is_flagged_in_redundancy_for_both_features(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        trend_diag = compute_feature_signal_diagnostics("trend", facts)
        copy_diag = compute_feature_signal_diagnostics("trend_copy", facts)
        assert any("identical" in e.finding for e in trend_diag.dimension_result(FeatureDiscoveryDimensionKind.REDUNDANCY).evidence)
        assert any("identical" in e.finding for e in copy_diag.dimension_result(FeatureDiscoveryDimensionKind.REDUNDANCY).evidence)
        assert trend_diag.is_blocking is False and copy_diag.is_blocking is False


class TestMissingValues:
    def test_high_missingness_lowers_information_content_score(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        sparse = train_df["trend"].copy()
        sparse.iloc[: int(len(sparse) * 0.6)] = np.nan
        splits = {**facts.splits, "train": train_df.assign(trend=sparse)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        assert diagnostics.dimension_result(FeatureDiscoveryDimensionKind.INFORMATION_CONTENT).score < 1.0
        assert diagnostics.is_blocking is False


class TestNaNInjection:
    def test_injected_nan_shows_up_as_missing_ratio_without_blocking(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        nan_series = train_df["trend"].copy()
        nan_series.iloc[0:10] = np.nan
        splits = {**facts.splits, "train": train_df.assign(trend=nan_series)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        assert diagnostics.is_blocking is False
        stats = compute_feature_statistics(nan_series, feature_name="trend")
        assert stats.missing_ratio > 0.0


class TestInfinityInjection:
    def test_injected_infinity_is_detected_and_does_not_poison_other_statistics(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        inf_series = train_df["trend"].copy().astype("float64")
        inf_series.iloc[0] = np.inf
        inf_series.iloc[1] = -np.inf
        splits = {**facts.splits, "train": train_df.assign(trend=inf_series)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        info_content = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.INFORMATION_CONTENT)
        assert any("non-finite" in e.finding and e.severity.value == "CRITICAL" for e in info_content.evidence)
        assert diagnostics.is_blocking is False
        # every OTHER dimension must still have computed real (finite) scores, not crashed or produced NaN
        for result in diagnostics.dimension_results:
            assert result.score == result.score  # not NaN
            assert 0.0 <= result.score <= 1.0

    def test_injected_infinity_never_poisons_diagnostics_with_a_runtimewarning(self, discovered_manifest, research_store) -> None:
        """Regression test for a real defect found during this audit:
        `evaluate_regime_stability`/`evaluate_drift_behaviour`/
        `evaluate_temporal_stability` and the shared redundancy facts all
        originally fed raw (inf-containing) series/frames directly into
        rolling-window and `features.drift.population_stability_index`/
        `compare_splits` computations, producing `inf - inf = NaN`
        `RuntimeWarning`s. Fixed via `diagnostics.py`'s `_finite_series`
        helper (and an equivalent inline mask for the shared redundancy
        DataFrame), applied at every one of those call sites."""
        import warnings

        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        inf_series = train_df["trend"].copy().astype("float64")
        inf_series.iloc[0] = np.inf
        inf_series.iloc[50] = -np.inf
        splits = {**facts.splits, "train": train_df.assign(trend=inf_series)}
        tampered_facts = replace(facts, splits=splits)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_feature_signal_diagnostics("trend", tampered_facts)


class TestFeatureDrift:
    def test_monotonic_trend_feature_shows_severe_drift_between_train_and_eval_splits(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        diagnostics = compute_feature_signal_diagnostics("trend", facts)
        drift = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.DRIFT_BEHAVIOUR)
        assert drift.score < 1.0
        assert any(e.severity.value == "WARNING" for e in drift.evidence)


class TestCoverageLoss:
    def test_severe_coverage_loss_lowers_coverage_score(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        sparse = train_df["trend"].copy()
        sparse.iloc[: int(len(sparse) * 0.7)] = np.nan
        splits = {**facts.splits, "train": train_df.assign(trend=sparse)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        coverage = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.COVERAGE)
        assert coverage.score < 1.0
        assert any("coverage" in e.finding for e in coverage.evidence)


class TestWarmup:
    def test_leading_null_run_is_reported_as_warmup(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        warmed_up = train_df["trend"].copy()
        warmed_up.iloc[:30] = np.nan
        splits = {**facts.splits, "train": train_df.assign(trend=warmed_up)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        coverage = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.COVERAGE)
        warmup_evidence = [e for e in coverage.evidence if "warmup" in e.finding]
        assert warmup_evidence
        assert warmup_evidence[0].supporting_statistics["warmup_rows"] == 30.0


class TestAvailability:
    def test_no_market_data_lineage_is_reported_as_informational_not_a_penalty(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        diagnostics = compute_feature_signal_diagnostics("trend", facts)
        availability = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.AVAILABILITY)
        assert availability.score == 1.0
        assert availability.is_blocking is False

    def test_large_dark_period_lowers_availability_score(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        dark = train_df["trend"].copy()
        dark.iloc[100:400] = np.nan
        splits = {**facts.splits, "train": train_df.assign(trend=dark)}
        tampered_facts = replace(facts, splits=splits)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        availability = diagnostics.dimension_result(FeatureDiscoveryDimensionKind.AVAILABILITY)
        assert availability.score < 1.0


class TestLeakage:
    def test_feature_identical_to_label_is_a_blocking_leakage_finding(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        train_df = facts.splits["train"].copy()
        train_df["trend"] = train_df["label"].to_numpy(dtype="float64")
        splits = {**facts.splits, "train": train_df}
        validation_report = validate_research_dataset(
            train_df[["trend", "const", "trend_copy"]], timestamps=train_df["open_time"], labels=train_df["label"],
        )
        tampered_facts = replace(facts, splits=splits, validation_report=validation_report)
        diagnostics = compute_feature_signal_diagnostics("trend", tampered_facts)
        assert diagnostics.is_blocking is True
        codes = {e.blocking_code for e in diagnostics.blocking_evidence}
        assert BlockingFindingCode.LEAKAGE in codes


class TestDeterminism:
    def test_content_store_checksum_mismatch_blocks_every_feature(self, discovered_manifest, research_store) -> None:
        facts = _facts(discovered_manifest, research_store)
        tampered_facts = replace(facts, metadata_checksums_match_manifest=False)
        for feature_name in ("trend", "const", "trend_copy"):
            diagnostics = compute_feature_signal_diagnostics(feature_name, tampered_facts)
            assert diagnostics.is_blocking is True

    def test_repeated_engine_runs_produce_identical_reports(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        report1 = engine.discover(discovered_manifest, research_store)
        report2 = engine.discover(discovered_manifest, research_store)
        raw1, raw2 = report1.to_json_dict(), report2.to_json_dict()
        raw1.pop("evaluation_time"), raw2.pop("evaluation_time")
        assert raw1 == raw2


class TestReproducibility:
    def test_tampered_dataset_id_blocks_with_identity_mismatch(self, discovered_manifest, research_store) -> None:
        tampered = replace(discovered_manifest, dataset_id="0" * 16)
        facts = _facts(tampered, research_store)
        diagnostics = compute_feature_signal_diagnostics("trend", facts)
        assert diagnostics.is_blocking is True
        codes = {e.blocking_code for e in diagnostics.blocking_evidence}
        assert BlockingFindingCode.IDENTITY_MISMATCH in codes

    def test_unreadable_content_directory_blocks_with_manifest_mismatch(self, discovered_manifest, research_store) -> None:
        bogus = replace(discovered_manifest, content_id="0" * 64)
        facts = _facts(bogus, research_store)
        diagnostics = compute_feature_signal_diagnostics("trend", facts)
        assert diagnostics.is_blocking is True
        codes = {e.blocking_code for e in diagnostics.blocking_evidence}
        assert BlockingFindingCode.MANIFEST_MISMATCH in codes
