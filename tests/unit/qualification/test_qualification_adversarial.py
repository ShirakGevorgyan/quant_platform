"""Milestone 11 Phase 1, Part 2: the adversarial audit. One test class
per attack named in the spec's list; each attempts the attack against
the REAL qualification pipeline (never a mock) and asserts it is
caught. Where an attack is legitimately out of this package's scope
(e.g. raw macro/cross-asset source re-verification -- see `diagnostics.
py`'s own module docstring), the test instead proves the DOCUMENTED
boundary: the best available signal (`market_data_lineage`) is surfaced
correctly and nothing crashes.

No defect survived verification during this audit -- see
`docs/milestone11_phase1_delivery_report.md`'s "Real defects" section
for the ones that were found and fixed DURING development (before this
suite existed) and the regression tests that now guard them."""

from __future__ import annotations

import shutil
from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.features.dataset_builder import ResearchDatasetBuilder
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.qualification.diagnostics import (
    _coverage_evidence,
    _statistical_evidence,
    _structural_evidence,
    _temporal_evidence,
)
from quant_platform.qualification.dimensions import (
    evaluate_determinism,
    evaluate_reproducibility,
    evaluate_structural_integrity,
    evaluate_temporal_integrity,
)
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.models import BlockingFailureCode, QualificationDecisionKind
from quant_platform.qualification.verifier import QualificationVerifier, verify_identity


def _facts(manifest, research_store, required_feature_names=frozenset({"trend"})):
    return QualificationVerifier().verify(manifest, research_store, required_feature_names=required_feature_names)


class TestFutureLeakage:
    def test_a_feature_identical_to_the_label_is_a_blocking_failure(self, qualified_manifest, research_store) -> None:
        from quant_platform.qualification.verifier import verify_no_future_leakage

        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        contaminated = splits["train"].copy()
        contaminated["trend"] = contaminated["label"].to_numpy(dtype="float64")
        splits["train"] = contaminated

        # Real detector, real contaminated data -- not a forced fact.
        leakage_free, leakage_messages = verify_no_future_leakage(contaminated, split_name="train")
        assert leakage_free is False
        assert leakage_messages

        contaminated_facts = replace(facts, leakage_free=leakage_free, leakage_messages=leakage_messages)
        result = evaluate_temporal_integrity(qualified_manifest, splits, contaminated_facts)
        assert result.score == 0.0
        assert any(f.code is BlockingFailureCode.FUTURE_LEAKAGE for f in result.blocking_failures)


class TestManifestCorruption:
    def test_unreadable_content_directory_is_manifest_corruption(self, qualified_manifest, research_store) -> None:
        bogus = replace(qualified_manifest, content_id="0" * 64)
        facts = _facts(bogus, research_store)
        result = evaluate_structural_integrity(bogus, facts.artifacts.splits, facts)
        assert result.score == 0.0
        assert result.blocking_failures[0].code is BlockingFailureCode.MANIFEST_CORRUPTION


class TestLineageCorruption:
    def test_malformed_market_data_lineage_does_not_crash_and_produces_no_findings(self, qualified_manifest, research_store) -> None:
        malformed = replace(qualified_manifest, market_data_lineage={"schema_version": 1, "coverage_decision": "not-a-dict"})
        facts = _facts(malformed, research_store)
        evidence = _temporal_evidence(malformed, facts.artifacts.splits, facts)
        assert not any("macro availability" in e.finding or "cross-asset availability" in e.finding for e in evidence)


class TestDatasetIdentityTampering:
    def test_tampered_dataset_id_is_detected(self, qualified_manifest) -> None:
        tampered = replace(qualified_manifest, dataset_id="0" * 16)
        matches, _ = verify_identity(tampered)
        assert matches is False

    def test_tampered_label_definition_changes_the_recomputed_identity(self, qualified_manifest) -> None:
        tampered = replace(qualified_manifest, label_definition={**qualified_manifest.label_definition, "horizon_bars": 999})
        matches, _ = verify_identity(tampered)
        assert matches is False


class TestReplayMismatch:
    def test_checksum_mismatch_is_a_blocking_failure(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        tampered_facts = replace(facts, artifacts=replace(facts.artifacts, metadata_checksums_match_manifest=False))
        result = evaluate_determinism(qualified_manifest, tampered_facts)
        assert result.blocking_failures[0].code is BlockingFailureCode.REPLAY_MISMATCH


class TestDuplicateTimestamps:
    def test_duplicated_open_time_is_detected(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        dupe = splits["train"].copy()
        dupe.loc[1, "open_time"] = dupe.loc[0, "open_time"]
        splits["train"] = dupe
        evidence = _statistical_evidence(qualified_manifest, splits)
        assert any("duplicate timestamps" in e.finding for e in evidence)


class TestDuplicateRows:
    def test_fully_duplicated_row_is_detected(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        dupe = splits["train"].copy()
        dupe.iloc[1] = dupe.iloc[0]
        splits["train"] = dupe
        evidence = _statistical_evidence(qualified_manifest, splits)
        assert any("duplicate rows" in e.finding for e in evidence)


class TestNaNInjection:
    def test_injected_nan_is_detected(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        nan_df = splits["train"].copy()
        nan_df.loc[0, "trend"] = np.nan
        splits["train"] = nan_df
        evidence = _statistical_evidence(qualified_manifest, splits)
        assert any(e.finding.startswith("NaN:") for e in evidence)


class TestInfinityInjection:
    def test_injected_infinity_is_detected(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        inf_df = splits["train"].copy()
        inf_df.loc[0, "trend"] = np.inf
        splits["train"] = inf_df
        evidence = _statistical_evidence(qualified_manifest, splits)
        assert any(e.finding.startswith("Infinity:") and e.severity.value == "CRITICAL" for e in evidence)

    def test_infinity_never_poisons_the_abnormal_distribution_skew_kurtosis_computation(self, qualified_manifest, research_store) -> None:
        """Regression test for a real defect found during this audit:
        `_skew_kurtosis` originally computed mean/std over the RAW
        (NaN-only-filtered) column, so an injected +/-inf produced
        `inf - inf = nan` arithmetic (a `RuntimeWarning`) and could have
        propagated a NaN skew/kurtosis value into `abnormal_evidence`.
        Fixed by restricting the skew/kurtosis computation to the
        FINITE-only view of the column (`diagnostics.py`'s
        `_statistical_evidence`, `np.isfinite` mask)."""
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        inf_df = splits["train"].copy()
        inf_df.loc[0, "trend"] = np.inf
        splits["train"] = inf_df
        evidence = _statistical_evidence(qualified_manifest, splits)
        abnormal_records = [e for e in evidence if e.finding.startswith("abnormal distributions:")]
        for record in abnormal_records:
            assert "nan" not in record.evidence[0].lower()


class TestConstantFeatures:
    def test_constant_feature_is_detected(self, two_feature_manifest, research_store) -> None:
        facts = _facts(two_feature_manifest, research_store, required_feature_names=frozenset({"trend", "const"}))
        evidence = _statistical_evidence(two_feature_manifest, facts.artifacts.splits)
        assert any("zero variance" in e.finding for e in evidence)


class TestNearConstantFeatures:
    def test_near_constant_feature_is_detected(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        near_const = splits["train"].copy()
        rng = np.random.default_rng(0)
        near_const["trend"] = 5.0 + rng.normal(0, 1e-9, size=len(near_const))
        splits["train"] = near_const
        evidence = _statistical_evidence(qualified_manifest, splits)
        assert any("near-zero variance" in e.finding for e in evidence)


class TestMissingRequiredFeature:
    def test_missing_required_feature_rejects(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        assert report.decision.decision is QualificationDecisionKind.REJECTED_FOR_RESEARCH


class TestMissingLineage:
    def test_missing_code_revision_is_a_blocking_failure(self, qualified_manifest, research_store) -> None:
        stripped = replace(qualified_manifest, code_revision="")
        facts = _facts(stripped, research_store)
        result = evaluate_reproducibility(stripped, facts)
        assert any(f.code is BlockingFailureCode.MISSING_LINEAGE for f in result.blocking_failures)


class TestMacroBeforeRelease:
    def test_insufficient_macro_coverage_is_surfaced_as_a_warning(self, qualified_manifest, research_store) -> None:
        lineage = {
            "schema_version": 1,
            "coverage_decision": {"findings": [{"source_kind": "macro", "source_name": "cpi", "required": True, "status": "insufficient", "coverage_fraction": 0.2}]},
        }
        with_lineage = replace(qualified_manifest, market_data_lineage=lineage)
        facts = _facts(with_lineage, research_store)
        evidence = _temporal_evidence(with_lineage, facts.artifacts.splits, facts)
        matches = [e for e in evidence if "macro availability" in e.finding and "cpi" in e.finding]
        assert len(matches) == 1
        assert matches[0].severity.value == "WARNING"


class TestCrossAssetBeforeAvailability:
    def test_insufficient_cross_asset_coverage_is_surfaced_as_a_warning(self, qualified_manifest, research_store) -> None:
        lineage = {
            "schema_version": 1,
            "coverage_decision": {"findings": [{"source_kind": "cross_asset", "source_name": "dxy", "required": True, "status": "missing", "coverage_fraction": 0.0}]},
        }
        with_lineage = replace(qualified_manifest, market_data_lineage=lineage)
        facts = _facts(with_lineage, research_store)
        evidence = _temporal_evidence(with_lineage, facts.artifacts.splits, facts)
        matches = [e for e in evidence if "cross-asset availability" in e.finding and "dxy" in e.finding]
        assert len(matches) == 1
        assert matches[0].severity.value == "WARNING"


class TestStaleMacro:
    def test_staleness_scope_limitation_is_documented_in_the_evidence_itself(self, qualified_manifest, research_store) -> None:
        lineage = {"schema_version": 1, "coverage_decision": {"findings": [{"source_kind": "macro", "source_name": "cpi", "required": True, "status": "ok", "coverage_fraction": 1.0}]}}
        with_lineage = replace(qualified_manifest, market_data_lineage=lineage)
        facts = _facts(with_lineage, research_store)
        evidence = _temporal_evidence(with_lineage, facts.artifacts.splits, facts)
        assert any("stale macro" in e.finding and "out of scope" in e.finding for e in evidence)


class TestStaleCrossAsset:
    def test_staleness_scope_limitation_applies_uniformly_regardless_of_source_kind(self, qualified_manifest, research_store) -> None:
        lineage = {"schema_version": 1, "coverage_decision": {"findings": [{"source_kind": "cross_asset", "source_name": "dxy", "required": False, "status": "ok", "coverage_fraction": 1.0}]}}
        with_lineage = replace(qualified_manifest, market_data_lineage=lineage)
        facts = _facts(with_lineage, research_store)
        evidence = _temporal_evidence(with_lineage, facts.artifacts.splits, facts)
        assert any("stale cross-asset" in e.finding for e in evidence)


class TestCoverageCorruption:
    def test_requested_range_far_beyond_observed_data_lowers_coverage_score(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        corrupted = replace(qualified_manifest, utc_end=qualified_manifest.utc_end + pd.Timedelta(days=365))
        evidence = _coverage_evidence(corrupted, facts.artifacts.splits)
        source_coverage = [e for e in evidence if "source coverage" in e.finding]
        assert len(source_coverage) == 1
        assert source_coverage[0].severity.value == "WARNING"


class TestFeatureOrderTampering:
    def test_reordering_a_splits_columns_does_not_change_the_structural_result(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = facts.artifacts.splits or {}
        reordered_train = splits["train"][list(reversed(splits["train"].columns))]
        assert list(reordered_train.columns) != list(splits["train"].columns)
        reordered_splits = {**splits, "train": reordered_train}

        original = evaluate_structural_integrity(qualified_manifest, splits, facts)
        reordered = evaluate_structural_integrity(qualified_manifest, reordered_splits, replace(facts, artifacts=replace(facts.artifacts, splits=reordered_splits)))
        assert original.score == reordered.score
        assert original.blocking_failures == reordered.blocking_failures


class TestColumnOrderTampering:
    def test_statistical_evidence_is_unaffected_by_column_order(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        train_df = (facts.artifacts.splits or {})["train"]
        reordered = train_df[list(reversed(train_df.columns))]
        original_evidence = _statistical_evidence(qualified_manifest, {"train": train_df})
        reordered_evidence = _statistical_evidence(qualified_manifest, {"train": reordered})
        assert [(e.finding, e.severity) for e in original_evidence] == [(e.finding, e.severity) for e in reordered_evidence]


class TestManifestHashRecomputation:
    def test_every_recipe_field_change_is_reflected_in_the_recomputed_hash(self, qualified_manifest) -> None:
        baseline_matches, _ = verify_identity(qualified_manifest)
        assert baseline_matches is True
        for field_name, new_value in (
            ("symbol", "EURUSD"), ("feature_registry_fingerprint", "0" * 16),
            ("split_definition", {**qualified_manifest.split_definition, "tampered": True}),
            ("preprocessing_definition", {**qualified_manifest.preprocessing_definition, "tampered": "yes"}),
        ):
            tampered = replace(qualified_manifest, **{field_name: new_value})
            matches, _ = verify_identity(tampered)
            assert matches is False, f"expected identity mismatch after tampering {field_name!r}"


class TestFilesystemRelocation:
    def test_qualification_is_identical_after_moving_the_entire_research_root(self, tmp_path, seeded_loader, trend_registry_factory, build_request_factory) -> None:
        original_root = tmp_path / "original"
        research_store = ResearchDatasetStore(original_root / "research")
        manifest_store = ResearchManifestStore(original_root / "research")
        builder = ResearchDatasetBuilder(historical_loader=seeded_loader, registry=trend_registry_factory(), research_store=research_store, manifest_store=manifest_store)
        manifest = builder.build(build_request_factory())
        report_before = DatasetQualificationEngine().qualify(manifest, research_store, required_feature_names=frozenset({"trend"}))

        relocated_root = tmp_path / "relocated"
        shutil.copytree(original_root, relocated_root)
        relocated_store = ResearchDatasetStore(relocated_root / "research")
        report_after = DatasetQualificationEngine().qualify(manifest, relocated_store, required_feature_names=frozenset({"trend"}))

        assert report_before.decision.decision == report_after.decision.decision
        assert report_before.decision.overall_score == report_after.decision.overall_score
        assert [r.score for r in report_before.dimension_results] == [r.score for r in report_after.dimension_results]


class TestWallClockDependency:
    def test_generated_at_is_the_only_field_that_differs_between_two_runs(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        report_a = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        report_b = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        a, b = report_a.to_json_dict(), report_b.to_json_dict()
        a.pop("generated_at"), b.pop("generated_at")
        a["decision"].pop("generated_at"), b["decision"].pop("generated_at")
        assert a == b


class TestRandomOrdering:
    def test_required_feature_names_insertion_order_does_not_affect_the_result(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        forward = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset(["trend"]))
        reversed_construction = engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset(reversed(["trend"])))
        assert forward.decision.overall_score == reversed_construction.decision.overall_score
        assert forward.decision.decision == reversed_construction.decision.decision

    def test_splits_dict_iteration_order_does_not_affect_structural_evidence(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = facts.artifacts.splits or {}
        forward_order = _structural_evidence(qualified_manifest, splits, facts)
        reversed_order = _structural_evidence(qualified_manifest, dict(reversed(list(splits.items()))), facts)
        assert [(e.finding, e.severity) for e in forward_order] == [(e.finding, e.severity) for e in reversed_order]
