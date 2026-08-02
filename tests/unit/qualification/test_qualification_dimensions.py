from __future__ import annotations

from dataclasses import replace

import numpy as np

from quant_platform.qualification.dimensions import (
    evaluate_coverage,
    evaluate_determinism,
    evaluate_reproducibility,
    evaluate_safety,
    evaluate_stability,
    evaluate_statistical_integrity,
    evaluate_structural_integrity,
    evaluate_temporal_integrity,
)
from quant_platform.qualification.models import BlockingFailureCode, QualificationDimensionKind
from quant_platform.qualification.verifier import QualificationVerifier


def _facts(manifest, research_store, **kwargs):
    return QualificationVerifier().verify(manifest, research_store, required_feature_names=kwargs.get("required_feature_names", frozenset({"trend"})))


class TestStructuralIntegrity:
    def test_clean_dataset_scores_perfectly_with_no_blocking(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_structural_integrity(qualified_manifest, facts.artifacts.splits, facts)
        assert result.score == 1.0
        assert result.blocking_failures == ()

    def test_missing_required_feature_is_a_blocking_failure(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}))
        result = evaluate_structural_integrity(qualified_manifest, facts.artifacts.splits, facts)
        assert result.score == 0.0
        assert len(result.blocking_failures) == 1
        assert result.blocking_failures[0].code is BlockingFailureCode.REQUIRED_FEATURE_MISSING
        assert result.blocking_failures[0].dimension is QualificationDimensionKind.STRUCTURAL_INTEGRITY

    def test_unreadable_artifacts_is_manifest_corruption(self, qualified_manifest, research_store) -> None:
        bogus = replace(qualified_manifest, content_id="0" * 64)
        facts = _facts(bogus, research_store)
        result = evaluate_structural_integrity(bogus, facts.artifacts.splits, facts)
        assert result.score == 0.0
        assert result.blocking_failures[0].code is BlockingFailureCode.MANIFEST_CORRUPTION


class TestTemporalIntegrity:
    def test_clean_dataset_scores_perfectly(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_temporal_integrity(qualified_manifest, facts.artifacts.splits, facts)
        assert result.score == 1.0
        assert result.blocking_failures == ()

    def test_non_monotonic_open_time_is_a_warning_not_blocking(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        shuffled = splits["train"].sample(frac=1.0, random_state=0).reset_index(drop=True)
        splits["train"] = shuffled
        result = evaluate_temporal_integrity(qualified_manifest, splits, facts)
        assert result.blocking_failures == ()
        assert any("not monotonically increasing" in w for w in result.warnings)
        assert result.score < 1.0


class TestStatisticalIntegrity:
    def test_never_blocks(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_statistical_integrity(qualified_manifest, facts.artifacts.splits)
        assert result.blocking_failures == ()
        assert result.score == 1.0

    def test_no_train_split_scores_zero(self, qualified_manifest) -> None:
        result = evaluate_statistical_integrity(qualified_manifest, {})
        assert result.score == 0.0
        assert result.blocking_failures == ()


class TestCoverage:
    def test_never_blocks_and_scores_near_one_for_full_coverage(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_coverage(qualified_manifest, facts.artifacts.splits)
        assert result.blocking_failures == ()
        assert result.score > 0.9

    def test_missing_artifacts_scores_zero(self, qualified_manifest) -> None:
        result = evaluate_coverage(qualified_manifest, None)
        assert result.score == 0.0
        assert result.blocking_failures == ()


class TestStability:
    def test_never_blocks_and_detects_severe_shift_in_the_monotonic_trend_feature(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_stability(qualified_manifest, facts.artifacts.splits)
        assert result.blocking_failures == ()
        assert result.score < 0.5
        assert any("PSI" in w for w in result.warnings)


class TestDeterminism:
    def test_clean_dataset_scores_one_with_no_blocking(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_determinism(qualified_manifest, facts)
        assert result.score == 1.0
        assert result.blocking_failures == ()

    def test_checksum_mismatch_is_replay_mismatch(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        tampered_facts = replace(facts, artifacts=replace(facts.artifacts, metadata_checksums_match_manifest=False))
        result = evaluate_determinism(qualified_manifest, tampered_facts)
        assert result.score == 0.0
        assert result.blocking_failures[0].code is BlockingFailureCode.REPLAY_MISMATCH
        assert result.blocking_failures[0].dimension is QualificationDimensionKind.DETERMINISM


class TestReproducibility:
    def test_clean_dataset_scores_one_with_no_blocking(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_reproducibility(qualified_manifest, facts)
        assert result.score == 1.0
        assert result.blocking_failures == ()

    def test_identity_mismatch_is_blocking(self, qualified_manifest, research_store) -> None:
        tampered = replace(qualified_manifest, dataset_id="0" * 16)
        facts = _facts(tampered, research_store)
        result = evaluate_reproducibility(tampered, facts)
        assert result.score == 0.0
        codes = {f.code for f in result.blocking_failures}
        assert BlockingFailureCode.IDENTITY_MISMATCH in codes

    def test_missing_lineage_is_blocking(self, qualified_manifest, research_store) -> None:
        stripped = replace(qualified_manifest, code_revision="")
        facts = _facts(stripped, research_store)
        result = evaluate_reproducibility(stripped, facts)
        assert result.score == 0.0
        codes = {f.code for f in result.blocking_failures}
        assert BlockingFailureCode.MISSING_LINEAGE in codes


class TestSafety:
    def test_clean_dataset_never_blocks_and_scores_perfectly(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        result = evaluate_safety(qualified_manifest, facts.artifacts.splits)
        assert result.blocking_failures == ()
        assert result.score == 1.0

    def test_reserved_prefix_column_is_flagged_as_a_warning(self, qualified_manifest, research_store) -> None:
        facts = _facts(qualified_manifest, research_store)
        splits = dict(facts.artifacts.splits or {})
        contaminated = splits["train"].copy()
        contaminated["label_leak"] = np.arange(len(contaminated), dtype="float64")
        splits["train"] = contaminated
        result = evaluate_safety(qualified_manifest, splits)
        assert result.blocking_failures == ()
        assert any("label_leak" in w for w in result.warnings)
        assert result.score < 1.0
