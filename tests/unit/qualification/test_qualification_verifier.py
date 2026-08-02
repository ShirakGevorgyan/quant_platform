from __future__ import annotations

from dataclasses import replace

from quant_platform.qualification.verifier import (
    QualificationVerifier,
    verify_identity,
    verify_lineage,
    verify_required_features,
)


class TestVerifyIdentity:
    def test_matches_for_an_unmodified_manifest(self, qualified_manifest) -> None:
        matches, message = verify_identity(qualified_manifest)
        assert matches is True
        assert message is None

    def test_detects_a_tampered_dataset_id(self, qualified_manifest) -> None:
        tampered = replace(qualified_manifest, dataset_id="0" * 16)
        matches, message = verify_identity(tampered)
        assert matches is False
        assert message is not None
        assert "0" * 16 in message


class TestVerifyLineage:
    def test_present_for_a_freshly_built_manifest(self, qualified_manifest) -> None:
        present, missing = verify_lineage(qualified_manifest)
        assert present is True
        assert missing == ()

    def test_detects_missing_lineage_field(self, qualified_manifest) -> None:
        stripped = replace(qualified_manifest, code_revision="")
        present, missing = verify_lineage(stripped)
        assert present is False
        assert "code_revision" in missing


class TestVerifyRequiredFeatures:
    def test_present_when_all_required_features_exist(self, qualified_manifest) -> None:
        present, missing = verify_required_features(qualified_manifest, frozenset({"trend"}))
        assert present is True
        assert missing == ()

    def test_detects_missing_required_feature(self, qualified_manifest) -> None:
        present, missing = verify_required_features(qualified_manifest, frozenset({"trend", "does_not_exist"}))
        assert present is False
        assert missing == ("does_not_exist",)


class TestQualificationVerifier:
    def test_verify_reports_clean_facts_for_a_healthy_dataset(self, qualified_manifest, research_store) -> None:
        facts = QualificationVerifier().verify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        assert facts.identity_matches is True
        assert facts.artifacts.readable is True
        assert facts.artifacts.metadata_checksums_match_manifest is True
        assert facts.artifacts.row_counts_match_manifest is True
        assert facts.lineage_present is True
        assert facts.required_features_present is True
        assert facts.leakage_free is True
        assert facts.artifacts.splits is not None
        assert set(facts.artifacts.splits) == {"train", "validation", "test"}

    def test_verify_flags_missing_required_feature(self, qualified_manifest, research_store) -> None:
        facts = QualificationVerifier().verify(
            qualified_manifest, research_store, required_feature_names=frozenset({"trend", "does_not_exist"}),
        )
        assert facts.required_features_present is False
        assert facts.missing_required_features == ("does_not_exist",)

    def test_verify_flags_tampered_identity(self, qualified_manifest, research_store) -> None:
        tampered = replace(qualified_manifest, dataset_id="0" * 16)
        facts = QualificationVerifier().verify(tampered, research_store, required_feature_names=frozenset())
        assert facts.identity_matches is False

    def test_verify_flags_unreadable_artifacts_for_a_nonexistent_content_id(self, qualified_manifest, research_store) -> None:
        bogus = replace(qualified_manifest, content_id="0" * 64)
        facts = QualificationVerifier().verify(bogus, research_store, required_feature_names=frozenset())
        assert facts.artifacts.readable is False
        assert facts.artifacts.splits is None
