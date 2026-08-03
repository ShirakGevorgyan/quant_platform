from __future__ import annotations

from dataclasses import replace

from quant_platform.label_validation.engine import LabelQualificationDecision, LabelQualificationEngine
from quant_platform.label_validation.verification import (
    LabelValidationVerificationResult,
    LabelValidationVerifier,
    verify_report_self_consistency,
)
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest


class TestVerifyReportSelfConsistency:
    def test_clean_report_is_self_consistent(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        consistent, issues = verify_report_self_consistency(report)
        assert consistent is True
        assert issues == ()

    def test_tampered_decision_caught(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        tampered = replace(report, decision=LabelQualificationDecision.REJECTED)
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("decision" in i for i in issues)

    def test_tampered_overall_score_caught(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        tampered_diagnostics = replace(report.diagnostics, overall_score=0.1)
        tampered = replace(report, diagnostics=tampered_diagnostics)
        consistent, issues = verify_report_self_consistency(tampered)
        assert consistent is False
        assert any("overall_score" in i for i in issues)


class TestLabelValidationVerifier:
    def test_clean_bundle_verifies(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, next_return_definition: LabelDefinition, ohlcv_source_data) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationVerifier().verify(report, next_return_bundle, next_return_manifest)
        assert result.verified is True
        assert result.self_consistent is True
        assert result.reconciliation.reconciled is True

    def test_tampered_bundle_fails_verification(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        tampered_values = next_return_bundle.values.copy()
        tampered_values.iloc[0] = 999.0
        tampered_bundle = replace(next_return_bundle, values=tampered_values)
        result = LabelValidationVerifier().verify(report, tampered_bundle, next_return_manifest)
        assert result.verified is False

    def test_json_round_trip(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationVerifier().verify(report, next_return_bundle, next_return_manifest)
        restored = LabelValidationVerificationResult.from_json_dict(result.to_json_dict())
        assert restored == result
