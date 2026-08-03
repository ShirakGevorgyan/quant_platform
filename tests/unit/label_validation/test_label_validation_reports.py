from __future__ import annotations

from quant_platform.label_validation import reports as lv_reports
from quant_platform.label_validation.diagnostics import compute_label_diagnostics
from quant_platform.label_validation.engine import LabelQualificationEngine
from quant_platform.label_validation.horizon import compare_horizons
from quant_platform.label_validation.overlap import detect_overlap
from quant_platform.label_validation.reconciliation import LabelValidationReconciliation
from quant_platform.label_validation.replay import LabelValidationReplay
from quant_platform.label_validation.verification import LabelValidationVerifier
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.multi_horizon_return import (
    MULTI_HORIZON_RETURN_MINIMUM_HORIZONS,
    build_multi_horizon_return_specifications,
    generate_multi_horizon_return_labels,
)
from quant_platform.labels.pricing import PriceBasis


class TestRenderDiagnosticsReport:
    def test_includes_every_dimension(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        text = lv_reports.render_diagnostics_report(diagnostics)
        for result in diagnostics.dimension_results:
            assert result.dimension.value in text

    def test_deterministic(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        assert lv_reports.render_diagnostics_report(diagnostics) == lv_reports.render_diagnostics_report(diagnostics)


class TestRenderQualificationReport:
    def test_includes_decision(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        text = lv_reports.render_qualification_report(report)
        assert f"decision: {report.decision.value}" in text


class TestRenderVerificationReport:
    def test_includes_verified_status(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationVerifier().verify(report, next_return_bundle, next_return_manifest)
        text = lv_reports.render_verification_report(result)
        assert f"verified: {result.verified}" in text


class TestRenderReconciliationReport:
    def test_includes_reconciled_status(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationReconciliation().reconcile(report, report)
        text = lv_reports.render_reconciliation_report(result)
        assert f"reconciled={result.reconciled}" in text


class TestRenderReplayReport:
    def test_includes_qualification_identical(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, ohlcv_source_data, source_content_id: str,
    ) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationReplay().replay_and_requalify(
            next_return_definition, ohlcv_source_data, source_content_id=source_content_id, manifest=next_return_manifest, original_report=report,
        )
        text = lv_reports.render_replay_report(result)
        assert f"qualification_identical: {result.qualification_identical}" in text


class TestRenderHorizonReport:
    def test_includes_every_horizon(self, ohlcv_source_data) -> None:
        specs = build_multi_horizon_return_specifications(
            horizons=MULTI_HORIZON_RETURN_MINIMUM_HORIZONS, price_basis=PriceBasis.CLOSE_TO_CLOSE, created_from_dataset="ds1", created_from_manifest="m1",
        )
        bundles = tuple(
            LabelBuilder().build(LabelDefinition(specification=s, generate=generate_multi_horizon_return_labels), ohlcv_source_data, source_content_id="src1")
            for s in specs
        )
        report = compare_horizons(bundles)
        text = lv_reports.render_horizon_report(report)
        for horizon in report.horizons:
            assert f"horizon_bars={horizon.horizon_bars}" in text


class TestRenderOverlapReport:
    def test_includes_findings(self, next_return_bundle: LabelBundle) -> None:
        report = detect_overlap((next_return_bundle, next_return_bundle))
        text = lv_reports.render_overlap_report(report)
        assert "duplicate_target" in text
