from __future__ import annotations

import pandas as pd

from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.diagnostics import compute_label_diagnostics
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.models import LabelSpecification
from quant_platform.labels.reconciliation import LabelReconciliation
from quant_platform.labels.registry import LabelRegistry
from quant_platform.labels.reports import (
    render_bundle_report,
    render_diagnostics_report,
    render_manifest_report,
    render_reconciliation_report,
    render_specification_report,
    render_verification_report,
    render_version_history_report,
)
from quant_platform.labels.verification import LabelVerifier


class TestRenderSpecificationReport:
    def test_includes_key_fields(self, specification: LabelSpecification) -> None:
        text = render_specification_report(specification)
        assert specification.label_specification_id in text
        assert specification.label_family.value in text

    def test_deterministic(self, specification: LabelSpecification) -> None:
        assert render_specification_report(specification) == render_specification_report(specification)


class TestRenderManifestReport:
    def test_includes_lineage(self, manifest: LabelManifest) -> None:
        text = render_manifest_report(manifest)
        assert manifest.dataset_identity in text
        for step in manifest.dependency_chain:
            assert step in text


class TestRenderBundleReport:
    def test_includes_counts(self, bundle: LabelBundle) -> None:
        text = render_bundle_report(bundle)
        assert str(bundle.row_count) in text
        assert bundle.identity.content_id in text


class TestRenderDiagnosticsReport:
    def test_includes_every_dimension(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(bundle, manifest)
        text = render_diagnostics_report(diagnostics)
        for result in diagnostics.dimension_results:
            assert result.dimension.value in text


class TestRenderVerificationReport:
    def test_includes_verified_status(
        self, bundle: LabelBundle, manifest: LabelManifest, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        result = LabelVerifier().verify(bundle, manifest, definition, source_data, source_content_id=source_content_id)
        text = render_verification_report(result)
        assert f"verified: {result.verified}" in text


class TestRenderReconciliationReport:
    def test_includes_reconciled_status(self, bundle: LabelBundle, manifest: LabelManifest) -> None:
        result = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=manifest)
        text = render_reconciliation_report(result)
        assert "reconciled=True" in text


class TestRenderVersionHistoryReport:
    def test_includes_every_version(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        history = registry.versions(specification.label_family)
        text = render_version_history_report(history)
        assert specification.label_specification_id in text
        assert specification.generation_version in text
