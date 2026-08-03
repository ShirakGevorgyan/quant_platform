from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelValidationReconciliationError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.engine import LabelQualificationEngine
from quant_platform.label_validation.evidence import make_evidence
from quant_platform.label_validation.reconciliation import (
    LabelValidationReconciliation,
    LabelValidationReconciliationResult,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest


class TestLabelValidationReconciliation:
    def test_self_reconciliation_is_clean(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationReconciliation().reconcile(report, report)
        assert result.reconciled is True
        assert result.issues == ()

    def test_different_specification_ids_raise(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, direction_bundle: LabelBundle, direction_manifest: LabelManifest) -> None:
        baseline = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        candidate = LabelQualificationEngine().qualify(direction_bundle, direction_manifest)
        with pytest.raises(LabelValidationReconciliationError):
            LabelValidationReconciliation().reconcile(baseline, candidate)

    def test_decision_drift_detected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        baseline = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        constant_bundle = replace(next_return_bundle, values=pd.Series(np.full(next_return_bundle.row_count, 0.5)))
        rejected_report = LabelQualificationEngine().qualify(constant_bundle, next_return_manifest)
        candidate = replace(rejected_report, label_specification_id=baseline.label_specification_id)
        result = LabelValidationReconciliation().reconcile(baseline, candidate)
        assert result.reconciled is False
        assert any(i.kind == "decision_drift" for i in result.issues)

    def test_score_drift_detected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        baseline = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        tampered_diagnostics = replace(baseline.diagnostics, overall_score=baseline.diagnostics.overall_score - 0.5)
        candidate = replace(baseline, diagnostics=tampered_diagnostics)
        result = LabelValidationReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "score_drift" for i in result.issues)

    def test_evidence_and_warning_drift_detected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        baseline = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        first_result = baseline.diagnostics.dimension_results[0]
        extra_evidence = make_evidence(
            finding="synthetic warning for drift testing", evidence=("synthetic",), dimension=first_result.dimension, severity=Severity.WARNING,
            affected_labels=(baseline.label_specification_id,),
        )
        tampered_first_result = replace(first_result, evidence=(*first_result.evidence, extra_evidence))
        tampered_dimension_results = (tampered_first_result, *baseline.diagnostics.dimension_results[1:])
        tampered_diagnostics = replace(baseline.diagnostics, dimension_results=tampered_dimension_results)
        candidate = replace(baseline, diagnostics=tampered_diagnostics)

        result = LabelValidationReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "evidence_drift" for i in result.issues)
        assert any(i.kind == "warning_drift" for i in result.issues)

    def test_json_round_trip(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationReconciliation().reconcile(report, report)
        restored = LabelValidationReconciliationResult.from_json_dict(result.to_json_dict())
        assert restored == result
