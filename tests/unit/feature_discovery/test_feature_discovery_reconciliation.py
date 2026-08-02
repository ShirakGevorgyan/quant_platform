from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import FeatureDiscoveryReconciliationError
from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.evidence import FeatureDiscoveryDimensionKind
from quant_platform.feature_discovery.models import FeatureSignalDiagnostics
from quant_platform.feature_discovery.reconciliation import (
    FeatureDiscoveryReconciliation,
    FeatureDiscoveryReconciliationResult,
)


def _replace_feature_dimension(report, feature_name, dimension, **kwargs):
    def _replace_diag(diag: FeatureSignalDiagnostics) -> FeatureSignalDiagnostics:
        if diag.feature_name != feature_name:
            return diag
        new_results = tuple(replace(r, **kwargs) if r.dimension is dimension else r for r in diag.dimension_results)
        return replace(diag, dimension_results=new_results)

    return replace(report, per_feature_diagnostics=tuple(_replace_diag(d) for d in report.per_feature_diagnostics))


class TestFeatureDiscoveryReconciliation:
    def test_identical_reruns_fully_reconcile(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        a = engine.discover(discovered_manifest, research_store)
        b = engine.discover(discovered_manifest, research_store)
        result = FeatureDiscoveryReconciliation().reconcile(a, b)
        assert result.reconciled is True
        assert result.issues == ()

    def test_feature_subset_difference_is_feature_set_drift(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        full = engine.discover(discovered_manifest, research_store)
        subset = engine.discover(discovered_manifest, research_store, feature_names=frozenset({"trend"}))
        result = FeatureDiscoveryReconciliation().reconcile(full, subset)
        assert result.reconciled is False
        assert any(i.kind == "feature_set_drift" for i in result.issues)

    def test_different_dataset_ids_raise(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        other = replace(report, dataset_id="f" * 16)
        with pytest.raises(FeatureDiscoveryReconciliationError):
            FeatureDiscoveryReconciliation().reconcile(report, other)

    def test_score_drift_detected(self, discovered_manifest, research_store) -> None:
        baseline = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        candidate = _replace_feature_dimension(baseline, "trend", FeatureDiscoveryDimensionKind.INFORMATION_CONTENT, score=0.1)
        result = FeatureDiscoveryReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "score_drift" and i.feature_name == "trend" for i in result.issues)

    def test_finding_drift_detected(self, discovered_manifest, research_store) -> None:
        baseline = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        candidate = _replace_feature_dimension(baseline, "trend", FeatureDiscoveryDimensionKind.COVERAGE, evidence=())
        result = FeatureDiscoveryReconciliation().reconcile(baseline, candidate)
        assert any(i.kind == "finding_drift" and i.feature_name == "trend" for i in result.issues)

    def test_json_round_trip(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        a = engine.discover(discovered_manifest, research_store)
        subset = engine.discover(discovered_manifest, research_store, feature_names=frozenset({"trend"}))
        result = FeatureDiscoveryReconciliation().reconcile(a, subset)
        assert FeatureDiscoveryReconciliationResult.from_json_dict(result.to_json_dict()) == result
