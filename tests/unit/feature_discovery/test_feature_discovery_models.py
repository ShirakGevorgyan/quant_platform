from __future__ import annotations

import pytest

from quant_platform.core.exceptions import FeatureDiscoveryError
from quant_platform.feature_discovery.evidence import (
    FEATURE_DISCOVERY_DIMENSION_ORDER,
    BlockingFindingCode,
    FeatureDiscoveryDimensionKind,
    FeatureDiscoveryEvidence,
    make_evidence,
)
from quant_platform.feature_discovery.models import (
    FeatureDimensionResult,
    FeatureDiscoveryReport,
    FeatureDiscoverySummary,
    FeatureSignalDiagnostics,
    compute_feature_set_id,
)
from quant_platform.historical.quality import Severity


def _evidence(dimension: FeatureDiscoveryDimensionKind, feature_name: str = "trend", *, severity: Severity = Severity.INFO, **kwargs) -> FeatureDiscoveryEvidence:
    return make_evidence(finding="ok", evidence=(), dimension=dimension, severity=severity, affected_feature=feature_name, **kwargs)


class TestFeatureDiscoveryEvidence:
    def test_json_round_trip(self) -> None:
        e = make_evidence(
            finding="constant", evidence=("std=0",), dimension=FeatureDiscoveryDimensionKind.INFORMATION_CONTENT, severity=Severity.WARNING,
            affected_feature="trend", supporting_statistics={"variance": 0.0}, recommendation="check",
        )
        assert FeatureDiscoveryEvidence.from_json_dict(e.to_json_dict()) == e

    def test_blocking_requires_blocking_code(self) -> None:
        with pytest.raises(ValueError):
            FeatureDiscoveryEvidence(
                finding="x", evidence=(), dimension=FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY, severity=Severity.CRITICAL,
                recommendation=None, affected_feature="trend", blocking=True, blocking_code=None,
            )

    def test_blocking_code_requires_blocking_true(self) -> None:
        with pytest.raises(ValueError):
            FeatureDiscoveryEvidence(
                finding="x", evidence=(), dimension=FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY, severity=Severity.CRITICAL,
                recommendation=None, affected_feature="trend", blocking=False, blocking_code=BlockingFindingCode.LEAKAGE,
            )


class TestFeatureDimensionResult:
    def test_rejects_score_outside_unit_interval(self) -> None:
        with pytest.raises(FeatureDiscoveryError):
            FeatureDimensionResult(dimension=FeatureDiscoveryDimensionKind.COVERAGE, feature_name="trend", score=1.5)

    def test_rejects_evidence_with_mismatched_dimension(self) -> None:
        bad = _evidence(FeatureDiscoveryDimensionKind.SAFETY if False else FeatureDiscoveryDimensionKind.AVAILABILITY)
        with pytest.raises(FeatureDiscoveryError):
            FeatureDimensionResult(dimension=FeatureDiscoveryDimensionKind.COVERAGE, feature_name="trend", score=1.0, evidence=(bad,))

    def test_rejects_evidence_with_mismatched_feature_name(self) -> None:
        bad = _evidence(FeatureDiscoveryDimensionKind.COVERAGE, feature_name="other_feature")
        with pytest.raises(FeatureDiscoveryError):
            FeatureDimensionResult(dimension=FeatureDiscoveryDimensionKind.COVERAGE, feature_name="trend", score=1.0, evidence=(bad,))

    def test_json_round_trip(self) -> None:
        result = FeatureDimensionResult(
            dimension=FeatureDiscoveryDimensionKind.COVERAGE, feature_name="trend", score=0.8,
            evidence=(_evidence(FeatureDiscoveryDimensionKind.COVERAGE),),
        )
        assert FeatureDimensionResult.from_json_dict(result.to_json_dict()) == result


class TestFeatureSignalDiagnostics:
    def test_requires_exactly_the_ten_canonical_dimensions(self) -> None:
        results = tuple(FeatureDimensionResult(dimension=d, feature_name="trend", score=1.0) for d in FEATURE_DISCOVERY_DIMENSION_ORDER[:-1])
        with pytest.raises(FeatureDiscoveryError):
            FeatureSignalDiagnostics(feature_name="trend", dimension_results=results, overall_score=1.0)

    def test_requires_matching_feature_name_on_every_dimension_result(self) -> None:
        results = tuple(
            FeatureDimensionResult(dimension=d, feature_name=("other" if i == 0 else "trend"), score=1.0)
            for i, d in enumerate(FEATURE_DISCOVERY_DIMENSION_ORDER)
        )
        with pytest.raises(FeatureDiscoveryError):
            FeatureSignalDiagnostics(feature_name="trend", dimension_results=results, overall_score=1.0)

    def test_dimension_result_lookup_and_blocking_properties(self) -> None:
        blocking_evidence = _evidence(
            FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY, blocking=True, blocking_code=BlockingFindingCode.LEAKAGE, severity=Severity.CRITICAL,
        )
        results = tuple(
            FeatureDimensionResult(
                dimension=d, feature_name="trend", score=(0.0 if d is FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY else 1.0),
                evidence=((blocking_evidence,) if d is FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY else ()),
            )
            for d in FEATURE_DISCOVERY_DIMENSION_ORDER
        )
        diagnostics = FeatureSignalDiagnostics(feature_name="trend", dimension_results=results, overall_score=0.9)
        assert diagnostics.is_blocking is True
        assert len(diagnostics.blocking_evidence) == 1
        assert diagnostics.dimension_result(FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY).score == 0.0

    def test_json_round_trip(self) -> None:
        results = tuple(FeatureDimensionResult(dimension=d, feature_name="trend", score=1.0) for d in FEATURE_DISCOVERY_DIMENSION_ORDER)
        diagnostics = FeatureSignalDiagnostics(feature_name="trend", dimension_results=results, overall_score=1.0)
        assert FeatureSignalDiagnostics.from_json_dict(diagnostics.to_json_dict()) == diagnostics


class TestFeatureDiscoveryReport:
    def _diagnostics(self, feature_name: str) -> FeatureSignalDiagnostics:
        results = tuple(FeatureDimensionResult(dimension=d, feature_name=feature_name, score=1.0) for d in FEATURE_DISCOVERY_DIMENSION_ORDER)
        return FeatureSignalDiagnostics(feature_name=feature_name, dimension_results=results, overall_score=1.0)

    def test_feature_count_must_match_per_feature_diagnostics_length(self) -> None:
        with pytest.raises(FeatureDiscoveryError):
            FeatureDiscoveryReport(
                schema_version=1, dataset_id="abc", feature_set_id="xyz", feature_count=2,
                evaluation_time="2026-01-01T00:00:00+00:00", summary=FeatureDiscoverySummary(1, 1, 0, 0, 1.0),
                per_feature_diagnostics=(self._diagnostics("trend"),), dimension_scores={}, warnings=(), blocking_findings=(), recommendations=(),
            )

    def test_feature_diagnostics_lookup(self) -> None:
        report = FeatureDiscoveryReport(
            schema_version=1, dataset_id="abc", feature_set_id="xyz", feature_count=1, evaluation_time="2026-01-01T00:00:00+00:00",
            summary=FeatureDiscoverySummary(1, 1, 0, 0, 1.0), per_feature_diagnostics=(self._diagnostics("trend"),), dimension_scores={},
            warnings=(), blocking_findings=(), recommendations=(),
        )
        assert report.feature_diagnostics("trend").feature_name == "trend"
        with pytest.raises(FeatureDiscoveryError):
            report.feature_diagnostics("does_not_exist")

    def test_json_round_trip(self) -> None:
        report = FeatureDiscoveryReport(
            schema_version=1, dataset_id="abc", feature_set_id="xyz", feature_count=1, evaluation_time="2026-01-01T00:00:00+00:00",
            summary=FeatureDiscoverySummary(1, 1, 0, 0, 1.0), per_feature_diagnostics=(self._diagnostics("trend"),),
            dimension_scores={"information_content": 1.0}, warnings=("w1",), blocking_findings=(), recommendations=("r1",),
        )
        assert FeatureDiscoveryReport.from_json_dict(report.to_json_dict()) == report


class TestComputeFeatureSetId:
    def test_deterministic_regardless_of_input_order(self) -> None:
        a = compute_feature_set_id(dataset_id="abc", feature_registry_fingerprint="fp", feature_names=("trend", "const"))
        b = compute_feature_set_id(dataset_id="abc", feature_registry_fingerprint="fp", feature_names=("const", "trend"))
        assert a == b

    def test_sensitive_to_every_input(self) -> None:
        base = compute_feature_set_id(dataset_id="abc", feature_registry_fingerprint="fp", feature_names=("trend",))
        assert base != compute_feature_set_id(dataset_id="xyz", feature_registry_fingerprint="fp", feature_names=("trend",))
        assert base != compute_feature_set_id(dataset_id="abc", feature_registry_fingerprint="fp2", feature_names=("trend",))
        assert base != compute_feature_set_id(dataset_id="abc", feature_registry_fingerprint="fp", feature_names=("trend", "const"))
