from __future__ import annotations

import pytest

from quant_platform.core.exceptions import QualificationError
from quant_platform.qualification.models import (
    QUALIFICATION_DIMENSION_ORDER,
    BlockingFailure,
    BlockingFailureCode,
    DatasetQualificationReport,
    DimensionResult,
    QualificationDecision,
    QualificationDecisionKind,
    QualificationDimensionKind,
)


def _blocking_failure(dimension: QualificationDimensionKind = QualificationDimensionKind.STRUCTURAL_INTEGRITY) -> BlockingFailure:
    return BlockingFailure(
        code=BlockingFailureCode.REQUIRED_FEATURE_MISSING, dimension=dimension, message="missing feature 'foo'",
        context={"feature": "foo"},
    )


def _dimension_result(dimension: QualificationDimensionKind, *, score: float = 1.0, blocking: tuple[BlockingFailure, ...] = ()) -> DimensionResult:
    return DimensionResult(dimension=dimension, score=score, findings=("ok",), warnings=(), blocking_failures=blocking, recommendations=())


def _full_report(*, rejected: bool = False) -> DatasetQualificationReport:
    results = []
    for dimension in QUALIFICATION_DIMENSION_ORDER:
        if rejected and dimension is QualificationDimensionKind.STRUCTURAL_INTEGRITY:
            results.append(_dimension_result(dimension, score=0.0, blocking=(_blocking_failure(dimension),)))
        else:
            results.append(_dimension_result(dimension))
    decision = QualificationDecision(
        schema_version=1, dataset_id="abc123", version="000001-abc", content_id="c" * 16,
        decision=(QualificationDecisionKind.REJECTED_FOR_RESEARCH if rejected else QualificationDecisionKind.APPROVED_FOR_RESEARCH),
        decision_reason="test", blocking_failure_count=(1 if rejected else 0), overall_score=(0.5 if rejected else 1.0),
        generated_at="2026-01-01T00:00:00+00:00",
    )
    return DatasetQualificationReport(
        schema_version=1, dataset_id="abc123", version="000001-abc", content_id="c" * 16,
        dimension_results=tuple(results), decision=decision, generated_at="2026-01-01T00:00:00+00:00",
    )


class TestBlockingFailure:
    def test_json_round_trip(self) -> None:
        failure = _blocking_failure()
        restored = BlockingFailure.from_json_dict(failure.to_json_dict())
        assert restored == failure


class TestDimensionResult:
    def test_rejects_score_outside_unit_interval(self) -> None:
        with pytest.raises(QualificationError):
            DimensionResult(dimension=QualificationDimensionKind.SAFETY, score=1.5)
        with pytest.raises(QualificationError):
            DimensionResult(dimension=QualificationDimensionKind.SAFETY, score=-0.1)

    def test_rejects_blocking_failure_attributed_to_a_different_dimension(self) -> None:
        mismatched = _blocking_failure(dimension=QualificationDimensionKind.SAFETY)
        with pytest.raises(QualificationError):
            DimensionResult(dimension=QualificationDimensionKind.STRUCTURAL_INTEGRITY, score=0.0, blocking_failures=(mismatched,))

    def test_is_blocking_reflects_blocking_failures(self) -> None:
        clean = _dimension_result(QualificationDimensionKind.SAFETY)
        assert clean.is_blocking is False
        dirty = _dimension_result(QualificationDimensionKind.SAFETY, score=0.0, blocking=(_blocking_failure(QualificationDimensionKind.SAFETY),))
        assert dirty.is_blocking is True

    def test_json_round_trip(self) -> None:
        result = _dimension_result(QualificationDimensionKind.COVERAGE, score=0.75)
        restored = DimensionResult.from_json_dict(result.to_json_dict())
        assert restored == result


class TestQualificationDecision:
    def test_json_round_trip(self) -> None:
        decision = QualificationDecision(
            schema_version=1, dataset_id="abc123", version="000001-abc", content_id="c" * 16,
            decision=QualificationDecisionKind.APPROVED_FOR_RESEARCH, decision_reason="clean", blocking_failure_count=0,
            overall_score=0.9, generated_at="2026-01-01T00:00:00+00:00",
        )
        restored = QualificationDecision.from_json_dict(decision.to_json_dict())
        assert restored == decision


class TestDatasetQualificationReport:
    def test_requires_exactly_the_eight_canonical_dimensions(self) -> None:
        incomplete = _full_report()
        with pytest.raises(QualificationError):
            DatasetQualificationReport(
                schema_version=1, dataset_id="abc123", version="000001-abc", content_id="c" * 16,
                dimension_results=incomplete.dimension_results[:-1], decision=incomplete.decision, generated_at=incomplete.generated_at,
            )

    def test_dimension_result_lookup(self) -> None:
        report = _full_report()
        result = report.dimension_result(QualificationDimensionKind.SAFETY)
        assert result.dimension is QualificationDimensionKind.SAFETY

    def test_all_blocking_failures_flattens_across_dimensions(self) -> None:
        approved = _full_report(rejected=False)
        assert approved.all_blocking_failures == ()
        rejected = _full_report(rejected=True)
        assert len(rejected.all_blocking_failures) == 1
        assert rejected.all_blocking_failures[0].dimension is QualificationDimensionKind.STRUCTURAL_INTEGRITY

    def test_json_round_trip_approved(self) -> None:
        report = _full_report(rejected=False)
        restored = DatasetQualificationReport.from_json_dict(report.to_json_dict())
        assert restored == report

    def test_json_round_trip_rejected(self) -> None:
        report = _full_report(rejected=True)
        restored = DatasetQualificationReport.from_json_dict(report.to_json_dict())
        assert restored == report
