from __future__ import annotations

import pytest

from quant_platform.execution.results import AggregatedExecutionResult, FoldResult, FoldStatus
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_NOW = format_utc_timestamp(utc_now())


def _completed_fold(**overrides: object) -> FoldResult:
    base: dict[str, object] = {
        "schema_version": 1, "fold_index": 0, "train_start": _NOW, "train_end": _NOW,
        "test_start": _NOW, "test_end": _NOW, "train_size": 100, "test_size": 20,
        "status": FoldStatus.COMPLETED, "duration_seconds": 1.5,
    }
    base.update(overrides)
    return FoldResult(**base)  # type: ignore[arg-type]


class TestFoldResultConstruction:
    def test_valid_completed_result_builds(self) -> None:
        result = _completed_fold()
        assert result.status is FoldStatus.COMPLETED
        assert result.failure_reason is None

    def test_round_trip(self) -> None:
        result = _completed_fold(metrics={"a": 1.0})
        assert FoldResult.from_json_dict(result.to_json_dict()) == result

    def test_negative_fold_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_index"):
            _completed_fold(fold_index=-1)

    def test_negative_sizes_rejected(self) -> None:
        with pytest.raises(ValueError, match="sizes"):
            _completed_fold(train_size=-1)

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds"):
            _completed_fold(duration_seconds=-1.0)

    def test_failed_requires_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="failure_reason is required"):
            _completed_fold(status=FoldStatus.FAILED)

    def test_completed_forbids_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="must be None unless"):
            _completed_fold(failure_reason="oops")

    def test_failed_with_reason_is_valid(self) -> None:
        result = _completed_fold(status=FoldStatus.FAILED, failure_reason="bad data")
        assert result.status is FoldStatus.FAILED

    def test_validation_start_end_must_both_be_set_or_both_none(self) -> None:
        with pytest.raises(ValueError, match="validation_start/validation_end"):
            _completed_fold(validation_start=_NOW)

    def test_validation_size_positive_requires_validation_timestamps(self) -> None:
        with pytest.raises(ValueError, match="validation_size > 0"):
            _completed_fold(validation_size=5)

    def test_valid_validation_fields_together(self) -> None:
        result = _completed_fold(validation_start=_NOW, validation_end=_NOW, validation_size=10)
        assert result.validation_size == 10

    def test_metrics_must_be_json_primitives(self) -> None:
        with pytest.raises(ValueError):
            _completed_fold(metrics={"bad": [1, 2]})  # type: ignore[dict-item]

    def test_invalid_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="not timezone-aware"):
            _completed_fold(train_start="2024-01-01T00:00:00")


def _aggregate(**overrides: object) -> AggregatedExecutionResult:
    base: dict[str, object] = {
        "schema_version": 1, "experiment_id": "a" * 64, "total_folds": 3,
        "completed_fold_indices": (0, 1, 2), "failed_fold_indices": (),
        "overall_status": ExecutionStage.COMPLETED, "started_at": _NOW, "completed_at": _NOW,
        "execution_duration_seconds": 12.3,
    }
    base.update(overrides)
    return AggregatedExecutionResult(**base)  # type: ignore[arg-type]


class TestAggregatedExecutionResultConstruction:
    def test_valid_result_builds(self) -> None:
        agg = _aggregate()
        assert agg.is_fully_completed

    def test_round_trip(self) -> None:
        agg = _aggregate()
        assert AggregatedExecutionResult.from_json_dict(agg.to_json_dict()) == agg

    def test_duplicate_completed_indices_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            _aggregate(completed_fold_indices=(0, 0, 1))

    def test_duplicate_failed_indices_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            _aggregate(failed_fold_indices=(0, 0))

    def test_overlap_between_completed_and_failed_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be both"):
            _aggregate(completed_fold_indices=(0, 1), failed_fold_indices=(1, 2))

    def test_negative_total_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_folds"):
            _aggregate(total_folds=-1)

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="execution_duration_seconds"):
            _aggregate(execution_duration_seconds=-1.0)

    def test_negative_resume_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="resume_count"):
            _aggregate(resume_count=-1)

    def test_is_fully_completed_false_when_folds_failed(self) -> None:
        agg = _aggregate(overall_status=ExecutionStage.FAILED, completed_fold_indices=(0, 1), failed_fold_indices=(2,))
        assert not agg.is_fully_completed

    def test_is_fully_completed_false_when_not_all_folds_done(self) -> None:
        agg = _aggregate(total_folds=5, completed_fold_indices=(0, 1, 2))
        assert not agg.is_fully_completed
