"""Immutable fold-level and execution-level result records (Milestone 4B,
Sections 7-8). Both are frozen dataclasses with the platform's standard
canonical `to_json_dict`/`from_json_dict` round-trip -- there is no
mutable "result object updated in place" anywhere; a `FoldResult` is
built once, after a fold finishes, from data that cannot change afterward.

METRICS ARE AN EXPLICIT PLACEHOLDER, NOT A REAL EVALUATION
--------------------------------------------------------------------------
`FoldResult.metrics` is a validated (JSON-primitive-only, via
`models.validate_json_primitive_mapping`) but INTENTIONALLY EMPTY-BY-
DEFAULT mapping. This milestone deliberately does not define what a
"correct" performance metric is (RMSE vs. MAE vs. accuracy vs. a
trading-specific measure is a real modeling decision, explicitly out of
scope alongside hyperparameter optimization/feature selection/ensembles)
-- the field exists, is fully round-trippable, and is documented as
reserved for a future milestone to populate; nothing here computes a
score from predictions and truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.ml.models import ArtifactReference, JsonPrimitive, validate_json_primitive_mapping
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    parse_utc_timestamp,
    require_schema_version,
)

_SCHEMA_VERSION = 1


class FoldStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FoldResult:
    schema_version: int
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    status: FoldStatus
    duration_seconds: float
    validation_start: str | None = None
    validation_end: str | None = None
    validation_size: int = 0
    artifact_references: tuple[ArtifactReference, ...] = ()
    metrics: Mapping[str, JsonPrimitive] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError(f"FoldResult.fold_index must be >= 0, got {self.fold_index}")
        if self.train_size < 0 or self.validation_size < 0 or self.test_size < 0:
            raise ValueError("FoldResult train/validation/test sizes must be >= 0")
        if self.duration_seconds < 0:
            raise ValueError(f"FoldResult.duration_seconds must be >= 0, got {self.duration_seconds}")
        parse_utc_timestamp(self.train_start)
        parse_utc_timestamp(self.train_end)
        parse_utc_timestamp(self.test_start)
        parse_utc_timestamp(self.test_end)
        if (self.validation_start is None) != (self.validation_end is None):
            raise ValueError("FoldResult validation_start/validation_end must both be set or both be None")
        if self.validation_start is not None:
            parse_utc_timestamp(self.validation_start)
            parse_utc_timestamp(self.validation_end)  # type: ignore[arg-type]
        if (self.validation_size > 0) != (self.validation_start is not None):
            raise ValueError("FoldResult.validation_size > 0 requires validation_start/validation_end to be set")
        if self.status is FoldStatus.FAILED and not self.failure_reason:
            raise ValueError("FoldResult.failure_reason is required when status=FAILED")
        if self.status is not FoldStatus.FAILED and self.failure_reason is not None:
            raise ValueError("FoldResult.failure_reason must be None unless status=FAILED")
        validate_json_primitive_mapping(self.metrics, field_name="FoldResult.metrics")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
            "metrics": dict(sorted(self.metrics.items())),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FoldResult:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="FoldResult")
        return cls(
            schema_version=_SCHEMA_VERSION,
            fold_index=int(str(raw["fold_index"])),
            train_start=str(raw["train_start"]), train_end=str(raw["train_end"]),
            validation_start=(None if raw.get("validation_start") is None else str(raw["validation_start"])),
            validation_end=(None if raw.get("validation_end") is None else str(raw["validation_end"])),
            test_start=str(raw["test_start"]), test_end=str(raw["test_end"]),
            train_size=int(str(raw["train_size"])), validation_size=int(str(raw.get("validation_size", 0))),
            test_size=int(str(raw["test_size"])),
            status=FoldStatus(raw["status"]), duration_seconds=float(str(raw["duration_seconds"])),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
            metrics=dict(as_json_dict(raw.get("metrics") or {}, field_name="metrics")),
            failure_reason=(None if raw.get("failure_reason") is None else str(raw["failure_reason"])),
        )


@dataclass(frozen=True, slots=True)
class AggregatedExecutionResult:
    """The final, immutable summary of one execution run -- written once,
    at the very end, when the execution reaches a terminal
    `ExecutionStage`. Never rewritten afterward (a resumed execution that
    eventually completes writes ONE new aggregate reflecting the final,
    true outcome, not an in-place edit of a previous partial one)."""

    schema_version: int
    experiment_id: str
    total_folds: int
    completed_fold_indices: tuple[int, ...]
    failed_fold_indices: tuple[int, ...]
    overall_status: ExecutionStage
    started_at: str
    completed_at: str
    execution_duration_seconds: float
    artifact_references: tuple[ArtifactReference, ...] = ()
    resume_count: int = 0

    def __post_init__(self) -> None:
        if self.total_folds < 0:
            raise ValueError(f"AggregatedExecutionResult.total_folds must be >= 0, got {self.total_folds}")
        if len(set(self.completed_fold_indices)) != len(self.completed_fold_indices):
            raise ValueError("AggregatedExecutionResult.completed_fold_indices must not contain duplicates")
        if len(set(self.failed_fold_indices)) != len(self.failed_fold_indices):
            raise ValueError("AggregatedExecutionResult.failed_fold_indices must not contain duplicates")
        overlap = set(self.completed_fold_indices) & set(self.failed_fold_indices)
        if overlap:
            raise ValueError(f"A fold index cannot be both completed and failed: {sorted(overlap)}")
        if self.execution_duration_seconds < 0:
            raise ValueError("AggregatedExecutionResult.execution_duration_seconds must be >= 0")
        if self.resume_count < 0:
            raise ValueError("AggregatedExecutionResult.resume_count must be >= 0")
        parse_utc_timestamp(self.started_at)
        parse_utc_timestamp(self.completed_at)

    @property
    def is_fully_completed(self) -> bool:
        return self.overall_status is ExecutionStage.COMPLETED and len(self.completed_fold_indices) == self.total_folds

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "total_folds": self.total_folds,
            "completed_fold_indices": list(self.completed_fold_indices),
            "failed_fold_indices": list(self.failed_fold_indices),
            "overall_status": self.overall_status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_duration_seconds": self.execution_duration_seconds,
            "artifact_references": [a.to_json_dict() for a in self.artifact_references],
            "resume_count": self.resume_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AggregatedExecutionResult:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="AggregatedExecutionResult")
        return cls(
            schema_version=_SCHEMA_VERSION,
            experiment_id=str(raw["experiment_id"]),
            total_folds=int(str(raw["total_folds"])),
            completed_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("completed_fold_indices") or [], field_name="completed_fold_indices")
            ),
            failed_fold_indices=tuple(
                int(str(i)) for i in as_json_list(raw.get("failed_fold_indices") or [], field_name="failed_fold_indices")
            ),
            overall_status=ExecutionStage(raw["overall_status"]),
            started_at=str(raw["started_at"]), completed_at=str(raw["completed_at"]),
            execution_duration_seconds=float(str(raw["execution_duration_seconds"])),
            artifact_references=tuple(
                ArtifactReference.from_json_dict(as_json_dict(a, field_name="artifact_references[]"))
                for a in as_json_list(raw.get("artifact_references") or [], field_name="artifact_references")
            ),
            resume_count=int(str(raw.get("resume_count", 0))),
        )


__all__ = ["AggregatedExecutionResult", "FoldResult", "FoldStatus"]
