"""Shared vocabulary for `quant_platform.qualification` (Milestone 11,
Phase 1) -- enums and small, JSON-serializable result types every other
module in this package imports. Mirrors `robustness.models`'s role one
layer up in that package: a single place every enum lives, so no two
modules independently redefine the same closed set of names.

`Severity` is reused directly from `historical.quality` (the SAME enum
`features.validation.validate_research_dataset` already uses) -- never
redeclared, matching this repository's own established cross-package
reuse convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quant_platform.core.exceptions import QualificationError
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "BLOCKING_FAILURE_SCHEMA_VERSION",
    "DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION",
    "DIMENSION_RESULT_SCHEMA_VERSION",
    "QUALIFICATION_DECISION_SCHEMA_VERSION",
    "QUALIFICATION_DIMENSION_ORDER",
    "BlockingFailure",
    "BlockingFailureCode",
    "DatasetQualificationReport",
    "DimensionResult",
    "QualificationDecision",
    "QualificationDecisionKind",
    "QualificationDimensionKind",
    "Severity",
]


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise QualificationError(f"Expected a JSON object, got {type(value).__name__}")
    return value


class QualificationDimensionKind(Enum):
    STRUCTURAL_INTEGRITY = "structural_integrity"
    TEMPORAL_INTEGRITY = "temporal_integrity"
    STATISTICAL_INTEGRITY = "statistical_integrity"
    COVERAGE = "coverage"
    STABILITY = "stability"
    DETERMINISM = "determinism"
    REPRODUCIBILITY = "reproducibility"
    SAFETY = "safety"


QUALIFICATION_DIMENSION_ORDER: tuple[QualificationDimensionKind, ...] = (
    QualificationDimensionKind.STRUCTURAL_INTEGRITY,
    QualificationDimensionKind.TEMPORAL_INTEGRITY,
    QualificationDimensionKind.STATISTICAL_INTEGRITY,
    QualificationDimensionKind.COVERAGE,
    QualificationDimensionKind.STABILITY,
    QualificationDimensionKind.DETERMINISM,
    QualificationDimensionKind.REPRODUCIBILITY,
    QualificationDimensionKind.SAFETY,
)
"""Fixed, canonical iteration order -- every report/engine/reporting
function in this package iterates dimensions in EXACTLY this order,
never dict/set iteration order, so two qualification runs over the same
dataset always produce byte-identical `DatasetQualificationReport.
dimension_results` ordering."""


class BlockingFailureCode(Enum):
    FUTURE_LEAKAGE = "future_leakage"
    MANIFEST_CORRUPTION = "manifest_corruption"
    REPLAY_MISMATCH = "replay_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    MISSING_LINEAGE = "missing_lineage"
    REQUIRED_FEATURE_MISSING = "required_feature_missing"


class QualificationDecisionKind(Enum):
    APPROVED_FOR_RESEARCH = "approved_for_research"
    REJECTED_FOR_RESEARCH = "rejected_for_research"


BLOCKING_FAILURE_SCHEMA_VERSION = 1
DIMENSION_RESULT_SCHEMA_VERSION = 1
QUALIFICATION_DECISION_SCHEMA_VERSION = 1
DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BlockingFailure:
    """A REJECTED-triggering finding -- one of the 6 named blocking-
    failure codes, always attributed to the exact dimension that raised
    it (never a bare, unattributed global failure)."""

    code: BlockingFailureCode
    dimension: QualificationDimensionKind
    message: str
    context: dict[str, object] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value, "dimension": self.dimension.value, "message": self.message,
            "context": self.context,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BlockingFailure:
        return cls(
            code=BlockingFailureCode(raw["code"]), dimension=QualificationDimensionKind(raw["dimension"]),
            message=str(raw["message"]), context=_as_dict(raw.get("context") or {}),
        )


@dataclass(frozen=True, slots=True)
class DimensionResult:
    """One qualification dimension's own scored, self-contained
    evaluation. `findings` are neutral observations (always populated,
    even for a perfect score, e.g. "1200 rows, 0 nulls"); `warnings` are
    non-blocking concerns; `blocking_failures` are exactly the findings
    severe enough to force `REJECTED_FOR_RESEARCH`; `recommendations` are
    free-text, actionable suggestions -- never an instruction to
    retrain/reselect features (this package makes no such decision)."""

    dimension: QualificationDimensionKind
    score: float
    findings: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blocking_failures: tuple[BlockingFailure, ...] = ()
    recommendations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise QualificationError(f"DimensionResult.score must be in [0, 1], got {self.score}", context={"dimension": self.dimension.value})
        for failure in self.blocking_failures:
            if failure.dimension is not self.dimension:
                raise QualificationError(
                    f"BlockingFailure.dimension={failure.dimension.value!r} does not match "
                    f"DimensionResult.dimension={self.dimension.value!r} -- every blocking failure must be "
                    "attributed to the dimension that actually raised it.",
                    context={"dimension": self.dimension.value, "failure_code": failure.code.value},
                )

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_failures)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": DIMENSION_RESULT_SCHEMA_VERSION, "dimension": self.dimension.value, "score": self.score,
            "findings": list(self.findings), "warnings": list(self.warnings),
            "blocking_failures": [b.to_json_dict() for b in self.blocking_failures],
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DimensionResult:
        require_schema_version(raw, supported=DIMENSION_RESULT_SCHEMA_VERSION, context="DimensionResult")
        return cls(
            dimension=QualificationDimensionKind(raw["dimension"]), score=float(str(raw["score"])),
            findings=tuple(str(s) for s in as_json_list(raw.get("findings") or [], field_name="findings")),
            warnings=tuple(str(s) for s in as_json_list(raw.get("warnings") or [], field_name="warnings")),
            blocking_failures=tuple(
                BlockingFailure.from_json_dict(as_json_dict(b, field_name="blocking_failures[]"))
                for b in as_json_list(raw.get("blocking_failures") or [], field_name="blocking_failures")
            ),
            recommendations=tuple(str(s) for s in as_json_list(raw.get("recommendations") or [], field_name="recommendations")),
        )


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    """The single, definitive outcome of one qualification run --
    deliberately a separate type from `DatasetQualificationReport`
    (mirrors `robustness.promotion.PromotionDecision`'s own separation
    from its broader evidence bundle), so a caller that only cares about
    "may I use this dataset" never has to parse the full report."""

    schema_version: int
    dataset_id: str
    version: str
    content_id: str
    decision: QualificationDecisionKind
    decision_reason: str
    blocking_failure_count: int
    overall_score: float
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "version": self.version,
            "content_id": self.content_id, "decision": self.decision.value, "decision_reason": self.decision_reason,
            "blocking_failure_count": self.blocking_failure_count, "overall_score": self.overall_score,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QualificationDecision:
        require_schema_version(raw, supported=QUALIFICATION_DECISION_SCHEMA_VERSION, context="QualificationDecision")
        return cls(
            schema_version=QUALIFICATION_DECISION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), version=str(raw["version"]),
            content_id=str(raw["content_id"]), decision=QualificationDecisionKind(raw["decision"]),
            decision_reason=str(raw["decision_reason"]), blocking_failure_count=int(str(raw["blocking_failure_count"])),
            overall_score=float(str(raw["overall_score"])), generated_at=str(raw["generated_at"]),
        )


@dataclass(frozen=True, slots=True)
class DatasetQualificationReport:
    """The complete, top-level deliverable: every dimension's own result
    plus the single overall `QualificationDecision` derived from them."""

    schema_version: int
    dataset_id: str
    version: str
    content_id: str
    dimension_results: tuple[DimensionResult, ...]
    decision: QualificationDecision
    generated_at: str

    def __post_init__(self) -> None:
        found = tuple(r.dimension for r in self.dimension_results)
        if found != QUALIFICATION_DIMENSION_ORDER:
            raise QualificationError(
                f"DatasetQualificationReport.dimension_results must cover exactly the 8 dimensions in "
                f"QUALIFICATION_DIMENSION_ORDER, got {[d.value for d in found]!r}",
                context={"expected": [d.value for d in QUALIFICATION_DIMENSION_ORDER], "found": [d.value for d in found]},
            )

    @property
    def all_blocking_failures(self) -> tuple[BlockingFailure, ...]:
        return tuple(failure for result in self.dimension_results for failure in result.blocking_failures)

    def dimension_result(self, dimension: QualificationDimensionKind) -> DimensionResult:
        for result in self.dimension_results:
            if result.dimension is dimension:
                return result
        raise QualificationError(f"No DimensionResult for dimension={dimension.value!r} in this report", context={"dimension": dimension.value})

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "version": self.version,
            "content_id": self.content_id, "dimension_results": [r.to_json_dict() for r in self.dimension_results],
            "decision": self.decision.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DatasetQualificationReport:
        require_schema_version(raw, supported=DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION, context="DatasetQualificationReport")
        return cls(
            schema_version=DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), version=str(raw["version"]),
            content_id=str(raw["content_id"]),
            dimension_results=tuple(
                DimensionResult.from_json_dict(as_json_dict(r, field_name="dimension_results[]"))
                for r in as_json_list(raw.get("dimension_results") or [], field_name="dimension_results")
            ),
            decision=QualificationDecision.from_json_dict(as_json_dict(raw["decision"], field_name="decision")),
            generated_at=str(raw["generated_at"]),
        )
