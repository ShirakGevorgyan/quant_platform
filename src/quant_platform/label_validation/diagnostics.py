"""`LabelDiagnostics` (Milestone 11, Phase 3, Part C): the complete,
8-dimension diagnostic report for one already-generated `labels.
builder.LabelBundle`, aggregating `distribution.py`/`balance.py`/
`degeneracy.py`/`coverage.py`/`stability.py`/`drift.py`/`leakage.py`
into one fixed-order, scored report -- exactly the SAME "compute each
section once, score it, never recompute" shape `qualification.engine`/
`feature_discovery.diagnostics` already established, one package over.

Every dimension's score starts at `1.0` and is reduced by a fixed,
documented penalty per non-blocking WARNING/CRITICAL evidence record it
carries; ANY blocking evidence anywhere in a dimension forces that
dimension's score to `0.0` -- no hidden weighting, no randomization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelValidationError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.balance import LabelBalance, compute_label_balance
from quant_platform.label_validation.coverage import LabelCoverage, compute_label_coverage
from quant_platform.label_validation.degeneracy import LabelDegeneracy, compute_label_degeneracy
from quant_platform.label_validation.distribution import LabelDistribution, compute_label_distribution
from quant_platform.label_validation.drift import LabelDrift, compute_label_drift
from quant_platform.label_validation.evidence import (
    LABEL_VALIDATION_DIMENSION_ORDER,
    LabelEvidence,
    LabelValidationDimensionKind,
)
from quant_platform.label_validation.leakage import LeakageValidationResult, validate_leakage
from quant_platform.label_validation.stability import LabelStability, compute_label_stability
from quant_platform.label_validation.statistics import LabelStatistics, compute_label_statistics
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.records import LabelRecord
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_DIAGNOSTICS_SCHEMA_VERSION",
    "LABEL_VALIDATION_DIMENSION_RESULT_SCHEMA_VERSION",
    "LabelDiagnostics",
    "LabelValidationDimensionResult",
    "compute_label_diagnostics",
]

LABEL_VALIDATION_DIMENSION_RESULT_SCHEMA_VERSION = 1
LABEL_DIAGNOSTICS_SCHEMA_VERSION = 1

_CRITICAL_PENALTY = 0.4
_WARNING_PENALTY = 0.15


def _score_from_evidence(evidence: tuple[LabelEvidence, ...]) -> float:
    if any(e.blocking for e in evidence):
        return 0.0
    score = 1.0
    for e in evidence:
        if e.severity is Severity.CRITICAL:
            score -= _CRITICAL_PENALTY
        elif e.severity is Severity.WARNING:
            score -= _WARNING_PENALTY
    return max(0.0, min(1.0, score))


@dataclass(frozen=True, slots=True)
class LabelValidationDimensionResult:
    dimension: LabelValidationDimensionKind
    label_specification_id: str
    score: float
    evidence: tuple[LabelEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise LabelValidationError(f"LabelValidationDimensionResult.score must be in [0, 1], got {self.score}", context={"dimension": self.dimension.value})
        for record in self.evidence:
            if record.dimension is not self.dimension:
                raise LabelValidationError(
                    f"LabelEvidence.dimension={record.dimension.value!r} does not match LabelValidationDimensionResult.dimension={self.dimension.value!r}",
                    context={"dimension": self.dimension.value},
                )

    @property
    def blocking_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for e in self.evidence if e.blocking)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": LABEL_VALIDATION_DIMENSION_RESULT_SCHEMA_VERSION, "dimension": self.dimension.value,
            "label_specification_id": self.label_specification_id, "score": self.score, "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelValidationDimensionResult:
        require_schema_version(raw, supported=LABEL_VALIDATION_DIMENSION_RESULT_SCHEMA_VERSION, context="LabelValidationDimensionResult")
        return cls(
            dimension=LabelValidationDimensionKind(raw["dimension"]), label_specification_id=str(raw["label_specification_id"]),
            score=float(str(raw["score"])),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class LabelDiagnostics:
    schema_version: int
    label_specification_id: str
    dimension_results: tuple[LabelValidationDimensionResult, ...]
    overall_score: float
    statistics: LabelStatistics
    distribution: LabelDistribution
    balance: LabelBalance
    degeneracy: LabelDegeneracy
    coverage: LabelCoverage
    stability: LabelStability
    drift: LabelDrift | None
    leakage: LeakageValidationResult

    def __post_init__(self) -> None:
        found = tuple(r.dimension for r in self.dimension_results)
        if found != LABEL_VALIDATION_DIMENSION_ORDER:
            raise LabelValidationError(
                f"LabelDiagnostics.dimension_results must cover exactly LABEL_VALIDATION_DIMENSION_ORDER, got {[d.value for d in found]!r}",
                context={"label_specification_id": self.label_specification_id},
            )

    @property
    def all_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.evidence)

    @property
    def blocking_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.blocking_evidence)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def dimension_result(self, dimension: LabelValidationDimensionKind) -> LabelValidationDimensionResult:
        for result in self.dimension_results:
            if result.dimension is dimension:
                return result
        raise LabelValidationError(f"No LabelValidationDimensionResult for dimension={dimension.value!r}", context={"label_specification_id": self.label_specification_id})

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "dimension_results": [r.to_json_dict() for r in self.dimension_results], "overall_score": self.overall_score,
            "statistics": self.statistics.to_json_dict(), "distribution": self.distribution.to_json_dict(),
            "balance": self.balance.to_json_dict(), "degeneracy": self.degeneracy.to_json_dict(), "coverage": self.coverage.to_json_dict(),
            "stability": self.stability.to_json_dict(), "drift": (None if self.drift is None else self.drift.to_json_dict()),
            "leakage": self.leakage.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDiagnostics:
        require_schema_version(raw, supported=LABEL_DIAGNOSTICS_SCHEMA_VERSION, context="LabelDiagnostics")
        drift_raw = raw.get("drift")
        return cls(
            schema_version=LABEL_DIAGNOSTICS_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            dimension_results=tuple(
                LabelValidationDimensionResult.from_json_dict(as_json_dict(r, field_name="dimension_results[]"))
                for r in as_json_list(raw.get("dimension_results") or [], field_name="dimension_results")
            ),
            overall_score=float(str(raw["overall_score"])), statistics=LabelStatistics.from_json_dict(as_json_dict(raw["statistics"], field_name="statistics")),
            distribution=LabelDistribution.from_json_dict(as_json_dict(raw["distribution"], field_name="distribution")),
            balance=LabelBalance.from_json_dict(as_json_dict(raw["balance"], field_name="balance")),
            degeneracy=LabelDegeneracy.from_json_dict(as_json_dict(raw["degeneracy"], field_name="degeneracy")),
            coverage=LabelCoverage.from_json_dict(as_json_dict(raw["coverage"], field_name="coverage")),
            stability=LabelStability.from_json_dict(as_json_dict(raw["stability"], field_name="stability")),
            drift=(None if drift_raw is None else LabelDrift.from_json_dict(as_json_dict(drift_raw, field_name="drift"))),
            leakage=LeakageValidationResult.from_json_dict(as_json_dict(raw["leakage"], field_name="leakage")),
        )


def compute_label_diagnostics(
    bundle: LabelBundle, manifest: LabelManifest, *, drift_baseline: LabelBundle | None = None, regime_assignment: pd.Series | None = None,
    records: tuple[LabelRecord, ...] | None = None,
) -> LabelDiagnostics:
    spec_id = bundle.specification.label_specification_id

    statistics = compute_label_statistics(bundle)
    distribution = compute_label_distribution(bundle)
    balance = compute_label_balance(bundle)
    degeneracy = compute_label_degeneracy(bundle)
    coverage = compute_label_coverage(bundle)
    stability = compute_label_stability(bundle, regime_assignment=regime_assignment)
    drift = compute_label_drift(drift_baseline, bundle) if drift_baseline is not None else None
    leakage = validate_leakage(bundle, manifest, records=records)

    temporal_evidence = tuple(e for e in stability.evidence if e.dimension is LabelValidationDimensionKind.TEMPORAL_STABILITY)
    regime_evidence = tuple(e for e in stability.evidence if e.dimension is LabelValidationDimensionKind.REGIME_STABILITY)
    drift_evidence: tuple[LabelEvidence, ...] = drift.evidence if drift is not None else ()

    dimension_results = (
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.DISTRIBUTION, label_specification_id=spec_id, score=_score_from_evidence(distribution.evidence), evidence=distribution.evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.BALANCE, label_specification_id=spec_id, score=_score_from_evidence(balance.evidence), evidence=balance.evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.DEGENERACY, label_specification_id=spec_id, score=_score_from_evidence(degeneracy.evidence), evidence=degeneracy.evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.COVERAGE, label_specification_id=spec_id, score=_score_from_evidence(coverage.evidence), evidence=coverage.evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.TEMPORAL_STABILITY, label_specification_id=spec_id, score=_score_from_evidence(temporal_evidence), evidence=temporal_evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.REGIME_STABILITY, label_specification_id=spec_id, score=_score_from_evidence(regime_evidence), evidence=regime_evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.DRIFT, label_specification_id=spec_id, score=_score_from_evidence(drift_evidence), evidence=drift_evidence),
        LabelValidationDimensionResult(dimension=LabelValidationDimensionKind.LEAKAGE, label_specification_id=spec_id, score=_score_from_evidence(leakage.evidence), evidence=leakage.evidence),
    )
    overall_score = sum(r.score for r in dimension_results) / len(dimension_results)

    return LabelDiagnostics(
        schema_version=LABEL_DIAGNOSTICS_SCHEMA_VERSION, label_specification_id=spec_id, dimension_results=dimension_results,
        overall_score=overall_score, statistics=statistics, distribution=distribution, balance=balance, degeneracy=degeneracy,
        coverage=coverage, stability=stability, drift=drift, leakage=leakage,
    )
