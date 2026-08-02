"""Shared composite vocabulary for `quant_platform.feature_discovery`
(Milestone 11, Phase 2, Part 1): the result types every dimension
evaluator (`diagnostics.py`) produces and every consumer (`engine.py`,
`verification.py`, `reconciliation.py`, `reports.py`) reads. Builds
directly on `evidence.py`'s `FeatureDiscoveryEvidence`/
`FeatureDiscoveryDimensionKind`/`BlockingFindingCode` -- never
redeclared here."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from quant_platform.core.exceptions import FeatureDiscoveryError
from quant_platform.feature_discovery.evidence import (
    FEATURE_DISCOVERY_DIMENSION_ORDER,
    FeatureDiscoveryDimensionKind,
    FeatureDiscoveryEvidence,
)
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "FEATURE_DIMENSION_RESULT_SCHEMA_VERSION",
    "FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION",
    "FEATURE_DISCOVERY_SUMMARY_SCHEMA_VERSION",
    "FEATURE_SIGNAL_DIAGNOSTICS_SCHEMA_VERSION",
    "FeatureDimensionResult",
    "FeatureDiscoveryReport",
    "FeatureDiscoverySummary",
    "FeatureSignalDiagnostics",
    "compute_feature_set_id",
]

FEATURE_DIMENSION_RESULT_SCHEMA_VERSION = 1
FEATURE_SIGNAL_DIAGNOSTICS_SCHEMA_VERSION = 1
FEATURE_DISCOVERY_SUMMARY_SCHEMA_VERSION = 1
FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION = 1


def compute_feature_set_id(*, dataset_id: str, feature_registry_fingerprint: str, feature_names: tuple[str, ...]) -> str:
    """A deterministic "recipe id" for the EXACT set of features being
    evaluated -- distinct from `dataset_id` (the whole dataset's own
    identity). Mirrors `features.manifests.compute_dataset_id`'s own
    plain-`hashlib.sha256`-over-a-canonical-string convention rather
    than reaching into an unrelated package's content-hash helper
    (`market_data`/`paper_trading` identity utilities are a different
    package family, not this package's natural sibling)."""
    payload = f"{dataset_id}|{feature_registry_fingerprint}|{','.join(sorted(feature_names))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class FeatureDimensionResult:
    """One dimension's own scored evaluation of ONE feature. Every
    conclusion is an `Evidence` record -- there is no separate plain-
    string `findings`/`warnings` field (unlike `qualification.models.
    DimensionResult`, which predates this package and whose evidence
    model was added later, additively, in a Part 2). Feature Discovery
    was built with the evidence model as its PRIMARY finding
    representation from the start."""

    dimension: FeatureDiscoveryDimensionKind
    feature_name: str
    score: float
    evidence: tuple[FeatureDiscoveryEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise FeatureDiscoveryError(
                f"FeatureDimensionResult.score must be in [0, 1], got {self.score}",
                context={"dimension": self.dimension.value, "feature_name": self.feature_name},
            )
        for record in self.evidence:
            if record.dimension is not self.dimension:
                raise FeatureDiscoveryError(
                    f"FeatureDiscoveryEvidence.dimension={record.dimension.value!r} does not match "
                    f"FeatureDimensionResult.dimension={self.dimension.value!r}",
                    context={"dimension": self.dimension.value, "feature_name": self.feature_name},
                )
            if record.affected_feature != self.feature_name:
                raise FeatureDiscoveryError(
                    f"FeatureDiscoveryEvidence.affected_feature={record.affected_feature!r} does not match "
                    f"FeatureDimensionResult.feature_name={self.feature_name!r}",
                    context={"dimension": self.dimension.value, "feature_name": self.feature_name},
                )

    @property
    def blocking_evidence(self) -> tuple[FeatureDiscoveryEvidence, ...]:
        return tuple(e for e in self.evidence if e.blocking)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_DIMENSION_RESULT_SCHEMA_VERSION, "dimension": self.dimension.value,
            "feature_name": self.feature_name, "score": self.score, "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDimensionResult:
        require_schema_version(raw, supported=FEATURE_DIMENSION_RESULT_SCHEMA_VERSION, context="FeatureDimensionResult")
        return cls(
            dimension=FeatureDiscoveryDimensionKind(raw["dimension"]), feature_name=str(raw["feature_name"]),
            score=float(str(raw["score"])),
            evidence=tuple(
                FeatureDiscoveryEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class FeatureSignalDiagnostics:
    """The complete, per-feature bundle: all 10 dimensions' own scored
    results for exactly one feature."""

    feature_name: str
    dimension_results: tuple[FeatureDimensionResult, ...]
    overall_score: float

    def __post_init__(self) -> None:
        found = tuple(r.dimension for r in self.dimension_results)
        if found != FEATURE_DISCOVERY_DIMENSION_ORDER:
            raise FeatureDiscoveryError(
                f"FeatureSignalDiagnostics.dimension_results must cover exactly the 10 dimensions in "
                f"FEATURE_DISCOVERY_DIMENSION_ORDER, got {[d.value for d in found]!r}",
                context={"feature_name": self.feature_name},
            )
        if any(r.feature_name != self.feature_name for r in self.dimension_results):
            raise FeatureDiscoveryError(
                f"every FeatureDimensionResult.feature_name must equal {self.feature_name!r}",
                context={"feature_name": self.feature_name},
            )

    @property
    def all_evidence(self) -> tuple[FeatureDiscoveryEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.evidence)

    @property
    def blocking_evidence(self) -> tuple[FeatureDiscoveryEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.blocking_evidence)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def dimension_result(self, dimension: FeatureDiscoveryDimensionKind) -> FeatureDimensionResult:
        for result in self.dimension_results:
            if result.dimension is dimension:
                return result
        raise FeatureDiscoveryError(f"No FeatureDimensionResult for dimension={dimension.value!r}", context={"feature_name": self.feature_name})

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_SIGNAL_DIAGNOSTICS_SCHEMA_VERSION, "feature_name": self.feature_name,
            "dimension_results": [r.to_json_dict() for r in self.dimension_results], "overall_score": self.overall_score,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureSignalDiagnostics:
        require_schema_version(raw, supported=FEATURE_SIGNAL_DIAGNOSTICS_SCHEMA_VERSION, context="FeatureSignalDiagnostics")
        return cls(
            feature_name=str(raw["feature_name"]),
            dimension_results=tuple(
                FeatureDimensionResult.from_json_dict(as_json_dict(r, field_name="dimension_results[]"))
                for r in as_json_list(raw.get("dimension_results") or [], field_name="dimension_results")
            ),
            overall_score=float(str(raw["overall_score"])),
        )


@dataclass(frozen=True, slots=True)
class FeatureDiscoverySummary:
    feature_count: int
    approved_count: int
    """Features with zero warnings and zero blocking evidence."""
    flagged_count: int
    """Features with at least one WARNING/CRITICAL-severity, non-blocking finding."""
    blocked_count: int
    """Features with at least one blocking finding."""
    mean_overall_score: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_count": self.feature_count, "approved_count": self.approved_count, "flagged_count": self.flagged_count,
            "blocked_count": self.blocked_count, "mean_overall_score": self.mean_overall_score,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoverySummary:
        return cls(
            feature_count=int(str(raw["feature_count"])), approved_count=int(str(raw["approved_count"])),
            flagged_count=int(str(raw["flagged_count"])), blocked_count=int(str(raw["blocked_count"])),
            mean_overall_score=float(str(raw["mean_overall_score"])),
        )


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryReport:
    schema_version: int
    dataset_id: str
    feature_set_id: str
    feature_count: int
    evaluation_time: str
    summary: FeatureDiscoverySummary
    per_feature_diagnostics: tuple[FeatureSignalDiagnostics, ...]
    dimension_scores: dict[str, float]
    """Mean score per dimension across every evaluated feature."""
    warnings: tuple[str, ...]
    """Dataset-level (not attributable to one feature) warnings, e.g. a
    cross-feature redundant-pair summary."""
    blocking_findings: tuple[FeatureDiscoveryEvidence, ...]
    """Every blocking `FeatureDiscoveryEvidence` across every feature,
    flattened -- a caller who only wants "what is blocked" never has to
    walk `per_feature_diagnostics` itself."""
    recommendations: tuple[str, ...]
    """Every unique, non-null recommendation across every feature."""

    def __post_init__(self) -> None:
        if len(self.per_feature_diagnostics) != self.feature_count:
            raise FeatureDiscoveryError(
                f"feature_count={self.feature_count} does not match len(per_feature_diagnostics)={len(self.per_feature_diagnostics)}",
                context={"dataset_id": self.dataset_id},
            )

    def feature_diagnostics(self, feature_name: str) -> FeatureSignalDiagnostics:
        for diagnostics in self.per_feature_diagnostics:
            if diagnostics.feature_name == feature_name:
                return diagnostics
        raise FeatureDiscoveryError(f"No FeatureSignalDiagnostics for feature_name={feature_name!r}", context={"dataset_id": self.dataset_id})

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "feature_set_id": self.feature_set_id,
            "feature_count": self.feature_count, "evaluation_time": self.evaluation_time, "summary": self.summary.to_json_dict(),
            "per_feature_diagnostics": [d.to_json_dict() for d in self.per_feature_diagnostics], "dimension_scores": self.dimension_scores,
            "warnings": list(self.warnings), "blocking_findings": [b.to_json_dict() for b in self.blocking_findings],
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoveryReport:
        require_schema_version(raw, supported=FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION, context="FeatureDiscoveryReport")
        return cls(
            schema_version=FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            feature_set_id=str(raw["feature_set_id"]), feature_count=int(str(raw["feature_count"])),
            evaluation_time=str(raw["evaluation_time"]), summary=FeatureDiscoverySummary.from_json_dict(as_json_dict(raw["summary"], field_name="summary")),
            per_feature_diagnostics=tuple(
                FeatureSignalDiagnostics.from_json_dict(as_json_dict(d, field_name="per_feature_diagnostics[]"))
                for d in as_json_list(raw.get("per_feature_diagnostics") or [], field_name="per_feature_diagnostics")
            ),
            dimension_scores={str(k): float(str(v)) for k, v in as_json_dict(raw.get("dimension_scores") or {}, field_name="dimension_scores").items()},
            warnings=tuple(str(s) for s in as_json_list(raw.get("warnings") or [], field_name="warnings")),
            blocking_findings=tuple(
                FeatureDiscoveryEvidence.from_json_dict(as_json_dict(b, field_name="blocking_findings[]"))
                for b in as_json_list(raw.get("blocking_findings") or [], field_name="blocking_findings")
            ),
            recommendations=tuple(str(s) for s in as_json_list(raw.get("recommendations") or [], field_name="recommendations")),
        )
