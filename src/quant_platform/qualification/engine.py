"""`DatasetQualificationEngine` (Milestone 11, Phase 1): the single
orchestration entry point. Runs `QualificationVerifier` exactly once,
feeds its facts into all 8 dimension evaluators (`dimensions.py`) in the
fixed `QUALIFICATION_DIMENSION_ORDER`, and derives exactly one
`QualificationDecision` from whatever `BlockingFailure`s those
dimensions produced.

DECISION RULE (deliberately the simplest rule that satisfies the spec):
any blocking failure, from any dimension, forces `REJECTED_FOR_RESEARCH`;
zero blocking failures means `APPROVED_FOR_RESEARCH`. Unlike `robustness.
promotion`'s 4-tier mandatory/advisory gate precedence, this package's
spec names only 2 decisions and 6 blocking codes with no "advisory
gate"/"skip" concept -- a WARNING-level finding on any dimension never
by itself blocks approval, only a named `BlockingFailure` does.

This module never trains a model, never computes feature importance,
never performs feature selection, and never constructs a second
`FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder` -- it only
reads an already-built `ResearchDatasetManifest` and its durable
artifacts through the existing `features.manifests.ResearchDatasetStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.qualification.dimensions import (
    evaluate_coverage,
    evaluate_determinism,
    evaluate_reproducibility,
    evaluate_safety,
    evaluate_stability,
    evaluate_statistical_integrity,
    evaluate_structural_integrity,
    evaluate_temporal_integrity,
)
from quant_platform.qualification.models import (
    DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION,
    QUALIFICATION_DECISION_SCHEMA_VERSION,
    QUALIFICATION_DIMENSION_ORDER,
    DatasetQualificationReport,
    DimensionResult,
    QualificationDecision,
    QualificationDecisionKind,
    QualificationDimensionKind,
)
from quant_platform.qualification.verifier import QualificationVerifier, VerificationFacts

__all__ = ["DatasetQualificationEngine"]


def _evaluate_all_dimensions(manifest: ResearchDatasetManifest, facts: VerificationFacts) -> dict[QualificationDimensionKind, DimensionResult]:
    splits = facts.artifacts.splits
    return {
        QualificationDimensionKind.STRUCTURAL_INTEGRITY: evaluate_structural_integrity(manifest, splits, facts),
        QualificationDimensionKind.TEMPORAL_INTEGRITY: evaluate_temporal_integrity(manifest, splits, facts),
        QualificationDimensionKind.STATISTICAL_INTEGRITY: evaluate_statistical_integrity(manifest, splits),
        QualificationDimensionKind.COVERAGE: evaluate_coverage(manifest, splits),
        QualificationDimensionKind.STABILITY: evaluate_stability(manifest, splits),
        QualificationDimensionKind.DETERMINISM: evaluate_determinism(manifest, facts),
        QualificationDimensionKind.REPRODUCIBILITY: evaluate_reproducibility(manifest, facts),
        QualificationDimensionKind.SAFETY: evaluate_safety(manifest, splits),
    }


def _decide(dataset_id: str, version: str, content_id: str, dimension_results: tuple[DimensionResult, ...]) -> QualificationDecision:
    blocking_failures = [f for r in dimension_results for f in r.blocking_failures]
    overall_score = sum(r.score for r in dimension_results) / len(dimension_results)
    generated_at = format_utc_timestamp(utc_now())

    if blocking_failures:
        codes = ", ".join(sorted({f"{f.dimension.value}:{f.code.value}" for f in blocking_failures}))
        decision = QualificationDecisionKind.REJECTED_FOR_RESEARCH
        decision_reason = f"REJECTED_FOR_RESEARCH: {len(blocking_failures)} blocking failure(s): {codes}"
    else:
        decision = QualificationDecisionKind.APPROVED_FOR_RESEARCH
        decision_reason = "APPROVED_FOR_RESEARCH: no blocking failure was found in any of the 8 qualification dimensions."

    return QualificationDecision(
        schema_version=QUALIFICATION_DECISION_SCHEMA_VERSION, dataset_id=dataset_id, version=version, content_id=content_id,
        decision=decision, decision_reason=decision_reason, blocking_failure_count=len(blocking_failures), overall_score=overall_score,
        generated_at=generated_at,
    )


@dataclass(frozen=True, slots=True)
class DatasetQualificationEngine:
    verifier: QualificationVerifier = field(default_factory=QualificationVerifier)

    def qualify(
        self, manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore, *, required_feature_names: frozenset[str] = frozenset(),
    ) -> DatasetQualificationReport:
        facts = self.verifier.verify(manifest, research_store, required_feature_names=required_feature_names)
        by_dimension = _evaluate_all_dimensions(manifest, facts)
        dimension_results = tuple(by_dimension[dimension] for dimension in QUALIFICATION_DIMENSION_ORDER)
        decision = _decide(manifest.dataset_id, manifest.version, manifest.content_id, dimension_results)
        return DatasetQualificationReport(
            schema_version=DATASET_QUALIFICATION_REPORT_SCHEMA_VERSION, dataset_id=manifest.dataset_id, version=manifest.version,
            content_id=manifest.content_id, dimension_results=dimension_results, decision=decision, generated_at=decision.generated_at,
        )
