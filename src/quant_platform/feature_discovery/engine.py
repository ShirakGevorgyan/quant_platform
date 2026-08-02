"""`FeatureDiscoveryEngine` (Milestone 11, Phase 2, Part 1): the single
orchestration entry point. Computes `SharedDiscoveryFacts` exactly
once, evaluates every requested feature's `FeatureSignalDiagnostics`
against those shared facts, and rolls the results up into one
`FeatureDiscoveryReport`.

This module never trains a model, never computes feature importance,
never performs feature selection, and never constructs a second
`FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder`/
`DatasetQualificationEngine` -- it only reads an already-built
`ResearchDatasetManifest` and its durable artifacts through the
existing `features.manifests.ResearchDatasetStore`, exactly like
`quant_platform.qualification`."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import FeatureDiscoveryRequestError
from quant_platform.feature_discovery.diagnostics import (
    compute_feature_signal_diagnostics,
    compute_shared_discovery_facts,
)
from quant_platform.feature_discovery.evidence import (
    FEATURE_DISCOVERY_DIMENSION_ORDER,
    FeatureDiscoveryDimensionKind,
    FeatureDiscoveryEvidence,
)
from quant_platform.feature_discovery.models import (
    FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION,
    FeatureDiscoveryReport,
    FeatureDiscoverySummary,
    FeatureSignalDiagnostics,
    compute_feature_set_id,
)
from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

__all__ = ["FeatureDiscoveryEngine"]


def _resolve_target_names(manifest: ResearchDatasetManifest, feature_names: frozenset[str] | None) -> tuple[str, ...]:
    if feature_names is None:
        return manifest.feature_names
    unknown = feature_names - set(manifest.feature_names)
    if unknown:
        raise FeatureDiscoveryRequestError(
            f"requested feature_names not declared on this manifest: {sorted(unknown)}",
            context={"dataset_id": manifest.dataset_id, "unknown_feature_names": sorted(unknown)},
        )
    return tuple(name for name in manifest.feature_names if name in feature_names)


def _summarize(per_feature: tuple[FeatureSignalDiagnostics, ...]) -> FeatureDiscoverySummary:
    approved = flagged = blocked = 0
    for diagnostics in per_feature:
        if diagnostics.is_blocking:
            blocked += 1
        elif any(e.severity in (Severity.WARNING, Severity.CRITICAL) for e in diagnostics.all_evidence):
            flagged += 1
        else:
            approved += 1
    mean_score = sum(d.overall_score for d in per_feature) / len(per_feature) if per_feature else 0.0
    return FeatureDiscoverySummary(
        feature_count=len(per_feature), approved_count=approved, flagged_count=flagged, blocked_count=blocked, mean_overall_score=mean_score,
    )


def _dataset_level_warnings(per_feature: tuple[FeatureSignalDiagnostics, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    redundant_features = sorted({
        d.feature_name for d in per_feature
        if any("redundan" in e.finding.lower() or "duplicat" in e.finding.lower() for e in d.dimension_result(FeatureDiscoveryDimensionKind.REDUNDANCY).evidence if e.severity is not Severity.INFO)
    })
    if redundant_features:
        warnings.append(f"{len(redundant_features)} feature(s) have at least one redundancy finding: {redundant_features}")
    return tuple(warnings)


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryEngine:
    def discover(
        self, manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore, *, feature_names: frozenset[str] | None = None,
    ) -> FeatureDiscoveryReport:
        target_names = _resolve_target_names(manifest, feature_names)
        facts = compute_shared_discovery_facts(manifest, research_store)
        per_feature = tuple(compute_feature_signal_diagnostics(name, facts) for name in target_names)

        dimension_scores = {
            dimension.value: (sum(d.dimension_result(dimension).score for d in per_feature) / len(per_feature) if per_feature else 0.0)
            for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER
        }
        blocking_findings: tuple[FeatureDiscoveryEvidence, ...] = tuple(e for d in per_feature for e in d.blocking_evidence)
        recommendations = tuple(sorted({e.recommendation for d in per_feature for e in d.all_evidence if e.recommendation}))
        summary = _summarize(per_feature)
        feature_set_id = compute_feature_set_id(
            dataset_id=manifest.dataset_id, feature_registry_fingerprint=manifest.feature_registry_fingerprint, feature_names=target_names,
        )
        evaluation_time = format_utc_timestamp(utc_now())

        return FeatureDiscoveryReport(
            schema_version=FEATURE_DISCOVERY_REPORT_SCHEMA_VERSION, dataset_id=manifest.dataset_id, feature_set_id=feature_set_id,
            feature_count=len(target_names), evaluation_time=evaluation_time, summary=summary, per_feature_diagnostics=per_feature,
            dimension_scores=dimension_scores, warnings=_dataset_level_warnings(per_feature), blocking_findings=blocking_findings,
            recommendations=recommendations,
        )
