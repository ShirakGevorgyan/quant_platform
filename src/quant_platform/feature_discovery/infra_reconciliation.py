"""`FeatureInfrastructureReconciliation` (Milestone 11, Phase 2, Part
2): compares two `FeatureInfrastructureBundle`s (two inventories, two
manifests, two catalogs -- all bundled together, since they are built
from the same snapshot) for the SAME `dataset_id`, detecting the 5
named drift kinds: feature drift, metadata drift, dependency drift,
lineage drift, manifest drift.

Reconciling two bundles for different `dataset_id`s is a structural
precondition violation (there is nothing to reconcile) and raises
`FeatureDiscoveryReconciliationError`; every other disagreement is a
normal, expected, non-raising `FeatureInfrastructureReconciliationIssue`
-- exactly the same shape `qualification.reconciliation`/
`feature_discovery.reconciliation` (Part 1) already established."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import FeatureDiscoveryReconciliationError
from quant_platform.feature_discovery.catalog import FeatureInfrastructureBundle
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION",
    "FeatureInfrastructureReconciliation",
    "FeatureInfrastructureReconciliationIssue",
    "FeatureInfrastructureReconciliationResult",
]

FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureInfrastructureReconciliationIssue:
    kind: str
    feature_name: str | None
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "feature_name": self.feature_name, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureInfrastructureReconciliationIssue:
        return cls(kind=str(raw["kind"]), feature_name=(None if raw.get("feature_name") is None else str(raw["feature_name"])), message=str(raw["message"]))


@dataclass(frozen=True, slots=True)
class FeatureInfrastructureReconciliationResult:
    schema_version: int
    dataset_id: str
    baseline_manifest_id: str
    candidate_manifest_id: str
    reconciled: bool
    issues: tuple[FeatureInfrastructureReconciliationIssue, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "baseline_manifest_id": self.baseline_manifest_id,
            "candidate_manifest_id": self.candidate_manifest_id, "reconciled": self.reconciled,
            "issues": [i.to_json_dict() for i in self.issues], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureInfrastructureReconciliationResult:
        require_schema_version(raw, supported=FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION, context="FeatureInfrastructureReconciliationResult")
        return cls(
            schema_version=FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            baseline_manifest_id=str(raw["baseline_manifest_id"]), candidate_manifest_id=str(raw["candidate_manifest_id"]),
            reconciled=bool(raw["reconciled"]),
            issues=tuple(
                FeatureInfrastructureReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


class FeatureInfrastructureReconciliation:
    def reconcile(self, baseline: FeatureInfrastructureBundle, candidate: FeatureInfrastructureBundle) -> FeatureInfrastructureReconciliationResult:
        if baseline.snapshot.dataset_id != candidate.snapshot.dataset_id:
            raise FeatureDiscoveryReconciliationError(
                f"Cannot reconcile bundles for different dataset_id values: baseline={baseline.snapshot.dataset_id!r} "
                f"candidate={candidate.snapshot.dataset_id!r}",
                context={"baseline_dataset_id": baseline.snapshot.dataset_id, "candidate_dataset_id": candidate.snapshot.dataset_id},
            )

        issues: list[FeatureInfrastructureReconciliationIssue] = []
        baseline_names = {m.feature_name for m in baseline.snapshot.metadata}
        candidate_names = {m.feature_name for m in candidate.snapshot.metadata}
        if baseline_names != candidate_names:
            issues.append(FeatureInfrastructureReconciliationIssue(
                kind="feature_drift", feature_name=None,
                message=f"feature sets differ: only_in_baseline={sorted(baseline_names - candidate_names)} only_in_candidate={sorted(candidate_names - baseline_names)}",
            ))

        baseline_lineage_by_name = {ln.feature_name: ln for ln in baseline.snapshot.lineages}
        candidate_lineage_by_name = {ln.feature_name: ln for ln in candidate.snapshot.lineages}
        for name in sorted(baseline_names & candidate_names):
            baseline_metadata = baseline.catalog.entry(name)
            candidate_metadata = candidate.catalog.entry(name)
            if baseline_metadata != candidate_metadata:
                issues.append(FeatureInfrastructureReconciliationIssue(
                    kind="metadata_drift", feature_name=name, message=f"metadata changed: baseline={baseline_metadata} candidate={candidate_metadata}",
                ))
            baseline_lineage = baseline_lineage_by_name.get(name)
            candidate_lineage = candidate_lineage_by_name.get(name)
            if baseline_lineage != candidate_lineage:
                issues.append(FeatureInfrastructureReconciliationIssue(
                    kind="lineage_drift", feature_name=name, message=f"lineage changed: baseline={baseline_lineage} candidate={candidate_lineage}",
                ))

        if (baseline.graph.edges, baseline.graph.cycles, baseline.graph.missing_parents, baseline.graph.orphan_features) != (
            candidate.graph.edges, candidate.graph.cycles, candidate.graph.missing_parents, candidate.graph.orphan_features
        ):
            issues.append(FeatureInfrastructureReconciliationIssue(
                kind="dependency_drift", feature_name=None,
                message=(
                    f"dependency graph changed: baseline(edges={len(baseline.graph.edges)}, cycles={len(baseline.graph.cycles)}, "
                    f"missing_parents={len(baseline.graph.missing_parents)}) candidate(edges={len(candidate.graph.edges)}, "
                    f"cycles={len(candidate.graph.cycles)}, missing_parents={len(candidate.graph.missing_parents)})"
                ),
            ))

        if baseline.manifest.manifest_id != candidate.manifest.manifest_id:
            issues.append(FeatureInfrastructureReconciliationIssue(
                kind="manifest_drift", feature_name=None,
                message=f"manifest_id changed: baseline={baseline.manifest.manifest_id} candidate={candidate.manifest.manifest_id}",
            ))

        return FeatureInfrastructureReconciliationResult(
            schema_version=FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION, dataset_id=baseline.snapshot.dataset_id,
            baseline_manifest_id=baseline.manifest.manifest_id, candidate_manifest_id=candidate.manifest.manifest_id,
            reconciled=not issues, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()),
        )
