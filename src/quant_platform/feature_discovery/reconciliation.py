"""`FeatureDiscoveryReconciliation` (Milestone 11, Phase 2, Part 1):
compares two `FeatureDiscoveryReport`s for the SAME `dataset_id` --
typically a freshly re-run candidate against a previously stored
baseline, or two reports covering different (possibly overlapping)
feature subsets -- and surfaces every disagreement as a structured,
non-raising issue.

Reconciling two reports for different `dataset_id`s is a structural
precondition violation (there is nothing to reconcile) and raises
`FeatureDiscoveryReconciliationError`; every other disagreement --
the two reports covering different feature sets, or any per-feature,
per-dimension score/finding/warning/recommendation/evidence drift -- is
a normal, expected, non-raising `FeatureDiscoveryReconciliationIssue`."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import FeatureDiscoveryReconciliationError
from quant_platform.feature_discovery.evidence import FEATURE_DISCOVERY_DIMENSION_ORDER
from quant_platform.feature_discovery.models import FeatureDiscoveryReport
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION",
    "FeatureDiscoveryReconciliation",
    "FeatureDiscoveryReconciliationIssue",
    "FeatureDiscoveryReconciliationResult",
]

FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION = 1
_DEFAULT_SCORE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryReconciliationIssue:
    kind: str
    feature_name: str | None
    dimension: str | None
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "feature_name": self.feature_name, "dimension": self.dimension, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoveryReconciliationIssue:
        return cls(
            kind=str(raw["kind"]), feature_name=(None if raw.get("feature_name") is None else str(raw["feature_name"])),
            dimension=(None if raw.get("dimension") is None else str(raw["dimension"])), message=str(raw["message"]),
        )


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryReconciliationResult:
    schema_version: int
    dataset_id: str
    baseline_feature_set_id: str
    candidate_feature_set_id: str
    reconciled: bool
    issues: tuple[FeatureDiscoveryReconciliationIssue, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "baseline_feature_set_id": self.baseline_feature_set_id,
            "candidate_feature_set_id": self.candidate_feature_set_id, "reconciled": self.reconciled,
            "issues": [i.to_json_dict() for i in self.issues], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoveryReconciliationResult:
        require_schema_version(raw, supported=FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION, context="FeatureDiscoveryReconciliationResult")
        return cls(
            schema_version=FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            baseline_feature_set_id=str(raw["baseline_feature_set_id"]), candidate_feature_set_id=str(raw["candidate_feature_set_id"]),
            reconciled=bool(raw["reconciled"]),
            issues=tuple(
                FeatureDiscoveryReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


class FeatureDiscoveryReconciliation:
    def reconcile(
        self, baseline: FeatureDiscoveryReport, candidate: FeatureDiscoveryReport, *, score_tolerance: float = _DEFAULT_SCORE_TOLERANCE,
    ) -> FeatureDiscoveryReconciliationResult:
        if baseline.dataset_id != candidate.dataset_id:
            raise FeatureDiscoveryReconciliationError(
                f"Cannot reconcile reports for different dataset_id values: baseline={baseline.dataset_id!r} candidate={candidate.dataset_id!r}",
                context={"baseline_dataset_id": baseline.dataset_id, "candidate_dataset_id": candidate.dataset_id},
            )

        issues: list[FeatureDiscoveryReconciliationIssue] = []
        baseline_by_name = {d.feature_name: d for d in baseline.per_feature_diagnostics}
        candidate_by_name = {d.feature_name: d for d in candidate.per_feature_diagnostics}
        only_in_baseline = sorted(set(baseline_by_name) - set(candidate_by_name))
        only_in_candidate = sorted(set(candidate_by_name) - set(baseline_by_name))
        if only_in_baseline or only_in_candidate:
            issues.append(FeatureDiscoveryReconciliationIssue(
                kind="feature_set_drift", feature_name=None, dimension=None,
                message=f"feature sets differ: only_in_baseline={only_in_baseline} only_in_candidate={only_in_candidate}",
            ))

        for feature_name in sorted(set(baseline_by_name) & set(candidate_by_name)):
            baseline_diag, candidate_diag = baseline_by_name[feature_name], candidate_by_name[feature_name]

            overall_delta = abs(baseline_diag.overall_score - candidate_diag.overall_score)
            if overall_delta > score_tolerance:
                issues.append(FeatureDiscoveryReconciliationIssue(
                    kind="score_drift", feature_name=feature_name, dimension=None,
                    message=f"overall_score drifted by {overall_delta:.4f} (baseline={baseline_diag.overall_score:.4f}, candidate={candidate_diag.overall_score:.4f})",
                ))

            for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER:
                baseline_result = baseline_diag.dimension_result(dimension)
                candidate_result = candidate_diag.dimension_result(dimension)

                dimension_delta = abs(baseline_result.score - candidate_result.score)
                if dimension_delta > score_tolerance:
                    issues.append(FeatureDiscoveryReconciliationIssue(
                        kind="score_drift", feature_name=feature_name, dimension=dimension.value,
                        message=f"{dimension.value} score drifted by {dimension_delta:.4f} (baseline={baseline_result.score:.4f}, candidate={candidate_result.score:.4f})",
                    ))

                baseline_findings = frozenset(e.finding for e in baseline_result.evidence)
                candidate_findings = frozenset(e.finding for e in candidate_result.evidence)
                if baseline_findings != candidate_findings:
                    issues.append(FeatureDiscoveryReconciliationIssue(
                        kind="finding_drift", feature_name=feature_name, dimension=dimension.value,
                        message=f"{dimension.value} findings changed: baseline={sorted(baseline_findings)} candidate={sorted(candidate_findings)}",
                    ))

                baseline_warnings = frozenset(e.finding for e in baseline_result.evidence if e.severity in (Severity.WARNING, Severity.CRITICAL))
                candidate_warnings = frozenset(e.finding for e in candidate_result.evidence if e.severity in (Severity.WARNING, Severity.CRITICAL))
                if baseline_warnings != candidate_warnings:
                    issues.append(FeatureDiscoveryReconciliationIssue(
                        kind="warning_drift", feature_name=feature_name, dimension=dimension.value,
                        message=f"{dimension.value} warnings changed: baseline={sorted(baseline_warnings)} candidate={sorted(candidate_warnings)}",
                    ))

                baseline_recommendations = frozenset(e.recommendation for e in baseline_result.evidence if e.recommendation)
                candidate_recommendations = frozenset(e.recommendation for e in candidate_result.evidence if e.recommendation)
                if baseline_recommendations != candidate_recommendations:
                    issues.append(FeatureDiscoveryReconciliationIssue(
                        kind="recommendation_drift", feature_name=feature_name, dimension=dimension.value,
                        message=f"{dimension.value} recommendations changed: baseline={sorted(baseline_recommendations)} candidate={sorted(candidate_recommendations)}",
                    ))

                baseline_blocking = frozenset((e.blocking_code.value if e.blocking_code else "") for e in baseline_result.blocking_evidence)
                candidate_blocking = frozenset((e.blocking_code.value if e.blocking_code else "") for e in candidate_result.blocking_evidence)
                if baseline_blocking != candidate_blocking:
                    issues.append(FeatureDiscoveryReconciliationIssue(
                        kind="evidence_drift", feature_name=feature_name, dimension=dimension.value,
                        message=f"{dimension.value} blocking evidence codes changed: baseline={sorted(baseline_blocking)} candidate={sorted(candidate_blocking)}",
                    ))

        return FeatureDiscoveryReconciliationResult(
            schema_version=FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION, dataset_id=baseline.dataset_id,
            baseline_feature_set_id=baseline.feature_set_id, candidate_feature_set_id=candidate.feature_set_id,
            reconciled=not issues, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()),
        )
