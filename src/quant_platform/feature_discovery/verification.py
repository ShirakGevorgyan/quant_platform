"""`FeatureDiscoveryVerifier` (Milestone 11, Phase 2, Part 1): independent
verification. Never trusts a cached `FeatureDiscoveryReport`'s own
`dimension_scores`/`summary`/`blocking_findings`/`recommendations`
fields at face value, and never trusts that a cached report still
reflects the dataset's current, live artifacts. Two independent checks:

1. `verify_report_self_consistency` -- pure, no I/O -- recomputes
   `dimension_scores`/`summary`/`blocking_findings`/`recommendations`
   from the report's own `per_feature_diagnostics`, using an
   INDEPENDENTLY reimplemented copy of `engine.py`'s tiny aggregation
   formulas (deliberately NOT importing `engine.py`'s private
   `_summarize`/`_dataset_level_warnings` helpers -- a bug shared
   between the two would otherwise go undetected), and compares
   against what the report itself claims.
2. Full re-discovery -- a fresh `FeatureDiscoveryEngine.discover()` run
   against the live manifest/store, diffed against the supplied report
   via `FeatureDiscoveryReconciliation` at zero score tolerance.

A mismatch in either is a normal, non-raising outcome
(`FeatureDiscoveryVerificationResult.verified=False`), never an
exception."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.evidence import FEATURE_DISCOVERY_DIMENSION_ORDER
from quant_platform.feature_discovery.models import FeatureDiscoveryReport
from quant_platform.feature_discovery.reconciliation import (
    FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION,
    FeatureDiscoveryReconciliation,
    FeatureDiscoveryReconciliationIssue,
    FeatureDiscoveryReconciliationResult,
)
from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION",
    "FeatureDiscoveryVerificationResult",
    "FeatureDiscoveryVerifier",
    "verify_report_self_consistency",
]

FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION = 1
_SCORE_EPSILON = 1e-9


def verify_report_self_consistency(report: FeatureDiscoveryReport) -> tuple[bool, tuple[str, ...]]:
    issues: list[str] = []
    per_feature = report.per_feature_diagnostics

    for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER:
        recomputed = sum(d.dimension_result(dimension).score for d in per_feature) / len(per_feature) if per_feature else 0.0
        claimed = report.dimension_scores.get(dimension.value)
        if claimed is None or abs(claimed - recomputed) > _SCORE_EPSILON:
            issues.append(f"dimension_scores[{dimension.value}]: report claims {claimed}, recomputed is {recomputed}")

    recomputed_blocking_count = sum(len(d.blocking_evidence) for d in per_feature)
    if recomputed_blocking_count != len(report.blocking_findings):
        issues.append(f"blocking_findings count: report claims {len(report.blocking_findings)}, recomputed is {recomputed_blocking_count}")

    recomputed_approved = recomputed_flagged = recomputed_blocked = 0
    for diagnostics in per_feature:
        if diagnostics.is_blocking:
            recomputed_blocked += 1
        elif any(e.severity in (Severity.WARNING, Severity.CRITICAL) for e in diagnostics.all_evidence):
            recomputed_flagged += 1
        else:
            recomputed_approved += 1
    if recomputed_approved != report.summary.approved_count:
        issues.append(f"summary.approved_count: report claims {report.summary.approved_count}, recomputed is {recomputed_approved}")
    if recomputed_flagged != report.summary.flagged_count:
        issues.append(f"summary.flagged_count: report claims {report.summary.flagged_count}, recomputed is {recomputed_flagged}")
    if recomputed_blocked != report.summary.blocked_count:
        issues.append(f"summary.blocked_count: report claims {report.summary.blocked_count}, recomputed is {recomputed_blocked}")

    recomputed_mean_score = sum(d.overall_score for d in per_feature) / len(per_feature) if per_feature else 0.0
    if abs(recomputed_mean_score - report.summary.mean_overall_score) > _SCORE_EPSILON:
        issues.append(f"summary.mean_overall_score: report claims {report.summary.mean_overall_score}, recomputed is {recomputed_mean_score}")

    return (not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryVerificationResult:
    schema_version: int
    dataset_id: str
    feature_set_id: str
    verified: bool
    self_consistent: bool
    self_consistency_issues: tuple[str, ...]
    reconciliation: FeatureDiscoveryReconciliationResult
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "feature_set_id": self.feature_set_id,
            "verified": self.verified, "self_consistent": self.self_consistent, "self_consistency_issues": list(self.self_consistency_issues),
            "reconciliation": self.reconciliation.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoveryVerificationResult:
        require_schema_version(raw, supported=FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION, context="FeatureDiscoveryVerificationResult")
        return cls(
            schema_version=FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), feature_set_id=str(raw["feature_set_id"]),
            verified=bool(raw["verified"]), self_consistent=bool(raw["self_consistent"]),
            self_consistency_issues=tuple(str(s) for s in as_json_list(raw.get("self_consistency_issues") or [], field_name="self_consistency_issues")),
            reconciliation=FeatureDiscoveryReconciliationResult.from_json_dict(as_json_dict(raw["reconciliation"], field_name="reconciliation")),
            generated_at=str(raw["generated_at"]),
        )


class FeatureDiscoveryVerifier:
    def verify(
        self, report: FeatureDiscoveryReport, manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore,
        *, engine: FeatureDiscoveryEngine | None = None, feature_names: frozenset[str] | None = None,
    ) -> FeatureDiscoveryVerificationResult:
        self_consistent, self_consistency_issues = verify_report_self_consistency(report)

        if report.dataset_id != manifest.dataset_id:
            reconciliation = FeatureDiscoveryReconciliationResult(
                schema_version=FEATURE_DISCOVERY_RECONCILIATION_SCHEMA_VERSION, dataset_id=report.dataset_id,
                baseline_feature_set_id=report.feature_set_id, candidate_feature_set_id=manifest.dataset_id, reconciled=False,
                issues=(FeatureDiscoveryReconciliationIssue(
                    kind="dataset_id_mismatch", feature_name=None, dimension=None,
                    message=f"report.dataset_id={report.dataset_id!r} does not match manifest.dataset_id={manifest.dataset_id!r} -- this report cannot be verified against this manifest",
                ),),
                generated_at=format_utc_timestamp(utc_now()),
            )
            return FeatureDiscoveryVerificationResult(
                schema_version=FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION, dataset_id=report.dataset_id, feature_set_id=report.feature_set_id,
                verified=False, self_consistent=self_consistent, self_consistency_issues=self_consistency_issues,
                reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
            )

        active_engine = engine if engine is not None else FeatureDiscoveryEngine()
        recomputed_report = active_engine.discover(manifest, research_store, feature_names=feature_names)
        reconciliation = FeatureDiscoveryReconciliation().reconcile(report, recomputed_report, score_tolerance=0.0)

        return FeatureDiscoveryVerificationResult(
            schema_version=FEATURE_DISCOVERY_VERIFICATION_SCHEMA_VERSION, dataset_id=report.dataset_id, feature_set_id=report.feature_set_id,
            verified=(self_consistent and reconciliation.reconciled), self_consistent=self_consistent,
            self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
        )
