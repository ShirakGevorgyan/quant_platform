"""Independent qualification verification (Milestone 11, Phase 1, Part
2). Answers a different question than Part 1's `QualificationVerifier`
(which computes the raw FACTS -- identity/artifact/lineage/leakage --
that `dimensions.py`'s evaluators consume): given an ALREADY-PRODUCED
`DatasetQualificationReport` (e.g. one loaded back from a persisted JSON
file, or handed to a caller by some other component), can it be
TRUSTED, or might it be stale, hand-edited, or the product of a bug?

This module never trusts:
  - the report's own `decision.decision`/`decision.overall_score`/
    `decision.blocking_failure_count` fields at face value (they are
    independently RECOMPUTED from the report's own `dimension_results`
    and compared -- `verify_report_self_consistency`, pure, no I/O);
  - the report as a whole being still an accurate description of the
    dataset's current, live artifacts (the dataset is independently
    RE-QUALIFIED from scratch via a fresh `DatasetQualificationEngine.
    qualify()` call against the live manifest/store, then diffed against
    the supplied report using Part 1's own `QualificationReconciliation`
    at zero score tolerance -- reused, not reimplemented).

A mismatch in either check is a NORMAL, expected, non-raising outcome
(`IndependentVerificationResult.verified=False`) -- exactly like a
`ReconciliationIssue`, never an exception. `QualificationVerificationError`
is reserved, as in Part 1, for genuinely being unable to attempt the
checks at all (e.g. the live artifacts are unreadable) -- it is allowed
to propagate from the inner `engine.qualify()` call unchanged."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.models import (
    DatasetQualificationReport,
    QualificationDecisionKind,
)
from quant_platform.qualification.reconciliation import (
    QUALIFICATION_RECONCILIATION_SCHEMA_VERSION,
    QualificationReconciliation,
    QualificationReconciliationResult,
    ReconciliationIssue,
)

__all__ = [
    "INDEPENDENT_VERIFICATION_SCHEMA_VERSION",
    "IndependentVerificationResult",
    "QualificationIndependentVerifier",
    "verify_report_self_consistency",
]

INDEPENDENT_VERIFICATION_SCHEMA_VERSION = 1
_SCORE_EPSILON = 1e-9


def verify_report_self_consistency(report: DatasetQualificationReport) -> tuple[bool, tuple[str, ...]]:
    """Pure, no I/O: recomputes `overall_score`/`blocking_failure_count`/
    `decision` from `report.dimension_results` alone, using an
    INDEPENDENTLY reimplemented copy of `engine.py`'s tiny 2-tier
    decision rule (never importing `engine._decide`, which is private
    and unexported for exactly this reason -- a bug shared between the
    two implementations would otherwise go undetected), and compares
    against what `report.decision` itself claims."""
    issues: list[str] = []

    recomputed_score = sum(r.score for r in report.dimension_results) / len(report.dimension_results)
    if abs(recomputed_score - report.decision.overall_score) > _SCORE_EPSILON:
        issues.append(f"overall_score: report claims {report.decision.overall_score}, recomputed from dimension_results is {recomputed_score}")

    recomputed_blocking = [f for r in report.dimension_results for f in r.blocking_failures]
    if len(recomputed_blocking) != report.decision.blocking_failure_count:
        issues.append(f"blocking_failure_count: report claims {report.decision.blocking_failure_count}, recomputed from dimension_results is {len(recomputed_blocking)}")

    recomputed_decision = QualificationDecisionKind.REJECTED_FOR_RESEARCH if recomputed_blocking else QualificationDecisionKind.APPROVED_FOR_RESEARCH
    if recomputed_decision is not report.decision.decision:
        issues.append(f"decision: report claims {report.decision.decision.value!r}, recomputed from dimension_results is {recomputed_decision.value!r}")

    return (not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class IndependentVerificationResult:
    schema_version: int
    dataset_id: str
    version: str
    content_id: str
    verified: bool
    self_consistent: bool
    self_consistency_issues: tuple[str, ...]
    reconciliation: QualificationReconciliationResult
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "version": self.version, "content_id": self.content_id,
            "verified": self.verified, "self_consistent": self.self_consistent, "self_consistency_issues": list(self.self_consistency_issues),
            "reconciliation": self.reconciliation.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> IndependentVerificationResult:
        require_schema_version(raw, supported=INDEPENDENT_VERIFICATION_SCHEMA_VERSION, context="IndependentVerificationResult")
        reconciliation_raw = as_json_dict(raw["reconciliation"], field_name="reconciliation")
        return cls(
            schema_version=INDEPENDENT_VERIFICATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), version=str(raw["version"]),
            content_id=str(raw["content_id"]), verified=bool(raw["verified"]), self_consistent=bool(raw["self_consistent"]),
            self_consistency_issues=tuple(str(s) for s in as_json_list(raw.get("self_consistency_issues") or [], field_name="self_consistency_issues")),
            reconciliation=QualificationReconciliationResult.from_json_dict(reconciliation_raw), generated_at=str(raw["generated_at"]),
        )


class QualificationIndependentVerifier:
    def verify(
        self, report: DatasetQualificationReport, manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore,
        *, engine: DatasetQualificationEngine | None = None, required_feature_names: frozenset[str] = frozenset(),
    ) -> IndependentVerificationResult:
        self_consistent, self_consistency_issues = verify_report_self_consistency(report)

        if report.dataset_id != manifest.dataset_id:
            reconciliation = QualificationReconciliationResult(
                schema_version=QUALIFICATION_RECONCILIATION_SCHEMA_VERSION, dataset_id=report.dataset_id,
                baseline_version=report.version, candidate_version=manifest.version, baseline_content_id=report.content_id,
                candidate_content_id=manifest.content_id, reconciled=False,
                issues=(ReconciliationIssue(
                    kind="dataset_id_mismatch", dimension=None,
                    message=f"report.dataset_id={report.dataset_id!r} does not match manifest.dataset_id={manifest.dataset_id!r} -- this report cannot be verified against this manifest",
                ),),
                generated_at=format_utc_timestamp(utc_now()),
            )
            return IndependentVerificationResult(
                schema_version=INDEPENDENT_VERIFICATION_SCHEMA_VERSION, dataset_id=report.dataset_id, version=report.version,
                content_id=report.content_id, verified=False, self_consistent=self_consistent,
                self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
            )

        active_engine = engine if engine is not None else DatasetQualificationEngine()
        recomputed_report = active_engine.qualify(manifest, research_store, required_feature_names=required_feature_names)
        reconciliation = QualificationReconciliation().reconcile(report, recomputed_report, score_tolerance=0.0)

        return IndependentVerificationResult(
            schema_version=INDEPENDENT_VERIFICATION_SCHEMA_VERSION, dataset_id=report.dataset_id, version=report.version,
            content_id=report.content_id, verified=(self_consistent and reconciliation.reconciled), self_consistent=self_consistent,
            self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
        )
