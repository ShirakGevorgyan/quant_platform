"""`QualificationReconciliation` (Milestone 11, Phase 1): compares two
`DatasetQualificationReport`s for the SAME `dataset_id` -- typically a
freshly re-run candidate report against a previously stored baseline
report over the same manifest, or over two different `version`/
`content_id`s of the same recipe -- and surfaces any disagreement as
structured `ReconciliationIssue`s.

This is the report-level counterpart to `QualificationVerifier`'s
artifact-level determinism/reproducibility checks: `QualificationVerifier`
asks "does THIS ONE report's own facts agree with themselves"; this
module asks "do TWO INDEPENDENT qualification runs over the same dataset
agree with EACH OTHER" -- the practical question a caller asks before
trusting a cached/stored report instead of re-running
`DatasetQualificationEngine.qualify` from scratch.

Reconciling two reports for different `dataset_id`s is a structural
precondition violation (there is nothing to reconcile) and raises
`QualificationReconciliationError`; every other disagreement -- a
different decision, a dimension score drifting beyond `score_tolerance`,
a different set of blocking-failure codes, or a different set of
findings/warnings/recommendations on some dimension -- is a normal,
expected, non-raising `ReconciliationIssue` finding.

Milestone 11 Phase 1, Part 2 additionally diffs each dimension's
`findings`/`warnings`/`recommendations` (as sets, order-independent):
`warning_drift`/`recommendation_drift` for any dimension, and
`finding_drift` for any dimension EXCEPT `REPRODUCIBILITY`, whose own
finding drift is reported as `lineage_drift` instead -- the spec names
"lineage drift" as its own category, and `REPRODUCIBILITY` is the one
dimension `dimensions.py`'s own module docstring documents as owning
lineage (`MISSING_LINEAGE -> Reproducibility`), so its finding text is
where a genuine lineage change (e.g. `source_historical_dataset_id`
changing) would actually show up.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import QualificationReconciliationError
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.qualification.models import (
    QUALIFICATION_DIMENSION_ORDER,
    DatasetQualificationReport,
    QualificationDimensionKind,
)

__all__ = [
    "QUALIFICATION_RECONCILIATION_SCHEMA_VERSION",
    "QualificationReconciliation",
    "QualificationReconciliationResult",
    "ReconciliationIssue",
]

QUALIFICATION_RECONCILIATION_SCHEMA_VERSION = 1
_DEFAULT_SCORE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    kind: str
    dimension: str | None
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "dimension": self.dimension, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReconciliationIssue:
        return cls(
            kind=str(raw["kind"]), dimension=(None if raw.get("dimension") is None else str(raw["dimension"])),
            message=str(raw["message"]),
        )


@dataclass(frozen=True, slots=True)
class QualificationReconciliationResult:
    schema_version: int
    dataset_id: str
    baseline_version: str
    candidate_version: str
    baseline_content_id: str
    candidate_content_id: str
    reconciled: bool
    issues: tuple[ReconciliationIssue, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version, "baseline_content_id": self.baseline_content_id,
            "candidate_content_id": self.candidate_content_id, "reconciled": self.reconciled,
            "issues": [i.to_json_dict() for i in self.issues], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QualificationReconciliationResult:
        require_schema_version(raw, supported=QUALIFICATION_RECONCILIATION_SCHEMA_VERSION, context="QualificationReconciliationResult")
        return cls(
            schema_version=QUALIFICATION_RECONCILIATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            baseline_version=str(raw["baseline_version"]), candidate_version=str(raw["candidate_version"]),
            baseline_content_id=str(raw["baseline_content_id"]), candidate_content_id=str(raw["candidate_content_id"]),
            reconciled=bool(raw["reconciled"]),
            issues=tuple(
                ReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


class QualificationReconciliation:
    def reconcile(
        self, baseline: DatasetQualificationReport, candidate: DatasetQualificationReport, *, score_tolerance: float = _DEFAULT_SCORE_TOLERANCE,
    ) -> QualificationReconciliationResult:
        if baseline.dataset_id != candidate.dataset_id:
            raise QualificationReconciliationError(
                f"Cannot reconcile reports for different dataset_id values: baseline={baseline.dataset_id!r} candidate={candidate.dataset_id!r}",
                context={"baseline_dataset_id": baseline.dataset_id, "candidate_dataset_id": candidate.dataset_id},
            )

        issues: list[ReconciliationIssue] = []
        if baseline.decision.decision is not candidate.decision.decision:
            issues.append(ReconciliationIssue(
                kind="decision_mismatch", dimension=None,
                message=f"baseline decision {baseline.decision.decision.value!r} != candidate decision {candidate.decision.decision.value!r}",
            ))

        for dimension in QUALIFICATION_DIMENSION_ORDER:
            baseline_result = baseline.dimension_result(dimension)
            candidate_result = candidate.dimension_result(dimension)

            score_delta = abs(baseline_result.score - candidate_result.score)
            if score_delta > score_tolerance:
                issues.append(ReconciliationIssue(
                    kind="dimension_score_drift", dimension=dimension.value,
                    message=(
                        f"{dimension.value} score drifted by {score_delta:.4f} "
                        f"(baseline={baseline_result.score:.4f}, candidate={candidate_result.score:.4f}, tolerance={score_tolerance:.4f})"
                    ),
                ))

            baseline_codes = frozenset(f.code.value for f in baseline_result.blocking_failures)
            candidate_codes = frozenset(f.code.value for f in candidate_result.blocking_failures)
            if baseline_codes != candidate_codes:
                issues.append(ReconciliationIssue(
                    kind="blocking_failure_set_changed", dimension=dimension.value,
                    message=f"{dimension.value} blocking failure codes changed: baseline={sorted(baseline_codes)} candidate={sorted(candidate_codes)}",
                ))

            baseline_findings = frozenset(baseline_result.findings)
            candidate_findings = frozenset(candidate_result.findings)
            if baseline_findings != candidate_findings:
                kind = "lineage_drift" if dimension is QualificationDimensionKind.REPRODUCIBILITY else "finding_drift"
                issues.append(ReconciliationIssue(
                    kind=kind, dimension=dimension.value,
                    message=f"{dimension.value} findings changed: baseline={sorted(baseline_findings)} candidate={sorted(candidate_findings)}",
                ))

            baseline_warnings = frozenset(baseline_result.warnings)
            candidate_warnings = frozenset(candidate_result.warnings)
            if baseline_warnings != candidate_warnings:
                issues.append(ReconciliationIssue(
                    kind="warning_drift", dimension=dimension.value,
                    message=f"{dimension.value} warnings changed: baseline={sorted(baseline_warnings)} candidate={sorted(candidate_warnings)}",
                ))

            baseline_recommendations = frozenset(baseline_result.recommendations)
            candidate_recommendations = frozenset(candidate_result.recommendations)
            if baseline_recommendations != candidate_recommendations:
                issues.append(ReconciliationIssue(
                    kind="recommendation_drift", dimension=dimension.value,
                    message=f"{dimension.value} recommendations changed: baseline={sorted(baseline_recommendations)} candidate={sorted(candidate_recommendations)}",
                ))

        return QualificationReconciliationResult(
            schema_version=QUALIFICATION_RECONCILIATION_SCHEMA_VERSION, dataset_id=baseline.dataset_id,
            baseline_version=baseline.version, candidate_version=candidate.version,
            baseline_content_id=baseline.content_id, candidate_content_id=candidate.content_id,
            reconciled=not issues, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()),
        )
