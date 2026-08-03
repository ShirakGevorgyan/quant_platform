"""`LabelValidationReconciliation` (Milestone 11, Phase 3, Part C):
compares two `engine.LabelQualificationReport`s for the SAME
`label_specification_id`, detecting the 5 named drift kinds: decision
drift, score drift, evidence drift, warning drift, distribution drift.
Mirrors `qualification.reconciliation`/`feature_discovery.
reconciliation`/`labels.reconciliation`'s established shape exactly:
reconciling two reports for DIFFERENT specifications is a structural
precondition violation (there is nothing to reconcile) and raises
`LabelValidationReconciliationError`; every other disagreement is a
normal, expected, non-raising `LabelValidationReconciliationIssue`."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import LabelValidationReconciliationError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.engine import LabelQualificationReport
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "LABEL_VALIDATION_RECONCILIATION_SCHEMA_VERSION",
    "LabelValidationReconciliation",
    "LabelValidationReconciliationIssue",
    "LabelValidationReconciliationResult",
]

LABEL_VALIDATION_RECONCILIATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelValidationReconciliationIssue:
    kind: str
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelValidationReconciliationIssue:
        return cls(kind=str(raw["kind"]), message=str(raw["message"]))


@dataclass(frozen=True, slots=True)
class LabelValidationReconciliationResult:
    schema_version: int
    label_specification_id: str
    baseline_decision: str
    candidate_decision: str
    reconciled: bool
    issues: tuple[LabelValidationReconciliationIssue, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "baseline_decision": self.baseline_decision, "candidate_decision": self.candidate_decision, "reconciled": self.reconciled,
            "issues": [i.to_json_dict() for i in self.issues], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelValidationReconciliationResult:
        require_schema_version(raw, supported=LABEL_VALIDATION_RECONCILIATION_SCHEMA_VERSION, context="LabelValidationReconciliationResult")
        return cls(
            schema_version=LABEL_VALIDATION_RECONCILIATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            baseline_decision=str(raw["baseline_decision"]), candidate_decision=str(raw["candidate_decision"]), reconciled=bool(raw["reconciled"]),
            issues=tuple(
                LabelValidationReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


class LabelValidationReconciliation:
    def reconcile(self, baseline: LabelQualificationReport, candidate: LabelQualificationReport, *, score_tolerance: float = 0.01) -> LabelValidationReconciliationResult:
        if baseline.label_specification_id != candidate.label_specification_id:
            raise LabelValidationReconciliationError(
                f"Cannot reconcile reports for different label_specification_id values: baseline={baseline.label_specification_id!r} "
                f"candidate={candidate.label_specification_id!r}",
                context={"baseline_label_specification_id": baseline.label_specification_id, "candidate_label_specification_id": candidate.label_specification_id},
            )

        issues: list[LabelValidationReconciliationIssue] = []

        if baseline.decision != candidate.decision:
            issues.append(LabelValidationReconciliationIssue(
                kind="decision_drift", message=f"decision changed: baseline={baseline.decision.value} candidate={candidate.decision.value}",
            ))

        score_delta = abs(baseline.diagnostics.overall_score - candidate.diagnostics.overall_score)
        if score_delta > score_tolerance:
            issues.append(LabelValidationReconciliationIssue(
                kind="score_drift",
                message=f"overall_score changed by {score_delta:.6f}: baseline={baseline.diagnostics.overall_score:.6f} candidate={candidate.diagnostics.overall_score:.6f}",
            ))

        baseline_evidence = {(r.dimension.value, e.finding, e.severity.value) for r in baseline.diagnostics.dimension_results for e in r.evidence}
        candidate_evidence = {(r.dimension.value, e.finding, e.severity.value) for r in candidate.diagnostics.dimension_results for e in r.evidence}
        if baseline_evidence != candidate_evidence:
            issues.append(LabelValidationReconciliationIssue(
                kind="evidence_drift",
                message=f"evidence changed: only_in_baseline={sorted(baseline_evidence - candidate_evidence)} only_in_candidate={sorted(candidate_evidence - baseline_evidence)}",
            ))

        baseline_warnings = {(d, f) for d, f, s in baseline_evidence if s == Severity.WARNING.value}
        candidate_warnings = {(d, f) for d, f, s in candidate_evidence if s == Severity.WARNING.value}
        if baseline_warnings != candidate_warnings:
            issues.append(LabelValidationReconciliationIssue(
                kind="warning_drift",
                message=f"warnings changed: only_in_baseline={sorted(baseline_warnings - candidate_warnings)} only_in_candidate={sorted(candidate_warnings - baseline_warnings)}",
            ))

        if baseline.diagnostics.distribution.class_ratios != candidate.diagnostics.distribution.class_ratios:
            issues.append(LabelValidationReconciliationIssue(
                kind="distribution_drift",
                message=f"class_ratios changed: baseline={baseline.diagnostics.distribution.class_ratios} candidate={candidate.diagnostics.distribution.class_ratios}",
            ))

        return LabelValidationReconciliationResult(
            schema_version=LABEL_VALIDATION_RECONCILIATION_SCHEMA_VERSION, label_specification_id=baseline.label_specification_id,
            baseline_decision=baseline.decision.value, candidate_decision=candidate.decision.value, reconciled=not issues,
            issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()),
        )
