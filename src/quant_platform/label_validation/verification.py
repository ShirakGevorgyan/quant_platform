"""`LabelValidationVerifier` (Milestone 11, Phase 3, Part C): independent
verification of a `engine.LabelQualificationReport`. Never trusts a
cached report. Two checks, mirroring `qualification.verification`/
`feature_discovery.verification`/`labels.verification`'s established
pattern exactly:

1. `verify_report_self_consistency` -- pure, no I/O -- independently
   recomputes the decision and `overall_score` from the report's own
   `diagnostics.dimension_results`, using a DELIBERATELY separate
   reimplementation of `engine._decide`'s tiny rule (never importing
   the private function, so a shared bug between the two would not go
   undetected).
2. Full re-qualification -- a fresh `engine.LabelQualificationEngine.
   qualify()` run against the live bundle/manifest, diffed against the
   supplied report via `reconciliation.LabelValidationReconciliation`."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelValidationError, LabelValidationVerificationError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.engine import (
    LabelQualificationDecision,
    LabelQualificationEngine,
    LabelQualificationReport,
)
from quant_platform.label_validation.reconciliation import (
    LabelValidationReconciliation,
    LabelValidationReconciliationResult,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.records import LabelRecord
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_VALIDATION_VERIFICATION_SCHEMA_VERSION", "LabelValidationVerificationResult", "LabelValidationVerifier", "verify_report_self_consistency"]

LABEL_VALIDATION_VERIFICATION_SCHEMA_VERSION = 1


def _independent_decide(report: LabelQualificationReport) -> LabelQualificationDecision:
    all_evidence = tuple(e for r in report.diagnostics.dimension_results for e in r.evidence)
    if any(e.blocking for e in all_evidence):
        return LabelQualificationDecision.REJECTED
    if any(e.severity in (Severity.WARNING, Severity.CRITICAL) for e in all_evidence):
        return LabelQualificationDecision.CONDITIONALLY_APPROVED
    return LabelQualificationDecision.APPROVED


def verify_report_self_consistency(report: LabelQualificationReport) -> tuple[bool, tuple[str, ...]]:
    issues: list[str] = []

    recomputed_decision = _independent_decide(report)
    if recomputed_decision != report.decision:
        issues.append(f"decision: report claims {report.decision.value!r}, recomputed is {recomputed_decision.value!r}")

    dimension_results = report.diagnostics.dimension_results
    recomputed_overall_score = sum(r.score for r in dimension_results) / len(dimension_results) if dimension_results else 0.0
    if abs(recomputed_overall_score - report.diagnostics.overall_score) > 1e-9:
        issues.append(f"overall_score: report claims {report.diagnostics.overall_score!r}, recomputed is {recomputed_overall_score!r}")

    return (not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class LabelValidationVerificationResult:
    schema_version: int
    label_specification_id: str
    verified: bool
    self_consistent: bool
    self_consistency_issues: tuple[str, ...]
    reconciliation: LabelValidationReconciliationResult
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "verified": self.verified,
            "self_consistent": self.self_consistent, "self_consistency_issues": list(self.self_consistency_issues),
            "reconciliation": self.reconciliation.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelValidationVerificationResult:
        require_schema_version(raw, supported=LABEL_VALIDATION_VERIFICATION_SCHEMA_VERSION, context="LabelValidationVerificationResult")
        return cls(
            schema_version=LABEL_VALIDATION_VERIFICATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            verified=bool(raw["verified"]), self_consistent=bool(raw["self_consistent"]),
            self_consistency_issues=tuple(str(s) for s in as_json_list(raw.get("self_consistency_issues") or [], field_name="self_consistency_issues")),
            reconciliation=LabelValidationReconciliationResult.from_json_dict(as_json_dict(raw["reconciliation"], field_name="reconciliation")),
            generated_at=str(raw["generated_at"]),
        )


class LabelValidationVerifier:
    def verify(
        self, report: LabelQualificationReport, bundle: LabelBundle, manifest: LabelManifest, *, drift_baseline: LabelBundle | None = None,
        regime_assignment: pd.Series | None = None, records: tuple[LabelRecord, ...] | None = None,
    ) -> LabelValidationVerificationResult:
        self_consistent, self_consistency_issues = verify_report_self_consistency(report)

        try:
            fresh_report = LabelQualificationEngine().qualify(bundle, manifest, drift_baseline=drift_baseline, regime_assignment=regime_assignment, records=records)
            reconciliation = LabelValidationReconciliation().reconcile(report, fresh_report)
        except LabelValidationError as exc:
            raise LabelValidationVerificationError(
                f"LabelValidationVerifier.verify could not complete re-qualification for label_specification_id={report.label_specification_id!r}: {exc}",
                context={"label_specification_id": report.label_specification_id},
            ) from exc

        return LabelValidationVerificationResult(
            schema_version=LABEL_VALIDATION_VERIFICATION_SCHEMA_VERSION, label_specification_id=report.label_specification_id,
            verified=(self_consistent and reconciliation.reconciled), self_consistent=self_consistent,
            self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
        )
