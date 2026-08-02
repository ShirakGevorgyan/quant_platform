"""Deterministic, plain-text reports (Milestone 11, Phase 1) -- mirrors
`features.market_data_bridge.reports`'s own established convention
(stable, sorted ordering; one block per item; never a dict-iteration-
order-dependent rendering) for every qualification artifact worth a
human- and diff-friendly rendering. Renders ALREADY-COMPUTED objects --
this module performs no qualification logic of its own.

Part 1 built the Qualification/Diagnostics/Reconciliation reports. Part
2 adds the remaining 4 the spec names: the Verification Report
(`IndependentVerificationResult`), the Evidence Report (every `Evidence`
record across a `QualificationDiagnostics`' 6 sections, grouped by
dimension), the Blocking Failure Report and the Recommendation Report
(both views over a single `DatasetQualificationReport`'s
`dimension_results` -- everything they show is already in the
Qualification Report too, just filtered down to one concern so a caller
who only wants "what's blocking this dataset" or "what should I do
about it" isn't parsing the full report)."""

from __future__ import annotations

from quant_platform.qualification.diagnostics import QualificationDiagnostics
from quant_platform.qualification.evidence import Evidence
from quant_platform.qualification.models import QUALIFICATION_DIMENSION_ORDER, DatasetQualificationReport
from quant_platform.qualification.reconciliation import QualificationReconciliationResult
from quant_platform.qualification.verification import IndependentVerificationResult

__all__ = [
    "render_blocking_failure_report",
    "render_dataset_qualification_report",
    "render_evidence_report",
    "render_independent_verification_report",
    "render_qualification_diagnostics",
    "render_qualification_reconciliation",
    "render_recommendation_report",
]


def render_dataset_qualification_report(report: DatasetQualificationReport) -> str:
    lines = [
        f"Dataset qualification report: {report.dataset_id} v{report.version} ({report.content_id})", "=" * 60,
        f"decision: {report.decision.decision.value} (score={report.decision.overall_score:.4f})",
        f"reason: {report.decision.decision_reason}", f"generated_at: {report.generated_at}",
    ]
    for dimension in QUALIFICATION_DIMENSION_ORDER:
        result = report.dimension_result(dimension)
        lines.append(f"\n[{dimension.value}] score={result.score:.4f} blocking={len(result.blocking_failures)}")
        for finding in result.findings:
            lines.append(f"  finding: {finding}")
        for warning in result.warnings:
            lines.append(f"  warning: {warning}")
        for failure in result.blocking_failures:
            lines.append(f"  BLOCKING [{failure.code.value}]: {failure.message}")
        for recommendation in result.recommendations:
            lines.append(f"  recommendation: {recommendation}")
    return "\n".join(lines)


_EVIDENCE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("structural_evidence", "Structural"), ("temporal_evidence", "Temporal"), ("statistical_evidence", "Statistical"),
    ("coverage_evidence", "Coverage"), ("stability_evidence", "Stability"), ("safety_evidence", "Safety"),
)


def render_qualification_diagnostics(diagnostics: QualificationDiagnostics) -> str:
    lines = [
        f"Qualification diagnostics: {diagnostics.dataset_id} v{diagnostics.version} ({diagnostics.content_id})", "=" * 60,
        f"decision: {diagnostics.decision} (overall_score={diagnostics.overall_score:.4f})", "\ndimension scores:",
    ]
    for name in sorted(diagnostics.dimension_scores):
        lines.append(f"  {name}: {diagnostics.dimension_scores[name]:.4f}")
    lines.append("\nsplits:")
    for split in sorted(diagnostics.split_diagnostics, key=lambda s: s.split_name):
        lines.append(f"  {split.split_name}: rows={split.row_count} span=[{split.open_time_min}, {split.open_time_max}]")
        for feature in sorted(split.feature_null_fractions):
            fraction = split.feature_null_fractions[feature]
            if fraction > 0:
                lines.append(f"    null_fraction[{feature}]={fraction:.4f}")
    for field_name, label in _EVIDENCE_SECTIONS:
        section: tuple[Evidence, ...] = getattr(diagnostics, field_name)
        lines.append(f"\n{label} diagnostics ({len(section)} finding(s)):")
        for evidence in sorted(section, key=lambda e: (e.severity.value, e.finding)):
            lines.append(f"  [{evidence.severity.value}] {evidence.finding}")
    return "\n".join(lines)


def render_qualification_reconciliation(result: QualificationReconciliationResult) -> str:
    lines = [
        f"Qualification reconciliation: {result.dataset_id} (reconciled={result.reconciled}, {len(result.issues)} issue(s))", "=" * 60,
        f"baseline: v{result.baseline_version} ({result.baseline_content_id})",
        f"candidate: v{result.candidate_version} ({result.candidate_content_id})",
    ]
    for issue in sorted(result.issues, key=lambda i: (i.kind, i.dimension or "", i.message)):
        dimension_prefix = f"{issue.dimension}: " if issue.dimension else ""
        lines.append(f"  [{issue.kind}] {dimension_prefix}{issue.message}")
    return "\n".join(lines)


def render_independent_verification_report(result: IndependentVerificationResult) -> str:
    lines = [
        f"Independent verification report: {result.dataset_id} v{result.version} ({result.content_id})", "=" * 60,
        f"verified: {result.verified}", f"self_consistent: {result.self_consistent}",
    ]
    for self_consistency_issue in result.self_consistency_issues:
        lines.append(f"  self-consistency issue: {self_consistency_issue}")
    lines.append(f"reconciliation.reconciled: {result.reconciliation.reconciled} ({len(result.reconciliation.issues)} issue(s))")
    for reconciliation_issue in sorted(result.reconciliation.issues, key=lambda i: (i.kind, i.dimension or "", i.message)):
        dimension_prefix = f"{reconciliation_issue.dimension}: " if reconciliation_issue.dimension else ""
        lines.append(f"  [{reconciliation_issue.kind}] {dimension_prefix}{reconciliation_issue.message}")
    lines.append(f"generated_at: {result.generated_at}")
    return "\n".join(lines)


def render_evidence_report(diagnostics: QualificationDiagnostics) -> str:
    lines = [
        f"Evidence report: {diagnostics.dataset_id} v{diagnostics.version} ({diagnostics.content_id})", "=" * 60,
        f"{len(diagnostics.all_evidence)} evidence record(s) across 6 diagnostic section(s)",
    ]
    for field_name, label in _EVIDENCE_SECTIONS:
        section: tuple[Evidence, ...] = getattr(diagnostics, field_name)
        lines.append(f"\n{label} ({len(section)} record(s)):")
        for evidence in sorted(section, key=lambda e: (e.severity.value, e.finding)):
            lines.append(f"  [{evidence.severity.value}] {evidence.finding} (blocking={evidence.blocking})")
            for item in evidence.evidence:
                lines.append(f"      evidence: {item}")
            for artifact in evidence.affected_artifacts:
                lines.append(f"      affected: {artifact}")
            if evidence.recommendation:
                lines.append(f"      recommendation: {evidence.recommendation}")
    return "\n".join(lines)


def render_blocking_failure_report(report: DatasetQualificationReport) -> str:
    failures = report.all_blocking_failures
    lines = [
        f"Blocking failure report: {report.dataset_id} v{report.version} ({report.content_id})", "=" * 60,
        f"decision: {report.decision.decision.value} ({len(failures)} blocking failure(s))",
    ]
    for failure in sorted(failures, key=lambda f: (f.dimension.value, f.code.value, f.message)):
        lines.append(f"  [{failure.dimension.value}:{failure.code.value}] {failure.message}")
    return "\n".join(lines)


def render_recommendation_report(report: DatasetQualificationReport) -> str:
    entries = [(dimension.value, recommendation) for dimension in QUALIFICATION_DIMENSION_ORDER for recommendation in report.dimension_result(dimension).recommendations]
    lines = [
        f"Recommendation report: {report.dataset_id} v{report.version} ({report.content_id})", "=" * 60,
        f"{len(entries)} recommendation(s)",
    ]
    for dimension_value, recommendation in sorted(entries):
        lines.append(f"  [{dimension_value}] {recommendation}")
    return "\n".join(lines)
