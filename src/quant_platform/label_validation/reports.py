"""Deterministic, plain-text label-validation reports (Milestone 11,
Phase 3, Part C) -- mirrors `qualification.reports`/`feature_discovery.
reports`/`labels.reports`'s established convention (stable, sorted
ordering; one block per item; never a dict-iteration-order-dependent
rendering). Renders ALREADY-COMPUTED objects only -- this module
performs no qualification/verification/reconciliation logic of its own.

7 named reports: Diagnostics Report, Qualification Report, Verification
Report, Reconciliation Report, Replay Report, Horizon Report, Overlap
Report."""

from __future__ import annotations

from quant_platform.label_validation.diagnostics import LabelDiagnostics
from quant_platform.label_validation.engine import LabelQualificationReport
from quant_platform.label_validation.horizon import HorizonComparisonReport
from quant_platform.label_validation.overlap import OverlapReport
from quant_platform.label_validation.reconciliation import LabelValidationReconciliationResult
from quant_platform.label_validation.replay import LabelValidationReplayResult
from quant_platform.label_validation.verification import LabelValidationVerificationResult

__all__ = [
    "render_diagnostics_report",
    "render_horizon_report",
    "render_overlap_report",
    "render_qualification_report",
    "render_reconciliation_report",
    "render_replay_report",
    "render_verification_report",
]


def render_diagnostics_report(diagnostics: LabelDiagnostics) -> str:
    lines = [f"Label diagnostics report: {diagnostics.label_specification_id} (overall_score={diagnostics.overall_score:.4f})", "=" * 60]
    for result in diagnostics.dimension_results:
        lines.append(f"\n{result.dimension.value}: score={result.score:.4f}")
        for evidence in result.evidence:
            marker = "BLOCKING" if evidence.blocking else evidence.severity.value
            lines.append(f"  [{marker}] {evidence.finding}")
            for fact in evidence.evidence:
                lines.append(f"    - {fact}")
    return "\n".join(lines)


def render_qualification_report(report: LabelQualificationReport) -> str:
    lines = [
        f"Label qualification report: {report.label_specification_id}", "=" * 60, f"decision: {report.decision.value}",
        f"overall_score: {report.diagnostics.overall_score:.4f}", f"qualified_at: {report.qualified_at}",
    ]
    if report.blocking_reasons:
        lines.append(f"\nblocking reasons ({len(report.blocking_reasons)}):")
        for reason in report.blocking_reasons:
            lines.append(f"  {reason}")
    return "\n".join(lines)


def render_verification_report(result: LabelValidationVerificationResult) -> str:
    lines = [
        f"Label validation verification report: {result.label_specification_id}", "=" * 60,
        f"verified: {result.verified}", f"self_consistent: {result.self_consistent}",
    ]
    for issue in result.self_consistency_issues:
        lines.append(f"  self-consistency issue: {issue}")
    lines.append(f"reconciliation.reconciled: {result.reconciliation.reconciled} ({len(result.reconciliation.issues)} issue(s))")
    for reconciliation_issue in sorted(result.reconciliation.issues, key=lambda i: (i.kind, i.message)):
        lines.append(f"  [{reconciliation_issue.kind}] {reconciliation_issue.message}")
    lines.append(f"generated_at: {result.generated_at}")
    return "\n".join(lines)


def render_reconciliation_report(result: LabelValidationReconciliationResult) -> str:
    lines = [
        f"Label validation reconciliation report: {result.label_specification_id} (reconciled={result.reconciled}, {len(result.issues)} issue(s))",
        "=" * 60, f"baseline_decision: {result.baseline_decision}", f"candidate_decision: {result.candidate_decision}",
    ]
    for issue in sorted(result.issues, key=lambda i: (i.kind, i.message)):
        lines.append(f"  [{issue.kind}] {issue.message}")
    return "\n".join(lines)


def render_replay_report(result: LabelValidationReplayResult) -> str:
    lines = [
        f"Label validation replay report: {result.label_specification_id}", "=" * 60,
        f"qualification_identical: {result.qualification_identical}", f"original_decision: {result.original_decision}",
        f"replayed_decision: {result.replayed_decision}",
    ]
    for issue in result.issues:
        lines.append(f"  issue: {issue}")
    return "\n".join(lines)


def render_horizon_report(report: HorizonComparisonReport) -> str:
    lines = [f"Horizon comparison report: {report.label_family} ({len(report.horizons)} horizon(s))", "=" * 60]
    for horizon in report.horizons:
        lines.append(f"\nhorizon_bars={horizon.horizon_bars} ({horizon.label_specification_id})")
        lines.append(f"  coverage_fraction={horizon.coverage_fraction:.4f} is_degenerate={horizon.is_degenerate}")
        lines.append(f"  imbalance_ratio={horizon.imbalance_ratio} window_stability_score={horizon.window_stability_score}")
    return "\n".join(lines)


def render_overlap_report(report: OverlapReport) -> str:
    lines = [f"Overlap report ({len(report.findings)} finding(s))", "=" * 60]
    for finding in sorted(report.findings, key=lambda f: (f.kind, f.label_specification_id_a, f.label_specification_id_b)):
        lines.append(f"  [{finding.kind}] {finding.label_specification_id_a} ~ {finding.label_specification_id_b}: {finding.message}")
    return "\n".join(lines)
