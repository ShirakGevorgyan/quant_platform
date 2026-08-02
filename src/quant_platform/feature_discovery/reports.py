"""`FeatureDiscoveryReports` (Milestone 11, Phase 2, Part 1): deterministic,
plain-text reports -- mirrors `qualification.reports`/`features.
market_data_bridge.reports`'s own established convention (stable, sorted
ordering; one block per item; never a dict-iteration-order-dependent
rendering). Renders ALREADY-COMPUTED objects only -- this module
performs no discovery logic of its own."""

from __future__ import annotations

from quant_platform.feature_discovery.evidence import FEATURE_DISCOVERY_DIMENSION_ORDER
from quant_platform.feature_discovery.models import FeatureDiscoveryReport, FeatureSignalDiagnostics
from quant_platform.feature_discovery.reconciliation import FeatureDiscoveryReconciliationResult
from quant_platform.feature_discovery.verification import FeatureDiscoveryVerificationResult

__all__ = [
    "render_blocking_findings_report",
    "render_feature_discovery_reconciliation",
    "render_feature_discovery_report",
    "render_feature_discovery_verification",
    "render_feature_signal_diagnostics",
    "render_recommendations_report",
]


def render_feature_signal_diagnostics(diagnostics: FeatureSignalDiagnostics) -> str:
    lines = [
        f"Feature signal diagnostics: {diagnostics.feature_name} (overall_score={diagnostics.overall_score:.4f})", "=" * 60,
    ]
    for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER:
        result = diagnostics.dimension_result(dimension)
        lines.append(f"\n[{dimension.value}] score={result.score:.4f} blocking={result.is_blocking}")
        for evidence in sorted(result.evidence, key=lambda e: (e.severity.value, e.finding)):
            marker = "BLOCKING " if evidence.blocking else ""
            lines.append(f"  [{marker}{evidence.severity.value}] {evidence.finding}")
            if evidence.recommendation:
                lines.append(f"      recommendation: {evidence.recommendation}")
    return "\n".join(lines)


def render_feature_discovery_report(report: FeatureDiscoveryReport) -> str:
    lines = [
        f"Feature discovery report: {report.dataset_id} feature_set={report.feature_set_id} ({report.feature_count} feature(s))", "=" * 60,
        f"evaluation_time: {report.evaluation_time}",
        f"summary: approved={report.summary.approved_count} flagged={report.summary.flagged_count} blocked={report.summary.blocked_count} "
        f"mean_overall_score={report.summary.mean_overall_score:.4f}",
        "\ndimension scores (mean across every evaluated feature):",
    ]
    for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER:
        lines.append(f"  {dimension.value}: {report.dimension_scores.get(dimension.value, 0.0):.4f}")
    if report.warnings:
        lines.append("\ndataset-level warnings:")
        for warning in report.warnings:
            lines.append(f"  {warning}")
    lines.append(f"\n{len(report.blocking_findings)} blocking finding(s) across the feature set.")
    lines.append("\nper-feature summary (sorted by feature name):")
    for diagnostics in sorted(report.per_feature_diagnostics, key=lambda d: d.feature_name):
        lines.append(f"  {diagnostics.feature_name}: overall_score={diagnostics.overall_score:.4f} blocking={diagnostics.is_blocking}")
    return "\n".join(lines)


def render_blocking_findings_report(report: FeatureDiscoveryReport) -> str:
    lines = [
        f"Blocking findings report: {report.dataset_id} feature_set={report.feature_set_id} ({len(report.blocking_findings)} finding(s))", "=" * 60,
    ]
    for finding in sorted(report.blocking_findings, key=lambda e: (e.affected_feature, e.dimension.value, e.finding)):
        code = finding.blocking_code.value if finding.blocking_code else "unknown"
        lines.append(f"  [{finding.affected_feature}:{finding.dimension.value}:{code}] {finding.finding}")
    return "\n".join(lines)


def render_recommendations_report(report: FeatureDiscoveryReport) -> str:
    lines = [
        f"Recommendations report: {report.dataset_id} feature_set={report.feature_set_id} ({len(report.recommendations)} recommendation(s))", "=" * 60,
    ]
    for recommendation in sorted(report.recommendations):
        lines.append(f"  {recommendation}")
    return "\n".join(lines)


def render_feature_discovery_reconciliation(result: FeatureDiscoveryReconciliationResult) -> str:
    lines = [
        f"Feature discovery reconciliation: {result.dataset_id} (reconciled={result.reconciled}, {len(result.issues)} issue(s))", "=" * 60,
        f"baseline_feature_set_id: {result.baseline_feature_set_id}", f"candidate_feature_set_id: {result.candidate_feature_set_id}",
    ]
    for issue in sorted(result.issues, key=lambda i: (i.kind, i.feature_name or "", i.dimension or "", i.message)):
        feature_prefix = f"{issue.feature_name}: " if issue.feature_name else ""
        dimension_prefix = f"{issue.dimension}: " if issue.dimension else ""
        lines.append(f"  [{issue.kind}] {feature_prefix}{dimension_prefix}{issue.message}")
    return "\n".join(lines)


def render_feature_discovery_verification(result: FeatureDiscoveryVerificationResult) -> str:
    lines = [
        f"Feature discovery verification: {result.dataset_id} feature_set={result.feature_set_id}", "=" * 60,
        f"verified: {result.verified}", f"self_consistent: {result.self_consistent}",
    ]
    for self_consistency_issue in result.self_consistency_issues:
        lines.append(f"  self-consistency issue: {self_consistency_issue}")
    lines.append(f"reconciliation.reconciled: {result.reconciliation.reconciled} ({len(result.reconciliation.issues)} issue(s))")
    for reconciliation_issue in sorted(result.reconciliation.issues, key=lambda i: (i.kind, i.feature_name or "", i.dimension or "", i.message)):
        feature_prefix = f"{reconciliation_issue.feature_name}: " if reconciliation_issue.feature_name else ""
        dimension_prefix = f"{reconciliation_issue.dimension}: " if reconciliation_issue.dimension else ""
        lines.append(f"  [{reconciliation_issue.kind}] {feature_prefix}{dimension_prefix}{reconciliation_issue.message}")
    lines.append(f"generated_at: {result.generated_at}")
    return "\n".join(lines)
