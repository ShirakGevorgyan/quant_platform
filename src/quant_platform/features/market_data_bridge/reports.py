"""Deterministic, plain-text reports (Milestone 10, Phase 4D, spec
Section 25) -- mirrors `features.lineage.render_lineage_report`'s own
established convention (stable, sorted ordering; one block per item;
never a dict-iteration-order-dependent rendering) for every bridge
artifact worth a human- and diff-friendly rendering."""

from __future__ import annotations

from quant_platform.features.market_data_bridge.bindings import (
    BaseAssetDatasetBinding,
    CrossAssetDatasetBinding,
    MacroDatasetBinding,
)
from quant_platform.features.market_data_bridge.coverage import CoverageReport
from quant_platform.features.market_data_bridge.rebuild_planner import RebuildPlan
from quant_platform.features.market_data_bridge.reconciliation import ReconciliationReport
from quant_platform.features.market_data_bridge.staleness import StalenessFinding

__all__ = [
    "render_binding_inventory_report",
    "render_coverage_report",
    "render_rebuild_plan_report",
    "render_reconciliation_report",
    "render_staleness_report",
]


def render_binding_inventory_report(
    *, base_binding: BaseAssetDatasetBinding, macro_bindings: dict[str, MacroDatasetBinding],
    cross_asset_bindings: dict[str, CrossAssetDatasetBinding],
) -> str:
    lines = ["Market-data source binding inventory", "=" * 60]
    lines.append(
        f"\nbase :: {base_binding.canonical_instrument_id} [{base_binding.provider}] {base_binding.timeframe.value}"
    )
    lines.append(f"  pinned_dataset_id: {base_binding.pinned_dataset_id}")
    lines.append(f"  binding_id: {base_binding.binding_id}")
    for name in sorted(macro_bindings):
        macro_binding = macro_bindings[name]
        lines.append(f"\nmacro:{name} :: {macro_binding.series_id} [{macro_binding.provider}] required={macro_binding.required}")
        lines.append(f"  component_manifest_id: {macro_binding.component_manifest_id}")
        lines.append(f"  revision_policy: {macro_binding.revision_policy_kind.value}")
        lines.append(f"  binding_id: {macro_binding.binding_id}")
    for name in sorted(cross_asset_bindings):
        cross_binding = cross_asset_bindings[name]
        lines.append(
            f"\ncross_asset:{name} :: {cross_binding.canonical_driver_id} "
            f"[{cross_binding.provider}:{cross_binding.provider_symbol}] form={cross_binding.instrument_form.value} "
            f"proxy={cross_binding.proxy_policy.is_proxy} required={cross_binding.required}"
        )
        lines.append(f"  component_manifest_id: {cross_binding.component_manifest_id}")
        lines.append(f"  binding_id: {cross_binding.binding_id}")
    return "\n".join(lines)


def render_coverage_report(report: CoverageReport) -> str:
    lines = [
        f"Source coverage report (trimmed={report.trimmed})", "=" * 60,
        f"safe_range: [{report.safe_start}, {report.safe_end})",
    ]
    if report.trim_reason:
        lines.append(f"trim_reason: {report.trim_reason}")
    for finding in sorted(report.findings, key=lambda f: (f.source_kind, f.source_name)):
        lines.append(
            f"\n[{finding.status}] {finding.source_kind}:{finding.source_name} required={finding.required} "
            f"fraction={finding.coverage_fraction:.3f} covered=[{finding.covered_start}, {finding.covered_end}]"
        )
    return "\n".join(lines)


def render_staleness_report(findings: list[StalenessFinding]) -> str:
    lines = ["Staleness report", "=" * 60]
    for finding in sorted(findings, key=lambda f: (f.source_kind, f.source_name)):
        lines.append(
            f"\n{finding.source_kind}:{finding.source_name} rows={finding.total_row_count} "
            f"unavailable={finding.unavailable_row_count} stale={finding.stale_row_count} "
            f"fraction={finding.stale_fraction:.3f} max_age_s={finding.max_observed_age_seconds} "
            f"threshold_s={finding.threshold_seconds}"
        )
    return "\n".join(lines)


def render_reconciliation_report(report: ReconciliationReport) -> str:
    lines = [f"Reconciliation report (clean={report.is_clean}, {len(report.issues)} issue(s))", "=" * 60]
    for issue in sorted(report.issues, key=lambda i: (i.code.value, i.message)):
        lines.append(f"  [{issue.code.value}] {issue.message}")
    return "\n".join(lines)


def render_rebuild_plan_report(plan: RebuildPlan) -> str:
    lines = [
        f"Incremental rebuild plan: {plan.kind.value}", "=" * 60,
        f"plan_id: {plan.plan_id}", f"expected_output_dataset_id: {plan.expected_output_dataset_id}",
        f"required_warmup_from: {plan.required_warmup_from}",
        f"affected_source_names: {list(plan.affected_source_names)}",
        f"reason_codes: {list(plan.reason_codes)}",
    ]
    return "\n".join(lines)
