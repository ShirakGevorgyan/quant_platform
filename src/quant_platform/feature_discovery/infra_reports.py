"""Deterministic, plain-text infrastructure reports (Milestone 11,
Phase 2, Part 2) -- mirrors `qualification.reports`/`feature_discovery.
reports`'s (Part 1) own established convention (stable, sorted
ordering; one block per item; never a dict-iteration-order-dependent
rendering). Renders ALREADY-COMPUTED objects only -- this module
performs no discovery/verification/reconciliation logic of its own.

The 7 named reports: Feature Catalog Report, Feature Inventory Report,
Dependency Report, Health Report, Metadata Report, Verification
Report, Reconciliation Report."""

from __future__ import annotations

from quant_platform.feature_discovery.catalog import FeatureCatalog, FeatureInventory
from quant_platform.feature_discovery.graph import FeatureDependencyGraph
from quant_platform.feature_discovery.health import FeatureHealthReport
from quant_platform.feature_discovery.infra_reconciliation import FeatureInfrastructureReconciliationResult
from quant_platform.feature_discovery.infra_verification import FeatureInfrastructureVerificationResult
from quant_platform.feature_discovery.metadata import FeatureMetadata

__all__ = [
    "render_dependency_report",
    "render_feature_catalog_report",
    "render_feature_inventory_report",
    "render_health_report",
    "render_infrastructure_reconciliation_report",
    "render_infrastructure_verification_report",
    "render_metadata_report",
]


def render_feature_catalog_report(catalog: FeatureCatalog) -> str:
    lines = [f"Feature catalog report: {catalog.dataset_id} ({len(catalog.entries)} feature(s))", "=" * 60]
    for entry in catalog.entries:
        lines.append(f"\n{entry.feature_name} ({entry.feature_id})  [{entry.feature_group}]")
        lines.append(f"  creation_stage: {entry.creation_stage}")
        lines.append(f"  availability_rule: {entry.availability_rule}")
        lines.append(f"  warmup_requirement: {entry.warmup_requirement}")
        if entry.dependencies:
            lines.append(f"  dependencies: {list(entry.dependencies)}")
        lines.append(f"  deterministic_identity: {entry.deterministic_identity[:16]}...")
    return "\n".join(lines)


def render_feature_inventory_report(inventory: FeatureInventory) -> str:
    lines = [f"Feature inventory report: {inventory.dataset_id} ({len(inventory.complete_catalog.entries)} feature(s))", "=" * 60]
    for label, view in (
        ("grouped (by feature_group)", inventory.grouped_catalog), ("by origin dataset", inventory.dataset_catalog),
        ("by creation stage", inventory.origin_catalog), ("by availability rule", inventory.availability_catalog),
    ):
        lines.append(f"\n{label}:")
        for key, names in sorted(view.items()):
            lines.append(f"  {key}: {list(names)}")
    return "\n".join(lines)


def render_dependency_report(graph: FeatureDependencyGraph) -> str:
    lines = [
        f"Dependency report: {graph.dataset_id} ({len(graph.nodes)} node(s), {len(graph.edges)} edge(s))", "=" * 60,
        f"is_valid: {graph.is_valid}",
    ]
    for kind_label, kind_value in (("raw_source", "raw_source"), ("market_data", "market_data"), ("derived_feature", "derived_feature"), ("higher_order_feature", "higher_order_feature")):
        matching = sorted(n.label for n in graph.nodes if n.kind.value == kind_value)
        if matching:
            lines.append(f"\n{kind_label} nodes: {matching}")
    if graph.cycles:
        lines.append(f"\nCYCLES ({len(graph.cycles)}):")
        for cycle in graph.cycles:
            lines.append(f"  {' -> '.join(cycle)}")
    if graph.missing_parents:
        lines.append(f"\nMISSING PARENTS ({len(graph.missing_parents)}):")
        for feature_name, missing in sorted(graph.missing_parents):
            lines.append(f"  {feature_name} -> {missing}")
    if graph.duplicate_derivations:
        lines.append(f"\nduplicate derivations ({len(graph.duplicate_derivations)}):")
        for a, b in sorted(graph.duplicate_derivations):
            lines.append(f"  {a} ~ {b}")
    if graph.orphan_features:
        lines.append(f"\norphan features: {list(graph.orphan_features)}")
    return "\n".join(lines)


def render_health_report(health_reports: tuple[FeatureHealthReport, ...]) -> str:
    healthy = sum(1 for h in health_reports if h.is_healthy)
    lines = [f"Health report: {len(health_reports)} feature(s), {healthy} healthy, {len(health_reports) - healthy} unhealthy", "=" * 60]
    for health in sorted(health_reports, key=lambda h: h.feature_name):
        lines.append(f"\n{health.feature_name}: healthy={health.is_healthy} overall_score={health.signal_diagnostics.overall_score:.4f}")
        if not health.lineage_present:
            lines.append(f"  lineage: MISSING fields={list(health.missing_lineage_fields)}")
        if health.signal_diagnostics.is_blocking:
            for evidence in health.signal_diagnostics.blocking_evidence:
                lines.append(f"  BLOCKING [{evidence.dimension.value}:{evidence.blocking_code.value if evidence.blocking_code else '?'}]: {evidence.finding}")
    return "\n".join(lines)


def render_metadata_report(metadata: tuple[FeatureMetadata, ...]) -> str:
    lines = [f"Metadata report: {len(metadata)} feature(s)", "=" * 60]
    for entry in sorted(metadata, key=lambda m: m.feature_name):
        lines.append(f"\n{entry.feature_name}:")
        lines.append(f"  feature_id={entry.feature_id} feature_group={entry.feature_group} creation_stage={entry.creation_stage}")
        lines.append(f"  origin_dataset={entry.origin_dataset} origin_manifest={entry.origin_manifest}")
        lines.append(f"  availability_rule={entry.availability_rule!r} warmup_requirement={entry.warmup_requirement}")
        lines.append(f"  dependencies={list(entry.dependencies)}")
        lines.append(f"  deterministic_identity={entry.deterministic_identity}")
    return "\n".join(lines)


def render_infrastructure_verification_report(result: FeatureInfrastructureVerificationResult) -> str:
    lines = [
        f"Infrastructure verification report: {result.dataset_id} manifest_id={result.manifest_id}", "=" * 60,
        f"verified: {result.verified}", f"self_consistent: {result.self_consistent}",
    ]
    for issue in result.self_consistency_issues:
        lines.append(f"  self-consistency issue: {issue}")
    lines.append(f"reconciliation.reconciled: {result.reconciliation.reconciled} ({len(result.reconciliation.issues)} issue(s))")
    for reconciliation_issue in sorted(result.reconciliation.issues, key=lambda i: (i.kind, i.feature_name or "", i.message)):
        feature_prefix = f"{reconciliation_issue.feature_name}: " if reconciliation_issue.feature_name else ""
        lines.append(f"  [{reconciliation_issue.kind}] {feature_prefix}{reconciliation_issue.message}")
    lines.append(f"generated_at: {result.generated_at}")
    return "\n".join(lines)


def render_infrastructure_reconciliation_report(result: FeatureInfrastructureReconciliationResult) -> str:
    lines = [
        f"Infrastructure reconciliation report: {result.dataset_id} (reconciled={result.reconciled}, {len(result.issues)} issue(s))", "=" * 60,
        f"baseline_manifest_id: {result.baseline_manifest_id}", f"candidate_manifest_id: {result.candidate_manifest_id}",
    ]
    for issue in sorted(result.issues, key=lambda i: (i.kind, i.feature_name or "", i.message)):
        feature_prefix = f"{issue.feature_name}: " if issue.feature_name else ""
        lines.append(f"  [{issue.kind}] {feature_prefix}{issue.message}")
    return "\n".join(lines)
