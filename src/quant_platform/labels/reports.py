"""Deterministic, plain-text label reports (Milestone 11, Phase 3, Part
A) -- mirrors `qualification.reports`/`feature_discovery.reports`'s
established convention (stable, sorted ordering; one block per item;
never a dict-iteration-order-dependent rendering). Renders ALREADY-
COMPUTED objects only -- this module performs no discovery/
verification/reconciliation logic of its own.

7 named reports: Specification Report, Manifest Report, Bundle Report,
Diagnostics Report, Verification Report, Reconciliation Report, Version
History Report."""

from __future__ import annotations

from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.diagnostics import LabelDiagnostics
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.models import LabelSpecification
from quant_platform.labels.reconciliation import LabelReconciliationResult
from quant_platform.labels.verification import LabelVerificationResult
from quant_platform.labels.versioning import LabelVersionHistory

__all__ = [
    "render_bundle_report",
    "render_diagnostics_report",
    "render_manifest_report",
    "render_reconciliation_report",
    "render_specification_report",
    "render_verification_report",
    "render_version_history_report",
]


def render_specification_report(specification: LabelSpecification) -> str:
    lines = [
        f"Label specification report: {specification.label_specification_id}", "=" * 60,
        f"label_family: {specification.label_family.value}", f"generation_version: {specification.generation_version}",
        f"parameter_hash: {specification.parameter_hash}", f"price_basis: {specification.price_basis}",
        f"prediction_horizon: {specification.prediction_horizon}", f"availability_rule: {specification.availability_rule}",
        f"reference_price: {specification.reference_price}", f"event_time_rule: {specification.event_time_rule}",
        f"generation_rule: {specification.generation_rule}", f"identity_algorithm: {specification.identity_algorithm}",
        f"created_from_dataset: {specification.created_from_dataset}", f"created_from_manifest: {specification.created_from_manifest}",
        f"parameters: {specification.parameters}",
    ]
    return "\n".join(lines)


def render_manifest_report(manifest: LabelManifest) -> str:
    lines = [
        f"Label manifest report: {manifest.label_specification_id} (checksum={manifest.manifest_checksum[:16]}...)", "=" * 60,
        f"dataset_identity: {manifest.dataset_identity}", f"manifest_identity: {manifest.manifest_identity}",
        f"feature_identity: {manifest.feature_identity}", f"qualification_identity: {manifest.qualification_identity}",
        f"availability_semantics: {manifest.availability_semantics}", f"generation_timestamp: {manifest.generation_timestamp}",
        f"generation_parameters: {manifest.generation_parameters}", "dependency_chain:",
    ]
    lines.extend(f"  {step}" for step in manifest.dependency_chain)
    return "\n".join(lines)


def render_bundle_report(bundle: LabelBundle) -> str:
    lines = [
        f"Label bundle report: {bundle.specification.label_specification_id}", "=" * 60,
        f"content_id: {bundle.identity.content_id}", f"source_content_id: {bundle.identity.source_content_id}",
        f"row_count: {bundle.row_count}", f"valid_count: {bundle.valid_count}",
        f"missing_count: {bundle.row_count - bundle.valid_count}", f"generated_at: {bundle.generated_at}",
    ]
    return "\n".join(lines)


def render_diagnostics_report(diagnostics: LabelDiagnostics) -> str:
    lines = [
        f"Label diagnostics report: {diagnostics.label_specification_id} (overall_score={diagnostics.overall_score:.4f})", "=" * 60,
    ]
    for result in diagnostics.dimension_results:
        lines.append(f"\n{result.dimension.value}: score={result.score:.4f}")
        for evidence in result.evidence:
            marker = "BLOCKING" if evidence.blocking else evidence.severity.value
            lines.append(f"  [{marker}] {evidence.finding}")
            for fact in evidence.evidence:
                lines.append(f"    - {fact}")
    return "\n".join(lines)


def render_verification_report(result: LabelVerificationResult) -> str:
    lines = [
        f"Label verification report: {result.label_specification_id}", "=" * 60,
        f"verified: {result.verified}", f"self_consistent: {result.self_consistent}",
    ]
    for consistency_issue in result.self_consistency_issues:
        lines.append(f"  self-consistency issue: {consistency_issue}")
    lines.append(f"reconciliation.reconciled: {result.reconciliation.reconciled} ({len(result.reconciliation.issues)} issue(s))")
    for reconciliation_issue in sorted(result.reconciliation.issues, key=lambda i: (i.kind, i.message)):
        lines.append(f"  [{reconciliation_issue.kind}] {reconciliation_issue.message}")
    lines.append(f"generated_at: {result.generated_at}")
    return "\n".join(lines)


def render_reconciliation_report(result: LabelReconciliationResult) -> str:
    lines = [
        f"Label reconciliation report: {result.label_specification_id} (reconciled={result.reconciled}, {len(result.issues)} issue(s))",
        "=" * 60, f"baseline_content_id: {result.baseline_content_id}", f"candidate_content_id: {result.candidate_content_id}",
    ]
    for issue in sorted(result.issues, key=lambda i: (i.kind, i.message)):
        lines.append(f"  [{issue.kind}] {issue.message}")
    return "\n".join(lines)


def render_version_history_report(history: LabelVersionHistory) -> str:
    lines = [f"Label version history report: {history.label_family} ({len(history.versions)} version(s))", "=" * 60]
    for version in history.versions:
        lines.append(f"\ngeneration_version: {version.generation_version}")
        lines.append(f"  label_specification_id: {version.label_specification_id}")
        lines.append(f"  parameter_hash: {version.parameter_hash}")
        lines.append(f"  registered_at: {version.registered_at}")
    return "\n".join(lines)
