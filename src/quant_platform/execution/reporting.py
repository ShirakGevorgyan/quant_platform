"""Human-readable and machine-readable summaries of an `ExecutionManifest`
(Milestone 4B, Section 14) -- the execution-engine analogue of
`ml.reporting`'s `build_report_json`/`render_report_markdown`, same
"deterministic JSON + companion Markdown, no dashboard" scope and same
"pure function of already-loaded data" design: neither function performs
I/O. `ExecutionManifest` only stores content-hash `ArtifactReference`s,
not the full `AggregatedExecutionResult`/`Timeline` content -- a caller
wanting those expanded fetches them (e.g. via `MLArtifactStore.
read_artifact`) and passes them in explicitly.
"""

from __future__ import annotations

from quant_platform.execution.manifests import ExecutionManifest
from quant_platform.execution.results import AggregatedExecutionResult
from quant_platform.execution.timeline import Timeline

_SCHEMA_VERSION = 1

_STANDARD_LIMITATIONS = (
    "This report describes EXECUTION of a walk-forward fold plan only: fit/predict ran against the "
    "registry's model definition, exercising the full pipeline end-to-end. `metrics` on every fold is a "
    "reserved placeholder -- no real performance score is computed here, and no profitability or "
    "predictive-validity claim is made or implied.",
)


def build_execution_report_json(
    execution_manifest: ExecutionManifest, *, aggregate: AggregatedExecutionResult | None = None,
    timeline: Timeline | None = None,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment_id": execution_manifest.experiment_id,
        "stage": execution_manifest.stage.value,
        "fold_plan_strategy": execution_manifest.fold_plan_strategy,
        "total_folds": execution_manifest.total_folds,
        "completed_fold_indices": list(execution_manifest.completed_fold_indices),
        "failed_fold_indices": list(execution_manifest.failed_fold_indices),
        "current_fold_index": execution_manifest.current_fold_index,
        "resume_count": execution_manifest.resume_count,
        "created_at": execution_manifest.created_at,
        "updated_at": execution_manifest.updated_at,
        "completed_at": execution_manifest.completed_at,
        "failure_summary": execution_manifest.failure_summary,
        "artifact_references": [
            {"category": a.category.value, "content_hash": a.content_hash, "size_bytes": a.size_bytes}
            for a in execution_manifest.artifact_references
        ],
        "aggregate": (None if aggregate is None else {
            "total_folds": aggregate.total_folds,
            "completed_fold_indices": list(aggregate.completed_fold_indices),
            "failed_fold_indices": list(aggregate.failed_fold_indices),
            "overall_status": aggregate.overall_status.value,
            "started_at": aggregate.started_at, "completed_at": aggregate.completed_at,
            "execution_duration_seconds": aggregate.execution_duration_seconds,
            "resume_count": aggregate.resume_count,
        }),
        "timeline": (None if timeline is None else [e.to_json_dict() for e in timeline.entries]),
        "limitations": list(_STANDARD_LIMITATIONS),
    }


def render_execution_report_markdown(
    execution_manifest: ExecutionManifest, *, aggregate: AggregatedExecutionResult | None = None,
    timeline: Timeline | None = None,
) -> str:
    lines = [
        "# Execution Report",
        "",
        f"- **Experiment ID:** `{execution_manifest.experiment_id}`",
        f"- **Stage:** `{execution_manifest.stage.value}`",
        f"- **Created at:** {execution_manifest.created_at}",
        f"- **Updated at:** {execution_manifest.updated_at}",
    ]
    if execution_manifest.completed_at is not None:
        lines.append(f"- **Completed at:** {execution_manifest.completed_at}")
    lines += [
        "",
        "## Status History",
        f"- fold plan strategy: `{execution_manifest.fold_plan_strategy}`",
        f"- total folds: {execution_manifest.total_folds}",
        f"- completed folds: {list(execution_manifest.completed_fold_indices)}",
        f"- failed folds: {list(execution_manifest.failed_fold_indices)}",
        "",
        "## Resume History",
        f"- resume_count: {execution_manifest.resume_count}",
    ]
    lines += ["", "## Duration"]
    if aggregate is None:
        lines.append("- (aggregate result not loaded -- pass `aggregate=` to expand)")
    else:
        lines.append(f"- overall_status: `{aggregate.overall_status.value}`")
        lines.append(f"- execution_duration_seconds: {aggregate.execution_duration_seconds:.3f}")
    lines += ["", "## Timeline"]
    if timeline is None:
        lines.append("- (timeline not loaded -- pass `timeline=` to expand)")
    else:
        lines.append("| Fold | Train | Test | Status |")
        lines.append("| --- | --- | --- | --- |")
        for e in timeline.entries:
            lines.append(f"| {e.fold_index} | {e.train_start} .. {e.train_end} | {e.test_start} .. {e.test_end} | {e.status or '-'} |")
    if execution_manifest.failure_summary:
        lines += ["", "## Failure Summary", execution_manifest.failure_summary]
    lines += ["", "## Limitations"]
    for limitation in _STANDARD_LIMITATIONS:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = ["build_execution_report_json", "render_execution_report_markdown"]
