"""Human-readable and machine-readable summaries of an
`ExperimentManifest` -- no charts, no dashboard, just a deterministic
JSON summary (`build_report_json`) and a companion Markdown rendering
(`render_report_markdown`) of the same data, matching Section 15's
"reporting, not a dashboard" scope for this milestone.

PURE FUNCTIONS OF ALREADY-LOADED DATA
--------------------------------------------------------------------------
Neither function performs any I/O. `ExperimentManifest` itself only
stores a `validation_report_reference` (a content hash), not the
`ValidationReport`'s full content -- callers that want the validation
issue list expanded in the report must fetch it themselves (e.g. via
`MLArtifactStore.read_artifact` + `parse_json_strict`) and pass it in as
`validation_report`. This keeps reporting a pure, side-effect-free
summarization step, consistent with `ml/validation.py`'s own design.
"""

from __future__ import annotations

from quant_platform.ml.manifests import ExperimentManifest
from quant_platform.ml.models import ValidationReport

_SCHEMA_VERSION = 1

_STANDARD_LIMITATIONS = (
    "This report certifies experiment PREPARATION only: dataset/feature/label/model/seed/environment "
    "bindings were validated for consistency. No model was fitted, no prediction was produced, and no "
    "profitability or predictive-validity claim is made or implied.",
)


def build_report_json(manifest: ExperimentManifest, *, validation_report: ValidationReport | None = None) -> dict[str, object]:
    """Deterministic JSON summary -- every key present for every manifest
    (never conditionally omitted), so two reports are diff-able field by
    field regardless of status."""
    spec = manifest.spec
    validation_summary: dict[str, object]
    if validation_report is None:
        validation_summary = {
            "performed": manifest.validation_report_reference is not None,
            "is_ready": None,
            "critical_count": None, "error_count": None, "warning_count": None, "info_count": None,
            "report_content_hash": (
                None if manifest.validation_report_reference is None else manifest.validation_report_reference.content_hash
            ),
            "warnings": [],
        }
    else:
        validation_summary = {
            "performed": True,
            "is_ready": validation_report.is_ready,
            "critical_count": len(validation_report.criticals),
            "error_count": len(validation_report.errors),
            "warning_count": len(validation_report.warnings),
            "info_count": len(validation_report.infos),
            "report_content_hash": (
                None if manifest.validation_report_reference is None else manifest.validation_report_reference.content_hash
            ),
            "warnings": [f"[{i.code}] {i.message}" for i in validation_report.warnings],
        }

    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment_id": manifest.identity.experiment_id,
        "status": manifest.status.value,
        "dataset": {
            "dataset_id": spec.dataset_binding.dataset_id,
            "manifest_version": spec.dataset_binding.manifest_version,
            "content_id": spec.dataset_binding.content_id,
            "symbol": spec.dataset_binding.symbol,
            "base_timeframe": spec.dataset_binding.base_timeframe,
        },
        "model": {
            "name": spec.model_name, "version": spec.model_version,
            "definition_fingerprint": manifest.model_definition_fingerprint,
        },
        "objective": spec.objective.value,
        "feature_count": len(spec.feature_binding.feature_names),
        "label": {
            "name": spec.label_binding.name, "kind": spec.label_binding.kind,
            "horizon_bars": spec.label_binding.horizon_bars, "label_type": spec.label_binding.label_type.value,
        },
        "split": {"strategy": spec.split_binding.strategy},
        "preprocessing": {
            "fitted_preprocessing_fingerprint": spec.preprocessing_binding.fitted_preprocessing_fingerprint,
        },
        "seed_fingerprint": spec.seed_configuration.fingerprint(),
        "code_revision": {
            "revision": spec.code_revision_binding.revision, "source": spec.code_revision_binding.source,
            "is_dirty": spec.code_revision_binding.is_dirty,
        },
        "environment_summary": {
            "python_version": manifest.environment_snapshot.python_version,
            "platform_system": manifest.environment_snapshot.platform_system,
            "platform_release": manifest.environment_snapshot.platform_release,
            "architecture": manifest.environment_snapshot.architecture,
        },
        "validation": validation_summary,
        "created_at": manifest.created_at,
        "completed_at": manifest.completed_at,
        "failure_summary": manifest.failure_summary,
        "artifact_references": [
            {"category": a.category.value, "content_hash": a.content_hash, "size_bytes": a.size_bytes} for a in manifest.artifact_references
        ],
        "limitations": list(_STANDARD_LIMITATIONS) + list(manifest.limitations),
    }


def render_report_markdown(manifest: ExperimentManifest, *, validation_report: ValidationReport | None = None) -> str:
    """Renders directly from `manifest`/`spec`/`validation_report` (the
    same source data `build_report_json` summarizes) rather than by
    re-parsing that function's untyped JSON output."""
    spec = manifest.spec
    env = manifest.environment_snapshot
    lines = [
        "# Experiment Preparation Report",
        "",
        f"- **Experiment ID:** `{manifest.identity.experiment_id}`",
        f"- **Status:** `{manifest.status.value}`",
        f"- **Created at:** {manifest.created_at}",
    ]
    if manifest.completed_at is not None:
        lines.append(f"- **Completed at:** {manifest.completed_at}")
    lines += [
        "",
        "## Dataset",
        f"- dataset_id: `{spec.dataset_binding.dataset_id}`",
        f"- manifest_version: `{spec.dataset_binding.manifest_version}`",
        f"- content_id: `{spec.dataset_binding.content_id}`",
        f"- symbol/timeframe: {spec.dataset_binding.symbol} / {spec.dataset_binding.base_timeframe}",
        "",
        "## Model",
        f"- {spec.model_name}@{spec.model_version} (definition fingerprint `{manifest.model_definition_fingerprint[:12]}...`)",
        f"- objective: `{spec.objective.value}`",
        f"- feature_count: {len(spec.feature_binding.feature_names)}",
        "",
        "## Label / Split / Preprocessing",
        f"- label: {spec.label_binding.name} ({spec.label_binding.kind}, horizon={spec.label_binding.horizon_bars}, "
        f"type={spec.label_binding.label_type.value})",
        f"- split strategy: `{spec.split_binding.strategy}`",
        f"- fitted preprocessing fingerprint: `{spec.preprocessing_binding.fitted_preprocessing_fingerprint}`",
        "",
        "## Reproducibility",
        f"- seed fingerprint: `{spec.seed_configuration.fingerprint()[:12]}...`",
        f"- code revision: `{spec.code_revision_binding.revision[:12]}...` "
        f"(source={spec.code_revision_binding.source}, dirty={spec.code_revision_binding.is_dirty})",
        f"- environment: Python {env.python_version} on {env.platform_system} {env.platform_release} ({env.architecture})",
        "",
        "## Validation",
    ]
    if validation_report is None:
        lines.append(f"- performed: {manifest.validation_report_reference is not None}")
        lines.append("- (full validation report not loaded -- pass `validation_report=` to expand)")
    else:
        lines.append("- performed: True")
        lines.append(f"- is_ready: {validation_report.is_ready}")
        lines.append(
            f"- critical/error/warning/info: {len(validation_report.criticals)}/{len(validation_report.errors)}/"
            f"{len(validation_report.warnings)}/{len(validation_report.infos)}"
        )
        for issue in validation_report.warnings:
            lines.append(f"  - warning: [{issue.code}] {issue.message}")
    if manifest.failure_summary:
        lines += ["", "## Failure Summary", manifest.failure_summary]
    lines += ["", "## Limitations"]
    for limitation in (*_STANDARD_LIMITATIONS, *manifest.limitations):
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = ["build_report_json", "render_report_markdown"]
