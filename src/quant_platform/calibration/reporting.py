"""Human-readable and machine-readable calibration reports (Milestone
4E) -- the calibration-framework analogue of `optimization.reporting`'s
"deterministic JSON + companion Markdown, no dashboard" scope. Every
function here is a PURE function of already-loaded data -- no I/O, no
re-fetching artifacts, no re-running anything.

`_STANDARD_LIMITATIONS` is included, verbatim, in EVERY report this
module produces (Section 22/40's explicit unsupported-claims list): this
framework never claims a calibrated probability is guaranteed accurate
under a future regime change, never claims its uncertainty proxies are a
complete Bayesian posterior, never equates "confidence" with certainty,
and never claims high confidence implies profitable trading -- this
milestone performs no backtesting, PnL simulation, transaction-cost, or
risk modeling at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from quant_platform.calibration.manifests import CalibrationManifest
from quant_platform.calibration.runner import OuterFoldCalibrationResult
from quant_platform.calibration.specs import CalibrationSpec

_SCHEMA_VERSION = 1

_STANDARD_LIMITATIONS = (
    "This report describes a LEAKAGE-SAFE POST-PROCESSING pipeline (probability calibration, decision "
    "thresholding, confidence, and uncertainty estimation) applied to an already-selected, already-refit base "
    "model's raw outputs -- it performs no feature selection, hyperparameter search, backtesting, PnL "
    "simulation, transaction-cost modeling, position sizing, or portfolio construction.",
    "A calibrated probability is an estimate fit on historical inner out-of-fold data; it is NOT guaranteed to "
    "remain accurate under a future regime change, and this framework makes no claim otherwise.",
    "\"Confidence\" and \"uncertainty\" scores in this report are transparent, DOCUMENTED proxies (distance from "
    "decision threshold, probability entropy, reliability-bin support, calibrator disagreement) -- they are not "
    "a claim of exact Bayesian posterior uncertainty, and confidence is not the same thing as certainty.",
    "High confidence or low uncertainty for a prediction does NOT imply that acting on that prediction would be "
    "profitable -- this milestone performs no financial evaluation of any kind.",
    "Outer-fold evaluation metrics are computed ONCE, after the calibrator/threshold/confidence/uncertainty "
    "policy was frozen on inner out-of-fold data alone, on a partition never used for any post-processing "
    "decision -- but a small number of outer folds (typical for walk-forward evaluation) limits how strongly "
    "any single calibration run's outer-fold results generalize.",
)


def _outer_fold_section(result: OuterFoldCalibrationResult) -> dict[str, object]:
    return {
        "outer_fold_index": result.outer_fold_index, "outer_train_row_count": result.outer_train_row_count,
        "outer_test_row_count": result.outer_test_row_count, "training_duration_seconds": result.training_duration_seconds,
        "classification_metrics": dict(sorted(result.classification_metrics.items())),
        "calibration_metrics_on_outer_test": dict(sorted(result.calibration_metrics_on_outer_test.items())),
        "selective_prediction_summary": dict(sorted(result.selective_prediction_summary.items())),
        "decision_counts": {
            decision: result.decisions.count(decision) for decision in sorted(set(result.decisions))
        },
        "mean_confidence": (sum(result.confidence_scores) / len(result.confidence_scores)) if result.confidence_scores else None,
        "mean_uncertainty": (sum(result.uncertainty_scores) / len(result.uncertainty_scores)) if result.uncertainty_scores else None,
    }


def _identity_section(spec: CalibrationSpec) -> dict[str, object]:
    return {
        "task": spec.task.value, "source_experiment_id": spec.source_experiment_id,
        "source_optimization_id": spec.source_optimization_id, "base_model_definition_identity": spec.base_model_definition_identity,
        "calibration_method_candidates": [m.value for m in spec.calibration_method_candidates],
        "calibration_selection_metric": spec.calibration_selection_metric.value,
        "threshold_policy": spec.threshold_spec.policy.value, "abstention_policy": spec.abstention_spec.policy.value,
    }


def _aggregate_metrics_section(outer_fold_results: Sequence[OuterFoldCalibrationResult]) -> dict[str, object]:
    """Section 27: per-fold values PLUS statistics, never hiding fold
    instability by silently averaging it away -- every fold's own metric
    values are listed, alongside cross-fold mean/std, for EVERY metric
    name that appears in every fold's `classification_metrics`."""
    if not outer_fold_results:
        return {"per_fold": {}, "cross_fold_mean": {}, "cross_fold_std": {}}
    common_names = set(outer_fold_results[0].classification_metrics)
    for r in outer_fold_results[1:]:
        common_names &= set(r.classification_metrics)
    per_fold: dict[str, list[object]] = {name: [] for name in sorted(common_names)}
    for r in outer_fold_results:
        for name in per_fold:
            per_fold[name].append(r.classification_metrics[name])
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name, values in per_fold.items():
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if len(numeric) == len(values) and numeric:
            mean = sum(numeric) / len(numeric)
            means[name] = mean
            stds[name] = (sum((v - mean) ** 2 for v in numeric) / len(numeric)) ** 0.5
    return {"per_fold": per_fold, "cross_fold_mean": means, "cross_fold_std": stds}


def build_calibration_report_json(
    manifest: CalibrationManifest, *, spec: CalibrationSpec | None = None,
    outer_fold_results: Sequence[OuterFoldCalibrationResult] = (),
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION, "calibration_id": manifest.calibration_id,
        "source_experiment_id": manifest.source_experiment_id, "source_optimization_id": manifest.source_optimization_id,
        "stage": manifest.stage.value, "created_at": manifest.created_at, "updated_at": manifest.updated_at,
        "completed_at": manifest.completed_at, "resume_count": manifest.resume_count,
        "total_outer_folds": manifest.total_outer_folds, "completed_outer_fold_indices": list(manifest.completed_outer_fold_indices),
        "identity": (None if spec is None else _identity_section(spec)),
        "outer_fold_results": [_outer_fold_section(r) for r in outer_fold_results],
        "aggregate_metrics": _aggregate_metrics_section(outer_fold_results),
        "failure_summary": manifest.failure_summary,
        "limitations": list(_STANDARD_LIMITATIONS),
    }


def render_calibration_report_markdown(
    manifest: CalibrationManifest, *, spec: CalibrationSpec | None = None,
    outer_fold_results: Sequence[OuterFoldCalibrationResult] = (),
) -> str:
    lines = [
        "# Calibration Report", "",
        f"- **Calibration ID:** `{manifest.calibration_id}`", f"- **Source experiment ID:** `{manifest.source_experiment_id}`",
        f"- **Stage:** `{manifest.stage.value}`", f"- **Created at:** {manifest.created_at}", f"- **Updated at:** {manifest.updated_at}",
    ]
    if manifest.source_optimization_id is not None:
        lines.append(f"- **Source optimization ID:** `{manifest.source_optimization_id}`")
    if manifest.completed_at is not None:
        lines.append(f"- **Completed at:** {manifest.completed_at}")
    lines.append(f"- **Resume count:** {manifest.resume_count}")

    if spec is not None:
        lines += [
            "", "## Identity",
            f"- task: `{spec.task.value}`", f"- base model definition identity: `{spec.base_model_definition_identity}`",
            f"- calibration method candidates: {', '.join(m.value for m in spec.calibration_method_candidates)}",
            f"- selection metric: `{spec.calibration_selection_metric.value}`",
            f"- threshold policy: `{spec.threshold_spec.policy.value}`", f"- abstention policy: `{spec.abstention_spec.policy.value}`",
        ]

    lines += ["", "## Outer-Fold Results (untouched outer-test partition, evaluated once per fold)"]
    if not outer_fold_results:
        lines.append("- (no outer-fold results loaded -- pass `outer_fold_results=` to expand)")
    else:
        lines.append("| Outer Fold | # Test Rows | Classification Metrics | Coverage | Mean Confidence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in outer_fold_results:
            metrics_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in sorted(r.classification_metrics.items()))
            coverage = r.selective_prediction_summary.get("coverage")
            coverage_str = f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "-"
            mean_conf = (sum(r.confidence_scores) / len(r.confidence_scores)) if r.confidence_scores else None
            mean_conf_str = f"{mean_conf:.3f}" if mean_conf is not None else "-"
            lines.append(f"| {r.outer_fold_index} | {r.outer_test_row_count} | {metrics_str} | {coverage_str} | {mean_conf_str} |")

    if outer_fold_results:
        aggregate = _aggregate_metrics_section(outer_fold_results)
        lines += ["", "## Cross-Fold Metric Stability (never hides fold instability by averaging alone)"]
        means = aggregate["cross_fold_mean"]
        stds = aggregate["cross_fold_std"]
        if isinstance(means, dict) and isinstance(stds, dict) and means:
            lines.append("| Metric | Mean | Std |")
            lines.append("| --- | --- | --- |")
            for name in sorted(means):
                lines.append(f"| {name} | {means[name]:.4g} | {stds[name]:.4g} |")
        else:
            lines.append("- (metrics were not uniformly numeric/present across all outer folds)")

    if manifest.failure_summary:
        lines += ["", "## Failure Summary", manifest.failure_summary]

    lines += ["", "## Limitations"]
    for limitation in _STANDARD_LIMITATIONS:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = ["build_calibration_report_json", "render_calibration_report_markdown"]
