"""Human-readable and machine-readable optimization reports (Milestone
4D) -- the optimization-engine analogue of `execution.reporting`/`ml.
reporting`'s "deterministic JSON + companion Markdown, no dashboard"
scope. Every function here is a PURE function of already-loaded data --
no I/O, no re-fetching artifacts, no re-running anything. A caller
wanting expanded content (ranking tables, stability reports, a baseline
comparison) fetches/computes it separately and passes it in.

"Do not label a result profitable. Do not claim production readiness. Do
not call a candidate statistically superior unless the required
comparison criteria are actually satisfied." -- `_STANDARD_LIMITATIONS`
is included, verbatim, in EVERY report this module produces, and
`_baseline_comparison_section` only ever states "outperforms all
baselines" when `ModelComparisonReport.outperforms_all_baselines` (`ml.
comparison`'s own, already-audited "never declare success" gate) actually
says so for the SAME primary metric -- never inferred from a raw score
comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from quant_platform.ml.comparison import ModelComparisonReport
from quant_platform.optimization.candidates import RankingTable
from quant_platform.optimization.manifests import OptimizationManifest
from quant_platform.optimization.models import OptimizationSpec
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.stability import FeatureStabilityReport, HyperparameterStabilityReport

_SCHEMA_VERSION = 1

_STANDARD_LIMITATIONS = (
    "This report describes a LEAKAGE-SAFE SEARCH PROCESS (nested walk-forward feature selection and "
    "hyperparameter optimization) and its resulting outer-fold evaluation metrics -- it does not label any "
    "result profitable, does not claim production readiness, and does not call a candidate statistically "
    "superior to a baseline unless ml.comparison's own paired-significance gate actually says so for the "
    "declared primary metric.",
    "Outer-fold metrics are computed ONCE, after candidate selection, on a partition never used for feature "
    "selection, hyperparameter search, pruning, early stopping, threshold selection, or model comparison -- but "
    "a small number of outer folds (typical for walk-forward evaluation) limits how strongly any single "
    "optimization's outer-fold results generalize.",
    "Feature/hyperparameter stability sections describe SEARCH BEHAVIOR (how consistent selections/winners were "
    "across inner folds and outer folds) -- instability is flagged, never silently hidden, but flagging is not "
    "itself a correctness defect; it is information for a human deciding how much to trust the winning candidate.",
)


def _split_config_section(spec: OptimizationSpec) -> dict[str, object]:
    return {
        "outer_split_binding": spec.outer_split_binding.to_json_dict(),
        "inner_split_config": spec.inner_split_config.to_json_dict(),
    }


def _search_section(spec: OptimizationSpec) -> dict[str, object]:
    return {
        "search_space": spec.search_space.to_json_dict(),
        "sampler_kind": spec.sampler_kind.value,
        "pruning_config": spec.pruning_config.to_json_dict(),
        "early_stopping_config": spec.early_stopping_config.to_json_dict(),
        "feature_selection_spec": spec.feature_selection_spec.to_json_dict(),
        "max_trials": spec.max_trials, "min_successful_inner_folds": spec.min_successful_inner_folds,
        "timeout_seconds": spec.timeout_seconds, "max_failed_trials": spec.max_failed_trials,
    }


def _outer_fold_section(result: OuterFoldResult) -> dict[str, object]:
    return {
        "outer_fold_index": result.outer_fold_index, "winning_trial_number": result.winning_trial_number,
        "final_selected_features": list(result.final_selected_features),
        "final_hyperparameters": dict(sorted(result.final_hyperparameters.items())),
        "final_round_source": result.final_round_source,
        "outer_train_row_count": result.outer_train_row_count, "outer_test_row_count": result.outer_test_row_count,
        "outer_test_metrics": dict(sorted(result.outer_test_metrics.items())),
    }


def _baseline_comparison_section(baseline_comparison: ModelComparisonReport | None, *, primary_metric: str) -> dict[str, object]:
    if baseline_comparison is None:
        return {"performed": False, "reason": "no baseline comparison was supplied to this report"}
    return {
        "performed": True, "outperforms_all_baselines": baseline_comparison.outperforms_all_baselines(primary_metric),
        "report": baseline_comparison.to_json_dict(),
    }


def build_optimization_report_json(
    manifest: OptimizationManifest, *, spec: OptimizationSpec | None = None,
    outer_fold_results: Sequence[OuterFoldResult] = (), ranking_tables: Mapping[int, RankingTable] | None = None,
    feature_stability: FeatureStabilityReport | None = None, hyperparameter_stability: HyperparameterStabilityReport | None = None,
    baseline_comparison: ModelComparisonReport | None = None,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "optimization_id": manifest.optimization_id, "parent_experiment_id": manifest.parent_experiment_id,
        "stage": manifest.stage.value, "created_at": manifest.created_at, "updated_at": manifest.updated_at,
        "completed_at": manifest.completed_at, "resume_count": manifest.resume_count,
        "trial_counts": {
            "completed": manifest.total_trials_completed, "failed": manifest.total_trials_failed,
            "invalid": manifest.total_trials_invalid, "pruned": manifest.total_trials_pruned,
        },
        "identity": (None if spec is None else {
            "model_name": spec.model_name, "model_version": spec.model_version, "objective": spec.objective.value,
            "primary_metric": spec.primary_metric, "metric_direction": spec.metric_direction,
            "dataset_binding": spec.dataset_binding.to_json_dict(), "preprocessing_policy": spec.preprocessing_policy.value,
        }),
        "split_config": (None if spec is None else _split_config_section(spec)),
        "search_config": (None if spec is None else _search_section(spec)),
        "winning_trial_by_outer_fold": {str(k): v for k, v in sorted(manifest.winning_trial_by_outer_fold.items())},
        "ranking_tables": (
            None if ranking_tables is None else {str(k): v.to_json_dict() for k, v in sorted(ranking_tables.items())}
        ),
        "outer_fold_results": [_outer_fold_section(r) for r in outer_fold_results],
        "feature_stability": (None if feature_stability is None else feature_stability.to_json_dict()),
        "hyperparameter_stability": (None if hyperparameter_stability is None else hyperparameter_stability.to_json_dict()),
        "baseline_comparison": _baseline_comparison_section(baseline_comparison, primary_metric=(spec.primary_metric if spec is not None else "")),
        "failure_summary": manifest.failure_summary,
        "limitations": list(_STANDARD_LIMITATIONS),
    }


def render_optimization_report_markdown(
    manifest: OptimizationManifest, *, spec: OptimizationSpec | None = None,
    outer_fold_results: Sequence[OuterFoldResult] = (), ranking_tables: Mapping[int, RankingTable] | None = None,
    feature_stability: FeatureStabilityReport | None = None, hyperparameter_stability: HyperparameterStabilityReport | None = None,
    baseline_comparison: ModelComparisonReport | None = None,
) -> str:
    lines = [
        "# Optimization Report", "",
        f"- **Optimization ID:** `{manifest.optimization_id}`", f"- **Parent experiment ID:** `{manifest.parent_experiment_id}`",
        f"- **Stage:** `{manifest.stage.value}`", f"- **Created at:** {manifest.created_at}", f"- **Updated at:** {manifest.updated_at}",
    ]
    if manifest.completed_at is not None:
        lines.append(f"- **Completed at:** {manifest.completed_at}")
    lines.append(f"- **Resume count:** {manifest.resume_count}")

    if spec is not None:
        lines += [
            "", "## Identity",
            f"- model: `{spec.model_name}@{spec.model_version}`", f"- objective: `{spec.objective.value}`",
            f"- primary metric: `{spec.primary_metric}` (direction: `{spec.metric_direction}`)",
            f"- preprocessing policy: `{spec.preprocessing_policy.value}`",
            "", "## Split Configuration",
            f"- outer split strategy: `{spec.outer_split_binding.strategy}`",
            f"- inner split strategy: `{spec.inner_split_config.strategy}` (n_splits={spec.inner_split_config.n_splits}, "
            f"test_size_fraction={spec.inner_split_config.test_size_fraction})",
            "", "## Search Configuration",
            f"- feature selection: `{spec.feature_selection_spec.strategy.value}`",
            f"- sampler: `{spec.sampler_kind.value}`", f"- pruning: `{spec.pruning_config.kind.value}`",
            f"- early stopping enabled: `{spec.early_stopping_config.enabled}`",
            f"- max trials: {spec.max_trials}, min successful inner folds: {spec.min_successful_inner_folds}",
        ]

    lines += [
        "", "## Trial Counts",
        f"- completed: {manifest.total_trials_completed}", f"- failed: {manifest.total_trials_failed}",
        f"- invalid: {manifest.total_trials_invalid}", f"- pruned: {manifest.total_trials_pruned}",
    ]

    lines += ["", "## Outer-Fold Results (untouched outer-test partition, evaluated once per fold)"]
    if not outer_fold_results:
        lines.append("- (no outer-fold results loaded -- pass `outer_fold_results=` to expand)")
    else:
        lines.append("| Outer Fold | Winning Trial | # Features | Outer-Test Metrics |")
        lines.append("| --- | --- | --- | --- |")
        for r in outer_fold_results:
            metrics_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in sorted(r.outer_test_metrics.items()))
            lines.append(f"| {r.outer_fold_index} | {r.winning_trial_number} | {len(r.final_selected_features)} | {metrics_str} |")

    lines += ["", "## Ranking Tables"]
    if ranking_tables is None:
        lines.append("- (ranking tables not loaded -- pass `ranking_tables=` to expand)")
    else:
        for outer_fold_index, table in sorted(ranking_tables.items()):
            lines.append(f"### Outer fold {outer_fold_index}")
            lines.append("| Rank | Trial | Valid | Primary Metric | Successful Inner Folds |")
            lines.append("| --- | --- | --- | --- | --- |")
            for entry in table.entries:
                lines.append(
                    f"| {entry.rank} | {entry.trial_number} | {entry.is_valid} | "
                    f"{entry.primary_metric_aggregate if entry.primary_metric_aggregate is not None else '-'} | "
                    f"{entry.successful_inner_folds} |"
                )

    lines += ["", "## Feature Stability"]
    if feature_stability is None:
        lines.append("- (feature stability not computed -- pass `feature_stability=` to expand)")
    else:
        if feature_stability.pairwise_jaccard is not None:
            lines.append(f"- pairwise Jaccard similarity across winning feature sets: mean={feature_stability.pairwise_jaccard.mean:.3f}")
        for warning in feature_stability.warnings:
            lines.append(f"- WARNING: {warning}")

    lines += ["", "## Hyperparameter Stability"]
    if hyperparameter_stability is None:
        lines.append("- (hyperparameter stability not computed -- pass `hyperparameter_stability=` to expand)")
    else:
        for warning in hyperparameter_stability.warnings:
            lines.append(f"- WARNING: {warning}")
        if not hyperparameter_stability.warnings:
            lines.append("- no instability/boundary-hit warnings")

    lines += ["", "## Baseline Comparison"]
    primary_metric = spec.primary_metric if spec is not None else ""
    if baseline_comparison is None:
        lines.append("- not performed as part of this report")
    else:
        lines.append(f"- outperforms all baselines (`{primary_metric}`): {baseline_comparison.outperforms_all_baselines(primary_metric)}")

    if manifest.failure_summary:
        lines += ["", "## Failure Summary", manifest.failure_summary]

    lines += ["", "## Limitations"]
    for limitation in _STANDARD_LIMITATIONS:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = ["build_optimization_report_json", "render_optimization_report_markdown"]
