"""Milestone 4D: leakage-safe feature selection and hyperparameter
optimization.

Depends on `quant_platform.ml`, `quant_platform.execution`,
`quant_platform.features`, and `quant_platform.validation`. None of those
packages import anything from this one -- the dependency is one-way.

THE CENTRAL RULE
--------------------------------------------------------------------------
    OUTER TRAIN
        |-- inner time-safe walk-forward splits
                |-- inner train
                |-- inner validation

    OUTER TEST
        never used during: feature selection, hyperparameter search,
        pruning decisions, early stopping decisions, metric choice,
        threshold selection, model comparison, preprocessing fitting,
        candidate ranking.

After the best candidate is selected using only inner splits: (1) refit
the selected pipeline on the complete outer-train partition; (2) evaluate
once on the untouched outer-test partition; (3) persist the exact
selected feature set, hyperparameters, inner scores, outer score, seeds,
and artifacts. `optimization.outer_fold.finalize_outer_fold` is the ONLY
function in this package that ever reads an outer fold's test partition.

MODULE MAP
--------------------------------------------------------------------------
`search_space` (typed hyperparameter search-space model, Optuna-free) ->
`inner_splits` (nested walk-forward construction + independent validator,
reusing `execution.splitters`/`validation.walk_forward` directly) ->
`feature_selection` (candidate feature universe + the six required
selection strategies + `FeatureSelectionResult`) -> `models`
(`OptimizationSpec` identity, `OptimizationStage` state machine, the
seed-derivation hierarchy) -> `objectives` (primary-metric/objective
compatibility, reusing `ml.comparison`'s authoritative direction registry)
-> `candidates` (`TrialSpec`/`TrialResult`, the one deterministic ranking
policy) -> `trial_executor` (isolated, idempotent inner-CV loop) ->
`study` (Optuna TPE/Random integration, deterministic resume, platform-
owned median-stopping pruning) -> `outer_fold` (outer-train refit +
outer-test evaluation) -> `stability` (feature/hyperparameter stability
analysis) -> `manifests` (`OptimizationManifest` + append-only event
log) -> `resume` (verified-artifact resume planning) -> `verification`
(`verify_optimization`) -> `reporting` (JSON/Markdown reports) ->
`runner` (`OptimizationRunner`, the top-level orchestrator).

PREPROCESSING POLICY: OPTION A (see `optimization.models`' own docstring
for the full rationale) -- scale-sensitive models (Logistic Regression,
Elastic Net) are excluded from this milestone's optimization surface
entirely, fail closed.

EXPLICITLY OUT OF SCOPE (see the delivery report for the complete list):
SHAP/LIME/feature-attribution dashboards, probability calibration,
ensembles/stacking/blending, reinforcement learning, neural networks,
distributed training, online learning, live trading/order execution,
recursive feature elimination, and any use of outer-test performance for
optimization, pruning, or search-space adaptation.
"""

from __future__ import annotations

from quant_platform.optimization.candidates import (
    InnerFoldTrialMetrics,
    RankingEntry,
    RankingTable,
    TrialResult,
    TrialSpec,
    TrialStatus,
    estimate_model_complexity,
    rank_trials,
)
from quant_platform.optimization.feature_selection import (
    FEATURE_SELECTION_SCHEMA_VERSION,
    MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS,
    FeatureSelectionResult,
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
    run_feature_selection,
    validate_feature_selection_result,
)
from quant_platform.optimization.inner_splits import (
    InnerFold,
    InnerFoldPlan,
    InnerSplitConfig,
    build_inner_fold_plan,
    validate_nested_plan,
)
from quant_platform.optimization.manifests import (
    OptimizationEventStore,
    OptimizationEventType,
    OptimizationManifest,
    OptimizationManifestStore,
)
from quant_platform.optimization.models import (
    OPTIMIZATION_IDENTITY_SCHEMA_VERSION,
    OPTIMIZATION_SCHEMA_VERSION,
    EarlyStoppingConfig,
    OptimizationIdentity,
    OptimizationSpec,
    OptimizationStage,
    PreprocessingPolicy,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
    compute_optimization_identity,
    is_legal_optimization_transition,
    is_terminal_optimization_stage,
    verify_optimization_identity,
)
from quant_platform.optimization.objectives import (
    PrimaryMetricAggregationOutcome,
    aggregate_primary_metric,
    metric_direction_multiplier,
    validate_primary_metric,
)
from quant_platform.optimization.outer_fold import OuterFoldResult, finalize_outer_fold
from quant_platform.optimization.reporting import (
    build_optimization_report_json,
    render_optimization_report_markdown,
)
from quant_platform.optimization.resume import (
    TrialResumePlan,
    build_trial_resume_plan,
    can_resume,
    require_resumable,
)
from quant_platform.optimization.runner import OptimizationOutcome, OptimizationRunner
from quant_platform.optimization.search_space import (
    SEARCH_SPACE_SCHEMA_VERSION,
    BooleanParameter,
    CategoricalParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    ParameterDefinition,
    SearchSpace,
    default_search_space_for_model,
)
from quant_platform.optimization.stability import (
    FeatureStabilityReport,
    HyperparameterStabilityReport,
    JaccardSummary,
    pairwise_jaccard_similarity,
    summarize_feature_stability,
    summarize_hyperparameter_stability,
)
from quant_platform.optimization.study import create_study, rebuild_study_from_history
from quant_platform.optimization.trial_executor import run_trial
from quant_platform.optimization.verification import verify_optimization

OPTIMIZATION_ENGINE_VERSION = "1.0.0"
"""Version of this package's own optimization-lifecycle/identity/manifest
semantics -- recorded informationally, independent of `quant_platform.
__version__` and of Milestones 4A-4C's own version constants. Bump on any
change to those semantics."""

__all__ = [
    "FEATURE_SELECTION_SCHEMA_VERSION",
    "MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS",
    "OPTIMIZATION_ENGINE_VERSION",
    "OPTIMIZATION_IDENTITY_SCHEMA_VERSION",
    "OPTIMIZATION_SCHEMA_VERSION",
    "SEARCH_SPACE_SCHEMA_VERSION",
    "BooleanParameter",
    "CategoricalParameter",
    "EarlyStoppingConfig",
    "FeatureSelectionResult",
    "FeatureSelectionSpec",
    "FeatureSelectionStrategy",
    "FeatureStabilityReport",
    "FeatureUniverse",
    "FixedParameter",
    "FloatParameter",
    "HyperparameterStabilityReport",
    "InnerFold",
    "InnerFoldPlan",
    "InnerFoldTrialMetrics",
    "InnerSplitConfig",
    "IntegerParameter",
    "JaccardSummary",
    "OptimizationEventStore",
    "OptimizationEventType",
    "OptimizationIdentity",
    "OptimizationManifest",
    "OptimizationManifestStore",
    "OptimizationOutcome",
    "OptimizationRunner",
    "OptimizationSpec",
    "OptimizationStage",
    "OuterFoldResult",
    "ParameterDefinition",
    "PreprocessingPolicy",
    "PrimaryMetricAggregationOutcome",
    "PruningConfig",
    "PruningKind",
    "RankingEntry",
    "RankingTable",
    "SamplerKind",
    "SearchSpace",
    "TrialResult",
    "TrialResumePlan",
    "TrialSpec",
    "TrialStatus",
    "aggregate_primary_metric",
    "build_inner_fold_plan",
    "build_optimization_report_json",
    "build_optimization_spec",
    "build_trial_resume_plan",
    "can_resume",
    "compute_optimization_identity",
    "create_study",
    "default_search_space_for_model",
    "estimate_model_complexity",
    "finalize_outer_fold",
    "is_legal_optimization_transition",
    "is_terminal_optimization_stage",
    "metric_direction_multiplier",
    "pairwise_jaccard_similarity",
    "rank_trials",
    "rebuild_study_from_history",
    "render_optimization_report_markdown",
    "require_resumable",
    "run_feature_selection",
    "run_trial",
    "summarize_feature_stability",
    "summarize_hyperparameter_stability",
    "validate_feature_selection_result",
    "validate_nested_plan",
    "validate_primary_metric",
    "verify_optimization",
    "verify_optimization_identity",
]
