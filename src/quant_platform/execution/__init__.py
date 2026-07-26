"""Time-safe validation and experiment execution engine (Milestone 4B).

Consumes an already-`ready` `quant_platform.ml.experiment_spec.
ExperimentSpec` (Milestone 4A) and actually RUNS it over a walk-forward
fold plan: expanding/rolling/blocked/grouped time-safe splitting with
configurable purge/embargo, a deterministic execution-lifecycle state
machine, resumable/idempotent fold-by-fold execution, and content-
addressed fold/aggregate artifact storage -- all reusing Milestone 4A's
manifest/artifact/locking/event infrastructure rather than duplicating it.

See `docs/execution_engine.md` for the full architecture and
`ml.testing.ConstantTestModel` / `execution.executor.
DeterministicFoldExecutor`'s own docstrings for exactly why this
milestone calls real `fit`/`predict` yet still trains NO real predictive
model -- the ONLY model ever registered anywhere in this codebase is
that deterministic, test-only one, and it is intentionally NOT
re-exported here, for the identical reason `ml/__init__.py` excludes
`ml.testing.ConstantTestModel` from its own public surface.
"""

from __future__ import annotations

from quant_platform.execution.context import FoldExecutionContext
from quant_platform.execution.execution_validation import validate_fold_plan
from quant_platform.execution.executor import (
    DeterministicFoldExecutor,
    FoldData,
    FoldExecutionOutcome,
    FoldExecutor,
)
from quant_platform.execution.manifests import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST,
    SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE,
    ExecutionManifest,
    ExecutionManifestStore,
)
from quant_platform.execution.reporting import build_execution_report_json, render_execution_report_markdown
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult, FoldStatus
from quant_platform.execution.resume import ResumePlan, build_resume_plan, can_resume, verify_completed_folds
from quant_platform.execution.runner import (
    ExecutionOutcome,
    ExecutionRunner,
    assert_preprocessing_is_safe_for_execution,
    extract_label_horizon_bars,
    resolve_serializer,
)
from quant_platform.execution.splitters import (
    EmbargoSpec,
    Fold,
    FoldPlan,
    PurgeSpec,
    build_folds_from_split_binding,
    fold_row_counts,
    generate_blocked_time_folds,
    generate_expanding_folds,
    generate_grouped_walk_forward_folds,
    generate_rolling_folds,
    iter_fold_bounds,
    reconstruct_dataset_timeline,
    required_label_purge_bars_for,
)
from quant_platform.execution.state_machine import (
    TERMINAL_STAGES,
    ExecutionStage,
    is_legal_execution_transition,
    is_terminal_stage,
)
from quant_platform.execution.timeline import Timeline, TimelineEntry, render_timeline_markdown
from quant_platform.execution.verification import verify_execution

EXECUTION_ENGINE_VERSION = "1.0.0"
"""Version of this package's own execution-lifecycle/fold-plan/manifest
semantics -- recorded informationally, independent of
`quant_platform.__version__` and of Milestone 4A's own
`ml.ML_INFRASTRUCTURE_VERSION`. Bump on any change to those semantics."""

__all__ = [
    "EXECUTION_ENGINE_VERSION",
    "EXECUTION_MANIFEST_SCHEMA_VERSION",
    "LABEL_HORIZON_SOURCE_RESEARCH_DATASET_MANIFEST",
    "SPLIT_POLICY_REJECT_INSUFFICIENT_LABEL_PURGE",
    "TERMINAL_STAGES",
    "AggregatedExecutionResult",
    "DeterministicFoldExecutor",
    "EmbargoSpec",
    "ExecutionManifest",
    "ExecutionManifestStore",
    "ExecutionOutcome",
    "ExecutionRunner",
    "ExecutionStage",
    "Fold",
    "FoldData",
    "FoldExecutionContext",
    "FoldExecutionOutcome",
    "FoldExecutor",
    "FoldPlan",
    "FoldResult",
    "FoldStatus",
    "PurgeSpec",
    "ResumePlan",
    "Timeline",
    "TimelineEntry",
    "assert_preprocessing_is_safe_for_execution",
    "build_execution_report_json",
    "build_folds_from_split_binding",
    "build_resume_plan",
    "can_resume",
    "extract_label_horizon_bars",
    "fold_row_counts",
    "generate_blocked_time_folds",
    "generate_expanding_folds",
    "generate_grouped_walk_forward_folds",
    "generate_rolling_folds",
    "is_legal_execution_transition",
    "is_terminal_stage",
    "iter_fold_bounds",
    "reconstruct_dataset_timeline",
    "render_execution_report_markdown",
    "render_timeline_markdown",
    "required_label_purge_bars_for",
    "resolve_serializer",
    "validate_fold_plan",
    "verify_completed_folds",
    "verify_execution",
]
