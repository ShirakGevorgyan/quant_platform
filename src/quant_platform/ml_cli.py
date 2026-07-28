"""Command-line interface for the ML core infrastructure and artifact
foundation (Milestone 4A), the time-safe execution engine (Milestone 4B),
the baseline predictive model framework (Milestone 4C), and the leakage-
safe feature-selection/hyperparameter-optimization engine (Milestone 4D).

    python -m quant_platform.ml_cli list-model-definitions
    python -m quant_platform.ml_cli describe-model-definition --name constant_test_model --version 1
    python -m quant_platform.ml_cli prepare-experiment --config config.json
    python -m quant_platform.ml_cli validate-experiment --config config.json
    python -m quant_platform.ml_cli inspect-experiment --config config.json --experiment-id ID
    python -m quant_platform.ml_cli inspect-experiment-manifest --config config.json --experiment-id ID
    python -m quant_platform.ml_cli verify-artifact --config config.json --content-hash HASH
    python -m quant_platform.ml_cli list-experiment-events --config config.json --experiment-id ID
    python -m quant_platform.ml_cli execute --config config.json --experiment-id ID
    python -m quant_platform.ml_cli resume --config config.json --experiment-id ID
    python -m quant_platform.ml_cli inspect-execution --config config.json --experiment-id ID
    python -m quant_platform.ml_cli inspect-fold --config config.json --experiment-id ID --fold-index N
    python -m quant_platform.ml_cli list-folds --config config.json --experiment-id ID
    python -m quant_platform.ml_cli verify-execution --config config.json --experiment-id ID
    python -m quant_platform.ml_cli list-models
    python -m quant_platform.ml_cli inspect-model --name lightgbm --version 1
    python -m quant_platform.ml_cli validate-model --config config.json
    python -m quant_platform.ml_cli train --config config.json --experiment-id ID
    python -m quant_platform.ml_cli compare --config config.json --candidate-experiment-id ID --baseline-experiment-id ID [--baseline-experiment-id ID ...] --primary-metric roc_auc
    python -m quant_platform.ml_cli optimize --config opt_config.json
    python -m quant_platform.ml_cli resume-optimization --config opt_config.json --optimization-id ID
    python -m quant_platform.ml_cli inspect-optimization --config opt_config.json --optimization-id ID
    python -m quant_platform.ml_cli list-trials --config opt_config.json --optimization-id ID --outer-fold-index N
    python -m quant_platform.ml_cli inspect-trial --config opt_config.json --optimization-id ID --outer-fold-index N --trial-number M
    python -m quant_platform.ml_cli verify-optimization --config opt_config.json --optimization-id ID
    python -m quant_platform.ml_cli compare-optimization-candidates --config opt_config.json --optimization-id ID --outer-fold-index N
    python -m quant_platform.ml_cli feature-stability --config opt_config.json --optimization-id ID
    python -m quant_platform.ml_cli hyperparameter-stability --config opt_config.json --optimization-id ID
    python -m quant_platform.ml_cli create-calibration-spec --config cal_config.json
    python -m quant_platform.ml_cli run-calibration --config cal_config.json
    python -m quant_platform.ml_cli resume-calibration --config cal_config.json --calibration-id ID
    python -m quant_platform.ml_cli inspect-calibration --config cal_config.json --calibration-id ID
    python -m quant_platform.ml_cli report-calibration --config cal_config.json --calibration-id ID
    python -m quant_platform.ml_cli inspect-calibration-fold --config cal_config.json --calibration-id ID --outer-fold-index N
    python -m quant_platform.ml_cli verify-calibration --config cal_config.json --calibration-id ID
    python -m quant_platform.ml_cli compare-calibration --config cal_config.json --calibration-id ID --baseline-calibration-id ID --metric accuracy
    python -m quant_platform.ml_cli create-backtest-spec --config bt_config.json
    python -m quant_platform.ml_cli run-backtest --config bt_config.json
    python -m quant_platform.ml_cli resume-backtest --config bt_config.json --backtest-id ID
    python -m quant_platform.ml_cli inspect-backtest --config bt_config.json --backtest-id ID
    python -m quant_platform.ml_cli report-backtest --config bt_config.json --backtest-id ID
    python -m quant_platform.ml_cli inspect-backtest-fold --config bt_config.json --backtest-id ID --outer-fold-index N
    python -m quant_platform.ml_cli verify-backtest --config bt_config.json --backtest-id ID
    python -m quant_platform.ml_cli inspect-backtest-lock --config bt_config.json --backtest-id ID
    python -m quant_platform.ml_cli recover-backtest-lock --config bt_config.json --backtest-id ID [--force]
    python -m quant_platform.ml_cli compare-backtests --config bt_config.json --backtest-id ID --baseline-backtest-id ID --metric total_net_return
    python -m quant_platform.ml_cli create-robustness-spec --config rb_config.json
    python -m quant_platform.ml_cli run-robustness --config rb_config.json
    python -m quant_platform.ml_cli resume-robustness --config rb_config.json --robustness-id ID
    python -m quant_platform.ml_cli inspect-robustness --config rb_config.json --robustness-id ID
    python -m quant_platform.ml_cli report-robustness --config rb_config.json --robustness-id ID
    python -m quant_platform.ml_cli verify-robustness --config rb_config.json --robustness-id ID
    python -m quant_platform.ml_cli compare-robustness --config rb_config.json --robustness-id ID --baseline-robustness-id ID
    python -m quant_platform.ml_cli inspect-promotion-decision --config rb_config.json --robustness-id ID
    python -m quant_platform.ml_cli inspect-strategy-family --config rb_config.json --content-hash HASH
    python -m quant_platform.ml_cli compare-strategy-candidates --config rb_config.json --robustness-id ID [--robustness-id ID ...]

Same operability conventions as `data_cli`/`feature_cli`: every command
returns 0 on success, non-zero on failure, and prints an actionable
stderr message -- never a raw traceback. `prepare-experiment` and
`validate-experiment` return 2 (not 1) when the experiment/spec itself
is not ready -- distinct from 1, which means the COMMAND itself failed
(bad config, missing dataset, etc). `execute`/`resume`/`train` follow
the same convention: 2 when the execution ends with one or more failed
folds, never a traceback. `validate-model` mirrors `validate-experiment`:
2 when `ml.model_validation.validate_training_data`'s report is not
ready.

`build_model_registry()` NOW REGISTERS REAL MODELS TOO (MILESTONE 4C)
--------------------------------------------------------------------------
Every command that resolves a model definition (`list-model-definitions`,
`describe-model-definition`, `prepare-experiment`, `validate-experiment`,
`execute`, `resume`, `list-models`, `inspect-model`, `validate-model`,
`train`) now sees `ml.testing.ConstantTestModelFactory` (still
explicitly test-only) AND all nine Milestone 4C production models
(`ml.model_zoo.register_default_models`) -- purely additive; nothing
about an experiment referencing the test model changes.

`execute`/`resume` STILL USE `DeterministicFoldExecutor` (UNCHANGED)
--------------------------------------------------------------------------
`train` is the NEW, additional command that runs the identical pipeline
`execute` does (same "already `ready`, resolved via `prepare-experiment`"
requirement) but with `execution.executor.MetricsFoldExecutor` --
computing real metrics, capturing `predict_proba`, and persisting a
`TrainingMetadata` provenance artifact. `execute`/`resume` are left
exactly as Milestone 4B shipped them (`DeterministicFoldExecutor`, no
metrics) -- a real model run through `execute` instead of `train` still
completes successfully (every 4C model satisfies `DeterministicFoldExecutor`'s
requirements too), just without the 4C-specific artifacts/metrics. This
is a deliberate, conservative choice: no existing command's default
behavior changes at all.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from quant_platform.backtesting.manifests import (
    BacktestEventStore,
    BacktestManifest,
    BacktestManifestStore,
)
from quant_platform.backtesting.models import BacktestStage
from quant_platform.backtesting.reporting import (
    build_backtest_report_json,
    render_backtest_report_markdown,
)
from quant_platform.backtesting.runner import (
    BacktestLockDiagnostics,
    BacktestRunner,
    OuterFoldBacktestResult,
    inspect_backtest_lock,
    recover_backtest_lock,
)
from quant_platform.backtesting.specs import BacktestSpec, compute_backtest_identity
from quant_platform.backtesting.stitching import StitchedWalkForwardEquity
from quant_platform.backtesting.verification import verify_backtest
from quant_platform.calibration.manifests import (
    CalibrationEventStore,
    CalibrationManifest,
    CalibrationManifestStore,
)
from quant_platform.calibration.models import CalibrationStage
from quant_platform.calibration.reporting import (
    build_calibration_report_json,
    render_calibration_report_markdown,
)
from quant_platform.calibration.runner import CalibrationRunner, OuterFoldCalibrationResult
from quant_platform.calibration.specs import CalibrationSpec, compute_calibration_identity
from quant_platform.calibration.verification import verify_calibration
from quant_platform.config.backtesting_schemas import BacktestConfig
from quant_platform.config.calibration_schemas import CalibrationConfig
from quant_platform.config.ml_schemas import MLExperimentConfig
from quant_platform.config.optimization_schemas import OptimizationConfig
from quant_platform.config.robustness_schemas import RobustnessConfig
from quant_platform.core.exceptions import QuantPlatformError
from quant_platform.execution.executor import DeterministicFoldExecutor, MetricsFoldExecutor
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.execution.reporting import build_execution_report_json, render_execution_report_markdown
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.splitters import reconstruct_dataset_timeline
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.timeline import Timeline
from quant_platform.execution.verification import verify_execution
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.comparison import ModelFoldMetrics, compare_to_baselines
from quant_platform.ml.environment import capture_code_revision_binding
from quant_platform.ml.experiment_identity import compute_experiment_identity
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.manifests import ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.model_validation import validate_training_data
from quant_platform.ml.models import (
    ArtifactCategory,
    DatasetBinding,
    ExperimentStatus,
    FeatureBinding,
    ModelCapabilities,
    ObjectiveType,
    PreprocessingBinding,
    ValidationReport,
)
from quant_platform.ml.persistence import parse_json_strict
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.reporting import build_report_json, render_report_markdown
from quant_platform.ml.testing import TEST_MODEL_NAME, TEST_MODEL_VERSION, ConstantTestModelFactory
from quant_platform.ml.tracking import ExperimentEventStore
from quant_platform.ml.validation import validate_experiment_spec
from quant_platform.optimization.candidates import RankingTable, TrialResult
from quant_platform.optimization.manifests import (
    OptimizationEventStore,
    OptimizationManifest,
    OptimizationManifestStore,
    trial_references_for_outer_fold,
)
from quant_platform.optimization.models import (
    OptimizationSpec,
    OptimizationStage,
)
from quant_platform.optimization.outer_fold import OuterFoldResult
from quant_platform.optimization.reporting import (
    build_optimization_report_json,
    render_optimization_report_markdown,
)
from quant_platform.optimization.runner import OptimizationRunner
from quant_platform.optimization.stability import (
    FeatureStabilityReport,
    HyperparameterStabilityReport,
    flag_near_tied_top_candidates,
)
from quant_platform.optimization.verification import verify_optimization
from quant_platform.robustness.manifests import RobustnessManifest, RobustnessManifestStore
from quant_platform.robustness.models import RobustnessStage
from quant_platform.robustness.multiple_testing import StrategyFamily
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.runner import RobustnessRunner
from quant_platform.robustness.selection import SelectionReport
from quant_platform.robustness.specs import RobustnessSpec, compute_robustness_identity
from quant_platform.robustness.verification import verify_robustness

_TIMESTAMP_COLUMN = "open_time"
_LABEL_COLUMN = "label"


def build_model_registry() -> ModelRegistry:
    """The test-only model (`ml.testing.ConstantTestModelFactory`,
    explicitly labeled as such, never a real predictive algorithm) PLUS
    every Milestone 4C production model (`ml.model_zoo.
    register_default_models`) -- LightGBM, XGBoost, CatBoost, Logistic
    Regression, Elastic Net, and the four mandatory baselines."""
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            name=TEST_MODEL_NAME, version=TEST_MODEL_VERSION,
            description="TEST-ONLY deterministic model (predicts the training label mean/positive-rate "
            "regardless of input) -- NOT a real predictive algorithm. See ml.testing module docstring.",
            capabilities=ModelCapabilities(
                supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
                supports_predict_proba=True,
            ),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        )
    )
    mz.register_default_models(registry)
    return registry


def _load_config(path: Path) -> MLExperimentConfig:
    return MLExperimentConfig.model_validate_json(path.read_text())


def _load_dataset_manifest(config: MLExperimentConfig) -> ResearchDatasetManifest:
    store = ResearchManifestStore(config.dataset.research_storage_root)
    return store.load(config.dataset.dataset_id, config.dataset.manifest_version)


def build_experiment_spec(config: MLExperimentConfig) -> tuple[ExperimentSpec, ResearchDatasetManifest]:
    """Assembles an `ExperimentSpec` from validated config plus whatever
    the referenced research dataset manifest actually records -- the
    dataset's own `feature_versions`/`feature_registry_fingerprint`/
    `preprocessing_definition`/`fitted_preprocessing_fingerprint` are
    used directly, never re-typed into the config (see
    `config.ml_schemas`'s module docstring)."""
    dataset_manifest = _load_dataset_manifest(config)

    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value,
        source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_names = tuple(config.dataset.feature_names) if config.dataset.feature_names is not None else dataset_manifest.feature_names
    feature_binding = FeatureBinding(
        feature_names=feature_names,
        feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    code_revision_binding = capture_code_revision_binding()

    spec = ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding, label_binding=config.label.build(),
        split_binding=config.split.build(), preprocessing_binding=preprocessing_binding,
        model_name=config.model.name, model_version=config.model.version,
        hyperparameters=config.model.build_hyperparameters(), objective=config.model.build_objective(),
        seed_configuration=config.seeds.build(), code_revision_binding=code_revision_binding,
        primary_metric=config.primary_metric, environment_requirements=dict(config.environment_requirements),
        tags=tuple(config.tags), notes=config.notes,
    )
    return spec, dataset_manifest


def cmd_list_model_definitions(args: argparse.Namespace) -> int:  # noqa: ARG001 -- uniform handler signature
    registry = build_model_registry()
    for definition in registry.list_definitions():
        deprecated = " [DEPRECATED]" if definition.deprecated else ""
        print(f"{definition.qualified_name}{deprecated} -- {definition.description}")
    return 0


def cmd_describe_model_definition(args: argparse.Namespace) -> int:
    registry = build_model_registry()
    definition = registry.get(args.name, args.version)
    for key, value in definition.to_json_dict().items():
        print(f"{key}: {value}")
    print(f"fingerprint: {definition.fingerprint()}")
    return 0


def _build_execution_runner(config: MLExperimentConfig, *, fold_executor: DeterministicFoldExecutor | MetricsFoldExecutor | None = None) -> ExecutionRunner:
    return ExecutionRunner(
        ml_artifacts_root=config.ml_artifacts_root, model_registry=build_model_registry(),
        research_manifest_store=ResearchManifestStore(config.dataset.research_storage_root),
        research_dataset_store=ResearchDatasetStore(config.dataset.research_storage_root),
        fold_executor=fold_executor if fold_executor is not None else DeterministicFoldExecutor(),
        additional_serializers=mz.default_serializer_registry(),
    )


def _build_preparer(config: MLExperimentConfig) -> ExperimentPreparer:
    return ExperimentPreparer(
        ml_artifacts_root=config.ml_artifacts_root, model_registry=build_model_registry(),
        research_manifest_store=ResearchManifestStore(config.dataset.research_storage_root),
    )


def cmd_prepare_experiment(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    spec, _ = build_experiment_spec(config)
    manifest = _build_preparer(config).prepare(spec)
    print(f"experiment_id: {manifest.identity.experiment_id}")
    print(f"status: {manifest.status.value}")
    if manifest.failure_summary:
        print(f"failure_summary: {manifest.failure_summary}")
    return 0 if manifest.status is ExperimentStatus.READY else 2


def cmd_validate_experiment(args: argparse.Namespace) -> int:
    """Dry run: builds the spec and runs preflight validation WITHOUT
    calling `ExperimentPreparer.prepare` -- no manifest, artifact, or
    event is written. Useful for iterating on a config before committing."""
    config = _load_config(Path(args.config))
    spec, dataset_manifest = build_experiment_spec(config)
    registry = build_model_registry()
    report = validate_experiment_spec(
        spec, model_registry=registry, dataset_manifest=dataset_manifest, ml_artifacts_root=config.ml_artifacts_root,
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def _load_manifest(config: MLExperimentConfig, experiment_id: str) -> ExperimentManifest:
    return ExperimentManifestStore(config.ml_artifacts_root).load(experiment_id)


def cmd_inspect_experiment(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = _load_manifest(config, args.experiment_id)
    validation_report = None
    if manifest.validation_report_reference is not None:
        raw = MLArtifactStore(config.ml_artifacts_root).read_artifact(manifest.validation_report_reference.content_hash)
        validation_report = ValidationReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    if args.format == "json":
        import json

        print(json.dumps(build_report_json(manifest, validation_report=validation_report), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_report_markdown(manifest, validation_report=validation_report))
    return 0


def cmd_inspect_experiment_manifest(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = _load_manifest(config, args.experiment_id)
    for key, value in manifest.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def cmd_verify_artifact(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    store = MLArtifactStore(config.ml_artifacts_root)
    ref = store.artifact_reference(args.content_hash)
    print(f"content_hash: {ref.content_hash}")
    print(f"category: {ref.category.value}")
    print(f"size_bytes: {ref.size_bytes}")
    print(f"created_at: {ref.created_at}")
    print("verified: OK (content hash matches, metadata sidecar consistent)")
    return 0


def cmd_list_experiment_events(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    events = ExperimentEventStore(config.ml_artifacts_root).read_events(args.experiment_id)
    if not events:
        print(f"No events recorded for experiment_id={args.experiment_id!r}", file=sys.stderr)
        return 1
    for event in events:
        print(f"{event.sequence:04d} {event.occurred_at} {event.event_type.value} {dict(sorted(event.details.items()))}")
    return 0


def _run_or_resume(
    args: argparse.Namespace, *, is_resume: bool, fold_executor: DeterministicFoldExecutor | MetricsFoldExecutor | None = None,
) -> int:
    config = _load_config(Path(args.config))
    runner = _build_execution_runner(config, fold_executor=fold_executor)
    force_rerun_folds = frozenset(getattr(args, "force_rerun_fold", None) or [])
    outcome = (
        runner.resume(args.experiment_id, force_rerun_folds=force_rerun_folds) if is_resume
        else runner.run(args.experiment_id, force_rerun_folds=force_rerun_folds)
    )
    print(f"experiment_id: {args.experiment_id}")
    print(f"overall_status: {outcome.aggregate.overall_status.value}")
    print(f"completed_folds: {list(outcome.aggregate.completed_fold_indices)}")
    print(f"failed_folds: {list(outcome.aggregate.failed_fold_indices)}")
    print(f"idempotent_no_op: {outcome.was_idempotent_no_op}")
    return 0 if outcome.aggregate.overall_status is ExecutionStage.COMPLETED else 2


def cmd_execute_experiment(args: argparse.Namespace) -> int:
    return _run_or_resume(args, is_resume=False)


def cmd_resume_execution(args: argparse.Namespace) -> int:
    return _run_or_resume(args, is_resume=True)


def cmd_train(args: argparse.Namespace) -> int:
    """Identical requirement as `execute` (the experiment must already be
    `ready`, via a prior `prepare-experiment`) and identical transparent-
    resume behavior -- the only difference is `fold_executor=
    MetricsFoldExecutor()`: real per-fold metrics, `predict_proba`
    capture, and a persisted `TrainingMetadata` artifact, on top of
    everything `execute` already does."""
    return _run_or_resume(args, is_resume=False, fold_executor=MetricsFoldExecutor())


def _execution_manifest_store(config: MLExperimentConfig) -> ExecutionManifestStore:
    return ExecutionManifestStore(config.ml_artifacts_root)


def cmd_inspect_execution(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    store = _execution_manifest_store(config)
    manifest = store.load(args.experiment_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)

    aggregate: AggregatedExecutionResult | None = None
    timeline: Timeline | None = None
    for ref in manifest.artifact_references:
        raw = artifact_store.read_artifact(ref.content_hash)
        if ref.category.value == "execution_summary":
            aggregate = AggregatedExecutionResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        elif ref.category.value == "timeline":
            timeline = Timeline.from_json_dict(parse_json_strict(raw.decode("utf-8")))

    if args.format == "json":
        import json

        print(json.dumps(build_execution_report_json(manifest, aggregate=aggregate, timeline=timeline), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_execution_report_markdown(manifest, aggregate=aggregate, timeline=timeline))
    return 0


def cmd_inspect_fold(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    store = _execution_manifest_store(config)
    manifest = store.load(args.experiment_id)
    ref = manifest.fold_result_references.get(args.fold_index)
    if ref is None:
        print(
            f"No fold result recorded for experiment_id={args.experiment_id!r} fold_index={args.fold_index} "
            f"(known folds: {sorted(manifest.fold_result_references)})",
            file=sys.stderr,
        )
        return 1
    raw = MLArtifactStore(config.ml_artifacts_root).read_artifact(ref.content_hash)
    fold_result = FoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    for key, value in fold_result.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def cmd_list_folds(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    store = _execution_manifest_store(config)
    manifest = store.load(args.experiment_id)
    completed = set(manifest.completed_fold_indices)
    failed = set(manifest.failed_fold_indices)
    known = sorted(manifest.fold_result_references)
    if not known:
        print(f"No folds recorded yet for experiment_id={args.experiment_id!r} (stage={manifest.stage.value})")
        return 0
    for fold_index in known:
        status = "completed" if fold_index in completed else "failed" if fold_index in failed else "unknown"
        print(f"fold {fold_index}: {status}")
    return 0


def cmd_verify_execution(args: argparse.Namespace) -> int:
    """Re-audits everything the named experiment's execution has ever
    recorded across its four separate stores (`ExecutionManifest`,
    `ExperimentManifest`, the artifact store, the event log) -- see
    `execution.verification`'s module docstring for exactly what is
    checked and why. Returns 2 (not 1) when the report contains any
    CRITICAL/ERROR issue -- consistent with `validate-experiment`'s own
    "0 unless not ready" convention; a WARNING-only report (e.g. the
    documented manifest-before-event crash window) still returns 0."""
    config = _load_config(Path(args.config))
    report = verify_execution(
        args.experiment_id,
        execution_manifest_store=_execution_manifest_store(config),
        experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        artifact_store=MLArtifactStore(config.ml_artifacts_root),
        event_store=ExperimentEventStore(config.ml_artifacts_root),
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_list_models(args: argparse.Namespace) -> int:  # noqa: ARG001 -- uniform handler signature
    """Milestone 4C: distinct from `list-model-definitions` (a plain
    name/description dump reused from Milestone 4A) -- one line per
    registered model summarizing exactly the capability fields the
    "MODEL REGISTRY" section requires every model to declare."""
    registry = build_model_registry()
    for definition in registry.list_definitions():
        cap = definition.capabilities
        objectives = "/".join(o.value for o in cap.supported_objectives)
        print(
            f"{definition.qualified_name}: objectives=[{objectives}] predict_proba={cap.supports_predict_proba} "
            f"feature_importance={cap.supports_feature_importance} missing_values={cap.supports_missing_values} "
            f"categorical={cap.supports_categorical_features} deterministic={cap.is_deterministic} "
            f"library={cap.library_name}"
        )
    return 0


def cmd_inspect_model(args: argparse.Namespace) -> int:
    """Milestone 4C: distinct from `describe-model-definition` (a raw
    `to_json_dict()` dump) -- prints every declared capability field on
    its own line, including the free-text `seed_usage`/
    `required_preprocessing` declarations `describe-model-definition`'s
    JSON dump renders less readably."""
    registry = build_model_registry()
    definition = registry.get(args.name, args.version)
    cap = definition.capabilities
    print(f"model: {definition.qualified_name}")
    print(f"description: {definition.description}")
    print(f"serializer_id: {definition.serializer_id}")
    print(f"fingerprint: {definition.fingerprint()}")
    print(f"supported_objectives: {[o.value for o in cap.supported_objectives]}")
    print(f"supports_predict_proba: {cap.supports_predict_proba}")
    print(f"supports_feature_importance: {cap.supports_feature_importance}")
    print(f"supports_incremental_fit: {cap.supports_incremental_fit}")
    print(f"supports_missing_values: {cap.supports_missing_values}")
    print(f"supports_categorical_features: {cap.supports_categorical_features}")
    print(f"is_deterministic: {cap.is_deterministic}")
    print(f"seed_usage: {cap.seed_usage}")
    print(f"required_preprocessing: {cap.required_preprocessing}")
    print(f"library_name: {cap.library_name}")
    return 0


def cmd_validate_model(args: argparse.Namespace) -> int:
    """Milestone 4C "MODEL VALIDATION": a standalone, dry-run (writes
    nothing) check of `ml.model_validation.validate_training_data`
    against the FULL reconstructed dataset timeline this config's
    `dataset_binding` refers to -- a coarser, whole-dataset version of
    the SAME per-fold gate `execution.executor.MetricsFoldExecutor` runs
    automatically immediately before every fold's `fit`. Useful for
    catching a gross model/dataset incompatibility (missing values, a
    non-numeric column, a fully-constant label) before ever calling
    `train`."""
    config = _load_config(Path(args.config))
    spec, _ = build_experiment_spec(config)
    definition = build_model_registry().get(spec.model_name, spec.model_version)
    feature_schema = FeatureSchema(feature_names=spec.feature_binding.feature_names)
    trainable = definition.factory.create(
        hyperparameters=spec.hyperparameters, feature_schema=feature_schema, objective=spec.objective,
    )

    research_dataset_store = ResearchDatasetStore(config.dataset.research_storage_root)
    timeline = reconstruct_dataset_timeline(
        research_dataset_store, dataset_id=spec.dataset_binding.dataset_id,
        content_id=spec.dataset_binding.content_id, timestamp_column=_TIMESTAMP_COLUMN,
    )
    features = timeline[list(feature_schema.feature_names)]
    labels = timeline[_LABEL_COLUMN]

    report = validate_training_data(metadata=trainable.metadata, features=features, labels=labels)
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def _load_fold_metrics(config: MLExperimentConfig, experiment_id: str) -> ModelFoldMetrics:
    experiment_manifest = _load_manifest(config, experiment_id)
    execution_manifest = _execution_manifest_store(config).load(experiment_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    fold_identities: list[int] = []
    per_fold_metrics: list[dict[str, float]] = []
    for fold_index in sorted(execution_manifest.fold_result_references):
        ref = execution_manifest.fold_result_references[fold_index]
        raw = artifact_store.read_artifact(ref.content_hash)
        fold_result = FoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        # `FoldResult.metrics` is declared as `Mapping[str, JsonPrimitive]`
        # (Milestone 4B's general-purpose placeholder type); every value
        # `MetricsFoldExecutor` actually stores is a `float` (see `ml.
        # metrics.MetricComputationReport.values`) -- filtered the same
        # way `ml.metrics.aggregate_fold_metrics` already does, never
        # trusting the wider declared type is narrower than it says.
        fold_identities.append(fold_result.fold_index)
        per_fold_metrics.append({
            name: float(value) for name, value in fold_result.metrics.items()
            if not isinstance(value, bool) and isinstance(value, (int, float))
        })
    if not per_fold_metrics:
        raise ValueError(f"No fold results recorded yet for experiment_id={experiment_id!r} -- run 'train' first")
    model_label = f"{experiment_manifest.spec.model_name}@{experiment_manifest.spec.model_version}:{experiment_id[:12]}"
    # `fold_identities` is `FoldResult.fold_index` itself (never just this
    # loop's own enumeration position) -- `ml.comparison` pairs candidate
    # and baseline folds by this IDENTITY, never by raw list position.
    return ModelFoldMetrics(model_name=model_label, per_fold_metrics=tuple(per_fold_metrics), fold_identities=tuple(fold_identities))


def cmd_compare(args: argparse.Namespace) -> int:
    """Milestone 4C "MODEL COMPARISON": loads the candidate's and every
    named baseline's already-persisted per-fold `FoldResult.metrics`
    (never re-running anything), then reports `ml.comparison.
    compare_to_baselines`'s full per-metric statistical comparison plus
    the single `outperforms_all_baselines` gate for `--primary-metric`.
    Returns 2 (not 1) when that gate is `False` -- "never declare a
    model successful unless it statistically outperforms every
    baseline" made an exit-code-checkable fact, not just a printed
    report a caller must parse."""
    config = _load_config(Path(args.config))
    candidate = _load_fold_metrics(config, args.candidate_experiment_id)
    baselines = [_load_fold_metrics(config, eid) for eid in args.baseline_experiment_id]
    report = compare_to_baselines(candidate, baselines)

    for baseline_report in report.baseline_reports:
        print(f"=== {report.candidate_name} vs {baseline_report.baseline_name} ===")
        for mc in baseline_report.metric_comparisons:
            candidate_mean = "n/a" if mc.candidate_aggregate is None else f"{mc.candidate_aggregate.mean:.6f}"
            baseline_mean = "n/a" if mc.baseline_aggregate is None else f"{mc.baseline_aggregate.mean:.6f}"
            p_value = "n/a" if mc.p_value is None else f"{mc.p_value:.6f}"
            reason = f" ({mc.reason})" if mc.reason else ""
            print(
                f"  {mc.metric_name}: candidate_mean={candidate_mean} baseline_mean={baseline_mean} "
                f"paired_n={mc.paired_n} p_value={p_value} outcome={mc.outcome.value}{reason}"
            )

    outperforms = report.outperforms_all_baselines(args.primary_metric)
    print(f"outperforms_all_baselines ({args.primary_metric}): {outperforms}")
    return 0 if outperforms else 2


def _load_optimization_config(path: Path) -> OptimizationConfig:
    return OptimizationConfig.model_validate_json(path.read_text())


def _resolve_optimization_spec(config: OptimizationConfig, *, model_registry: ModelRegistry) -> tuple[OptimizationSpec, str]:
    experiment_config = _load_config(config.experiment_config_path)
    experiment_spec, _ = build_experiment_spec(experiment_config)
    parent_experiment_id = compute_experiment_identity(experiment_spec).experiment_id
    optimization_spec = config.build(experiment=experiment_spec, parent_experiment_id=parent_experiment_id, model_registry=model_registry)
    return optimization_spec, parent_experiment_id


def _build_optimization_runner(config: OptimizationConfig) -> OptimizationRunner:
    experiment_config = _load_config(config.experiment_config_path)
    return OptimizationRunner(
        ml_artifacts_root=config.ml_artifacts_root, model_registry=build_model_registry(),
        research_manifest_store=ResearchManifestStore(experiment_config.dataset.research_storage_root),
        research_dataset_store=ResearchDatasetStore(experiment_config.dataset.research_storage_root),
        experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        additional_serializers=mz.default_serializer_registry(),
    )


def cmd_optimize(args: argparse.Namespace) -> int:
    """Runs (or transparently resumes, if a manifest already exists for
    this exact `OptimizationSpec`'s identity) a full nested walk-forward
    search: for every outer fold, an inner trial search selects a winner
    using ONLY inner-fold evidence, then that winner is refit on the
    complete outer-train partition and evaluated exactly once on the
    untouched outer-test partition. Requires the referenced parent
    experiment to already be prepared (see `prepare-experiment`)."""
    config = _load_optimization_config(Path(args.config))
    registry = build_model_registry()
    optimization_spec, _ = _resolve_optimization_spec(config, model_registry=registry)
    runner = _build_optimization_runner(config)
    outcome = runner.run(optimization_spec)
    print(f"optimization_id: {outcome.manifest.optimization_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(
        f"trials: completed={outcome.manifest.total_trials_completed} failed={outcome.manifest.total_trials_failed} "
        f"invalid={outcome.manifest.total_trials_invalid} pruned={outcome.manifest.total_trials_pruned}"
    )
    print(f"winning_trial_by_outer_fold: {dict(sorted(outcome.manifest.winning_trial_by_outer_fold.items()))}")
    if outcome.manifest.failure_summary:
        print(f"failure_summary: {outcome.manifest.failure_summary}")
    return 0 if outcome.manifest.stage is OptimizationStage.COMPLETED else 2


def cmd_resume_optimization(args: argparse.Namespace) -> int:
    """Resumes a prior, non-terminal optimization -- fails if there is
    nothing to resume. The `OptimizationSpec` is re-loaded from the
    optimization's OWN recorded `OPTIMIZATION_SPEC` artifact (never
    rebuilt from `--config` again), so this command only needs `--config`
    to construct the runner's stores/registry."""
    config = _load_optimization_config(Path(args.config))
    runner = _build_optimization_runner(config)
    outcome = runner.resume(args.optimization_id)
    print(f"optimization_id: {outcome.manifest.optimization_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(
        f"trials: completed={outcome.manifest.total_trials_completed} failed={outcome.manifest.total_trials_failed} "
        f"invalid={outcome.manifest.total_trials_invalid} pruned={outcome.manifest.total_trials_pruned}"
    )
    print(f"resume_count: {outcome.manifest.resume_count}")
    return 0 if outcome.manifest.stage is OptimizationStage.COMPLETED else 2


def _optimization_manifest_store(config: OptimizationConfig) -> OptimizationManifestStore:
    return OptimizationManifestStore(config.ml_artifacts_root)


def _load_outer_fold_results(manifest: OptimizationManifest, artifact_store: MLArtifactStore) -> dict[int, OuterFoldResult]:
    results: dict[int, OuterFoldResult] = {}
    for outer_fold_index, ref in manifest.outer_fold_result_references.items():
        raw = artifact_store.read_artifact(ref.content_hash)
        results[outer_fold_index] = OuterFoldResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    return results


def _load_ranking_tables(outer_fold_results: dict[int, OuterFoldResult], artifact_store: MLArtifactStore) -> dict[int, RankingTable]:
    tables: dict[int, RankingTable] = {}
    for outer_fold_index, result in outer_fold_results.items():
        if result.search_summary_reference is None:
            continue
        raw = artifact_store.read_artifact(result.search_summary_reference.content_hash)
        tables[outer_fold_index] = RankingTable.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    return tables


def _load_optimization_spec(manifest: OptimizationManifest, artifact_store: MLArtifactStore) -> OptimizationSpec | None:
    ref = next((r for r in manifest.artifact_references if r.category is ArtifactCategory.OPTIMIZATION_SPEC), None)
    if ref is None:
        return None
    raw = artifact_store.read_artifact(ref.content_hash)
    return OptimizationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))


def _load_summary_by_category(manifest: OptimizationManifest, artifact_store: MLArtifactStore, category: ArtifactCategory) -> bytes | None:
    ref = next((r for r in manifest.summary_references if r.category is category), None)
    if ref is None:
        return None
    return artifact_store.read_artifact(ref.content_hash)


def cmd_inspect_optimization(args: argparse.Namespace) -> int:
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)

    spec = _load_optimization_spec(manifest, artifact_store)
    outer_fold_results = _load_outer_fold_results(manifest, artifact_store)
    ranking_tables = _load_ranking_tables(outer_fold_results, artifact_store)
    feature_stability_raw = _load_summary_by_category(manifest, artifact_store, ArtifactCategory.FEATURE_STABILITY)
    hyperparameter_stability_raw = _load_summary_by_category(manifest, artifact_store, ArtifactCategory.HYPERPARAMETER_STABILITY)
    feature_stability = FeatureStabilityReport.from_json_dict(parse_json_strict(feature_stability_raw.decode("utf-8"))) if feature_stability_raw else None
    hyperparameter_stability = (
        HyperparameterStabilityReport.from_json_dict(parse_json_strict(hyperparameter_stability_raw.decode("utf-8")))
        if hyperparameter_stability_raw else None
    )

    if args.format == "json":
        import json

        payload = build_optimization_report_json(
            manifest, spec=spec, outer_fold_results=list(outer_fold_results.values()), ranking_tables=ranking_tables,
            feature_stability=feature_stability, hyperparameter_stability=hyperparameter_stability,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_optimization_report_markdown(
            manifest, spec=spec, outer_fold_results=list(outer_fold_results.values()), ranking_tables=ranking_tables,
            feature_stability=feature_stability, hyperparameter_stability=hyperparameter_stability,
        ))
    return 0


def cmd_list_trials(args: argparse.Namespace) -> int:
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    refs = trial_references_for_outer_fold(manifest, args.outer_fold_index)
    if not refs:
        print(f"No trials recorded yet for optimization_id={args.optimization_id!r} outer_fold_index={args.outer_fold_index}", file=sys.stderr)
        return 1
    for trial_number in sorted(refs):
        raw = artifact_store.read_artifact(refs[trial_number].content_hash)
        trial = TrialResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        print(
            f"trial {trial.trial_number}: status={trial.status.value} "
            f"primary_metric_aggregate={trial.primary_metric_aggregate} "
            f"successful_inner_folds={trial.successful_inner_folds}/{trial.total_inner_folds}"
        )
    return 0


def cmd_inspect_trial(args: argparse.Namespace) -> int:
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    refs = trial_references_for_outer_fold(manifest, args.outer_fold_index)
    ref = refs.get(args.trial_number)
    if ref is None:
        print(f"No trial recorded for optimization_id={args.optimization_id!r} outer_fold_index={args.outer_fold_index} trial_number={args.trial_number}", file=sys.stderr)
        return 1
    raw = artifact_store.read_artifact(ref.content_hash)
    trial = TrialResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    for key, value in trial.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def cmd_verify_optimization(args: argparse.Namespace) -> int:
    """Re-audits everything the named optimization has ever recorded --
    see `optimization.verification.verify_optimization`'s module
    docstring for exactly what is checked, including the outer-test-
    isolation event-ordering proof. Returns 2 (not 1) when the report
    contains any CRITICAL/ERROR issue."""
    config = _load_optimization_config(Path(args.config))
    report = verify_optimization(
        args.optimization_id, optimization_manifest_store=_optimization_manifest_store(config),
        experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        artifact_store=MLArtifactStore(config.ml_artifacts_root),
        event_store=OptimizationEventStore(config.ml_artifacts_root),
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_compare_optimization_candidates(args: argparse.Namespace) -> int:
    """Prints one outer fold's complete, deterministic trial ranking
    table (see `optimization.candidates.rank_trials`) plus a near-tie
    flag between the winner and its closest competitor -- never a
    statistical test between every pair of trials."""
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    outer_fold_results = _load_outer_fold_results(manifest, artifact_store)
    result = outer_fold_results.get(args.outer_fold_index)
    if result is None or result.search_summary_reference is None:
        print(f"No completed ranking table for optimization_id={args.optimization_id!r} outer_fold_index={args.outer_fold_index}", file=sys.stderr)
        return 1
    raw = artifact_store.read_artifact(result.search_summary_reference.content_hash)
    table = RankingTable.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    print(f"primary_metric: {table.primary_metric}")
    for entry in table.entries:
        print(
            f"rank {entry.rank}: trial={entry.trial_number} valid={entry.is_valid} "
            f"primary_metric_aggregate={entry.primary_metric_aggregate} "
            f"successful_inner_folds={entry.successful_inner_folds} "
            f"mean_selected_feature_count={entry.mean_selected_feature_count}"
        )
    for warning in flag_near_tied_top_candidates(table):
        print(f"WARNING: {warning}")
    return 0


def cmd_feature_stability(args: argparse.Namespace) -> int:
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    raw = _load_summary_by_category(manifest, artifact_store, ArtifactCategory.FEATURE_STABILITY)
    if raw is None:
        print(f"No feature-stability summary recorded yet for optimization_id={args.optimization_id!r}", file=sys.stderr)
        return 1
    report = FeatureStabilityReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    print(f"total_evaluations: {report.total_evaluations}")
    for entry in report.entries:
        print(
            f"{entry.feature_name}: selection_frequency={entry.selection_frequency:.3f} "
            f"selected_in_winning_candidate_frequency={entry.selected_in_winning_candidate_frequency:.3f} "
            f"mean_rank={entry.mean_rank} mean_score={entry.mean_score}"
        )
    if report.pairwise_jaccard is not None:
        print(f"pairwise_jaccard_mean: {report.pairwise_jaccard.mean:.3f}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    return 0


def cmd_hyperparameter_stability(args: argparse.Namespace) -> int:
    config = _load_optimization_config(Path(args.config))
    manifest = _optimization_manifest_store(config).load(args.optimization_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    raw = _load_summary_by_category(manifest, artifact_store, ArtifactCategory.HYPERPARAMETER_STABILITY)
    if raw is None:
        print(f"No hyperparameter-stability summary recorded yet for optimization_id={args.optimization_id!r}", file=sys.stderr)
        return 1
    report = HyperparameterStabilityReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    for numeric in report.numeric_parameters:
        print(f"{numeric.parameter_name}: mean={numeric.mean:.6g} std={numeric.std:.6g} boundary_hit_frequency={numeric.boundary_hit_frequency:.3f}")
    for categorical in report.categorical_parameters:
        print(f"{categorical.parameter_name}: {dict(sorted(categorical.choice_frequencies.items()))}")
    print(f"trial_score_dispersion: {report.trial_score_dispersion}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    return 0


def _load_calibration_config(path: Path) -> CalibrationConfig:
    return CalibrationConfig.model_validate_json(path.read_text())


def _calibration_manifest_store(config: CalibrationConfig) -> CalibrationManifestStore:
    return CalibrationManifestStore(config.ml_artifacts_root)


def _resolve_calibration_spec(config: CalibrationConfig, *, model_registry: ModelRegistry) -> CalibrationSpec:
    experiment_manifest_store = ExperimentManifestStore(config.ml_artifacts_root)
    experiment_manifest = experiment_manifest_store.load(config.source_experiment_id)
    model_definition = model_registry.get(experiment_manifest.spec.model_name, experiment_manifest.spec.model_version)
    return config.build(experiment_spec=experiment_manifest.spec, model_definition=model_definition)


def _build_calibration_runner(config: CalibrationConfig) -> CalibrationRunner:
    return CalibrationRunner(
        ml_artifacts_root=config.ml_artifacts_root, model_registry=build_model_registry(),
        research_manifest_store=ResearchManifestStore(config.research_storage_root),
        research_dataset_store=ResearchDatasetStore(config.research_storage_root),
        experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        optimization_manifest_store=(OptimizationManifestStore(config.ml_artifacts_root) if config.source_optimization_id else None),
        additional_serializers=mz.default_serializer_registry(),
    )


def cmd_create_calibration_spec(args: argparse.Namespace) -> int:
    """Dry-run: resolves `--source-experiment-id`'s bound experiment,
    builds and validates a `CalibrationSpec` from `--config`, and prints
    its deterministic `calibration_id` -- writes nothing. Mirrors
    `validate-experiment`'s "preflight, no side effects" convention."""
    config = _load_calibration_config(Path(args.config))
    spec = _resolve_calibration_spec(config, model_registry=build_model_registry())
    identity = compute_calibration_identity(spec)
    print(f"calibration_id: {identity.calibration_id}")
    print(f"source_experiment_id: {spec.source_experiment_id}")
    print(f"source_optimization_id: {spec.source_optimization_id}")
    print(f"calibration_method_candidates: {[m.value for m in spec.calibration_method_candidates]}")
    print(f"threshold_policy: {spec.threshold_spec.policy.value}")
    return 0


def cmd_run_calibration(args: argparse.Namespace) -> int:
    """Runs (or transparently resumes, if a manifest already exists for
    this exact `CalibrationSpec`'s identity) a full leakage-safe
    calibration: for every outer fold, inner out-of-fold predictions fit
    and freeze a calibrator/threshold/confidence/uncertainty/abstention
    policy using ONLY inner-fold evidence, then the base model is refit
    on the complete outer-train partition and evaluated exactly once on
    the untouched outer-test partition. Requires the referenced source
    experiment to already be prepared (see `prepare-experiment`)."""
    config = _load_calibration_config(Path(args.config))
    spec = _resolve_calibration_spec(config, model_registry=build_model_registry())
    runner = _build_calibration_runner(config)
    outcome = runner.run(spec)
    print(f"calibration_id: {outcome.manifest.calibration_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(f"completed_outer_fold_indices: {list(outcome.manifest.completed_outer_fold_indices)}")
    if outcome.manifest.failure_summary:
        print(f"failure_summary: {outcome.manifest.failure_summary}")
    return 0 if outcome.manifest.stage is CalibrationStage.COMPLETED else 2


def cmd_resume_calibration(args: argparse.Namespace) -> int:
    """Resumes a prior calibration. An already-`COMPLETED` calibration is
    a safe idempotent no-op (exit 0, matching `run-calibration`'s own
    idempotency guarantee) -- this command does NOT raise for "nothing
    left to resume" the way `resume-optimization` does; only a calibration
    with no manifest at all, or one already `FAILED`, raises
    `CalibrationResumeError` (exit 1). The `CalibrationSpec` is re-loaded
    from the calibration's OWN recorded `CALIBRATION_SPEC` artifact (never
    rebuilt from `--config` again), so this command only needs `--config`
    to construct the runner's stores/registry."""
    config = _load_calibration_config(Path(args.config))
    runner = _build_calibration_runner(config)
    outcome = runner.resume(args.calibration_id)
    print(f"calibration_id: {outcome.manifest.calibration_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(f"completed_outer_fold_indices: {list(outcome.manifest.completed_outer_fold_indices)}")
    print(f"resume_count: {outcome.manifest.resume_count}")
    return 0 if outcome.manifest.stage is CalibrationStage.COMPLETED else 2


def _load_calibration_outer_fold_results(manifest: CalibrationManifest, artifact_store: MLArtifactStore) -> dict[int, OuterFoldCalibrationResult]:
    results: dict[int, OuterFoldCalibrationResult] = {}
    for outer_fold_index, ref in manifest.outer_fold_result_references.items():
        raw = artifact_store.read_artifact(ref.content_hash)
        results[outer_fold_index] = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    return results


def _load_calibration_spec_artifact(manifest: CalibrationManifest, artifact_store: MLArtifactStore) -> CalibrationSpec | None:
    if manifest.spec_reference is None:
        return None
    raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
    return CalibrationSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))


def cmd_inspect_calibration(args: argparse.Namespace) -> int:
    config = _load_calibration_config(Path(args.config))
    manifest = _calibration_manifest_store(config).load(args.calibration_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    spec = _load_calibration_spec_artifact(manifest, artifact_store)
    outer_fold_results = _load_calibration_outer_fold_results(manifest, artifact_store)

    if args.format == "json":
        import json

        payload = build_calibration_report_json(manifest, spec=spec, outer_fold_results=list(outer_fold_results.values()))
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_calibration_report_markdown(manifest, spec=spec, outer_fold_results=list(outer_fold_results.values())))
    return 0


def cmd_report_calibration(args: argparse.Namespace) -> int:
    """Alias for `inspect-calibration` -- a completed calibration's
    aggregate report IS what `inspect-calibration` prints (mirrors
    `optimization`'s identical choice not to duplicate `inspect-
    optimization`'s content under a second command name)."""
    return cmd_inspect_calibration(args)


def cmd_inspect_calibration_fold(args: argparse.Namespace) -> int:
    config = _load_calibration_config(Path(args.config))
    manifest = _calibration_manifest_store(config).load(args.calibration_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    reference = manifest.outer_fold_result_references.get(args.outer_fold_index)
    if reference is None:
        print(f"No recorded result for calibration_id={args.calibration_id!r} outer_fold_index={args.outer_fold_index}", file=sys.stderr)
        return 1
    raw = artifact_store.read_artifact(reference.content_hash)
    result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    print(f"outer_fold_index: {result.outer_fold_index}")
    print(f"outer_train_row_count: {result.outer_train_row_count}, outer_test_row_count: {result.outer_test_row_count}")
    print(f"classification_metrics: {dict(sorted(result.classification_metrics.items()))}")
    print(f"calibration_metrics_on_outer_test: {dict(sorted(result.calibration_metrics_on_outer_test.items()))}")
    print(f"selective_prediction_summary: {dict(sorted(result.selective_prediction_summary.items()))}")
    decision_counts = {d: result.decisions.count(d) for d in sorted(set(result.decisions))}
    print(f"decision_counts: {decision_counts}")
    return 0


def cmd_verify_calibration(args: argparse.Namespace) -> int:
    """Re-audits everything the named calibration has ever recorded --
    see `calibration.verification.verify_calibration`'s module docstring
    for exactly what is checked, including the recomputation proof that
    persisted calibrated probabilities/decisions are actually
    reproducible from persisted parameters. Returns 2 (not 1) when the
    report contains any CRITICAL/ERROR issue."""
    config = _load_calibration_config(Path(args.config))
    report = verify_calibration(
        args.calibration_id, calibration_manifest_store=_calibration_manifest_store(config),
        artifact_store=MLArtifactStore(config.ml_artifacts_root), event_store=CalibrationEventStore(config.ml_artifacts_root),
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_compare_calibration(args: argparse.Namespace) -> int:
    """Prints one metric's per-outer-fold values, side by side, for two
    completed calibrations -- never a statistical claim, just the raw
    numbers a human compares."""
    config = _load_calibration_config(Path(args.config))
    store = _calibration_manifest_store(config)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    left = _load_calibration_outer_fold_results(store.load(args.calibration_id), artifact_store)
    right = _load_calibration_outer_fold_results(store.load(args.baseline_calibration_id), artifact_store)
    for outer_fold_index in sorted(set(left) | set(right)):
        left_value = left[outer_fold_index].classification_metrics.get(args.metric) if outer_fold_index in left else None
        right_value = right[outer_fold_index].classification_metrics.get(args.metric) if outer_fold_index in right else None
        print(f"outer_fold {outer_fold_index}: {args.calibration_id[:12]}={left_value} {args.baseline_calibration_id[:12]}={right_value}")
    return 0


def _load_backtest_config(path: Path) -> BacktestConfig:
    return BacktestConfig.model_validate_json(path.read_text())


def _backtest_manifest_store(config: BacktestConfig) -> BacktestManifestStore:
    return BacktestManifestStore(config.ml_artifacts_root)


def _resolve_backtest_spec(config: BacktestConfig) -> BacktestSpec:
    calibration_manifest = CalibrationManifestStore(config.ml_artifacts_root).load(config.source_calibration_id)
    experiment_manifest = ExperimentManifestStore(config.ml_artifacts_root).load(calibration_manifest.source_experiment_id)
    return config.build(calibration_manifest=calibration_manifest, experiment_spec=experiment_manifest.spec)


@dataclass(frozen=True, slots=True)
class _BacktestSourceStores:
    """Milestone 5.2, Section 3: the same source-resolution stores
    `BacktestRunner` itself needs (`resolve_backtest_inputs`'s exact
    dependency list) -- extracted so `cmd_verify_backtest` can pass them
    to `verify_backtest`'s raw-source reconstruction without duplicating
    `_build_backtest_runner`'s construction logic."""

    calibration_manifest_store: CalibrationManifestStore
    experiment_manifest_store: ExperimentManifestStore
    execution_manifest_store: ExecutionManifestStore
    research_manifest_store: ResearchManifestStore
    research_dataset_store: ResearchDatasetStore
    dataset_loader: DatasetLoader


def _backtest_source_stores(config: BacktestConfig) -> _BacktestSourceStores:
    return _BacktestSourceStores(
        calibration_manifest_store=CalibrationManifestStore(config.ml_artifacts_root),
        experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root), execution_manifest_store=ExecutionManifestStore(config.ml_artifacts_root),
        research_manifest_store=ResearchManifestStore(config.research_storage_root), research_dataset_store=ResearchDatasetStore(config.research_storage_root),
        dataset_loader=DatasetLoader(CanonicalStore(config.historical_storage_root), ManifestStore(config.historical_storage_root)),
    )


def _build_backtest_runner(config: BacktestConfig) -> BacktestRunner:
    stores = _backtest_source_stores(config)
    return BacktestRunner(
        ml_artifacts_root=config.ml_artifacts_root, calibration_manifest_store=stores.calibration_manifest_store,
        experiment_manifest_store=stores.experiment_manifest_store, execution_manifest_store=stores.execution_manifest_store,
        research_manifest_store=stores.research_manifest_store, research_dataset_store=stores.research_dataset_store,
        dataset_loader=stores.dataset_loader,
    )


def cmd_create_backtest_spec(args: argparse.Namespace) -> int:
    """Dry-run: resolves `--source-calibration-id`'s bound calibration/
    experiment, builds and validates a `BacktestSpec` from `--config`, and
    prints its deterministic `backtest_id` -- writes nothing. Mirrors
    `create-calibration-spec`'s identical "preflight, no side effects"
    convention."""
    config = _load_backtest_config(Path(args.config))
    spec = _resolve_backtest_spec(config)
    identity = compute_backtest_identity(spec)
    print(f"backtest_id: {identity.backtest_id}")
    print(f"source_calibration_id: {spec.source_calibration_id}")
    print(f"source_experiment_id: {spec.source_experiment_id}")
    print(f"instrument_identity: {spec.instrument_identity} @ {spec.bar_interval.value}")
    print(f"signal_mapping: {spec.signal_mapping.kind.value}, position_mode: {spec.position_mode.value}")
    return 0


def cmd_run_backtest(args: argparse.Namespace) -> int:
    """Runs (or transparently resumes, if a manifest already exists for
    this exact `BacktestSpec`'s identity) a full leakage-safe backtest:
    for every outer fold, independently re-verified calibrated
    predictions are mapped to signals, simulated into fills and trades
    under the declared cost model, and evaluated into equity curves,
    drawdown, financial metrics, benchmarks, cost sensitivity, and
    confidence/uncertainty bucket analysis. Requires the referenced
    source calibration to already be `COMPLETED` (see `run-calibration`)
    and the referenced source execution to already be `COMPLETED`."""
    config = _load_backtest_config(Path(args.config))
    spec = _resolve_backtest_spec(config)
    runner = _build_backtest_runner(config)
    outcome = runner.run(spec)
    print(f"backtest_id: {outcome.manifest.backtest_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(f"completed_outer_fold_indices: {list(outcome.manifest.completed_outer_fold_indices)}")
    if outcome.manifest.failure_summary:
        print(f"failure_summary: {outcome.manifest.failure_summary}")
    return 0 if outcome.manifest.stage is BacktestStage.COMPLETED else 2


def cmd_resume_backtest(args: argparse.Namespace) -> int:
    """Resumes a prior backtest. An already-`COMPLETED` backtest is a safe
    idempotent no-op (exit 0, matching `run-backtest`'s own idempotency
    guarantee). The `BacktestSpec` is re-loaded from the backtest's OWN
    recorded `BACKTEST_SPEC` artifact (never rebuilt from `--config`
    again), so this command only needs `--config` to construct the
    runner's stores."""
    config = _load_backtest_config(Path(args.config))
    runner = _build_backtest_runner(config)
    outcome = runner.resume(args.backtest_id)
    print(f"backtest_id: {outcome.manifest.backtest_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(f"completed_outer_fold_indices: {list(outcome.manifest.completed_outer_fold_indices)}")
    print(f"resume_count: {outcome.manifest.resume_count}")
    return 0 if outcome.manifest.stage is BacktestStage.COMPLETED else 2


def _print_backtest_lock_diagnostics(diagnostics: BacktestLockDiagnostics) -> None:
    for key, value in diagnostics.to_json_dict().items():
        print(f"{key}: {value}")


def cmd_inspect_backtest_lock(args: argparse.Namespace) -> int:
    """Milestone 5.2, Section 6: read-only lock diagnostics -- never
    acquires, contests, or removes the lock. Safe to run at any time,
    including against a genuinely live run."""
    config = _load_backtest_config(Path(args.config))
    diagnostics = inspect_backtest_lock(args.backtest_id, ml_artifacts_root=config.ml_artifacts_root)
    _print_backtest_lock_diagnostics(diagnostics)
    return 0


def cmd_recover_backtest_lock(args: argparse.Namespace) -> int:
    """Milestone 5.2, Section 6: the explicit, diagnosed, auditable
    alternative to undocumented manual deletion of `.backtest_run.lock`.
    Always prints the diagnostics it based its decision on. Without
    `--force`, refuses to touch a lock that has not gone stale by age
    (never steals a live lock). `--force` must only be passed after a
    human has reviewed the printed diagnostics and independently
    confirmed the owning process is dead -- see `recover_backtest_lock`'s
    own docstring."""
    config = _load_backtest_config(Path(args.config))
    diagnostics = recover_backtest_lock(args.backtest_id, ml_artifacts_root=config.ml_artifacts_root, force=args.force)
    _print_backtest_lock_diagnostics(diagnostics)
    print("recovered: " + ("nothing to do (no lock present)" if not diagnostics.lock_exists else "yes"))
    return 0


def _load_backtest_outer_fold_results(manifest: BacktestManifest, artifact_store: MLArtifactStore) -> dict[int, OuterFoldBacktestResult]:
    results: dict[int, OuterFoldBacktestResult] = {}
    for outer_fold_index, ref in manifest.outer_fold_result_references.items():
        raw = artifact_store.read_artifact(ref.content_hash)
        results[outer_fold_index] = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    return results


def _load_backtest_spec_artifact(manifest: BacktestManifest, artifact_store: MLArtifactStore) -> BacktestSpec | None:
    if manifest.spec_reference is None:
        return None
    raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
    return BacktestSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))


def _load_backtest_stitched_equity(manifest: BacktestManifest, artifact_store: MLArtifactStore) -> StitchedWalkForwardEquity | None:
    if manifest.stitched_equity_reference is None:
        return None
    raw = artifact_store.read_artifact(manifest.stitched_equity_reference.content_hash)
    return StitchedWalkForwardEquity.from_json_dict(parse_json_strict(raw.decode("utf-8")))


def cmd_inspect_backtest(args: argparse.Namespace) -> int:
    config = _load_backtest_config(Path(args.config))
    manifest = _backtest_manifest_store(config).load(args.backtest_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    spec = _load_backtest_spec_artifact(manifest, artifact_store)
    outer_fold_results = _load_backtest_outer_fold_results(manifest, artifact_store)
    stitched = _load_backtest_stitched_equity(manifest, artifact_store)

    if args.format == "json":
        import json

        payload = build_backtest_report_json(
            manifest, spec=spec, outer_fold_results=list(outer_fold_results.values()), stitched=stitched,
            stitched_equity_reference=manifest.stitched_equity_reference,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_backtest_report_markdown(
            manifest, spec=spec, outer_fold_results=list(outer_fold_results.values()), stitched=stitched,
            stitched_equity_reference=manifest.stitched_equity_reference,
        ))
    return 0


def cmd_report_backtest(args: argparse.Namespace) -> int:
    """Alias for `inspect-backtest` -- a completed backtest's aggregate
    report IS what `inspect-backtest` prints (mirrors `calibration`'s
    identical choice not to duplicate `inspect-calibration`'s content
    under a second command name)."""
    return cmd_inspect_backtest(args)


def cmd_inspect_backtest_fold(args: argparse.Namespace) -> int:
    config = _load_backtest_config(Path(args.config))
    manifest = _backtest_manifest_store(config).load(args.backtest_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    reference = manifest.outer_fold_result_references.get(args.outer_fold_index)
    if reference is None:
        print(f"No recorded result for backtest_id={args.backtest_id!r} outer_fold_index={args.outer_fold_index}", file=sys.stderr)
        return 1
    raw = artifact_store.read_artifact(reference.content_hash)
    result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    print(f"outer_fold_index: {result.outer_fold_index}")
    print(f"outer_test_row_count: {result.outer_test_row_count}, closed_trade_count: {result.closed_trade_count}")
    print(f"meets_minimum_trade_threshold: {result.meets_minimum_trade_threshold}")
    print(f"financial_metrics: {dict(sorted(result.financial_metrics.items()))}")
    if result.skipped_metrics:
        print(f"skipped_metrics: {dict(sorted(result.skipped_metrics.items()))}")
    return 0


def cmd_verify_backtest(args: argparse.Namespace) -> int:
    """Re-audits everything the named backtest has ever recorded -- see
    `backtesting.verification.verify_backtest`'s module docstring for
    exactly what is checked, including the recomputation proof that
    persisted financial metrics are actually reproducible from a
    persisted `TradeSet`. Returns 2 (not 1) when the report contains any
    CRITICAL/ERROR issue."""
    config = _load_backtest_config(Path(args.config))
    stores = _backtest_source_stores(config)
    report = verify_backtest(
        args.backtest_id, backtest_manifest_store=_backtest_manifest_store(config),
        artifact_store=MLArtifactStore(config.ml_artifacts_root), event_store=BacktestEventStore(config.ml_artifacts_root),
        calibration_manifest_store=stores.calibration_manifest_store, experiment_manifest_store=stores.experiment_manifest_store,
        execution_manifest_store=stores.execution_manifest_store, research_manifest_store=stores.research_manifest_store,
        research_dataset_store=stores.research_dataset_store, dataset_loader=stores.dataset_loader,
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_compare_backtests(args: argparse.Namespace) -> int:
    """Prints one metric's per-outer-fold values, side by side, for two
    completed backtests -- never a statistical claim, just the raw
    numbers a human compares."""
    config = _load_backtest_config(Path(args.config))
    store = _backtest_manifest_store(config)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    left = _load_backtest_outer_fold_results(store.load(args.backtest_id), artifact_store)
    right = _load_backtest_outer_fold_results(store.load(args.baseline_backtest_id), artifact_store)
    for outer_fold_index in sorted(set(left) | set(right)):
        left_value = left[outer_fold_index].financial_metrics.get(args.metric) if outer_fold_index in left else None
        right_value = right[outer_fold_index].financial_metrics.get(args.metric) if outer_fold_index in right else None
        print(f"outer_fold {outer_fold_index}: {args.backtest_id[:12]}={left_value} {args.baseline_backtest_id[:12]}={right_value}")
    return 0


# --------------------------------------------------------------------------
# Milestone 6: statistical robustness / promotion-gate commands
# --------------------------------------------------------------------------
def _load_robustness_config(path: Path) -> RobustnessConfig:
    return RobustnessConfig.model_validate_json(path.read_text())


def _robustness_manifest_store(config: RobustnessConfig) -> RobustnessManifestStore:
    return RobustnessManifestStore(config.ml_artifacts_root)


@dataclass(frozen=True, slots=True)
class _RobustnessSourceStores:
    """The same source-resolution stores `RobustnessRunner` itself needs
    -- extracted so `cmd_verify_robustness`/`cmd_create_robustness_spec`
    can reuse them without duplicating `_build_robustness_runner`'s
    construction logic (mirrors `_BacktestSourceStores`'s identical role
    one layer down)."""

    backtest_manifest_store: BacktestManifestStore
    backtest_event_store: BacktestEventStore
    calibration_manifest_store: CalibrationManifestStore
    experiment_manifest_store: ExperimentManifestStore
    execution_manifest_store: ExecutionManifestStore
    research_manifest_store: ResearchManifestStore
    research_dataset_store: ResearchDatasetStore
    dataset_loader: DatasetLoader


def _robustness_source_stores(config: RobustnessConfig) -> _RobustnessSourceStores:
    return _RobustnessSourceStores(
        backtest_manifest_store=BacktestManifestStore(config.ml_artifacts_root), backtest_event_store=BacktestEventStore(config.ml_artifacts_root),
        calibration_manifest_store=CalibrationManifestStore(config.ml_artifacts_root), experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        execution_manifest_store=ExecutionManifestStore(config.ml_artifacts_root), research_manifest_store=ResearchManifestStore(config.research_storage_root),
        research_dataset_store=ResearchDatasetStore(config.research_storage_root),
        dataset_loader=DatasetLoader(CanonicalStore(config.historical_storage_root), ManifestStore(config.historical_storage_root)),
    )


def _build_robustness_runner(config: RobustnessConfig) -> RobustnessRunner:
    stores = _robustness_source_stores(config)
    return RobustnessRunner(
        ml_artifacts_root=config.ml_artifacts_root, backtest_manifest_store=stores.backtest_manifest_store, backtest_event_store=stores.backtest_event_store,
        calibration_manifest_store=stores.calibration_manifest_store, experiment_manifest_store=stores.experiment_manifest_store,
        execution_manifest_store=stores.execution_manifest_store, research_manifest_store=stores.research_manifest_store,
        research_dataset_store=stores.research_dataset_store, dataset_loader=stores.dataset_loader,
    )


def _resolve_robustness_spec(config: RobustnessConfig) -> RobustnessSpec:
    """Loads `--source-backtest-id`'s OWN persisted `BacktestSpec` and
    derives every identity-bearing `RobustnessSpec` field from it --
    exactly `RobustnessConfig.build`'s own documented reasoning, never a
    hand-typed, driftable duplicate."""
    backtest_manifest = BacktestManifestStore(config.ml_artifacts_root).load(config.source_backtest_id)
    if backtest_manifest.spec_reference is None:
        raise ValueError(f"Source backtest {config.source_backtest_id!r} has no recorded BACKTEST_SPEC artifact")
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    raw = artifact_store.read_artifact(backtest_manifest.spec_reference.content_hash)
    backtest_spec = BacktestSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    return config.build(source_backtest_spec=backtest_spec)


_T = TypeVar("_T")


def _load_robustness_named_artifact(
    manifest: RobustnessManifest, artifact_store: MLArtifactStore, *, kind: str, decoder: Callable[[dict[str, object]], _T],
) -> _T | None:
    reference = manifest.artifact(kind)
    if reference is None:
        return None
    raw = artifact_store.read_artifact(reference.content_hash)
    return decoder(parse_json_strict(raw.decode("utf-8")))


def cmd_create_robustness_spec(args: argparse.Namespace) -> int:
    """Milestone 6: dry-run -- resolves `--source-backtest-id`'s own
    persisted `BacktestSpec`, builds and validates a `RobustnessSpec` from
    `--config`, and prints its deterministic `robustness_id`. Writes
    nothing. Mirrors `create-backtest-spec`'s identical convention."""
    config = _load_robustness_config(Path(args.config))
    spec = _resolve_robustness_spec(config)
    identity = compute_robustness_identity(spec)
    print(f"robustness_id: {identity.robustness_id}")
    print(f"source_backtest_id: {spec.source_backtest_id}")
    print(f"return_series_kind: {spec.return_series_kind.value}")
    print(f"bootstrap: method={spec.bootstrap_spec.method.value} repetitions={spec.bootstrap_spec.repetitions} confidence_level={spec.bootstrap_spec.confidence_level}")
    print(f"multiple_testing_correction: {spec.multiple_testing_correction.value}")
    print(f"minimum_fold_count={spec.minimum_fold_count} minimum_trade_count={spec.minimum_trade_count} minimum_effective_sample_size={spec.minimum_effective_sample_size}")
    return 0


def cmd_run_robustness(args: argparse.Namespace) -> int:
    """Milestone 6: runs (or transparently resumes) a full statistical-
    robustness analysis of `--source-backtest-id`'s own COMPLETED,
    independently re-verified backtest: source verification, return-
    series construction, dependence-aware bootstrap, downside analysis,
    fold-stability/concentration analysis, parameter-sensitivity and
    cost/latency stress analysis, regime analysis, standalone selection
    eligibility, promotion-gate evaluation, and independent verification."""
    config = _load_robustness_config(Path(args.config))
    spec = _resolve_robustness_spec(config)
    runner = _build_robustness_runner(config)
    outcome = runner.run(spec)
    print(f"robustness_id: {outcome.manifest.robustness_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    if outcome.manifest.failure_summary:
        print(f"failure_summary: {outcome.manifest.failure_summary}")
    return 0 if outcome.manifest.stage is RobustnessStage.COMPLETED else 2


def cmd_resume_robustness(args: argparse.Namespace) -> int:
    """Milestone 6: resumes a prior, non-terminal robustness run. An
    already-`COMPLETED` run is a safe idempotent no-op (exit 0). The
    `RobustnessSpec` is re-loaded from the run's OWN recorded
    `ROBUSTNESS_SPEC` artifact, never rebuilt from `--config` again."""
    config = _load_robustness_config(Path(args.config))
    runner = _build_robustness_runner(config)
    outcome = runner.resume(args.robustness_id)
    print(f"robustness_id: {outcome.manifest.robustness_id}")
    print(f"stage: {outcome.manifest.stage.value}")
    print(f"resume_count: {outcome.manifest.resume_count}")
    return 0 if outcome.manifest.stage is RobustnessStage.COMPLETED else 2


def cmd_inspect_robustness(args: argparse.Namespace) -> int:
    """Milestone 6: prints a human-readable (or JSON) summary of one
    robustness run -- its manifest, and, once recorded, its promotion
    decision and every gate's measured value/outcome/reason."""
    config = _load_robustness_config(Path(args.config))
    manifest = _robustness_manifest_store(config).load(args.robustness_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)

    if args.format == "json":
        import json

        payload = {
            "robustness_id": manifest.robustness_id, "source_backtest_id": manifest.source_backtest_id, "stage": manifest.stage.value,
            "created_at": manifest.created_at, "updated_at": manifest.updated_at, "completed_at": manifest.completed_at,
            "failure_summary": manifest.failure_summary, "resume_count": manifest.resume_count,
            "named_artifacts": {k: v.to_json_dict() for k, v in sorted(manifest.named_artifacts.items())},
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0

    print(f"robustness_id: {manifest.robustness_id}")
    print(f"source_backtest_id: {manifest.source_backtest_id}")
    print(f"stage: {manifest.stage.value}")
    print(f"resume_count: {manifest.resume_count}")
    if manifest.failure_summary:
        print(f"failure_summary: {manifest.failure_summary}")
    promotion = _load_robustness_named_artifact(manifest, artifact_store, kind="promotion_decision", decoder=PromotionDecision.from_json_dict)
    if promotion is not None:
        print(f"promotion_decision: {promotion.decision.value}")
        print(f"promotion_decision_reason: {promotion.decision_reason}")
        for gate in promotion.gate_evaluations:
            print(f"  gate[{gate.gate_name}]: mandatory={gate.mandatory} outcome={gate.outcome.value} measured={gate.measured_value} min={gate.minimum_value} max={gate.maximum_value}")
        print(f"disclaimer: {promotion.disclaimer}")
    return 0


def cmd_report_robustness(args: argparse.Namespace) -> int:
    """Alias for `inspect-robustness` -- mirrors `report-backtest`'s
    identical choice not to duplicate `inspect-backtest`'s content under a
    second command name."""
    return cmd_inspect_robustness(args)


def cmd_verify_robustness(args: argparse.Namespace) -> int:
    """Milestone 6: independently RECOMPUTES every deterministic analysis
    this robustness run persisted and compares it against what was
    recorded -- see `robustness.verification.verify_robustness`'s module
    docstring. Returns 2 (not 1) when the report contains any CRITICAL/
    ERROR issue."""
    config = _load_robustness_config(Path(args.config))
    stores = _robustness_source_stores(config)
    report = verify_robustness(
        args.robustness_id, robustness_manifest_store=_robustness_manifest_store(config), artifact_store=MLArtifactStore(config.ml_artifacts_root),
        backtest_manifest_store=stores.backtest_manifest_store, backtest_event_store=stores.backtest_event_store,
        calibration_manifest_store=stores.calibration_manifest_store, experiment_manifest_store=stores.experiment_manifest_store,
        execution_manifest_store=stores.execution_manifest_store, research_manifest_store=stores.research_manifest_store,
        research_dataset_store=stores.research_dataset_store, dataset_loader=stores.dataset_loader,
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_compare_robustness(args: argparse.Namespace) -> int:
    """Milestone 6: prints two robustness runs' headline stage/promotion
    outcome side by side -- never a statistical claim, just the raw
    recorded outcomes a human compares."""
    config = _load_robustness_config(Path(args.config))
    store = _robustness_manifest_store(config)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    left = store.load(args.robustness_id)
    right = store.load(args.baseline_robustness_id)
    left_promotion = _load_robustness_named_artifact(left, artifact_store, kind="promotion_decision", decoder=PromotionDecision.from_json_dict)
    right_promotion = _load_robustness_named_artifact(right, artifact_store, kind="promotion_decision", decoder=PromotionDecision.from_json_dict)
    print(f"{args.robustness_id[:12]}: stage={left.stage.value} promotion_decision={left_promotion.decision.value if left_promotion else None}")
    print(f"{args.baseline_robustness_id[:12]}: stage={right.stage.value} promotion_decision={right_promotion.decision.value if right_promotion else None}")
    return 0


def cmd_inspect_promotion_decision(args: argparse.Namespace) -> int:
    """Milestone 6: prints one robustness run's `PromotionDecision` in
    full -- every gate's name/measured value/required bound(s)/pass-fail-
    skip outcome/reason, never merely the final verdict."""
    config = _load_robustness_config(Path(args.config))
    manifest = _robustness_manifest_store(config).load(args.robustness_id)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    promotion = _load_robustness_named_artifact(manifest, artifact_store, kind="promotion_decision", decoder=PromotionDecision.from_json_dict)
    if promotion is None:
        print(f"No promotion decision recorded yet for robustness_id={args.robustness_id!r} (stage={manifest.stage.value!r})", file=sys.stderr)
        return 1
    print(f"robustness_id: {promotion.robustness_id}")
    print(f"decision: {promotion.decision.value}")
    print(f"decision_reason: {promotion.decision_reason}")
    for gate in promotion.gate_evaluations:
        print(f"gate[{gate.gate_name}] mandatory={gate.mandatory} outcome={gate.outcome.value} measured={gate.measured_value} minimum={gate.minimum_value} maximum={gate.maximum_value} reason={gate.reason}")
    print(f"disclaimer: {promotion.disclaimer}")
    return 0


def cmd_inspect_strategy_family(args: argparse.Namespace) -> int:
    """Milestone 6: prints one durable `StrategyFamily` record, looked up
    by its content-addressed `--content-hash` (mirrors `verify-artifact`'s
    identical content-hash-based lookup convention -- `StrategyFamily`
    has no dedicated manifest store of its own; the caller who originally
    built and persisted it via `multiple_testing.build_strategy_family`
    already has the `ArtifactReference` this command needs)."""
    config = _load_robustness_config(Path(args.config))
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    raw = artifact_store.read_artifact(args.content_hash)
    family = StrategyFamily.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    print(f"family_id: {family.family_id}")
    print(f"candidate_count: {family.candidate_count}")
    print(f"candidate_backtest_ids: {list(family.candidate_backtest_ids)}")
    print(f"candidate_experiment_ids: {list(family.candidate_experiment_ids)}")
    print(f"candidate_calibration_ids: {list(family.candidate_calibration_ids)}")
    print(f"candidate_optimization_identities: {list(family.candidate_optimization_identities)}")
    print(f"search_space_identity: {family.search_space_identity}")
    print(f"selection_metric: {family.selection_metric}")
    print(f"eligibility_rules_description: {family.eligibility_rules_description}")
    print(f"created_at: {family.created_at}")
    return 0


def cmd_compare_strategy_candidates(args: argparse.Namespace) -> int:
    """Milestone 6: prints each `--robustness-id` candidate's OWN
    standalone selection evaluation (Section 12) side by side --
    eligibility, rejection reasons (if any), and ranking metric values.
    Each robustness run's own `selection_report` was already computed as
    a single-candidate `SelectionReport` during its own pipeline; this
    command loads and juxtaposes what was already persisted for each,
    never re-running `selection.compute_selection_report` itself."""
    config = _load_robustness_config(Path(args.config))
    store = _robustness_manifest_store(config)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    for robustness_id in args.robustness_id:
        manifest = store.load(robustness_id)
        selection = _load_robustness_named_artifact(manifest, artifact_store, kind="selection_report", decoder=SelectionReport.from_json_dict)
        print(f"robustness_id: {robustness_id}")
        if selection is None:
            print(f"  no selection_report recorded yet (stage={manifest.stage.value!r})")
            continue
        for eligibility in selection.candidate_eligibility:
            print(f"  eligible: {eligibility.eligible}")
            for reason in eligibility.rejection_reasons:
                print(f"    rejected: {reason}")
        for ranking in selection.ranking:
            print(f"  ranking_metrics: {dict(sorted(ranking.metric_values.items()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.ml_cli",
        description="ML core infrastructure and artifact foundation CLI: experiment preparation (Milestone "
        "4A), the time-safe execution engine (Milestone 4B), and the baseline predictive model framework "
        "(Milestone 4C).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-model-definitions", help="List every registered model definition.")
    list_parser.set_defaults(handler=cmd_list_model_definitions)

    describe_parser = subparsers.add_parser("describe-model-definition", help="Print one model definition's full metadata.")
    describe_parser.add_argument("--name", required=True)
    describe_parser.add_argument("--version", required=True)
    describe_parser.set_defaults(handler=cmd_describe_model_definition)

    prepare_parser = subparsers.add_parser("prepare-experiment", help="Prepare (validate + create manifest for) an experiment.")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.set_defaults(handler=cmd_prepare_experiment)

    validate_parser = subparsers.add_parser("validate-experiment", help="Dry-run preflight validation; writes nothing.")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(handler=cmd_validate_experiment)

    inspect_parser = subparsers.add_parser("inspect-experiment", help="Print a human-readable (or JSON) experiment report.")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--experiment-id", required=True)
    inspect_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    inspect_parser.set_defaults(handler=cmd_inspect_experiment)

    manifest_parser = subparsers.add_parser("inspect-experiment-manifest", help="Print the raw experiment manifest.")
    manifest_parser.add_argument("--config", required=True)
    manifest_parser.add_argument("--experiment-id", required=True)
    manifest_parser.set_defaults(handler=cmd_inspect_experiment_manifest)

    verify_parser = subparsers.add_parser("verify-artifact", help="Verify a content-addressed artifact's integrity.")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--content-hash", required=True)
    verify_parser.set_defaults(handler=cmd_verify_artifact)

    events_parser = subparsers.add_parser("list-experiment-events", help="Print an experiment's append-only event log.")
    events_parser.add_argument("--config", required=True)
    events_parser.add_argument("--experiment-id", required=True)
    events_parser.set_defaults(handler=cmd_list_experiment_events)

    execute_parser = subparsers.add_parser(
        "execute", help="Run (or transparently resume) a READY experiment's walk-forward fold plan."
    )
    execute_parser.add_argument("--config", required=True)
    execute_parser.add_argument("--experiment-id", required=True)
    execute_parser.add_argument(
        "--force-rerun-fold", type=int, action="append", default=None,
        help="Re-run this fold index even if already verified complete (repeatable).",
    )
    execute_parser.set_defaults(handler=cmd_execute_experiment)

    resume_parser = subparsers.add_parser(
        "resume", help="Resume a prior, non-terminal execution -- fails if there is nothing to resume."
    )
    resume_parser.add_argument("--config", required=True)
    resume_parser.add_argument("--experiment-id", required=True)
    resume_parser.add_argument("--force-rerun-fold", type=int, action="append", default=None)
    resume_parser.set_defaults(handler=cmd_resume_execution)

    inspect_execution_parser = subparsers.add_parser(
        "inspect-execution", help="Print a human-readable (or JSON) execution report."
    )
    inspect_execution_parser.add_argument("--config", required=True)
    inspect_execution_parser.add_argument("--experiment-id", required=True)
    inspect_execution_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    inspect_execution_parser.set_defaults(handler=cmd_inspect_execution)

    inspect_fold_parser = subparsers.add_parser("inspect-fold", help="Print one fold's persisted FoldResult.")
    inspect_fold_parser.add_argument("--config", required=True)
    inspect_fold_parser.add_argument("--experiment-id", required=True)
    inspect_fold_parser.add_argument("--fold-index", type=int, required=True)
    inspect_fold_parser.set_defaults(handler=cmd_inspect_fold)

    list_folds_parser = subparsers.add_parser("list-folds", help="List every fold this execution has recorded, with status.")
    list_folds_parser.add_argument("--config", required=True)
    list_folds_parser.add_argument("--experiment-id", required=True)
    list_folds_parser.set_defaults(handler=cmd_list_folds)

    verify_execution_parser = subparsers.add_parser(
        "verify-execution", help="Re-verify every artifact (folds, timeline, aggregate) an execution has recorded."
    )
    verify_execution_parser.add_argument("--config", required=True)
    verify_execution_parser.add_argument("--experiment-id", required=True)
    verify_execution_parser.set_defaults(handler=cmd_verify_execution)

    list_models_parser = subparsers.add_parser(
        "list-models", help="Milestone 4C: list every registered model with a one-line capability summary."
    )
    list_models_parser.set_defaults(handler=cmd_list_models)

    inspect_model_parser = subparsers.add_parser(
        "inspect-model", help="Milestone 4C: print one model's full declared capabilities."
    )
    inspect_model_parser.add_argument("--name", required=True)
    inspect_model_parser.add_argument("--version", required=True)
    inspect_model_parser.set_defaults(handler=cmd_inspect_model)

    validate_model_parser = subparsers.add_parser(
        "validate-model", help="Milestone 4C: dry-run pre-fit model/data compatibility check; writes nothing."
    )
    validate_model_parser.add_argument("--config", required=True)
    validate_model_parser.set_defaults(handler=cmd_validate_model)

    train_parser = subparsers.add_parser(
        "train", help="Milestone 4C: run (or transparently resume) a READY experiment with real metrics/probabilities/training-metadata capture.",
    )
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--experiment-id", required=True)
    train_parser.add_argument(
        "--force-rerun-fold", type=int, action="append", default=None,
        help="Re-run this fold index even if already verified complete (repeatable).",
    )
    train_parser.set_defaults(handler=cmd_train)

    compare_parser = subparsers.add_parser(
        "compare", help="Milestone 4C: statistically compare a candidate experiment's per-fold metrics against one or more baseline experiments.",
    )
    compare_parser.add_argument("--config", required=True)
    compare_parser.add_argument("--candidate-experiment-id", required=True)
    compare_parser.add_argument("--baseline-experiment-id", action="append", required=True, help="Repeatable -- at least one required.")
    compare_parser.add_argument("--primary-metric", required=True)
    compare_parser.set_defaults(handler=cmd_compare)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Milestone 4D: run (or transparently resume) a leakage-safe nested feature-selection/hyperparameter-optimization search.",
    )
    optimize_parser.add_argument("--config", required=True)
    optimize_parser.set_defaults(handler=cmd_optimize)

    resume_optimization_parser = subparsers.add_parser(
        "resume-optimization", help="Milestone 4D: resume a prior, non-terminal optimization.",
    )
    resume_optimization_parser.add_argument("--config", required=True)
    resume_optimization_parser.add_argument("--optimization-id", required=True)
    resume_optimization_parser.set_defaults(handler=cmd_resume_optimization)

    inspect_optimization_parser = subparsers.add_parser(
        "inspect-optimization", help="Milestone 4D: print a human-readable (or JSON) optimization report.",
    )
    inspect_optimization_parser.add_argument("--config", required=True)
    inspect_optimization_parser.add_argument("--optimization-id", required=True)
    inspect_optimization_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    inspect_optimization_parser.set_defaults(handler=cmd_inspect_optimization)

    list_trials_parser = subparsers.add_parser(
        "list-trials", help="Milestone 4D: list every trial recorded for one outer fold, with status and score.",
    )
    list_trials_parser.add_argument("--config", required=True)
    list_trials_parser.add_argument("--optimization-id", required=True)
    list_trials_parser.add_argument("--outer-fold-index", type=int, required=True)
    list_trials_parser.set_defaults(handler=cmd_list_trials)

    inspect_trial_parser = subparsers.add_parser(
        "inspect-trial", help="Milestone 4D: print one trial's full persisted TrialResult.",
    )
    inspect_trial_parser.add_argument("--config", required=True)
    inspect_trial_parser.add_argument("--optimization-id", required=True)
    inspect_trial_parser.add_argument("--outer-fold-index", type=int, required=True)
    inspect_trial_parser.add_argument("--trial-number", type=int, required=True)
    inspect_trial_parser.set_defaults(handler=cmd_inspect_trial)

    verify_optimization_parser = subparsers.add_parser(
        "verify-optimization", help="Milestone 4D: re-verify every artifact/event an optimization has recorded, including outer-test isolation.",
    )
    verify_optimization_parser.add_argument("--config", required=True)
    verify_optimization_parser.add_argument("--optimization-id", required=True)
    verify_optimization_parser.set_defaults(handler=cmd_verify_optimization)

    compare_candidates_parser = subparsers.add_parser(
        "compare-optimization-candidates", help="Milestone 4D: print one outer fold's complete deterministic trial ranking table.",
    )
    compare_candidates_parser.add_argument("--config", required=True)
    compare_candidates_parser.add_argument("--optimization-id", required=True)
    compare_candidates_parser.add_argument("--outer-fold-index", type=int, required=True)
    compare_candidates_parser.set_defaults(handler=cmd_compare_optimization_candidates)

    feature_stability_parser = subparsers.add_parser(
        "feature-stability", help="Milestone 4D: print the feature-selection stability report for a completed optimization.",
    )
    feature_stability_parser.add_argument("--config", required=True)
    feature_stability_parser.add_argument("--optimization-id", required=True)
    feature_stability_parser.set_defaults(handler=cmd_feature_stability)

    hyperparameter_stability_parser = subparsers.add_parser(
        "hyperparameter-stability", help="Milestone 4D: print the hyperparameter stability report for a completed optimization.",
    )
    hyperparameter_stability_parser.add_argument("--config", required=True)
    hyperparameter_stability_parser.add_argument("--optimization-id", required=True)
    hyperparameter_stability_parser.set_defaults(handler=cmd_hyperparameter_stability)

    create_calibration_spec_parser = subparsers.add_parser(
        "create-calibration-spec", help="Milestone 4E: dry-run -- build/validate a CalibrationSpec from --config and print its calibration_id. Writes nothing.",
    )
    create_calibration_spec_parser.add_argument("--config", required=True)
    create_calibration_spec_parser.set_defaults(handler=cmd_create_calibration_spec)

    run_calibration_parser = subparsers.add_parser(
        "run-calibration", help="Milestone 4E: run (or transparently resume) a full leakage-safe calibration.",
    )
    run_calibration_parser.add_argument("--config", required=True)
    run_calibration_parser.set_defaults(handler=cmd_run_calibration)

    resume_calibration_parser = subparsers.add_parser(
        "resume-calibration", help="Milestone 4E: resume a prior, non-terminal calibration run.",
    )
    resume_calibration_parser.add_argument("--config", required=True)
    resume_calibration_parser.add_argument("--calibration-id", required=True)
    resume_calibration_parser.set_defaults(handler=cmd_resume_calibration)

    inspect_calibration_parser = subparsers.add_parser(
        "inspect-calibration", help="Milestone 4E: print a human-readable (or JSON) calibration report.",
    )
    inspect_calibration_parser.add_argument("--config", required=True)
    inspect_calibration_parser.add_argument("--calibration-id", required=True)
    inspect_calibration_parser.add_argument("--format", choices=["text", "json"], default="text")
    inspect_calibration_parser.set_defaults(handler=cmd_inspect_calibration)

    report_calibration_parser = subparsers.add_parser(
        "report-calibration", help="Milestone 4E: alias for inspect-calibration.",
    )
    report_calibration_parser.add_argument("--config", required=True)
    report_calibration_parser.add_argument("--calibration-id", required=True)
    report_calibration_parser.add_argument("--format", choices=["text", "json"], default="text")
    report_calibration_parser.set_defaults(handler=cmd_report_calibration)

    inspect_calibration_fold_parser = subparsers.add_parser(
        "inspect-calibration-fold", help="Milestone 4E: print one outer fold's persisted OuterFoldCalibrationResult.",
    )
    inspect_calibration_fold_parser.add_argument("--config", required=True)
    inspect_calibration_fold_parser.add_argument("--calibration-id", required=True)
    inspect_calibration_fold_parser.add_argument("--outer-fold-index", type=int, required=True)
    inspect_calibration_fold_parser.set_defaults(handler=cmd_inspect_calibration_fold)

    verify_calibration_parser = subparsers.add_parser(
        "verify-calibration", help="Milestone 4E: re-verify every artifact/event a calibration has recorded, including the calibrated-probability recomputation proof.",
    )
    verify_calibration_parser.add_argument("--config", required=True)
    verify_calibration_parser.add_argument("--calibration-id", required=True)
    verify_calibration_parser.set_defaults(handler=cmd_verify_calibration)

    compare_calibration_parser = subparsers.add_parser(
        "compare-calibration", help="Milestone 4E: print one metric's per-outer-fold values, side by side, for two completed calibrations.",
    )
    compare_calibration_parser.add_argument("--config", required=True)
    compare_calibration_parser.add_argument("--calibration-id", required=True)
    compare_calibration_parser.add_argument("--baseline-calibration-id", required=True)
    compare_calibration_parser.add_argument("--metric", required=True)
    compare_calibration_parser.set_defaults(handler=cmd_compare_calibration)

    create_backtest_spec_parser = subparsers.add_parser(
        "create-backtest-spec", help="Milestone 5: dry-run -- build/validate a BacktestSpec from --config and print its backtest_id. Writes nothing.",
    )
    create_backtest_spec_parser.add_argument("--config", required=True)
    create_backtest_spec_parser.set_defaults(handler=cmd_create_backtest_spec)

    run_backtest_parser = subparsers.add_parser(
        "run-backtest", help="Milestone 5: run (or transparently resume) a full leakage-safe backtest.",
    )
    run_backtest_parser.add_argument("--config", required=True)
    run_backtest_parser.set_defaults(handler=cmd_run_backtest)

    resume_backtest_parser = subparsers.add_parser(
        "resume-backtest", help="Milestone 5: resume a prior, non-terminal backtest run.",
    )
    resume_backtest_parser.add_argument("--config", required=True)
    resume_backtest_parser.add_argument("--backtest-id", required=True)
    resume_backtest_parser.set_defaults(handler=cmd_resume_backtest)

    inspect_backtest_parser = subparsers.add_parser(
        "inspect-backtest", help="Milestone 5: print a human-readable (or JSON) backtest report.",
    )
    inspect_backtest_parser.add_argument("--config", required=True)
    inspect_backtest_parser.add_argument("--backtest-id", required=True)
    inspect_backtest_parser.add_argument("--format", choices=["text", "json"], default="text")
    inspect_backtest_parser.set_defaults(handler=cmd_inspect_backtest)

    report_backtest_parser = subparsers.add_parser(
        "report-backtest", help="Milestone 5: alias for inspect-backtest.",
    )
    report_backtest_parser.add_argument("--config", required=True)
    report_backtest_parser.add_argument("--backtest-id", required=True)
    report_backtest_parser.add_argument("--format", choices=["text", "json"], default="text")
    report_backtest_parser.set_defaults(handler=cmd_report_backtest)

    inspect_backtest_fold_parser = subparsers.add_parser(
        "inspect-backtest-fold", help="Milestone 5: print one outer fold's persisted OuterFoldBacktestResult.",
    )
    inspect_backtest_fold_parser.add_argument("--config", required=True)
    inspect_backtest_fold_parser.add_argument("--backtest-id", required=True)
    inspect_backtest_fold_parser.add_argument("--outer-fold-index", type=int, required=True)
    inspect_backtest_fold_parser.set_defaults(handler=cmd_inspect_backtest_fold)

    verify_backtest_parser = subparsers.add_parser(
        "verify-backtest", help="Milestone 5: re-verify every artifact/event a backtest has recorded, including the financial-metrics recomputation proof.",
    )
    verify_backtest_parser.add_argument("--config", required=True)
    verify_backtest_parser.add_argument("--backtest-id", required=True)
    verify_backtest_parser.set_defaults(handler=cmd_verify_backtest)

    inspect_backtest_lock_parser = subparsers.add_parser(
        "inspect-backtest-lock",
        help="Milestone 5.2: read-only diagnostics for a backtest's run lock (age, owner PID/host, manifest state) -- never touches the lock.",
    )
    inspect_backtest_lock_parser.add_argument("--config", required=True)
    inspect_backtest_lock_parser.add_argument("--backtest-id", required=True)
    inspect_backtest_lock_parser.set_defaults(handler=cmd_inspect_backtest_lock)

    recover_backtest_lock_parser = subparsers.add_parser(
        "recover-backtest-lock",
        help="Milestone 5.2: explicit, diagnosed recovery for an abandoned .backtest_run.lock -- replaces undocumented manual deletion. "
        "Refuses to touch a lock that has not gone stale by age unless --force is passed.",
    )
    recover_backtest_lock_parser.add_argument("--config", required=True)
    recover_backtest_lock_parser.add_argument("--backtest-id", required=True)
    recover_backtest_lock_parser.add_argument("--force", action="store_true", help="Reclaim the lock even if it has not gone stale by age. Only use after confirming the owning process is dead.")
    recover_backtest_lock_parser.set_defaults(handler=cmd_recover_backtest_lock)

    compare_backtests_parser = subparsers.add_parser(
        "compare-backtests", help="Milestone 5: print one metric's per-outer-fold values, side by side, for two completed backtests.",
    )
    compare_backtests_parser.add_argument("--config", required=True)
    compare_backtests_parser.add_argument("--backtest-id", required=True)
    compare_backtests_parser.add_argument("--baseline-backtest-id", required=True)
    compare_backtests_parser.add_argument("--metric", required=True)
    compare_backtests_parser.set_defaults(handler=cmd_compare_backtests)

    create_robustness_spec_parser = subparsers.add_parser(
        "create-robustness-spec", help="Milestone 6: dry-run -- build/validate a RobustnessSpec from --config and print its robustness_id. Writes nothing.",
    )
    create_robustness_spec_parser.add_argument("--config", required=True)
    create_robustness_spec_parser.set_defaults(handler=cmd_create_robustness_spec)

    run_robustness_parser = subparsers.add_parser(
        "run-robustness", help="Milestone 6: run (or transparently resume) a full statistical-robustness analysis of an already-COMPLETED backtest.",
    )
    run_robustness_parser.add_argument("--config", required=True)
    run_robustness_parser.set_defaults(handler=cmd_run_robustness)

    resume_robustness_parser = subparsers.add_parser(
        "resume-robustness", help="Milestone 6: resume a prior, non-terminal robustness run.",
    )
    resume_robustness_parser.add_argument("--config", required=True)
    resume_robustness_parser.add_argument("--robustness-id", required=True)
    resume_robustness_parser.set_defaults(handler=cmd_resume_robustness)

    inspect_robustness_parser = subparsers.add_parser(
        "inspect-robustness", help="Milestone 6: print a human-readable (or JSON) robustness run summary, including its promotion decision.",
    )
    inspect_robustness_parser.add_argument("--config", required=True)
    inspect_robustness_parser.add_argument("--robustness-id", required=True)
    inspect_robustness_parser.add_argument("--format", choices=["text", "json"], default="text")
    inspect_robustness_parser.set_defaults(handler=cmd_inspect_robustness)

    report_robustness_parser = subparsers.add_parser(
        "report-robustness", help="Milestone 6: alias for inspect-robustness.",
    )
    report_robustness_parser.add_argument("--config", required=True)
    report_robustness_parser.add_argument("--robustness-id", required=True)
    report_robustness_parser.add_argument("--format", choices=["text", "json"], default="text")
    report_robustness_parser.set_defaults(handler=cmd_report_robustness)

    verify_robustness_parser = subparsers.add_parser(
        "verify-robustness", help="Milestone 6: independently reconstruct and cross-check every deterministic analysis a robustness run persisted.",
    )
    verify_robustness_parser.add_argument("--config", required=True)
    verify_robustness_parser.add_argument("--robustness-id", required=True)
    verify_robustness_parser.set_defaults(handler=cmd_verify_robustness)

    compare_robustness_parser = subparsers.add_parser(
        "compare-robustness", help="Milestone 6: print two robustness runs' stage/promotion-decision outcome side by side.",
    )
    compare_robustness_parser.add_argument("--config", required=True)
    compare_robustness_parser.add_argument("--robustness-id", required=True)
    compare_robustness_parser.add_argument("--baseline-robustness-id", required=True)
    compare_robustness_parser.set_defaults(handler=cmd_compare_robustness)

    inspect_promotion_decision_parser = subparsers.add_parser(
        "inspect-promotion-decision", help="Milestone 6: print one robustness run's full PromotionDecision -- every gate's name/measured value/bound(s)/outcome/reason.",
    )
    inspect_promotion_decision_parser.add_argument("--config", required=True)
    inspect_promotion_decision_parser.add_argument("--robustness-id", required=True)
    inspect_promotion_decision_parser.set_defaults(handler=cmd_inspect_promotion_decision)

    inspect_strategy_family_parser = subparsers.add_parser(
        "inspect-strategy-family", help="Milestone 6: print one durable StrategyFamily record, looked up by its content-addressed --content-hash.",
    )
    inspect_strategy_family_parser.add_argument("--config", required=True)
    inspect_strategy_family_parser.add_argument("--content-hash", required=True)
    inspect_strategy_family_parser.set_defaults(handler=cmd_inspect_strategy_family)

    compare_strategy_candidates_parser = subparsers.add_parser(
        "compare-strategy-candidates", help="Milestone 6: print each candidate's own standalone selection evaluation (eligibility, ranking metrics) side by side.",
    )
    compare_strategy_candidates_parser.add_argument("--config", required=True)
    compare_strategy_candidates_parser.add_argument("--robustness-id", required=True, action="append", help="May be passed more than once, one per candidate to compare.")
    compare_strategy_candidates_parser.set_defaults(handler=cmd_compare_strategy_candidates)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (QuantPlatformError, ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
