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
    python -m quant_platform.ml_cli create-paper-trading-spec --config pt_config.json
    python -m quant_platform.ml_cli run-paper-session --config pt_config.json --replay-source events.jsonl --feature-name candle_body_ratio [--feature-name ...]
    python -m quant_platform.ml_cli resume-paper-session --config pt_config.json --paper-session-id ID --replay-source events.jsonl --feature-name candle_body_ratio
    python -m quant_platform.ml_cli pause-paper-session --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli inspect-paper-session --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli report-paper-session --config pt_config.json --paper-session-id ID [--verify]
    python -m quant_platform.ml_cli verify-paper-session --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli compare-paper-to-backtest --config pt_config.json --paper-session-id ID --backtest-id ID
    python -m quant_platform.ml_cli inspect-paper-orders --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli inspect-paper-fills --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli inspect-paper-risk-events --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli inspect-paper-reconciliation --config pt_config.json --paper-session-id ID
    python -m quant_platform.ml_cli run-shadow-session --config pt_config.json --replay-source events.jsonl --feature-name candle_body_ratio
    python -m quant_platform.ml_cli report-shadow-session --config pt_config.json --paper-session-id ID

MILESTONE 7 (PAPER TRADING / SHADOW EXECUTION) -- NO LIVE ORDER TRANSMISSION
--------------------------------------------------------------------------
Every `run-paper-session`/`resume-paper-session`/`run-shadow-session`
invocation prints an explicit "no live orders are sent" notice. There is
no command named `run-live`/`submit-live-order`/`connect-broker`/
`execute-mt5`/`deploy-live` anywhere on this parser, and none can be --
`paper_trading` contains no network/broker client of any kind (Section 35's
safety scan proves this structurally, not by convention). `--paper-
session-id` for shadow sessions is `run-shadow-session`'s own printed
`paper_session_id` -- the SAME identity space as a real paper session
(both are `PaperTradingSpec`-addressed), never a separate namespace.

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
from decimal import Decimal
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
from quant_platform.config.execution_gateway_schemas import ExecutionGatewayConfigSchema
from quant_platform.config.ml_schemas import MLExperimentConfig
from quant_platform.config.optimization_schemas import OptimizationConfig
from quant_platform.config.paper_trading_schemas import PaperTradingConfig
from quant_platform.config.robustness_schemas import RobustnessConfig
from quant_platform.core.exceptions import (
    ExecutionGatewayIdentityError,
    PaperTradingEligibilityError,
    PaperTradingIdentityError,
    PaperTradingStateError,
    QuantPlatformError,
)
from quant_platform.core.json import write_json_atomic
from quant_platform.execution.executor import DeterministicFoldExecutor, MetricsFoldExecutor
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.execution.reporting import build_execution_report_json, render_execution_report_markdown
from quant_platform.execution.results import AggregatedExecutionResult, FoldResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.splitters import reconstruct_dataset_timeline
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.execution.timeline import Timeline
from quant_platform.execution.verification import verify_execution
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind, ExecutionSessionStage
from quant_platform.execution_gateway.paper_bridge import PaperBridgeEnvironment
from quant_platform.execution_gateway.persistence import ExecutionLedgerEntry, ExecutionSessionEventStore
from quant_platform.execution_gateway.portfolio_risk_gate import PortfolioRiskGatewayContext
from quant_platform.execution_gateway.reconciliation import reconcile_execution_session
from quant_platform.execution_gateway.reports import generate_execution_session_report
from quant_platform.execution_gateway.runner import (
    RunnerEnvironment as ExecutionGatewayRunnerEnvironment,
)
from quant_platform.execution_gateway.runner import (
    current_kill_switch_state,
    pause_execution_session,
    run_execution_session,
)
from quant_platform.execution_gateway.specs import compute_execution_gateway_spec_id
from quant_platform.execution_gateway.state_machine import ExecutionOrderStateEvent
from quant_platform.execution_gateway.states import (
    ExecutionFill,
    compute_execution_order_id,
    reconstruct_execution_order,
)
from quant_platform.execution_gateway.verification import verify_execution_session
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
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    parse_json_strict,
    read_json_file,
    utc_now,
)
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
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.eligibility import EligibilityVerificationEnvironment
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.model_strategy import ModelStrategyRuntime
from quant_platform.paper_trading.models import LedgerEntryKind, OrderState, PaperSessionStage, SessionMode
from quant_platform.paper_trading.orders import OrderRequest, OrderStateEvent, resolve_order_state
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.reconciliation import reconcile_session
from quant_platform.paper_trading.replay import load_replay_events
from quant_platform.paper_trading.reports import (
    BacktestComparisonMetrics,
    build_paper_session_report,
    compare_paper_to_backtest,
)
from quant_platform.paper_trading.risk import KillSwitchTransitionEvent
from quant_platform.paper_trading.runner import (
    RunnerEnvironment,
    pause_paper_session,
    run_paper_trading_session,
    run_shadow_session,
)
from quant_platform.paper_trading.specs import PaperTradingSpec, compute_paper_session_spec_id
from quant_platform.paper_trading.verification import verify_paper_session
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.models import PORTFOLIO_RISK_SPEC_SCHEMA_VERSION
from quant_platform.portfolio_risk.snapshots import create_portfolio_snapshot, create_price_snapshot
from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec
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


# --------------------------------------------------------------------------
# Milestone 7: deterministic paper trading / shadow execution commands
#
# Every command below prints "no live orders are sent" context only where
# genuinely informative (run/resume/shadow); no command anywhere in this
# section can transmit a real order -- `run_paper_trading_session`/`run_
# shadow_session` structurally cannot (Section 35's safety scan proves no
# network/broker client exists in `paper_trading` at all).
# --------------------------------------------------------------------------
_NO_LIVE_ORDERS_NOTICE = "NOTICE: this is a simulated paper/shadow session. No order is ever sent to any broker."

_DEFAULT_INSPECTION_ROW_LIMIT = 200
"""Release-audit finding, fixed (Section 11): `inspect-paper-orders`/
`inspect-paper-fills`/`inspect-paper-risk-events` used to print EVERY
matching record unconditionally, with no cap at all -- a long-running
session's own order/fill count is unbounded, so this was a genuine
operator-safety gap (unbounded terminal output), not merely a cosmetic
one. `--limit` (overridable per invocation) caps how many rows print;
the underlying ledger itself is never touched or truncated."""


def _load_paper_trading_config(path: Path) -> PaperTradingConfig:
    return PaperTradingConfig.model_validate_json(path.read_text())


def _paper_trading_manifest_store(config: PaperTradingConfig) -> PaperSessionManifestStore:
    return PaperSessionManifestStore(config.ml_artifacts_root)


def _paper_trading_event_store(config: PaperTradingConfig) -> PaperSessionEventStore:
    return PaperSessionEventStore(config.ml_artifacts_root)


def _paper_trading_eligibility_environment(config: PaperTradingConfig) -> EligibilityVerificationEnvironment:
    return EligibilityVerificationEnvironment(
        robustness_manifest_store=RobustnessManifestStore(config.ml_artifacts_root), artifact_store=MLArtifactStore(config.ml_artifacts_root),
        backtest_manifest_store=BacktestManifestStore(config.ml_artifacts_root), backtest_event_store=BacktestEventStore(config.ml_artifacts_root),
        calibration_manifest_store=CalibrationManifestStore(config.ml_artifacts_root), experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        execution_manifest_store=ExecutionManifestStore(config.ml_artifacts_root), research_manifest_store=ResearchManifestStore(config.research_storage_root),
        research_dataset_store=ResearchDatasetStore(config.research_storage_root),
        dataset_loader=DatasetLoader(CanonicalStore(config.historical_storage_root), ManifestStore(config.historical_storage_root)),
    )


def _paper_trading_runner_environment(config: PaperTradingConfig) -> RunnerEnvironment:
    return RunnerEnvironment(
        manifest_store=_paper_trading_manifest_store(config), event_store=_paper_trading_event_store(config),
        eligibility_environment=_paper_trading_eligibility_environment(config),
    )


def _paper_session_model_pin_path(config: PaperTradingConfig, paper_session_id: str) -> Path:
    return config.ml_artifacts_root / "paper_sessions" / paper_session_id / "resolved_model_selection.json"


def _resolve_fitted_strategy_runtime(
    config: PaperTradingConfig, spec: PaperTradingSpec, *, feature_names: tuple[str, ...], long_threshold: float, short_threshold: float, target_quantity: float,
) -> ModelStrategyRuntime:
    """Resolves a real, already-fitted model for `spec.model_artifact_
    identity` (the source experiment's own id -- see `eligibility.py`'s
    `resolved_model_artifact_identity`) and wraps it in `model_strategy.
    ModelStrategyRuntime`.

    WHICH FOLD'S MODEL: a walk-forward experiment fits one model PER
    OUTER FOLD, never a single "the" model -- this resolves the model
    from the HIGHEST-indexed (temporally last, most-training-data) outer
    fold, a documented, deliberate simplification (Section 39's delivery
    report classifies this choice explicitly) rather than an unstated
    assumption. This is NEVER a test-performance-driven "best fold" pick
    -- it is a pure index rule, independent of any fold's own metrics.

    PINNED, release-audit finding, fixed: `fold_index`/the resolved
    model's own content hash used to be re-derived from LIVE, mutable
    `ExecutionManifestStore`/`MLArtifactStore` state on EVERY call --
    `run-paper-session`, EVERY `resume-paper-session`, and `run-shadow-
    session` each independently re-ran this exact resolution. Nothing
    about which fold/model was actually used was ever part of `spec`
    (hashed into `paper_session_spec_id`) OR recorded anywhere durable,
    so if the underlying experiment's fold set ever changed between two
    calls for the SAME session (e.g. an operator re-runs a fold with
    `--force-rerun-fold`, or a later fold is added), a `resume-paper-
    session` call could SILENTLY swap in a different fitted model mid-
    session while `paper_session_spec_id` stayed byte-identical -- the
    literal defect class Section 4 exists to catch ("the selected fold
    cannot change without changing identity"). Fixed by pinning the
    resolved `(experiment_id, fold_index, model_content_hash)` triple to
    a durable, session-scoped file the FIRST time it is resolved, and
    fail-closed (`PaperTradingEligibilityError`, before any event is
    processed) if a later resolution for the SAME `paper_session_id`
    would pick something different."""
    experiment_id = spec.model_artifact_identity
    experiment_manifest = ExperimentManifestStore(config.ml_artifacts_root).load(experiment_id)
    execution_manifest = ExecutionManifestStore(config.ml_artifacts_root).load(experiment_id)
    if not execution_manifest.fold_result_references:
        raise ValueError(f"experiment {experiment_id!r} has no recorded fold results to load a fitted model from")
    fold_index = max(execution_manifest.fold_result_references)
    artifact_store = MLArtifactStore(config.ml_artifacts_root)
    fold_result_ref = execution_manifest.fold_result_references[fold_index]
    fold_result = FoldResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_result_ref.content_hash).decode("utf-8")))
    model_ref = next((r for r in fold_result.artifact_references if r.category is ArtifactCategory.MODEL), None)
    if model_ref is None:
        raise ValueError(f"fold {fold_index} of experiment {experiment_id!r} has no recorded MODEL artifact")

    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    resolved_pin: dict[str, object] = {"experiment_id": experiment_id, "fold_index": fold_index, "model_content_hash": model_ref.content_hash}
    pin_path = _paper_session_model_pin_path(config, paper_session_id)
    if pin_path.is_file():
        pinned = as_json_dict(read_json_file(pin_path), field_name="resolved_model_selection")
        if pinned != resolved_pin:
            raise PaperTradingEligibilityError(
                f"Resolved model for paper session {paper_session_id!r} has changed since this session began "
                f"(pinned={pinned!r}, now resolves to={resolved_pin!r}) -- refusing to swap the fitted model mid-session.",
                context={"paper_session_id": paper_session_id},
            )
    else:
        write_json_atomic(pin_path, resolved_pin)

    model_registry = build_model_registry()
    model_definition = model_registry.get(experiment_manifest.spec.model_name, experiment_manifest.spec.model_version)
    _serializer, deserializer = mz.default_serializer_registry()[model_definition.serializer_id]
    fitted_model = deserializer.deserialize(artifact_store.read_artifact(model_ref.content_hash))

    return ModelStrategyRuntime(
        strategy_identity=spec.strategy_candidate_identity, fitted_model=fitted_model, feature_names=feature_names,
        long_threshold=long_threshold, short_threshold=short_threshold, target_quantity=target_quantity,
    )


def _resolve_paper_trading_spec(config: PaperTradingConfig) -> PaperTradingSpec:
    return config.build()


def cmd_create_paper_trading_spec(args: argparse.Namespace) -> int:
    """Milestone 7: dry-run -- build/validate a `PaperTradingSpec` from
    `--config` and print its deterministic `paper_session_spec_id`.
    Writes nothing; does NOT verify eligibility (that happens at session
    creation, fail-closed)."""
    config = _load_paper_trading_config(Path(args.config))
    spec = _resolve_paper_trading_spec(config)
    identity = compute_paper_session_spec_id(spec)
    print(f"paper_session_spec_id: {identity.paper_session_spec_id}")
    print(f"session_mode: {spec.session_mode.value}")
    print(f"instrument: {spec.instrument.symbol}")
    print(f"starting_cash: {spec.starting_cash}")
    print(f"seed: {spec.seed}")
    print(_NO_LIVE_ORDERS_NOTICE)
    return 0


def cmd_run_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7: creates (or transparently resumes) a `REPLAY_PAPER`/
    `FORWARD_PAPER` session and runs it against `--replay-source`'s
    bounded, pre-validated `MarketEvent` sequence (Section 32). Fails
    closed before a single event is processed unless the spec's declared
    `ELIGIBLE_FOR_PAPER_TRADING` chain independently re-verifies."""
    config = _load_paper_trading_config(Path(args.config))
    spec = _resolve_paper_trading_spec(config)
    environment = _paper_trading_runner_environment(config)
    events = load_replay_events(Path(args.replay_source))
    strategy_runtime = _resolve_fitted_strategy_runtime(
        config, spec, feature_names=tuple(args.feature_name), long_threshold=args.long_threshold, short_threshold=args.short_threshold, target_quantity=args.target_quantity,
    )
    manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=events)
    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"stage: {manifest.stage.value}")
    print(_NO_LIVE_ORDERS_NOTICE)
    return 0 if manifest.stage is PaperSessionStage.COMPLETED else 2


def cmd_resume_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7: resumes a prior, non-terminal paper session --
    `--paper-session-id` must already exist (fails otherwise, never
    silently creates a new one). `--replay-source` must be the SAME
    deterministic source the original run used for resume to reproduce
    identical results.

    IDENTITY-BOUND, release-audit finding, fixed: `--paper-session-id`
    used to be checked ONLY for existence, then completely ignored --
    the actual session touched below was whatever `compute_paper_
    session_spec_id(spec)` resolves `--config` to, with no requirement
    that the two agree. A `--config` that (by drift, typo, or a stale/
    mismatched file) resolves to a DIFFERENT spec than the one that
    produced `--paper-session-id` would silently resume (or, if that
    other id happened not to exist yet, silently CREATE) a completely
    different session while the operator believed they were resuming
    the one they named -- the exact identity-binding gap Section 5/8
    exist to close ("resume cannot cross-load another session's
    artifacts"). Fixed: fail closed before anything else happens if the
    two identities disagree."""
    config = _load_paper_trading_config(Path(args.config))
    manifest_store = _paper_trading_manifest_store(config)
    manifest_store.load(args.paper_session_id)  # fails closed if the session does not already exist
    spec = _resolve_paper_trading_spec(config)
    resolved_paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    if resolved_paper_session_id != args.paper_session_id:
        raise PaperTradingIdentityError(
            f"--paper-session-id {args.paper_session_id!r} does not match the session {resolved_paper_session_id!r} that --config resolves to "
            "-- refusing to resume a different session than the one named.",
            context={"requested_paper_session_id": args.paper_session_id, "config_resolved_paper_session_id": resolved_paper_session_id},
        )
    environment = _paper_trading_runner_environment(config)
    events = load_replay_events(Path(args.replay_source))
    strategy_runtime = _resolve_fitted_strategy_runtime(
        config, spec, feature_names=tuple(args.feature_name), long_threshold=args.long_threshold, short_threshold=args.short_threshold, target_quantity=args.target_quantity,
    )
    manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=events)
    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"stage: {manifest.stage.value}")
    return 0 if manifest.stage is PaperSessionStage.COMPLETED else 2


def cmd_pause_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 23): durably pauses a `RUNNING` paper
    session -- a subsequent `resume-paper-session` continues from the
    ledger's own last completed event."""
    config = _load_paper_trading_config(Path(args.config))
    environment = _paper_trading_runner_environment(config)
    manifest = pause_paper_session(environment, args.paper_session_id)
    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"stage: {manifest.stage.value}")
    return 0


def cmd_inspect_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7: prints a human-readable (or JSON) summary of one
    paper/shadow session's manifest."""
    config = _load_paper_trading_config(Path(args.config))
    manifest = _paper_trading_manifest_store(config).load(args.paper_session_id)

    if args.format == "json":
        import json

        print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0

    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"session_mode: {manifest.session_mode.value}")
    print(f"stage: {manifest.stage.value}")
    print(f"created_at: {manifest.created_at}")
    print(f"updated_at: {manifest.updated_at}")
    if manifest.completed_at:
        print(f"completed_at: {manifest.completed_at}")
    if manifest.failure_category:
        print(f"failure_category: {manifest.failure_category}")
        print(f"failure_stage: {manifest.failure_stage}")
        print(f"failure_recoverable: {manifest.failure_recoverable}")
    return 0


def cmd_report_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 27): prints the 15-summary durable session
    report (decisions, orders, fills, execution quality, costs,
    positions, account/equity, drawdown, risk events, rejections, halts,
    reconciliation, shadow observations). Does NOT run `verify-paper-
    session` itself (a separate, heavier command) -- `report.verification`
    is left un-run (`was_run=False`) unless `--verify` is passed."""
    config = _load_paper_trading_config(Path(args.config))
    manifest_store = _paper_trading_manifest_store(config)
    event_store = _paper_trading_event_store(config)
    manifest = manifest_store.load(args.paper_session_id)
    spec = _resolve_paper_trading_spec(config)
    if manifest.session_mode is SessionMode.SHADOW_OBSERVATION:
        # Release-audit finding, fixed (Section 11): neither report command previously checked the
        # session's own declared mode -- calling the wrong report command against the wrong session
        # id silently produced a misleadingly-labeled (but not incorrect) report instead of a clean
        # error. `report-shadow-session` is the correct command for a SHADOW_OBSERVATION session.
        # Checked against `manifest.session_mode` (the session's OWN persisted record of what it
        # actually was created/run as), never `spec.session_mode` (merely what `--config` currently
        # declares, which could be stale/misconfigured/unrelated to the session actually being asked about).
        raise PaperTradingStateError(f"paper_session_id={args.paper_session_id!r} is session_mode=shadow_observation -- use 'report-shadow-session', not 'report-paper-session'")
    ledger = event_store.read_events(args.paper_session_id)
    reconciliation_report = reconcile_session(ledger, session_id=args.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
    verification_report = None
    if args.verify:
        eligibility_environment = _paper_trading_eligibility_environment(config)
        verification_report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=eligibility_environment)
    report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report, verification_report=verification_report)

    if args.format == "json":
        import json

        print(json.dumps(report.to_json_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0

    print(f"paper_session_id: {report.session.session_id} ({report.session.session_mode})")
    print(f"event_count={report.session.event_count} decision_count={report.decisions.decision_count} abstention_count={report.decisions.abstention_count}")
    print(f"order_count={report.orders.order_count} rejected_count={report.orders.rejected_count} fill_count={report.fills.fill_count}")
    print(f"starting_cash={report.account_equity.starting_cash} final_equity={report.account_equity.final_equity} net_pnl={report.account_equity.net_pnl}")
    print(f"total_costs={report.costs.total_costs} maximum_drawdown_fraction={report.drawdown.maximum_drawdown_fraction}")
    print(f"reconciled={report.reconciliation.is_reconciled} failed_checks={list(report.reconciliation.failed_check_identities)}")
    if report.verification.was_run:
        print(f"verification.is_ready={report.verification.is_ready}")
    print(f"disclaimer: {report.disclaimer}")
    return 0


def cmd_verify_paper_session(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 26): independently re-verifies spec identity,
    the full eligibility chain, ledger/manifest integrity, and
    reconciliation -- never trusting the persisted manifest/report at
    face value. Returns 2 (not 1) when the report contains any CRITICAL/
    ERROR issue. See `verification.INDEPENDENCE_CLASSIFICATION` for what
    is/is not independently re-derived."""
    config = _load_paper_trading_config(Path(args.config))
    manifest = _paper_trading_manifest_store(config).load(args.paper_session_id)
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    spec = _resolve_paper_trading_spec(config)
    eligibility_environment = _paper_trading_eligibility_environment(config)
    report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=eligibility_environment)
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def cmd_compare_paper_to_backtest(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 28): diagnostic-only comparison between a
    paper session and its source backtest's own aggregate metrics.
    `decision_count`/`abstention_count`/`rejected_order_count` are not
    concepts a vectorized backtest tracks the same way paper trading
    does -- they are reported as `0` on the backtest side (a documented
    simplification), so those three comparisons are EXPECTED to surface
    as `unexpected_decision_mismatch` and should be read accordingly,
    never as a defect."""
    config = _load_paper_trading_config(Path(args.config))
    manifest = _paper_trading_manifest_store(config).load(args.paper_session_id)
    spec = _resolve_paper_trading_spec(config)
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    reconciliation_report = reconcile_session(ledger, session_id=args.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
    paper_report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

    backtest_manifest = BacktestManifestStore(config.ml_artifacts_root).load(args.backtest_id)
    backtest_report = build_backtest_report_json(backtest_manifest)
    aggregate_metrics = as_json_dict(backtest_report["aggregate_metrics"], field_name="aggregate_metrics")
    fold_wise_mean = as_json_dict(aggregate_metrics["fold_wise_mean"], field_name="fold_wise_mean")

    def _metric(name: str) -> float:
        value = fold_wise_mean.get(name)
        return float(value) if isinstance(value, (int, float)) else 0.0

    backtest_metrics = BacktestComparisonMetrics(
        decision_count=0, order_count=0, gross_return=_metric("total_gross_return"), net_return=_metric("total_net_return"),
        total_costs=_metric("total_cost_notional"), turnover=_metric("turnover_notional_ratio"), max_drawdown_fraction=_metric("maximum_drawdown"),
        rejected_order_count=0, abstention_count=0,
    )
    comparison = compare_paper_to_backtest(backtest_metrics, paper_report, source_backtest_id=args.backtest_id)
    for metric in comparison.comparisons:
        print(f"{metric.metric_name}: backtest={metric.backtest_value} paper={metric.paper_value} matches={metric.matches} classification={metric.classification}")
    print(f"disclaimer: {comparison.disclaimer}")
    return 0


def _require_non_negative_limit(limit: int) -> int:
    if limit < 0:
        raise ValueError(f"--limit must be >= 0, got {limit}")
    return limit


def cmd_inspect_paper_orders(args: argparse.Namespace) -> int:
    """Milestone 7: lists every order this paper session's ledger has
    recorded, reconstructed from `ORDER_STATE_EVENT` entries alone."""
    config = _load_paper_trading_config(Path(args.config))
    _paper_trading_manifest_store(config).load(args.paper_session_id)  # fails closed if the session does not exist
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    orders: dict[str, tuple[dict[str, object], list[OrderStateEvent]]] = {}
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.ORDER_STATE_EVENT:
            continue
        order_json = entry.payload["order"]
        state_event = OrderStateEvent.from_json_dict(entry.payload["order_state_event"])  # type: ignore[arg-type]
        if state_event.order_id not in orders:
            orders[state_event.order_id] = (order_json, [])  # type: ignore[assignment]
        orders[state_event.order_id][1].append(state_event)
    if not orders:
        print(f"No orders recorded for paper_session_id={args.paper_session_id!r}")
        return 0
    limit = _require_non_negative_limit(args.limit)
    for order_id, (order_json, events) in list(orders.items())[:limit]:
        final_state: OrderState = resolve_order_state(order_id, events)
        print(f"order_id={order_id} side={order_json['side']} type={order_json['order_type']} quantity={order_json['quantity']} final_state={final_state.value}")
    if len(orders) > limit:
        print(f"... {len(orders) - limit} more order(s) not shown (total={len(orders)}, --limit={limit})")
    return 0


def cmd_inspect_paper_fills(args: argparse.Namespace) -> int:
    """Milestone 7: lists every `FILL` this paper session's ledger has
    recorded."""
    config = _load_paper_trading_config(Path(args.config))
    _paper_trading_manifest_store(config).load(args.paper_session_id)  # fails closed if the session does not exist
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    fills = [e.payload for e in ledger if e.kind is LedgerEntryKind.FILL]
    if not fills:
        print(f"No fills recorded for paper_session_id={args.paper_session_id!r}")
        return 0
    limit = _require_non_negative_limit(args.limit)
    for fill in fills[:limit]:
        print(f"fill_id={fill['fill_id']} order_id={fill['order_id']} side={fill['side']} quantity={fill['quantity']} price={fill['price']} is_final={fill['is_final']}")
    if len(fills) > limit:
        print(f"... {len(fills) - limit} more fill(s) not shown (total={len(fills)}, --limit={limit})")
    return 0


def cmd_inspect_paper_risk_events(args: argparse.Namespace) -> int:
    """Milestone 7: lists every pre-trade/continuous `RISK_DECISION`
    batch and every kill-switch `HALT_TRIGGERED` transition this paper
    session's ledger has recorded."""
    config = _load_paper_trading_config(Path(args.config))
    _paper_trading_manifest_store(config).load(args.paper_session_id)  # fails closed if the session does not exist
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    limit = _require_non_negative_limit(args.limit)
    risk_decisions = [e.payload for e in ledger if e.kind is LedgerEntryKind.RISK_DECISION]
    halts = [KillSwitchTransitionEvent.from_json_dict(e.payload) for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
    failed_results: list[dict[str, object]] = []
    for batch in risk_decisions:
        for raw_result in as_json_list(batch.get("results") or [], field_name="results"):
            result = as_json_dict(raw_result, field_name="results[]")
            if not result.get("passed"):
                failed_results.append(result)
    print(f"risk_decision_batches: {len(risk_decisions)} (failed_checks={len(failed_results)})")
    for result in failed_results[:limit]:
        print(f"  FAILED check={result['check_identity']} measured={result['measured_value']} limit={result['limit']} action={result['action']} reason={result['reason_code']}")
    if len(failed_results) > limit:
        print(f"  ... {len(failed_results) - limit} more failed check(s) not shown (--limit={limit})")
    print(f"kill_switch_transitions: {len(halts)}")
    for halt in halts[:limit]:
        print(f"  {halt.from_state.value} -> {halt.to_state.value} trigger={halt.trigger.value} detail={halt.detail!r}")
    if len(halts) > limit:
        print(f"  ... {len(halts) - limit} more transition(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_paper_reconciliation(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 25): runs the 11 independent reconciliation
    checks against this paper session's ledger and prints every check's
    pass/fail status."""
    config = _load_paper_trading_config(Path(args.config))
    _paper_trading_manifest_store(config).load(args.paper_session_id)  # fails closed if the session does not exist
    spec = _resolve_paper_trading_spec(config)
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    report = reconcile_session(ledger, session_id=args.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.check_identity}: expected={check.expected_value!r} observed={check.observed_value!r}")
    print(f"is_reconciled: {report.is_reconciled}")
    return 0 if report.is_reconciled else 2


def cmd_run_shadow_session(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 19): creates (or transparently resumes --
    resume is NOT supported for shadow sessions, see `runner.run_shadow_
    session`'s own module docstring) a `SHADOW_OBSERVATION` session.
    Hypothetical orders/fills are recorded; the real simulated account
    (if this config even has one) is never touched."""
    config = _load_paper_trading_config(Path(args.config))
    spec = _resolve_paper_trading_spec(config)
    environment = _paper_trading_runner_environment(config)
    events = load_replay_events(Path(args.replay_source))
    strategy_runtime = _resolve_fitted_strategy_runtime(
        config, spec, feature_names=tuple(args.feature_name), long_threshold=args.long_threshold, short_threshold=args.short_threshold, target_quantity=args.target_quantity,
    )
    manifest = run_shadow_session(spec, environment=environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=events)
    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"stage: {manifest.stage.value}")
    print(_NO_LIVE_ORDERS_NOTICE)
    return 0 if manifest.stage is PaperSessionStage.COMPLETED else 2


def cmd_report_shadow_session(args: argparse.Namespace) -> int:
    """Milestone 7 (Section 19): prints the SHADOW-labeled observation
    summary only (hypothetical order/fill counts, counterfactual realized
    P&L) -- never folded into any real-account figure, exactly matching
    `reports.ShadowObservationSummary`'s own field-naming discipline."""
    config = _load_paper_trading_config(Path(args.config))
    manifest = _paper_trading_manifest_store(config).load(args.paper_session_id)
    spec = _resolve_paper_trading_spec(config)
    if manifest.session_mode is not SessionMode.SHADOW_OBSERVATION:
        # Same release-audit fix as `cmd_report_paper_session`'s own -- see that function's comment.
        raise PaperTradingStateError(f"paper_session_id={args.paper_session_id!r} is session_mode={manifest.session_mode.value!r}, not shadow_observation -- use 'report-paper-session', not 'report-shadow-session'")
    ledger = _paper_trading_event_store(config).read_events(args.paper_session_id)
    reconciliation_report = reconcile_session(ledger, session_id=args.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
    report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)
    print(f"paper_session_id: {report.session.session_id} (SHADOW_OBSERVATION)")
    print(f"observation_count={report.shadow.observation_count} observations_with_hypothetical_fill_count={report.shadow.observations_with_hypothetical_fill_count}")
    print(f"total_counterfactual_realized_pnl={report.shadow.total_counterfactual_realized_pnl}")
    print(f"disclaimer: {report.disclaimer}")
    return 0


# --------------------------------------------------------------------------
# Milestone 8: broker-neutral deterministic execution gateway commands
#
# TEST-ONLY. Every command below dispatches exclusively to
# `execution_gateway.dummy_broker.DeterministicDummyBrokerAdapter` -- an
# in-process, seeded, deterministic simulator. No command in this section
# can transmit a real order: `execution_mode`/`adapter_kind` are each
# single-member enums (`TEST_ONLY`/`DETERMINISTIC_DUMMY`), and Section 35's
# safety scan proves no network/broker-client/MT5 import exists anywhere
# in `execution_gateway` at all.
# --------------------------------------------------------------------------
_NO_LIVE_EXECUTION_NOTICE = "NOTICE: this is a TEST-ONLY deterministic dummy-broker execution session. No order is ever sent to any real broker."


def _load_execution_gateway_config(path: Path) -> ExecutionGatewayConfigSchema:
    return ExecutionGatewayConfigSchema.model_validate_json(path.read_text())


def _execution_gateway_manifest_store(config: ExecutionGatewayConfigSchema) -> ExecutionSessionManifestStore:
    return ExecutionSessionManifestStore(config.ml_artifacts_root)


def _execution_gateway_event_store(config: ExecutionGatewayConfigSchema) -> ExecutionSessionEventStore:
    return ExecutionSessionEventStore(config.ml_artifacts_root)


def _execution_gateway_eligibility_environment(config: ExecutionGatewayConfigSchema) -> EligibilityVerificationEnvironment:
    return EligibilityVerificationEnvironment(
        robustness_manifest_store=RobustnessManifestStore(config.ml_artifacts_root), artifact_store=MLArtifactStore(config.ml_artifacts_root),
        backtest_manifest_store=BacktestManifestStore(config.ml_artifacts_root), backtest_event_store=BacktestEventStore(config.ml_artifacts_root),
        calibration_manifest_store=CalibrationManifestStore(config.ml_artifacts_root), experiment_manifest_store=ExperimentManifestStore(config.ml_artifacts_root),
        execution_manifest_store=ExecutionManifestStore(config.ml_artifacts_root), research_manifest_store=ResearchManifestStore(config.research_storage_root),
        research_dataset_store=ResearchDatasetStore(config.research_storage_root),
        dataset_loader=DatasetLoader(CanonicalStore(config.historical_storage_root), ManifestStore(config.historical_storage_root)),
    )


def _execution_gateway_paper_bridge_environment(config: ExecutionGatewayConfigSchema) -> PaperBridgeEnvironment:
    return PaperBridgeEnvironment(
        manifest_store=PaperSessionManifestStore(config.ml_artifacts_root), event_store=PaperSessionEventStore(config.ml_artifacts_root),
        artifact_store=MLArtifactStore(config.ml_artifacts_root), eligibility_environment=_execution_gateway_eligibility_environment(config),
    )


def _execution_gateway_default_portfolio_risk_context(config: ExecutionGatewayConfigSchema, *, storage_root: str | None = None) -> PortfolioRiskGatewayContext:
    """Milestone 9 Phase 4: the CLI wires a MINIMAL, always-present
    portfolio-risk context automatically -- deliberately NO new CLI flags
    or `ExecutionGatewayConfigSchema` fields (out of this phase's own
    scope: "no CLI expansion"). Every limit on the resulting
    `PortfolioRiskPolicy` is `None` ("not configured", Phase 1's own
    pre-existing convention -- NOT a bypass flag; no field anywhere in
    this package exists whose purpose is "skip risk evaluation"). The
    gate itself still runs for REAL, through the genuine `evaluate_risk`
    pipeline, on every dispatch through this CLI -- it will simply always
    approve until an operator-supplied policy exists. Documented as a
    known, honest limitation in `docs/milestone9_phase4_delivery_report.md`,
    not silently glossed over."""
    event_time = utc_now().to_pydatetime()
    paper_orders = _load_paper_orders_for_session(config, config.paper_session_id)
    instrument_id = paper_orders[0].instrument if paper_orders else "unconfigured_instrument"
    equity = Decimal("1000000")
    portfolio_snapshot = create_portfolio_snapshot(
        portfolio_id=config.paper_session_id, event_time=event_time, cash=equity, equity=equity, realized_pnl=Decimal(0), unrealized_pnl=Decimal(0),
        peak_equity=equity, daily_start_equity=equity, positions=(), source_execution_session_id=None,
    )
    price_snapshot = create_price_snapshot(
        instrument_id=instrument_id, bid=Decimal(1), ask=Decimal(1), reference_price=Decimal(1), event_time=event_time, source_event_id=None,
    )
    policy = PortfolioRiskPolicy(
        max_order_notional=None, max_position_notional=None, max_instrument_gross_exposure=None, max_strategy_gross_exposure=None,
        max_portfolio_gross_exposure=None, max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None,
        max_daily_realized_loss=None, max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None, minimum_cash_buffer=None,
        maximum_price_age=None, maximum_portfolio_snapshot_age=None, allow_reduce_only_during_halt=True,
    )
    return PortfolioRiskGatewayContext(
        store=PortfolioRiskLedgerStore(storage_root if storage_root is not None else config.ml_artifacts_root), portfolio_id=config.paper_session_id,
        portfolio_snapshot=portfolio_snapshot, price_snapshot=price_snapshot,
        risk_spec=PortfolioRiskSpec(schema_version=PORTFOLIO_RISK_SPEC_SCHEMA_VERSION, policy=policy), portfolio_halted=False, consecutive_losses=0,
    )


def _execution_gateway_runner_environment(config: ExecutionGatewayConfigSchema) -> ExecutionGatewayRunnerEnvironment:
    return ExecutionGatewayRunnerEnvironment(
        manifest_store=_execution_gateway_manifest_store(config), event_store=_execution_gateway_event_store(config),
        paper_bridge_environment=_execution_gateway_paper_bridge_environment(config),
        portfolio_risk_context=_execution_gateway_default_portfolio_risk_context(config),
    )


def _load_paper_orders_for_session(config: ExecutionGatewayConfigSchema, paper_session_id: str) -> list[OrderRequest]:
    """Extracts every distinct source `OrderRequest` a paper session's own
    ledger recorded, in first-seen order -- every `ORDER_STATE_EVENT`
    ledger entry embeds the order's full economic detail alongside its
    own transition, so any single entry per order is enough to recover
    it (see `paper_trading.runner._order_state_payload`)."""
    event_store = PaperSessionEventStore(config.ml_artifacts_root)
    ledger = event_store.read_events(paper_session_id)
    seen: dict[str, OrderRequest] = {}
    for entry in ledger:
        if entry.kind is LedgerEntryKind.ORDER_STATE_EVENT:
            order_raw = entry.payload.get("order")
            if order_raw is None:
                continue
            order = OrderRequest.from_json_dict(as_json_dict(order_raw, field_name="order"))
            seen.setdefault(order.order_id, order)
    return list(seen.values())


def cmd_create_execution_gateway_spec(args: argparse.Namespace) -> int:
    """Milestone 8: dry-run -- build/validate an `ExecutionGatewaySpec`
    from `--config` and print its deterministic `execution_gateway_spec_id`.
    Writes nothing; does NOT verify source eligibility (that happens at
    session start, fail-closed)."""
    config = _load_execution_gateway_config(Path(args.config))
    spec = config.build()
    identity = compute_execution_gateway_spec_id(spec)
    print(f"execution_gateway_spec_id: {identity.execution_gateway_spec_id}")
    print(f"execution_mode: {spec.execution_mode.value}")
    print(f"adapter_kind: {spec.adapter_kind.value}")
    print(f"paper_session_id: {spec.paper_session_id}")
    print(_NO_LIVE_EXECUTION_NOTICE)
    return 0


def _run_or_resume_dummy_execution_session(args: argparse.Namespace) -> int:
    config = _load_execution_gateway_config(Path(args.config))
    spec = config.build()
    environment = _execution_gateway_runner_environment(config)
    paper_orders = _load_paper_orders_for_session(config, spec.paper_session_id)
    market_events = load_replay_events(Path(args.replay_source))
    adapter = DeterministicDummyBrokerAdapter(adapter_id=args.adapter_id, scenario=spec.dummy_broker_scenario)
    manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=paper_orders, market_events=market_events, event_time=utc_now().to_pydatetime())
    print(f"execution_session_id: {manifest.execution_session_id}")
    print(f"current_stage: {manifest.current_stage.value}")
    print(_NO_LIVE_EXECUTION_NOTICE)
    return 0 if manifest.current_stage is ExecutionSessionStage.COMPLETED else 2


def cmd_run_dummy_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8: creates (or transparently resumes) a TEST_ONLY
    execution session against the DETERMINISTIC_DUMMY adapter, bridging
    `--paper-session-id`'s own orders and running them against
    `--replay-source`'s bounded, pre-validated market-event sequence.
    Fails closed before a single command is dispatched unless the source
    paper session's full eligibility chain independently re-verifies."""
    return _run_or_resume_dummy_execution_session(args)


def cmd_resume_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8: resumes a prior, non-terminal execution session --
    `--execution-session-id` must already exist and match what `--config`
    resolves to (fails otherwise, never silently creates or cross-loads a
    different session), mirroring the Milestone 7 release-audit's own
    identity-binding fix for `resume-paper-session`."""
    config = _load_execution_gateway_config(Path(args.config))
    _execution_gateway_manifest_store(config).load(args.execution_session_id)  # fails closed if the session does not already exist
    spec = config.build()
    resolved_execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id
    if resolved_execution_session_id != args.execution_session_id:
        raise ExecutionGatewayIdentityError(
            f"--execution-session-id {args.execution_session_id!r} does not match the session {resolved_execution_session_id!r} that --config resolves to "
            "-- refusing to resume a different session than the one named.",
            context={"requested_execution_session_id": args.execution_session_id, "config_resolved_execution_session_id": resolved_execution_session_id},
        )
    return _run_or_resume_dummy_execution_session(args)


def cmd_pause_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8: durably pauses a `RUNNING` execution session -- a
    subsequent `resume-execution-session` continues from the ledger's own
    last completed step."""
    config = _load_execution_gateway_config(Path(args.config))
    environment = _execution_gateway_runner_environment(config)
    manifest = pause_execution_session(execution_session_id=args.execution_session_id, environment=environment, event_time=utc_now().to_pydatetime())
    print(f"execution_session_id: {manifest.execution_session_id}")
    print(f"current_stage: {manifest.current_stage.value}")
    return 0


def cmd_inspect_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8: prints a human-readable (or JSON) summary of one
    execution session's manifest."""
    config = _load_execution_gateway_config(Path(args.config))
    manifest = _execution_gateway_manifest_store(config).load(args.execution_session_id)
    if args.format == "json":
        import json

        print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    print(f"execution_session_id: {manifest.execution_session_id}")
    print(f"paper_session_id: {manifest.paper_session_id}")
    print(f"adapter_id: {manifest.adapter_id}")
    print(f"execution_mode: {manifest.execution_mode.value}")
    print(f"current_stage: {manifest.current_stage.value}")
    print(f"created_event_time: {manifest.created_event_time}")
    print(f"last_transition_event_time: {manifest.last_transition_event_time}")
    if manifest.semantic_digest:
        print(f"semantic_digest: {manifest.semantic_digest}")
    if manifest.failure_category:
        print(f"failure_category: {manifest.failure_category}")
        print(f"failure_stage: {manifest.failure_stage}")
        print(f"recoverable: {manifest.recoverable}")
    return 0


def _load_execution_session_ledger(args: argparse.Namespace) -> tuple[ExecutionGatewayConfigSchema, list[ExecutionLedgerEntry]]:
    config = _load_execution_gateway_config(Path(args.config))
    _execution_gateway_manifest_store(config).load(args.execution_session_id)  # fails closed if the session does not exist
    ledger = _execution_gateway_event_store(config).read_events(args.execution_session_id)
    return config, ledger


def cmd_inspect_execution_intents(args: argparse.Namespace) -> int:
    """Milestone 8: lists every execution intent an execution session's
    ledger has recorded (accepted and rejected)."""
    _config, ledger = _load_execution_session_ledger(args)
    limit = _require_non_negative_limit(args.limit)
    intents = [e for e in ledger if e.entry_kind in (ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED, ExecutionLedgerEntryKind.EXECUTION_INTENT_REJECTED)]
    for entry in intents[:limit]:
        print(f"{entry.entry_kind.value}: execution_intent_id={entry.payload.get('execution_intent_id')} instrument_id={entry.payload.get('instrument_id')} side={entry.payload.get('side')} quantity={entry.payload.get('quantity')}")
    if len(intents) > limit:
        print(f"... truncated: {len(intents) - limit} additional intent(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_execution_commands(args: argparse.Namespace) -> int:
    """Milestone 8: lists every command an execution session's ledger has
    recorded, along with its dispatch outcome."""
    _config, ledger = _load_execution_session_ledger(args)
    limit = _require_non_negative_limit(args.limit)
    commands = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED]
    outcome_kinds = (ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED, ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED, ExecutionLedgerEntryKind.COMMAND_MARKED_UNKNOWN, ExecutionLedgerEntryKind.COMMAND_REJECTED)
    outcomes = {e.payload.get("command_id"): e.entry_kind.value for e in ledger if e.entry_kind in outcome_kinds}
    for entry in commands[:limit]:
        command_id = entry.payload.get("command_id")
        print(f"command_id={command_id} command_type={entry.payload.get('command_type')} outcome={outcomes.get(command_id, 'unresolved')}")
    if len(commands) > limit:
        print(f"... truncated: {len(commands) - limit} additional command(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_execution_orders(args: argparse.Namespace) -> int:
    """Milestone 8: reconstructs and lists every execution order an
    execution session's ledger implies, purely from the ledger (never a
    cached view)."""
    _config, ledger = _load_execution_session_ledger(args)
    limit = _require_non_negative_limit(args.limit)
    from quant_platform.execution_gateway.commands import SubmitOrderCommand
    from quant_platform.execution_gateway.events import BrokerEvent as _BrokerEvent

    submits = {compute_execution_order_id(SubmitOrderCommand.from_json_dict(e.payload)): SubmitOrderCommand.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and e.payload.get("command_type") == "submit_order"}
    for order_id, command in list(submits.items())[:limit]:
        state_events = [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload.get("execution_order_id") == order_id]
        broker_events = [_BrokerEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED and e.payload.get("client_order_id") == command.client_order_id]
        fills = [ExecutionFill.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED and e.payload.get("execution_order_id") == order_id]
        order = reconstruct_execution_order(submit_command=command, state_events=state_events, broker_events=broker_events, fills=fills)
        print(f"execution_order_id={order_id} state={order.current_state.value} filled={order.filled_quantity} remaining={order.remaining_quantity}")
    if len(submits) > limit:
        print(f"... truncated: {len(submits) - limit} additional order(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_execution_fills(args: argparse.Namespace) -> int:
    """Milestone 8: lists every fill an execution session's ledger has
    recorded."""
    _config, ledger = _load_execution_session_ledger(args)
    limit = _require_non_negative_limit(args.limit)
    fills = [ExecutionFill.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED]
    for fill in fills[:limit]:
        print(f"execution_fill_id={fill.execution_fill_id} execution_order_id={fill.execution_order_id} quantity={fill.quantity} price={fill.price} gross_notional={fill.gross_notional}")
    if len(fills) > limit:
        print(f"... truncated: {len(fills) - limit} additional fill(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_broker_events(args: argparse.Namespace) -> int:
    """Milestone 8: lists every normalized broker event an execution
    session's ledger has recorded (received, duplicate, out-of-order,
    conflict)."""
    _config, ledger = _load_execution_session_ledger(args)
    limit = _require_non_negative_limit(args.limit)
    kinds = (ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED, ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE, ExecutionLedgerEntryKind.BROKER_EVENT_OUT_OF_ORDER, ExecutionLedgerEntryKind.BROKER_EVENT_CONFLICT)
    events = [e for e in ledger if e.entry_kind in kinds]
    for entry in events[:limit]:
        broker_event = BrokerEvent.from_json_dict(entry.payload)
        print(f"{entry.entry_kind.value}: broker_sequence={broker_event.broker_sequence} event_type={broker_event.event_type.value} client_order_id={broker_event.client_order_id}")
    if len(events) > limit:
        print(f"... truncated: {len(events) - limit} additional event(s) not shown (--limit={limit})")
    return 0


def cmd_inspect_execution_health(args: argparse.Namespace) -> int:
    """Milestone 8: prints the current kill-switch state (event-sourced,
    reconstructed from the ledger, never a cached field)."""
    config = _load_execution_gateway_config(Path(args.config))
    _execution_gateway_manifest_store(config).load(args.execution_session_id)
    state = current_kill_switch_state(execution_session_id=args.execution_session_id, event_store=_execution_gateway_event_store(config))
    print(f"execution_session_id: {args.execution_session_id}")
    print(f"kill_switch_state: {state.value}")
    return 0


def cmd_inspect_execution_reconciliation(args: argparse.Namespace) -> int:
    """Milestone 8: runs independent reconciliation between the ledger's
    own reconstruction and the (dummy) broker's snapshots. NOTE: this
    inspection command constructs a FRESH `DeterministicDummyBrokerAdapter`
    with no prior state, so it can only meaningfully reconcile a session
    whose ledger alone fully determines the expected broker state (a
    COMPLETED session) -- a still-RUNNING session's in-memory adapter
    state cannot be reconstructed this way; use `verify-execution-session`
    for a ledger-only-based check instead."""
    config, ledger = _load_execution_session_ledger(args)
    spec = config.build()
    adapter = DeterministicDummyBrokerAdapter(adapter_id=args.execution_session_id[:16], scenario=spec.dummy_broker_scenario)
    adapter.initialize(execution_session_id=args.execution_session_id, event_time=utc_now().to_pydatetime())
    report = reconcile_execution_session(execution_session_id=args.execution_session_id, ledger=ledger, adapter=adapter, event_time=utc_now().to_pydatetime(), policy=spec.reconciliation_policy)
    print(f"is_reconciled: {report.is_reconciled}")
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.issue_code}: {issue.description} (expected={issue.expected!r} actual={issue.actual!r})")
    return 0 if report.is_reconciled else 1


def cmd_report_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8 (Section 27): prints the durable session report,
    recomputed purely from the ledger."""
    _config, ledger = _load_execution_session_ledger(args)
    report = generate_execution_session_report(execution_session_id=args.execution_session_id, ledger=ledger)
    if args.format == "json":
        import json

        print(json.dumps(report.to_json_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    for section_name, section in report.sections.items():
        print(f"{section_name}: {section}")
    print(_NO_LIVE_EXECUTION_NOTICE)
    return 0


def cmd_verify_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8 (Section 28): independently re-verifies spec identity,
    ledger integrity, order-state legality, fill identity, and FOK/IOC
    semantics -- never trusting the persisted manifest."""
    config, ledger = _load_execution_session_ledger(args)
    spec = config.build()
    report = verify_execution_session(spec, execution_session_id=args.execution_session_id, ledger=ledger)
    print(f"is_ready: {report.is_ready}")
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")
    return 0 if report.is_ready else 1


def cmd_replay_execution_session(args: argparse.Namespace) -> int:
    """Milestone 8 (Section 30): runs `--config`'s spec end-to-end
    against a FRESH, isolated store rooted at `--replay-storage-root`,
    printing the resulting semantic digest -- used to independently
    confirm two separate runs of the same immutable inputs produce the
    same deterministic outcome."""
    from quant_platform.execution_gateway.replay import replay_execution_session

    config = _load_execution_gateway_config(Path(args.config))
    spec = config.build()
    paper_orders = _load_paper_orders_for_session(config, spec.paper_session_id)
    market_events = load_replay_events(Path(args.replay_source))
    result = replay_execution_session(
        spec, storage_root=Path(args.replay_storage_root), paper_bridge_environment=_execution_gateway_paper_bridge_environment(config),
        portfolio_risk_context=_execution_gateway_default_portfolio_risk_context(config, storage_root=args.replay_storage_root), paper_orders=paper_orders,
        market_events=market_events, adapter_id=args.adapter_id, event_time=utc_now().to_pydatetime(),
    )
    print(f"execution_session_id: {result.execution_session_id}")
    print(f"current_stage: {result.manifest.current_stage.value}")
    print(f"semantic_digest: {result.semantic_digest}")
    print(f"is_reconciled: {result.reconciliation_report.is_reconciled}")
    print(f"verification_is_ready: {result.verification_report.is_ready}")
    return 0 if result.manifest.current_stage is ExecutionSessionStage.COMPLETED else 2


def cmd_compare_execution_to_paper(args: argparse.Namespace) -> int:
    """Milestone 8: diagnostic-only comparison between an execution
    session's own fill/order counts and its source paper session's --
    genuine decision/order-count mismatches (not merely cost/latency
    differences) warrant investigation, exactly like Milestone 7's own
    `compare-paper-to-backtest`."""
    config, ledger = _load_execution_session_ledger(args)
    execution_report = generate_execution_session_report(execution_session_id=args.execution_session_id, ledger=ledger)
    paper_orders = _load_paper_orders_for_session(config, args.paper_session_id)
    print(f"paper_order_count={len(paper_orders)}")
    print(f"execution_order_count={execution_report.sections['OrderSummary']['total_orders']}")  # type: ignore[index]
    print(f"execution_fill_count={execution_report.sections['FillSummary']['fill_count']}")  # type: ignore[index]
    print("NOTE: this comparison is diagnostic only, never a promotion or completion decision.")
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

    def _add_strategy_runtime_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--replay-source", required=True, help="Path to a bounded, deterministic JSONL replay source (Section 32).")
        p.add_argument("--feature-name", required=True, action="append", help="One supported feature name (repeatable); must cover the fitted model's own declared feature schema.")
        p.add_argument("--long-threshold", type=float, default=0.6, help="p_positive >= this fires a LONG decision. Must be in (0.5, 1.0].")
        p.add_argument("--short-threshold", type=float, default=0.4, help="p_positive <= this fires a SHORT decision. Must be in [0.0, 0.5).")
        p.add_argument("--target-quantity", type=float, default=1.0, help="Fixed order quantity for every non-abstain decision.")

    create_paper_spec_parser = subparsers.add_parser(
        "create-paper-trading-spec", help="Milestone 7: dry-run -- build/validate a PaperTradingSpec from --config and print its paper_session_spec_id. Writes nothing.",
    )
    create_paper_spec_parser.add_argument("--config", required=True)
    create_paper_spec_parser.set_defaults(handler=cmd_create_paper_trading_spec)

    run_paper_session_parser = subparsers.add_parser(
        "run-paper-session", help="Milestone 7: create (or transparently resume) and run a REPLAY_PAPER/FORWARD_PAPER session against a bounded replay source. No live order is ever sent.",
    )
    run_paper_session_parser.add_argument("--config", required=True)
    _add_strategy_runtime_arguments(run_paper_session_parser)
    run_paper_session_parser.set_defaults(handler=cmd_run_paper_session)

    resume_paper_session_parser = subparsers.add_parser(
        "resume-paper-session", help="Milestone 7: resume a prior, non-terminal paper session -- fails if --paper-session-id does not already exist.",
    )
    resume_paper_session_parser.add_argument("--config", required=True)
    resume_paper_session_parser.add_argument("--paper-session-id", required=True)
    _add_strategy_runtime_arguments(resume_paper_session_parser)
    resume_paper_session_parser.set_defaults(handler=cmd_resume_paper_session)

    pause_paper_session_parser = subparsers.add_parser(
        "pause-paper-session", help="Milestone 7: durably pause a RUNNING paper session.",
    )
    pause_paper_session_parser.add_argument("--config", required=True)
    pause_paper_session_parser.add_argument("--paper-session-id", required=True)
    pause_paper_session_parser.set_defaults(handler=cmd_pause_paper_session)

    inspect_paper_session_parser = subparsers.add_parser(
        "inspect-paper-session", help="Milestone 7: print a human-readable (or JSON) paper/shadow session manifest summary.",
    )
    inspect_paper_session_parser.add_argument("--config", required=True)
    inspect_paper_session_parser.add_argument("--paper-session-id", required=True)
    inspect_paper_session_parser.add_argument("--format", choices=["text", "json"], default="text")
    inspect_paper_session_parser.set_defaults(handler=cmd_inspect_paper_session)

    report_paper_session_parser = subparsers.add_parser(
        "report-paper-session", help="Milestone 7: print the 15-summary durable session report (Section 27).",
    )
    report_paper_session_parser.add_argument("--config", required=True)
    report_paper_session_parser.add_argument("--paper-session-id", required=True)
    report_paper_session_parser.add_argument("--format", choices=["text", "json"], default="text")
    report_paper_session_parser.add_argument("--verify", action="store_true", help="Also run verify-paper-session and include its outcome in the report.")
    report_paper_session_parser.set_defaults(handler=cmd_report_paper_session)

    verify_paper_session_parser = subparsers.add_parser(
        "verify-paper-session", help="Milestone 7: independently re-verify spec identity, eligibility chain, ledger/manifest integrity, and reconciliation (Section 26).",
    )
    verify_paper_session_parser.add_argument("--config", required=True)
    verify_paper_session_parser.add_argument("--paper-session-id", required=True)
    verify_paper_session_parser.set_defaults(handler=cmd_verify_paper_session)

    compare_paper_to_backtest_parser = subparsers.add_parser(
        "compare-paper-to-backtest", help="Milestone 7: diagnostic-only comparison between a paper session and its source backtest's aggregate metrics (Section 28).",
    )
    compare_paper_to_backtest_parser.add_argument("--config", required=True)
    compare_paper_to_backtest_parser.add_argument("--paper-session-id", required=True)
    compare_paper_to_backtest_parser.add_argument("--backtest-id", required=True)
    compare_paper_to_backtest_parser.set_defaults(handler=cmd_compare_paper_to_backtest)

    inspect_paper_orders_parser = subparsers.add_parser(
        "inspect-paper-orders", help="Milestone 7: list every order a paper session's ledger has recorded.",
    )
    inspect_paper_orders_parser.add_argument("--config", required=True)
    inspect_paper_orders_parser.add_argument("--paper-session-id", required=True)
    inspect_paper_orders_parser.add_argument("--limit", type=int, default=_DEFAULT_INSPECTION_ROW_LIMIT, help=f"Maximum rows to print (default {_DEFAULT_INSPECTION_ROW_LIMIT}); a long-running session's own order count is unbounded, this is a terminal/operator-safety cap, never a data-loss risk (the full ledger is untouched).")
    inspect_paper_orders_parser.set_defaults(handler=cmd_inspect_paper_orders)

    inspect_paper_fills_parser = subparsers.add_parser(
        "inspect-paper-fills", help="Milestone 7: list every fill a paper session's ledger has recorded.",
    )
    inspect_paper_fills_parser.add_argument("--config", required=True)
    inspect_paper_fills_parser.add_argument("--paper-session-id", required=True)
    inspect_paper_fills_parser.add_argument("--limit", type=int, default=_DEFAULT_INSPECTION_ROW_LIMIT, help=f"Maximum rows to print (default {_DEFAULT_INSPECTION_ROW_LIMIT}).")
    inspect_paper_fills_parser.set_defaults(handler=cmd_inspect_paper_fills)

    inspect_paper_risk_events_parser = subparsers.add_parser(
        "inspect-paper-risk-events", help="Milestone 7: list every risk-check failure and kill-switch transition a paper session's ledger has recorded.",
    )
    inspect_paper_risk_events_parser.add_argument("--config", required=True)
    inspect_paper_risk_events_parser.add_argument("--paper-session-id", required=True)
    inspect_paper_risk_events_parser.add_argument("--limit", type=int, default=_DEFAULT_INSPECTION_ROW_LIMIT, help=f"Maximum rows to print per section (default {_DEFAULT_INSPECTION_ROW_LIMIT}).")
    inspect_paper_risk_events_parser.set_defaults(handler=cmd_inspect_paper_risk_events)

    inspect_paper_reconciliation_parser = subparsers.add_parser(
        "inspect-paper-reconciliation", help="Milestone 7: run the 11 independent reconciliation checks against a paper session's ledger (Section 25).",
    )
    inspect_paper_reconciliation_parser.add_argument("--config", required=True)
    inspect_paper_reconciliation_parser.add_argument("--paper-session-id", required=True)
    inspect_paper_reconciliation_parser.set_defaults(handler=cmd_inspect_paper_reconciliation)

    run_shadow_session_parser = subparsers.add_parser(
        "run-shadow-session", help="Milestone 7: create and run a SHADOW_OBSERVATION session. Hypothetical only -- never touches a real simulated account.",
    )
    run_shadow_session_parser.add_argument("--config", required=True)
    _add_strategy_runtime_arguments(run_shadow_session_parser)
    run_shadow_session_parser.set_defaults(handler=cmd_run_shadow_session)

    report_shadow_session_parser = subparsers.add_parser(
        "report-shadow-session", help="Milestone 7: print the SHADOW-labeled observation summary only -- never folded into any real-account figure.",
    )
    report_shadow_session_parser.add_argument("--config", required=True)
    report_shadow_session_parser.add_argument("--paper-session-id", required=True)
    report_shadow_session_parser.set_defaults(handler=cmd_report_shadow_session)

    # ----------------------------------------------------------------
    # Milestone 8: broker-neutral deterministic execution gateway (TEST-ONLY)
    # ----------------------------------------------------------------
    create_execution_gateway_spec_parser = subparsers.add_parser(
        "create-execution-gateway-spec", help="Milestone 8: dry-run -- build/validate an ExecutionGatewaySpec from --config and print its execution_gateway_spec_id. Writes nothing.",
    )
    create_execution_gateway_spec_parser.add_argument("--config", required=True)
    create_execution_gateway_spec_parser.set_defaults(handler=cmd_create_execution_gateway_spec)

    def _add_execution_replay_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--replay-source", required=True, help="Path to a bounded, deterministic JSONL replay source (Milestone 7 Section 32, reused directly).")
        p.add_argument("--adapter-id", default="dummy-broker-1", help="Identifier for this run's DeterministicDummyBrokerAdapter instance.")

    run_dummy_execution_session_parser = subparsers.add_parser(
        "run-dummy-execution-session", help="Milestone 8: create (or transparently resume) and run a TEST_ONLY execution session against the deterministic dummy broker. No live order is ever sent.",
    )
    run_dummy_execution_session_parser.add_argument("--config", required=True)
    _add_execution_replay_arguments(run_dummy_execution_session_parser)
    run_dummy_execution_session_parser.set_defaults(handler=cmd_run_dummy_execution_session)

    resume_execution_session_parser = subparsers.add_parser(
        "resume-execution-session", help="Milestone 8: resume a prior, non-terminal execution session -- fails if --execution-session-id does not already exist or does not match --config.",
    )
    resume_execution_session_parser.add_argument("--config", required=True)
    resume_execution_session_parser.add_argument("--execution-session-id", required=True)
    _add_execution_replay_arguments(resume_execution_session_parser)
    resume_execution_session_parser.set_defaults(handler=cmd_resume_execution_session)

    pause_execution_session_parser = subparsers.add_parser(
        "pause-execution-session", help="Milestone 8: durably pause a RUNNING execution session.",
    )
    pause_execution_session_parser.add_argument("--config", required=True)
    pause_execution_session_parser.add_argument("--execution-session-id", required=True)
    pause_execution_session_parser.set_defaults(handler=cmd_pause_execution_session)

    inspect_execution_session_parser = subparsers.add_parser(
        "inspect-execution-session", help="Milestone 8: print a human-readable (or JSON) execution session manifest summary.",
    )
    inspect_execution_session_parser.add_argument("--config", required=True)
    inspect_execution_session_parser.add_argument("--execution-session-id", required=True)
    inspect_execution_session_parser.add_argument("--format", choices=["text", "json"], default="text")
    inspect_execution_session_parser.set_defaults(handler=cmd_inspect_execution_session)

    def _add_execution_session_and_limit_arguments(p: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
        p.add_argument("--config", required=True)
        p.add_argument("--execution-session-id", required=True)
        p.add_argument("--limit", type=int, default=_DEFAULT_INSPECTION_ROW_LIMIT, help=f"Maximum rows to print (default {_DEFAULT_INSPECTION_ROW_LIMIT}).{help_suffix}")

    inspect_execution_intents_parser = subparsers.add_parser("inspect-execution-intents", help="Milestone 8: list every execution intent an execution session's ledger has recorded.")
    _add_execution_session_and_limit_arguments(inspect_execution_intents_parser)
    inspect_execution_intents_parser.set_defaults(handler=cmd_inspect_execution_intents)

    inspect_execution_commands_parser = subparsers.add_parser("inspect-execution-commands", help="Milestone 8: list every command an execution session's ledger has recorded, with its dispatch outcome.")
    _add_execution_session_and_limit_arguments(inspect_execution_commands_parser)
    inspect_execution_commands_parser.set_defaults(handler=cmd_inspect_execution_commands)

    inspect_execution_orders_parser = subparsers.add_parser("inspect-execution-orders", help="Milestone 8: reconstruct and list every execution order an execution session's ledger implies.")
    _add_execution_session_and_limit_arguments(inspect_execution_orders_parser)
    inspect_execution_orders_parser.set_defaults(handler=cmd_inspect_execution_orders)

    inspect_execution_fills_parser = subparsers.add_parser("inspect-execution-fills", help="Milestone 8: list every fill an execution session's ledger has recorded.")
    _add_execution_session_and_limit_arguments(inspect_execution_fills_parser)
    inspect_execution_fills_parser.set_defaults(handler=cmd_inspect_execution_fills)

    inspect_broker_events_parser = subparsers.add_parser("inspect-broker-events", help="Milestone 8: list every normalized broker event an execution session's ledger has recorded.")
    _add_execution_session_and_limit_arguments(inspect_broker_events_parser)
    inspect_broker_events_parser.set_defaults(handler=cmd_inspect_broker_events)

    inspect_execution_health_parser = subparsers.add_parser("inspect-execution-health", help="Milestone 8: print the current kill-switch state, reconstructed from the ledger.")
    inspect_execution_health_parser.add_argument("--config", required=True)
    inspect_execution_health_parser.add_argument("--execution-session-id", required=True)
    inspect_execution_health_parser.set_defaults(handler=cmd_inspect_execution_health)

    inspect_execution_reconciliation_parser = subparsers.add_parser("inspect-execution-reconciliation", help="Milestone 8: run independent reconciliation between the ledger's own reconstruction and a fresh dummy-broker snapshot.")
    inspect_execution_reconciliation_parser.add_argument("--config", required=True)
    inspect_execution_reconciliation_parser.add_argument("--execution-session-id", required=True)
    inspect_execution_reconciliation_parser.set_defaults(handler=cmd_inspect_execution_reconciliation)

    report_execution_session_parser = subparsers.add_parser("report-execution-session", help="Milestone 8: print the durable execution session report, recomputed purely from the ledger.")
    report_execution_session_parser.add_argument("--config", required=True)
    report_execution_session_parser.add_argument("--execution-session-id", required=True)
    report_execution_session_parser.add_argument("--format", choices=["text", "json"], default="text")
    report_execution_session_parser.set_defaults(handler=cmd_report_execution_session)

    verify_execution_session_parser = subparsers.add_parser("verify-execution-session", help="Milestone 8: independently re-verify spec identity, ledger integrity, order-state legality, fill identity, and FOK/IOC semantics.")
    verify_execution_session_parser.add_argument("--config", required=True)
    verify_execution_session_parser.add_argument("--execution-session-id", required=True)
    verify_execution_session_parser.set_defaults(handler=cmd_verify_execution_session)

    replay_execution_session_parser = subparsers.add_parser("replay-execution-session", help="Milestone 8: run --config's spec end-to-end against a fresh, isolated store and print the resulting semantic digest.")
    replay_execution_session_parser.add_argument("--config", required=True)
    _add_execution_replay_arguments(replay_execution_session_parser)
    replay_execution_session_parser.add_argument("--replay-storage-root", required=True, help="Fresh, isolated storage root for this replay run.")
    replay_execution_session_parser.set_defaults(handler=cmd_replay_execution_session)

    compare_execution_to_paper_parser = subparsers.add_parser("compare-execution-to-paper", help="Milestone 8: diagnostic-only comparison between an execution session and its source paper session's order/fill counts.")
    compare_execution_to_paper_parser.add_argument("--config", required=True)
    compare_execution_to_paper_parser.add_argument("--execution-session-id", required=True)
    compare_execution_to_paper_parser.add_argument("--paper-session-id", required=True)
    compare_execution_to_paper_parser.set_defaults(handler=cmd_compare_execution_to_paper)

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
