"""End-to-end Milestone 5 integration tests: real synthetic historical
data -> a real Milestone 3 research dataset -> a real Milestone 4A
prepared experiment -> a real Milestone 4B execution -> a real Milestone
4E leakage-safe calibration -> a real `BacktestRunner` leakage-safe
financial evaluation run, using the ACTUAL production stores. Mirrors
`tests/integration/test_calibration_engine.py`'s conventions exactly, one
layer up: this package needs a COMPLETED execution (a genuinely new
requirement `calibration`'s own fixture never had, since
`BacktestSpec.source_execution_id` must reference a completed
`ExecutionManifest` -- see `backtesting.runner.resolve_backtest_inputs`).

`test_backtest_runner_end_to_end` IS this milestone's bounded end-to-end
infrastructure acceptance run (Section 41 lineage): deterministic
synthetic data, a real completed calibration, signal mapping, fill/trade
simulation, equity/drawdown, financial metrics, benchmarks, cost
sensitivity, bucket analysis, persistence, resume (exercised separately
by the crash-window tests below), and independent verification (including
the financial-metrics recomputation proof) -- an INFRASTRUCTURE
acceptance test, not evidence of profitability (the test-only constant
model predicts nothing meaningful; see `ml.testing`'s own docstring, and
`test_backtesting_real_model_acceptance.py` for the real-model Section 48
acceptance run)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifestStore
from quant_platform.backtesting.metrics import compute_financial_metrics
from quant_platform.backtesting.models import (
    BacktestStage,
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionDirection,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.runner import (
    BacktestOutcome,
    BacktestRunner,
    OuterFoldBacktestResult,
    inspect_backtest_lock,
    recover_backtest_lock,
)
from quant_platform.backtesting.signals import SignalSet
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
    compute_backtest_identity,
)
from quant_platform.backtesting.timeline import BarReturnTimeline, bar_return_timeline_to_equity_curve
from quant_platform.backtesting.trades import TradeRecord, TradeSet
from quant_platform.backtesting.verification import verify_backtest
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationStage,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.calibration.runner import CalibrationRunner
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
)
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    BacktestResumeError,
    ExperimentLockError,
)
from quant_platform.core.types import Timeframe
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.locking import LockInfo
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
    CodeRevisionBinding,
    DatasetBinding,
    ExperimentStatus,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict, write_json_atomic
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import TEST_MODEL_NAME, TEST_MODEL_VERSION, ConstantTestModelFactory
from quant_platform.optimization.inner_splits import InnerSplitConfig

_WALK_FORWARD_SPLIT = {"strategy": "expanding_walk_forward", "params": {"n_splits": 2, "test_size": 150, "purge_bars": 5, "embargo_bars": 2}}


def _build_dataset(tmp_path: Path):
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(1500, seed=11)
    seed_canonical_dataset(historical_root, df)

    canonical_store = CanonicalStore(historical_root)
    manifest_store = ManifestStore(historical_root)
    historical_loader = DatasetLoader(canonical_store, manifest_store)

    registry = FeatureRegistry()
    register_core_technical_features(
        registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1, 5), momentum_windows=(10,), atr_window=14)
    )

    research_store = ResearchDatasetStore(research_root)
    research_manifest_store = ResearchManifestStore(research_root)
    builder = ResearchDatasetBuilder(historical_loader=historical_loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store)
    feature_names = tuple(spec.name for spec in registry.list_features())
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=start, end=end, feature_names=feature_names,
        label_definition=LabelDefinition(name="fut5", kind=LabelKind.BINARY_DIRECTION, horizon_bars=5),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        preprocessing={},  # type: ignore[arg-type]
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, research_store, historical_root


def _build_ready_setup(
    tmp_path: Path, *, seed: int = 42,
) -> tuple[BacktestSpec, BacktestRunner, Path, ResearchManifestStore, ResearchDatasetStore, Path]:
    """Builds a real synthetic dataset, a real READY experiment, a real
    COMPLETED execution, a real COMPLETED calibration, and a valid
    `BacktestSpec` + `BacktestRunner` wired to real, on-disk stores.
    Returns `(spec, runner, ml_artifacts_root, research_manifest_store,
    research_store, historical_root)`."""
    dataset_manifest, research_manifest_store, research_store, historical_root = _build_dataset(tmp_path)
    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value, source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_binding = FeatureBinding(
        feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    experiment_spec = ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding,
        label_binding=LabelBinding(name="fut5", kind=LabelKind.BINARY_DIRECTION.value, horizon_bars=5, label_type=LabelType.BINARY),
        split_binding=SplitBinding(strategy=_WALK_FORWARD_SPLIT["strategy"], params=_WALK_FORWARD_SPLIT["params"]),  # type: ignore[arg-type]
        preprocessing_binding=preprocessing_binding, model_name=TEST_MODEL_NAME, model_version=TEST_MODEL_VERSION,
        hyperparameters=ModelHyperparameters(values={}), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=1), code_revision_binding=CodeRevisionBinding(revision="a" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )
    model_registry = ModelRegistry()
    model_registry.register(ModelDefinition(
        name=TEST_MODEL_NAME, version=TEST_MODEL_VERSION, description="TEST-ONLY deterministic model",
        capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION), supports_predict_proba=True),
        factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
    ))
    ml_artifacts_root = tmp_path / "ml_artifacts"
    preparer = ExperimentPreparer(ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store)
    experiment_manifest = preparer.prepare(experiment_spec)
    assert experiment_manifest.status is ExperimentStatus.READY, experiment_manifest.failure_summary
    experiment_id = experiment_manifest.identity.experiment_id

    execution_runner = ExecutionRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store,
    )
    execution_runner.run(experiment_id)
    execution_manifest = execution_runner.execution_manifest_store.load(experiment_id)
    assert execution_manifest.stage is ExecutionStage.COMPLETED, execution_manifest.failure_summary

    calibration_spec = CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id=experiment_id,
        base_model_definition_identity=model_registry.get(TEST_MODEL_NAME, TEST_MODEL_VERSION).fingerprint(),
        dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        calibration_method_candidates=(CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=10, minimum_samples_per_class=2,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2, embargo_bars=1),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.NONE),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy", "margin", "bin_support"), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
        seed=seed, determinism_policy=DeterminismPolicy.STRICT,
    )
    calibration_runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
    )
    calibration_outcome = calibration_runner.run(calibration_spec)
    assert calibration_outcome.manifest.stage is CalibrationStage.COMPLETED, calibration_outcome.manifest.failure_summary
    calibration_id = calibration_outcome.manifest.calibration_id

    backtest_spec = BacktestSpec(
        schema_version=1, source_calibration_id=calibration_id, source_experiment_id=experiment_id, source_execution_id=experiment_id,
        dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        instrument_identity=dataset_manifest.symbol, market_timezone="UTC", bar_interval=dataset_manifest.base_timeframe,
        decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT,
        entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=3, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        overlap_policy=OverlapPolicyKind.IGNORE, price_basis=PriceBasisKind.CLOSE,
        spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=5.0),
        commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
        slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
        return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.NON_COMPOUNDED,
        initial_notional=10000.0, determinism_policy=DeterminismPolicy.STRICT, respect_calibration_abstention=True, seed=0,
    )

    dataset_loader = DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root))
    backtest_runner = BacktestRunner(
        ml_artifacts_root=ml_artifacts_root, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root), execution_manifest_store=execution_runner.execution_manifest_store,
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )
    return backtest_spec, backtest_runner, ml_artifacts_root, research_manifest_store, research_store, historical_root


def _new_backtest_runner(ml_artifacts_root: Path, *, research_manifest_store: ResearchManifestStore, research_store: ResearchDatasetStore, historical_root: Path) -> BacktestRunner:
    dataset_loader = DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root))
    return BacktestRunner(
        ml_artifacts_root=ml_artifacts_root, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root), execution_manifest_store=ExecutionManifestStore(ml_artifacts_root),
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )


def _verify(
    backtest_id: str, ml_artifacts_root: Path, *, research_manifest_store: ResearchManifestStore, research_store: ResearchDatasetStore, historical_root: Path,
):
    return verify_backtest(
        backtest_id, backtest_manifest_store=BacktestManifestStore(ml_artifacts_root),
        artifact_store=MLArtifactStore(ml_artifacts_root), event_store=BacktestEventStore(ml_artifacts_root),
        calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root), experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
        execution_manifest_store=ExecutionManifestStore(ml_artifacts_root), research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, dataset_loader=DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root)),
    )


def test_backtest_runner_end_to_end(tmp_path: Path) -> None:
    spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
    outcome = runner.run(spec)
    assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary
    assert not outcome.was_idempotent_no_op
    assert outcome.manifest.total_outer_folds == 2
    assert len(outcome.manifest.completed_outer_fold_indices) == 2

    identity = compute_backtest_identity(spec)
    assert identity.backtest_id == outcome.manifest.backtest_id

    store = BacktestManifestStore(ml_artifacts_root)
    reloaded = store.load(outcome.manifest.backtest_id)
    assert reloaded.stage is BacktestStage.COMPLETED

    second = runner.run(spec)
    assert second.was_idempotent_no_op
    assert second.manifest.backtest_id == outcome.manifest.backtest_id

    artifact_store = MLArtifactStore(ml_artifacts_root)
    ref = outcome.manifest.outer_fold_result_references[0]
    raw = artifact_store.read_artifact(ref.content_hash)
    result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    assert result.outer_test_row_count > 0
    assert "trade_count" in result.financial_metrics or "trade_count" in result.skipped_metrics

    trade_set_raw = artifact_store.read_artifact(result.trade_set_reference.content_hash)
    trade_set_json = parse_json_strict(trade_set_raw.decode("utf-8"))
    assert trade_set_json["trades"], "expected at least one trade to confirm the decision-timestamp policy against"
    for trade_raw in trade_set_json["trades"]:
        trade = TradeRecord.from_json_dict(trade_raw)
        assert abs((trade.gross_return - trade.cost_breakdown.total_cost) - trade.net_return) < 1e-9
        # Milestone 5.1, Section 5: under AFTER_BAR_CLOSE with delay_bars=1
        # over this fixture's gapless synthetic bars, the earliest legal
        # entry instant (the target bar's own open) is EXACTLY the signal
        # bar's close -- i.e. decision_timestamp == entry_timestamp, not
        # merely "some time after" (TradeRecord.__post_init__ already
        # rejects entry BEFORE decision; this confirms the real pipeline
        # produces the tight, expected boundary, not a looser one).
        assert trade.entry_timestamp == trade.decision_timestamp

    report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]
    assert any(i.code == "fold_financials_reproduce" for i in report.infos)
    assert any(i.code == "stitched_equity_reproduces" for i in report.infos)

    assert outcome.manifest.aggregate_report_reference is not None
    report_raw = artifact_store.read_artifact(outcome.manifest.aggregate_report_reference.content_hash)
    report_json = json.loads(report_raw.decode("utf-8"))
    assert "limitations" in report_json and len(report_json["limitations"]) > 0
    assert any("HYPOTHETICAL" in limitation for limitation in report_json["limitations"])

    # Milestone 5.1, Section 3: the stitched whole-backtest chronological
    # equity path is persisted, spans BOTH outer folds contiguously, and
    # its summary metrics are embedded in the aggregate report.
    from quant_platform.backtesting.stitching import StitchedWalkForwardEquity

    assert outcome.manifest.stitched_equity_reference is not None
    stitched_raw = artifact_store.read_artifact(outcome.manifest.stitched_equity_reference.content_hash)
    stitched = StitchedWalkForwardEquity.from_json_dict(json.loads(stitched_raw.decode("utf-8")))
    assert [b.outer_fold_index for b in stitched.fold_boundaries] == [0, 1]
    assert stitched.fold_boundaries[0].stitched_point_start_index == 0
    assert stitched.fold_boundaries[-1].stitched_point_end_index == len(stitched.points) - 1
    stitched_section = report_json["stitched_equity"]
    assert stitched_section is not None
    assert stitched_section["point_count"] == len(stitched.points)
    assert "stitched_total_net_return" in stitched_section["metrics"]


class _CountingArtifactStoreProxy:
    """Delegates to a real `MLArtifactStore`, raising `ExperimentLockError`
    the instant `write_artifact` call number `crash_after_n_writes + 1` is
    attempted -- simulating a process death at a precise point inside
    `run_outer_fold_backtest`'s own write sequence (signals, trades,
    equity curve, gross/net drawdown, benchmarks, cost sensitivity,
    bucket analysis). Mirrors `test_calibration_engine._CountingArtifactStoreProxy`
    exactly."""

    def __init__(self, real_store, *, crash_after_n_writes: int) -> None:
        self._real_store = real_store
        self._crash_after_n_writes = crash_after_n_writes
        self.write_calls = 0

    def write_artifact(self, *args, **kwargs):
        self.write_calls += 1
        if self.write_calls > self._crash_after_n_writes:
            raise ExperimentLockError(f"simulated crash after {self._crash_after_n_writes} artifact write(s) inside run_outer_fold_backtest")
        return self._real_store.write_artifact(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_store, name)


@pytest.mark.parametrize("crash_after_n_writes", [0, 1, 3, 5, 7])
def test_backtest_runner_resumes_after_mid_fold_crash(tmp_path: Path, crash_after_n_writes: int) -> None:
    """A crash partway through outer fold 0's `run_outer_fold_backtest`
    call must leave the manifest resumable (never at a stage claiming
    completed work it does not have), and resume must redo that ENTIRE
    fold from scratch and reach an identical COMPLETED terminal state --
    never partially resuming mid-fold. Five distinct crash points span
    every artifact this fold writes (signal_set, trade_set, equity_curve,
    gross/net drawdown, benchmark/cost-sensitivity/bucket reports)."""
    spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

    import quant_platform.backtesting.runner as runner_module
    real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest

    def crashing_run_outer_fold(*, artifact_store, **kwargs):
        proxy = _CountingArtifactStoreProxy(artifact_store, crash_after_n_writes=crash_after_n_writes)
        return real_run_outer_fold_backtest(artifact_store=proxy, **kwargs)

    runner_module.run_outer_fold_backtest = crashing_run_outer_fold
    try:
        with pytest.raises(ExperimentLockError):
            runner.run(spec)
    finally:
        runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

    backtest_id = compute_backtest_identity(spec).backtest_id
    manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
    assert manifest_after_crash.stage not in (BacktestStage.COMPLETED, BacktestStage.FAILED)
    assert manifest_after_crash.completed_outer_fold_indices == ()

    outcome_resumed = runner.resume(backtest_id)
    assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
    assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)
    assert outcome_resumed.manifest.resume_count >= 1

    report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]


def test_genuine_domain_exception_mid_run_leaves_an_accurate_failed_record(tmp_path: Path) -> None:
    """The lesson carried forward from the Milestone 4E audit
    (`CalibrationRunner._fail` was defined but never called, making
    `FAILED` unreachable): `BacktestRunner._fail` must actually be wired
    into `_run_locked`'s exception handler and reachable for a genuine
    mid-run domain error (as opposed to the `ExperimentLockError`-
    simulated process-crash tests elsewhere in this file)."""
    from quant_platform.core.exceptions import BacktestValidationError

    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

    import quant_platform.backtesting.runner as runner_module
    real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest

    def always_fails(**kwargs):
        raise BacktestValidationError("simulated genuine domain failure: e.g. a malformed market bar frame")

    runner_module.run_outer_fold_backtest = always_fails
    try:
        with pytest.raises(BacktestValidationError, match="simulated genuine domain failure"):
            runner.run(spec)
    finally:
        runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

    backtest_id = compute_backtest_identity(spec).backtest_id
    manifest = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
    assert manifest.stage is BacktestStage.FAILED
    assert manifest.failure_summary is not None
    assert "simulated genuine domain failure" in manifest.failure_summary

    events = BacktestEventStore(ml_artifacts_root).read_events(backtest_id)
    assert events[-1].event_type.value == "run_failed"

    with pytest.raises(BacktestResumeError):
        runner.run(spec)
    with pytest.raises(BacktestResumeError):
        runner.resume(backtest_id)


def test_crash_during_post_fold_stage_transition_burst_is_still_resumable(tmp_path: Path) -> None:
    """A DISTINCT crash window from the mid-fold test above: this test
    crashes AFTER `run_outer_fold_backtest` has already returned
    successfully (every one of its artifacts already exists on disk) but
    DURING `_execute_pipeline`'s subsequent manifest stage-transition
    burst (`SIGNALS_READY` -> ... -> `METRICS_READY`), before the
    `OUTER_FOLD_BACKTEST_RESULT` bundle is written and before
    `completed_outer_fold_indices` is updated. Resume must still redo the
    WHOLE fold (never partially trust the already-written sub-artifacts)."""
    spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

    real_transition = BacktestManifestStore.transition
    call_count = {"n": 0}

    def crashing_transition(self, *args, **kwargs):
        call_count["n"] += 1
        # Allow CREATED->SOURCES_VERIFIED (1st call) through, then crash on
        # the SECOND transition call (SIGNALS_READY) -- i.e. strictly
        # after run_outer_fold_backtest has already returned and every
        # one of its artifacts is durably written.
        if call_count["n"] == 2:
            raise ExperimentLockError("simulated crash during the post-fold manifest stage-transition burst")
        return real_transition(self, *args, **kwargs)

    import quant_platform.backtesting.manifests as manifests_module

    manifests_module.BacktestManifestStore.transition = crashing_transition
    try:
        with pytest.raises(ExperimentLockError):
            runner.run(spec)
    finally:
        manifests_module.BacktestManifestStore.transition = real_transition

    backtest_id = compute_backtest_identity(spec).backtest_id
    manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
    assert manifest_after_crash.stage not in (BacktestStage.COMPLETED, BacktestStage.FAILED)
    assert manifest_after_crash.completed_outer_fold_indices == ()

    outcome_resumed = runner.resume(backtest_id)
    assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
    assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)

    report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]


def test_resume_rejects_a_spec_that_reproduces_a_different_backtest_id(tmp_path: Path) -> None:
    """Milestone 5.1, Section 6: "source identity changes invalidate
    resume." `BacktestRunner.resume(backtest_id, spec=...)` must refuse an
    explicitly-passed `spec` that does not reproduce the SAME `backtest_id`
    being resumed -- without this check, a caller could silently resume
    an in-progress backtest UNDER A DIFFERENT declared configuration
    (e.g. a different cost model), never caught until a later, separate
    `verify_backtest` call."""
    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

    import quant_platform.backtesting.runner as runner_module
    real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest

    def always_crash(**kwargs):
        raise ExperimentLockError("simulated crash to leave a resumable, non-terminal manifest")

    runner_module.run_outer_fold_backtest = always_crash
    try:
        with pytest.raises(ExperimentLockError):
            runner.run(spec)
    finally:
        runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

    backtest_id = compute_backtest_identity(spec).backtest_id
    manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
    assert manifest_after_crash.stage not in (BacktestStage.COMPLETED, BacktestStage.FAILED)

    from dataclasses import replace

    mismatched_spec = replace(spec, initial_notional=spec.initial_notional * 2.0)
    assert compute_backtest_identity(mismatched_spec).backtest_id != backtest_id

    with pytest.raises(BacktestResumeError, match="mismatched source identity"):
        runner.resume(backtest_id, spec=mismatched_spec)

    # The correct spec (reproducing the SAME backtest_id) still resumes cleanly.
    outcome_resumed = runner.resume(backtest_id, spec=spec)
    assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary


class TestRunLevelCrashWindows:
    """Milestone 5.1, Section 6: crash-window coverage BEYOND one outer
    fold's own processing (already covered above by the parametrized
    mid-fold test and `test_crash_during_post_fold_stage_transition_
    burst_is_still_resumable`) -- the run-level boundaries introduced or
    newly reachable in this milestone: before source verification, during
    the whole-backtest stitched-equity build, during the final aggregate
    report build, and immediately before the terminal COMPLETED
    transition. Every case proves: resume reaches COMPLETED; the crash
    left no partial/half-written artifact trusted; `verify_backtest`
    passes afterward."""

    def test_crash_before_source_verification_is_resumable(self, tmp_path: Path) -> None:
        """The very first boundary: a crash inside `resolve_backtest_
        inputs` itself, before the CREATED -> SOURCES_VERIFIED transition
        -- the manifest exists (created by `_run_locked` before
        `_execute_pipeline` is ever entered) but has done zero further
        work."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.runner as runner_module
        real_resolve_backtest_inputs = runner_module.resolve_backtest_inputs

        def crashing_resolve(*args, **kwargs):
            raise ExperimentLockError("simulated crash before source verification")

        runner_module.resolve_backtest_inputs = crashing_resolve
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.resolve_backtest_inputs = real_resolve_backtest_inputs

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_after_crash.stage is BacktestStage.CREATED
        assert manifest_after_crash.completed_outer_fold_indices == ()

        outcome_resumed = runner.resume(backtest_id)
        assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)
        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]

    def test_crash_during_stitched_equity_build_after_verified_is_resumable(self, tmp_path: Path) -> None:
        """Both outer folds fully complete (manifest reaches VERIFIED),
        THEN the crash hits -- inside `build_stitched_walk_forward_equity`
        itself, before that artifact is ever written and before the
        aggregate report/COMPLETED transition."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.runner as runner_module
        real_build_stitched = runner_module.build_stitched_walk_forward_equity

        def crashing_build_stitched(*args, **kwargs):
            raise ExperimentLockError("simulated crash during stitched-equity build")

        runner_module.build_stitched_walk_forward_equity = crashing_build_stitched
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.build_stitched_walk_forward_equity = real_build_stitched

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_after_crash.stage is BacktestStage.VERIFIED
        assert manifest_after_crash.completed_outer_fold_indices == (0, 1)
        assert manifest_after_crash.stitched_equity_reference is None

        outcome_resumed = runner.resume(backtest_id)
        assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.stitched_equity_reference is not None
        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]
        assert any(i.code == "stitched_equity_reproduces" for i in report.infos)

    def test_crash_during_final_report_build_is_resumable(self, tmp_path: Path) -> None:
        """The stitched-equity artifact is already durably written; the
        crash hits inside `build_backtest_report_json` itself, before the
        `BACKTEST_REPORT` artifact exists and before COMPLETED."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.reporting as reporting_module
        real_build_report = reporting_module.build_backtest_report_json

        def crashing_build_report(*args, **kwargs):
            raise ExperimentLockError("simulated crash during aggregate report build")

        reporting_module.build_backtest_report_json = crashing_build_report
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            reporting_module.build_backtest_report_json = real_build_report

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_after_crash.stage is BacktestStage.VERIFIED
        assert manifest_after_crash.aggregate_report_reference is None

        outcome_resumed = runner.resume(backtest_id)
        assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.aggregate_report_reference is not None
        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]

    def test_crash_immediately_before_completed_transition_is_resumable(self, tmp_path: Path) -> None:
        """Both folds are already durably written; the crash hits at the
        very last step -- the single, atomic COMPLETED transition call
        that ALSO records `stitched_equity_reference`/`aggregate_report_
        reference` (distinguished from the earlier per-fold transition-
        burst crash by filtering on `new_stage=COMPLETED` specifically,
        not a raw call count). Because that one transition call never
        completes, none of the three fields it would have set land on the
        persisted manifest -- proving resume must rebuild (not merely
        reference) the stitched-equity/report artifacts too, exactly as
        `test_crash_during_stitched_equity_build_after_verified_is_
        resumable`/`test_crash_during_final_report_build_is_resumable`
        prove for their own, earlier crash points."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        real_transition = BacktestManifestStore.transition

        def crashing_transition(self, *args, **kwargs):
            if kwargs.get("new_stage") is BacktestStage.COMPLETED:
                raise ExperimentLockError("simulated crash immediately before the COMPLETED transition")
            return real_transition(self, *args, **kwargs)

        import quant_platform.backtesting.manifests as manifests_module

        manifests_module.BacktestManifestStore.transition = crashing_transition
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            manifests_module.BacktestManifestStore.transition = real_transition

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_after_crash.stage is BacktestStage.VERIFIED
        assert manifest_after_crash.completed_outer_fold_indices == (0, 1)
        # The crashed transition call never completed, so NONE of the
        # fields it would have set (stage, stitched_equity_reference,
        # aggregate_report_reference) landed on the persisted manifest --
        # even though the underlying artifact BYTES were already written
        # to the content store beforehand.
        assert manifest_after_crash.aggregate_report_reference is None
        assert manifest_after_crash.stitched_equity_reference is None

        outcome_resumed = runner.resume(backtest_id)
        assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.aggregate_report_reference is not None
        assert outcome_resumed.manifest.stitched_equity_reference is not None
        assert outcome_resumed.manifest.completed_at is not None
        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]

    def test_resume_does_not_trust_a_stale_completed_claim_for_a_corrupted_fold(self, tmp_path: Path) -> None:
        """`resume()` must never silently TRUST a fold the manifest claims
        is complete without re-verifying its artifact. Fold 0 genuinely
        completes; its OWN persisted `OUTER_FOLD_BACKTEST_RESULT` artifact
        is then bit-flipped on disk (mirroring `TestCorruptionAndTampering`'s
        technique) BEFORE a crash on fold 1; resume must then treat fold 0
        as `needs_rerun` (never trust the stale claim) rather than
        propagating the corrupted reference into the final manifest.

        `OuterFoldBacktestResult.evaluated_at` is a fresh wall-clock
        timestamp on every call, so genuinely re-running fold 0 produces a
        DIFFERENT (correct) content hash, not a dedup no-op against the
        corrupted bytes at the old hash -- the corrupted artifact is
        simply orphaned (no longer referenced by anything), and the fold's
        reference is updated to point at the freshly-written, valid one.
        This is what a caller actually observes and what matters for
        safety: the corrupted artifact never ends up referenced by a
        COMPLETED manifest, and `verify_backtest` reports clean afterward."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.runner as runner_module
        real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest
        call_count = {"n": 0}

        def crash_on_second_fold(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash before fold 1 completes")
            return real_run_outer_fold_backtest(**kwargs)

        runner_module.run_outer_fold_backtest = crash_on_second_fold
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_after_crash = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_after_crash.completed_outer_fold_indices == (0,)

        fold0_ref = manifest_after_crash.outer_fold_result_references[0]
        content_path = ml_artifacts_root / "content" / fold0_ref.content_hash[:2] / fold0_ref.content_hash
        original = content_path.read_bytes()
        content_path.write_bytes(original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01"))

        artifact_store = MLArtifactStore(ml_artifacts_root)
        with pytest.raises(ArtifactCorruptionError):
            artifact_store.read_artifact(fold0_ref.content_hash)

        outcome_resumed = runner.resume(backtest_id)
        assert outcome_resumed.manifest.stage is BacktestStage.COMPLETED, outcome_resumed.manifest.failure_summary
        assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)
        # Fold 0 was genuinely RE-RUN (not trusted): its reference in the
        # final manifest now points to a FRESH artifact, distinct from the
        # corrupted one left behind at the old hash.
        assert outcome_resumed.manifest.outer_fold_result_references[0].content_hash != fold0_ref.content_hash
        artifact_store.read_artifact(outcome_resumed.manifest.outer_fold_result_references[0].content_hash)  # does not raise

        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]


class TestUninterruptedVsResumedArtifactEquality:
    """Milestone 5.1, Section 6: promotes the uninterrupted-vs-resumed
    normalized artifact equality check into the PERMANENT automated
    suite. Runs the SAME `BacktestSpec` (same `backtest_id`) twice, in
    two entirely separate temp roots: once straight through, once with an
    injected mid-fold crash-then-resume. Every DETERMINISTIC financial
    payload artifact's content_hash must match EXACTLY between the two;
    only OPERATIONAL/wall-clock provenance fields on the top-level
    manifest/report (timestamps, `resume_count`) are permitted to differ."""

    def test_uninterrupted_and_resumed_runs_produce_byte_identical_financial_artifacts(self, tmp_path: Path) -> None:
        uninterrupted_spec, uninterrupted_runner, uninterrupted_root, *_ = _build_ready_setup(tmp_path / "uninterrupted", seed=123)
        uninterrupted_outcome = uninterrupted_runner.run(uninterrupted_spec)
        assert uninterrupted_outcome.manifest.stage is BacktestStage.COMPLETED

        resumed_spec, resumed_runner, resumed_root, *_ = _build_ready_setup(tmp_path / "resumed", seed=123)
        assert compute_backtest_identity(resumed_spec).backtest_id == compute_backtest_identity(uninterrupted_spec).backtest_id

        import quant_platform.backtesting.runner as runner_module
        real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest

        def crashing_run_outer_fold(*, artifact_store, **kwargs):
            proxy = _CountingArtifactStoreProxy(artifact_store, crash_after_n_writes=3)
            return real_run_outer_fold_backtest(artifact_store=proxy, **kwargs)

        runner_module.run_outer_fold_backtest = crashing_run_outer_fold
        try:
            with pytest.raises(ExperimentLockError):
                resumed_runner.run(resumed_spec)
        finally:
            runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest
        backtest_id = compute_backtest_identity(resumed_spec).backtest_id
        resumed_outcome = resumed_runner.resume(backtest_id)
        assert resumed_outcome.manifest.stage is BacktestStage.COMPLETED, resumed_outcome.manifest.failure_summary
        assert resumed_outcome.manifest.resume_count >= 1

        uninterrupted_store = MLArtifactStore(uninterrupted_root)
        resumed_store = MLArtifactStore(resumed_root)

        # Every per-fold sub-artifact: bit-for-bit identical content hash.
        # `evaluated_at` (fresh wall-clock every run) and each embedded
        # `ArtifactReference.created_at` (when THIS store first saw those
        # bytes -- the two runs use entirely separate stores/roots) are
        # the only fields legitimately allowed to differ, so this compares
        # `content_hash` per reference plus every content-derived scalar
        # field directly, rather than the whole `to_json_dict()`.
        for outer_fold_index in (0, 1):
            u_ref = uninterrupted_outcome.manifest.outer_fold_result_references[outer_fold_index]
            r_ref = resumed_outcome.manifest.outer_fold_result_references[outer_fold_index]
            u_result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(uninterrupted_store.read_artifact(u_ref.content_hash).decode("utf-8")))
            r_result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(resumed_store.read_artifact(r_ref.content_hash).decode("utf-8")))
            assert u_result.outer_test_row_count == r_result.outer_test_row_count
            assert u_result.closed_trade_count == r_result.closed_trade_count
            assert u_result.meets_minimum_trade_threshold == r_result.meets_minimum_trade_threshold
            assert dict(u_result.financial_metrics) == dict(r_result.financial_metrics), f"outer fold {outer_fold_index}: financial_metrics diverge"
            assert dict(u_result.skipped_metrics) == dict(r_result.skipped_metrics)

            for sub_ref_name in (
                "signal_set_reference", "trade_set_reference", "bar_return_timeline_reference", "equity_curve_reference", "gross_drawdown_reference",
                "net_drawdown_reference", "benchmark_report_reference", "cost_sensitivity_report_reference", "bucket_analysis_report_reference",
            ):
                u_sub_ref = getattr(u_result, sub_ref_name)
                r_sub_ref = getattr(r_result, sub_ref_name)
                assert u_sub_ref.content_hash == r_sub_ref.content_hash, f"outer fold {outer_fold_index}: {sub_ref_name} content_hash diverges"

        # The run-level stitched-equity artifact: also bit-for-bit identical.
        assert uninterrupted_outcome.manifest.stitched_equity_reference is not None
        assert resumed_outcome.manifest.stitched_equity_reference is not None
        assert uninterrupted_outcome.manifest.stitched_equity_reference.content_hash == resumed_outcome.manifest.stitched_equity_reference.content_hash

        # The aggregate report: byte-identical after normalizing wall-clock/operational fields.
        u_report_ref = uninterrupted_outcome.manifest.aggregate_report_reference
        r_report_ref = resumed_outcome.manifest.aggregate_report_reference
        assert u_report_ref is not None and r_report_ref is not None
        u_report = json.loads(uninterrupted_store.read_artifact(u_report_ref.content_hash).decode("utf-8"))
        r_report = json.loads(resumed_store.read_artifact(r_report_ref.content_hash).decode("utf-8"))
        for report in (u_report, r_report):
            for operational_field in ("created_at", "updated_at", "completed_at", "resume_count", "backtest_id"):
                del report[operational_field]
            for entry in report["outer_fold_results"]:
                entry["evaluated_at"] = ""  # fresh wall-clock every run -- normalized out
        assert u_report == r_report, "uninterrupted vs resumed aggregate BacktestReport diverges beyond declared operational/wall-clock fields"


class TestCorruptionAndTampering:
    """Section 34/41: corrupted/tampered artifacts must be detected and
    reported (or rejected outright), never silently trusted."""

    def test_bitflipped_artifact_content_is_detected_on_read(self, tmp_path: Path) -> None:
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED

        ref = outcome.manifest.outer_fold_result_references[0]
        artifact_store = MLArtifactStore(ml_artifacts_root)
        content_path = ml_artifacts_root / "content" / ref.content_hash[:2] / ref.content_hash
        assert content_path.is_file()
        original = content_path.read_bytes()
        tampered = original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01")
        content_path.write_bytes(tampered)

        with pytest.raises(ArtifactCorruptionError):
            artifact_store.read_artifact(ref.content_hash)

    def test_verify_backtest_fails_closed_on_corrupted_outer_fold_artifact(self, tmp_path: Path) -> None:
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED

        ref = outcome.manifest.outer_fold_result_references[0]
        content_path = ml_artifacts_root / "content" / ref.content_hash[:2] / ref.content_hash
        original = content_path.read_bytes()
        content_path.write_bytes(original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01"))

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_unverifiable" for i in report.criticals)

    def test_verify_backtest_fails_closed_on_financial_metrics_semantic_tampering(self, tmp_path: Path) -> None:
        """The strongest tampering case: a NEW, byte-VALID
        `OuterFoldBacktestResult` artifact (parses cleanly, correct
        schema, correct hash for ITS OWN new content) whose
        `financial_metrics` have been altered -- content-hash validity
        alone is insufficient; `verify_backtest`'s recomputation check
        must still catch that the persisted metrics no longer reproduce
        from the persisted `TradeSet`."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        tampered_metrics = dict(result.financial_metrics)
        tampered_metrics["total_net_return"] = float(tampered_metrics.get("total_net_return", 0.0)) + 999.0
        tampered = OuterFoldBacktestResult.from_json_dict({**result.to_json_dict(), "financial_metrics": tampered_metrics})
        tampered_ref = artifact_store.write_artifact(canonical_json_bytes(tampered.to_json_dict()), category=fold_ref.category)

        manifest_path = ml_artifacts_root / "backtests" / outcome.manifest.backtest_id / "backtest_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        assert any(i.code == "financial_metrics_do_not_reproduce" for i in report.criticals)

    def test_verify_backtest_rejects_a_timeline_and_metrics_tampered_consistently_with_each_other(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 3's explicit threat model: an attacker
        who fabricates a `BarReturnTimeline` (inflating every point's
        cumulative equity by a constant factor -- still passing every one
        of `BarReturnPoint`/`BarReturnTimeline.__post_init__`'s OWN
        structural checks, since none of them recompute the cumulative-
        equity recursion from `gross_return`/`net_return`), recomputes
        `financial_metrics` FROM that same fabricated timeline (so the two
        artifacts are perfectly self-consistent with each other), and
        writes both under fresh, individually-valid content hashes. This
        defeats `_verify_fold_financials_reproduce` (which only checks
        that persisted metrics reproduce from the persisted timeline --
        and here they genuinely do) -- but `_verify_raw_source_
        reconstruction` must still reject it, because raw-source
        reconstruction never reads the persisted (tampered) timeline as an
        input; it rebuilds an entirely independent timeline from the real
        market bars and real predictions, which will not match the
        inflated one."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        timeline_raw = parse_json_strict(artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash).decode("utf-8"))
        timeline = BarReturnTimeline.from_json_dict(timeline_raw)

        inflation_factor = 1.5
        tampered_points = tuple(
            type(p)(**{
                **p.to_json_dict(),
                "cumulative_gross_equity": p.cumulative_gross_equity * inflation_factor,
                "cumulative_net_equity": p.cumulative_net_equity * inflation_factor,
                "peak_equity": p.peak_equity * inflation_factor,
            })
            for p in timeline.points
        )
        tampered_timeline = BarReturnTimeline(
            schema_version=timeline.schema_version, outer_fold_index=timeline.outer_fold_index, return_basis=timeline.return_basis,
            compounded=timeline.compounded, fold_start_position=timeline.fold_start_position, fold_end_position=timeline.fold_end_position,
            points=tuple(tampered_points),
        )
        tampered_timeline_ref = artifact_store.write_artifact(
            canonical_json_bytes(tampered_timeline.to_json_dict()), category=result.bar_return_timeline_reference.category,
        )

        signal_raw = parse_json_strict(artifact_store.read_artifact(result.signal_set_reference.content_hash).decode("utf-8"))
        signals = SignalSet.from_json_dict(signal_raw)
        tampered_equity_curve = bar_return_timeline_to_equity_curve(tampered_timeline)
        tampered_gross_dd = compute_drawdown_report(tampered_equity_curve, equity_basis="gross")
        tampered_net_dd = compute_drawdown_report(tampered_equity_curve, equity_basis="net")
        trade_set_raw = parse_json_strict(artifact_store.read_artifact(result.trade_set_reference.content_hash).decode("utf-8"))
        trade_set = TradeSet.from_json_dict(trade_set_raw)
        tampered_metrics_report = compute_financial_metrics(
            trades=trade_set, equity_curve=tampered_equity_curve, gross_drawdown=tampered_gross_dd, net_drawdown=tampered_net_dd,
            signals=signals, bar_timeline=tampered_timeline, bar_interval=spec.bar_interval, annual_risk_free_rate=spec.annual_risk_free_rate,
            initial_notional=spec.initial_notional, exposure_cap=spec.exposure_cap,
        )

        # Sanity: this fabricated pair genuinely IS self-consistent (the
        # narrower, persisted-artifact-only check would be fooled).
        assert tampered_metrics_report.values["total_net_return"] != pytest.approx(result.financial_metrics["total_net_return"])

        tampered_result = OuterFoldBacktestResult.from_json_dict({
            **result.to_json_dict(), "bar_return_timeline_reference": tampered_timeline_ref.to_json_dict(),
            "financial_metrics": dict(tampered_metrics_report.values), "skipped_metrics": dict(tampered_metrics_report.skipped),
        })
        tampered_result_ref = artifact_store.write_artifact(canonical_json_bytes(tampered_result.to_json_dict()), category=fold_ref.category)

        manifest_path = ml_artifacts_root / "backtests" / outcome.manifest.backtest_id / "backtest_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_result_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        # The narrower, persisted-artifact-only check is fooled (proving
        # this tamper is a genuine test of raw-source reconstruction, not
        # a trivially-caught inconsistency).
        assert any(i.code == "fold_financials_reproduce" for i in report.infos)
        assert not any(i.code == "financial_metrics_do_not_reproduce" for i in report.criticals)
        # Raw-source reconstruction still catches it.
        assert any(i.code in ("raw_source_timeline_mismatch", "raw_source_artifact_mismatch", "raw_source_financial_metrics_mismatch") for i in report.criticals)

    def test_verify_backtest_rejects_a_benchmark_report_tamper_with_no_structural_self_check(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 4: `BenchmarkReport` (unlike
        `BarReturnTimeline`) has essentially no internal cross-field
        structural validation of its own -- a tampered value decodes
        cleanly. Only raw-source reconstruction's independent rebuild +
        content-hash comparison catches this, not decode-time validation
        and not the narrower persisted-artifact-only check."""
        from quant_platform.backtesting.runner import BenchmarkReport, BenchmarkResult

        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        original_bench_raw = parse_json_strict(artifact_store.read_artifact(result.benchmark_report_reference.content_hash).decode("utf-8"))
        original = BenchmarkReport.from_json_dict(original_bench_raw)
        tampered_benchmarks = tuple(
            BenchmarkResult(name=b.name, description=b.description, gross_return=b.gross_return + 999.0, net_return=b.net_return + 999.0)
            if b.name == "always_flat" else b
            for b in original.benchmarks
        )
        tampered = BenchmarkReport(schema_version=original.schema_version, outer_fold_index=original.outer_fold_index, benchmarks=tampered_benchmarks)
        tampered_ref = artifact_store.write_artifact(canonical_json_bytes(tampered.to_json_dict()), category=result.benchmark_report_reference.category)

        tampered_result = OuterFoldBacktestResult.from_json_dict({**result.to_json_dict(), "benchmark_report_reference": tampered_ref.to_json_dict()})
        tampered_result_ref = artifact_store.write_artifact(canonical_json_bytes(tampered_result.to_json_dict()), category=fold_ref.category)
        manifest_path = ml_artifacts_root / "backtests" / outcome.manifest.backtest_id / "backtest_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_result_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        assert any(i.code == "raw_source_artifact_mismatch" and i.context.get("artifact") == "benchmark_report" for i in report.criticals)

    def test_shared_production_and_verification_logic_bug_is_not_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 4's explicit honesty
        requirement: raw-source reconstruction deliberately reuses the
        EXACT SAME production code path (`runner.recompute_outer_fold_
        backtest_artifacts`) the original run used, rather than a
        separately re-implemented computation (see that function's own
        docstring: "never a re-implemented, drift-prone parallel
        computation"). This buys strong protection against a tampered
        ARTIFACT, but means a SYSTEMATIC bug in code BOTH the original run
        and the verifier's rebuild call is, by construction, reproduced
        IDENTICALLY by both -- and therefore CANNOT be caught by this
        verifier. This test proves that limit concretely: a small,
        plausible bug (a silent 1% undervaluation of every LONG position,
        not an obviously-broken change) is injected into `_local_value_
        multiplier` -- shared by both the original timeline build and the
        verifier's independent rebuild -- for the DURATION of both the run
        and the verify call. Verification must still report is_ready=True,
        despite the systematic error, because both sides compute the
        SAME wrong answer. Raw-source reconstruction is therefore
        correctly classified as protecting against ARTIFACT tampering,
        not as an independent proof of the underlying FORMULA's
        correctness."""
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.timeline as timeline_module
        real_local_value_multiplier = timeline_module._local_value_multiplier

        def buggy_local_value_multiplier(direction, entry_price, price, policy):
            result = real_local_value_multiplier(direction, entry_price, price, policy)
            return result * 0.99 if direction.value == "long" else result

        monkeypatch.setattr(timeline_module, "_local_value_multiplier", buggy_local_value_multiplier)

        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        # THE POINT: verification does NOT catch this -- it reruns the
        # SAME buggy shared function to build its own comparison values.
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]
        assert any(i.code == "raw_source_reconstruction_verified" for i in report.infos)

    def test_verify_backtest_rejects_a_stitched_equity_tamper(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 4: `manifest.stitched_equity_
        reference` pointed at a fabricated (but internally valid,
        self-consistent) `StitchedWalkForwardEquity` artifact -- caught
        because raw-source reconstruction rebuilds the stitched series
        from the RAW-SOURCE-RECONSTRUCTED per-fold timelines (never the
        persisted stitched artifact, and never the persisted per-fold
        timelines either), then compares by content hash."""
        from quant_platform.backtesting.stitching import StitchedWalkForwardEquity

        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        assert outcome.manifest.stitched_equity_reference is not None
        artifact_store = MLArtifactStore(ml_artifacts_root)

        original_raw = parse_json_strict(artifact_store.read_artifact(outcome.manifest.stitched_equity_reference.content_hash).decode("utf-8"))
        original = StitchedWalkForwardEquity.from_json_dict(original_raw)
        tampered_points = tuple(
            type(p)(**{**p.to_json_dict(), "stitched_gross_equity": p.stitched_gross_equity * 1.5, "stitched_net_equity": p.stitched_net_equity * 1.5})
            for p in original.points
        )
        tampered = StitchedWalkForwardEquity(
            schema_version=original.schema_version, backtest_id=original.backtest_id,
            compounded=original.compounded, fold_boundaries=original.fold_boundaries, points=tampered_points,
        )
        tampered_ref = artifact_store.write_artifact(canonical_json_bytes(tampered.to_json_dict()), category=outcome.manifest.stitched_equity_reference.category)

        manifest_path = ml_artifacts_root / "backtests" / outcome.manifest.backtest_id / "backtest_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["stitched_equity_reference"] = tampered_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        assert any(i.code in ("raw_source_stitched_equity_mismatch", "stitched_equity_does_not_reproduce") for i in report.criticals)

    def test_verify_backtest_rejects_backtest_id_mismatch_tamper(self, tmp_path: Path) -> None:
        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        tampered = OuterFoldBacktestResult.from_json_dict({**result.to_json_dict(), "backtest_id": "f" * 64})
        tampered_ref = artifact_store.write_artifact(canonical_json_bytes(tampered.to_json_dict()), category=fold_ref.category)

        manifest_path = ml_artifacts_root / "backtests" / outcome.manifest.backtest_id / "backtest_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(
        outcome.manifest.backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_key_mismatch" for i in report.criticals)

    def test_tampered_trade_record_entry_timestamp_is_rejected_at_decode(self, tmp_path: Path) -> None:
        """`trade_id` is a pure function of IDENTITY fields (source
        calibration, outer fold, signal sample position, direction, entry/
        exit timestamps -- see `trades.compute_trade_id`), deliberately
        NOT of financial-outcome fields like `net_return` (a trade's
        identity and its realized return are different concerns). Editing
        an identity field without recomputing `trade_id` must fail at
        DECODE time (`TradeRecord.__post_init__`'s own recomputation
        check)."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        trade_set_raw = json.loads(artifact_store.read_artifact(result.trade_set_reference.content_hash).decode("utf-8"))
        assert trade_set_raw["trades"], "expected at least one trade to tamper with"
        trade_set_raw["trades"][0]["signal_sample_position"] += 1

        from quant_platform.core.exceptions import TradeConstructionError

        with pytest.raises(TradeConstructionError, match="does not match the recomputed deterministic id"):
            TradeRecord.from_json_dict(trade_set_raw["trades"][0])

    def test_tampered_entry_timestamp_before_decision_timestamp_is_rejected_at_decode(self, tmp_path: Path) -> None:
        """Milestone 5.1, Section 5's semantic-verification tampering
        case: a real, produced `TradeRecord` with its `entry_timestamp`
        rewound to one hour before its own `decision_timestamp` (and
        `trade_id` correctly recomputed to match, so this is NOT caught by
        the identity-mismatch check above) -- an IMPOSSIBLE fill that
        `TradeRecord.__post_init__`'s decision-timestamp-policy invariant
        must reject regardless of trade_id validity."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        trade_set_raw = json.loads(artifact_store.read_artifact(result.trade_set_reference.content_hash).decode("utf-8"))
        assert trade_set_raw["trades"], "expected at least one trade to tamper with"
        raw_trade = dict(trade_set_raw["trades"][0])

        import pandas as pd

        from quant_platform.backtesting.trades import compute_trade_id

        impossible_entry = (pd.Timestamp(raw_trade["decision_timestamp"]) - pd.Timedelta(hours=1)).isoformat()
        raw_trade["entry_timestamp"] = impossible_entry
        raw_trade["trade_id"] = compute_trade_id(
            source_calibration_id=raw_trade["source_calibration_id"], outer_fold_index=raw_trade["outer_fold_index"],
            signal_sample_position=raw_trade["signal_sample_position"], direction=PositionDirection(raw_trade["direction"]),
            entry_timestamp=impossible_entry, exit_timestamp=raw_trade["exit_timestamp"],
        )

        from quant_platform.core.exceptions import TradeConstructionError

        with pytest.raises(TradeConstructionError, match="IMPOSSIBLE fill"):
            TradeRecord.from_json_dict(raw_trade)

    def test_tampered_cost_breakdown_total_is_rejected_at_decode(self, tmp_path: Path) -> None:
        """`CostBreakdown.total_cost` is a computed property, but the
        PERSISTED `total_cost` field is cross-checked against the sum of
        its own itemized components on every decode (`CostBreakdown.
        from_json_dict`) -- editing one itemized cost component without
        updating the persisted total must fail at DECODE time."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        trade_set_raw = json.loads(artifact_store.read_artifact(result.trade_set_reference.content_hash).decode("utf-8"))
        assert trade_set_raw["trades"], "expected at least one trade to tamper with"
        trade_set_raw["trades"][0]["cost_breakdown"]["entry_spread_cost"] += 0.01

        from quant_platform.backtesting.costs import CostBreakdown
        from quant_platform.core.exceptions import CostModelError

        with pytest.raises(CostModelError, match="does not match the sum of its own itemized components"):
            CostBreakdown.from_json_dict(trade_set_raw["trades"][0]["cost_breakdown"])

    def test_read_artifact_on_a_nonexistent_hash_fails_closed(self, tmp_path: Path) -> None:
        _, _, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        artifact_store = MLArtifactStore(ml_artifacts_root)
        with pytest.raises(ArtifactNotFoundError):
            artifact_store.read_artifact("f" * 64)


class TestConcurrency:
    """Deterministic, Windows-compatible concurrency proof -- never
    `time.sleep`-based synchronization. Both threads are released from a
    `threading.Barrier` at the precise `os.link` call `DatasetLock.
    acquire()` uses to publish its lock file. Proves: exactly one active
    owner; the loser fails FAST (never hangs); no double publication."""

    def test_two_simultaneous_run_attempts_for_the_same_backtest_exactly_one_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os
        import threading

        spec, _runner_unused, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[tuple[str, object]] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            runner = _new_backtest_runner(ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root)
            try:
                outcome = runner.run(spec)
                with results_lock:
                    results.append(("completed", outcome))
            except ExperimentLockError as exc:
                with results_lock:
                    results.append(("rejected", exc))

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        monkeypatch.setattr(os, "link", real_link)
        assert not any(t.is_alive() for t in threads), "a losing attempt hung instead of failing fast"

        outcomes = sorted(r[0] for r in results)
        assert outcomes == ["completed", "rejected"], f"expected exactly one winner and one fast-failing loser, got {outcomes}"

        backtest_id = compute_backtest_identity(spec).backtest_id
        reloaded = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert reloaded.stage is BacktestStage.COMPLETED

        report = _verify(
        backtest_id, ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root,
    )
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]

    def test_two_simultaneous_resume_attempts_on_the_same_partially_complete_backtest_exactly_one_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scenario 2 ("same fold, two workers"): this platform has no
        per-fold lock -- one outer fold's whole `run_outer_fold_backtest`
        call is always processed inside the SAME `.backtest_run.lock`-held
        run, so "two workers racing the same fold" reduces to two workers
        racing to RESUME the same partially-complete backtest and both
        reach for fold 1. Distinguishes itself from the test above (which
        exercises `.run()` from a fresh `CREATED` manifest) by exercising
        `.resume()` against a manifest already sitting mid-pipeline
        (fold 0 done, fold 1 pending)."""
        import os
        import threading

        spec, runner, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.runner as runner_module
        real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest
        call_count = {"n": 0}

        def crash_on_second_fold(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ExperimentLockError("simulated crash before fold 1 completes")
            return real_run_outer_fold_backtest(**kwargs)

        runner_module.run_outer_fold_backtest = crash_on_second_fold
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

        backtest_id = compute_backtest_identity(spec).backtest_id
        manifest_before = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert manifest_before.completed_outer_fold_indices == (0,)

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[tuple[str, object]] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            worker_runner = _new_backtest_runner(ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root)
            try:
                outcome = worker_runner.resume(backtest_id)
                with results_lock:
                    results.append(("completed", outcome))
            except ExperimentLockError as exc:
                with results_lock:
                    results.append(("rejected", exc))

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        monkeypatch.setattr(os, "link", real_link)
        assert not any(t.is_alive() for t in threads), "a losing resume attempt hung instead of failing fast"

        outcomes = sorted(r[0] for r in results)
        assert outcomes == ["completed", "rejected"], f"expected exactly one winner and one fast-failing loser, got {outcomes}"
        reloaded2 = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        assert reloaded2.stage is BacktestStage.COMPLETED
        assert reloaded2.completed_outer_fold_indices == (0, 1)

    def test_reader_never_observes_a_partially_published_artifact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scenario 3 ("reader during atomic artifact publication"):
        `MLArtifactStore.write_artifact` publishes content via `Path.
        replace` (an atomic rename, the same publish primitive `DatasetLock`
        itself relies on -- see `TestNoDoubleAcquisitionUnderForcedInterleaving`
        for that lower-level proof). A concurrent reader paused exactly
        before that rename completes must see EITHER "not found" (never
        published yet) or the full, correct, hash-verified content --
        never a torn/partial file."""
        import threading

        store = MLArtifactStore(tmp_path)
        data = b'{"deterministic": "financial payload"}' * 50
        expected_hash = MLArtifactStore(tmp_path).write_artifact(data, category=ArtifactCategory.BACKTEST_REPORT).content_hash
        # Reset to a fresh, empty store root for the real test.
        import shutil

        shutil.rmtree(tmp_path)
        tmp_path.mkdir()
        store = MLArtifactStore(tmp_path)

        writer_paused_before_replace = threading.Event()
        allow_writer_to_replace = threading.Event()
        real_replace = Path.replace

        def paused_replace(self: Path, target):
            if self.name.startswith(".") and self.suffix == ".tmp":
                writer_paused_before_replace.set()
                assert allow_writer_to_replace.wait(timeout=5), "test failed to release the writer in time"
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", paused_replace)

        reader_observations: list[str] = []

        def reader() -> None:
            assert writer_paused_before_replace.wait(timeout=5), "reader did not see the writer pause in time"
            try:
                store.read_artifact(expected_hash)
                reader_observations.append("read_succeeded_with_valid_content")
            except ArtifactNotFoundError:
                reader_observations.append("not_found")
            allow_writer_to_replace.set()

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        store.write_artifact(data, category=ArtifactCategory.BACKTEST_REPORT)
        reader_thread.join(timeout=10)

        assert reader_observations == ["not_found"], (
            f"reader must observe ONLY 'not found' before atomic publication completes, never a torn file: {reader_observations}"
        )
        # After publication completes, the same read succeeds cleanly.
        assert store.read_artifact(expected_hash) == data

    def test_inspection_succeeds_while_a_backtest_run_is_actively_in_progress(self, tmp_path: Path) -> None:
        """Scenario 4 ("inspect during active run"): `BacktestManifestStore.
        load`/`verify_backtest` are read-only and take no run-level lock --
        a caller inspecting at a few distinct, realistically-spaced
        moments during an in-progress run (an expected NOT-ready
        verification report while folds are still pending is a valid
        outcome; an unhandled exception is not) must always succeed.

        DISCOVERED, WINDOWS-SPECIFIC RACE (documented, not fixed here --
        lives in SHARED `core.json.write_json_atomic`/`read_json_file`,
        used by every manifest store across Milestones 4A-5.1, out of this
        backtesting-specific milestone's scope to change): a reader with
        `backtest_manifest.json` open for reading at the EXACT instant a
        writer's `Path.replace` publishes a new version can, on Windows
        specifically, cause `PermissionError` on EITHER side (POSIX
        `rename` has no such failure mode against an open reader). A
        genuinely TIGHT read-polling loop reproduces this reliably; a
        realistic caller (a human or CLI invocation checking in
        occasionally, which this test deliberately models with real,
        clock-spaced inspection points -- not a busy-spin) essentially
        never collides with the sub-millisecond rename window in
        practice. Worth a dedicated `core.json` hardening pass (e.g.
        retry-on-transient-PermissionError) outside this milestone."""
        import threading
        import time as time_module

        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        manifest_store = BacktestManifestStore(ml_artifacts_root)
        backtest_id = compute_backtest_identity(spec).backtest_id

        run_outcome: dict[str, object] = {}
        run_exception: dict[str, BaseException] = {}

        def run() -> None:
            try:
                run_outcome["outcome"] = runner.run(spec)
            except BaseException as exc:  # pragma: no cover - only on the documented, rare Windows race
                run_exception["exc"] = exc

        run_thread = threading.Thread(target=run)
        run_thread.start()
        try:
            observed_exceptions: list[BaseException] = []
            inspections = 0
            # A FEW realistically-spaced inspection points (not a tight
            # spin-loop) -- see the Windows-race note above for why.
            for _ in range(8):
                time_module.sleep(0.1)
                if not run_thread.is_alive():
                    break
                try:
                    if manifest_store.exists(backtest_id):
                        manifest_store.load(backtest_id)
                        inspections += 1
                except (ArtifactNotFoundError, ArtifactCorruptionError) as exc:  # pragma: no cover - would indicate a real defect
                    observed_exceptions.append(exc)
        finally:
            run_thread.join(timeout=30)
        assert not run_thread.is_alive(), "the background run hung"
        assert not observed_exceptions, f"inspection during an active run raised: {observed_exceptions}"
        assert "exc" not in run_exception, f"the background run itself raised: {run_exception.get('exc')!r}"
        assert isinstance(run_outcome.get("outcome"), BacktestOutcome)
        assert run_outcome["outcome"].manifest.stage is BacktestStage.COMPLETED

    def test_abandoned_run_lock_is_reclaimed(self, tmp_path: Path) -> None:
        """Scenario 5 ("abandoned lock recovery"): the underlying
        stale-lock reclaim mechanism is already thoroughly proven at the
        `DatasetLock` layer (`tests/unit/historical/test_locking.py::
        TestStaleLockRecovery`) -- this is the one backtesting-layer
        integration proof that `BacktestRunner`'s OWN `.backtest_run.lock`
        path benefits from that exact mechanism: a lock file pre-written
        as if abandoned by a long-dead process (aged well past the
        default `stale_after`) must not block a fresh `run()`."""
        spec, runner, _ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        abandoned = LockInfo(pid=999_999, hostname="long-dead-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(abandoned.to_json_dict()))

        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary

    def test_fresh_lock_from_a_now_dead_owner_is_not_prematurely_reclaimed(self, tmp_path: Path) -> None:
        """Scenario 6 ("lock owner disappearance"): distinguishes itself
        from scenario 5 above -- staleness in this platform is AGE-based
        (`stale_after`), never a liveness probe (`DatasetLock` never calls
        `os.kill(pid, 0)` or similar) -- a deliberate, conservative design
        (PID reuse makes liveness probing unreliable). A lock whose
        "owner" no longer exists but whose lock file is still YOUNG must
        therefore still be honored as held, not silently stolen."""
        spec, runner, _ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fresh_but_owner_gone = LockInfo(pid=999_999, hostname="just-died-host", acquired_at=pd.Timestamp.now(tz="UTC"))
        lock_path.write_text(json.dumps(fresh_but_owner_gone.to_json_dict()))

        with pytest.raises(ExperimentLockError):
            runner.run(spec)
        # Cleanup: remove the fake lock so it does not leak into other tests.
        lock_path.unlink(missing_ok=True)

    def test_lock_is_released_and_reacquirable_after_a_genuine_domain_failure(self, tmp_path: Path) -> None:
        """Scenario 7 ("re-acquisition after domain failure"): a genuine
        domain exception (never `ExperimentLockError` itself) must still
        release the run-level lock -- `experiment_lock`'s context-manager
        semantics already guarantee this generically (`tests/unit/ml/
        test_concurrency.py::TestHappyPath::test_body_return_value_and_
        exceptions_pass_through_untouched`); this proves it for
        `BacktestRunner._run_lock_path` specifically."""
        from quant_platform.core.exceptions import BacktestValidationError

        spec, runner, _ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

        import quant_platform.backtesting.runner as runner_module
        real_run_outer_fold_backtest = runner_module.run_outer_fold_backtest

        def always_fails(**kwargs):
            raise BacktestValidationError("simulated genuine domain failure")

        runner_module.run_outer_fold_backtest = always_fails
        try:
            with pytest.raises(BacktestValidationError):
                runner.run(spec)
        finally:
            runner_module.run_outer_fold_backtest = real_run_outer_fold_backtest

        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        assert not lock_path.exists(), "the run-level lock must be released after a genuine domain failure, not left held"

        # A fresh acquisition on the exact same path succeeds immediately.
        with experiment_lock(lock_path):
            pass

    def test_recover_backtest_lock_refuses_a_live_owner_without_force(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "live owner": a fresh lock genuinely
        held by THIS process (a real, verifiably-alive PID) must never be
        stolen, force or no force is irrelevant here since age already
        refuses it -- `force=True` alone must not bypass a fresh lock
        that a human has not separately investigated."""
        import os as os_module

        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        live = LockInfo(pid=os_module.getpid(), hostname="this-test-host", acquired_at=pd.Timestamp.now(tz="UTC"))
        lock_path.write_text(json.dumps(live.to_json_dict()))

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert diagnostics.lock_exists
        assert diagnostics.owner_pid == os_module.getpid()
        assert diagnostics.owner_pid_alive is True
        assert not diagnostics.is_stale_by_age

        with pytest.raises(BacktestResumeError, match="refusing to steal"):
            recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert lock_path.is_file(), "an un-forced recovery attempt must not remove the lock"
        lock_path.unlink()

    def test_recover_backtest_lock_refuses_a_dead_owner_that_is_not_yet_stale_without_force(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "dead owner": a lock whose owner PID
        is confirmed dead (via the best-effort liveness probe) but whose
        AGE has not yet crossed `stale_after` must STILL be refused
        without `--force` -- this platform's policy is age-based, not
        liveness-based (liveness is diagnostic-only, per the module's own
        "Windows PID liveness is not reliable" caveat), and `force=True`
        is exactly the documented escape hatch for this case."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        dead_owner_fresh_lock = LockInfo(pid=999_999, hostname="just-died-host", acquired_at=pd.Timestamp.now(tz="UTC"))
        lock_path.write_text(json.dumps(dead_owner_fresh_lock.to_json_dict()))

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert not diagnostics.is_stale_by_age
        with pytest.raises(BacktestResumeError, match="refusing to steal"):
            recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert lock_path.is_file()

        recovered = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=True)
        assert not lock_path.is_file()
        assert recovered.owner_pid == 999_999

    def test_recover_backtest_lock_reclaims_a_stale_lock_without_force(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "stale lock": a lock aged past
        `stale_after` is reclaimable WITHOUT `--force` -- the same
        age-based policy `DatasetLock.acquire()` already applies
        transparently on the next `run()`/`resume()`, now also available
        as an explicit, diagnosed, operator-triggered action."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale = LockInfo(pid=999_999, hostname="long-dead-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(stale.to_json_dict()))

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert diagnostics.is_stale_by_age
        recovered = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert recovered.is_stale_by_age
        assert not lock_path.is_file()

    def test_recover_backtest_lock_on_a_fresh_abandoned_lock_requires_force(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "fresh abandoned lock": distinct from
        the stale-lock case above -- a lock that is YOUNG (well within
        `stale_after`) must never be silently reclaimed no matter how
        "obviously abandoned" it might look to a human, without an
        explicit `--force`."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fresh = LockInfo(pid=999_999, hostname="abandoned-a-second-ago", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1))
        lock_path.write_text(json.dumps(fresh.to_json_dict()))

        with pytest.raises(BacktestResumeError):
            recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert lock_path.is_file()

    def test_recover_backtest_lock_with_a_nonexistent_backtest_id_is_a_safe_no_op(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "wrong backtest ID": an operator
        typo, or an id that genuinely has no lock and no manifest, must
        never crash or fabricate a diagnosis -- both `inspect` and
        `recover` treat "nothing there" as a trivially safe no-op."""
        _spec, _runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        bogus_backtest_id = "0" * 64

        diagnostics = inspect_backtest_lock(bogus_backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert not diagnostics.lock_exists
        assert not diagnostics.manifest_exists
        assert diagnostics.manifest_backtest_id_matches is None
        assert diagnostics.owner_pid is None

        recovered = recover_backtest_lock(bogus_backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert not recovered.lock_exists

    def test_recover_backtest_lock_on_corrupted_lock_metadata_requires_force(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "corrupted lock metadata": a lock
        file present but unparseable (matches `DatasetLock.peek()`'s own
        fail-open-to-`None` semantics) cannot be confirmed stale by age --
        `is_stale_by_age` is conservatively `False` for it, so recovery
        still requires an explicit `--force`, never silently trusting
        that "unreadable" means "safe to remove."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("{ this is not valid json at all")

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert diagnostics.lock_exists
        assert diagnostics.owner_pid is None
        assert diagnostics.lock_age_seconds is None
        assert not diagnostics.is_stale_by_age

        with pytest.raises(BacktestResumeError):
            recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert lock_path.is_file()

        recovered = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=True)
        assert not lock_path.is_file()
        assert recovered.owner_pid is None

    def test_recovery_followed_by_resume_completes_the_backtest(self, tmp_path: Path) -> None:
        """Milestone 5.2, Section 6, "recovery followed by resume": the
        end-to-end operator workflow -- an abandoned lock is recovered,
        and the run then proceeds and completes normally through the
        ORDINARY `experiment_lock` acquisition path (recovery itself
        never acquires anything, it only removes an abandoned file)."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        abandoned = LockInfo(pid=999_999, hostname="long-dead-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(abandoned.to_json_dict()))

        recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert not lock_path.is_file()

        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary

    def test_recover_backtest_lock_reports_a_mismatched_host_without_special_casing(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 6, "wrong host": this is a
        purely LOCAL-filesystem advisory lock (`historical.locking`'s own
        module docstring) -- it never compares a lock's recorded
        `hostname` against the current machine's own hostname, on either
        the inspect or recover path. A lock recorded under a different
        host is diagnosed identically to any other lock (by age alone);
        this test confirms that reporting path does not crash and
        surfaces the foreign hostname verbatim for a human to judge."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        foreign = LockInfo(pid=999_999, hostname="a-totally-different-machine.example", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(foreign.to_json_dict()))

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert diagnostics.owner_hostname == "a-totally-different-machine.example"
        assert diagnostics.is_stale_by_age  # age-based, unaffected by hostname
        recovered = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert not lock_path.is_file()
        assert recovered.owner_hostname == "a-totally-different-machine.example"

    def test_recover_backtest_lock_pid_reuse_does_not_cause_a_false_still_alive_block(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 6, "PID reuse ambiguity": the
        exact scenario the platform's age-based (never liveness-based)
        policy exists to defend against -- this test process's OWN,
        genuinely-alive PID is used as the lock's `owner_pid` (so
        `owner_pid_alive` will report `True`, indistinguishable from a
        real live owner), but the lock's `acquired_at` is old enough to
        be stale. If liveness were EVER load-bearing, this would
        incorrectly block recovery (a classic PID-reuse false positive);
        the platform must still allow recovery without `--force`, because
        only age decides."""
        import os as os_module

        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        reused_pid = LockInfo(pid=os_module.getpid(), hostname="reused-pid-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(reused_pid.to_json_dict()))

        diagnostics = inspect_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert diagnostics.owner_pid_alive is True  # genuinely alive (it's this process) -- yet...
        assert diagnostics.is_stale_by_age  # ...age still says stale, and age is what governs.
        recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)  # must NOT raise
        assert not lock_path.is_file()

    def test_recover_backtest_lock_refuses_a_genuinely_held_lock_from_an_active_writer(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 6, "recovery during an active
        writer": uses the REAL `experiment_lock` context manager to hold
        a genuinely live lock (not merely a hand-written `LockInfo` file)
        while a concurrent recovery attempt is made -- must refuse
        without `--force`, and the ACTIVE holder's own lock must remain
        intact and re-releasable normally afterward."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        with experiment_lock(lock_path):
            assert lock_path.is_file()
            with pytest.raises(BacktestResumeError, match="refusing to steal"):
                recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
            assert lock_path.is_file(), "the active writer's own lock must not have been removed"
        assert not lock_path.is_file(), "the active writer's own release must still work normally afterward"

    def test_recover_backtest_lock_command_is_idempotent_when_repeated(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 6, "repeated recovery
        command": calling recovery twice in a row (the second call
        finding nothing left to do) must never raise or behave
        differently the second time."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        lock_path = runner._run_lock_path(backtest_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale = LockInfo(pid=999_999, hostname="long-dead-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(stale.to_json_dict()))

        first = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert first.lock_exists
        second = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert not second.lock_exists
        third = recover_backtest_lock(backtest_id, ml_artifacts_root=ml_artifacts_root, force=True)
        assert not third.lock_exists

    def test_recover_backtest_lock_on_a_completed_backtest_is_a_safe_no_op_with_full_diagnostics(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 6, "recovery on a completed
        backtest": distinct from the "nonexistent backtest ID" no-op --
        here the manifest genuinely exists and reached COMPLETED (the
        lock was released normally on success), so diagnostics must
        report `manifest_exists=True`/`manifest_stage="completed"` while
        `lock_exists=False`, and recovery must be a harmless no-op."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED

        diagnostics = inspect_backtest_lock(outcome.manifest.backtest_id, ml_artifacts_root=ml_artifacts_root)
        assert not diagnostics.lock_exists
        assert diagnostics.manifest_exists
        assert diagnostics.manifest_stage == "completed"
        assert diagnostics.manifest_backtest_id_matches is True

        recovered = recover_backtest_lock(outcome.manifest.backtest_id, ml_artifacts_root=ml_artifacts_root, force=False)
        assert not recovered.lock_exists
        # A resume on the already-COMPLETED backtest remains a safe idempotent no-op afterward.
        second_outcome = runner.resume(outcome.manifest.backtest_id)
        assert second_outcome.was_idempotent_no_op

    def test_no_self_deadlock_within_one_run(self, tmp_path: Path) -> None:
        """Scenario 8 ("no self-deadlock"): `BacktestRunner.run()` holds
        `.backtest_run.lock` for the WHOLE run while internally calling
        `BacktestManifestStore.transition`/`create`, each of which
        acquires a DIFFERENT, distinct lock path (`.backtest.lock`) --
        never the SAME path twice. Every other test in this file that
        completes a full run without hanging (e.g. `test_backtest_runner_
        end_to_end`) is itself a live proof this never deadlocks; this
        test makes the "different paths" structural guarantee explicit
        and bounds one full run's wall-clock time as an additional,
        direct anti-hang check."""
        import time as time_module

        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        backtest_id = compute_backtest_identity(spec).backtest_id
        run_lock_path = runner._run_lock_path(backtest_id)
        manifest_lock_path = BacktestManifestStore(ml_artifacts_root)._lock_path(backtest_id)
        assert run_lock_path != manifest_lock_path

        started = time_module.monotonic()
        outcome = runner.run(spec)
        elapsed = time_module.monotonic() - started
        assert outcome.manifest.stage is BacktestStage.COMPLETED
        assert elapsed < 60.0, f"a full run took {elapsed:.1f}s -- unexpectedly slow, possibly contending with itself"

    def test_no_nested_lock_inversion_between_two_contending_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scenario 9 ("no nested lock inversion"): lock acquisition order
        is always `.backtest_run.lock` THEN (briefly, internally)
        `.backtest.lock` -- never the reverse in any code path. If two
        concurrent runs ever acquired these two locks in opposite orders,
        classic lock-ordering deadlock would be possible; since only ONE
        order ever occurs here, two contenders can only ever race on the
        SAME first lock (`.backtest_run.lock`) -- exactly the scenario the
        first test in this class exercises, reused here with a bounded
        join timeout as the direct anti-deadlock proof (a lock-ordering
        inversion would manifest as a hang, not a clean fast-fail)."""
        import os
        import threading

        spec, _runner_unused, ml_artifacts_root, research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        def attempt() -> None:
            worker_runner = _new_backtest_runner(ml_artifacts_root, research_manifest_store=research_manifest_store, research_store=research_store, historical_root=historical_root)
            with contextlib.suppress(ExperimentLockError):
                worker_runner.run(spec)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        monkeypatch.setattr(os, "link", real_link)
        assert not any(t.is_alive() for t in threads), "a lock-ordering inversion would manifest as a hang here"
