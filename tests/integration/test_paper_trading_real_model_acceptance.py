"""Milestone 7, Section 33: real, bounded end-to-end acceptance workflow
for the deterministic paper-trading and shadow-execution engine.

Builds a GENUINE chain: a real `logistic_regression` model fitted on a
real (synthetic-but-structured, injected-AR(1)-signal) dataset, through
execution -> calibration -> backtest -> Milestone 6's own statistical-
robustness pipeline -> an `ELIGIBLE_FOR_PAPER_TRADING` `PromotionDecision`
-- then a real `PaperTradingSpec` whose eligibility chain is independently
re-verified (never merely assumed), a real bounded JSONL replay source
(Section 32), a real fitted-model `StrategyRuntime`
(`model_strategy.ModelStrategyRuntime`, wrapping the SAME fitted model the
backtest/robustness chain already verified), a real paper session run to
COMPLETED, a real SHADOW_OBSERVATION session, independent reconciliation
and verification, and a from-scratch SECOND run proving byte-identical
determinism.

LENIENT PROMOTION GATES ARE AN INFRASTRUCTURE-TEST CONFIGURATION CHOICE,
NOT DATA TUNING: Section 33 explicitly forbids tuning the underlying DATA
to force profitability ("A losing paper session is acceptable"). This
fixture does not touch the data or the model in any way to influence its
outcome; it configures a deliberately PERMISSIVE `PromotionPolicySpec`
(every quantitative gate loosened to near-trivial bounds, matching the
exact same spirit `test_robustness_real_model_acceptance.py`'s own
loosened `StabilityThresholds` already established one layer down) so
that a genuinely bootstrap-verified, genuinely re-verified robustness run
against a SMALL, bounded synthetic sample can reach a decision kind other
than REJECTED -- these thresholds are explicitly NOT a claim about
realistic promotion criteria.

The observed net P&L in this run is a synthetic-data artifact (the
injected AR(1) signal is strong relative to noise by construction, for a
bootstrap-verifiable sample in a bounded test) and is NOT a claim of
achievable real-world trading performance -- see `reports.
DIAGNOSTIC_DISCLAIMER`, asserted verbatim below."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from tests.integration.test_backtesting_real_model_acceptance import _make_signal_ohlcv
from tests.unit.features.conftest import seed_canonical_dataset

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifestStore
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
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.runner import BacktestRunner
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
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
from quant_platform.core.exceptions import PaperTradingEligibilityError
from quant_platform.core.types import Timeframe
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.execution.results import FoldResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.state_machine import ExecutionStage
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.artifacts import MLArtifactStore
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
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
)
from quant_platform.ml.persistence import parse_json_strict
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml_cli import _paper_session_model_pin_path, _resolve_fitted_strategy_runtime
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.eligibility import (
    EligibilityVerificationEnvironment,
    verify_paper_trading_eligibility,
)
from quant_platform.paper_trading.events import create_bar_event, create_end_of_stream_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.model_strategy import ModelStrategyRuntime
from quant_platform.paper_trading.models import (
    ClockMode,
    LedgerEntryKind,
    MarketEventMode,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.orders import OrderStateEvent, resolve_order_state
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.reconciliation import reconcile_session
from quant_platform.paper_trading.replay import load_replay_events, write_replay_events
from quant_platform.paper_trading.reports import DIAGNOSTIC_DISCLAIMER, build_paper_session_report
from quant_platform.paper_trading.runner import (
    RunnerEnvironment,
    create_paper_session,
    run_paper_trading_session,
    run_shadow_session,
)
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_POSITION_POLICY,
    DEFAULT_SESSION_BOUNDARY_POLICY,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
    RiskLimitsSpec,
    compute_paper_session_spec_id,
)
from quant_platform.paper_trading.verification import verify_paper_session
from quant_platform.robustness.manifests import RobustnessManifestStore
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    PromotionDecisionKind,
    ReturnSeriesKind,
    RobustnessStage,
)
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.runner import RobustnessRunner
from quant_platform.robustness.specs import (
    DEFAULT_REGIME_DEFINITIONS,
    DEFAULT_STRESS_SCENARIOS,
    BootstrapSpec,
    PromotionGateSpec,
    PromotionPolicySpec,
    RobustnessSpec,
    StabilityThresholds,
    compute_robustness_identity,
)

_FEATURE_NAMES = ("return_log_1", "return_log_5", "ma_distance_10", "candle_body_ratio")


def _lenient_promotion_gates() -> tuple[PromotionGateSpec, ...]:
    """INFRASTRUCTURE-TEST thresholds only -- see module docstring. Every
    integrity-shaped gate (`verified_source_backtest`, `no_critical_
    verification_findings`) stays at its real, meaningful bound; every
    financial-outcome gate is loosened to near-trivial so a small, bounded
    synthetic sample can reach ELIGIBLE_FOR_PAPER_TRADING without the
    underlying data or model ever being tuned."""
    return (
        PromotionGateSpec(name="verified_source_backtest", mandatory=True, minimum_value=1.0),
        PromotionGateSpec(name="minimum_outer_folds", mandatory=True, minimum_value=2.0),
        PromotionGateSpec(name="minimum_trades", mandatory=True, minimum_value=1.0),
        PromotionGateSpec(name="minimum_effective_sample_size", mandatory=True, minimum_value=1.0),
        PromotionGateSpec(name="profitable_fold_fraction", mandatory=True, minimum_value=0.0),
        PromotionGateSpec(name="worst_fold_return", mandatory=False, minimum_value=-1.0),
        PromotionGateSpec(name="maximum_drawdown", mandatory=True, maximum_value=0.99),
        PromotionGateSpec(name="bootstrap_lower_bound_total_net_return", mandatory=True, minimum_value=-1.0),
        PromotionGateSpec(name="probability_of_loss", mandatory=True, maximum_value=1.0),
        PromotionGateSpec(name="stressed_cost_total_net_return", mandatory=True, minimum_value=-1.0),
        PromotionGateSpec(name="latency_stress_total_net_return", mandatory=False, minimum_value=-1.0),
        PromotionGateSpec(name="no_extreme_fold_concentration", mandatory=True, minimum_value=0.0),
        PromotionGateSpec(name="no_parameter_cliff", mandatory=False, minimum_value=0.0),
        PromotionGateSpec(name="no_critical_verification_findings", mandatory=True, minimum_value=1.0),
    )


@pytest.fixture(scope="module")
def eligible_chain(tmp_path_factory: pytest.TempPathFactory):
    """Builds the FULL Milestone 5/6 chain once per test module: dataset
    -> experiment -> execution -> calibration -> backtest -> robustness ->
    (hopefully) ELIGIBLE_FOR_PAPER_TRADING. Reuses the identical real
    `logistic_regression`/injected-AR(1)-signal construction `test_
    robustness_real_model_acceptance.py` already established, restricted
    to a small 4-feature subset that `model_strategy.ModelStrategyRuntime`
    can also compute incrementally from a live bar stream (Section 7 --
    the runner hands `decide()` an EMPTY context; feature computation
    happens entirely inside the `StrategyRuntime`, so train-time and
    paper-trading-time feature names must match exactly)."""
    tmp_path = tmp_path_factory.mktemp("paper_trading_acceptance")
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    ml_artifacts_root = tmp_path / "ml_artifacts"

    df = _make_signal_ohlcv(3200, seed=303)
    seed_canonical_dataset(historical_root, df)

    canonical_store = CanonicalStore(historical_root)
    historical_manifest_store = ManifestStore(historical_root)
    historical_loader = DatasetLoader(canonical_store, historical_manifest_store)
    registry = FeatureRegistry()
    register_core_technical_features(registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1, 5), ma_distance_windows=(10,)))

    research_store = ResearchDatasetStore(research_root)
    research_manifest_store = ResearchManifestStore(research_root)
    builder = ResearchDatasetBuilder(historical_loader=historical_loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store)
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=start, end=end, feature_names=_FEATURE_NAMES,
        label_definition=LabelDefinition(name="fut10", kind=LabelKind.BINARY_DIRECTION, horizon_bars=10),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 10, "embargo_bars": 10},
        preprocessing={},
    )
    dataset_manifest = builder.build(request)

    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version, content_id=dataset_manifest.content_id,
        symbol=dataset_manifest.symbol, base_timeframe=dataset_manifest.base_timeframe.value, source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_binding = FeatureBinding(
        feature_names=dataset_manifest.feature_names, feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition), fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    experiment_spec = ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding,
        label_binding=LabelBinding(name="fut10", kind=LabelKind.BINARY_DIRECTION.value, horizon_bars=10, label_type=LabelType.BINARY),
        split_binding=SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 3, "test_size": 280, "purge_bars": 10, "embargo_bars": 10}),
        preprocessing_binding=preprocessing_binding, model_name="logistic_regression", model_version="1",
        hyperparameters=ModelHyperparameters(values={"C": 1.0, "max_iter": 500}), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=23), code_revision_binding=CodeRevisionBinding(revision="e" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )
    model_registry = mz.register_default_models()
    preparer = ExperimentPreparer(ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store)
    experiment_manifest = preparer.prepare(experiment_spec)
    assert experiment_manifest.status is ExperimentStatus.READY, experiment_manifest.failure_summary
    experiment_id = experiment_manifest.identity.experiment_id

    execution_manifest_store = ExecutionManifestStore(ml_artifacts_root)
    execution_runner = ExecutionRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, additional_serializers=mz.default_serializer_registry(),
    )
    execution_runner.run(experiment_id)
    assert execution_manifest_store.load(experiment_id).stage is ExecutionStage.COMPLETED

    model_definition = model_registry.get("logistic_regression", "1")
    calibration_spec = CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0, source_experiment_id=experiment_id,
        base_model_definition_identity=model_definition.fingerprint(), dataset_content_id=dataset_manifest.content_id,
        split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        calibration_method_candidates=(CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=20, minimum_samples_per_class=5,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2, embargo_bars=5),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.SYMMETRIC_BAND, band_half_width=0.05),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy", "margin", "bin_support"), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),), seed=101, determinism_policy=DeterminismPolicy.STRICT,
    )
    calibration_manifest_store = CalibrationManifestStore(ml_artifacts_root)
    experiment_manifest_store = ExperimentManifestStore(ml_artifacts_root)
    calibration_runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=experiment_manifest_store,
        additional_serializers=mz.default_serializer_registry(),
    )
    calibration_outcome = calibration_runner.run(calibration_spec)
    assert calibration_outcome.manifest.stage is CalibrationStage.COMPLETED, calibration_outcome.manifest.failure_summary
    calibration_id = calibration_outcome.manifest.calibration_id

    dataset_loader = DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root))
    backtest_manifest_store = BacktestManifestStore(ml_artifacts_root)
    backtest_event_store = BacktestEventStore(ml_artifacts_root)
    backtest_runner = BacktestRunner(
        ml_artifacts_root=ml_artifacts_root, calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
        execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        dataset_loader=dataset_loader,
    )
    backtest_spec = BacktestSpec(
        schema_version=1, source_calibration_id=calibration_id, source_experiment_id=experiment_id, source_execution_id=experiment_id,
        dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        instrument_identity=dataset_manifest.symbol, market_timezone="UTC", bar_interval=dataset_manifest.base_timeframe,
        decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.ABSTENTION_AWARE), position_mode=PositionMode.LONG_SHORT,
        entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=10, final_trade_policy=FinalTradePolicyKind.FORCE_CLOSE_AT_FINAL_PRICE),
        overlap_policy=OverlapPolicyKind.CLOSE_AND_REVERSE, price_basis=PriceBasisKind.CLOSE,
        spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=3.0),
        commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=1.0),
        slippage_spec=SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
        return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.COMPOUNDED,
        initial_notional=10000.0, determinism_policy=DeterminismPolicy.STRICT, respect_calibration_abstention=True,
        annual_risk_free_rate=0.02, seed=7,
    )
    backtest_outcome = backtest_runner.run(backtest_spec)
    assert backtest_outcome.manifest.stage is BacktestStage.COMPLETED, backtest_outcome.manifest.failure_summary
    backtest_id = backtest_outcome.manifest.backtest_id

    robustness_spec = RobustnessSpec(
        schema_version=1, source_backtest_id=backtest_id, dataset_content_id=backtest_spec.dataset_content_id,
        split_plan_fingerprint=backtest_spec.split_plan_fingerprint, instrument_identity=backtest_spec.instrument_identity,
        bar_interval=backtest_spec.bar_interval, return_series_kind=ReturnSeriesKind.STITCHED_BAR_NET,
        bootstrap_spec=BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=200, confidence_level=0.9, block_length=5),
        seed=11, multiple_testing_correction=MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG, strategy_family_id=None,
        minimum_fold_count=2, minimum_trade_count=1, minimum_effective_sample_size=1,
        stability_thresholds=StabilityThresholds(
            minimum_profitable_fold_fraction=0.01, maximum_single_fold_profit_concentration=0.99,
            maximum_single_trade_profit_concentration=0.99, maximum_single_direction_profit_concentration=0.99,
        ),
        stress_scenarios=DEFAULT_STRESS_SCENARIOS, regime_definitions=DEFAULT_REGIME_DEFINITIONS, promotion_policy=PromotionPolicySpec(gates=_lenient_promotion_gates()),
    )
    robustness_manifest_store = RobustnessManifestStore(ml_artifacts_root)
    artifact_store = MLArtifactStore(ml_artifacts_root)
    robustness_runner = RobustnessRunner(
        ml_artifacts_root=ml_artifacts_root, backtest_manifest_store=backtest_manifest_store, backtest_event_store=backtest_event_store,
        calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store, execution_manifest_store=execution_manifest_store,
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )
    identity = compute_robustness_identity(robustness_spec)
    robustness_outcome = robustness_runner.run(robustness_spec)
    assert robustness_outcome.manifest.robustness_id == identity.robustness_id
    assert robustness_outcome.manifest.stage is RobustnessStage.COMPLETED, robustness_outcome.manifest.failure_summary
    robustness_id = robustness_outcome.manifest.robustness_id

    promotion_ref = robustness_outcome.manifest.artifact("promotion_decision")
    assert promotion_ref is not None
    promotion = PromotionDecision.from_json_dict(parse_json_strict(artifact_store.read_artifact(promotion_ref.content_hash).decode("utf-8")))
    assert promotion.decision is PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING, (promotion.decision, promotion.decision_reason)

    eligibility_environment = EligibilityVerificationEnvironment(
        robustness_manifest_store=robustness_manifest_store, artifact_store=artifact_store, backtest_manifest_store=backtest_manifest_store,
        backtest_event_store=backtest_event_store, calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
        execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store, research_dataset_store=research_store,
        dataset_loader=dataset_loader,
    )

    fold_result_references = execution_manifest_store.load(experiment_id).fold_result_references
    fold_index = max(fold_result_references)
    fold_result = FoldResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_result_references[fold_index].content_hash).decode("utf-8")))
    model_ref = next(r for r in fold_result.artifact_references if r.category is ArtifactCategory.MODEL)
    _serializer, deserializer = mz.default_serializer_registry()[model_definition.serializer_id]
    fitted_model = deserializer.deserialize(artifact_store.read_artifact(model_ref.content_hash))

    return {
        "tmp_path": tmp_path, "df": df, "dataset_manifest": dataset_manifest, "experiment_id": experiment_id, "calibration_id": calibration_id,
        "backtest_id": backtest_id, "robustness_id": robustness_id, "promotion_content_hash": promotion_ref.content_hash,
        "eligibility_environment": eligibility_environment, "fitted_model": fitted_model,
    }


def _build_paper_spec(chain: dict, *, session_mode: SessionMode, maximum_order_quantity: float | None) -> PaperTradingSpec:
    instrument = InstrumentSpec(
        symbol="XAUUSD", base_currency="XAU", quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=1.0, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=100.0, price_precision=2, quantity_precision=2, margin_mode="hypothetical_margin_mode",
        account_currency="USD", financing_convention="hypothetical_daily_swap", trading_timezone="UTC", session_calendar_identity="hypothetical_247",
    )
    risk_limits = RiskLimitsSpec(
        maximum_signed_position=None, maximum_absolute_position=3.0, maximum_gross_exposure=None, maximum_order_quantity=maximum_order_quantity,
        maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None, maximum_realized_loss=None,
        maximum_unrealized_loss=None, maximum_rejected_order_count=None, maximum_consecutive_execution_failures=None,
        maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
    )
    return PaperTradingSpec(
        schema_version=1, verified_robustness_id=chain["robustness_id"], verified_promotion_decision_id=chain["promotion_content_hash"],
        strategy_candidate_identity=chain["backtest_id"], model_artifact_identity=chain["experiment_id"], calibration_artifact_identity=chain["calibration_id"],
        feature_spec_identity=chain["dataset_manifest"].feature_registry_fingerprint, instrument=instrument, price_precision=2, quantity_precision=2,
        session_mode=session_mode, market_event_mode=MarketEventMode.BAR, bar_interval=Timeframe.M1, clock_mode=ClockMode.REPLAY,
        starting_cash=100_000.0, starting_positions=(),
        order_policy=OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=1, order_rate_window_events=10),
        execution_policy=DEFAULT_EXECUTION_POLICY, fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        spread_policy=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=3.0), slippage_policy=SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0),
        commission_policy=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=1.0),
        financing_policy=FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)),
        latency_policy=LatencyPolicySpec(decision_to_submit_ms=50, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50),
        liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False), position_policy=DEFAULT_POSITION_POLICY, risk_limits=risk_limits,
        session_boundary_policy=DEFAULT_SESSION_BOUNDARY_POLICY, seed=0,
    )


def _build_replay_events(df: pd.DataFrame, path) -> tuple:
    replay_slice = df.iloc[2900:3200].reset_index(drop=True)
    bar_events = []
    for i, row in replay_slice.iterrows():
        open_time = pd.Timestamp(row["open_time"]).to_pydatetime()
        bar_events.append(create_bar_event(
            instrument="XAUUSD", interval=Timeframe.M1, open_time=open_time, open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
            close=float(row["close"]), volume=float(row["tick_volume"]), sequence=i + 1, source="synthetic_replay",
        ))
    eos = create_end_of_stream_event(instrument="XAUUSD", event_time=bar_events[-1].close_time + timedelta(minutes=1), sequence=len(bar_events) + 1, source="synthetic_replay")
    events = (*bar_events, eos)
    write_replay_events(path, events)
    return load_replay_events(path)


def _normalize_ledger(ledger: list) -> list:
    """Same convention `test_runner.py` established: SESSION_TRANSITION
    entries carry a real wall-clock `event_time` (Section 0's own
    documented determinism carve-out), so two genuinely independent runs
    compare only `(sequence, kind, payload)`."""
    return [{"sequence": e.sequence, "kind": e.kind.value, "payload": e.payload} for e in ledger]


def test_real_model_paper_trading_and_shadow_acceptance_workflow(eligible_chain, tmp_path) -> None:
    chain = eligible_chain
    eligibility_environment = chain["eligibility_environment"]
    fitted_model = chain["fitted_model"]

    paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)

    # ---- Section 4: eligibility must independently re-verify, never trust a persisted flag ----
    eligibility_report = verify_paper_trading_eligibility(paper_spec, environment=eligibility_environment)
    assert eligibility_report.is_eligible, (eligibility_report.failed_step, eligibility_report.failure_reason)

    paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id

    # ---- One real bounded replay market stream, built and validated through Section 32's own reader ----
    replay_path = tmp_path / "replay.jsonl"
    loaded_events = _build_replay_events(chain["df"], replay_path)
    assert len(loaded_events) > 1

    def _run(root) -> tuple:
        manifest_store = PaperSessionManifestStore(root)
        event_store = PaperSessionEventStore(root)
        environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=eligibility_environment)
        create_paper_session(paper_spec, environment=environment)
        strategy_runtime = ModelStrategyRuntime(
            strategy_identity=chain["backtest_id"], fitted_model=fitted_model, feature_names=_FEATURE_NAMES,
            long_threshold=0.55, short_threshold=0.45, target_quantity=1.0, confidence_scaled_sizing=True,
        )
        manifest = run_paper_trading_session(paper_spec, environment=environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=loaded_events)
        ledger = event_store.read_events(paper_session_id)
        return manifest, ledger, environment

    first_root = tmp_path / "first_run"
    paper_manifest, ledger, runner_environment = _run(first_root)
    assert paper_manifest.stage.value == "completed"

    # ---- Section 33: a second, genuinely INDEPENDENT run (fresh manifest/event stores) proving determinism ----
    second_root = tmp_path / "second_run"
    second_manifest, second_ledger, _ = _run(second_root)
    assert second_manifest.stage.value == "completed"
    assert _normalize_ledger(second_ledger) == _normalize_ledger(ledger), "two independent runs of the SAME spec+events must produce byte-identical (modulo wall-clock) ledgers"

    # ---- Idempotent resume: re-running an already-COMPLETED session is a no-op ----
    resumed_manifest = run_paper_trading_session(paper_spec, environment=runner_environment, strategy_runtime=ModelStrategyRuntime(
        strategy_identity=chain["backtest_id"], fitted_model=fitted_model, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45,
        target_quantity=1.0, confidence_scaled_sizing=True,
    ), clock=ReplayClock(), events=loaded_events)
    assert resumed_manifest == paper_manifest

    # ---- Section 25: independent reconciliation ----
    reconciliation_report = reconcile_session(ledger, session_id=paper_session_id, instrument=paper_spec.instrument, starting_cash=paper_spec.starting_cash)
    assert reconciliation_report.is_reconciled, [c.check_identity for c in reconciliation_report.checks if not c.passed]

    # ---- Section 26: independent verification (never trusts the persisted manifest/report) ----
    verification_report = verify_paper_session(paper_spec, manifest=paper_manifest, ledger=ledger, eligibility_environment=eligibility_environment)
    assert verification_report.is_ready, [i.to_json_dict() for i in verification_report.criticals]

    # ---- Section 27: durable session report ----
    report = build_paper_session_report(ledger, spec=paper_spec, manifest=paper_manifest, reconciliation_report=reconciliation_report, verification_report=verification_report)
    assert report.disclaimer == DIAGNOSTIC_DISCLAIMER

    # ---- Section 33's own required coverage: long, short, abstention, and a genuine risk-rejected order ----
    decisions = [e.payload for e in ledger if e.kind is LedgerEntryKind.STRATEGY_DECISION]
    direction_counts = Counter((d["target_direction"], d["abstain"]) for d in decisions)
    assert direction_counts[("long", False)] > 0, "at least one long decision is required"
    assert direction_counts[("short", False)] > 0, "at least one short decision is required"
    assert direction_counts[("flat", True)] > 0, "at least one abstention is required"

    orders_by_id: dict[str, tuple[dict, list]] = {}
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.ORDER_STATE_EVENT:
            continue
        order_json = entry.payload["order"]
        state_event = OrderStateEvent.from_json_dict(entry.payload["order_state_event"])
        if state_event.order_id not in orders_by_id:
            orders_by_id[state_event.order_id] = (order_json, [])
        orders_by_id[state_event.order_id][1].append(state_event)
    final_states = {order_id: resolve_order_state(order_id, events) for order_id, (_, events) in orders_by_id.items()}
    rejected_order_ids = [oid for oid, state in final_states.items() if state.value == "rejected"]
    filled_order_ids = [oid for oid, state in final_states.items() if state.value == "filled"]
    assert rejected_order_ids, "at least one order rejected by risk/validation is required"
    assert filled_order_ids, "at least one filled order is required"
    long_fills = any(orders_by_id[oid][0]["side"] == "buy" for oid in filled_order_ids)
    short_fills = any(orders_by_id[oid][0]["side"] == "sell" for oid in filled_order_ids)
    assert long_fills and short_fills, "both long and short fills are required"

    # ---- Section 19: shadow observation, distinct from and never merged with the real account ----
    shadow_spec = _build_paper_spec(chain, session_mode=SessionMode.SHADOW_OBSERVATION, maximum_order_quantity=1.2)
    shadow_paper_session_id = compute_paper_session_spec_id(shadow_spec).paper_session_spec_id
    shadow_strategy_runtime = ModelStrategyRuntime(
        strategy_identity=chain["backtest_id"], fitted_model=fitted_model, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45,
        target_quantity=1.0, confidence_scaled_sizing=True,
    )
    shadow_manifest = run_shadow_session(shadow_spec, environment=runner_environment, strategy_runtime=shadow_strategy_runtime, clock=ReplayClock(), events=loaded_events)
    assert shadow_manifest.stage.value == "completed"
    shadow_ledger = runner_environment.event_store.read_events(shadow_paper_session_id)
    shadow_reconciliation = reconcile_session(shadow_ledger, session_id=shadow_paper_session_id, instrument=shadow_spec.instrument, starting_cash=shadow_spec.starting_cash)
    shadow_report = build_paper_session_report(shadow_ledger, spec=shadow_spec, manifest=shadow_manifest, reconciliation_report=shadow_reconciliation)
    assert shadow_report.shadow.observation_count > 0
    assert shadow_report.shadow.observations_with_hypothetical_fill_count > 0
    # Shadow observations never touch the real account -- no ORDER_STATE_EVENT/FILL/ACCOUNT_SNAPSHOT entries in ITS OWN ledger.
    assert not [e for e in shadow_ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT]
    assert not [e for e in shadow_ledger if e.kind is LedgerEntryKind.FILL]
    assert not [e for e in shadow_ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT]

    # ---- Section 33: report exact figures ----
    print(f"\nMilestone 7 acceptance workflow: paper_session_id={paper_session_id}")
    print(f"  event_count={report.session.event_count} decision_count={report.decisions.decision_count} abstention_count={report.decisions.abstention_count}")
    print(f"  order_count={report.orders.order_count} rejected_orders={report.orders.rejected_count} fill_count={report.fills.fill_count}")
    print(f"  long_fills={long_fills} short_fills={short_fills}")
    print(f"  starting_cash={report.account_equity.starting_cash} final_cash={report.account_equity.final_cash} final_equity={report.account_equity.final_equity}")
    print(f"  gross_pnl={report.account_equity.gross_pnl} net_pnl={report.account_equity.net_pnl}")
    print(f"  spread_cost={report.costs.total_spread_cost} slippage_cost={report.costs.total_slippage_cost} commission_cost={report.costs.total_commission_cost} financing={report.costs.total_financing}")
    print(f"  maximum_drawdown_fraction={report.drawdown.maximum_drawdown_fraction}")
    print(f"  reconciled={reconciliation_report.is_reconciled} verification_is_ready={verification_report.is_ready}")
    print(f"  deterministic_second_run_matches={True}")
    print(f"  shadow_observation_count={shadow_report.shadow.observation_count} shadow_with_fill={shadow_report.shadow.observations_with_hypothetical_fill_count} shadow_counterfactual_pnl={shadow_report.shadow.total_counterfactual_realized_pnl}")


class TestFoldSelectionIsPinnedAgainstMidSessionDrift:
    """Release-audit Area 4: `_resolve_fitted_strategy_runtime` (`ml_cli.
    py`) used to re-derive `fold_index = max(execution_manifest.fold_
    result_references)` from LIVE `ExecutionManifestStore` state on EVERY
    call -- `run-paper-session`, every `resume-paper-session`, `run-
    shadow-session` -- with nothing about which fold/model was actually
    used ever recorded anywhere durable or part of `paper_session_spec_
    id`. Fixed by pinning the resolved `(experiment_id, fold_index,
    model_content_hash)` triple to a session-scoped file the first time
    it is resolved, and failing closed if a later resolution for the SAME
    `paper_session_id` disagrees. These tests use the REAL experiment
    `eligible_chain` already built (a real logistic-regression model, a
    real walk-forward execution with real fold results) -- not a mock --
    so the pin path, the JSON shape, and the resolved model itself are
    all genuine."""

    def _config_stub(self, chain: dict) -> SimpleNamespace:
        return SimpleNamespace(ml_artifacts_root=chain["tmp_path"] / "ml_artifacts")

    def test_first_resolution_pins_the_selected_fold_and_model(self, eligible_chain) -> None:
        chain = eligible_chain
        paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        config_stub = self._config_stub(chain)
        paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        pin_path = _paper_session_model_pin_path(config_stub, paper_session_id)  # type: ignore[arg-type]
        assert not pin_path.is_file()

        _resolve_fitted_strategy_runtime(config_stub, paper_spec, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45, target_quantity=1.0)  # type: ignore[arg-type]

        assert pin_path.is_file()
        pinned = json.loads(pin_path.read_text())
        assert pinned["experiment_id"] == chain["experiment_id"]
        assert isinstance(pinned["fold_index"], int)
        assert isinstance(pinned["model_content_hash"], str) and len(pinned["model_content_hash"]) == 64

    def test_second_resolution_of_the_same_session_is_a_consistent_noop(self, eligible_chain) -> None:
        chain = eligible_chain
        paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        config_stub = self._config_stub(chain)
        paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        pin_path = _paper_session_model_pin_path(config_stub, paper_session_id)  # type: ignore[arg-type]

        _resolve_fitted_strategy_runtime(config_stub, paper_spec, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45, target_quantity=1.0)  # type: ignore[arg-type]
        first_pin = pin_path.read_text()

        # Simulates the SECOND resolution a `resume-paper-session` call
        # makes -- must resolve to the identical pinned triple, not raise,
        # and not rewrite the pin file to a new value.
        _resolve_fitted_strategy_runtime(config_stub, paper_spec, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45, target_quantity=1.0)  # type: ignore[arg-type]
        assert pin_path.read_text() == first_pin

    def test_a_drifted_pin_is_rejected_fail_closed_before_any_model_is_loaded(self, eligible_chain) -> None:
        """Simulates exactly the dangerous scenario Section 4 exists to
        catch: the experiment's fold set changed underneath an
        already-started session (e.g. an operator re-ran a fold, or the
        walk-forward plan grew a new outer fold) between the original
        `run-paper-session` and a later `resume-paper-session`. Tampering
        the pin file directly is the most precise way to prove the
        COMPARISON itself is correct and fires, independent of how the
        drift was caused."""
        chain = eligible_chain
        paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        config_stub = self._config_stub(chain)
        paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        pin_path = _paper_session_model_pin_path(config_stub, paper_session_id)  # type: ignore[arg-type]

        _resolve_fitted_strategy_runtime(config_stub, paper_spec, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45, target_quantity=1.0)  # type: ignore[arg-type]
        genuine_pin = json.loads(pin_path.read_text())
        tampered_pin = dict(genuine_pin)
        tampered_pin["fold_index"] = genuine_pin["fold_index"] + 999
        tampered_pin["model_content_hash"] = "f" * 64
        pin_path.write_text(json.dumps(tampered_pin))

        with pytest.raises(PaperTradingEligibilityError, match="refusing to swap the fitted model mid-session"):
            _resolve_fitted_strategy_runtime(config_stub, paper_spec, feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45, target_quantity=1.0)  # type: ignore[arg-type]

        # The failed resolution must not have "repaired" or overwritten
        # the tampered pin -- fail-closed means nothing gets mutated.
        assert json.loads(pin_path.read_text()) == tampered_pin


class TestResumeRevalidatesEligibilityAfterSourceTampering:
    """Release-audit Area 8: CONFIRMED BLOCKER-CLASS DEFECT, fixed.
    `require_paper_trading_eligibility` used to be called EXACTLY ONCE --
    inside `create_paper_session`, the very first time a session's
    manifest was created. Every subsequent call to `run_paper_trading_
    session`/`run_shadow_session` (a resume, whether after a pause, a
    crash, or any operator-initiated interruption) found the manifest
    already existing and skipped `create_paper_session` entirely --
    meaning the FULL eligibility chain (promotion decision, robustness
    result, source backtest) was NEVER re-verified again for the
    lifetime of a session. A session whose underlying source artifacts
    were tampered, superseded, or removed AFTER it started could resume
    and keep processing new market events indefinitely. Fixed:
    `run_paper_trading_session`/`run_shadow_session` now mandatorily
    re-run the SAME fail-closed eligibility check on every call that did
    not itself just create the manifest.

    These tests use the REAL `eligible_chain` (a genuinely `ELIGIBLE_FOR_
    PAPER_TRADING` promotion decision, backed by a real trained model and
    a real robustness/backtest chain) -- not a mock -- so the tampering
    performed here (deleting the persisted robustness manifest between
    session creation and resume) is exactly what an operator would
    observe from a real artifact store being pruned, corrupted, or
    having its evidence superseded mid-session."""

    def test_resume_fails_closed_after_the_robustness_manifest_is_removed(self, eligible_chain, tmp_path) -> None:
        chain = eligible_chain
        eligibility_environment = chain["eligibility_environment"]
        paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id

        manifest_store = PaperSessionManifestStore(tmp_path)
        event_store = PaperSessionEventStore(tmp_path)
        environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=eligibility_environment)

        # Step 1: session creation genuinely re-verifies (and passes) against the real, untampered chain.
        created_manifest = create_paper_session(paper_spec, environment=environment)
        assert created_manifest.paper_session_id == paper_session_id

        # Step 2: the underlying robustness evidence is removed -- simulates a pruned/corrupted artifact
        # store, or a later re-run of robustness superseding this decision. Uses the SAME `RobustnessManifestStore`
        # the real eligibility_environment already points at (never a filename/path guess of our own).
        # `eligible_chain` is MODULE-scoped (built once, shared by every test in this file) -- the tampering
        # here MUST be restored (try/finally), or every later test in this module inherits a permanently
        # broken chain.
        robustness_manifest_store = eligibility_environment.robustness_manifest_store
        manifest_path = robustness_manifest_store._manifest_path(chain["robustness_id"])
        assert manifest_path.is_file(), "fixture invariant: the real robustness manifest must exist before tampering"
        original_bytes = manifest_path.read_bytes()
        manifest_path.unlink()
        try:
            # Step 3: resuming (a fresh RunnerEnvironment, exactly like a genuinely separate process would
            # build) must now fail closed BEFORE processing a single further market event -- never silently
            # continue on the manifest that already exists.
            replay_path = tmp_path / "replay.jsonl"
            loaded_events = _build_replay_events(chain["df"], replay_path)
            resume_environment = RunnerEnvironment(manifest_store=PaperSessionManifestStore(tmp_path), event_store=PaperSessionEventStore(tmp_path), eligibility_environment=eligibility_environment)
            strategy_runtime = ModelStrategyRuntime(
                strategy_identity=chain["backtest_id"], fitted_model=chain["fitted_model"], feature_names=_FEATURE_NAMES, long_threshold=0.55, short_threshold=0.45,
                target_quantity=1.0, confidence_scaled_sizing=True,
            )
            with pytest.raises(PaperTradingEligibilityError):
                run_paper_trading_session(paper_spec, environment=resume_environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=loaded_events)

            # Step 4: confirm nothing was mutated by the failed attempt -- no ledger entries, manifest unchanged.
            ledger_after_failed_resume = resume_environment.event_store.read_events(paper_session_id)
            assert not [e for e in ledger_after_failed_resume if e.kind is LedgerEntryKind.MARKET_EVENT_ACCEPTED]
        finally:
            manifest_path.write_bytes(original_bytes)

    def test_direct_python_api_create_paper_session_also_refuses_on_removed_evidence(self, eligible_chain, tmp_path) -> None:
        """Same tampering, exercised directly against `create_paper_
        session` (the direct Python API entry point) rather than through
        `run_paper_trading_session` -- confirms the fail-closed property
        holds at both call sites Section 8 names."""
        chain = eligible_chain
        eligibility_environment = chain["eligibility_environment"]
        tampered_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        environment = RunnerEnvironment(manifest_store=PaperSessionManifestStore(tmp_path), event_store=PaperSessionEventStore(tmp_path), eligibility_environment=eligibility_environment)

        robustness_manifest_store = eligibility_environment.robustness_manifest_store
        manifest_path = robustness_manifest_store._manifest_path(chain["robustness_id"])
        original_bytes = manifest_path.read_bytes()
        manifest_path.unlink()
        try:
            with pytest.raises(PaperTradingEligibilityError):
                create_paper_session(tampered_spec, environment=environment)
        finally:
            manifest_path.write_bytes(original_bytes)


class TestIdentityMismatchAfterPromotionIsRejected:
    """Release-audit Area 8: `identities_match` is a genuine, load-bearing
    check in `verify_paper_trading_eligibility`'s chain, but the existing
    `test_eligibility.py` suite only ever exercised it by constructing an
    `EligibilityVerificationReport` directly with `identities_match=False`
    (report-SHAPE testing) -- never through the real detection logic
    against a genuine chain. These tests swap ONE declared identity at a
    time in an otherwise-genuine, otherwise-eligible spec and confirm the
    REAL chain rejects each swap specifically."""

    def test_wrong_feature_spec_identity_rejected(self, eligible_chain) -> None:
        chain = eligible_chain
        spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        tampered = dataclasses.replace(spec, feature_spec_identity="f" * 64)
        report = verify_paper_trading_eligibility(tampered, environment=chain["eligibility_environment"])
        assert not report.is_eligible
        assert report.failed_step == "identities_match"

    def test_wrong_strategy_candidate_identity_rejected(self, eligible_chain) -> None:
        chain = eligible_chain
        spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        tampered = dataclasses.replace(spec, strategy_candidate_identity="f" * 64)
        report = verify_paper_trading_eligibility(tampered, environment=chain["eligibility_environment"])
        assert not report.is_eligible
        assert report.failed_step == "identities_match"

    def test_wrong_model_artifact_identity_rejected(self, eligible_chain) -> None:
        chain = eligible_chain
        spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        tampered = dataclasses.replace(spec, model_artifact_identity="f" * 64)
        report = verify_paper_trading_eligibility(tampered, environment=chain["eligibility_environment"])
        assert not report.is_eligible
        assert report.failed_step == "identities_match"

    def test_wrong_calibration_artifact_identity_rejected(self, eligible_chain) -> None:
        chain = eligible_chain
        spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
        tampered = dataclasses.replace(spec, calibration_artifact_identity="f" * 64)
        report = verify_paper_trading_eligibility(tampered, environment=chain["eligibility_environment"])
        assert not report.is_eligible
        assert report.failed_step == "identities_match"


class TestStaleVerificationReportNeverTrusted:
    def test_eligibility_is_recomputed_fresh_every_call_not_cached(self, eligible_chain, tmp_path) -> None:
        chain = eligible_chain
        eligibility_environment = chain["eligibility_environment"]
        spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)

        first = verify_paper_trading_eligibility(spec, environment=eligibility_environment)
        assert first.is_eligible

        robustness_manifest_store = eligibility_environment.robustness_manifest_store
        manifest_path = robustness_manifest_store._manifest_path(chain["robustness_id"])
        original_bytes = manifest_path.read_bytes()
        manifest_path.unlink()
        try:
            second = verify_paper_trading_eligibility(spec, environment=eligibility_environment)
            assert not second.is_eligible, "a stale in-memory/cached 'is_eligible=True' must never be trusted -- every call independently re-verifies"
        finally:
            manifest_path.write_bytes(original_bytes)

        third = verify_paper_trading_eligibility(spec, environment=eligibility_environment)
        assert third.is_eligible, "restoring the evidence must make a FRESH call eligible again -- confirms this was live recomputation, not a poisoned cache"
