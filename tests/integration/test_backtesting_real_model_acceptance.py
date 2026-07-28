"""Section 48: a bounded, deterministic, real-model end-to-end backtest
acceptance test -- NOT `ConstantTestModelFactory`. Uses a real production
`logistic_regression` model (registered in `ml.model_zoo`) against a
synthetic dataset with GENUINE injected predictive structure (an AR(1)
latent momentum signal drives forward returns), carried all the way
through a real execution run, a real calibration run, and a real
leakage-safe backtest.

This test asserts INFRASTRUCTURE correctness (the pipeline runs to
completion, produces finite, internally-consistent, independently
re-verifiable financial artifacts) -- it deliberately does NOT assert
that the resulting backtest is profitable. A positive or negative net
return here is not evidence of live trading viability; see
`backtesting.reporting`'s mandatory hypothetical/non-live disclaimers,
reproduced in every report this framework produces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
from quant_platform.backtesting.runner import BacktestRunner, OuterFoldBacktestResult
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
from quant_platform.historical.manifest import ManifestStore
from quant_platform.ml import model_zoo as mz
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
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
from quant_platform.optimization.inner_splits import InnerSplitConfig

_RAW_HISTORICAL_COLUMNS = ("open_time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread")


def _make_signal_ohlcv(n: int, *, seed: int, ar_coefficient: float = 0.92, signal_strength: float = 0.006, noise_std: float = 0.0009) -> pd.DataFrame:
    """Identical construction to `test_calibration_real_model_acceptance.
    _make_signal_ohlcv`: a genuine injected AR(1) momentum signal driving
    forward returns, so both the model AND the resulting backtest have
    real (not fabricated) structure to work with."""
    rng = np.random.default_rng(seed)
    open_time = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    signal = np.zeros(n)
    for t in range(1, n):
        signal[t] = ar_coefficient * signal[t - 1] + rng.normal(0, 1.0)
    signal = signal / (signal.std() + 1e-12)
    log_returns = np.zeros(n)
    for t in range(1, n):
        log_returns[t] = signal_strength * signal[t - 1] + rng.normal(0, noise_std)
    close = 2000.0 * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    if n > 0:
        open_[0] = 2000.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, size=n)))
    return pd.DataFrame({
        "open_time": open_time, "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(10, 1000, size=n), "real_volume": np.zeros(n, dtype=np.int64), "spread": rng.integers(1, 30, size=n),
    })[list(_RAW_HISTORICAL_COLUMNS)]


@pytest.fixture(scope="module")
def signal_dataset(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("backtest_real_model_dataset")
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = _make_signal_ohlcv(4200, seed=202)
    seed_canonical_dataset(historical_root, df)

    canonical_store = CanonicalStore(historical_root)
    manifest_store = ManifestStore(historical_root)
    historical_loader = DatasetLoader(canonical_store, manifest_store)
    registry = FeatureRegistry()
    register_core_technical_features(registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1, 5, 10), momentum_windows=(10,), atr_window=14))

    research_store = ResearchDatasetStore(research_root)
    research_manifest_store = ResearchManifestStore(research_root)
    builder = ResearchDatasetBuilder(historical_loader=historical_loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store)
    feature_names = (
        "return_log_1", "return_log_5", "return_log_10", "candle_body_ratio", "candle_lower_wick_ratio",
        "candle_upper_wick_ratio", "high_low_distance_20", "ma_distance_20", "ma_distance_50", "rolling_zscore_close_20",
    )
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=start, end=end, feature_names=feature_names,
        label_definition=LabelDefinition(name="fut10", kind=LabelKind.BINARY_DIRECTION, horizon_bars=10),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 10, "embargo_bars": 10},
        preprocessing={},
    )
    manifest = builder.build(request)
    return manifest, research_manifest_store, research_store, historical_root, tmp_path


def test_logistic_regression_real_model_end_to_end_backtest(signal_dataset) -> None:
    dataset_manifest, research_manifest_store, research_store, historical_root, tmp_path = signal_dataset
    ml_artifacts_root = tmp_path / "ml_artifacts_logreg_backtest"

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
        label_binding=LabelBinding(name="fut10", kind=LabelKind.BINARY_DIRECTION.value, horizon_bars=10, label_type=LabelType.BINARY),
        split_binding=SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 3, "test_size": 350, "purge_bars": 10, "embargo_bars": 10}),
        preprocessing_binding=preprocessing_binding, model_name="logistic_regression", model_version="1",
        hyperparameters=ModelHyperparameters(values={"C": 1.0, "max_iter": 500}), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=17), code_revision_binding=CodeRevisionBinding(revision="d" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )
    model_registry = mz.register_default_models()
    preparer = ExperimentPreparer(ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store)
    experiment_manifest = preparer.prepare(experiment_spec)
    assert experiment_manifest.status is ExperimentStatus.READY, experiment_manifest.failure_summary
    experiment_id = experiment_manifest.identity.experiment_id

    execution_runner = ExecutionRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, additional_serializers=mz.default_serializer_registry(),
    )
    execution_runner.run(experiment_id)
    execution_manifest = execution_runner.execution_manifest_store.load(experiment_id)
    assert execution_manifest.stage is ExecutionStage.COMPLETED, execution_manifest.failure_summary

    model_definition = model_registry.get("logistic_regression", "1")
    calibration_spec = CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id=experiment_id, base_model_definition_identity=model_definition.fingerprint(),
        dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        calibration_method_candidates=(CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=20, minimum_samples_per_class=5,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2, embargo_bars=5),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.SYMMETRIC_BAND, band_half_width=0.05),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy", "margin", "bin_support"), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
        seed=99, determinism_policy=DeterminismPolicy.STRICT,
    )
    calibration_runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
        additional_serializers=mz.default_serializer_registry(),
    )
    calibration_outcome = calibration_runner.run(calibration_spec)
    assert calibration_outcome.manifest.stage is CalibrationStage.COMPLETED, calibration_outcome.manifest.failure_summary
    calibration_id = calibration_outcome.manifest.calibration_id

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
    dataset_loader = DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root))
    backtest_runner = BacktestRunner(
        ml_artifacts_root=ml_artifacts_root, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root), execution_manifest_store=ExecutionManifestStore(ml_artifacts_root),
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )

    identity = compute_backtest_identity(backtest_spec)
    outcome = backtest_runner.run(backtest_spec)
    assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary
    assert outcome.manifest.backtest_id == identity.backtest_id

    artifact_store = MLArtifactStore(ml_artifacts_root)
    total_closed_trades = 0
    for ref in outcome.manifest.outer_fold_result_references.values():
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(ref.content_hash).decode("utf-8")))
        total_closed_trades += result.closed_trade_count
        for name, value in result.financial_metrics.items():
            assert np.isfinite(value) if isinstance(value, (int, float)) else True, f"non-finite metric {name}={value!r}"

        benchmark_raw = parse_json_strict(artifact_store.read_artifact(result.benchmark_report_reference.content_hash).decode("utf-8"))
        # Milestone 5.1, Section 4: the completed benchmark matrix --
        # always_flat + always_long/short (buy-and-hold and full-pipeline
        # "always" strategies, both present since this spec uses
        # PositionMode.LONG_SHORT) + raw-threshold + calibrated
        # with/without abstention, each of the latter 5 as a zero/net-cost
        # pair: 1 + 2 + 2 + 2 + 2 + 5*2 = 15.
        benchmark_names = {b["name"] for b in benchmark_raw["benchmarks"]}
        assert len(benchmark_names) == 15
        assert {
            "always_flat", "always_long_zero_cost", "always_long_net_cost", "always_short_zero_cost", "always_short_net_cost",
            "always_long_strategy_zero_cost", "always_short_strategy_zero_cost", "raw_uncalibrated_threshold_zero_cost",
            "calibrated_no_abstention_zero_cost", "calibrated_with_abstention_zero_cost",
        } <= benchmark_names
        cost_sensitivity_raw = parse_json_strict(artifact_store.read_artifact(result.cost_sensitivity_report_reference.content_hash).decode("utf-8"))
        by_scenario = {r["scenario_name"]: r["total_net_return"] for r in cost_sensitivity_raw["results"]}
        if result.closed_trade_count > 0 and by_scenario.get("zero_cost") != by_scenario.get("2x_spread"):
            assert by_scenario["zero_cost"] >= by_scenario["2x_spread"], "zero-cost scenario must never underperform the doubled-spread scenario"

    # A real model with genuine injected signal, evaluated through a
    # realistic (long/short, abstention-aware) mapping, should place at
    # least SOME trades across 3 outer folds -- proving this is not just
    # infrastructure plumbing around a strategy that never fires.
    assert total_closed_trades > 0, "expected at least one closed trade across all outer folds"

    report = verify_backtest(
        outcome.manifest.backtest_id, backtest_manifest_store=BacktestManifestStore(ml_artifacts_root),
        artifact_store=artifact_store, event_store=BacktestEventStore(ml_artifacts_root),
        calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root), experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
        execution_manifest_store=ExecutionManifestStore(ml_artifacts_root), research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, dataset_loader=dataset_loader,
    )
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]
    assert any(i.code == "fold_financials_reproduce" for i in report.infos)
    assert any(i.code == "raw_source_reconstruction_verified" for i in report.infos)

    second = backtest_runner.run(backtest_spec)
    assert second.was_idempotent_no_op
    assert second.manifest.backtest_id == outcome.manifest.backtest_id
