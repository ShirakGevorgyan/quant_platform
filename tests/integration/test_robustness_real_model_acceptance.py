"""Milestone 6, Section 17: bounded, real-model acceptance workflow for
the statistical-robustness/promotion-gate framework. Reuses the IDENTICAL
real `logistic_regression` model, injected-AR(1)-signal dataset,
experiment, execution, and calibration setup Milestone 5's own acceptance
test (`test_backtesting_real_model_acceptance.py`) already established
and verified -- carried forward into TWO distinct backtest candidates
(differing in holding period), a `StrategyFamily` linking them, one
`RobustnessRunner` analysis per candidate, a cross-candidate
`SelectionReport`, and each candidate's own `PromotionDecision`.

Synthetic data is used only as infrastructure evidence, exactly as
Section 17 requires. This test asserts INFRASTRUCTURE correctness (the
full pipeline runs to completion, produces finite, independently
re-verifiable artifacts, and reaches SOME valid `PromotionDecisionKind`)
-- it does NOT assert or require a particular promotion outcome.
REJECTED, RESEARCH_ONLY, MANUAL_REVIEW_REQUIRED, and ELIGIBLE_FOR_
PAPER_TRADING are all treated as valid, successful acceptance results;
this test never tunes data or thresholds to force a passing decision."""

from __future__ import annotations

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
from quant_platform.robustness.bootstrap import BootstrapReport
from quant_platform.robustness.manifests import RobustnessManifestStore
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    PromotionDecisionKind,
    ReturnSeriesKind,
    RobustnessStage,
)
from quant_platform.robustness.multiple_testing import build_strategy_family
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.runner import RobustnessRunner
from quant_platform.robustness.selection import CandidateEvidence, compute_selection_report
from quant_platform.robustness.sensitivity import SensitivityReport
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.specs import (
    DEFAULT_PROMOTION_GATES,
    DEFAULT_REGIME_DEFINITIONS,
    DEFAULT_STRESS_SCENARIOS,
    BootstrapSpec,
    PromotionPolicySpec,
    RobustnessSpec,
    StabilityThresholds,
    compute_robustness_identity,
)
from quant_platform.robustness.stability import FoldStabilityReport
from quant_platform.robustness.stress import StressReport
from quant_platform.robustness.verification import load_fold_evidence, verify_robustness


@pytest.fixture(scope="module")
def signal_dataset(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("robustness_real_model_dataset")
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = _make_signal_ohlcv(3200, seed=303)
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


def _build_candidate_backtest(*, holding_period_bars: int, seed: int, calibration_id: str, experiment_id: str, dataset_manifest, experiment_spec) -> BacktestSpec:
    return BacktestSpec(
        schema_version=1, source_calibration_id=calibration_id, source_experiment_id=experiment_id, source_execution_id=experiment_id,
        dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
        instrument_identity=dataset_manifest.symbol, market_timezone="UTC", bar_interval=dataset_manifest.base_timeframe,
        decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.ABSTENTION_AWARE), position_mode=PositionMode.LONG_SHORT,
        entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=holding_period_bars, final_trade_policy=FinalTradePolicyKind.FORCE_CLOSE_AT_FINAL_PRICE),
        overlap_policy=OverlapPolicyKind.CLOSE_AND_REVERSE, price_basis=PriceBasisKind.CLOSE,
        spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=3.0),
        commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=1.0),
        slippage_spec=SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
        return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.COMPOUNDED,
        initial_notional=10000.0, determinism_policy=DeterminismPolicy.STRICT, respect_calibration_abstention=True,
        annual_risk_free_rate=0.02, seed=seed,
    )


def test_two_candidate_robustness_and_promotion_acceptance_workflow(signal_dataset) -> None:
    dataset_manifest, research_manifest_store, research_store, historical_root, tmp_path = signal_dataset
    ml_artifacts_root = tmp_path / "ml_artifacts_robustness"

    # ---- Section 17 Step 1: one real Logistic Regression candidate (experiment/execution/calibration) ----
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

    execution_runner = ExecutionRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, additional_serializers=mz.default_serializer_registry(),
    )
    execution_runner.run(experiment_id)
    assert execution_runner.execution_manifest_store.load(experiment_id).stage is ExecutionStage.COMPLETED

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
    calibration_runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root), additional_serializers=mz.default_serializer_registry(),
    )
    calibration_outcome = calibration_runner.run(calibration_spec)
    assert calibration_outcome.manifest.stage is CalibrationStage.COMPLETED, calibration_outcome.manifest.failure_summary
    calibration_id = calibration_outcome.manifest.calibration_id

    # ---- Section 17 Step 2: at least TWO distinct backtest candidates from the SAME calibration ----
    dataset_loader = DatasetLoader(CanonicalStore(historical_root), ManifestStore(historical_root))
    backtest_runner = BacktestRunner(
        ml_artifacts_root=ml_artifacts_root, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root), execution_manifest_store=ExecutionManifestStore(ml_artifacts_root),
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )
    candidate_specs = [
        _build_candidate_backtest(holding_period_bars=10, seed=7, calibration_id=calibration_id, experiment_id=experiment_id, dataset_manifest=dataset_manifest, experiment_spec=experiment_spec),
        _build_candidate_backtest(holding_period_bars=20, seed=7, calibration_id=calibration_id, experiment_id=experiment_id, dataset_manifest=dataset_manifest, experiment_spec=experiment_spec),
    ]
    candidate_backtest_ids: list[str] = []
    for spec in candidate_specs:
        outcome = backtest_runner.run(spec)
        assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary
        candidate_backtest_ids.append(outcome.manifest.backtest_id)
    assert len(set(candidate_backtest_ids)) == 2, "the two candidates must have distinct backtest_ids"

    # ---- Section 17 Step 3: one StrategyFamily linking both candidates ----
    family = build_strategy_family(
        candidate_backtest_ids=tuple(candidate_backtest_ids), candidate_experiment_ids=(experiment_id,), candidate_calibration_ids=(calibration_id,),
        search_space_identity="holding_period_bars in {10, 20}", selection_metric="total_net_return",
        eligibility_rules_description="Milestone 6 acceptance workflow -- illustrative 2-candidate family, holding period axis only.",
    )
    assert family.candidate_count == 2

    # ---- Section 17 Step 4: one robustness analysis PER candidate ----
    backtest_manifest_store = BacktestManifestStore(ml_artifacts_root)
    backtest_event_store = BacktestEventStore(ml_artifacts_root)
    calibration_manifest_store = CalibrationManifestStore(ml_artifacts_root)
    experiment_manifest_store = ExperimentManifestStore(ml_artifacts_root)
    execution_manifest_store = ExecutionManifestStore(ml_artifacts_root)
    artifact_store = MLArtifactStore(ml_artifacts_root)
    robustness_manifest_store = RobustnessManifestStore(ml_artifacts_root)

    robustness_runner = RobustnessRunner(
        ml_artifacts_root=ml_artifacts_root, backtest_manifest_store=backtest_manifest_store, backtest_event_store=backtest_event_store,
        calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store, execution_manifest_store=execution_manifest_store,
        research_manifest_store=research_manifest_store, research_dataset_store=research_store, dataset_loader=dataset_loader,
    )

    robustness_ids: list[str] = []
    for source_backtest_id, candidate_backtest_spec in zip(candidate_backtest_ids, candidate_specs, strict=True):
        robustness_spec = RobustnessSpec(
            schema_version=1, source_backtest_id=source_backtest_id, dataset_content_id=candidate_backtest_spec.dataset_content_id,
            split_plan_fingerprint=candidate_backtest_spec.split_plan_fingerprint, instrument_identity=candidate_backtest_spec.instrument_identity,
            bar_interval=candidate_backtest_spec.bar_interval, return_series_kind=ReturnSeriesKind.STITCHED_BAR_NET,
            bootstrap_spec=BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=200, confidence_level=0.9, block_length=5),
            seed=11, multiple_testing_correction=MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG, strategy_family_id=family.family_id,
            minimum_fold_count=2, minimum_trade_count=1, minimum_effective_sample_size=1,
            stability_thresholds=StabilityThresholds(
                minimum_profitable_fold_fraction=0.01, maximum_single_fold_profit_concentration=0.99,
                maximum_single_trade_profit_concentration=0.99, maximum_single_direction_profit_concentration=0.99,
            ),
            stress_scenarios=DEFAULT_STRESS_SCENARIOS, regime_definitions=DEFAULT_REGIME_DEFINITIONS, promotion_policy=PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES),
        )
        identity = compute_robustness_identity(robustness_spec)
        outcome = robustness_runner.run(robustness_spec)
        assert outcome.manifest.robustness_id == identity.robustness_id
        # Section 17 explicitly permits (indeed expects, honestly) a
        # non-COMPLETED terminal-ish outcome for a bounded, small-sample
        # synthetic acceptance run -- what matters is that the pipeline
        # made real, verifiable progress, never that it is "successful"
        # in a trading sense.
        assert outcome.manifest.stage in (RobustnessStage.VERIFIED, RobustnessStage.COMPLETED), (
            f"robustness run for {source_backtest_id!r} stalled at {outcome.manifest.stage.value!r}: {outcome.manifest.failure_summary}"
        )
        robustness_ids.append(outcome.manifest.robustness_id)

        # Independent re-verification must also pass -- this is the SAME
        # check `run()` itself performs at the VERIFIED stage, re-run here
        # explicitly for visibility into what an operator would see from
        # `verify-robustness`.
        validation_report = verify_robustness(
            outcome.manifest.robustness_id, robustness_manifest_store=robustness_manifest_store, artifact_store=artifact_store,
            backtest_manifest_store=backtest_manifest_store, backtest_event_store=backtest_event_store, calibration_manifest_store=calibration_manifest_store,
            experiment_manifest_store=experiment_manifest_store, execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store,
            research_dataset_store=research_store, dataset_loader=dataset_loader,
        )
        assert validation_report.is_ready, [i.to_json_dict() for i in validation_report.criticals]

    # ---- Section 17 Step 5: champion/challenger selection across BOTH real candidates ----
    def _load_named_artifact(manifest, kind: str, decoder):
        ref = manifest.artifact(kind)
        assert ref is not None, f"{manifest.robustness_id!r} missing {kind!r}"
        return decoder(parse_json_strict(artifact_store.read_artifact(ref.content_hash).decode("utf-8")))

    candidate_evidences = []
    for source_backtest_id, robustness_id in zip(candidate_backtest_ids, robustness_ids, strict=True):
        manifest = robustness_manifest_store.load(robustness_id)
        source_verification = _load_named_artifact(manifest, "source_verification_report", SourceVerificationReport.from_json_dict)
        bootstrap_report = _load_named_artifact(manifest, "bootstrap_report", BootstrapReport.from_json_dict)
        stability_report = _load_named_artifact(manifest, "fold_stability_report", FoldStabilityReport.from_json_dict)
        stress_report = _load_named_artifact(manifest, "stress_report", StressReport.from_json_dict)
        sensitivity_report = _load_named_artifact(manifest, "sensitivity_report", SensitivityReport.from_json_dict) if manifest.artifact("sensitivity_report") is not None else None

        source_backtest_manifest = backtest_manifest_store.load(source_backtest_id)
        fold_evidence = load_fold_evidence(source_backtest_manifest.outer_fold_result_references, artifact_store=artifact_store)

        candidate_evidences.append(CandidateEvidence(
            robustness_id=robustness_id, source_backtest_id=source_backtest_id, source_verification=source_verification,
            total_outer_folds=source_verification.total_outer_folds, total_closed_trade_count=len(fold_evidence.all_closed_trades),
            bootstrap=bootstrap_report, fold_stability=stability_report, stress=stress_report, sensitivity=sensitivity_report,
        ))

        # Sanity: this candidate's OWN persisted PromotionDecision exists and is one of the 4 valid kinds.
        promotion_decision = _load_named_artifact(manifest, "promotion_decision", PromotionDecision.from_json_dict)
        assert promotion_decision.decision in PromotionDecisionKind
        assert not hasattr(PromotionDecisionKind, "ELIGIBLE_FOR_LIVE_TRADING")

    selection_report = compute_selection_report(candidates=tuple(candidate_evidences), family_id=family.family_id)
    assert len(selection_report.candidate_eligibility) == 2, "neither real candidate may disappear from the cross-candidate selection report"

    print(f"\nMilestone 6 acceptance workflow: family_id={family.family_id}")
    for evidence, eligibility in zip(candidate_evidences, selection_report.candidate_eligibility, strict=True):
        print(f"  candidate robustness_id={evidence.robustness_id[:12]} eligible={eligibility.eligible} rejection_reasons={eligibility.rejection_reasons}")
    print(f"  selected_candidate_robustness_id={selection_report.selected_candidate_robustness_id}")
