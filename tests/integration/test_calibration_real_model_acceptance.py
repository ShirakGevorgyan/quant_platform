"""Release audit Section 3: a bounded, deterministic, real-model
end-to-end calibration acceptance test -- NOT `ConstantTestModelFactory`.
Uses a real production `logistic_regression` model (registered in
`ml.model_zoo`) against a synthetic dataset with GENUINE injected
predictive structure (an AR(1) latent momentum signal drives forward
returns -- unlike `tests/unit/features/conftest.py::make_synthetic_ohlcv`,
a pure random walk with no injectable signal at all).

This is a SLIMMED, CI-fast companion to the full manual audit run (which
additionally covered LightGBM and a real crash/resume cycle -- see the
delivery report); this permanent test exists so real-model capability
stays regression-protected going forward, not just verified once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.features.conftest import seed_canonical_dataset

from quant_platform.calibration.manifests import CalibrationEventStore, CalibrationManifestStore
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
from quant_platform.calibration.runner import CalibrationRunner, OuterFoldCalibrationResult
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
    compute_calibration_identity,
)
from quant_platform.calibration.verification import verify_calibration
from quant_platform.core.types import Timeframe
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
    """A synthetic OHLCV series with a GENUINE injected AR(1) momentum
    signal driving forward returns -- deliberately NOT a pure random walk
    (unlike `make_synthetic_ohlcv`), so a real model has real signal to
    learn, and calibration is exercised against non-trivial (not 50/50,
    not perfectly separable) real predictive accuracy."""
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
    tmp_path = tmp_path_factory.mktemp("real_model_dataset")
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
    # A scale-homogeneous feature subset (all roughly O(1) or smaller) --
    # `momentum_10`/`rolling_volume_mean_20`/`rolling_volume_std_20`/`atr_14`
    # span 4-5 orders of magnitude larger than the return-based features,
    # which starves `LogisticRegression`'s LBFGS solver of real convergence
    # (this is standard unscaled-feature behavior for a linear model, not a
    # calibration defect -- excluded here rather than papered over with a
    # suppressed ConvergenceWarning, per the audit's "no broad warning
    # suppression" requirement).
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
    return manifest, research_manifest_store, research_store, tmp_path


def _run_real_model_acceptance(signal_dataset, *, model_name: str, model_version: str, hyperparameters: dict, min_mean_accuracy: float) -> None:
    dataset_manifest, research_manifest_store, research_store, tmp_path = signal_dataset
    ml_artifacts_root = tmp_path / f"ml_artifacts_{model_name}"

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
        preprocessing_binding=preprocessing_binding, model_name=model_name, model_version=model_version,
        hyperparameters=ModelHyperparameters(values=hyperparameters), objective=ObjectiveType.BINARY_CLASSIFICATION,
        seed_configuration=SeedConfiguration(master_seed=17), code_revision_binding=CodeRevisionBinding(revision="d" * 40, source="git", is_dirty=False),
        primary_metric="accuracy",
    )
    model_registry = mz.register_default_models()
    preparer = ExperimentPreparer(ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store)
    experiment_manifest = preparer.prepare(experiment_spec)
    assert experiment_manifest.status is ExperimentStatus.READY, experiment_manifest.failure_summary

    model_definition = model_registry.get(model_name, model_version)
    spec = CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id=experiment_manifest.identity.experiment_id, base_model_definition_identity=model_definition.fingerprint(),
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
    runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
        additional_serializers=mz.default_serializer_registry(),
    )

    identity = compute_calibration_identity(spec)
    outcome = runner.run(spec)
    assert outcome.manifest.stage is CalibrationStage.COMPLETED, outcome.manifest.failure_summary
    assert outcome.manifest.calibration_id == identity.calibration_id

    artifact_store = MLArtifactStore(ml_artifacts_root)
    accuracies = []
    for ref in outcome.manifest.outer_fold_result_references.values():
        result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(ref.content_hash).decode("utf-8")))
        assert all(0.0 <= p <= 1.0 for p in result.raw_probabilities)
        assert all(np.isfinite(p) for p in result.calibrated_probabilities)
        assert all(0.0 <= c <= 1.0 for c in result.confidence_scores)
        assert all(0.0 <= u <= 1.0 for u in result.uncertainty_scores)
        accuracy = result.classification_metrics.get("accuracy")
        assert isinstance(accuracy, (int, float))
        accuracies.append(accuracy)

        _serializer, deserializer = mz.default_serializer_registry()[model_definition.serializer_id]
        reloaded = deserializer.deserialize(artifact_store.read_artifact(result.model_reference.content_hash))
        assert 1.0 in reloaded.class_labels

    # The whole point of using a REAL model on data with genuine injected
    # signal: average accuracy across folds must be meaningfully above
    # chance (0.5) -- proving this is not just infrastructure plumbing
    # around a model that predicts nothing.
    assert sum(accuracies) / len(accuracies) > min_mean_accuracy, accuracies

    report = verify_calibration(
        outcome.manifest.calibration_id, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        artifact_store=artifact_store, event_store=CalibrationEventStore(ml_artifacts_root),
    )
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]
    assert any(i.code == "calibrated_probabilities_reproduce" for i in report.infos)

    second = runner.run(spec)
    assert second.was_idempotent_no_op
    assert second.manifest.calibration_id == outcome.manifest.calibration_id


def test_logistic_regression_real_model_end_to_end_calibration(signal_dataset) -> None:
    _run_real_model_acceptance(
        signal_dataset, model_name="logistic_regression", model_version="1",
        hyperparameters={"C": 1.0, "max_iter": 500}, min_mean_accuracy=0.52,
    )


def test_lightgbm_real_model_end_to_end_calibration(signal_dataset) -> None:
    _run_real_model_acceptance(
        signal_dataset, model_name="lightgbm", model_version="1",
        hyperparameters={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1}, min_mean_accuracy=0.55,
    )
