"""Performance benchmarks for the Milestone 4E leakage-safe calibration,
thresholding, confidence, and uncertainty framework. Same philosophy as
`test_optimization_throughput.py`: conservative floors (roughly 10x-100x
below measured numbers on reference hardware) to catch a severe
accidental regression without being flaky on a slower CI runner -- these
are NOT production throughput guarantees, and no safety check (leakage
validation, artifact hash re-verification, recomputation proof) is ever
skipped to make a number look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%):
  - `select_calibrator` (4 candidates, 500 pooled inner-OOF samples), 100
    iterations: 11.8ms/iter median, ~85 iter/sec.
  - `evaluate_threshold_candidates` (F1, 101-point grid, 500 samples),
    100 iterations: 248ms/iter median, ~4/sec (sklearn per-candidate
    metric calls dominate, not this platform's own grid-search loop).
  - `generate_inner_oof_predictions` (3 inner folds, `ConstantTestModelFactory`,
    400-row outer-train partition), 50 iterations: 5.2ms/iter median,
    ~193 iter/sec.
  - `fit_decision_policy` (calibrator selection + threshold + stability +
    reliability -- internally calls `evaluate_threshold_candidates` once
    pooled plus once per inner fold), 50 iterations: 991ms/iter median,
    ~1/sec.
  - `CalibrationRunner.run` (full 2-outer-fold pipeline, distinct
    calibration every iteration, `constant_test_model`), 10 iterations:
    2.23s/iter median, ~0.4/sec.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.calibration.fitting import (
    fit_decision_policy,
    generate_inner_oof_predictions,
    select_calibrator,
)
from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
)
from quant_platform.calibration.thresholds import evaluate_threshold_candidates
from quant_platform.execution.splitters import Fold
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.inner_splits import InnerSplitConfig

pytestmark = pytest.mark.performance


def _timed_iterations(fn, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> float:
    median = statistics.median(timings)
    rate = 1.0 / median if median > 0 else float("inf")
    print(f"\n{label}: n={len(timings)} median={median * 1000:.3f}ms p95={sorted(timings)[int(len(timings) * 0.95)] * 1000:.3f}ms rate={rate:,.0f}/sec")
    return median


def _correlated_probabilities_and_labels(n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    probabilities = 1.0 / (1.0 + np.exp(-1.5 * latent))
    labels = (rng.uniform(size=n) < probabilities).astype(float)
    labels[0], labels[1] = 0.0, 1.0
    return probabilities, labels


def _spec(seed: int) -> CalibrationSpec:
    return CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id="a" * 64, base_model_definition_identity="constant_test_model:1",
        dataset_content_id="b" * 64, split_plan_fingerprint="c" * 64,
        calibration_method_candidates=(
            CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA,
        ),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=10, minimum_samples_per_class=2,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15, embargo_bars=1),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=101),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.NONE),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy",), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
        seed=seed, determinism_policy=DeterminismPolicy.STRICT,
    )


class TestCalibratorSelectionThroughput:
    def test_select_calibrator(self) -> None:
        """All 4 candidates (identity/Platt/isotonic/beta), 500 pooled
        inner-OOF samples."""
        from quant_platform.calibration.models import RawPredictionSet

        probabilities, labels = _correlated_probabilities_and_labels(500, seed=0)
        spec = _spec(seed=1)
        timestamps = tuple(ts.isoformat() for ts in pd.date_range("2020-01-01", periods=500, freq="1min", tz="UTC"))
        oof = RawPredictionSet(
            schema_version=1, outer_fold_index=0, inner_fold_index=None,
            sample_positions=tuple(range(500)), timestamps=timestamps,
            raw_scores=None, raw_probabilities=tuple(float(v) for v in probabilities), class_labels=(0.0, 1.0), positive_class_index=1,
            source_model_identity="m", source_experiment_id="e", true_labels=tuple(float(v) for v in labels),
        )
        median = _report("select_calibrator (4 candidates, 500 samples)", _timed_iterations(lambda: select_calibrator(oof, spec=spec), 100))
        assert median < 0.5, "selecting among 4 calibrator candidates over 500 samples should not take >500ms (a generous floor)"


class TestThresholdEvaluationThroughput:
    def test_evaluate_threshold_candidates(self) -> None:
        """F1 policy, 101-point grid, 500 samples."""
        probabilities, labels = _correlated_probabilities_and_labels(500, seed=0)
        spec = ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=101)
        median = _report(
            "evaluate_threshold_candidates (F1, 101-point grid, 500 samples)",
            _timed_iterations(lambda: evaluate_threshold_candidates(probabilities, labels, spec=spec), 100),
        )
        assert median < 1.0, "a 101-point F1 threshold grid search over 500 samples should not take >1s (~4x the measured ~249ms floor)"


class TestInnerOofGenerationThroughput:
    def test_generate_inner_oof_predictions(self) -> None:
        """3 inner folds, ConstantTestModelFactory, 400-row outer-train
        partition."""
        n = 500
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        feature_a = rng.normal(size=n)
        labels = (feature_a + rng.normal(scale=0.5, size=n) > 0).astype(float)
        timeline = pd.DataFrame({"open_time": timestamps, "feature_a": feature_a, "label": labels})
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 400), test_indices=np.arange(410, 500),
            train_start=timestamps[0], train_end=timestamps[399], test_start=timestamps[410], test_end=timestamps[499],
        )
        spec = _spec(seed=2)

        def run() -> None:
            generate_inner_oof_predictions(
                outer_fold=fold, timeline=timeline, feature_names=["feature_a"], label_column="label", label_horizon_bars=1,
                model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(values={}),
                objective=ObjectiveType.BINARY_CLASSIFICATION, seed_configuration=SeedConfiguration(master_seed=spec.seed), spec=spec,
                source_model_identity="constant_test_model:1", source_experiment_id=spec.source_experiment_id,
            )

        median = _report("generate_inner_oof_predictions (3 inner folds, 400-row outer-train)", _timed_iterations(run, 50))
        assert median < 1.0, "generating 3 inner folds' OOF predictions over a 400-row partition should not take >1s (a generous floor)"

    def test_fit_decision_policy(self) -> None:
        n = 500
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        feature_a = rng.normal(size=n)
        labels = (feature_a + rng.normal(scale=0.5, size=n) > 0).astype(float)
        timeline = pd.DataFrame({"open_time": timestamps, "feature_a": feature_a, "label": labels})
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 400), test_indices=np.arange(410, 500),
            train_start=timestamps[0], train_end=timestamps[399], test_start=timestamps[410], test_end=timestamps[499],
        )
        spec = _spec(seed=3)
        oof = generate_inner_oof_predictions(
            outer_fold=fold, timeline=timeline, feature_names=["feature_a"], label_column="label", label_horizon_bars=1,
            model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(values={}),
            objective=ObjectiveType.BINARY_CLASSIFICATION, seed_configuration=SeedConfiguration(master_seed=spec.seed), spec=spec,
            source_model_identity="constant_test_model:1", source_experiment_id=spec.source_experiment_id,
        )
        median = _report(
            "fit_decision_policy (calibrator selection + threshold + stability + reliability)",
            _timed_iterations(lambda: fit_decision_policy(oof, spec=spec), 50),
        )
        assert median < 3.5, "fitting the full decision policy (calibrator + threshold + stability) should not take >3.5s (~3.5x the measured ~1s floor)"


class TestCalibrationRunnerThroughput:
    def test_run_distinct_calibrations(self, tmp_path: Path) -> None:
        """A full 2-outer-fold `CalibrationRunner.run()` against a fresh,
        DISTINCT calibration every iteration (never an idempotent
        no-op) -- one research dataset built ONCE (content-addressed,
        written directly through `ResearchDatasetStore`, bypassing the
        full historical/feature-engineering pipeline for speed, exactly
        `test_optimization_throughput.py`'s own established pattern),
        with a fresh experiment (distinct `code_revision_binding`) and
        fresh `CalibrationSpec` (distinct `seed`) prepared per iteration."""
        from tests.unit.execution.conftest import (
            build_registry,
            make_experiment_spec_kwargs,
            write_synthetic_research_dataset,
        )

        from quant_platform.calibration.runner import CalibrationRunner
        from quant_platform.ml.experiment_manager import ExperimentPreparer
        from quant_platform.ml.experiment_spec import ExperimentSpec
        from quant_platform.ml.fingerprints import fingerprint_json
        from quant_platform.ml.manifests import ExperimentManifestStore
        from quant_platform.ml.models import CodeRevisionBinding, LabelBinding, LabelType

        n = 300
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        f1 = rng.normal(size=n)
        binary_timeline = pd.DataFrame({
            "open_time": timestamps, "f1": f1, "f2": rng.normal(size=n),
            "label": (f1 + rng.normal(scale=0.5, size=n) > 0).astype(float),
        })
        dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path, timeline=binary_timeline)

        registry = build_registry(supported_objectives=(ObjectiveType.BINARY_CLASSIFICATION,))
        ml_artifacts_root = tmp_path / "ml"
        preparer = ExperimentPreparer(ml_artifacts_root=ml_artifacts_root, model_registry=registry, research_manifest_store=research_manifest_store)
        runner = CalibrationRunner(
            ml_artifacts_root=ml_artifacts_root, model_registry=registry, research_manifest_store=research_manifest_store,
            research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
        )
        model_definition_fingerprint = registry.get("constant_test_model", "1").fingerprint()
        policy = _spec(seed=0)  # only its policy fields are reused below -- identity fields are per-iteration
        counter = {"i": 0}

        def run() -> None:
            counter["i"] += 1
            exp_spec = ExperimentSpec(**make_experiment_spec_kwargs(  # type: ignore[arg-type]
                dataset_manifest=dataset_manifest, split_params={"n_splits": 2, "test_size": 60, "purge_bars": 5, "embargo_bars": 2},
                code_revision_binding=CodeRevisionBinding(revision=f"{counter['i']:040x}", source="git", is_dirty=True),
                objective=ObjectiveType.BINARY_CLASSIFICATION,
                label_binding=LabelBinding(name="fwd_ret_5", kind="binary_direction", horizon_bars=5, label_type=LabelType.BINARY),
            ))
            experiment_manifest = preparer.prepare(exp_spec)
            spec = CalibrationSpec(
                schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
                source_experiment_id=experiment_manifest.identity.experiment_id, base_model_definition_identity=model_definition_fingerprint,
                dataset_content_id=dataset_manifest.content_id, split_plan_fingerprint=fingerprint_json(exp_spec.split_binding.to_json_dict()),
                calibration_method_candidates=policy.calibration_method_candidates, calibration_selection_metric=policy.calibration_selection_metric,
                calibration_tie_break_policy=policy.calibration_tie_break_policy, minimum_calibration_sample_count=10, minimum_samples_per_class=2,
                inner_oof_policy=policy.inner_oof_policy, threshold_spec=policy.threshold_spec, abstention_spec=policy.abstention_spec,
                confidence_spec=policy.confidence_spec, uncertainty_spec=policy.uncertainty_spec, probability_clipping=policy.probability_clipping,
                reliability_binning_specs=policy.reliability_binning_specs, seed=counter["i"], determinism_policy=policy.determinism_policy,
            )
            runner.run(spec)

        median = _report("CalibrationRunner.run (distinct calibrations, 2 outer folds x 2 inner folds, constant_test_model)", _timed_iterations(run, 10))
        assert median < 8.0, "a full small 2-outer-fold calibration run should not take >8s (~3.6x the measured ~2.2s floor)"
