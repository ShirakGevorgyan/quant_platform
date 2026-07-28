"""End-to-end Milestone 4E integration tests: real synthetic historical
data -> a real Milestone 3 research dataset -> a real Milestone 4A
prepared experiment -> a real `CalibrationRunner` leakage-safe
calibration run, using the ACTUAL production stores/registry. Mirrors
`tests/integration/test_optimization_engine.py`'s conventions exactly.

`test_calibration_runner_end_to_end` IS this milestone's bounded
end-to-end acceptance run (Section 41): deterministic synthetic data, a
real immutable source experiment, multiple outer folds, real inner OOF
generation, identity+Platt+isotonic candidates, deterministic selection,
threshold, confidence, uncertainty, abstention, persistence, resume
(exercised separately by the crash-window tests below), and independent
verification (including the recomputation proof) -- an INFRASTRUCTURE
acceptance test, not evidence of market edge (the test-only constant
model predicts nothing meaningful; see `ml.testing`'s own docstring)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

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
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    CalibrationResumeError,
    ExperimentLockError,
)
from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
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
    return manifest, research_manifest_store, research_store


def _build_ready_setup(
    tmp_path: Path, *, seed: int = 42,
) -> tuple[CalibrationSpec, CalibrationRunner, Path, ModelRegistry, ResearchManifestStore, ResearchDatasetStore]:
    """Builds a real synthetic dataset, a real READY experiment (bound to
    the test-only deterministic model), and a valid `CalibrationSpec` +
    `CalibrationRunner` wired to real, on-disk stores. Returns `(spec,
    runner, ml_artifacts_root, model_registry, research_manifest_store,
    research_store)` -- callers decide whether to `.run()`, interrupt it,
    or corrupt artifacts afterward; the trailing three are only needed by
    concurrency tests building a SECOND runner via `_new_runner` (see its
    own docstring for why re-calling this whole function is unsafe)."""
    dataset_manifest, research_manifest_store, research_store = _build_dataset(tmp_path)
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

    spec = CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id=experiment_manifest.identity.experiment_id,
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
    runner = CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
    )
    return spec, runner, ml_artifacts_root, model_registry, research_manifest_store, research_store


def _new_runner(ml_artifacts_root: Path, *, model_registry: ModelRegistry, research_manifest_store: ResearchManifestStore, research_store: ResearchDatasetStore) -> CalibrationRunner:
    """A SECOND, independent `CalibrationRunner` instance over the SAME
    already-built on-disk stores -- for concurrency tests that need two
    genuinely separate runner objects (mirroring two separate processes)
    without re-running `ExperimentPreparer.prepare()`/`ResearchDatasetBuilder.
    build()` a second time (not guaranteed idempotent-safe to repeat)."""
    return CalibrationRunner(
        ml_artifacts_root=ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store,
        research_dataset_store=research_store, experiment_manifest_store=ExperimentManifestStore(ml_artifacts_root),
    )


def _verify(calibration_id: str, ml_artifacts_root: Path):
    return verify_calibration(
        calibration_id, calibration_manifest_store=CalibrationManifestStore(ml_artifacts_root),
        artifact_store=MLArtifactStore(ml_artifacts_root), event_store=CalibrationEventStore(ml_artifacts_root),
    )


def test_calibration_runner_end_to_end(tmp_path: Path) -> None:
    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
    outcome = runner.run(spec)
    assert outcome.manifest.stage is CalibrationStage.COMPLETED, outcome.manifest.failure_summary
    assert not outcome.was_idempotent_no_op
    assert outcome.manifest.total_outer_folds == 2
    assert len(outcome.manifest.completed_outer_fold_indices) == 2

    identity = compute_calibration_identity(spec)
    assert identity.calibration_id == outcome.manifest.calibration_id

    store = CalibrationManifestStore(ml_artifacts_root)
    reloaded = store.load(outcome.manifest.calibration_id)
    assert reloaded.stage is CalibrationStage.COMPLETED

    # Idempotent re-run.
    second = runner.run(spec)
    assert second.was_idempotent_no_op
    assert second.manifest.calibration_id == outcome.manifest.calibration_id

    # Inspect one outer fold's persisted result.
    artifact_store = MLArtifactStore(ml_artifacts_root)
    ref = outcome.manifest.outer_fold_result_references[0]
    raw = artifact_store.read_artifact(ref.content_hash)
    result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    assert result.outer_test_row_count == len(result.sample_positions)
    assert all(0.0 <= c <= 1.0 for c in result.confidence_scores)
    assert all(0.0 <= u <= 1.0 for u in result.uncertainty_scores)

    report = _verify(outcome.manifest.calibration_id, ml_artifacts_root)
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]
    assert any(i.code == "calibrated_probabilities_reproduce" for i in report.infos)

    # Aggregate report: JSON is well-formed and rejects NaN/Infinity by construction.
    assert outcome.manifest.aggregate_report_reference is not None
    report_raw = artifact_store.read_artifact(outcome.manifest.aggregate_report_reference.content_hash)
    report_json = json.loads(report_raw.decode("utf-8"))
    assert "limitations" in report_json and len(report_json["limitations"]) > 0


class _CountingArtifactStoreProxy:
    """Delegates to a real `MLArtifactStore`, raising `ExperimentLockError`
    the instant `write_artifact` call number `crash_after_n_writes + 1` is
    attempted -- simulating a process death at a precise point inside
    `run_outer_fold_calibration`'s own write sequence (inner OOF,
    calibrator selection report, threshold report, decision policy,
    model). Mirrors `test_optimization_engine._CountingArtifactStoreProxy`
    exactly."""

    def __init__(self, real_store, *, crash_after_n_writes: int) -> None:
        self._real_store = real_store
        self._crash_after_n_writes = crash_after_n_writes
        self.write_calls = 0

    def write_artifact(self, *args, **kwargs):
        self.write_calls += 1
        if self.write_calls > self._crash_after_n_writes:
            raise ExperimentLockError(f"simulated crash after {self._crash_after_n_writes} artifact write(s) inside run_outer_fold_calibration")
        return self._real_store.write_artifact(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_store, name)


@pytest.mark.parametrize("crash_after_n_writes", [0, 2, 4])
def test_calibration_runner_resumes_after_mid_fold_crash(tmp_path: Path, crash_after_n_writes: int) -> None:
    """A crash partway through outer fold 0's `run_outer_fold_calibration`
    call must leave the manifest resumable (never at a stage claiming
    completed work it does not have), and resume must redo that ENTIRE
    fold from scratch and reach an identical COMPLETED terminal state --
    never partially resuming mid-fold (see `CalibrationStage`'s own
    docstring for why that is the deliberate design)."""
    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

    import quant_platform.calibration.runner as runner_module
    real_run_outer_fold_calibration = runner_module.run_outer_fold_calibration

    def crashing_run_outer_fold(*, artifact_store, **kwargs):
        proxy = _CountingArtifactStoreProxy(artifact_store, crash_after_n_writes=crash_after_n_writes)
        return real_run_outer_fold_calibration(artifact_store=proxy, **kwargs)

    runner_module.run_outer_fold_calibration = crashing_run_outer_fold
    try:
        with pytest.raises(ExperimentLockError):
            runner.run(spec)
    finally:
        runner_module.run_outer_fold_calibration = real_run_outer_fold_calibration

    calibration_id = compute_calibration_identity(spec).calibration_id
    manifest_after_crash = CalibrationManifestStore(ml_artifacts_root).load(calibration_id)
    assert manifest_after_crash.stage not in (CalibrationStage.COMPLETED, CalibrationStage.FAILED)
    assert manifest_after_crash.completed_outer_fold_indices == ()

    outcome_resumed = runner.resume(calibration_id)
    assert outcome_resumed.manifest.stage is CalibrationStage.COMPLETED, outcome_resumed.manifest.failure_summary
    assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)
    assert outcome_resumed.manifest.resume_count >= 1

    report = _verify(calibration_id, ml_artifacts_root)
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]


def test_genuine_domain_exception_mid_run_leaves_an_accurate_failed_record(tmp_path: Path) -> None:
    """Release audit finding: `CalibrationRunner._fail` mirrors
    `OptimizationRunner._fail`'s exact shape but was never actually
    CALLED anywhere -- `CalibrationStage.FAILED` was unreachable in the
    real implementation, so a genuine mid-run domain error (as opposed to
    the `ExperimentLockError`-simulated process-crash tests elsewhere in
    this file) left the manifest silently stuck at an intermediate stage
    forever, with no persisted reason and no RUN_FAILED event. Now fixed:
    `_run_locked` records `stage=FAILED` (with the exception message as
    `failure_summary`) before re-raising, for any domain exception OTHER
    than `ExperimentLockError` (which represents lock contention/an
    aborted process, never a real "this calibration's data/config is
    wrong" verdict -- see that except clause's own comment)."""
    from quant_platform.core.exceptions import CalibrationFitError

    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

    import quant_platform.calibration.runner as runner_module
    real_run_outer_fold_calibration = runner_module.run_outer_fold_calibration

    def always_fails(**kwargs):
        raise CalibrationFitError("simulated genuine domain failure: e.g. a malformed base model")

    runner_module.run_outer_fold_calibration = always_fails
    try:
        with pytest.raises(CalibrationFitError, match="simulated genuine domain failure"):
            runner.run(spec)
    finally:
        runner_module.run_outer_fold_calibration = real_run_outer_fold_calibration

    calibration_id = compute_calibration_identity(spec).calibration_id
    manifest = CalibrationManifestStore(ml_artifacts_root).load(calibration_id)
    assert manifest.stage is CalibrationStage.FAILED
    assert manifest.failure_summary is not None
    assert "simulated genuine domain failure" in manifest.failure_summary

    events = CalibrationEventStore(ml_artifacts_root).read_events(calibration_id)
    assert events[-1].event_type.value == "run_failed"

    # A FAILED calibration is terminal -- a further run() attempt must
    # raise cleanly (not silently resurrect or hang), and resume() must
    # equally refuse.
    with pytest.raises(CalibrationResumeError):
        runner.run(spec)
    with pytest.raises(CalibrationResumeError):
        runner.resume(calibration_id)


def test_crash_during_post_fold_stage_transition_burst_is_still_resumable(tmp_path: Path) -> None:
    """Release audit Section 10, a DISTINCT crash window from `test_
    calibration_runner_resumes_after_mid_fold_crash` above: that test
    crashes INSIDE `run_outer_fold_calibration` (before any of its 5
    artifacts, or some subset, are written). This test crashes AFTER
    `run_outer_fold_calibration` has already returned successfully (every
    one of its artifacts -- inner OOF, calibrator selection, threshold
    report, decision policy, model -- already exist on disk) but DURING
    `_execute_pipeline`'s subsequent manifest stage-transition burst
    (`CALIBRATORS_EVALUATED` -> ... -> `OUTER_PREDICTIONS_READY`), before
    the `OUTER_FOLD_CALIBRATION_RESULT` bundle is written and before
    `completed_outer_fold_indices` is updated. This is the "manifest
    updated but artifact [bundle] missing" window: sub-artifacts exist,
    but nothing yet claims the fold complete. Resume must still redo the
    WHOLE fold (never partially trust the already-written sub-artifacts)
    and reach an identical terminal state."""
    spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)

    from quant_platform.calibration.manifests import CalibrationManifestStore

    real_transition = CalibrationManifestStore.transition
    call_count = {"n": 0}

    def crashing_transition(self, *args, **kwargs):
        call_count["n"] += 1
        # Allow CREATED->INNER_PREDICTIONS_READY (1st call) through, then
        # crash on the SECOND transition call (CALIBRATORS_EVALUATED) --
        # i.e. strictly after run_outer_fold_calibration has already
        # returned and every one of its 5 artifacts is durably written.
        if call_count["n"] == 2:
            raise ExperimentLockError("simulated crash during the post-fold manifest stage-transition burst")
        return real_transition(self, *args, **kwargs)

    import quant_platform.calibration.manifests as manifests_module

    manifests_module.CalibrationManifestStore.transition = crashing_transition
    try:
        with pytest.raises(ExperimentLockError):
            runner.run(spec)
    finally:
        manifests_module.CalibrationManifestStore.transition = real_transition

    calibration_id = compute_calibration_identity(spec).calibration_id
    manifest_after_crash = CalibrationManifestStore(ml_artifacts_root).load(calibration_id)
    assert manifest_after_crash.stage not in (CalibrationStage.COMPLETED, CalibrationStage.FAILED)
    assert manifest_after_crash.completed_outer_fold_indices == ()

    outcome_resumed = runner.resume(calibration_id)
    assert outcome_resumed.manifest.stage is CalibrationStage.COMPLETED, outcome_resumed.manifest.failure_summary
    assert outcome_resumed.manifest.completed_outer_fold_indices == (0, 1)

    report = _verify(calibration_id, ml_artifacts_root)
    assert report.is_ready, [i.to_json_dict() for i in report.criticals]


class TestCorruptionAndTampering:
    """Section 34: corrupted/tampered artifacts must be detected and
    reported (or rejected outright), never silently trusted."""

    def test_bitflipped_artifact_content_is_detected_on_read(self, tmp_path: Path) -> None:
        """`MLArtifactStore` is content-addressed (SHA-256 keyed) --
        directly overwriting the bytes on disk for a completed
        calibration's artifact must be caught by hash verification on
        the very next read, not silently accepted."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED

        ref = outcome.manifest.outer_fold_result_references[0]
        artifact_store = MLArtifactStore(ml_artifacts_root)
        content_path = ml_artifacts_root / "content" / ref.content_hash[:2] / ref.content_hash
        assert content_path.is_file(), f"expected content file at {content_path}"
        original = content_path.read_bytes()
        tampered = original.replace(b"positive", b"NEGATIVE!", 1) if b"positive" in original else original[:-1] + b"\x00"
        content_path.write_bytes(tampered)

        with pytest.raises(ArtifactCorruptionError):
            artifact_store.read_artifact(ref.content_hash)

    def test_verify_calibration_fails_closed_on_corrupted_outer_fold_artifact(self, tmp_path: Path) -> None:
        """The SAME corruption, surfaced through `verify_calibration`
        rather than a raw `read_artifact` call: must report a CRITICAL
        issue (is_ready=False), never crash uncaught, never silently
        skip the corrupted fold."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED

        ref = outcome.manifest.outer_fold_result_references[0]
        content_path = ml_artifacts_root / "content" / ref.content_hash[:2] / ref.content_hash
        original = content_path.read_bytes()
        content_path.write_bytes(original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01"))

        report = _verify(outcome.manifest.calibration_id, ml_artifacts_root)
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_unverifiable" for i in report.criticals)

    def test_verify_calibration_fails_closed_on_decision_policy_semantic_tampering(self, tmp_path: Path) -> None:
        """The strongest tampering case: a NEW, byte-VALID
        `FrozenDecisionPolicy` artifact (parses cleanly, correct schema,
        correct hash for ITS OWN new content) whose selected calibrator's
        parameters have been altered -- content-hash validity alone is
        insufficient (Section 25); `verify_calibration`'s recomputation
        check must still catch that the persisted calibrated
        probabilities no longer reproduce."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        policy_raw = json.loads(artifact_store.read_artifact(result.decision_policy_reference.content_hash).decode("utf-8"))

        selected_kind = policy_raw["calibrator_selection"]["selected_kind"]
        for candidate in policy_raw["calibrator_selection"]["candidates"]:
            if candidate["kind"] == selected_kind and candidate["fitted"] is not None:
                fitted = candidate["fitted"]
                if fitted["kind"] == "platt":
                    fitted["coefficient"] = fitted["coefficient"] + 5.0
                elif fitted["kind"] == "isotonic":
                    fitted["y_thresholds"] = [min(1.0, y + 0.3) for y in fitted["y_thresholds"]]
                # identity has no parameters to tamper with -- covered by
                # the other two branches across repeated runs/seeds.

        # A NEW, byte-valid artifact under a NEW content hash (tampering
        # a persisted file in place would just be the previous test's
        # hash-mismatch case again -- this one is byte-valid and self-
        # consistent, only semantically wrong).
        tampered_policy_ref = artifact_store.write_artifact(canonical_json_bytes(policy_raw), category=result.decision_policy_reference.category)
        tampered_result = OuterFoldCalibrationResult.from_json_dict(
            {**result.to_json_dict(), "decision_policy_reference": tampered_policy_ref.to_json_dict()}
        )
        tampered_result_ref = artifact_store.write_artifact(canonical_json_bytes(tampered_result.to_json_dict()), category=fold_ref.category)

        # `CalibrationManifest.transition()` only allows FORWARD stage
        # moves (no same-stage update) -- to simulate "the reference on
        # disk was swapped" without inventing a new legal transition,
        # write the manifest JSON directly. This test's whole point is
        # to bypass the runner's own honest write path.
        manifest_path = ml_artifacts_root / "calibrations" / outcome.manifest.calibration_id / "calibration_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_result_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(outcome.manifest.calibration_id, ml_artifacts_root)
        assert not report.is_ready
        assert any(i.code == "calibrated_probabilities_do_not_reproduce" for i in report.criticals)

    def test_verify_calibration_rejects_calibration_id_mismatch_tamper(self, tmp_path: Path) -> None:
        """Release audit Section 7, tamper dimension 9 (calibration_id):
        a hash-VALID `OuterFoldCalibrationResult` whose OWN `calibration_id`
        field has been changed to a different (also valid-looking)
        identity, filed under the ORIGINAL calibration's manifest, must be
        rejected -- proving the manifest<->artifact cross-reference is
        actually checked, not just each side's internal validity."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        tampered = OuterFoldCalibrationResult.from_json_dict({**result.to_json_dict(), "calibration_id": "f" * 64})
        tampered_ref = artifact_store.write_artifact(canonical_json_bytes(tampered.to_json_dict()), category=fold_ref.category)

        manifest_path = ml_artifacts_root / "calibrations" / outcome.manifest.calibration_id / "calibration_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(outcome.manifest.calibration_id, ml_artifacts_root)
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_key_mismatch" for i in report.criticals)

    def test_verify_calibration_rejects_hash_valid_oof_artifact_with_tainted_provenance(self, tmp_path: Path) -> None:
        """Release audit Section 5: a hash-VALID `InnerOofPredictionSet`
        artifact whose `fitted_on_rows` has been tampered to overlap its
        own `sample_positions` (i.e. claims a validation row was also in
        its own model's training set) must be rejected by `verify_
        calibration`, not silently trusted because the bytes are
        self-consistent. Exercises the verification.py fix that now
        DECODES (not just reads) the inner-OOF artifact -- decoding
        re-runs `RawPredictionSet.__post_init__`'s leakage check, which
        previously never ran at verify time because only raw bytes were
        read."""
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        outcome = runner.run(spec)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED
        artifact_store = MLArtifactStore(ml_artifacts_root)

        fold_ref = outcome.manifest.outer_fold_result_references[0]
        result = OuterFoldCalibrationResult.from_json_dict(parse_json_strict(artifact_store.read_artifact(fold_ref.content_hash).decode("utf-8")))
        oof_raw = json.loads(artifact_store.read_artifact(result.inner_oof_reference.content_hash).decode("utf-8"))

        # Taint the first inner fold's provenance: claim its model was
        # ALSO fit on the very first row it predicted -- a genuine
        # leakage violation `RawPredictionSet.__post_init__` structurally
        # rejects at construction time (see test_leakage_adversarial.py's
        # `test_fitted_on_rows_overlapping_sample_positions_is_rejected`)
        # but which a hand-crafted, re-hashed artifact could otherwise
        # smuggle past a bytes-only verification read.
        first_inner = oof_raw["per_inner_fold"][0]
        overlapping_row = first_inner["sample_positions"][0]
        first_inner["fitted_on_rows"] = [*first_inner["fitted_on_rows"], overlapping_row]

        tampered_oof_ref = artifact_store.write_artifact(canonical_json_bytes(oof_raw), category=result.inner_oof_reference.category)
        tampered_result = OuterFoldCalibrationResult.from_json_dict(
            {**result.to_json_dict(), "inner_oof_reference": tampered_oof_ref.to_json_dict()}
        )
        tampered_result_ref = artifact_store.write_artifact(canonical_json_bytes(tampered_result.to_json_dict()), category=fold_ref.category)

        manifest_path = ml_artifacts_root / "calibrations" / outcome.manifest.calibration_id / "calibration_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["outer_fold_result_references"]["0"] = tampered_result_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        report = _verify(outcome.manifest.calibration_id, ml_artifacts_root)
        assert not report.is_ready
        assert any(i.code == "outer_fold_result_dependent_artifact_unverifiable" for i in report.criticals)

    def test_read_artifact_on_a_nonexistent_hash_fails_closed(self, tmp_path: Path) -> None:
        _, _, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        artifact_store = MLArtifactStore(ml_artifacts_root)
        with pytest.raises(ArtifactNotFoundError):
            artifact_store.read_artifact("f" * 64)


class TestResumeEnvironmentCompatibility:
    """Release audit Section 10 ("environment version changed after
    interruption"): `CalibrationRunner._require_compatible_environment`
    (newly added by this audit -- previously `DeterminismPolicy`/
    `CalibrationResumeError`'s own docstrings promised this check but no
    code implemented it) must fail closed under `DeterminismPolicy.STRICT`
    when the installed scikit-learn version differs from the one recorded
    at calibration-creation time, and merely warn-and-proceed under
    `DeterminismPolicy.WARN`."""

    @staticmethod
    def _interrupt_after_first_fold(tmp_path: Path):
        spec, runner, ml_artifacts_root, *_ = _build_ready_setup(tmp_path)
        import quant_platform.calibration.runner as runner_module
        real_run_outer_fold_calibration = runner_module.run_outer_fold_calibration

        call_count = {"n": 0}

        def crash_after_first_fold(**kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise ExperimentLockError("simulated crash before the second outer fold")
            return real_run_outer_fold_calibration(**kwargs)

        runner_module.run_outer_fold_calibration = crash_after_first_fold
        try:
            with pytest.raises(ExperimentLockError):
                runner.run(spec)
        finally:
            runner_module.run_outer_fold_calibration = real_run_outer_fold_calibration
        calibration_id = compute_calibration_identity(spec).calibration_id
        return spec, runner, ml_artifacts_root, calibration_id

    def test_strict_policy_refuses_to_resume_under_a_different_scikit_learn_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _, runner, _, calibration_id = self._interrupt_after_first_fold(tmp_path)

        import importlib.metadata as importlib_metadata

        real_version = importlib_metadata.version

        def fake_version(name: str) -> str:
            if name == "scikit-learn":
                return "0.0.1-simulated-mismatch"
            return real_version(name)

        monkeypatch.setattr(importlib_metadata, "version", fake_version)
        with pytest.raises(CalibrationResumeError, match="scikit-learn"):
            runner.resume(calibration_id)

    def test_warn_policy_proceeds_with_a_different_scikit_learn_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        _, runner, ml_artifacts_root, calibration_id = self._interrupt_after_first_fold(tmp_path)
        # `DeterminismPolicy` is bound into the spec at CREATION time and
        # re-loaded from the calibration's OWN recorded artifact on
        # resume (never re-read from a caller-supplied spec by default) --
        # patch the manifest's recorded spec artifact directly so resume's
        # `_load_spec_artifact` path picks up WARN, exactly like an
        # operator re-authoring their config before a real resume attempt.
        from quant_platform.calibration.manifests import CalibrationManifestStore
        from quant_platform.ml.artifacts import MLArtifactStore

        manifest = CalibrationManifestStore(ml_artifacts_root).load(calibration_id)
        artifact_store = MLArtifactStore(ml_artifacts_root)
        assert manifest.spec_reference is not None
        spec_raw = json.loads(artifact_store.read_artifact(manifest.spec_reference.content_hash).decode("utf-8"))
        spec_raw["determinism_policy"] = "warn"
        new_spec_ref = artifact_store.write_artifact(canonical_json_bytes(spec_raw), category=manifest.spec_reference.category)
        manifest_path = ml_artifacts_root / "calibrations" / calibration_id / "calibration_manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["spec_reference"] = new_spec_ref.to_json_dict()
        write_json_atomic(manifest_path, manifest_json)

        import importlib.metadata as importlib_metadata

        real_version = importlib_metadata.version

        def fake_version(name: str) -> str:
            if name == "scikit-learn":
                return "0.0.1-simulated-mismatch"
            return real_version(name)

        monkeypatch.setattr(importlib_metadata, "version", fake_version)
        with caplog.at_level("WARNING"):
            outcome = runner.resume(calibration_id)
        assert outcome.manifest.stage is CalibrationStage.COMPLETED, outcome.manifest.failure_summary
        assert any("scikit-learn" in record.message for record in caplog.records)


class TestConcurrency:
    """Section 35: deterministic, Windows-compatible concurrency proof --
    never `time.sleep`-based synchronization. Both threads are released
    from a `threading.Barrier` at the precise `os.link` call `DatasetLock.
    acquire()` uses to publish its lock file, the exact technique
    `test_optimization_engine.TestConcurrencyStress` already established
    for the identical race one layer down (optimization's own run-lock).
    Proves: exactly one active owner; the loser fails FAST (never hangs);
    no double publication (one self-consistent COMPLETED manifest,
    independently re-verifiable)."""

    def test_two_simultaneous_run_attempts_for_the_same_calibration_exactly_one_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os
        import threading

        spec, _runner_unused, ml_artifacts_root, model_registry, research_manifest_store, research_store = _build_ready_setup(tmp_path)

        real_link = os.link
        barrier = threading.Barrier(2, timeout=15)
        already_synced = threading.local()

        def synchronized_link(src, dst):
            # A full run() makes MANY os.link calls (the one outer run-
            # lock, plus one per manifest transition) -- only the FIRST
            # per thread (the run-lock race this test targets) is
            # barrier-synchronized; later calls proceed normally, or an
            # already-consumed 2-party barrier would corrupt itself
            # (BrokenBarrierError) instead of testing the intended race.
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[tuple[str, object]] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            # Each thread gets its OWN CalibrationRunner instance (a
            # fresh MLArtifactStore/CalibrationManifestStore wrapper over
            # the SAME on-disk root) -- mirrors two independent processes
            # racing for the same calibration_id, never two threads
            # sharing one Python-level runner object. Reuses the ALREADY-
            # built dataset/experiment rather than re-running
            # ExperimentPreparer.prepare()/ResearchDatasetBuilder.build()
            # a second time (see _new_runner's own docstring for why).
            runner = _new_runner(ml_artifacts_root, model_registry=model_registry, research_manifest_store=research_manifest_store, research_store=research_store)
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
        monkeypatch.setattr(os, "link", real_link)  # restore before any further single-threaded lock use below
        assert not any(t.is_alive() for t in threads), "a losing attempt hung instead of failing fast"

        outcomes = sorted(r[0] for r in results)
        assert outcomes == ["completed", "rejected"], f"expected exactly one winner and one fast-failing loser, got {outcomes}"

        calibration_id = compute_calibration_identity(spec).calibration_id
        reloaded = CalibrationManifestStore(ml_artifacts_root).load(calibration_id)
        assert reloaded.stage is CalibrationStage.COMPLETED

        report = _verify(calibration_id, ml_artifacts_root)
        assert report.is_ready, [i.to_json_dict() for i in report.criticals]
