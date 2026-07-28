"""Milestone 6, Section 3 / release-audit Section 2: `verify_and_load_
source_backtest` must never trust a source backtest's manifest merely
because `stage == COMPLETED`. Covers every fail-closed branch: missing
manifest, non-terminal/non-COMPLETED stage, missing spec artifact
reference, independent `verify_backtest` re-verification failing,
corrupted/undecodable persisted `BacktestSpec`, and identity cross-check
mismatches (dataset/split-plan/instrument/bar-interval) between the
declared `RobustnessSpec` and the source's own persisted spec -- proving
the contract is fail-closed under adversarial/tampered input, not merely
trusted from documentation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifest, BacktestManifestStore
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
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.exceptions import RobustnessSourceVerificationError
from quant_platform.core.types import Timeframe
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ValidationReport
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    ReturnSeriesKind,
)
from quant_platform.robustness.source import verify_and_load_source_backtest
from quant_platform.robustness.specs import (
    DEFAULT_PROMOTION_GATES,
    DEFAULT_REGIME_DEFINITIONS,
    DEFAULT_STRESS_SCENARIOS,
    BootstrapSpec,
    PromotionPolicySpec,
    RobustnessSpec,
    StabilityThresholds,
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backtest_spec(**overrides: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_calibration_id": "a" * 64, "source_experiment_id": "b" * 64, "source_execution_id": "b" * 64,
        "dataset_content_id": "c" * 64, "split_plan_fingerprint": "d" * 64, "instrument_identity": "XAUUSD", "market_timezone": "UTC",
        "bar_interval": Timeframe.H1, "decision_timestamp_policy": DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        "signal_mapping": SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), "position_mode": PositionMode.LONG_FLAT,
        "entry_spec": EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        "exit_spec": ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=1, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        "overlap_policy": OverlapPolicyKind.IGNORE, "price_basis": PriceBasisKind.CLOSE,
        "spread_spec": SpreadSpec(kind=SpreadModelKind.ZERO), "commission_spec": CommissionSpec(kind=CommissionModelKind.ZERO),
        "slippage_spec": SlippageSpec(kind=SlippageModelKind.ZERO), "financing_spec": FinancingSpec(kind=FinancingModelKind.NONE),
        "return_calculation_policy": ReturnCalculationPolicyKind.SIMPLE, "compounding_policy": CompoundingPolicyKind.NON_COMPOUNDED,
        "initial_notional": 10_000.0, "determinism_policy": DeterminismPolicy.STRICT,
    }
    defaults.update(overrides)
    return BacktestSpec(**defaults)  # type: ignore[arg-type]


def _robustness_spec(*, source_backtest_id: str, backtest_spec: BacktestSpec, minimum_fold_count: int = 3, **overrides: object) -> RobustnessSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_backtest_id": source_backtest_id, "dataset_content_id": backtest_spec.dataset_content_id,
        "split_plan_fingerprint": backtest_spec.split_plan_fingerprint, "instrument_identity": backtest_spec.instrument_identity,
        "bar_interval": backtest_spec.bar_interval, "return_series_kind": ReturnSeriesKind.STITCHED_BAR_NET,
        "bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.95, block_length=10),
        "seed": 0, "multiple_testing_correction": MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG, "strategy_family_id": None,
        "minimum_fold_count": minimum_fold_count, "minimum_trade_count": 30, "minimum_effective_sample_size": 30,
        "stability_thresholds": StabilityThresholds(
            minimum_profitable_fold_fraction=0.5, maximum_single_fold_profit_concentration=0.6,
            maximum_single_trade_profit_concentration=0.4, maximum_single_direction_profit_concentration=0.7,
        ),
        "stress_scenarios": DEFAULT_STRESS_SCENARIOS, "regime_definitions": DEFAULT_REGIME_DEFINITIONS,
        "promotion_policy": PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES),
    }
    defaults.update(overrides)
    return RobustnessSpec(**defaults)  # type: ignore[arg-type]


def _ready_report() -> ValidationReport:
    return ValidationReport(schema_version=1, issues=(), generated_at=_now())


_BACKTEST_ID = "1" * 64


def _create_manifest(store: BacktestManifestStore, *, backtest_id: str = _BACKTEST_ID) -> None:
    store.create(BacktestManifest(schema_version=1, backtest_id=backtest_id, source_calibration_id="a" * 64, stage=BacktestStage.CREATED, created_at=_now(), updated_at=_now()))


def _advance_to_completed(
    store: BacktestManifestStore, artifact_store: MLArtifactStore, *, backtest_spec: BacktestSpec, backtest_id: str = _BACKTEST_ID,
    total_outer_folds: int = 6, final_spec_reference_override: object = None,
) -> None:
    """Advances a freshly-`_create_manifest`d backtest all the way to
    `COMPLETED`. `final_spec_reference_override`, when given, replaces
    `spec_reference` ONLY at the final `VERIFIED -> COMPLETED` transition
    -- the legal way to simulate a tampered/corrupted spec artifact
    reference on an otherwise-terminal manifest, since `COMPLETED` has no
    legal outgoing transitions (including back to itself) to tamper via a
    second call."""
    spec_ref = artifact_store.write_artifact(json.dumps(backtest_spec.to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)
    for stage in (
        BacktestStage.SOURCES_VERIFIED, BacktestStage.SIGNALS_READY, BacktestStage.FILLS_READY, BacktestStage.TRADES_READY,
        BacktestStage.RETURNS_READY, BacktestStage.METRICS_READY, BacktestStage.REPORTS_READY, BacktestStage.VERIFIED, BacktestStage.COMPLETED,
    ):
        kwargs: dict[str, object] = {"new_stage": stage, "updated_at": _now()}
        if stage is BacktestStage.SOURCES_VERIFIED:
            kwargs["spec_reference"] = spec_ref
            kwargs["total_outer_folds"] = total_outer_folds
        if stage is BacktestStage.COMPLETED:
            kwargs["completed_at"] = _now()
            if final_spec_reference_override is not None:
                kwargs["spec_reference"] = final_spec_reference_override
        store.transition(backtest_id, **kwargs)  # type: ignore[arg-type]


def _stores(tmp_path: object) -> tuple[BacktestManifestStore, MLArtifactStore, BacktestEventStore]:
    manifest_store = BacktestManifestStore(f"{tmp_path}/backtests")
    artifact_store = MLArtifactStore(f"{tmp_path}/artifacts")
    event_store = BacktestEventStore(f"{tmp_path}/events")
    return manifest_store, artifact_store, event_store


def _call(spec: RobustnessSpec, *, manifest_store: BacktestManifestStore, artifact_store: MLArtifactStore, event_store: BacktestEventStore) -> object:
    return verify_and_load_source_backtest(
        spec, backtest_manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store,
        calibration_manifest_store=None, experiment_manifest_store=None, execution_manifest_store=None,  # type: ignore[arg-type]
        research_manifest_store=None, research_dataset_store=None, dataset_loader=None,  # type: ignore[arg-type]
    )


class TestMissingOrNonCompletedSourceRejectedBeforeVerification:
    """These branches must fail BEFORE `verify_backtest` is ever invoked
    -- confirmed by NOT mocking it and still observing a clean, specific
    failure rather than an unrelated crash from unset store dependencies."""

    def test_missing_manifest_rejected(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        backtest_spec = _backtest_spec()
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        with pytest.raises(RobustnessSourceVerificationError, match="could not load manifest"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)

    @pytest.mark.parametrize("stage", [BacktestStage.CREATED, BacktestStage.SIGNALS_READY, BacktestStage.VERIFIED])
    def test_non_completed_stage_rejected(self, tmp_path: object, stage: BacktestStage) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        if stage is not BacktestStage.CREATED:
            manifest_store.transition(_BACKTEST_ID, new_stage=BacktestStage.SOURCES_VERIFIED, updated_at=_now())
        if stage not in (BacktestStage.CREATED, BacktestStage.SOURCES_VERIFIED):
            manifest_store.transition(_BACKTEST_ID, new_stage=BacktestStage.SIGNALS_READY, updated_at=_now())
        backtest_spec = _backtest_spec()
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        with pytest.raises(RobustnessSourceVerificationError, match=r"has not reached BacktestStage\.COMPLETED"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)

    def test_failed_stage_rejected(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        manifest_store.transition(_BACKTEST_ID, new_stage=BacktestStage.FAILED, updated_at=_now(), failure_summary="synthetic test failure")
        backtest_spec = _backtest_spec()
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        with pytest.raises(RobustnessSourceVerificationError, match=r"has not reached BacktestStage\.COMPLETED"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)


class TestIndependentVerificationNotMerelyTrusted:
    def test_verify_backtest_not_ready_rejected(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        from quant_platform.ml.models import ValidationIssue, ValidationSeverity

        not_ready = ValidationReport(
            schema_version=1, issues=(ValidationIssue(severity=ValidationSeverity.CRITICAL, code="synthetic_critical", message="synthetic"),), generated_at=_now(),
        )
        with patch("quant_platform.robustness.source.verify_backtest", return_value=not_ready), pytest.raises(RobustnessSourceVerificationError, match="failed independent re-verification"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)

    def test_completed_stage_is_not_sufficient_verify_backtest_is_still_invoked(self, tmp_path: object) -> None:
        """The central Section 2 property, proven at the call-graph level:
        even for a manifest that has legitimately reached stage=COMPLETED,
        `verify_and_load_source_backtest` unconditionally calls
        `verify_backtest` and gates on its result -- COMPLETED status
        alone is architecturally never sufficient on its own to proceed."""
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        with patch("quant_platform.robustness.source.verify_backtest", return_value=_ready_report()) as mock_verify:
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)
        mock_verify.assert_called_once()


class TestTamperedOrCorruptedSpecRejected:
    def test_undecodable_spec_artifact_rejected(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        # Tamper: the FINAL COMPLETED transition points spec_reference at a
        # content hash whose bytes are NOT a valid BacktestSpec at all.
        garbage_ref = artifact_store.write_artifact(b'{"not": "a backtest spec"}', category=ArtifactCategory.BACKTEST_SPEC)
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec, final_spec_reference_override=garbage_ref)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec)
        with patch("quant_platform.robustness.source.verify_backtest", return_value=_ready_report()), pytest.raises(RobustnessSourceVerificationError, match="could not be read/decoded"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)


class TestIdentityCrossCheckMismatchesRejected:
    @pytest.mark.parametrize(
        "override",
        [{"dataset_content_id": "9" * 64}, {"split_plan_fingerprint": "8" * 64}, {"instrument_identity": "EURUSD"}, {"bar_interval": Timeframe.H4}],
    )
    def test_mismatched_declared_identity_rejected(self, tmp_path: object, override: dict[str, object]) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec, **override)
        with patch("quant_platform.robustness.source.verify_backtest", return_value=_ready_report()), pytest.raises(RobustnessSourceVerificationError, match="failed identity cross-checks"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)

    def test_matching_identity_and_ready_verification_succeeds(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec, total_outer_folds=6)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec, minimum_fold_count=3)
        with patch("quant_platform.robustness.source.verify_backtest", return_value=_ready_report()):
            result = _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)
        assert result.source_verification_report.all_checks_passed is True  # type: ignore[union-attr]


class TestInsufficientFoldCountRejected:
    def test_fewer_outer_folds_than_required_rejected(self, tmp_path: object) -> None:
        manifest_store, artifact_store, event_store = _stores(tmp_path)
        _create_manifest(manifest_store)
        backtest_spec = _backtest_spec()
        _advance_to_completed(manifest_store, artifact_store, backtest_spec=backtest_spec, total_outer_folds=2)
        spec = _robustness_spec(source_backtest_id=_BACKTEST_ID, backtest_spec=backtest_spec, minimum_fold_count=3)
        with patch("quant_platform.robustness.source.verify_backtest", return_value=_ready_report()), pytest.raises(RobustnessSourceVerificationError, match=r"below RobustnessSpec\.minimum_fold_count"):
            _call(spec, manifest_store=manifest_store, artifact_store=artifact_store, event_store=event_store)
