"""Closure-audit Section 10: `verify_robustness`'s call graph and
independence classification. Every recomputed report is produced by the
SAME production function the forward pipeline itself uses (never a
second, parallel, drift-prone implementation) -- `verify_robustness` is
therefore RECOMPUTING / STRUCTURALLY INDEPENDENT (it independently
reconstructs from raw/verified source data and would catch persistence
tampering, non-determinism, and reload/round-trip bugs -- exactly what
caught this session's own order-dependence regression, live, against the
full acceptance-workflow test), but it is NOT algorithmically independent
(a bug baked into a shared `compute_*` function would be recomputed
identically and falsely "verified" as matching). This file mocks every
`compute_*`/`build_*`/`load_*`/`verify_and_load_source_backtest`
dependency `verify_robustness` calls, so each artifact kind's tamper-
detection can be tested in isolation without the expensive real pipeline
-- proving the CONTROL FLOW (does verify_robustness actually load and
compare X) is correct, since the underlying formulas are already
exhaustively hand-verified elsewhere in this suite.

RELEASE-AUDIT REGRESSION: `source_verification_report` was the ONE
required artifact kind `verify_robustness` never actually compared
against its own freshly-recomputed value (every other kind was) --
tampering with it went completely undetected. Fixed by adding the
missing `_load`+`_compare` pair; `TestSourceVerificationReportTampering`
below is the regression coverage."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from tests.unit.robustness.test_source_verification import _backtest_spec, _robustness_spec

from quant_platform.backtesting.manifests import BacktestManifest
from quant_platform.backtesting.models import BacktestStage
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ValidationReport
from quant_platform.robustness.bootstrap import BootstrapEstimate, BootstrapReport, DownsideAnalysisReport
from quant_platform.robustness.manifests import RobustnessManifest, RobustnessManifestStore
from quant_platform.robustness.models import (
    PromotionDecisionKind,
    ReturnSeriesKind,
    RobustnessStage,
)
from quant_platform.robustness.promotion import GateEvaluation as PromotionGateEvaluation
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.selection import (
    CandidateEligibility,
    GateEvaluation,
    GateOutcomeKind,
    RankingEntry,
    SelectionReport,
)
from quant_platform.robustness.series import ReturnSeriesBundle
from quant_platform.robustness.source import SourceVerificationReport, VerifiedBacktestSource
from quant_platform.robustness.stability import ConcentrationReport, FoldStabilityReport
from quant_platform.robustness.stress import StressReport
from quant_platform.robustness.verification import FoldEvidence, verify_robustness

_ROBUSTNESS_ID = "1" * 64
_SOURCE_BACKTEST_ID = "2" * 64


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_verification_report() -> SourceVerificationReport:
    return SourceVerificationReport(
        schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, verify_backtest_is_ready=True, verify_backtest_critical_count=0,
        verify_backtest_issue_codes=(), dataset_content_id_matches=True, split_plan_fingerprint_matches=True, instrument_identity_matches=True,
        bar_interval_matches=True, total_outer_folds=3, generated_at=_now(),
    )


def _series_bundle() -> ReturnSeriesBundle:
    return ReturnSeriesBundle(
        schema_version=1, kind=ReturnSeriesKind.STITCHED_BAR_NET, sampling_frequency="bar", observation_count=2, effective_sample_count=2,
        values=(0.01, -0.005), fold_boundaries=(), source_artifact_content_hashes=("a" * 64,), time_range_start=None, time_range_end=None, built_at=_now(),
    )


def _bootstrap_report() -> BootstrapReport:
    return BootstrapReport(
        schema_version=1, series_kind="stitched_bar_net", method="stationary", repetitions=100, confidence_level=0.95, seed=0,
        estimates=(BootstrapEstimate(statistic_name="total_return", point_estimate=0.02, lower_bound=0.0, upper_bound=0.04, valid_repetitions=100, skipped_repetitions=0, failure_reasons={}),),
        generated_at=_now(),
    )


def _downside_report() -> DownsideAnalysisReport:
    return DownsideAnalysisReport(
        schema_version=1, probability_total_net_return_non_positive=0.1, probability_mean_return_non_positive=0.1, probability_sharpe_non_positive=0.1,
        probability_underperforms_always_flat=None, probability_underperforms_always_long=None, probability_cost_stressed_unprofitable=None,
        probability_maximum_drawdown_exceeds_limit=None, drawdown_limit=None, repetitions=100, seed=0, generated_at=_now(),
    )


def _stability_report() -> FoldStabilityReport:
    concentration = ConcentrationReport(
        schema_version=1, single_fold_profit_concentration=0.5, single_trade_profit_concentration=0.5, single_day_profit_concentration=0.5,
        single_direction_profit_concentration=0.5, single_confidence_bucket_profit_concentration=0.5, warning_codes=(),
    )
    return FoldStabilityReport(
        schema_version=1, fold_count=3, profitable_fold_fraction=0.66, positive_sharpe_fold_fraction=0.66, median_fold_return=0.01,
        worst_fold_return=-0.005, fold_return_stdev=0.01, fold_sharpe_dispersion=0.1, maximum_fold_drawdown=0.05, worst_fold_cost_drag=0.001,
        fold_trade_count_dispersion=1.0, fold_exposure_dispersion=0.1, direction_consistency=0.8, benchmark_outperformance_fraction=0.6,
        concentration=concentration, generated_at=_now(),
    )


def _stress_report() -> StressReport:
    return StressReport(
        schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, baseline_total_net_return=0.02, baseline_total_gross_return=0.025,
        baseline_closed_trade_count=20, baseline_maximum_drawdown=0.05, scenario_results=(), breakeven_results=(), generated_at=_now(),
    )


def _selection_report() -> SelectionReport:
    eligibility = CandidateEligibility(
        robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, eligible=True,
        gate_evaluations=(GateEvaluation(gate_name="minimum_fold_count", mandatory=True, measured_value=3.0, minimum_value=3.0, maximum_value=None, outcome=GateOutcomeKind.PASS, reason=None),),
        rejection_reasons=(),
    )
    return SelectionReport(
        schema_version=1, family_id=None, policy_identity="a" * 64, candidate_eligibility=(eligibility,),
        ranking=(RankingEntry(robustness_id=_ROBUSTNESS_ID, metric_values={"worst_fold_return": -0.005}),),
        tie_break_procedure="deterministic", selected_candidate_robustness_id=_ROBUSTNESS_ID, selection_identity="b" * 64, generated_at=_now(),
    )


def _promotion_decision() -> PromotionDecision:
    return PromotionDecision(
        schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, decision=PromotionDecisionKind.RESEARCH_ONLY,
        decision_reason="advisory gate failed", gate_evaluations=(PromotionGateEvaluation(gate_name="no_parameter_cliff", mandatory=False, measured_value=0.0, minimum_value=1.0, maximum_value=None, outcome=GateOutcomeKind.FAIL, reason="cliff"),),
        disclaimer="not a live-trading recommendation", generated_at=_now(),
    )


def _write(store: MLArtifactStore, obj: object) -> object:
    from quant_platform.ml.persistence import canonical_json_bytes

    return store.write_artifact(canonical_json_bytes(obj.to_json_dict()), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)  # type: ignore[attr-defined]


def _build_rig(tmp_path: object, *, artifact_overrides: dict[str, object] | None = None, spec: object = None) -> dict:
    """Builds a full, self-consistent manifest (every required artifact
    persisted, matching what the mocks below will "recompute") at stage
    PROMOTION_EVALUATED, plus every mock target `verify_robustness`
    depends on. `artifact_overrides` replaces the PERSISTED bytes for the
    named kind(s) before the manifest is built (used to simulate "the
    forward pass itself already persisted a bugged/wrong artifact", as
    opposed to a test-time mock override which only affects the
    RECOMPUTED side). `spec` overrides the persisted `RobustnessSpec`
    itself (default: the standard `_robustness_spec` fixture)."""
    manifest_store = RobustnessManifestStore(f"{tmp_path}/robustness")
    artifact_store = MLArtifactStore(f"{tmp_path}/artifacts")
    spec = spec if spec is not None else _robustness_spec(source_backtest_id=_SOURCE_BACKTEST_ID, backtest_spec=_backtest_spec())
    spec_ref = _write(artifact_store, spec)

    now = _now()
    manifest = RobustnessManifest(
        schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.CREATED,
        created_at=now, updated_at=now, spec_reference=spec_ref,
    )
    manifest_store.create(manifest)

    artifacts: dict[str, object] = {
        "source_verification_report": _source_verification_report(),
        "return_series_bundle": _series_bundle(),
        "bootstrap_report": _bootstrap_report(),
        "downside_analysis_report": _downside_report(),
        "fold_stability_report": _stability_report(),
        "stress_report": _stress_report(),
        "selection_report": _selection_report(),
        "promotion_decision": _promotion_decision(),
    }
    artifacts.update(artifact_overrides or {})
    refs = {kind: _write(artifact_store, obj) for kind, obj in artifacts.items()}
    stage_chain: tuple[tuple[RobustnessStage, tuple[str, ...]], ...] = (
        (RobustnessStage.SOURCE_VERIFIED, ("source_verification_report",)),
        (RobustnessStage.SERIES_BUILT, ("return_series_bundle",)),
        (RobustnessStage.BOOTSTRAP_COMPLETED, ("bootstrap_report", "downside_analysis_report")),
        (RobustnessStage.STABILITY_COMPLETED, ("fold_stability_report",)),
        (RobustnessStage.STRESS_COMPLETED, ("stress_report",)),
        (RobustnessStage.REGIMES_COMPLETED, ()),
        (RobustnessStage.SELECTION_COMPLETED, ("selection_report",)),
        (RobustnessStage.PROMOTION_EVALUATED, ("promotion_decision",)),
    )
    for stage, kinds in stage_chain:
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=stage, updated_at=now, new_named_artifacts=tuple((k, refs[k]) for k in kinds))

    backtest_manifest = BacktestManifest(
        schema_version=1, backtest_id=_SOURCE_BACKTEST_ID, source_calibration_id="a" * 64, stage=BacktestStage.COMPLETED, created_at=now, updated_at=now,
    )
    verified_source = VerifiedBacktestSource(
        manifest=backtest_manifest, backtest_spec=_backtest_spec(), verification_report=ValidationReport(schema_version=1, issues=(), generated_at=now),
        source_verification_report=artifacts["source_verification_report"],  # type: ignore[arg-type]
    )
    fold_evidence = FoldEvidence(fold_results=(), bar_timelines=(), all_closed_trades=(), benchmark_reports=())

    return {
        "manifest_store": manifest_store, "artifact_store": artifact_store, "artifacts": artifacts, "verified_source": verified_source,
        "fold_evidence": fold_evidence,
    }


@pytest.fixture
def rig(tmp_path: object) -> dict:
    return _build_rig(tmp_path)


def _call_verify_robustness(rig: dict, **overrides: object) -> ValidationReport:
    a = rig["artifacts"]
    patches = {
        "verify_and_load_source_backtest": lambda *_args, **_kwargs: overrides.get("source", rig["verified_source"]),
        "build_return_series": lambda *_args, **_kwargs: overrides.get("return_series_bundle", a["return_series_bundle"]),
        "compute_bootstrap_report": lambda *_args, **_kwargs: overrides.get("bootstrap_report", a["bootstrap_report"]),
        "compute_downside_analysis": lambda *_args, **_kwargs: overrides.get("downside_analysis_report", a["downside_analysis_report"]),
        "load_fold_evidence": lambda *_args, **_kwargs: rig["fold_evidence"],
        "compute_fold_stability_report": lambda *_args, **_kwargs: overrides.get("fold_stability_report", a["fold_stability_report"]),
        "resolve_backtest_inputs": lambda *_args, **_kwargs: None,
        "compute_stress_report": lambda *_args, **_kwargs: overrides.get("stress_report", a["stress_report"]),
        "compute_selection_report": lambda *_args, **_kwargs: overrides.get("selection_report", a["selection_report"]),
        "evaluate_promotion": overrides.get("_evaluate_promotion_fn") or (lambda *_args, **_kwargs: overrides.get("promotion_decision", a["promotion_decision"])),
    }
    with patch.multiple("quant_platform.robustness.verification", **patches):
        return verify_robustness(
            _ROBUSTNESS_ID, robustness_manifest_store=rig["manifest_store"], artifact_store=rig["artifact_store"],
            backtest_manifest_store=None, backtest_event_store=None, calibration_manifest_store=None, experiment_manifest_store=None,  # type: ignore[arg-type]
            execution_manifest_store=None, research_manifest_store=None, research_dataset_store=None, dataset_loader=None,  # type: ignore[arg-type]
        )


class TestNoTamperingProducesACleanReport:
    def test_fully_consistent_rig_verifies_with_no_critical_or_error_issues(self, rig: dict) -> None:
        report = _call_verify_robustness(rig)
        assert report.is_ready is True
        assert report.criticals == ()


class TestSharedComputeFunctionBugIsTheHonestBlindSpot:
    """Closure-audit Section 10's explicit request: inject one plausible
    bug into a SHARED compute function and confirm honestly whether
    `verify_robustness` catches it. It CANNOT, by construction: the
    forward pass and the verification pass both call the exact same
    production function (`compute_bootstrap_report` et al.), so a bug
    baked into that shared function produces the SAME wrong answer both
    times, and `_compare`'s equality check sees no mismatch. This is not
    a defect to fix -- `verification.py`'s own module docstring already
    states it recomputes "using the exact same production functions...
    never a second, parallel, drift-prone implementation" -- but it is
    the precise, honest boundary of what this verification layer can and
    cannot prove: it is RECOMPUTING / STRUCTURALLY INDEPENDENT (catches
    persistence tampering, non-determinism, reload/round-trip bugs -- see
    every other test class in this file, and this session's own
    order-dependence regression, which this exact mechanism caught
    live), but it is NOT algorithmically independent (cannot catch a bug
    shared by both the original computation and its own re-verification
    of that computation)."""

    def test_a_bug_present_in_both_the_persisted_artifact_and_the_recomputation_is_not_detected(self, tmp_path: object) -> None:
        """Simulates "the forward pass ran a bugged `compute_bootstrap_
        report` and persisted its (wrong) output; verify_robustness later
        re-runs the SAME bugged function" by building the rig with the
        BUGGED bootstrap_report already persisted from the start (not a
        test-time mock override, which would only affect the recomputed
        side and correctly get caught as a mismatch instead)."""
        baseline = _bootstrap_report()
        bugged_estimate = dataclasses.replace(baseline.estimates[0], point_estimate=-999.0, lower_bound=-999.0, upper_bound=-999.0)
        bugged = dataclasses.replace(baseline, estimates=(bugged_estimate,))
        rig = _build_rig(tmp_path, artifact_overrides={"bootstrap_report": bugged})
        report = _call_verify_robustness(rig, bootstrap_report=bugged)  # recomputation calls the SAME (bugged) logic -> same wrong answer
        assert not any(i.code == "bootstrap_report_mismatch" for i in report.criticals), (
            "a bug shared by both the original computation and its own re-verification is NOT detectable by "
            "recomputation-and-compare -- this is the honest, disclosed limit of this verification layer's independence"
        )


class TestSourceVerificationReportTampering:
    """Regression coverage for the defect found and fixed during this
    audit pass: `source_verification_report` tampering must now be
    detected, exactly like every other artifact kind."""

    def test_tampered_source_verification_report_is_detected(self, rig: dict) -> None:
        tampered_source = dataclasses.replace(rig["verified_source"], source_verification_report=dataclasses.replace(rig["artifacts"]["source_verification_report"], verify_backtest_critical_count=99))
        report = _call_verify_robustness(rig, source=tampered_source)
        assert any(i.code == "source_verification_report_mismatch" for i in report.criticals)


class TestPerArtifactTamperingMatrix:
    """Tamper separately with each artifact kind `verify_robustness`
    handles and confirm each produces its OWN specific mismatch code --
    proving no two artifact kinds share a blind spot."""

    def test_return_series_bundle_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["return_series_bundle"], values=(0.99, 0.99))
        report = _call_verify_robustness(rig, return_series_bundle=tampered)
        assert any(i.code == "return_series_mismatch" for i in report.criticals)

    def test_bootstrap_report_tampering_detected(self, rig: dict) -> None:
        tampered_estimate = dataclasses.replace(rig["artifacts"]["bootstrap_report"].estimates[0], point_estimate=999.0)
        tampered = dataclasses.replace(rig["artifacts"]["bootstrap_report"], estimates=(tampered_estimate,))
        report = _call_verify_robustness(rig, bootstrap_report=tampered)
        assert any(i.code == "bootstrap_report_mismatch" for i in report.criticals)

    def test_downside_analysis_report_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["downside_analysis_report"], probability_total_net_return_non_positive=0.999)
        report = _call_verify_robustness(rig, downside_analysis_report=tampered)
        assert any(i.code == "downside_analysis_mismatch" for i in report.criticals)

    def test_fold_stability_report_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["fold_stability_report"], profitable_fold_fraction=0.01)
        report = _call_verify_robustness(rig, fold_stability_report=tampered)
        assert any(i.code == "fold_stability_mismatch" for i in report.criticals)

    def test_stress_report_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["stress_report"], baseline_total_net_return=999.0)
        report = _call_verify_robustness(rig, stress_report=tampered)
        assert any(i.code == "stress_report_mismatch" for i in report.criticals)

    def test_selection_report_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["selection_report"], selected_candidate_robustness_id=None)
        report = _call_verify_robustness(rig, selection_report=tampered)
        assert any(i.code == "selection_report_mismatch" for i in report.criticals)

    def test_promotion_decision_tampering_detected(self, rig: dict) -> None:
        tampered = dataclasses.replace(rig["artifacts"]["promotion_decision"], decision=PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING)
        report = _call_verify_robustness(rig, promotion_decision=tampered)
        assert any(i.code == "promotion_decision_mismatch" for i in report.criticals)


class TestMissingRequiredArtifactsFailsClosedBeforeAnyComparison:
    def test_missing_spec_reference(self, tmp_path: object) -> None:
        manifest_store = RobustnessManifestStore(f"{tmp_path}/robustness")
        artifact_store = MLArtifactStore(f"{tmp_path}/artifacts")
        now = _now()
        manifest_store.create(RobustnessManifest(schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.CREATED, created_at=now, updated_at=now))
        report = verify_robustness(
            _ROBUSTNESS_ID, robustness_manifest_store=manifest_store, artifact_store=artifact_store,
            backtest_manifest_store=None, backtest_event_store=None, calibration_manifest_store=None, experiment_manifest_store=None,  # type: ignore[arg-type]
            execution_manifest_store=None, research_manifest_store=None, research_dataset_store=None, dataset_loader=None,  # type: ignore[arg-type]
        )
        assert any(i.code == "missing_spec_reference" for i in report.criticals)

    def test_missing_required_artifact_fails_before_source_verification_even_runs(self, rig: dict) -> None:
        """Only the spec exists -- no downstream artifacts. Must fail
        closed on `missing_required_artifacts` WITHOUT ever attempting
        `verify_and_load_source_backtest` (which would otherwise need
        real store dependencies this test never provides)."""
        spec = _robustness_spec(source_backtest_id=_SOURCE_BACKTEST_ID, backtest_spec=_backtest_spec())
        # Build a fresh manifest store/artifact store with ONLY the spec artifact persisted.
        import tempfile

        tmp = tempfile.mkdtemp()
        manifest_store = RobustnessManifestStore(tmp)
        artifact_store = MLArtifactStore(tmp)
        spec_ref = _write(artifact_store, spec)
        now = _now()
        manifest_store.create(RobustnessManifest(schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.CREATED, created_at=now, updated_at=now, spec_reference=spec_ref))
        report = verify_robustness(
            _ROBUSTNESS_ID, robustness_manifest_store=manifest_store, artifact_store=artifact_store,
            backtest_manifest_store=None, backtest_event_store=None, calibration_manifest_store=None, experiment_manifest_store=None,  # type: ignore[arg-type]
            execution_manifest_store=None, research_manifest_store=None, research_dataset_store=None, dataset_loader=None,  # type: ignore[arg-type]
        )
        assert any(i.code == "missing_required_artifacts" for i in report.criticals)


class TestPromotionPolicyWiringRegression:
    """Release-audit regression (closure-audit Section 12/documentation
    audit): a REAL defect found and fixed during this pass.
    `RobustnessSpec.promotion_policy` is validated, hashed into
    `robustness_id`, and persisted, but `RobustnessRunner`/
    `verify_robustness` both previously called `evaluate_promotion` with
    the hardcoded `promotion.DEFAULT_PROMOTION_POLICY`, silently ignoring
    any operator-declared custom policy. Fixed in BOTH call sites (a fix
    in only one would reintroduce the forward/verify divergence class
    this session already found and fixed once). This test proves
    `verify_robustness` now passes `spec.promotion_policy` itself --
    not the module-level default -- to `evaluate_promotion`."""

    def test_verify_robustness_recomputes_promotion_using_the_declared_spec_policy_not_the_hardcoded_default(self, tmp_path: object) -> None:
        from quant_platform.robustness.promotion import DEFAULT_PROMOTION_POLICY
        from quant_platform.robustness.specs import PromotionGateSpec, PromotionPolicySpec

        custom_policy = PromotionPolicySpec(gates=(PromotionGateSpec(name="verified_source_backtest", mandatory=True, minimum_value=1.0),))
        assert custom_policy.to_json_dict() != DEFAULT_PROMOTION_POLICY.to_json_dict()  # sanity: genuinely different from the default

        spec = _robustness_spec(source_backtest_id=_SOURCE_BACKTEST_ID, backtest_spec=_backtest_spec(), promotion_policy=custom_policy)
        rig = _build_rig(tmp_path, spec=spec)

        captured: dict[str, object] = {}

        def _spy_evaluate_promotion(*, evidence: object, policy: object) -> object:
            captured["policy"] = policy
            return rig["artifacts"]["promotion_decision"]

        _call_verify_robustness(rig, _evaluate_promotion_fn=_spy_evaluate_promotion)

        # `verify_robustness` reloads the spec from its own persisted JSON
        # (a fresh, content-equal object, not the same Python instance) --
        # compare by CONTENT, which is what actually matters here.
        assert captured["policy"] == custom_policy  # type: ignore[union-attr]
        assert captured["policy"] != DEFAULT_PROMOTION_POLICY  # type: ignore[union-attr]
