"""Independent verification (Milestone 6, Section 14). `verify_robustness`
NEVER trusts a persisted `BootstrapReport`/`FoldStabilityReport`/
`StressReport`/`SensitivityReport`/`RegimeReport`/`SelectionReport`/
`PromotionDecision` at face value -- it independently RECOMPUTES every
one of them from the verified source backtest's own raw artifacts, the
declared `RobustnessSpec`, and the declared random seed (using the exact
same production functions this package's own runner uses -- never a
second, parallel, drift-prone implementation), then compares the fresh
result against what was persisted. A mismatch on ANY field other than a
wall-clock timestamp is a CRITICAL finding: it means either the artifact
store was tampered with, or the recorded `RobustnessSpec`/seed do not
actually reproduce what was persisted.

`load_fold_evidence` is exported and reused UNCHANGED by `robustness.
runner` -- the forward pipeline and this verification path load fold-
level evidence (bar timelines, closed trades, benchmark reports) through
the exact same function, so there is no risk of the two silently
diverging in what "the fold evidence" even means."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifestStore
from quant_platform.backtesting.runner import (
    BenchmarkReport,
    OuterFoldBacktestResult,
    resolve_backtest_inputs,
)
from quant_platform.backtesting.timeline import BarReturnTimeline
from quant_platform.backtesting.trades import TradeRecord, TradeSet
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactReference, ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp, parse_json_strict, utc_now
from quant_platform.robustness.bootstrap import compute_bootstrap_report, compute_downside_analysis
from quant_platform.robustness.manifests import RobustnessManifestStore
from quant_platform.robustness.models import RobustnessStage
from quant_platform.robustness.promotion import (
    PromotionEvidence,
    evaluate_promotion,
)
from quant_platform.robustness.regimes import compute_regime_report
from quant_platform.robustness.selection import (
    DEFAULT_SELECTION_POLICY,
    CandidateEvidence,
    compute_selection_report,
)
from quant_platform.robustness.sensitivity import compute_sensitivity_report
from quant_platform.robustness.series import build_return_series
from quant_platform.robustness.source import SourceVerificationReport, verify_and_load_source_backtest
from quant_platform.robustness.specs import RobustnessSpec
from quant_platform.robustness.stability import compute_fold_stability_report
from quant_platform.robustness.stress import compute_stress_report

_VERIFICATION_REPORT_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True, slots=True)
class FoldEvidence:
    fold_results: tuple[OuterFoldBacktestResult, ...]
    bar_timelines: tuple[BarReturnTimeline, ...]
    all_closed_trades: tuple[TradeRecord, ...]
    benchmark_reports: tuple[BenchmarkReport, ...]


def load_fold_evidence(outer_fold_result_references: Mapping[int, ArtifactReference], *, artifact_store: MLArtifactStore) -> FoldEvidence:
    fold_results: list[OuterFoldBacktestResult] = []
    bar_timelines: list[BarReturnTimeline] = []
    all_closed_trades: list[TradeRecord] = []
    benchmark_reports: list[BenchmarkReport] = []
    for _fold_index, reference in sorted(outer_fold_result_references.items()):
        raw = artifact_store.read_artifact(reference.content_hash)
        result = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        fold_results.append(result)
        bt_raw = artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash)
        bar_timelines.append(BarReturnTimeline.from_json_dict(parse_json_strict(bt_raw.decode("utf-8"))))
        ts_raw = artifact_store.read_artifact(result.trade_set_reference.content_hash)
        trade_set = TradeSet.from_json_dict(parse_json_strict(ts_raw.decode("utf-8")))
        all_closed_trades.extend(trade_set.closed_trades)
        bench_raw = artifact_store.read_artifact(result.benchmark_report_reference.content_hash)
        benchmark_reports.append(BenchmarkReport.from_json_dict(parse_json_strict(bench_raw.decode("utf-8"))))
    return FoldEvidence(
        fold_results=tuple(fold_results), bar_timelines=tuple(bar_timelines), all_closed_trades=tuple(all_closed_trades),
        benchmark_reports=tuple(benchmark_reports),
    )


def _mean_turnover(fold_results: tuple[OuterFoldBacktestResult, ...]) -> float | None:
    values = []
    for r in fold_results:
        raw = r.financial_metrics.get("turnover_notional_ratio")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values.append(float(raw))
    return (sum(values) / len(values)) if values else None


def _strip_timestamp(obj: object, *, field_name: str) -> object:
    return dataclasses.replace(obj, **{field_name: "0"})  # type: ignore[type-var]


def _compare(issues: list[ValidationIssue], *, code: str, label: str, persisted: object, recomputed: object, timestamp_field: str) -> None:
    normalized_persisted = _strip_timestamp(persisted, field_name=timestamp_field)
    normalized_recomputed = _strip_timestamp(recomputed, field_name=timestamp_field)
    if normalized_persisted != normalized_recomputed:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.CRITICAL, code=code,
            message=f"{label}: independently recomputed result does not match the persisted artifact -- the artifact store may be "
                    "tampered with, or the recorded RobustnessSpec/seed do not reproduce what was persisted",
        ))


def verify_robustness(
    robustness_id: str, *, robustness_manifest_store: RobustnessManifestStore, artifact_store: MLArtifactStore,
    backtest_manifest_store: BacktestManifestStore, backtest_event_store: BacktestEventStore,
    calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
    execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore,
    research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    manifest = robustness_manifest_store.load(robustness_id)
    if manifest.spec_reference is None:
        issues.append(ValidationIssue(severity=ValidationSeverity.CRITICAL, code="missing_spec_reference", message="RobustnessManifest has no recorded ROBUSTNESS_SPEC artifact"))
        return ValidationReport(schema_version=_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))
    spec_raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
    spec = RobustnessSpec.from_json_dict(parse_json_strict(spec_raw.decode("utf-8")))

    required_kinds = ("source_verification_report", "return_series_bundle", "bootstrap_report", "downside_analysis_report", "fold_stability_report", "stress_report", "selection_report", "promotion_decision")
    missing = [k for k in required_kinds if manifest.artifact(k) is None]
    if missing:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.CRITICAL, code="missing_required_artifacts",
            message=f"RobustnessManifest is missing required artifact(s) for independent verification: {sorted(missing)}",
        ))
        return ValidationReport(schema_version=_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))

    source = verify_and_load_source_backtest(
        spec, backtest_manifest_store=backtest_manifest_store, artifact_store=artifact_store, event_store=backtest_event_store,
        calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
        execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store,
        research_dataset_store=research_dataset_store, dataset_loader=dataset_loader,
    )

    def _load(kind: str, decoder: type) -> object:
        ref = manifest.artifact(kind)
        assert ref is not None
        raw = artifact_store.read_artifact(ref.content_hash)
        return decoder.from_json_dict(parse_json_strict(raw.decode("utf-8")))  # type: ignore[attr-defined]

    from quant_platform.robustness.bootstrap import BootstrapReport, DownsideAnalysisReport
    from quant_platform.robustness.promotion import PromotionDecision
    from quant_platform.robustness.regimes import RegimeReport
    from quant_platform.robustness.selection import SelectionReport
    from quant_platform.robustness.sensitivity import SensitivityReport
    from quant_platform.robustness.series import ReturnSeriesBundle
    from quant_platform.robustness.stability import FoldStabilityReport
    from quant_platform.robustness.stress import StressReport

    # Closure-audit finding (Section 10): every OTHER required artifact
    # kind below is loaded and compared against a freshly recomputed
    # value via `_compare` -- `source_verification_report` was the one
    # exception: `required_kinds` above demands it be PRESENT, but its
    # CONTENT was never compared against `source.source_verification_
    # report` (already freshly recomputed by `verify_and_load_source_
    # backtest` just above). This meant tampering with the PERSISTED
    # source_verification_report artifact specifically (e.g. flipping
    # `dataset_content_id_matches` from False to True, or zeroing
    # `verify_backtest_critical_count`) went completely undetected by
    # this function, silently contradicting its own module docstring's
    # universal claim ("A mismatch on ANY field... is a CRITICAL
    # finding"). Downstream promotion evidence already correctly used the
    # FRESH `source.source_verification_report` (never the persisted,
    # possibly-tampered one), so this gap never caused a wrong promotion
    # decision -- but it did mean the persisted artifact itself, and this
    # function's own verification report, could misrepresent a tampered
    # source-verification record as unverified/unflagged.
    persisted_source_verification = _load("source_verification_report", SourceVerificationReport)
    _compare(
        issues, code="source_verification_report_mismatch", label="SourceVerificationReport",
        persisted=persisted_source_verification, recomputed=source.source_verification_report, timestamp_field="generated_at",
    )

    persisted_series = _load("return_series_bundle", ReturnSeriesBundle)
    recomputed_series = build_return_series(spec.return_series_kind, source=source, artifact_store=artifact_store)
    _compare(issues, code="return_series_mismatch", label="ReturnSeriesBundle", persisted=persisted_series, recomputed=recomputed_series, timestamp_field="built_at")

    persisted_bootstrap = _load("bootstrap_report", BootstrapReport)
    recomputed_bootstrap = compute_bootstrap_report(recomputed_series, spec=spec.bootstrap_spec, seed=spec.seed)
    _compare(issues, code="bootstrap_report_mismatch", label="BootstrapReport", persisted=persisted_bootstrap, recomputed=recomputed_bootstrap, timestamp_field="generated_at")

    persisted_downside = _load("downside_analysis_report", DownsideAnalysisReport)
    recomputed_downside = compute_downside_analysis(recomputed_series, spec=spec.bootstrap_spec, seed=spec.seed)
    _compare(issues, code="downside_analysis_mismatch", label="DownsideAnalysisReport", persisted=persisted_downside, recomputed=recomputed_downside, timestamp_field="generated_at")

    fold_evidence = load_fold_evidence(source.manifest.outer_fold_result_references, artifact_store=artifact_store)

    persisted_stability = _load("fold_stability_report", FoldStabilityReport)
    recomputed_stability = compute_fold_stability_report(
        fold_evidence.fold_results, all_closed_trades=fold_evidence.all_closed_trades, thresholds=spec.stability_thresholds,
        benchmark_reports=fold_evidence.benchmark_reports,
    )
    _compare(issues, code="fold_stability_mismatch", label="FoldStabilityReport", persisted=persisted_stability, recomputed=recomputed_stability, timestamp_field="generated_at")

    resolved_inputs = resolve_backtest_inputs(
        source.backtest_spec, calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
        execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store,
        research_dataset_store=research_dataset_store, dataset_loader=dataset_loader, artifact_store=artifact_store,
    )

    recomputed_sensitivity = None
    if manifest.artifact("sensitivity_report") is not None:
        persisted_sensitivity = _load("sensitivity_report", SensitivityReport)
        recomputed_sensitivity = compute_sensitivity_report(source=source, resolved_inputs=resolved_inputs, spec=spec, artifact_store=artifact_store)
        _compare(issues, code="sensitivity_report_mismatch", label="SensitivityReport", persisted=persisted_sensitivity, recomputed=recomputed_sensitivity, timestamp_field="generated_at")

    persisted_stress = _load("stress_report", StressReport)
    recomputed_stress = compute_stress_report(source=source, resolved_inputs=resolved_inputs, spec=spec, artifact_store=artifact_store)
    _compare(issues, code="stress_report_mismatch", label="StressReport", persisted=persisted_stress, recomputed=recomputed_stress, timestamp_field="generated_at")

    if manifest.artifact("regime_report") is not None:
        persisted_regime = _load("regime_report", RegimeReport)
        recomputed_regime = compute_regime_report(
            spec=spec, bar_interval=source.backtest_spec.bar_interval, bar_timelines=fold_evidence.bar_timelines,
            bars=resolved_inputs.bars, all_closed_trades=fold_evidence.all_closed_trades,
        )
        _compare(issues, code="regime_report_mismatch", label="RegimeReport", persisted=persisted_regime, recomputed=recomputed_regime, timestamp_field="generated_at")

    candidate_evidence = CandidateEvidence(
        robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id, source_verification=source.source_verification_report,
        total_outer_folds=len(fold_evidence.fold_results), total_closed_trade_count=len(fold_evidence.all_closed_trades),
        bootstrap=recomputed_bootstrap, fold_stability=recomputed_stability, stress=recomputed_stress, sensitivity=recomputed_sensitivity,
        mean_turnover_notional_ratio=_mean_turnover(fold_evidence.fold_results),
    )
    persisted_selection = _load("selection_report", SelectionReport)
    recomputed_selection = compute_selection_report(candidates=(candidate_evidence,), policy=DEFAULT_SELECTION_POLICY, family_id=spec.strategy_family_id)
    _compare(issues, code="selection_report_mismatch", label="SelectionReport", persisted=persisted_selection, recomputed=recomputed_selection, timestamp_field="generated_at")

    promotion_evidence = PromotionEvidence(
        robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id, source_verification=source.source_verification_report,
        total_outer_folds=len(fold_evidence.fold_results), observation_count=recomputed_series.observation_count,
        effective_sample_count=recomputed_series.effective_sample_count, total_closed_trade_count=len(fold_evidence.all_closed_trades),
        bootstrap=recomputed_bootstrap, downside=recomputed_downside, fold_stability=recomputed_stability, stress=recomputed_stress,
        sensitivity=recomputed_sensitivity, regime=(recomputed_regime if manifest.artifact("regime_report") is not None else None),
    )
    persisted_promotion = _load("promotion_decision", PromotionDecision)
    # Closure-audit fix: must recompute with the SAME policy the forward
    # pass actually used (spec.promotion_policy) -- using the hardcoded
    # DEFAULT_PROMOTION_POLICY here while the forward pass now correctly
    # uses spec.promotion_policy would reintroduce exactly the forward/
    # verify divergence class this session already found and fixed once
    # (Section 1's RobustnessSpec identity round-trip regression).
    recomputed_promotion = evaluate_promotion(evidence=promotion_evidence, policy=spec.promotion_policy)
    _compare(issues, code="promotion_decision_mismatch", label="PromotionDecision", persisted=persisted_promotion, recomputed=recomputed_promotion, timestamp_field="generated_at")

    if manifest.stage not in (RobustnessStage.PROMOTION_EVALUATED, RobustnessStage.VERIFIED, RobustnessStage.COMPLETED):
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR, code="premature_verification",
            message=f"verify_robustness was called before the pipeline reached PROMOTION_EVALUATED (current stage={manifest.stage.value!r})",
        ))

    return ValidationReport(schema_version=_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


__all__ = ["FoldEvidence", "load_fold_evidence", "verify_robustness"]
