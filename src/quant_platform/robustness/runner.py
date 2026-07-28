"""Pipeline orchestration (Milestone 6, Section 14). `RobustnessRunner`
drives ONE `RobustnessSpec` through its entire LINEAR pipeline (Sections
3-13), persisting one artifact per stage via `robustness.manifests`, and
resuming from wherever `robustness.resume.verify_completed_robustness_
stages` independently determines the run genuinely reached -- never from
what the manifest merely claims.

Mirrors `backtesting.runner.BacktestRunner`'s `run`/`resume` shape and
its `ExperimentLockError`-excluded, fail-closed `FAILED`-recording
exception handling exactly (see `_run_locked`'s own comment for why that
exclusion matters -- carried forward from the Milestone 4E audit that
found an analogous `_fail` defined but never called elsewhere in this
codebase).

STAGE-TO-ARTIFACT MAPPING: each `if manifest.stage is X:` block below
computes and persists exactly the artifact(s) `resume._STAGE_ARTIFACT_
KINDS[next stage]` expects, then transitions forward -- a resumed run
simply skips every block whose stage has already been passed, falling
through to the first one whose precondition (`manifest.stage is ...`)
still holds. `SENSITIVITY_REPORT` has no dedicated `RobustnessStage`
(the enum has none for Section 9) -- it is computed and persisted
alongside `STRESS_REPORT` during the `STRESS_COMPLETED` transition,
since both reuse `resimulation.py`'s re-simulation mechanism."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifestStore
from quant_platform.backtesting.runner import OuterFoldBacktestResult, resolve_backtest_inputs
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.core.exceptions import ExperimentLockError, QuantPlatformError, RobustnessResumeError
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import (
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    utc_now,
)
from quant_platform.robustness.bootstrap import (
    BootstrapReport,
    DownsideAnalysisReport,
    compute_bootstrap_report,
    compute_downside_analysis,
)
from quant_platform.robustness.manifests import (
    ROBUSTNESS_MANIFEST_SCHEMA_VERSION,
    RobustnessEventStore,
    RobustnessEventType,
    RobustnessManifest,
    RobustnessManifestStore,
)
from quant_platform.robustness.models import RobustnessStage
from quant_platform.robustness.promotion import (
    PromotionDecision,
    PromotionEvidence,
    evaluate_promotion,
)
from quant_platform.robustness.regimes import RegimeReport, compute_regime_report
from quant_platform.robustness.reporting import ROBUSTNESS_REPORT_SCHEMA_VERSION, RobustnessReport
from quant_platform.robustness.resume import require_robustness_resumable, resolve_resume_start_stage
from quant_platform.robustness.selection import (
    DEFAULT_SELECTION_POLICY,
    CandidateEvidence,
    compute_selection_report,
)
from quant_platform.robustness.sensitivity import SensitivityReport, compute_sensitivity_report
from quant_platform.robustness.series import build_return_series
from quant_platform.robustness.source import verify_and_load_source_backtest
from quant_platform.robustness.specs import RobustnessSpec, compute_robustness_identity
from quant_platform.robustness.stability import FoldStabilityReport, compute_fold_stability_report
from quant_platform.robustness.stress import StressReport, compute_stress_report
from quant_platform.robustness.verification import load_fold_evidence, verify_robustness

_RUN_LOCK_FILE_NAME = ".robustness_run.lock"

_T = TypeVar("_T")


class _JsonSerializable(Protocol):
    def to_json_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class RobustnessOutcome:
    manifest: RobustnessManifest
    was_idempotent_no_op: bool


def _mean_turnover(fold_results: tuple[OuterFoldBacktestResult, ...]) -> float | None:
    values: list[float] = []
    for r in fold_results:
        raw = r.financial_metrics.get("turnover_notional_ratio")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values.append(float(raw))
    return (sum(values) / len(values)) if values else None


class RobustnessRunner:
    def __init__(
        self, *, ml_artifacts_root: Path | str, backtest_manifest_store: BacktestManifestStore, backtest_event_store: BacktestEventStore,
        calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
        execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore,
        research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
    ) -> None:
        self._root = Path(ml_artifacts_root).resolve()
        self._artifact_store = MLArtifactStore(self._root)
        self._manifest_store = RobustnessManifestStore(self._root)
        self._event_store = RobustnessEventStore(self._root)
        self._backtest_manifest_store = backtest_manifest_store
        self._backtest_event_store = backtest_event_store
        self._calibration_manifest_store = calibration_manifest_store
        self._experiment_manifest_store = experiment_manifest_store
        self._execution_manifest_store = execution_manifest_store
        self._research_manifest_store = research_manifest_store
        self._research_dataset_store = research_dataset_store
        self._dataset_loader = dataset_loader

    def _run_lock_path(self, robustness_id: str) -> Path:
        return self._root / "robustness" / robustness_id / _RUN_LOCK_FILE_NAME

    def run(self, spec: RobustnessSpec) -> RobustnessOutcome:
        identity = compute_robustness_identity(spec)
        with experiment_lock(self._run_lock_path(identity.robustness_id)):
            return self._run_locked(spec, identity.robustness_id, require_existing=False)

    def resume(self, robustness_id: str, *, spec: RobustnessSpec | None = None) -> RobustnessOutcome:
        existing = self._manifest_store.load_if_exists(robustness_id)
        if existing is not None and existing.stage is RobustnessStage.COMPLETED:
            return RobustnessOutcome(manifest=existing, was_idempotent_no_op=True)
        require_robustness_resumable(existing, robustness_id=robustness_id)
        assert existing is not None
        resolved_spec = spec if spec is not None else self._load_spec_artifact(existing)
        resolved_identity = compute_robustness_identity(resolved_spec)
        if resolved_identity.robustness_id != robustness_id:
            raise RobustnessResumeError(
                f"resume: the provided RobustnessSpec reproduces robustness_id={resolved_identity.robustness_id!r}, which does "
                f"not match the robustness_id being resumed ({robustness_id!r}) -- refusing to resume under a mismatched source identity",
                context={"robustness_id": robustness_id, "resolved_robustness_id": resolved_identity.robustness_id},
            )
        with experiment_lock(self._run_lock_path(robustness_id)):
            return self._run_locked(resolved_spec, robustness_id, require_existing=True)

    def _load_spec_artifact(self, manifest: RobustnessManifest) -> RobustnessSpec:
        if manifest.spec_reference is None:
            raise RobustnessResumeError(f"Robustness run {manifest.robustness_id!r} has no recorded ROBUSTNESS_SPEC artifact to resume from")
        raw = self._artifact_store.read_artifact(manifest.spec_reference.content_hash)
        return RobustnessSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))

    def _run_locked(self, spec: RobustnessSpec, robustness_id: str, *, require_existing: bool) -> RobustnessOutcome:
        manifest = self._manifest_store.load_if_exists(robustness_id)
        if require_existing and manifest is None:
            raise RobustnessResumeError(f"No robustness manifest exists for robustness_id={robustness_id!r}")  # pragma: no cover - guarded identically before locking

        if manifest is not None and manifest.stage in (RobustnessStage.COMPLETED, RobustnessStage.FAILED):
            if manifest.stage is RobustnessStage.COMPLETED:
                return RobustnessOutcome(manifest=manifest, was_idempotent_no_op=True)
            raise RobustnessResumeError(
                f"Robustness run {robustness_id!r} already reached terminal stage {manifest.stage.value!r}",
                context={"robustness_id": robustness_id, "stage": manifest.stage.value},
            )

        if manifest is None:
            now = format_utc_timestamp(utc_now())
            spec_ref = self._artifact_store.write_artifact(canonical_json_bytes(spec.to_json_dict()), category=ArtifactCategory.ROBUSTNESS_SPEC)
            manifest = RobustnessManifest(
                schema_version=ROBUSTNESS_MANIFEST_SCHEMA_VERSION, robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id,
                stage=RobustnessStage.CREATED, created_at=now, updated_at=now, spec_reference=spec_ref, artifact_references=(spec_ref,),
            )
            self._manifest_store.create(manifest)
            self._event_store.append(robustness_id, RobustnessEventType.ROBUSTNESS_CREATED)
            self._event_store.append(robustness_id, RobustnessEventType.RUN_STARTED)
        else:
            start_stage = resolve_resume_start_stage(manifest, artifact_store=self._artifact_store)
            if start_stage is not manifest.stage:
                manifest = self._manifest_store.transition(robustness_id, new_stage=start_stage, updated_at=format_utc_timestamp(utc_now()))
            manifest = self._manifest_store.bump_resume_count(robustness_id)
            self._event_store.append(robustness_id, RobustnessEventType.ROBUSTNESS_RESUMED, details={"resume_count": manifest.resume_count})

        try:
            manifest = self._execute_pipeline(spec, manifest)
        except QuantPlatformError as exc:
            if not isinstance(exc, ExperimentLockError):
                self._fail(robustness_id, str(exc))
            raise
        return RobustnessOutcome(manifest=manifest, was_idempotent_no_op=False)

    def _fail(self, robustness_id: str, failure_summary: str) -> None:
        now = format_utc_timestamp(utc_now())
        self._manifest_store.transition(robustness_id, new_stage=RobustnessStage.FAILED, updated_at=now, completed_at=now, failure_summary=failure_summary)
        self._event_store.append(robustness_id, RobustnessEventType.RUN_FAILED, details={"reason": failure_summary[:200]})

    def _write(self, obj: _JsonSerializable, *, category: ArtifactCategory) -> ArtifactReference:
        return self._artifact_store.write_artifact(canonical_json_bytes(obj.to_json_dict()), category=category)

    def _read(self, manifest: RobustnessManifest, *, kind: str, decoder: Callable[[dict[str, object]], _T]) -> _T:
        reference = manifest.artifact(kind)
        if reference is None:
            raise RobustnessResumeError(f"Robustness run {manifest.robustness_id!r} is missing its {kind!r} artifact -- cannot proceed")
        raw = self._artifact_store.read_artifact(reference.content_hash)
        return decoder(parse_json_strict(raw.decode("utf-8")))

    def _require_artifact(self, manifest: RobustnessManifest, *, kind: str) -> ArtifactReference:
        reference = manifest.artifact(kind)
        if reference is None:
            raise RobustnessResumeError(f"Robustness run {manifest.robustness_id!r} is missing its {kind!r} artifact reference")
        return reference

    def _execute_pipeline(self, spec: RobustnessSpec, manifest: RobustnessManifest) -> RobustnessManifest:
        robustness_id = manifest.robustness_id

        source = verify_and_load_source_backtest(
            spec, backtest_manifest_store=self._backtest_manifest_store, artifact_store=self._artifact_store, event_store=self._backtest_event_store,
            calibration_manifest_store=self._calibration_manifest_store, experiment_manifest_store=self._experiment_manifest_store,
            execution_manifest_store=self._execution_manifest_store, research_manifest_store=self._research_manifest_store,
            research_dataset_store=self._research_dataset_store, dataset_loader=self._dataset_loader,
        )

        if manifest.stage is RobustnessStage.CREATED:
            source_ref = self._write(source.source_verification_report, category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("source_verification_report", source_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.SOURCE_VERIFIED)

        series_bundle = build_return_series(spec.return_series_kind, source=source, artifact_store=self._artifact_store)
        if manifest.stage is RobustnessStage.SOURCE_VERIFIED:
            series_ref = self._write(series_bundle, category=ArtifactCategory.RETURN_SERIES_BUNDLE)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.SERIES_BUILT, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("return_series_bundle", series_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.SERIES_BUILT)

        if manifest.stage is RobustnessStage.SERIES_BUILT:
            bootstrap_report = compute_bootstrap_report(series_bundle, spec=spec.bootstrap_spec, seed=spec.seed)
            downside_report = compute_downside_analysis(series_bundle, spec=spec.bootstrap_spec, seed=spec.seed)
            bootstrap_ref = self._write(bootstrap_report, category=ArtifactCategory.BOOTSTRAP_REPORT)
            downside_ref = self._write(downside_report, category=ArtifactCategory.DOWNSIDE_ANALYSIS_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.BOOTSTRAP_COMPLETED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("bootstrap_report", bootstrap_ref), ("downside_analysis_report", downside_ref)),
            )
            self._event_store.append(robustness_id, RobustnessEventType.BOOTSTRAP_COMPLETED)

        fold_evidence = load_fold_evidence(source.manifest.outer_fold_result_references, artifact_store=self._artifact_store)

        if manifest.stage is RobustnessStage.BOOTSTRAP_COMPLETED:
            stability_report = compute_fold_stability_report(
                fold_evidence.fold_results, all_closed_trades=fold_evidence.all_closed_trades, thresholds=spec.stability_thresholds,
                benchmark_reports=fold_evidence.benchmark_reports,
            )
            stability_ref = self._write(stability_report, category=ArtifactCategory.FOLD_STABILITY_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.STABILITY_COMPLETED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("fold_stability_report", stability_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.STABILITY_COMPLETED)

        resolved_inputs = resolve_backtest_inputs(
            source.backtest_spec, calibration_manifest_store=self._calibration_manifest_store, experiment_manifest_store=self._experiment_manifest_store,
            execution_manifest_store=self._execution_manifest_store, research_manifest_store=self._research_manifest_store,
            research_dataset_store=self._research_dataset_store, dataset_loader=self._dataset_loader, artifact_store=self._artifact_store,
        )

        if manifest.stage is RobustnessStage.STABILITY_COMPLETED:
            sensitivity_report = compute_sensitivity_report(source=source, resolved_inputs=resolved_inputs, spec=spec, artifact_store=self._artifact_store)
            stress_report = compute_stress_report(source=source, resolved_inputs=resolved_inputs, spec=spec, artifact_store=self._artifact_store)
            sensitivity_ref = self._write(sensitivity_report, category=ArtifactCategory.SENSITIVITY_REPORT)
            stress_ref = self._write(stress_report, category=ArtifactCategory.STRESS_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.STRESS_COMPLETED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("sensitivity_report", sensitivity_ref), ("stress_report", stress_ref)),
            )
            self._event_store.append(robustness_id, RobustnessEventType.STRESS_COMPLETED)

        if manifest.stage is RobustnessStage.STRESS_COMPLETED:
            regime_report = compute_regime_report(
                spec=spec, bar_interval=source.backtest_spec.bar_interval, bar_timelines=fold_evidence.bar_timelines,
                bars=resolved_inputs.bars, all_closed_trades=fold_evidence.all_closed_trades,
            )
            regime_ref = self._write(regime_report, category=ArtifactCategory.REGIME_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.REGIMES_COMPLETED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("regime_report", regime_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.REGIMES_COMPLETED)

        # A resumed run skips every block above whose stage already
        # passed -- re-read whatever THIS run cycle didn't just compute,
        # so every step below always has a full, consistent evidence set.
        bootstrap_report = self._read(manifest, kind="bootstrap_report", decoder=BootstrapReport.from_json_dict)
        downside_report = self._read(manifest, kind="downside_analysis_report", decoder=DownsideAnalysisReport.from_json_dict)
        stability_report = self._read(manifest, kind="fold_stability_report", decoder=FoldStabilityReport.from_json_dict)
        stress_report = self._read(manifest, kind="stress_report", decoder=StressReport.from_json_dict)
        sensitivity_report = self._read(manifest, kind="sensitivity_report", decoder=SensitivityReport.from_json_dict)

        if manifest.stage is RobustnessStage.REGIMES_COMPLETED:
            candidate_evidence = CandidateEvidence(
                robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id, source_verification=source.source_verification_report,
                total_outer_folds=len(fold_evidence.fold_results), total_closed_trade_count=len(fold_evidence.all_closed_trades),
                bootstrap=bootstrap_report, fold_stability=stability_report, stress=stress_report, sensitivity=sensitivity_report,
                mean_turnover_notional_ratio=_mean_turnover(fold_evidence.fold_results),
            )
            selection_report = compute_selection_report(candidates=(candidate_evidence,), policy=DEFAULT_SELECTION_POLICY, family_id=spec.strategy_family_id)
            selection_ref = self._write(selection_report, category=ArtifactCategory.SELECTION_REPORT)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.SELECTION_COMPLETED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("selection_report", selection_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.SELECTION_COMPLETED)

        if manifest.stage is RobustnessStage.SELECTION_COMPLETED:
            regime_report = self._read(manifest, kind="regime_report", decoder=RegimeReport.from_json_dict)
            promotion_evidence = PromotionEvidence(
                robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id, source_verification=source.source_verification_report,
                total_outer_folds=len(fold_evidence.fold_results), observation_count=series_bundle.observation_count,
                effective_sample_count=series_bundle.effective_sample_count, total_closed_trade_count=len(fold_evidence.all_closed_trades),
                bootstrap=bootstrap_report, downside=downside_report, fold_stability=stability_report, stress=stress_report,
                sensitivity=sensitivity_report, regime=regime_report,
            )
            # Closure-audit fix: use the DECLARED spec.promotion_policy, not
            # the hardcoded DEFAULT_PROMOTION_POLICY -- see RobustnessSpec.
            # promotion_policy's own docstring for the defect this corrects.
            promotion_decision = evaluate_promotion(evidence=promotion_evidence, policy=spec.promotion_policy)
            promotion_ref = self._write(promotion_decision, category=ArtifactCategory.PROMOTION_DECISION)
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.PROMOTION_EVALUATED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("promotion_decision", promotion_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.PROMOTION_EVALUATED)

        if manifest.stage is RobustnessStage.PROMOTION_EVALUATED:
            validation_report = verify_robustness(
                robustness_id, robustness_manifest_store=self._manifest_store, artifact_store=self._artifact_store,
                backtest_manifest_store=self._backtest_manifest_store, backtest_event_store=self._backtest_event_store,
                calibration_manifest_store=self._calibration_manifest_store, experiment_manifest_store=self._experiment_manifest_store,
                execution_manifest_store=self._execution_manifest_store, research_manifest_store=self._research_manifest_store,
                research_dataset_store=self._research_dataset_store, dataset_loader=self._dataset_loader,
            )
            verification_ref = self._write(validation_report, category=ArtifactCategory.ROBUSTNESS_VERIFICATION_REPORT)
            if not validation_report.is_ready:
                codes = sorted({i.code for i in validation_report.criticals} | {i.code for i in validation_report.errors})
                raise RobustnessResumeError(
                    f"Independent verification failed for robustness_id={robustness_id!r}: {len(validation_report.criticals)} "
                    f"critical issue(s), {len(validation_report.errors)} error(s): {codes}",
                    context={"robustness_id": robustness_id, "issue_codes": ",".join(codes)},
                )
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.VERIFIED, updated_at=format_utc_timestamp(utc_now()),
                new_named_artifacts=(("verification_report", verification_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.RUN_VERIFIED)

        if manifest.stage is RobustnessStage.VERIFIED:
            promotion_decision = self._read(manifest, kind="promotion_decision", decoder=PromotionDecision.from_json_dict)
            robustness_report = RobustnessReport(
                schema_version=ROBUSTNESS_REPORT_SCHEMA_VERSION, robustness_id=robustness_id, source_backtest_id=spec.source_backtest_id,
                source_verification_reference=self._require_artifact(manifest, kind="source_verification_report"),
                return_series_reference=self._require_artifact(manifest, kind="return_series_bundle"),
                bootstrap_report_reference=self._require_artifact(manifest, kind="bootstrap_report"),
                downside_analysis_reference=self._require_artifact(manifest, kind="downside_analysis_report"),
                fold_stability_reference=self._require_artifact(manifest, kind="fold_stability_report"),
                sensitivity_reference=manifest.artifact("sensitivity_report"), stress_reference=self._require_artifact(manifest, kind="stress_report"),
                regime_reference=manifest.artifact("regime_report"), selection_reference=self._require_artifact(manifest, kind="selection_report"),
                promotion_decision_reference=self._require_artifact(manifest, kind="promotion_decision"),
                promotion_decision=promotion_decision.decision, generated_at=format_utc_timestamp(utc_now()),
            )
            report_ref = self._write(robustness_report, category=ArtifactCategory.ROBUSTNESS_REPORT)
            now = format_utc_timestamp(utc_now())
            manifest = self._manifest_store.transition(
                robustness_id, new_stage=RobustnessStage.COMPLETED, updated_at=now, completed_at=now,
                new_named_artifacts=(("robustness_report", report_ref),),
            )
            self._event_store.append(robustness_id, RobustnessEventType.RUN_COMPLETED)

        return manifest


__all__ = ["RobustnessOutcome", "RobustnessRunner"]
