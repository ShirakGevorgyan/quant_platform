"""`verify_backtest`: an independent, read-only re-audit of everything a
backtest run has recorded -- its own `BacktestManifest`, every outer-fold
result artifact and the sub-artifacts it references, and the append-only
event log -- proving they still agree with each other. Mirrors
`calibration.verification.verify_calibration`'s exact philosophy and
structure.

HASH VALIDITY IS NOT ENOUGH: THE RECOMPUTATION CHECK (Section 30)
--------------------------------------------------------------------------
`_verify_fold_financials_reproduce` re-derives the equity curve, gross/net
drawdown reports, and financial metrics DIRECTLY from the persisted
`TradeSet` and persisted `BarReturnTimeline` alone, and asserts every
recomputed scalar metric matches the persisted `OuterFoldBacktestResult.
financial_metrics` value (within float tolerance). Combined with
`TradeRecord.__post_init__`/`CostBreakdown.from_json_dict`'s own self-
verifying `trade_id`/`total_cost` recomputation (which runs automatically
on every decode, not as a separate opt-in step here), this catches
Section 30's semantic-tampering examples where `financial_metrics` was
edited independently of the trades/timeline it claims to summarize.

MILESTONE 5.2, SECTION 3: RAW-SOURCE RECONSTRUCTION -- A VALID CONTENT
HASH MUST NEVER BE SUFFICIENT
--------------------------------------------------------------------------
`_verify_fold_financials_reproduce` alone is insufficient against an
attacker who tampers a `BarReturnTimeline` (and the `financial_metrics`
derived from it) CONSISTENTLY -- both would still "reproduce" each other,
and the persisted content hash would still validate, because the check
never leaves this backtest's own artifact tree. `_verify_raw_source_
reconstruction` closes this gap: it independently re-resolves the FULL
source context this backtest was originally run against (the source
calibration/experiment/execution manifests, the research dataset
timeline, and raw market bars -- via `runner.resolve_backtest_inputs`,
the exact function the original run used) and independently RECOMPUTES
every outer fold's signals, trades, bar-level timeline, equity, drawdown,
financial metrics, benchmarks, cost-sensitivity report, and bucket
analysis from those raw sources using the SAME production code path the
original run used (`runner.recompute_outer_fold_backtest_artifacts`,
never a re-implemented, drift-prone parallel computation) -- then
compares the rebuild to what was actually persisted, field by field for
the `BarReturnTimeline` and by content hash for every other artifact. It
then independently rebuilds the stitched walk-forward equity from the
raw-source-reconstructed per-fold timelines (not the persisted ones) and
the aggregate report's substantive sections from the raw-source-verified
per-fold results. Because the rebuild never reads the persisted timeline
or persisted metrics at any point, a consistently-tampered timeline,
metrics, AND content hash (Section 3's explicit threat model) cannot pass
undetected -- the rebuild is derived entirely from data outside this
backtest's own artifact tree.

WHAT THIS MODULE STILL CANNOT INDEPENDENTLY VERIFY
--------------------------------------------------------------------------
It does not re-verify the SOURCE calibration's own model-fitting/
calibration-fitting correctness (only that its persisted calibrator
reproduces its own persisted calibrated probabilities, via `runner.
verify_and_load_predictions`'s recomputation check, which now genuinely
runs INSIDE this module's raw-source reconstruction rather than merely
being documented as "already ran once, trusted here"). Raw-source
reconstruction also has a new operational requirement it did not
previously have: the source calibration, source experiment, source
execution, research dataset, and raw historical market data must all
still be present and resolvable through the stores/loader passed to
`verify_backtest` -- if any of those have since been garbage-collected or
moved, `_verify_raw_source_reconstruction` fails closed (a CRITICAL
`raw_source_resolution_failed`/`raw_source_recomputation_failed` issue),
even for a backtest whose own persisted artifacts are perfectly
consistent. This is a deliberate, documented tradeoff (see the Milestone
5.2 delivery report's remaining-limitations section), not an oversight."""

from __future__ import annotations

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.manifests import (
    BacktestEventStore,
    BacktestEventType,
    BacktestManifest,
    BacktestManifestStore,
)
from quant_platform.backtesting.metrics import compute_financial_metrics
from quant_platform.backtesting.models import BacktestStage
from quant_platform.backtesting.reporting import build_backtest_report_json
from quant_platform.backtesting.runner import (
    OuterFoldBacktestResult,
    recompute_outer_fold_backtest_artifacts,
    resolve_backtest_inputs,
)
from quant_platform.backtesting.signals import SignalSet
from quant_platform.backtesting.specs import BacktestSpec, compute_backtest_identity
from quant_platform.backtesting.stitching import StitchedWalkForwardEquity, build_stitched_walk_forward_equity
from quant_platform.backtesting.timeline import BarReturnTimeline, bar_return_timeline_to_equity_curve
from quant_platform.backtesting.trades import TradeSet
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    BacktestValidationError,
    BacktestVerificationError,
    QuantPlatformError,
    SchemaVersionError,
)
from quant_platform.core.types import Timeframe
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
    JsonPrimitive,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from quant_platform.ml.persistence import (
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    sha256_hex_bytes,
    utc_now,
)

_SCHEMA_VERSION = 1

_UNVERIFIABLE_ARTIFACT_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, BacktestValidationError,
    BacktestVerificationError, KeyError, ValueError, TypeError,
)
# Raw-source reconstruction (Section 3) spans every subsystem a backtest's
# source context touches (calibration, experiment, execution, research
# dataset, historical market data) -- `QuantPlatformError` is the shared
# root of every domain exception across ALL of those subsystems, so
# catching it here (rather than an artifact-decode-scoped tuple) is the
# correct breadth for this specific check, not overly broad.
_RAW_SOURCE_UNVERIFIABLE_ERRORS: tuple[type[Exception], ...] = (QuantPlatformError, KeyError, ValueError, TypeError)
_RECOMPUTATION_TOLERANCE = 1e-6
_TERMINAL_STAGE_TO_RUN_EVENT: dict[BacktestStage, BacktestEventType] = {
    BacktestStage.COMPLETED: BacktestEventType.RUN_COMPLETED, BacktestStage.FAILED: BacktestEventType.RUN_FAILED,
}


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def verify_backtest(
    backtest_id: str, *, backtest_manifest_store: BacktestManifestStore, artifact_store: MLArtifactStore, event_store: BacktestEventStore,
    calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
    execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore,
    research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
) -> ValidationReport:
    """Milestone 5.2, Section 3: the six additional stores/loader beyond
    `backtest_manifest_store`/`artifact_store`/`event_store` are what let
    `_verify_raw_source_reconstruction` independently re-resolve this
    backtest's full source context (calibration/experiment/execution/
    research dataset/raw market data) rather than merely trusting this
    backtest's own persisted artifact tree -- see the module docstring's
    "RAW-SOURCE RECONSTRUCTION" section for why this is required, not
    optional, for a genuine re-audit."""
    manifest = backtest_manifest_store.load(backtest_id)

    issues: list[ValidationIssue] = []
    spec, spec_issues = _verify_backtest_spec(manifest, artifact_store=artifact_store)
    issues += spec_issues

    results, result_issues = _verify_outer_fold_results(manifest, artifact_store=artifact_store)
    issues += result_issues
    if spec is not None:
        for result in results:
            issues += _verify_fold_financials_reproduce(result, spec=spec, artifact_store=artifact_store)
        issues += _verify_raw_source_reconstruction(
            manifest, spec, results, artifact_store=artifact_store, calibration_manifest_store=calibration_manifest_store,
            experiment_manifest_store=experiment_manifest_store, execution_manifest_store=execution_manifest_store,
            research_manifest_store=research_manifest_store, research_dataset_store=research_dataset_store, dataset_loader=dataset_loader,
        )
    issues += _verify_stitched_equity_reproduces(manifest, results, artifact_store=artifact_store)

    issues += _verify_manifest_stage_consistency(manifest)
    issues += _verify_events(manifest, event_store=event_store)

    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _verify_backtest_spec(manifest: BacktestManifest, *, artifact_store: MLArtifactStore) -> tuple[BacktestSpec | None, list[ValidationIssue]]:
    if manifest.spec_reference is None:
        return None, [_issue(ValidationSeverity.CRITICAL, "missing_spec_reference", "BacktestManifest has no spec_reference")]
    if manifest.spec_reference.category is not ArtifactCategory.BACKTEST_SPEC:
        return None, [_issue(ValidationSeverity.CRITICAL, "spec_reference_wrong_category", "spec_reference has the wrong ArtifactCategory")]
    try:
        raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
        spec = BacktestSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return None, [_issue(ValidationSeverity.CRITICAL, "spec_unverifiable", f"BacktestSpec could not be read and decoded: {exc}")]
    identity = compute_backtest_identity(spec)
    if identity.backtest_id != manifest.backtest_id:
        return spec, [_issue(
            ValidationSeverity.CRITICAL, "spec_identity_mismatch",
            f"Recomputed backtest_id={identity.backtest_id!r} does not match manifest.backtest_id={manifest.backtest_id!r}",
        )]
    return spec, [_issue(ValidationSeverity.INFO, "spec_verified", "BacktestSpec decoded and its identity matches the manifest")]


def _verify_outer_fold_results(manifest: BacktestManifest, *, artifact_store: MLArtifactStore) -> tuple[list[OuterFoldBacktestResult], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    results: list[OuterFoldBacktestResult] = []
    for outer_fold_index, reference in manifest.outer_fold_result_references.items():
        if reference.category is not ArtifactCategory.OUTER_FOLD_BACKTEST_RESULT:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_wrong_category",
                f"Outer fold {outer_fold_index}: OuterFoldBacktestResult reference has the wrong category", outer_fold_index=outer_fold_index,
            ))
            continue
        try:
            raw = artifact_store.read_artifact(reference.content_hash)
            decoded = OuterFoldBacktestResult.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_unverifiable",
                f"Outer fold {outer_fold_index}: OuterFoldBacktestResult could not be read and decoded: {exc}", outer_fold_index=outer_fold_index,
            ))
            continue
        if decoded.outer_fold_index != outer_fold_index or decoded.backtest_id != manifest.backtest_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "outer_fold_result_key_mismatch",
                f"Outer fold {outer_fold_index}: decoded OuterFoldBacktestResult does not match its own filing key", outer_fold_index=outer_fold_index,
            ))
            continue
        # Decoding EVERY dependent artifact (not just reading bytes)
        # re-runs that type's own `__post_init__` structural validation --
        # this is what actually catches a hash-valid-but-semantically-
        # tampered artifact (e.g. a `TradeRecord` whose `net_return` was
        # edited without recomputing `trade_id`, Section 30).
        for ref_name, ref, decoder in (
            ("signal_set_reference", decoded.signal_set_reference, SignalSet.from_json_dict),
            ("trade_set_reference", decoded.trade_set_reference, TradeSet.from_json_dict),
        ):
            try:
                raw_sub = artifact_store.read_artifact(ref.content_hash)
                decoder(parse_json_strict(raw_sub.decode("utf-8")))
            except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "outer_fold_result_dependent_artifact_unverifiable",
                    f"Outer fold {outer_fold_index}: {ref_name} could not be read, decoded, and structurally validated: {exc}",
                    outer_fold_index=outer_fold_index,
                ))
        for ref_name, ref in (
            ("equity_curve_reference", decoded.equity_curve_reference), ("gross_drawdown_reference", decoded.gross_drawdown_reference),
            ("bar_return_timeline_reference", decoded.bar_return_timeline_reference),
            ("net_drawdown_reference", decoded.net_drawdown_reference), ("benchmark_report_reference", decoded.benchmark_report_reference),
            ("cost_sensitivity_report_reference", decoded.cost_sensitivity_report_reference),
            ("bucket_analysis_report_reference", decoded.bucket_analysis_report_reference),
        ):
            try:
                artifact_store.read_artifact(ref.content_hash)
            except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "outer_fold_result_dependent_artifact_unverifiable",
                    f"Outer fold {outer_fold_index}: {ref_name} could not be read: {exc}", outer_fold_index=outer_fold_index,
                ))
        results.append(decoded)
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "all_outer_fold_results_verified", f"All {len(results)} recorded OuterFoldBacktestResult artifact(s) verified"))
    return results, issues


def _verify_fold_financials_reproduce(result: OuterFoldBacktestResult, *, spec: BacktestSpec, artifact_store: MLArtifactStore) -> list[ValidationIssue]:
    """THE recomputation check (see module docstring): DECODES (not just
    reads) the persisted `BarReturnTimeline` -- which re-runs its own
    `__post_init__` structural invariants (every point's `gross_return ==
    realized_return + unrealized_return`, exposure identities, exactly
    one point per bar in the fold's own declared range, positive equity,
    non-negative drawdown) -- then rebuilds the equity curve, gross/net
    drawdown, and financial metrics FROM that decoded, validated timeline
    plus the persisted `TradeSet`, and confirms every recomputed scalar
    matches the persisted `financial_metrics`.

    WHAT THIS DOES NOT INDEPENDENTLY RE-DERIVE (a documented, narrower
    limitation than Milestone 5's original claim): it does not re-fetch
    raw market bars and rebuild `BarReturnTimeline` from scratch (that
    would require re-resolving the full experiment/dataset/execution
    context inside `verify_backtest`, a materially heavier operation than
    an artifact-only audit). A `financial_metrics` value tampered
    independently of `bar_return_timeline` is still caught here (the
    common case, and the one Section 30's tampering tests exercise); a
    `bar_return_timeline` tampered consistently with a matching
    `financial_metrics` value (both derived from a fabricated price
    series never matching real market data) would not be -- see
    `docs/financial_evaluation_backtesting.md`'s known limitations."""
    try:
        raw = artifact_store.read_artifact(result.trade_set_reference.content_hash)
        trade_set = TradeSet.from_json_dict(parse_json_strict(raw.decode("utf-8")))
        signal_raw = artifact_store.read_artifact(result.signal_set_reference.content_hash)
        signals = SignalSet.from_json_dict(parse_json_strict(signal_raw.decode("utf-8")))
        timeline_raw = artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash)
        bar_timeline: BarReturnTimeline = BarReturnTimeline.from_json_dict(parse_json_strict(timeline_raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(
            ValidationSeverity.CRITICAL, "fold_financials_unverifiable",
            f"Outer fold {result.outer_fold_index}: dependent artifacts required to recompute financials could not be read: {exc}",
            outer_fold_index=result.outer_fold_index,
        )]

    if bar_timeline.outer_fold_index != result.outer_fold_index:
        return [_issue(
            ValidationSeverity.CRITICAL, "bar_return_timeline_key_mismatch",
            f"Outer fold {result.outer_fold_index}: persisted BarReturnTimeline.outer_fold_index={bar_timeline.outer_fold_index} does not match its own filing key",
            outer_fold_index=result.outer_fold_index,
        )]

    try:
        equity_curve = bar_return_timeline_to_equity_curve(bar_timeline)
        gross_drawdown = compute_drawdown_report(equity_curve, equity_basis="gross")
        net_drawdown = compute_drawdown_report(equity_curve, equity_basis="net")
        recomputed = compute_financial_metrics(
            trades=trade_set, equity_curve=equity_curve, gross_drawdown=gross_drawdown, net_drawdown=net_drawdown,
            signals=signals, bar_timeline=bar_timeline, bar_interval=Timeframe(spec.bar_interval.value), annual_risk_free_rate=spec.annual_risk_free_rate,
            initial_notional=spec.initial_notional, exposure_cap=spec.exposure_cap,
        )
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(
            ValidationSeverity.CRITICAL, "fold_financials_recomputation_failed",
            f"Outer fold {result.outer_fold_index}: recomputing equity/drawdown/metrics from the persisted BarReturnTimeline raised: {exc}",
            outer_fold_index=result.outer_fold_index,
        )]

    mismatches: list[str] = []
    for name, recomputed_value in recomputed.values.items():
        persisted_value = result.financial_metrics.get(name)
        if persisted_value is None:
            mismatches.append(f"{name} (missing from persisted financial_metrics)")
            continue
        if isinstance(persisted_value, bool) or not isinstance(persisted_value, (int, float)):
            mismatches.append(f"{name} (persisted value is non-numeric: {persisted_value!r})")
            continue
        if abs(float(persisted_value) - float(recomputed_value)) > _RECOMPUTATION_TOLERANCE:
            mismatches.append(f"{name} (persisted={persisted_value!r}, recomputed={recomputed_value!r})")
    if mismatches:
        return [_issue(
            ValidationSeverity.CRITICAL, "financial_metrics_do_not_reproduce",
            f"Outer fold {result.outer_fold_index}: {len(mismatches)} financial metric(s) recomputed from the persisted "
            f"TradeSet do not match the persisted financial_metrics -- the persisted artifact is hash-consistent but "
            f"SEMANTICALLY WRONG: {'; '.join(mismatches[:5])}",
            outer_fold_index=result.outer_fold_index, mismatch_count=len(mismatches),
        )]
    return [_issue(
        ValidationSeverity.INFO, "fold_financials_reproduce",
        f"Outer fold {result.outer_fold_index}: persisted financial metrics are reproducible from the persisted "
        "TradeSet alone", outer_fold_index=result.outer_fold_index,
    )]


def _diff_bar_return_timelines(persisted: BarReturnTimeline, rebuilt: BarReturnTimeline, *, tolerance: float = 1e-9) -> list[str]:
    """Milestone 5.2, Section 3's explicit "compare field by field"
    requirement: a content-hash comparison alone cannot distinguish
    "identical" from "an attacker tampered the timeline and recomputed
    its hash to match" -- but a per-field comparison against a timeline
    rebuilt from data the attacker never touched (raw market bars, raw
    predictions) cannot be fooled this way. Returns a list of human-
    readable mismatch descriptions (empty if the timelines agree)."""
    mismatches: list[str] = []
    for name in ("outer_fold_index", "compounded", "fold_start_position", "fold_end_position"):
        persisted_value, rebuilt_value = getattr(persisted, name), getattr(rebuilt, name)
        if persisted_value != rebuilt_value:
            mismatches.append(f"{name}: persisted={persisted_value!r} rebuilt={rebuilt_value!r}")
    if persisted.return_basis != rebuilt.return_basis:
        mismatches.append(f"return_basis: persisted={persisted.return_basis!r} rebuilt={rebuilt.return_basis!r}")
    if len(persisted.points) != len(rebuilt.points):
        mismatches.append(f"points: persisted has {len(persisted.points)} point(s), rebuilt has {len(rebuilt.points)}")
        return mismatches
    for i, (p, r) in enumerate(zip(persisted.points, rebuilt.points, strict=True)):
        p_dict, r_dict = p.to_json_dict(), r.to_json_dict()
        for key, persisted_value in p_dict.items():
            rebuilt_value = r_dict[key]
            if isinstance(persisted_value, (int, float)) and not isinstance(persisted_value, bool) and isinstance(rebuilt_value, (int, float)) and not isinstance(rebuilt_value, bool):
                if abs(float(persisted_value) - float(rebuilt_value)) > tolerance:
                    mismatches.append(f"points[{i}].{key}: persisted={persisted_value!r} rebuilt={rebuilt_value!r}")
            elif persisted_value != rebuilt_value:
                mismatches.append(f"points[{i}].{key}: persisted={persisted_value!r} rebuilt={rebuilt_value!r}")
    return mismatches


def _verify_raw_source_reconstruction(
    manifest: BacktestManifest, spec: BacktestSpec, results: list[OuterFoldBacktestResult], *, artifact_store: MLArtifactStore,
    calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
    execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore,
    research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
) -> list[ValidationIssue]:
    """Milestone 5.2, Section 3 (see module docstring for the full
    rationale): independently re-resolves this backtest's full source
    context and re-derives every outer fold's signals/trades/timeline/
    equity/drawdown/metrics/benchmarks/cost-sensitivity/bucket-analysis
    from raw market bars and raw source predictions ALONE, using the same
    production code path (`runner.recompute_outer_fold_backtest_
    artifacts`) the original run used -- then compares the rebuild to what
    was actually persisted. Never reads the persisted `BarReturnTimeline`
    or persisted `financial_metrics` as an input to its own rebuild (only
    as the THING BEING CHECKED), so a consistently-tampered timeline,
    metrics, and content hash cannot pass undetected."""
    try:
        inputs = resolve_backtest_inputs(
            spec, calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
            execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store,
            research_dataset_store=research_dataset_store, dataset_loader=dataset_loader, artifact_store=artifact_store,
        )
    except _RAW_SOURCE_UNVERIFIABLE_ERRORS as exc:
        return [_issue(
            ValidationSeverity.CRITICAL, "raw_source_resolution_failed",
            f"Could not independently re-resolve this backtest's source calibration/experiment/execution/dataset "
            f"context in order to rebuild it from raw sources: {exc}",
        )]

    folds_by_index = {f.fold_index: f for f in inputs.outer_folds}
    ordered_results = sorted(results, key=lambda r: r.outer_fold_index)
    issues: list[ValidationIssue] = []
    rebuilt_timelines: list[BarReturnTimeline] = []

    for result in ordered_results:
        fold = folds_by_index.get(result.outer_fold_index)
        if fold is None:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_fold_not_in_recomputed_plan",
                f"Outer fold {result.outer_fold_index}: no matching fold in the independently re-resolved outer "
                "fold plan -- the persisted result cannot be raw-source-verified", outer_fold_index=result.outer_fold_index,
            ))
            continue
        try:
            recomputed = recompute_outer_fold_backtest_artifacts(
                spec=spec, outer_fold=fold, timeline=inputs.timeline, bars=inputs.bars,
                calibration_manifest=inputs.calibration_manifest, calibration_spec=inputs.calibration_spec, artifact_store=artifact_store,
            )
        except _RAW_SOURCE_UNVERIFIABLE_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_recomputation_failed",
                f"Outer fold {result.outer_fold_index}: independently recomputing this fold from raw sources raised: {exc}",
                outer_fold_index=result.outer_fold_index,
            ))
            continue
        rebuilt_timelines.append(recomputed.bar_timeline)

        try:
            persisted_timeline_raw = artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash)
            persisted_timeline = BarReturnTimeline.from_json_dict(parse_json_strict(persisted_timeline_raw.decode("utf-8")))
        except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_persisted_timeline_unreadable",
                f"Outer fold {result.outer_fold_index}: could not read the persisted BarReturnTimeline to compare "
                f"against the raw-source rebuild: {exc}", outer_fold_index=result.outer_fold_index,
            ))
            continue
        timeline_mismatches = _diff_bar_return_timelines(persisted_timeline, recomputed.bar_timeline)
        if timeline_mismatches:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_timeline_mismatch",
                f"Outer fold {result.outer_fold_index}: the persisted BarReturnTimeline does not match one "
                f"independently rebuilt from raw market bars and raw predictions -- {len(timeline_mismatches)} "
                f"field mismatch(es): {'; '.join(timeline_mismatches[:5])}",
                outer_fold_index=result.outer_fold_index, mismatch_count=len(timeline_mismatches),
            ))

        for label, ref, recomputed_obj in (
            ("signal_set", result.signal_set_reference, recomputed.signals), ("trade_set", result.trade_set_reference, recomputed.trade_set),
            ("equity_curve", result.equity_curve_reference, recomputed.equity_curve), ("gross_drawdown", result.gross_drawdown_reference, recomputed.gross_drawdown),
            ("net_drawdown", result.net_drawdown_reference, recomputed.net_drawdown), ("benchmark_report", result.benchmark_report_reference, recomputed.benchmark_report),
            ("cost_sensitivity_report", result.cost_sensitivity_report_reference, recomputed.cost_sensitivity_report),
            ("bucket_analysis_report", result.bucket_analysis_report_reference, recomputed.bucket_report),
        ):
            rebuilt_hash = sha256_hex_bytes(canonical_json_bytes(recomputed_obj.to_json_dict()))
            if rebuilt_hash != ref.content_hash:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "raw_source_artifact_mismatch",
                    f"Outer fold {result.outer_fold_index}: independently rebuilding {label} from raw sources "
                    f"produced content_hash={rebuilt_hash!r}, which does not match the persisted reference "
                    f"content_hash={ref.content_hash!r}", outer_fold_index=result.outer_fold_index, artifact=label,
                ))

        metric_mismatches: list[str] = []
        for name, recomputed_value in recomputed.metrics_report.values.items():
            persisted_value = result.financial_metrics.get(name)
            if persisted_value is None or isinstance(persisted_value, bool) or not isinstance(persisted_value, (int, float)):
                metric_mismatches.append(f"{name} (persisted={persisted_value!r})")
            elif abs(float(persisted_value) - float(recomputed_value)) > _RECOMPUTATION_TOLERANCE:
                metric_mismatches.append(f"{name} (persisted={persisted_value!r}, raw-source-recomputed={recomputed_value!r})")
        if set(recomputed.metrics_report.skipped) != set(result.skipped_metrics):
            metric_mismatches.append(
                f"skipped_metrics keys differ: persisted={sorted(result.skipped_metrics)!r} "
                f"raw-source-recomputed={sorted(recomputed.metrics_report.skipped)!r}"
            )
        if metric_mismatches:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_financial_metrics_mismatch",
                f"Outer fold {result.outer_fold_index}: {len(metric_mismatches)} financial metric(s) independently "
                f"recomputed from raw sources do not match the persisted financial_metrics: {'; '.join(metric_mismatches[:5])}",
                outer_fold_index=result.outer_fold_index, mismatch_count=len(metric_mismatches),
            ))

    if not rebuilt_timelines:
        if not issues:
            issues.append(_issue(ValidationSeverity.INFO, "raw_source_reconstruction_no_folds", "No outer fold results were available to raw-source-verify"))
        return issues

    try:
        rebuilt_stitched = build_stitched_walk_forward_equity(backtest_id=manifest.backtest_id, timelines=rebuilt_timelines)
    except _RAW_SOURCE_UNVERIFIABLE_ERRORS as exc:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "raw_source_stitching_failed",
            f"Rebuilding stitched equity from raw-source-reconstructed per-fold timelines raised: {exc}",
        ))
        return issues

    if manifest.stage in (BacktestStage.COMPLETED, BacktestStage.VERIFIED) and manifest.stitched_equity_reference is not None:
        rebuilt_stitched_hash = sha256_hex_bytes(canonical_json_bytes(rebuilt_stitched.to_json_dict()))
        if rebuilt_stitched_hash != manifest.stitched_equity_reference.content_hash:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_stitched_equity_mismatch",
                f"Independently rebuilding stitched equity from raw-source-reconstructed per-fold timelines "
                f"produced content_hash={rebuilt_stitched_hash!r}, which does not match the persisted "
                f"stitched_equity_reference content_hash={manifest.stitched_equity_reference.content_hash!r}",
            ))

    if manifest.stage is BacktestStage.COMPLETED and manifest.aggregate_report_reference is not None:
        try:
            persisted_report_raw = artifact_store.read_artifact(manifest.aggregate_report_reference.content_hash)
            persisted_report = parse_json_strict(persisted_report_raw.decode("utf-8"))
            rebuilt_report = build_backtest_report_json(
                manifest, spec=spec, outer_fold_results=ordered_results, stitched=rebuilt_stitched,
                stitched_equity_reference=manifest.stitched_equity_reference,
            )
        except _RAW_SOURCE_UNVERIFIABLE_ERRORS as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "raw_source_aggregate_report_failed", f"Rebuilding the aggregate report raised: {exc}"))
            return issues
        # `stage`/`created_at`/`updated_at`/`completed_at`/`resume_count`
        # legitimately differ between the manifest snapshot captured at
        # original report-generation time (necessarily BEFORE the final
        # COMPLETED transition -- the report must be written before the
        # manifest can reference it) and the manifest loaded here, now, at
        # its final state; only the substantive, raw-source-derived
        # sections are compared.
        mismatched_sections = [
            key for key in ("identity", "outer_fold_results", "aggregate_metrics", "stitched_equity", "limitations")
            if persisted_report.get(key) != rebuilt_report.get(key)
        ]
        if mismatched_sections:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "raw_source_aggregate_report_mismatch",
                f"The persisted aggregate report's {mismatched_sections} section(s) do not match a rebuild derived "
                "from raw-source-verified per-fold results and a raw-source-rebuilt stitched equity",
                sections=",".join(mismatched_sections),
            ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(
            ValidationSeverity.INFO, "raw_source_reconstruction_verified",
            f"All {len(ordered_results)} outer fold result(s), the stitched equity, and the aggregate report are "
            "independently reproducible from raw market bars and raw source predictions alone -- a valid content "
            "hash was not merely trusted",
        ))
    return issues


def _verify_stitched_equity_reproduces(
    manifest: BacktestManifest, results: list[OuterFoldBacktestResult], *, artifact_store: MLArtifactStore,
) -> list[ValidationIssue]:
    """Milestone 5.1, Section 3: "verification independently reconstructs
    it" -- rebuilds `stitching.StitchedWalkForwardEquity` from the SAME
    persisted per-fold `BarReturnTimeline`s `_verify_fold_financials_
    reproduce` already decodes and validates, and asserts the rebuild's
    content hash matches the persisted `stitched_equity_reference.
    content_hash` EXACTLY. An exact hash comparison (not a float-tolerance
    field comparison) is the correct and strongest possible check here,
    because `build_stitched_walk_forward_equity` is a pure, deterministic
    function of its inputs -- any divergence, however small, means the
    persisted artifact was not actually produced by this construction from
    these exact per-fold timelines."""
    if manifest.stage not in (BacktestStage.COMPLETED, BacktestStage.VERIFIED):
        return []
    if manifest.stitched_equity_reference is None:
        return [_issue(
            ValidationSeverity.CRITICAL, "missing_stitched_equity_reference",
            f"BacktestManifest.stage={manifest.stage.value!r} but stitched_equity_reference is None",
        )]
    if manifest.stitched_equity_reference.category is not ArtifactCategory.STITCHED_WALK_FORWARD_EQUITY:
        return [_issue(ValidationSeverity.CRITICAL, "stitched_equity_reference_wrong_category", "stitched_equity_reference has the wrong ArtifactCategory")]
    try:
        raw = artifact_store.read_artifact(manifest.stitched_equity_reference.content_hash)
        decoded = StitchedWalkForwardEquity.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(ValidationSeverity.CRITICAL, "stitched_equity_unverifiable", f"StitchedWalkForwardEquity could not be read and decoded: {exc}")]
    if decoded.backtest_id != manifest.backtest_id:
        return [_issue(ValidationSeverity.CRITICAL, "stitched_equity_key_mismatch", "Decoded StitchedWalkForwardEquity.backtest_id does not match its own filing key")]

    timelines: list[BarReturnTimeline] = []
    try:
        for result in sorted(results, key=lambda r: r.outer_fold_index):
            timeline_raw = artifact_store.read_artifact(result.bar_return_timeline_reference.content_hash)
            timelines.append(BarReturnTimeline.from_json_dict(parse_json_strict(timeline_raw.decode("utf-8"))))
        rebuilt = build_stitched_walk_forward_equity(backtest_id=manifest.backtest_id, timelines=timelines)
        rebuilt_hash = sha256_hex_bytes(canonical_json_bytes(rebuilt.to_json_dict()))
    except _UNVERIFIABLE_ARTIFACT_ERRORS as exc:
        return [_issue(
            ValidationSeverity.CRITICAL, "stitched_equity_recomputation_failed",
            f"Rebuilding StitchedWalkForwardEquity from the persisted per-fold BarReturnTimelines raised: {exc}",
        )]

    if rebuilt_hash != manifest.stitched_equity_reference.content_hash:
        return [_issue(
            ValidationSeverity.CRITICAL, "stitched_equity_does_not_reproduce",
            f"Independently rebuilding StitchedWalkForwardEquity from the persisted per-fold BarReturnTimelines "
            f"produced content_hash={rebuilt_hash!r}, which does not match the persisted stitched_equity_reference "
            f"content_hash={manifest.stitched_equity_reference.content_hash!r} -- the persisted artifact is not "
            "reproducible from this backtest's own per-fold timelines",
        )]
    return [_issue(
        ValidationSeverity.INFO, "stitched_equity_reproduces",
        "The persisted StitchedWalkForwardEquity artifact is independently reproducible (exact content-hash match) "
        "from the persisted per-fold BarReturnTimelines",
    )]


def _verify_manifest_stage_consistency(manifest: BacktestManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.stage in (BacktestStage.COMPLETED, BacktestStage.VERIFIED) and (
        manifest.total_outer_folds is None or len(manifest.completed_outer_fold_indices) != manifest.total_outer_folds
    ):
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "terminal_stage_outer_fold_count_mismatch",
            f"stage={manifest.stage.value!r} but completed_outer_fold_indices has {len(manifest.completed_outer_fold_indices)} "
            f"entries, expected total_outer_folds={manifest.total_outer_folds!r}",
        ))
    if manifest.stage is BacktestStage.COMPLETED and manifest.aggregate_report_reference is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "completed_without_aggregate_report", "stage=COMPLETED but aggregate_report_reference is None"))
    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "manifest_stage_consistent", f"BacktestManifest.stage={manifest.stage.value!r} is consistent with its own recorded progress fields"))
    return issues


def _verify_events(manifest: BacktestManifest, *, event_store: BacktestEventStore) -> list[ValidationIssue]:
    try:
        events = event_store.read_events(manifest.backtest_id)
    except ArtifactCorruptionError as exc:
        return [_issue(ValidationSeverity.CRITICAL, "event_log_corrupted", f"Event log could not be read: {exc}")]

    issues: list[ValidationIssue] = []
    run_started = [e for e in events if e.event_type is BacktestEventType.RUN_STARTED]
    if len(run_started) > 1:
        issues.append(_issue(ValidationSeverity.CRITICAL, "multiple_run_started_events", f"{len(run_started)} RUN_STARTED events recorded -- expected at most 1"))

    for index, event in enumerate(events):
        if event.event_type in _TERMINAL_STAGE_TO_RUN_EVENT.values() and index != len(events) - 1:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "event_after_terminal_run_event",
                f"{event.event_type.value!r} at sequence={event.sequence} is not the last event ({len(events)} total)", sequence=event.sequence,
            ))

    # Fold boundary ordering: SOURCES_VERIFIED must precede that fold's
    # own SIGNALS_GENERATED/TRADES_CONSTRUCTED events (Section 24: no
    # cross-fold positions, every fold starts flat -- evidenced here by
    # the event log itself always re-verifying sources before generating
    # that fold's signals).
    verified_sequence: dict[int, int] = {}
    for event in events:
        raw_index = event.details.get("outer_fold_index")
        outer_fold_index = None if raw_index is None else int(str(raw_index))
        if event.event_type is BacktestEventType.SOURCES_VERIFIED and outer_fold_index is not None:
            verified_sequence[outer_fold_index] = event.sequence
        elif event.event_type in (BacktestEventType.SIGNALS_GENERATED, BacktestEventType.TRADES_CONSTRUCTED) and outer_fold_index is not None:
            verified_at = verified_sequence.get(outer_fold_index)
            if verified_at is None or verified_at >= event.sequence:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "signal_event_before_sources_verified",
                    f"Outer fold {outer_fold_index}: {event.event_type.value!r} at sequence={event.sequence} has no earlier "
                    "SOURCES_VERIFIED event for the same fold", sequence=event.sequence, outer_fold_index=outer_fold_index,
                ))

    if not any(i.severity is ValidationSeverity.CRITICAL for i in issues):
        issues.append(_issue(ValidationSeverity.INFO, "events_consistent", f"All {len(events)} recorded event(s) are internally consistent"))
    return issues


__all__ = ["verify_backtest"]
