"""Human-readable and machine-readable backtest reports (Milestone 5,
Section 31) -- the backtesting-framework analogue of `calibration.
reporting`'s "deterministic JSON + companion Markdown, no dashboard"
scope. Every function here is a PURE function of already-loaded data --
no I/O, no re-fetching artifacts, no re-running anything.

`_STANDARD_LIMITATIONS` is included, verbatim, in EVERY report this
module produces (Section 31's mandatory hypothetical/non-live disclaimer
requirement): this framework never claims a backtested result predicts
future, live-trading performance, never claims the modeled costs match
any specific broker's real costs, and never claims a positive backtested
return demonstrates a profitable trading strategy.

SECTION 25/MILESTONE 5.1 SECTION 3: WHY THIS MODULE NEVER *AVERAGES*
EQUITY CURVES -- BUT DOES NOW STITCH THEM
--------------------------------------------------------------------------
`_aggregate_metrics_section` aggregates each outer fold's already-computed
SCALAR financial metrics (fold-wise mean/std) -- it never concatenates or
averages the folds' own `EquityCurve` objects point-by-point into one
"combined" curve; that would produce a number with no coherent financial
meaning for non-overlapping calendar windows. A genuine chronological
STITCHED walk-forward equity curve (chaining each fold's curve after the
previous one's ending equity, per `backtesting.stitching`) is a distinct,
correctly-defined construction -- Milestone 5.1 implements it in
`stitching.build_stitched_walk_forward_equity`; `_stitched_equity_section`
below surfaces its summary metrics (never a per-fold average) whenever the
caller supplies an already-built `StitchedWalkForwardEquity`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.manifests import BacktestManifest
from quant_platform.backtesting.runner import OuterFoldBacktestResult
from quant_platform.backtesting.specs import BacktestSpec
from quant_platform.backtesting.stitching import (
    StitchedWalkForwardEquity,
    compute_stitched_financial_metrics,
    stitched_walk_forward_equity_to_equity_curve,
)
from quant_platform.ml.models import ArtifactReference

_SCHEMA_VERSION = 1

_STANDARD_LIMITATIONS = (
    "HYPOTHETICAL, NON-LIVE RESULTS: every return, trade, and metric in this report comes from a simulated "
    "reconstruction of what a declared trading rule would have done against historical bars under explicitly "
    "modeled execution assumptions -- no live order was ever submitted, no real capital was ever at risk, and no "
    "broker, exchange, or liquidity venue was ever involved.",
    "PAST PERFORMANCE, EVEN SIMULATED, DOES NOT INDICATE FUTURE RESULTS: a positive backtested return does not "
    "demonstrate that the underlying strategy is, or will be, profitable in live trading.",
    "MODELED COSTS ARE ASSUMPTIONS, NOT MEASUREMENTS: spread, commission, slippage, and financing figures are "
    "the DECLARED, deterministic cost model this backtest was configured with (see `identity.cost_model`) -- "
    "they are not a claim that any specific broker's real-world costs, fills, latency, or liquidity would match.",
    "NO LIVE EXECUTION, PORTFOLIO ALLOCATION, OR RISK MANAGEMENT WAS PERFORMED: this framework evaluates ONE "
    "declared strategy in isolation, with no capital-allocation-across-strategies, leverage optimization, or "
    "production risk-limit logic of any kind (explicitly out of scope for this milestone).",
    "A SMALL NUMBER OF OUTER FOLDS (typical for walk-forward evaluation) LIMITS STATISTICAL CONFIDENCE: fold-wise "
    "metric means/standard deviations below are descriptive, not a claim of statistical significance.",
    "PER-FOLD EQUITY CURVES ARE NEVER AVERAGED TOGETHER: `aggregate_metrics` reports fold-wise mean/std of each "
    "fold's own SCALAR metrics only; the separate `stitched_equity` section (when present) is a chronologically "
    "CHAINED equity path, not an average -- see this module's own docstring.",
)


def _identity_section(spec: BacktestSpec) -> dict[str, object]:
    return {
        "source_calibration_id": spec.source_calibration_id, "source_experiment_id": spec.source_experiment_id,
        "instrument_identity": spec.instrument_identity, "bar_interval": spec.bar_interval.value,
        "decision_timestamp_policy": spec.decision_timestamp_policy.value, "signal_mapping": spec.signal_mapping.kind.value,
        "position_mode": spec.position_mode.value, "entry_policy": spec.entry_spec.kind.value, "exit_policy": spec.exit_spec.kind.value,
        "overlap_policy": spec.overlap_policy.value, "price_basis": spec.price_basis.value,
        "cost_model": {
            "spread": spec.spread_spec.kind.value, "commission": spec.commission_spec.kind.value,
            "slippage": spec.slippage_spec.kind.value, "financing": spec.financing_spec.kind.value,
        },
        "return_calculation_policy": spec.return_calculation_policy.value, "compounding_policy": spec.compounding_policy.value,
        "initial_notional": spec.initial_notional, "exposure_cap": spec.exposure_cap, "annual_risk_free_rate": spec.annual_risk_free_rate,
        "respect_calibration_abstention": spec.respect_calibration_abstention, "determinism_policy": spec.determinism_policy.value,
    }


def _outer_fold_section(result: OuterFoldBacktestResult) -> dict[str, object]:
    return {
        "outer_fold_index": result.outer_fold_index, "outer_test_row_count": result.outer_test_row_count,
        "closed_trade_count": result.closed_trade_count, "meets_minimum_trade_threshold": result.meets_minimum_trade_threshold,
        "financial_metrics": dict(sorted(result.financial_metrics.items())), "skipped_metrics": dict(sorted(result.skipped_metrics.items())),
        "evaluated_at": result.evaluated_at,
    }


def _aggregate_metrics_section(outer_fold_results: Sequence[OuterFoldBacktestResult]) -> dict[str, object]:
    """Section 25/27: per-fold values PLUS fold-wise statistics, never
    hiding fold instability by silently averaging it away -- every fold's
    own metric values are listed, alongside cross-fold mean/std, for
    EVERY metric name common to every fold's `financial_metrics`."""
    if not outer_fold_results:
        return {"per_fold": {}, "fold_wise_mean": {}, "fold_wise_std": {}}
    common_names = set(outer_fold_results[0].financial_metrics)
    for r in outer_fold_results[1:]:
        common_names &= set(r.financial_metrics)
    per_fold: dict[str, list[object]] = {name: [] for name in sorted(common_names)}
    for r in outer_fold_results:
        for name in per_fold:
            per_fold[name].append(r.financial_metrics[name])
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name, values in per_fold.items():
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if len(numeric) == len(values) and numeric:
            mean = sum(numeric) / len(numeric)
            means[name] = mean
            stds[name] = (sum((v - mean) ** 2 for v in numeric) / len(numeric)) ** 0.5
    return {"per_fold": per_fold, "fold_wise_mean": means, "fold_wise_std": stds}


def _pooled_transacted_notional(outer_fold_results: Sequence[OuterFoldBacktestResult]) -> float:
    """Milestone 5.2, Section 4: the stitched-level transacted-notional
    total is the SUM of each fold's own exact, fill-price-derived
    `total_transacted_notional` (`metrics.compute_financial_metrics`'s
    corrected formula) -- a fold with no closed trades contributes 0.0
    (that metric is `skip`ped, not persisted as zero, for such a fold;
    treating "skipped because no trades" as a zero contribution to a SUM
    is correct here, unlike averaging a skipped metric would be)."""
    total = 0.0
    for r in outer_fold_results:
        value = r.financial_metrics.get("total_transacted_notional")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return total


def _stitched_equity_section(
    stitched: StitchedWalkForwardEquity | None, *, spec: BacktestSpec | None, stitched_equity_reference: ArtifactReference | None,
    outer_fold_results: Sequence[OuterFoldBacktestResult] = (),
) -> dict[str, object] | None:
    """Milestone 5.1, Section 3: summarizes an already-built
    `StitchedWalkForwardEquity` -- `None` when the caller has none to
    report (e.g. `create-backtest-spec`'s dry-run path, or a not-yet-
    COMPLETED backtest), never fabricated from partial data. Requires
    `spec` (for `bar_interval`/`annual_risk_free_rate`/`initial_notional`,
    exactly what `compute_stitched_financial_metrics` needs) -- absent
    `spec`, only the raw fold-boundary/point-count structure is reported.
    `outer_fold_results` supplies the pooled transacted-notional total
    (Milestone 5.2, Section 4) -- see `_pooled_transacted_notional`."""
    if stitched is None:
        return None
    section: dict[str, object] = {
        "stitched_equity_reference_content_hash": (None if stitched_equity_reference is None else stitched_equity_reference.content_hash),
        "point_count": len(stitched.points),
        "compounded": stitched.compounded,
        "fold_boundaries": [b.to_json_dict() for b in stitched.fold_boundaries],
    }
    if spec is None:
        section["metrics"] = None
        section["skipped_metrics"] = None
        return section
    equity_curve = stitched_walk_forward_equity_to_equity_curve(stitched)
    gross_drawdown = compute_drawdown_report(equity_curve, equity_basis="gross")
    net_drawdown = compute_drawdown_report(equity_curve, equity_basis="net")
    metrics_report = compute_stitched_financial_metrics(
        stitched=stitched, gross_drawdown=gross_drawdown, net_drawdown=net_drawdown, bar_interval=spec.bar_interval,
        annual_risk_free_rate=spec.annual_risk_free_rate, initial_notional=spec.initial_notional,
        total_transacted_notional=_pooled_transacted_notional(outer_fold_results),
    )
    section["metrics"] = dict(sorted(metrics_report.values.items()))
    section["skipped_metrics"] = dict(sorted(metrics_report.skipped.items()))
    return section


def build_backtest_report_json(
    manifest: BacktestManifest, *, spec: BacktestSpec | None = None, outer_fold_results: Sequence[OuterFoldBacktestResult] = (),
    stitched: StitchedWalkForwardEquity | None = None, stitched_equity_reference: ArtifactReference | None = None,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION, "backtest_id": manifest.backtest_id, "source_calibration_id": manifest.source_calibration_id,
        "stage": manifest.stage.value, "created_at": manifest.created_at, "updated_at": manifest.updated_at, "completed_at": manifest.completed_at,
        "resume_count": manifest.resume_count, "total_outer_folds": manifest.total_outer_folds,
        "completed_outer_fold_indices": list(manifest.completed_outer_fold_indices),
        "identity": (None if spec is None else _identity_section(spec)),
        "outer_fold_results": [_outer_fold_section(r) for r in outer_fold_results],
        "aggregate_metrics": _aggregate_metrics_section(outer_fold_results),
        "stitched_equity": _stitched_equity_section(stitched, spec=spec, stitched_equity_reference=stitched_equity_reference, outer_fold_results=outer_fold_results),
        "failure_summary": manifest.failure_summary, "limitations": list(_STANDARD_LIMITATIONS),
    }


def render_backtest_report_markdown(
    manifest: BacktestManifest, *, spec: BacktestSpec | None = None, outer_fold_results: Sequence[OuterFoldBacktestResult] = (),
    stitched: StitchedWalkForwardEquity | None = None, stitched_equity_reference: ArtifactReference | None = None,
) -> str:
    lines = [
        "# Backtest Report (Hypothetical, Non-Live Results)", "",
        f"- **Backtest ID:** `{manifest.backtest_id}`", f"- **Source calibration ID:** `{manifest.source_calibration_id}`",
        f"- **Stage:** `{manifest.stage.value}`", f"- **Created at:** {manifest.created_at}", f"- **Updated at:** {manifest.updated_at}",
    ]
    if manifest.completed_at is not None:
        lines.append(f"- **Completed at:** {manifest.completed_at}")
    lines.append(f"- **Resume count:** {manifest.resume_count}")

    if spec is not None:
        lines += [
            "", "## Identity",
            f"- instrument: `{spec.instrument_identity}` @ `{spec.bar_interval.value}`",
            f"- position mode: `{spec.position_mode.value}`, signal mapping: `{spec.signal_mapping.kind.value}`",
            f"- entry policy: `{spec.entry_spec.kind.value}`, exit policy: `{spec.exit_spec.kind.value}`, overlap policy: `{spec.overlap_policy.value}`",
            f"- cost model: spread=`{spec.spread_spec.kind.value}` commission=`{spec.commission_spec.kind.value}` "
            f"slippage=`{spec.slippage_spec.kind.value}` financing=`{spec.financing_spec.kind.value}`",
            f"- initial notional: {spec.initial_notional}, exposure cap: {spec.exposure_cap}, annual risk-free rate: {spec.annual_risk_free_rate}",
        ]

    lines += ["", "## Outer-Fold Results (untouched outer-test partition, evaluated once per fold)"]
    if not outer_fold_results:
        lines.append("- (no outer-fold results loaded -- pass `outer_fold_results=` to expand)")
    else:
        lines.append("| Outer Fold | # Test Rows | # Closed Trades | Total Net Return | Sharpe (bar) | Max Drawdown |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        def _fmt(metrics: Mapping[str, object], key: str) -> str:
            v = metrics.get(key)
            return f"{v:.4g}" if isinstance(v, (int, float)) else "-"

        for r in outer_fold_results:
            m = r.financial_metrics
            lines.append(f"| {r.outer_fold_index} | {r.outer_test_row_count} | {r.closed_trade_count} | {_fmt(m, 'total_net_return')} | {_fmt(m, 'bar_return_sharpe')} | {_fmt(m, 'maximum_drawdown')} |")

    if outer_fold_results:
        aggregate = _aggregate_metrics_section(outer_fold_results)
        lines += ["", "## Cross-Fold Metric Stability (fold-wise mean/std, never an averaged equity curve)"]
        means = aggregate["fold_wise_mean"]
        stds = aggregate["fold_wise_std"]
        if isinstance(means, dict) and isinstance(stds, dict) and means:
            lines.append("| Metric | Mean | Std |")
            lines.append("| --- | --- | --- |")
            for name in sorted(means):
                lines.append(f"| {name} | {means[name]:.4g} | {stds[name]:.4g} |")
        else:
            lines.append("- (metrics were not uniformly numeric/present across all outer folds)")

    stitched_section = _stitched_equity_section(stitched, spec=spec, stitched_equity_reference=stitched_equity_reference, outer_fold_results=outer_fold_results)
    if stitched_section is not None:
        fold_boundary_count = len(stitched_section["fold_boundaries"]) if isinstance(stitched_section["fold_boundaries"], list) else 0
        lines += ["", "## Stitched Walk-Forward Equity (whole-backtest chronological path, never an average)"]
        lines.append(f"- points: {stitched_section['point_count']}, fold boundaries: {fold_boundary_count}, compounded: {stitched_section['compounded']}")
        metrics = stitched_section.get("metrics")
        if isinstance(metrics, dict) and metrics:
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for name in sorted(metrics):
                value = metrics[name]
                lines.append(f"| {name} | {value:.4g} |" if isinstance(value, (int, float)) else f"| {name} | {value} |")

    if manifest.failure_summary:
        lines += ["", "## Failure Summary", manifest.failure_summary]

    lines += ["", "## Limitations"]
    for limitation in _STANDARD_LIMITATIONS:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = ["build_backtest_report_json", "render_backtest_report_markdown"]
