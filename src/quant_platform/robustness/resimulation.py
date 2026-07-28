"""Shared re-simulation helper for `sensitivity.py` (Section 9) and
`stress.py` (Section 10) -- both need the SAME underlying operation:
re-run the source backtest's OWN already-verified predictions/market bars
through the FULL production simulation pipeline under a MODIFIED
`BacktestSpec`, and reduce the result to a small set of scalar outcomes.
Reuses `backtesting.runner.recompute_outer_fold_backtest_artifacts` (the
exact production code path, never a re-implemented parallel one) for
every outer fold, then stitches. Nothing computed here is ever persisted
as a `StitchedWalkForwardEquity`/`OuterFoldBacktestResult` artifact --
these are ephemeral, in-memory "what if" evaluations, never confused with
a genuine backtest run."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.runner import ResolvedBacktestInputs, recompute_outer_fold_backtest_artifacts
from quant_platform.backtesting.specs import BacktestSpec
from quant_platform.backtesting.stitching import (
    build_stitched_walk_forward_equity,
    stitched_walk_forward_equity_to_equity_curve,
)
from quant_platform.core.exceptions import BacktestError
from quant_platform.ml.artifacts import MLArtifactStore


@dataclass(frozen=True, slots=True)
class ResimulationResult:
    total_net_return: float
    total_gross_return: float
    closed_trade_count: int
    maximum_drawdown: float


_EXPECTED_DOMAIN_INVALID_PERTURBATION_ERRORS: tuple[type[Exception], ...] = (BacktestError,)
"""Every domain exception `backtesting`'s own simulation pipeline can
legitimately raise for a MODIFIED spec that is internally invalid or
drives a result out of this platform's modeled scope (e.g. a holding
period that violates `ExitSpec`'s own bounds, or an extreme cost
multiplier that pushes a LOG-policy net value non-positive). Caught here
and reported as "this perturbation could not be simulated", never
silently -- see `resimulate_stitched_outcome`'s own docstring. A genuine
bug (TypeError/AttributeError/KeyError from this module's own code) is
deliberately NOT in this tuple and always propagates."""


def resimulate_stitched_outcome(
    *, modified_spec: BacktestSpec, resolved_inputs: ResolvedBacktestInputs, label_backtest_id: str, artifact_store: MLArtifactStore,
) -> ResimulationResult | None:
    """Returns `None` (never raises) if the modified spec cannot be
    simulated at all (e.g. an internally inconsistent perturbation) --
    the caller reports this as an explicit skip, not a crash of the
    whole sensitivity/stress sweep."""
    try:
        timelines = []
        total_closed = 0
        for fold in resolved_inputs.outer_folds:
            recomputed = recompute_outer_fold_backtest_artifacts(
                spec=modified_spec, outer_fold=fold, timeline=resolved_inputs.timeline, bars=resolved_inputs.bars,
                calibration_manifest=resolved_inputs.calibration_manifest, calibration_spec=resolved_inputs.calibration_spec,
                artifact_store=artifact_store,
            )
            timelines.append(recomputed.bar_timeline)
            total_closed += len(recomputed.trade_set.closed_trades)
        stitched = build_stitched_walk_forward_equity(backtest_id=label_backtest_id, timelines=timelines)
    except _EXPECTED_DOMAIN_INVALID_PERTURBATION_ERRORS:
        return None
    last = stitched.points[-1]
    equity_curve = stitched_walk_forward_equity_to_equity_curve(stitched)
    net_drawdown = compute_drawdown_report(equity_curve, equity_basis="net")
    return ResimulationResult(
        total_net_return=last.stitched_net_equity - 1.0, total_gross_return=last.stitched_gross_equity - 1.0,
        closed_trade_count=total_closed, maximum_drawdown=net_drawdown.maximum_drawdown,
    )


__all__ = ["ResimulationResult", "resimulate_stitched_outcome"]
