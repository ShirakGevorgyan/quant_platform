"""Parameter and decision stability analysis (Milestone 6, Section 9).
Tests whether the source backtest's ALREADY-SELECTED operating point is
stable under small, pre-declared perturbations -- never a search for a
better parameter. Every perturbed point is produced by re-simulating the
source backtest's OWN verified predictions/market bars through the
EXACT production pipeline (`resimulation.resimulate_stitched_outcome`)
under a `dataclasses.replace`-modified `BacktestSpec`; nothing here
re-fits a model or looks for an improvement to feed back in.

APPLICABILITY: not every declared `PerturbationAxisKind` applies to every
source `BacktestSpec` (e.g. `PROBABILITY_THRESHOLD` only means something
when `signal_mapping.kind == PROBABILITY_BANDS`). An axis that does not
apply to the source spec is reported as an explicit, whole-axis skip
(`AxisSensitivityResult.skipped=True`) with a stated reason -- never
silently omitted and never silently coerced into a different axis.
`ABSTENTION_THRESHOLD` is unconditionally skipped: this platform's
abstention control (`BacktestSpec.respect_calibration_abstention`) is a
boolean, and the actual abstention THRESHOLD is a calibration-time
concept outside this backtest-level re-simulation's reach.

A single declared perturbation `relative_delta` can also independently
fail to produce a valid point -- e.g. shrinking `holding_period_bars`
below 1, or inverting a probability band -- WITHOUT invalidating the rest
of the axis; that one point is recorded with
`status="skipped_invalid_perturbed_value"` and an explicit reason. If the
production pipeline itself refuses to simulate an otherwise
structurally-valid perturbed spec, the point is recorded with
`status="skipped_unsimulable"`. No perturbation point ever silently
disappears from the report."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from quant_platform.backtesting.models import (
    CommissionModelKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinancingModelKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.runner import ResolvedBacktestInputs
from quant_platform.backtesting.specs import BacktestSpec
from quant_platform.core.exceptions import SensitivityAnalysisError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import PerturbationAxisKind
from quant_platform.robustness.resimulation import ResimulationResult, resimulate_stitched_outcome
from quant_platform.robustness.source import VerifiedBacktestSource
from quant_platform.robustness.specs import PerturbationSpec, RobustnessSpec

SENSITIVITY_REPORT_SCHEMA_VERSION = 1


def _bounded_or_skip(value: float, *, lo: float, hi: float, field_name: str) -> tuple[float | None, str | None]:
    if not math.isfinite(value):
        return None, f"perturbed {field_name} is not finite ({value!r})"
    if not (lo <= value <= hi):
        return None, f"perturbed {field_name}={value!r} is outside its valid domain [{lo}, {hi}]"
    return value, None


# --------------------------------------------------------------------------
# Per-axis structural applicability (Section 9: an axis that does not apply
# to the source spec's own modeling choices is skipped, never coerced).
# --------------------------------------------------------------------------
def axis_structural_skip_reason(axis: PerturbationAxisKind, spec: BacktestSpec) -> str | None:
    """Returns `None` if `axis` is structurally applicable to `spec`, else
    a human-readable reason the WHOLE axis is skipped before any
    individual `relative_delta` is even attempted."""
    if axis is PerturbationAxisKind.ABSTENTION_THRESHOLD:
        return (
            "not supported for re-simulation at this layer: abstention is controlled by "
            "BacktestSpec.respect_calibration_abstention (a boolean), and the underlying abstention THRESHOLD "
            "is a calibration-time concept this backtest-level re-simulation cannot perturb"
        )
    sm = spec.signal_mapping
    if axis is PerturbationAxisKind.PROBABILITY_THRESHOLD and sm.kind is not SignalMappingPolicyKind.PROBABILITY_BANDS:
        return f"not applicable: signal_mapping.kind={sm.kind.value!r} != probability_bands"
    if axis is PerturbationAxisKind.CONFIDENCE_THRESHOLD and sm.kind not in (
        SignalMappingPolicyKind.CONFIDENCE_FLOOR, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY,
    ):
        return f"not applicable: signal_mapping.kind={sm.kind.value!r} has no confidence_floor"
    if axis is PerturbationAxisKind.UNCERTAINTY_THRESHOLD and sm.kind not in (
        SignalMappingPolicyKind.UNCERTAINTY_CEILING, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY,
    ):
        return f"not applicable: signal_mapping.kind={sm.kind.value!r} has no uncertainty_ceiling"
    if axis is PerturbationAxisKind.HOLDING_PERIOD_BARS and spec.exit_spec.kind is not ExitPolicyKind.FIXED_HORIZON:
        return f"not applicable: exit_spec.kind={spec.exit_spec.kind.value!r} != fixed_horizon (no holding_period_bars)"
    if axis is PerturbationAxisKind.SPREAD_BASIS_POINTS and spec.spread_spec.kind is not SpreadModelKind.FIXED_BASIS_POINTS:
        return f"not applicable: spread_spec.kind={spec.spread_spec.kind.value!r} != fixed_basis_points"
    if axis is PerturbationAxisKind.COMMISSION_BASIS_POINTS and spec.commission_spec.kind is not CommissionModelKind.PER_SIDE_BASIS_POINTS:
        return f"not applicable: commission_spec.kind={spec.commission_spec.kind.value!r} != per_side_basis_points"
    if axis is PerturbationAxisKind.SLIPPAGE_BASIS_POINTS and spec.slippage_spec.kind is not SlippageModelKind.FIXED_BASIS_POINTS:
        return f"not applicable: slippage_spec.kind={spec.slippage_spec.kind.value!r} != fixed_basis_points"
    if axis is PerturbationAxisKind.FINANCING_BASIS_POINTS and spec.financing_spec.kind is not FinancingModelKind.FIXED_DAILY_BASIS_POINTS:
        return f"not applicable: financing_spec.kind={spec.financing_spec.kind.value!r} != fixed_daily_basis_points"
    # ENTRY_DELAY_BARS and EXPOSURE_CAP are always structurally applicable.
    return None


# --------------------------------------------------------------------------
# Per-axis spec modifiers: (source spec, relative_delta) -> (modified spec,
# reported applied value, skip reason). Exactly one of (modified spec,
# skip reason) is non-None. Never clamps a perturbed value into range --
# an out-of-domain perturbation is an explicit skipped point, not a
# silently distorted one.
# --------------------------------------------------------------------------
_PerturberFn = Callable[[BacktestSpec, float], "tuple[BacktestSpec | None, float | None, str | None]"]


def _perturb_probability_threshold(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    sm = spec.signal_mapping
    assert sm.probability_band_long_min is not None and sm.probability_band_short_max is not None
    new_long_min, reason = _bounded_or_skip(sm.probability_band_long_min * (1.0 + delta), lo=0.0, hi=1.0, field_name="probability_band_long_min")
    if new_long_min is None:
        return None, None, reason
    new_short_max, reason = _bounded_or_skip(sm.probability_band_short_max * (1.0 + delta), lo=0.0, hi=1.0, field_name="probability_band_short_max")
    if new_short_max is None:
        return None, None, reason
    if new_short_max >= new_long_min:
        return None, None, f"perturbed probability band is invalid: short_max={new_short_max!r} >= long_min={new_long_min!r}"
    new_sm = replace(sm, probability_band_long_min=new_long_min, probability_band_short_max=new_short_max)
    return replace(spec, signal_mapping=new_sm), new_long_min, None


def _perturb_confidence_threshold(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    sm = spec.signal_mapping
    assert sm.confidence_floor is not None
    value, reason = _bounded_or_skip(sm.confidence_floor * (1.0 + delta), lo=0.0, hi=1.0, field_name="confidence_floor")
    if value is None:
        return None, None, reason
    return replace(spec, signal_mapping=replace(sm, confidence_floor=value)), value, None


def _perturb_uncertainty_threshold(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    sm = spec.signal_mapping
    assert sm.uncertainty_ceiling is not None
    value, reason = _bounded_or_skip(sm.uncertainty_ceiling * (1.0 + delta), lo=0.0, hi=1.0, field_name="uncertainty_ceiling")
    if value is None:
        return None, None, reason
    return replace(spec, signal_mapping=replace(sm, uncertainty_ceiling=value)), value, None


def _perturb_holding_period(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    ex = spec.exit_spec
    assert ex.holding_period_bars is not None
    candidate = round(ex.holding_period_bars * (1.0 + delta))
    if candidate < 1:
        return None, None, f"perturbed holding_period_bars={candidate} < 1"
    return replace(spec, exit_spec=replace(ex, holding_period_bars=candidate)), float(candidate), None


def _perturb_entry_delay(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    en = spec.entry_spec
    candidate = round(en.delay_bars * (1.0 + delta))
    minimum = 1 if en.kind is EntryPolicyKind.DELAYED_BAR else 0
    if candidate < minimum:
        return None, None, f"perturbed delay_bars={candidate} < minimum {minimum} required for kind={en.kind.value!r}"
    if candidate == 0 and not en.allow_same_bar_close:
        return None, None, "perturbed delay_bars=0 requires allow_same_bar_close=True"
    return replace(spec, entry_spec=replace(en, delay_bars=candidate)), float(candidate), None


def _perturb_spread_bps(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    sp = spec.spread_spec
    assert sp.basis_points is not None
    value, reason = _bounded_or_skip(sp.basis_points * (1.0 + delta), lo=0.0, hi=math.inf, field_name="spread_spec.basis_points")
    if value is None:
        return None, None, reason
    return replace(spec, spread_spec=replace(sp, basis_points=value)), value, None


def _perturb_commission_bps(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    cs = spec.commission_spec
    assert cs.per_side_basis_points is not None
    value, reason = _bounded_or_skip(cs.per_side_basis_points * (1.0 + delta), lo=0.0, hi=math.inf, field_name="commission_spec.per_side_basis_points")
    if value is None:
        return None, None, reason
    return replace(spec, commission_spec=replace(cs, per_side_basis_points=value)), value, None


def _perturb_slippage_bps(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    sl = spec.slippage_spec
    assert sl.basis_points is not None
    value, reason = _bounded_or_skip(sl.basis_points * (1.0 + delta), lo=0.0, hi=math.inf, field_name="slippage_spec.basis_points")
    if value is None:
        return None, None, reason
    return replace(spec, slippage_spec=replace(sl, basis_points=value)), value, None


def _perturb_financing_bps(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    fs = spec.financing_spec
    assert fs.daily_basis_points is not None
    value, reason = _bounded_or_skip(fs.daily_basis_points * (1.0 + delta), lo=-math.inf, hi=math.inf, field_name="financing_spec.daily_basis_points")
    if value is None:
        return None, None, reason
    return replace(spec, financing_spec=replace(fs, daily_basis_points=value)), value, None


def _perturb_exposure_cap(spec: BacktestSpec, delta: float) -> tuple[BacktestSpec | None, float | None, str | None]:
    candidate = spec.exposure_cap * (1.0 + delta)
    if not math.isfinite(candidate) or candidate <= 0.0:
        return None, None, f"perturbed exposure_cap={candidate!r} must be finite and > 0"
    return replace(spec, exposure_cap=candidate), candidate, None


_PERTURBERS: dict[PerturbationAxisKind, _PerturberFn] = {
    PerturbationAxisKind.PROBABILITY_THRESHOLD: _perturb_probability_threshold,
    PerturbationAxisKind.CONFIDENCE_THRESHOLD: _perturb_confidence_threshold,
    PerturbationAxisKind.UNCERTAINTY_THRESHOLD: _perturb_uncertainty_threshold,
    PerturbationAxisKind.HOLDING_PERIOD_BARS: _perturb_holding_period,
    PerturbationAxisKind.ENTRY_DELAY_BARS: _perturb_entry_delay,
    PerturbationAxisKind.SPREAD_BASIS_POINTS: _perturb_spread_bps,
    PerturbationAxisKind.COMMISSION_BASIS_POINTS: _perturb_commission_bps,
    PerturbationAxisKind.SLIPPAGE_BASIS_POINTS: _perturb_slippage_bps,
    PerturbationAxisKind.FINANCING_BASIS_POINTS: _perturb_financing_bps,
    PerturbationAxisKind.EXPOSURE_CAP: _perturb_exposure_cap,
}


# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PerturbationPointResult:
    relative_delta: float
    applied_value: float | None
    status: str
    """One of: `evaluated`, `skipped_invalid_perturbed_value` (the
    perturbed value itself falls outside this axis's valid domain, e.g.
    an inverted probability band or a sub-1 holding period),
    `skipped_unsimulable` (the value was structurally valid but the
    production pipeline itself could not simulate it)."""
    skip_reason: str | None
    total_net_return: float | None
    total_gross_return: float | None
    closed_trade_count: int | None
    maximum_drawdown: float | None
    is_profitable: bool | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relative_delta": self.relative_delta, "applied_value": self.applied_value, "status": self.status,
            "skip_reason": self.skip_reason, "total_net_return": self.total_net_return, "total_gross_return": self.total_gross_return,
            "closed_trade_count": self.closed_trade_count, "maximum_drawdown": self.maximum_drawdown, "is_profitable": self.is_profitable,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PerturbationPointResult:
        def _opt_float(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            relative_delta=float(str(raw["relative_delta"])), applied_value=_opt_float("applied_value"), status=str(raw["status"]),
            skip_reason=(None if raw.get("skip_reason") is None else str(raw["skip_reason"])), total_net_return=_opt_float("total_net_return"),
            total_gross_return=_opt_float("total_gross_return"),
            closed_trade_count=(None if raw.get("closed_trade_count") is None else int(str(raw["closed_trade_count"]))),
            maximum_drawdown=_opt_float("maximum_drawdown"),
            is_profitable=(None if raw.get("is_profitable") is None else bool(raw["is_profitable"])),
        )


@dataclass(frozen=True, slots=True)
class AxisSensitivityResult:
    """Section 9's per-axis report. When `skipped` is True every
    aggregate field is `None` and `points` is empty -- the axis was never
    attempted, not attempted-and-empty.

    Aggregate field definitions (deliberately simple, literal, and
    documented here rather than borrowed from any named statistical
    procedure -- there is no canonical "parameter sensitivity score" in
    the literature this platform can cite, so the exact formula is
    spelled out to avoid any appearance of borrowed authority):

    - `monotonicity_violations`: among the baseline (delta=0) point plus
      every `evaluated` point, sorted by `relative_delta`, the number of
      times the sign of the return change between consecutive points
      flips (a zero-change step is not counted as a direction).
    - `cliff_detected`: True if the evaluated point with the smallest
      nonzero |relative_delta| immediately adjacent to the baseline (on
      either side) has `is_profitable != ` the baseline's profitability
      -- a small perturbation flips the qualitative outcome.
    - `rank_stable`: True if the baseline's `total_net_return` is not the
      worst value among {baseline} union {evaluated points} -- a coarse,
      single-axis, single-candidate proxy. This is NOT the same concept
      as Section 12's cross-candidate selection-ranking stability.
    - `profitable_neighborhood_fraction`: fraction of `evaluated` points
      (excluding the baseline) with `is_profitable=True`.
    - `parameter_sensitivity_score`: `max(|evaluated.total_net_return -
      baseline_total_net_return|) / max(|baseline_total_net_return|,
      1e-9)` -- the largest fractional swing in total net return observed
      anywhere in the declared perturbation range for this axis.
    """

    axis: PerturbationAxisKind
    skipped: bool
    skip_reason: str | None
    baseline_total_net_return: float | None
    points: tuple[PerturbationPointResult, ...]
    monotonicity_violations: int | None
    cliff_detected: bool | None
    rank_stable: bool | None
    profitable_neighborhood_fraction: float | None
    parameter_sensitivity_score: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis.value, "skipped": self.skipped, "skip_reason": self.skip_reason,
            "baseline_total_net_return": self.baseline_total_net_return, "points": [p.to_json_dict() for p in self.points],
            "monotonicity_violations": self.monotonicity_violations, "cliff_detected": self.cliff_detected, "rank_stable": self.rank_stable,
            "profitable_neighborhood_fraction": self.profitable_neighborhood_fraction, "parameter_sensitivity_score": self.parameter_sensitivity_score,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AxisSensitivityResult:
        def _opt_float(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            axis=PerturbationAxisKind(raw["axis"]), skipped=bool(raw["skipped"]),
            skip_reason=(None if raw.get("skip_reason") is None else str(raw["skip_reason"])),
            baseline_total_net_return=_opt_float("baseline_total_net_return"),
            points=tuple(PerturbationPointResult.from_json_dict(as_json_dict(p, field_name="points[]")) for p in as_json_list(raw.get("points") or [], field_name="points")),
            monotonicity_violations=(None if raw.get("monotonicity_violations") is None else int(str(raw["monotonicity_violations"]))),
            cliff_detected=(None if raw.get("cliff_detected") is None else bool(raw["cliff_detected"])),
            rank_stable=(None if raw.get("rank_stable") is None else bool(raw["rank_stable"])),
            profitable_neighborhood_fraction=_opt_float("profitable_neighborhood_fraction"),
            parameter_sensitivity_score=_opt_float("parameter_sensitivity_score"),
        )


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    schema_version: int
    source_backtest_id: str
    baseline_total_net_return: float
    baseline_total_gross_return: float
    baseline_closed_trade_count: int
    baseline_maximum_drawdown: float
    axis_results: tuple[AxisSensitivityResult, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_backtest_id": self.source_backtest_id,
            "baseline_total_net_return": self.baseline_total_net_return, "baseline_total_gross_return": self.baseline_total_gross_return,
            "baseline_closed_trade_count": self.baseline_closed_trade_count, "baseline_maximum_drawdown": self.baseline_maximum_drawdown,
            "axis_results": [a.to_json_dict() for a in self.axis_results], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SensitivityReport:
        require_schema_version(raw, supported=SENSITIVITY_REPORT_SCHEMA_VERSION, context="SensitivityReport")
        return cls(
            schema_version=SENSITIVITY_REPORT_SCHEMA_VERSION, source_backtest_id=str(raw["source_backtest_id"]),
            baseline_total_net_return=float(str(raw["baseline_total_net_return"])), baseline_total_gross_return=float(str(raw["baseline_total_gross_return"])),
            baseline_closed_trade_count=int(str(raw["baseline_closed_trade_count"])), baseline_maximum_drawdown=float(str(raw["baseline_maximum_drawdown"])),
            axis_results=tuple(
                AxisSensitivityResult.from_json_dict(as_json_dict(a, field_name="axis_results[]")) for a in as_json_list(raw.get("axis_results") or [], field_name="axis_results")
            ),
            generated_at=str(raw["generated_at"]),
        )


# --------------------------------------------------------------------------
# Aggregate statistics over one axis's evaluated points
# --------------------------------------------------------------------------
def _count_monotonicity_violations(sorted_deltas_and_returns: list[tuple[float, float]]) -> int:
    diffs = [b[1] - a[1] for a, b in itertools.pairwise(sorted_deltas_and_returns)]
    signs = [0 if d == 0.0 else (1 if d > 0.0 else -1) for d in diffs]
    nonzero_signs = [s for s in signs if s != 0]
    return sum(1 for a, b in itertools.pairwise(nonzero_signs) if a != b)


def _compute_axis_aggregates(
    *, baseline_net_return: float, baseline_profitable: bool, evaluated: list[PerturbationPointResult],
) -> tuple[int | None, bool | None, bool | None, float | None, float | None]:
    if not evaluated:
        return None, None, None, None, None

    points_by_delta = sorted(((p.relative_delta, p.total_net_return) for p in evaluated if p.total_net_return is not None), key=lambda t: t[0])
    all_points = sorted({(0.0, baseline_net_return), *points_by_delta}, key=lambda t: t[0])
    monotonicity_violations = _count_monotonicity_violations(all_points) if len(all_points) >= 3 else 0

    negative_neighbor = max((p for p in evaluated if p.relative_delta < 0.0), key=lambda p: p.relative_delta, default=None)
    positive_neighbor = min((p for p in evaluated if p.relative_delta > 0.0), key=lambda p: p.relative_delta, default=None)
    cliff_detected = any(
        neighbor is not None and neighbor.is_profitable is not None and neighbor.is_profitable != baseline_profitable
        for neighbor in (negative_neighbor, positive_neighbor)
    )

    all_returns = [baseline_net_return, *(p.total_net_return for p in evaluated if p.total_net_return is not None)]
    rank_stable = baseline_net_return > min(all_returns) or len(all_returns) == 1 or all(r == baseline_net_return for r in all_returns)

    profitable_flags = [p.is_profitable for p in evaluated if p.is_profitable is not None]
    profitable_neighborhood_fraction = (sum(1 for f in profitable_flags if f) / len(profitable_flags)) if profitable_flags else None

    swings = [abs(p.total_net_return - baseline_net_return) for p in evaluated if p.total_net_return is not None]
    parameter_sensitivity_score = (max(swings) / max(abs(baseline_net_return), 1e-9)) if swings else None

    return monotonicity_violations, cliff_detected, rank_stable, profitable_neighborhood_fraction, parameter_sensitivity_score


def _compute_axis_sensitivity(
    perturbation: PerturbationSpec, *, source_backtest_spec: BacktestSpec, resolved_inputs: ResolvedBacktestInputs,
    baseline: ResimulationResult, robustness_source_backtest_id: str, artifact_store: MLArtifactStore,
) -> AxisSensitivityResult:
    axis = perturbation.axis
    structural_reason = axis_structural_skip_reason(axis, source_backtest_spec)
    if structural_reason is not None:
        return AxisSensitivityResult(
            axis=axis, skipped=True, skip_reason=structural_reason, baseline_total_net_return=None, points=(),
            monotonicity_violations=None, cliff_detected=None, rank_stable=None, profitable_neighborhood_fraction=None, parameter_sensitivity_score=None,
        )

    perturber = _PERTURBERS[axis]
    baseline_profitable = baseline.total_net_return > 0.0
    points: list[PerturbationPointResult] = []
    for i, delta in enumerate(perturbation.relative_deltas):
        modified_spec, applied_value, skip_reason = perturber(source_backtest_spec, delta)
        if modified_spec is None:
            points.append(PerturbationPointResult(
                relative_delta=delta, applied_value=None, status="skipped_invalid_perturbed_value", skip_reason=skip_reason,
                total_net_return=None, total_gross_return=None, closed_trade_count=None, maximum_drawdown=None, is_profitable=None,
            ))
            continue
        label = f"{robustness_source_backtest_id}:sensitivity:{axis.value}:{i}"
        result = resimulate_stitched_outcome(modified_spec=modified_spec, resolved_inputs=resolved_inputs, label_backtest_id=label, artifact_store=artifact_store)
        if result is None:
            points.append(PerturbationPointResult(
                relative_delta=delta, applied_value=applied_value, status="skipped_unsimulable",
                skip_reason="the perturbed spec was structurally valid but the production simulation pipeline could not evaluate it",
                total_net_return=None, total_gross_return=None, closed_trade_count=None, maximum_drawdown=None, is_profitable=None,
            ))
            continue
        points.append(PerturbationPointResult(
            relative_delta=delta, applied_value=applied_value, status="evaluated", skip_reason=None,
            total_net_return=result.total_net_return, total_gross_return=result.total_gross_return,
            closed_trade_count=result.closed_trade_count, maximum_drawdown=result.maximum_drawdown,
            is_profitable=result.total_net_return > 0.0,
        ))

    evaluated = [p for p in points if p.status == "evaluated"]
    monotonicity_violations, cliff_detected, rank_stable, profitable_neighborhood_fraction, parameter_sensitivity_score = _compute_axis_aggregates(
        baseline_net_return=baseline.total_net_return, baseline_profitable=baseline_profitable, evaluated=evaluated,
    )
    return AxisSensitivityResult(
        axis=axis, skipped=False, skip_reason=None, baseline_total_net_return=baseline.total_net_return, points=tuple(points),
        monotonicity_violations=monotonicity_violations, cliff_detected=cliff_detected, rank_stable=rank_stable,
        profitable_neighborhood_fraction=profitable_neighborhood_fraction, parameter_sensitivity_score=parameter_sensitivity_score,
    )


def compute_sensitivity_report(
    *, source: VerifiedBacktestSource, resolved_inputs: ResolvedBacktestInputs, spec: RobustnessSpec, artifact_store: MLArtifactStore,
) -> SensitivityReport:
    """Section 9's entry point: re-simulates the UNPERTURBED source spec
    once as the baseline, then every declared `spec.perturbations` axis
    around it. Never searches for a better parameter -- every perturbed
    point is independently reported, never selected or fed back."""
    baseline = resimulate_stitched_outcome(
        modified_spec=source.backtest_spec, resolved_inputs=resolved_inputs, label_backtest_id=f"{spec.source_backtest_id}:sensitivity:baseline",
        artifact_store=artifact_store,
    )
    if baseline is None:
        raise SensitivityAnalysisError(
            f"compute_sensitivity_report: re-simulating the UNPERTURBED source BacktestSpec for "
            f"source_backtest_id={spec.source_backtest_id!r} itself failed to produce a result -- this indicates a "
            "reproducibility defect (the original backtest completed, but re-running its own exact spec through the "
            "same pipeline did not), not a perturbation-domain issue",
            context={"source_backtest_id": spec.source_backtest_id},
        )

    axis_results = tuple(
        _compute_axis_sensitivity(
            p, source_backtest_spec=source.backtest_spec, resolved_inputs=resolved_inputs, baseline=baseline,
            robustness_source_backtest_id=spec.source_backtest_id, artifact_store=artifact_store,
        )
        for p in spec.perturbations
    )
    return SensitivityReport(
        schema_version=SENSITIVITY_REPORT_SCHEMA_VERSION, source_backtest_id=spec.source_backtest_id,
        baseline_total_net_return=baseline.total_net_return, baseline_total_gross_return=baseline.total_gross_return,
        baseline_closed_trade_count=baseline.closed_trade_count, baseline_maximum_drawdown=baseline.maximum_drawdown,
        axis_results=axis_results, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "SENSITIVITY_REPORT_SCHEMA_VERSION",
    "AxisSensitivityResult",
    "PerturbationPointResult",
    "SensitivityReport",
    "axis_structural_skip_reason",
    "compute_sensitivity_report",
]
