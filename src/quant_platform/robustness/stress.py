"""Cost, latency, and execution stress analysis (Milestone 6, Section
10). Every scenario is a pre-declared, deterministic configuration --
never a random or adaptive search -- re-simulated through the exact
production pipeline (`resimulation.resimulate_stitched_outcome`) under a
`dataclasses.replace`-modified `BacktestSpec`. `NAMED_XAUUSD_STRESS_
PROFILES` are illustrative CONFIGURATION IDENTITIES for a future
XAUUSD/MT5 integration -- they are not claims about actual broker
behavior, spread widening, or slippage observed in any real market or
account.

APPLICABILITY: a multiplier only means something if the source spec's
own cost model has a scalable numeric magnitude for that axis (e.g.
`spread_multiplier` has nothing to scale when `spread_spec.kind ==
bid_ask_observed`, since that model has no configured magnitude field at
all). Such a scenario/search point is reported as an explicit skip
(`status="skipped_not_applicable"`), never silently coerced or dropped.

BREAK-EVEN SEARCH: for each of spread/slippage/commission/financing
multiplier and additional latency bars, this module evaluates a FIXED,
DOCUMENTED, deterministic grid of stress magnitudes (`_MULTIPLIER_
SEARCH_GRID` / `_LATENCY_SEARCH_GRID`) and reports the tightest bracket
on that grid where `total_net_return` crosses from positive to
non-positive. This is NOT root-finding by interpolation -- no value
between two searched grid points is ever fabricated; the report states
the bracket, or explicitly that no crossing was found within the
declared bounds. The search is a left-to-right FIRST-crossing scan
(`itertools.pairwise`, early return) -- for a non-monotonic response it
reports only the first `positive -> non-positive` bracket encountered on
the grid, never every root.

MONOTONICITY (closure-audit finding, Sections 4-5): `_apply_cost_stress`
modifies the underlying `BacktestSpec` and RE-RUNS the full production
simulation (`resimulate_stitched_outcome`) -- it is never a fixed-trade
arithmetic replay. This matters differently per axis:
- `spread_multiplier`/`slippage_multiplier`/`commission_multiplier`/
  `financing_multiplier`: entry/exit bar position and direction are
  decided entirely by `EntryPolicyKind`/`ExitPolicyKind`/signal logic,
  NONE of which reads any cost spec (cost specs are consumed only in
  `backtesting.runner`'s post-hoc price/return adjustment, strictly
  AFTER the trade's timing is already fixed) -- so for a FIXED trade
  path, increasing any of these four multipliers can only increase
  (never decrease) total deducted cost, since `CostBreakdown` rejects
  negative components. `total_net_return` is therefore structurally
  monotonic non-increasing in each of these four multipliers alone.
- `additional_latency_bars`: shifts `entry_spec.delay_bars`, which
  shifts WHICH bar a trade actually enters on -- a genuine execution-
  decision change (different entry price, different holding-period
  alignment, a trade possibly pushed past the fold boundary and
  discarded/force-closed differently). Monotonicity of `total_net_
  return` in this axis is NOT structurally guaranteed the way it is for
  the four pure-cost multipliers; its break-even search must be read as
  "response to a genuinely different trade path under added latency",
  not "response to a fixed trade path under heavier cost"."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.runner import ResolvedBacktestInputs
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    FinancingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.core.exceptions import StressAnalysisError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import StressAxisKind
from quant_platform.robustness.resimulation import ResimulationResult, resimulate_stitched_outcome
from quant_platform.robustness.source import VerifiedBacktestSource
from quant_platform.robustness.specs import RobustnessSpec, StressScenarioSpec

STRESS_REPORT_SCHEMA_VERSION = 1

NAMED_XAUUSD_STRESS_PROFILES: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec(name="normal_liquidity", axis=StressAxisKind.NAMED_PROFILE, named_profile="normal_liquidity"),
    StressScenarioSpec(name="rollover_spread_expansion", axis=StressAxisKind.NAMED_PROFILE, named_profile="rollover_spread_expansion", spread_multiplier=3.0),
    StressScenarioSpec(
        name="high_impact_macro_release", axis=StressAxisKind.NAMED_PROFILE, named_profile="high_impact_macro_release",
        spread_multiplier=2.5, slippage_multiplier=3.0, additional_latency_bars=1,
    ),
    StressScenarioSpec(name="thin_session", axis=StressAxisKind.NAMED_PROFILE, named_profile="thin_session", spread_multiplier=1.8, slippage_multiplier=1.5),
    StressScenarioSpec(
        name="broker_degradation", axis=StressAxisKind.NAMED_PROFILE, named_profile="broker_degradation",
        spread_multiplier=2.0, slippage_multiplier=2.0, commission_multiplier=1.5, additional_latency_bars=2,
    ),
)
"""Section 10's own instruction: 'these are configuration identities, not
claims about actual broker behavior.' Not part of `DEFAULT_STRESS_
SCENARIOS` -- an operator opts in by including these (or their own) in
`RobustnessSpec.stress_scenarios`."""

_MULTIPLIER_SEARCH_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
_LATENCY_SEARCH_GRID: tuple[float, ...] = tuple(float(b) for b in range(0, 11))


# --------------------------------------------------------------------------
# Spec modification: one shared helper for both named scenarios and the
# break-even grid search.
# --------------------------------------------------------------------------
def _apply_cost_stress(
    spec: BacktestSpec, *, spread_multiplier: float = 1.0, slippage_multiplier: float = 1.0, commission_multiplier: float = 1.0,
    financing_multiplier: float = 1.0, additional_latency_bars: int = 0, force_zero_cost: bool = False,
) -> tuple[BacktestSpec | None, str | None]:
    if force_zero_cost:
        return (
            replace(
                spec, spread_spec=SpreadSpec(kind=SpreadModelKind.ZERO), commission_spec=CommissionSpec(kind=CommissionModelKind.ZERO),
                slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
            ),
            None,
        )

    new_spread = spec.spread_spec
    if spread_multiplier != 1.0:
        if spec.spread_spec.kind is SpreadModelKind.FIXED_BASIS_POINTS:
            assert spec.spread_spec.basis_points is not None
            new_spread = replace(spec.spread_spec, basis_points=spec.spread_spec.basis_points * spread_multiplier)
        elif spec.spread_spec.kind is SpreadModelKind.FIXED_PRICE_UNITS:
            assert spec.spread_spec.price_units is not None
            new_spread = replace(spec.spread_spec, price_units=spec.spread_spec.price_units * spread_multiplier)
        elif spec.spread_spec.kind is not SpreadModelKind.ZERO:
            return None, f"spread_multiplier={spread_multiplier!r} not applicable: spread_spec.kind={spec.spread_spec.kind.value!r} has no scalable magnitude"

    new_commission = spec.commission_spec
    if commission_multiplier != 1.0:
        if spec.commission_spec.kind is CommissionModelKind.PER_SIDE_BASIS_POINTS:
            assert spec.commission_spec.per_side_basis_points is not None
            new_commission = replace(spec.commission_spec, per_side_basis_points=spec.commission_spec.per_side_basis_points * commission_multiplier)
        elif spec.commission_spec.kind is CommissionModelKind.FIXED_PER_TRADE:
            assert spec.commission_spec.fixed_per_trade is not None
            new_commission = replace(spec.commission_spec, fixed_per_trade=spec.commission_spec.fixed_per_trade * commission_multiplier)

    new_slippage = spec.slippage_spec
    if slippage_multiplier != 1.0:
        if spec.slippage_spec.kind is SlippageModelKind.FIXED_BASIS_POINTS:
            assert spec.slippage_spec.basis_points is not None
            new_slippage = replace(spec.slippage_spec, basis_points=spec.slippage_spec.basis_points * slippage_multiplier)
        elif spec.slippage_spec.kind is SlippageModelKind.FIXED_PRICE_UNITS:
            assert spec.slippage_spec.price_units is not None
            new_slippage = replace(spec.slippage_spec, price_units=spec.slippage_spec.price_units * slippage_multiplier)

    new_financing = spec.financing_spec
    if financing_multiplier != 1.0 and spec.financing_spec.kind is FinancingModelKind.FIXED_DAILY_BASIS_POINTS:
        assert spec.financing_spec.daily_basis_points is not None
        new_financing = replace(spec.financing_spec, daily_basis_points=spec.financing_spec.daily_basis_points * financing_multiplier)

    new_entry = spec.entry_spec
    if additional_latency_bars != 0:
        new_entry = replace(spec.entry_spec, delay_bars=spec.entry_spec.delay_bars + additional_latency_bars)

    return replace(spec, spread_spec=new_spread, commission_spec=new_commission, slippage_spec=new_slippage, financing_spec=new_financing, entry_spec=new_entry), None


def _apply_scenario(spec: BacktestSpec, scenario: StressScenarioSpec) -> tuple[BacktestSpec | None, str | None]:
    if scenario.axis is StressAxisKind.ZERO_COST:
        return _apply_cost_stress(spec, force_zero_cost=True)
    return _apply_cost_stress(
        spec, spread_multiplier=scenario.spread_multiplier, slippage_multiplier=scenario.slippage_multiplier,
        commission_multiplier=scenario.commission_multiplier, financing_multiplier=scenario.financing_multiplier,
        additional_latency_bars=scenario.additional_latency_bars,
    )


# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StressScenarioResult:
    name: str
    axis: StressAxisKind
    named_profile: str | None
    status: str
    """`evaluated` | `skipped_not_applicable` (a declared multiplier has
    no scalable magnitude in the source spec's cost model)."""
    skip_reason: str | None
    total_net_return: float | None
    total_gross_return: float | None
    closed_trade_count: int | None
    maximum_drawdown: float | None
    is_profitable: bool | None
    net_return_degradation_vs_baseline: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "axis": self.axis.value, "named_profile": self.named_profile, "status": self.status, "skip_reason": self.skip_reason,
            "total_net_return": self.total_net_return, "total_gross_return": self.total_gross_return, "closed_trade_count": self.closed_trade_count,
            "maximum_drawdown": self.maximum_drawdown, "is_profitable": self.is_profitable, "net_return_degradation_vs_baseline": self.net_return_degradation_vs_baseline,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StressScenarioResult:
        def _opt_float(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            name=str(raw["name"]), axis=StressAxisKind(raw["axis"]), named_profile=(None if raw.get("named_profile") is None else str(raw["named_profile"])),
            status=str(raw["status"]), skip_reason=(None if raw.get("skip_reason") is None else str(raw["skip_reason"])),
            total_net_return=_opt_float("total_net_return"), total_gross_return=_opt_float("total_gross_return"),
            closed_trade_count=(None if raw.get("closed_trade_count") is None else int(str(raw["closed_trade_count"]))),
            maximum_drawdown=_opt_float("maximum_drawdown"), is_profitable=(None if raw.get("is_profitable") is None else bool(raw["is_profitable"])),
            net_return_degradation_vs_baseline=_opt_float("net_return_degradation_vs_baseline"),
        )


@dataclass(frozen=True, slots=True)
class BreakEvenSearchPoint:
    value: float
    status: str
    skip_reason: str | None
    total_net_return: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {"value": self.value, "status": self.status, "skip_reason": self.skip_reason, "total_net_return": self.total_net_return}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BreakEvenSearchPoint:
        v = raw.get("total_net_return")
        return cls(
            value=float(str(raw["value"])), status=str(raw["status"]), skip_reason=(None if raw.get("skip_reason") is None else str(raw["skip_reason"])),
            total_net_return=(None if v is None else float(str(v))),
        )


@dataclass(frozen=True, slots=True)
class BreakEvenResult:
    axis_name: str
    found: bool
    breakeven_lower_bound: float | None
    """The largest searched value at which the strategy was still
    profitable (`total_net_return > 0`)."""
    breakeven_upper_bound: float | None
    """The smallest searched value at which `total_net_return <= 0`."""
    reason: str | None
    searched_points: tuple[BreakEvenSearchPoint, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "axis_name": self.axis_name, "found": self.found, "breakeven_lower_bound": self.breakeven_lower_bound,
            "breakeven_upper_bound": self.breakeven_upper_bound, "reason": self.reason,
            "searched_points": [p.to_json_dict() for p in self.searched_points],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BreakEvenResult:
        def _opt_float(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            axis_name=str(raw["axis_name"]), found=bool(raw["found"]), breakeven_lower_bound=_opt_float("breakeven_lower_bound"),
            breakeven_upper_bound=_opt_float("breakeven_upper_bound"), reason=(None if raw.get("reason") is None else str(raw["reason"])),
            searched_points=tuple(
                BreakEvenSearchPoint.from_json_dict(as_json_dict(p, field_name="searched_points[]")) for p in as_json_list(raw.get("searched_points") or [], field_name="searched_points")
            ),
        )


@dataclass(frozen=True, slots=True)
class StressReport:
    schema_version: int
    source_backtest_id: str
    baseline_total_net_return: float
    baseline_total_gross_return: float
    baseline_closed_trade_count: int
    baseline_maximum_drawdown: float
    scenario_results: tuple[StressScenarioResult, ...]
    breakeven_results: tuple[BreakEvenResult, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_backtest_id": self.source_backtest_id,
            "baseline_total_net_return": self.baseline_total_net_return, "baseline_total_gross_return": self.baseline_total_gross_return,
            "baseline_closed_trade_count": self.baseline_closed_trade_count, "baseline_maximum_drawdown": self.baseline_maximum_drawdown,
            "scenario_results": [s.to_json_dict() for s in self.scenario_results], "breakeven_results": [b.to_json_dict() for b in self.breakeven_results],
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StressReport:
        require_schema_version(raw, supported=STRESS_REPORT_SCHEMA_VERSION, context="StressReport")
        return cls(
            schema_version=STRESS_REPORT_SCHEMA_VERSION, source_backtest_id=str(raw["source_backtest_id"]),
            baseline_total_net_return=float(str(raw["baseline_total_net_return"])), baseline_total_gross_return=float(str(raw["baseline_total_gross_return"])),
            baseline_closed_trade_count=int(str(raw["baseline_closed_trade_count"])), baseline_maximum_drawdown=float(str(raw["baseline_maximum_drawdown"])),
            scenario_results=tuple(
                StressScenarioResult.from_json_dict(as_json_dict(s, field_name="scenario_results[]")) for s in as_json_list(raw.get("scenario_results") or [], field_name="scenario_results")
            ),
            breakeven_results=tuple(
                BreakEvenResult.from_json_dict(as_json_dict(b, field_name="breakeven_results[]")) for b in as_json_list(raw.get("breakeven_results") or [], field_name="breakeven_results")
            ),
            generated_at=str(raw["generated_at"]),
        )


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _evaluate_scenario(
    scenario: StressScenarioSpec, *, source_backtest_spec: BacktestSpec, resolved_inputs: ResolvedBacktestInputs, baseline: ResimulationResult,
    robustness_source_backtest_id: str, artifact_store: MLArtifactStore,
) -> StressScenarioResult:
    modified_spec, skip_reason = _apply_scenario(source_backtest_spec, scenario)
    if modified_spec is None:
        return StressScenarioResult(
            name=scenario.name, axis=scenario.axis, named_profile=scenario.named_profile, status="skipped_not_applicable", skip_reason=skip_reason,
            total_net_return=None, total_gross_return=None, closed_trade_count=None, maximum_drawdown=None, is_profitable=None, net_return_degradation_vs_baseline=None,
        )
    label = f"{robustness_source_backtest_id}:stress:{scenario.name}"
    result = resimulate_stitched_outcome(modified_spec=modified_spec, resolved_inputs=resolved_inputs, label_backtest_id=label, artifact_store=artifact_store)
    if result is None:
        return StressScenarioResult(
            name=scenario.name, axis=scenario.axis, named_profile=scenario.named_profile, status="skipped_unsimulable",
            skip_reason="the stressed spec was structurally valid but the production simulation pipeline could not evaluate it",
            total_net_return=None, total_gross_return=None, closed_trade_count=None, maximum_drawdown=None, is_profitable=None, net_return_degradation_vs_baseline=None,
        )
    return StressScenarioResult(
        name=scenario.name, axis=scenario.axis, named_profile=scenario.named_profile, status="evaluated", skip_reason=None,
        total_net_return=result.total_net_return, total_gross_return=result.total_gross_return, closed_trade_count=result.closed_trade_count,
        maximum_drawdown=result.maximum_drawdown, is_profitable=result.total_net_return > 0.0,
        net_return_degradation_vs_baseline=baseline.total_net_return - result.total_net_return,
    )


def _search_breakeven(
    *, axis_name: str, grid: tuple[float, ...], modifier: Callable[[BacktestSpec, float], tuple[BacktestSpec | None, str | None]],
    source_backtest_spec: BacktestSpec, resolved_inputs: ResolvedBacktestInputs, robustness_source_backtest_id: str, artifact_store: MLArtifactStore,
) -> BreakEvenResult:
    points: list[BreakEvenSearchPoint] = []
    for value in grid:
        modified_spec, skip_reason = modifier(source_backtest_spec, value)
        if modified_spec is None:
            points.append(BreakEvenSearchPoint(value=value, status="skipped_not_applicable", skip_reason=skip_reason, total_net_return=None))
            continue
        label = f"{robustness_source_backtest_id}:stress:breakeven:{axis_name}:{value}"
        result = resimulate_stitched_outcome(modified_spec=modified_spec, resolved_inputs=resolved_inputs, label_backtest_id=label, artifact_store=artifact_store)
        if result is None:
            points.append(BreakEvenSearchPoint(
                value=value, status="skipped_unsimulable", skip_reason="the stressed spec could not be evaluated by the production pipeline", total_net_return=None,
            ))
            continue
        points.append(BreakEvenSearchPoint(value=value, status="evaluated", skip_reason=None, total_net_return=result.total_net_return))

    evaluated = [p for p in points if p.status == "evaluated"]
    if len(evaluated) <= 1:
        non_trivial_skips = [p for p in points if p.status != "evaluated"]
        reason = non_trivial_skips[0].skip_reason if non_trivial_skips else "fewer than two searched points could be evaluated"
        return BreakEvenResult(axis_name=axis_name, found=False, breakeven_lower_bound=None, breakeven_upper_bound=None, reason=reason, searched_points=tuple(points))

    assert evaluated[0].total_net_return is not None
    if evaluated[0].total_net_return <= 0.0:
        return BreakEvenResult(
            axis_name=axis_name, found=True, breakeven_lower_bound=None, breakeven_upper_bound=evaluated[0].value,
            reason="already at or below break-even at the smallest searched value", searched_points=tuple(points),
        )

    for lo, hi in itertools.pairwise(evaluated):
        assert lo.total_net_return is not None and hi.total_net_return is not None
        if lo.total_net_return > 0.0 and hi.total_net_return <= 0.0:
            return BreakEvenResult(axis_name=axis_name, found=True, breakeven_lower_bound=lo.value, breakeven_upper_bound=hi.value, reason=None, searched_points=tuple(points))

    return BreakEvenResult(
        axis_name=axis_name, found=False, breakeven_lower_bound=None, breakeven_upper_bound=None,
        reason=f"no finite break-even point exists within the declared search bounds [{evaluated[0].value}, {evaluated[-1].value}]",
        searched_points=tuple(points),
    )


def compute_stress_report(
    *, source: VerifiedBacktestSource, resolved_inputs: ResolvedBacktestInputs, spec: RobustnessSpec, artifact_store: MLArtifactStore,
) -> StressReport:
    """Section 10's entry point: re-simulates the UNSTRESSED source spec
    once as the baseline, every declared `spec.stress_scenarios`, and a
    fixed break-even search grid for each of the four cost multipliers
    plus additional latency."""
    baseline = resimulate_stitched_outcome(
        modified_spec=source.backtest_spec, resolved_inputs=resolved_inputs, label_backtest_id=f"{spec.source_backtest_id}:stress:baseline",
        artifact_store=artifact_store,
    )
    if baseline is None:
        raise StressAnalysisError(
            f"compute_stress_report: re-simulating the UNSTRESSED source BacktestSpec for source_backtest_id={spec.source_backtest_id!r} "
            "itself failed to produce a result -- this indicates a reproducibility defect, not a stress-scenario issue",
            context={"source_backtest_id": spec.source_backtest_id},
        )

    scenario_results = tuple(
        _evaluate_scenario(
            s, source_backtest_spec=source.backtest_spec, resolved_inputs=resolved_inputs, baseline=baseline,
            robustness_source_backtest_id=spec.source_backtest_id, artifact_store=artifact_store,
        )
        for s in spec.stress_scenarios
    )

    def _spread(s: BacktestSpec, v: float) -> tuple[BacktestSpec | None, str | None]:
        return _apply_cost_stress(s, spread_multiplier=v)

    def _slippage(s: BacktestSpec, v: float) -> tuple[BacktestSpec | None, str | None]:
        return _apply_cost_stress(s, slippage_multiplier=v)

    def _commission(s: BacktestSpec, v: float) -> tuple[BacktestSpec | None, str | None]:
        return _apply_cost_stress(s, commission_multiplier=v)

    def _financing(s: BacktestSpec, v: float) -> tuple[BacktestSpec | None, str | None]:
        return _apply_cost_stress(s, financing_multiplier=v)

    def _latency(s: BacktestSpec, v: float) -> tuple[BacktestSpec | None, str | None]:
        return _apply_cost_stress(s, additional_latency_bars=round(v))

    breakeven_axes: tuple[tuple[str, tuple[float, ...], Callable[[BacktestSpec, float], tuple[BacktestSpec | None, str | None]]], ...] = (
        ("spread_multiplier", _MULTIPLIER_SEARCH_GRID, _spread),
        ("slippage_multiplier", _MULTIPLIER_SEARCH_GRID, _slippage),
        ("commission_multiplier", _MULTIPLIER_SEARCH_GRID, _commission),
        ("financing_multiplier", _MULTIPLIER_SEARCH_GRID, _financing),
        ("additional_latency_bars", _LATENCY_SEARCH_GRID, _latency),
    )
    breakeven_results = tuple(
        _search_breakeven(
            axis_name=axis_name, grid=grid, modifier=modifier, source_backtest_spec=source.backtest_spec, resolved_inputs=resolved_inputs,
            robustness_source_backtest_id=spec.source_backtest_id, artifact_store=artifact_store,
        )
        for axis_name, grid, modifier in breakeven_axes
    )

    return StressReport(
        schema_version=STRESS_REPORT_SCHEMA_VERSION, source_backtest_id=spec.source_backtest_id, baseline_total_net_return=baseline.total_net_return,
        baseline_total_gross_return=baseline.total_gross_return, baseline_closed_trade_count=baseline.closed_trade_count,
        baseline_maximum_drawdown=baseline.maximum_drawdown, scenario_results=scenario_results, breakeven_results=breakeven_results,
        generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "NAMED_XAUUSD_STRESS_PROFILES",
    "STRESS_REPORT_SCHEMA_VERSION",
    "BreakEvenResult",
    "BreakEvenSearchPoint",
    "StressReport",
    "StressScenarioResult",
    "compute_stress_report",
]
