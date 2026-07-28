"""Return calculation and equity-curve construction (Milestone 5,
Sections 14/15; NET formula corrected in Milestone 5.2, Section 2).

THE SHARED-DENOMINATOR DESIGN (why `net_return = gross_return -
total_cost` holds EXACTLY for SIMPLE returns)
--------------------------------------------------------------------------
`gross_return` is computed from OBSERVED (unadjusted) entry/exit prices;
every itemized cost component (`backtesting.costs.CostBreakdown`) is
computed as a return-FRACTION relative to the SAME `entry_observed_price`
denominator, never the trade's own effective price. Under `SIMPLE`
returns this makes `net_return := gross_return - total_cost` hold EXACTLY
by construction: `gross_return` is already a linear (dollar-proportional)
quantity, so a linear cost subtraction composes with it exactly -- the
`M_net_exit = M_gross_exit - total_cost` identity that `backtesting.
timeline` relies on for its own bar-level exactness proof (see that
module's docstring) is the SAME linear-subtraction identity applied at
trade level.

MILESTONE 5.2 CORRECTION -- `LOG` returns and `net_return`
--------------------------------------------------------------------------
Under `LOG` returns, `gross_return = log(M_gross_exit)` (a log-space
quantity), so a LINEAR subtraction of a dollar-fraction cost
(`gross_return - total_cost`, the ORIGINAL Milestone 5 formula) does NOT
correspond to any coherent economic operation -- it mixes a log quantity
with a linear one. The corrected formula computes the cost-adjusted
"local value multiplier" `M_net_exit = M_gross_exit - total_cost` (a
LINEAR/dollar-space subtraction, matching how a real cost actually
reduces cash) and THEN reports `net_return = log(M_net_exit)` -- i.e. the
cost is applied in the SAME dollar/value space `M` already lives in, and
`return_calculation_policy` only governs how the FINAL value is reported
(linear delta for SIMPLE, log for LOG), never how costs are combined with
price moves. `SIMPLE` policy's formula is unchanged (it already worked
this way, since `M_gross_exit = 1 + gross_return` there). This is what
makes bar-level, fold-level, and stitched-level net equity reconcile
EXACTLY to `trade.net_return` for every direction/policy/compounding
combination -- see `backtesting.timeline`'s module docstring for the
complete proof and `tests/unit/backtesting/test_exact_portfolio_
accounting.py` for the hand-verified reconciliation tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.costs import CostBreakdown, financing_fraction, per_side_commission_fraction
from quant_platform.backtesting.models import PositionDirection, ReturnCalculationPolicyKind
from quant_platform.backtesting.specs import CommissionSpec, CostSensitivityScenario, FinancingSpec
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.ml.persistence import as_json_list, parse_utc_timestamp, require_schema_version


def _direction_sign(direction: PositionDirection) -> int:
    if direction is PositionDirection.LONG:
        return 1
    if direction is PositionDirection.SHORT:
        return -1
    raise FinancialMetricError(f"_direction_sign: PositionDirection.FLAT has no sign -- got {direction!r}")


def compute_gross_return(
    *, direction: PositionDirection, entry_observed_price: float, exit_observed_price: float,
    return_calculation_policy: ReturnCalculationPolicyKind,
) -> float:
    if entry_observed_price <= 0.0 or exit_observed_price <= 0.0:
        raise FinancialMetricError("compute_gross_return: prices must be positive")
    sign = _direction_sign(direction)
    if return_calculation_policy is ReturnCalculationPolicyKind.SIMPLE:
        return sign * (exit_observed_price - entry_observed_price) / entry_observed_price
    if return_calculation_policy is ReturnCalculationPolicyKind.LOG:
        return sign * math.log(exit_observed_price / entry_observed_price)
    raise FinancialMetricError(f"Unsupported ReturnCalculationPolicyKind: {return_calculation_policy!r}")  # pragma: no cover - exhaustive enum


@dataclass(frozen=True, slots=True)
class TradeReturnResult:
    gross_return: float
    net_return: float
    cost_breakdown: CostBreakdown


def compute_trade_return_result(
    *, direction: PositionDirection, entry_observed_price: float, exit_observed_price: float,
    entry_spread_adjustment: float, exit_spread_adjustment: float, entry_slippage_adjustment: float,
    exit_slippage_adjustment: float, commission_spec: CommissionSpec, financing_spec: FinancingSpec,
    holding_days: float, notional: float, return_calculation_policy: ReturnCalculationPolicyKind,
    cost_scenario: CostSensitivityScenario | None = None,
) -> TradeReturnResult:
    """The one authoritative per-trade return computation. `cost_scenario`
    (Section 20) scales every cost COMPONENT (never the gross return
    itself) by its declared multipliers -- sensitivity analysis, never a
    second, different cost MODEL."""
    scenario = cost_scenario or CostSensitivityScenario(name="base_cost")
    gross_return = compute_gross_return(
        direction=direction, entry_observed_price=entry_observed_price, exit_observed_price=exit_observed_price,
        return_calculation_policy=return_calculation_policy,
    )
    entry_spread_cost = abs(entry_spread_adjustment) / entry_observed_price * scenario.spread_multiplier
    exit_spread_cost = abs(exit_spread_adjustment) / entry_observed_price * scenario.spread_multiplier
    entry_slippage_cost = abs(entry_slippage_adjustment) / entry_observed_price * scenario.slippage_multiplier
    exit_slippage_cost = abs(exit_slippage_adjustment) / entry_observed_price * scenario.slippage_multiplier
    entry_commission = per_side_commission_fraction(commission_spec, notional=notional) * scenario.commission_multiplier
    exit_commission = per_side_commission_fraction(commission_spec, notional=notional) * scenario.commission_multiplier
    financing_cost = financing_fraction(financing_spec, holding_days=holding_days)

    breakdown = CostBreakdown(
        entry_spread_cost=entry_spread_cost, exit_spread_cost=exit_spread_cost, entry_commission=entry_commission,
        exit_commission=exit_commission, entry_slippage=entry_slippage_cost, exit_slippage=exit_slippage_cost,
        financing_cost=financing_cost,
    )
    # Milestone 5.2, Section 2: costs are applied in the LINEAR "local
    # value multiplier" space (M_gross = 1+gross_return for SIMPLE,
    # M_gross = exp(gross_return) for LOG) -- never by linearly
    # subtracting a dollar-fraction cost from a log-space return. For
    # SIMPLE this reduces to the original `gross_return - total_cost`
    # (M_gross is already linear); for LOG it does not.
    if return_calculation_policy is ReturnCalculationPolicyKind.SIMPLE:
        net_return = gross_return - breakdown.total_cost
    elif return_calculation_policy is ReturnCalculationPolicyKind.LOG:
        m_gross_exit = math.exp(gross_return)
        m_net_exit = m_gross_exit - breakdown.total_cost
        if m_net_exit <= 0.0:
            raise FinancialMetricError(
                f"compute_trade_return_result: total_cost ({breakdown.total_cost!r}) exceeds this position's "
                f"gross value under LOG returns (M_gross_exit={m_gross_exit!r}) -- the resulting net value would "
                "be non-positive (a total-loss-and-beyond outcome), which is out of this milestone's scope",
                context={"total_cost": breakdown.total_cost, "m_gross_exit": m_gross_exit},
            )
        net_return = math.log(m_net_exit)
    else:
        raise FinancialMetricError(f"Unsupported ReturnCalculationPolicyKind: {return_calculation_policy!r}")  # pragma: no cover - exhaustive enum
    return TradeReturnResult(gross_return=gross_return, net_return=net_return, cost_breakdown=breakdown)


# --------------------------------------------------------------------------
# Equity curve (Section 15)
# --------------------------------------------------------------------------
EQUITY_POINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: str
    period_gross_return: float
    period_net_return: float
    cumulative_gross_equity: float
    cumulative_net_equity: float
    exposure: float
    trade_count_to_date: int

    def __post_init__(self) -> None:
        parse_utc_timestamp(self.timestamp)
        for name, value in (
            ("period_gross_return", self.period_gross_return), ("period_net_return", self.period_net_return),
            ("cumulative_gross_equity", self.cumulative_gross_equity), ("cumulative_net_equity", self.cumulative_net_equity),
            ("exposure", self.exposure),
        ):
            if not math.isfinite(value):
                raise FinancialMetricError(f"EquityPoint.{name} must be finite, got {value!r}")
        if self.cumulative_gross_equity <= 0.0 or self.cumulative_net_equity <= 0.0:
            raise FinancialMetricError("EquityPoint: cumulative equity must remain positive (a total-loss/negative-equity outcome is out of this milestone's scope)")
        if self.exposure < 0.0:
            raise FinancialMetricError(f"EquityPoint.exposure must be >= 0, got {self.exposure!r}")
        if self.trade_count_to_date < 0:
            raise FinancialMetricError(f"EquityPoint.trade_count_to_date must be >= 0, got {self.trade_count_to_date}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp, "period_gross_return": self.period_gross_return, "period_net_return": self.period_net_return,
            "cumulative_gross_equity": self.cumulative_gross_equity, "cumulative_net_equity": self.cumulative_net_equity,
            "exposure": self.exposure, "trade_count_to_date": self.trade_count_to_date,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EquityPoint:
        return cls(
            timestamp=str(raw["timestamp"]), period_gross_return=float(str(raw["period_gross_return"])),
            period_net_return=float(str(raw["period_net_return"])), cumulative_gross_equity=float(str(raw["cumulative_gross_equity"])),
            cumulative_net_equity=float(str(raw["cumulative_net_equity"])), exposure=float(str(raw["exposure"])),
            trade_count_to_date=int(str(raw["trade_count_to_date"])),
        )


@dataclass(frozen=True, slots=True)
class EquityCurve:
    """Section 15: one chronological equity curve for one outer fold.
    `compounded` determines how `period_*_return` values accumulate into
    `cumulative_*_equity` -- see `build_equity_curve`."""

    schema_version: int
    outer_fold_index: int
    compounded: bool
    points: tuple[EquityPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise FinancialMetricError("EquityCurve.points must not be empty")
        timestamps = [p.timestamp for p in self.points]
        if timestamps != sorted(timestamps):
            raise FinancialMetricError("EquityCurve.points must be in non-decreasing timestamp order")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "compounded": self.compounded,
            "points": [p.to_json_dict() for p in self.points],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EquityCurve:
        require_schema_version(raw, supported=EQUITY_POINT_SCHEMA_VERSION, context="EquityCurve")
        return cls(
            schema_version=EQUITY_POINT_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])), compounded=bool(raw["compounded"]),
            points=tuple(EquityPoint.from_json_dict(p) for p in as_json_list(raw["points"], field_name="points")),
        )


def build_equity_curve(
    *, outer_fold_index: int, period_returns: list[tuple[str, float, float, float, int]], compounded: bool, initial_notional: float,
) -> EquityCurve:
    """`period_returns`: chronologically ordered
    `(timestamp, gross_return, net_return, exposure, trade_count_to_date)`
    tuples -- ONE point per period a trade closed (Section 15: "Define
    how overlapping trades contribute to period returns" -- this
    reference implementation defines one period per trade EXIT event;
    concurrently-open independent-overlapping trades each contribute
    their own period return at their own exit time, summed with any other
    trade closing at that exact timestamp before being applied, so
    exposure is never silently double-counted past `exposure_cap`)."""
    if initial_notional <= 0.0:
        raise FinancialMetricError(f"build_equity_curve: initial_notional must be > 0, got {initial_notional!r}")
    points: list[EquityPoint] = []
    gross_equity = 1.0
    net_equity = 1.0
    gross_sum = 0.0
    net_sum = 0.0
    for timestamp, gross_return, net_return, exposure, trade_count in period_returns:
        if compounded:
            gross_equity *= (1.0 + gross_return)
            net_equity *= (1.0 + net_return)
        else:
            gross_sum += gross_return
            net_sum += net_return
            gross_equity = 1.0 + gross_sum
            net_equity = 1.0 + net_sum
        points.append(EquityPoint(
            timestamp=timestamp, period_gross_return=gross_return, period_net_return=net_return,
            cumulative_gross_equity=gross_equity, cumulative_net_equity=net_equity, exposure=exposure,
            trade_count_to_date=trade_count,
        ))
    return EquityCurve(schema_version=EQUITY_POINT_SCHEMA_VERSION, outer_fold_index=outer_fold_index, compounded=compounded, points=tuple(points))


__all__ = [
    "EQUITY_POINT_SCHEMA_VERSION",
    "EquityCurve",
    "EquityPoint",
    "TradeReturnResult",
    "build_equity_curve",
    "compute_gross_return",
    "compute_trade_return_result",
]
