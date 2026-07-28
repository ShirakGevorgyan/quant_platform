"""True chronological bar-level mark-to-market strategy-return timeline
(Milestone 5.1, Section 1; EXACT portfolio accounting per Milestone 5.2,
Sections 1-2) -- the corrective replacement for treating trade-EXIT-event
returns as if they were BAR returns, now using a mathematically exact
per-bar accounting model for every supported direction/return-policy/
compounding combination (no "approximately reconciles" case remains).

ROOT CAUSE THIS MODULE FIXES (Milestone 5.1)
--------------------------------------------------------------------------
The original `returns.build_equity_curve` produced exactly one point per
DISTINCT TRADE-EXIT TIMESTAMP, then `metrics.compute_financial_metrics`
annualized that series using the market's BAR frequency. This module
instead walks EVERY outer-test market bar -- including bars with no trade
activity at all -- and marks every open position to market at each bar's
close, so the resulting series is genuinely bar-sampled.

MILESTONE 5.2: THE EXACT ACCOUNTING MODEL
--------------------------------------------------------------------------
Milestone 5.1 shipped a per-bar return formula that compounded correctly
ONLY for LONG+SIMPLE (and was INCORRECTLY documented as also exact for
LOG -- a documentation error caught and corrected here, not merely
SHORT+SIMPLE's already-known asymmetry). The root cause: multiplicative
COMPOUNDING of percentage returns only telescopes exactly when the
per-bar return is a true RATIO of consecutive values of the SAME
quantity; the old formula used the bar's own OHLC price ratio directly,
which is that quantity only for a LONG position (asset value scales
linearly with price). A SHORT position's economics are a LIABILITY, not
an asset value, so no bar-return formula derived from raw price ratios
can ever compound exactly for SHORT -- this is a mathematical fact, not
an implementation gap.

THE FIX: instead of inventing a per-bar "return", this module computes a
closed-form LOCAL VALUE MULTIPLIER `M(P)` for each trade at each bar --
the trade's own mark-to-market value, as a multiple of its value at
entry (`M(entry_price) == 1.0` always) -- and DERIVES the per-bar return
from the RATIO (compounded) or DELTA (non-compounded) of consecutive `M`
values. Because `M` is defined so that `M(exit_price)` exactly equals
`1 + trade.gross_return` (SIMPLE) or `exp(trade.gross_return)` (LOG), and
because ANY ratio sequence telescopes exactly under multiplication and
ANY value/log-value sequence telescopes exactly under addition (both
pure algebraic identities, not approximations), this construction is
exact BY CONSTRUCTION for every direction and every policy:

    M(direction=LONG,  price) = price / entry_price
    M(direction=SHORT, price, policy=SIMPLE) = 2.0 - price / entry_price
    M(direction=SHORT, price, policy=LOG)    = entry_price / price

(SIMPLE+LONG's `M` is the classic "asset value ratio"; SIMPLE+SHORT's `M`
is `1 + (1 - price/entry)`, i.e. "cash-plus-shrinking-liability" -- see
`_local_value_multiplier`'s own docstring for the cash/liability
derivation. LOG's two formulas are the same construction in log-space:
`log(M)` is exactly `sign * log(price/entry)`, the ALREADY-existing,
UNCHANGED per-bar LOG formula -- Milestone 5.2 did not need to change
LOG's per-bar VALUE, only how it accumulates, see below.)

EXPOSURE-CAP BLENDING (why a partially-sized position also reconciles
exactly, not merely at `exposure_cap=1.0`)
--------------------------------------------------------------------------
A trade sized at `exposure_cap` fraction of capital is modeled as
`exposure_cap` invested in the position (worth `M(P)` per unit) and
`(1 - exposure_cap)` held in un-invested cash (worth `1.0`, unchanged).
The blended portfolio-level value of this ONE trade's sleeve is:

    V(M, exposure_cap, policy=SIMPLE) = 1 + exposure_cap * (M - 1)
                                       = (1 - exposure_cap) + exposure_cap * M   [implemented form -- see
                                         `_exposure_blended_value`'s own docstring for why this algebraically
                                         identical rearrangement is the one actually computed]
    V(M, exposure_cap, policy=LOG)    = M ** exposure_cap

(SIMPLE blends linearly, matching real dollar-cash blending exactly; LOG
blends via exponentiation-by-a-fraction, the log-space analogue of linear
blending: `log(V) = exposure_cap * log(M)`.) `V` is, again, a pure ratio
sequence in `M`/price, so it ALSO telescopes exactly. `V(exposure_cap=1)`
reduces to `M` itself. This is what makes Section 4's "exposure cap below
1" reconciliation tests pass exactly, not merely at full sizing.

COSTS: EXACT NET ACCOUNTING (Milestone 5.2, Section 2)
--------------------------------------------------------------------------
Net value is `M_net(P, t) = M_gross(P) - cost_to_date(t)`, where
`cost_to_date(t)` is the trade's OWN cumulative dollar-fraction cost
incurred by bar `t` (0 before entry; entry-side costs from the entry bar
onward; additionally exit-side costs + financing from the exit bar
onward) -- a pure LINEAR subtraction in the SAME value space `M_gross`
already lives in, at the EXACT bar each cost is incurred, matching
`returns.compute_trade_return_result`'s Milestone-5.2-corrected trade-
level formula (`M_net_exit = M_gross_exit - total_cost`) exactly. Because
subtracting a scalar from a value sequence does not break ratio/delta
telescoping (it is still just "the same construction applied to a
different starting sequence"), `M_net`/`V_net` reconcile exactly to
`trade.net_return` by the identical argument as gross -- proven
numerically, per combination, in `tests/unit/backtesting/test_exact_
portfolio_accounting.py`, not merely asserted.

ACCUMULATION -- WHAT `compounding_policy` ACTUALLY GOVERNS
--------------------------------------------------------------------------
`compounding_policy` selects WHICH telescoping identity is used to derive
a bar's reported return from consecutive `V` values -- it does not, and
in this milestone still does not, make position SIZING equity-dependent
(sizing remains `exposure_cap * spec.initial_notional`, fixed, regardless
of compounding_policy -- true equity-scaled reinvestment sizing is a
materially larger feature, out of this milestone's scope, noted in the
delivery report's remaining-limitations section):

    COMPOUNDED:     bar_return = V(curr)/V(prev) - 1;  cum *= (1+bar_return)
    NON_COMPOUNDED (SIMPLE): bar_return = V(curr)-V(prev);      cum += bar_return
    NON_COMPOUNDED (LOG):    bar_return = log(V(curr))-log(V(prev)); cum += bar_return

WHICH TRADES CONTRIBUTE (unchanged from Milestone 5.1)
--------------------------------------------------------------------------
Only `TradeSet.closed_trades` (status `CLOSED` or `INCOMPLETE_FORCE_
CLOSED`) -- the SAME set trade-level statistics already use.

NO DOUBLE COUNTING (unchanged)
--------------------------------------------------------------------------
Every trade contributes its return to EXACTLY ONE of {realized_return,
unrealized_return} at EACH bar it touches (realized on its own exit bar,
unrealized otherwise), and its cost components are split so that
`sum(transaction_costs across bars) == exposure_cap * trade.cost_
breakdown.total_cost` exactly (entry-side on the entry bar, exit-side +
financing on the exit bar)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.backtesting.models import PositionDirection, ReturnCalculationPolicyKind, TradeStatus
from quant_platform.backtesting.returns import (
    EQUITY_POINT_SCHEMA_VERSION,
    EquityCurve,
    EquityPoint,
)
from quant_platform.backtesting.trades import TradeRecord
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.ml.persistence import (
    as_json_list,
    format_utc_timestamp,
    parse_utc_timestamp,
    require_schema_version,
)

BAR_RETURN_TIMELINE_SCHEMA_VERSION = 1


class BarReturnBasis(Enum):
    """Section 1: which convention bar-level returns use. Exactly one
    value is implemented -- see module docstring for its precise
    definition (previous valuation timestamp -> current valuation
    timestamp, with fill-price substitution at entry/exit bars)."""

    PREVIOUS_VALUATION_TO_CURRENT_VALUATION = "previous_valuation_to_current_valuation"


def _local_value_multiplier(direction: PositionDirection, entry_price: float, price: float, policy: ReturnCalculationPolicyKind) -> float:
    """`M(P)`: Milestone 5.2's "local trade value multiplier" -- the
    trade's own mark-to-market value at price `P`, as a multiple of its
    value at entry (`M(entry_price) == 1.0` always, for every direction
    and policy). Derivation:

    LONG: holding `quantity = notional/entry_price` shares, asset value
    at P is `quantity*P`; as a multiple of the entry value
    (`quantity*entry_price`), that is `P/entry_price` directly.

    SHORT, SIMPLE: at entry, short-sale proceeds of `notional` are
    banked as cash; the mark-to-market LIABILITY to buy back is
    `quantity*P = notional*P/entry_price`. Value = cash - liability =
    `notional - notional*P/entry_price = notional*(2 - P/entry_price)`;
    as a multiple of the entry value (`notional`), that is
    `2 - P/entry_price`.

    SHORT, LOG: `entry_price/P` -- chosen so that `log(M(P))` is exactly
    `-log(P/entry_price)`, i.e. the UNCHANGED, already-correct per-bar
    LOG return formula this platform has always used for SHORT
    positions; `M` is simply that formula's own multiplicative
    (`exp`-space) representation."""
    if direction is PositionDirection.LONG:
        return price / entry_price
    if direction is PositionDirection.SHORT:
        if policy is ReturnCalculationPolicyKind.SIMPLE:
            return 2.0 - price / entry_price
        if policy is ReturnCalculationPolicyKind.LOG:
            return entry_price / price
        raise FinancialMetricError(f"Unsupported ReturnCalculationPolicyKind: {policy!r}")  # pragma: no cover - exhaustive enum
    raise FinancialMetricError(f"_local_value_multiplier: PositionDirection.FLAT has no local value -- got {direction!r}")


def _exposure_blended_value(m: float, exposure_cap: float, policy: ReturnCalculationPolicyKind) -> float:
    """`V(M, exposure_cap)`: blends a trade's own local value `M` with its
    `(1 - exposure_cap)` un-invested cash cushion -- see module docstring
    for the linear (SIMPLE) vs power (LOG) blending derivation. Raises if
    the blended value would be non-positive (an extreme loss/cost
    scenario this milestone does not attempt to model past total loss of
    the allocated sleeve).

    FINAL MILESTONE 5 AUDIT: the SIMPLE branch computes `(1 - exposure_
    cap) + exposure_cap * m`, algebraically identical to `1 + exposure_
    cap * (m - 1)` but NOT numerically identical -- the latter suffers
    catastrophic cancellation when `m` is many orders of magnitude
    smaller than 1 (a near-total-loss LONG position, `m = price /
    entry_price`, under an extreme adverse price move): `m - 1` rounds to
    exactly `-1.0` once `m` is smaller than float64's precision relative
    to 1 (roughly `m < 1.1e-16`), silently discarding `m` entirely and
    reporting `V = 1 - exposure_cap` -- exactly `0.0` at `exposure_cap =
    1.0` -- even though the true blended value is a tiny but genuinely
    POSITIVE number, not an actual insolvency. The reformulation avoids
    the `m - 1` subtraction (verified, via 200k randomized samples across
    the normal operating range, to agree with the old formula to within
    ~1.5e-14 -- pure floating-point noise, far inside this platform's
    existing 1e-9 reconciliation tolerances -- while remaining exact at
    the genuine insolvency boundary, e.g. SHORT+SIMPLE at
    `price == 2 * entry_price`, `exposure_cap == 1.0`, still yields
    `V == 0.0` precisely)."""
    if policy is ReturnCalculationPolicyKind.SIMPLE:
        value = (1.0 - exposure_cap) + exposure_cap * m
    elif policy is ReturnCalculationPolicyKind.LOG:
        if m <= 0.0:
            raise FinancialMetricError(f"_exposure_blended_value: local value multiplier must be positive under LOG returns, got {m!r}")
        value = m**exposure_cap
    else:
        raise FinancialMetricError(f"Unsupported ReturnCalculationPolicyKind: {policy!r}")  # pragma: no cover - exhaustive enum
    if not math.isfinite(value) or value <= 0.0:
        raise FinancialMetricError(
            f"_exposure_blended_value: blended value must remain finite and positive, got {value!r} "
            f"(m={m!r}, exposure_cap={exposure_cap!r}) -- an extreme loss/cost scenario pushed this "
            "position's contribution non-positive"
        )
    return value


@dataclass(frozen=True, slots=True)
class _TradeBarContribution:
    gross: float
    net: float
    cost: float
    realized: bool


def _trade_bar_contribution(
    *, trade: TradeRecord, bar_position: int, prev_close: float | None, curr_close: float,
    return_calculation_policy: ReturnCalculationPolicyKind, exposure_cap: float, compounded: bool,
) -> _TradeBarContribution:
    """One trade's exact gross/net contribution to ONE bar, unifying what
    Milestone 5.1 handled as four separate cases (same-bar, entry-only,
    mid-position, exit-only) into a single formula: the valuation
    endpoints for this bar are `(prev_price, curr_price)`, substituting
    the trade's own fill price at the entry and/or exit instant -- Section
    1's `PREVIOUS_VALUATION_TO_CURRENT_VALUATION` basis, unchanged."""
    entry_costs = trade.cost_breakdown.entry_spread_cost + trade.cost_breakdown.entry_commission + trade.cost_breakdown.entry_slippage
    exit_costs = trade.cost_breakdown.exit_spread_cost + trade.cost_breakdown.exit_commission + trade.cost_breakdown.exit_slippage + trade.cost_breakdown.financing_cost

    is_entry_bar = bar_position == trade.entry_bar_position
    is_exit_bar = bar_position == trade.exit_bar_position

    if is_entry_bar:
        prev_price = trade.entry_observed_price
    else:
        if prev_close is None:
            raise FinancialMetricError(  # pragma: no cover - structurally unreachable (see module docstring: trades never cross fold boundaries)
                f"_trade_bar_contribution: bar {bar_position} needs a previous close but none was supplied "
                f"(trade {trade.trade_id!r} entered at bar {trade.entry_bar_position})"
            )
        prev_price = prev_close
    prev_cost_to_date = 0.0 if is_entry_bar else entry_costs

    curr_price = trade.exit_observed_price if is_exit_bar else curr_close
    curr_cost_to_date = (entry_costs + exit_costs) if is_exit_bar else entry_costs

    m_gross_prev = _local_value_multiplier(trade.direction, trade.entry_observed_price, prev_price, return_calculation_policy)
    m_gross_curr = _local_value_multiplier(trade.direction, trade.entry_observed_price, curr_price, return_calculation_policy)
    m_net_prev = m_gross_prev - prev_cost_to_date
    m_net_curr = m_gross_curr - curr_cost_to_date

    v_gross_prev = _exposure_blended_value(m_gross_prev, exposure_cap, return_calculation_policy)
    v_gross_curr = _exposure_blended_value(m_gross_curr, exposure_cap, return_calculation_policy)
    v_net_prev = _exposure_blended_value(m_net_prev, exposure_cap, return_calculation_policy)
    v_net_curr = _exposure_blended_value(m_net_curr, exposure_cap, return_calculation_policy)

    if compounded:
        gross = v_gross_curr / v_gross_prev - 1.0
        net = v_net_curr / v_net_prev - 1.0
    elif return_calculation_policy is ReturnCalculationPolicyKind.SIMPLE:
        gross = v_gross_curr - v_gross_prev
        net = v_net_curr - v_net_prev
    else:
        gross = math.log(v_gross_curr) - math.log(v_gross_prev)
        net = math.log(v_net_curr) - math.log(v_net_prev)

    cost_this_bar = exposure_cap * (curr_cost_to_date - prev_cost_to_date)
    return _TradeBarContribution(gross=gross, net=net, cost=cost_this_bar, realized=is_exit_bar)


@dataclass(frozen=True, slots=True)
class BarReturnPoint:
    """One outer-test market bar's complete mark-to-market strategy
    state (Section 1). Every bar in `[fold_start_position,
    fold_end_position]` gets exactly one point -- bars with no trade
    activity still exist, with `entries_count=exits_count=0` and
    `unrealized_return` reflecting mark-to-market on whatever remains
    open (`0.0` if flat)."""

    schema_version: int
    bar_position: int
    timestamp: str
    gross_return: float
    net_return: float
    realized_return: float
    unrealized_return: float
    active_long_exposure: float
    active_short_exposure: float
    total_absolute_exposure: float
    net_exposure: float
    open_trade_count: int
    entries_count: int
    exits_count: int
    transaction_costs: float
    cumulative_gross_equity: float
    cumulative_net_equity: float
    peak_equity: float
    drawdown: float

    def __post_init__(self) -> None:
        parse_utc_timestamp(self.timestamp)
        if self.bar_position < 0:
            raise FinancialMetricError(f"BarReturnPoint.bar_position must be >= 0, got {self.bar_position}")
        for name, value in (
            ("gross_return", self.gross_return), ("net_return", self.net_return), ("realized_return", self.realized_return),
            ("unrealized_return", self.unrealized_return), ("active_long_exposure", self.active_long_exposure),
            ("active_short_exposure", self.active_short_exposure), ("total_absolute_exposure", self.total_absolute_exposure),
            ("net_exposure", self.net_exposure), ("transaction_costs", self.transaction_costs),
            ("cumulative_gross_equity", self.cumulative_gross_equity), ("cumulative_net_equity", self.cumulative_net_equity),
            ("peak_equity", self.peak_equity), ("drawdown", self.drawdown),
        ):
            if not math.isfinite(value):
                raise FinancialMetricError(f"BarReturnPoint.{name} must be finite, got {value!r}")
        if abs(self.gross_return - (self.realized_return + self.unrealized_return)) > 1e-9:
            raise FinancialMetricError(
                f"BarReturnPoint.gross_return ({self.gross_return}) must equal realized_return + unrealized_return "
                f"({self.realized_return + self.unrealized_return})"
            )
        if abs(self.active_long_exposure + self.active_short_exposure - self.total_absolute_exposure) > 1e-9:
            raise FinancialMetricError("BarReturnPoint.total_absolute_exposure must equal active_long_exposure + active_short_exposure")
        if abs((self.active_long_exposure - self.active_short_exposure) - self.net_exposure) > 1e-9:
            raise FinancialMetricError("BarReturnPoint.net_exposure must equal active_long_exposure - active_short_exposure")
        for name, value in (("active_long_exposure", self.active_long_exposure), ("active_short_exposure", self.active_short_exposure), ("transaction_costs", self.transaction_costs)):
            if value < 0.0:
                raise FinancialMetricError(f"BarReturnPoint.{name} must be >= 0, got {value!r}")
        if self.cumulative_gross_equity <= 0.0 or self.cumulative_net_equity <= 0.0:
            raise FinancialMetricError("BarReturnPoint: cumulative equity must remain positive")
        if self.open_trade_count < 0 or self.entries_count < 0 or self.exits_count < 0:
            raise FinancialMetricError("BarReturnPoint: open_trade_count/entries_count/exits_count must be >= 0")
        if self.drawdown < 0.0:
            raise FinancialMetricError(f"BarReturnPoint.drawdown must be >= 0, got {self.drawdown!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "bar_position": self.bar_position, "timestamp": self.timestamp,
            "gross_return": self.gross_return, "net_return": self.net_return, "realized_return": self.realized_return,
            "unrealized_return": self.unrealized_return, "active_long_exposure": self.active_long_exposure,
            "active_short_exposure": self.active_short_exposure, "total_absolute_exposure": self.total_absolute_exposure,
            "net_exposure": self.net_exposure, "open_trade_count": self.open_trade_count, "entries_count": self.entries_count,
            "exits_count": self.exits_count, "transaction_costs": self.transaction_costs,
            "cumulative_gross_equity": self.cumulative_gross_equity, "cumulative_net_equity": self.cumulative_net_equity,
            "peak_equity": self.peak_equity, "drawdown": self.drawdown,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BarReturnPoint:
        return cls(
            schema_version=int(str(raw["schema_version"])), bar_position=int(str(raw["bar_position"])), timestamp=str(raw["timestamp"]),
            gross_return=float(str(raw["gross_return"])), net_return=float(str(raw["net_return"])), realized_return=float(str(raw["realized_return"])),
            unrealized_return=float(str(raw["unrealized_return"])), active_long_exposure=float(str(raw["active_long_exposure"])),
            active_short_exposure=float(str(raw["active_short_exposure"])), total_absolute_exposure=float(str(raw["total_absolute_exposure"])),
            net_exposure=float(str(raw["net_exposure"])), open_trade_count=int(str(raw["open_trade_count"])),
            entries_count=int(str(raw["entries_count"])), exits_count=int(str(raw["exits_count"])), transaction_costs=float(str(raw["transaction_costs"])),
            cumulative_gross_equity=float(str(raw["cumulative_gross_equity"])), cumulative_net_equity=float(str(raw["cumulative_net_equity"])),
            peak_equity=float(str(raw["peak_equity"])), drawdown=float(str(raw["drawdown"])),
        )


@dataclass(frozen=True, slots=True)
class BarReturnTimeline:
    schema_version: int
    outer_fold_index: int
    return_basis: BarReturnBasis
    compounded: bool
    fold_start_position: int
    fold_end_position: int
    points: tuple[BarReturnPoint, ...]

    def __post_init__(self) -> None:
        expected_count = self.fold_end_position - self.fold_start_position + 1
        if len(self.points) != expected_count:
            raise FinancialMetricError(
                f"BarReturnTimeline must contain exactly one point per bar in [{self.fold_start_position}, "
                f"{self.fold_end_position}] ({expected_count} bars) -- got {len(self.points)}. The financial "
                "timeline must never be compressed to trade-exit events."
            )
        for i, p in enumerate(self.points):
            if p.bar_position != self.fold_start_position + i:
                raise FinancialMetricError(f"BarReturnTimeline.points[{i}].bar_position={p.bar_position} is out of chronological order")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index, "return_basis": self.return_basis.value,
            "compounded": self.compounded, "fold_start_position": self.fold_start_position, "fold_end_position": self.fold_end_position,
            "points": [p.to_json_dict() for p in self.points],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BarReturnTimeline:
        require_schema_version(raw, supported=BAR_RETURN_TIMELINE_SCHEMA_VERSION, context="BarReturnTimeline")
        return cls(
            schema_version=BAR_RETURN_TIMELINE_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            return_basis=BarReturnBasis(raw["return_basis"]), compounded=bool(raw["compounded"]),
            fold_start_position=int(str(raw["fold_start_position"])), fold_end_position=int(str(raw["fold_end_position"])),
            points=tuple(BarReturnPoint.from_json_dict(p) for p in as_json_list(raw["points"], field_name="points")),
        )


def build_bar_return_timeline(
    *, trades: tuple[TradeRecord, ...], bars: pd.DataFrame, fold_start_position: int, fold_end_position: int,
    outer_fold_index: int, return_calculation_policy: ReturnCalculationPolicyKind, exposure_cap: float, compounded: bool,
) -> BarReturnTimeline:
    """Section 1 / Milestone 5.2 Sections 1-2: walk every bar in
    `[fold_start_position, fold_end_position]`, marking every closed
    trade (see module docstring for which trades qualify) to market at
    each bar it touches using the EXACT local-value-multiplier accounting
    model. No trade-exit compression; bars without any trade activity
    still get a point."""
    closed = [t for t in trades if t.status in (TradeStatus.CLOSED, TradeStatus.INCOMPLETE_FORCE_CLOSED)]

    points: list[BarReturnPoint] = []
    cum_gross = 1.0
    cum_net = 1.0
    peak = 1.0

    for bar_position in range(fold_start_position, fold_end_position + 1):
        bar = bars.iloc[bar_position]
        curr_close = float(bar["close"])
        prev_close = float(bars.iloc[bar_position - 1]["close"]) if bar_position > fold_start_position else None

        touching = [t for t in closed if t.entry_bar_position <= bar_position <= t.exit_bar_position]
        open_after = [t for t in closed if t.entry_bar_position <= bar_position < t.exit_bar_position]

        realized_return = 0.0
        unrealized_return = 0.0
        net_bar_return = 0.0
        transaction_costs = 0.0
        entries_count = 0
        exits_count = 0

        for t in touching:
            contribution = _trade_bar_contribution(
                trade=t, bar_position=bar_position, prev_close=prev_close, curr_close=curr_close,
                return_calculation_policy=return_calculation_policy, exposure_cap=exposure_cap, compounded=compounded,
            )
            if contribution.realized:
                realized_return += contribution.gross
            else:
                unrealized_return += contribution.gross
            net_bar_return += contribution.net
            transaction_costs += contribution.cost
            if bar_position == t.entry_bar_position:
                entries_count += 1
            if bar_position == t.exit_bar_position:
                exits_count += 1

        gross_bar_return = realized_return + unrealized_return

        if compounded:
            cum_gross *= 1.0 + gross_bar_return
            cum_net *= 1.0 + net_bar_return
        else:
            cum_gross = cum_gross + gross_bar_return
            cum_net = cum_net + net_bar_return

        peak = max(peak, cum_net)
        drawdown = (peak - cum_net) / peak if cum_net < peak else 0.0

        long_open = [t for t in open_after if t.direction is PositionDirection.LONG]
        short_open = [t for t in open_after if t.direction is PositionDirection.SHORT]
        active_long = len(long_open) * exposure_cap
        active_short = len(short_open) * exposure_cap

        points.append(BarReturnPoint(
            schema_version=1, bar_position=bar_position, timestamp=format_utc_timestamp(bar["open_time"]),
            gross_return=gross_bar_return, net_return=net_bar_return, realized_return=realized_return, unrealized_return=unrealized_return,
            active_long_exposure=active_long, active_short_exposure=active_short, total_absolute_exposure=active_long + active_short,
            net_exposure=active_long - active_short, open_trade_count=len(open_after), entries_count=entries_count,
            exits_count=exits_count, transaction_costs=transaction_costs, cumulative_gross_equity=cum_gross,
            cumulative_net_equity=cum_net, peak_equity=peak, drawdown=drawdown,
        ))

    return BarReturnTimeline(
        schema_version=BAR_RETURN_TIMELINE_SCHEMA_VERSION, outer_fold_index=outer_fold_index,
        return_basis=BarReturnBasis.PREVIOUS_VALUATION_TO_CURRENT_VALUATION, compounded=compounded,
        fold_start_position=fold_start_position, fold_end_position=fold_end_position, points=tuple(points),
    )


def bar_return_timeline_to_equity_curve(timeline: BarReturnTimeline) -> EquityCurve:
    """Projects the rich bar-level timeline down to `returns.EquityCurve`'s
    existing (bar-granular, now correctly so) shape, for reuse by
    `drawdown.compute_drawdown_report` unchanged."""
    trade_count_to_date = 0
    points: list[EquityPoint] = []
    for p in timeline.points:
        trade_count_to_date += p.exits_count
        points.append(EquityPoint(
            timestamp=p.timestamp, period_gross_return=p.gross_return, period_net_return=p.net_return,
            cumulative_gross_equity=p.cumulative_gross_equity, cumulative_net_equity=p.cumulative_net_equity,
            exposure=p.total_absolute_exposure, trade_count_to_date=trade_count_to_date,
        ))
    return EquityCurve(schema_version=EQUITY_POINT_SCHEMA_VERSION, outer_fold_index=timeline.outer_fold_index, compounded=timeline.compounded, points=tuple(points))


__all__ = [
    "BAR_RETURN_TIMELINE_SCHEMA_VERSION",
    "BarReturnBasis",
    "BarReturnPoint",
    "BarReturnTimeline",
    "bar_return_timeline_to_equity_curve",
    "build_bar_return_timeline",
]
