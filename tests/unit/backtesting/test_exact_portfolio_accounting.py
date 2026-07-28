"""Milestone 5.2, Sections 1-2: EXACT portfolio accounting. Proves, for
every supported (direction, return_calculation_policy, compounding_policy)
combination -- including `exposure_cap < 1.0` -- that fill-level economics,
trade-level gross/net PnL, bar-level cumulative equity, and fold-level
final equity all reconcile EXACTLY (not approximately). No "approximately
reconciles" case remains: SHORT+SIMPLE (Milestone 5.1's known gap) and
LOG-under-COMPOUNDED (a Milestone 5.1 documentation error, caught and
fixed here) are now exact, proven the same way as LONG+SIMPLE always was.

WHY THE TARGETS DIFFER BY (policy, compounding) -- READ BEFORE EXTENDING
--------------------------------------------------------------------------
`cumulative_gross_equity` is always a genuine multiplicative wealth
factor (`M_gross_exit`); `trade.gross_return` is reported in the policy's
OWN units (a linear fraction for SIMPLE, a log value for LOG). The
correct target is therefore:
    SIMPLE: cumulative_gross_equity_at_exit == 1 + trade.gross_return
    LOG:    cumulative_gross_equity_at_exit == exp(trade.gross_return)
(equivalently: `log(cumulative_gross_equity_at_exit) == trade.gross_return`
under LOG). The SAME pattern applies to `cumulative_net_equity` vs
`trade.net_return`. See `backtesting.timeline`'s module docstring for the
full derivation of why this is exact by construction, not merely
verified for these specific numbers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    ExitReasonCode,
    FinancingModelKind,
    PositionDirection,
    ReturnCalculationPolicyKind,
    SignalReasonCode,
    TradeStatus,
)
from quant_platform.backtesting.returns import compute_trade_return_result
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec
from quant_platform.backtesting.timeline import build_bar_return_timeline
from quant_platform.backtesting.trades import TradeRecord, compute_trade_id

_CALIBRATION_ID = "a" * 64
_EXPERIMENT_ID = "b" * 64


def _bars(prices: list[float], *, start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    return pd.DataFrame({
        "open_time": timestamps, "open": prices, "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices], "close": [p + 0.05 for p in prices],
    })


def _trade_from_result(
    *, entry_bar: int, exit_bar: int, entry_price: float, exit_price: float, direction: PositionDirection,
    entry_spread: float = 0.0, exit_spread: float = 0.0, entry_slippage: float = 0.0, exit_slippage: float = 0.0,
    commission_spec: CommissionSpec | None = None, financing_spec: FinancingSpec | None = None, holding_days: float = 0.0,
    return_policy: ReturnCalculationPolicyKind = ReturnCalculationPolicyKind.SIMPLE, notional: float = 1.0, sig_pos: int = 0,
) -> TradeRecord:
    """Builds a `TradeRecord` whose gross/net return and cost breakdown
    come from the SAME `compute_trade_return_result` the real pipeline
    uses -- never hand-computed -- so these tests exercise the ACTUAL
    fill-to-trade-level formula, not a parallel reimplementation of it."""
    result = compute_trade_return_result(
        direction=direction, entry_observed_price=entry_price, exit_observed_price=exit_price,
        entry_spread_adjustment=entry_spread, exit_spread_adjustment=exit_spread, entry_slippage_adjustment=entry_slippage,
        exit_slippage_adjustment=exit_slippage, commission_spec=commission_spec or CommissionSpec(kind=CommissionModelKind.ZERO),
        financing_spec=financing_spec or FinancingSpec(kind=FinancingModelKind.NONE), holding_days=holding_days,
        notional=notional, return_calculation_policy=return_policy,
    )
    base = pd.Timestamp("2024-01-01", tz="UTC")
    entry_ts = (base + pd.Timedelta(hours=entry_bar)).isoformat()
    exit_ts = (base + pd.Timedelta(hours=exit_bar)).isoformat()
    tid = compute_trade_id(source_calibration_id=_CALIBRATION_ID, outer_fold_index=0, signal_sample_position=sig_pos, direction=direction, entry_timestamp=entry_ts, exit_timestamp=exit_ts)
    return TradeRecord(
        schema_version=1, trade_id=tid, signal_sample_position=sig_pos, outer_fold_index=0, direction=direction,
        signal_timestamp=entry_ts, decision_timestamp=entry_ts, entry_timestamp=entry_ts, entry_bar_position=entry_bar,
        entry_observed_price=entry_price, entry_effective_price=entry_price + entry_spread, exit_timestamp=exit_ts, exit_bar_position=exit_bar,
        exit_observed_price=exit_price, exit_effective_price=exit_price + exit_spread, holding_bars=exit_bar - entry_bar,
        gross_return=result.gross_return, net_return=result.net_return, cost_breakdown=result.cost_breakdown,
        confidence=0.8, uncertainty=0.2, calibrated_probability=0.7, entry_reason=SignalReasonCode.ACCEPTED_POSITIVE,
        exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED,
        source_calibration_id=_CALIBRATION_ID, source_experiment_id=_EXPERIMENT_ID,
    )


def _scaled_target(trade_return: float, policy: ReturnCalculationPolicyKind, exposure_cap: float, compounded: bool) -> float:
    """The exact `cumulative_*_equity` a trade should reach at exit, given
    its OWN trade-level (gross or net) return, `exposure_cap`, and the
    fold's `compounding_policy` -- see module docstring for the full
    derivation of why this target differs between COMPOUNDED (a genuine
    multiplicative wealth factor, `M_exit**exposure_cap` under LOG) and
    NON_COMPOUNDED (LOG's preserved, Milestone-5.1-established "linear
    accumulation of the log return itself" convention,
    `1 + exposure_cap*trade_return`)."""
    if policy is ReturnCalculationPolicyKind.SIMPLE:
        m_exit = 1.0 + trade_return
        return 1.0 + exposure_cap * (m_exit - 1.0)
    if compounded:
        m_exit = math.exp(trade_return)
        return m_exit**exposure_cap
    return 1.0 + exposure_cap * trade_return


_PRICES = [100.0, 102.0, 99.0, 105.0, 101.0, 108.0]  # 6 bars, entry at [0], exit at [-1]
_DIRECTIONS = (PositionDirection.LONG, PositionDirection.SHORT)
_POLICIES = (ReturnCalculationPolicyKind.SIMPLE, ReturnCalculationPolicyKind.LOG)
_COMPOUNDING = (True, False)
_EXPOSURE_CAPS = (1.0, 0.5, 0.3)


class TestExactSingleTradeReconciliation:
    """Section 1's core requirement: fill-level economics -> trade-level
    gross PnL -> bar-level cumulative gross/net equity, exact, for EVERY
    (direction, policy, compounding, exposure_cap) combination."""

    @pytest.mark.parametrize("direction", _DIRECTIONS)
    @pytest.mark.parametrize("policy", _POLICIES)
    @pytest.mark.parametrize("compounded", _COMPOUNDING)
    @pytest.mark.parametrize("exposure_cap", _EXPOSURE_CAPS)
    def test_gross_and_net_equity_reconcile_exactly(
        self, direction: PositionDirection, policy: ReturnCalculationPolicyKind, compounded: bool, exposure_cap: float,
    ) -> None:
        bars = _bars(_PRICES)
        trade = _trade_from_result(
            entry_bar=0, exit_bar=len(_PRICES) - 1, entry_price=_PRICES[0], exit_price=_PRICES[-1], direction=direction,
            entry_spread=0.05, exit_spread=0.04, entry_slippage=0.02, exit_slippage=0.02,
            commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=3.0),
            financing_spec=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=1.0),
            holding_days=2.0, return_policy=policy, notional=1.0,
        )
        timeline = build_bar_return_timeline(
            trades=(trade,), bars=bars, fold_start_position=0, fold_end_position=len(_PRICES) - 1, outer_fold_index=0,
            return_calculation_policy=policy, exposure_cap=exposure_cap, compounded=compounded,
        )
        final = timeline.points[-1]

        gross_target = _scaled_target(trade.gross_return, policy, exposure_cap, compounded)
        net_target = _scaled_target(trade.net_return, policy, exposure_cap, compounded)

        assert final.cumulative_gross_equity == pytest.approx(gross_target, abs=1e-9), f"gross mismatch: {direction} {policy} compounded={compounded} cap={exposure_cap}"
        assert final.cumulative_net_equity == pytest.approx(net_target, abs=1e-9), f"net mismatch: {direction} {policy} compounded={compounded} cap={exposure_cap}"


class TestEntryOnlyAndExitOnlyCosts:
    def test_entry_only_cost_attributed_and_reconciles(self) -> None:
        bars = _bars(_PRICES)
        trade = _trade_from_result(
            entry_bar=1, exit_bar=4, entry_price=_PRICES[1], exit_price=_PRICES[4], direction=PositionDirection.LONG,
            entry_spread=0.10, exit_spread=0.0, entry_slippage=0.0, exit_slippage=0.0,
        )
        assert trade.cost_breakdown.exit_spread_cost == 0.0 and trade.cost_breakdown.entry_spread_cost > 0.0
        timeline = build_bar_return_timeline(
            trades=(trade,), bars=bars, fold_start_position=0, fold_end_position=len(_PRICES) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[1].transaction_costs == pytest.approx(trade.cost_breakdown.entry_spread_cost, abs=1e-12)
        assert by_bar[4].transaction_costs == 0.0
        assert timeline.points[-1].cumulative_net_equity == pytest.approx(1.0 + trade.net_return, abs=1e-9)

    def test_exit_only_cost_attributed_and_reconciles(self) -> None:
        bars = _bars(_PRICES)
        trade = _trade_from_result(
            entry_bar=1, exit_bar=4, entry_price=_PRICES[1], exit_price=_PRICES[4], direction=PositionDirection.SHORT,
            entry_spread=0.0, exit_spread=0.12, entry_slippage=0.0, exit_slippage=0.0,
        )
        assert trade.cost_breakdown.entry_spread_cost == 0.0 and trade.cost_breakdown.exit_spread_cost > 0.0
        timeline = build_bar_return_timeline(
            trades=(trade,), bars=bars, fold_start_position=0, fold_end_position=len(_PRICES) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False,
        )
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[1].transaction_costs == 0.0
        assert by_bar[4].transaction_costs == pytest.approx(trade.cost_breakdown.exit_spread_cost, abs=1e-12)
        assert timeline.points[-1].cumulative_net_equity == pytest.approx(1.0 + trade.net_return, abs=1e-9)

    def test_financing_only_cost_attributed_on_exit_bar_and_reconciles(self) -> None:
        bars = _bars(_PRICES)
        trade = _trade_from_result(
            entry_bar=0, exit_bar=5, entry_price=_PRICES[0], exit_price=_PRICES[5], direction=PositionDirection.LONG,
            financing_spec=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=5.0), holding_days=5.0,
        )
        assert trade.cost_breakdown.financing_cost > 0.0
        assert trade.cost_breakdown.entry_spread_cost == 0.0 and trade.cost_breakdown.exit_spread_cost == 0.0
        timeline = build_bar_return_timeline(
            trades=(trade,), bars=bars, fold_start_position=0, fold_end_position=len(_PRICES) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[5].transaction_costs == pytest.approx(trade.cost_breakdown.financing_cost, abs=1e-12)
        assert timeline.points[-1].cumulative_net_equity == pytest.approx(1.0 + trade.net_return, abs=1e-9)


class TestSameBarRoundTrip:
    @pytest.mark.parametrize("direction", _DIRECTIONS)
    @pytest.mark.parametrize("policy", _POLICIES)
    def test_same_bar_entry_and_exit_reconciles_exactly(self, direction: PositionDirection, policy: ReturnCalculationPolicyKind) -> None:
        bars = _bars(_PRICES)
        trade = _trade_from_result(entry_bar=2, exit_bar=2, entry_price=_PRICES[2], exit_price=_PRICES[2] + 3.0, direction=direction, return_policy=policy)
        timeline = build_bar_return_timeline(
            trades=(trade,), bars=bars, fold_start_position=0, fold_end_position=len(_PRICES) - 1, outer_fold_index=0,
            return_calculation_policy=policy, exposure_cap=1.0, compounded=True,
        )
        p2 = next(p for p in timeline.points if p.bar_position == 2)
        # `p2.realized_return` is always a RATIO-style bar return
        # (`V_exit/V_entry - 1`, COMPOUNDED mode's own unit) -- under LOG
        # policy this is `exp(trade.gross_return) - 1`, NOT `trade.
        # gross_return` itself (a log value); under SIMPLE they coincide.
        expected_bar_return = math.exp(trade.gross_return) - 1.0 if policy is ReturnCalculationPolicyKind.LOG else trade.gross_return
        assert p2.realized_return == pytest.approx(expected_bar_return, abs=1e-9)
        assert p2.unrealized_return == 0.0
        assert p2.entries_count == 1 and p2.exits_count == 1
        assert timeline.points[-1].cumulative_gross_equity == pytest.approx(_scaled_target(trade.gross_return, policy, 1.0, True), abs=1e-9)


class TestSequentialTradesCloseAndReverse:
    def test_close_and_reverse_is_exact_under_non_compounded_accumulation(self) -> None:
        """Close-and-reverse: trade 1 exits on bar 3, trade 2 enters
        (opposite direction) on the SAME bar 3. Under NON_COMPOUNDED
        accumulation, bar-level contributions from DIFFERENT trades
        sharing a bar are pure ADDITION (`cum = 1 + sum(...)`), which is
        commutative/associative regardless of how contributions are
        grouped -- so the combined run's final equity is EXACTLY the sum
        of each trade's own isolated total, with NO cross-term, even
        though they share a bar."""
        prices = [100.0, 103.0, 98.0, 106.0, 104.0, 110.0, 107.0]
        bars = _bars(prices)
        t1 = _trade_from_result(entry_bar=0, exit_bar=3, entry_price=prices[0], exit_price=prices[3], direction=PositionDirection.LONG, sig_pos=0)
        t2 = _trade_from_result(entry_bar=3, exit_bar=6, entry_price=prices[3], exit_price=prices[6], direction=PositionDirection.SHORT, sig_pos=1)
        combined = build_bar_return_timeline(
            trades=(t1, t2), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False,
        )
        alone1 = build_bar_return_timeline(
            trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False,
        )
        alone2 = build_bar_return_timeline(
            trades=(t2,), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False,
        )
        combined_total = combined.points[-1].cumulative_gross_equity - 1.0
        isolated_sum = (alone1.points[-1].cumulative_gross_equity - 1.0) + (alone2.points[-1].cumulative_gross_equity - 1.0)
        assert combined_total == pytest.approx(isolated_sum, abs=1e-12)

        # Bar 3 (the reversal bar) sees BOTH an exit and an entry.
        bar3 = next(p for p in combined.points if p.bar_position == 3)
        assert bar3.exits_count == 1 and bar3.entries_count == 1

    def test_close_and_reverse_under_compounding_has_a_documented_second_order_cross_term(self) -> None:
        """Under COMPOUNDED accumulation, when two DIFFERENT trades'
        contributions are summed into ONE bar's return and that SINGLE
        combined bar return is compounded (`cum *= 1+bar_return`), the
        result is `1 + a + b` for that bar, not `(1+a)*(1+b)` -- a
        second-order cross-term (`a*b`) versus treating the two trades as
        perfectly sequential (exit fully settled, THEN entry begins).
        This is the SAME mathematical shape as Milestone 5.1's documented
        gross/net compounding cross-term, now documented for same-bar
        multi-trade compounding too -- small at realistic per-bar return
        magnitudes, never silently hidden."""
        prices = [100.0, 103.0, 98.0, 106.0, 104.0, 110.0, 107.0]
        bars = _bars(prices)
        t1 = _trade_from_result(entry_bar=0, exit_bar=3, entry_price=prices[0], exit_price=prices[3], direction=PositionDirection.LONG, sig_pos=0)
        t2 = _trade_from_result(entry_bar=3, exit_bar=6, entry_price=prices[3], exit_price=prices[6], direction=PositionDirection.SHORT, sig_pos=1)
        combined = build_bar_return_timeline(
            trades=(t1, t2), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        naive_sequential = (1.0 + t1.gross_return) * (1.0 + t2.gross_return)
        actual = combined.points[-1].cumulative_gross_equity
        # Close (same order of magnitude as the per-bar returns squared), but NOT bit-exact.
        assert abs(actual - naive_sequential) < 1e-3
        assert abs(actual - naive_sequential) > 1e-9


class TestOverlappingTradesWhereSupported:
    def test_overlapping_trades_each_contribute_independently_without_double_counting(self) -> None:
        """Where overlap is supported (`OverlapPolicyKind.INDEPENDENT_
        OVERLAPPING` upstream), each trade's OWN bar-level contribution
        remains independently exact (proven by `TestExactSingleTrade
        Reconciliation`); the portfolio-level combination is the SUM of
        those independently-exact contributions (never double-counted,
        never silently dropped) -- proven here via a 2-trade overlap
        where each trade's OWN net contribution can be isolated by
        running it alone and comparing to its share of the combined run."""
        prices = [100.0, 103.0, 98.0, 106.0, 104.0]
        bars = _bars(prices)
        t1 = _trade_from_result(entry_bar=0, exit_bar=4, entry_price=prices[0], exit_price=prices[4], direction=PositionDirection.LONG, sig_pos=0)
        t2 = _trade_from_result(entry_bar=1, exit_bar=3, entry_price=prices[1], exit_price=prices[3], direction=PositionDirection.SHORT, sig_pos=1)
        exposure_cap = 0.4

        combined = build_bar_return_timeline(
            trades=(t1, t2), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=exposure_cap, compounded=False,
        )
        alone1 = build_bar_return_timeline(
            trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=exposure_cap, compounded=False,
        )
        alone2 = build_bar_return_timeline(
            trades=(t2,), bars=bars, fold_start_position=0, fold_end_position=len(prices) - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=exposure_cap, compounded=False,
        )
        # NON-COMPOUNDED mode: contributions are additive, so combined total gross
        # return must equal the SUM of each trade's own isolated total gross return exactly.
        combined_total = combined.points[-1].cumulative_gross_equity - 1.0
        isolated_sum = (alone1.points[-1].cumulative_gross_equity - 1.0) + (alone2.points[-1].cumulative_gross_equity - 1.0)
        assert combined_total == pytest.approx(isolated_sum, abs=1e-9)

        # No double counting: bar 1 (t2 enters, t1 still open) must show exactly one entry.
        by_bar = {p.bar_position: p for p in combined.points}
        assert by_bar[1].entries_count == 1
        assert by_bar[1].open_trade_count == 2  # t1 (opened bar 0) and t2 (just opened) both open


class TestFoldAndStitchedLevelReconciliation:
    def test_fold_final_gross_equity_matches_chained_trade_totals_and_stitches_exactly(self) -> None:
        """Fold-level AND stitched-level reconciliation in one proof: two
        outer folds, each with one trade, stitched together -- the
        stitched final gross equity must equal the product (COMPOUNDED)
        of each fold's own exact trade-chained equity."""
        from quant_platform.backtesting.stitching import build_stitched_walk_forward_equity

        prices_fold0 = [100.0, 104.0, 101.0, 109.0]
        prices_fold1 = [200.0, 195.0, 203.0, 198.0]
        bars0 = _bars(prices_fold0, start="2024-01-01")
        bars1 = _bars(prices_fold1, start="2024-02-01")  # strictly after fold 0's own bars

        t0 = _trade_from_result(entry_bar=0, exit_bar=3, entry_price=prices_fold0[0], exit_price=prices_fold0[3], direction=PositionDirection.LONG, sig_pos=0)
        t1 = _trade_from_result(entry_bar=0, exit_bar=3, entry_price=prices_fold1[0], exit_price=prices_fold1[3], direction=PositionDirection.SHORT, sig_pos=0)

        timeline0 = build_bar_return_timeline(
            trades=(t0,), bars=bars0, fold_start_position=0, fold_end_position=3, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        timeline1 = build_bar_return_timeline(
            trades=(t1,), bars=bars1, fold_start_position=0, fold_end_position=3, outer_fold_index=1,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        assert timeline0.points[-1].cumulative_gross_equity == pytest.approx(1.0 + t0.gross_return, abs=1e-9)
        assert timeline1.points[-1].cumulative_gross_equity == pytest.approx(1.0 + t1.gross_return, abs=1e-9)

        stitched = build_stitched_walk_forward_equity(backtest_id="c" * 64, timelines=[timeline0, timeline1])
        expected_stitched_final = (1.0 + t0.gross_return) * (1.0 + t1.gross_return)
        assert stitched.points[-1].stitched_gross_equity == pytest.approx(expected_stitched_final, abs=1e-9)
