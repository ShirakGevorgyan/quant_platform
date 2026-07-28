"""Hand-calculated tests for `backtesting.timeline.build_bar_return_timeline`
(Milestone 5.1, Section 1) -- the corrective bar-level mark-to-market
timeline that replaces trade-exit-event compression. Every test asserts
an exact, hand-derivable numeric value, not merely "does not crash"."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.models import (
    ExitReasonCode,
    PositionDirection,
    ReturnCalculationPolicyKind,
    SignalReasonCode,
    TradeStatus,
)
from quant_platform.backtesting.timeline import BarReturnBasis, build_bar_return_timeline
from quant_platform.backtesting.trades import TradeRecord, compute_trade_id
from quant_platform.core.exceptions import FinancialMetricError

_CALIBRATION_ID = "a" * 64
_EXPERIMENT_ID = "b" * 64


def _bars(n: int, *, start_price: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    prices = [start_price + i * step for i in range(n)]
    return pd.DataFrame({
        "open_time": timestamps, "open": prices, "high": [p + 0.2 for p in prices],
        "low": [p - 0.2 for p in prices], "close": [p + 0.1 for p in prices],
    })


def _trade(
    entry_bar: int, exit_bar: int, entry_price: float, exit_price: float, *, direction: PositionDirection = PositionDirection.LONG,
    sig_pos: int | None = None, entry_cost: float = 0.001, exit_cost: float = 0.001, return_policy: ReturnCalculationPolicyKind = ReturnCalculationPolicyKind.SIMPLE,
) -> TradeRecord:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    entry_ts = (base + pd.Timedelta(hours=entry_bar)).isoformat()
    exit_ts = (base + pd.Timedelta(hours=exit_bar)).isoformat()
    cb = CostBreakdown(entry_spread_cost=entry_cost / 2, exit_spread_cost=exit_cost / 2, entry_commission=0.0, exit_commission=0.0, entry_slippage=entry_cost / 2, exit_slippage=exit_cost / 2, financing_cost=0.0)
    sign = 1 if direction is PositionDirection.LONG else -1
    if return_policy is ReturnCalculationPolicyKind.SIMPLE:
        gross = sign * (exit_price - entry_price) / entry_price
        net = gross - cb.total_cost
    else:
        gross = sign * math.log(exit_price / entry_price)
        # Milestone 5.2, Section 2: costs are applied in the LINEAR value
        # space (M_gross = exp(gross)), not by linearly subtracting a
        # dollar-fraction cost from a log-space return -- matches
        # `returns.compute_trade_return_result`'s corrected formula.
        net = math.log(math.exp(gross) - cb.total_cost)
    sp = sig_pos if sig_pos is not None else entry_bar - 1
    tid = compute_trade_id(source_calibration_id=_CALIBRATION_ID, outer_fold_index=0, signal_sample_position=sp, direction=direction, entry_timestamp=entry_ts, exit_timestamp=exit_ts)
    return TradeRecord(
        schema_version=1, trade_id=tid, signal_sample_position=sp, outer_fold_index=0, direction=direction,
        signal_timestamp=entry_ts, decision_timestamp=entry_ts, entry_timestamp=entry_ts, entry_bar_position=entry_bar,
        entry_observed_price=entry_price, entry_effective_price=entry_price, exit_timestamp=exit_ts, exit_bar_position=exit_bar,
        exit_observed_price=exit_price, exit_effective_price=exit_price, holding_bars=exit_bar - entry_bar, gross_return=gross, net_return=net,
        cost_breakdown=cb, confidence=0.8, uncertainty=0.2, calibrated_probability=0.7,
        entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED,
        source_calibration_id=_CALIBRATION_ID, source_experiment_id=_EXPERIMENT_ID,
    )


class TestNoCompressionToExitEvents:
    def test_every_bar_in_range_gets_exactly_one_point_even_with_no_trades(self) -> None:
        bars = _bars(20)
        timeline = build_bar_return_timeline(trades=(), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        assert len(timeline.points) == 20
        assert all(p.entries_count == 0 and p.exits_count == 0 and p.gross_return == 0.0 for p in timeline.points)
        assert timeline.return_basis is BarReturnBasis.PREVIOUS_VALUATION_TO_CURRENT_VALUATION

    def test_many_bars_with_no_trade_event_still_produce_points_around_one_trade(self) -> None:
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        assert len(timeline.points) == 20
        zero_event = [p for p in timeline.points if p.entries_count == 0 and p.exits_count == 0]
        assert len(zero_event) == 20 - 2  # every bar except entry (3) and exit (8)


class TestPositionOpenAcrossMultipleBars:
    def test_open_trade_count_and_exposure_correct_across_the_hold(self) -> None:
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[2].open_trade_count == 0
        for bp in (3, 4, 5, 6, 7):
            assert by_bar[bp].open_trade_count == 1, bp
            assert by_bar[bp].active_long_exposure == 1.0
        assert by_bar[8].open_trade_count == 0  # closed exactly on the exit bar
        assert by_bar[9].open_trade_count == 0


class TestEntryAndExitOnDifferentBars:
    def test_gross_return_compounds_exactly_to_the_trade_total_for_long_simple(self) -> None:
        """Section 1's precise, verified reconciliation claim: compounding
        (multiplying) `1 + bar.gross_return` across every bar a LONG trade
        touches reproduces `1 + trade.gross_return` EXACTLY under SIMPLE
        returns (prices telescope multiplicatively)."""
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True)
        product = 1.0
        for p in timeline.points:
            if t1.entry_bar_position <= p.bar_position <= t1.exit_bar_position:
                product *= 1.0 + p.gross_return
        assert abs((product - 1.0) - t1.gross_return) < 1e-9

    def test_gross_return_compounds_exactly_for_short_under_log_returns(self) -> None:
        """Milestone 5.2: `cumulative_gross_equity` (built from the exact
        local-value-multiplier model) reconciles exactly to `exp(trade.
        gross_return)` for SHORT+LOG under COMPOUNDED accumulation -- the
        precise, corrected form of "LOG returns reconcile regardless of
        direction" (summing the raw per-bar `gross_return` figures is NOT
        the right check under compounding; see `test_exact_portfolio_
        accounting.py` for the complete, hand-derived proof of this)."""
        import math

        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.97, direction=PositionDirection.SHORT, return_policy=ReturnCalculationPolicyKind.LOG)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.LOG, exposure_cap=1.0, compounded=True)
        assert abs(timeline.points[-1].cumulative_gross_equity - math.exp(t1.gross_return)) < 1e-9

    def test_same_bar_entry_and_exit_is_fully_realized_on_that_one_bar(self) -> None:
        bars = _bars(20)
        t = _trade(12, 12, bars.iloc[12]["open"], bars.iloc[12]["close"])
        timeline = build_bar_return_timeline(trades=(t,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        p12 = next(p for p in timeline.points if p.bar_position == 12)
        assert abs(p12.gross_return - t.gross_return) < 1e-9
        assert abs(p12.realized_return - p12.gross_return) < 1e-9
        assert p12.unrealized_return == 0.0
        assert p12.entries_count == 1
        assert p12.exits_count == 1


class TestSimultaneousExitsAndOverlap:
    def test_two_trades_exiting_at_the_same_bar_are_both_counted(self) -> None:
        bars = _bars(15)
        t1 = _trade(2, 7, bars.iloc[2]["open"], bars.iloc[7]["close"], sig_pos=1)
        t2 = _trade(4, 9, bars.iloc[4]["open"], bars.iloc[9]["close"], direction=PositionDirection.SHORT, sig_pos=3)
        t3 = _trade(6, 9, bars.iloc[6]["open"], bars.iloc[9]["close"], sig_pos=5)
        timeline = build_bar_return_timeline(trades=(t1, t2, t3), bars=bars, fold_start_position=0, fold_end_position=14, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=0.5, compounded=True)
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[9].exits_count == 2
        assert by_bar[9].open_trade_count == 0

    def test_overlapping_positions_report_correct_net_and_absolute_exposure(self) -> None:
        """Bar 6: t1 (LONG, opened bar 2) + t2 (SHORT, opened bar 4) + t3
        (LONG, opened bar 6) all open simultaneously -- 2 long + 1 short at
        `exposure_cap=0.5` each: long=1.0, short=0.5, net=0.5, total_abs=1.5."""
        bars = _bars(15)
        t1 = _trade(2, 7, bars.iloc[2]["open"], bars.iloc[7]["close"], sig_pos=1)
        t2 = _trade(4, 9, bars.iloc[4]["open"], bars.iloc[9]["close"], direction=PositionDirection.SHORT, sig_pos=3)
        t3 = _trade(6, 9, bars.iloc[6]["open"], bars.iloc[9]["close"], sig_pos=5)
        timeline = build_bar_return_timeline(trades=(t1, t2, t3), bars=bars, fold_start_position=0, fold_end_position=14, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=0.5, compounded=True)
        by_bar = {p.bar_position: p for p in timeline.points}
        assert by_bar[6].open_trade_count == 3
        assert by_bar[6].active_long_exposure == 1.0
        assert by_bar[6].active_short_exposure == 0.5
        assert by_bar[6].net_exposure == 0.5
        assert by_bar[6].total_absolute_exposure == 1.5


class TestCostsOnEntryAndExitBars:
    def test_transaction_costs_split_across_entry_and_exit_bars_sum_exactly(self) -> None:
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999, entry_cost=0.002, exit_cost=0.0015)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        by_bar = {p.bar_position: p for p in timeline.points}
        assert abs(by_bar[3].transaction_costs - (t1.cost_breakdown.entry_spread_cost + t1.cost_breakdown.entry_commission + t1.cost_breakdown.entry_slippage)) < 1e-12
        assert abs(by_bar[8].transaction_costs - (t1.cost_breakdown.exit_spread_cost + t1.cost_breakdown.exit_commission + t1.cost_breakdown.exit_slippage + t1.cost_breakdown.financing_cost)) < 1e-12
        for bp in (4, 5, 6, 7):
            assert by_bar[bp].transaction_costs == 0.0
        total = sum(p.transaction_costs for p in timeline.points)
        assert abs(total - t1.cost_breakdown.total_cost) < 1e-12

    def test_overlapping_trades_costs_never_double_counted(self) -> None:
        """Milestone 5.2: bar-level `transaction_costs` are portfolio-level
        (scaled by `exposure_cap`, consistent with the gross/net return
        accounting), never the raw per-unit-notional `CostBreakdown.
        total_cost` figures -- see Section 4's turnover-notional fix for
        the same scaling convention applied to transacted notional."""
        bars = _bars(15)
        t1 = _trade(2, 7, bars.iloc[2]["open"], bars.iloc[7]["close"], sig_pos=1, entry_cost=0.001, exit_cost=0.001)
        t2 = _trade(4, 9, bars.iloc[4]["open"], bars.iloc[9]["close"], direction=PositionDirection.SHORT, sig_pos=3, entry_cost=0.0012, exit_cost=0.0008)
        t3 = _trade(6, 9, bars.iloc[6]["open"], bars.iloc[9]["close"], sig_pos=5, entry_cost=0.0005, exit_cost=0.0005)
        exposure_cap = 0.5
        timeline = build_bar_return_timeline(trades=(t1, t2, t3), bars=bars, fold_start_position=0, fold_end_position=14, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=exposure_cap, compounded=True)
        total_from_bars = sum(p.transaction_costs for p in timeline.points)
        total_from_trades = exposure_cap * sum(t.cost_breakdown.total_cost for t in (t1, t2, t3))
        assert abs(total_from_bars - total_from_trades) < 1e-9


class TestZeroEventBarsPreserveFinalPnlButChangeSamplingAndExposure:
    def test_appending_trailing_zero_event_bars_changes_sample_count_and_leaves_earlier_equity_untouched(self) -> None:
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        t2 = _trade(12, 12, bars.iloc[12]["open"], bars.iloc[12]["close"])
        trades = (t1, t2)

        short_timeline = build_bar_return_timeline(trades=trades, bars=bars, fold_start_position=0, fold_end_position=14, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True)
        long_timeline = build_bar_return_timeline(trades=trades, bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True)

        # Sample count changes exactly by the number of appended bars.
        assert len(long_timeline.points) - len(short_timeline.points) == 5

        # The 5 appended bars (15-19) are all zero-event (no trade touches them).
        appended = [p for p in long_timeline.points if p.bar_position >= 15]
        assert len(appended) == 5
        assert all(p.entries_count == 0 and p.exits_count == 0 and p.gross_return == 0.0 and p.net_return == 0.0 for p in appended)

        # Cumulative equity at bar 14 is IDENTICAL in both timelines --
        # appending trailing zero-event bars never changes any EARLIER
        # (or, since zero contributes a strict compounding identity, LATER)
        # cumulative equity value. This is the precise, provable sense of
        # "zero-event bars change sample count/exposure but not final PnL."
        assert abs(short_timeline.points[-1].cumulative_net_equity - long_timeline.points[14].cumulative_net_equity) < 1e-12
        assert abs(long_timeline.points[-1].cumulative_net_equity - long_timeline.points[14].cumulative_net_equity) < 1e-12
        assert abs(long_timeline.points[-1].cumulative_gross_equity - long_timeline.points[14].cumulative_gross_equity) < 1e-12

    def test_trade_level_pnl_is_never_recomputed_or_mutated_by_timeline_construction(self) -> None:
        """The strongest form of the guarantee: `TradeRecord` objects are
        passed BY REFERENCE into `build_bar_return_timeline` and never
        mutated (they are frozen dataclasses, so this is also structurally
        enforced, not just a runtime observation) -- trade-level PnL is
        computed entirely upstream, in `execution.py`, before this module
        is ever called."""
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        original_gross, original_net = t1.gross_return, t1.net_return
        build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True)
        assert t1.gross_return == original_gross
        assert t1.net_return == original_net


class TestStructuralInvariants:
    def test_gross_return_always_equals_realized_plus_unrealized(self) -> None:
        bars = _bars(20)
        t1 = _trade(3, 8, bars.iloc[3]["open"], bars.iloc[8]["close"] * 0.999)
        timeline = build_bar_return_timeline(trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=19, outer_fold_index=0, return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=False)
        for p in timeline.points:
            assert abs(p.gross_return - (p.realized_return + p.unrealized_return)) < 1e-12

    def test_point_count_must_equal_fold_bar_count_or_construction_rejects_it(self) -> None:
        from quant_platform.backtesting.timeline import BarReturnPoint, BarReturnTimeline

        with pytest.raises(Exception, match="exactly one point per bar"):
            BarReturnTimeline(
                schema_version=1, outer_fold_index=0, return_basis=BarReturnBasis.PREVIOUS_VALUATION_TO_CURRENT_VALUATION,
                compounded=False, fold_start_position=0, fold_end_position=4,
                points=(BarReturnPoint(
                    schema_version=1, bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.0, net_return=0.0,
                    realized_return=0.0, unrealized_return=0.0, active_long_exposure=0.0, active_short_exposure=0.0,
                    total_absolute_exposure=0.0, net_exposure=0.0, open_trade_count=0, entries_count=0, exits_count=0,
                    transaction_costs=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0,
                ),),  # only 1 point for a 5-bar range
            )


class TestExposureBlendedValueCatastrophicCancellationRegression:
    """FINAL MILESTONE 5 RELEASE AUDIT, Section 1: `_exposure_blended_
    value`'s SIMPLE branch previously computed `1.0 + exposure_cap *
    (m - 1.0)`, which suffers catastrophic cancellation once `m` is more
    than ~16 orders of magnitude below 1.0 (`m - 1.0` rounds to exactly
    `-1.0`, silently discarding `m`) -- at `exposure_cap == 1.0` this
    reported a FALSE `V == 0.0` (triggering the insolvency guard) for a
    position that had merely lost an extreme-but-not-total fraction of
    its value. Fixed by reformulating as `(1 - exposure_cap) + exposure_
    cap * m`, algebraically identical but numerically stable. This is
    most concretely observable across TWO consecutive bars that are both
    individually deep in the cancellation zone (so the OLD formula would
    raise `FinancialMetricError` at the very first such bar) but whose
    RATIO to each other is an ordinary, moderate further loss -- the
    correct behavior is to keep marking the position to market, not to
    falsely declare it insolvent."""

    def test_gradual_extreme_long_decay_through_the_cancellation_zone_does_not_falsely_fail_closed(self) -> None:
        """A GRADUAL decay (10% lost per bar, ~349 bars to cross the
        cancellation threshold) so that EVERY individual bar-to-bar
        transition is an ordinary, moderate loss -- never itself spanning
        enough orders of magnitude to hit the separate, genuinely
        unavoidable "ratio-of-vastly-different-magnitudes minus 1"
        floating-point floor (which is NOT this bug, and is NOT
        fixable -- see this class's own docstring). A single-bar jump
        straight from entry into the cancellation zone would hit that
        OTHER, correct-and-intentional fail-closed floor instead, and
        would not isolate the specific defect this test targets."""
        entry_price = 100.0
        n_bars = 400
        decay = 0.9
        prices = [entry_price * (decay**i) for i in range(n_bars)]
        timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
        bars = pd.DataFrame({
            "open_time": timestamps, "open": prices, "high": prices, "low": prices, "close": prices,
        })
        t1 = _trade(0, n_bars - 1, entry_price, prices[-1], direction=PositionDirection.LONG, entry_cost=0.0, exit_cost=0.0)

        # Must not raise anywhere along the whole decay (the regression:
        # the old formula raised INSIDE `_exposure_blended_value` as soon
        # as a bar's OWN local value multiplier crossed the cancellation
        # threshold, even though every bar-to-bar RATIO stays a normal 0.9).
        timeline = build_bar_return_timeline(
            trades=(t1,), bars=bars, fold_start_position=0, fold_end_position=n_bars - 1, outer_fold_index=0,
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=1.0, compounded=True,
        )
        # Every bar past the entry bar should show an ordinary ~-10% local return.
        for p in timeline.points[1:]:
            assert p.gross_return == pytest.approx(-0.1, abs=1e-6), p.bar_position
            assert math.isfinite(p.cumulative_gross_equity)
            assert p.cumulative_gross_equity > 0.0
        # The final point is deep in the cancellation zone (m < 1.1e-16 relative to entry).
        assert prices[-1] / entry_price < 1.1e-16

    def test_exposure_blended_value_reports_a_tiny_positive_value_not_a_false_zero(self) -> None:
        from quant_platform.backtesting.timeline import _exposure_blended_value

        # m = 1e-17 is deep in the cancellation zone at exposure_cap=1.0.
        v = _exposure_blended_value(1e-17, 1.0, ReturnCalculationPolicyKind.SIMPLE)
        assert v == pytest.approx(1e-17, rel=1e-9)
        assert v > 0.0  # the old formula returned exactly 0.0 here

    def test_exposure_blended_value_still_fails_closed_at_the_genuine_insolvency_boundary(self) -> None:
        """The fix must not weaken the GENUINE insolvency boundary: a
        SHORT+SIMPLE position at exactly price == 2 * entry_price with
        exposure_cap == 1.0 has a true, exact local value of zero (not a
        floating-point artifact) and must still raise."""
        from quant_platform.backtesting.timeline import _exposure_blended_value

        m_at_true_zero = 2.0 - 200.0 / 100.0  # == 0.0 exactly, not a cancellation artifact
        with pytest.raises(FinancialMetricError, match="must remain finite and positive"):
            _exposure_blended_value(m_at_true_zero, 1.0, ReturnCalculationPolicyKind.SIMPLE)
