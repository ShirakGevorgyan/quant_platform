"""Milestone 5.1, Section 3: hand-calculated tests for `backtesting.
stitching` -- proves the carry-forward stitching identity exactly for
both compounding conventions, the fold-ordering/overlap fail-closed
checks, the `EquityCurve` projection, and `compute_stitched_financial_
metrics`'s basic values. Uses hand-built minimal `BarReturnTimeline`s
(never a full trade simulation) so every expected number is computed by
hand in this file's own comments, exactly `test_bar_return_timeline.py`'s
convention one layer down."""

from __future__ import annotations

import pytest

from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.stitching import (
    STITCHED_EQUITY_OUTER_FOLD_INDEX,
    StitchedWalkForwardEquity,
    build_stitched_walk_forward_equity,
    compute_stitched_financial_metrics,
    stitched_walk_forward_equity_to_equity_curve,
)
from quant_platform.backtesting.timeline import BarReturnBasis, BarReturnPoint, BarReturnTimeline
from quant_platform.core.exceptions import FinancialMetricError
from quant_platform.core.types import Timeframe


def _point(
    *, bar_position: int, timestamp: str, gross_return: float, net_return: float, cumulative_gross_equity: float,
    cumulative_net_equity: float, peak_equity: float, drawdown: float, entries_count: int = 0, exits_count: int = 0,
    open_trade_count: int = 0,
) -> BarReturnPoint:
    return BarReturnPoint(
        schema_version=1, bar_position=bar_position, timestamp=timestamp, gross_return=gross_return, net_return=net_return,
        realized_return=gross_return, unrealized_return=0.0, active_long_exposure=0.0, active_short_exposure=0.0,
        total_absolute_exposure=0.0, net_exposure=0.0, open_trade_count=open_trade_count, entries_count=entries_count,
        exits_count=exits_count, transaction_costs=gross_return - net_return, cumulative_gross_equity=cumulative_gross_equity,
        cumulative_net_equity=cumulative_net_equity, peak_equity=peak_equity, drawdown=drawdown,
    )


def _timeline(*, outer_fold_index: int, compounded: bool, fold_start_position: int, points: list[BarReturnPoint]) -> BarReturnTimeline:
    return BarReturnTimeline(
        schema_version=1, outer_fold_index=outer_fold_index, return_basis=BarReturnBasis.PREVIOUS_VALUATION_TO_CURRENT_VALUATION,
        compounded=compounded, fold_start_position=fold_start_position, fold_end_position=fold_start_position + len(points) - 1, points=tuple(points),
    )


class TestCompoundedCarryForwardIdentity:
    """Fold 0: gross +10% then -5% (compounded: 1.10 -> 1.045); net +8%
    then -6% (1.08 -> 1.0152). Fold 1 (independently starts flat and its
    OWN local equity restarts at 1.0): gross +20% then flat (1.20 -> 1.20);
    net +18% then flat (1.18 -> 1.18). Stitched must CONTINUE from fold
    0's ending equity: stitched_gross = 1.045 * 1.20 = 1.254, stitched_net
    = 1.0152 * 1.18 = 1.197936 -- hand-computed, not merely re-executing
    the implementation."""

    def _timelines(self) -> list[BarReturnTimeline]:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.10, net_return=0.08, cumulative_gross_equity=1.10, cumulative_net_equity=1.08, peak_equity=1.08, drawdown=0.0),
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=-0.05, net_return=-0.06, cumulative_gross_equity=1.045, cumulative_net_equity=1.0152, peak_equity=1.08, drawdown=0.06),
        ])
        fold1 = _timeline(outer_fold_index=1, compounded=True, fold_start_position=10, points=[
            _point(bar_position=10, timestamp="2024-01-02T00:00:00+00:00", gross_return=0.20, net_return=0.18, cumulative_gross_equity=1.20, cumulative_net_equity=1.18, peak_equity=1.18, drawdown=0.0),
            _point(bar_position=11, timestamp="2024-01-02T01:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.20, cumulative_net_equity=1.18, peak_equity=1.18, drawdown=0.0),
        ])
        return [fold0, fold1]

    def test_carry_forward_matches_hand_computation(self) -> None:
        stitched = build_stitched_walk_forward_equity(backtest_id="b" * 64, timelines=self._timelines())
        assert len(stitched.points) == 4
        assert stitched.points[0].stitched_gross_equity == pytest.approx(1.10)
        assert stitched.points[0].stitched_net_equity == pytest.approx(1.08)
        assert stitched.points[1].stitched_gross_equity == pytest.approx(1.045)
        assert stitched.points[1].stitched_net_equity == pytest.approx(1.0152)
        # Fold 1 CONTINUES from fold 0's ending equity, never restarting at 1.0.
        assert stitched.points[2].stitched_gross_equity == pytest.approx(1.045 * 1.20)
        assert stitched.points[2].stitched_net_equity == pytest.approx(1.0152 * 1.18)
        assert stitched.points[3].stitched_gross_equity == pytest.approx(1.254)
        assert stitched.points[3].stitched_net_equity == pytest.approx(1.197936)

    def test_fold_boundaries_record_carry_in_and_contiguous_point_ranges(self) -> None:
        stitched = build_stitched_walk_forward_equity(backtest_id="b" * 64, timelines=self._timelines())
        assert len(stitched.fold_boundaries) == 2
        b0, b1 = stitched.fold_boundaries
        assert (b0.outer_fold_index, b0.stitched_point_start_index, b0.stitched_point_end_index) == (0, 0, 1)
        assert (b1.outer_fold_index, b1.stitched_point_start_index, b1.stitched_point_end_index) == (1, 2, 3)
        assert b0.carry_in_gross_equity == pytest.approx(1.0)
        assert b0.carry_in_net_equity == pytest.approx(1.0)
        # Fold 1's carry-in IS fold 0's own final stitched equity.
        assert b1.carry_in_gross_equity == pytest.approx(1.045)
        assert b1.carry_in_net_equity == pytest.approx(1.0152)
        assert b0.fold_start_position == 0 and b0.fold_end_position == 1
        assert b1.fold_start_position == 10 and b1.fold_end_position == 11

    def test_bar_local_returns_are_unaffected_by_stitching(self) -> None:
        """The per-bar LOCAL return fields must be copied through
        unchanged -- only the CUMULATIVE level is carry-rescaled."""
        stitched = build_stitched_walk_forward_equity(backtest_id="b" * 64, timelines=self._timelines())
        assert [p.bar_gross_return for p in stitched.points] == [0.10, -0.05, 0.20, 0.0]
        assert [p.bar_net_return for p in stitched.points] == [0.08, -0.06, 0.18, 0.0]


class TestNonCompoundedCarryForwardIdentity:
    """Fold 0 (non-compounded, additive): +5% then +3% -> local gross
    equity 1.08; net +4% then +2% -> 1.06. Fold 1: +10% then -2% -> local
    gross 1.08; net +9% then -3% -> 1.06. Stitched (additive carry):
    fold 1's points = carry_in + (local - 1.0), so final stitched_gross =
    1.08 + (1.08-1.0) = 1.16; final stitched_net = 1.06 + (1.06-1.0) = 1.12."""

    def _timelines(self) -> list[BarReturnTimeline]:
        fold0 = _timeline(outer_fold_index=0, compounded=False, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.05, net_return=0.04, cumulative_gross_equity=1.05, cumulative_net_equity=1.04, peak_equity=1.04, drawdown=0.0),
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=0.03, net_return=0.02, cumulative_gross_equity=1.08, cumulative_net_equity=1.06, peak_equity=1.06, drawdown=0.0),
        ])
        fold1 = _timeline(outer_fold_index=1, compounded=False, fold_start_position=10, points=[
            _point(bar_position=10, timestamp="2024-01-02T00:00:00+00:00", gross_return=0.10, net_return=0.09, cumulative_gross_equity=1.10, cumulative_net_equity=1.09, peak_equity=1.09, drawdown=0.0),
            _point(bar_position=11, timestamp="2024-01-02T01:00:00+00:00", gross_return=-0.02, net_return=-0.03, cumulative_gross_equity=1.08, cumulative_net_equity=1.06, peak_equity=1.09, drawdown=(1.09 - 1.06) / 1.09),
        ])
        return [fold0, fold1]

    def test_additive_carry_forward_matches_hand_computation(self) -> None:
        stitched = build_stitched_walk_forward_equity(backtest_id="c" * 64, timelines=self._timelines())
        assert stitched.points[1].stitched_gross_equity == pytest.approx(1.08)
        assert stitched.points[1].stitched_net_equity == pytest.approx(1.06)
        assert stitched.points[2].stitched_gross_equity == pytest.approx(1.18)
        assert stitched.points[2].stitched_net_equity == pytest.approx(1.15)
        assert stitched.points[3].stitched_gross_equity == pytest.approx(1.16)
        assert stitched.points[3].stitched_net_equity == pytest.approx(1.12)


class TestRejectsOutOfOrderOrOverlappingFolds:
    def test_rejects_non_increasing_outer_fold_index(self) -> None:
        fold_a = _timeline(outer_fold_index=1, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        fold_b = _timeline(outer_fold_index=0, compounded=True, fold_start_position=1, points=[
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        with pytest.raises(FinancialMetricError, match="strictly increasing"):
            build_stitched_walk_forward_equity(backtest_id="d" * 64, timelines=[fold_a, fold_b])

    def test_rejects_overlapping_or_non_chronological_timestamps(self) -> None:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-02T00:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        fold1 = _timeline(outer_fold_index=1, compounded=True, fold_start_position=1, points=[
            # Begins BEFORE fold 0's own bar -- must be rejected, not silently accepted.
            _point(bar_position=1, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        with pytest.raises(FinancialMetricError, match="chronologically ordered"):
            build_stitched_walk_forward_equity(backtest_id="e" * 64, timelines=[fold0, fold1])

    def test_rejects_mismatched_compounding_modes(self) -> None:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        fold1 = _timeline(outer_fold_index=1, compounded=False, fold_start_position=1, points=[
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=0.0, net_return=0.0, cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0),
        ])
        with pytest.raises(FinancialMetricError, match="same compounding mode"):
            build_stitched_walk_forward_equity(backtest_id="f" * 64, timelines=[fold0, fold1])

    def test_rejects_empty_timelines(self) -> None:
        with pytest.raises(FinancialMetricError, match="non-empty"):
            build_stitched_walk_forward_equity(backtest_id="a" * 64, timelines=[])


class TestEquityCurveProjectionAndDrawdown:
    def _stitched(self) -> StitchedWalkForwardEquity:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.10, net_return=0.08, cumulative_gross_equity=1.10, cumulative_net_equity=1.08, peak_equity=1.08, drawdown=0.0, exits_count=1),
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=-0.10, net_return=-0.10, cumulative_gross_equity=0.99, cumulative_net_equity=0.972, peak_equity=1.08, drawdown=(1.08 - 0.972) / 1.08, exits_count=1),
        ])
        return build_stitched_walk_forward_equity(backtest_id="9" * 64, timelines=[fold0])

    def test_projection_preserves_local_returns_and_running_trade_count(self) -> None:
        stitched = self._stitched()
        curve = stitched_walk_forward_equity_to_equity_curve(stitched)
        assert curve.outer_fold_index == STITCHED_EQUITY_OUTER_FOLD_INDEX
        assert [p.period_gross_return for p in curve.points] == [0.10, -0.10]
        assert [p.trade_count_to_date for p in curve.points] == [1, 2]
        assert curve.points[-1].cumulative_net_equity == pytest.approx(0.972)

    def test_drawdown_report_reflects_the_stitched_series(self) -> None:
        stitched = self._stitched()
        curve = stitched_walk_forward_equity_to_equity_curve(stitched)
        net_dd = compute_drawdown_report(curve, equity_basis="net")
        assert net_dd.maximum_drawdown == pytest.approx((1.08 - 0.972) / 1.08)


class TestComputeStitchedFinancialMetrics:
    def test_total_returns_and_sharpe_skip_reason_for_single_point(self) -> None:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.05, net_return=0.04, cumulative_gross_equity=1.05, cumulative_net_equity=1.04, peak_equity=1.04, drawdown=0.0),
        ])
        stitched = build_stitched_walk_forward_equity(backtest_id="7" * 64, timelines=[fold0])
        curve = stitched_walk_forward_equity_to_equity_curve(stitched)
        gross_dd = compute_drawdown_report(curve, equity_basis="gross")
        net_dd = compute_drawdown_report(curve, equity_basis="net")
        report = compute_stitched_financial_metrics(
            stitched=stitched, gross_drawdown=gross_dd, net_drawdown=net_dd, bar_interval=Timeframe.H1,
            annual_risk_free_rate=0.0, initial_notional=10000.0, total_transacted_notional=0.0,
        )
        assert report.values["stitched_total_gross_return"] == pytest.approx(0.05)
        assert report.values["stitched_total_net_return"] == pytest.approx(0.04)
        # Fewer than 2 bar observations -- Sharpe/Sortino/volatility must be SKIPPED, never fabricated.
        assert "stitched_bar_return_sharpe" in report.skipped
        assert "stitched_sortino_ratio" in report.skipped
        assert "stitched_volatility" in report.skipped

    def test_transaction_count_is_a_bar_sampled_aggregate_and_exposure_too(self) -> None:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.02, net_return=0.01, cumulative_gross_equity=1.02, cumulative_net_equity=1.01, peak_equity=1.01, drawdown=0.0, entries_count=1, open_trade_count=1),
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=0.01, net_return=0.005, cumulative_gross_equity=1.0302, cumulative_net_equity=1.015, peak_equity=1.015, drawdown=0.0, exits_count=1),
        ])
        stitched = build_stitched_walk_forward_equity(backtest_id="6" * 64, timelines=[fold0])
        curve = stitched_walk_forward_equity_to_equity_curve(stitched)
        gross_dd = compute_drawdown_report(curve, equity_basis="gross")
        net_dd = compute_drawdown_report(curve, equity_basis="net")
        report = compute_stitched_financial_metrics(
            stitched=stitched, gross_drawdown=gross_dd, net_drawdown=net_dd, bar_interval=Timeframe.H1,
            annual_risk_free_rate=0.0, initial_notional=10000.0, total_transacted_notional=5000.0,
        )
        assert report.values["stitched_transaction_count"] == pytest.approx(2.0)  # 1 entry + 1 exit
        assert report.values["stitched_time_in_market_fraction"] == pytest.approx(0.5)  # 1 of 2 bars has open_trade_count > 0

    def test_turnover_notional_ratio_uses_the_caller_supplied_pooled_total_not_transaction_count(self) -> None:
        """Milestone 5.2, Section 4: `stitched_turnover_notional_ratio`
        must be `total_transacted_notional / initial_notional` -- NOT
        `stitched_transaction_count * exposure_cap` (the corrected-away
        formula, which this test would still pass under the OLD, wrong
        implementation only by coincidence; the values below are chosen
        so the two formulas disagree, proving the new one is what runs)."""
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.02, net_return=0.01, cumulative_gross_equity=1.02, cumulative_net_equity=1.01, peak_equity=1.01, drawdown=0.0, entries_count=1, open_trade_count=1),
            _point(bar_position=1, timestamp="2024-01-01T01:00:00+00:00", gross_return=0.01, net_return=0.005, cumulative_gross_equity=1.0302, cumulative_net_equity=1.015, peak_equity=1.015, drawdown=0.0, exits_count=1),
        ])
        stitched = build_stitched_walk_forward_equity(backtest_id="9" * 64, timelines=[fold0])
        curve = stitched_walk_forward_equity_to_equity_curve(stitched)
        gross_dd = compute_drawdown_report(curve, equity_basis="gross")
        net_dd = compute_drawdown_report(curve, equity_basis="net")
        report = compute_stitched_financial_metrics(
            stitched=stitched, gross_drawdown=gross_dd, net_drawdown=net_dd, bar_interval=Timeframe.H1,
            annual_risk_free_rate=0.0, initial_notional=10000.0, total_transacted_notional=7500.0,
        )
        assert report.values["stitched_turnover_notional_ratio"] == pytest.approx(0.75)  # 7500 / 10000
        # The old (wrong) formula would have produced `transaction_count * exposure_cap` -- a
        # different number entirely, with no `exposure_cap` even in scope any more.
        assert report.values["stitched_turnover_notional_ratio"] != pytest.approx(report.values["stitched_transaction_count"])


class TestStructuralInvariants:
    def test_rejects_non_contiguous_fold_boundaries(self) -> None:
        from quant_platform.backtesting.stitching import StitchedEquityPoint, StitchedFoldBoundary

        point = StitchedEquityPoint(
            schema_version=1, outer_fold_index=0, bar_position=0, timestamp="2024-01-01T00:00:00+00:00",
            bar_gross_return=0.0, bar_net_return=0.0, stitched_gross_equity=1.0, stitched_net_equity=1.0,
            total_absolute_exposure=0.0, net_exposure=0.0, open_trade_count=0, entries_count=0, exits_count=0,
        )
        bad_boundary = StitchedFoldBoundary(
            outer_fold_index=0, stitched_point_start_index=1, stitched_point_end_index=1,  # should start at 0
            fold_start_position=0, fold_end_position=0, carry_in_gross_equity=1.0, carry_in_net_equity=1.0,
        )
        with pytest.raises(FinancialMetricError, match="contiguously cover"):
            StitchedWalkForwardEquity(schema_version=1, backtest_id="8" * 64, compounded=True, fold_boundaries=(bad_boundary,), points=(point,))

    def test_round_trips_through_json(self) -> None:
        fold0 = _timeline(outer_fold_index=0, compounded=True, fold_start_position=0, points=[
            _point(bar_position=0, timestamp="2024-01-01T00:00:00+00:00", gross_return=0.01, net_return=0.005, cumulative_gross_equity=1.01, cumulative_net_equity=1.005, peak_equity=1.005, drawdown=0.0),
        ])
        stitched = build_stitched_walk_forward_equity(backtest_id="5" * 64, timelines=[fold0])
        roundtripped = StitchedWalkForwardEquity.from_json_dict(stitched.to_json_dict())
        assert roundtripped.to_json_dict() == stitched.to_json_dict()
