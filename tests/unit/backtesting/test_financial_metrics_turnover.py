"""Milestone 5.2, Section 4: `compute_financial_metrics`'s corrected
turnover formula. `turnover_notional_ratio` must be computed from each
closed trade's own persisted entry/exit EFFECTIVE fill prices -- never
`transaction_count * exposure_cap` (an identity that only happens to hold
when every trade's exit leg transacts the identical dollar notional as
its entry leg, which is false whenever price moved between entry and
exit). Every test here hand-derives its expected numeric value from the
`entry_notional = exposure_cap * initial_notional`, `exit_notional =
entry_notional * (exit_effective_price / entry_effective_price)` formula
directly, rather than merely asserting "does not crash".

ALSO Milestone 5.2, Section 7: `trades_per_bar` (genuine round trips per
bar) vs. `transaction_sides_per_bar` (entry+exit SIDES per bar, always
exactly double `trades_per_bar`) -- a prior delivery report conflated the
two, calling `transaction_sides_per_bar`'s value "round-trips per bar".
`TestTradesPerBarAndTransactionSidesPerBar`/`TestReversalRate` below
prove they are distinct, correctly-named, separately-persisted metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.drawdown import compute_drawdown_report
from quant_platform.backtesting.metrics import compute_financial_metrics
from quant_platform.backtesting.models import (
    ExitReasonCode,
    PositionDirection,
    ReturnCalculationPolicyKind,
    SignalReasonCode,
    TradeStatus,
)
from quant_platform.backtesting.signals import Signal, SignalSet
from quant_platform.backtesting.timeline import bar_return_timeline_to_equity_curve, build_bar_return_timeline
from quant_platform.backtesting.trades import TradeRecord, TradeSet, compute_trade_id
from quant_platform.core.types import Timeframe

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
    entry_bar: int, exit_bar: int, entry_price: float, exit_price: float, *, direction: PositionDirection = PositionDirection.LONG, sig_pos: int | None = None,
) -> TradeRecord:
    """Zero-cost trade (so `gross_return`/`net_return` need no cost-space
    reconciliation for these turnover-only tests) -- `entry_effective_
    price`/`exit_effective_price` equal the observed prices, which is
    what `compute_financial_metrics`'s turnover formula actually reads."""
    base = pd.Timestamp("2024-01-01", tz="UTC")
    entry_ts = (base + pd.Timedelta(hours=entry_bar)).isoformat()
    exit_ts = (base + pd.Timedelta(hours=exit_bar)).isoformat()
    cb = CostBreakdown(entry_spread_cost=0.0, exit_spread_cost=0.0, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0)
    sign = 1 if direction is PositionDirection.LONG else -1
    gross = sign * (exit_price - entry_price) / entry_price
    sp = sig_pos if sig_pos is not None else entry_bar
    tid = compute_trade_id(source_calibration_id=_CALIBRATION_ID, outer_fold_index=0, signal_sample_position=sp, direction=direction, entry_timestamp=entry_ts, exit_timestamp=exit_ts)
    return TradeRecord(
        schema_version=1, trade_id=tid, signal_sample_position=sp, outer_fold_index=0, direction=direction,
        signal_timestamp=entry_ts, decision_timestamp=entry_ts, entry_timestamp=entry_ts, entry_bar_position=entry_bar,
        entry_observed_price=entry_price, entry_effective_price=entry_price, exit_timestamp=exit_ts, exit_bar_position=exit_bar,
        exit_observed_price=exit_price, exit_effective_price=exit_price, holding_bars=exit_bar - entry_bar, gross_return=gross, net_return=gross,
        cost_breakdown=cb, confidence=0.8, uncertainty=0.2, calibrated_probability=0.7,
        entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED,
        source_calibration_id=_CALIBRATION_ID, source_experiment_id=_EXPERIMENT_ID,
    )


def _signal(sample_position: int, *, direction: PositionDirection = PositionDirection.LONG, accepted: bool = True) -> Signal:
    reason = SignalReasonCode.ACCEPTED_POSITIVE if accepted else SignalReasonCode.BELOW_CONFIDENCE_FLOOR
    return Signal(
        sample_position=sample_position, decision_timestamp=pd.Timestamp("2024-01-01", tz="UTC").isoformat(), direction=direction, strength=1.0,
        accepted=accepted, reason_code=reason, confidence=0.8, uncertainty=0.2, threshold=0.5, calibrated_probability=0.7,
        source_calibration_id=_CALIBRATION_ID, source_experiment_id=_EXPERIMENT_ID, outer_fold_index=0,
    )


def _metrics(
    trades: tuple[TradeRecord, ...], *, exposure_cap: float, initial_notional: float, n_bars: int = 12, signals: SignalSet | None = None,
):
    bars = _bars(n_bars)
    bar_timeline = build_bar_return_timeline(
        trades=trades, bars=bars, fold_start_position=0, fold_end_position=n_bars - 1, outer_fold_index=0,
        return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, exposure_cap=exposure_cap, compounded=False,
    )
    equity_curve = bar_return_timeline_to_equity_curve(bar_timeline)
    gross_dd = compute_drawdown_report(equity_curve, equity_basis="gross")
    net_dd = compute_drawdown_report(equity_curve, equity_basis="net")
    trade_set = TradeSet(schema_version=1, outer_fold_index=0, trades=trades)
    if signals is None:
        positions = {t.signal_sample_position for t in trades} or {0}
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=tuple(_signal(p) for p in sorted(positions)))
    return compute_financial_metrics(
        trades=trade_set, equity_curve=equity_curve, gross_drawdown=gross_dd, net_drawdown=net_dd, signals=signals,
        bar_timeline=bar_timeline, bar_interval=Timeframe.H1, annual_risk_free_rate=0.0, initial_notional=initial_notional, exposure_cap=exposure_cap,
    )


def _expected_total_transacted_notional(trades: tuple[TradeRecord, ...], *, exposure_cap: float, initial_notional: float) -> float:
    entry_notional = exposure_cap * initial_notional
    return sum(entry_notional * (1.0 + t.exit_effective_price / t.entry_effective_price) for t in trades)


class TestPartialExposureAndBelowOneCaps:
    def test_partial_exposure_scales_transacted_notional_linearly(self) -> None:
        t1 = _trade(0, 5, 100.0, 110.0)
        report = _metrics((t1,), exposure_cap=0.6, initial_notional=10000.0)
        expected_total = _expected_total_transacted_notional((t1,), exposure_cap=0.6, initial_notional=10000.0)
        assert expected_total == pytest.approx(6000.0 + 6600.0)
        assert report.values["total_transacted_notional"] == pytest.approx(expected_total)
        assert report.values["turnover_notional_ratio"] == pytest.approx(expected_total / 10000.0)
        assert report.values["transaction_count"] == pytest.approx(2.0)

    def test_exposure_cap_well_below_one(self) -> None:
        t1 = _trade(0, 5, 100.0, 105.0)
        report = _metrics((t1,), exposure_cap=0.1, initial_notional=10000.0)
        # entry_notional = 1000, exit_notional = 1000 * 1.05 = 1050
        assert report.values["total_transacted_notional"] == pytest.approx(2050.0)
        assert report.values["turnover_notional_ratio"] == pytest.approx(0.205)


class TestCloseAndReverseAndOverlapping:
    def test_close_and_reverse_sums_both_legs_notional_independently(self) -> None:
        t1 = _trade(0, 3, 100.0, 105.0, direction=PositionDirection.LONG, sig_pos=0)
        t2 = _trade(3, 6, 105.0, 98.0, direction=PositionDirection.SHORT, sig_pos=3)
        report = _metrics((t1, t2), exposure_cap=0.5, initial_notional=10000.0)
        expected_total = _expected_total_transacted_notional((t1, t2), exposure_cap=0.5, initial_notional=10000.0)
        assert report.values["total_transacted_notional"] == pytest.approx(expected_total)
        assert report.values["transaction_count"] == pytest.approx(4.0)

    def test_overlapping_trades_notional_summed_independently(self) -> None:
        t1 = _trade(0, 5, 100.0, 108.0, sig_pos=0)
        t2 = _trade(2, 4, 103.0, 101.0, sig_pos=2)
        report = _metrics((t1, t2), exposure_cap=0.4, initial_notional=20000.0)
        expected_total = _expected_total_transacted_notional((t1, t2), exposure_cap=0.4, initial_notional=20000.0)
        assert report.values["total_transacted_notional"] == pytest.approx(expected_total)
        assert report.values["turnover_notional_ratio"] == pytest.approx(expected_total / 20000.0)


class TestSameBarRoundTrip:
    def test_same_bar_entry_and_exit_still_contributes_full_round_trip_notional(self) -> None:
        t1 = _trade(4, 4, 100.0, 101.0, sig_pos=4)
        report = _metrics((t1,), exposure_cap=1.0, initial_notional=5000.0)
        # entry_notional = 5000, exit_notional = 5000 * 1.01 = 5050
        assert report.values["total_transacted_notional"] == pytest.approx(5000.0 + 5050.0)
        assert report.values["transaction_count"] == pytest.approx(2.0)


class TestFlatAcceptedSignalProducesNoTrades:
    def test_flat_accepted_signal_skips_turnover_rather_than_reporting_zero(self) -> None:
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, direction=PositionDirection.FLAT, accepted=True),))
        report = _metrics((), exposure_cap=0.5, initial_notional=10000.0, signals=signals)
        for name in ("transaction_count", "total_transacted_notional", "turnover_notional_ratio", "annualized_turnover"):
            assert name in report.skipped, f"{name} should be skipped (no closed trades), not fabricated as zero"
            assert name not in report.values
        assert report.values["accepted_signal_rate"] == pytest.approx(1.0)


class TestDifferentInitialNotionals:
    def test_total_transacted_notional_scales_but_ratio_is_invariant(self) -> None:
        t1 = _trade(0, 5, 100.0, 112.0, sig_pos=0)
        small = _metrics((t1,), exposure_cap=0.5, initial_notional=10000.0)
        large = _metrics((t1,), exposure_cap=0.5, initial_notional=50000.0)
        assert large.values["total_transacted_notional"] == pytest.approx(small.values["total_transacted_notional"] * 5.0)
        assert large.values["turnover_notional_ratio"] == pytest.approx(small.values["turnover_notional_ratio"])


class TestNotTheOldFormula:
    def test_turnover_notional_ratio_is_not_transaction_count_times_exposure_cap(self) -> None:
        """The defect this milestone fixes: the OLD formula was
        `transaction_count * exposure_cap`, which ignores price movement
        entirely. Choosing a large price move makes the two formulas
        diverge by a wide margin, proving the corrected formula is what
        actually runs (not merely a coincidental match)."""
        t1 = _trade(0, 5, 100.0, 150.0, sig_pos=0)  # a deliberately large 50% move
        report = _metrics((t1,), exposure_cap=0.5, initial_notional=10000.0)
        old_wrong_formula = report.values["transaction_count"] * 0.5
        assert report.values["turnover_notional_ratio"] != pytest.approx(old_wrong_formula)
        # entry_notional = 5000, exit_notional = 5000 * 1.5 = 7500, total = 12500, ratio = 1.25
        assert report.values["turnover_notional_ratio"] == pytest.approx(1.25)
        assert old_wrong_formula == pytest.approx(1.0)  # 2 * 0.5 -- the old, wrong answer

    def test_annualized_turnover_is_ratio_divided_by_fold_duration_years(self) -> None:
        t1 = _trade(0, 5, 100.0, 110.0, sig_pos=0)
        report = _metrics((t1,), exposure_cap=1.0, initial_notional=10000.0, n_bars=12)
        periods_per_year = 365.25 * 24 * 60 / Timeframe.H1.minutes
        duration_years = 12 / periods_per_year
        assert report.values["annualized_turnover"] == pytest.approx(report.values["turnover_notional_ratio"] / duration_years)


class TestTradesPerBarAndTransactionSidesPerBar:
    def test_trades_per_bar_is_round_trip_count_over_bar_count(self) -> None:
        trades = (_trade(0, 2, 100.0, 101.0, sig_pos=0), _trade(3, 5, 101.0, 99.0, sig_pos=3), _trade(6, 8, 99.0, 103.0, sig_pos=6))
        report = _metrics(trades, exposure_cap=1.0, initial_notional=10000.0, n_bars=10)
        assert report.values["trade_count"] == pytest.approx(3.0)
        assert report.values["trades_per_bar"] == pytest.approx(3.0 / 10.0)

    def test_transaction_sides_per_bar_is_exactly_double_trades_per_bar(self) -> None:
        """The identity this milestone's terminology fix depends on:
        every closed trade contributes exactly 2 transaction SIDES (one
        entry, one exit), so `transaction_sides_per_bar` is ALWAYS
        `2 * trades_per_bar` -- never the same number as `trades_per_bar`
        itself, which is exactly the confusion Section 7 corrects."""
        trades = (_trade(0, 2, 100.0, 101.0, sig_pos=0), _trade(3, 5, 101.0, 99.0, sig_pos=3), _trade(6, 8, 99.0, 103.0, sig_pos=6))
        report = _metrics(trades, exposure_cap=1.0, initial_notional=10000.0, n_bars=10)
        assert report.values["transaction_sides_per_bar"] == pytest.approx(2.0 * report.values["trades_per_bar"])
        assert report.values["transaction_sides_per_bar"] == pytest.approx(6.0 / 10.0)
        assert report.values["transaction_sides_per_bar"] != pytest.approx(report.values["trades_per_bar"])

    def test_trades_per_bar_is_reported_even_with_zero_closed_trades(self) -> None:
        """Unlike `transaction_sides_per_bar` (which lives inside the
        trade-dependent `if closed:` block and is SKIPPED with zero
        trades), `trades_per_bar` is a pure count-over-bars ratio,
        well-defined at zero, and reported unconditionally."""
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, direction=PositionDirection.FLAT, accepted=True),))
        report = _metrics((), exposure_cap=1.0, initial_notional=10000.0, n_bars=10, signals=signals)
        assert report.values["trade_count"] == pytest.approx(0.0)
        assert report.values["trades_per_bar"] == pytest.approx(0.0)
        assert "transaction_sides_per_bar" in report.skipped
        assert "transaction_sides_per_bar" not in report.values


class TestReversalRate:
    def test_reversal_rate_counts_direction_flips_between_chronologically_adjacent_trades(self) -> None:
        # LONG -> SHORT -> SHORT -> LONG: 3 adjacent pairs, 2 flips.
        trades = (
            _trade(0, 1, 100.0, 101.0, direction=PositionDirection.LONG, sig_pos=0),
            _trade(2, 3, 101.0, 100.0, direction=PositionDirection.SHORT, sig_pos=2),
            _trade(4, 5, 100.0, 99.0, direction=PositionDirection.SHORT, sig_pos=4),
            _trade(6, 7, 99.0, 100.0, direction=PositionDirection.LONG, sig_pos=6),
        )
        report = _metrics(trades, exposure_cap=1.0, initial_notional=10000.0, n_bars=10)
        assert report.values["reversal_rate"] == pytest.approx(2.0 / 3.0)

    def test_reversal_rate_is_zero_when_every_trade_shares_the_same_direction(self) -> None:
        trades = (
            _trade(0, 1, 100.0, 101.0, sig_pos=0), _trade(2, 3, 101.0, 102.0, sig_pos=2), _trade(4, 5, 102.0, 103.0, sig_pos=4),
        )
        report = _metrics(trades, exposure_cap=1.0, initial_notional=10000.0, n_bars=10)
        assert report.values["reversal_rate"] == pytest.approx(0.0)

    def test_reversal_rate_skipped_with_fewer_than_two_closed_trades(self) -> None:
        t1 = _trade(0, 1, 100.0, 101.0, sig_pos=0)
        report = _metrics((t1,), exposure_cap=1.0, initial_notional=10000.0, n_bars=10)
        assert "reversal_rate" in report.skipped
        assert "reversal_rate" not in report.values

        report_zero = _metrics((), exposure_cap=1.0, initial_notional=10000.0, n_bars=10, signals=SignalSet(
            schema_version=1, outer_fold_index=0, signals=(_signal(0, direction=PositionDirection.FLAT, accepted=True),),
        ))
        assert "reversal_rate" in report_zero.skipped
