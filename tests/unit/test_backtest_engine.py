"""Unit tests for BacktestEngine mechanics: entry/exit lifecycle, reversal
handling, zero-quantity skipping, end-of-data forced closure, and
multi-timeframe wiring -- using small, hand-verified scripted fixtures
rather than realistic market data, so expected behavior is checkable by
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import ConfigurationError, InsufficientDataError
from quant_platform.core.types import OHLCV_COLUMNS, Signal, SignalAction, Timeframe
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.engine.backtest_engine import BacktestEngine
from quant_platform.risk.position_sizing import (
    FixedFractionalSizer,
    PositionSizer,
    VolatilityTargetSizer,
)
from quant_platform.strategy.interfaces import Strategy, StrategyContext

UTC = timezone.utc
ZERO_COST = FixedSpreadCostModel(spread_points=0.0, slippage_points=0.0, point_value=1.0)


def _bars(rows: list[tuple[float, float, float, float]], timeframe: Timeframe = Timeframe.M15) -> pd.DataFrame:
    """rows are (open, high, low, close) tuples, one M15 bar apart."""
    n = len(rows)
    open_times = pd.date_range(start=datetime(2024, 1, 1, tzinfo=UTC), periods=n, freq="15min")
    opens, highs, lows, closes = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


@dataclass
class ScriptedStrategy(Strategy):
    """Returns one predetermined Signal per call, in order; repeats the
    final signal if called more times than scripted (defensive default)."""

    signals: list[Signal]
    _call_count: int = field(default=0, init=False)

    def on_bar(self, context: StrategyContext) -> Signal:
        index = min(self._call_count, len(self.signals) - 1)
        self._call_count += 1
        return self.signals[index]


@dataclass
class ZeroSizer(PositionSizer):
    """A PositionSizer test double that always sizes to exactly zero."""

    def __init__(self) -> None:
        super().__init__(max_position_fraction=1.0, on_limit_exceeded="clamp")

    def _raw_size(self, **_: object) -> float:
        return 0.0


def _hold(n: int) -> list[Signal]:
    return [Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD) for _ in range(n)]


class TestConstructionValidation:
    def test_rejects_missing_base_timeframe(self) -> None:
        data = {Timeframe.M15: _bars([(100, 101, 99, 100)] * 5)}
        with pytest.raises(ConfigurationError, match="base_timeframe"):
            BacktestEngine(
                data=data, base_timeframe=Timeframe.H1,
                strategy=ScriptedStrategy(_hold(5)), cost_model=ZERO_COST,
                position_sizer=FixedFractionalSizer(1.0), initial_capital=10_000.0,
            )

    def test_rejects_non_positive_max_window_bars(self) -> None:
        data = {Timeframe.M15: _bars([(100, 101, 99, 100)] * 5)}
        with pytest.raises(ValueError, match="max_window_bars"):
            BacktestEngine(
                data=data, base_timeframe=Timeframe.M15,
                strategy=ScriptedStrategy(_hold(5)), cost_model=ZERO_COST,
                position_sizer=FixedFractionalSizer(1.0), initial_capital=10_000.0,
                max_window_bars=0,
            )

    def test_rejects_empty_base_data(self) -> None:
        empty = pd.DataFrame(columns=list(OHLCV_COLUMNS))
        data = {Timeframe.M15: empty}
        engine = BacktestEngine(
            data=data, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy([]), cost_model=ZERO_COST,
            position_sizer=FixedFractionalSizer(1.0), initial_capital=10_000.0,
        )
        with pytest.raises(InsufficientDataError):
            engine.run()


class TestEntryAndStopLossExit:
    def test_long_entry_then_stop_loss_exit(self) -> None:
        bars = _bars(
            [
                (100, 101, 99, 100),   # bar0: HOLD
                (100, 101, 99, 100),   # bar1: LONG signal fires, entry at close=100
                (100, 100, 94, 96),    # bar2: low=94 breaches stop=95 -> SL exit at 95
                (96, 97, 95, 96),      # bar3: flat
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, stop_loss=95.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=ZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(100.0)
        assert trade.exit_price == pytest.approx(95.0)
        assert trade.exit_reason == "SL"
        # risk_amount=100 (1% of 10000), stop_distance=5 -> quantity=20
        assert trade.quantity == pytest.approx(20.0)
        assert trade.gross_pnl == pytest.approx(-100.0)  # (95-100)*20
        assert not trade.is_win

    def test_no_exit_check_on_the_entry_bar_itself(self) -> None:
        # bar1 (the entry bar) has a low of 90, which WOULD breach a
        # stop_loss=95 if checked -- but the entry bar must never be
        # checked for its own exit, since the fill only happens at its close.
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 90, 100),   # entry bar; low=90 must be ignored for exit purposes
                (100, 101, 99, 100),
                (100, 101, 99, 100),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, stop_loss=95.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=ZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()
        # Position survives to end of data (never breaches stop on bars 2/3) -> END_OF_DATA close.
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "END_OF_DATA"


class TestReversalAndEndOfData:
    def test_reversal_closes_existing_then_opens_opposite(self) -> None:
        bars = _bars(
            [
                (100, 101, 99, 100),   # bar0: LONG entry @ 100
                (100, 101, 99, 100),   # bar1: HOLD
                (110, 111, 109, 110),  # bar2: SHORT signal -> reversal (close long @110, open short @110)
                (110, 111, 109, 110),  # bar3: HOLD -> forced close at end of data
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, current_volatility=5.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.SHORT, current_volatility=5.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        # FixedFractionalSizer requires a stop_loss on every signal; these
        # signals deliberately don't provide one, so a volatility-based
        # sizer (which needs current_volatility instead) is used here.
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=ZERO_COST,
            position_sizer=VolatilityTargetSizer(target_volatility_percent=10.0, max_position_fraction=0.5),
            initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 2
        first, second = result.trades
        assert first.side.value == "BUY"
        assert first.exit_reason == "REVERSAL"
        assert first.entry_price == pytest.approx(100.0)
        assert first.exit_price == pytest.approx(110.0)
        assert first.is_win

        assert second.side.value == "SELL"
        assert second.exit_reason == "END_OF_DATA"
        assert second.entry_price == pytest.approx(110.0)
        assert second.exit_price == pytest.approx(110.0)  # last bar's close, unchanged


class TestZeroQuantitySkip:
    def test_zero_sized_signal_never_opens_a_position(self) -> None:
        bars = _bars([(100, 101, 99, 100)] * 3)
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, current_volatility=5.0),
            *_hold(2),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=ZERO_COST,
            position_sizer=ZeroSizer(), initial_capital=10_000.0,
        )
        result = engine.run()
        assert result.trades == []
        assert result.final_equity == pytest.approx(10_000.0)


class TestMultiTimeframeWiring:
    def test_higher_timeframe_window_is_populated_and_grows_correctly(self) -> None:
        # 8 M15 bars = exactly 2 H1 bars once both have closed.
        m15 = _bars([(100, 101, 99, 100)] * 8, timeframe=Timeframe.M15)
        h1_rows = [(100, 102, 98, 101), (101, 103, 99, 102)]
        h1_open_times = pd.date_range(start=datetime(2024, 1, 1, tzinfo=UTC), periods=2, freq="1h")
        opens, highs, lows, closes = zip(*h1_rows, strict=True)
        h1 = pd.DataFrame(
            {"open_time": h1_open_times, "open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [400.0] * 2}
        )

        seen_h1_lengths: list[int] = []

        @dataclass
        class ProbeStrategy(Strategy):
            def on_bar(self, context: StrategyContext) -> Signal:
                seen_h1_lengths.append(len(context.window(Timeframe.H1)))
                return Signal(timestamp=context.timestamp, action=SignalAction.HOLD)

        engine = BacktestEngine(
            data={Timeframe.M15: m15, Timeframe.H1: h1}, base_timeframe=Timeframe.M15,
            strategy=ProbeStrategy(), cost_model=ZERO_COST,
            position_sizer=FixedFractionalSizer(1.0), initial_capital=10_000.0,
        )
        engine.run()

        # First H1 bar (00:00-01:00) closes once the M15 clock reaches 01:00,
        # i.e. at M15 bar index 3 (bars are 0:00,0:15,0:30,0:45 -> close of
        # bar index 3 is 01:00). Bars 0-2 must see 0 H1 bars; bars 3-6 see 1;
        # bar 7 (closing at 02:00) sees 2.
        assert seen_h1_lengths == [0, 0, 0, 1, 1, 1, 1, 2]


class TestStopLossTakeProfitCostInteraction:
    """Golden-master regression tests for a real bug found during adversarial
    audit: `_open_position` used to re-anchor `signal.stop_loss`/`take_profit`
    to the (cost-adjusted) FILL price rather than using them as the absolute
    mid-price levels the strategy specified. That re-anchoring, combined with
    `resolve_intrabar_exit`'s result being routed back through `fill_exit`
    (which applies the exit-side spread again), double-counted the spread:
    every long/short stop-loss triggered on a smaller adverse move than
    requested (tighter by exactly half the spread), and every take-profit
    required a LARGER favorable move than requested (wider by half the
    spread) -- silently deviating from the strategy's stated risk on every
    trade that set stop_loss/take_profit, on every non-zero cost model.

    Every value below is computed independently by hand (see the comments),
    not derived from calling the production formula -- these numbers would
    have failed against the pre-fix implementation.
    """

    # spread=2 -> half_spread=1; slippage=1; point_value=1 (chosen so every
    # number is a clean integer and independently checkable by hand).
    NONZERO_COST = FixedSpreadCostModel(
        spread_points=2.0, slippage_points=1.0, point_value=1.0, commission_per_unit=0.0
    )

    def test_long_stop_loss_triggers_at_the_exact_intended_mid_price_level(self) -> None:
        # By hand: mid entry reference = 100. Entry fill (long pays the ask,
        # plus slippage) = 100 + half_spread(1) + slippage(1) = 102.
        # Strategy wants a stop 10 points below its OWN mid reference: stop=90.
        # Correct behavior: the stop must fire when the market MID reaches
        # 90, i.e. exactly the level the strategy specified -- NOT 91 (what
        # the pre-fix re-anchored formula would have produced: 102 - 10 = 92,
        # nor any other re-derived value).
        bars = _bars(
            [
                (100, 101, 99, 100),   # bar0: HOLD (warm-up)
                (100, 101, 99, 100),   # bar1: LONG signal fires, entry mid=100
                (95, 96, 85, 90),      # bar2: low=85 <= stop(90); open=95 > 90, so NOT a gap -> nominal-level fill
                (90, 91, 89, 90),      # bar3: flat (unreached if bar2 already exited)
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG,
                   stop_loss=90.0, take_profit=120.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]

        # By hand: risk_amount = 1% of 10,000 = 100. stop_distance (mid-based,
        # from the RAW signal, independent of any fill-price adjustment) =
        # |100 - 90| = 10. quantity = 100 / 10 = 10.
        assert trade.quantity == pytest.approx(10.0)
        assert trade.entry_price == pytest.approx(102.0)  # ask + slippage, by hand above
        # Exit: nominal stop level (90, the TRUE intended mid) sold at bid:
        # 90 - half_spread(1) = 89. NOT 91 (the pre-fix buggy value: had the
        # bug still been present, active_stop_loss would have been
        # 102 - 10 = 92, exit = 92 - 1 = 91).
        assert trade.exit_price == pytest.approx(89.0)
        assert trade.exit_reason == "SL"
        # gross_pnl = (89 - 102) * 10 = -130 -- a true 10-point adverse mid
        # move (100->90, exactly as the strategy specified) plus one full
        # round-trip spread (1+1=2) and one entry slippage (1), all *10 qty:
        # -(10 + 2 + 1) * 10 = -130. The pre-fix bug would have shown -110
        # for this SAME bar (understating the loss by a full 20).
        assert trade.gross_pnl == pytest.approx(-130.0)

    def test_long_take_profit_requires_the_exact_intended_mid_price_level(self) -> None:
        # Mirror case: intended take-profit at mid=120 (20 points above the
        # 100 mid entry). The bar's high reaches exactly 120 -- correct
        # behavior fires here; the pre-fix bug would have required mid=121
        # (an extra half-spread of favorable movement) and NOT fired on
        # this exact bar.
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),   # entry mid=100
                (115, 120, 114, 118),  # high=120 == take_profit -> fires under the fix
                (118, 119, 117, 118),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG,
                   stop_loss=90.0, take_profit=120.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "TP"
        # Exit: nominal TP level (120, the intended mid) sold at bid:
        # 120 - half_spread(1) = 119. Under the pre-fix bug this bar would
        # NOT have triggered TP at all (its re-anchored level would have
        # been 102 + 20 = 122, requiring high >= 122).
        assert trade.exit_price == pytest.approx(119.0)

    def test_short_stop_loss_triggers_at_the_exact_intended_mid_price_level(self) -> None:
        # Short entry: mid=100. Entry fill (short sells the bid, minus
        # slippage) = 100 - half_spread(1) - slippage(1) = 98. Strategy
        # stop 10 points above mid: stop=110 (a short's stop is above entry).
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),   # entry mid=100
                (105, 110, 104, 106),  # high=110 == stop_loss(110); open=105 < 110, not a gap
                (106, 107, 105, 106),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.SHORT,
                   stop_loss=110.0, take_profit=80.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(98.0)
        # Exit: nominal stop level (110, the true intended mid) bought back
        # at ask: 110 + half_spread(1) = 111. The pre-fix bug would have
        # re-anchored to 98 + 10 = 108, exiting at 108 + 1 = 109 instead --
        # again understating the loss (109 is a smaller adverse fill than 111).
        assert trade.exit_price == pytest.approx(111.0)
        assert trade.exit_reason == "SL"
        # gross_pnl for a short = (entry - exit) * quantity = (98-111)*10 = -130.
        assert trade.gross_pnl == pytest.approx(-130.0)


class TestGapThroughStopLoss:
    """Golden-master regression test for a second bug found during the same
    audit: a bar whose OPEN has already gapped past the stop-loss level used
    to fill at the exact nominal stop price regardless -- a price the market
    never actually traded at that bar. A real stop becomes a market order
    once triggered, so the realistic fill in a gap is the bar's open (the
    first price actually available), not the stale nominal level.
    """

    NONZERO_COST = FixedSpreadCostModel(
        spread_points=2.0, slippage_points=0.0, point_value=1.0, commission_per_unit=0.0
    )

    def test_long_stop_loss_fills_at_the_gapped_open_not_the_nominal_level(self) -> None:
        # Stop=90. The exit bar GAPS DOWN: it opens at 80, well below the
        # stop -- the market never traded at 90 during this bar. By hand,
        # the correct fill must use the worse (lower) of {bar.open, stop} =
        # min(80, 90) = 80, then the exit-side half-spread(1) applies:
        # 80 - 1 = 79. The old (pre-fix) behavior would have filled at
        # exactly 90 - 1 = 89, a materially better (less realistic) price.
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),   # entry mid=100
                (80, 81, 75, 78),      # GAP DOWN: opens at 80, already past stop=90
                (78, 79, 77, 78),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, stop_loss=90.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "SL"
        assert trade.exit_price == pytest.approx(79.0)

    def test_short_stop_loss_fills_at_the_gapped_open_not_the_nominal_level(self) -> None:
        # Mirror: short stop=110, bar GAPS UP opening at 125 (already past
        # the stop). Worse (higher) of {125, 110} = 125, plus exit-side
        # half-spread(1): 125 + 1 = 126.
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),
                (125, 130, 124, 128),  # GAP UP: opens at 125, already past stop=110
                (128, 129, 127, 128),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.SHORT, stop_loss=110.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "SL"
        assert trade.exit_price == pytest.approx(126.0)

    def test_no_gap_still_fills_at_nominal_level(self) -> None:
        # Control case: bar.open has NOT gapped past the stop (open=95 > 90),
        # only the low touches it intrabar -- fill must remain at the
        # nominal level (90), unaffected by the gap-through logic.
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),
                (95, 96, 85, 90),
                (90, 91, 89, 90),
            ]
        )
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, stop_loss=90.0),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.HOLD),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=self.NONZERO_COST,
            position_sizer=FixedFractionalSizer(risk_percent=1.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        # Nominal stop (90) minus exit-side half-spread(1) = 89.
        assert result.trades[0].exit_price == pytest.approx(89.0)


class TestEndOfDataSingleEquityPoint:
    """Golden-master regression test for a third bug found during the
    audit: forcing closure of a still-open position AFTER the loop's own
    final equity-point recording produced TWO points at the same timestamp
    (mark-to-market, then realized), distorting the return series analytics
    consume. Exactly one point per bar is the correct, checked-by-hand
    invariant regardless of whether the last bar forces a closure.
    """

    def test_exactly_one_equity_point_per_bar_when_forced_closed(self) -> None:
        bars = _bars([(100, 101, 99, 100)] * 4)
        signals = [
            Signal(timestamp=datetime(2024, 1, 1, tzinfo=UTC), action=SignalAction.LONG, current_volatility=5.0),
            *_hold(3),
        ]
        engine = BacktestEngine(
            data={Timeframe.M15: bars}, base_timeframe=Timeframe.M15,
            strategy=ScriptedStrategy(signals), cost_model=ZERO_COST,
            position_sizer=VolatilityTargetSizer(target_volatility_percent=10.0), initial_capital=10_000.0,
        )
        result = engine.run()

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "END_OF_DATA"
        # Hand count: 4 bars in the series -> exactly 4 equity points, never 5.
        assert len(result.equity_curve) == 4
        timestamps = [p.timestamp for p in result.equity_curve]
        assert len(timestamps) == len(set(timestamps)), "no duplicate-timestamp equity points"
