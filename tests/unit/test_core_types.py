"""Direct unit tests for core domain types: every dataclass invariant and
enum behavior should be verifiable in isolation, independent of any
higher-level component that happens to exercise it incidentally."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import pairwise

import pytest

from quant_platform.core.types import (
    Bar,
    EquityPoint,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    Signal,
    SignalAction,
    Timeframe,
    Trade,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 1, 0, 15, tzinfo=UTC)


class TestTimeframe:
    def test_durations_are_correct(self) -> None:
        assert Timeframe.M1.duration == timedelta(minutes=1)
        assert Timeframe.M5.duration == timedelta(minutes=5)
        assert Timeframe.M15.duration == timedelta(minutes=15)
        assert Timeframe.M30.duration == timedelta(minutes=30)
        assert Timeframe.H1.duration == timedelta(hours=1)
        assert Timeframe.H4.duration == timedelta(hours=4)
        assert Timeframe.D1.duration == timedelta(days=1)

    def test_minutes_matches_duration(self) -> None:
        assert Timeframe.H1.minutes == 60.0
        assert Timeframe.M15.minutes == 15.0

    def test_ordering_across_all_timeframes(self) -> None:
        ordered = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        for smaller, larger in pairwise(ordered):
            assert smaller < larger
            assert smaller <= larger
            assert not (larger < smaller)

    def test_le_is_true_for_equal_timeframes(self) -> None:
        assert Timeframe.M15 <= Timeframe.M15

    def test_comparison_with_non_timeframe_is_not_implemented(self) -> None:
        assert Timeframe.M15.__lt__("not a timeframe") is NotImplemented
        assert Timeframe.M15.__le__("not a timeframe") is NotImplemented


class TestBar:
    def test_valid_bar_constructs(self) -> None:
        bar = Bar(open_time=T0, close_time=T1, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0)
        assert bar.open == 100.0

    def test_rejects_high_less_than_low(self) -> None:
        with pytest.raises(ValueError, match=r"high .* < low"):
            Bar(open_time=T0, close_time=T1, open=100.0, high=98.0, low=99.0, close=100.0, volume=10.0)

    def test_rejects_open_outside_range(self) -> None:
        with pytest.raises(ValueError, match=r"open .* outside"):
            Bar(open_time=T0, close_time=T1, open=200.0, high=101.0, low=99.0, close=100.0, volume=10.0)

    def test_rejects_close_outside_range(self) -> None:
        with pytest.raises(ValueError, match=r"close .* outside"):
            Bar(open_time=T0, close_time=T1, open=100.0, high=101.0, low=99.0, close=200.0, volume=10.0)

    def test_rejects_close_time_not_after_open_time(self) -> None:
        with pytest.raises(ValueError, match=r"close_time .* must be after"):
            Bar(open_time=T0, close_time=T0, open=100.0, high=101.0, low=99.0, close=100.0, volume=10.0)


class TestOrderSide:
    def test_buy_sign_is_positive(self) -> None:
        assert OrderSide.BUY.sign == 1

    def test_sell_sign_is_negative(self) -> None:
        assert OrderSide.SELL.sign == -1


class TestSignal:
    def test_valid_signal_constructs_with_defaults(self) -> None:
        signal = Signal(timestamp=T0, action=SignalAction.LONG)
        assert signal.confidence == 1.0
        assert signal.stop_loss is None
        assert signal.metadata == {}

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_rejects_confidence_out_of_range(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Signal(timestamp=T0, action=SignalAction.LONG, confidence=confidence)

    def test_rejects_equal_stop_loss_and_take_profit(self) -> None:
        with pytest.raises(ValueError, match="must not be equal"):
            Signal(timestamp=T0, action=SignalAction.LONG, stop_loss=100.0, take_profit=100.0)

    def test_allows_only_one_of_stop_loss_take_profit(self) -> None:
        signal = Signal(timestamp=T0, action=SignalAction.LONG, stop_loss=95.0)
        assert signal.stop_loss == 95.0
        assert signal.take_profit is None


class TestOrder:
    def test_valid_market_order_constructs(self) -> None:
        order = Order(timestamp=T0, side=OrderSide.BUY, quantity=1.0)
        assert order.order_type is OrderType.MARKET

    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            Order(timestamp=T0, side=OrderSide.BUY, quantity=0.0)

    def test_limit_order_requires_limit_price(self) -> None:
        with pytest.raises(ValueError, match="LIMIT orders require"):
            Order(timestamp=T0, side=OrderSide.BUY, quantity=1.0, order_type=OrderType.LIMIT)

    def test_limit_order_with_price_constructs(self) -> None:
        order = Order(timestamp=T0, side=OrderSide.BUY, quantity=1.0, order_type=OrderType.LIMIT, limit_price=100.0)
        assert order.limit_price == 100.0

    def test_stop_order_requires_stop_price(self) -> None:
        with pytest.raises(ValueError, match="STOP orders require"):
            Order(timestamp=T0, side=OrderSide.BUY, quantity=1.0, order_type=OrderType.STOP)

    def test_stop_order_with_price_constructs(self) -> None:
        order = Order(timestamp=T0, side=OrderSide.BUY, quantity=1.0, order_type=OrderType.STOP, stop_price=105.0)
        assert order.stop_price == 105.0


class TestFill:
    def test_constructs_with_expected_fields(self) -> None:
        fill = Fill(timestamp=T0, side=OrderSide.BUY, quantity=2.0, price=100.5, commission=1.0)
        assert fill.price == 100.5
        assert fill.commission == 1.0


class TestPosition:
    def test_default_position_is_flat(self) -> None:
        position = Position(symbol="EURUSD")
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short

    def test_positive_quantity_is_long(self) -> None:
        position = Position(symbol="EURUSD", quantity=5.0)
        assert position.is_long
        assert not position.is_short
        assert not position.is_flat

    def test_negative_quantity_is_short(self) -> None:
        position = Position(symbol="EURUSD", quantity=-5.0)
        assert position.is_short
        assert not position.is_long

    def test_unrealized_pnl_is_zero_when_flat(self) -> None:
        position = Position(symbol="EURUSD")
        assert position.unrealized_pnl(current_price=999.0) == 0.0

    def test_unrealized_pnl_for_long_position(self) -> None:
        position = Position(symbol="EURUSD", quantity=10.0, average_entry_price=100.0)
        assert position.unrealized_pnl(current_price=105.0) == pytest.approx(50.0)

    def test_unrealized_pnl_scales_with_point_value(self) -> None:
        position = Position(symbol="XAUUSD", quantity=1.0, average_entry_price=2000.0)
        assert position.unrealized_pnl(current_price=2010.0, point_value=100.0) == pytest.approx(1000.0)


class TestEquityPoint:
    def test_constructs_with_expected_fields(self) -> None:
        point = EquityPoint(timestamp=T0, cash=1000.0, equity=1050.0, drawdown_pct=2.5)
        assert point.equity == 1050.0


class TestTrade:
    def _trade(self, gross_pnl: float, total_cost: float = 0.0) -> Trade:
        return Trade(
            entry_time=T0, exit_time=T1, side=OrderSide.BUY, entry_price=100.0, exit_price=105.0,
            quantity=1.0, gross_pnl=gross_pnl, total_cost=total_cost, exit_reason="TP",
        )

    def test_net_pnl_subtracts_total_cost(self) -> None:
        trade = self._trade(gross_pnl=50.0, total_cost=5.0)
        assert trade.net_pnl == pytest.approx(45.0)

    def test_is_win_true_for_positive_net_pnl(self) -> None:
        assert self._trade(gross_pnl=10.0).is_win

    def test_is_win_false_for_zero_or_negative_net_pnl(self) -> None:
        assert not self._trade(gross_pnl=0.0).is_win
        assert not self._trade(gross_pnl=-10.0).is_win

    def test_holding_period_is_exit_minus_entry(self) -> None:
        trade = self._trade(gross_pnl=10.0)
        assert trade.holding_period == T1 - T0
