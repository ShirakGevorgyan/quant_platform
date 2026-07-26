"""Tests for BrokerSimulator: fill pricing delegation and intrabar
SL/TP/time-stop resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.types import Bar, OrderSide
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.engine.broker_simulator import (
    SL_REASON,
    TIME_STOP_REASON,
    TP_REASON,
    BrokerSimulator,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bar(*, open_time: datetime, high: float, low: float, close: float, open_: float | None = None) -> Bar:
    return Bar(
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15),
        open=open_ if open_ is not None else close,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


@pytest.fixture
def cost_model() -> FixedSpreadCostModel:
    return FixedSpreadCostModel(spread_points=20.0, slippage_points=10.0, point_value=0.01, commission_per_unit=1.0)


@pytest.fixture
def broker(cost_model: FixedSpreadCostModel) -> BrokerSimulator:
    return BrokerSimulator(cost_model)


class TestFillEntry:
    def test_delegates_to_cost_model_for_price_and_commission(self, broker: BrokerSimulator) -> None:
        fill = broker.fill_entry(T0, OrderSide.BUY, quantity=2.0, reference_price=2000.0)
        assert fill.price == pytest.approx(2000.20)  # half spread + slippage
        assert fill.commission == pytest.approx(2.0)  # 2 units * $1
        assert fill.side is OrderSide.BUY
        assert fill.quantity == 2.0

    def test_rejects_non_positive_quantity(self, broker: BrokerSimulator) -> None:
        with pytest.raises(ValueError, match="quantity"):
            broker.fill_entry(T0, OrderSide.BUY, quantity=0.0, reference_price=2000.0)


class TestFillExit:
    def test_closing_a_long_fills_via_sell_side(self, broker: BrokerSimulator) -> None:
        fill = broker.fill_exit(T0, OrderSide.BUY, quantity=2.0, reference_price=2000.0)
        assert fill.side is OrderSide.SELL
        assert fill.price == pytest.approx(1999.90)

    def test_closing_a_short_fills_via_buy_side(self, broker: BrokerSimulator) -> None:
        fill = broker.fill_exit(T0, OrderSide.SELL, quantity=2.0, reference_price=2000.0)
        assert fill.side is OrderSide.BUY
        assert fill.price == pytest.approx(2000.10)

    def test_rejects_non_positive_quantity(self, broker: BrokerSimulator) -> None:
        with pytest.raises(ValueError, match="quantity"):
            broker.fill_exit(T0, OrderSide.BUY, quantity=-1.0, reference_price=2000.0)


class TestResolveIntrabarExitLong:
    def test_stop_loss_hit(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=101.0, low=94.0, close=95.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=95.0, take_profit=110.0,
        )
        assert result is not None
        assert result.exit_reason == SL_REASON
        assert result.exit_price == 95.0

    def test_take_profit_hit(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=111.0, low=99.0, close=110.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=90.0, take_profit=110.0,
        )
        assert result is not None
        assert result.exit_reason == TP_REASON

    def test_both_touched_sl_wins(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=111.0, low=89.0, close=100.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=90.0, take_profit=110.0,
        )
        assert result is not None
        assert result.exit_reason == SL_REASON

    def test_neither_touched_no_time_stop_stays_open(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=105.0, low=95.0, close=100.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=90.0, take_profit=110.0,
        )
        assert result is None

    def test_time_stop_triggers_when_max_hold_elapsed(self, broker: BrokerSimulator) -> None:
        entry_time = T0 - timedelta(minutes=30)
        bar = _bar(open_time=T0, high=105.0, low=95.0, close=100.0)  # close_time = T0+15min
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=entry_time,
            stop_loss=90.0, take_profit=110.0, max_hold=timedelta(minutes=30),
        )
        assert result is not None
        assert result.exit_reason == TIME_STOP_REASON
        assert result.exit_price == 100.0  # bar close

    def test_time_stop_does_not_trigger_before_elapsed(self, broker: BrokerSimulator) -> None:
        entry_time = T0 - timedelta(minutes=5)
        bar = _bar(open_time=T0, high=105.0, low=95.0, close=100.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=entry_time,
            stop_loss=90.0, take_profit=110.0, max_hold=timedelta(minutes=30),
        )
        assert result is None

    def test_no_stops_and_no_max_hold_never_exits(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=1000.0, low=0.01, close=100.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.BUY, bar=bar, entry_time=T0 - timedelta(days=365),
        )
        assert result is None


class TestResolveIntrabarExitShort:
    def test_stop_loss_hit(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=111.0, low=95.0, close=105.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.SELL, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=110.0, take_profit=90.0,
        )
        assert result is not None
        assert result.exit_reason == SL_REASON
        assert result.exit_price == 110.0

    def test_take_profit_hit(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=101.0, low=89.0, close=90.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.SELL, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=110.0, take_profit=90.0,
        )
        assert result is not None
        assert result.exit_reason == TP_REASON

    def test_both_touched_sl_wins(self, broker: BrokerSimulator) -> None:
        bar = _bar(open_time=T0, high=111.0, low=89.0, close=100.0)
        result = broker.resolve_intrabar_exit(
            position_side=OrderSide.SELL, bar=bar, entry_time=T0 - timedelta(minutes=15),
            stop_loss=110.0, take_profit=90.0,
        )
        assert result is not None
        assert result.exit_reason == SL_REASON
