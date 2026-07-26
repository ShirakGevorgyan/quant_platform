"""Tests for single-symbol Portfolio state management."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import EngineError
from quant_platform.core.types import OrderSide
from quant_platform.engine.portfolio import Portfolio

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)


class TestConstruction:
    def test_starts_flat_with_full_cash(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        assert portfolio.cash == 10_000.0
        assert portfolio.position is None
        assert not portfolio.has_open_position()
        assert portfolio.closed_trades == []
        assert portfolio.equity_curve == []

    def test_rejects_non_positive_initial_capital(self) -> None:
        with pytest.raises(ValueError, match="initial_capital"):
            Portfolio(initial_capital=0.0)

    def test_rejects_non_positive_point_value(self) -> None:
        with pytest.raises(ValueError, match="point_value"):
            Portfolio(initial_capital=10_000.0, point_value=0.0)


class TestOpenPosition:
    def test_open_long_sets_signed_positive_quantity(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, quantity=2.0, fill_price=100.0, commission=1.0, timestamp=T0)
        assert portfolio.has_open_position()
        assert portfolio.position.quantity == 2.0
        assert portfolio.position.average_entry_price == 100.0
        assert portfolio.cash == 10_000.0 - 1.0

    def test_open_short_sets_signed_negative_quantity(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.open_position("EURUSD", OrderSide.SELL, quantity=2.0, fill_price=100.0, commission=1.0, timestamp=T0)
        assert portfolio.position.quantity == -2.0

    def test_rejects_opening_when_already_open(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, 1.0, 100.0, 1.0, T0)
        with pytest.raises(EngineError, match="already open"):
            portfolio.open_position("EURUSD", OrderSide.BUY, 1.0, 101.0, 1.0, T0)

    def test_rejects_non_positive_quantity(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        with pytest.raises(ValueError, match="quantity"):
            portfolio.open_position("EURUSD", OrderSide.BUY, 0.0, 100.0, 1.0, T0)


class TestClosePosition:
    def test_closing_a_winning_long_increases_cash(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0, point_value=1.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, quantity=10.0, fill_price=100.0, commission=1.0, timestamp=T0)
        trade = portfolio.close_position(exit_price=105.0, commission=1.0, timestamp=T1, exit_reason="TP")

        assert trade.gross_pnl == pytest.approx(50.0)  # (105-100)*10
        assert trade.net_pnl == pytest.approx(48.0)  # minus entry commission (1) and exit commission (1)
        assert trade.is_win
        # cash = 10000 - entry_commission(1) + gross_pnl(50) - exit_commission(1)
        assert portfolio.cash == pytest.approx(10_048.0)
        assert not portfolio.has_open_position()

    def test_closing_a_losing_short_decreases_cash(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0, point_value=1.0)
        portfolio.open_position("EURUSD", OrderSide.SELL, quantity=10.0, fill_price=100.0, commission=1.0, timestamp=T0)
        # price rose against the short
        trade = portfolio.close_position(exit_price=105.0, commission=1.0, timestamp=T1, exit_reason="SL")

        assert trade.gross_pnl == pytest.approx(-50.0)  # (105-100)*(-10)
        assert not trade.is_win
        assert portfolio.cash == pytest.approx(10_000.0 - 1.0 - 50.0 - 1.0)

    def test_point_value_scales_pnl(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0, point_value=100.0)
        portfolio.open_position("XAUUSD", OrderSide.BUY, quantity=1.0, fill_price=2000.0, commission=0.0, timestamp=T0)
        trade = portfolio.close_position(exit_price=2010.0, commission=0.0, timestamp=T1, exit_reason="TP")
        assert trade.gross_pnl == pytest.approx(10.0 * 100.0)

    def test_records_entry_and_exit_time_and_side(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, 1.0, 100.0, 0.0, T0)
        trade = portfolio.close_position(105.0, 0.0, T1, "TP")
        assert trade.entry_time == T0
        assert trade.exit_time == T1
        assert trade.side is OrderSide.BUY
        assert trade.holding_period == T1 - T0

    def test_rejects_closing_when_flat(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        with pytest.raises(EngineError, match="No open position"):
            portfolio.close_position(100.0, 0.0, T0, "TP")

    def test_closed_trade_is_appended_to_history(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, 1.0, 100.0, 0.0, T0)
        portfolio.close_position(105.0, 0.0, T1, "TP")
        assert len(portfolio.closed_trades) == 1

    def test_trade_net_pnl_reconciles_exactly_with_cash_change(self) -> None:
        """Regression guard: total_cost must include BOTH entry and exit
        commission, or the trade log silently understates cost relative to
        the actual change in account cash."""
        portfolio = Portfolio(initial_capital=10_000.0, point_value=1.0)
        starting_cash = portfolio.cash

        portfolio.open_position("EURUSD", OrderSide.BUY, quantity=10.0, fill_price=100.0, commission=2.5, timestamp=T0)
        trade = portfolio.close_position(exit_price=103.0, commission=3.5, timestamp=T1, exit_reason="TP")

        assert trade.total_cost == pytest.approx(2.5 + 3.5)
        assert portfolio.cash == pytest.approx(starting_cash + trade.net_pnl)


class TestEquityAndDrawdown:
    def test_equity_is_cash_when_flat(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        assert portfolio.equity() == 10_000.0
        assert portfolio.equity(current_price=999.0) == 10_000.0  # ignored while flat

    def test_equity_includes_unrealized_pnl_when_open(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0, point_value=1.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, 10.0, 100.0, 0.0, T0)
        assert portfolio.equity(current_price=103.0) == pytest.approx(10_030.0)

    def test_record_equity_point_tracks_peak_and_drawdown(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0, point_value=1.0)
        portfolio.open_position("EURUSD", OrderSide.BUY, 10.0, 100.0, 0.0, T0)

        p1 = portfolio.record_equity_point(T0, current_price=110.0)  # equity=10100, new peak
        assert p1.equity == pytest.approx(10_100.0)
        assert p1.drawdown_pct == pytest.approx(0.0)

        p2 = portfolio.record_equity_point(T1, current_price=90.0)  # equity=9900, drawdown from 10100
        assert p2.equity == pytest.approx(9_900.0)
        expected_dd = (10_100.0 - 9_900.0) / 10_100.0 * 100.0
        assert p2.drawdown_pct == pytest.approx(expected_dd)

    def test_equity_curve_accumulates_points_in_order(self) -> None:
        portfolio = Portfolio(initial_capital=10_000.0)
        portfolio.record_equity_point(T0)
        portfolio.record_equity_point(T1)
        assert [p.timestamp for p in portfolio.equity_curve] == [T0, T1]
