"""Unit tests for `portfolio_risk.valuation`: position/portfolio
post-trade projection across every required scenario."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.portfolio_risk.models import OrderSide
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    PositionSnapshot,
    create_portfolio_snapshot,
    create_price_snapshot,
)
from quant_platform.portfolio_risk.valuation import (
    TradeRiskClassification,
    classify_trade_risk,
    project_fill_price,
    project_portfolio,
    project_position,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _price(**overrides: object):
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "bid": Decimal("1.0995"), "ask": Decimal("1.1005"), "reference_price": Decimal("1.10"), "event_time": _T0,
        "source_event_id": "e1",
    }
    base.update(overrides)
    return create_price_snapshot(**base)  # type: ignore[arg-type]


def _position(**overrides: object) -> PositionSnapshot:
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "strategy_id": "s1", "side": OrderSide.BUY, "quantity": Decimal("1000"), "average_entry_price": Decimal("1.10"),
        "mark_price": Decimal("1.10"), "unrealized_pnl": Decimal("0"), "realized_pnl": Decimal("0"), "contract_multiplier": Decimal("1"),
    }
    base.update(overrides)
    return PositionSnapshot(**base)  # type: ignore[arg-type]


def _portfolio(*, positions: tuple[PositionSnapshot, ...], cash: Decimal = Decimal("100000")) -> PortfolioSnapshot:
    marked_value = sum((p.market_value for p in positions), start=Decimal(0))
    unrealized = sum((p.unrealized_pnl for p in positions), start=Decimal(0))
    return create_portfolio_snapshot(
        portfolio_id="p1", event_time=_T0, cash=cash, equity=cash + marked_value, realized_pnl=Decimal("0"), unrealized_pnl=unrealized,
        peak_equity=cash + marked_value, daily_start_equity=cash, positions=positions, source_execution_session_id=None,
    )


class TestProjectFillPrice:
    def test_buy_fills_at_ask(self) -> None:
        assert project_fill_price(_price(), side=OrderSide.BUY) == Decimal("1.1005")

    def test_sell_fills_at_bid(self) -> None:
        assert project_fill_price(_price(), side=OrderSide.SELL) == Decimal("1.0995")


class TestClassifyTradeRisk:
    def test_open_from_flat_is_increasing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(0), projected_signed_quantity=Decimal(1000)) is TradeRiskClassification.INCREASING

    def test_add_same_direction_is_increasing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(1000), projected_signed_quantity=Decimal(1500)) is TradeRiskClassification.INCREASING

    def test_partial_reduce_is_reducing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(1000), projected_signed_quantity=Decimal(500)) is TradeRiskClassification.REDUCING

    def test_full_close_is_reducing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(1000), projected_signed_quantity=Decimal(0)) is TradeRiskClassification.REDUCING

    def test_cross_long_to_short_is_increasing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(1000), projected_signed_quantity=Decimal(-500)) is TradeRiskClassification.INCREASING

    def test_cross_short_to_long_is_increasing(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(-1000), projected_signed_quantity=Decimal(500)) is TradeRiskClassification.INCREASING

    def test_unchanged_is_neutral(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(1000), projected_signed_quantity=Decimal(1000)) is TradeRiskClassification.NEUTRAL

    def test_flat_to_flat_is_neutral(self) -> None:
        assert classify_trade_risk(current_signed_quantity=Decimal(0), projected_signed_quantity=Decimal(0)) is TradeRiskClassification.NEUTRAL


class TestProjectPositionOpenLong:
    def test_opens_from_flat(self) -> None:
        result = project_position(None, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), fill_price=Decimal("1.10"), mark_price=Decimal("1.10"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING
        assert result.new_position is not None
        assert result.new_position.side is OrderSide.BUY
        assert result.new_position.quantity == Decimal("1000")
        assert result.new_position.average_entry_price == Decimal("1.10")
        assert result.realized_pnl_delta == Decimal("0")


class TestProjectPositionOpenShort:
    def test_opens_from_flat(self) -> None:
        result = project_position(None, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1000"), fill_price=Decimal("1.10"), mark_price=Decimal("1.10"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING
        assert result.new_position is not None
        assert result.new_position.side is OrderSide.SELL
        assert result.new_position.quantity == Decimal("1000")


class TestProjectPositionAdd:
    def test_add_same_direction_weighted_averages(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), fill_price=Decimal("1.20"), mark_price=Decimal("1.20"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING
        assert result.new_position.quantity == Decimal("2000")
        assert result.new_position.average_entry_price == Decimal("1.15")
        assert result.realized_pnl_delta == Decimal("0")


class TestProjectPositionReduce:
    def test_partial_reduce_realizes_pnl_keeps_average_price(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("400"), fill_price=Decimal("1.20"), mark_price=Decimal("1.20"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.REDUCING
        assert result.new_position.quantity == Decimal("600")
        assert result.new_position.average_entry_price == Decimal("1.10")
        assert result.realized_pnl_delta == Decimal("400") * (Decimal("1.20") - Decimal("1.10"))


class TestProjectPositionClose:
    def test_full_close_returns_none_position_but_reports_realized_delta(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1000"), fill_price=Decimal("1.25"), mark_price=Decimal("1.25"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.REDUCING
        assert result.new_position is None
        assert result.realized_pnl_delta == Decimal("1000") * (Decimal("1.25") - Decimal("1.10"))

    def test_over_closing_quantity_caps_realization_at_current_size(self) -> None:
        # Requesting MORE than the current position holds crosses through
        # zero -- covered separately by the crossing tests; this test
        # confirms closing_quantity is capped at abs(current_signed) for
        # the realized-pnl calculation portion of a reversal.
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1500"), fill_price=Decimal("1.20"), mark_price=Decimal("1.20"), contract_multiplier=Decimal("1"))
        assert result.realized_pnl_delta == Decimal("1000") * (Decimal("1.20") - Decimal("1.10"))


class TestProjectPositionCrossing:
    def test_long_to_short_realizes_and_opens_fresh_at_fill_price(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1500"), fill_price=Decimal("1.20"), mark_price=Decimal("1.20"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING
        assert result.new_position.side is OrderSide.SELL
        assert result.new_position.quantity == Decimal("500")
        assert result.new_position.average_entry_price == Decimal("1.20")

    def test_short_to_long_realizes_and_opens_fresh_at_fill_price(self) -> None:
        current = _position(side=OrderSide.SELL, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1500"), fill_price=Decimal("1.05"), mark_price=Decimal("1.05"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING
        assert result.new_position.side is OrderSide.BUY
        assert result.new_position.quantity == Decimal("500")
        assert result.new_position.average_entry_price == Decimal("1.05")
        # short reduced by 1000 units, profit when exit BELOW entry: 1000 * (1.10 - 1.05)
        assert result.realized_pnl_delta == Decimal("1000") * (Decimal("1.10") - Decimal("1.05"))


class TestProjectPositionMismatchedIdentity:
    def test_rejects_current_position_with_different_instrument(self) -> None:
        from quant_platform.core.exceptions import ExposureCalculationError

        current = _position(instrument_id="GBPUSD")
        with pytest.raises(ExposureCalculationError):
            project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("100"), fill_price=Decimal("1.1"), mark_price=Decimal("1.1"), contract_multiplier=Decimal("1"))


class TestProjectPortfolioDoesNotMutateOriginal:
    def test_original_snapshot_unchanged_after_projection(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        portfolio = _portfolio(positions=(current,))
        original_json = portfolio.to_json_dict()
        project_portfolio(portfolio, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("500"), price=_price(), contract_multiplier=Decimal("1"), evaluation_time=_T0)
        assert portfolio.to_json_dict() == original_json

    def test_projected_portfolio_is_a_different_object_with_valid_invariants(self) -> None:
        portfolio = _portfolio(positions=())
        result = project_portfolio(portfolio, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), price=_price(), contract_multiplier=Decimal("1"), evaluation_time=_T0)
        assert result.portfolio is not portfolio
        assert result.portfolio.equity == result.portfolio.cash + sum(p.market_value for p in result.portfolio.positions)


class TestReduceOnlySemantics:
    def test_reduce_only_valid_reduction_is_reducing(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("500"), fill_price=Decimal("1.10"), mark_price=Decimal("1.10"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.REDUCING

    def test_reduce_only_invalid_increase_is_increasing(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("500"), fill_price=Decimal("1.10"), mark_price=Decimal("1.10"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING

    def test_reduce_only_crossing_through_zero_is_increasing_never_reducing(self) -> None:
        current = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.10"))
        result = project_position(current, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("2000"), fill_price=Decimal("1.10"), mark_price=Decimal("1.10"), contract_multiplier=Decimal("1"))
        assert result.classification is TradeRiskClassification.INCREASING


class TestContractMultiplierCorrectness:
    def test_cash_delta_and_position_value_scale_with_multiplier(self) -> None:
        portfolio = _portfolio(positions=())
        result = project_portfolio(portfolio, instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("10"), price=_price(reference_price=Decimal("1.10")), contract_multiplier=Decimal("100"), evaluation_time=_T0)
        position = result.portfolio.position_for(instrument_id="EURUSD", strategy_id="s1")
        assert position is not None
        assert position.contract_multiplier == Decimal("100")
        expected_cash = Decimal("100000") - Decimal("10") * Decimal("1.1005") * Decimal("100")
        assert result.portfolio.cash == expected_cash
