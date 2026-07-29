"""Milestone 7, Section 13: `PortfolioState` construction, the exact
`equity = cash + marked_position_value - liabilities - accrued_costs`
formula, and the reconciliation property `equity - starting_cash ==
realized_pnl + unrealized_pnl - accrued_costs + total_financing` that the
cash/accrued_costs convention is designed to hold exactly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import PortfolioReconciliationError
from quant_platform.paper_trading.fills import create_fill
from quant_platform.paper_trading.models import OrderSide, PartialFillPolicyKind
from quant_platform.paper_trading.portfolio import (
    PortfolioState,
    apply_fill_to_portfolio,
    apply_financing_to_portfolio,
    apply_mark_to_portfolio,
    apply_order_created_to_portfolio,
    apply_order_rejected_to_portfolio,
    initial_portfolio,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_EVENT = "a" * 64


def _fill(*, side: OrderSide, quantity: float, price: float, contract_multiplier: float = 1.0, spread_cost: float = 0.0, slippage_cost: float = 0.0, commission_cost: float = 0.0):
    return create_fill(
        order_id="order-1", session_id="session-1", instrument="X", side=side, quantity=quantity, price=price, contract_multiplier=contract_multiplier,
        spread_cost=spread_cost, slippage_cost=slippage_cost, commission_cost=commission_cost, execution_time=_T0, source_market_event_identity=_HEX_EVENT,
        liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=True,
    )


class TestInitialPortfolio:
    def test_starts_with_cash_equal_to_starting_cash(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        assert portfolio.cash == 100_000.0
        assert portfolio.equity == 100_000.0
        assert portfolio.peak_equity == 100_000.0

    def test_empty_positions(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        assert portfolio.positions == {}
        assert portfolio.gross_exposure == 0.0


class TestApplyFillToPortfolio:
    def test_buy_decreases_cash_by_gross_notional(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        assert portfolio.cash == pytest.approx(100_000.0 - 1000.0)

    def test_sell_to_open_short_increases_cash_by_gross_notional(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        assert portfolio.cash == pytest.approx(100_000.0 + 1000.0)

    def test_equity_unchanged_immediately_after_opening_at_entry_price(self) -> None:
        """Opening a position (long or short) at exactly the current
        market price, with zero cost, must not change equity at all --
        cash and marked_position_value move by equal and opposite
        amounts."""
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        assert portfolio.equity == pytest.approx(100_000.0)

    def test_fill_count_increments(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=1.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        assert portfolio.fill_count == 1

    def test_turnover_accumulates_gross_notional(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.SELL, quantity=10.0, price=110.0), event_time=_T0, contract_multiplier=1.0)
        assert portfolio.turnover == pytest.approx(1000.0 + 1100.0)


class TestOrderAndRejectionCounts:
    def test_order_created_increments_order_count(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_order_created_to_portfolio(portfolio, event_time=_T0)
        assert portfolio.order_count == 1

    def test_order_rejected_increments_rejected_count(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_order_rejected_to_portfolio(portfolio, event_time=_T0)
        assert portfolio.rejected_order_count == 1


class TestApplyMarkToPortfolio:
    def test_mark_updates_unrealized_pnl_and_equity(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=110.0, event_time=_T0)
        assert portfolio.unrealized_pnl == pytest.approx(100.0)
        assert portfolio.equity == pytest.approx(100_100.0)

    def test_mark_with_no_existing_position_is_a_no_op(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        marked = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=100.0, event_time=_T0)
        assert marked.positions == {}

    def test_peak_equity_tracks_the_maximum(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=120.0, event_time=_T0)
        assert portfolio.peak_equity == pytest.approx(100_200.0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=105.0, event_time=_T0)
        assert portfolio.equity == pytest.approx(100_050.0)
        assert portfolio.peak_equity == pytest.approx(100_200.0)

    def test_drawdown_fraction_computed_from_peak(self) -> None:
        # cash after buy 100@100 = 100,000 - 10,000 = 90,000
        # peak equity at mark=120: cash(90,000) + 100*120 = 102,000
        # equity at mark=108: cash(90,000) + 100*108 = 100,800
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=100.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=120.0, event_time=_T0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=108.0, event_time=_T0)
        expected_drawdown = (102_000.0 - 100_800.0) / 102_000.0
        assert portfolio.drawdown_fraction == pytest.approx(expected_drawdown)


class TestApplyFinancingToPortfolio:
    def test_financing_cost_decreases_cash_and_equity(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        equity_before = portfolio.equity
        portfolio = apply_financing_to_portfolio(portfolio, instrument="X", cash_delta=-10.0, event_time=_T0 + timedelta(days=1))
        assert portfolio.cash == pytest.approx(100_000.0 - 1000.0 - 10.0)
        assert portfolio.equity == pytest.approx(equity_before - 10.0)

    def test_financing_with_no_existing_position_rejected(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        with pytest.raises(PortfolioReconciliationError, match="no position"):
            apply_financing_to_portfolio(portfolio, instrument="X", cash_delta=-1.0, event_time=_T0)


class TestExactReconciliation:
    """The property the cash/accrued_costs design is built around:
    `equity - starting_cash == realized_pnl + unrealized_pnl - accrued_
    costs + total_financing`, holding EXACTLY (not approximately by some
    tolerance chosen to hide a bug)."""

    def test_reconciliation_holds_after_round_trip_with_costs_and_financing(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(
            portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0, spread_cost=0.5, slippage_cost=0.25, commission_cost=1.0),
            event_time=_T0, contract_multiplier=1.0,
        )
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=105.0, event_time=_T0)
        portfolio = apply_financing_to_portfolio(portfolio, instrument="X", cash_delta=-3.0, event_time=_T0 + timedelta(days=1))
        portfolio = apply_fill_to_portfolio(
            portfolio, _fill(side=OrderSide.SELL, quantity=10.0, price=108.0, spread_cost=0.5, slippage_cost=0.25, commission_cost=1.0),
            event_time=_T0, contract_multiplier=1.0,
        )
        lhs = portfolio.equity - portfolio.starting_cash
        rhs = portfolio.realized_pnl + portfolio.unrealized_pnl - portfolio.accrued_costs + portfolio.total_financing
        assert lhs == pytest.approx(rhs, abs=1e-9)

    def test_reconciliation_holds_for_short_position_with_costs(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=50_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.SELL, quantity=5.0, price=200.0, commission_cost=2.0), event_time=_T0, contract_multiplier=1.0)
        portfolio = apply_mark_to_portfolio(portfolio, instrument="X", mark_price=190.0, event_time=_T0)
        lhs = portfolio.equity - portfolio.starting_cash
        rhs = portfolio.realized_pnl + portfolio.unrealized_pnl - portfolio.accrued_costs + portfolio.total_financing
        assert lhs == pytest.approx(rhs, abs=1e-9)


class TestToStrategySnapshot:
    def test_snapshot_reflects_current_position_and_account_state(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        snapshot = portfolio.to_strategy_snapshot("X")
        assert snapshot.signed_quantity == 10.0
        assert snapshot.average_entry_price == 100.0
        assert snapshot.cash == portfolio.cash
        assert snapshot.equity == portfolio.equity

    def test_snapshot_for_untraded_instrument_is_flat(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        snapshot = portfolio.to_strategy_snapshot("X")
        assert snapshot.signed_quantity == 0.0
        assert snapshot.average_entry_price is None


class TestPortfolioStateValidation:
    def test_negative_starting_cash_rejected(self) -> None:
        with pytest.raises(PortfolioReconciliationError, match="starting_cash"):
            initial_portfolio("session-1", starting_cash=-1.0)

    def test_positions_key_mismatch_rejected(self) -> None:
        from quant_platform.paper_trading.accounting import flat_position

        mismatched = flat_position("OTHER", contract_multiplier=1.0)
        with pytest.raises(PortfolioReconciliationError, match="positions key"):
            PortfolioState(
                session_id="s1", starting_cash=1000.0, cash=1000.0, positions={"X": mismatched}, order_count=0, fill_count=0,
                rejected_order_count=0, turnover=0.0, peak_equity=1000.0, last_event_time=None, portfolio_version=0,
            )

    def test_json_round_trip(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        portfolio = apply_fill_to_portfolio(portfolio, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0, contract_multiplier=1.0)
        roundtripped = PortfolioState.from_json_dict(portfolio.to_json_dict())
        assert roundtripped == portfolio
