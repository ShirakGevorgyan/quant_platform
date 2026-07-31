"""Unit tests for `portfolio_risk.exposure`: long/short/mixed portfolios,
concentration, leverage, contract-multiplier correctness, strategy
aggregation, and fail-closed behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExposureCalculationError
from quant_platform.portfolio_risk.exposure import (
    compute_available_cash,
    compute_concentration_fraction,
    compute_daily_loss,
    compute_leverage,
    compute_long_gross_exposure,
    compute_portfolio_exposure,
    compute_short_gross_exposure,
    compute_total_loss,
)
from quant_platform.portfolio_risk.models import OrderSide
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    PositionSnapshot,
    create_portfolio_snapshot,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _position(**overrides: object) -> PositionSnapshot:
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "strategy_id": "s1", "side": OrderSide.BUY, "quantity": Decimal("1000"),
        "average_entry_price": Decimal("1.10"), "mark_price": Decimal("1.10"), "unrealized_pnl": Decimal("0"), "realized_pnl": Decimal("0"),
        "contract_multiplier": Decimal("1"),
    }
    base.update(overrides)
    return PositionSnapshot(**base)  # type: ignore[arg-type]


def _portfolio(*, positions: tuple[PositionSnapshot, ...], cash: Decimal = Decimal("100000"), **overrides: object) -> PortfolioSnapshot:
    marked_value = sum((p.market_value for p in positions), start=Decimal(0))
    unrealized = sum((p.unrealized_pnl for p in positions), start=Decimal(0))
    base: dict[str, object] = {
        "portfolio_id": "p1", "event_time": _T0, "cash": cash, "equity": cash + marked_value, "realized_pnl": Decimal("0"),
        "unrealized_pnl": unrealized, "peak_equity": cash + marked_value, "daily_start_equity": cash, "positions": positions,
        "source_execution_session_id": None,
    }
    base.update(overrides)
    return create_portfolio_snapshot(**base)  # type: ignore[arg-type]


class TestLongOnlyPortfolio:
    def test_gross_and_net_are_equal_and_positive(self) -> None:
        p = _position(side=OrderSide.BUY, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(p,))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure == exposure.net_exposure == Decimal("1100")

    def test_long_gross_equals_total_short_gross_is_zero(self) -> None:
        p = _position()
        portfolio = _portfolio(positions=(p,))
        assert compute_long_gross_exposure(portfolio) == Decimal("1100")
        assert compute_short_gross_exposure(portfolio) == Decimal("0")


class TestShortOnlyPortfolio:
    def test_gross_positive_net_negative(self) -> None:
        p = _position(side=OrderSide.SELL, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(p,))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure == Decimal("1100")
        assert exposure.net_exposure == Decimal("-1100")

    def test_short_gross_equals_total_long_gross_is_zero(self) -> None:
        p = _position(side=OrderSide.SELL, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(p,))
        assert compute_short_gross_exposure(portfolio) == Decimal("1100")
        assert compute_long_gross_exposure(portfolio) == Decimal("0")


class TestMixedLongShortPortfolio:
    def test_gross_sums_absolute_net_nets_out(self) -> None:
        long_p = _position(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        short_p = _position(instrument_id="GBPUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(long_p, short_p))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure == Decimal("2200")
        assert exposure.net_exposure == Decimal("0")

    def test_long_and_short_split_matches_gross(self) -> None:
        long_p = _position(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        short_p = _position(instrument_id="GBPUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("500"), average_entry_price=Decimal("1.30"), mark_price=Decimal("1.30"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(long_p, short_p))
        long_gross = compute_long_gross_exposure(portfolio)
        short_gross = compute_short_gross_exposure(portfolio)
        assert long_gross == Decimal("1100")
        assert short_gross == Decimal("650")
        assert long_gross + short_gross == compute_portfolio_exposure(portfolio).gross_exposure


class TestConcentrationFraction:
    def test_single_instrument_is_fully_concentrated(self) -> None:
        p = _position()
        portfolio = _portfolio(positions=(p,))
        assert compute_concentration_fraction(portfolio) == Decimal("1")

    def test_two_equal_instruments_split_evenly(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="s1", quantity=Decimal("1000"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        b = _position(instrument_id="GBPUSD", strategy_id="s1", quantity=Decimal("1000"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(a, b))
        assert compute_concentration_fraction(portfolio) == Decimal("0.5")

    def test_flat_portfolio_has_zero_concentration(self) -> None:
        portfolio = _portfolio(positions=())
        assert compute_concentration_fraction(portfolio) == Decimal("0")

    def test_concentration_aggregates_same_instrument_across_strategies(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="s1", quantity=Decimal("1000"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        b = _position(instrument_id="EURUSD", strategy_id="s2", quantity=Decimal("1000"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        c = _position(instrument_id="GBPUSD", strategy_id="s1", quantity=Decimal("500"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(a, b, c))
        # EURUSD (across s1+s2) = 2000, GBPUSD = 500, total = 2500 -> concentration = 2000/2500 = 0.8
        assert compute_concentration_fraction(portfolio) == Decimal("0.8")


class TestLeverage:
    def test_computed_as_gross_exposure_over_equity(self) -> None:
        p = _position(quantity=Decimal("1000"), average_entry_price=Decimal("1.00"), mark_price=Decimal("1.00"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(p,), cash=Decimal("500"), equity=Decimal("1500"))
        assert compute_leverage(portfolio) == Decimal("1000") / Decimal("1500")

    def test_zero_equity_raises(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("0"), equity=Decimal("0"), peak_equity=Decimal("0"), daily_start_equity=Decimal("0"))
        with pytest.raises(ExposureCalculationError):
            compute_leverage(portfolio)

    def test_negative_equity_raises(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("-100"), equity=Decimal("-100"), peak_equity=Decimal("0"), daily_start_equity=Decimal("0"))
        with pytest.raises(ExposureCalculationError):
            compute_leverage(portfolio)


class TestContractMultiplierCorrectness:
    def test_market_value_scales_with_multiplier(self) -> None:
        p = _position(quantity=Decimal("10"), average_entry_price=Decimal("100"), mark_price=Decimal("100"), unrealized_pnl=Decimal("0"), contract_multiplier=Decimal("50"))
        portfolio = _portfolio(positions=(p,))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure == Decimal("10") * Decimal("100") * Decimal("50")


class TestAvailableCash:
    def test_subtracts_buffer(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("10000"))
        assert compute_available_cash(portfolio, minimum_cash_buffer=Decimal("2000")) == Decimal("8000")

    def test_can_go_negative_without_clamping(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("1000"))
        assert compute_available_cash(portfolio, minimum_cash_buffer=Decimal("2000")) == Decimal("-1000")


class TestDailyAndTotalLoss:
    def test_daily_loss_floors_at_zero_on_gain(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("110000"), equity=Decimal("110000"), daily_start_equity=Decimal("100000"), peak_equity=Decimal("110000"))
        assert compute_daily_loss(portfolio) == Decimal("0")

    def test_daily_loss_measures_equity_decline_since_day_start(self) -> None:
        portfolio = _portfolio(positions=(), cash=Decimal("90000"), equity=Decimal("90000"), daily_start_equity=Decimal("100000"), peak_equity=Decimal("100000"))
        assert compute_daily_loss(portfolio) == Decimal("10000")

    def test_total_loss_floors_at_zero_on_gain(self) -> None:
        portfolio = _portfolio(positions=(), realized_pnl=Decimal("500"), unrealized_pnl=Decimal("0"))
        assert compute_total_loss(portfolio) == Decimal("0")

    def test_total_loss_measures_negative_total_pnl(self) -> None:
        # positions=() forces unrealized_pnl=0 (Phase 1's own reconciliation
        # invariant -- sum of zero open positions) -- total loss here comes
        # entirely from realized_pnl.
        portfolio = _portfolio(positions=(), cash=Decimal("99000"), equity=Decimal("99000"), realized_pnl=Decimal("-1000"), daily_start_equity=Decimal("100000"))
        assert compute_total_loss(portfolio) == Decimal("1000")


class TestStrategyAggregation:
    def test_strategy_exposure_only_sums_matching_strategy(self) -> None:
        from quant_platform.portfolio_risk.exposure import compute_strategy_exposure

        a = _position(instrument_id="EURUSD", strategy_id="alpha", quantity=Decimal("1000"), average_entry_price=Decimal("1.0"), mark_price=Decimal("1.0"), unrealized_pnl=Decimal("0"))
        b = _position(instrument_id="GBPUSD", strategy_id="beta", quantity=Decimal("500"), average_entry_price=Decimal("1.0"), mark_price=Decimal("1.0"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(a, b))
        assert compute_strategy_exposure(portfolio, strategy_id="alpha").gross_exposure == Decimal("1000")
        assert compute_strategy_exposure(portfolio, strategy_id="beta").gross_exposure == Decimal("500")


class TestMissingPriceFailClosed:
    def test_position_snapshot_requires_mark_price_structurally(self) -> None:
        # Phase 1's own PositionSnapshot invariant already makes "missing
        # price for an open position" structurally impossible -- mark_price
        # is a required, validated field, never optional. This test
        # documents that guarantee at the exposure-calculation boundary.
        from quant_platform.core.exceptions import PortfolioSnapshotValidationError

        with pytest.raises(PortfolioSnapshotValidationError):
            _position(mark_price=Decimal("0"))
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(mark_price=Decimal("-1"))
