"""Unit tests for `portfolio_risk.sizing`: each independent constraint,
combinations of several simultaneous constraints, conservative
quantity-step rounding, zero capacity, non-positive step rejection, and
symmetric short-side handling."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PositionSizingError
from quant_platform.portfolio_risk.models import OrderSide
from quant_platform.portfolio_risk.sizing import (
    compute_maximum_allowed_quantity,
    max_quantity_by_cash_buffer,
    max_quantity_by_instrument_exposure,
    max_quantity_by_leverage,
    max_quantity_by_order_notional,
    max_quantity_by_portfolio_gross_exposure,
    max_quantity_by_position_notional,
    quantize_quantity_down,
)


class TestMaxQuantityByOrderNotional:
    def test_none_limit_is_unconstrained(self) -> None:
        assert max_quantity_by_order_notional(reference_price=Decimal("1.10"), contract_multiplier=Decimal("1"), limit_value=None) is None

    def test_computes_exact_capacity(self) -> None:
        result = max_quantity_by_order_notional(reference_price=Decimal("2"), contract_multiplier=Decimal("1"), limit_value=Decimal("1000"))
        assert result == Decimal("500")

    def test_scales_with_contract_multiplier(self) -> None:
        result = max_quantity_by_order_notional(reference_price=Decimal("10"), contract_multiplier=Decimal("10"), limit_value=Decimal("1000"))
        assert result == Decimal("10")


class TestMaxQuantityByPositionNotional:
    def test_accounts_for_existing_notional(self) -> None:
        result = max_quantity_by_position_notional(current_position_notional=Decimal("400"), reference_price=Decimal("2"), contract_multiplier=Decimal("1"), limit_value=Decimal("1000"))
        assert result == Decimal("300")

    def test_already_over_limit_yields_zero_not_negative(self) -> None:
        result = max_quantity_by_position_notional(current_position_notional=Decimal("1500"), reference_price=Decimal("2"), contract_multiplier=Decimal("1"), limit_value=Decimal("1000"))
        assert result == Decimal("0")


class TestMaxQuantityByInstrumentExposure:
    def test_remaining_capacity(self) -> None:
        result = max_quantity_by_instrument_exposure(current_instrument_gross_exposure=Decimal("0"), reference_price=Decimal("5"), contract_multiplier=Decimal("1"), limit_value=Decimal("500"))
        assert result == Decimal("100")


class TestMaxQuantityByPortfolioGrossExposure:
    def test_remaining_capacity(self) -> None:
        result = max_quantity_by_portfolio_gross_exposure(current_portfolio_gross_exposure=Decimal("9000"), reference_price=Decimal("10"), contract_multiplier=Decimal("1"), limit_value=Decimal("10000"))
        assert result == Decimal("100")


class TestMaxQuantityByLeverage:
    def test_computes_remaining_leverage_headroom(self) -> None:
        # max_gross = 3 * 10000 = 30000; remaining = 30000 - 20000 = 10000; qty = 10000 / (10*1) = 1000
        result = max_quantity_by_leverage(current_portfolio_gross_exposure=Decimal("20000"), equity=Decimal("10000"), reference_price=Decimal("10"), contract_multiplier=Decimal("1"), limit_value=Decimal("3"))
        assert result == Decimal("1000")

    def test_non_positive_equity_yields_zero(self) -> None:
        result = max_quantity_by_leverage(current_portfolio_gross_exposure=Decimal("0"), equity=Decimal("0"), reference_price=Decimal("10"), contract_multiplier=Decimal("1"), limit_value=Decimal("3"))
        assert result == Decimal("0")

    def test_none_limit_is_unconstrained(self) -> None:
        assert max_quantity_by_leverage(current_portfolio_gross_exposure=Decimal("0"), equity=Decimal("1000"), reference_price=Decimal("1"), contract_multiplier=Decimal("1"), limit_value=None) is None


class TestMaxQuantityByCashBuffer:
    def test_buy_is_constrained(self) -> None:
        result = max_quantity_by_cash_buffer(cash=Decimal("10000"), minimum_cash_buffer=Decimal("2000"), reference_price=Decimal("8"), contract_multiplier=Decimal("1"), side=OrderSide.BUY)
        assert result == Decimal("1000")

    def test_sell_is_never_buffer_constrained(self) -> None:
        result = max_quantity_by_cash_buffer(cash=Decimal("0"), minimum_cash_buffer=Decimal("999999"), reference_price=Decimal("1"), contract_multiplier=Decimal("1"), side=OrderSide.SELL)
        assert result is None

    def test_none_buffer_is_unconstrained(self) -> None:
        assert max_quantity_by_cash_buffer(cash=Decimal("100"), minimum_cash_buffer=None, reference_price=Decimal("1"), contract_multiplier=Decimal("1"), side=OrderSide.BUY) is None

    def test_already_below_buffer_yields_zero(self) -> None:
        result = max_quantity_by_cash_buffer(cash=Decimal("500"), minimum_cash_buffer=Decimal("2000"), reference_price=Decimal("1"), contract_multiplier=Decimal("1"), side=OrderSide.BUY)
        assert result == Decimal("0")


class TestQuantizeQuantityDown:
    def test_rounds_down_to_nearest_step(self) -> None:
        assert quantize_quantity_down(Decimal("1234"), step=Decimal("100")) == Decimal("1200")

    def test_exact_multiple_unchanged(self) -> None:
        assert quantize_quantity_down(Decimal("1200"), step=Decimal("100")) == Decimal("1200")

    def test_never_rounds_up(self) -> None:
        result = quantize_quantity_down(Decimal("199.999"), step=Decimal("100"))
        assert result == Decimal("100")

    def test_zero_or_negative_quantity_yields_zero(self) -> None:
        assert quantize_quantity_down(Decimal("0"), step=Decimal("1")) == Decimal("0")
        assert quantize_quantity_down(Decimal("-5"), step=Decimal("1")) == Decimal("0")

    def test_non_positive_step_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            quantize_quantity_down(Decimal("100"), step=Decimal("0"))
        with pytest.raises(PositionSizingError):
            quantize_quantity_down(Decimal("100"), step=Decimal("-1"))


class TestComputeMaximumAllowedQuantity:
    def test_single_binding_constraint(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("10000"), quantity_step=Decimal("1"), constraints=(Decimal("500"),))
        assert result == Decimal("500")

    def test_several_simultaneous_constraints_take_the_minimum(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("10000"), quantity_step=Decimal("1"), constraints=(Decimal("500"), Decimal("300"), Decimal("800"), None))
        assert result == Decimal("300")

    def test_never_increases_requested_quantity(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("100"), quantity_step=Decimal("1"), constraints=(Decimal("999999"),))
        assert result == Decimal("100")

    def test_conservative_step_rounding_applied_after_min(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("10000"), quantity_step=Decimal("1000"), constraints=(Decimal("4321"),))
        assert result == Decimal("4000")

    def test_zero_capacity_constraint_yields_zero(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("100"), quantity_step=Decimal("1"), constraints=(Decimal("0"),))
        assert result == Decimal("0")

    def test_negative_capacity_constraint_treated_as_zero(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("100"), quantity_step=Decimal("1"), constraints=(Decimal("-50"),))
        assert result == Decimal("0")

    def test_no_constraints_returns_requested_quantized(self) -> None:
        result = compute_maximum_allowed_quantity(requested_quantity=Decimal("123"), quantity_step=Decimal("10"), constraints=())
        assert result == Decimal("120")

    def test_non_positive_quantity_step_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            compute_maximum_allowed_quantity(requested_quantity=Decimal("100"), quantity_step=Decimal("0"), constraints=())

    def test_non_positive_requested_quantity_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            compute_maximum_allowed_quantity(requested_quantity=Decimal("0"), quantity_step=Decimal("1"), constraints=())


class TestShortOrdersHandledSymmetrically:
    def test_sell_side_is_never_cash_constrained_matching_buy_which_is(self) -> None:
        buy_capacity = max_quantity_by_cash_buffer(cash=Decimal("1000"), minimum_cash_buffer=Decimal("500"), reference_price=Decimal("1"), contract_multiplier=Decimal("1"), side=OrderSide.BUY)
        sell_capacity = max_quantity_by_cash_buffer(cash=Decimal("1000"), minimum_cash_buffer=Decimal("500"), reference_price=Decimal("1"), contract_multiplier=Decimal("1"), side=OrderSide.SELL)
        assert buy_capacity == Decimal("500")
        assert sell_capacity is None

    def test_notional_and_exposure_constraints_are_side_agnostic(self) -> None:
        # max_quantity_by_order_notional/position_notional/instrument_exposure/portfolio_gross_exposure/leverage
        # do not take a `side` parameter at all -- the same magnitude-based
        # capacity applies whether the order increases a long or a short
        # position; this test documents that symmetry explicitly.
        long_capacity = max_quantity_by_order_notional(reference_price=Decimal("10"), contract_multiplier=Decimal("1"), limit_value=Decimal("1000"))
        short_capacity = max_quantity_by_order_notional(reference_price=Decimal("10"), contract_multiplier=Decimal("1"), limit_value=Decimal("1000"))
        assert long_capacity == short_capacity == Decimal("100")
