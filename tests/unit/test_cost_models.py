"""Tests for transaction cost models."""

from __future__ import annotations

import pytest

from quant_platform.core.types import OrderSide
from quant_platform.costs.models import FixedSpreadCostModel, VolatilityScaledSlippageModel


class TestFixedSpreadCostModel:
    @pytest.fixture
    def model(self) -> FixedSpreadCostModel:
        # 20-point spread, 10-point slippage, point_value=0.01 (e.g. XAUUSD 2dp quote)
        return FixedSpreadCostModel(
            spread_points=20.0, slippage_points=10.0, point_value=0.01, commission_per_unit=2.5
        )

    def test_buy_entry_is_worse_than_reference(self, model: FixedSpreadCostModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.BUY)
        # half spread (0.10) + slippage (0.10) = 0.20 worse for a buyer
        assert fill == pytest.approx(2000.20)

    def test_sell_entry_is_worse_than_reference(self, model: FixedSpreadCostModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.SELL)
        assert fill == pytest.approx(1999.80)

    def test_closing_a_long_sells_at_bid(self, model: FixedSpreadCostModel) -> None:
        fill = model.exit_fill_price(2000.0, OrderSide.BUY)
        assert fill == pytest.approx(1999.90)  # half spread only, no slippage on exit

    def test_closing_a_short_buys_at_ask(self, model: FixedSpreadCostModel) -> None:
        fill = model.exit_fill_price(2000.0, OrderSide.SELL)
        assert fill == pytest.approx(2000.10)

    def test_round_trip_cost_is_spread_plus_entry_slippage(self, model: FixedSpreadCostModel) -> None:
        reference = 2000.0
        entry = model.entry_fill_price(reference, OrderSide.BUY)
        exit_ = model.exit_fill_price(reference, OrderSide.BUY)
        round_trip_cost = entry - exit_
        assert round_trip_cost == pytest.approx(0.30)  # spread (0.20) + slippage (0.10)

    def test_commission_scales_with_quantity(self, model: FixedSpreadCostModel) -> None:
        assert model.commission(quantity=3.0, price=2000.0) == pytest.approx(7.5)
        assert model.commission(quantity=-3.0, price=2000.0) == pytest.approx(7.5)  # sign-agnostic

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spread_points", -1.0),
            ("slippage_points", -1.0),
            ("point_value", 0.0),
            ("point_value", -0.01),
            ("commission_per_unit", -1.0),
        ],
    )
    def test_rejects_invalid_construction(self, field: str, value: float) -> None:
        kwargs = {"spread_points": 20.0, "slippage_points": 10.0, "point_value": 0.01}
        kwargs[field] = value
        with pytest.raises(ValueError):
            FixedSpreadCostModel(**kwargs)


class TestVolatilityScaledSlippageModel:
    @pytest.fixture
    def model(self) -> VolatilityScaledSlippageModel:
        return VolatilityScaledSlippageModel(
            spread_points=20.0,
            base_slippage_points=10.0,
            reference_volatility=1.0,
            point_value=0.01,
        )

    def test_no_volatility_supplied_uses_base_slippage(self, model: VolatilityScaledSlippageModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.BUY)
        assert fill == pytest.approx(2000.20)

    def test_double_volatility_doubles_slippage(self, model: VolatilityScaledSlippageModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.BUY, current_volatility=2.0)
        # half spread (0.10) + slippage (0.10 * 2x = 0.20) = 0.30
        assert fill == pytest.approx(2000.30)

    def test_slippage_multiplier_is_clamped_to_max(self, model: VolatilityScaledSlippageModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.BUY, current_volatility=1000.0)
        max_slippage = 10.0 * 0.01 * 5.0  # clamped to max_slippage_multiplier=5.0
        assert fill == pytest.approx(2000.0 + 0.10 + max_slippage)

    def test_slippage_multiplier_is_clamped_to_min(self, model: VolatilityScaledSlippageModel) -> None:
        fill = model.entry_fill_price(2000.0, OrderSide.BUY, current_volatility=0.0001)
        min_slippage = 10.0 * 0.01 * 0.25  # clamped to min_slippage_multiplier=0.25
        assert fill == pytest.approx(2000.0 + 0.10 + min_slippage)

    def test_exit_price_ignores_volatility(self, model: VolatilityScaledSlippageModel) -> None:
        fill_no_vol = model.exit_fill_price(2000.0, OrderSide.BUY)
        fill_with_vol = model.exit_fill_price(2000.0, OrderSide.BUY, current_volatility=5.0)
        assert fill_no_vol == fill_with_vol

    def test_rejects_non_positive_reference_volatility(self) -> None:
        with pytest.raises(ValueError):
            VolatilityScaledSlippageModel(
                spread_points=20.0, base_slippage_points=10.0, reference_volatility=0.0, point_value=0.01
            )

    def test_rejects_inverted_multiplier_bounds(self) -> None:
        with pytest.raises(ValueError):
            VolatilityScaledSlippageModel(
                spread_points=20.0,
                base_slippage_points=10.0,
                reference_volatility=1.0,
                point_value=0.01,
                min_slippage_multiplier=5.0,
                max_slippage_multiplier=1.0,
            )
