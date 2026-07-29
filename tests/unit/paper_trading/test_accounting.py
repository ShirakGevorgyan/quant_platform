"""Milestone 7, Section 12: position accounting hand fixtures. All 11
required named fixtures (one long round trip; one short round trip;
partial long close; partial short close; scale-in; reversal; fee-only
loss; spread/slippage; financing; zero-price-change round trip; price-gap
fill), with exact expected numbers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import PositionAccountingError
from quant_platform.paper_trading.accounting import (
    apply_fill_to_position,
    apply_financing_to_position,
    apply_mark_to_position,
    flat_position,
)
from quant_platform.paper_trading.fills import create_fill
from quant_platform.paper_trading.models import OrderSide, PartialFillPolicyKind

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_EVENT = "a" * 64


def _fill(*, side: OrderSide, quantity: float, price: float, contract_multiplier: float = 1.0, spread_cost: float = 0.0, slippage_cost: float = 0.0, commission_cost: float = 0.0, is_final: bool = True, execution_time: datetime = _T0):
    return create_fill(
        order_id="order-1", session_id="session-1", instrument="X", side=side, quantity=quantity, price=price, contract_multiplier=contract_multiplier,
        spread_cost=spread_cost, slippage_cost=slippage_cost, commission_cost=commission_cost, execution_time=execution_time,
        source_market_event_identity=_HEX_EVENT, liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=is_final,
    )


class TestElevenRequiredHandFixtures:
    def test_one_long_round_trip(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        assert position.signed_quantity == 10.0
        assert position.average_entry_price == 100.0
        assert position.realized_pnl == 0.0
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=110.0), event_time=_T0)
        assert position.signed_quantity == 0.0
        assert position.average_entry_price is None
        assert position.realized_pnl == pytest.approx(100.0)

    def test_one_short_round_trip(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0)
        assert position.signed_quantity == -10.0
        assert position.average_entry_price == 100.0
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=90.0), event_time=_T0)
        assert position.signed_quantity == 0.0
        assert position.realized_pnl == pytest.approx(100.0)

    def test_partial_long_close(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=4.0, price=105.0), event_time=_T0)
        assert position.signed_quantity == pytest.approx(6.0)
        assert position.average_entry_price == pytest.approx(100.0)
        assert position.realized_pnl == pytest.approx(20.0)
        assert position.gross_cost_basis == pytest.approx(600.0)

    def test_partial_short_close(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=4.0, price=95.0), event_time=_T0)
        assert position.signed_quantity == pytest.approx(-6.0)
        assert position.average_entry_price == pytest.approx(100.0)
        assert position.realized_pnl == pytest.approx(20.0)

    def test_scale_in(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=5.0, price=110.0), event_time=_T0)
        assert position.signed_quantity == pytest.approx(15.0)
        assert position.average_entry_price == pytest.approx((10.0 * 100.0 + 5.0 * 110.0) / 15.0)
        assert position.realized_pnl == 0.0

    def test_reversal(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=15.0, price=105.0), event_time=_T0)
        assert position.signed_quantity == pytest.approx(-5.0)
        assert position.average_entry_price == pytest.approx(105.0)
        assert position.realized_pnl == pytest.approx(10.0 * (105.0 - 100.0))

    def test_fee_only_loss(self) -> None:
        """Zero price movement, but nonzero commission -- realized P&L
        from PRICE alone is zero; the "loss" lives entirely in
        `accumulated_transaction_costs`, tracked separately."""
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0, commission_cost=2.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0, commission_cost=2.0), event_time=_T0)
        assert position.realized_pnl == 0.0
        assert position.accumulated_transaction_costs == pytest.approx(4.0)

    def test_spread_and_slippage_accumulate_across_fills(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=5.0, price=100.0, spread_cost=0.5, slippage_cost=0.25), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=5.0, price=101.0, spread_cost=0.5, slippage_cost=0.25), event_time=_T0)
        assert position.accumulated_transaction_costs == pytest.approx(1.5)

    def test_financing(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_financing_to_position(position, cash_delta=-5.0, event_time=_T0 + timedelta(days=1))
        position = apply_financing_to_position(position, cash_delta=-5.0, event_time=_T0 + timedelta(days=2))
        assert position.accumulated_financing == pytest.approx(-10.0)

    def test_zero_price_change_round_trip(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0)
        assert position.realized_pnl == 0.0
        assert position.signed_quantity == 0.0

    def test_price_gap_fill(self) -> None:
        """A large, discontinuous jump between the entry fill and the next
        mark-to-market event must be handled exactly, with no clamping."""
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=150.0, event_time=_T0 + timedelta(hours=1))
        assert position.unrealized_pnl == pytest.approx(500.0)
        assert position.last_mark == 150.0


class TestMarkToMarketUnifiedFormula:
    def test_long_gains_when_price_rises(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=110.0, event_time=_T0)
        assert position.unrealized_pnl == pytest.approx(100.0)

    def test_long_loses_when_price_falls(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=90.0, event_time=_T0)
        assert position.unrealized_pnl == pytest.approx(-100.0)

    def test_short_gains_when_price_falls(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=90.0, event_time=_T0)
        assert position.unrealized_pnl == pytest.approx(100.0)

    def test_short_loses_when_price_rises(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=10.0, price=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=110.0, event_time=_T0)
        assert position.unrealized_pnl == pytest.approx(-100.0)

    def test_flat_position_has_zero_unrealized_pnl(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        position = apply_mark_to_position(position, mark_price=100.0, event_time=_T0)
        assert position.unrealized_pnl == 0.0


class TestContractMultiplierAppliedConsistently:
    """Regression coverage: a non-1.0 `contract_multiplier` (e.g. a
    XAUUSD-like instrument with multiplier=100) must scale realized P&L,
    unrealized P&L, and gross_cost_basis identically -- these formulas
    must never silently assume a unit multiplier."""

    def test_realized_pnl_scales_by_contract_multiplier(self) -> None:
        position = flat_position("X", contract_multiplier=100.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=1.0, price=1900.0, contract_multiplier=100.0), event_time=_T0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.SELL, quantity=1.0, price=1910.0, contract_multiplier=100.0), event_time=_T0)
        # price moved by 10 per unit, 1 unit, multiplier 100 -> 1000, not 10
        assert position.realized_pnl == pytest.approx(1000.0)

    def test_unrealized_pnl_scales_by_contract_multiplier(self) -> None:
        position = flat_position("X", contract_multiplier=100.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=1.0, price=1900.0, contract_multiplier=100.0), event_time=_T0)
        position = apply_mark_to_position(position, mark_price=1910.0, event_time=_T0)
        assert position.unrealized_pnl == pytest.approx(1000.0)

    def test_gross_cost_basis_scales_by_contract_multiplier(self) -> None:
        position = flat_position("X", contract_multiplier=100.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=1.0, price=1900.0, contract_multiplier=100.0), event_time=_T0)
        assert position.gross_cost_basis == pytest.approx(190_000.0)

    def test_fill_implying_a_different_multiplier_than_the_position_rejected(self) -> None:
        """A fill built with a DIFFERENT contract_multiplier than the
        position's own is a data-consistency error, not a silently
        accepted mismatch."""
        position = flat_position("X", contract_multiplier=100.0)
        mismatched_fill = _fill(side=OrderSide.BUY, quantity=1.0, price=1900.0, contract_multiplier=1.0)
        with pytest.raises(PositionAccountingError, match="contract_multiplier"):
            apply_fill_to_position(position, mismatched_fill, event_time=_T0)


class TestPositionStateValidation:
    def test_flat_with_average_entry_price_rejected(self) -> None:
        with pytest.raises(PositionAccountingError, match="average_entry_price"):
            from quant_platform.paper_trading.accounting import PositionState

            PositionState(
                instrument="X", contract_multiplier=1.0, signed_quantity=0.0, average_entry_price=100.0, gross_cost_basis=0.0, realized_pnl=0.0,
                unrealized_pnl=0.0, accumulated_transaction_costs=0.0, accumulated_financing=0.0, last_mark=None, last_event_time=None, position_version=0,
            )

    def test_mismatched_instrument_fill_rejected(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        wrong_instrument_fill = create_fill(
            order_id="o1", session_id="s1", instrument="OTHER", side=OrderSide.BUY, quantity=1.0, price=100.0, contract_multiplier=1.0,
            spread_cost=0.0, slippage_cost=0.0, commission_cost=0.0, execution_time=_T0, source_market_event_identity=_HEX_EVENT,
            liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=True,
        )
        with pytest.raises(PositionAccountingError, match="instrument"):
            apply_fill_to_position(position, wrong_instrument_fill, event_time=_T0)

    def test_json_round_trip(self) -> None:
        from quant_platform.paper_trading.accounting import PositionState

        position = flat_position("X", contract_multiplier=1.0)
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=10.0, price=100.0), event_time=_T0)
        roundtripped = PositionState.from_json_dict(position.to_json_dict())
        assert roundtripped == position

    def test_position_version_increments_on_every_mutation(self) -> None:
        position = flat_position("X", contract_multiplier=1.0)
        assert position.position_version == 0
        position = apply_fill_to_position(position, _fill(side=OrderSide.BUY, quantity=1.0, price=100.0), event_time=_T0)
        assert position.position_version == 1
        position = apply_mark_to_position(position, mark_price=101.0, event_time=_T0)
        assert position.position_version == 2
        position = apply_financing_to_position(position, cash_delta=-1.0, event_time=_T0)
        assert position.position_version == 3
