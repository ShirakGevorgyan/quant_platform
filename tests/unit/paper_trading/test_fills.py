"""Milestone 7, Section 11: `Fill` construction/validation, deterministic
identity, JSON round-trip, and `validate_fill_sequence_for_order`'s
cross-fill invariants (cumulative quantity never exceeds the order's own
quantity; at most one final fill; a final fill's cumulative quantity
matches the order exactly)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import FillValidationError
from quant_platform.paper_trading.fills import Fill, create_fill, validate_fill_sequence_for_order
from quant_platform.paper_trading.models import (
    OrderSide,
    OrderTypeKind,
    PartialFillPolicyKind,
    PositionIntentKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import create_order_request

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_EVENT = "a" * 64
_HEX_DECISION = "b" * 64


def _fill(**overrides: object) -> Fill:
    defaults: dict[str, object] = {
        "order_id": "order-1", "session_id": "session-1", "instrument": "X", "side": OrderSide.BUY, "quantity": 1.0, "price": 100.0,
        "contract_multiplier": 1.0, "spread_cost": 0.1, "slippage_cost": 0.05, "commission_cost": 0.2, "execution_time": _T0,
        "source_market_event_identity": _HEX_EVENT, "liquidity_assumption": PartialFillPolicyKind.FULL_FILL_ONLY, "is_final": True,
    }
    defaults.update(overrides)
    return create_fill(**defaults)  # type: ignore[arg-type]


def _order(quantity: float = 3.0, side: OrderSide = OrderSide.BUY):
    return create_order_request(
        client_order_id="c1", session_id="session-1", strategy_decision_id=_HEX_DECISION, instrument="X", side=side, order_type=OrderTypeKind.MARKET,
        quantity=quantity, time_in_force=TimeInForceKind.DAY, create_time=_T0, submit_time=_T0, reduce_only=False, position_intent=PositionIntentKind.OPEN,
    )


class TestFillValidation:
    def test_valid_fill(self) -> None:
        fill = _fill()
        assert fill.quantity == 1.0

    def test_gross_notional_computed_from_price_quantity_multiplier(self) -> None:
        fill = _fill(quantity=2.0, price=50.0, contract_multiplier=10.0)
        assert fill.gross_notional == pytest.approx(2.0 * 50.0 * 10.0)

    def test_financing_component_always_zero(self) -> None:
        """Mutation-test-style pin: `create_fill` never accepts a caller-
        supplied `financing_component` -- it is always exactly 0.0
        (Section 16's recognition-timing rule: financing is recognized
        only at `FinancingEvent` processing, never at fill time)."""
        assert _fill().financing_component == 0.0

    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(FillValidationError, match="quantity"):
            _fill(quantity=0.0)

    def test_non_positive_price_rejected(self) -> None:
        with pytest.raises(FillValidationError, match="price"):
            _fill(price=0.0)

    def test_negative_spread_cost_rejected(self) -> None:
        with pytest.raises(FillValidationError, match="spread_cost"):
            _fill(spread_cost=-0.1)

    def test_invalid_source_market_event_identity_rejected(self) -> None:
        with pytest.raises(FillValidationError, match="source_market_event_identity"):
            _fill(source_market_event_identity="not-a-hash")

    def test_naive_execution_time_rejected(self) -> None:
        with pytest.raises(FillValidationError, match="timezone-aware"):
            _fill(execution_time=datetime(2026, 1, 5, 10, 0, 0))

    def test_json_round_trip(self) -> None:
        fill = _fill()
        assert Fill.from_json_dict(fill.to_json_dict()) == fill


class TestFillIdentity:
    def test_identical_arguments_produce_identical_fill_id(self) -> None:
        assert _fill().fill_id == _fill().fill_id

    def test_different_price_changes_fill_id(self) -> None:
        assert _fill(price=100.0).fill_id != _fill(price=101.0).fill_id

    def test_different_execution_time_changes_fill_id(self) -> None:
        from datetime import timedelta

        assert _fill(execution_time=_T0).fill_id != _fill(execution_time=_T0 + timedelta(seconds=1)).fill_id


class TestValidateFillSequenceForOrder:
    def test_single_full_fill_is_valid(self) -> None:
        order = _order(quantity=2.0)
        fill = _fill(order_id=order.order_id, quantity=2.0, is_final=True)
        validate_fill_sequence_for_order(order, [fill])

    def test_two_partial_fills_summing_to_order_quantity_is_valid(self) -> None:
        order = _order(quantity=3.0)
        first = _fill(order_id=order.order_id, quantity=1.0, is_final=False)
        second = _fill(order_id=order.order_id, quantity=2.0, is_final=True, price=101.0)
        validate_fill_sequence_for_order(order, [first, second])

    def test_fill_for_wrong_order_rejected(self) -> None:
        order = _order(quantity=2.0)
        fill = _fill(order_id="different-order", quantity=2.0, is_final=True)
        with pytest.raises(FillValidationError, match="belongs to order"):
            validate_fill_sequence_for_order(order, [fill])

    def test_fill_side_mismatch_rejected(self) -> None:
        order = _order(quantity=2.0, side=OrderSide.BUY)
        fill = _fill(order_id=order.order_id, side=OrderSide.SELL, quantity=2.0, is_final=True)
        with pytest.raises(FillValidationError, match="side"):
            validate_fill_sequence_for_order(order, [fill])

    def test_cumulative_exceeding_order_quantity_rejected(self) -> None:
        order = _order(quantity=1.0)
        over_fill = _fill(order_id=order.order_id, quantity=2.0, is_final=True)
        with pytest.raises(FillValidationError, match="exceeds order quantity"):
            validate_fill_sequence_for_order(order, [over_fill])

    def test_final_fill_with_quantity_short_of_order_rejected(self) -> None:
        order = _order(quantity=3.0)
        short_final_fill = _fill(order_id=order.order_id, quantity=1.0, is_final=True)
        with pytest.raises(FillValidationError, match="is_final"):
            validate_fill_sequence_for_order(order, [short_final_fill])

    def test_fill_after_final_fill_rejected(self) -> None:
        order = _order(quantity=3.0)
        final_fill = _fill(order_id=order.order_id, quantity=3.0, is_final=True)
        extra_fill = _fill(order_id=order.order_id, quantity=1.0, is_final=False, price=105.0)
        with pytest.raises(FillValidationError, match="already-final"):
            validate_fill_sequence_for_order(order, [final_fill, extra_fill])

    def test_empty_fill_list_is_valid(self) -> None:
        order = _order(quantity=1.0)
        validate_fill_sequence_for_order(order, [])
