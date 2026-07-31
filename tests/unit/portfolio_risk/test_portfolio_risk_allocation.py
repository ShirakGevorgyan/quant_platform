"""Unit tests for `portfolio_risk.allocation`: `CapitalAllocation` and
`PositionSizeProposal`."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PositionSizingError
from quant_platform.portfolio_risk.allocation import (
    CapitalAllocation,
    PositionSizeProposal,
    create_capital_allocation,
    create_position_size_proposal,
)
from quant_platform.portfolio_risk.models import OrderSide

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _allocation(**overrides: object) -> CapitalAllocation:
    base: dict[str, object] = {
        "portfolio_id": "portfolio-1", "strategy_id": "strategy-a", "allocated_capital": Decimal("100000"),
        "utilized_capital": Decimal("25000"), "allocation_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_capital_allocation(**base)  # type: ignore[arg-type]


def _proposal(**overrides: object) -> PositionSizeProposal:
    base: dict[str, object] = {
        "portfolio_id": "portfolio-1", "strategy_id": "strategy-a", "instrument_id": "EURUSD", "side": OrderSide.BUY,
        "proposed_quantity": Decimal("1000"), "reference_price": Decimal("1.10"), "proposed_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_position_size_proposal(**base)  # type: ignore[arg-type]


class TestCapitalAllocationInvariants:
    def test_default_constructs(self) -> None:
        allocation = _allocation()
        assert allocation.available_capital == Decimal("75000")

    def test_utilized_exceeding_allocated_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _allocation(allocated_capital=Decimal("100"), utilized_capital=Decimal("101"))

    def test_utilized_equal_to_allocated_accepted(self) -> None:
        allocation = _allocation(allocated_capital=Decimal("100"), utilized_capital=Decimal("100"))
        assert allocation.available_capital == Decimal("0")

    def test_negative_allocated_capital_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _allocation(allocated_capital=Decimal("-1"), utilized_capital=Decimal("0"))

    def test_empty_strategy_id_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _allocation(strategy_id="")

    def test_negative_sequence_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _allocation(allocation_sequence=-1)

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _allocation(event_time=datetime(2026, 1, 1))


class TestCapitalAllocationIdentity:
    def test_deterministic(self) -> None:
        assert _allocation().capital_allocation_id == _allocation().capital_allocation_id

    def test_utilized_capital_participates_in_identity(self) -> None:
        a = _allocation(utilized_capital=Decimal("1000")).capital_allocation_id
        b = _allocation(utilized_capital=Decimal("2000")).capital_allocation_id
        assert a != b

    def test_round_trips_through_json(self) -> None:
        allocation = _allocation()
        restored = CapitalAllocation.from_json_dict(allocation.to_json_dict())
        assert restored.to_json_dict() == allocation.to_json_dict()


class TestPositionSizeProposalInvariants:
    def test_default_constructs(self) -> None:
        proposal = _proposal()
        assert proposal.proposed_notional == Decimal("1000") * Decimal("1.10")

    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _proposal(proposed_quantity=Decimal("0"))

    def test_non_positive_reference_price_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _proposal(reference_price=Decimal("0"))

    def test_empty_instrument_id_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _proposal(instrument_id="")

    def test_negative_sequence_rejected(self) -> None:
        with pytest.raises(PositionSizingError):
            _proposal(proposed_sequence=-1)


class TestPositionSizeProposalIdentity:
    def test_deterministic(self) -> None:
        assert _proposal().position_size_proposal_id == _proposal().position_size_proposal_id

    def test_side_participates_in_identity(self) -> None:
        a = _proposal(side=OrderSide.BUY).position_size_proposal_id
        b = _proposal(side=OrderSide.SELL).position_size_proposal_id
        assert a != b

    def test_round_trips_through_json(self) -> None:
        proposal = _proposal()
        restored = PositionSizeProposal.from_json_dict(proposal.to_json_dict())
        assert restored.to_json_dict() == proposal.to_json_dict()
