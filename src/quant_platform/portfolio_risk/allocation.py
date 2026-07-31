"""`CapitalAllocation` and `PositionSizeProposal` -- small, standalone,
content-addressed value objects for `quant_platform.portfolio_risk`
(Milestone 9, Phase 1). Neither is consumed by an evaluator in this
phase (none exists yet); both exist so a future evaluator has a
well-defined, already-tested shape to build against rather than
inventing one ad hoc mid-Phase-2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import PositionSizingError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import compute_content_id, decimal_to_json, parse_decimal
from quant_platform.portfolio_risk.models import OrderSide

CAPITAL_ALLOCATION_KIND = "capital_allocation"
POSITION_SIZE_PROPOSAL_KIND = "position_size_proposal"


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PositionSizingError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    _require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PositionSizingError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PositionSizingError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PositionSizingError(f"{field_name}: {exc}") from exc


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise PositionSizingError(f"{field_name} must be finite and > 0, got {value!r}")


def _non_negative_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value < 0:
        raise PositionSizingError(f"{field_name} must be finite and >= 0, got {value!r}")


# --------------------------------------------------------------------------
# CapitalAllocation
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CapitalAllocation:
    capital_allocation_id: str
    portfolio_id: str
    strategy_id: str
    allocated_capital: Decimal
    utilized_capital: Decimal
    allocation_sequence: int
    event_time: datetime

    def __post_init__(self) -> None:
        if not self.portfolio_id:
            raise PositionSizingError("CapitalAllocation.portfolio_id must not be empty")
        if not self.strategy_id:
            raise PositionSizingError("CapitalAllocation.strategy_id must not be empty")
        _non_negative_decimal(self.allocated_capital, field_name="CapitalAllocation.allocated_capital")
        _non_negative_decimal(self.utilized_capital, field_name="CapitalAllocation.utilized_capital")
        if self.utilized_capital > self.allocated_capital:
            raise PositionSizingError(
                f"CapitalAllocation.utilized_capital ({self.utilized_capital!r}) must not exceed allocated_capital "
                f"({self.allocated_capital!r})"
            )
        if self.allocation_sequence < 0:
            raise PositionSizingError(f"CapitalAllocation.allocation_sequence must be >= 0, got {self.allocation_sequence}")
        _require_tz_aware(self.event_time, field_name="CapitalAllocation.event_time")

    @property
    def available_capital(self) -> Decimal:
        return self.allocated_capital - self.utilized_capital

    def to_json_dict(self) -> dict[str, object]:
        return {
            "capital_allocation_id": self.capital_allocation_id, "portfolio_id": self.portfolio_id, "strategy_id": self.strategy_id,
            "allocated_capital": decimal_to_json(self.allocated_capital), "utilized_capital": decimal_to_json(self.utilized_capital),
            "allocation_sequence": self.allocation_sequence, "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["capital_allocation_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CapitalAllocation:
        return cls(
            capital_allocation_id=str(raw["capital_allocation_id"]), portfolio_id=str(raw["portfolio_id"]), strategy_id=str(raw["strategy_id"]),
            allocated_capital=parse_decimal(raw["allocated_capital"], field_name="allocated_capital"),
            utilized_capital=parse_decimal(raw["utilized_capital"], field_name="utilized_capital"),
            allocation_sequence=int(str(raw["allocation_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_capital_allocation(
    *, portfolio_id: str, strategy_id: str, allocated_capital: Decimal, utilized_capital: Decimal, allocation_sequence: int, event_time: datetime,
) -> CapitalAllocation:
    provisional = CapitalAllocation(
        capital_allocation_id="0" * 64, portfolio_id=portfolio_id, strategy_id=strategy_id, allocated_capital=allocated_capital,
        utilized_capital=utilized_capital, allocation_sequence=allocation_sequence, event_time=event_time,
    )
    capital_allocation_id = compute_content_id(CAPITAL_ALLOCATION_KIND, provisional.to_identity_payload())
    return CapitalAllocation(
        capital_allocation_id=capital_allocation_id, portfolio_id=portfolio_id, strategy_id=strategy_id, allocated_capital=allocated_capital,
        utilized_capital=utilized_capital, allocation_sequence=allocation_sequence, event_time=event_time,
    )


# --------------------------------------------------------------------------
# PositionSizeProposal
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionSizeProposal:
    position_size_proposal_id: str
    portfolio_id: str
    strategy_id: str
    instrument_id: str
    side: OrderSide
    proposed_quantity: Decimal
    reference_price: Decimal
    proposed_sequence: int
    event_time: datetime

    def __post_init__(self) -> None:
        for field_name, value in (("portfolio_id", self.portfolio_id), ("strategy_id", self.strategy_id), ("instrument_id", self.instrument_id)):
            if not value:
                raise PositionSizingError(f"PositionSizeProposal.{field_name} must not be empty")
        _positive_decimal(self.proposed_quantity, field_name="PositionSizeProposal.proposed_quantity")
        _positive_decimal(self.reference_price, field_name="PositionSizeProposal.reference_price")
        if self.proposed_sequence < 0:
            raise PositionSizingError(f"PositionSizeProposal.proposed_sequence must be >= 0, got {self.proposed_sequence}")
        _require_tz_aware(self.event_time, field_name="PositionSizeProposal.event_time")

    @property
    def proposed_notional(self) -> Decimal:
        return self.proposed_quantity * self.reference_price

    def to_json_dict(self) -> dict[str, object]:
        return {
            "position_size_proposal_id": self.position_size_proposal_id, "portfolio_id": self.portfolio_id, "strategy_id": self.strategy_id,
            "instrument_id": self.instrument_id, "side": self.side.value, "proposed_quantity": decimal_to_json(self.proposed_quantity),
            "reference_price": decimal_to_json(self.reference_price), "proposed_sequence": self.proposed_sequence,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["position_size_proposal_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PositionSizeProposal:
        return cls(
            position_size_proposal_id=str(raw["position_size_proposal_id"]), portfolio_id=str(raw["portfolio_id"]),
            strategy_id=str(raw["strategy_id"]), instrument_id=str(raw["instrument_id"]), side=OrderSide(raw["side"]),
            proposed_quantity=parse_decimal(raw["proposed_quantity"], field_name="proposed_quantity"),
            reference_price=parse_decimal(raw["reference_price"], field_name="reference_price"),
            proposed_sequence=int(str(raw["proposed_sequence"])),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
        )


def create_position_size_proposal(
    *, portfolio_id: str, strategy_id: str, instrument_id: str, side: OrderSide, proposed_quantity: Decimal, reference_price: Decimal,
    proposed_sequence: int, event_time: datetime,
) -> PositionSizeProposal:
    provisional = PositionSizeProposal(
        position_size_proposal_id="0" * 64, portfolio_id=portfolio_id, strategy_id=strategy_id, instrument_id=instrument_id, side=side,
        proposed_quantity=proposed_quantity, reference_price=reference_price, proposed_sequence=proposed_sequence, event_time=event_time,
    )
    position_size_proposal_id = compute_content_id(POSITION_SIZE_PROPOSAL_KIND, provisional.to_identity_payload())
    return PositionSizeProposal(
        position_size_proposal_id=position_size_proposal_id, portfolio_id=portfolio_id, strategy_id=strategy_id, instrument_id=instrument_id,
        side=side, proposed_quantity=proposed_quantity, reference_price=reference_price, proposed_sequence=proposed_sequence,
        event_time=event_time,
    )


__all__ = [
    "CAPITAL_ALLOCATION_KIND",
    "POSITION_SIZE_PROPOSAL_KIND",
    "CapitalAllocation",
    "PositionSizeProposal",
    "create_capital_allocation",
    "create_position_size_proposal",
]
