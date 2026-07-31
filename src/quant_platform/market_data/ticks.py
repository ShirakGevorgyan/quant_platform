"""Immutable `Tick` events (Milestone 10, Phase 1) -- the finest-grained
market data event this package models: a single price (and optionally a
size) observed at an instant, with no timeframe. See `candles.py`'s
module docstring for the shared envelope-plus-typed-payload design this
mirrors exactly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import MarketDataEventError
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_non_negative_sequence,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = ["TICK_KIND", "Tick", "create_tick"]

TICK_KIND = "tick"
_PAYLOAD_KEYS = ("price", "volume")


def _payload_to_json(*, price: Decimal, volume: Decimal | None) -> dict[str, object]:
    return {"price": str(price), "volume": (None if volume is None else str(volume))}


def _parse_payload(payload: dict[str, object]) -> tuple[Decimal, Decimal | None]:
    extra_keys = set(payload.keys()) - set(_PAYLOAD_KEYS)
    if extra_keys:
        raise MarketDataEventError(f"Tick.payload has unexpected key(s): {sorted(extra_keys)}")
    missing_keys = set(_PAYLOAD_KEYS) - set(payload.keys())
    if missing_keys:
        raise MarketDataEventError(f"Tick.payload is missing required key(s): {sorted(missing_keys)}")
    price = parse_decimal(payload["price"], field_name="Tick.payload['price']")
    raw_volume = payload["volume"]
    volume = None if raw_volume is None else parse_decimal(raw_volume, field_name="Tick.payload['volume']")
    return price, volume


@dataclass(frozen=True, slots=True)
class Tick:
    event_id: str
    instrument_id: str
    provider: str
    symbol: str
    event_time: datetime
    arrival_time: datetime
    timeframe: None
    """Always `None` -- a tick has no timeframe. Present as a field
    (rather than omitted) so every event kind shares the identical
    envelope schema (see `candles.py`'s module docstring)."""
    sequence: int
    source_event_id: str | None
    payload: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.instrument_id, field_name="Tick.instrument_id")
        require_non_empty(self.provider, field_name="Tick.provider")
        require_non_empty(self.symbol, field_name="Tick.symbol")
        require_tz_aware(self.event_time, field_name="Tick.event_time")
        require_tz_aware(self.arrival_time, field_name="Tick.arrival_time")
        if self.arrival_time < self.event_time:
            raise MarketDataEventError(f"Tick.arrival_time ({self.arrival_time}) must be >= event_time ({self.event_time})")
        if self.timeframe is not None:
            raise MarketDataEventError("Tick.timeframe must be None -- a tick has no timeframe")
        require_non_negative_sequence(self.sequence)
        price, volume = _parse_payload(self.payload)
        if price <= 0:
            raise MarketDataEventError(f"Tick.price must be > 0, got {price}")
        if volume is not None and volume < 0:
            raise MarketDataEventError(f"Tick.volume must be >= 0, got {volume}")

    @property
    def price(self) -> Decimal:
        return _parse_payload(self.payload)[0]

    @property
    def volume(self) -> Decimal | None:
        return _parse_payload(self.payload)[1]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": TICK_KIND, "event_id": self.event_id, "instrument_id": self.instrument_id, "provider": self.provider,
            "symbol": self.symbol, "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
            "arrival_time": serialize_timestamp(self.arrival_time, field_name="arrival_time"), "timeframe": None,
            "sequence": self.sequence, "source_event_id": self.source_event_id, "payload": self.payload,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Tick:
        from quant_platform.ml.persistence import as_json_dict

        if raw.get("timeframe") is not None:
            raise MarketDataEventError(f"Tick.timeframe must be None, got {raw.get('timeframe')!r}")
        return cls(
            event_id=str(raw["event_id"]), instrument_id=str(raw["instrument_id"]), provider=str(raw["provider"]),
            symbol=str(raw["symbol"]), event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
            arrival_time=deserialize_timestamp(raw["arrival_time"], field_name="arrival_time"), timeframe=None,
            sequence=int(str(raw["sequence"])),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            payload=as_json_dict(raw["payload"], field_name="payload"),
        )


def create_tick(
    *, instrument_id: str, provider: str, symbol: str, event_time: datetime, sequence: int, price: Decimal,
    volume: Decimal | None = None, arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Tick:
    resolved_arrival_time = event_time if arrival_time is None else arrival_time
    payload = _payload_to_json(price=price, volume=volume)
    provisional = Tick(
        event_id="0" * 64, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
    event_id = compute_content_id(TICK_KIND, provisional.to_identity_payload())
    return Tick(
        event_id=event_id, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
