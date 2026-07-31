"""Immutable OHLCV `Candle` events (Milestone 10, Phase 1).

Every event kind in this package shares one envelope shape -- `event_id`,
`instrument_id`, `provider`, `symbol`, `event_time`, `arrival_time`,
`timeframe`, `sequence`, `source_event_id`, `payload` -- with the kind-
specific numeric fields living INSIDE `payload` (a JSON-safe dict of
`Decimal`-as-string values) rather than as separate top-level dataclass
fields. `payload` is exposed back out through typed, validating
properties (`open`/`high`/`low`/`close`/`volume` here) so callers never
touch the raw dict directly; this keeps the envelope schema byte-for-byte
identical across `Tick`/`Quote`/`Trade`/`Candle` (see `events.py`'s
`MarketDataEventKind` dispatch) while still giving construction-time
validation and ergonomic typed access.

For a `Candle`, `event_time` is the bar's OPEN time (the conventional
index key for an OHLCV series -- see `core.types.OHLCV_COLUMNS`); the
close time is not a separately stored field (which could drift out of
sync with `timeframe`) but a derived property computed the same way the
rest of the platform does (`core.time_utils.compute_close_time`)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import MarketDataEventError
from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import Timeframe
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_non_negative_sequence,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "CANDLE_KIND",
    "Candle",
    "candle_body_size",
    "candle_is_bullish",
    "candle_lower_wick",
    "candle_range",
    "candle_upper_wick",
    "create_candle",
]

CANDLE_KIND = "candle"
_PAYLOAD_KEYS = ("open", "high", "low", "close", "volume")


def _payload_to_json(*, open_: Decimal, high: Decimal, low: Decimal, close: Decimal, volume: Decimal | None) -> dict[str, object]:
    return {
        "open": str(open_), "high": str(high), "low": str(low), "close": str(close),
        "volume": (None if volume is None else str(volume)),
    }


def _parse_payload(payload: dict[str, object]) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal | None]:
    extra_keys = set(payload.keys()) - set(_PAYLOAD_KEYS)
    if extra_keys:
        raise MarketDataEventError(f"Candle.payload has unexpected key(s): {sorted(extra_keys)}")
    missing_keys = set(_PAYLOAD_KEYS) - set(payload.keys())
    if missing_keys:
        raise MarketDataEventError(f"Candle.payload is missing required key(s): {sorted(missing_keys)}")
    open_ = parse_decimal(payload["open"], field_name="Candle.payload['open']")
    high = parse_decimal(payload["high"], field_name="Candle.payload['high']")
    low = parse_decimal(payload["low"], field_name="Candle.payload['low']")
    close = parse_decimal(payload["close"], field_name="Candle.payload['close']")
    raw_volume = payload["volume"]
    volume = None if raw_volume is None else parse_decimal(raw_volume, field_name="Candle.payload['volume']")
    return open_, high, low, close, volume


@dataclass(frozen=True, slots=True)
class Candle:
    event_id: str
    instrument_id: str
    provider: str
    symbol: str
    event_time: datetime
    """The bar's OPEN time."""
    arrival_time: datetime
    timeframe: Timeframe
    sequence: int
    source_event_id: str | None
    payload: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.instrument_id, field_name="Candle.instrument_id")
        require_non_empty(self.provider, field_name="Candle.provider")
        require_non_empty(self.symbol, field_name="Candle.symbol")
        require_tz_aware(self.event_time, field_name="Candle.event_time")
        require_tz_aware(self.arrival_time, field_name="Candle.arrival_time")
        if self.arrival_time < self.event_time:
            raise MarketDataEventError(f"Candle.arrival_time ({self.arrival_time}) must be >= event_time ({self.event_time})")
        require_non_negative_sequence(self.sequence)
        open_, high, low, close, volume = _parse_payload(self.payload)
        if high < low:
            raise MarketDataEventError(f"Candle.high ({high}) must be >= low ({low})")
        if not (low <= open_ <= high):
            raise MarketDataEventError(f"Candle.open ({open_}) must be within [low, high] ([{low}, {high}])")
        if not (low <= close <= high):
            raise MarketDataEventError(f"Candle.close ({close}) must be within [low, high] ([{low}, {high}])")
        if volume is not None and volume < 0:
            raise MarketDataEventError(f"Candle.volume must be >= 0, got {volume}")

    @property
    def open(self) -> Decimal:
        return _parse_payload(self.payload)[0]

    @property
    def high(self) -> Decimal:
        return _parse_payload(self.payload)[1]

    @property
    def low(self) -> Decimal:
        return _parse_payload(self.payload)[2]

    @property
    def close(self) -> Decimal:
        return _parse_payload(self.payload)[3]

    @property
    def volume(self) -> Decimal | None:
        return _parse_payload(self.payload)[4]

    @property
    def close_time(self) -> datetime:
        return compute_close_time(self.event_time, self.timeframe)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CANDLE_KIND, "event_id": self.event_id, "instrument_id": self.instrument_id, "provider": self.provider,
            "symbol": self.symbol, "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
            "arrival_time": serialize_timestamp(self.arrival_time, field_name="arrival_time"), "timeframe": self.timeframe.value,
            "sequence": self.sequence, "source_event_id": self.source_event_id, "payload": self.payload,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Candle:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            event_id=str(raw["event_id"]), instrument_id=str(raw["instrument_id"]), provider=str(raw["provider"]),
            symbol=str(raw["symbol"]), event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
            arrival_time=deserialize_timestamp(raw["arrival_time"], field_name="arrival_time"),
            timeframe=Timeframe(raw["timeframe"]), sequence=int(str(raw["sequence"])),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            payload=as_json_dict(raw["payload"], field_name="payload"),
        )


def create_candle(
    *, instrument_id: str, provider: str, symbol: str, event_time: datetime, timeframe: Timeframe, sequence: int,
    open: Decimal, high: Decimal, low: Decimal, close: Decimal, volume: Decimal | None = None,
    arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Candle:
    """The only supported way to mint a fresh `Candle` -- computes its
    deterministic `event_id` from every other field before constructing
    it. `arrival_time` defaults to `event_time` (a batch/backfill ingest
    has no separately observable arrival latency)."""
    resolved_arrival_time = event_time if arrival_time is None else arrival_time
    payload = _payload_to_json(open_=open, high=high, low=low, close=close, volume=volume)
    provisional = Candle(
        event_id="0" * 64, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=timeframe, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
    event_id = compute_content_id(CANDLE_KIND, provisional.to_identity_payload())
    return Candle(
        event_id=event_id, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=timeframe, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )


# --------------------------------------------------------------------------
# Pure structural math -- shared by `quality.py` (heuristic checks) and
# `feature_generation.py` (high-low range / body size / wick ratio
# features), so the arithmetic is defined exactly once.
# --------------------------------------------------------------------------
def candle_range(candle: Candle) -> Decimal:
    return candle.high - candle.low


def candle_body_size(candle: Candle) -> Decimal:
    return abs(candle.close - candle.open)


def candle_upper_wick(candle: Candle) -> Decimal:
    return candle.high - max(candle.open, candle.close)


def candle_lower_wick(candle: Candle) -> Decimal:
    return min(candle.open, candle.close) - candle.low


def candle_is_bullish(candle: Candle) -> bool:
    return candle.close >= candle.open
