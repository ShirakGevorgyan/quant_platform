"""`Quote`/`Trade` events, the `MarketDataEvent` union, and the append-only
raw event store (Milestone 10, Phase 1). See `candles.py`'s module
docstring for the shared envelope-plus-typed-payload design every event
kind in this package follows.

This module sits ABOVE `candles.py`/`ticks.py` in the dependency graph
(it imports `Candle`/`Tick` to build the union and dispatch table) --
never the other way around, which is why the shared validators live in
`identity.py` rather than here (see that module's docstring).

`MarketEventStore` deliberately has NO separate hash-chain field (unlike
`portfolio_risk.ledger.PortfolioRiskLedgerStore`'s `previous_entry_hash`):
every event already carries its own content-addressed `event_id`, so
physical append-order integrity reduces to "the sequence numbers are
gapless and no id repeats with different content" -- `verification.py`
checks exactly that. A separate chain field would only re-prove what the
per-event id already proves."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    MarketDataEventError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.core.types import OrderSide
from quant_platform.market_data.candles import CANDLE_KIND, Candle
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_non_negative_sequence,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.ticks import TICK_KIND, Tick
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import as_json_dict, parse_json_strict

__all__ = [
    "MarketDataEvent",
    "MarketDataEventKind",
    "MarketEventStore",
    "Quote",
    "Trade",
    "create_quote",
    "create_trade",
    "market_data_event_from_json_dict",
    "market_data_event_id",
    "market_data_event_sequence",
    "market_data_event_time",
]


class MarketDataEventKind(Enum):
    TICK = TICK_KIND
    QUOTE = "quote"
    TRADE = "trade"
    CANDLE = CANDLE_KIND


# --------------------------------------------------------------------------
# Quote
# --------------------------------------------------------------------------
_QUOTE_PAYLOAD_KEYS = ("bid", "ask", "bid_size", "ask_size")


def _quote_payload_to_json(*, bid: Decimal, ask: Decimal, bid_size: Decimal | None, ask_size: Decimal | None) -> dict[str, object]:
    return {"bid": str(bid), "ask": str(ask), "bid_size": (None if bid_size is None else str(bid_size)), "ask_size": (None if ask_size is None else str(ask_size))}


def _parse_quote_payload(payload: dict[str, object]) -> tuple[Decimal, Decimal, Decimal | None, Decimal | None]:
    extra_keys = set(payload.keys()) - set(_QUOTE_PAYLOAD_KEYS)
    if extra_keys:
        raise MarketDataEventError(f"Quote.payload has unexpected key(s): {sorted(extra_keys)}")
    missing_keys = set(_QUOTE_PAYLOAD_KEYS) - set(payload.keys())
    if missing_keys:
        raise MarketDataEventError(f"Quote.payload is missing required key(s): {sorted(missing_keys)}")
    bid = parse_decimal(payload["bid"], field_name="Quote.payload['bid']")
    ask = parse_decimal(payload["ask"], field_name="Quote.payload['ask']")
    raw_bid_size = payload["bid_size"]
    raw_ask_size = payload["ask_size"]
    bid_size = None if raw_bid_size is None else parse_decimal(raw_bid_size, field_name="Quote.payload['bid_size']")
    ask_size = None if raw_ask_size is None else parse_decimal(raw_ask_size, field_name="Quote.payload['ask_size']")
    return bid, ask, bid_size, ask_size


@dataclass(frozen=True, slots=True)
class Quote:
    event_id: str
    instrument_id: str
    provider: str
    symbol: str
    event_time: datetime
    arrival_time: datetime
    timeframe: None
    sequence: int
    source_event_id: str | None
    payload: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.instrument_id, field_name="Quote.instrument_id")
        require_non_empty(self.provider, field_name="Quote.provider")
        require_non_empty(self.symbol, field_name="Quote.symbol")
        require_tz_aware(self.event_time, field_name="Quote.event_time")
        require_tz_aware(self.arrival_time, field_name="Quote.arrival_time")
        if self.arrival_time < self.event_time:
            raise MarketDataEventError(f"Quote.arrival_time ({self.arrival_time}) must be >= event_time ({self.event_time})")
        if self.timeframe is not None:
            raise MarketDataEventError("Quote.timeframe must be None -- a quote has no timeframe")
        require_non_negative_sequence(self.sequence)
        bid, ask, bid_size, ask_size = _parse_quote_payload(self.payload)
        if bid <= 0:
            raise MarketDataEventError(f"Quote.bid must be > 0, got {bid}")
        if ask <= 0:
            raise MarketDataEventError(f"Quote.ask must be > 0, got {ask}")
        if ask < bid:
            raise MarketDataEventError(f"Quote.ask ({ask}) must be >= bid ({bid})")
        if bid_size is not None and bid_size < 0:
            raise MarketDataEventError(f"Quote.bid_size must be >= 0, got {bid_size}")
        if ask_size is not None and ask_size < 0:
            raise MarketDataEventError(f"Quote.ask_size must be >= 0, got {ask_size}")

    @property
    def bid(self) -> Decimal:
        return _parse_quote_payload(self.payload)[0]

    @property
    def ask(self) -> Decimal:
        return _parse_quote_payload(self.payload)[1]

    @property
    def bid_size(self) -> Decimal | None:
        return _parse_quote_payload(self.payload)[2]

    @property
    def ask_size(self) -> Decimal | None:
        return _parse_quote_payload(self.payload)[3]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketDataEventKind.QUOTE.value, "event_id": self.event_id, "instrument_id": self.instrument_id,
            "provider": self.provider, "symbol": self.symbol, "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
            "arrival_time": serialize_timestamp(self.arrival_time, field_name="arrival_time"), "timeframe": None,
            "sequence": self.sequence, "source_event_id": self.source_event_id, "payload": self.payload,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Quote:
        if raw.get("timeframe") is not None:
            raise MarketDataEventError(f"Quote.timeframe must be None, got {raw.get('timeframe')!r}")
        return cls(
            event_id=str(raw["event_id"]), instrument_id=str(raw["instrument_id"]), provider=str(raw["provider"]),
            symbol=str(raw["symbol"]), event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
            arrival_time=deserialize_timestamp(raw["arrival_time"], field_name="arrival_time"), timeframe=None,
            sequence=int(str(raw["sequence"])),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            payload=as_json_dict(raw["payload"], field_name="payload"),
        )


def create_quote(
    *, instrument_id: str, provider: str, symbol: str, event_time: datetime, sequence: int, bid: Decimal, ask: Decimal,
    bid_size: Decimal | None = None, ask_size: Decimal | None = None, arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Quote:
    resolved_arrival_time = event_time if arrival_time is None else arrival_time
    payload = _quote_payload_to_json(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)
    provisional = Quote(
        event_id="0" * 64, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
    event_id = compute_content_id(MarketDataEventKind.QUOTE.value, provisional.to_identity_payload())
    return Quote(
        event_id=event_id, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )


# --------------------------------------------------------------------------
# Trade
# --------------------------------------------------------------------------
_TRADE_PAYLOAD_KEYS = ("price", "size", "side")


def _trade_payload_to_json(*, price: Decimal, size: Decimal, side: OrderSide | None) -> dict[str, object]:
    return {"price": str(price), "size": str(size), "side": (None if side is None else side.value)}


def _parse_trade_payload(payload: dict[str, object]) -> tuple[Decimal, Decimal, OrderSide | None]:
    extra_keys = set(payload.keys()) - set(_TRADE_PAYLOAD_KEYS)
    if extra_keys:
        raise MarketDataEventError(f"Trade.payload has unexpected key(s): {sorted(extra_keys)}")
    missing_keys = set(_TRADE_PAYLOAD_KEYS) - set(payload.keys())
    if missing_keys:
        raise MarketDataEventError(f"Trade.payload is missing required key(s): {sorted(missing_keys)}")
    price = parse_decimal(payload["price"], field_name="Trade.payload['price']")
    size = parse_decimal(payload["size"], field_name="Trade.payload['size']")
    raw_side = payload["side"]
    side = None if raw_side is None else OrderSide(raw_side)
    return price, size, side


@dataclass(frozen=True, slots=True)
class Trade:
    event_id: str
    instrument_id: str
    provider: str
    symbol: str
    event_time: datetime
    arrival_time: datetime
    timeframe: None
    sequence: int
    source_event_id: str | None
    payload: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.instrument_id, field_name="Trade.instrument_id")
        require_non_empty(self.provider, field_name="Trade.provider")
        require_non_empty(self.symbol, field_name="Trade.symbol")
        require_tz_aware(self.event_time, field_name="Trade.event_time")
        require_tz_aware(self.arrival_time, field_name="Trade.arrival_time")
        if self.arrival_time < self.event_time:
            raise MarketDataEventError(f"Trade.arrival_time ({self.arrival_time}) must be >= event_time ({self.event_time})")
        if self.timeframe is not None:
            raise MarketDataEventError("Trade.timeframe must be None -- a trade has no timeframe")
        require_non_negative_sequence(self.sequence)
        price, size, _side = _parse_trade_payload(self.payload)
        if price <= 0:
            raise MarketDataEventError(f"Trade.price must be > 0, got {price}")
        if size <= 0:
            raise MarketDataEventError(f"Trade.size must be > 0, got {size}")

    @property
    def price(self) -> Decimal:
        return _parse_trade_payload(self.payload)[0]

    @property
    def size(self) -> Decimal:
        return _parse_trade_payload(self.payload)[1]

    @property
    def side(self) -> OrderSide | None:
        return _parse_trade_payload(self.payload)[2]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketDataEventKind.TRADE.value, "event_id": self.event_id, "instrument_id": self.instrument_id,
            "provider": self.provider, "symbol": self.symbol, "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
            "arrival_time": serialize_timestamp(self.arrival_time, field_name="arrival_time"), "timeframe": None,
            "sequence": self.sequence, "source_event_id": self.source_event_id, "payload": self.payload,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Trade:
        if raw.get("timeframe") is not None:
            raise MarketDataEventError(f"Trade.timeframe must be None, got {raw.get('timeframe')!r}")
        return cls(
            event_id=str(raw["event_id"]), instrument_id=str(raw["instrument_id"]), provider=str(raw["provider"]),
            symbol=str(raw["symbol"]), event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
            arrival_time=deserialize_timestamp(raw["arrival_time"], field_name="arrival_time"), timeframe=None,
            sequence=int(str(raw["sequence"])),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            payload=as_json_dict(raw["payload"], field_name="payload"),
        )


def create_trade(
    *, instrument_id: str, provider: str, symbol: str, event_time: datetime, sequence: int, price: Decimal, size: Decimal,
    side: OrderSide | None = None, arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Trade:
    resolved_arrival_time = event_time if arrival_time is None else arrival_time
    payload = _trade_payload_to_json(price=price, size=size, side=side)
    provisional = Trade(
        event_id="0" * 64, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
    event_id = compute_content_id(MarketDataEventKind.TRADE.value, provisional.to_identity_payload())
    return Trade(
        event_id=event_id, instrument_id=instrument_id, provider=provider, symbol=symbol, event_time=event_time,
        arrival_time=resolved_arrival_time, timeframe=None, sequence=sequence, source_event_id=source_event_id, payload=payload,
    )


# --------------------------------------------------------------------------
# Union, dispatch, and shared accessors.
# --------------------------------------------------------------------------
MarketDataEvent = Tick | Quote | Trade | Candle

_EVENT_CLASSES_BY_KIND: dict[str, type[MarketDataEvent]] = {
    MarketDataEventKind.TICK.value: Tick,
    MarketDataEventKind.QUOTE.value: Quote,
    MarketDataEventKind.TRADE.value: Trade,
    MarketDataEventKind.CANDLE.value: Candle,
}


def market_data_event_from_json_dict(raw: dict[str, object]) -> MarketDataEvent:
    kind = raw.get("kind")
    if kind not in _EVENT_CLASSES_BY_KIND:
        raise MarketDataEventError(f"Unknown market data event kind {kind!r}")
    return _EVENT_CLASSES_BY_KIND[str(kind)].from_json_dict(raw)


def market_data_event_id(event: MarketDataEvent) -> str:
    return event.event_id


def market_data_event_time(event: MarketDataEvent) -> datetime:
    """The single ordering timestamp every event kind exposes for
    cross-event ordering checks (`quality.py`). A `Candle`'s `event_time`
    is its OPEN time; unlike `paper_trading.events.market_event_time`
    (which returns a bar's CLOSE time for look-ahead safety in a live
    execution context), this package's replay/quality concerns care about
    the bar's own INDEX timestamp, not when its data became fully known
    -- point-in-time safety for a downstream consumer is that consumer's
    own responsibility (see `docs/market_data_architecture.md`)."""
    return event.event_time


def market_data_event_sequence(event: MarketDataEvent) -> int:
    return event.sequence


# --------------------------------------------------------------------------
# MarketEventStore -- append-only, partitioned by (provider, instrument_id).
# All four event kinds interleave in one partition (sequence-ordered),
# since `replay.py` needs the FULL raw stream for one instrument to
# rebuild derived features deterministically.
# --------------------------------------------------------------------------
@contextmanager
def market_data_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire market data event lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        # Mirrors `portfolio_risk.ledger.portfolio_risk_lock`'s identical,
        # already-documented Windows release-side sharing-violation
        # translation -- `historical.locking` is shared infrastructure
        # outside this package's own scope to modify.
        raise MarketDataLockError(f"Market data event lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class MarketEventStore:
    """Storage layout: `{storage_root}/market_events/{provider}/
    {instrument_id}/events.jsonl`."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _partition_dir(self, provider: str, instrument_id: str) -> Path:
        return self._root / "market_events" / provider / instrument_id

    def _events_path(self, provider: str, instrument_id: str) -> Path:
        return self._partition_dir(provider, instrument_id) / "events.jsonl"

    def events_path(self, provider: str, instrument_id: str) -> Path:
        """Public accessor for `recovery.py`'s tolerant-read path -- the
        only legitimate reason outside this class to know the physical
        file location; ordinary reads always go through `read_events`."""
        return self._events_path(provider, instrument_id)

    def _lock_path(self, provider: str, instrument_id: str) -> Path:
        return self._partition_dir(provider, instrument_id) / ".events.lock"

    def read_events(self, provider: str, instrument_id: str) -> list[MarketDataEvent]:
        path = self._events_path(provider, instrument_id)
        if not path.is_file():
            return []
        events: list[MarketDataEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted market event line for {provider}/{instrument_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted market event line for {provider}/{instrument_id}: expected a JSON object")
            events.append(market_data_event_from_json_dict(raw))
        return events

    def append(self, event: MarketDataEvent) -> MarketDataEvent:
        provider, instrument_id = event.provider, event.instrument_id
        lock_path = self._lock_path(provider, instrument_id)
        self._partition_dir(provider, instrument_id).mkdir(parents=True, exist_ok=True)
        with market_data_lock(lock_path):
            existing_events = self.read_events(provider, instrument_id)
            if event.sequence < len(existing_events):
                existing = existing_events[event.sequence]
                if market_data_event_id(existing) == market_data_event_id(event):
                    return existing  # idempotent no-op: identical append
                raise MarketDataPersistenceError(
                    f"Conflicting append at sequence {event.sequence} for {provider}/{instrument_id}: existing event_id "
                    f"{market_data_event_id(existing)!r} != new event_id {market_data_event_id(event)!r}"
                )
            if event.sequence > len(existing_events):
                raise MarketDataPersistenceError(
                    f"Event sequence {event.sequence} leaves a gap for {provider}/{instrument_id} (expected {len(existing_events)})"
                )
            path = self._events_path(provider, instrument_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(event.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def next_sequence(self, provider: str, instrument_id: str) -> int:
        return len(self.read_events(provider, instrument_id))
