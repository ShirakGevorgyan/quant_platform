"""Ingestion boundary: turns raw, untrusted provider data (a plain
mapping -- one CSV/DataFrame row, one JSON record from a vendor feed)
into this package's strict, validated, content-addressed domain objects
(`Tick`/`Quote`/`Trade`/`Candle`). `events.py`/`candles.py`/`ticks.py`
define the TRUSTED domain model and its `create_*` factories, which
require already-clean `Decimal` values and a resolved `instrument_id`;
this module is the layer in front of them that tolerates raw
`str`/`int`/`float` numeric input and derives `instrument_id` when a
caller has not already resolved one.

`instrument_id` defaults to `f"{provider}:{symbol}"` when not explicitly
supplied: Phase 1 has no separate instrument registry/master-data
service (out of this milestone's scope -- see `docs/
market_data_architecture.md`'s Known Limitations), so a provider+symbol
pair is the only identity Phase 1 can derive deterministically. A future
milestone that adds a real instrument registry can supply `instrument_id`
explicitly without changing this module's signature.

Every `normalize_*` function raises `MarketDataEventError` (via the
`create_*` factory it delegates to, or its own upfront checks) on the
FIRST invalid row -- for validating a whole batch and collecting every
issue rather than stopping at the first one, see `quality.py`."""

from __future__ import annotations

from datetime import datetime

from quant_platform.core.exceptions import MarketDataEventError
from quant_platform.core.types import OrderSide, Timeframe
from quant_platform.market_data.candles import Candle, create_candle
from quant_platform.market_data.events import Quote, Trade, create_quote, create_trade
from quant_platform.market_data.identity import parse_decimal, require_non_empty
from quant_platform.market_data.ticks import Tick, create_tick

__all__ = ["derive_instrument_id", "normalize_candle_row", "normalize_quote_row", "normalize_tick_row", "normalize_trade_row"]


def derive_instrument_id(*, provider: str, symbol: str) -> str:
    """`__` (not `:` or `/`) joins `provider` and `symbol` -- both
    `MarketEventStore` and `FeatureStore` use `instrument_id` directly as
    a filesystem path component, and `:` is a reserved drive-separator
    character on Windows (confirmed via a `NotADirectoryError` during
    this phase's own smoke testing); `__` is safe on every platform this
    repository targets."""
    require_non_empty(provider, field_name="provider")
    require_non_empty(symbol, field_name="symbol")
    return f"{provider}__{symbol}"


def _resolve_instrument_id(*, provider: str, symbol: str, instrument_id: str | None) -> str:
    return derive_instrument_id(provider=provider, symbol=symbol) if instrument_id is None else instrument_id


def normalize_tick_row(
    *, provider: str, symbol: str, event_time: datetime, sequence: int, price: object, volume: object = None,
    instrument_id: str | None = None, arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Tick:
    resolved_instrument_id = _resolve_instrument_id(provider=provider, symbol=symbol, instrument_id=instrument_id)
    parsed_price = parse_decimal(price, field_name="price")
    parsed_volume = None if volume is None else parse_decimal(volume, field_name="volume")
    return create_tick(
        instrument_id=resolved_instrument_id, provider=provider, symbol=symbol, event_time=event_time, sequence=sequence,
        price=parsed_price, volume=parsed_volume, arrival_time=arrival_time, source_event_id=source_event_id,
    )


def normalize_quote_row(
    *, provider: str, symbol: str, event_time: datetime, sequence: int, bid: object, ask: object,
    bid_size: object = None, ask_size: object = None, instrument_id: str | None = None,
    arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> Quote:
    resolved_instrument_id = _resolve_instrument_id(provider=provider, symbol=symbol, instrument_id=instrument_id)
    parsed_bid = parse_decimal(bid, field_name="bid")
    parsed_ask = parse_decimal(ask, field_name="ask")
    parsed_bid_size = None if bid_size is None else parse_decimal(bid_size, field_name="bid_size")
    parsed_ask_size = None if ask_size is None else parse_decimal(ask_size, field_name="ask_size")
    return create_quote(
        instrument_id=resolved_instrument_id, provider=provider, symbol=symbol, event_time=event_time, sequence=sequence,
        bid=parsed_bid, ask=parsed_ask, bid_size=parsed_bid_size, ask_size=parsed_ask_size, arrival_time=arrival_time,
        source_event_id=source_event_id,
    )


def normalize_trade_row(
    *, provider: str, symbol: str, event_time: datetime, sequence: int, price: object, size: object,
    side: OrderSide | str | None = None, instrument_id: str | None = None, arrival_time: datetime | None = None,
    source_event_id: str | None = None,
) -> Trade:
    resolved_instrument_id = _resolve_instrument_id(provider=provider, symbol=symbol, instrument_id=instrument_id)
    parsed_price = parse_decimal(price, field_name="price")
    parsed_size = parse_decimal(size, field_name="size")
    resolved_side = OrderSide(side) if isinstance(side, str) else side
    return create_trade(
        instrument_id=resolved_instrument_id, provider=provider, symbol=symbol, event_time=event_time, sequence=sequence,
        price=parsed_price, size=parsed_size, side=resolved_side, arrival_time=arrival_time, source_event_id=source_event_id,
    )


def normalize_candle_row(
    *, provider: str, symbol: str, event_time: datetime, timeframe: Timeframe, sequence: int, open: object, high: object,
    low: object, close: object, volume: object = None, instrument_id: str | None = None, arrival_time: datetime | None = None,
    source_event_id: str | None = None,
) -> Candle:
    resolved_instrument_id = _resolve_instrument_id(provider=provider, symbol=symbol, instrument_id=instrument_id)
    parsed_open = parse_decimal(open, field_name="open")
    parsed_high = parse_decimal(high, field_name="high")
    parsed_low = parse_decimal(low, field_name="low")
    parsed_close = parse_decimal(close, field_name="close")
    parsed_volume = None if volume is None else parse_decimal(volume, field_name="volume")
    if parsed_open <= 0 or parsed_high <= 0 or parsed_low <= 0 or parsed_close <= 0:
        raise MarketDataEventError("Candle open/high/low/close must all be > 0")
    return create_candle(
        instrument_id=resolved_instrument_id, provider=provider, symbol=symbol, event_time=event_time, timeframe=timeframe,
        sequence=sequence, open=parsed_open, high=parsed_high, low=parsed_low, close=parsed_close, volume=parsed_volume,
        arrival_time=arrival_time, source_event_id=source_event_id,
    )
