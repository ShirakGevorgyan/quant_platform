"""Unit tests for `market_data.candles`/`market_data.ticks`/
`market_data.events`: construction/validation, content identity, JSON
round-tripping, and `MarketEventStore`'s append-only/idempotent/
conflict/gap semantics."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import MarketDataError, MarketDataEventError, MarketDataPersistenceError
from quant_platform.core.types import OrderSide, Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.events import (
    MarketEventStore,
    create_quote,
    create_trade,
    market_data_event_from_json_dict,
)
from quant_platform.market_data.ticks import create_tick

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _candle(**overrides: object):
    base: dict[str, object] = {
        "instrument_id": "mt5__XAUUSD", "provider": "mt5", "symbol": "XAUUSD", "event_time": _T0, "timeframe": Timeframe.H1,
        "sequence": 0, "open": Decimal("2000"), "high": Decimal("2005"), "low": Decimal("1995"), "close": Decimal("2002"),
        "volume": Decimal("100"),
    }
    base.update(overrides)
    return create_candle(**base)  # type: ignore[arg-type]


class TestCandleConstruction:
    def test_valid_candle_round_trips_through_json(self) -> None:
        candle = _candle()
        restored = type(candle).from_json_dict(candle.to_json_dict())
        assert restored == candle

    def test_two_calls_with_identical_arguments_produce_the_same_id(self) -> None:
        assert _candle().event_id == _candle().event_id

    def test_changing_any_field_changes_the_id(self) -> None:
        assert _candle().event_id != _candle(close=Decimal("2003")).event_id

    def test_high_less_than_low_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            _candle(high=Decimal("1990"), low=Decimal("1995"))

    def test_open_outside_high_low_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            _candle(open=Decimal("2010"))

    def test_close_outside_high_low_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            _candle(close=Decimal("1990"))

    def test_negative_volume_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            _candle(volume=Decimal("-1"))

    def test_naive_event_time_is_rejected(self) -> None:
        with pytest.raises(MarketDataError):
            _candle(event_time=datetime(2026, 1, 5))

    def test_close_time_is_derived_from_open_time_and_timeframe(self) -> None:
        candle = _candle(timeframe=Timeframe.M15)
        assert candle.close_time == _T0 + timedelta(minutes=15)

    def test_arrival_time_defaults_to_event_time(self) -> None:
        assert _candle().arrival_time == _T0

    def test_arrival_time_before_event_time_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            _candle(arrival_time=_T0 - timedelta(seconds=1))


class TestTickConstruction:
    def test_round_trips_through_json(self) -> None:
        tick = create_tick(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=Decimal("2000.5"), volume=Decimal("1"))
        assert type(tick).from_json_dict(tick.to_json_dict()) == tick

    def test_non_positive_price_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            create_tick(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=Decimal("0"))

    def test_timeframe_is_always_none(self) -> None:
        tick = create_tick(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=Decimal("1"))
        assert tick.timeframe is None


class TestQuoteConstruction:
    def test_round_trips_through_json(self) -> None:
        quote = create_quote(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, bid=Decimal("1999"), ask=Decimal("2001"))
        assert type(quote).from_json_dict(quote.to_json_dict()) == quote

    def test_ask_below_bid_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            create_quote(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, bid=Decimal("2001"), ask=Decimal("1999"))


class TestTradeConstruction:
    def test_round_trips_through_json(self) -> None:
        trade = create_trade(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=Decimal("2000"), size=Decimal("1"), side=OrderSide.BUY)
        assert type(trade).from_json_dict(trade.to_json_dict()) == trade

    def test_non_positive_size_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            create_trade(instrument_id="i", provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=Decimal("2000"), size=Decimal("0"))


class TestMarketDataEventDispatch:
    def test_from_json_dict_dispatches_on_kind(self) -> None:
        candle = _candle()
        restored = market_data_event_from_json_dict(candle.to_json_dict())
        assert restored == candle

    def test_unknown_kind_is_rejected(self) -> None:
        raw = dict(_candle().to_json_dict())
        raw["kind"] = "unknown_kind"
        with pytest.raises(MarketDataEventError):
            market_data_event_from_json_dict(raw)


class TestMarketEventStore:
    def test_append_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            candle = _candle(sequence=0)
            store.append(candle)
            events = store.read_events("mt5", "mt5__XAUUSD")
            assert events == [candle]

    def test_sequence_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            with pytest.raises(MarketDataPersistenceError):
                store.append(_candle(sequence=1))

    def test_identical_reappend_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            candle = _candle(sequence=0)
            store.append(candle)
            store.append(candle)
            assert len(store.read_events("mt5", "mt5__XAUUSD")) == 1

    def test_conflicting_append_at_same_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            store.append(_candle(sequence=0))
            with pytest.raises(MarketDataPersistenceError):
                store.append(_candle(sequence=0, open=Decimal("2001")))

    def test_multiple_event_kinds_interleave_in_one_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            candle = _candle(sequence=0)
            store.append(candle)
            tick = create_tick(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(minutes=1), sequence=1, price=Decimal("2001"))
            store.append(tick)
            events = store.read_events("mt5", "mt5__XAUUSD")
            assert events == [candle, tick]

    def test_missing_partition_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            assert store.read_events("mt5", "does-not-exist") == []

    def test_next_sequence_reflects_store_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            assert store.next_sequence("mt5", "mt5__XAUUSD") == 0
            store.append(_candle(sequence=0))
            assert store.next_sequence("mt5", "mt5__XAUUSD") == 1
