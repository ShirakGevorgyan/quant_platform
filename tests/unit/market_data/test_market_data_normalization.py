"""Unit tests for `market_data.normalization`: raw-input tolerance
(str/int/float), instrument_id derivation, and delegation to the
trusted `create_*` factories' own validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import MarketDataError, MarketDataEventError
from quant_platform.core.types import OrderSide, Timeframe
from quant_platform.market_data.normalization import (
    derive_instrument_id,
    normalize_candle_row,
    normalize_quote_row,
    normalize_tick_row,
    normalize_trade_row,
)

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


class TestDeriveInstrumentId:
    def test_joins_provider_and_symbol_with_a_path_safe_separator(self) -> None:
        assert derive_instrument_id(provider="mt5", symbol="XAUUSD") == "mt5__XAUUSD"

    def test_never_contains_a_colon(self) -> None:
        # A ':' is a reserved drive-separator character on Windows and
        # would break MarketEventStore/FeatureStore's own path-based
        # partitioning (see normalization.py's own docstring).
        assert ":" not in derive_instrument_id(provider="mt5", symbol="XAUUSD")


class TestNormalizeCandleRow:
    def test_accepts_string_and_float_numeric_input(self) -> None:
        candle = normalize_candle_row(
            provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=0, open="2000.5", high=2005.25, low="1995", close=2001,
        )
        assert candle.open.as_tuple() is not None

    def test_derives_instrument_id_when_not_supplied(self) -> None:
        candle = normalize_candle_row(provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=0, open="1", high="2", low="1", close="1")
        assert candle.instrument_id == "mt5__XAUUSD"

    def test_non_positive_open_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            normalize_candle_row(provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=0, open="0", high="2", low="0", close="1")

    def test_nan_input_is_rejected(self) -> None:
        with pytest.raises(MarketDataError):
            normalize_candle_row(provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=0, open=float("nan"), high="2", low="1", close="1")


class TestNormalizeTickRow:
    def test_accepts_int_price(self) -> None:
        tick = normalize_tick_row(provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price=2000)
        assert str(tick.price) == "2000"


class TestNormalizeQuoteRow:
    def test_ask_below_bid_is_rejected(self) -> None:
        with pytest.raises(MarketDataEventError):
            normalize_quote_row(provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, bid="2001", ask="1999")


class TestNormalizeTradeRow:
    def test_accepts_side_as_string(self) -> None:
        trade = normalize_trade_row(provider="mt5", symbol="XAUUSD", event_time=_T0, sequence=0, price="2000", size="1", side="BUY")
        assert trade.side is OrderSide.BUY
