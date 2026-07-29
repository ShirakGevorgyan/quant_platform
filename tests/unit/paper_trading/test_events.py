"""Milestone 7, Section 5: normalized market-event model. Covers per-kind
construction, validation (finite/positive prices, ask>=bid, OHLC bounds,
timezone-aware timestamps, non-negative sequence, non-empty required
strings, duplicate quality-flag rejection), deterministic content-
addressed identity, and lossless JSON round-trip via the `market_event_
from_json_dict` dispatcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import MarketEventError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.events import (
    BarEvent,
    QuoteEvent,
    create_bar_event,
    create_end_of_stream_event,
    create_financing_event,
    create_quote_event,
    create_session_close_event,
    create_session_open_event,
    create_trading_halt_event,
    create_trading_resume_event,
    market_event_from_json_dict,
    market_event_time,
)
from quant_platform.paper_trading.models import MarketEventQualityFlagKind

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)


class TestQuoteEvent:
    def test_construction_succeeds_with_valid_fields(self) -> None:
        event = create_quote_event(instrument="HYPOTHETICAL_XAU", event_time=_T0, sequence=1, bid=1900.10, ask=1900.30, source="test_source")
        assert event.bid == 1900.10
        assert event.ask == 1900.30

    def test_identical_arguments_produce_identical_event_id(self) -> None:
        a = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        b = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        assert a.event_id == b.event_id

    def test_different_sequence_produces_different_event_id(self) -> None:
        a = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        b = create_quote_event(instrument="X", event_time=_T0, sequence=2, bid=1.0, ask=1.1, source="s")
        assert a.event_id != b.event_id

    def test_ask_below_bid_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="ask"):
            create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.1, ask=1.0, source="s")

    def test_non_positive_bid_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="bid"):
            create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=0.0, ask=1.0, source="s")

    def test_non_finite_ask_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="ask"):
            create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=float("nan"), source="s")

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="timezone-aware"):
            create_quote_event(instrument="X", event_time=datetime(2026, 1, 5, 10, 0, 0), sequence=1, bid=1.0, ask=1.1, source="s")

    def test_naive_receive_time_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="timezone-aware"):
            create_quote_event(instrument="X", event_time=_T0, receive_time=datetime(2026, 1, 5, 10, 0, 1), sequence=1, bid=1.0, ask=1.1, source="s")

    def test_receive_time_before_event_time_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="receive_time"):
            create_quote_event(instrument="X", event_time=_T0, receive_time=_T0 - timedelta(seconds=1), sequence=1, bid=1.0, ask=1.1, source="s")

    def test_receive_time_equal_to_event_time_allowed(self) -> None:
        event = create_quote_event(instrument="X", event_time=_T0, receive_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        assert event.receive_time == _T0

    def test_negative_sequence_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="sequence"):
            create_quote_event(instrument="X", event_time=_T0, sequence=-1, bid=1.0, ask=1.1, source="s")

    def test_empty_instrument_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="instrument"):
            create_quote_event(instrument="", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")

    def test_negative_bid_size_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="bid_size"):
            create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s", bid_size=-1.0)

    def test_duplicate_quality_flags_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="quality_flags"):
            create_quote_event(
                instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s",
                quality_flags=(MarketEventQualityFlagKind.STALE, MarketEventQualityFlagKind.STALE),
            )

    def test_json_round_trip_preserves_all_fields(self) -> None:
        original = create_quote_event(
            instrument="X", event_time=_T0, receive_time=_T0, sequence=3, bid=1.0, ask=1.1, source="s", bid_size=10.0, ask_size=12.0,
            source_event_identity="src-1", quality_flags=(MarketEventQualityFlagKind.WIDE_SPREAD, MarketEventQualityFlagKind.STALE),
        )
        roundtripped = QuoteEvent.from_json_dict(original.to_json_dict())
        assert roundtripped == original

    def test_round_trip_preserves_declared_quality_flag_order(self) -> None:
        original = create_quote_event(
            instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s",
            quality_flags=(MarketEventQualityFlagKind.WIDE_SPREAD, MarketEventQualityFlagKind.STALE),
        )
        assert original.to_json_dict()["quality_flags"] == ["wide_spread", "stale"]

    def test_reordered_quality_flags_produce_identical_event_id(self) -> None:
        a = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s", quality_flags=(MarketEventQualityFlagKind.WIDE_SPREAD, MarketEventQualityFlagKind.STALE))
        b = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s", quality_flags=(MarketEventQualityFlagKind.STALE, MarketEventQualityFlagKind.WIDE_SPREAD))
        assert a.event_id == b.event_id

    def test_dispatcher_reconstructs_quote_event(self) -> None:
        original = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_market_event_time_returns_event_time(self) -> None:
        event = create_quote_event(instrument="X", event_time=_T0, sequence=1, bid=1.0, ask=1.1, source="s")
        assert market_event_time(event) == _T0


class TestBarEvent:
    def test_construction_succeeds_with_valid_fields(self) -> None:
        event = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s")
        assert event.close_time == _T0 + Timeframe.H1.duration

    def test_high_below_close_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="high"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=101.0, low=99.0, close=102.0, sequence=1, source="s")

    def test_low_above_open_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="low"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=100.5, close=102.0, sequence=1, source="s")

    def test_non_positive_open_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="open"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=0.0, high=1.0, low=0.0, close=0.5, sequence=1, source="s")

    def test_ask_close_below_bid_close_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="ask_close"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s", bid_close=101.9, ask_close=101.8)

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="volume"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s", volume=-1.0)

    def test_close_time_mismatch_rejected_directly(self) -> None:
        """`BarEvent` may also be constructed directly (not only via the
        factory) -- e.g. by `from_json_dict` reloading a persisted ledger
        entry -- and must reject an inconsistent `close_time` either way."""
        with pytest.raises(MarketEventError, match="close_time"):
            BarEvent(
                event_id="0" * 64, instrument="X", interval=Timeframe.H1, open_time=_T0, close_time=_T0 + timedelta(hours=2), open=100.0,
                high=105.0, low=99.0, close=102.0, volume=None, bid_close=None, ask_close=None, is_complete=True, sequence=1, source="s",
                source_event_identity=None, quality_flags=(),
            )

    def test_naive_open_time_rejected(self) -> None:
        with pytest.raises(MarketEventError, match="timezone-aware"):
            create_bar_event(instrument="X", interval=Timeframe.H1, open_time=datetime(2026, 1, 5, 10, 0, 0), open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s")

    def test_json_round_trip_preserves_all_fields(self) -> None:
        original = create_bar_event(
            instrument="X", interval=Timeframe.M15, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=7, source="s",
            volume=1234.0, bid_close=101.9, ask_close=102.1, is_complete=False, source_event_identity="src-7", quality_flags=(MarketEventQualityFlagKind.NORMALIZED_FROM_SOURCE,),
        )
        roundtripped = BarEvent.from_json_dict(original.to_json_dict())
        assert roundtripped == original

    def test_market_event_time_returns_close_time_not_open_time(self) -> None:
        event = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s")
        assert market_event_time(event) == event.close_time
        assert market_event_time(event) != event.open_time

    def test_dispatcher_reconstructs_bar_event(self) -> None:
        original = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original


class TestSessionBoundaryHaltResumeFinancingEndOfStreamEvents:
    def test_session_open_round_trips(self) -> None:
        original = create_session_open_event(instrument="X", event_time=_T0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_session_close_round_trips(self) -> None:
        original = create_session_close_event(instrument="X", event_time=_T0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_trading_halt_requires_non_empty_reason(self) -> None:
        with pytest.raises(MarketEventError, match="reason"):
            create_trading_halt_event(instrument="X", event_time=_T0, sequence=1, reason="", source="s")

    def test_trading_halt_round_trips(self) -> None:
        original = create_trading_halt_event(instrument="X", event_time=_T0, sequence=1, reason="stale_data", source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_trading_resume_round_trips(self) -> None:
        original = create_trading_resume_event(instrument="X", event_time=_T0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_financing_event_round_trips(self) -> None:
        original = create_financing_event(instrument="X", event_time=_T0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_end_of_stream_event_round_trips(self) -> None:
        original = create_end_of_stream_event(instrument="X", event_time=_T0, sequence=1, source="s")
        assert market_event_from_json_dict(original.to_json_dict()) == original

    def test_unknown_kind_rejected_by_dispatcher(self) -> None:
        with pytest.raises(MarketEventError, match="Unknown"):
            market_event_from_json_dict({"kind": "not_a_real_kind"})
