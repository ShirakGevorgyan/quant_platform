"""Unit tests for `market_data.reports`: fresh-recompute-every-time
session reporting."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.events import MarketEventStore
from quant_platform.market_data.feature_generation import generate_candle_features
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.reports import generate_market_data_report

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_AS_OF = _T0 + timedelta(days=5)


def _rising_candles(count: int) -> list[object]:
    candles = []
    price = Decimal("2000")
    for h in range(count):
        candles.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=h), timeframe=Timeframe.H1,
            sequence=h, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    return candles


class TestGenerateMarketDataReport:
    def test_report_reflects_event_and_feature_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_store = MarketEventStore(root / "events")
            candles = _rising_candles(30)
            for candle in candles:
                event_store.append(candle)
            feature_store = FeatureStore(root / "features")
            generate_candle_features(candles, feature_version=1, store=feature_store)

            report = generate_market_data_report(
                event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store,
                feature_partitions=(("sma_20", 1),), report_time=_AS_OF,
            )
            summary = report.sections["MarketEventSummary"]
            assert summary["total_events"] == 30
            feature_summary = report.sections["FeatureStoreSummary"]["sma_20_v1"]
            assert feature_summary["record_count"] == 30 - 19  # sma_20 warms up after 19 bars
            assert feature_summary["critical_issue_count"] == 0

    def test_two_calls_against_unchanged_state_produce_identical_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_store = MarketEventStore(root / "events")
            candles = _rising_candles(10)
            for candle in candles:
                event_store.append(candle)
            report_a = generate_market_data_report(event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", report_time=_AS_OF)
            report_b = generate_market_data_report(event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", report_time=_AS_OF)
            assert report_a.to_json_dict() == report_b.to_json_dict()

    def test_report_without_a_feature_store_has_an_empty_feature_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event_store = MarketEventStore(Path(tmp) / "events")
            report = generate_market_data_report(event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", report_time=_AS_OF)
            assert report.sections["FeatureStoreSummary"] == {}
