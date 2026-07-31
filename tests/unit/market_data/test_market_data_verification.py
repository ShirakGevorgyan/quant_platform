"""Unit tests for `market_data.verification`: forged-identity detection,
append-only/sequence integrity, cross-event/cross-feature ordering, and
duplicate detection -- all recomputed independently from the store's own
raw entries."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.events import MarketEventStore
from quant_platform.market_data.feature_store import FeatureStore, create_feature_record
from quant_platform.market_data.verification import verify_feature_store, verify_market_event_store

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_AS_OF = _T0 + timedelta(days=1)


def _candle(hour: int, sequence: int | None = None):
    return create_candle(
        instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=hour), timeframe=Timeframe.H1,
        sequence=(hour if sequence is None else sequence), open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("10"),
    )


def _corrupt_last_line(path: Path, mutate) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    raw = json.loads(lines[-1])
    mutate(raw)
    lines[-1] = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestVerifyMarketEventStoreCleanCase:
    def test_no_issues_for_a_well_formed_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            for h in range(5):
                store.append(_candle(h))
            report = verify_market_event_store(store=store, provider="mt5", instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            assert report.criticals == ()


class TestVerifyMarketEventStoreForgedIdentity:
    def test_a_hand_edited_payload_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MarketEventStore(root)
            store.append(_candle(0))
            events_path = root / "market_events" / "mt5" / "mt5__XAUUSD" / "events.jsonl"
            _corrupt_last_line(events_path, lambda raw: raw["payload"].__setitem__("close", "2004"))
            report = verify_market_event_store(store=store, provider="mt5", instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            assert "forged_event_identity" in {i.code for i in report.criticals}


class TestVerifyMarketEventStoreSequenceIntegrity:
    def test_a_reordered_physical_line_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MarketEventStore(root)
            store.append(_candle(0))
            store.append(_candle(1))
            events_path = root / "market_events" / "mt5" / "mt5__XAUUSD" / "events.jsonl"
            lines = events_path.read_text(encoding="utf-8").splitlines()
            events_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
            report = verify_market_event_store(store=store, provider="mt5", instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            codes = {i.code for i in report.criticals}
            assert "event_sequence_gap_or_reorder" in codes or "event_ordering_violation" in codes

    def test_duplicate_event_id_in_the_raw_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MarketEventStore(root)
            store.append(_candle(0))
            store.append(_candle(1))
            events_path = root / "market_events" / "mt5" / "mt5__XAUUSD" / "events.jsonl"
            lines = events_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            second["event_id"] = first["event_id"]
            lines[1] = json.dumps(second, sort_keys=True, separators=(",", ":"))
            events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_market_event_store(store=store, provider="mt5", instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            codes = {i.code for i in report.criticals}
            assert "duplicate_event_id_in_store" in codes


class TestVerifyMarketEventStoreFutureTimestamp:
    def test_an_event_after_as_of_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MarketEventStore(Path(tmp))
            store.append(_candle(0))
            report = verify_market_event_store(store=store, provider="mt5", instrument_id="mt5__XAUUSD", as_of=_T0 - timedelta(hours=1))
            assert "future_event_timestamp" in {i.code for i in report.criticals}


class TestVerifyFeatureStoreCleanCase:
    def test_no_issues_for_a_well_formed_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            for h in range(5):
                store.append(create_feature_record(feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0 + timedelta(hours=h), timeframe=Timeframe.H1, value=Decimal("2000")))
            report = verify_feature_store(store=store, feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            assert report.criticals == ()


class TestVerifyFeatureStoreForgedIdentity:
    def test_a_hand_edited_value_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FeatureStore(root)
            store.append(create_feature_record(feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("2000")))
            records_path = root / "features" / "sma_20" / "v1" / "mt5__XAUUSD" / "records.jsonl"
            _corrupt_last_line(records_path, lambda raw: raw.__setitem__("value", "9999"))
            report = verify_feature_store(store=store, feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            assert "forged_feature_identity" in {i.code for i in report.criticals}


class TestVerifyFeatureStoreConflictingHistory:
    def test_two_different_ids_at_the_same_timestamp_in_the_raw_file_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FeatureStore(root)
            store.append(create_feature_record(feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("2000")))
            records_path = root / "features" / "sma_20" / "v1" / "mt5__XAUUSD" / "records.jsonl"
            # Hand-append a second, conflicting record at the SAME timestamp --
            # something FeatureStore.append itself would refuse, simulating a
            # corrupted/hand-edited file to prove verification catches it
            # independently rather than trusting append()'s own guarantee.
            conflicting = create_feature_record(feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("2500"))
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(conflicting.to_json_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            report = verify_feature_store(store=store, feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", as_of=_AS_OF)
            assert "conflicting_feature_value_at_timestamp" in {i.code for i in report.criticals}


class TestVerifyFeatureStoreFutureTimestamp:
    def test_a_record_after_as_of_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            store.append(create_feature_record(feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("2000")))
            report = verify_feature_store(store=store, feature_name="sma_20", feature_version=1, instrument_id="mt5__XAUUSD", as_of=_T0 - timedelta(hours=1))
            assert "future_feature_timestamp" in {i.code for i in report.criticals}
