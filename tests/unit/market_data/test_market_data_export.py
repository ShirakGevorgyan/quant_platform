"""Unit tests for `market_data.export`: deterministic bytes, Decimal
preservation, timestamp preservation, stable row ordering, identical
export across different filesystem roots, and export-digest mutation
detection."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.export import export_feature_dataset_jsonl, export_raw_dataset_jsonl
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.repository import MarketDataRepository

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


def _ingest(repo: MarketDataRepository, count: int) -> tuple:
    seq = next_sequence_for(repo, _RAW_KEY)
    price = Decimal("2000.25")
    events = []
    for i in range(count):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + Decimal("5.5"), low=price - Decimal("5.5"), close=price + Decimal("1.1"), volume=Decimal("10.75"),
        ))
        price += Decimal("1.1")
    events_tuple = tuple(events)
    ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=events_tuple, partitioning=_SPEC)
    return events_tuple


class TestDeterministicBytesAndOrdering:
    def test_two_exports_of_the_same_state_produce_byte_identical_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 15)
            path_a = Path(tmp) / "a.jsonl"
            path_b = Path(tmp) / "b.jsonl"
            export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path_a)
            export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path_b)
            assert path_a.read_bytes() == path_b.read_bytes()

    def test_row_order_is_by_event_time_not_append_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            # ingest out of event-time order in one batch
            seq = next_sequence_for(repo, _RAW_KEY)
            c1 = create_candle(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=1), timeframe=Timeframe.H1, sequence=seq, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"))
            c0 = create_candle(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=seq + 1, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"))
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=(c1, c0), partitioning=_SPEC)
            path = Path(tmp) / "export.jsonl"
            export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert lines[0]["event_id"] == c0.event_id
            assert lines[1]["event_id"] == c1.event_id


class TestDecimalAndTimestampPreservation:
    def test_decimal_values_are_exported_as_exact_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 3)
            path = Path(tmp) / "export.jsonl"
            export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path)
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            assert first["payload"]["open"] == "2000.25"
            assert first["payload"]["volume"] == "10.75"

    def test_timestamps_preserve_utc_iso8601_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 3)
            path = Path(tmp) / "export.jsonl"
            export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path)
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            assert first["event_time"] == "2026-01-05T00:00:00+00:00"


class TestIdenticalExportAcrossDifferentRoots:
    def test_export_digest_is_identical_across_independent_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = MarketDataRepository.open(Path(tmp_a))
            repo_b = MarketDataRepository.open(Path(tmp_b))
            events = _ingest(repo_a, 10)
            ingest_raw_events(repository=repo_b, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            path_a = Path(tmp_a) / "export.jsonl"
            path_b = Path(tmp_b) / "export.jsonl"
            result_a = export_raw_dataset_jsonl(repository=repo_a, dataset_key=_RAW_KEY, destination=path_a)
            result_b = export_raw_dataset_jsonl(repository=repo_b, dataset_key=_RAW_KEY, destination=path_b)
            assert result_a.export_semantic_digest == result_b.export_semantic_digest
            assert path_a.read_bytes() == path_b.read_bytes()

    def test_a_deeply_nested_root_produces_the_same_export_as_a_shallow_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shallow_repo = MarketDataRepository.open(Path(tmp) / "shallow")
            events = _ingest(shallow_repo, 10)
            nested_root = Path(tmp) / "a" / "b" / "c" / "d" / "nested"
            nested_repo = MarketDataRepository.open(nested_root)
            ingest_raw_events(repository=nested_repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            shallow_export = export_raw_dataset_jsonl(repository=shallow_repo, dataset_key=_RAW_KEY, destination=Path(tmp) / "shallow_export.jsonl")
            nested_export = export_raw_dataset_jsonl(repository=nested_repo, dataset_key=_RAW_KEY, destination=Path(tmp) / "nested_export.jsonl")
            assert shallow_export.export_semantic_digest == nested_export.export_semantic_digest


class TestExportDigestMutationDetection:
    def test_a_different_underlying_dataset_produces_a_different_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = MarketDataRepository.open(Path(tmp_a))
            repo_b = MarketDataRepository.open(Path(tmp_b))
            _ingest(repo_a, 10)
            _ingest(repo_b, 12)  # genuinely different data
            result_a = export_raw_dataset_jsonl(repository=repo_a, dataset_key=_RAW_KEY, destination=Path(tmp_a) / "e.jsonl")
            result_b = export_raw_dataset_jsonl(repository=repo_b, dataset_key=_RAW_KEY, destination=Path(tmp_b) / "e.jsonl")
            assert result_a.export_semantic_digest != result_b.export_semantic_digest


class TestFeatureExport:
    def test_feature_export_row_count_matches_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            export_path = Path(tmp) / "features.jsonl"
            export_result = export_feature_dataset_jsonl(repository=repo, dataset_key=result.feature_dataset_key, destination=export_path)
            stored_count = len(repo.feature_store.read_records("sma_5", 1, "mt5__XAUUSD"))
            assert export_result.row_count == stored_count
