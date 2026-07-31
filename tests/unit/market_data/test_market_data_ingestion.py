"""Unit tests for `market_data.ingestion`: batch idempotency/conflict
detection, first/continuation/retry/overlap/late-arrival/out-of-order
ingestion, sequence-gap rejection, and manifest/partition rebuild
correctness -- the specification's own required ingestion scenario list."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import IngestionConflictError, IngestionError, MarketDataPersistenceError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
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
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


def _repo(tmp: str) -> MarketDataRepository:
    return MarketDataRepository.open(Path(tmp))


def _candles_at(repo: MarketDataRepository, hours: list[int]) -> tuple:
    seq = next_sequence_for(repo, _KEY)
    price = Decimal("2000")
    result = []
    for offset, hour in enumerate(hours):
        result.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=hour), timeframe=Timeframe.H1,
            sequence=seq + offset, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    return tuple(result)


class TestFirstIngestion:
    def test_appends_events_and_creates_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            events = _candles_at(repo, [0, 1, 2])
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            assert result.appended_event_count == 3
            assert result.was_idempotent_replay is False
            manifest = repo.manifest_store.read_current(_KEY)
            assert manifest.event_count == 3

    def test_empty_batch_is_legal_and_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="empty", ingestion_time=_T0, events=(), partitioning=_SPEC)
            assert result.appended_event_count == 0
            assert repo.manifest_store.read_current(_KEY).event_count == 0


class TestContinuationBatch:
    def test_second_batch_extends_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            first = _candles_at(repo, [0, 1])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=first, partitioning=_SPEC)
            second = _candles_at(repo, [2, 3])
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b2", ingestion_time=_T0 + timedelta(hours=1), events=second, partitioning=_SPEC)
            assert result.appended_event_count == 2
            assert repo.manifest_store.read_current(_KEY).event_count == 4


class TestExactRetry:
    def test_identical_resubmission_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            events = _candles_at(repo, [0, 1])
            first = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            second = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            assert second.was_idempotent_replay is True
            assert second.resulting_dataset_id == first.resulting_dataset_id
            assert repo.manifest_store.read_current(_KEY).event_count == 2  # not duplicated

    def test_retry_creates_no_duplicate_partition_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            events = _candles_at(repo, [0, 1])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            assert len(repo.manifest_store.read_history(_KEY)) == 1


class TestConflictingRetry:
    def test_same_batch_id_different_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            first_events = _candles_at(repo, [0, 1])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=first_events, partitioning=_SPEC)
            different_events = _candles_at(repo, [5, 6])
            with pytest.raises(IngestionConflictError):
                ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=different_events, partitioning=_SPEC)

    def test_same_batch_id_different_ingestion_time_is_also_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            events = _candles_at(repo, [0, 1])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=events, partitioning=_SPEC)
            with pytest.raises(IngestionConflictError):
                ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0 + timedelta(days=1), events=events, partitioning=_SPEC)


class TestLateArrival:
    def test_a_historical_event_into_an_already_partitioned_day_rebuilds_that_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            first = _candles_at(repo, [0, 1, 2])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=first, partitioning=_SPEC)
            first_manifest = repo.manifest_store.read_current(_KEY)

            # a late-arriving bar for hour 1 (between two already-ingested hours) -- a genuinely new,
            # distinct event (different close), not a duplicate of any existing one.
            seq = next_sequence_for(repo, _KEY)
            late = create_candle(
                instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=0, minutes=30), timeframe=Timeframe.H1,
                sequence=seq, open=Decimal("1990"), high=Decimal("1995"), low=Decimal("1985"), close=Decimal("1992"), volume=Decimal("3"),
            )
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="late-1", ingestion_time=_T0 + timedelta(days=1), events=(late,), partitioning=_SPEC)
            assert result.rebuilt_partition_keys == ("2026-01-05",)
            second_manifest = repo.manifest_store.read_current(_KEY)
            assert second_manifest.event_count == 4
            assert second_manifest.dataset_id != first_manifest.dataset_id
            # the prior manifest VERSION remains independently retrievable, unmutated.
            history = repo.manifest_store.read_history(_KEY)
            assert history[0] == first_manifest


class TestOutOfOrderArrival:
    def test_arrival_order_may_differ_from_event_time_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            # submit hour 2 before hour 0/1 in ONE batch -- repository append
            # sequence is arrival order, but partition membership must still
            # reflect event-time order.
            seq = next_sequence_for(repo, _KEY)
            c2 = create_candle(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=2), timeframe=Timeframe.H1, sequence=seq, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"))
            c0 = create_candle(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=seq + 1, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"))
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="ooo", ingestion_time=_T0, events=(c2, c0), partitioning=_SPEC)
            assert result.appended_event_count == 2
            events = repo.event_store.read_events("mt5", "mt5__XAUUSD")
            assert [e.sequence for e in events] == [seq, seq + 1]  # physical arrival order preserved
            partition = repo.partition_store.read(_KEY, "2026-01-05")
            assert partition.ordered_member_ids == (c0.event_id, c2.event_id)  # logical (event-time) order


class TestPartialOverlapWithExactDuplicates:
    def test_a_batch_re_submitting_some_already_ingested_events_plus_new_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            first = _candles_at(repo, [0, 1])
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=first, partitioning=_SPEC)
            new_only = _candles_at(repo, [2])
            overlapping_batch = first + new_only
            # the overlap re-submits `first`'s own already-appended events with
            # their own already-assigned sequence numbers -- append() absorbs
            # them idempotently at those positions before the genuinely new one.
            result = ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b2", ingestion_time=_T0 + timedelta(hours=1), events=overlapping_batch, partitioning=_SPEC)
            assert repo.manifest_store.read_current(_KEY).event_count == 3
            assert result.appended_event_count == 1  # only the genuinely new event


class TestSequenceGapIsRejected:
    def test_a_gap_in_repository_append_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            seq = next_sequence_for(repo, _KEY)
            gapped = create_candle(
                instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=seq + 5,
                open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2001"), volume=Decimal("1"),
            )
            with pytest.raises(MarketDataPersistenceError):
                ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="gap", ingestion_time=_T0, events=(gapped,), partitioning=_SPEC)


class TestWrongDatasetKeyOrEventMismatch:
    def test_wrong_dataset_kind_is_rejected(self) -> None:
        feature_key = DatasetKey(dataset_kind=DatasetKind.DERIVED_FEATURES, instrument_id="mt5__XAUUSD", feature_name="sma_20", feature_version=1)
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            with pytest.raises(IngestionError):
                ingest_raw_events(repository=repo, dataset_key=feature_key, batch_id="b1", ingestion_time=_T0, events=(), partitioning=_SPEC)

    def test_an_event_belonging_to_a_different_instrument_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo(tmp)
            wrong = create_candle(
                instrument_id="mt5__EURUSD", provider="mt5", symbol="EURUSD", event_time=_T0, timeframe=Timeframe.H1, sequence=0,
                open=Decimal("1"), high=Decimal("2"), low=Decimal("1"), close=Decimal("1.5"), volume=Decimal("1"),
            )
            with pytest.raises(IngestionError):
                ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=(wrong,), partitioning=_SPEC)
