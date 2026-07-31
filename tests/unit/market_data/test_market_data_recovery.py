"""Unit tests for `market_data.recovery`: every interrupted-operation
scenario the specification names (truncated trailing record, batch
reserved-but-not-completed, checkpoint behind durable data), deterministic
repeated recovery, no duplicate append after recovery, and corruption
that is NOT a clean truncation is never silently accepted."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import RepositoryCorruptionError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.ingestion import IngestionBatchStore, ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.recovery import (
    read_jsonl_tolerating_truncated_tail,
    recover_feature_dataset,
    recover_raw_dataset,
)
from quant_platform.market_data.repository import MarketDataRepository

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


def _ingest(repo: MarketDataRepository, count: int, *, batch_id: str = "b1", start_hour: int = 0) -> tuple:
    seq = next_sequence_for(repo, _RAW_KEY)
    price = Decimal("2000")
    events = []
    for i in range(count):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=start_hour + i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    events_tuple = tuple(events)
    ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id=batch_id, ingestion_time=_T0, events=events_tuple, partitioning=_SPEC)
    return events_tuple


class TestTruncatedTrailingRecord:
    def test_a_truncated_last_line_is_discarded_and_the_file_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            events_path = repo.event_store.events_path("mt5", "mt5__XAUUSD")
            with events_path.open("a", encoding="utf-8") as f:
                f.write('{"kind": "candle", "event_id": "truncated')
            report = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            assert report.discarded_truncated_tail is True
            # normal strict reads must now succeed again (file was repaired).
            events = repo.event_store.read_events("mt5", "mt5__XAUUSD")
            assert len(events) == 10

    def test_a_non_trailing_corruption_is_never_silently_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            events_path = repo.event_store.events_path("mt5", "mt5__XAUUSD")
            lines = events_path.read_text(encoding="utf-8").splitlines()
            lines[3] = "{not-valid-json-in-the-middle"
            events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with pytest.raises(RepositoryCorruptionError):
                read_jsonl_tolerating_truncated_tail(events_path)


class TestBatchReservedButNotCompleted:
    def test_a_reserved_batch_with_no_matching_commit_is_reported_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            batch_store = IngestionBatchStore(repo.root)
            batch_store.reserve(dataset_key=_RAW_KEY, batch_id="crashed", content_digest="a" * 64, ingestion_time=_T0)
            report = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            assert report.pending_batch_ids == ("crashed",)

    def test_a_committed_batch_is_never_reported_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            report = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            assert report.pending_batch_ids == ()


class TestPartitionWrittenButManifestBehind:
    def test_recovery_advances_a_stale_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            manifest_before = repo.manifest_store.read_current(_RAW_KEY)
            # simulate "checkpoint/manifest fell behind": manually wipe the
            # manifest history file so the manifest layer looks uninitialized
            # even though events/partitions are already durable.
            manifest_path = repo.manifest_store._manifests_path(_RAW_KEY)
            manifest_path.unlink()
            report = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            assert report.manifest_advanced is True
            assert report.resulting_dataset_id == manifest_before.dataset_id  # deterministically reconstructed to the SAME content


class TestDeterministicRepeatedRecovery:
    def test_recovering_twice_in_a_row_yields_the_same_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 8)
            first = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            second = recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=2))
            assert first.resulting_dataset_id == second.resulting_dataset_id
            assert second.manifest_advanced is False  # nothing new to do the second time


class TestNoDuplicateAppendAfterRecovery:
    def test_event_count_is_unchanged_by_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 8)
            before = len(repo.event_store.read_events("mt5", "mt5__XAUUSD"))
            recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
            after = len(repo.event_store.read_events("mt5", "mt5__XAUUSD"))
            assert before == after


class TestFeatureDatasetRecovery:
    def test_recovery_completes_pending_feature_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            feature_key = result.feature_dataset_key

            # simulate the checkpoint never having been advanced (crash right
            # after the manifest write but before the checkpoint append).
            from quant_platform.market_data.checkpoints import CheckpointStore

            checkpoints_path = CheckpointStore(repo.root)._checkpoints_path(feature_key)
            checkpoints_path.unlink()

            report = recover_feature_dataset(
                repository=repo, raw_dataset_key=_RAW_KEY, feature_dataset_key=feature_key, feature_base_name="sma", feature_version=1,
                window=5, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1),
            )
            assert report.resulting_dataset_id == result.resulting_feature_dataset_id

    def test_empty_raw_dataset_is_a_clean_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            feature_key = DatasetKey(dataset_kind=DatasetKind.DERIVED_FEATURES, instrument_id="mt5__XAUUSD", feature_name="sma_5", feature_version=1)
            report = recover_feature_dataset(
                repository=repo, raw_dataset_key=_RAW_KEY, feature_dataset_key=feature_key, feature_base_name="sma", feature_version=1,
                window=5, partitioning=_SPEC, recovery_time=_T0,
            )
            assert report.manifest_advanced is False
            assert report.resulting_dataset_id is None


class TestAmbiguousCorruptionIsNeverSilentlyAccepted:
    def test_a_sequence_gap_in_the_recovered_stream_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            events_path = repo.event_store.events_path("mt5", "mt5__XAUUSD")
            lines = events_path.read_text(encoding="utf-8").splitlines()
            del lines[2]  # remove a middle line -- creates a sequence gap
            events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with pytest.raises(RepositoryCorruptionError):
                recover_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, recovery_time=_T0 + timedelta(days=1))
