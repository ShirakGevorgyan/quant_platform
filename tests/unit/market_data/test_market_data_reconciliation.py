"""Unit tests for `market_data.reconciliation`: clean repository, and
every structured issue category the specification names -- missing
partition, orphan partition, wrong digest, wrong count, duplicate
coordinate, broken lineage, stale checkpoint, semantic mismatch."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.partitions import build_partition, partition_key_for
from quant_platform.market_data.reconciliation import reconcile_feature_dataset, reconcile_raw_dataset
from quant_platform.market_data.repository import MarketDataRepository

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")
_GENERATED_AT = "2026-02-01T00:00:00Z"


def _ingest(repo: MarketDataRepository, count: int) -> tuple:
    seq = next_sequence_for(repo, _RAW_KEY)
    price = Decimal("2000")
    events = []
    for i in range(count):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    events_tuple = tuple(events)
    ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=events_tuple, partitioning=_SPEC)
    return events_tuple


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


class TestCleanRepository:
    def test_clean_raw_dataset_has_no_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert report.criticals == ()

    def test_clean_feature_dataset_has_no_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            report = reconcile_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert report.criticals == ()

    def test_empty_repository_has_no_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert report.criticals == ()


class TestMissingPartition:
    def test_a_partition_file_removed_after_ingestion_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            path = repo.partition_store._partition_path(_RAW_KEY, "2026-01-05")
            path.unlink()
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "missing_partition" in _codes(report)


class TestOrphanPartition:
    def test_a_partition_with_no_corresponding_events_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            far_past = _T0 - timedelta(days=365)
            key = partition_key_for(far_past, _SPEC)
            fake = build_partition(dataset_key=_RAW_KEY, partition_key=key, spec=_SPEC, members=[("nonexistent-id", far_past)])
            repo.partition_store.write(fake)
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "orphan_partition" in _codes(report)


class TestWrongPartitionDigest:
    def test_a_hand_edited_partition_digest_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            path = repo.partition_store._partition_path(_RAW_KEY, "2026-01-05")
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["content_digest"] = "f" * 64
            path.write_text(json.dumps(raw), encoding="utf-8")
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "wrong_partition_digest" in _codes(report)


class TestWrongEventCount:
    def test_a_manifest_declaring_the_wrong_count_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            manifest_path = repo.manifest_store._manifests_path(_RAW_KEY)
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[-1])
            raw["event_count"] = 999
            lines[-1] = json.dumps(raw)
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "wrong_event_count" in _codes(report)


class TestDuplicateCoordinate:
    def test_a_hand_appended_conflicting_feature_record_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            from quant_platform.market_data.feature_store import create_feature_record

            records = repo.feature_store.read_records("sma_5", 1, "mt5__XAUUSD")
            target = records[0]
            conflicting = create_feature_record(feature_name="sma_5", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=target.timestamp, timeframe=Timeframe.H1, value=Decimal("999999"))
            path = repo.feature_store.records_path("sma_5", 1, "mt5__XAUUSD")
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(conflicting.to_json_dict()))
                f.write("\n")
            report = reconcile_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "feature_coordinate_conflict" in _codes(report)


class TestBrokenLineage:
    def test_a_feature_manifest_referencing_a_nonexistent_raw_version_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            manifest_path = repo.manifest_store._manifests_path(result.feature_dataset_key)
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[-1])
            raw["raw_source_dataset_id"] = "f" * 64
            lines[-1] = json.dumps(raw)
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = reconcile_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "broken_lineage" in _codes(report)


class TestSemanticDigestMismatch:
    def test_a_hand_edited_manifest_semantic_digest_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            manifest_path = repo.manifest_store._manifests_path(_RAW_KEY)
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[-1])
            raw["semantic_digest"] = "0" * 64
            lines[-1] = json.dumps(raw)
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "semantic_digest_mismatch" in _codes(report)


class TestStaleCheckpoint:
    def test_a_hand_edited_raw_checkpoint_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 5)
            from quant_platform.market_data.checkpoints import (
                CheckpointStore,
                compute_raw_ingestion_checkpoint,
            )

            checkpoint_store = CheckpointStore(repo.root)
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_RAW_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            checkpoint_store.append(_RAW_KEY, checkpoint)
            # advance the repository past the checkpoint
            more = create_candle(
                instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=50), timeframe=Timeframe.H1,
                sequence=next_sequence_for(repo, _RAW_KEY), open=Decimal("2010"), high=Decimal("2015"), low=Decimal("2005"), close=Decimal("2011"), volume=Decimal("1"),
            )
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b2", ingestion_time=_T0, events=(more,), partitioning=_SPEC)
            report = reconcile_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, generated_at=_GENERATED_AT)
            assert "stale_checkpoint" in _codes(report)
