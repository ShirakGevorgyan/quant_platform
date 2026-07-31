"""Unit tests for `market_data.checkpoints`: self-verification, forged
checkpoint detection, stale (behind/ahead) checkpoint detection, and
exact-retry idempotency of `CheckpointStore`."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import CheckpointError, StaleCheckpointError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.checkpoints import (
    CheckpointStore,
    compute_raw_ingestion_checkpoint,
    verify_raw_ingestion_checkpoint,
)
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


def _repo_with_data(tmp: str, count: int = 5):
    repo = MarketDataRepository.open(Path(tmp))
    seq = next_sequence_for(repo, _KEY)
    price = Decimal("2000")
    events = []
    for i in range(count):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b1", ingestion_time=_T0, events=tuple(events), partitioning=_SPEC)
    return repo


class TestComputeAndVerifyRawCheckpoint:
    def test_a_freshly_computed_checkpoint_verifies_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            verify_raw_ingestion_checkpoint(checkpoint, repository=repo)  # must not raise

    def test_round_trips_through_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            from quant_platform.market_data.checkpoints import RawIngestionCheckpoint

            assert RawIngestionCheckpoint.from_json_dict(checkpoint.to_json_dict()) == checkpoint


class TestForgedCheckpoint:
    def test_a_hand_edited_field_fails_self_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            from quant_platform.market_data.checkpoints import RawIngestionCheckpoint

            tampered = RawIngestionCheckpoint(
                checkpoint_id=checkpoint.checkpoint_id, dataset_key=checkpoint.dataset_key, last_committed_sequence=999,
                last_committed_batch_id=checkpoint.last_committed_batch_id, last_canonical_partition_id=checkpoint.last_canonical_partition_id,
                semantic_digest=checkpoint.semantic_digest, checkpoint_time=checkpoint.checkpoint_time,
            )
            with pytest.raises(CheckpointError):
                verify_raw_ingestion_checkpoint(tampered, repository=repo)


class TestStaleCheckpoint:
    def test_checkpoint_behind_durable_data_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp, count=3)
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            # advance the repository past what the checkpoint reflects
            more = create_candle(
                instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=10), timeframe=Timeframe.H1,
                sequence=next_sequence_for(repo, _KEY), open=Decimal("2010"), high=Decimal("2015"), low=Decimal("2005"), close=Decimal("2011"), volume=Decimal("1"),
            )
            ingest_raw_events(repository=repo, dataset_key=_KEY, batch_id="b2", ingestion_time=_T0, events=(more,), partitioning=_SPEC)
            with pytest.raises(StaleCheckpointError):
                verify_raw_ingestion_checkpoint(checkpoint, repository=repo)

    def test_checkpoint_ahead_of_durable_data_is_detected(self) -> None:
        # Simulate "ahead" by hand-constructing a checkpoint claiming more
        # sequence than the repository actually has.
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp, count=3)
            real = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            from quant_platform.market_data.checkpoints import (
                _create_raw_checkpoint,  # type: ignore[attr-defined]
            )

            ahead = _create_raw_checkpoint(
                dataset_key=_KEY, last_committed_sequence=real.last_committed_sequence + 100, last_committed_batch_id="b1",
                last_canonical_partition_id=real.last_canonical_partition_id, semantic_digest=real.semantic_digest, checkpoint_time=_T0,
            )
            with pytest.raises(StaleCheckpointError):
                verify_raw_ingestion_checkpoint(ahead, repository=repo)


class TestCheckpointStore:
    def test_append_and_read_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            store = CheckpointStore(Path(tmp))
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            store.append(_KEY, checkpoint)
            assert store.read_current(_KEY) == checkpoint

    def test_exact_retry_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            store = CheckpointStore(Path(tmp))
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            store.append(_KEY, checkpoint)
            store.append(_KEY, checkpoint)
            assert len(store.read_history(_KEY)) == 1

    def test_missing_dataset_reads_as_no_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            assert store.read_current(_KEY) is None

    def test_corrupted_kind_discriminator_is_rejected_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_data(tmp)
            store = CheckpointStore(Path(tmp))
            checkpoint = compute_raw_ingestion_checkpoint(repository=repo, dataset_key=_KEY, last_committed_batch_id="b1", checkpoint_time=_T0)
            store.append(_KEY, checkpoint)
            path = store._checkpoints_path(_KEY)
            lines = path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[0])
            raw["kind"] = "not_a_real_kind"
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            from quant_platform.core.exceptions import MarketDataPersistenceError

            with pytest.raises(MarketDataPersistenceError):
                store.read_history(_KEY)
