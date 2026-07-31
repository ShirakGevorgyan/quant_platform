"""Unit tests for `market_data.compaction`: the required invariants --
event/feature identities, semantic digest, logical ordering, and replay
result are never changed by compaction; a physically drifted partition is
corrected back to its canonical form."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.compaction import compact_feature_dataset, compact_raw_dataset
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.replay import compute_replay_result, replay_candle_features_from_events
from quant_platform.market_data.repository import MarketDataRepository

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


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


class TestCompactionPreservesSemanticDigest:
    def test_raw_compaction_never_changes_semantic_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 30)
            result = compact_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, compaction_time=_T0 + timedelta(days=1))
            assert result.semantic_digest_preserved is True

    def test_feature_compaction_never_changes_semantic_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 30)
            gen_result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            compaction_result = compact_feature_dataset(repository=repo, dataset_key=gen_result.feature_dataset_key, partitioning=_SPEC, compaction_time=_T0 + timedelta(days=1))
            assert compaction_result.semantic_digest_preserved is True


class TestCompactionNeverChangesEventOrFeatureIdentities:
    def test_raw_events_are_untouched_by_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            events_before = _ingest(repo, 20)
            compact_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, compaction_time=_T0 + timedelta(days=1))
            events_after = repo.event_store.read_events("mt5", "mt5__XAUUSD")
            assert tuple(events_after) == events_before


class TestCompactionNeverChangesReplayResult:
    def test_replay_after_compaction_matches_replay_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 30)
            from quant_platform.market_data.feature_store import FeatureStore

            store_a = FeatureStore(Path(tmp) / "replay_before")
            records_before = replay_candle_features_from_events(event_store=repo.event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=store_a, feature_version=1, feature_names=("sma",), windows={"sma": 5})
            result_before = compute_replay_result(records_before, instrument_id="mt5__XAUUSD")

            compact_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, compaction_time=_T0 + timedelta(days=1))

            store_b = FeatureStore(Path(tmp) / "replay_after")
            records_after = replay_candle_features_from_events(event_store=repo.event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=store_b, feature_version=1, feature_names=("sma",), windows={"sma": 5})
            result_after = compute_replay_result(records_after, instrument_id="mt5__XAUUSD")

            assert result_before.feature_semantic_digest == result_after.feature_semantic_digest


class TestCompactionCorrectsPhysicalDrift:
    def test_a_hand_edited_partition_is_restored_to_canonical_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            path = repo.partition_store._partition_path(_RAW_KEY, "2026-01-05")
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["content_digest"] = "f" * 64  # drift: no longer matches its own recorded members
            path.write_text(json.dumps(raw), encoding="utf-8")

            result = compact_raw_dataset(repository=repo, dataset_key=_RAW_KEY, partitioning=_SPEC, compaction_time=_T0 + timedelta(days=1))
            restored = repo.partition_store.read(_RAW_KEY, "2026-01-05")
            assert restored.content_digest != "f" * 64
            assert result.semantic_digest_preserved is True  # economic content was never actually wrong
