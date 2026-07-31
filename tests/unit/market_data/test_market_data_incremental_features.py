"""Unit tests for `feature_generation.generate_feature_dataset_incremental`:
the core Phase 2 correctness property (incremental output equals full
fresh recomputation) for returns, rolling mean/std, ATR, RSI, EMA, SMA,
and VWAP, with partition boundaries deliberately cutting through rolling
windows -- plus lineage/no-overwrite requirements."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import FeatureStoreError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.feature_generation import (
    generate_candle_features,
    generate_feature_dataset_incremental,
)
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.ingestion import ingest_raw_events
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.repository import MarketDataRepository

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)  # a Monday
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


def _candles(count: int, *, start_hour: int = 0):
    price = Decimal("2000")
    result = []
    for i in range(count):
        h = start_hour + i
        # deterministic but non-monotonic close path so RSI/ATR are non-trivial
        delta = Decimal(2) if i % 3 == 0 else (Decimal(-1) if i % 3 == 1 else Decimal(1))
        result.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=h), timeframe=Timeframe.H1,
            sequence=h, open=price, high=price + 6, low=price - 6, close=price + delta, volume=Decimal(str(10 + i % 4)),
        ))
        price += delta
    return result


def _ingest_all(repo: MarketDataRepository, candles: list, *, batches: list[tuple[int, int]]) -> None:
    """`batches` is a list of `(start_index, end_index)` slices (relative
    to `candles`, ALREADY sorted by hour) to ingest as successive batches
    -- letting tests cut a rolling window across arbitrary batch/partition
    boundaries."""
    for i, (start, end) in enumerate(batches):
        chunk = tuple(candles[start:end])
        if not chunk:
            continue
        ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id=f"batch-{i}", ingestion_time=_T0 + timedelta(days=i), events=chunk, partitioning=_SPEC)


def _full_fresh_values(candles: list, *, feature_base_name: str, window: int | None, feature_version: int = 1) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = FeatureStore(Path(tmp))
        windows = {feature_base_name: window} if window is not None else None
        records = generate_candle_features(candles, feature_version=feature_version, store=store, feature_names=(feature_base_name,), windows=windows)
    return {r.timestamp: r.value for r in records}


@pytest.mark.parametrize(
    "feature_base_name,window",
    [("return", None), ("rolling_mean", 5), ("rolling_std", 5), ("atr", 5), ("rsi", 5), ("ema", 5), ("sma", 5), ("vwap", None)],
)
def test_incremental_equals_full_recomputation_across_partition_boundaries(feature_base_name: str, window: int | None) -> None:
    candles = _candles(50)
    # 3 batches, boundaries deliberately NOT aligned to the rolling window
    # (window=5): sizes 13, 17, 20 -- guarantees at least one batch
    # boundary falls strictly inside a would-be rolling window.
    with tempfile.TemporaryDirectory() as tmp:
        repo = MarketDataRepository.open(Path(tmp))
        batches = [(0, 13), (13, 30), (30, 50)]
        checkpoint_time = _T0
        for start, end in batches:
            chunk = tuple(candles[start:end])
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id=f"b-{start}", ingestion_time=checkpoint_time, events=chunk, partitioning=_SPEC)
            generate_feature_dataset_incremental(
                repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name=feature_base_name, feature_version=1, partitioning=_SPEC,
                checkpoint_time=checkpoint_time, window=window,
            )
            checkpoint_time += timedelta(days=1)

        stored_name = f"{feature_base_name}_{window}" if window is not None else feature_base_name
        incremental_records = repo.feature_store.read_records(stored_name, 1, "mt5__XAUUSD")
        incremental_values = {r.timestamp: r.value for r in incremental_records}

    fresh_values = _full_fresh_values(candles, feature_base_name=feature_base_name, window=window)
    assert incremental_values == fresh_values
    assert len(incremental_values) > 0


class TestNoOpWhenNoNewRawData:
    def test_a_second_call_with_no_new_ingestion_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            candles = tuple(_candles(20))
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=candles, partitioning=_SPEC)
            first = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            second = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            assert second.was_no_op is True
            assert second.new_record_count == 0
            assert first.resulting_feature_dataset_id == second.resulting_feature_dataset_id


class TestNoDuplicateFeatureCoordinates:
    def test_reingesting_the_same_batch_id_never_duplicates_feature_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            candles = tuple(_candles(10))
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=candles, partitioning=_SPEC)
            generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="return", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=None)
            records_before = len(repo.feature_store.read_records("return", 1, "mt5__XAUUSD"))
            # retry the exact same ingestion batch -- idempotent no-op at the raw layer -- then re-run generation.
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=candles, partitioning=_SPEC)
            generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="return", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=None)
            records_after = len(repo.feature_store.read_records("return", 1, "mt5__XAUUSD"))
            assert records_before == records_after


class TestChangedFeatureVersionCreatesNewLineage:
    def test_two_feature_versions_are_completely_independent_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            candles = tuple(_candles(20))
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=candles, partitioning=_SPEC)
            v1 = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            v2 = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=2, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            assert v1.feature_dataset_key != v2.feature_dataset_key
            assert len(repo.feature_store.read_records("sma_5", 1, "mt5__XAUUSD")) == len(repo.feature_store.read_records("sma_5", 2, "mt5__XAUUSD"))
            manifest_v1 = repo.manifest_store.read_current(v1.feature_dataset_key)
            manifest_v2 = repo.manifest_store.read_current(v2.feature_dataset_key)
            assert manifest_v1.dataset_id != manifest_v2.dataset_id  # different lineage entirely


class TestChangedRawVersionCreatesNewDerivedVersion:
    def test_new_raw_data_produces_a_new_feature_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            candles = _candles(30)
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=tuple(candles[:15]), partitioning=_SPEC)
            first = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b2", ingestion_time=_T0 + timedelta(days=1), events=tuple(candles[15:]), partitioning=_SPEC)
            second = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0 + timedelta(days=1), window=5)
            assert first.resulting_feature_dataset_id != second.resulting_feature_dataset_id


class TestExistingCompletedHistoryNeverOverwritten:
    def test_the_feature_store_itself_still_rejects_a_conflicting_value(self) -> None:
        # generate_feature_dataset_incremental can never produce a
        # conflicting value for the SAME raw data (determinism), but the
        # underlying FeatureStore's own no-overwrite guarantee is what
        # makes that a STRUCTURAL guarantee, not just an observed one --
        # confirmed directly here.
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            from quant_platform.market_data.feature_store import create_feature_record

            record = create_feature_record(feature_name="sma_5", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("2000"))
            store.append(record)
            conflicting = create_feature_record(feature_name="sma_5", feature_version=1, instrument_id="mt5__XAUUSD", timestamp=_T0, timeframe=Timeframe.H1, value=Decimal("9999"))
            with pytest.raises(FeatureStoreError):
                store.append(conflicting)


class TestEmptyOrMissingRawDataset:
    def test_no_raw_manifest_yet_is_a_clean_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            assert result.was_no_op is True
            assert result.resulting_feature_dataset_id is None
