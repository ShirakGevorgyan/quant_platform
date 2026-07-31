"""Unit tests for `market_data.replay`: rebuilding a feature store purely
from raw events, determinism across independent stores/roots, divergence
detection, and cross-process reproducibility (a fresh OS process, a
different `PYTHONHASHSEED`)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import MarketDataReplayError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.events import MarketEventStore
from quant_platform.market_data.feature_generation import generate_candle_features
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.replay import (
    assert_replay_deterministic,
    compute_replay_result,
    replay_candle_features_from_events,
)

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


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


def _populate_event_store(root: Path) -> MarketEventStore:
    store = MarketEventStore(root)
    for candle in _rising_candles(30):
        store.append(candle)
    return store


class TestReplayRebuildsFromRawEventsOnly:
    def test_rebuilt_records_match_a_direct_generation_from_the_same_candles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_store = _populate_event_store(root / "events")
            direct_store = FeatureStore(root / "direct")
            direct_records = generate_candle_features(_rising_candles(30), feature_version=1, store=direct_store)

            replay_store = FeatureStore(root / "replay")
            replayed = replay_candle_features_from_events(
                event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=replay_store, feature_version=1,
            )
            assert {r.feature_id for r in direct_records} == {r.feature_id for r in replayed}

    def test_non_candle_events_in_the_same_partition_are_ignored(self) -> None:
        from quant_platform.market_data.ticks import create_tick

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_store = MarketEventStore(root / "events")
            candles = _rising_candles(25)
            for candle in candles:
                event_store.append(candle)
            event_store.append(create_tick(instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=25), sequence=25, price=Decimal("2030")))

            replay_store = FeatureStore(root / "replay")
            replayed = replay_candle_features_from_events(
                event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=replay_store, feature_version=1, feature_names=("return",),
            )
            assert len(replayed) == len(candles) - 1  # first bar has no return


class TestReplayDeterminismAcrossIndependentStores:
    def test_two_independent_temp_roots_replay_identically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = _populate_event_store(Path(tmp_a) / "events")
            store_b = _populate_event_store(Path(tmp_b) / "events")
            feature_store_a = FeatureStore(Path(tmp_a) / "features")
            feature_store_b = FeatureStore(Path(tmp_b) / "features")
            records_a = replay_candle_features_from_events(event_store=store_a, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store_a, feature_version=1)
            records_b = replay_candle_features_from_events(event_store=store_b, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store_b, feature_version=1)
            result_a = compute_replay_result(records_a, instrument_id="mt5__XAUUSD")
            result_b = compute_replay_result(records_b, instrument_id="mt5__XAUUSD")
            assert_replay_deterministic(result_a, result_b)

    def test_repeated_computation_never_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _populate_event_store(Path(tmp) / "events")
            feature_store = FeatureStore(Path(tmp) / "features")
            records = replay_candle_features_from_events(event_store=store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store, feature_version=1)
            first = compute_replay_result(records, instrument_id="mt5__XAUUSD")
            second = compute_replay_result(records, instrument_id="mt5__XAUUSD")
            assert first == second


class TestReplayDivergenceIsDetected:
    def test_a_genuinely_different_feature_set_raises_on_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = _populate_event_store(Path(tmp_a) / "events")
            feature_store_a = FeatureStore(Path(tmp_a) / "features")
            records_a = replay_candle_features_from_events(event_store=store_a, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store_a, feature_version=1)

            store_b = MarketEventStore(Path(tmp_b) / "events")
            for candle in _rising_candles(35):  # a different, longer series
                store_b.append(candle)
            feature_store_b = FeatureStore(Path(tmp_b) / "features")
            records_b = replay_candle_features_from_events(event_store=store_b, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store_b, feature_version=1)

            result_a = compute_replay_result(records_a, instrument_id="mt5__XAUUSD")
            result_b = compute_replay_result(records_b, instrument_id="mt5__XAUUSD")
            with pytest.raises(MarketDataReplayError):
                assert_replay_deterministic(result_a, result_b)


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    from pathlib import Path

    from quant_platform.core.types import Timeframe
    from quant_platform.market_data.candles import create_candle
    from quant_platform.market_data.events import MarketEventStore
    from quant_platform.market_data.feature_store import FeatureStore
    from quant_platform.market_data.replay import compute_replay_result, replay_candle_features_from_events

    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    event_store = MarketEventStore(Path(sys.argv[1]) / "events")
    price = Decimal("2000")
    for h in range(30):
        candle = create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=t0 + timedelta(hours=h), timeframe=Timeframe.H1,
            sequence=h, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        )
        event_store.append(candle)
        price += 1
    feature_store = FeatureStore(Path(sys.argv[1]) / "features")
    records = replay_candle_features_from_events(event_store=event_store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store, feature_version=1)
    result = compute_replay_result(records, instrument_id="mt5__XAUUSD")
    print(result.feature_semantic_digest)
    print(",".join(result.feature_ids))
    """
)


def _run_in_subprocess(root: str, *, hashseed: str) -> tuple[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, root], env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    lines = completed.stdout.strip().splitlines()
    return lines[0], lines[1]


class TestCrossProcessReproducibility:
    def test_semantic_digest_is_stable_across_separate_processes_with_different_hash_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            digest_a, ids_a = _run_in_subprocess(tmp_a, hashseed="0")
            digest_b, ids_b = _run_in_subprocess(tmp_b, hashseed="4294967295")
            assert digest_a == digest_b
            assert ids_a == ids_b

    def test_subprocess_digest_matches_in_process_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            digest_subprocess, _ = _run_in_subprocess(tmp_a, hashseed="12345")
            store = _populate_event_store(Path(tmp_b) / "events")
            feature_store = FeatureStore(Path(tmp_b) / "features")
            records = replay_candle_features_from_events(event_store=store, provider="mt5", instrument_id="mt5__XAUUSD", feature_store=feature_store, feature_version=1)
            result_in_process = compute_replay_result(records, instrument_id="mt5__XAUUSD")
            assert digest_subprocess == result_in_process.feature_semantic_digest
