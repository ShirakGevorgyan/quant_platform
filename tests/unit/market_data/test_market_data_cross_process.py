"""Cross-process determinism tests for the Phase 2 repository layer:
different `PYTHONHASHSEED`, different temp filesystem roots, and
repeated replay/verification/export -- mirroring
`portfolio_risk.replay`'s and `market_data.replay`'s own established
subprocess-proof pattern exactly."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.export import export_raw_dataset_jsonl
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.verification import verify_raw_dataset

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_RAW_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


def _ingest_and_generate(repo: MarketDataRepository) -> None:
    seq = next_sequence_for(repo, _RAW_KEY)
    price = Decimal("2000")
    events = []
    for i in range(30):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    ingest_raw_events(repository=repo, dataset_key=_RAW_KEY, batch_id="b1", ingestion_time=_T0, events=tuple(events), partitioning=_SPEC)
    generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)


class TestRepeatedReplayVerificationExport:
    def test_repeated_verification_of_unchanged_state_never_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest_and_generate(repo)
            first = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_T0 + timedelta(days=10))
            second = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_T0 + timedelta(days=10))
            assert first.criticals == second.criticals == ()

    def test_repeated_export_of_unchanged_state_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest_and_generate(repo)
            path_a = Path(tmp) / "a.jsonl"
            path_b = Path(tmp) / "b.jsonl"
            result_a = export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path_a)
            result_b = export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=path_b)
            assert result_a.export_semantic_digest == result_b.export_semantic_digest
            assert path_a.read_bytes() == path_b.read_bytes()


class TestDifferentFilesystemRoots:
    def test_two_independent_roots_produce_the_same_dataset_id_and_export_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = MarketDataRepository.open(Path(tmp_a))
            repo_b = MarketDataRepository.open(Path(tmp_b))
            _ingest_and_generate(repo_a)
            _ingest_and_generate(repo_b)
            manifest_a = repo_a.manifest_store.read_current(_RAW_KEY)
            manifest_b = repo_b.manifest_store.read_current(_RAW_KEY)
            assert manifest_a.dataset_id == manifest_b.dataset_id
            export_a = export_raw_dataset_jsonl(repository=repo_a, dataset_key=_RAW_KEY, destination=Path(tmp_a) / "e.jsonl")
            export_b = export_raw_dataset_jsonl(repository=repo_b, dataset_key=_RAW_KEY, destination=Path(tmp_b) / "e.jsonl")
            assert export_a.export_semantic_digest == export_b.export_semantic_digest


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    from pathlib import Path

    from quant_platform.core.types import Timeframe
    from quant_platform.market_data.candles import create_candle
    from quant_platform.market_data.export import export_raw_dataset_jsonl
    from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
    from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
    from quant_platform.market_data.manifests import DatasetKey, DatasetKind, PartitionGranularity, PartitioningSpec
    from quant_platform.market_data.repository import MarketDataRepository

    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    spec = PartitioningSpec(granularity=PartitionGranularity.DAILY)
    raw_key = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")
    root = Path(sys.argv[1])
    repo = MarketDataRepository.open(root)
    seq = next_sequence_for(repo, raw_key)
    price = Decimal("2000")
    events = []
    for i in range(30):
        events.append(create_candle(
            instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=t0 + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=seq + i, open=price, high=price + 5, low=price - 5, close=price + 1, volume=Decimal("10"),
        ))
        price += 1
    ingest_raw_events(repository=repo, dataset_key=raw_key, batch_id="b1", ingestion_time=t0, events=tuple(events), partitioning=spec)
    feature_result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=raw_key, feature_base_name="sma", feature_version=1, partitioning=spec, checkpoint_time=t0, window=5)
    manifest = repo.manifest_store.read_current(raw_key)
    export_result = export_raw_dataset_jsonl(repository=repo, dataset_key=raw_key, destination=root / "export.jsonl")
    print(manifest.dataset_id)
    print(feature_result.resulting_feature_dataset_id)
    print(export_result.export_semantic_digest)
    """
)


def _run_in_subprocess(root: str, *, hashseed: str) -> tuple[str, str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, root], env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    lines = completed.stdout.strip().splitlines()
    return lines[0], lines[1], lines[2]


class TestPythonHashSeedIndependence:
    def test_dataset_and_export_digests_are_stable_across_separate_processes_with_different_hash_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            dataset_id_a, feature_id_a, export_digest_a = _run_in_subprocess(tmp_a, hashseed="0")
            dataset_id_b, feature_id_b, export_digest_b = _run_in_subprocess(tmp_b, hashseed="4294967295")
            assert dataset_id_a == dataset_id_b
            assert feature_id_a == feature_id_b
            assert export_digest_a == export_digest_b

    def test_subprocess_result_matches_in_process_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            dataset_id_subprocess, _, export_digest_subprocess = _run_in_subprocess(tmp_a, hashseed="12345")
            repo = MarketDataRepository.open(Path(tmp_b))
            _ingest_and_generate(repo)
            manifest = repo.manifest_store.read_current(_RAW_KEY)
            export_result = export_raw_dataset_jsonl(repository=repo, dataset_key=_RAW_KEY, destination=Path(tmp_b) / "e.jsonl")
            assert dataset_id_subprocess == manifest.dataset_id
            assert export_digest_subprocess == export_result.export_semantic_digest
