"""Unit tests for `verification.verify_raw_dataset`/`verify_feature_dataset`
(Milestone 10, Phase 2): clean datasets, forged manifest, forged
partition, an event removed and the store re-chained, a feature value
changed and indexes rebuilt, and the fresh-recompute cross-check catching
coherent feature tampering (an attacker who consistently re-hashed a
tampered value) that pure identity recomputation cannot."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.feature_generation import generate_feature_dataset_incremental
from quant_platform.market_data.feature_store import FEATURE_RECORD_KIND, FeatureRecord
from quant_platform.market_data.identity import compute_content_id
from quant_platform.market_data.ingestion import ingest_raw_events, next_sequence_for
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.verification import verify_feature_dataset, verify_raw_dataset

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_AS_OF = _T0 + timedelta(days=30)
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


def _codes(report) -> set[str]:
    return {i.code for i in report.criticals}


class TestCleanDatasets:
    def test_clean_raw_dataset_has_no_criticals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            report = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert report.criticals == ()

    def test_clean_feature_dataset_has_no_criticals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            report = verify_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert report.criticals == ()


class TestForgedManifest:
    def test_a_hand_edited_manifest_field_fails_dataset_identity_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            manifest_path = repo.manifest_store._manifests_path(_RAW_KEY)
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[-1])
            raw["event_count"] = 999  # dataset_id itself is untouched -- now inconsistent with its own payload
            lines[-1] = json.dumps(raw)
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert "forged_dataset_identity" in _codes(report)


class TestForgedPartition:
    def test_a_hand_edited_partition_field_fails_partition_identity_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            path = repo.partition_store._partition_path(_RAW_KEY, "2026-01-05")
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["ordered_member_ids"] = list(reversed(raw["ordered_member_ids"]))  # partition_id itself untouched
            path.write_text(json.dumps(raw), encoding="utf-8")
            report = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert "forged_partition_identity" in _codes(report)


class TestEventRemovedAndStoreReChained:
    def test_removing_a_middle_event_and_renumbering_sequences_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 10)
            events_path = repo.event_store.events_path("mt5", "mt5__XAUUSD")
            lines = events_path.read_text(encoding="utf-8").splitlines()
            parsed = [json.loads(line) for line in lines]
            del parsed[3]
            # "re-chain": renumber sequence to look superficially consistent
            # (gapless) -- but each event's OWN event_id still bakes in its
            # ORIGINAL sequence, so this must still be caught.
            for i, raw in enumerate(parsed):
                raw["sequence"] = i
            events_path.write_text("\n".join(json.dumps(r) for r in parsed) + "\n", encoding="utf-8")
            report = verify_raw_dataset(repository=repo, dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert "forged_event_identity" in _codes(report)


class TestFeatureChangedAndIndexesRebuilt:
    def test_a_changed_feature_value_with_a_stale_partition_index_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            records_path = repo.feature_store.records_path("sma_5", 1, "mt5__XAUUSD")
            lines = records_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[0])
            raw["value"] = "12345"  # feature_id now stale relative to its own payload
            lines[0] = json.dumps(raw)
            records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert "forged_feature_identity" in _codes(report)


class TestFreshRecomputeCatchesCoherentTampering:
    def test_a_consistently_rehashed_tampered_value_is_invisible_to_identity_checks_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketDataRepository.open(Path(tmp))
            _ingest(repo, 20)
            result = generate_feature_dataset_incremental(repository=repo, raw_dataset_key=_RAW_KEY, feature_base_name="sma", feature_version=1, partitioning=_SPEC, checkpoint_time=_T0, window=5)
            records_path = repo.feature_store.records_path("sma_5", 1, "mt5__XAUUSD")
            lines = records_path.read_text(encoding="utf-8").splitlines()
            raw = json.loads(lines[0])
            raw["value"] = "999999"
            tampered = FeatureRecord.from_json_dict({**raw, "feature_id": "0" * 64})
            raw["feature_id"] = compute_content_id(FEATURE_RECORD_KIND, tampered.to_identity_payload())
            lines[0] = json.dumps(raw)
            records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            basic_report = verify_feature_dataset(repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, as_of=_AS_OF)
            assert basic_report.criticals == ()  # coherent tampering is invisible here

            crosscheck_report = verify_feature_dataset(
                repository=repo, dataset_key=result.feature_dataset_key, raw_dataset_key=_RAW_KEY, as_of=_AS_OF,
                cross_check_against_fresh_recomputation=True, feature_base_name="sma", window=5,
            )
            assert "coherent_feature_tampering_detected" in _codes(crosscheck_report)
