"""Throughput benchmarks for the Milestone 3 feature engineering platform.
Same philosophy as `test_historical_pipeline_throughput.py`: conservative
floors (10x-100x below measured numbers) to catch a severe accidental
regression (an O(n^2) operation, a per-row Python loop replacing a
vectorized one) without being flaky on a slower runner -- not an attempt to
chase this specific machine's absolute numbers.

Measured on reference hardware (informational, one real run of this file's
own benchmarks; expect run-to-run variance of at least +/-30%):
  - Raw technical feature computation (19 registered features, trailing
    rolling windows, 500,000 rows): 0.254s, ~1,971,000 rows/sec.
  - `align_higher_timeframe` (500,000 base rows against 2,000 H1 bars):
    0.441s, ~1,134,000 rows/sec.
  - Full `FeatureEngine.compute` (technical + temporal + multi-timeframe,
    35 features, dependency-resolved, 200,000 rows): 1.652s, ~121,000
    rows/sec -- markedly slower than raw technical-only computation
    because multi-timeframe features each independently recompute their
    own alignment pass (a documented simplicity-over-micro-optimization
    tradeoff, see `features.multi_timeframe`'s module docstring).
  - Research dataset artifact write (`ResearchDatasetStore.write_artifacts`,
    3 splits, 200,000 total rows, zstd-compressed Parquet): 0.148s,
    ~1,349,000 rows/sec.
  - Manifest + artifact reconstruction round-trip (save manifest, load by
    version, read artifacts back, verify checksums, 200,000-row dataset):
    34.3ms.
"""

from __future__ import annotations

import time

import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.features.multi_timeframe import MultiTimeframeWindows, register_multi_timeframe_features
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.features.temporal.calendar_features import register_core_temporal_features

pytestmark = pytest.mark.performance


def _engine_df(raw_df):
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


class TestTechnicalFeatureThroughput:
    def test_processes_at_least_100000_rows_per_second(self) -> None:
        registry = FeatureRegistry()
        register_core_technical_features(registry, timeframe=Timeframe.M1, windows=TechnicalWindows())
        base_df = _engine_df(make_synthetic_ohlcv(500_000, seed=1))
        names = tuple(s.name for s in registry.list_features())

        engine = FeatureEngine(registry)
        started = time.perf_counter()
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=names)
        elapsed = time.perf_counter() - started

        rows_per_sec = len(base_df) / elapsed
        print(f"\nTechnical feature computation: {len(base_df)} rows, {len(names)} features, {elapsed:.3f}s, {rows_per_sec:,.0f} rows/sec")
        assert len(result.features) == len(base_df)
        assert rows_per_sec > 100_000


class TestAlignmentThroughput:
    def test_align_higher_timeframe_processes_at_least_500000_rows_per_second(self) -> None:
        base_df = make_synthetic_ohlcv(500_000, freq_minutes=1, seed=2)
        higher_df = make_synthetic_ohlcv(2000, freq_minutes=60, seed=3)
        base_close_times = base_df["open_time"] + Timeframe.M1.duration

        started = time.perf_counter()
        aligned = align_higher_timeframe(base_close_times, higher_df, Timeframe.H1)
        elapsed = time.perf_counter() - started

        rows_per_sec = len(base_df) / elapsed
        print(f"\nalign_higher_timeframe: {len(base_df)} rows, {elapsed:.3f}s, {rows_per_sec:,.0f} rows/sec")
        assert len(aligned) == len(base_df)
        assert rows_per_sec > 500_000


class TestFullEngineThroughput:
    def test_full_feature_set_processes_at_least_20000_rows_per_second(self) -> None:
        registry = FeatureRegistry()
        register_core_technical_features(registry, timeframe=Timeframe.M1, windows=TechnicalWindows())
        register_core_temporal_features(registry, timeframe=Timeframe.M1, calendar=None)
        register_multi_timeframe_features(
            registry, base_timeframe=Timeframe.M1, higher_timeframe=Timeframe.H1, windows=MultiTimeframeWindows()
        )

        base_df = _engine_df(make_synthetic_ohlcv(200_000, freq_minutes=1, seed=4))
        higher_df = _engine_df(make_synthetic_ohlcv(len(base_df) // 60 + 5, freq_minutes=60, seed=5))
        names = tuple(s.name for s in registry.list_features())

        engine = FeatureEngine(registry)
        started = time.perf_counter()
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=names,
            higher_timeframe_data={Timeframe.H1: higher_df},
        )
        elapsed = time.perf_counter() - started

        rows_per_sec = len(base_df) / elapsed
        print(f"\nFull engine ({len(names)} features incl. multi-timeframe): {len(base_df)} rows, {elapsed:.3f}s, {rows_per_sec:,.0f} rows/sec")
        assert len(result.features) == len(base_df)
        assert rows_per_sec > 20_000


class TestDatasetWriteThroughput:
    def test_write_artifacts_processes_at_least_50000_rows_per_second(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        train = _engine_df(make_synthetic_ohlcv(140_000, seed=6))
        validation = _engine_df(make_synthetic_ohlcv(30_000, seed=7))
        test = _engine_df(make_synthetic_ohlcv(30_000, seed=8))
        splits = {"train": train, "validation": validation, "test": test}
        total_rows = sum(len(df) for df in splits.values())

        started = time.perf_counter()
        _content_id, checksums = store.write_artifacts("perf_ds", splits=splits, preprocessing_json={"global": {}})
        elapsed = time.perf_counter() - started

        rows_per_sec = total_rows / elapsed
        print(f"\nResearchDatasetStore.write_artifacts: {total_rows} rows, {elapsed:.3f}s, {rows_per_sec:,.0f} rows/sec")
        assert len(checksums) == 3
        assert rows_per_sec > 50_000


class TestManifestReconstructionLatency:
    def test_save_load_and_read_round_trip_completes_quickly(self, tmp_path) -> None:
        research_store = ResearchDatasetStore(tmp_path)
        manifest_store = ResearchManifestStore(tmp_path)
        splits = {
            "train": _engine_df(make_synthetic_ohlcv(140_000, seed=9)),
            "validation": _engine_df(make_synthetic_ohlcv(30_000, seed=10)),
            "test": _engine_df(make_synthetic_ohlcv(30_000, seed=11)),
        }
        content_id, output_hashes = research_store.write_artifacts(
            "perf_ds2", splits=splits, preprocessing_json={"global": {}}
        )
        manifest = ResearchDatasetManifest(
            dataset_id="perf_ds2", version="", source_historical_dataset_id="hist", source_historical_manifest_version="v1",
            symbol="XAUUSD", base_timeframe=Timeframe.M1, utc_start=splits["train"]["open_time"].iloc[0],
            utc_end=splits["test"]["open_time"].iloc[-1], feature_names=("open",), feature_versions={"open": "1"},
            feature_registry_fingerprint="fp", label_definition={}, split_definition={}, preprocessing_definition={},
            fitted_preprocessing_fingerprint=None, code_revision="content:abc", input_content_hashes={},
            output_content_hashes=output_hashes, row_counts={k: len(v) for k, v in splits.items()},
            missing_data_summary={}, leakage_validation_result={"is_valid": True},
            created_at=splits["train"]["open_time"].iloc[0], content_id=content_id,
        )

        started = time.perf_counter()
        version = manifest_store.save(manifest)
        loaded = manifest_store.load("perf_ds2", version)
        reconstructed = research_store.read_artifacts(loaded.dataset_id, loaded.content_id)
        elapsed = time.perf_counter() - started

        print(f"\nManifest save+load+read round trip: {elapsed * 1000:.1f}ms")
        assert reconstructed is not None
        assert elapsed < 1.0
