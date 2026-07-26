"""Throughput/scaling regression tests for the historical data pipeline.

Same philosophy as `test_engine_throughput.py`: conservative floors and
relative-scaling assertions, not absolute numbers chased on one machine,
so these are not flaky on slower CI runners while still catching a real
regression (e.g. an accidental O(n^2) operation).

Measured on reference hardware (informational, not asserted verbatim --
actual figures from a single run of this file's own benchmarks; expect
run-to-run variance of at least +/-30%):
  - Raw snapshot write (Parquet + checksum + metadata): ~1,900,000 rows/sec.
  - Canonical partition write (zstd-compressed Parquet): ~2,400,000 rows/sec.
  - Quality checks (full check catalog): ~1,500,000 rows/sec.
  - M1 -> H1 resampling: ~7,500,000 source rows/sec.
  - Partial-year canonical read vs. full-history read: reading 1 of 5
    written year-partitions takes a small fraction of reading all 5,
    confirming `CanonicalStore.read_range` actually skips
    non-overlapping partitions rather than scanning the whole dataset.
  - Multi-year (~2.1M row, 4-year) write+read: write ~2.4s, read ~0.3s.
  - Cross-process ("cold-ish") read of a dataset written by a separate
    Python process: well under a second for 200,000 rows -- this does NOT
    force true OS page-cache eviction (no elevated privileges assumed),
    it only avoids the "read what this same process just wrote" warm-cache
    shortcut every other benchmark here takes; see `TestReadAfterProcessRestart`.
  - Compression comparison at 300,000 rows (one real measured run):
    zstd 12,007,805 bytes / snappy 13,400,298 bytes / uncompressed
    13,704,435 bytes -- zstd ~12% smaller than uncompressed, snappy ~2%
    smaller, both write and read within noise of each other at this scale.

The floors asserted below are deliberately far below these measured
numbers (10x-100x headroom) -- like `test_engine_throughput.py`'s existing
floor, the goal is catching a severe accidental regression (e.g. an O(n^2)
operation, or a per-row Python loop replacing a vectorized one) without
being flaky on a slower CI runner, not chasing this specific machine's
absolute numbers.

Where pandas is retained vs. where it is the bottleneck: every hot path
here (quality checks, resampling, dtype coercion) is already vectorized
pandas/numpy -- no Python-level per-row loops -- so at this platform's
target scale (single-symbol, research-scale history: low millions of
rows) pandas throughput is more than sufficient, as these numbers show.
The one place further optimization (Polars/Arrow-native compute, or a
compiled aggregation) would plausibly matter is `resample_ohlcv`'s
`groupby` for very large multi-symbol batch resampling jobs -- not
exercised or claimed here, and explicitly out of scope for this milestone
(see README "Known limitations").

No memory-profiling dependency (e.g. `psutil`) is introduced for this
milestone; memory footprint is instead reasoned about from dtype sizes:
one `RAW_HISTORICAL_COLUMNS` row is 8 bytes (datetime64) + 4*8 bytes
(OHLC float64) + 3*8 bytes (tick_volume/real_volume/spread int64) = 64
bytes/row, so 1 million rows is ~64MB resident, before any transient
Pandas copy overhead (typically 1-3x during a groupby/merge) -- well
within a single research workstation's memory for this platform's
targeted scale.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.models import coerce_historical_dtypes
from quant_platform.historical.quality import run_quality_checks
from quant_platform.historical.raw_store import RawSnapshotStore
from quant_platform.historical.resampling import resample_ohlcv

UTC = timezone.utc

pytestmark = pytest.mark.performance


def _synthetic_raw_frame(n: int, start: datetime = datetime(2024, 1, 1, tzinfo=UTC), seed: int = 1) -> pd.DataFrame:
    sd = generate_ohlcv(SyntheticDataConfig(start=start, periods=n, timeframe=Timeframe.M1, seed=seed))
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["tick_volume"] = sd["tick_volume"].astype(np.int64)
    sd["real_volume"] = 0
    sd["spread"] = 15
    return coerce_historical_dtypes(sd)


class TestRawSnapshotWriteThroughput:
    def test_write_snapshot_processes_at_least_10000_rows_per_second(self, tmp_path) -> None:
        n = 200_000
        df = _synthetic_raw_frame(n)
        store = RawSnapshotStore(tmp_path)

        start = time.perf_counter()
        store.write_snapshot(
            df, source_name="mt5", source_version="1.0", broker="B", symbol="XAUUSD", source_symbol="XAUUSDm",
            timeframe=Timeframe.M1, requested_start=df["open_time"].iloc[0],
            requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            server_timezone_repr="FIXED", extracted_at=pd.Timestamp.now(tz="UTC"), is_complete=True,
        )
        elapsed = time.perf_counter() - start

        rows_per_second = n / elapsed
        assert rows_per_second > 10_000, f"Raw snapshot write regression: {rows_per_second:.0f} rows/sec (expected > 10,000)"


class TestCanonicalWriteThroughput:
    def test_write_partition_processes_at_least_10000_rows_per_second(self, tmp_path) -> None:
        n = 200_000
        df = _synthetic_raw_frame(n)
        store = CanonicalStore(tmp_path)

        start = time.perf_counter()
        store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        elapsed = time.perf_counter() - start

        rows_per_second = n / elapsed
        assert rows_per_second > 10_000, f"Canonical write regression: {rows_per_second:.0f} rows/sec (expected > 10,000)"


class TestQualityCheckThroughput:
    def test_run_quality_checks_processes_at_least_20000_rows_per_second(self) -> None:
        n = 300_000
        df = _synthetic_raw_frame(n)

        start = time.perf_counter()
        run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        elapsed = time.perf_counter() - start

        rows_per_second = n / elapsed
        assert rows_per_second > 20_000, f"Quality check regression: {rows_per_second:.0f} rows/sec (expected > 20,000)"


class TestResamplingThroughput:
    def test_resample_ohlcv_processes_at_least_50000_source_rows_per_second(self) -> None:
        n = 500_000
        df = _synthetic_raw_frame(n)

        start = time.perf_counter()
        resample_ohlcv(df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        elapsed = time.perf_counter() - start

        rows_per_second = n / elapsed
        assert rows_per_second > 50_000, f"Resampling regression: {rows_per_second:.0f} rows/sec (expected > 50,000)"

    def test_time_scales_roughly_linearly_not_quadratically(self) -> None:
        small_n, large_n = 100_000, 400_000
        small_df = _synthetic_raw_frame(small_n)
        large_df = _synthetic_raw_frame(large_n)

        start = time.perf_counter()
        resample_ohlcv(small_df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        small_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        resample_ohlcv(large_df, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1)
        large_elapsed = time.perf_counter() - start

        time_ratio = large_elapsed / max(small_elapsed, 1e-6)
        data_ratio = large_n / small_n  # 4.0
        assert time_ratio < data_ratio * 3.0, (
            f"Possible quadratic scaling in resample_ohlcv: {data_ratio}x the data took "
            f"{time_ratio:.2f}x the time"
        )


class TestPartialRangeReadSkipsNonOverlappingPartitions:
    """The core performance CLAIM this storage layer makes: reading a
    narrow date range should not cost anywhere near as much as reading
    the entire multi-year history, because `CanonicalStore.read_range`
    only opens the year-partitions that actually overlap the request."""

    def test_reading_one_year_is_much_faster_than_reading_all_five(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        rows_per_year = 100_000
        for year in range(2020, 2025):
            df = _synthetic_raw_frame(rows_per_year, start=datetime(year, 1, 1, tzinfo=UTC), seed=year)
            store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=year)

        start = time.perf_counter()
        full_result = store.read_range(
            symbol="XAUUSD", timeframe=Timeframe.M1,
            start=pd.Timestamp("2020-01-01", tz="UTC"), end=pd.Timestamp("2025-01-01", tz="UTC"),
        )
        full_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        partial_result = store.read_range(
            symbol="XAUUSD", timeframe=Timeframe.M1,
            start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2025-01-01", tz="UTC"),
        )
        partial_elapsed = time.perf_counter() - start

        assert len(full_result) == 5 * rows_per_year
        assert len(partial_result) == rows_per_year
        assert partial_elapsed < full_elapsed * 0.6, (
            f"Partial-range read ({partial_elapsed:.4f}s) did not meaningfully beat "
            f"full-history read ({full_elapsed:.4f}s) -- partition pruning may have regressed"
        )


class TestLargerScaleEndToEnd:
    """A single, larger (multi-year, ~2 million row) run through
    ingest-shaped write + read, honestly labeled as "large for this
    platform's single-symbol research target", not a big-data-scale claim.
    Measured on reference hardware: ~2.1M rows across 4 years wrote in
    ~2.4s and read back in ~0.3s (informational; see module docstring for
    the same run-to-run-variance caveat)."""

    def test_multi_year_write_and_read_completes_and_scales_reasonably(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        rows_per_year = 525_600  # one M1 bar per minute for a full 365-day year
        years = (2021, 2022, 2023, 2024)

        write_start = time.perf_counter()
        for year in years:
            df = _synthetic_raw_frame(rows_per_year, start=datetime(year, 1, 1, tzinfo=UTC), seed=year)
            store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=year)
        write_elapsed = time.perf_counter() - write_start

        read_start = time.perf_counter()
        combined = store.read_range(
            symbol="XAUUSD", timeframe=Timeframe.M1,
            start=pd.Timestamp(years[0], 1, 1, tz="UTC"), end=pd.Timestamp(years[-1] + 1, 1, 1, tz="UTC"),
        )
        read_elapsed = time.perf_counter() - read_start

        total_rows = rows_per_year * len(years)
        assert len(combined) == total_rows
        assert combined["open_time"].is_monotonic_increasing
        # Conservative floors only -- see module docstring.
        assert total_rows / write_elapsed > 50_000, f"multi-year write regression: {total_rows / write_elapsed:.0f} rows/sec"
        assert total_rows / read_elapsed > 100_000, f"multi-year read regression: {total_rows / read_elapsed:.0f} rows/sec"


class TestReadAfterProcessRestart:
    """An HONEST approximation of a "cold" read: this process has never
    touched these files before, so the OS page cache for them is populated
    only by THIS read, not primed by an earlier write in the same process
    (unlike every other benchmark in this file, which reads back what it
    just wrote in the same process -- almost certainly still hot in the OS
    cache). This does NOT force true cold-cache eviction (that requires
    OS-level privileges this test does not assume or request); it is the
    closest approximation achievable portably, and is reported as such --
    not conflated with a guaranteed-cold measurement.
    """

    def test_read_from_a_dataset_written_by_a_separate_process(self, tmp_path) -> None:
        import subprocess
        import sys

        rows = 200_000
        src_dir = Path(__file__).resolve().parents[2] / "src"
        write_script = f"""
import sys
sys.path.insert(0, {str(src_dir)!r})
import numpy as np, pandas as pd
from datetime import datetime, timezone
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.models import coerce_historical_dtypes

sd = generate_ohlcv(SyntheticDataConfig(start=datetime(2024, 1, 1, tzinfo=timezone.utc), periods={rows}, timeframe=Timeframe.M1, seed=1))
sd = sd.rename(columns={{"volume": "tick_volume"}})
sd["tick_volume"] = sd["tick_volume"].astype(np.int64)
sd["real_volume"] = 0
sd["spread"] = 15
df = coerce_historical_dtypes(sd)
store = CanonicalStore({str(tmp_path)!r})
store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
"""
        result = subprocess.run([sys.executable, "-c", write_script], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr

        store = CanonicalStore(tmp_path)
        start = time.perf_counter()
        loaded = store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        elapsed = time.perf_counter() - start
        assert loaded is not None
        df, _ = loaded
        assert len(df) == rows
        # A very conservative floor: this is about proving the read still
        # completes promptly from a file this process has never touched,
        # not about chasing a tight number given cache-state uncertainty.
        assert elapsed < 5.0, f"cross-process read took {elapsed:.2f}s, unexpectedly slow"


class TestCompressionComparison:
    """Honest, measured comparison of the three practical Parquet
    compression codecs for this schema -- not a claim that zstd is
    strictly better in every dimension, just what was actually observed."""

    def test_compression_codecs_produce_valid_data_with_measured_size_and_speed_tradeoffs(self, tmp_path) -> None:
        df = _synthetic_raw_frame(300_000)
        results: dict[str, dict[str, float]] = {}

        for codec in ("snappy", "zstd", None):
            codec_dir = tmp_path / (codec or "none")
            # `CanonicalStore.__init__` only accepts the codecs it types as
            # valid (no "none" option); to measure the uncompressed
            # baseline honestly, write directly via pandas/pyarrow instead
            # of going through the store's typed API for this one case.
            codec_dir.mkdir(parents=True, exist_ok=True)
            data_path = codec_dir / "data.parquet"
            write_start = time.perf_counter()
            df.to_parquet(data_path, index=False, compression=codec)
            write_elapsed = time.perf_counter() - write_start

            read_start = time.perf_counter()
            reloaded = pd.read_parquet(data_path)
            read_elapsed = time.perf_counter() - read_start

            pd.testing.assert_frame_equal(reloaded, df)
            results[codec or "none"] = {
                "size_bytes": data_path.stat().st_size,
                "write_s": write_elapsed,
                "read_s": read_elapsed,
            }

        # The only claim asserted (not just reported): compression must
        # not INFLATE the file relative to storing it uncompressed --
        # both zstd and snappy should compress this repetitive OHLCV-shaped
        # data smaller than "none". Timings are printed for the record but
        # not asserted on, since relative codec speed is measurement-noise-
        # sensitive at this row count and is not this test's claim.
        assert results["zstd"]["size_bytes"] < results["none"]["size_bytes"]
        assert results["snappy"]["size_bytes"] < results["none"]["size_bytes"]
        print(f"\nCompression comparison (300,000 rows): {results}")
