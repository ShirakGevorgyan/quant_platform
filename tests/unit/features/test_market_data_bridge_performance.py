"""Performance and memory benchmarks (Milestone 10, Phase 4D, spec
Section 24). Correctness priority over throughput -- these are generous,
non-flaky wall-clock sanity bounds (never a tight microbenchmark floor),
proving the alignment/verification/coverage/build paths scale to a
"complete small research dataset" size (spec's own phrase) without a
pathological (e.g. accidentally quadratic) blowup, not asserting a
specific throughput number this platform commits to.

COMPLEXITY DOCUMENTED HERE, NOT RE-PROVEN: `features.alignment.
align_higher_timeframe`/`as_of_join_external` are vectorized
`numpy.searchsorted`/`pandas.merge_asof` operations -- O(n log m) in the
base-row count `n` and source-row count `m` -- already benchmarked by
M3's own `tests/performance/test_feature_throughput.py`; this file only
confirms the BRIDGE's own additive work (binding verification's fresh
store read + digest recomputation, coverage evaluation, lineage
assembly) does not introduce an unexpected extra pass over the data
per source added."""

from __future__ import annotations

import time
from datetime import timedelta
from decimal import Decimal

import pandas as pd
from _market_data_bridge_test_helpers import (
    BASE_TIME,
    make_cross_asset_fixture,
    make_macro_fixture,
    open_repository,
)

from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.base_asset_adapter import (
    resolve_base_asset_dataframe,
    verify_base_asset_binding,
)
from quant_platform.features.market_data_bridge.coverage import (
    SourceCoveragePolicy,
    SourceCoveragePolicyKind,
    evaluate_source_coverage,
)
from quant_platform.features.market_data_bridge.cross_asset_adapter import resolve_cross_asset_dataframe
from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage, lineage_content_id
from quant_platform.features.market_data_bridge.macro_adapter import resolve_macro_dataframe
from quant_platform.features.market_data_bridge.verification import (
    verify_truncation_invariance_cross_asset,
    verify_truncation_invariance_macro,
)
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.ingestion import ingest_raw_events
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)

_GENEROUS_BOUND_SECONDS = 60.0
_HOURS = 800  # ~1 month of H1 bars -- large enough to expose an O(n^2) alignment regression,
# small enough that this suite's own per-event durable-store fsync writes (`MarketEventStore.
# append`'s own real disk-sync-per-event discipline -- deliberate durability, not a test
# artifact) don't dominate wall time on a slow filesystem.


def _build_large_base_repo(tmp_path):
    repo = open_repository(tmp_path / "md")
    key = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="mt5")
    candles = []
    for i in range(_HOURS):
        close_val = 2000 + (i % 10) * 0.4
        candles.append(create_candle(
            instrument_id="XAUUSD", provider="mt5", symbol="XAUUSD", event_time=BASE_TIME + timedelta(hours=i), timeframe=Timeframe.H1,
            sequence=i, open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal(str(close_val)), volume=Decimal("100"),
        ))
    result = ingest_raw_events(
        repository=repo, dataset_key=key, batch_id="perf", ingestion_time=BASE_TIME, events=tuple(candles),
        partitioning=PartitioningSpec(granularity=PartitionGranularity.DAILY),
    )
    from quant_platform.features.market_data_bridge.bindings import create_base_asset_binding

    binding = create_base_asset_binding(canonical_instrument_id="XAUUSD", provider="mt5", pinned_dataset_id=result.resulting_dataset_id, timeframe=Timeframe.H1)
    return repo, binding


class TestBaseAssetAdapterPerformance:
    def test_verify_and_resolve_completes_within_generous_bound(self, tmp_path) -> None:
        repo, binding = _build_large_base_repo(tmp_path)
        started = time.perf_counter()
        candles = verify_base_asset_binding(repo, binding)
        df = resolve_base_asset_dataframe(repo, binding, start=BASE_TIME, end=BASE_TIME + timedelta(hours=_HOURS))
        elapsed = time.perf_counter() - started
        assert len(candles) == _HOURS
        assert len(df) == _HOURS
        assert elapsed < _GENEROUS_BOUND_SECONDS, f"base-asset verify+resolve took {elapsed:.2f}s for {_HOURS} bars"


class TestMacroAlignmentPerformance:
    def test_as_of_alignment_over_a_large_base_timeline(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=60)
        macro_df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        base_avail = pd.Series(pd.date_range(BASE_TIME, periods=_HOURS, freq="h", tz="UTC"))
        started = time.perf_counter()
        result = verify_truncation_invariance_macro(base_avail, macro_df, source_name="dfii10", truncate_after=base_avail.iloc[_HOURS // 2])
        elapsed = time.perf_counter() - started
        assert result.is_invariant
        assert elapsed < _GENEROUS_BOUND_SECONDS, f"macro truncation-invariance check took {elapsed:.2f}s for {_HOURS} base rows"


class TestCrossAssetAlignmentPerformance:
    def test_close_time_alignment_over_a_large_base_timeline(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=60)
        cross_df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        base_avail = pd.Series(pd.date_range(BASE_TIME, periods=_HOURS, freq="h", tz="UTC"))
        started = time.perf_counter()
        result = verify_truncation_invariance_cross_asset(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1, truncate_after=base_avail.iloc[_HOURS // 2])
        elapsed = time.perf_counter() - started
        assert result.is_invariant
        assert elapsed < _GENEROUS_BOUND_SECONDS, f"cross-asset truncation-invariance check took {elapsed:.2f}s for {_HOURS} base rows"


class TestCoverageEvaluationPerformance:
    def test_multi_source_coverage_evaluation_scales_with_source_count(self, tmp_path) -> None:
        base_df = pd.DataFrame({"open_time": pd.date_range(BASE_TIME, periods=_HOURS, freq="h", tz="UTC")})
        macro_frames = {}
        macro_bindings = {}
        for i in range(5):
            fixture = make_macro_fixture(tmp_path / f"m{i}", series_id=f"SERIES{i}", days=15)
            macro_frames[f"SERIES{i}"] = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
            macro_bindings[f"SERIES{i}"] = fixture.binding
        started = time.perf_counter()
        report = evaluate_source_coverage(
            base_df=base_df, base_timeframe=Timeframe.H1, macro_frames=macro_frames, macro_bindings=macro_bindings, cross_asset_frames={},
            cross_asset_bindings={}, requested_start=pd.Timestamp(BASE_TIME), requested_end=pd.Timestamp(BASE_TIME) + pd.Timedelta(hours=_HOURS),
            policy=SourceCoveragePolicy(kind=SourceCoveragePolicyKind.ALLOW_OPTIONAL_MISSING_AND_REPORT),
        )
        elapsed = time.perf_counter() - started
        assert len(report.findings) == 6  # base + 5 macro sources
        assert elapsed < _GENEROUS_BOUND_SECONDS, f"5-source coverage evaluation took {elapsed:.2f}s"


class TestLineageAssemblyPerformance:
    def test_lineage_assembly_and_fingerprint_over_many_sources(self, tmp_path) -> None:
        _repo, base_binding = _build_large_base_repo(tmp_path)
        macro_bindings = {}
        for i in range(8):
            fixture = make_macro_fixture(tmp_path / f"lm{i}", series_id=f"SERIES{i}", days=5)
            macro_bindings[f"SERIES{i}"] = fixture.binding
        started = time.perf_counter()
        lineage = build_market_data_lineage(base_binding=base_binding, macro_bindings=macro_bindings, cross_asset_bindings={})
        content_id = lineage_content_id(lineage)
        elapsed = time.perf_counter() - started
        assert content_id
        assert elapsed < _GENEROUS_BOUND_SECONDS, f"8-source lineage assembly took {elapsed:.2f}s"
