"""The concrete reproducibility proof required by the Milestone 2
release-readiness audit.

Scenario, exactly as specified:

  1. Build dataset version V1 (a fresh ingest through the full pipeline).
  2. Run a backtest against V1 (pinned `dataset_version`) and fingerprint
     the complete result (every trade, every equity point, final equity).
  3. Revise one historical source bar (`RevisionPolicy.ACCEPT_NEWER_SOURCE`).
  4. Build V2 (the manifest version produced by that revision).
  5. Reload V1 explicitly, months "later" in wall-clock terms (nothing
     about `DatasetLoader.load` depends on when it is called).
  6. Prove V1's data and backtest fingerprint are IDENTICAL to the
     original run -- the revision must not have touched what V1 means.
  7. Prove V2 is DISTINCT -- both in data and in backtest fingerprint.
  8. Prove both V1 and V2 remain independently loadable at the end.

This is what actually validates the redesigned content-addressed
`CanonicalStore` + `DatasetManifest.partition_content_ids` +
`DatasetLoader` reconstruction path end to end, against a real
(fake-MT5-backed) ingestion pipeline and the real `BacktestEngine` -- not
a unit-level assertion about one function in isolation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, replace
from datetime import timedelta

import pandas as pd
import pytest

from quant_platform.core.types import Signal, SignalAction, Timeframe
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.engine.backtest_engine import BacktestEngine, BacktestResult
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.mt5_adapter import MT5AdapterConfig, MT5HistoricalSource
from quant_platform.historical.mt5_testing import FakeMt5Client
from quant_platform.historical.quality import run_quality_checks
from quant_platform.historical.raw_store import RawSnapshotStore
from quant_platform.historical.repair import SeverityPolicy, apply_repair_policy
from quant_platform.historical.source import SourceRequest
from quant_platform.historical.timezones import FixedOffsetTimezone
from quant_platform.historical.update_pipeline import RevisionPolicy, apply_incremental_update
from quant_platform.risk.position_sizing import KellyCriterionSizer
from quant_platform.strategy.interfaces import Strategy, StrategyContext

UTC = "UTC"
SYMBOL = "XAUUSD"
BROKER = "ReproAuditBroker"
SOURCE_NAME = "mt5"
REPRODUCIBILITY_SEED = 42


@dataclass(slots=True)
class DeterministicCrossoverStrategy(Strategy):
    """A plain, deterministic SMA-crossover strategy -- no randomness of
    its own. Any run-to-run difference in a fingerprinted backtest against
    this strategy can only come from the DATA it was fed, which is exactly
    the property this test needs to isolate."""

    timeframe: Timeframe
    fast_period: int = 5
    slow_period: int = 15

    def required_warmup(self, timeframe: Timeframe) -> int:
        return self.slow_period + 1 if timeframe is self.timeframe else 0

    def on_bar(self, context: StrategyContext) -> Signal:
        window = context.window(self.timeframe)
        if len(window) < self.slow_period + 1:
            return Signal(timestamp=context.timestamp, action=SignalAction.HOLD)
        closes = window["close"]
        fast_now = closes.iloc[-self.fast_period :].mean()
        slow_now = closes.iloc[-self.slow_period :].mean()
        fast_prev = closes.iloc[-self.fast_period - 1 : -1].mean()
        slow_prev = closes.iloc[-self.slow_period - 1 : -1].mean()
        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal(timestamp=context.timestamp, action=SignalAction.LONG)
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(timestamp=context.timestamp, action=SignalAction.FLAT)
        return Signal(timestamp=context.timestamp, action=SignalAction.HOLD)


@dataclass(frozen=True, slots=True)
class ReproducibilityBundle:
    """Everything that must be pinned for a backtest to be exactly
    reproducible: the dataset version (not just symbol/timeframe/range --
    the EXACT content), the strategy's own parameters, the cost model's
    parameters, the position sizer's parameters, and the seed that
    generated the underlying fixture data. Serializing this alongside a
    result's fingerprint is what would let a real experiment be re-run
    months later with confidence it is the SAME experiment, not merely a
    similar one."""

    dataset_version: str
    symbol: str
    timeframe: str
    start: str
    end: str
    strategy_fast_period: int
    strategy_slow_period: int
    cost_spread_points: float
    cost_slippage_points: float
    cost_commission_per_unit: float
    sizer_win_rate: float
    sizer_win_loss_ratio: float
    sizer_kelly_fraction: float
    initial_capital: float
    reproducibility_seed: int

    def fingerprint(self) -> str:
        payload = "|".join(f"{f.name}={getattr(self, f.name)!r}" for f in sorted(fields(self), key=lambda f: f.name))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fingerprint_backtest_result(result: BacktestResult) -> str:
    """A complete, deterministic fingerprint of a `BacktestResult`: every
    trade's every field, every equity point, and final equity. Two
    fingerprints matching is exactly as strong a claim as "these two runs
    produced byte-for-byte identical trades and equity curves" -- not an
    approximation."""
    parts: list[str] = [f"initial_capital={result.initial_capital!r}", f"final_equity={result.final_equity!r}"]
    for trade in result.trades:
        parts.append(
            f"trade|{trade.entry_time}|{trade.exit_time}|{trade.side.value}|{trade.entry_price!r}|"
            f"{trade.exit_price!r}|{trade.quantity!r}|{trade.gross_pnl!r}|{trade.total_cost!r}|{trade.exit_reason}"
        )
    for point in result.equity_curve:
        parts.append(f"equity|{point.timestamp}|{point.cash!r}|{point.equity!r}|{point.drawdown_pct!r}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _run_backtest(engine_df_m1: pd.DataFrame, bundle: ReproducibilityBundle) -> BacktestResult:
    strategy = DeterministicCrossoverStrategy(
        timeframe=Timeframe.M1, fast_period=bundle.strategy_fast_period, slow_period=bundle.strategy_slow_period
    )
    cost_model = FixedSpreadCostModel(
        spread_points=bundle.cost_spread_points, slippage_points=bundle.cost_slippage_points,
        point_value=1.0, commission_per_unit=bundle.cost_commission_per_unit,
    )
    sizer = KellyCriterionSizer(
        win_rate=bundle.sizer_win_rate, win_loss_ratio=bundle.sizer_win_loss_ratio,
        kelly_fraction=bundle.sizer_kelly_fraction, max_position_fraction=0.5,
    )
    engine = BacktestEngine(
        data={Timeframe.M1: engine_df_m1}, base_timeframe=Timeframe.M1, strategy=strategy,
        cost_model=cost_model, position_sizer=sizer, initial_capital=bundle.initial_capital,
        point_value=1.0, symbol=bundle.symbol,
    )
    return engine.run()


@pytest.fixture
def pipeline_storage(tmp_path):
    return RawSnapshotStore(tmp_path), CanonicalStore(tmp_path), ManifestStore(tmp_path)


def _ingest_range(
    raw_store: RawSnapshotStore, canonical_store: CanonicalStore, manifest_store: ManifestStore,
    *, start: pd.Timestamp, end: pd.Timestamp, seed: int,
) -> str:
    """Ingest `[start, end)` of fixture M1 data through the full pipeline
    (source -> raw snapshot -> quality -> repair -> canonical update) and
    return the resulting manifest version."""
    server_tz = FixedOffsetTimezone(timedelta(hours=2), name="EET")
    adapter_config = MT5AdapterConfig(broker=BROKER, source_symbol="XAUUSDm", server_timezone=server_tz)
    source = MT5HistoricalSource(adapter_config, client=FakeMt5Client(seed=seed, base_price=2000.0))
    source.connect()
    try:
        request = SourceRequest(symbol="XAUUSDm", timeframe=Timeframe.M1, start=start, end=end, max_batch_size=100_000)
        raw_df = pd.concat([b.data for b in source.fetch_all(request)], ignore_index=True)
    finally:
        source.disconnect()

    extracted_at = pd.Timestamp.now(tz=UTC)
    snapshot_metadata = raw_store.write_snapshot(
        raw_df, source_name=SOURCE_NAME, source_version="fake-1.0", broker=BROKER, symbol=SYMBOL,
        source_symbol="XAUUSDm", timeframe=Timeframe.M1, requested_start=start, requested_end=end,
        server_timezone_repr=str(server_tz), extracted_at=extracted_at, is_complete=True,
    )
    quality_report = run_quality_checks(raw_df, symbol=SYMBOL, timeframe=Timeframe.M1)
    assert quality_report.is_valid, quality_report.summary()
    repair_result = apply_repair_policy(
        raw_df, quality_report, policy=SeverityPolicy.STRICT, input_snapshot_id=snapshot_metadata.snapshot_id,
    )
    update_report = apply_incremental_update(
        canonical_store, manifest_store, repair_result.data,
        symbol=SYMBOL, timeframe=Timeframe.M1, source_name=SOURCE_NAME, broker=BROKER,
        pipeline_version="1.0.0", parent_snapshot_ids=(snapshot_metadata.snapshot_id,),
        requested_start=start, requested_end=end, reproducibility_seed=seed,
        quality_summary={"critical": 0, "warning": 0}, repair_summary={"rows_removed": 0},
    )
    return update_report.manifest_version


class TestV1V2ReproducibilityProof:
    def test_pinned_dataset_version_reproduces_byte_identical_data_and_backtest_after_a_later_revision(
        self, pipeline_storage
    ) -> None:
        raw_store, canonical_store, manifest_store = pipeline_storage
        start = pd.Timestamp("2024-03-04T00:00:00", tz=UTC)
        end = pd.Timestamp("2024-03-06T00:00:00", tz=UTC)

        # --- Step 1: build dataset version V1 ---
        v1 = _ingest_range(raw_store, canonical_store, manifest_store, start=start, end=end, seed=REPRODUCIBILITY_SEED)

        loader = DatasetLoader(canonical_store, manifest_store)
        v1_bundle = ReproducibilityBundle(
            dataset_version=v1, symbol=SYMBOL, timeframe=Timeframe.M1.value, start=str(start), end=str(end),
            strategy_fast_period=5, strategy_slow_period=15, cost_spread_points=2.0, cost_slippage_points=1.0,
            cost_commission_per_unit=0.01, sizer_win_rate=0.5, sizer_win_loss_ratio=1.5, sizer_kelly_fraction=0.3,
            initial_capital=10_000.0, reproducibility_seed=REPRODUCIBILITY_SEED,
        )

        # --- Step 2: run and fingerprint a backtest against V1 ---
        v1_engine_df_original = loader.load_for_engine(
            LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v1)
        )
        v1_result_original = _run_backtest(v1_engine_df_original, v1_bundle)
        v1_fingerprint_original = _fingerprint_backtest_result(v1_result_original)
        assert len(v1_result_original.trades) > 0, "expected at least one trade to make this proof non-trivial"

        # --- Step 3+4: revise one historical source bar -> build V2 ---
        v1_data = loader.load(LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v1))
        revised_row = v1_data.iloc[[20]].copy()
        # Shift the WHOLE bar (not just close) by the same offset, so the
        # revision stays OHLC-valid (high >= open/close >= low) while still
        # being an unmistakable, large, easily-asserted-on revision.
        for price_column in ("open", "high", "low", "close"):
            revised_row[price_column] = revised_row[price_column] + 500.0
        revised_open_time = revised_row["open_time"].iloc[0]
        v2 = apply_incremental_update(
            canonical_store, manifest_store, revised_row,
            symbol=SYMBOL, timeframe=Timeframe.M1, source_name=SOURCE_NAME, broker=BROKER,
            pipeline_version="1.0.0", parent_snapshot_ids=(),
            requested_start=revised_open_time, requested_end=revised_open_time + timedelta(minutes=1),
            revision_policy=RevisionPolicy.ACCEPT_NEWER_SOURCE, reproducibility_seed=REPRODUCIBILITY_SEED,
        ).manifest_version
        assert v2 != v1

        # --- Step 5: reload V1 explicitly (as if "months later") ---
        v1_engine_df_reloaded = loader.load_for_engine(
            LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v1)
        )

        # --- Step 6: V1 data and backtest fingerprint are IDENTICAL ---
        pd.testing.assert_frame_equal(v1_engine_df_original, v1_engine_df_reloaded)
        v1_result_reloaded = _run_backtest(v1_engine_df_reloaded, v1_bundle)
        v1_fingerprint_reloaded = _fingerprint_backtest_result(v1_result_reloaded)
        assert v1_fingerprint_reloaded == v1_fingerprint_original
        assert v1_bundle.fingerprint() == v1_bundle.fingerprint()  # config bundle itself is stable/hashable

        # --- Step 7: V2 is DISTINCT, both in data and in backtest outcome ---
        v2_bundle = replace(v1_bundle, dataset_version=v2)
        v2_engine_df = loader.load_for_engine(
            LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v2)
        )
        assert not v1_engine_df_original["close"].equals(v2_engine_df["close"])
        v2_result = _run_backtest(v2_engine_df, v2_bundle)
        v2_fingerprint = _fingerprint_backtest_result(v2_result)
        assert v2_fingerprint != v1_fingerprint_original
        assert v2_bundle.fingerprint() != v1_bundle.fingerprint()

        # --- Step 8: both V1 and V2 remain independently loadable ---
        final_v1_check = loader.load(LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v1))
        final_v2_check = loader.load(LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=start, end=end, dataset_version=v2))
        assert len(final_v1_check) == len(v1_data)
        assert len(final_v2_check) == len(v1_data)
        original_close = v1_data.loc[v1_data["open_time"] == revised_open_time, "close"].iloc[0]
        assert final_v1_check.loc[final_v1_check["open_time"] == revised_open_time, "close"].iloc[0] == original_close
        assert final_v2_check.loc[final_v2_check["open_time"] == revised_open_time, "close"].iloc[0] == original_close + 500.0

    def test_manifest_records_reproducibility_seed_and_code_revision(self, pipeline_storage) -> None:
        raw_store, canonical_store, manifest_store = pipeline_storage
        start = pd.Timestamp("2024-03-04T00:00:00", tz=UTC)
        end = pd.Timestamp("2024-03-05T00:00:00", tz=UTC)
        version = _ingest_range(raw_store, canonical_store, manifest_store, start=start, end=end, seed=REPRODUCIBILITY_SEED)
        manifest = manifest_store.load(symbol=SYMBOL, timeframe=Timeframe.M1, version=version)
        assert manifest.reproducibility_seed == REPRODUCIBILITY_SEED
        assert manifest.code_revision is not None
        assert manifest.code_revision.startswith("git:") or manifest.code_revision.startswith("content:")
