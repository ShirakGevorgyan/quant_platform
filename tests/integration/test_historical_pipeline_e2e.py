"""End-to-end integration test for the Milestone 2 historical data
pipeline: fixture source -> raw snapshot -> quality validation -> repair
-> canonical Parquet -> M1-to-H1 resampling -> manifest -> dataset loader
-> the EXISTING (Milestone 1) `BacktestEngine`.

This is the proof the Milestone 2 spec explicitly requires: that every new
pipeline stage, composed end to end, still preserves the platform's core
guarantee -- no look-ahead leakage -- and that cash/equity reconciliation
(a Milestone 1 invariant) continues to hold when the data feeding the
engine came from this pipeline rather than being hand-constructed.

The no-look-ahead proof does NOT just check that the run completes without
`LookaheadViolationError` (a useful but weak signal, since that guard is a
defensive assertion inside `TimeframeCursor` itself). It independently
recomputes, via plain pandas over the actual H1 DataFrame the engine was
given, which H1 bar SHOULD have been visible at every single base-clock
step, and compares that to what the strategy actually observed through
`StrategyContext.window` -- proving the full pipeline's output is leak-free
against a computation that does not reuse `TimeframeCursor`'s own logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd
import pytest

from quant_platform.core.types import Signal, SignalAction, Timeframe
from quant_platform.costs.models import FixedSpreadCostModel
from quant_platform.engine.backtest_engine import BacktestEngine
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.mt5_adapter import MT5AdapterConfig, MT5HistoricalSource
from quant_platform.historical.mt5_testing import FakeMt5Client
from quant_platform.historical.quality import run_quality_checks
from quant_platform.historical.raw_store import RawSnapshotStore
from quant_platform.historical.repair import SeverityPolicy, apply_repair_policy
from quant_platform.historical.resampling import DerivedBarPolicy, resample_ohlcv
from quant_platform.historical.source import SourceRequest
from quant_platform.historical.timezones import FixedOffsetTimezone
from quant_platform.historical.update_pipeline import apply_incremental_update
from quant_platform.risk.position_sizing import KellyCriterionSizer
from quant_platform.strategy.interfaces import Strategy, StrategyContext

UTC = "UTC"
SYMBOL = "XAUUSD"
BROKER = "E2ETestBroker"
SOURCE_NAME = "mt5"


@dataclass(slots=True)
class RecordingStrategy(Strategy):
    """Never trades on its own signal logic beyond a single, deliberately
    simple SMA-crossover rule -- its real job is recording, at every base
    (M1) step, exactly which H1 bar (`open_time`, or `None`) was visible
    through `StrategyContext.window`, so the test can independently verify
    that sequence never leaks a not-yet-closed H1 bar."""

    higher_timeframe: Timeframe
    fast_period: int = 5
    slow_period: int = 15
    observed: list[tuple[pd.Timestamp, pd.Timestamp | None]] = field(default_factory=list)

    def required_warmup(self, timeframe: Timeframe) -> int:
        return self.slow_period + 1 if timeframe is self.higher_timeframe else 0

    def on_bar(self, context: StrategyContext) -> Signal:
        window = context.window(self.higher_timeframe)
        visible_open_time = pd.Timestamp(window["open_time"].iloc[-1]) if len(window) else None
        self.observed.append((pd.Timestamp(context.timestamp), visible_open_time))

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


def _independent_expected_h1_open_time(h1_open_times: pd.Series, as_of: pd.Timestamp) -> pd.Timestamp | None:
    """Recomputed via plain pandas (never via `TimeframeCursor` or
    `resample_ohlcv`): the latest H1 bar whose CLOSE time (open_time + 1h)
    is <= `as_of`, or None if no H1 bar has closed yet."""
    eligible = h1_open_times[h1_open_times + timedelta(hours=1) <= as_of]
    return pd.Timestamp(eligible.max()) if len(eligible) else None


@pytest.fixture
def pipeline_storage(tmp_path):
    return RawSnapshotStore(tmp_path), CanonicalStore(tmp_path), ManifestStore(tmp_path)


class TestHistoricalPipelineEndToEnd:
    def test_full_pipeline_feeds_backtest_engine_without_lookahead_leakage(self, pipeline_storage) -> None:
        raw_store, canonical_store, manifest_store = pipeline_storage
        server_tz = FixedOffsetTimezone(timedelta(hours=2), name="EET")

        # --- Stage 1: fixture source (through the REAL MT5 adapter code
        # path, backed by the fake client -- not a hand-built DataFrame) ---
        adapter_config = MT5AdapterConfig(broker=BROKER, source_symbol="XAUUSDm", server_timezone=server_tz)
        source = MT5HistoricalSource(adapter_config, client=FakeMt5Client(seed=11, base_price=2000.0))
        source.connect()
        request = SourceRequest(
            symbol="XAUUSDm", timeframe=Timeframe.M1,
            start=pd.Timestamp("2024-03-04T00:00:00", tz=UTC),  # a Monday
            end=pd.Timestamp("2024-03-11T00:00:00", tz=UTC),    # 7 days later
            max_batch_size=100_000,
        )
        batches = list(source.fetch_all(request))
        source.disconnect()
        raw_df = pd.concat([b.data for b in batches], ignore_index=True)
        assert len(raw_df) == 7 * 24 * 60  # 7 days of M1 bars, hand-computed

        # --- Stage 2: immutable raw snapshot ---
        extracted_at = pd.Timestamp.now(tz=UTC)
        snapshot_metadata = raw_store.write_snapshot(
            raw_df, source_name=SOURCE_NAME, source_version="fake-1.0", broker=BROKER, symbol=SYMBOL,
            source_symbol="XAUUSDm", timeframe=Timeframe.M1, requested_start=request.start,
            requested_end=request.end, server_timezone_repr=str(server_tz), extracted_at=extracted_at,
            is_complete=True,
        )
        loaded_raw_df, _ = raw_store.read_snapshot(
            raw_store.dataset_dir(source_name=SOURCE_NAME, broker=BROKER, symbol=SYMBOL, timeframe=Timeframe.M1)
            / f"snapshot={snapshot_metadata.snapshot_id}"
        )

        # --- Stage 3: quality validation ---
        quality_report = run_quality_checks(loaded_raw_df, symbol=SYMBOL, timeframe=Timeframe.M1)
        assert quality_report.is_valid, quality_report.summary()

        # --- Stage 4: repair/quarantine policy (expected no-op on clean fixture data) ---
        repair_result = apply_repair_policy(
            loaded_raw_df, quality_report, policy=SeverityPolicy.STRICT,
            input_snapshot_id=snapshot_metadata.snapshot_id,
        )
        assert repair_result.lineage.rows_out == len(loaded_raw_df)
        assert repair_result.lineage.rows_quarantined == 0

        # --- Stage 5: canonical Parquet storage (M1) ---
        m1_update = apply_incremental_update(
            canonical_store, manifest_store, repair_result.data,
            symbol=SYMBOL, timeframe=Timeframe.M1, source_name=SOURCE_NAME, broker=BROKER,
            pipeline_version="1.0.0", parent_snapshot_ids=(snapshot_metadata.snapshot_id,),
            requested_start=request.start, requested_end=request.end,
            quality_summary={"critical": len(quality_report.critical_issues), "warning": len(quality_report.warnings)},
            repair_summary={"rows_removed": repair_result.lineage.rows_in - repair_result.lineage.rows_out},
        )
        assert m1_update.rows_inserted == len(repair_result.data)

        # --- Stage 6: leak-free M1 -> H1 resampling ---
        h1_derived = resample_ohlcv(
            repair_result.data, source_timeframe=Timeframe.M1, target_timeframe=Timeframe.H1,
            policy=DerivedBarPolicy.REJECT_INCOMPLETE,
        )
        assert len(h1_derived) == 7 * 24  # exactly 168 complete H1 bars from 7 clean days of M1
        assert h1_derived["is_complete"].all()
        h1_canonical = h1_derived[
            ["open_time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
        ]

        # --- Stage 7: canonical storage + manifest for the DERIVED dataset ---
        h1_update = apply_incremental_update(
            canonical_store, manifest_store, h1_canonical,
            symbol=SYMBOL, timeframe=Timeframe.H1, source_name=SOURCE_NAME, broker=BROKER,
            pipeline_version="1.0.0", parent_snapshot_ids=(snapshot_metadata.snapshot_id,),
            requested_start=request.start, requested_end=request.end,
            quality_summary={"critical": 0, "warning": 0},
            repair_summary={"rows_removed": 0},
            resampling_config={"source_timeframe": "M1", "policy": "REJECT_INCOMPLETE"},
        )
        assert h1_update.rows_inserted == len(h1_canonical)

        # --- Stage 8: reproducible dataset loader ---
        loader = DatasetLoader(canonical_store, manifest_store)
        m1_engine_df = loader.load_for_engine(
            LoadRequest(symbol=SYMBOL, timeframe=Timeframe.M1, start=request.start, end=request.end)
        )
        h1_engine_df = loader.load_for_engine(
            LoadRequest(
                symbol=SYMBOL, timeframe=Timeframe.H1, start=request.start,
                end=h1_canonical["open_time"].iloc[-1] + timedelta(hours=1),
            )
        )
        assert len(m1_engine_df) == len(repair_result.data)
        assert len(h1_engine_df) == len(h1_canonical)

        # --- Stage 9: feed the EXISTING (Milestone 1) BacktestEngine ---
        strategy = RecordingStrategy(higher_timeframe=Timeframe.H1)
        cost_model = FixedSpreadCostModel(spread_points=2.0, slippage_points=1.0, point_value=1.0, commission_per_unit=0.01)
        # RecordingStrategy's crossover signals never set `stop_loss` (it
        # exits via the opposite crossover, like `SmaCrossoverStrategy`),
        # so a sizer that needs no stop-loss distance is required here --
        # same reasoning as `test_full_backtest_sma.py`.
        sizer = KellyCriterionSizer(win_rate=0.5, win_loss_ratio=1.5, kelly_fraction=0.3, max_position_fraction=0.5)
        engine = BacktestEngine(
            data={Timeframe.M1: m1_engine_df, Timeframe.H1: h1_engine_df},
            base_timeframe=Timeframe.M1, strategy=strategy, cost_model=cost_model,
            position_sizer=sizer, initial_capital=10_000.0, point_value=1.0, symbol=SYMBOL,
        )
        result = engine.run()  # must not raise LookaheadViolationError

        # --- Proof 1: cash/equity reconciliation (Milestone 1 invariant) ---
        reconciled_equity = result.initial_capital + sum(t.net_pnl for t in result.trades)
        assert result.final_equity == pytest.approx(reconciled_equity, abs=1e-6)
        assert len(result.equity_curve) == len(m1_engine_df)

        # --- Proof 2: independent, non-cursor-based no-look-ahead check ---
        h1_open_times = h1_canonical["open_time"]
        mismatches = [
            (as_of, visible, _independent_expected_h1_open_time(h1_open_times, as_of))
            for as_of, visible in strategy.observed
            if visible != _independent_expected_h1_open_time(h1_open_times, as_of)
        ]
        assert mismatches == [], f"look-ahead leak detected at {len(mismatches)} step(s): {mismatches[:5]}"

        # Sanity: the strategy actually exercised its crossover logic (not
        # just HOLD the whole way through) -- otherwise proofs 1 and 2
        # above would hold trivially without exercising much.
        assert any(v is not None for _, v in strategy.observed)
        assert len(result.trades) > 0, "expected at least one trade over 7 days of volatile fixture data"
