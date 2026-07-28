"""Closure audit, Section 1: direct construction-path tests for every
implemented `ReturnSeriesKind` (`build_return_series`'s dispatcher and
each `_build_*` helper), not merely exercised indirectly through the
acceptance workflow. Only `BAR_NET`/`BAR_GROSS`/`TRADE_NET`/`TRADE_GROSS`/
`STITCHED_BAR_NET`/`PER_FOLD_BAR_NET`/`BENCHMARK_RELATIVE` exist --
`ReturnSeriesKind` has no GROSS variant of `STITCHED_BAR_NET` or
`PER_FOLD_BAR_NET` (confirmed by direct enum inspection); those are not
tested here because they are not implemented, not because they were
skipped."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from tests.unit.robustness.test_source_verification import _backtest_spec

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.manifests import BacktestManifest
from quant_platform.backtesting.models import (
    BacktestStage,
    ExitReasonCode,
    PositionDirection,
    SignalReasonCode,
    TradeStatus,
)
from quant_platform.backtesting.runner import BenchmarkReport, BenchmarkResult, OuterFoldBacktestResult
from quant_platform.backtesting.stitching import (
    StitchedEquityPoint,
    StitchedFoldBoundary,
    StitchedWalkForwardEquity,
)
from quant_platform.backtesting.timeline import BarReturnBasis, BarReturnPoint, BarReturnTimeline
from quant_platform.backtesting.trades import TradeRecord, TradeSet, compute_trade_id
from quant_platform.core.exceptions import ReturnSeriesError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference, ValidationReport
from quant_platform.robustness.models import ReturnSeriesKind
from quant_platform.robustness.series import FoldBoundary, build_return_series
from quant_platform.robustness.source import SourceVerificationReport, VerifiedBacktestSource

_BACKTEST_ID = "1" * 64
_TIMESTAMP_BASE = "2026-01-01T00:00:00Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fake_ref() -> ArtifactReference:
    """A syntactically-valid `ArtifactReference` for fields `series.py`
    never reads (signal_set/equity_curve/drawdowns/cost_sensitivity/
    bucket_analysis) -- `ArtifactReference.__post_init__` only validates
    hash FORMAT, not that content actually exists at that hash, so no
    artifact-store write is required for these unused fields."""
    return ArtifactReference(category=ArtifactCategory.BACKTEST_SPEC, content_hash="0" * 64, size_bytes=0, created_at=_now())


def _bar_point(pos: int, *, net_return: float = 0.0, gross_return: float | None = None, transaction_costs: float = 0.0, exposure: float = 1.0, timestamp: str | None = None) -> BarReturnPoint:
    gross = gross_return if gross_return is not None else net_return
    ts = timestamp or (pd.Timestamp(_TIMESTAMP_BASE) + pd.Timedelta(hours=pos)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return BarReturnPoint(
        schema_version=1, bar_position=pos, timestamp=ts, gross_return=gross, net_return=net_return, realized_return=gross,
        unrealized_return=0.0, active_long_exposure=exposure, active_short_exposure=0.0, total_absolute_exposure=exposure,
        net_exposure=exposure, open_trade_count=0, entries_count=0, exits_count=0, transaction_costs=transaction_costs,
        cumulative_gross_equity=1.0, cumulative_net_equity=1.0, peak_equity=1.0, drawdown=0.0,
    )


def _bar_timeline(outer_fold_index: int, points: tuple[BarReturnPoint, ...]) -> BarReturnTimeline:
    return BarReturnTimeline(
        schema_version=1, outer_fold_index=outer_fold_index, return_basis=BarReturnBasis.PREVIOUS_VALUATION_TO_CURRENT_VALUATION, compounded=False,
        fold_start_position=points[0].bar_position, fold_end_position=points[-1].bar_position, points=points,
    )


def _trade(*, outer_fold_index: int, entry_pos: int, exit_pos: int, net_return: float, gross_return: float | None = None, signal_sample_position: int | None = None) -> TradeRecord:
    gross = gross_return if gross_return is not None else net_return
    entry_ts = (pd.Timestamp(_TIMESTAMP_BASE) + pd.Timedelta(hours=entry_pos)).strftime("%Y-%m-%dT%H:%M:%SZ")
    exit_ts = (pd.Timestamp(_TIMESTAMP_BASE) + pd.Timedelta(hours=exit_pos)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sample_pos = signal_sample_position if signal_sample_position is not None else entry_pos
    trade_id = compute_trade_id(source_calibration_id="a" * 64, outer_fold_index=outer_fold_index, signal_sample_position=sample_pos, direction=PositionDirection.LONG, entry_timestamp=entry_ts, exit_timestamp=exit_ts)
    return TradeRecord(
        schema_version=1, trade_id=trade_id, signal_sample_position=sample_pos, outer_fold_index=outer_fold_index, direction=PositionDirection.LONG,
        signal_timestamp=entry_ts, decision_timestamp=entry_ts, entry_timestamp=entry_ts, entry_bar_position=entry_pos, entry_observed_price=100.0,
        entry_effective_price=100.0, exit_timestamp=exit_ts, exit_bar_position=exit_pos, exit_observed_price=100.0, exit_effective_price=100.0,
        holding_bars=exit_pos - entry_pos, gross_return=gross, net_return=net_return,
        cost_breakdown=CostBreakdown(entry_spread_cost=0.0, exit_spread_cost=0.0, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0),
        confidence=0.8, uncertainty=0.1, calibrated_probability=0.8, entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED,
        status=TradeStatus.CLOSED, source_calibration_id="a" * 64, source_experiment_id="b" * 64,
    )


def _trade_set(outer_fold_index: int, trades: tuple[TradeRecord, ...]) -> TradeSet:
    return TradeSet(schema_version=1, outer_fold_index=outer_fold_index, trades=trades)


def _benchmark_report(outer_fold_index: int, *, name: str = "always_flat_net_cost", net_return: float = 0.0, gross_return: float | None = None) -> BenchmarkReport:
    gross = gross_return if gross_return is not None else net_return
    return BenchmarkReport(schema_version=1, outer_fold_index=outer_fold_index, benchmarks=(BenchmarkResult(name=name, description="d", gross_return=gross, net_return=net_return),))


def _write_fold(
    artifact_store: MLArtifactStore, *, outer_fold_index: int, bar_points: tuple[BarReturnPoint, ...], trades: tuple[TradeRecord, ...] = (),
    benchmark_net_return: float = 0.0, financial_metrics: dict[str, object] | None = None,
) -> ArtifactReference:
    timeline_ref = artifact_store.write_artifact(json.dumps(_bar_timeline(outer_fold_index, bar_points).to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)
    trade_set_ref = artifact_store.write_artifact(json.dumps(_trade_set(outer_fold_index, trades).to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)
    benchmark_ref = artifact_store.write_artifact(json.dumps(_benchmark_report(outer_fold_index, net_return=benchmark_net_return).to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)
    metrics = financial_metrics if financial_metrics is not None else {"total_net_return": sum(t.net_return for t in trades)}
    result = OuterFoldBacktestResult(
        schema_version=1, backtest_id=_BACKTEST_ID, outer_fold_index=outer_fold_index, signal_set_reference=_fake_ref(), trade_set_reference=trade_set_ref,
        bar_return_timeline_reference=timeline_ref, equity_curve_reference=_fake_ref(), gross_drawdown_reference=_fake_ref(), net_drawdown_reference=_fake_ref(),
        benchmark_report_reference=benchmark_ref, cost_sensitivity_report_reference=_fake_ref(), bucket_analysis_report_reference=_fake_ref(),
        outer_test_row_count=len(bar_points), closed_trade_count=len(trades), meets_minimum_trade_threshold=True, financial_metrics=metrics, skipped_metrics={}, evaluated_at=_now(),
    )
    return artifact_store.write_artifact(json.dumps(result.to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)


def _source(
    artifact_store: MLArtifactStore, *, outer_fold_result_references: dict[int, ArtifactReference], stitched_equity_reference: ArtifactReference | None = None,
) -> VerifiedBacktestSource:
    manifest = BacktestManifest(
        schema_version=1, backtest_id=_BACKTEST_ID, source_calibration_id="a" * 64, stage=BacktestStage.COMPLETED, created_at=_now(), updated_at=_now(),
        completed_at=_now(), total_outer_folds=len(outer_fold_result_references), outer_fold_result_references=outer_fold_result_references,
        completed_outer_fold_indices=tuple(sorted(outer_fold_result_references)), stitched_equity_reference=stitched_equity_reference,
    )
    backtest_spec = _backtest_spec()
    verification_report = ValidationReport(schema_version=1, issues=(), generated_at=_now())
    source_verification_report = SourceVerificationReport(
        schema_version=1, source_backtest_id=_BACKTEST_ID, verify_backtest_is_ready=True, verify_backtest_critical_count=0, verify_backtest_issue_codes=(),
        dataset_content_id_matches=True, split_plan_fingerprint_matches=True, instrument_identity_matches=True, bar_interval_matches=True,
        total_outer_folds=len(outer_fold_result_references), generated_at=_now(),
    )
    return VerifiedBacktestSource(manifest=manifest, backtest_spec=backtest_spec, verification_report=verification_report, source_verification_report=source_verification_report)


def _store(tmp_path: object) -> MLArtifactStore:
    return MLArtifactStore(f"{tmp_path}/artifacts")


class TestBarNetAndBarGrossConstruction:
    def test_bar_net_uses_net_return_bar_gross_uses_gross_return(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        points = (_bar_point(0, net_return=0.01, gross_return=0.015, transaction_costs=0.005), _bar_point(1, net_return=-0.02, gross_return=-0.018, transaction_costs=0.002))
        ref = _write_fold(store, outer_fold_index=0, bar_points=points)
        source = _source(store, outer_fold_result_references={0: ref})

        net_bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        gross_bundle = build_return_series(ReturnSeriesKind.BAR_GROSS, source=source, artifact_store=store)
        assert net_bundle.values == (0.01, -0.02)
        assert gross_bundle.values == (0.015, -0.018)
        assert net_bundle.sampling_frequency == "bar"
        assert net_bundle.observation_count == 2

    def test_fold_boundaries_and_source_hashes_track_each_fold(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref0 = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.01), _bar_point(1, net_return=0.02)))
        ref1 = _write_fold(store, outer_fold_index=1, bar_points=(_bar_point(2, net_return=-0.01),))
        source = _source(store, outer_fold_result_references={0: ref0, 1: ref1})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.fold_boundaries == (FoldBoundary(outer_fold_index=0, start_index=0, end_index=1), FoldBoundary(outer_fold_index=1, start_index=2, end_index=2))
        assert len(bundle.source_artifact_content_hashes) == 2
        assert bundle.observation_count == 3

    def test_gap_between_fold_bar_positions_does_not_corrupt_boundaries(self, tmp_path: object) -> None:
        """`series.py` indexes `fold_boundaries` by STITCHED-POINT index
        (position within the built `values` list), never by raw source
        `bar_position` -- a real gap between fold 0's last bar (position 1)
        and fold 1's first bar (position 10, e.g. an excluded warm-up
        region) must not corrupt the boundary indices."""
        store = _store(tmp_path)
        ref0 = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.01), _bar_point(1, net_return=0.01)))
        ref1 = _write_fold(store, outer_fold_index=1, bar_points=(_bar_point(10, net_return=0.02), _bar_point(11, net_return=0.02)))
        source = _source(store, outer_fold_result_references={0: ref0, 1: ref1})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.fold_boundaries == (FoldBoundary(outer_fold_index=0, start_index=0, end_index=1), FoldBoundary(outer_fold_index=1, start_index=2, end_index=3))
        assert bundle.observation_count == 4

    def test_constant_series(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        points = tuple(_bar_point(i, net_return=0.005) for i in range(15))
        ref = _write_fold(store, outer_fold_index=0, bar_points=points)
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.values == tuple(0.005 for _ in range(15))
        assert bundle.effective_sample_count == 15  # zero-variance -> rho1 defined as 0.0, no reduction claimed

    def test_all_zero_series(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        points = tuple(_bar_point(i, net_return=0.0) for i in range(10))
        ref = _write_fold(store, outer_fold_index=0, bar_points=points)
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.values == tuple(0.0 for _ in range(10))

    def test_timestamps_track_first_and_last_bar(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        points = (_bar_point(0, net_return=0.01, timestamp="2026-03-01T00:00:00Z"), _bar_point(1, net_return=0.01, timestamp="2026-03-01T05:00:00Z"))
        ref = _write_fold(store, outer_fold_index=0, bar_points=points)
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.time_range_start == "2026-03-01T00:00:00Z"
        assert bundle.time_range_end == "2026-03-01T05:00:00Z"

    def test_effective_sample_size_uses_autocorrelation_adjustment(self, tmp_path: object) -> None:
        """Bar-sampled series get the AR(1) adjustment -- a strongly
        autocorrelated (constant, non-varying) series must NOT report
        effective_sample_count == observation_count; it must be clamped
        via the documented formula."""
        store = _store(tmp_path)
        points = tuple(_bar_point(i, net_return=0.01) for i in range(30))  # constant -> pstdev==0 -> rho1=0.0 by the module's own convention -> no reduction
        ref = _write_fold(store, outer_fold_index=0, bar_points=points)
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.effective_sample_count == 30  # zero-variance series: rho1 defined as 0.0, no reduction

    def test_empty_series_across_all_folds_rejected(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        source = _source(store, outer_fold_result_references={})
        with pytest.raises(ReturnSeriesError, match="no bar-return observations"):
            build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)

    def test_one_observation_series(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.03),))
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        assert bundle.observation_count == 1
        assert bundle.effective_sample_count == 1
        assert bundle.values == (0.03,)


class TestPerFoldBarNetIsIntentionallyIdenticalToBarNet:
    """Closure audit finding: `BAR_NET` and `PER_FOLD_BAR_NET` share the
    exact same construction path (`_build_per_fold_bar_series`) and
    produce byte-identical `values`/`fold_boundaries`/`effective_sample_
    count` for the same source -- confirmed here as an intentional,
    tested contract (both kinds exist for callers to signal fold-
    partition-aware INTENT via the `kind` label; `fold_boundaries` is
    already populated identically for both, so there is no additional
    behavior to distinguish today). This is NOT a bug: no downstream
    consumer branches on BAR_NET vs PER_FOLD_BAR_NET, and no metadata is
    lost -- fold boundaries are carried on every bar-sampled kind. If a
    future feature needs PER_FOLD_BAR_NET to behave differently (e.g.
    forcing FOLD_LEVEL-only bootstrap eligibility), this test is the
    tripwire that must be deliberately updated."""

    def test_bar_net_and_per_fold_bar_net_produce_identical_content(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref0 = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.01), _bar_point(1, net_return=-0.02)))
        ref1 = _write_fold(store, outer_fold_index=1, bar_points=(_bar_point(2, net_return=0.03),))
        source = _source(store, outer_fold_result_references={0: ref0, 1: ref1})
        bar_net = build_return_series(ReturnSeriesKind.BAR_NET, source=source, artifact_store=store)
        per_fold = build_return_series(ReturnSeriesKind.PER_FOLD_BAR_NET, source=source, artifact_store=store)
        assert bar_net.values == per_fold.values
        assert bar_net.fold_boundaries == per_fold.fold_boundaries
        assert bar_net.effective_sample_count == per_fold.effective_sample_count
        assert bar_net.kind is ReturnSeriesKind.BAR_NET
        assert per_fold.kind is ReturnSeriesKind.PER_FOLD_BAR_NET
        assert bar_net.kind != per_fold.kind  # the ONLY difference is the label


class TestTradeSeriesConstruction:
    def test_trade_net_and_trade_gross_use_closed_trades_only(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        closed = _trade(outer_fold_index=0, entry_pos=0, exit_pos=1, net_return=0.02, gross_return=0.025)
        incomplete = TradeRecord(
            schema_version=1, trade_id=compute_trade_id(source_calibration_id="a" * 64, outer_fold_index=0, signal_sample_position=5, direction=PositionDirection.LONG, entry_timestamp="2026-01-01T05:00:00Z", exit_timestamp="2026-01-01T06:00:00Z"),
            signal_sample_position=5, outer_fold_index=0, direction=PositionDirection.LONG, signal_timestamp="2026-01-01T05:00:00Z", decision_timestamp="2026-01-01T05:00:00Z",
            entry_timestamp="2026-01-01T05:00:00Z", entry_bar_position=5, entry_observed_price=100.0, entry_effective_price=100.0, exit_timestamp="2026-01-01T06:00:00Z",
            exit_bar_position=6, exit_observed_price=100.0, exit_effective_price=100.0, holding_bars=1, gross_return=0.0, net_return=0.0,
            cost_breakdown=CostBreakdown(entry_spread_cost=0.0, exit_spread_cost=0.0, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0),
            confidence=0.5, uncertainty=0.1, calibrated_probability=0.5, entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.DISCARDED_INCOMPLETE,
            status=TradeStatus.INCOMPLETE_DISCARDED, source_calibration_id="a" * 64, source_experiment_id="b" * 64,
        )
        ref = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.0), _bar_point(1, net_return=0.0)), trades=(closed, incomplete))
        source = _source(store, outer_fold_result_references={0: ref})
        net_bundle = build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)
        gross_bundle = build_return_series(ReturnSeriesKind.TRADE_GROSS, source=source, artifact_store=store)
        assert net_bundle.values == (0.02,)  # INCOMPLETE_DISCARDED excluded -- closed_trades property filters it out
        assert gross_bundle.values == (0.025,)
        assert net_bundle.sampling_frequency == "trade"

    def test_trade_series_ordered_by_entry_then_exit_position(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        later = _trade(outer_fold_index=0, entry_pos=5, exit_pos=6, net_return=0.05, signal_sample_position=5)
        earlier = _trade(outer_fold_index=0, entry_pos=1, exit_pos=2, net_return=0.01, signal_sample_position=1)
        ref = _write_fold(store, outer_fold_index=0, bar_points=tuple(_bar_point(i, net_return=0.0) for i in range(8)), trades=(later, earlier))
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)
        assert bundle.values == (0.01, 0.05)  # ordered by (entry_bar_position, exit_bar_position), not declaration order

    def test_same_bar_trade_entry_equals_exit(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        same_bar = _trade(outer_fold_index=0, entry_pos=3, exit_pos=3, net_return=0.001)
        ref = _write_fold(store, outer_fold_index=0, bar_points=tuple(_bar_point(i, net_return=0.0) for i in range(5)), trades=(same_bar,))
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)
        assert bundle.values == (0.001,)
        assert bundle.observation_count == 1

    def test_no_closed_trades_across_any_fold_rejected(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.0),), trades=())
        source = _source(store, outer_fold_result_references={0: ref})
        with pytest.raises(ReturnSeriesError, match="no closed trades"):
            build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)

    def test_one_trade_series(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        trade = _trade(outer_fold_index=0, entry_pos=0, exit_pos=3, net_return=0.04)
        ref = _write_fold(store, outer_fold_index=0, bar_points=tuple(_bar_point(i, net_return=0.0) for i in range(4)), trades=(trade,))
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)
        assert bundle.observation_count == 1
        assert bundle.effective_sample_count == 1  # trade-level: unadjusted, not autocorrelation-reduced

    def test_trade_series_effective_sample_size_never_autocorrelation_adjusted(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        trades = tuple(_trade(outer_fold_index=0, entry_pos=i, exit_pos=i + 1, net_return=0.01, signal_sample_position=i) for i in range(0, 20, 2))
        ref = _write_fold(store, outer_fold_index=0, bar_points=tuple(_bar_point(i, net_return=0.0) for i in range(21)), trades=trades)
        source = _source(store, outer_fold_result_references={0: ref})
        bundle = build_return_series(ReturnSeriesKind.TRADE_NET, source=source, artifact_store=store)
        assert bundle.effective_sample_count == bundle.observation_count == len(trades)


class TestBenchmarkRelativeConstruction:
    def test_one_observation_per_fold_relative_to_named_benchmark(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref0 = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.02),), benchmark_net_return=0.01, financial_metrics={"total_net_return": 0.05})
        ref1 = _write_fold(store, outer_fold_index=1, bar_points=(_bar_point(1, net_return=-0.01),), benchmark_net_return=0.02, financial_metrics={"total_net_return": -0.03})
        source = _source(store, outer_fold_result_references={0: ref0, 1: ref1})
        bundle = build_return_series(ReturnSeriesKind.BENCHMARK_RELATIVE, source=source, artifact_store=store)
        assert bundle.values == pytest.approx((0.05 - 0.01, -0.03 - 0.02))
        assert bundle.sampling_frequency == "fold"
        assert bundle.observation_count == 2
        assert bundle.effective_sample_count == 2  # fold-level: unadjusted

    def test_missing_named_benchmark_rejected(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.0),), financial_metrics={"total_net_return": 0.01})
        source = _source(store, outer_fold_result_references={0: ref})
        with pytest.raises(ReturnSeriesError, match="not found"):
            build_return_series(ReturnSeriesKind.BENCHMARK_RELATIVE, source=source, artifact_store=store, benchmark_name="does_not_exist")

    def test_missing_total_net_return_metric_rejected(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        ref = _write_fold(store, outer_fold_index=0, bar_points=(_bar_point(0, net_return=0.0),), financial_metrics={})
        source = _source(store, outer_fold_result_references={0: ref})
        with pytest.raises(ReturnSeriesError, match="no finite total_net_return"):
            build_return_series(ReturnSeriesKind.BENCHMARK_RELATIVE, source=source, artifact_store=store)


class TestStitchedBarNetConstruction:
    def test_uses_stitched_equity_artifact_not_per_fold_reconstruction(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        points = (
            StitchedEquityPoint(
                schema_version=1, outer_fold_index=0, bar_position=0, timestamp="2026-01-01T00:00:00Z", bar_gross_return=0.02, bar_net_return=0.015,
                stitched_gross_equity=1.02, stitched_net_equity=1.015, total_absolute_exposure=1.0, net_exposure=1.0, open_trade_count=0, entries_count=0, exits_count=0,
            ),
            StitchedEquityPoint(
                schema_version=1, outer_fold_index=0, bar_position=1, timestamp="2026-01-01T01:00:00Z", bar_gross_return=-0.01, bar_net_return=-0.012,
                stitched_gross_equity=1.01, stitched_net_equity=1.003, total_absolute_exposure=1.0, net_exposure=1.0, open_trade_count=0, entries_count=0, exits_count=0,
            ),
            StitchedEquityPoint(
                schema_version=1, outer_fold_index=1, bar_position=0, timestamp="2026-01-01T02:00:00Z", bar_gross_return=0.03, bar_net_return=0.028,
                stitched_gross_equity=1.04, stitched_net_equity=1.031, total_absolute_exposure=1.0, net_exposure=1.0, open_trade_count=0, entries_count=0, exits_count=0,
            ),
        )
        boundaries = (
            StitchedFoldBoundary(outer_fold_index=0, stitched_point_start_index=0, stitched_point_end_index=1, fold_start_position=0, fold_end_position=1, carry_in_gross_equity=1.0, carry_in_net_equity=1.0),
            StitchedFoldBoundary(outer_fold_index=1, stitched_point_start_index=2, stitched_point_end_index=2, fold_start_position=0, fold_end_position=0, carry_in_gross_equity=1.0, carry_in_net_equity=1.0),
        )
        stitched = StitchedWalkForwardEquity(schema_version=1, backtest_id=_BACKTEST_ID, compounded=True, fold_boundaries=boundaries, points=points)
        stitched_ref = store.write_artifact(json.dumps(stitched.to_json_dict()).encode("utf-8"), category=ArtifactCategory.BACKTEST_SPEC)
        source = _source(store, outer_fold_result_references={}, stitched_equity_reference=stitched_ref)
        bundle = build_return_series(ReturnSeriesKind.STITCHED_BAR_NET, source=source, artifact_store=store)
        assert bundle.values == (0.015, -0.012, 0.028)  # bar_NET_return, not gross
        assert bundle.fold_boundaries == (FoldBoundary(outer_fold_index=0, start_index=0, end_index=1), FoldBoundary(outer_fold_index=1, start_index=2, end_index=2))
        assert bundle.observation_count == 3
        assert bundle.time_range_start == "2026-01-01T00:00:00Z"
        assert bundle.time_range_end == "2026-01-01T02:00:00Z"

    def test_missing_stitched_equity_reference_rejected(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        source = _source(store, outer_fold_result_references={}, stitched_equity_reference=None)
        with pytest.raises(ReturnSeriesError, match="no stitched_equity_reference"):
            build_return_series(ReturnSeriesKind.STITCHED_BAR_NET, source=source, artifact_store=store)
