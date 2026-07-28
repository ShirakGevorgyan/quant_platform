"""Milestone 6, Section 16 / closure-audit Sections 2-3: fold stability
and concentration-risk detection, against hand-computed reference
concentration ratios and dispersion statistics. Expected values are
computed independently (literal arithmetic or direct `statistics.*`
stdlib calls on hand-picked numbers) -- never by calling
`compute_fold_stability_report`/`compute_concentration_report`
themselves to derive their own expected output.

DENOMINATOR POLICY (read directly from `stability.py`, documented here
for the audit record):
- `profitable_fold_fraction`, `median_fold_return`, `worst_fold_return`,
  `fold_return_stdev`: denominator is VALID folds only (folds with a
  defined `total_net_return`) -- a fold missing this metric is silently
  excluded from these five, but still counts toward `fold_count`.
- `positive_sharpe_fold_fraction`, `fold_sharpe_dispersion`: VALID folds
  only (defined `bar_return_sharpe`); `None` if zero are defined.
- `maximum_fold_drawdown`: VALID folds only, but defaults to `0.0` (not
  `None`) if NONE are defined -- a deliberate "no evidence of drawdown"
  reading, not "drawdown is unknown".
- `worst_fold_cost_drag`, `fold_exposure_dispersion`: VALID folds only,
  `None` if zero are defined.
- `fold_trade_count_dispersion`: ALL folds (`closed_trade_count` is never
  "undefined" -- a zero-trade fold contributes `0.0`).
- `direction_consistency`: ELIGIBLE folds only (folds where trades exist
  AND are not perfectly long/short balanced); `None` if no fold is
  eligible.
- `benchmark_outperformance_fraction`: ELIGIBLE folds only (a matching
  named benchmark result AND a defined fold return both present); `None`
  if zero are eligible.
- Concentration ratios (`_concentration_ratio`): POSITIVE contributions
  only, `max(positive) / sum(positive)`; `None` (never `0.0`) if there is
  no positive total to divide by."""

from __future__ import annotations

import statistics

import pytest

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.models import ExitReasonCode, PositionDirection, SignalReasonCode, TradeStatus
from quant_platform.backtesting.runner import BenchmarkReport, BenchmarkResult, OuterFoldBacktestResult
from quant_platform.backtesting.trades import TradeRecord, compute_trade_id
from quant_platform.core.exceptions import StabilityAnalysisError
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.robustness.specs import StabilityThresholds
from quant_platform.robustness.stability import compute_concentration_report, compute_fold_stability_report

_DUMMY_REF = ArtifactReference(content_hash="a" * 64, category=ArtifactCategory.TRADE_SET, size_bytes=1, created_at="2026-01-01T00:00:00Z")
_THRESHOLDS = StabilityThresholds(
    minimum_profitable_fold_fraction=0.5, maximum_single_fold_profit_concentration=0.6,
    maximum_single_trade_profit_concentration=0.5, maximum_single_direction_profit_concentration=0.9,
)


_OMIT = object()  # sentinel: omit this metric from financial_metrics entirely ("skipped fold metric")


def _fold(
    outer_fold_index: int, *, total_net_return: object = 0.0, closed_trade_count: int = 2, sharpe: object = _OMIT,
    max_dd: object = 0.05, cost_drag: object = 0.001, exposure: object = 0.4,
) -> OuterFoldBacktestResult:
    metrics: dict[str, object] = {}
    if total_net_return is not _OMIT:
        metrics["total_net_return"] = total_net_return
    if sharpe is not _OMIT:
        metrics["bar_return_sharpe"] = sharpe
    if max_dd is not _OMIT:
        metrics["maximum_drawdown"] = max_dd
    if cost_drag is not _OMIT:
        metrics["mean_trade_cost"] = cost_drag
    if exposure is not _OMIT:
        metrics["time_in_market_fraction"] = exposure
    return OuterFoldBacktestResult(
        schema_version=1, backtest_id="a" * 64, outer_fold_index=outer_fold_index, signal_set_reference=_DUMMY_REF, trade_set_reference=_DUMMY_REF,
        bar_return_timeline_reference=_DUMMY_REF, equity_curve_reference=_DUMMY_REF, gross_drawdown_reference=_DUMMY_REF, net_drawdown_reference=_DUMMY_REF,
        benchmark_report_reference=_DUMMY_REF, cost_sensitivity_report_reference=_DUMMY_REF, bucket_analysis_report_reference=_DUMMY_REF,
        outer_test_row_count=100, closed_trade_count=closed_trade_count, meets_minimum_trade_threshold=True, financial_metrics=metrics,
        skipped_metrics={}, evaluated_at="2026-01-01T00:00:00Z",
    )


def _trade(
    signal_sample_position: int, *, outer_fold_index: int, net_return: float, direction: PositionDirection = PositionDirection.LONG, confidence: float = 0.6,
    exit_timestamp: str = "2024-01-01T05:00:00+00:00",
) -> TradeRecord:
    cb = CostBreakdown(entry_spread_cost=0.0005, exit_spread_cost=0.0005, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0)
    entry_ts = "2024-01-01T01:00:00+00:00"
    exit_ts = exit_timestamp
    trade_id = compute_trade_id(
        source_calibration_id="a" * 64, outer_fold_index=outer_fold_index, signal_sample_position=signal_sample_position, direction=direction,
        entry_timestamp=entry_ts, exit_timestamp=exit_ts,
    )
    return TradeRecord(
        schema_version=1, trade_id=trade_id, signal_sample_position=signal_sample_position, outer_fold_index=outer_fold_index, direction=direction,
        signal_timestamp="2024-01-01T00:00:00+00:00", decision_timestamp="2024-01-01T00:00:00+00:00", entry_timestamp=entry_ts,
        entry_bar_position=1, entry_observed_price=100.0, entry_effective_price=100.0, exit_timestamp=exit_ts, exit_bar_position=5,
        exit_observed_price=101.0, exit_effective_price=101.0, holding_bars=4, gross_return=net_return + 0.001, net_return=net_return, cost_breakdown=cb,
        confidence=confidence, uncertainty=0.2, calibrated_probability=0.7, entry_reason=SignalReasonCode.ACCEPTED_POSITIVE,
        exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED, source_calibration_id="a" * 64, source_experiment_id="b" * 64,
    )


class TestConcentrationRatioHandComputed:
    def test_single_fold_concentration_matches_hand_computed_ratio(self) -> None:
        """Fold profits: 0.10, 0.05, -0.02 (a loss, excluded from the
        positive-contribution set). Ratio = max(0.10, 0.05) / (0.10+0.05)
        = 0.10 / 0.15 = 0.6666..."""
        folds = (_fold(0, total_net_return=0.10), _fold(1, total_net_return=0.05), _fold(2, total_net_return=-0.02))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == pytest.approx(0.10 / 0.15, abs=1e-12)

    def test_single_trade_concentration_matches_hand_computed_ratio(self) -> None:
        """Trade net returns: 0.06, 0.03, 0.01 (all winners). Ratio =
        0.06 / (0.06+0.03+0.01) = 0.06 / 0.10 = 0.6."""
        trades = (_trade(0, outer_fold_index=0, net_return=0.06), _trade(1, outer_fold_index=0, net_return=0.03), _trade(2, outer_fold_index=0, net_return=0.01))
        report = compute_concentration_report((_fold(0, total_net_return=0.10),), all_closed_trades=trades, thresholds=_THRESHOLDS)
        assert report.single_trade_profit_concentration == pytest.approx(0.6, abs=1e-12)

    def test_no_positive_contributions_gives_none_not_zero(self) -> None:
        """Concentration is UNDEFINED (not 0.0) when there is no positive
        total to divide by -- a losing-only fold set."""
        folds = (_fold(0, total_net_return=-0.05), _fold(1, total_net_return=-0.02))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration is None

    def test_warning_code_emitted_when_threshold_exceeded(self) -> None:
        folds = (_fold(0, total_net_return=0.95), _fold(1, total_net_return=0.05))  # concentration = 0.95
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert "single_fold_profit_concentration_exceeded" in report.warning_codes

    def test_no_warning_when_under_threshold(self) -> None:
        folds = (_fold(0, total_net_return=0.10), _fold(1, total_net_return=0.09))  # concentration ~0.526, under 0.6
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert "single_fold_profit_concentration_exceeded" not in report.warning_codes


class TestFoldStabilityReportHandComputed:
    def test_profitable_fold_fraction_matches_hand_count(self) -> None:
        folds = (_fold(0, total_net_return=0.10), _fold(1, total_net_return=-0.02), _fold(2, total_net_return=0.03), _fold(3, total_net_return=0.01))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.profitable_fold_fraction == pytest.approx(3.0 / 4.0, abs=1e-12)

    def test_worst_and_median_fold_return_match_hand_computed_values(self) -> None:
        folds = (_fold(0, total_net_return=0.10), _fold(1, total_net_return=-0.02), _fold(2, total_net_return=0.03))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.worst_fold_return == pytest.approx(-0.02, abs=1e-12)
        assert report.median_fold_return == pytest.approx(0.03, abs=1e-12)  # median of [-0.02, 0.03, 0.10]

    def test_empty_fold_results_rejected(self) -> None:
        with pytest.raises(StabilityAnalysisError):
            compute_fold_stability_report((), all_closed_trades=(), thresholds=_THRESHOLDS)

    def test_direction_consistency_hand_computed(self) -> None:
        """Fold 0: 2 long trades (dominant=long). Fold 1: 1 long, 2 short
        (dominant=short). Overall: 3 long, 2 short -> overall dominant is
        long. direction_consistency = fraction of folds whose OWN
        dominant direction matches the OVERALL dominant direction = 1/2
        (only fold 0 matches)."""
        trades = (
            _trade(0, outer_fold_index=0, net_return=0.01, direction=PositionDirection.LONG),
            _trade(1, outer_fold_index=0, net_return=0.01, direction=PositionDirection.LONG),
            _trade(2, outer_fold_index=1, net_return=0.01, direction=PositionDirection.LONG),
            _trade(3, outer_fold_index=1, net_return=0.01, direction=PositionDirection.SHORT),
            _trade(4, outer_fold_index=1, net_return=0.01, direction=PositionDirection.SHORT),
        )
        folds = (_fold(0, total_net_return=0.02, closed_trade_count=2), _fold(1, total_net_return=0.03, closed_trade_count=3))
        report = compute_fold_stability_report(folds, all_closed_trades=trades, thresholds=_THRESHOLDS)
        assert report.direction_consistency == pytest.approx(0.5, abs=1e-12)

    def test_json_round_trip(self) -> None:
        folds = (_fold(0, total_net_return=0.10, sharpe=1.2), _fold(1, total_net_return=-0.02, sharpe=-0.3))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert type(report).from_json_dict(report.to_json_dict()) == report


class TestOneFoldAndTwoFoldEdgeCases:
    def test_one_fold_profitable_fraction_is_one_and_dispersions_are_zero_or_none(self) -> None:
        folds = (_fold(0, total_net_return=0.05, sharpe=1.0),)
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.profitable_fold_fraction == 1.0
        assert report.fold_return_stdev == 0.0  # < 2 valid folds -> 0.0, not undefined
        assert report.fold_sharpe_dispersion is None  # < 2 valid folds -> None (sharpe dispersion IS optional)
        assert report.fold_trade_count_dispersion == 0.0
        assert report.worst_fold_return == pytest.approx(0.05, abs=1e-12)
        assert report.median_fold_return == pytest.approx(0.05, abs=1e-12)

    def test_one_fold_unprofitable_fraction_is_zero(self) -> None:
        folds = (_fold(0, total_net_return=-0.03),)
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.profitable_fold_fraction == 0.0

    def test_two_folds_all_positive(self) -> None:
        folds = (_fold(0, total_net_return=0.02), _fold(1, total_net_return=0.05))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.profitable_fold_fraction == 1.0
        assert report.fold_return_stdev == pytest.approx(statistics.pstdev([0.02, 0.05]), abs=1e-12)

    def test_two_folds_all_negative(self) -> None:
        folds = (_fold(0, total_net_return=-0.02), _fold(1, total_net_return=-0.05))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.profitable_fold_fraction == 0.0
        assert report.worst_fold_return == pytest.approx(-0.05, abs=1e-12)

    def test_tied_fold_returns(self) -> None:
        folds = (_fold(0, total_net_return=0.03), _fold(1, total_net_return=0.03))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_return_stdev == 0.0  # tied values -> zero dispersion, not undefined
        assert report.median_fold_return == pytest.approx(0.03, abs=1e-12)

    def test_one_extreme_outlier_dominates_stdev(self) -> None:
        folds = (_fold(0, total_net_return=0.01), _fold(1, total_net_return=0.01), _fold(2, total_net_return=5.0))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_return_stdev == pytest.approx(statistics.pstdev([0.01, 0.01, 5.0]), abs=1e-9)
        assert report.median_fold_return == pytest.approx(0.01, abs=1e-12)  # median is robust to the outlier; stdev is not


class TestPositiveSharpeFractionAndDispersionHandComputed:
    def test_positive_sharpe_fraction_with_one_undefined_sharpe(self) -> None:
        """Sharpes: [1.2, -0.3, undefined]. sharpes_defined=[1.2,-0.3].
        positive_sharpe_fold_fraction = 1/2 = 0.5 -- the undefined fold
        contributes to neither the numerator nor denominator."""
        folds = (_fold(0, total_net_return=0.1, sharpe=1.2), _fold(1, total_net_return=-0.05, sharpe=-0.3), _fold(2, total_net_return=0.02, sharpe=_OMIT))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.positive_sharpe_fold_fraction == pytest.approx(0.5, abs=1e-12)
        assert report.fold_sharpe_dispersion == pytest.approx(statistics.pstdev([1.2, -0.3]), abs=1e-12)

    def test_all_sharpes_undefined_reports_none_not_zero(self) -> None:
        folds = (_fold(0, total_net_return=0.1, sharpe=_OMIT), _fold(1, total_net_return=0.02, sharpe=_OMIT))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.positive_sharpe_fold_fraction is None
        assert report.fold_sharpe_dispersion is None

    def test_sharpe_recorded_as_none_treated_same_as_absent_key(self) -> None:
        folds = (_fold(0, total_net_return=0.1, sharpe=None), _fold(1, total_net_return=0.02, sharpe=1.0))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.positive_sharpe_fold_fraction == 1.0  # only fold 1's sharpe counts


class TestMaximumDrawdownAndCostDragAndExposureHandComputed:
    def test_maximum_drawdown_is_the_max_across_valid_folds(self) -> None:
        folds = (_fold(0, total_net_return=0.1, max_dd=0.05), _fold(1, total_net_return=0.02, max_dd=0.30), _fold(2, total_net_return=0.01, max_dd=0.10))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.maximum_fold_drawdown == pytest.approx(0.30, abs=1e-12)

    def test_maximum_drawdown_defaults_to_zero_not_none_when_all_skipped(self) -> None:
        """Denominator policy: an entirely-missing drawdown metric across
        every fold reads as `0.0` ("no evidence of drawdown"), NOT `None`
        -- distinct from every other optional aggregate in this report."""
        folds = (_fold(0, total_net_return=0.1, max_dd=_OMIT), _fold(1, total_net_return=0.02, max_dd=_OMIT))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.maximum_fold_drawdown == 0.0

    def test_worst_fold_cost_drag_is_the_max_across_valid_folds(self) -> None:
        folds = (_fold(0, total_net_return=0.1, cost_drag=0.001), _fold(1, total_net_return=0.02, cost_drag=0.015))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.worst_fold_cost_drag == pytest.approx(0.015, abs=1e-12)

    def test_worst_fold_cost_drag_none_when_all_skipped(self) -> None:
        folds = (_fold(0, total_net_return=0.1, cost_drag=_OMIT), _fold(1, total_net_return=0.02, cost_drag=_OMIT))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.worst_fold_cost_drag is None

    def test_fold_exposure_dispersion_hand_computed(self) -> None:
        folds = (_fold(0, total_net_return=0.1, exposure=0.3), _fold(1, total_net_return=0.02, exposure=0.5))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_exposure_dispersion == pytest.approx(statistics.pstdev([0.3, 0.5]), abs=1e-12)

    def test_fold_exposure_dispersion_none_when_all_skipped(self) -> None:
        folds = (_fold(0, total_net_return=0.1, exposure=_OMIT), _fold(1, total_net_return=0.02, exposure=_OMIT))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_exposure_dispersion is None


class TestFoldTradeCountDispersionIncludesZeroTradeFolds:
    def test_zero_trade_fold_contributes_zero_not_excluded(self) -> None:
        """`fold_trade_count_dispersion` uses ALL folds' `closed_trade_
        count` -- a zero-trade fold is never "undefined" and must count
        toward the denominator, unlike every other optional metric."""
        folds = (_fold(0, total_net_return=0.1, closed_trade_count=2), _fold(1, total_net_return=0.0, closed_trade_count=0), _fold(2, total_net_return=-0.01, closed_trade_count=5))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_trade_count_dispersion == pytest.approx(statistics.pstdev([2.0, 0.0, 5.0]), abs=1e-12)


class TestSkippedFoldMetricStillCountsTowardFoldCount:
    def test_fold_with_missing_total_net_return_excluded_from_profitable_fraction_but_counted_in_fold_count(self) -> None:
        folds = (_fold(0, total_net_return=0.1), _fold(1, total_net_return=_OMIT), _fold(2, total_net_return=-0.02))
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.fold_count == 3  # the skipped fold is still counted as a fold
        assert report.profitable_fold_fraction == pytest.approx(0.5, abs=1e-12)  # denominator is 2 (valid), not 3

    def test_all_folds_missing_total_net_return_fails_closed(self) -> None:
        folds = (_fold(0, total_net_return=_OMIT), _fold(1, total_net_return=_OMIT))
        with pytest.raises(StabilityAnalysisError, match="no fold reports a defined total_net_return"):
            compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)


class TestBenchmarkOutperformanceFractionHandComputed:
    def test_wins_and_losses_against_named_benchmark(self) -> None:
        """Fold 0: fold return 0.05 > benchmark 0.01 -> win. Fold 1: fold
        return -0.02 < benchmark 0.00 -> loss. outperformance = 1/2."""
        folds = (_fold(0, total_net_return=0.05), _fold(1, total_net_return=-0.02))
        benchmarks = (
            BenchmarkReport(schema_version=1, outer_fold_index=0, benchmarks=(BenchmarkResult(name="always_flat_net_cost", description="d", gross_return=0.01, net_return=0.01),)),
            BenchmarkReport(schema_version=1, outer_fold_index=1, benchmarks=(BenchmarkResult(name="always_flat_net_cost", description="d", gross_return=0.0, net_return=0.0),)),
        )
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS, benchmark_reports=benchmarks)
        assert report.benchmark_outperformance_fraction == pytest.approx(0.5, abs=1e-12)

    def test_no_benchmark_reports_supplied_gives_none(self) -> None:
        folds = (_fold(0, total_net_return=0.05),)
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS, benchmark_reports=())
        assert report.benchmark_outperformance_fraction is None

    def test_benchmark_name_mismatch_excludes_fold_from_denominator(self) -> None:
        folds = (_fold(0, total_net_return=0.05),)
        benchmarks = (BenchmarkReport(schema_version=1, outer_fold_index=0, benchmarks=(BenchmarkResult(name="a_different_benchmark", description="d", gross_return=0.0, net_return=0.0),)),)
        report = compute_fold_stability_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS, benchmark_reports=benchmarks)
        assert report.benchmark_outperformance_fraction is None


class TestDayDirectionConfidenceBucketConcentrationHandComputed:
    def test_single_day_concentration_hand_computed(self) -> None:
        """Two trades on the SAME calendar day (0.04, 0.02) and one on a
        different day (0.01): day totals = {day1: 0.06, day2: 0.01}.
        Ratio = 0.06 / (0.06+0.01) = 0.06/0.07."""
        same_day_a = _trade(0, outer_fold_index=0, net_return=0.04)
        same_day_b = _trade(1, outer_fold_index=0, net_return=0.02)
        other_day = _trade(2, outer_fold_index=0, net_return=0.01, exit_timestamp="2024-01-05T05:00:00+00:00")
        report = compute_concentration_report((_fold(0, total_net_return=0.07),), all_closed_trades=(same_day_a, same_day_b, other_day), thresholds=_THRESHOLDS)
        assert report.single_day_profit_concentration == pytest.approx(0.06 / 0.07, abs=1e-12)

    def test_single_direction_concentration_hand_computed(self) -> None:
        """Long trades: 0.05, 0.03 (total 0.08). Short trades: 0.01.
        Ratio = 0.08 / (0.08 + 0.01) = 0.08/0.09."""
        trades = (
            _trade(0, outer_fold_index=0, net_return=0.05, direction=PositionDirection.LONG),
            _trade(1, outer_fold_index=0, net_return=0.03, direction=PositionDirection.LONG),
            _trade(2, outer_fold_index=0, net_return=0.01, direction=PositionDirection.SHORT),
        )
        report = compute_concentration_report((_fold(0, total_net_return=0.09),), all_closed_trades=trades, thresholds=_THRESHOLDS)
        assert report.single_direction_profit_concentration == pytest.approx(0.08 / 0.09, abs=1e-12)

    def test_single_confidence_bucket_concentration_hand_computed(self) -> None:
        """Confidence terciles: <1/3 low, <2/3 medium, else high (from
        `_tercile_bucket`). confidence=0.2 -> low; 0.6 -> medium; 0.9 ->
        high. Trades: (low, 0.05), (medium, 0.02), (medium, 0.01) ->
        bucket totals = {low: 0.05, medium: 0.03}. Ratio = 0.05/0.08."""
        trades = (
            _trade(0, outer_fold_index=0, net_return=0.05, confidence=0.2),
            _trade(1, outer_fold_index=0, net_return=0.02, confidence=0.6),
            _trade(2, outer_fold_index=0, net_return=0.01, confidence=0.65),
        )
        report = compute_concentration_report((_fold(0, total_net_return=0.08),), all_closed_trades=trades, thresholds=_THRESHOLDS)
        assert report.single_confidence_bucket_profit_concentration == pytest.approx(0.05 / 0.08, abs=1e-12)

    def test_winners_and_losers_cancel_leaves_only_positive_contributions(self) -> None:
        """Overall net could be zero or negative while SOME individual
        contributors are still positive -- concentration is computed over
        the positive subset regardless of the aggregate sign."""
        folds = (_fold(0, total_net_return=0.10), _fold(1, total_net_return=-0.10))  # net zero overall
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == 1.0  # only one positive contributor -> it IS the total

    def test_overall_net_loss_with_one_positive_fold(self) -> None:
        folds = (_fold(0, total_net_return=0.02), _fold(1, total_net_return=-0.20))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == 1.0

    def test_one_contributor_total_profit_positive(self) -> None:
        report = compute_concentration_report((_fold(0, total_net_return=0.05),), all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == 1.0

    def test_tied_contributors_split_evenly(self) -> None:
        folds = (_fold(0, total_net_return=0.05), _fold(1, total_net_return=0.05))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == pytest.approx(0.5, abs=1e-12)

    def test_negative_dominant_contributor_excluded_from_ratio(self) -> None:
        """The LARGEST-magnitude contributor is a loss -- it must be
        excluded entirely; concentration is computed only over the two
        positive contributors."""
        folds = (_fold(0, total_net_return=0.03), _fold(1, total_net_return=0.02), _fold(2, total_net_return=-100.0))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == pytest.approx(0.03 / 0.05, abs=1e-12)

    def test_total_profit_exactly_zero_gives_none(self) -> None:
        folds = (_fold(0, total_net_return=0.0), _fold(1, total_net_return=0.0))
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration is None

    def test_undefined_denominator_never_persists_a_misleading_percentage(self) -> None:
        """Every concentration dimension must independently report `None`
        (never a fabricated `0.0` or `1.0`) when its own denominator is
        undefined, even while OTHER dimensions on the same report are
        well-defined."""
        folds = (_fold(0, total_net_return=-0.05),)  # no positive fold contribution at all
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration is None
        assert report.single_trade_profit_concentration is None  # no trades at all either
        assert report.single_day_profit_concentration is None
        assert report.single_direction_profit_concentration is None
        assert report.single_confidence_bucket_profit_concentration is None


class TestConcentrationWarningThresholdEqualityBoundary:
    def test_concentration_exactly_at_threshold_does_not_warn(self) -> None:
        """Warning condition is `concentration > threshold` (strict), not
        `>=` -- a ratio landing EXACTLY on the threshold must not warn."""
        folds = (_fold(0, total_net_return=0.6), _fold(1, total_net_return=0.4))  # concentration exactly 0.6 == threshold
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert report.single_fold_profit_concentration == pytest.approx(0.6, abs=1e-12)
        assert "single_fold_profit_concentration_exceeded" not in report.warning_codes

    def test_concentration_one_epsilon_above_threshold_warns(self) -> None:
        folds = (_fold(0, total_net_return=0.601), _fold(1, total_net_return=0.399))  # concentration 0.601 > 0.6
        report = compute_concentration_report(folds, all_closed_trades=(), thresholds=_THRESHOLDS)
        assert "single_fold_profit_concentration_exceeded" in report.warning_codes
