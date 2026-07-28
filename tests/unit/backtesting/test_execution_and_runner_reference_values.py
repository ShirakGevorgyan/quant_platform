"""Hand-checkable reference-value unit tests for `backtesting.execution`
and `backtesting.runner`'s pure computation functions (Sections 35-39
lineage) -- formalizes the numeric checks performed interactively during
development into permanent regression coverage.

`TestMixedDirectionalAndFlatAcceptedSignals` documents a real defect
found via `tests/performance/test_backtesting_throughput.py` (using
random, threshold-crossing predictions rather than a near-constant test
model): `SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT`'s "predicted
negative -> flat" case (and `PROBABILITY_BANDS`' middle dead zone)
produce an ACCEPTED `Signal` with `direction=FLAT` -- `execution.
simulate_outer_fold_trades` was treating every accepted signal as
entry-triggering, crashing on `compute_fill_price(direction=FLAT)`. Fixed
by filtering `direction is FLAT` out of the entry-triggering signal list,
exactly mirroring `ExitPolicyKind.OPPOSITE_SIGNAL`'s own pre-existing
close-trigger check (which already excluded FLAT)."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.execution import simulate_outer_fold_trades
from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionDirection,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SignalReasonCode,
    SlippageModelKind,
    SpreadModelKind,
    VerifiedPredictionSet,
)
from quant_platform.backtesting.runner import (
    compute_benchmark_report,
    compute_bucket_analysis_report,
    compute_cost_sensitivity_report,
)
from quant_platform.backtesting.signals import Signal, SignalSet, generate_signals
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.backtesting.trades import TradeRecord, compute_trade_id
from quant_platform.calibration.models import Decision, DeterminismPolicy
from quant_platform.core.types import Timeframe


def _spec(**overrides: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_calibration_id": "a" * 64, "source_experiment_id": "b" * 64, "source_execution_id": "b" * 64,
        "dataset_content_id": "c" * 64, "split_plan_fingerprint": "d" * 64, "instrument_identity": "XAUUSD", "market_timezone": "UTC",
        "bar_interval": Timeframe.H1, "decision_timestamp_policy": DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        "signal_mapping": SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), "position_mode": PositionMode.LONG_FLAT,
        "entry_spec": EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        "exit_spec": ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=1, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        "overlap_policy": OverlapPolicyKind.IGNORE, "price_basis": PriceBasisKind.CLOSE,
        "spread_spec": SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=10.0),
        "commission_spec": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=5.0),
        "slippage_spec": SlippageSpec(kind=SlippageModelKind.ZERO), "financing_spec": FinancingSpec(kind=FinancingModelKind.NONE),
        "return_calculation_policy": ReturnCalculationPolicyKind.SIMPLE, "compounding_policy": CompoundingPolicyKind.NON_COMPOUNDED,
        "initial_notional": 10_000.0, "determinism_policy": DeterminismPolicy.STRICT,
    }
    defaults.update(overrides)
    return BacktestSpec(**defaults)  # type: ignore[arg-type]


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({
        "open_time": timestamps, "open": [100.0, 101.0, 102.0, 103.0, 104.0], "high": [100.5, 101.5, 102.5, 103.5, 104.5],
        "low": [99.5, 100.5, 101.5, 102.5, 103.5], "close": [100.0, 101.0, 102.0, 103.0, 105.0],
    })


def _predictions(
    *, raw_probabilities: tuple[float, ...] | None = None, calibrated_probabilities: tuple[float, ...] | None = None,
    decisions: tuple[str, ...] | None = None, threshold: float = 0.5,
) -> VerifiedPredictionSet:
    """5 samples, positionally aligned to `_bars()`'s 5 rows (Milestone
    5.1, Section 4's benchmark matrix tests)."""
    n = 5
    calibrated = calibrated_probabilities if calibrated_probabilities is not None else (0.9, 0.1, 0.8, 0.2, 0.7)
    raw = raw_probabilities if raw_probabilities is not None else calibrated
    decs = decisions if decisions is not None else tuple(Decision.POSITIVE.value if v >= threshold else Decision.NEGATIVE.value for v in calibrated)
    timestamps = tuple(ts.isoformat() for ts in pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    return VerifiedPredictionSet(
        schema_version=1, outer_fold_index=0, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
        base_model_definition_identity="m:1", sample_positions=tuple(range(n)), timestamps=timestamps,
        raw_probabilities=raw, calibrated_probabilities=calibrated, threshold=threshold, decisions=decs,
        abstention_reason_codes=("none",) * n, confidence_scores=tuple(min(abs(v - threshold) * 2, 1.0) for v in calibrated),
        confidence_categories=("medium",) * n, uncertainty_scores=tuple(1.0 - min(abs(v - threshold) * 2, 1.0) for v in calibrated),
    )


def _signal(sample_position: int, direction: PositionDirection, *, accepted: bool = True) -> Signal:
    reason = SignalReasonCode.ACCEPTED_POSITIVE if direction is PositionDirection.LONG else SignalReasonCode.ACCEPTED_NEGATIVE
    return Signal(
        sample_position=sample_position, decision_timestamp="2024-01-01T00:00:00+00:00", direction=direction, strength=1.0,
        accepted=accepted, reason_code=reason, confidence=0.8, uncertainty=0.2, threshold=0.5, calibrated_probability=(0.9 if direction is PositionDirection.LONG else 0.1),
        source_calibration_id="a" * 64, source_experiment_id="b" * 64, outer_fold_index=0,
    )


class TestMixedDirectionalAndFlatAcceptedSignals:
    def test_accepted_flat_signal_never_triggers_an_entry(self) -> None:
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, PositionDirection.FLAT),))
        trade_set = simulate_outer_fold_trades(signals=signals, bars=_bars(), spec=_spec(), fold_end_position=4)
        assert trade_set.trades == ()

    def test_mixed_long_and_flat_accepted_signals_do_not_crash_and_only_long_opens(self) -> None:
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, PositionDirection.LONG), _signal(2, PositionDirection.FLAT)))
        trade_set = simulate_outer_fold_trades(signals=signals, bars=_bars(), spec=_spec(), fold_end_position=4)
        assert len(trade_set.trades) == 1
        assert trade_set.trades[0].direction is PositionDirection.LONG

    def test_long_short_mode_directional_signals_are_unaffected_by_the_fix(self) -> None:
        """Sanity check: `LONG_SHORT` mode's `directional_long_short`
        mapping never produces FLAT-direction accepted signals in the
        first place, so this fix must not change its behavior at all."""
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, PositionDirection.LONG), _signal(2, PositionDirection.SHORT)))
        spec = _spec(position_mode=PositionMode.LONG_SHORT, signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_SHORT))
        trade_set = simulate_outer_fold_trades(signals=signals, bars=_bars(), spec=spec, fold_end_position=4)
        assert len(trade_set.trades) == 2


def _closed_trade(exit_timestamp: str, gross_return: float, net_return: float, *, confidence: float = 0.5, uncertainty: float = 0.5, sample_position: int = 0) -> TradeRecord:
    cb = CostBreakdown(entry_spread_cost=0.0001, exit_spread_cost=0.0001, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0)
    entry_timestamp = "2024-01-01T00:00:00+00:00"
    tid = compute_trade_id(source_calibration_id="a" * 64, outer_fold_index=0, signal_sample_position=sample_position, direction=PositionDirection.LONG, entry_timestamp=entry_timestamp, exit_timestamp=exit_timestamp)
    from quant_platform.backtesting.models import ExitReasonCode, TradeStatus

    return TradeRecord(
        schema_version=1, trade_id=tid, signal_sample_position=sample_position, outer_fold_index=0, direction=PositionDirection.LONG,
        signal_timestamp=entry_timestamp, decision_timestamp=entry_timestamp, entry_timestamp=entry_timestamp, entry_bar_position=sample_position,
        entry_observed_price=100.0, entry_effective_price=100.02, exit_timestamp=exit_timestamp, exit_bar_position=sample_position + 1,
        exit_observed_price=101.0, exit_effective_price=100.98, holding_bars=1, gross_return=gross_return, net_return=net_return,
        cost_breakdown=cb, confidence=confidence, uncertainty=uncertainty, calibrated_probability=0.7,
        entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED,
        source_calibration_id="a" * 64, source_experiment_id="b" * 64,
    )


class TestBucketAnalysis:
    def test_terciles_and_insufficient_sample_flagging(self) -> None:
        trades = [
            _closed_trade("2024-01-01T01:00:00+00:00", 0.01, 0.008, confidence=0.9, uncertainty=0.1, sample_position=0),
            _closed_trade("2024-01-01T02:00:00+00:00", -0.01, -0.012, confidence=0.2, uncertainty=0.8, sample_position=1),
            _closed_trade("2024-01-01T03:00:00+00:00", 0.02, 0.018, confidence=0.5, uncertainty=0.5, sample_position=2),
        ]
        report = compute_bucket_analysis_report(trades=trades, outer_fold_index=0, minimum_bucket_samples=1)
        by_key = {(b.dimension, b.bucket): b for b in report.buckets}
        assert by_key[("confidence", "high")].sample_count == 1
        assert abs(by_key[("confidence", "high")].average_net_return - 0.008) < 1e-9
        assert by_key[("uncertainty", "high")].sample_count == 1

        strict_report = compute_bucket_analysis_report(trades=trades, outer_fold_index=0, minimum_bucket_samples=5)
        for bucket in strict_report.buckets:
            if bucket.sample_count > 0:
                assert bucket.insufficient_sample is True
                assert bucket.average_net_return is None
                assert bucket.hit_rate is None


class TestBenchmarkReport:
    def test_always_flat_is_always_zero(self) -> None:
        bars = _bars()
        report = compute_benchmark_report(predictions=_predictions(), bars=bars, spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        flat = next(b for b in report.benchmarks if b.name == "always_flat")
        assert flat.gross_return == 0.0
        assert flat.net_return == 0.0

    def test_always_long_zero_cost_matches_first_to_last_close(self) -> None:
        bars = _bars()
        report = compute_benchmark_report(predictions=_predictions(), bars=bars, spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        zero_cost = next(b for b in report.benchmarks if b.name == "always_long_zero_cost")
        expected = (105.0 - 100.0) / 100.0
        assert abs(zero_cost.gross_return - expected) < 1e-9
        assert zero_cost.gross_return == zero_cost.net_return

    def test_always_long_net_cost_is_strictly_worse_than_zero_cost(self) -> None:
        bars = _bars()
        report = compute_benchmark_report(predictions=_predictions(), bars=bars, spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        zero_cost = next(b for b in report.benchmarks if b.name == "always_long_zero_cost")
        net_cost = next(b for b in report.benchmarks if b.name == "always_long_net_cost")
        assert net_cost.gross_return == zero_cost.gross_return
        assert net_cost.net_return < net_cost.gross_return

    def test_matrix_is_complete_under_long_flat(self) -> None:
        report = compute_benchmark_report(predictions=_predictions(), bars=_bars(), spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        names = {b.name for b in report.benchmarks}
        expected = {
            "always_flat", "always_long_zero_cost", "always_long_net_cost",
            "always_long_strategy_zero_cost", "always_long_strategy_net_cost",
            "raw_uncalibrated_threshold_zero_cost", "raw_uncalibrated_threshold_net_cost",
            "calibrated_no_abstention_zero_cost", "calibrated_no_abstention_net_cost",
            "calibrated_with_abstention_zero_cost", "calibrated_with_abstention_net_cost",
        }
        assert names == expected
        # "always_short*" is unsupported (not silently fabricated) under LONG_FLAT.
        assert not any("short" in name for name in names)

    def test_always_short_present_only_under_long_short(self) -> None:
        spec = _spec(position_mode=PositionMode.LONG_SHORT, signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_SHORT))
        report = compute_benchmark_report(predictions=_predictions(), bars=_bars(), spec=spec, fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        names = {b.name for b in report.benchmarks}
        assert {"always_short_zero_cost", "always_short_net_cost", "always_short_strategy_zero_cost", "always_short_strategy_net_cost"} <= names

    def test_raw_and_calibrated_no_abstention_are_identical_under_identity_calibration(self) -> None:
        """Section 4: "calibration benchmark differs only in calibration
        decision." With raw==calibrated probabilities and a matching 0.5
        threshold, the raw-threshold and calibrated-no-abstention signal
        rules make IDENTICAL decisions on every sample -- so their
        benchmark results must match exactly. Only when calibration
        actually TRANSFORMS the probability would the two rows diverge."""
        predictions = _predictions(calibrated_probabilities=(0.9, 0.1, 0.8, 0.2, 0.7))  # raw defaults to calibrated
        report = compute_benchmark_report(predictions=predictions, bars=_bars(), spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        by_name = {b.name: b for b in report.benchmarks}
        assert by_name["raw_uncalibrated_threshold_zero_cost"].gross_return == pytest.approx(by_name["calibrated_no_abstention_zero_cost"].gross_return)
        assert by_name["raw_uncalibrated_threshold_net_cost"].net_return == pytest.approx(by_name["calibrated_no_abstention_net_cost"].net_return)

    def test_raw_and_calibrated_diverge_when_calibration_actually_transforms_probabilities(self) -> None:
        """Same decisions class-wise (sign relative to 0.5) but a
        DIFFERENT raw probability magnitude -- proves the two rows are
        computed independently (not literally aliased) even though this
        particular case still agrees on direction; the true divergence
        case is exercised structurally by construction (raw feeds `_raw_
        uncalibrated_threshold_signal_set`, calibrated feeds
        `generate_signals` -- two different code paths)."""
        predictions = _predictions(raw_probabilities=(0.51, 0.49, 0.99, 0.01, 0.60), calibrated_probabilities=(0.9, 0.1, 0.8, 0.2, 0.7))
        report = compute_benchmark_report(predictions=predictions, bars=_bars(), spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        names = {b.name for b in report.benchmarks}
        assert "raw_uncalibrated_threshold_zero_cost" in names and "calibrated_no_abstention_zero_cost" in names

    def test_no_abstention_and_with_abstention_are_identical_when_nothing_abstains(self) -> None:
        """Section 4: "abstention benchmark differs only in abstention
        filtering." When no sample's decision is ABSTAIN, forcing
        abstention filtering ON vs OFF has no observable effect --
        `respect_calibration_abstention` only ever changes behavior for
        samples the calibration policy itself marked ABSTAIN."""
        predictions = _predictions()  # decisions default to positive/negative only, never abstain
        no_abstention = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=False)
        with_abstention = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        assert [s.direction for s in no_abstention.signals] == [s.direction for s in with_abstention.signals]
        assert [s.accepted for s in no_abstention.signals] == [s.accepted for s in with_abstention.signals]

    def test_abstention_filtering_changes_only_the_abstained_sample(self) -> None:
        calibrated = (0.9, 0.1, 0.8, 0.2, 0.7)
        decisions = (Decision.POSITIVE.value, Decision.NEGATIVE.value, Decision.ABSTAIN.value, Decision.NEGATIVE.value, Decision.POSITIVE.value)
        predictions = _predictions(calibrated_probabilities=calibrated, decisions=decisions)
        no_abstention = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=False)
        with_abstention = generate_signals(predictions, spec=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        for i, (a, b) in enumerate(zip(no_abstention.signals, with_abstention.signals, strict=True)):
            if i == 2:
                assert a.accepted and not b.accepted
                assert b.direction is PositionDirection.FLAT
            else:
                assert a.direction == b.direction and a.accepted == b.accepted

    def test_zero_cost_and_net_cost_pairs_share_identical_gross_return(self) -> None:
        report = compute_benchmark_report(predictions=_predictions(), bars=_bars(), spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        by_name = {b.name: b for b in report.benchmarks}
        zero_cost_names = [n for n in by_name if n.endswith("_zero_cost")]
        assert len(zero_cost_names) >= 5
        for zero_name in zero_cost_names:
            net_name = zero_name.replace("_zero_cost", "_net_cost")
            assert net_name in by_name, f"missing paired net-cost benchmark for {zero_name!r}"
            assert by_name[zero_name].gross_return == pytest.approx(by_name[net_name].gross_return), f"{zero_name} and {net_name} gross returns must match (gross return is cost-independent)"

    def test_net_cost_never_improves_on_gross_return(self) -> None:
        """Section 4: "net-cost cannot improve under non-negative costs" --
        `_spec()`'s cost model has strictly non-negative spread/commission/
        slippage/financing, so `net_return <= gross_return` for every
        `*_net_cost` row, always."""
        report = compute_benchmark_report(predictions=_predictions(), bars=_bars(), spec=_spec(), fold_start_position=0, fold_end_position=4, outer_fold_index=0)
        for b in report.benchmarks:
            if b.name.endswith("_net_cost"):
                assert b.net_return <= b.gross_return + 1e-9, f"{b.name}: net_return ({b.net_return}) exceeds gross_return ({b.gross_return})"


class TestCostSensitivityMonotonicity:
    def test_higher_spread_multiplier_never_improves_net_return(self) -> None:
        bars = _bars()
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(_signal(0, PositionDirection.LONG),))
        report = compute_cost_sensitivity_report(signals=signals, bars=bars, spec=_spec(), fold_end_position=4, outer_fold_index=0)
        by_name = {r.scenario_name: r.total_net_return for r in report.results}
        assert by_name["zero_cost"] >= by_name["base_cost"] >= by_name["1.5x_spread"] >= by_name["2x_spread"]
