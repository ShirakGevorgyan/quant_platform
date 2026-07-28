"""Adversarial leakage tests (Milestone 5, Section 34) -- fail-LOUD
sentinels that prove temporal/informational isolation structurally, not
just by comparing a final metric before/after a change. Mirrors
`tests/unit/calibration/test_leakage_adversarial.py`'s exact philosophy,
adapted to this milestone's own leakage boundary: this package has no
"outer-test labels" to protect (calibration already isolated those) --
instead the boundary is FUTURE MARKET DATA relative to a signal's own
decision timestamp, and the requirement that realized trading OUTCOMES
never feed back into signal generation.

Every test here either (a) replaces bar values beyond a legitimate
decision/fold boundary with a landmine object that raises on any numeric
use and proves leakage-critical functions complete without ever touching
them, (b) constructs deliberately-leaky configuration and proves the
platform's own structural guards reject it, or (c) proves via function
signature that a leakage-critical function has no parameter through which
future data or realized outcomes could arrive."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtesting.execution import simulate_outer_fold_trades
from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    Decision,
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
from quant_platform.backtesting.reporting import build_backtest_report_json, render_backtest_report_markdown
from quant_platform.backtesting.returns import compute_trade_return_result
from quant_platform.backtesting.runner import (
    compute_benchmark_report,
    compute_bucket_analysis_report,
    compute_cost_sensitivity_report,
    resolve_market_bars_for_timeline,
    run_outer_fold_backtest,
    verify_and_load_predictions,
)
from quant_platform.backtesting.signals import Signal, SignalSet, generate_signals
from quant_platform.backtesting.specs import (
    BacktestSpec,
    CommissionSpec,
    CostSensitivityScenario,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.exceptions import BacktestValidationError, MarketDataBindingError
from quant_platform.core.types import Timeframe


class _Landmine:
    """A value that raises `AssertionError` on almost every operation a
    numeric bar column value could be subjected to. Placed at every bar
    row a function under test must never read."""

    def __repr__(self) -> str:
        return "<LANDMINE: a bar beyond the legitimate boundary was touched>"

    def __eq__(self, other: object) -> bool:
        raise AssertionError("LEAKAGE: an out-of-bounds bar value was compared")

    def __hash__(self) -> int:
        raise AssertionError("LEAKAGE: an out-of-bounds bar value was hashed")

    def __float__(self) -> float:
        raise AssertionError("LEAKAGE: an out-of-bounds bar value was converted to float")

    def __add__(self, other: object) -> object:
        raise AssertionError("LEAKAGE: an out-of-bounds bar value was used in arithmetic")

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__


def _real_bars(n: int, *, base_price: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    prices = base_price + np.arange(n, dtype="float64") * 0.1
    return pd.DataFrame({
        "open_time": timestamps, "open": prices, "high": prices + 0.5, "low": prices - 0.5, "close": prices + 0.05,
    })


def _landmine_bars_beyond(n: int, *, boundary: int, base_price: float = 100.0) -> pd.DataFrame:
    """`n`-row bar frame identical to `_real_bars` for rows `<= boundary`,
    and containing `_Landmine()` object values for every price column on
    every row `> boundary`."""
    df = _real_bars(n, base_price=base_price).astype({"open": object, "high": object, "low": object, "close": object})
    landmine = _Landmine()
    for col in ("open", "high", "low", "close"):
        df.loc[df.index[boundary + 1 :], col] = landmine
    return df


def _prediction_set(n: int = 10, *, threshold: float = 0.5, positive_at: frozenset[int] = frozenset()) -> VerifiedPredictionSet:
    timestamps = tuple(ts.isoformat() for ts in pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    calibrated = tuple(0.9 if i in positive_at else 0.1 for i in range(n))
    return VerifiedPredictionSet(
        schema_version=1, outer_fold_index=0, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
        base_model_definition_identity="m:1", sample_positions=tuple(range(n)), timestamps=timestamps,
        raw_probabilities=calibrated, calibrated_probabilities=calibrated, threshold=threshold,
        decisions=tuple(Decision.POSITIVE.value if i in positive_at else Decision.NEGATIVE.value for i in range(n)),
        abstention_reason_codes=("none",) * n, confidence_scores=(0.8,) * n, confidence_categories=("high",) * n,
        uncertainty_scores=(0.2,) * n,
    )


def _signal_mapping() -> SignalMappingSpec:
    return SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT)


class TestSignalGenerationHasNoMarketDataChannel:
    """`generate_signals` must have no parameter through which a bar
    price, a future timestamp, or a realized trading outcome could
    arrive -- this is checked structurally (by signature), not merely by
    convention, mirroring `test_select_calibrator_never_receives_true_
    labels_from_outside_pooled_oof`'s identical technique one milestone
    down."""

    def test_generate_signals_signature_has_no_bar_price_or_outcome_parameter(self) -> None:
        signature = inspect.signature(generate_signals)
        forbidden_substrings = ("bar", "price", "trade", "equity", "drawdown", "return")
        for name in signature.parameters:
            lowered = name.lower()
            for forbidden in forbidden_substrings:
                assert forbidden not in lowered, f"generate_signals has an unexpected {forbidden!r}-shaped parameter: {name!r}"

    def test_signals_are_a_pure_function_of_predictions_alone(self) -> None:
        """Calling `generate_signals` twice with byte-identical
        predictions (and unrelated other arguments held fixed) must
        produce byte-identical signals -- proving no hidden global or
        time-dependent state influences the mapping."""
        predictions = _prediction_set(positive_at=frozenset({2, 5, 8}))
        first = generate_signals(predictions, spec=_signal_mapping(), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        second = generate_signals(predictions, spec=_signal_mapping(), position_mode=PositionMode.LONG_FLAT, respect_calibration_abstention=True)
        assert first.to_json_dict() == second.to_json_dict()


class TestNoBarTCloseEntryByDefault:
    """Section 1/7's central rule, enforced STRUCTURALLY: `EntrySpec`
    refuses `delay_bars=0` unless `allow_same_bar_close` is explicitly,
    deliberately set."""

    def test_zero_delay_without_explicit_opt_in_is_rejected(self) -> None:
        with pytest.raises(BacktestValidationError, match="look-ahead"):
            EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=0)

    def test_zero_delay_with_explicit_opt_in_is_accepted(self) -> None:
        spec = EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=0, allow_same_bar_close=True)
        assert spec.delay_bars == 0


class TestEntryFillNeverReadsTheSignalBarsOwnLandmine:
    """A `NEXT_BAR_OPEN` entry must price off bar `signal.sample_position
    + delay_bars`'s OWN `open` -- never the signal's origin bar. Poisons
    the ORIGIN bar's `close` (a landmine) while leaving every other bar
    real; a passing test proves entry pricing never reads it."""

    def _spec(self) -> BacktestSpec:
        return BacktestSpec(
            schema_version=1, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
            dataset_content_id="c" * 64, split_plan_fingerprint="d" * 64, instrument_identity="XAUUSD", market_timezone="UTC",
            bar_interval=Timeframe.H1, decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
            signal_mapping=_signal_mapping(), position_mode=PositionMode.LONG_FLAT,
            entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
            exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=2, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
            overlap_policy=OverlapPolicyKind.IGNORE, price_basis=PriceBasisKind.CLOSE,
            spread_spec=SpreadSpec(kind=SpreadModelKind.ZERO),
            commission_spec=CommissionSpec(kind=CommissionModelKind.ZERO),
            slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO),
            financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE,
            compounding_policy=CompoundingPolicyKind.NON_COMPOUNDED,
            initial_notional=10_000.0, determinism_policy=DeterminismPolicy.STRICT,
        )

    def test_entry_fill_never_reads_the_origin_bars_close(self) -> None:
        bars = _real_bars(10).astype({"close": object})
        landmine = _Landmine()
        origin_bar_position = 3
        bars.loc[bars.index[origin_bar_position], "close"] = landmine  # only the ORIGIN bar's close is poisoned

        signal = Signal(
            sample_position=origin_bar_position, decision_timestamp=bars.iloc[origin_bar_position]["open_time"].isoformat(),
            direction=PositionDirection.LONG, strength=1.0, accepted=True, reason_code=SignalReasonCode.ACCEPTED_POSITIVE,
            confidence=0.8, uncertainty=0.2, threshold=0.5, calibrated_probability=0.9,
            source_calibration_id="a" * 64, source_experiment_id="b" * 64, outer_fold_index=0,
        )
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(signal,))
        # If entry pricing ever reads the origin bar's `close` (a
        # landmine), this raises AssertionError immediately.
        trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=self._spec(), fold_end_position=9)
        assert len(trade_set.trades) == 1
        assert trade_set.trades[0].entry_bar_position == origin_bar_position + 1


class TestNoBarBeyondFoldEndPositionIsEverRead:
    """Section 24: no position ever crosses a fold boundary. Poisons
    every bar STRICTLY AFTER `fold_end_position`; a passing test proves
    `simulate_outer_fold_trades` never indexes past it, regardless of
    exit policy or final-trade policy."""

    def _spec(self, *, final_trade_policy: FinalTradePolicyKind, exit_kind: ExitPolicyKind = ExitPolicyKind.FIXED_HORIZON) -> BacktestSpec:
        return BacktestSpec(
            schema_version=1, source_calibration_id="a" * 64, source_experiment_id="b" * 64, source_execution_id="b" * 64,
            dataset_content_id="c" * 64, split_plan_fingerprint="d" * 64, instrument_identity="XAUUSD", market_timezone="UTC",
            bar_interval=Timeframe.H1, decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
            signal_mapping=_signal_mapping(), position_mode=PositionMode.LONG_FLAT,
            entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
            exit_spec=ExitSpec(kind=exit_kind, holding_period_bars=(6 if exit_kind is ExitPolicyKind.FIXED_HORIZON else None), final_trade_policy=final_trade_policy),
            overlap_policy=OverlapPolicyKind.IGNORE, price_basis=PriceBasisKind.CLOSE,
            spread_spec=SpreadSpec(kind=SpreadModelKind.ZERO), commission_spec=CommissionSpec(kind=CommissionModelKind.ZERO),
            slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
            return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.NON_COMPOUNDED,
            initial_notional=10_000.0, determinism_policy=DeterminismPolicy.STRICT,
        )

    @pytest.mark.parametrize("final_trade_policy", [FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE, FinalTradePolicyKind.FORCE_CLOSE_AT_FINAL_PRICE, FinalTradePolicyKind.DISCARD_INCOMPLETE])
    def test_fixed_horizon_exit_beyond_fold_end_never_reads_poisoned_bars(self, final_trade_policy: FinalTradePolicyKind) -> None:
        fold_end_position = 12
        bars = _landmine_bars_beyond(20, boundary=fold_end_position)
        # entry at bar 6 (sample_position 5 + delay 1), fixed-horizon exit
        # at bar 12 -- exactly ON the boundary, never past it.
        signal = Signal(
            sample_position=5, decision_timestamp=pd.Timestamp("2024-01-01", tz="UTC").isoformat(), direction=PositionDirection.LONG,
            strength=1.0, accepted=True, reason_code=SignalReasonCode.ACCEPTED_POSITIVE, confidence=0.8, uncertainty=0.2,
            threshold=0.5, calibrated_probability=0.9, source_calibration_id="a" * 64, source_experiment_id="b" * 64, outer_fold_index=0,
        )
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(signal,))
        trade_set = simulate_outer_fold_trades(signals=signals, bars=bars, spec=self._spec(final_trade_policy=final_trade_policy), fold_end_position=fold_end_position)
        assert len(trade_set.trades) == 1
        assert trade_set.trades[0].exit_bar_position <= fold_end_position

    def test_end_of_fold_exit_never_reads_past_the_boundary(self) -> None:
        fold_end_position = 12
        bars = _landmine_bars_beyond(20, boundary=fold_end_position)
        signal = Signal(
            sample_position=5, decision_timestamp=pd.Timestamp("2024-01-01", tz="UTC").isoformat(), direction=PositionDirection.LONG,
            strength=1.0, accepted=True, reason_code=SignalReasonCode.ACCEPTED_POSITIVE, confidence=0.8, uncertainty=0.2,
            threshold=0.5, calibrated_probability=0.9, source_calibration_id="a" * 64, source_experiment_id="b" * 64, outer_fold_index=0,
        )
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(signal,))
        trade_set = simulate_outer_fold_trades(
            signals=signals, bars=bars, spec=self._spec(final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE, exit_kind=ExitPolicyKind.END_OF_FOLD),
            fold_end_position=fold_end_position,
        )
        assert len(trade_set.trades) == 1
        assert trade_set.trades[0].exit_bar_position == fold_end_position

    def test_entry_beyond_fold_end_is_silently_excluded_never_fabricated(self) -> None:
        """A signal whose entry bar would land beyond `fold_end_position`
        must never open a position at all (MISSING_MARKET_BAR) -- proven
        by placing the landmine at the WOULD-BE entry bar itself."""
        fold_end_position = 10
        bars = _landmine_bars_beyond(20, boundary=fold_end_position)
        signal = Signal(
            sample_position=10, decision_timestamp=pd.Timestamp("2024-01-01", tz="UTC").isoformat(), direction=PositionDirection.LONG,
            strength=1.0, accepted=True, reason_code=SignalReasonCode.ACCEPTED_POSITIVE, confidence=0.8, uncertainty=0.2,
            threshold=0.5, calibrated_probability=0.9, source_calibration_id="a" * 64, source_experiment_id="b" * 64, outer_fold_index=0,
        )
        signals = SignalSet(schema_version=1, outer_fold_index=0, signals=(signal,))
        trade_set = simulate_outer_fold_trades(
            signals=signals, bars=bars, spec=self._spec(final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE), fold_end_position=fold_end_position,
        )
        assert len(trade_set.trades) == 0


class TestMarketDataNeverForwardFilled:
    """Section 6: a timeline timestamp with no matching raw market bar
    must fail closed (`MarketDataBindingError`), never silently
    forward-filled or interpolated."""

    class _FakeLoaderMissingOneRow:
        def __init__(self, bars: pd.DataFrame) -> None:
            self._bars = bars

        def load(self, request: object) -> pd.DataFrame:
            return self._bars

    def test_a_timeline_timestamp_with_no_matching_raw_bar_fails_closed(self) -> None:
        timeline = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")})
        raw_bars = _real_bars(5)
        raw_bars = raw_bars.drop(index=2).reset_index(drop=True)  # remove the 3rd bar -- a genuine gap

        loader = self._FakeLoaderMissingOneRow(raw_bars)
        with pytest.raises(MarketDataBindingError, match="no matching raw market bar"):
            resolve_market_bars_for_timeline(loader, symbol="XAUUSD", base_timeframe="H1", timeline=timeline, timestamp_column="open_time")


class TestCostSensitivityNeverAltersGrossReturn:
    """Section 20: cost-sensitivity scenario multipliers must scale ONLY
    cost components -- `gross_return` (the raw, cost-free price outcome)
    must be bit-for-bit identical across every scenario, proving the
    "signal" (price movement) can never be contaminated by a cost
    assumption."""

    def test_gross_return_is_identical_across_all_cost_scenarios(self) -> None:
        scenarios = (
            CostSensitivityScenario(name="zero_cost", spread_multiplier=0.0, slippage_multiplier=0.0, commission_multiplier=0.0),
            CostSensitivityScenario(name="base_cost"),
            CostSensitivityScenario(name="2x_spread", spread_multiplier=2.0),
        )
        results = [
            compute_trade_return_result(
                direction=PositionDirection.LONG, entry_observed_price=100.0, exit_observed_price=102.0,
                entry_spread_adjustment=0.05, exit_spread_adjustment=-0.05, entry_slippage_adjustment=0.02, exit_slippage_adjustment=-0.02,
                commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=5.0),
                financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
                holding_days=1.0, notional=10_000.0,
                return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE,
                cost_scenario=scenario,
            )
            for scenario in scenarios
        ]
        gross_returns = {r.gross_return for r in results}
        assert len(gross_returns) == 1, f"gross_return must be scenario-invariant, got {gross_returns}"
        net_returns = {r.net_return for r in results}
        assert len(net_returns) == 3, "net_return must actually differ across distinctly-multiplied scenarios"


class TestNoRealizedOutcomeFeedsBackIntoSignalGeneration:
    """Signals are generated ONCE, from predictions alone, strictly
    BEFORE any trade/equity/drawdown/benchmark/bucket-analysis object can
    exist -- proven structurally (signature-based, mirroring `test_
    reporting_functions_only_accept_already_evaluated_results_no_fold_
    or_timeline`'s identical technique) for every downstream analysis
    function that consumes ALREADY-simulated results."""

    def test_benchmark_computation_has_no_realized_outcome_parameter(self) -> None:
        """Milestone 5.1, Section 4: `compute_benchmark_report` now
        legitimately accepts `predictions` -- the SAME already-verified,
        pre-outcome `VerifiedPredictionSet` the primary strategy's own
        `signals` are built from (Section 4's raw-uncalibrated-threshold
        and calibrated-signal benchmark rows need it to construct their
        OWN independent signal variants). This is not a realized-outcome
        leak: predictions exist strictly BEFORE any trade/equity/return
        object, exactly like `signals` (already an accepted parameter of
        `compute_cost_sensitivity_report`, see the test below). What must
        still never happen: accepting an already-REALIZED trade/equity/
        return/metrics object that could let a benchmark be tuned after
        seeing outcomes."""
        signature = inspect.signature(compute_benchmark_report)
        params = set(signature.parameters)
        assert "trades" not in params and "equity_curve" not in params and "drawdown" not in params and "financial_metrics" not in params

    def test_bucket_analysis_only_accepts_already_closed_trades(self) -> None:
        signature = inspect.signature(compute_bucket_analysis_report)
        params = set(signature.parameters)
        assert "trades" in params
        assert "signals" not in params and "predictions" not in params

    def test_cost_sensitivity_recomputation_has_no_realized_outcome_parameter(self) -> None:
        """`compute_cost_sensitivity_report` re-simulates from `signals`
        (pre-outcome intent) -- it must not additionally accept an
        already-realized `trades`/`equity_curve` shortcut that could let
        a cost scenario be selected AFTER seeing results."""
        signature = inspect.signature(compute_cost_sensitivity_report)
        params = set(signature.parameters)
        assert "trades" not in params and "equity_curve" not in params

    def test_report_builders_take_only_already_evaluated_fold_results(self) -> None:
        for fn in (build_backtest_report_json, render_backtest_report_markdown):
            signature = inspect.signature(fn)
            for name in signature.parameters:
                lowered = name.lower()
                assert "signal" not in lowered and "bar" not in lowered, f"{fn!r} has an unexpected parameter: {name!r}"


class TestPredictionVerificationRejectsUnreproducibleCalibratedProbabilities:
    """Section 5: `verify_and_load_predictions` must have no route to
    trust a calibration's OWN unverified claim -- proven here at the
    signature level (no `trust`/`skip_verification` escape hatch exists)."""

    def test_verify_and_load_predictions_has_no_trust_or_skip_parameter(self) -> None:
        signature = inspect.signature(verify_and_load_predictions)
        for name in signature.parameters:
            lowered = name.lower()
            assert "trust" not in lowered and "skip" not in lowered, f"verify_and_load_predictions has an unexpected escape-hatch parameter: {name!r}"


class TestRunOuterFoldBacktestSignatureHasNoCrossFoldChannel:
    """`run_outer_fold_backtest` takes exactly ONE `outer_fold: Fold`
    (the fold currently being evaluated) -- no parameter through which a
    different, already-evaluated fold's results (or a "next fold" lookahead)
    could arrive."""

    def test_no_other_fold_or_lookahead_parameter_exists(self) -> None:
        signature = inspect.signature(run_outer_fold_backtest)
        names = {n.lower() for n in signature.parameters}
        assert names == {"spec", "backtest_id", "outer_fold", "timeline", "bars", "calibration_manifest", "calibration_spec", "artifact_store"}
