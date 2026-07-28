"""Milestone 6, Section 16: parameter/decision sensitivity -- structural
applicability of each perturbation axis, domain-boundary skips, and
hand-computed monotonicity-violation counting / cliff detection."""

from __future__ import annotations

from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
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
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.types import Timeframe
from quant_platform.robustness.models import PerturbationAxisKind
from quant_platform.robustness.sensitivity import (
    _PERTURBERS,
    PerturbationPointResult,
    _compute_axis_aggregates,
    _count_monotonicity_violations,
    axis_structural_skip_reason,
)


def _spec(**overrides: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_calibration_id": "a" * 64, "source_experiment_id": "b" * 64, "source_execution_id": "b" * 64,
        "dataset_content_id": "c" * 64, "split_plan_fingerprint": "d" * 64, "instrument_identity": "XAUUSD", "market_timezone": "UTC",
        "bar_interval": Timeframe.H1, "decision_timestamp_policy": DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
        "signal_mapping": SignalMappingSpec(kind=SignalMappingPolicyKind.PROBABILITY_BANDS, probability_band_long_min=0.6, probability_band_short_max=0.4),
        "position_mode": PositionMode.LONG_SHORT, "entry_spec": EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
        "exit_spec": ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=4, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
        "overlap_policy": OverlapPolicyKind.IGNORE, "price_basis": PriceBasisKind.CLOSE,
        "spread_spec": SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=2.0),
        "commission_spec": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=1.0),
        "slippage_spec": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=0.5),
        "financing_spec": FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=0.1),
        "return_calculation_policy": ReturnCalculationPolicyKind.SIMPLE, "compounding_policy": CompoundingPolicyKind.NON_COMPOUNDED,
        "initial_notional": 10_000.0, "determinism_policy": DeterminismPolicy.STRICT, "exposure_cap": 1.0,
    }
    defaults.update(overrides)
    return BacktestSpec(**defaults)  # type: ignore[arg-type]


class TestAxisStructuralApplicability:
    def test_abstention_threshold_is_always_skipped(self) -> None:
        assert axis_structural_skip_reason(PerturbationAxisKind.ABSTENTION_THRESHOLD, _spec()) is not None

    def test_probability_threshold_applies_only_under_probability_bands(self) -> None:
        assert axis_structural_skip_reason(PerturbationAxisKind.PROBABILITY_THRESHOLD, _spec()) is None
        flat_spec = _spec(signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT)
        assert axis_structural_skip_reason(PerturbationAxisKind.PROBABILITY_THRESHOLD, flat_spec) is not None

    def test_confidence_and_uncertainty_thresholds_apply_only_under_combined_kind(self) -> None:
        confidence_spec = _spec(signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY, confidence_floor=0.5, uncertainty_ceiling=0.3))
        assert axis_structural_skip_reason(PerturbationAxisKind.CONFIDENCE_THRESHOLD, confidence_spec) is None
        assert axis_structural_skip_reason(PerturbationAxisKind.UNCERTAINTY_THRESHOLD, confidence_spec) is None
        assert axis_structural_skip_reason(PerturbationAxisKind.CONFIDENCE_THRESHOLD, _spec()) is not None  # _spec() uses probability_bands

    def test_entry_delay_and_exposure_cap_always_applicable(self) -> None:
        assert axis_structural_skip_reason(PerturbationAxisKind.ENTRY_DELAY_BARS, _spec()) is None
        assert axis_structural_skip_reason(PerturbationAxisKind.EXPOSURE_CAP, _spec()) is None


class TestPerturberArithmetic:
    def test_probability_threshold_scales_both_band_edges_by_the_same_factor(self) -> None:
        modified_spec, applied_value, reason = _PERTURBERS[PerturbationAxisKind.PROBABILITY_THRESHOLD](_spec(), 0.1)
        assert reason is None and modified_spec is not None
        assert modified_spec.signal_mapping.probability_band_long_min == 0.6 * 1.1
        assert modified_spec.signal_mapping.probability_band_short_max == 0.4 * 1.1
        assert applied_value == 0.6 * 1.1

    def test_probability_threshold_out_of_domain_delta_is_skipped_not_clamped(self) -> None:
        """A same-sign multiplicative delta can never invert the two band
        edges (they scale by the identical factor), but CAN push
        long_min out of [0, 1] -- must be an explicit skip, never a
        silently clamped value."""
        modified_spec, applied_value, reason = _PERTURBERS[PerturbationAxisKind.PROBABILITY_THRESHOLD](_spec(), 1.0)  # long_min -> 1.2
        assert modified_spec is None and applied_value is None and reason is not None

    def test_holding_period_below_one_is_skipped(self) -> None:
        modified_spec, _applied_value, reason = _PERTURBERS[PerturbationAxisKind.HOLDING_PERIOD_BARS](_spec(), -0.99)  # 4 * 0.01 = 0.04 -> rounds to 0
        assert modified_spec is None and reason is not None and "< 1" in reason

    def test_exposure_cap_non_positive_is_skipped(self) -> None:
        modified_spec, _applied_value, reason = _PERTURBERS[PerturbationAxisKind.EXPOSURE_CAP](_spec(), -1.5)
        assert modified_spec is None and reason is not None


class TestMonotonicityViolationCounting:
    def test_strictly_increasing_sequence_has_zero_violations(self) -> None:
        assert _count_monotonicity_violations([(-0.1, 1.0), (0.0, 2.0), (0.1, 3.0)]) == 0

    def test_single_direction_reversal_counts_as_one_violation(self) -> None:
        assert _count_monotonicity_violations([(-0.1, 1.0), (0.0, 3.0), (0.1, 2.0)]) == 1

    def test_two_direction_reversals_count_as_two_violations(self) -> None:
        assert _count_monotonicity_violations([(-0.2, 1.0), (-0.1, 3.0), (0.0, 2.0), (0.1, 4.0)]) == 2

    def test_flat_steps_are_not_counted_as_a_direction(self) -> None:
        assert _count_monotonicity_violations([(-0.1, 1.0), (0.0, 1.0), (0.1, 2.0)]) == 0


class TestCliffDetectionAndAggregateStats:
    def test_cliff_detected_when_nearest_neighbor_flips_profitability(self) -> None:
        points = (
            PerturbationPointResult(relative_delta=-0.1, applied_value=0.9, status="evaluated", skip_reason=None, total_net_return=-0.02, total_gross_return=-0.01, closed_trade_count=5, maximum_drawdown=0.05, is_profitable=False),
            PerturbationPointResult(relative_delta=0.1, applied_value=1.1, status="evaluated", skip_reason=None, total_net_return=0.03, total_gross_return=0.04, closed_trade_count=5, maximum_drawdown=0.04, is_profitable=True),
        )
        _mv, cliff, _rank_stable, fraction, _score = _compute_axis_aggregates(baseline_net_return=0.02, baseline_profitable=True, evaluated=list(points))
        assert cliff is True
        assert fraction == 0.5

    def test_parameter_sensitivity_score_matches_hand_computed_formula(self) -> None:
        """score = max(|evaluated - baseline|) / max(|baseline|, 1e-9).
        baseline=0.02; evaluated={-0.02, 0.03} -> swings={0.04, 0.01} ->
        max=0.04 -> score = 0.04 / 0.02 = 2.0 exactly."""
        points = (
            PerturbationPointResult(relative_delta=-0.1, applied_value=0.9, status="evaluated", skip_reason=None, total_net_return=-0.02, total_gross_return=-0.01, closed_trade_count=5, maximum_drawdown=0.05, is_profitable=False),
            PerturbationPointResult(relative_delta=0.1, applied_value=1.1, status="evaluated", skip_reason=None, total_net_return=0.03, total_gross_return=0.04, closed_trade_count=5, maximum_drawdown=0.04, is_profitable=True),
        )
        _mv, _cliff, _rank_stable, _fraction, score = _compute_axis_aggregates(baseline_net_return=0.02, baseline_profitable=True, evaluated=list(points))
        assert score == 2.0

    def test_empty_evaluated_points_gives_all_none(self) -> None:
        result = _compute_axis_aggregates(baseline_net_return=0.02, baseline_profitable=True, evaluated=[])
        assert result == (None, None, None, None, None)
