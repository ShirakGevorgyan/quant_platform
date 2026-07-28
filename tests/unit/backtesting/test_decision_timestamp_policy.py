"""Milestone 5.1, Section 5: `decision_timestamp_policy` must not be
decorative. Covers: `prediction_availability_timestamp`'s own formula and
its internal-consistency guard; `BacktestSpec` rejecting the two
unimplementable policies (BEFORE_NEXT_BAR_OPEN/EXTERNALLY_TIMESTAMPED) at
construction, fail-closed; `BacktestSpec` rejecting `entry_spec.
delay_bars=0` under AFTER_BAR_CLOSE (an impossible fill, regardless of
`allow_same_bar_close`); `TradeRecord` rejecting an entry timestamped
before its own decision timestamp, at construction/decode time."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.backtesting.costs import CostBreakdown
from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    ExitReasonCode,
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
    TradeStatus,
    prediction_availability_timestamp,
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
from quant_platform.backtesting.trades import TradeRecord, compute_trade_id
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.exceptions import (
    BacktestValidationError,
    MarketDataBindingError,
    TradeConstructionError,
)
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
        "spread_spec": SpreadSpec(kind=SpreadModelKind.ZERO), "commission_spec": CommissionSpec(kind=CommissionModelKind.ZERO),
        "slippage_spec": SlippageSpec(kind=SlippageModelKind.ZERO), "financing_spec": FinancingSpec(kind=FinancingModelKind.NONE),
        "return_calculation_policy": ReturnCalculationPolicyKind.SIMPLE, "compounding_policy": CompoundingPolicyKind.NON_COMPOUNDED,
        "initial_notional": 10_000.0, "determinism_policy": DeterminismPolicy.STRICT,
    }
    defaults.update(overrides)
    return BacktestSpec(**defaults)  # type: ignore[arg-type]


class TestPredictionAvailabilityTimestamp:
    def test_after_bar_close_is_bar_open_plus_interval(self) -> None:
        bar_open = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        result = prediction_availability_timestamp(bar_open, bar_interval=Timeframe.H1, policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE)
        assert result == bar_open + Timeframe.H1.duration

    def test_rejects_unsupported_policy_as_internal_consistency_violation(self) -> None:
        bar_open = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        with pytest.raises(MarketDataBindingError, match="not a supported"):
            prediction_availability_timestamp(bar_open, bar_interval=Timeframe.H1, policy=DecisionTimestampPolicyKind.BEFORE_NEXT_BAR_OPEN)


class TestBacktestSpecRejectsUnsupportedDecisionTimestampPolicies:
    @pytest.mark.parametrize("policy", [DecisionTimestampPolicyKind.BEFORE_NEXT_BAR_OPEN, DecisionTimestampPolicyKind.EXTERNALLY_TIMESTAMPED])
    def test_unsupported_policy_rejected_at_construction(self, policy: DecisionTimestampPolicyKind) -> None:
        with pytest.raises(BacktestValidationError, match="not implementable"):
            _spec(decision_timestamp_policy=policy)

    def test_after_bar_close_is_accepted(self) -> None:
        spec = _spec(decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE)
        assert spec.decision_timestamp_policy is DecisionTimestampPolicyKind.AFTER_BAR_CLOSE


class TestBacktestSpecRejectsImpossibleFillsUnderAfterBarClose:
    def test_delay_bars_zero_is_rejected_even_without_allow_same_bar_close(self) -> None:
        with pytest.raises(BacktestValidationError, match=r"IMPOSSIBLE|delay_bars"):
            _spec(entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=0, allow_same_bar_close=True))

    def test_delay_bars_one_is_accepted(self) -> None:
        spec = _spec(entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1))
        assert spec.entry_spec.delay_bars == 1


def _trade(*, decision_timestamp: str, entry_timestamp: str, exit_timestamp: str = "2024-01-01T05:00:00+00:00") -> TradeRecord:
    cb = CostBreakdown(entry_spread_cost=0.0, exit_spread_cost=0.0, entry_commission=0.0, exit_commission=0.0, entry_slippage=0.0, exit_slippage=0.0, financing_cost=0.0)
    tid = compute_trade_id(source_calibration_id="a" * 64, outer_fold_index=0, signal_sample_position=0, direction=PositionDirection.LONG, entry_timestamp=entry_timestamp, exit_timestamp=exit_timestamp)
    return TradeRecord(
        schema_version=1, trade_id=tid, signal_sample_position=0, outer_fold_index=0, direction=PositionDirection.LONG,
        signal_timestamp=decision_timestamp, decision_timestamp=decision_timestamp, entry_timestamp=entry_timestamp, entry_bar_position=1,
        entry_observed_price=100.0, entry_effective_price=100.0, exit_timestamp=exit_timestamp, exit_bar_position=5,
        exit_observed_price=101.0, exit_effective_price=101.0, holding_bars=4, gross_return=0.01, net_return=0.01,
        cost_breakdown=cb, confidence=0.5, uncertainty=0.5, calibrated_probability=0.7,
        entry_reason=SignalReasonCode.ACCEPTED_POSITIVE, exit_reason=ExitReasonCode.FIXED_HORIZON_REACHED, status=TradeStatus.CLOSED,
        source_calibration_id="a" * 64, source_experiment_id="b" * 64,
    )


class TestTradeRecordRejectsImpossibleFills:
    def test_entry_at_or_after_decision_timestamp_is_accepted(self) -> None:
        trade = _trade(decision_timestamp="2024-01-01T01:00:00+00:00", entry_timestamp="2024-01-01T01:00:00+00:00")
        assert trade.entry_timestamp == "2024-01-01T01:00:00+00:00"
        trade_later = _trade(decision_timestamp="2024-01-01T01:00:00+00:00", entry_timestamp="2024-01-01T02:00:00+00:00")
        assert trade_later.entry_timestamp == "2024-01-01T02:00:00+00:00"

    def test_entry_before_decision_timestamp_is_rejected(self) -> None:
        """The semantic-verification tampering case: an entry_timestamp
        one hour BEFORE its own decision_timestamp -- an impossible fill,
        must be rejected at construction (and therefore at decode, since
        `TradeRecord.from_json_dict` always re-runs `__post_init__`)."""
        with pytest.raises(TradeConstructionError, match="IMPOSSIBLE fill"):
            _trade(decision_timestamp="2024-01-01T01:00:00+00:00", entry_timestamp="2024-01-01T00:00:00+00:00")

    def test_tampered_persisted_trade_record_is_rejected_at_decode(self) -> None:
        """Mirrors the tampering-test convention used throughout this
        milestone: a byte-valid, schema-valid JSON payload whose
        entry_timestamp was edited to precede decision_timestamp must
        fail exactly at `TradeRecord.from_json_dict`, never silently load."""
        trade = _trade(decision_timestamp="2024-01-01T01:00:00+00:00", entry_timestamp="2024-01-01T01:00:00+00:00")
        tampered_raw = {**trade.to_json_dict(), "entry_timestamp": "2024-01-01T00:00:00+00:00"}
        with pytest.raises(TradeConstructionError, match="IMPOSSIBLE fill"):
            TradeRecord.from_json_dict(tampered_raw)
