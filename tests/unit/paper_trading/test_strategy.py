"""Milestone 7, Section 7: `StrategyDecision`/`StrategyContext` and the
narrow strategy-visible snapshots. Covers construction/validation,
deterministic content-addressed `decision_id`, JSON round-trip, and a
unit-level leakage-sensitivity proof: `decision_id` (and therefore the
decision's own identity) changes whenever `event_identity`/`feature_
snapshot_identity`/`model_output_identity` changes -- the property
`verification.py`'s full leakage test (Section 26, built later against a
real pipeline) ultimately relies on."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import StrategyRuntimeError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.models import KillSwitchState, PaperSessionStage
from quant_platform.paper_trading.strategy import (
    PortfolioSnapshot,
    RiskState,
    SessionState,
    StopTargetIntent,
    StrategyContext,
    StrategyDecision,
    create_strategy_decision,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_STRATEGY = "a" * 64
_HEX_FEATURE = "b" * 64
_HEX_MODEL = "c" * 64


def _event(sequence: int = 1):
    return create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=105.0, low=99.0, close=102.0, sequence=sequence, source="s")


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(instrument="X", signed_quantity=0.0, average_entry_price=None, cash=100_000.0, equity=100_000.0, unrealized_pnl=0.0, realized_pnl=0.0)


def _risk() -> RiskState:
    return RiskState(trading_halted=False, kill_switch_state=KillSwitchState.ACTIVE)


def _session() -> SessionState:
    return SessionState(paper_session_id="a" * 64, stage=PaperSessionStage.RUNNING)


class TestPortfolioSnapshotValidation:
    def test_valid_flat_snapshot(self) -> None:
        snapshot = _portfolio()
        assert snapshot.signed_quantity == 0.0

    def test_empty_instrument_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="instrument"):
            PortfolioSnapshot(instrument="", signed_quantity=0.0, average_entry_price=None, cash=0.0, equity=0.0, unrealized_pnl=0.0, realized_pnl=0.0)

    def test_flat_with_average_entry_price_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="average_entry_price"):
            PortfolioSnapshot(instrument="X", signed_quantity=0.0, average_entry_price=100.0, cash=0.0, equity=0.0, unrealized_pnl=0.0, realized_pnl=0.0)

    def test_non_finite_cash_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="cash"):
            PortfolioSnapshot(instrument="X", signed_quantity=0.0, average_entry_price=None, cash=float("nan"), equity=0.0, unrealized_pnl=0.0, realized_pnl=0.0)


class TestStrategyContextValidation:
    def _context(self, **overrides: object) -> StrategyContext:
        defaults: dict[str, object] = {
            "event": _event(), "feature_snapshot": {"rsi": 55.0}, "feature_snapshot_identity": _HEX_FEATURE, "model_output": 0.7,
            "model_output_identity": _HEX_MODEL, "calibrated_probability": 0.65, "confidence": 0.8, "uncertainty": 0.1,
            "portfolio": _portfolio(), "risk": _risk(), "session": _session(), "decision_time": _T0,
        }
        defaults.update(overrides)
        return StrategyContext(**defaults)  # type: ignore[arg-type]

    def test_valid_context_constructs(self) -> None:
        assert self._context().confidence == 0.8

    def test_naive_decision_time_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="timezone-aware"):
            self._context(decision_time=datetime(2026, 1, 5, 10, 0, 0))

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="confidence"):
            self._context(confidence=1.5)

    def test_negative_uncertainty_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="uncertainty"):
            self._context(uncertainty=-0.1)

    def test_invalid_feature_snapshot_identity_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="feature_snapshot_identity"):
            self._context(feature_snapshot_identity="not-a-hash")

    def test_none_feature_snapshot_identity_allowed(self) -> None:
        context = self._context(feature_snapshot_identity=None)
        assert context.feature_snapshot_identity is None


class TestStopTargetIntent:
    def test_valid_intent(self) -> None:
        intent = StopTargetIntent(stop_loss_price=95.0, take_profit_price=110.0)
        assert intent.stop_loss_price == 95.0

    def test_non_positive_stop_loss_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="stop_loss_price"):
            StopTargetIntent(stop_loss_price=0.0, take_profit_price=None)

    def test_json_round_trip(self) -> None:
        intent = StopTargetIntent(stop_loss_price=95.0, take_profit_price=None)
        assert StopTargetIntent.from_json_dict(intent.to_json_dict()) == intent


class TestStrategyDecisionConstructionAndValidation:
    def test_valid_long_decision(self) -> None:
        decision = create_strategy_decision(
            strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.LONG, target_quantity=1.0,
            confidence=0.8, uncertainty=0.1, abstain=False, reason_codes=("model_signal_long",), model_output_identity=_HEX_MODEL,
            feature_snapshot_identity=_HEX_FEATURE,
        )
        assert decision.target_direction is PositionDirection.LONG
        assert decision.abstain is False

    def test_flat_direction_with_nonzero_quantity_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="target_quantity"):
            create_strategy_decision(
                strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=1.0,
                confidence=0.5, uncertainty=0.0, abstain=True, reason_codes=("no_signal",),
            )

    def test_empty_reason_codes_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="reason_codes"):
            create_strategy_decision(
                strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=0.0,
                confidence=0.5, uncertainty=0.0, abstain=True, reason_codes=(),
            )

    def test_invalid_strategy_identity_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="strategy_identity"):
            create_strategy_decision(
                strategy_identity="not-a-hash", event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=0.0,
                confidence=0.5, uncertainty=0.0, abstain=True, reason_codes=("no_signal",),
            )

    def test_non_json_safe_diagnostics_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="diagnostics"):
            create_strategy_decision(
                strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=0.0,
                confidence=0.5, uncertainty=0.0, abstain=True, reason_codes=("no_signal",), diagnostics={"bad": object()},  # type: ignore[dict-item]
            )

    def test_non_finite_diagnostic_float_rejected(self) -> None:
        with pytest.raises(StrategyRuntimeError, match="diagnostics"):
            create_strategy_decision(
                strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=0.0,
                confidence=0.5, uncertainty=0.0, abstain=True, reason_codes=("no_signal",), diagnostics={"bad": float("nan")},
            )

    def test_abstain_decision_persists_with_full_diagnostic_fields(self) -> None:
        """Section 7: "Persist every decision including abstentions" --
        an abstention is a full, valid `StrategyDecision`, not a stub."""
        decision = create_strategy_decision(
            strategy_identity=_HEX_STRATEGY, event=_event(), decision_time=_T0, target_direction=PositionDirection.FLAT, target_quantity=0.0,
            confidence=0.3, uncertainty=0.4, abstain=True, reason_codes=("low_confidence", "high_uncertainty"), model_output_identity=_HEX_MODEL,
            feature_snapshot_identity=_HEX_FEATURE, diagnostics={"raw_score": 0.12},
        )
        assert decision.abstain is True
        assert decision.reason_codes == ("low_confidence", "high_uncertainty")


class TestStrategyDecisionIdentityDeterminismAndLeakageSensitivity:
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "strategy_identity": _HEX_STRATEGY, "event": _event(sequence=1), "decision_time": _T0, "target_direction": PositionDirection.LONG,
            "target_quantity": 1.0, "confidence": 0.8, "uncertainty": 0.1, "abstain": False, "reason_codes": ("model_signal_long",),
            "model_output_identity": _HEX_MODEL, "feature_snapshot_identity": _HEX_FEATURE,
        }

    def test_identical_arguments_produce_identical_decision_id(self) -> None:
        a = create_strategy_decision(**self._base_kwargs())  # type: ignore[arg-type]
        b = create_strategy_decision(**self._base_kwargs())  # type: ignore[arg-type]
        assert a.decision_id == b.decision_id

    def test_different_event_identity_changes_decision_id(self) -> None:
        """The core leakage-sensitivity property: a decision computed
        against a DIFFERENT market event (e.g. one carrying future
        information the correctly-scoped version never saw) always gets a
        different, non-reproducible identity."""
        kwargs = self._base_kwargs()
        a = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        kwargs["event"] = _event(sequence=2)
        b = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        assert a.decision_id != b.decision_id
        assert a.event_identity != b.event_identity

    def test_different_feature_snapshot_identity_changes_decision_id(self) -> None:
        kwargs = self._base_kwargs()
        a = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        kwargs["feature_snapshot_identity"] = "d" * 64
        b = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        assert a.decision_id != b.decision_id

    def test_different_model_output_identity_changes_decision_id(self) -> None:
        kwargs = self._base_kwargs()
        a = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        kwargs["model_output_identity"] = "d" * 64
        b = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        assert a.decision_id != b.decision_id

    def test_different_confidence_changes_decision_id(self) -> None:
        kwargs = self._base_kwargs()
        a = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        kwargs["confidence"] = 0.5
        b = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        assert a.decision_id != b.decision_id

    def test_different_abstain_flag_changes_decision_id(self) -> None:
        kwargs = self._base_kwargs()
        a = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        kwargs["abstain"] = True
        kwargs["target_direction"] = PositionDirection.FLAT
        kwargs["target_quantity"] = 0.0
        b = create_strategy_decision(**kwargs)  # type: ignore[arg-type]
        assert a.decision_id != b.decision_id

    def test_json_round_trip_preserves_identity(self) -> None:
        decision = create_strategy_decision(**self._base_kwargs())  # type: ignore[arg-type]
        roundtripped = StrategyDecision.from_json_dict(decision.to_json_dict())
        assert roundtripped == decision
