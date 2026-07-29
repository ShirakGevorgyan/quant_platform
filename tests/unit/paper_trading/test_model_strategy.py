"""`paper_trading.model_strategy.ModelStrategyRuntime`. Uses a lightweight
FAKE `ProbabilisticPredictor`/`FittedModel` (structurally satisfies both
`@runtime_checkable` Protocols via plain attributes/methods -- exactly
`isinstance(..., ProbabilisticPredictor)` needs, no real sklearn model
required) to exercise the ADAPTER's own logic: bar buffering, feature
warm-up/NaN handling, threshold-based direction selection, and
non-bar-event abstention. The genuine end-to-end path (a REAL trained
model deserialized from `MLArtifactStore`) is exercised once, for real,
by the Section 33 acceptance workflow -- avoiding duplicating that
expensive fixture here, exactly the precedent `test_eligibility.py`
already set for the eligibility chain's own downstream steps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import StrategyRuntimeError
from quant_platform.core.types import Timeframe
from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.paper_trading.events import create_bar_event, create_session_open_event
from quant_platform.paper_trading.model_strategy import ModelStrategyRuntime
from quant_platform.paper_trading.models import KillSwitchState, PaperSessionStage
from quant_platform.paper_trading.strategy import PortfolioSnapshot, RiskState, SessionState, StrategyContext

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_A = "a" * 64


@dataclass
class _FakeFittedModel:
    """Structurally satisfies `ml.interfaces.FittedModel` AND
    `ProbabilisticPredictor` (both `@runtime_checkable`) via plain
    attributes/methods -- `isinstance` checks against those Protocols
    only look for member presence, not a declared base class."""

    feature_names: tuple[str, ...]
    fixed_p_positive: float

    @property
    def metadata(self) -> object:
        return SimpleNamespace(feature_schema=FeatureSchema(feature_names=self.feature_names))

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def class_labels(self) -> tuple[object, ...]:
        return (0.0, 1.0)

    def predict(self, features: pd.DataFrame, *, column_policy: object = None) -> np.ndarray:
        return np.array([1.0 if self.fixed_p_positive >= 0.5 else 0.0] * len(features))

    def predict_proba(self, features: pd.DataFrame, *, column_policy: object = None) -> np.ndarray:
        return np.array([[1.0 - self.fixed_p_positive, self.fixed_p_positive]] * len(features))


def _bars(closes: list[float]) -> list:
    events = []
    for i, close in enumerate(closes):
        open_time = _T0 + timedelta(hours=i)
        events.append(create_bar_event(instrument="X", interval=Timeframe.H1, open_time=open_time, open=close, high=close + 0.5, low=close - 0.5, close=close, sequence=i + 1, source="test"))
    return events


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(instrument="X", signed_quantity=0.0, average_entry_price=None, cash=100_000.0, equity=100_000.0, unrealized_pnl=0.0, realized_pnl=0.0)


def _context_for(event: object, decision_time: datetime) -> StrategyContext:
    return StrategyContext(
        event=event, feature_snapshot={}, feature_snapshot_identity=None, model_output=0.0, model_output_identity=None,
        calibrated_probability=None, confidence=0.5, uncertainty=0.0, portfolio=_snapshot(),
        risk=RiskState(trading_halted=False, kill_switch_state=KillSwitchState.ACTIVE), session=SessionState(paper_session_id=_HEX_A, stage=PaperSessionStage.RUNNING),
        decision_time=decision_time,
    )


class TestConstructionValidation:
    def test_model_not_probabilistic_rejected(self) -> None:
        class _NonProbabilistic:
            @property
            def metadata(self) -> object:
                return SimpleNamespace(feature_schema=FeatureSchema(feature_names=("candle_body_ratio",)))

            def predict(self, features: pd.DataFrame, *, column_policy: object = None) -> np.ndarray:
                return np.array([0.0])

        with pytest.raises(StrategyRuntimeError, match="predict_proba"):
            ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=_NonProbabilistic(), feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)  # type: ignore[arg-type]

    def test_unsupported_feature_name_rejected(self) -> None:
        model = _FakeFittedModel(feature_names=("not_a_real_feature",), fixed_p_positive=0.5)
        with pytest.raises(StrategyRuntimeError, match="unsupported feature"):
            ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("not_a_real_feature",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)

    def test_feature_names_must_cover_model_schema(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio", "ma_distance_10"), fixed_p_positive=0.5)
        with pytest.raises(StrategyRuntimeError, match="does not cover"):
            ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)

    @pytest.mark.parametrize("long_threshold", [0.5, 0.4, 1.1])
    def test_invalid_long_threshold_rejected(self, long_threshold: float) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.5)
        with pytest.raises(StrategyRuntimeError, match="long_threshold"):
            ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=long_threshold, short_threshold=0.4, target_quantity=1.0)


class TestDecide:
    def test_non_bar_event_abstains(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.9)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        event = create_session_open_event(instrument="X", event_time=_T0, sequence=1, source="test")
        decision = runtime.decide(_context_for(event, _T0))
        assert decision.abstain
        assert decision.reason_codes == ("non_bar_event",)

    def test_insufficient_history_abstains(self) -> None:
        model = _FakeFittedModel(feature_names=("ma_distance_20",), fixed_p_positive=0.9)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("ma_distance_20",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        bars = _bars([100.0, 101.0, 102.0])
        decision = None
        for bar in bars:
            decision = runtime.decide(_context_for(bar, bar.close_time))
        assert decision is not None
        assert decision.abstain
        assert decision.reason_codes == ("insufficient_history_or_warmup",)

    def test_high_p_positive_produces_long_decision_once_warmed_up(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.9)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=2.5)
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        assert not decision.abstain
        assert decision.target_direction is PositionDirection.LONG
        assert decision.target_quantity == 2.5
        assert decision.confidence == pytest.approx(0.8)
        assert decision.feature_snapshot_identity is not None
        assert decision.model_output_identity is not None

    def test_low_p_positive_produces_short_decision(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.1)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        assert not decision.abstain
        assert decision.target_direction is PositionDirection.SHORT

    def test_neutral_p_positive_abstains(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.5)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        assert decision.abstain
        assert decision.reason_codes == ("model_signal_neutral",)

    def test_decision_identity_is_deterministic_given_same_history(self) -> None:
        model_a = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.9)
        model_b = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.9)
        runtime_a = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model_a, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        runtime_b = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model_b, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        bar = _bars([100.0])[0]
        decision_a = runtime_a.decide(_context_for(bar, bar.close_time))
        decision_b = runtime_b.decide(_context_for(bar, bar.close_time))
        assert decision_a.decision_id == decision_b.decision_id
        assert decision_a.feature_snapshot_identity == decision_b.feature_snapshot_identity


class TestConfidenceScaledSizing:
    def test_disabled_by_default_uses_fixed_quantity(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.99)
        runtime = ModelStrategyRuntime(strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4, target_quantity=1.0)
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        assert decision.target_quantity == 1.0

    def test_enabled_scales_quantity_with_confidence(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.99)
        runtime = ModelStrategyRuntime(
            strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4,
            target_quantity=1.0, confidence_scaled_sizing=True,
        )
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        expected_confidence = min(1.0, abs(0.99 - 0.5) * 2.0)
        assert decision.target_quantity == pytest.approx(1.0 * (1.0 + expected_confidence))
        assert decision.target_quantity > 1.0

    def test_enabled_low_confidence_stays_near_base_quantity(self) -> None:
        model = _FakeFittedModel(feature_names=("candle_body_ratio",), fixed_p_positive=0.61)
        runtime = ModelStrategyRuntime(
            strategy_identity=_HEX_A, fitted_model=model, feature_names=("candle_body_ratio",), long_threshold=0.6, short_threshold=0.4,
            target_quantity=1.0, confidence_scaled_sizing=True,
        )
        bar = _bars([100.0])[0]
        decision = runtime.decide(_context_for(bar, bar.close_time))
        assert 1.0 < decision.target_quantity < 1.3
