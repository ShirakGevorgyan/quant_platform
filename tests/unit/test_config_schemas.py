"""Tests for Pydantic configuration schemas: validation rules and the
`.build()` factory methods that turn config into runtime objects."""

from __future__ import annotations

from datetime import datetime, timezone

import pydantic
import pytest

from quant_platform.config.schemas import BacktestConfig, CostModelConfig, RiskConfig
from quant_platform.core.types import Timeframe
from quant_platform.costs.models import FixedSpreadCostModel, VolatilityScaledSlippageModel
from quant_platform.risk.position_sizing import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    VolatilityTargetSizer,
)

UTC = timezone.utc


class TestCostModelConfig:
    def test_builds_fixed_spread_model(self) -> None:
        config = CostModelConfig(spread_points=20.0, slippage_points=10.0, point_value=0.01, commission_per_unit=1.0)
        model = config.build()
        assert isinstance(model, FixedSpreadCostModel)

    def test_volatility_scaled_requires_reference_volatility(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="reference_volatility"):
            CostModelConfig(model_type="volatility_scaled", spread_points=20.0, point_value=0.01)

    def test_builds_volatility_scaled_model_when_configured(self) -> None:
        config = CostModelConfig(
            model_type="volatility_scaled", spread_points=20.0, slippage_points=10.0,
            point_value=0.01, reference_volatility=1.5,
        )
        model = config.build()
        assert isinstance(model, VolatilityScaledSlippageModel)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"spread_points": -1.0, "point_value": 0.01},
            {"spread_points": 20.0, "point_value": 0.0},
            {"spread_points": 20.0, "point_value": 0.01, "commission_per_unit": -1.0},
            {"spread_points": 20.0, "point_value": 0.01, "slippage_points": -1.0},
        ],
    )
    def test_rejects_invalid_fields(self, kwargs: dict) -> None:
        with pytest.raises(pydantic.ValidationError):
            CostModelConfig(**kwargs)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CostModelConfig(spread_points=20.0, point_value=0.01, bogus_field=123)

    def test_is_frozen(self) -> None:
        config = CostModelConfig(spread_points=20.0, point_value=0.01)
        with pytest.raises(pydantic.ValidationError):
            config.spread_points = 999.0


class TestRiskConfig:
    def test_builds_fixed_fractional_sizer(self) -> None:
        config = RiskConfig(sizer_type="fixed_fractional", risk_percent=1.0)
        assert isinstance(config.build(), FixedFractionalSizer)

    def test_fixed_fractional_requires_risk_percent(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="risk_percent"):
            RiskConfig(sizer_type="fixed_fractional")

    def test_builds_volatility_target_sizer(self) -> None:
        config = RiskConfig(sizer_type="volatility_target", target_volatility_percent=2.0)
        assert isinstance(config.build(), VolatilityTargetSizer)

    def test_volatility_target_requires_its_field(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="target_volatility_percent"):
            RiskConfig(sizer_type="volatility_target")

    def test_builds_kelly_sizer(self) -> None:
        config = RiskConfig(sizer_type="kelly", win_rate=0.6, win_loss_ratio=2.0)
        assert isinstance(config.build(), KellyCriterionSizer)

    def test_kelly_requires_both_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="win_rate and win_loss_ratio"):
            RiskConfig(sizer_type="kelly", win_rate=0.6)


class TestBacktestConfig:
    def _valid_kwargs(self, **overrides: object) -> dict:
        base: dict[str, object] = {
            "symbol": "XAUUSD",
            "base_timeframe": Timeframe.M15,
            "start": datetime(2024, 1, 1, tzinfo=UTC),
            "end": datetime(2024, 6, 1, tzinfo=UTC),
            "initial_capital": 10_000.0,
            "cost": CostModelConfig(spread_points=20.0, point_value=0.01),
            "risk": RiskConfig(sizer_type="fixed_fractional", risk_percent=1.0),
        }
        base.update(overrides)
        return base

    def test_valid_config_constructs(self) -> None:
        config = BacktestConfig(**self._valid_kwargs())
        assert config.symbol == "XAUUSD"
        assert config.base_timeframe is Timeframe.M15

    def test_accepts_timeframe_as_string(self) -> None:
        config = BacktestConfig(**self._valid_kwargs(base_timeframe="H1"))
        assert config.base_timeframe is Timeframe.H1

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be strictly after"):
            BacktestConfig(**self._valid_kwargs(start=datetime(2024, 6, 1, tzinfo=UTC), end=datetime(2024, 1, 1, tzinfo=UTC)))

    def test_end_equal_to_start_raises(self) -> None:
        same = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(pydantic.ValidationError):
            BacktestConfig(**self._valid_kwargs(start=same, end=same))

    def test_rejects_path_traversal_symbol(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="Invalid symbol"):
            BacktestConfig(**self._valid_kwargs(symbol="../etc/passwd"))

    def test_rejects_non_positive_initial_capital(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            BacktestConfig(**self._valid_kwargs(initial_capital=0.0))

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            BacktestConfig(**self._valid_kwargs(bogus_field="oops"))

    def test_nested_configs_are_usable_as_factories(self) -> None:
        config = BacktestConfig(**self._valid_kwargs())
        assert isinstance(config.cost.build(), FixedSpreadCostModel)
        assert isinstance(config.risk.build(), FixedFractionalSizer)
