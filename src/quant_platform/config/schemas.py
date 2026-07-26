"""Pydantic configuration schemas.

Each schema is both a validated data holder AND a factory (`.build()`) for
the corresponding runtime object -- so a config file (or any external
input) can fully describe a backtest run, with every constraint enforced
at parse time rather than discovered mid-run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.core.exceptions import DataSourceError
from quant_platform.core.types import Timeframe
from quant_platform.costs.models import CostModel, FixedSpreadCostModel, VolatilityScaledSlippageModel
from quant_platform.risk.position_sizing import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    PositionSizer,
    VolatilityTargetSizer,
)

LimitPolicy = Literal["clamp", "raise"]


class CostModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: Literal["fixed_spread", "volatility_scaled"] = "fixed_spread"
    spread_points: float = Field(ge=0.0)
    point_value: float = Field(gt=0.0)
    commission_per_unit: float = Field(default=0.0, ge=0.0)
    slippage_points: float = Field(default=0.0, ge=0.0)
    reference_volatility: float | None = Field(default=None, gt=0.0)
    min_slippage_multiplier: float = Field(default=0.25, gt=0.0)
    max_slippage_multiplier: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def _check_volatility_scaled_requirements(self) -> CostModelConfig:
        if self.model_type == "volatility_scaled" and self.reference_volatility is None:
            raise ValueError("reference_volatility is required when model_type='volatility_scaled'")
        return self

    def build(self) -> CostModel:
        if self.model_type == "fixed_spread":
            return FixedSpreadCostModel(
                spread_points=self.spread_points,
                slippage_points=self.slippage_points,
                point_value=self.point_value,
                commission_per_unit=self.commission_per_unit,
            )
        assert self.reference_volatility is not None  # enforced by the model_validator above
        return VolatilityScaledSlippageModel(
            spread_points=self.spread_points,
            base_slippage_points=self.slippage_points,
            reference_volatility=self.reference_volatility,
            point_value=self.point_value,
            commission_per_unit=self.commission_per_unit,
            min_slippage_multiplier=self.min_slippage_multiplier,
            max_slippage_multiplier=self.max_slippage_multiplier,
        )


class RiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sizer_type: Literal["fixed_fractional", "volatility_target", "kelly"] = "fixed_fractional"
    max_position_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    on_limit_exceeded: LimitPolicy = "clamp"

    risk_percent: float | None = Field(default=None, gt=0.0, le=100.0)
    target_volatility_percent: float | None = Field(default=None, gt=0.0, le=100.0)
    win_rate: float | None = Field(default=None, gt=0.0, lt=1.0)
    win_loss_ratio: float | None = Field(default=None, gt=0.0)
    kelly_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_required_fields_for_sizer_type(self) -> RiskConfig:
        if self.sizer_type == "fixed_fractional" and self.risk_percent is None:
            raise ValueError("risk_percent is required when sizer_type='fixed_fractional'")
        if self.sizer_type == "volatility_target" and self.target_volatility_percent is None:
            raise ValueError("target_volatility_percent is required when sizer_type='volatility_target'")
        if self.sizer_type == "kelly" and (self.win_rate is None or self.win_loss_ratio is None):
            raise ValueError("win_rate and win_loss_ratio are both required when sizer_type='kelly'")
        return self

    def build(self) -> PositionSizer:
        if self.sizer_type == "fixed_fractional":
            assert self.risk_percent is not None
            return FixedFractionalSizer(
                self.risk_percent, self.max_position_fraction, self.on_limit_exceeded
            )
        if self.sizer_type == "volatility_target":
            assert self.target_volatility_percent is not None
            return VolatilityTargetSizer(
                self.target_volatility_percent, self.max_position_fraction, self.on_limit_exceeded
            )
        assert self.win_rate is not None and self.win_loss_ratio is not None
        return KellyCriterionSizer(
            self.win_rate,
            self.win_loss_ratio,
            self.kelly_fraction,
            self.max_position_fraction,
            self.on_limit_exceeded,
        )


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    base_timeframe: Timeframe
    start: datetime
    end: datetime
    initial_capital: float = Field(gt=0.0)
    point_value: float = Field(default=1.0, gt=0.0)
    max_window_bars: int = Field(default=500, gt=0)
    cost: CostModelConfig
    risk: RiskConfig

    @field_validator("symbol")
    @classmethod
    def _symbol_must_be_a_safe_identifier(cls, value: str) -> str:
        # Delegates to the same check DataSource implementations use, so a
        # config that would fail to load data fails at parse time instead.
        from quant_platform.data.interfaces import DataSource

        try:
            return DataSource.sanitize_identifier(value, field_name="symbol")
        except DataSourceError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _check_date_range(self) -> BacktestConfig:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be strictly after start ({self.start})")
        return self
