"""Pydantic configuration schema for Milestone 9's portfolio risk engine
(Phase 1). Same conventions as `config.execution_gateway_schemas`: every
model is frozen, `extra="forbid"` (no unknown field can ever be silently
accepted -- this is also what makes "no credential field", "no endpoint
URL" true BY CONSTRUCTION: no such field is ever defined here, and none
can be smuggled in through `extra`). Decimal-valued limits are plain `str
| None` fields (exactly like `config.execution_gateway_schemas.
ReconciliationPolicyConfigSchema`'s tolerances) -- Pydantic does not
natively constrain decimal-string magnitude, so numeric validation
(positive, finite, in-range) is deferred to `PortfolioRiskPolicy.
__post_init__` via `.build()`, never duplicated here."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.portfolio_risk.models import PORTFOLIO_RISK_SPEC_SCHEMA_VERSION
from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec


class PortfolioRiskPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_order_notional: str | None = None
    max_position_notional: str | None = None
    max_instrument_gross_exposure: str | None = None
    max_strategy_gross_exposure: str | None = None
    max_portfolio_gross_exposure: str | None = None
    max_portfolio_net_exposure: str | None = None
    max_concentration_fraction: str | None = None
    max_leverage: str | None = None
    max_daily_realized_loss: str | None = None
    max_total_loss: str | None = None
    max_drawdown_fraction: str | None = None
    max_consecutive_losses: int | None = Field(default=None, ge=1)
    minimum_cash_buffer: str | None = None
    maximum_price_age: int | None = Field(default=None, ge=0)
    maximum_portfolio_snapshot_age: int | None = Field(default=None, ge=0)
    allow_reduce_only_during_halt: bool = True

    def build(self) -> PortfolioRiskPolicy:
        def _opt_decimal(value: str | None) -> Decimal | None:
            return None if value is None else Decimal(value)

        return PortfolioRiskPolicy(
            max_order_notional=_opt_decimal(self.max_order_notional), max_position_notional=_opt_decimal(self.max_position_notional),
            max_instrument_gross_exposure=_opt_decimal(self.max_instrument_gross_exposure),
            max_strategy_gross_exposure=_opt_decimal(self.max_strategy_gross_exposure),
            max_portfolio_gross_exposure=_opt_decimal(self.max_portfolio_gross_exposure),
            max_portfolio_net_exposure=_opt_decimal(self.max_portfolio_net_exposure),
            max_concentration_fraction=_opt_decimal(self.max_concentration_fraction), max_leverage=_opt_decimal(self.max_leverage),
            max_daily_realized_loss=_opt_decimal(self.max_daily_realized_loss), max_total_loss=_opt_decimal(self.max_total_loss),
            max_drawdown_fraction=_opt_decimal(self.max_drawdown_fraction), max_consecutive_losses=self.max_consecutive_losses,
            minimum_cash_buffer=_opt_decimal(self.minimum_cash_buffer), maximum_price_age=self.maximum_price_age,
            maximum_portfolio_snapshot_age=self.maximum_portfolio_snapshot_age, allow_reduce_only_during_halt=self.allow_reduce_only_during_halt,
        )


class PortfolioRiskConfigSchema(BaseModel):
    """Top-level operator-facing config. Deliberately NOT bound to a
    `portfolio_id` -- see `specs.PortfolioRiskSpec`'s own docstring: one
    policy is reusable across many portfolios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: PortfolioRiskPolicyConfigSchema = PortfolioRiskPolicyConfigSchema()

    def build(self) -> PortfolioRiskSpec:
        return PortfolioRiskSpec(schema_version=PORTFOLIO_RISK_SPEC_SCHEMA_VERSION, policy=self.policy.build())


__all__ = ["PortfolioRiskConfigSchema", "PortfolioRiskPolicyConfigSchema"]
