"""Unit tests for `config.portfolio_risk_schemas`: strict `extra="forbid"`
rejection and `.build()` round trip into the frozen domain dataclasses."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_platform.config.portfolio_risk_schemas import (
    PortfolioRiskConfigSchema,
    PortfolioRiskPolicyConfigSchema,
)


class TestPortfolioRiskPolicyConfigSchemaStrictness:
    def test_default_constructs(self) -> None:
        schema = PortfolioRiskPolicyConfigSchema()
        assert schema.allow_reduce_only_during_halt is True

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioRiskPolicyConfigSchema(not_a_real_field=1)  # type: ignore[call-arg]

    def test_frozen_after_construction(self) -> None:
        schema = PortfolioRiskPolicyConfigSchema()
        with pytest.raises(ValidationError):
            schema.max_leverage = "5"  # type: ignore[misc]

    def test_negative_max_consecutive_losses_rejected_by_schema(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioRiskPolicyConfigSchema(max_consecutive_losses=0)

    def test_negative_maximum_price_age_rejected_by_schema(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioRiskPolicyConfigSchema(maximum_price_age=-1)

    def test_no_broker_credential_field_can_be_smuggled_in(self) -> None:
        for forbidden in ("broker_api_key", "mt5_login", "password", "endpoint_url", "account_number"):
            with pytest.raises(ValidationError):
                PortfolioRiskPolicyConfigSchema(**{forbidden: "x"})  # type: ignore[arg-type]


class TestPortfolioRiskPolicyConfigSchemaBuild:
    def test_build_produces_matching_domain_policy(self) -> None:
        schema = PortfolioRiskPolicyConfigSchema(
            max_order_notional="100000", max_leverage="3", max_concentration_fraction="0.25", max_consecutive_losses=5,
            minimum_cash_buffer="1000", maximum_price_age=30, allow_reduce_only_during_halt=False,
        )
        policy = schema.build()
        assert policy.max_order_notional == Decimal("100000")
        assert policy.max_leverage == Decimal("3")
        assert policy.allow_reduce_only_during_halt is False

    def test_build_with_all_defaults_produces_all_none_limits(self) -> None:
        policy = PortfolioRiskPolicyConfigSchema().build()
        assert policy.max_order_notional is None
        assert policy.max_leverage is None
        assert policy.allow_reduce_only_during_halt is True

    def test_build_rejects_invalid_decimal_string_via_domain_validation(self) -> None:
        from quant_platform.core.exceptions import PortfolioRiskPolicyError

        schema = PortfolioRiskPolicyConfigSchema(max_order_notional="-5")
        with pytest.raises(PortfolioRiskPolicyError):
            schema.build()


class TestPortfolioRiskConfigSchemaStrictness:
    def test_default_constructs(self) -> None:
        schema = PortfolioRiskConfigSchema()
        assert schema.policy is not None

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioRiskConfigSchema(not_a_real_field=1)  # type: ignore[call-arg]

    def test_no_portfolio_id_field_exists(self) -> None:
        # PortfolioRiskSpec is deliberately reusable across portfolios --
        # confirms the top-level config schema does not bind one in.
        assert "portfolio_id" not in PortfolioRiskConfigSchema.model_fields


class TestPortfolioRiskConfigSchemaBuild:
    def test_build_produces_a_spec_with_matching_policy(self) -> None:
        schema = PortfolioRiskConfigSchema(policy=PortfolioRiskPolicyConfigSchema(max_leverage="4"))
        spec = schema.build()
        assert spec.policy.max_leverage == Decimal("4")
        assert spec.schema_version == 1
