"""Unit tests for `portfolio_risk.specs`: `PortfolioRiskPolicy` validation
and `PortfolioRiskSpec` content-addressed identity."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PortfolioRiskPolicyError
from quant_platform.portfolio_risk.specs import (
    PortfolioRiskPolicy,
    PortfolioRiskSpec,
    compute_portfolio_risk_spec_id,
    verify_portfolio_risk_spec_identity,
)


def _policy(**overrides: object) -> PortfolioRiskPolicy:
    base: dict[str, object] = {
        "max_order_notional": Decimal("100000"), "max_position_notional": Decimal("500000"),
        "max_instrument_gross_exposure": Decimal("500000"), "max_strategy_gross_exposure": Decimal("1000000"),
        "max_portfolio_gross_exposure": Decimal("2000000"), "max_portfolio_net_exposure": Decimal("1000000"),
        "max_concentration_fraction": Decimal("0.25"), "max_leverage": Decimal("3"), "max_daily_realized_loss": Decimal("50000"),
        "max_total_loss": Decimal("200000"), "max_drawdown_fraction": Decimal("0.2"), "max_consecutive_losses": 5,
        "minimum_cash_buffer": Decimal("10000"), "maximum_price_age": 30, "maximum_portfolio_snapshot_age": 60,
        "allow_reduce_only_during_halt": True,
    }
    base.update(overrides)
    return PortfolioRiskPolicy(**base)  # type: ignore[arg-type]


def _spec(**overrides: object) -> PortfolioRiskSpec:
    base: dict[str, object] = {"schema_version": 1, "policy": _policy()}
    base.update(overrides)
    return PortfolioRiskSpec(**base)  # type: ignore[arg-type]


class TestPortfolioRiskPolicyAllFieldsOptionalExceptHaltSwitch:
    def test_all_none_limits_construct(self) -> None:
        policy = _policy(
            max_order_notional=None, max_position_notional=None, max_instrument_gross_exposure=None, max_strategy_gross_exposure=None,
            max_portfolio_gross_exposure=None, max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None,
            max_daily_realized_loss=None, max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None,
            minimum_cash_buffer=None, maximum_price_age=None, maximum_portfolio_snapshot_age=None,
        )
        assert policy.max_order_notional is None

    def test_reduce_only_switch_accepts_both_booleans(self) -> None:
        assert _policy(allow_reduce_only_during_halt=True).allow_reduce_only_during_halt is True
        assert _policy(allow_reduce_only_during_halt=False).allow_reduce_only_during_halt is False


class TestPortfolioRiskPolicyInvalidLimitsRejected:
    @pytest.mark.parametrize(
        "field_name",
        [
            "max_order_notional", "max_position_notional", "max_instrument_gross_exposure", "max_strategy_gross_exposure",
            "max_portfolio_gross_exposure", "max_portfolio_net_exposure", "max_leverage", "max_daily_realized_loss", "max_total_loss",
        ],
    )
    def test_zero_magnitude_limit_rejected(self, field_name: str) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(**{field_name: Decimal("0")})

    @pytest.mark.parametrize(
        "field_name",
        [
            "max_order_notional", "max_position_notional", "max_instrument_gross_exposure", "max_strategy_gross_exposure",
            "max_portfolio_gross_exposure", "max_portfolio_net_exposure", "max_leverage", "max_daily_realized_loss", "max_total_loss",
        ],
    )
    def test_negative_magnitude_limit_rejected(self, field_name: str) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(**{field_name: Decimal("-1")})

    @pytest.mark.parametrize("field_name", ["max_concentration_fraction", "max_drawdown_fraction"])
    def test_fraction_above_one_rejected(self, field_name: str) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(**{field_name: Decimal("1.01")})

    @pytest.mark.parametrize("field_name", ["max_concentration_fraction", "max_drawdown_fraction"])
    def test_fraction_of_zero_rejected(self, field_name: str) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(**{field_name: Decimal("0")})

    @pytest.mark.parametrize("field_name", ["max_concentration_fraction", "max_drawdown_fraction"])
    def test_fraction_of_exactly_one_accepted(self, field_name: str) -> None:
        assert _policy(**{field_name: Decimal("1")}) is not None

    def test_negative_max_consecutive_losses_rejected(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(max_consecutive_losses=0)

    def test_negative_minimum_cash_buffer_rejected(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(minimum_cash_buffer=Decimal("-1"))

    def test_zero_minimum_cash_buffer_accepted(self) -> None:
        assert _policy(minimum_cash_buffer=Decimal("0")).minimum_cash_buffer == Decimal("0")

    def test_negative_maximum_price_age_rejected(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(maximum_price_age=-1)

    def test_zero_maximum_price_age_accepted(self) -> None:
        assert _policy(maximum_price_age=0).maximum_price_age == 0

    def test_nan_decimal_rejected(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(max_order_notional=Decimal("NaN"))

    def test_infinite_decimal_rejected(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            _policy(max_order_notional=Decimal("Infinity"))


class TestPortfolioRiskPolicyRoundTrip:
    def test_round_trips_with_all_limits_set(self) -> None:
        policy = _policy()
        restored = PortfolioRiskPolicy.from_json_dict(policy.to_json_dict())
        assert restored.to_json_dict() == policy.to_json_dict()

    def test_round_trips_with_all_limits_none(self) -> None:
        policy = _policy(
            max_order_notional=None, max_position_notional=None, max_instrument_gross_exposure=None, max_strategy_gross_exposure=None,
            max_portfolio_gross_exposure=None, max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None,
            max_daily_realized_loss=None, max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None,
            minimum_cash_buffer=None, maximum_price_age=None, maximum_portfolio_snapshot_age=None,
        )
        restored = PortfolioRiskPolicy.from_json_dict(policy.to_json_dict())
        assert restored.to_json_dict() == policy.to_json_dict()


class TestPortfolioRiskSpecRoundTrip:
    def test_round_trips_through_json(self) -> None:
        spec = _spec()
        restored = PortfolioRiskSpec.from_json_dict(spec.to_json_dict())
        assert restored.to_json_dict() == spec.to_json_dict()


class TestPortfolioRiskSpecIdentity:
    def test_identity_is_deterministic(self) -> None:
        a = compute_portfolio_risk_spec_id(_spec()).portfolio_risk_spec_id
        b = compute_portfolio_risk_spec_id(_spec()).portfolio_risk_spec_id
        assert a == b
        assert len(a) == 64

    def test_identity_is_a_pure_function_verify_matches(self) -> None:
        spec = _spec()
        identity = compute_portfolio_risk_spec_id(spec)
        assert verify_portfolio_risk_spec_identity(spec, identity.portfolio_risk_spec_id)
        assert not verify_portfolio_risk_spec_identity(spec, "0" * 64)

    def test_schema_version_excluded_from_identity(self) -> None:
        a = compute_portfolio_risk_spec_id(_spec(schema_version=1)).portfolio_risk_spec_id
        # schema_version is currently fixed at 1 everywhere in this
        # milestone, so this test documents the exclusion via the
        # identity payload directly rather than constructing a second
        # schema version (none exists yet).
        assert "schema_version" not in _spec().to_identity_payload()
        assert a == compute_portfolio_risk_spec_id(_spec()).portfolio_risk_spec_id

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda s: replace(s, policy=replace(s.policy, max_order_notional=Decimal("999"))),
            lambda s: replace(s, policy=replace(s.policy, max_leverage=Decimal("10"))),
            lambda s: replace(s, policy=replace(s.policy, max_concentration_fraction=Decimal("0.5"))),
            lambda s: replace(s, policy=replace(s.policy, max_consecutive_losses=1)),
            lambda s: replace(s, policy=replace(s.policy, allow_reduce_only_during_halt=False)),
            lambda s: replace(s, policy=replace(s.policy, maximum_price_age=1)),
        ],
    )
    def test_changing_any_policy_field_changes_identity(self, mutate) -> None:
        original = _spec()
        mutated = mutate(original)
        assert compute_portfolio_risk_spec_id(original).portfolio_risk_spec_id != compute_portfolio_risk_spec_id(mutated).portfolio_risk_spec_id

    def test_identity_stable_under_pythonhashseed(self) -> None:
        # No dict/set iteration participates in this spec's identity
        # payload (no unordered collection exists in Phase 1's policy) --
        # this test documents that the identity is a pure function of
        # field values, independent of process-level hash randomization.
        # Mirrors `tests/unit/ml/test_seeds.py::
        # TestDeriveSeed.test_deterministic_across_processes`'s own
        # env-inheritance pattern exactly (never a hand-built minimal env,
        # which risks missing platform-required variables on Windows).
        import os
        import subprocess
        import sys

        script = (
            "from decimal import Decimal\n"
            "from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec, compute_portfolio_risk_spec_id\n"
            "policy = PortfolioRiskPolicy(max_order_notional=Decimal('100'), max_position_notional=None, "
            "max_instrument_gross_exposure=None, max_strategy_gross_exposure=None, max_portfolio_gross_exposure=None, "
            "max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None, max_daily_realized_loss=None, "
            "max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None, minimum_cash_buffer=None, "
            "maximum_price_age=None, maximum_portfolio_snapshot_age=None, allow_reduce_only_during_halt=True)\n"
            "spec = PortfolioRiskSpec(schema_version=1, policy=policy)\n"
            "print(compute_portfolio_risk_spec_id(spec).portfolio_risk_spec_id)\n"
        )
        results = set()
        for seed in ("0", "1", "42"):
            proc = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed},
            )
            results.add(proc.stdout.strip())
        assert len(results) == 1
