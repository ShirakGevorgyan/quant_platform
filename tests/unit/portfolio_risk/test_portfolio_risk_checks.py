"""Unit tests for `portfolio_risk.checks`: every one of the 18 required
policy checks, at and around their boundaries, plus the `None`-limit
"disabled, not globally approved" behavior."""

from __future__ import annotations

from decimal import Decimal

from quant_platform.portfolio_risk import checks
from quant_platform.portfolio_risk.models import RiskCheckSeverity, RiskDenialReason


class TestCheckOrderIsCanonicalAndComplete:
    def test_exactly_eighteen_checks_in_fixed_order(self) -> None:
        assert len(checks.CHECK_ORDER) == 18
        assert len(set(checks.CHECK_ORDER)) == 18

    def test_order_matches_the_required_specification_order(self) -> None:
        assert checks.CHECK_ORDER == (
            "order_notional_limit", "position_notional_limit", "instrument_gross_exposure_limit", "strategy_gross_exposure_limit",
            "portfolio_gross_exposure_limit", "portfolio_net_exposure_limit", "concentration_fraction_limit", "leverage_limit",
            "minimum_cash_buffer", "daily_realized_loss_limit", "total_loss_limit", "drawdown_limit", "consecutive_losses_limit",
            "stale_price", "stale_portfolio_snapshot", "portfolio_halted", "reduce_only_validity", "missing_or_inconsistent_valuation_data",
        )


class TestBoundaryBehaviorForCeilingChecks:
    """A representative ceiling check exercised at just-below / exactly-at
    / just-above its boundary -- every `_ceiling_check`-based function
    shares the identical `<=` semantics, verified once here in depth and
    spot-checked per-function below."""

    def test_just_below_boundary_passes(self) -> None:
        result = checks.check_order_notional_limit(order_notional=Decimal("9999.99"), limit_value=Decimal("10000"))
        assert result.passed is True
        assert result.severity is RiskCheckSeverity.INFO
        assert result.denial_reason is None

    def test_exactly_at_boundary_passes(self) -> None:
        result = checks.check_order_notional_limit(order_notional=Decimal("10000"), limit_value=Decimal("10000"))
        assert result.passed is True

    def test_just_above_boundary_fails(self) -> None:
        result = checks.check_order_notional_limit(order_notional=Decimal("10000.01"), limit_value=Decimal("10000"))
        assert result.passed is False
        assert result.severity is RiskCheckSeverity.DENY
        assert result.denial_reason is RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED

    def test_none_limit_disables_this_specific_check_only(self) -> None:
        result = checks.check_order_notional_limit(order_notional=Decimal("999999999"), limit_value=None)
        assert result.passed is True
        assert result.severity is RiskCheckSeverity.INFO
        assert result.measured_value == Decimal("999999999")
        assert result.limit_value == checks.NOT_CONFIGURED_LIMIT_SENTINEL


class TestEachLimitCheckIndividually:
    def test_position_notional_limit(self) -> None:
        assert checks.check_position_notional_limit(projected_position_notional=Decimal("5000"), limit_value=Decimal("5000")).passed
        assert not checks.check_position_notional_limit(projected_position_notional=Decimal("5000.01"), limit_value=Decimal("5000")).passed

    def test_instrument_gross_exposure_limit(self) -> None:
        assert checks.check_instrument_gross_exposure_limit(projected_instrument_gross_exposure=Decimal("1000"), limit_value=Decimal("1000")).passed
        r = checks.check_instrument_gross_exposure_limit(projected_instrument_gross_exposure=Decimal("1000.01"), limit_value=Decimal("1000"))
        assert not r.passed and r.denial_reason is RiskDenialReason.INSTRUMENT_GROSS_EXPOSURE_LIMIT_EXCEEDED

    def test_strategy_gross_exposure_limit(self) -> None:
        r = checks.check_strategy_gross_exposure_limit(projected_strategy_gross_exposure=Decimal("1000.01"), limit_value=Decimal("1000"))
        assert not r.passed and r.denial_reason is RiskDenialReason.STRATEGY_GROSS_EXPOSURE_LIMIT_EXCEEDED

    def test_portfolio_gross_exposure_limit(self) -> None:
        r = checks.check_portfolio_gross_exposure_limit(projected_portfolio_gross_exposure=Decimal("1000.01"), limit_value=Decimal("1000"))
        assert not r.passed and r.denial_reason is RiskDenialReason.PORTFOLIO_GROSS_EXPOSURE_LIMIT_EXCEEDED

    def test_portfolio_net_exposure_limit_uses_absolute_value(self) -> None:
        passing = checks.check_portfolio_net_exposure_limit(projected_portfolio_net_exposure=Decimal("-1000"), limit_value=Decimal("1000"))
        assert passing.passed and passing.measured_value == Decimal("1000")
        failing = checks.check_portfolio_net_exposure_limit(projected_portfolio_net_exposure=Decimal("-1000.01"), limit_value=Decimal("1000"))
        assert not failing.passed

    def test_concentration_fraction_limit(self) -> None:
        assert checks.check_concentration_fraction_limit(projected_concentration_fraction=Decimal("0.25"), limit_value=Decimal("0.25")).passed
        assert not checks.check_concentration_fraction_limit(projected_concentration_fraction=Decimal("0.2501"), limit_value=Decimal("0.25")).passed

    def test_leverage_limit_normal_case(self) -> None:
        assert checks.check_leverage_limit(projected_leverage=Decimal("3"), limit_value=Decimal("3")).passed
        r = checks.check_leverage_limit(projected_leverage=Decimal("3.01"), limit_value=Decimal("3"))
        assert not r.passed and r.severity is RiskCheckSeverity.DENY

    def test_leverage_limit_undefined_leverage_always_fails_at_halt_severity(self) -> None:
        r = checks.check_leverage_limit(projected_leverage=None, limit_value=Decimal("3"))
        assert not r.passed
        assert r.severity is RiskCheckSeverity.HALT
        assert r.denial_reason is RiskDenialReason.LEVERAGE_LIMIT_EXCEEDED

    def test_leverage_limit_undefined_fails_even_without_a_configured_limit(self) -> None:
        r = checks.check_leverage_limit(projected_leverage=None, limit_value=None)
        assert not r.passed
        assert r.severity is RiskCheckSeverity.HALT

    def test_minimum_cash_buffer_is_a_floor(self) -> None:
        assert checks.check_minimum_cash_buffer(projected_cash=Decimal("1000"), floor_value=Decimal("1000")).passed
        r = checks.check_minimum_cash_buffer(projected_cash=Decimal("999.99"), floor_value=Decimal("1000"))
        assert not r.passed and r.denial_reason is RiskDenialReason.CASH_BUFFER_BREACHED

    def test_minimum_cash_buffer_none_disables(self) -> None:
        r = checks.check_minimum_cash_buffer(projected_cash=Decimal("-999999"), floor_value=None)
        assert r.passed

    def test_daily_realized_loss_limit_is_halt_severity(self) -> None:
        r = checks.check_daily_realized_loss_limit(projected_daily_loss=Decimal("1000.01"), limit_value=Decimal("1000"))
        assert not r.passed and r.severity is RiskCheckSeverity.HALT and r.denial_reason is RiskDenialReason.DAILY_REALIZED_LOSS_LIMIT_EXCEEDED

    def test_total_loss_limit_is_halt_severity(self) -> None:
        r = checks.check_total_loss_limit(projected_total_loss=Decimal("1000.01"), limit_value=Decimal("1000"))
        assert not r.passed and r.severity is RiskCheckSeverity.HALT and r.denial_reason is RiskDenialReason.TOTAL_LOSS_LIMIT_EXCEEDED

    def test_drawdown_limit_is_halt_severity(self) -> None:
        assert checks.check_drawdown_limit(projected_drawdown_fraction=Decimal("0.2"), limit_value=Decimal("0.2")).passed
        r = checks.check_drawdown_limit(projected_drawdown_fraction=Decimal("0.2001"), limit_value=Decimal("0.2"))
        assert not r.passed and r.severity is RiskCheckSeverity.HALT

    def test_consecutive_losses_limit_is_halt_severity(self) -> None:
        assert checks.check_consecutive_losses_limit(consecutive_losses=5, limit_value=5).passed
        r = checks.check_consecutive_losses_limit(consecutive_losses=6, limit_value=5)
        assert not r.passed and r.severity is RiskCheckSeverity.HALT and r.denial_reason is RiskDenialReason.CONSECUTIVE_LOSSES_LIMIT_EXCEEDED

    def test_stale_price_boundary(self) -> None:
        assert checks.check_stale_price(age_seconds=Decimal("30"), limit_value=30).passed
        r = checks.check_stale_price(age_seconds=Decimal("30.01"), limit_value=30)
        assert not r.passed and r.denial_reason is RiskDenialReason.STALE_PRICE

    def test_stale_price_none_limit_disables(self) -> None:
        assert checks.check_stale_price(age_seconds=Decimal("999999"), limit_value=None).passed

    def test_stale_portfolio_snapshot_boundary(self) -> None:
        assert checks.check_stale_portfolio_snapshot(age_seconds=Decimal("60"), limit_value=60).passed
        r = checks.check_stale_portfolio_snapshot(age_seconds=Decimal("60.01"), limit_value=60)
        assert not r.passed and r.denial_reason is RiskDenialReason.STALE_PORTFOLIO_SNAPSHOT

    def test_portfolio_halted_check(self) -> None:
        assert checks.check_portfolio_halted(portfolio_halted=False).passed
        r = checks.check_portfolio_halted(portfolio_halted=True)
        assert not r.passed and r.severity is RiskCheckSeverity.HALT and r.denial_reason is RiskDenialReason.PORTFOLIO_HALTED

    def test_reduce_only_validity_check(self) -> None:
        assert checks.check_reduce_only_validity(reduce_only=True, is_risk_increasing=False).passed
        assert checks.check_reduce_only_validity(reduce_only=False, is_risk_increasing=True).passed
        r = checks.check_reduce_only_validity(reduce_only=True, is_risk_increasing=True)
        assert not r.passed and r.severity is RiskCheckSeverity.DENY and r.denial_reason is RiskDenialReason.INCOHERENT_EVALUATION_STATE

    def test_missing_or_inconsistent_valuation_data_check(self) -> None:
        assert checks.check_missing_or_inconsistent_valuation_data(reduce_only=True, has_existing_position=True).passed
        assert checks.check_missing_or_inconsistent_valuation_data(reduce_only=False, has_existing_position=False).passed
        r = checks.check_missing_or_inconsistent_valuation_data(reduce_only=True, has_existing_position=False)
        assert not r.passed and r.denial_reason is RiskDenialReason.INCOHERENT_EVALUATION_STATE


class TestNoneLimitMeansDisabledNotGlobalApproval:
    """A None limit disables only ITS OWN check -- it must never leak into
    approving anything else. Verified by confirming a None-limited check
    always reports passed=True/INFO regardless of how extreme the
    measured value is, while a DIFFERENT, still-configured check on the
    same inputs is unaffected."""

    def test_extreme_measured_value_with_none_limit_still_passes_only_that_check(self) -> None:
        order_check = checks.check_order_notional_limit(order_notional=Decimal("1e12"), limit_value=None)
        leverage_check = checks.check_leverage_limit(projected_leverage=Decimal("999"), limit_value=Decimal("3"))
        assert order_check.passed is True
        assert leverage_check.passed is False


class TestAllChecksAlwaysProduceAResult:
    def test_every_check_function_returns_exactly_one_result_never_none(self) -> None:
        results = [
            checks.check_order_notional_limit(order_notional=Decimal("1"), limit_value=None),
            checks.check_position_notional_limit(projected_position_notional=Decimal("1"), limit_value=None),
            checks.check_instrument_gross_exposure_limit(projected_instrument_gross_exposure=Decimal("1"), limit_value=None),
            checks.check_strategy_gross_exposure_limit(projected_strategy_gross_exposure=Decimal("1"), limit_value=None),
            checks.check_portfolio_gross_exposure_limit(projected_portfolio_gross_exposure=Decimal("1"), limit_value=None),
            checks.check_portfolio_net_exposure_limit(projected_portfolio_net_exposure=Decimal("1"), limit_value=None),
            checks.check_concentration_fraction_limit(projected_concentration_fraction=Decimal("1"), limit_value=None),
            checks.check_leverage_limit(projected_leverage=Decimal("1"), limit_value=None),
            checks.check_minimum_cash_buffer(projected_cash=Decimal("1"), floor_value=None),
            checks.check_daily_realized_loss_limit(projected_daily_loss=Decimal("0"), limit_value=None),
            checks.check_total_loss_limit(projected_total_loss=Decimal("0"), limit_value=None),
            checks.check_drawdown_limit(projected_drawdown_fraction=Decimal("0"), limit_value=None),
            checks.check_consecutive_losses_limit(consecutive_losses=0, limit_value=None),
            checks.check_stale_price(age_seconds=Decimal("0"), limit_value=None),
            checks.check_stale_portfolio_snapshot(age_seconds=Decimal("0"), limit_value=None),
            checks.check_portfolio_halted(portfolio_halted=False),
            checks.check_reduce_only_validity(reduce_only=False, is_risk_increasing=False),
            checks.check_missing_or_inconsistent_valuation_data(reduce_only=False, has_existing_position=False),
        ]
        assert len(results) == 18
        assert all(r is not None for r in results)
        assert tuple(r.check_identity for r in results) == checks.CHECK_ORDER
