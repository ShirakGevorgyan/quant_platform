"""The 18 required pre-trade policy checks for `quant_platform.portfolio_risk`
(Milestone 9, Phase 2). Every function here is PURE: given already-computed
inputs (exposure figures, projections, ages), it returns exactly one
`RiskCheckResult` -- it never recomputes exposure itself (that is
`exposure.py`'s job) and never decides the overall `RiskDecision` (that is
`evaluator.py`'s job). This separation is what makes every check
independently unit-testable against boundary values without needing a
real `PortfolioSnapshot`.

CHECK ORDER IS CANONICAL AND FIXED (`CHECK_ORDER` below) -- `evaluator.py`
runs and appends checks in EXACTLY this order, never via dict/set
iteration, so `RiskDecision.check_results` (and therefore
`RiskDecision.risk_decision_id`) is deterministic across processes and
`PYTHONHASHSEED` values.

SEVERITY ASSIGNMENT (a deliberate, documented Phase 2 design decision):
order/position/exposure/concentration/leverage/cash-buffer checks (1-9)
are DENY severity when breached -- each tests whether THIS ONE proposed
trade should be blocked, and because every one of these checks is
evaluated against the POST-TRADE PROJECTED state (see `evaluator.py`), a
risk-REDUCING trade naturally improves its own measured values and is
therefore never blocked by these checks without any special-casing.
Loss/drawdown/consecutive-loss checks (10-13) are HALT severity when
breached -- these reflect ACCOUNT-WIDE health, independent of any single
order, mirroring `paper_trading.risk`'s own mapping of loss/drawdown
triggers to portfolio-wide `FLATTEN_SIMULATED_POSITIONS`/
`HALT_NEW_ORDERS` actions (more severe than a single `REJECT_ORDER`).
Staleness/reduce-only/missing-data checks (14, 15, 17, 18) are DENY
(evaluation-specific, not account-wide); an ALREADY-halted portfolio (16)
is reported at HALT severity, since it is the halt state itself being
propagated forward, not a new condition being evaluated.

DENIAL REASON REUSE (not a Phase 1 gap): `RiskDenialReason.
INCOHERENT_EVALUATION_STATE` -- Phase 1's own documented "fail-closed
catch-all: required inputs were missing, mutually inconsistent, or
otherwise could not be safely evaluated" -- is reused for BOTH check 17
(reduce-only validity) and check 18 (missing/inconsistent valuation
data), since both are genuinely "the request was not a coherent,
evaluable state" rather than a distinct economic limit breach. Every
other check has its own 1:1 dedicated `RiskDenialReason` member already
defined in Phase 1 -- no new enum member was needed."""

from __future__ import annotations

from decimal import Decimal

from quant_platform.portfolio_risk.decisions import RiskCheckResult
from quant_platform.portfolio_risk.models import RiskCheckSeverity, RiskDenialReason

NOT_CONFIGURED_LIMIT_SENTINEL = Decimal("9" * 30)
"""Reported as `RiskCheckResult.limit_value` when the corresponding
`PortfolioRiskPolicy` field is `None` (not configured). `RiskCheckResult.
limit_value` is a required, FINITE `Decimal` (Phase 1's own invariant --
there is no `Decimal("Infinity")` sentinel available), so an explicit,
documented, very-large-but-finite sentinel is used instead. A check
against this sentinel always trivially passes for any economically
realistic measured value."""

CHECK_ORDER: tuple[str, ...] = (
    "order_notional_limit", "position_notional_limit", "instrument_gross_exposure_limit", "strategy_gross_exposure_limit",
    "portfolio_gross_exposure_limit", "portfolio_net_exposure_limit", "concentration_fraction_limit", "leverage_limit",
    "minimum_cash_buffer", "daily_realized_loss_limit", "total_loss_limit", "drawdown_limit", "consecutive_losses_limit",
    "stale_price", "stale_portfolio_snapshot", "portfolio_halted", "reduce_only_validity", "missing_or_inconsistent_valuation_data",
)

__all__ = [
    "CHECK_ORDER",
    "NOT_CONFIGURED_LIMIT_SENTINEL",
    "check_concentration_fraction_limit",
    "check_consecutive_losses_limit",
    "check_daily_realized_loss_limit",
    "check_drawdown_limit",
    "check_instrument_gross_exposure_limit",
    "check_leverage_limit",
    "check_minimum_cash_buffer",
    "check_missing_or_inconsistent_valuation_data",
    "check_order_notional_limit",
    "check_portfolio_gross_exposure_limit",
    "check_portfolio_halted",
    "check_portfolio_net_exposure_limit",
    "check_position_notional_limit",
    "check_reduce_only_validity",
    "check_stale_portfolio_snapshot",
    "check_stale_price",
    "check_strategy_gross_exposure_limit",
    "check_total_loss_limit",
]


def _ceiling_check(
    check_identity: str, *, measured_value: Decimal, limit_value: Decimal | None, denial_reason: RiskDenialReason,
    severity_if_failed: RiskCheckSeverity,
) -> RiskCheckResult:
    """`measured_value <= limit_value`. `limit_value=None` means "not
    configured" -- reported via `NOT_CONFIGURED_LIMIT_SENTINEL`, always
    passing."""
    if limit_value is None:
        return RiskCheckResult(
            check_identity=check_identity, measured_value=measured_value, limit_value=NOT_CONFIGURED_LIMIT_SENTINEL, passed=True,
            severity=RiskCheckSeverity.INFO, denial_reason=None,
        )
    passed = measured_value <= limit_value
    return RiskCheckResult(
        check_identity=check_identity, measured_value=measured_value, limit_value=limit_value, passed=passed,
        severity=(RiskCheckSeverity.INFO if passed else severity_if_failed), denial_reason=(None if passed else denial_reason),
    )


def _floor_check(
    check_identity: str, *, measured_value: Decimal, floor_value: Decimal | None, denial_reason: RiskDenialReason,
    severity_if_failed: RiskCheckSeverity,
) -> RiskCheckResult:
    """`measured_value >= floor_value`. `floor_value=None` means "not
    configured" -- reported via a `0` floor (always satisfied, since
    every relevant measured value in this package is representable and
    the check is a no-op)."""
    if floor_value is None:
        return RiskCheckResult(
            check_identity=check_identity, measured_value=measured_value, limit_value=Decimal(0), passed=True, severity=RiskCheckSeverity.INFO,
            denial_reason=None,
        )
    passed = measured_value >= floor_value
    return RiskCheckResult(
        check_identity=check_identity, measured_value=measured_value, limit_value=floor_value, passed=passed,
        severity=(RiskCheckSeverity.INFO if passed else severity_if_failed), denial_reason=(None if passed else denial_reason),
    )


def _boolean_check(check_identity: str, *, triggered: bool, denial_reason: RiskDenialReason, severity_if_triggered: RiskCheckSeverity) -> RiskCheckResult:
    return RiskCheckResult(
        check_identity=check_identity, measured_value=(Decimal(1) if triggered else Decimal(0)), limit_value=Decimal(0), passed=not triggered,
        severity=(severity_if_triggered if triggered else RiskCheckSeverity.INFO), denial_reason=(denial_reason if triggered else None),
    )


# --------------------------------------------------------------------------
# 1-9: order/position/exposure/concentration/leverage/cash-buffer -- DENY.
# --------------------------------------------------------------------------
def check_order_notional_limit(*, order_notional: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("order_notional_limit", measured_value=order_notional, limit_value=limit_value, denial_reason=RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_position_notional_limit(*, projected_position_notional: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("position_notional_limit", measured_value=projected_position_notional, limit_value=limit_value, denial_reason=RiskDenialReason.POSITION_NOTIONAL_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_instrument_gross_exposure_limit(*, projected_instrument_gross_exposure: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("instrument_gross_exposure_limit", measured_value=projected_instrument_gross_exposure, limit_value=limit_value, denial_reason=RiskDenialReason.INSTRUMENT_GROSS_EXPOSURE_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_strategy_gross_exposure_limit(*, projected_strategy_gross_exposure: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("strategy_gross_exposure_limit", measured_value=projected_strategy_gross_exposure, limit_value=limit_value, denial_reason=RiskDenialReason.STRATEGY_GROSS_EXPOSURE_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_portfolio_gross_exposure_limit(*, projected_portfolio_gross_exposure: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("portfolio_gross_exposure_limit", measured_value=projected_portfolio_gross_exposure, limit_value=limit_value, denial_reason=RiskDenialReason.PORTFOLIO_GROSS_EXPOSURE_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_portfolio_net_exposure_limit(*, projected_portfolio_net_exposure: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("portfolio_net_exposure_limit", measured_value=abs(projected_portfolio_net_exposure), limit_value=limit_value, denial_reason=RiskDenialReason.PORTFOLIO_NET_EXPOSURE_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_concentration_fraction_limit(*, projected_concentration_fraction: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("concentration_fraction_limit", measured_value=projected_concentration_fraction, limit_value=limit_value, denial_reason=RiskDenialReason.CONCENTRATION_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_leverage_limit(*, projected_leverage: Decimal | None, limit_value: Decimal | None) -> RiskCheckResult:
    """`projected_leverage=None` signals `exposure.compute_leverage`
    raised (`equity <= 0`) -- an automatic, unconditional failure at HALT
    severity, regardless of whether `limit_value` is configured, since
    non-positive equity is an account-health condition, not an ordinary
    over-limit trade."""
    if projected_leverage is None:
        reported_limit = limit_value if limit_value is not None else NOT_CONFIGURED_LIMIT_SENTINEL
        return RiskCheckResult(
            check_identity="leverage_limit", measured_value=NOT_CONFIGURED_LIMIT_SENTINEL, limit_value=reported_limit, passed=False,
            severity=RiskCheckSeverity.HALT, denial_reason=RiskDenialReason.LEVERAGE_LIMIT_EXCEEDED,
        )
    return _ceiling_check("leverage_limit", measured_value=projected_leverage, limit_value=limit_value, denial_reason=RiskDenialReason.LEVERAGE_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.DENY)


def check_minimum_cash_buffer(*, projected_cash: Decimal, floor_value: Decimal | None) -> RiskCheckResult:
    return _floor_check("minimum_cash_buffer", measured_value=projected_cash, floor_value=floor_value, denial_reason=RiskDenialReason.CASH_BUFFER_BREACHED, severity_if_failed=RiskCheckSeverity.DENY)


# --------------------------------------------------------------------------
# 10-13: account-health -- HALT.
# --------------------------------------------------------------------------
def check_daily_realized_loss_limit(*, projected_daily_loss: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("daily_realized_loss_limit", measured_value=projected_daily_loss, limit_value=limit_value, denial_reason=RiskDenialReason.DAILY_REALIZED_LOSS_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.HALT)


def check_total_loss_limit(*, projected_total_loss: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("total_loss_limit", measured_value=projected_total_loss, limit_value=limit_value, denial_reason=RiskDenialReason.TOTAL_LOSS_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.HALT)


def check_drawdown_limit(*, projected_drawdown_fraction: Decimal, limit_value: Decimal | None) -> RiskCheckResult:
    return _ceiling_check("drawdown_limit", measured_value=projected_drawdown_fraction, limit_value=limit_value, denial_reason=RiskDenialReason.DRAWDOWN_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.HALT)


def check_consecutive_losses_limit(*, consecutive_losses: int, limit_value: int | None) -> RiskCheckResult:
    limit_decimal = None if limit_value is None else Decimal(limit_value)
    return _ceiling_check("consecutive_losses_limit", measured_value=Decimal(consecutive_losses), limit_value=limit_decimal, denial_reason=RiskDenialReason.CONSECUTIVE_LOSSES_LIMIT_EXCEEDED, severity_if_failed=RiskCheckSeverity.HALT)


# --------------------------------------------------------------------------
# 14-18: staleness / halt / reduce-only / missing data -- DENY except an
# already-active halt (HALT, propagating the existing state forward).
# --------------------------------------------------------------------------
def check_stale_price(*, age_seconds: Decimal, limit_value: int | None) -> RiskCheckResult:
    limit_decimal = None if limit_value is None else Decimal(limit_value)
    return _ceiling_check("stale_price", measured_value=age_seconds, limit_value=limit_decimal, denial_reason=RiskDenialReason.STALE_PRICE, severity_if_failed=RiskCheckSeverity.DENY)


def check_stale_portfolio_snapshot(*, age_seconds: Decimal, limit_value: int | None) -> RiskCheckResult:
    limit_decimal = None if limit_value is None else Decimal(limit_value)
    return _ceiling_check("stale_portfolio_snapshot", measured_value=age_seconds, limit_value=limit_decimal, denial_reason=RiskDenialReason.STALE_PORTFOLIO_SNAPSHOT, severity_if_failed=RiskCheckSeverity.DENY)


def check_portfolio_halted(*, portfolio_halted: bool) -> RiskCheckResult:
    return _boolean_check("portfolio_halted", triggered=portfolio_halted, denial_reason=RiskDenialReason.PORTFOLIO_HALTED, severity_if_triggered=RiskCheckSeverity.HALT)


def check_reduce_only_validity(*, reduce_only: bool, is_risk_increasing: bool) -> RiskCheckResult:
    triggered = reduce_only and is_risk_increasing
    return _boolean_check("reduce_only_validity", triggered=triggered, denial_reason=RiskDenialReason.INCOHERENT_EVALUATION_STATE, severity_if_triggered=RiskCheckSeverity.DENY)


def check_missing_or_inconsistent_valuation_data(*, reduce_only: bool, has_existing_position: bool) -> RiskCheckResult:
    """Triggered when `reduce_only=True` but no existing position exists
    for the requested `(instrument_id, strategy_id)` -- there is nothing
    to reduce, so the request cannot be coherently evaluated."""
    triggered = reduce_only and not has_existing_position
    return _boolean_check("missing_or_inconsistent_valuation_data", triggered=triggered, denial_reason=RiskDenialReason.INCOHERENT_EVALUATION_STATE, severity_if_triggered=RiskCheckSeverity.DENY)
