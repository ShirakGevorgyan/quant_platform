"""Derived portfolio exposure and account metrics for
`quant_platform.portfolio_risk` (Milestone 9, Phase 2). Every function
here is PURE and DERIVES its result from a `PortfolioSnapshot`'s own
already-validated fields -- nothing here independently trusts a stored
figure Phase 1 didn't already validate, and nothing here stores a result
back onto a `PortfolioSnapshot`.

Reuses `snapshots.compute_portfolio_exposure`/`compute_instrument_
exposure`/`compute_strategy_exposure` and `PortfolioSnapshot.
drawdown_fraction` directly rather than recomputing gross/net exposure or
drawdown a second way (Phase 2's own "do not create redundant
abstractions when Phase 1 already covers the concept" instruction).

DOCUMENTED ACCOUNTING MODEL FOR LOSS METRICS (a genuine, deliberate
Phase 2 design decision, not an oversight): `PortfolioSnapshot` carries
`realized_pnl` (cumulative SINCE INCEPTION -- Phase 1's own documented
choice, since closed positions' realized pnl is not retained in the
point-in-time `positions` collection) and `daily_start_equity` (a
day-scoped EQUITY baseline), but no separate "realized pnl at the start
of today" baseline. Two DISTINCT, non-redundant loss measures are
therefore defined from what actually exists:

- `total_loss` = `max(0, -(realized_pnl + unrealized_pnl))` -- the
  portfolio's total pnl loss since inception, mirroring `paper_trading.
  risk.evaluate_continuous_risk`'s own `-(realized_pnl + unrealized_pnl)`
  formula exactly (that function calls this "daily_loss" only because
  `paper_trading.portfolio.PortfolioState` has no day-boundary concept at
  all; this package's own equivalent, inception-scoped figure is named
  `total_loss` instead, since a genuinely daily-scoped figure exists
  separately below).
- `daily_loss` = `max(0, daily_start_equity - equity)` -- the portfolio's
  TOTAL (realized + unrealized combined) equity decline since the start
  of the current day. This is what `PortfolioRiskPolicy.
  max_daily_realized_loss` is checked against in this phase. It is named
  "realized" in the policy field because a future phase that adds a
  `realized_pnl`-at-day-start baseline to `PortfolioSnapshot` could
  isolate the realized component specifically; Phase 1's snapshot does
  not carry that baseline, so Phase 2 measures the day's total equity
  decline instead. This is a documented, explicit Known Limitation (see
  `docs/portfolio_risk_architecture.md`), never a silent approximation."""

from __future__ import annotations

from decimal import Decimal

from quant_platform.core.exceptions import ExposureCalculationError
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    compute_instrument_exposure,
    compute_portfolio_exposure,
    compute_strategy_exposure,
)

__all__ = [
    "compute_available_cash",
    "compute_concentration_fraction",
    "compute_daily_loss",
    "compute_instrument_exposure",
    "compute_leverage",
    "compute_long_gross_exposure",
    "compute_portfolio_exposure",
    "compute_short_gross_exposure",
    "compute_strategy_exposure",
    "compute_total_loss",
]


def compute_long_gross_exposure(portfolio: PortfolioSnapshot) -> Decimal:
    """Sum of `abs(market_value)` across every LONG (positive
    `signed_quantity`) position. Always `>= 0`."""
    return sum((abs(p.market_value) for p in portfolio.positions if p.signed_quantity > 0), start=Decimal(0))


def compute_short_gross_exposure(portfolio: PortfolioSnapshot) -> Decimal:
    """Sum of `abs(market_value)` across every SHORT (negative
    `signed_quantity`) position. Always `>= 0` -- a MAGNITUDE, not a
    signed figure (long_gross_exposure + short_gross_exposure ==
    portfolio gross exposure, by construction)."""
    return sum((abs(p.market_value) for p in portfolio.positions if p.signed_quantity < 0), start=Decimal(0))


def compute_concentration_fraction(portfolio: PortfolioSnapshot) -> Decimal:
    """Largest fraction of PORTFOLIO gross exposure any single instrument
    represents -- `max(instrument gross exposure) / portfolio gross
    exposure`, aggregated ACROSS STRATEGIES for the same instrument (an
    instrument's concentration risk does not care which strategy holds
    it). `0` when the portfolio is flat (no exposure to concentrate)."""
    portfolio_gross = compute_portfolio_exposure(portfolio).gross_exposure
    if portfolio_gross == 0:
        return Decimal(0)
    instrument_ids = {p.instrument_id for p in portfolio.positions}
    largest = max((compute_instrument_exposure(portfolio, instrument_id=i).gross_exposure for i in instrument_ids), default=Decimal(0))
    return largest / portfolio_gross


def compute_leverage(portfolio: PortfolioSnapshot) -> Decimal:
    """`gross_exposure / equity`. Raises `ExposureCalculationError` when
    `equity <= 0` -- leverage is mathematically undefined (or infinite)
    against non-positive equity, and this package never silently reports
    a sentinel value (e.g. `0` or `Decimal("Infinity")`) for an undefined
    calculation; callers evaluating a leverage LIMIT must treat this
    exception as an automatic, fail-closed breach."""
    if portfolio.equity <= 0:
        raise ExposureCalculationError(f"compute_leverage: portfolio.equity must be > 0 to compute leverage, got {portfolio.equity!r}")
    gross_exposure = compute_portfolio_exposure(portfolio).gross_exposure
    return gross_exposure / portfolio.equity


def compute_available_cash(portfolio: PortfolioSnapshot, *, minimum_cash_buffer: Decimal) -> Decimal:
    """`cash - minimum_cash_buffer`. May be negative (a caller checking
    against a `minimum_cash_buffer` limit should treat a negative result
    as an already-breached buffer, not clamp it to zero and hide the
    breach)."""
    return portfolio.cash - minimum_cash_buffer


def compute_daily_loss(portfolio: PortfolioSnapshot) -> Decimal:
    """See module docstring's "DOCUMENTED ACCOUNTING MODEL FOR LOSS
    METRICS" -- the day's TOTAL (realized + unrealized) equity decline
    since `daily_start_equity`, floored at `0` (a gain is not a
    "loss")."""
    return max(Decimal(0), portfolio.daily_start_equity - portfolio.equity)


def compute_total_loss(portfolio: PortfolioSnapshot) -> Decimal:
    """See module docstring -- total pnl loss since inception, floored at
    `0`."""
    total_pnl = portfolio.realized_pnl + portfolio.unrealized_pnl
    return max(Decimal(0), -total_pnl)
