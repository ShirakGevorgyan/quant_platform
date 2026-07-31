"""Pure evaluation orchestration for `quant_platform.portfolio_risk`
(Milestone 9, Phase 2). `evaluate_risk` is the one function that ties
`exposure.py`/`valuation.py`/`checks.py`/`sizing.py` together into a
final, immutable `RiskDecision` -- and, only when `APPROVED`, a
`PositionSizeProposal`/`CapitalAllocation`.

FAIL-CLOSED BY CONSTRUCTION, NOT BY CONVENTION:
- `_verify_request_bindings` recomputes every referenced object's OWN
  identity (never trusts a caller-declared id) and raises
  `RiskEvaluationError` -- never returns a decision -- on any mismatch: a
  forged/tampered snapshot, a price snapshot for the wrong instrument, or
  a portfolio snapshot for the wrong portfolio. These are CALLER/
  INTEGRATION defects, not economic conditions a `RiskDecision` should
  ever represent.
- `_aggregate_decision_kind` raises `RiskEvaluationError` if
  `check_results` does not contain EXACTLY the 18 checks `checks.
  CHECK_ORDER` requires -- an `APPROVED` decision is structurally
  impossible unless every required check actually ran.
- No `except Exception`/bare `except` exists anywhere in this module.
  Domain-shaped conditions (stale data, exposure over a limit, a halted
  portfolio) become ordinary `RiskCheckResult`s; anything else (a
  programming defect) propagates as a real exception, visible to tests
  and callers, never silently turned into an approval OR a denial."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from quant_platform.core.exceptions import (
    ExposureCalculationError,
    RiskEvaluationError,
    StalePortfolioSnapshotError,
    StalePriceError,
)
from quant_platform.portfolio_risk import checks, exposure
from quant_platform.portfolio_risk.allocation import (
    CapitalAllocation,
    PositionSizeProposal,
    create_capital_allocation,
    create_position_size_proposal,
)
from quant_platform.portfolio_risk.decisions import (
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
    create_risk_decision,
)
from quant_platform.portfolio_risk.models import (
    RiskCheckSeverity,
    RiskDecisionKind,
    RiskDenialReason,
    most_severe_check_severity,
)
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    PriceSnapshot,
    verify_portfolio_snapshot_identity,
    verify_price_snapshot_identity,
)
from quant_platform.portfolio_risk.specs import PortfolioRiskSpec, compute_portfolio_risk_spec_id
from quant_platform.portfolio_risk.valuation import (
    TradeRiskClassification,
    project_fill_price,
    project_portfolio,
)

__all__ = ["EvaluationOutcome", "evaluate_risk"]


def _age_seconds(*, event_time: datetime, reference_time: datetime, error_cls: type[Exception]) -> Decimal:
    """Exact `Decimal` seconds between two timezone-aware datetimes --
    NEVER `timedelta.total_seconds()` (a `float`). Raises `error_cls` if
    `reference_time` precedes `event_time` (mirrors `snapshots.
    is_price_stale`/`is_portfolio_snapshot_stale`'s identical guard)."""
    if event_time.tzinfo is None or reference_time.tzinfo is None:
        raise error_cls("age computation requires timezone-aware datetimes")
    delta = reference_time - event_time
    if delta < timedelta(0):
        raise error_cls(f"reference_time {reference_time!r} precedes event_time {event_time!r}")
    return Decimal(delta.days) * Decimal(86400) + Decimal(delta.seconds) + Decimal(delta.microseconds) / Decimal(1_000_000)


def _verify_request_bindings(*, request: RiskEvaluationRequest, portfolio: PortfolioSnapshot, price: PriceSnapshot, spec: PortfolioRiskSpec) -> None:
    if not verify_portfolio_snapshot_identity(portfolio, request.portfolio_snapshot_id):
        raise RiskEvaluationError(
            f"evaluate_risk: portfolio snapshot identity mismatch -- portfolio.snapshot_id={portfolio.snapshot_id!r} does not match "
            f"request.portfolio_snapshot_id={request.portfolio_snapshot_id!r} (or the snapshot's own id does not reproduce its content -- forged/tampered snapshot)"
        )
    if not verify_price_snapshot_identity(price, request.price_snapshot_id):
        raise RiskEvaluationError(
            f"evaluate_risk: price snapshot identity mismatch -- price.price_snapshot_id={price.price_snapshot_id!r} does not match "
            f"request.price_snapshot_id={request.price_snapshot_id!r} (or the snapshot's own id does not reproduce its content -- forged/tampered snapshot)"
        )
    computed_policy_id = compute_portfolio_risk_spec_id(spec).portfolio_risk_spec_id
    if computed_policy_id != request.risk_policy_id:
        raise RiskEvaluationError(
            f"evaluate_risk: policy identity mismatch -- spec recomputes to {computed_policy_id!r}, request.risk_policy_id={request.risk_policy_id!r}"
        )
    if portfolio.portfolio_id != request.portfolio_id:
        raise RiskEvaluationError(f"evaluate_risk: portfolio.portfolio_id={portfolio.portfolio_id!r} does not match request.portfolio_id={request.portfolio_id!r}")
    if price.instrument_id != request.instrument_id:
        raise RiskEvaluationError(
            f"evaluate_risk: cross-instrument price snapshot -- price.instrument_id={price.instrument_id!r} does not match request.instrument_id={request.instrument_id!r}"
        )


def _aggregate_decision_kind(check_results: tuple[RiskCheckResult, ...]) -> tuple[RiskDecisionKind, tuple[RiskDenialReason, ...]]:
    """Raises `RiskEvaluationError` unless `check_results` contains
    EXACTLY the 18 checks `checks.CHECK_ORDER` requires -- an `APPROVED`
    decision is structurally impossible if even one required check did
    not run, regardless of what the ones that DID run would have
    concluded."""
    present = tuple(c.check_identity for c in check_results)
    if set(present) != set(checks.CHECK_ORDER) or len(present) != len(checks.CHECK_ORDER):
        missing = set(checks.CHECK_ORDER) - set(present)
        raise RiskEvaluationError(f"evaluate_risk: not every required check was executed -- missing {sorted(missing)!r}")

    overall_severity = most_severe_check_severity(tuple(c.severity for c in check_results))
    if overall_severity is RiskCheckSeverity.HALT:
        kind = RiskDecisionKind.HALTED
    elif overall_severity is RiskCheckSeverity.DENY:
        kind = RiskDecisionKind.DENIED
    else:
        kind = RiskDecisionKind.APPROVED

    denial_reasons = tuple(sorted({c.denial_reason for c in check_results if not c.passed and c.denial_reason is not None}, key=lambda r: r.value))
    return kind, denial_reasons


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    decision: RiskDecision
    position_size_proposal: PositionSizeProposal | None
    """`None` unless `decision.kind is RiskDecisionKind.APPROVED`."""
    capital_allocation: CapitalAllocation | None
    """`None` unless `decision.kind is RiskDecisionKind.APPROVED`."""


def evaluate_risk(
    *, request: RiskEvaluationRequest, portfolio: PortfolioSnapshot, price: PriceSnapshot, spec: PortfolioRiskSpec, evaluation_time: datetime,
    portfolio_halted: bool, consecutive_losses: int, contract_multiplier: Decimal, decision_sequence: int,
) -> EvaluationOutcome:
    """`portfolio_halted`/`consecutive_losses`/`contract_multiplier` are
    explicit, mandatory, caller-supplied parameters -- NOT fields on
    `RiskEvaluationRequest` (a Phase 1 model this phase does not modify).
    Phase 2 has no durable ledger yet to derive a pre-existing halt state
    or a losing-streak count from; a later phase's persistence layer
    would supply these from real history. `evaluation_time` is likewise
    always caller-supplied -- this function never reads the wall clock."""
    _verify_request_bindings(request=request, portfolio=portfolio, price=price, spec=spec)
    policy = spec.policy

    current_position = portfolio.position_for(instrument_id=request.instrument_id, strategy_id=request.strategy_id)
    has_existing_position = current_position is not None

    projection = project_portfolio(
        portfolio, instrument_id=request.instrument_id, strategy_id=request.strategy_id, side=request.side, quantity=request.quantity, price=price,
        contract_multiplier=contract_multiplier, evaluation_time=evaluation_time,
    )
    projected_portfolio = projection.portfolio
    is_risk_increasing = projection.classification is TradeRiskClassification.INCREASING

    fill_price = project_fill_price(price, side=request.side)
    order_notional = request.quantity * fill_price * contract_multiplier

    projected_position = projected_portfolio.position_for(instrument_id=request.instrument_id, strategy_id=request.strategy_id)
    projected_position_notional = abs(projected_position.market_value) if projected_position is not None else Decimal(0)

    projected_instrument_gross_exposure = exposure.compute_instrument_exposure(projected_portfolio, instrument_id=request.instrument_id).gross_exposure
    projected_strategy_gross_exposure = exposure.compute_strategy_exposure(projected_portfolio, strategy_id=request.strategy_id).gross_exposure
    projected_portfolio_exposure = exposure.compute_portfolio_exposure(projected_portfolio)
    projected_concentration_fraction = exposure.compute_concentration_fraction(projected_portfolio)
    try:
        projected_leverage: Decimal | None = exposure.compute_leverage(projected_portfolio)
    except ExposureCalculationError:
        projected_leverage = None
    projected_daily_loss = exposure.compute_daily_loss(projected_portfolio)
    projected_total_loss = exposure.compute_total_loss(projected_portfolio)
    projected_drawdown_fraction = projected_portfolio.drawdown_fraction

    price_age_seconds = _age_seconds(event_time=price.event_time, reference_time=evaluation_time, error_cls=StalePriceError)
    portfolio_age_seconds = _age_seconds(event_time=portfolio.event_time, reference_time=evaluation_time, error_cls=StalePortfolioSnapshotError)

    check_results: tuple[RiskCheckResult, ...] = (
        checks.check_order_notional_limit(order_notional=order_notional, limit_value=policy.max_order_notional),
        checks.check_position_notional_limit(projected_position_notional=projected_position_notional, limit_value=policy.max_position_notional),
        checks.check_instrument_gross_exposure_limit(projected_instrument_gross_exposure=projected_instrument_gross_exposure, limit_value=policy.max_instrument_gross_exposure),
        checks.check_strategy_gross_exposure_limit(projected_strategy_gross_exposure=projected_strategy_gross_exposure, limit_value=policy.max_strategy_gross_exposure),
        checks.check_portfolio_gross_exposure_limit(projected_portfolio_gross_exposure=projected_portfolio_exposure.gross_exposure, limit_value=policy.max_portfolio_gross_exposure),
        checks.check_portfolio_net_exposure_limit(projected_portfolio_net_exposure=projected_portfolio_exposure.net_exposure, limit_value=policy.max_portfolio_net_exposure),
        checks.check_concentration_fraction_limit(projected_concentration_fraction=projected_concentration_fraction, limit_value=policy.max_concentration_fraction),
        checks.check_leverage_limit(projected_leverage=projected_leverage, limit_value=policy.max_leverage),
        checks.check_minimum_cash_buffer(projected_cash=projected_portfolio.cash, floor_value=policy.minimum_cash_buffer),
        checks.check_daily_realized_loss_limit(projected_daily_loss=projected_daily_loss, limit_value=policy.max_daily_realized_loss),
        checks.check_total_loss_limit(projected_total_loss=projected_total_loss, limit_value=policy.max_total_loss),
        checks.check_drawdown_limit(projected_drawdown_fraction=projected_drawdown_fraction, limit_value=policy.max_drawdown_fraction),
        checks.check_consecutive_losses_limit(consecutive_losses=consecutive_losses, limit_value=policy.max_consecutive_losses),
        checks.check_stale_price(age_seconds=price_age_seconds, limit_value=policy.maximum_price_age),
        checks.check_stale_portfolio_snapshot(age_seconds=portfolio_age_seconds, limit_value=policy.maximum_portfolio_snapshot_age),
        checks.check_portfolio_halted(portfolio_halted=portfolio_halted),
        checks.check_reduce_only_validity(reduce_only=request.reduce_only, is_risk_increasing=is_risk_increasing),
        checks.check_missing_or_inconsistent_valuation_data(reduce_only=request.reduce_only, has_existing_position=has_existing_position),
    )
    kind, denial_reasons = _aggregate_decision_kind(check_results)

    decision = create_risk_decision(
        risk_evaluation_request_id=request.risk_evaluation_request_id, kind=kind, denial_reasons=denial_reasons, check_results=check_results,
        evaluated_quantity=request.quantity, evaluated_price=fill_price, portfolio_snapshot_id=portfolio.snapshot_id, price_snapshot_id=price.price_snapshot_id,
        risk_policy_id=request.risk_policy_id, decision_sequence=decision_sequence, event_time=evaluation_time,
    )

    position_size_proposal: PositionSizeProposal | None = None
    capital_allocation: CapitalAllocation | None = None
    if kind is RiskDecisionKind.APPROVED:
        position_size_proposal = create_position_size_proposal(
            portfolio_id=request.portfolio_id, strategy_id=request.strategy_id, instrument_id=request.instrument_id, side=request.side,
            proposed_quantity=request.quantity, reference_price=fill_price, proposed_sequence=decision_sequence, event_time=evaluation_time,
        )
        allocated_capital = policy.max_strategy_gross_exposure if policy.max_strategy_gross_exposure is not None else projected_strategy_gross_exposure
        capital_allocation = create_capital_allocation(
            portfolio_id=request.portfolio_id, strategy_id=request.strategy_id, allocated_capital=allocated_capital,
            utilized_capital=projected_strategy_gross_exposure, allocation_sequence=decision_sequence, event_time=evaluation_time,
        )

    return EvaluationOutcome(decision=decision, position_size_proposal=position_size_proposal, capital_allocation=capital_allocation)
