"""Shadow observation mode (Milestone 7, Section 19). A `SHADOW_
OBSERVATION` session runs the IDENTICAL decision -> order-policy ->
execution -> accounting pipeline every other session mode uses (Section
19: "hypothetical execution outcomes MAY be estimated" -- this
implementation computes them EXACTLY, via the same deterministic
functions, rather than approximating), but every result is captured in a
`ShadowObservation` and applied ONLY to a separate `PositionState` this
module owns -- never to `portfolio.PortfolioState`. There is no code path
from this module into `portfolio.apply_fill_to_portfolio`; a shadow
session's REAL account (if one even exists) is structurally impossible to
touch from here.

"Do not merge shadow outcomes with paper-account P&L. Reports must
clearly label SHADOW versus PAPER" (Section 19) -- `ShadowObservation`'s
own field names are all prefixed `hypothetical_`/`counterfactual_`, and
`reports.py` (built later) must never fold these into a `paper_session_
report`'s real P&L figures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.backtesting.models import PositionDirection
from quant_platform.backtesting.specs import CommissionSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.accounting import PositionState, apply_fill_to_position
from quant_platform.paper_trading.costs import (
    compute_commission_dollars,
    compute_slippage_cost_dollars,
    compute_spread_cost_dollars,
)
from quant_platform.paper_trading.events import MarketEvent, market_event_id
from quant_platform.paper_trading.execution import evaluate_order_against_event
from quant_platform.paper_trading.fills import create_fill
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import OrderSide
from quant_platform.paper_trading.order_policy import OrderPolicyState, apply_order_policy
from quant_platform.paper_trading.orders import OrderRequest
from quant_platform.paper_trading.specs import (
    ExecutionPolicySpec,
    FillPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    RiskLimitsSpec,
)
from quant_platform.paper_trading.strategy import PortfolioSnapshot, StrategyDecision

SHADOW_OBSERVATION_KIND = "shadow_observation"


def _direction_for_side(side: OrderSide) -> PositionDirection:
    return PositionDirection.LONG if side is OrderSide.BUY else PositionDirection.SHORT


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise PaperTradingError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PaperTradingError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PaperTradingError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PaperTradingError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    observation_id: str
    session_id: str
    decision_id: str
    instrument: str
    hypothetical_order_id: str | None
    hypothetical_fill_id: str | None
    hypothetical_fill_price: float | None
    hypothetical_fill_quantity: float | None
    counterfactual_realized_pnl_delta: float | None
    event_identity: str
    event_time: datetime
    sequence: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise PaperTradingError("ShadowObservation.session_id must not be empty")
        if not self.instrument:
            raise PaperTradingError("ShadowObservation.instrument must not be empty")
        _require_tz_aware(self.event_time, field_name="ShadowObservation.event_time")
        if self.sequence < 0:
            raise PaperTradingError(f"ShadowObservation.sequence must be >= 0, got {self.sequence}")
        if self.hypothetical_fill_id is not None and self.hypothetical_order_id is None:
            raise PaperTradingError("ShadowObservation.hypothetical_order_id is required when hypothetical_fill_id is present")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id, "session_id": self.session_id, "decision_id": self.decision_id, "instrument": self.instrument,
            "hypothetical_order_id": self.hypothetical_order_id, "hypothetical_fill_id": self.hypothetical_fill_id,
            "hypothetical_fill_price": self.hypothetical_fill_price, "hypothetical_fill_quantity": self.hypothetical_fill_quantity,
            "counterfactual_realized_pnl_delta": self.counterfactual_realized_pnl_delta, "event_identity": self.event_identity,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["observation_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ShadowObservation:
        return cls(
            observation_id=str(raw["observation_id"]), session_id=str(raw["session_id"]), decision_id=str(raw["decision_id"]),
            instrument=str(raw["instrument"]), hypothetical_order_id=(None if raw.get("hypothetical_order_id") is None else str(raw["hypothetical_order_id"])),
            hypothetical_fill_id=(None if raw.get("hypothetical_fill_id") is None else str(raw["hypothetical_fill_id"])),
            hypothetical_fill_price=(None if raw.get("hypothetical_fill_price") is None else float(str(raw["hypothetical_fill_price"]))),
            hypothetical_fill_quantity=(None if raw.get("hypothetical_fill_quantity") is None else float(str(raw["hypothetical_fill_quantity"]))),
            counterfactual_realized_pnl_delta=(None if raw.get("counterfactual_realized_pnl_delta") is None else float(str(raw["counterfactual_realized_pnl_delta"]))),
            event_identity=str(raw["event_identity"]), event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
        )


def _create_shadow_observation(
    *, session_id: str, decision_id: str, instrument: str, hypothetical_order_id: str | None, hypothetical_fill_id: str | None,
    hypothetical_fill_price: float | None, hypothetical_fill_quantity: float | None, counterfactual_realized_pnl_delta: float | None,
    event_identity: str, event_time: datetime, sequence: int,
) -> ShadowObservation:
    kwargs: dict[str, object] = {
        "session_id": session_id, "decision_id": decision_id, "instrument": instrument, "hypothetical_order_id": hypothetical_order_id,
        "hypothetical_fill_id": hypothetical_fill_id, "hypothetical_fill_price": hypothetical_fill_price,
        "hypothetical_fill_quantity": hypothetical_fill_quantity, "counterfactual_realized_pnl_delta": counterfactual_realized_pnl_delta,
        "event_identity": event_identity, "event_time": event_time, "sequence": sequence,
    }
    provisional = ShadowObservation(observation_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    observation_id = compute_content_id(SHADOW_OBSERVATION_KIND, provisional.to_identity_payload())
    return ShadowObservation(observation_id=observation_id, **kwargs)  # type: ignore[arg-type]


def evaluate_shadow_decision(
    decision: StrategyDecision, *, shadow_position: PositionState, instrument: InstrumentSpec, order_policy: OrderPolicySpec,
    execution_policy: ExecutionPolicySpec, spread_policy: SpreadSpec, slippage_policy: SlippageSpec, commission_policy: CommissionSpec,
    fill_policy: FillPolicySpec, liquidity_policy: LiquidityPolicySpec, latency_policy: LatencyPolicySpec, risk_limits: RiskLimitsSpec,
    session_id: str, event: MarketEvent, order_policy_state: OrderPolicyState, sequence: int,
) -> tuple[ShadowObservation, PositionState]:
    """Runs decision -> order-policy -> execution -> accounting exactly
    like the real pipeline, entirely against `shadow_position` (never a
    real `PortfolioState`). Returns exactly ONE observation per decision:
    an abstention or a no-op decision produces an observation with every
    hypothetical field `None` (still persisted -- Section 19: "observations
    persisted for later comparison" applies to every decision, not only
    ones that would have traded). Only the FIRST hypothetical order (if
    order-policy produces more than one, e.g. a close-then-reverse pair)
    is evaluated for an immediate fill against `event` -- a documented
    simplification consistent with shadow observation's diagnostic,
    single-event-horizon purpose."""
    event_identity = market_event_id(event)
    snapshot = PortfolioSnapshot(
        instrument=instrument.symbol, signed_quantity=shadow_position.signed_quantity, average_entry_price=shadow_position.average_entry_price,
        cash=0.0, equity=0.0, unrealized_pnl=shadow_position.unrealized_pnl, realized_pnl=shadow_position.realized_pnl,
    )
    hypothetical_orders = apply_order_policy(
        decision, portfolio=snapshot, instrument=instrument, policy=order_policy, risk_limits=risk_limits, latency_policy=latency_policy,
        session_id=session_id, create_time=decision.decision_time, state=order_policy_state,
    )

    if not hypothetical_orders:
        observation = _create_shadow_observation(
            session_id=session_id, decision_id=decision.decision_id, instrument=instrument.symbol, hypothetical_order_id=None,
            hypothetical_fill_id=None, hypothetical_fill_price=None, hypothetical_fill_quantity=None, counterfactual_realized_pnl_delta=None,
            event_identity=event_identity, event_time=decision.decision_time, sequence=sequence,
        )
        return observation, shadow_position

    order: OrderRequest = hypothetical_orders[0]
    candidate = evaluate_order_against_event(
        order, event, remaining_quantity=order.quantity, execution_policy=execution_policy, spread_policy=spread_policy, slippage_policy=slippage_policy,
        liquidity_policy=liquidity_policy, fill_policy=fill_policy,
    )

    if candidate is None:
        observation = _create_shadow_observation(
            session_id=session_id, decision_id=decision.decision_id, instrument=instrument.symbol, hypothetical_order_id=order.order_id,
            hypothetical_fill_id=None, hypothetical_fill_price=None, hypothetical_fill_quantity=None, counterfactual_realized_pnl_delta=None,
            event_identity=event_identity, event_time=decision.decision_time, sequence=sequence,
        )
        return observation, shadow_position

    direction = _direction_for_side(order.side)
    notional = candidate.price * candidate.quantity * instrument.contract_multiplier
    hypothetical_fill = create_fill(
        order_id=order.order_id, session_id=session_id, instrument=instrument.symbol, side=order.side, quantity=candidate.quantity,
        price=candidate.price, contract_multiplier=instrument.contract_multiplier,
        spread_cost=compute_spread_cost_dollars(spread_policy, candidate.price, direction, is_entry=True, quantity=candidate.quantity, contract_multiplier=instrument.contract_multiplier),
        slippage_cost=compute_slippage_cost_dollars(slippage_policy, candidate.price, direction, is_entry=True, quantity=candidate.quantity, contract_multiplier=instrument.contract_multiplier),
        commission_cost=compute_commission_dollars(commission_policy, notional=notional), execution_time=decision.decision_time,
        source_market_event_identity=event_identity, liquidity_assumption=candidate.liquidity_assumption, is_final=True,
    )
    realized_before = shadow_position.realized_pnl
    updated_shadow_position = apply_fill_to_position(shadow_position, hypothetical_fill, event_time=decision.decision_time)
    realized_delta = updated_shadow_position.realized_pnl - realized_before

    observation = _create_shadow_observation(
        session_id=session_id, decision_id=decision.decision_id, instrument=instrument.symbol, hypothetical_order_id=order.order_id,
        hypothetical_fill_id=hypothetical_fill.fill_id, hypothetical_fill_price=hypothetical_fill.price, hypothetical_fill_quantity=hypothetical_fill.quantity,
        counterfactual_realized_pnl_delta=realized_delta, event_identity=event_identity, event_time=decision.decision_time, sequence=sequence,
    )
    return observation, updated_shadow_position


__all__ = ["SHADOW_OBSERVATION_KIND", "ShadowObservation", "evaluate_shadow_decision"]
