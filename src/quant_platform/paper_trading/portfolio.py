"""Account/portfolio state (Milestone 7, Section 13). `PortfolioState`
wraps a `Mapping[instrument, accounting.PositionState]` with account-level
cash/exposure/equity/turnover. The initial implementation supports one
active instrument per session when `PositionPolicySpec.single_instrument_
only` is set (explicit and fail-closed, per Section 13's own permission
for this simplification) -- `positions` itself is not artificially
restricted to size 1 here, so lifting that constraint later does not
require rewriting this module.

EXACT RECONCILIATION FORMULA (Section 13, verbatim):
    equity = cash + marked_position_value - liabilities - accrued_costs
`liabilities` is always `0.0` in this milestone -- a DOCUMENTED
simplification (Section 13 explicitly permits "a clearly documented
margin/P&L abstraction" for derivatives): this is a fully cash-settled
model with no separate borrowed-margin liability line item.
`marked_position_value` and `accrued_costs` are both derived, summed
across `positions` from each `PositionState`'s own already-validated
fields -- never recomputed independently.

CASH CONVENTION: `cash` tracks GROSS trade cash flows only (a BUY
decreases cash by `fill.gross_notional`, a SELL increases it by the same,
neither adjusted for cost) plus financing cash deltas (`costs.compute_
financing_cash_delta`'s sign convention: negative = cost, positive =
credit). Transaction costs (spread/slippage/commission) are NEVER
subtracted from `cash` directly -- they accumulate in each `PositionState.
accumulated_transaction_costs` and are subtracted exactly once, in the
`equity` formula's `accrued_costs` term. This keeps `cash` and `accrued_
costs` two genuinely independent running totals with no double-counting,
and makes `equity - starting_cash == realized_pnl + unrealized_pnl -
accrued_costs + total_financing` hold EXACTLY (see `test_portfolio.py`'s
reconciliation tests)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import PortfolioReconciliationError
from quant_platform.ml.persistence import as_json_dict, format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.accounting import (
    PositionState,
    apply_fill_to_position,
    apply_financing_to_position,
    apply_mark_to_position,
    flat_position,
)
from quant_platform.paper_trading.fills import Fill
from quant_platform.paper_trading.models import OrderSide
from quant_platform.paper_trading.strategy import PortfolioSnapshot


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise PortfolioReconciliationError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise PortfolioReconciliationError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise PortfolioReconciliationError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PortfolioState:
    session_id: str
    starting_cash: float
    cash: float
    positions: Mapping[str, PositionState]
    order_count: int
    fill_count: int
    rejected_order_count: int
    turnover: float
    peak_equity: float
    last_event_time: datetime | None
    portfolio_version: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise PortfolioReconciliationError("PortfolioState.session_id must not be empty")
        if not math.isfinite(self.starting_cash) or self.starting_cash < 0.0:
            raise PortfolioReconciliationError(f"PortfolioState.starting_cash must be finite and >= 0, got {self.starting_cash!r}")
        for field_name, value in (("cash", self.cash), ("turnover", self.turnover), ("peak_equity", self.peak_equity)):
            if not math.isfinite(value):
                raise PortfolioReconciliationError(f"PortfolioState.{field_name} must be finite, got {value!r}")
        if self.turnover < 0.0:
            raise PortfolioReconciliationError(f"PortfolioState.turnover must be >= 0, got {self.turnover!r}")
        for field_name, value in (("order_count", self.order_count), ("fill_count", self.fill_count), ("rejected_order_count", self.rejected_order_count), ("portfolio_version", self.portfolio_version)):
            if value < 0:
                raise PortfolioReconciliationError(f"PortfolioState.{field_name} must be >= 0, got {value}")
        for instrument, position in self.positions.items():
            if position.instrument != instrument:
                raise PortfolioReconciliationError(f"PortfolioState.positions key {instrument!r} does not match PositionState.instrument {position.instrument!r}")

    def position_for(self, instrument: str) -> PositionState | None:
        return self.positions.get(instrument)

    @property
    def marked_position_value(self) -> float:
        return sum(p.signed_quantity * (p.last_mark if p.last_mark is not None else (p.average_entry_price or 0.0)) * p.contract_multiplier for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.signed_quantity) * (p.last_mark if p.last_mark is not None else (p.average_entry_price or 0.0)) * p.contract_multiplier for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return self.marked_position_value

    @property
    def accrued_costs(self) -> float:
        return sum(p.accumulated_transaction_costs for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_financing(self) -> float:
        return sum(p.accumulated_financing for p in self.positions.values())

    @property
    def liabilities(self) -> float:
        """Always `0.0` in this milestone -- see module docstring."""
        return 0.0

    @property
    def equity(self) -> float:
        return self.cash + self.marked_position_value - self.liabilities - self.accrued_costs

    @property
    def drawdown_fraction(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def to_strategy_snapshot(self, instrument: str) -> PortfolioSnapshot:
        """Narrows this full account state down to `strategy.
        PortfolioSnapshot` -- the deliberately limited view a
        `StrategyRuntime` is allowed to see (Section 7)."""
        position = self.positions.get(instrument)
        signed_quantity = 0.0 if position is None else position.signed_quantity
        average_entry_price = None if position is None else position.average_entry_price
        unrealized_pnl = 0.0 if position is None else position.unrealized_pnl
        realized_pnl = 0.0 if position is None else position.realized_pnl
        return PortfolioSnapshot(
            instrument=instrument, signed_quantity=signed_quantity, average_entry_price=average_entry_price, cash=self.cash, equity=self.equity,
            unrealized_pnl=unrealized_pnl, realized_pnl=realized_pnl,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id, "starting_cash": self.starting_cash, "cash": self.cash,
            "positions": {instrument: position.to_json_dict() for instrument, position in self.positions.items()},
            "order_count": self.order_count, "fill_count": self.fill_count, "rejected_order_count": self.rejected_order_count,
            "turnover": self.turnover, "peak_equity": self.peak_equity,
            "last_event_time": (None if self.last_event_time is None else _serialize_timestamp(self.last_event_time, field_name="last_event_time")),
            "portfolio_version": self.portfolio_version,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PortfolioState:
        last_event_time_raw = raw.get("last_event_time")
        positions_raw = as_json_dict(raw.get("positions") or {}, field_name="positions")
        return cls(
            session_id=str(raw["session_id"]), starting_cash=float(str(raw["starting_cash"])), cash=float(str(raw["cash"])),
            positions={instrument: PositionState.from_json_dict(as_json_dict(payload, field_name=f"positions[{instrument!r}]")) for instrument, payload in positions_raw.items()},
            order_count=int(str(raw["order_count"])), fill_count=int(str(raw["fill_count"])), rejected_order_count=int(str(raw["rejected_order_count"])),
            turnover=float(str(raw["turnover"])), peak_equity=float(str(raw["peak_equity"])),
            last_event_time=(None if last_event_time_raw is None else _deserialize_timestamp(last_event_time_raw, field_name="last_event_time")),
            portfolio_version=int(str(raw["portfolio_version"])),
        )


def initial_portfolio(session_id: str, *, starting_cash: float) -> PortfolioState:
    return PortfolioState(
        session_id=session_id, starting_cash=starting_cash, cash=starting_cash, positions={}, order_count=0, fill_count=0, rejected_order_count=0,
        turnover=0.0, peak_equity=starting_cash, last_event_time=None, portfolio_version=0,
    )


def _finalize(portfolio: PortfolioState) -> PortfolioState:
    """Recompute `peak_equity` -- every `apply_*` function below routes
    its result through this before returning, so `peak_equity` (and
    therefore `drawdown_fraction`) is always current."""
    equity = portfolio.equity
    if equity > portfolio.peak_equity:
        return replace(portfolio, peak_equity=equity)
    return portfolio


def apply_fill_to_portfolio(portfolio: PortfolioState, fill: Fill, *, event_time: datetime, contract_multiplier: float) -> PortfolioState:
    position = portfolio.positions.get(fill.instrument) or flat_position(fill.instrument, contract_multiplier=contract_multiplier)
    new_position = apply_fill_to_position(position, fill, event_time=event_time)
    cash_delta = -fill.gross_notional if fill.side is OrderSide.BUY else fill.gross_notional
    new_positions = dict(portfolio.positions)
    new_positions[fill.instrument] = new_position
    updated = replace(
        portfolio, cash=portfolio.cash + cash_delta, positions=new_positions, fill_count=portfolio.fill_count + 1,
        turnover=portfolio.turnover + fill.gross_notional, last_event_time=event_time, portfolio_version=portfolio.portfolio_version + 1,
    )
    return _finalize(updated)


def apply_order_created_to_portfolio(portfolio: PortfolioState, *, event_time: datetime) -> PortfolioState:
    updated = replace(portfolio, order_count=portfolio.order_count + 1, last_event_time=event_time, portfolio_version=portfolio.portfolio_version + 1)
    return _finalize(updated)


def apply_order_rejected_to_portfolio(portfolio: PortfolioState, *, event_time: datetime) -> PortfolioState:
    updated = replace(portfolio, rejected_order_count=portfolio.rejected_order_count + 1, last_event_time=event_time, portfolio_version=portfolio.portfolio_version + 1)
    return _finalize(updated)


def apply_financing_to_portfolio(portfolio: PortfolioState, *, instrument: str, cash_delta: float, event_time: datetime) -> PortfolioState:
    position = portfolio.positions.get(instrument)
    if position is None:
        raise PortfolioReconciliationError(f"apply_financing_to_portfolio: no position exists for instrument {instrument!r}")
    new_position = apply_financing_to_position(position, cash_delta=cash_delta, event_time=event_time)
    new_positions = dict(portfolio.positions)
    new_positions[instrument] = new_position
    updated = replace(portfolio, cash=portfolio.cash + cash_delta, positions=new_positions, last_event_time=event_time, portfolio_version=portfolio.portfolio_version + 1)
    return _finalize(updated)


def apply_mark_to_portfolio(portfolio: PortfolioState, *, instrument: str, mark_price: float, event_time: datetime) -> PortfolioState:
    """A no-op (beyond bookkeeping timestamps) if no position exists yet
    for `instrument` -- marking before ever trading is not an error."""
    position = portfolio.positions.get(instrument)
    if position is None:
        return portfolio
    new_position = apply_mark_to_position(position, mark_price=mark_price, event_time=event_time)
    new_positions = dict(portfolio.positions)
    new_positions[instrument] = new_position
    updated = replace(portfolio, positions=new_positions, last_event_time=event_time, portfolio_version=portfolio.portfolio_version + 1)
    return _finalize(updated)


__all__ = [
    "PortfolioState",
    "apply_fill_to_portfolio",
    "apply_financing_to_portfolio",
    "apply_mark_to_portfolio",
    "apply_order_created_to_portfolio",
    "apply_order_rejected_to_portfolio",
    "initial_portfolio",
]
