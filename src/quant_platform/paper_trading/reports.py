"""Durable session reporting (Milestone 7, Section 27) and the optional
backtest-comparison diagnostic (Section 28).

`build_paper_session_report` aggregates PRESENTATION data -- the 15 named
summaries Section 27 requires -- from the event ledger. This is
deliberately NOT a re-verification layer (that is `reconciliation.py`/
`verification.py`'s own job, Sections 25/26): it freely uses the LAST
`ACCOUNT_SNAPSHOT` entry for aggregate account/equity figures (Section 21
explicitly permits treating a snapshot as a cache), and accepts an
already-computed `ReconciliationReport` rather than recomputing one.
Where a cheap independent recomputation is easy and materially more
informative than the snapshot alone -- win/loss counts, maximum drawdown
over the WHOLE session rather than only its final value -- this module
replays the ledger's own `FILL`/`ACCOUNT_SNAPSHOT` entries through the
SAME production functions (`accounting.apply_fill_to_position`,
`PortfolioState.from_json_dict(...).drawdown_fraction`) the forward run
itself used, never a duplicate formula.

`compare_paper_to_backtest` (Section 28) intentionally does NOT reach into
`backtesting.reporting`'s own report-dict internals -- extracting
comparable numbers from a specific backtest report shape is the CALLER's
job (e.g. the real-acceptance-workflow test, Section 33, which already
has a `build_backtest_report_json(...)` result on hand); this module owns
only the comparison/classification structure (`BacktestComparisonMetrics`
in, `BacktestComparisonReport` out), keeping the two report formats
decoupled. The mapping from "which metric differed" to "which Section-28
bucket" is a fixed, documented heuristic (not a data-driven root-cause
analysis) -- any UNEXPECTED classification is a prompt for manual
investigation, not an automated verdict."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.ml.models import ValidationReport
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    parse_utc_timestamp,
    utc_now,
)
from quant_platform.paper_trading.accounting import apply_fill_to_position, flat_position
from quant_platform.paper_trading.fills import Fill
from quant_platform.paper_trading.manifests import PaperSessionManifest
from quant_platform.paper_trading.models import LedgerEntryKind, OrderState
from quant_platform.paper_trading.orders import OrderStateEvent, resolve_order_state
from quant_platform.paper_trading.persistence import LedgerEntry
from quant_platform.paper_trading.portfolio import PortfolioState
from quant_platform.paper_trading.reconciliation import ReconciliationReport
from quant_platform.paper_trading.risk import KillSwitchTransitionEvent
from quant_platform.paper_trading.specs import PaperTradingSpec

PAPER_SESSION_REPORT_SCHEMA_VERSION = 1
BACKTEST_COMPARISON_REPORT_SCHEMA_VERSION = 1

DIAGNOSTIC_DISCLAIMER = "This report is diagnostic, not a promotion decision. Simulated paper fills are not broker fills; paper trading does not prove profitability."

_STALE_DATA_CHECK_SUBSTRING = "stale_data"


# --------------------------------------------------------------------------
# Section 27 -- durable session report
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    session_mode: str
    instrument: str
    manifest_stage: str
    event_count: int
    starting_cash: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id, "session_mode": self.session_mode, "instrument": self.instrument,
            "manifest_stage": self.manifest_stage, "event_count": self.event_count, "starting_cash": self.starting_cash,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SessionSummary:
        return cls(
            session_id=str(raw["session_id"]), session_mode=str(raw["session_mode"]), instrument=str(raw["instrument"]),
            manifest_stage=str(raw["manifest_stage"]), event_count=int(str(raw["event_count"])), starting_cash=float(str(raw["starting_cash"])),
        )


@dataclass(frozen=True, slots=True)
class StrategyDecisionSummary:
    decision_count: int
    abstention_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {"decision_count": self.decision_count, "abstention_count": self.abstention_count}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StrategyDecisionSummary:
        return cls(decision_count=int(str(raw["decision_count"])), abstention_count=int(str(raw["abstention_count"])))


@dataclass(frozen=True, slots=True)
class OrderSummary:
    order_count: int
    rejected_count: int
    filled_count: int
    cancelled_count: int
    expired_count: int
    working_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "order_count": self.order_count, "rejected_count": self.rejected_count, "filled_count": self.filled_count,
            "cancelled_count": self.cancelled_count, "expired_count": self.expired_count, "working_count": self.working_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OrderSummary:
        return cls(
            order_count=int(str(raw["order_count"])), rejected_count=int(str(raw["rejected_count"])), filled_count=int(str(raw["filled_count"])),
            cancelled_count=int(str(raw["cancelled_count"])), expired_count=int(str(raw["expired_count"])), working_count=int(str(raw["working_count"])),
        )


@dataclass(frozen=True, slots=True)
class FillSummary:
    fill_count: int
    partial_fill_count: int
    final_fill_count: int
    total_filled_quantity: float
    total_gross_notional: float
    winning_fill_count: int
    losing_fill_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fill_count": self.fill_count, "partial_fill_count": self.partial_fill_count, "final_fill_count": self.final_fill_count,
            "total_filled_quantity": self.total_filled_quantity, "total_gross_notional": self.total_gross_notional,
            "winning_fill_count": self.winning_fill_count, "losing_fill_count": self.losing_fill_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FillSummary:
        return cls(
            fill_count=int(str(raw["fill_count"])), partial_fill_count=int(str(raw["partial_fill_count"])),
            final_fill_count=int(str(raw["final_fill_count"])), total_filled_quantity=float(str(raw["total_filled_quantity"])),
            total_gross_notional=float(str(raw["total_gross_notional"])), winning_fill_count=int(str(raw["winning_fill_count"])),
            losing_fill_count=int(str(raw["losing_fill_count"])),
        )


@dataclass(frozen=True, slots=True)
class ExecutionQualitySummary:
    average_execution_delay_seconds: float | None
    order_fill_rate: float
    quantity_fill_rate: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "average_execution_delay_seconds": self.average_execution_delay_seconds, "order_fill_rate": self.order_fill_rate,
            "quantity_fill_rate": self.quantity_fill_rate,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionQualitySummary:
        raw_delay = raw.get("average_execution_delay_seconds")
        return cls(
            average_execution_delay_seconds=(None if raw_delay is None else float(str(raw_delay))),
            order_fill_rate=float(str(raw["order_fill_rate"])), quantity_fill_rate=float(str(raw["quantity_fill_rate"])),
        )


@dataclass(frozen=True, slots=True)
class CostSummary:
    total_spread_cost: float
    total_slippage_cost: float
    total_commission_cost: float
    total_financing: float
    total_costs: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "total_spread_cost": self.total_spread_cost, "total_slippage_cost": self.total_slippage_cost,
            "total_commission_cost": self.total_commission_cost, "total_financing": self.total_financing, "total_costs": self.total_costs,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostSummary:
        return cls(
            total_spread_cost=float(str(raw["total_spread_cost"])), total_slippage_cost=float(str(raw["total_slippage_cost"])),
            total_commission_cost=float(str(raw["total_commission_cost"])), total_financing=float(str(raw["total_financing"])),
            total_costs=float(str(raw["total_costs"])),
        )


@dataclass(frozen=True, slots=True)
class PositionSummary:
    instrument: str
    final_signed_quantity: float
    final_average_entry_price: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {"instrument": self.instrument, "final_signed_quantity": self.final_signed_quantity, "final_average_entry_price": self.final_average_entry_price}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PositionSummary:
        raw_price = raw.get("final_average_entry_price")
        return cls(instrument=str(raw["instrument"]), final_signed_quantity=float(str(raw["final_signed_quantity"])), final_average_entry_price=(None if raw_price is None else float(str(raw_price))))


@dataclass(frozen=True, slots=True)
class AccountEquitySummary:
    starting_cash: float
    final_cash: float
    final_equity: float
    peak_equity: float
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    gross_pnl: float
    net_pnl: float
    turnover: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "starting_cash": self.starting_cash, "final_cash": self.final_cash, "final_equity": self.final_equity, "peak_equity": self.peak_equity,
            "gross_exposure": self.gross_exposure, "net_exposure": self.net_exposure, "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl, "gross_pnl": self.gross_pnl, "net_pnl": self.net_pnl, "turnover": self.turnover,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AccountEquitySummary:
        return cls(
            starting_cash=float(str(raw["starting_cash"])), final_cash=float(str(raw["final_cash"])), final_equity=float(str(raw["final_equity"])),
            peak_equity=float(str(raw["peak_equity"])), gross_exposure=float(str(raw["gross_exposure"])), net_exposure=float(str(raw["net_exposure"])),
            realized_pnl=float(str(raw["realized_pnl"])), unrealized_pnl=float(str(raw["unrealized_pnl"])), gross_pnl=float(str(raw["gross_pnl"])),
            net_pnl=float(str(raw["net_pnl"])), turnover=float(str(raw["turnover"])),
        )


@dataclass(frozen=True, slots=True)
class DrawdownSummary:
    peak_equity: float
    final_equity: float
    maximum_drawdown_fraction: float
    final_drawdown_fraction: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "peak_equity": self.peak_equity, "final_equity": self.final_equity, "maximum_drawdown_fraction": self.maximum_drawdown_fraction,
            "final_drawdown_fraction": self.final_drawdown_fraction,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DrawdownSummary:
        return cls(
            peak_equity=float(str(raw["peak_equity"])), final_equity=float(str(raw["final_equity"])),
            maximum_drawdown_fraction=float(str(raw["maximum_drawdown_fraction"])), final_drawdown_fraction=float(str(raw["final_drawdown_fraction"])),
        )


@dataclass(frozen=True, slots=True)
class RiskEventSummary:
    total_risk_checks: int
    failed_risk_checks: int
    stale_data_incidents: int
    halts_triggered: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "total_risk_checks": self.total_risk_checks, "failed_risk_checks": self.failed_risk_checks,
            "stale_data_incidents": self.stale_data_incidents, "halts_triggered": self.halts_triggered,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskEventSummary:
        return cls(
            total_risk_checks=int(str(raw["total_risk_checks"])), failed_risk_checks=int(str(raw["failed_risk_checks"])),
            stale_data_incidents=int(str(raw["stale_data_incidents"])), halts_triggered=int(str(raw["halts_triggered"])),
        )


@dataclass(frozen=True, slots=True)
class RejectionSummary:
    rejection_count: int
    rejections_by_reason: tuple[tuple[str, int], ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"rejection_count": self.rejection_count, "rejections_by_reason": dict(self.rejections_by_reason)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RejectionSummary:
        by_reason = as_json_dict(raw.get("rejections_by_reason") or {}, field_name="rejections_by_reason")
        return cls(rejection_count=int(str(raw["rejection_count"])), rejections_by_reason=tuple(sorted((str(k), int(str(v))) for k, v in by_reason.items())))


@dataclass(frozen=True, slots=True)
class HaltSummary:
    halt_count: int
    final_kill_switch_state: str | None
    transitions: tuple[dict[str, object], ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"halt_count": self.halt_count, "final_kill_switch_state": self.final_kill_switch_state, "transitions": list(self.transitions)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HaltSummary:
        transitions_raw = as_json_list(raw.get("transitions") or [], field_name="transitions")
        return cls(
            halt_count=int(str(raw["halt_count"])), final_kill_switch_state=(None if raw.get("final_kill_switch_state") is None else str(raw["final_kill_switch_state"])),
            transitions=tuple(as_json_dict(t, field_name="transitions[]") for t in transitions_raw),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    is_reconciled: bool
    total_checks: int
    failed_check_identities: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"is_reconciled": self.is_reconciled, "total_checks": self.total_checks, "failed_check_identities": list(self.failed_check_identities)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReconciliationSummary:
        return cls(
            is_reconciled=bool(raw["is_reconciled"]), total_checks=int(str(raw["total_checks"])),
            failed_check_identities=tuple(str(c) for c in as_json_list(raw.get("failed_check_identities") or [], field_name="failed_check_identities")),
        )


@dataclass(frozen=True, slots=True)
class ShadowObservationSummary:
    """Section 19: "Reports must clearly label SHADOW versus PAPER" --
    this summary's every field is explicitly `hypothetical_`/
    `counterfactual_`-scoped and is NEVER folded into `AccountEquitySummary`
    or any other real-account figure above."""

    observation_count: int
    observations_with_hypothetical_fill_count: int
    total_counterfactual_realized_pnl: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "observation_count": self.observation_count, "observations_with_hypothetical_fill_count": self.observations_with_hypothetical_fill_count,
            "total_counterfactual_realized_pnl": self.total_counterfactual_realized_pnl,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ShadowObservationSummary:
        return cls(
            observation_count=int(str(raw["observation_count"])), observations_with_hypothetical_fill_count=int(str(raw["observations_with_hypothetical_fill_count"])),
            total_counterfactual_realized_pnl=float(str(raw["total_counterfactual_realized_pnl"])),
        )


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """`verify_paper_session` (Section 26) is comparatively expensive
    (re-runs the full eligibility chain) and requires an `Eligibility
    VerificationEnvironment` this report-building step does not otherwise
    need -- `verification_report` is therefore OPTIONAL input; when the
    caller has not run it, every field here is `None`/`0`, never a
    fabricated "verified" claim."""

    was_run: bool
    is_ready: bool | None
    critical_count: int
    error_count: int
    warning_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {"was_run": self.was_run, "is_ready": self.is_ready, "critical_count": self.critical_count, "error_count": self.error_count, "warning_count": self.warning_count}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> VerificationSummary:
        raw_ready = raw.get("is_ready")
        return cls(
            was_run=bool(raw["was_run"]), is_ready=(None if raw_ready is None else bool(raw_ready)), critical_count=int(str(raw["critical_count"])),
            error_count=int(str(raw["error_count"])), warning_count=int(str(raw["warning_count"])),
        )


@dataclass(frozen=True, slots=True)
class PaperSessionReport:
    schema_version: int
    session: SessionSummary
    decisions: StrategyDecisionSummary
    orders: OrderSummary
    fills: FillSummary
    execution_quality: ExecutionQualitySummary
    costs: CostSummary
    positions: tuple[PositionSummary, ...]
    account_equity: AccountEquitySummary
    drawdown: DrawdownSummary
    risk_events: RiskEventSummary
    rejections: RejectionSummary
    halts: HaltSummary
    reconciliation: ReconciliationSummary
    shadow: ShadowObservationSummary
    verification: VerificationSummary
    disclaimer: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "session": self.session.to_json_dict(), "decisions": self.decisions.to_json_dict(),
            "orders": self.orders.to_json_dict(), "fills": self.fills.to_json_dict(), "execution_quality": self.execution_quality.to_json_dict(),
            "costs": self.costs.to_json_dict(), "positions": [p.to_json_dict() for p in self.positions],
            "account_equity": self.account_equity.to_json_dict(), "drawdown": self.drawdown.to_json_dict(), "risk_events": self.risk_events.to_json_dict(),
            "rejections": self.rejections.to_json_dict(), "halts": self.halts.to_json_dict(), "reconciliation": self.reconciliation.to_json_dict(),
            "shadow": self.shadow.to_json_dict(), "verification": self.verification.to_json_dict(), "disclaimer": self.disclaimer,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PaperSessionReport:
        positions_raw = as_json_list(raw.get("positions") or [], field_name="positions")
        return cls(
            schema_version=int(str(raw["schema_version"])), session=SessionSummary.from_json_dict(as_json_dict(raw["session"], field_name="session")),
            decisions=StrategyDecisionSummary.from_json_dict(as_json_dict(raw["decisions"], field_name="decisions")),
            orders=OrderSummary.from_json_dict(as_json_dict(raw["orders"], field_name="orders")), fills=FillSummary.from_json_dict(as_json_dict(raw["fills"], field_name="fills")),
            execution_quality=ExecutionQualitySummary.from_json_dict(as_json_dict(raw["execution_quality"], field_name="execution_quality")),
            costs=CostSummary.from_json_dict(as_json_dict(raw["costs"], field_name="costs")),
            positions=tuple(PositionSummary.from_json_dict(as_json_dict(p, field_name="positions[]")) for p in positions_raw),
            account_equity=AccountEquitySummary.from_json_dict(as_json_dict(raw["account_equity"], field_name="account_equity")),
            drawdown=DrawdownSummary.from_json_dict(as_json_dict(raw["drawdown"], field_name="drawdown")),
            risk_events=RiskEventSummary.from_json_dict(as_json_dict(raw["risk_events"], field_name="risk_events")),
            rejections=RejectionSummary.from_json_dict(as_json_dict(raw["rejections"], field_name="rejections")),
            halts=HaltSummary.from_json_dict(as_json_dict(raw["halts"], field_name="halts")),
            reconciliation=ReconciliationSummary.from_json_dict(as_json_dict(raw["reconciliation"], field_name="reconciliation")),
            shadow=ShadowObservationSummary.from_json_dict(as_json_dict(raw["shadow"], field_name="shadow")),
            verification=VerificationSummary.from_json_dict(as_json_dict(raw["verification"], field_name="verification")),
            disclaimer=str(raw["disclaimer"]), generated_at=str(raw["generated_at"]),
        )


def _extract_orders(ledger: list[LedgerEntry]) -> dict[str, tuple[dict[str, object], list[OrderStateEvent]]]:
    orders: dict[str, tuple[dict[str, object], list[OrderStateEvent]]] = {}
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.ORDER_STATE_EVENT:
            continue
        order_json = entry.payload["order"]
        state_event = OrderStateEvent.from_json_dict(entry.payload["order_state_event"])  # type: ignore[arg-type]
        if state_event.order_id not in orders:
            orders[state_event.order_id] = (order_json, [])  # type: ignore[assignment]
        orders[state_event.order_id][1].append(state_event)
    return orders


def build_paper_session_report(
    ledger: list[LedgerEntry], *, spec: PaperTradingSpec, manifest: PaperSessionManifest, reconciliation_report: ReconciliationReport,
    verification_report: ValidationReport | None = None,
) -> PaperSessionReport:
    event_count = sum(1 for e in ledger if e.kind is LedgerEntryKind.MARKET_EVENT_ACCEPTED)

    decision_count = 0
    abstention_count = 0
    for entry in ledger:
        if entry.kind is LedgerEntryKind.STRATEGY_DECISION:
            decision_count += 1
            if bool(entry.payload.get("abstain")):
                abstention_count += 1

    orders = _extract_orders(ledger)
    rejected_count = filled_count = cancelled_count = expired_count = working_count = 0
    rejections_by_reason: dict[str, int] = {}
    execution_delays: list[float] = []
    order_create_times: dict[str, object] = {}
    for order_id, (order_json, events) in orders.items():
        order_create_times[order_id] = order_json["create_time"]
        final_state = resolve_order_state(order_id, events)
        if final_state is OrderState.REJECTED:
            rejected_count += 1
            reject_event = next((e for e in events if e.to_state is OrderState.REJECTED), None)
            if reject_event is not None and reject_event.reason_code is not None:
                key = reject_event.reason_code.value
                rejections_by_reason[key] = rejections_by_reason.get(key, 0) + 1
        elif final_state is OrderState.FILLED:
            filled_count += 1
        elif final_state is OrderState.CANCELLED:
            cancelled_count += 1
        elif final_state is OrderState.EXPIRED:
            expired_count += 1
        else:
            working_count += 1

    fills = [Fill.from_json_dict(entry.payload) for entry in ledger if entry.kind is LedgerEntryKind.FILL]
    fill_count = len(fills)
    partial_fill_count = sum(1 for f in fills if not f.is_final)
    final_fill_count = fill_count - partial_fill_count
    total_filled_quantity = sum(f.quantity for f in fills)
    total_gross_notional = sum(f.gross_notional for f in fills)
    total_spread_cost = sum(f.spread_cost for f in fills)
    total_slippage_cost = sum(f.slippage_cost for f in fills)
    total_commission_cost = sum(f.commission_cost for f in fills)

    orders_with_fill = {f.order_id for f in fills}
    order_fill_rate = (len(orders_with_fill) / len(orders)) if orders else 0.0
    total_order_quantity = sum(float(str(order_json["quantity"])) for order_json, _ in orders.values())
    quantity_fill_rate = (total_filled_quantity / total_order_quantity) if total_order_quantity > 0.0 else 0.0
    for fill in fills:
        create_time = order_create_times.get(fill.order_id)
        if create_time is not None:
            delay = (fill.execution_time - parse_utc_timestamp(str(create_time)).to_pydatetime()).total_seconds()
            execution_delays.append(delay)
    average_execution_delay_seconds = (sum(execution_delays) / len(execution_delays)) if execution_delays else None

    total_financing = sum(float(str(e.payload["cash_delta"])) for e in ledger if e.kind is LedgerEntryKind.FINANCING_APPLIED)
    total_costs = total_spread_cost + total_slippage_cost + total_commission_cost

    # Win/loss counts: replay every fill through the SAME production
    # accounting function, classifying each fill by whether it moved
    # realized P&L up (win), down (loss), or left it unchanged (pure open,
    # neither -- not counted in either bucket).
    replay_position = flat_position(spec.instrument.symbol, contract_multiplier=spec.instrument.contract_multiplier)
    winning_fill_count = losing_fill_count = 0
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.FILL:
            continue
        fill = Fill.from_json_dict(entry.payload)
        realized_before = replay_position.realized_pnl
        replay_position = apply_fill_to_position(replay_position, fill, event_time=fill.execution_time)
        realized_delta = replay_position.realized_pnl - realized_before
        if realized_delta > 0.0:
            winning_fill_count += 1
        elif realized_delta < 0.0:
            losing_fill_count += 1

    snapshots = [PortfolioState.from_json_dict(e.payload) for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT]
    maximum_drawdown_fraction = max((s.drawdown_fraction for s in snapshots), default=0.0)
    if snapshots:
        last = snapshots[-1]
        final_cash, final_equity, peak_equity = last.cash, last.equity, last.peak_equity
        gross_exposure, net_exposure = last.gross_exposure, last.net_exposure
        realized_pnl, unrealized_pnl = last.realized_pnl, last.unrealized_pnl
        turnover = last.turnover
        final_drawdown_fraction = last.drawdown_fraction
        position = last.positions.get(spec.instrument.symbol)
        position_summaries = (PositionSummary(instrument=spec.instrument.symbol, final_signed_quantity=(0.0 if position is None else position.signed_quantity), final_average_entry_price=(None if position is None else position.average_entry_price)),)
    else:
        final_cash = spec.starting_cash
        final_equity = peak_equity = spec.starting_cash
        gross_exposure = net_exposure = realized_pnl = unrealized_pnl = turnover = final_drawdown_fraction = 0.0
        position_summaries = (PositionSummary(instrument=spec.instrument.symbol, final_signed_quantity=0.0, final_average_entry_price=None),)
    gross_pnl = realized_pnl + unrealized_pnl
    net_pnl = gross_pnl - total_costs + total_financing

    total_risk_checks = 0
    failed_risk_checks = 0
    stale_data_incidents = 0
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.RISK_DECISION:
            continue
        results = as_json_list(entry.payload.get("results") or [], field_name="results")
        for result in results:
            result_dict = as_json_dict(result, field_name="results[]")
            total_risk_checks += 1
            if not bool(result_dict.get("passed")):
                failed_risk_checks += 1
                if _STALE_DATA_CHECK_SUBSTRING in str(result_dict.get("check_identity", "")):
                    stale_data_incidents += 1

    halt_entries = [e for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
    halt_transitions = [KillSwitchTransitionEvent.from_json_dict(e.payload).to_json_dict() for e in halt_entries]
    final_kill_switch_state = str(halt_transitions[-1]["to_state"]) if halt_transitions else None

    shadow_entries = [e for e in ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]
    observations_with_fill = sum(1 for e in shadow_entries if e.payload.get("hypothetical_fill_id") is not None)
    total_counterfactual_pnl = sum(float(str(e.payload["counterfactual_realized_pnl_delta"])) for e in shadow_entries if e.payload.get("counterfactual_realized_pnl_delta") is not None)

    verification_summary = (
        VerificationSummary(was_run=False, is_ready=None, critical_count=0, error_count=0, warning_count=0)
        if verification_report is None
        else VerificationSummary(was_run=True, is_ready=verification_report.is_ready, critical_count=len(verification_report.criticals), error_count=len(verification_report.errors), warning_count=len(verification_report.warnings))
    )

    return PaperSessionReport(
        schema_version=PAPER_SESSION_REPORT_SCHEMA_VERSION,
        session=SessionSummary(session_id=manifest.paper_session_id, session_mode=spec.session_mode.value, instrument=spec.instrument.symbol, manifest_stage=manifest.stage.value, event_count=event_count, starting_cash=spec.starting_cash),
        decisions=StrategyDecisionSummary(decision_count=decision_count, abstention_count=abstention_count),
        orders=OrderSummary(order_count=len(orders), rejected_count=rejected_count, filled_count=filled_count, cancelled_count=cancelled_count, expired_count=expired_count, working_count=working_count),
        fills=FillSummary(fill_count=fill_count, partial_fill_count=partial_fill_count, final_fill_count=final_fill_count, total_filled_quantity=total_filled_quantity, total_gross_notional=total_gross_notional, winning_fill_count=winning_fill_count, losing_fill_count=losing_fill_count),
        execution_quality=ExecutionQualitySummary(average_execution_delay_seconds=average_execution_delay_seconds, order_fill_rate=order_fill_rate, quantity_fill_rate=quantity_fill_rate),
        costs=CostSummary(total_spread_cost=total_spread_cost, total_slippage_cost=total_slippage_cost, total_commission_cost=total_commission_cost, total_financing=total_financing, total_costs=total_costs),
        positions=position_summaries,
        account_equity=AccountEquitySummary(starting_cash=spec.starting_cash, final_cash=final_cash, final_equity=final_equity, peak_equity=peak_equity, gross_exposure=gross_exposure, net_exposure=net_exposure, realized_pnl=realized_pnl, unrealized_pnl=unrealized_pnl, gross_pnl=gross_pnl, net_pnl=net_pnl, turnover=turnover),
        drawdown=DrawdownSummary(peak_equity=peak_equity, final_equity=final_equity, maximum_drawdown_fraction=maximum_drawdown_fraction, final_drawdown_fraction=final_drawdown_fraction),
        risk_events=RiskEventSummary(total_risk_checks=total_risk_checks, failed_risk_checks=failed_risk_checks, stale_data_incidents=stale_data_incidents, halts_triggered=len(halt_entries)),
        rejections=RejectionSummary(rejection_count=rejected_count, rejections_by_reason=tuple(sorted(rejections_by_reason.items()))),
        halts=HaltSummary(halt_count=len(halt_entries), final_kill_switch_state=final_kill_switch_state, transitions=tuple(halt_transitions)),
        reconciliation=ReconciliationSummary(is_reconciled=reconciliation_report.is_reconciled, total_checks=len(reconciliation_report.checks), failed_check_identities=tuple(c.check_identity for c in reconciliation_report.checks if not c.passed)),
        shadow=ShadowObservationSummary(observation_count=len(shadow_entries), observations_with_hypothetical_fill_count=observations_with_fill, total_counterfactual_realized_pnl=total_counterfactual_pnl),
        verification=verification_summary, disclaimer=DIAGNOSTIC_DISCLAIMER, generated_at=format_utc_timestamp(utc_now()),
    )


# --------------------------------------------------------------------------
# Section 28 -- backtest-comparison diagnostic
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BacktestComparisonMetrics:
    """Caller-supplied (see module docstring): the CALLER extracts these
    from whatever backtest report format it already has on hand."""

    decision_count: int
    order_count: int
    gross_return: float
    net_return: float
    total_costs: float
    turnover: float
    max_drawdown_fraction: float
    rejected_order_count: int
    abstention_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "decision_count": self.decision_count, "order_count": self.order_count, "gross_return": self.gross_return, "net_return": self.net_return,
            "total_costs": self.total_costs, "turnover": self.turnover, "max_drawdown_fraction": self.max_drawdown_fraction,
            "rejected_order_count": self.rejected_order_count, "abstention_count": self.abstention_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestComparisonMetrics:
        return cls(
            decision_count=int(str(raw["decision_count"])), order_count=int(str(raw["order_count"])), gross_return=float(str(raw["gross_return"])),
            net_return=float(str(raw["net_return"])), total_costs=float(str(raw["total_costs"])), turnover=float(str(raw["turnover"])),
            max_drawdown_fraction=float(str(raw["max_drawdown_fraction"])), rejected_order_count=int(str(raw["rejected_order_count"])),
            abstention_count=int(str(raw["abstention_count"])),
        )


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric_name: str
    backtest_value: float
    paper_value: float
    absolute_difference: float
    matches: bool
    classification: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name, "backtest_value": self.backtest_value, "paper_value": self.paper_value,
            "absolute_difference": self.absolute_difference, "matches": self.matches, "classification": self.classification,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MetricComparison:
        return cls(
            metric_name=str(raw["metric_name"]), backtest_value=float(str(raw["backtest_value"])), paper_value=float(str(raw["paper_value"])),
            absolute_difference=float(str(raw["absolute_difference"])), matches=bool(raw["matches"]), classification=str(raw["classification"]),
        )


@dataclass(frozen=True, slots=True)
class BacktestComparisonReport:
    schema_version: int
    session_id: str
    source_backtest_id: str
    comparisons: tuple[MetricComparison, ...]
    disclaimer: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "session_id": self.session_id, "source_backtest_id": self.source_backtest_id,
            "comparisons": [c.to_json_dict() for c in self.comparisons], "disclaimer": self.disclaimer, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestComparisonReport:
        return cls(
            schema_version=int(str(raw["schema_version"])), session_id=str(raw["session_id"]), source_backtest_id=str(raw["source_backtest_id"]),
            comparisons=tuple(MetricComparison.from_json_dict(as_json_dict(c, field_name="comparisons[]")) for c in as_json_list(raw.get("comparisons") or [], field_name="comparisons")),
            disclaimer=str(raw["disclaimer"]), generated_at=str(raw["generated_at"]),
        )


# (metric_name, backtest_value, paper_value, tolerance, exact_int_comparison, classification_if_differs)
def _metric_rows(backtest: BacktestComparisonMetrics, paper: PaperSessionReport, *, tolerance: float) -> list[tuple[str, float, float, float, str]]:
    return [
        ("decision_count", float(backtest.decision_count), float(paper.decisions.decision_count), 0.0, "unexpected_decision_mismatch"),
        ("order_count", float(backtest.order_count), float(paper.orders.order_count), 0.0, "unexpected_decision_mismatch"),
        ("abstention_count", float(backtest.abstention_count), float(paper.decisions.abstention_count), 0.0, "unexpected_decision_mismatch"),
        ("rejected_order_count", float(backtest.rejected_order_count), float(paper.orders.rejected_count), 0.0, "expected_due_to_latency"),
        ("gross_return", backtest.gross_return, (paper.account_equity.gross_pnl / paper.account_equity.starting_cash if paper.account_equity.starting_cash else 0.0), tolerance, "expected_due_to_spread"),
        ("net_return", backtest.net_return, (paper.account_equity.net_pnl / paper.account_equity.starting_cash if paper.account_equity.starting_cash else 0.0), tolerance, "expected_due_to_spread"),
        ("total_costs", backtest.total_costs, paper.costs.total_costs, tolerance, "expected_due_to_spread"),
        ("turnover", backtest.turnover, paper.account_equity.turnover, tolerance, "expected_due_to_partial_fills"),
        ("max_drawdown_fraction", backtest.max_drawdown_fraction, paper.drawdown.maximum_drawdown_fraction, tolerance, "expected_due_to_spread"),
    ]


def compare_paper_to_backtest(
    backtest: BacktestComparisonMetrics, paper: PaperSessionReport, *, source_backtest_id: str, tolerance: float = 1e-6,
) -> BacktestComparisonReport:
    """Section 28: diagnostic only, never a promotion decision. A mismatch
    is classified via a fixed, documented per-metric heuristic (see module
    docstring) -- any `unexpected_*` classification is a signal for manual
    investigation, not an automated verdict of a defect."""
    comparisons = []
    for metric_name, backtest_value, paper_value, metric_tolerance, classification_if_differs in _metric_rows(backtest, paper, tolerance=tolerance):
        difference = abs(backtest_value - paper_value)
        matches = difference <= metric_tolerance
        comparisons.append(MetricComparison(metric_name=metric_name, backtest_value=backtest_value, paper_value=paper_value, absolute_difference=difference, matches=matches, classification=("matches" if matches else classification_if_differs)))
    return BacktestComparisonReport(
        schema_version=BACKTEST_COMPARISON_REPORT_SCHEMA_VERSION, session_id=paper.session.session_id, source_backtest_id=source_backtest_id,
        comparisons=tuple(comparisons), disclaimer=DIAGNOSTIC_DISCLAIMER, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "BACKTEST_COMPARISON_REPORT_SCHEMA_VERSION",
    "DIAGNOSTIC_DISCLAIMER",
    "PAPER_SESSION_REPORT_SCHEMA_VERSION",
    "AccountEquitySummary",
    "BacktestComparisonMetrics",
    "BacktestComparisonReport",
    "CostSummary",
    "DrawdownSummary",
    "ExecutionQualitySummary",
    "FillSummary",
    "HaltSummary",
    "MetricComparison",
    "OrderSummary",
    "PaperSessionReport",
    "PositionSummary",
    "ReconciliationSummary",
    "RejectionSummary",
    "RiskEventSummary",
    "SessionSummary",
    "ShadowObservationSummary",
    "StrategyDecisionSummary",
    "VerificationSummary",
    "build_paper_session_report",
    "compare_paper_to_backtest",
]
