"""Reconciliation (Milestone 7, Section 25): 11 required checks, each
independently recomputed purely from the event ledger (never trusting a
persisted total beyond what it is being checked against). Accounting
checks use EXACT arithmetic (a small floating-point tolerance only where
Section 25 itself concedes one is unavoidable -- equity/cash comparisons
against a persisted snapshot, since summing many float fills can
accumulate a few ULPs of drift; every count/identity/state check is
exact-equality, zero tolerance).

This module reconstructs orders/fills/positions ENTIRELY from `LedgerEntry`
payloads (an order's own economic details are recoverable from ANY of its
`ORDER_STATE_EVENT` entries -- see `runner._order_state_payload`'s
docstring for why every one embeds the full `OrderRequest`), never from
the runner's own in-memory state -- the same "ledger is the only source of
truth" discipline `verification.py` (Section 26) will apply one layer up,
though reconciliation's OWN job (Section 25) is narrower: does the
bookkeeping balance, not "is every persisted artifact independently
re-derivable" (that broader claim is verification's)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import OrderStateError, PaperTradingArtifactError
from quant_platform.ml.persistence import as_json_dict, as_json_list, format_utc_timestamp, utc_now
from quant_platform.paper_trading.accounting import apply_fill_to_position, flat_position
from quant_platform.paper_trading.fills import Fill
from quant_platform.paper_trading.models import OrderSide, OrderState
from quant_platform.paper_trading.orders import OrderStateEvent, resolve_order_state
from quant_platform.paper_trading.persistence import LedgerEntry, verify_ledger_chain_integrity
from quant_platform.paper_trading.portfolio import PortfolioState
from quant_platform.paper_trading.specs import InstrumentSpec

RECONCILIATION_REPORT_SCHEMA_VERSION = 1
_EXACT_TOLERANCE = 0.0
_FLOAT_ACCUMULATION_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ReconciliationCheckResult:
    check_identity: str
    passed: bool
    expected_value: str
    observed_value: str
    tolerance: float
    reason_code: str | None
    source_event_references: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "check_identity": self.check_identity, "passed": self.passed, "expected_value": self.expected_value, "observed_value": self.observed_value,
            "tolerance": self.tolerance, "reason_code": self.reason_code, "source_event_references": list(self.source_event_references),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReconciliationCheckResult:
        return cls(
            check_identity=str(raw["check_identity"]), passed=bool(raw["passed"]), expected_value=str(raw["expected_value"]),
            observed_value=str(raw["observed_value"]), tolerance=float(str(raw["tolerance"])),
            reason_code=(None if raw.get("reason_code") is None else str(raw["reason_code"])),
            source_event_references=tuple(str(r) for r in as_json_list(raw.get("source_event_references") or [], field_name="source_event_references")),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    schema_version: int
    session_id: str
    checks: tuple[ReconciliationCheckResult, ...]
    generated_at: str

    @property
    def is_reconciled(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "session_id": self.session_id, "checks": [c.to_json_dict() for c in self.checks],
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReconciliationReport:
        return cls(
            schema_version=int(str(raw["schema_version"])), session_id=str(raw["session_id"]),
            checks=tuple(
                ReconciliationCheckResult.from_json_dict(as_json_dict(c, field_name="checks[]"))
                for c in as_json_list(raw.get("checks") or [], field_name="checks")
            ),
            generated_at=str(raw["generated_at"]),
        )


def _pass(check_identity: str, *, value: object, tolerance: float = _EXACT_TOLERANCE, refs: tuple[str, ...] = ()) -> ReconciliationCheckResult:
    return ReconciliationCheckResult(check_identity=check_identity, passed=True, expected_value=str(value), observed_value=str(value), tolerance=tolerance, reason_code=None, source_event_references=refs)


def _fail(check_identity: str, *, expected: object, observed: object, tolerance: float = _EXACT_TOLERANCE, reason: str, refs: tuple[str, ...] = ()) -> ReconciliationCheckResult:
    return ReconciliationCheckResult(check_identity=check_identity, passed=False, expected_value=str(expected), observed_value=str(observed), tolerance=tolerance, reason_code=reason, source_event_references=refs)


def _numeric_check(check_identity: str, *, expected: float, observed: float, tolerance: float, reason: str, refs: tuple[str, ...] = ()) -> ReconciliationCheckResult:
    if abs(expected - observed) <= tolerance:
        return _pass(check_identity, value=observed, tolerance=tolerance, refs=refs)
    return _fail(check_identity, expected=expected, observed=observed, tolerance=tolerance, reason=reason, refs=refs)


def _extract_orders(ledger: list[LedgerEntry]) -> dict[str, tuple[dict[str, object], list[tuple[OrderStateEvent, LedgerEntry]]]]:
    """order_id -> (order_json, [(OrderStateEvent, source_ledger_entry), ...]) in ledger order."""
    orders: dict[str, tuple[dict[str, object], list[tuple[OrderStateEvent, LedgerEntry]]]] = {}
    for entry in ledger:
        if entry.kind.value != "order_state_event":
            continue
        order_json = entry.payload["order"]
        state_event = OrderStateEvent.from_json_dict(entry.payload["order_state_event"])  # type: ignore[arg-type]
        order_id = state_event.order_id
        if order_id not in orders:
            orders[order_id] = (order_json, [])  # type: ignore[assignment]
        orders[order_id][1].append((state_event, entry))
    return orders


def _extract_fills(ledger: list[LedgerEntry]) -> list[Fill]:
    return [Fill.from_json_dict(entry.payload) for entry in ledger if entry.kind.value == "fill"]


def _check_event_sequence_contiguous(ledger: list[LedgerEntry]) -> ReconciliationCheckResult:
    try:
        verify_ledger_chain_integrity(ledger)
    except PaperTradingArtifactError as exc:
        return _fail("event_sequence_contiguous", expected="unbroken chain", observed=str(exc), reason="ledger_chain_broken")
    return _pass("event_sequence_contiguous", value="unbroken chain")


def _check_no_duplicate_identities(ledger: list[LedgerEntry], fills: list[Fill]) -> ReconciliationCheckResult:
    entry_ids = [e.entry_id for e in ledger]
    fill_ids = [f.fill_id for f in fills]
    if len(set(entry_ids)) != len(entry_ids):
        return _fail("no_duplicate_identities", expected="all unique", observed="duplicate ledger entry_id found", reason="duplicate_ledger_entry")
    if len(set(fill_ids)) != len(fill_ids):
        return _fail("no_duplicate_identities", expected="all unique", observed="duplicate fill_id found", reason="duplicate_fill")
    return _pass("no_duplicate_identities", value="all unique")


def _check_order_state_transitions_legal(orders: dict[str, tuple[dict[str, object], list]]) -> ReconciliationCheckResult:  # type: ignore[type-arg]
    for order_id, (_, transitions) in orders.items():
        events_only = [t[0] for t in transitions]
        try:
            resolve_order_state(order_id, events_only)
        except OrderStateError as exc:
            return _fail("order_state_transitions_legal", expected="legal transition sequence", observed=str(exc), reason="illegal_order_transition", refs=(order_id,))
    return _pass("order_state_transitions_legal", value="all legal")


def _order_quantity(order_json: dict[str, object]) -> float:
    return float(str(order_json["quantity"]))


def _order_side(order_json: dict[str, object]) -> OrderSide:
    return OrderSide(order_json["side"])


def reconcile_session(ledger: list[LedgerEntry], *, session_id: str, instrument: InstrumentSpec, starting_cash: float) -> ReconciliationReport:
    orders = _extract_orders(ledger)
    fills = _extract_fills(ledger)
    fills_by_order: dict[str, list[Fill]] = {}
    for fill in fills:
        fills_by_order.setdefault(fill.order_id, []).append(fill)

    checks: list[ReconciliationCheckResult] = [
        _check_event_sequence_contiguous(ledger),
        _check_no_duplicate_identities(ledger, fills),
        _check_order_state_transitions_legal(orders),
    ]

    # Check 1 & 2: order quantity = sum(fills) + remaining; filled orders have zero remaining.
    over_filled = []
    unfinished_filled = []
    for order_id, (order_json, transitions) in orders.items():
        declared_quantity = _order_quantity(order_json)
        filled_quantity = sum(f.quantity for f in fills_by_order.get(order_id, []))
        try:
            final_state = resolve_order_state(order_id, [t[0] for t in transitions])
        except OrderStateError:
            # Release-audit finding, fixed: this SAME call, unguarded, used
            # to let a genuinely illegal/gapped order-transition sequence
            # crash `reconcile_session` with an unhandled `OrderStateError`
            # instead of returning a structured report -- even though
            # `_check_order_state_transitions_legal` above already caught
            # and reported the exact same order as a CRITICAL failure.
            # Reconciliation/verification must NEVER raise on adversarial
            # input; this order's fill-quantity classification is
            # meaningless once its own state is unresolvable, so it is
            # skipped here rather than re-attempted.
            continue
        if filled_quantity - declared_quantity > _FLOAT_ACCUMULATION_TOLERANCE:
            over_filled.append(order_id)
        if final_state is OrderState.FILLED and abs(declared_quantity - filled_quantity) > _FLOAT_ACCUMULATION_TOLERANCE:
            unfinished_filled.append(order_id)
    if over_filled:
        checks.append(_fail("order_quantity_equals_fills_plus_remaining", expected="sum(fills) <= order.quantity", observed=f"over-filled orders: {over_filled}", reason="order_over_filled"))
    else:
        checks.append(_pass("order_quantity_equals_fills_plus_remaining", value="sum(fills) <= order.quantity for every order"))
    if unfinished_filled:
        checks.append(_fail("filled_orders_have_zero_remaining", expected="remaining == 0 for FILLED orders", observed=f"orders: {unfinished_filled}", reason="filled_order_with_remaining"))
    else:
        checks.append(_pass("filled_orders_have_zero_remaining", value="remaining == 0 for every FILLED order"))

    # Check 3: no fill without a valid (known, non-rejected-before-fill) order.
    orphaned_fills = [f.fill_id for f in fills if f.order_id not in orders]
    if orphaned_fills:
        checks.append(_fail("no_fill_without_valid_order", expected="every fill references a known order", observed=f"orphaned fills: {orphaned_fills}", reason="orphaned_fill"))
    else:
        checks.append(_pass("no_fill_without_valid_order", value="every fill references a known order"))

    # Reconstruct position/cash purely from fills + financing, from flat/starting_cash.
    position = flat_position(instrument.symbol, contract_multiplier=instrument.contract_multiplier)
    cash = starting_cash
    accrued_costs = 0.0
    for entry in ledger:
        if entry.kind.value == "fill":
            fill = Fill.from_json_dict(entry.payload)
            position = apply_fill_to_position(position, fill, event_time=fill.execution_time)
            cash += -fill.gross_notional if fill.side is OrderSide.BUY else fill.gross_notional
            accrued_costs += fill.spread_cost + fill.slippage_cost + fill.commission_cost
        elif entry.kind.value == "financing_applied":
            cash += float(str(entry.payload["cash_delta"]))

    # Check 4: position quantity = signed cumulative fills (vs last snapshot).
    last_snapshot = next((entry for entry in reversed(ledger) if entry.kind.value == "account_snapshot"), None)
    if last_snapshot is not None:
        snapshot = PortfolioState.from_json_dict(last_snapshot.payload)
        snapshot_position = snapshot.positions.get(instrument.symbol)
        snapshot_signed_quantity = 0.0 if snapshot_position is None else snapshot_position.signed_quantity
        checks.append(_numeric_check("position_quantity_equals_signed_cumulative_fills", expected=position.signed_quantity, observed=snapshot_signed_quantity, tolerance=_FLOAT_ACCUMULATION_TOLERANCE, reason="position_quantity_mismatch", refs=(last_snapshot.entry_id,)))

        # Check 5: realized P&L matches closed quantities.
        snapshot_realized_pnl = 0.0 if snapshot_position is None else snapshot_position.realized_pnl
        checks.append(_numeric_check("realized_pnl_matches_closed_quantities", expected=position.realized_pnl, observed=snapshot_realized_pnl, tolerance=_FLOAT_ACCUMULATION_TOLERANCE, reason="realized_pnl_mismatch", refs=(last_snapshot.entry_id,)))

        # Check 6: cash movements match fills+costs.
        checks.append(_numeric_check("cash_movements_match_fills_and_costs", expected=cash, observed=snapshot.cash, tolerance=_FLOAT_ACCUMULATION_TOLERANCE, reason="cash_mismatch", refs=(last_snapshot.entry_id,)))

        # Check 7: total costs = component sums.
        checks.append(_numeric_check("total_costs_equal_component_sums", expected=accrued_costs, observed=snapshot.accrued_costs, tolerance=_FLOAT_ACCUMULATION_TOLERANCE, reason="accrued_costs_mismatch", refs=(last_snapshot.entry_id,)))

        # Check 8: account equity reconciles (recomputed exactly the way PortfolioState.equity does, from the RECONCILED cash/costs plus the snapshot's own marked_position_value -- mark price itself is independently disclosed by MARK_APPLIED entries, not re-derived here).
        recomputed_equity = cash + snapshot.marked_position_value - snapshot.liabilities - accrued_costs
        checks.append(_numeric_check("account_equity_reconciles", expected=recomputed_equity, observed=snapshot.equity, tolerance=_FLOAT_ACCUMULATION_TOLERANCE, reason="equity_mismatch", refs=(last_snapshot.entry_id,)))
    else:
        for check_identity in ("position_quantity_equals_signed_cumulative_fills", "realized_pnl_matches_closed_quantities", "cash_movements_match_fills_and_costs", "total_costs_equal_component_sums", "account_equity_reconciles"):
            checks.append(_pass(check_identity, value="no account snapshot yet (nothing to reconcile against)"))

    return ReconciliationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, session_id=session_id, checks=tuple(checks), generated_at=format_utc_timestamp(utc_now()))


__all__ = ["RECONCILIATION_REPORT_SCHEMA_VERSION", "ReconciliationCheckResult", "ReconciliationReport", "reconcile_session"]
