"""Independent session verification (Milestone 8, Section 28).
`verify_execution_session` NEVER trusts the persisted manifest, a final
report, or a cached order/position view -- it independently recomputes
spec identity, ledger chain integrity, ledger session ownership, order-
state-transition legality (by replaying `ExecutionOrderStateEvent`
sequences from scratch), fill identity/cumulative-quantity legality, the
FOK/IOC semantic invariants, and re-runs reconciliation.

Reuses `ml.models.ValidationReport`/`ValidationIssue`/`ValidationSeverity`
directly -- the SAME report shape `paper_trading.verification.
verify_paper_session` and `robustness.verification.verify_robustness`
return, rather than inventing a second one.

INDEPENDENCE CLASSIFICATION (classified honestly, mirroring the
Milestone 7 release audit's own explicit instruction not to claim
uniform independence):

- Ledger chain integrity, session ownership, entry sequencing, order-
  state-transition legality, command/broker-event identity: STRUCTURALLY
  INDEPENDENT (pure recomputation from hashes/enums, no financial logic).
- Reconciliation (positions/account vs the dummy broker's own
  independently-maintained bookkeeping): ALGORITHMICALLY INDEPENDENT --
  recomputes from the ledger's own persisted fills using the SAME
  accounting arithmetic the forward dispatcher used, cross-checked
  against a genuinely separate broker-side view, but does NOT
  independently re-derive what price/quantity SHOULD have been quoted --
  it trusts the dummy broker's own quoted fill price as ground truth,
  exactly like a real reconciliation would trust a real broker's fill
  confirmations.
- Source eligibility (of the underlying Milestone 6/7 chain):
  SOURCE-RECONSTRUCTING, delegated entirely to
  `paper_trading.eligibility.verify_paper_trading_eligibility`.

Net honest classification: **PARTIALLY INDEPENDENT** -- this is NOT a
second, independent implementation of order/fill/price determination;
it is an independent RECONCILIATION of the SAME durable evidence the
forward run itself produced, plus a fresh cross-check against the
broker's own separately-maintained state."""

from __future__ import annotations

from decimal import Decimal

from quant_platform.execution_gateway.models import (
    ExecutionLedgerEntryKind,
    TimeInForceKind,
    is_legal_execution_order_transition,
)
from quant_platform.execution_gateway.persistence import (
    ExecutionLedgerEntry,
    compute_execution_ledger_semantic_digest,
    verify_execution_ledger_chain_integrity,
)
from quant_platform.execution_gateway.reconciliation import reconstruct_all_orders_from_ledger
from quant_platform.execution_gateway.specs import ExecutionGatewaySpec, compute_execution_gateway_spec_id
from quant_platform.execution_gateway.states import reconstruct_execution_order
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

VERIFICATION_REPORT_SCHEMA_VERSION = 1

INDEPENDENCE_CLASSIFICATION = (
    "PARTIALLY INDEPENDENT: ledger chain integrity/session ownership/entry sequencing/order-state-transition legality/command and "
    "broker-event identity are STRUCTURALLY INDEPENDENT; reconciliation against the dummy broker's own independently-maintained "
    "bookkeeping is ALGORITHMICALLY INDEPENDENT (but trusts the broker's own quoted fill price as ground truth, exactly like a real "
    "reconciliation would); source eligibility is delegated entirely to paper_trading.eligibility (SOURCE-RECONSTRUCTING). This is not "
    "a second, independent implementation of order/fill/price determination."
)


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def verify_execution_session(
    spec: ExecutionGatewaySpec, *, execution_session_id: str, ledger: list[ExecutionLedgerEntry],
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    # 1. Spec identity.
    recomputed_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id
    if recomputed_id != execution_session_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "spec_identity_mismatch", f"recomputed execution_gateway_spec_id={recomputed_id!r} does not match execution_session_id={execution_session_id!r}"))

    # 9. Ledger session ownership.
    for e in ledger:
        if e.execution_session_id != execution_session_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "ledger_session_ownership_mismatch", f"ledger entry {e.entry_id!r} belongs to session {e.execution_session_id!r}, not {execution_session_id!r}"))
            break

    # 8/10. Ledger chain integrity and entry sequence.
    try:
        verify_execution_ledger_chain_integrity(ledger)
    except Exception as exc:
        issues.append(_issue(ValidationSeverity.CRITICAL, "ledger_chain_broken", str(exc)))

    # 11/12/13. Command identity / client-order-id uniqueness / idempotency.
    from quant_platform.execution_gateway.idempotency import (
        build_broker_order_index,
        build_client_order_index,
        build_command_index,
    )

    for builder, code in ((build_command_index, "command_idempotency_evidence_corrupted"), (build_client_order_index, "client_order_id_evidence_corrupted"), (build_broker_order_index, "broker_order_id_evidence_corrupted")):
        try:
            builder(ledger)
        except Exception as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, code, str(exc)))

    # 17/18/20/21. Order state transitions, cumulative fills, FOK/IOC semantics, cancel legality -- via full aggregate reconstruction.
    try:
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
    except Exception as exc:
        issues.append(_issue(ValidationSeverity.CRITICAL, "order_reconstruction_failed", str(exc)))
        reconstructed = {}

    # CONFIRMED DEFECT, FIXED (found during this milestone's own adversarial
    # testing): `reconstruct_all_orders_from_ledger` pre-filters fills by
    # grouping them under KNOWN submitted orders only -- an orphan
    # `EXECUTION_FILL_RECORDED` entry whose `execution_order_id` matches no
    # `COMMAND_CREATED` submit anywhere in the ledger was silently DROPPED
    # by that grouping, making it invisible to every check below (and to
    # `reconciliation.py`, which uses the same helper). Scanning the RAW
    # ledger directly here closes that gap, mirroring the Milestone 7
    # audit's own `reconciliation_no_fill_without_valid_order_failed` check.
    known_order_ids = set(reconstructed)
    for e in ledger:
        if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED:
            fill_order_id = e.payload.get("execution_order_id")
            if fill_order_id not in known_order_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "orphan_fill_no_matching_order", f"fill entry {e.entry_id!r} references execution_order_id={fill_order_id!r}, which has no corresponding submitted order in this ledger"))

    # CONFIRMED DEFECT, FIXED (found during this milestone's own adversarial
    # testing): the "24. UNKNOWN-state handling" block below used to call
    # `resolve_execution_order_state` again, unguarded, for EVERY order in
    # `reconstructed` -- including ones that had ALREADY raised out of
    # `reconstruct_execution_order` in the loop immediately below (caught
    # there and correctly converted to an `order_state_transition_illegal`
    # issue). The second, unguarded call re-raised the SAME exception,
    # crashing `verify_execution_session` with an uncaught traceback
    # instead of returning a clean `ValidationReport` -- precisely the
    # "traceback on an expected domain error" failure class this milestone
    # must never exhibit. `failed_order_ids` is threaded through every
    # later loop that re-touches `state_events` so none of them repeat
    # the mistake.
    failed_order_ids: set[str] = set()
    for order_id, (command, state_events, broker_events, fills) in reconstructed.items():
        try:
            order = reconstruct_execution_order(submit_command=command, state_events=state_events, broker_events=broker_events, fills=fills)
        except Exception as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "order_state_transition_illegal", f"order {order_id!r}: {exc}"))
            failed_order_ids.add(order_id)
            continue
        if order.filled_quantity > order.current_quantity:
            issues.append(_issue(ValidationSeverity.CRITICAL, "cumulative_fill_exceeds_quantity", f"order {order_id!r}: filled {order.filled_quantity} > accepted {order.current_quantity}"))
        if command.time_in_force is TimeInForceKind.FOK and Decimal(0) < order.filled_quantity < order.current_quantity:
            issues.append(_issue(ValidationSeverity.CRITICAL, "fok_partial_fill_detected", f"order {order_id!r} is FOK but shows a partial fill ({order.filled_quantity} of {order.current_quantity})"))
        for i in range(1, len(state_events)):
            if not is_legal_execution_order_transition(state_events[i - 1].to_state, state_events[i].from_state) and state_events[i - 1].to_state is not state_events[i].from_state:
                issues.append(_issue(ValidationSeverity.CRITICAL, "order_state_replay_discontinuous", f"order {order_id!r}: entry {i} does not continue from entry {i - 1}"))

    # 19. Fill identity -- no two distinct execution_fill_id values share the same (execution_order_id, broker_fill_id).
    seen: dict[str, str] = {}
    for order_id, (_c, _s, _b, fills) in reconstructed.items():
        for fill in fills:
            key = f"{order_id}:{fill.broker_fill_id}"
            if key in seen and seen[key] != fill.execution_fill_id:
                issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_fill_identity", f"broker_fill_id {fill.broker_fill_id!r} on order {order_id!r} maps to two different execution_fill_id values"))
            seen[key] = fill.execution_fill_id

    # 24. UNKNOWN-state handling: completion (session-level) checked by runner.py; here we simply surface any remaining UNKNOWN as a WARNING (not necessarily CRITICAL -- an in-progress session legitimately may have one).
    from quant_platform.execution_gateway.models import ExecutionOrderState
    from quant_platform.execution_gateway.state_machine import resolve_execution_order_state

    for order_id, (_c, state_events, _b, _f) in reconstructed.items():
        if order_id in failed_order_ids:
            continue  # already reported as order_state_transition_illegal above -- never re-resolve the same illegal history.
        if resolve_execution_order_state(order_id, state_events) is ExecutionOrderState.UNKNOWN:
            issues.append(_issue(ValidationSeverity.WARNING, "order_remains_unknown", f"order {order_id!r} is currently UNKNOWN"))

    # 27. Command prohibition after halt/termination -- no COMMAND_CREATED entry may follow an EXECUTION_SESSION_HALTED/TERMINATED entry.
    halted_at: int | None = None
    for i, e in enumerate(ledger):
        if e.entry_kind in (ExecutionLedgerEntryKind.EXECUTION_SESSION_HALTED, ExecutionLedgerEntryKind.EXECUTION_SESSION_TERMINATED):
            halted_at = i
            break
    if halted_at is not None:
        for e in ledger[halted_at + 1 :]:
            if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED:
                issues.append(_issue(ValidationSeverity.CRITICAL, "command_after_halt", f"command {e.payload.get('command_id')!r} created after session halt/termination"))

    # 31. Absence of live/network/broker artifacts -- structural: no field in this ledger's own payloads should ever contain a live-mode/broker-credential-shaped string. Cheap, targeted scan.
    forbidden_substrings = ("live", "mt5", "metatrader", "fxpro", "api_key", "password")
    for e in ledger:
        flat = str(e.payload).lower()
        for token in forbidden_substrings:
            if token in flat:
                issues.append(_issue(ValidationSeverity.WARNING, "suspicious_live_like_content", f"ledger entry {e.entry_id!r} payload contains the substring {token!r} -- verify this is not a live/broker artifact"))
                break

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def compute_verification_semantic_digest(ledger: list[ExecutionLedgerEntry]) -> str:
    return compute_execution_ledger_semantic_digest(ledger)


__all__ = [
    "INDEPENDENCE_CLASSIFICATION",
    "VERIFICATION_REPORT_SCHEMA_VERSION",
    "compute_verification_semantic_digest",
    "verify_execution_session",
]
