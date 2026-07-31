"""Portfolio-risk dispatch gate (Milestone 9 Phase 4). This is the ONE
integration point between `quant_platform.execution_gateway` (Milestone
8) and `quant_platform.portfolio_risk` (Milestone 9): `runner.py` calls
into this module, never the other way around, and `portfolio_risk`
itself still never imports `execution_gateway` -- the dependency
direction documented since Phase 1 (`docs/portfolio_risk_architecture.md`
"Package architecture and dependency direction") is realized here for
the first time, exactly as planned.

PRIMARY GUARANTEE: no `ExecutionIntent` may reach `dispatcher.
dispatch_command` without first passing through `authorize_portfolio_
risk_dispatch` (evaluate + issue) and `reserve_portfolio_risk_dispatch`
(reserve) -- both fail closed, raising `ExecutionPortfolioRiskAuthorizationError`
on any refusal, never a silent pass-through. `runner.py`'s own
`_run_intents_and_events` is the only call site that wires this
unconditionally into the dispatch path.

FLOW THIS MODULE IMPLEMENTS (Phase 4's own required architecture):

    ExecutionIntent -> [authorize_portfolio_risk_dispatch] -> RiskAuthorization
                     -> [reserve_portfolio_risk_dispatch]   -> RESERVED
                     -> dispatcher.dispatch_command (runner.py's own call)
                     -> [consume_portfolio_risk_dispatch]   -> CONSUMED  (only on COMMAND_DISPATCH_SUCCEEDED)

CONSUMPTION IS TIED TO `COMMAND_DISPATCH_SUCCEEDED` ONLY: a capability
rejection (`COMMAND_REJECTED`), an ambiguous adapter exception
(`COMMAND_MARKED_UNKNOWN`), or a synchronous broker refusal
(`COMMAND_DISPATCH_REJECTED`) all leave the authorization RESERVED,
never consumed and never auto-invalidated -- `recover_portfolio_risk_
dispatch_gate` (below) is what later resolves that ambiguity, using
BOTH packages' own durable evidence, never a guess.

`consumption_identity` is always `intent.execution_intent_id` -- already
a deterministic, unique-per-intent, content-addressed id; reusing it
(rather than minting a third id) makes an exact retry of the SAME
intent's dispatch attempt idempotent by construction (Phase 3's own
`validate_authorization_use` semantics), and gives `recover_portfolio_
risk_dispatch_gate` a direct, unambiguous way to find the corresponding
execution-gateway order for any RESERVED-but-unresolved authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import (
    ExecutionPortfolioRiskAuthorizationError,
    PortfolioRiskError,
    PortfolioRiskLockError,
    RiskAuthorizationReuseError,
)
from quant_platform.execution_gateway.commands import SubmitOrderCommand
from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind
from quant_platform.execution_gateway.paper_bridge import ExecutionIntent
from quant_platform.execution_gateway.persistence import ExecutionLedgerEntry
from quant_platform.execution_gateway.specs import ExecutionGatewaySpec
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    resolve_execution_order_state,
)
from quant_platform.execution_gateway.states import compute_execution_order_id
from quant_platform.execution_gateway.verification import verify_execution_session
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp
from quant_platform.portfolio_risk.authorization import RiskAuthorization
from quant_platform.portfolio_risk.decisions import create_risk_evaluation_request
from quant_platform.portfolio_risk.evaluator import evaluate_risk
from quant_platform.portfolio_risk.idempotency import (
    build_authorization_payload_index,
    build_authorization_status_index,
    build_execution_intent_index,
)
from quant_platform.portfolio_risk.issuance import issue_risk_authorization
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore, RiskLedgerEntryKind
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    invalidate_authorization,
    record_authorization_issuance,
    record_risk_decision,
    record_risk_evaluation_request,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import RiskAuthorizationStatus, RiskDecisionKind
from quant_platform.portfolio_risk.recovery import recover_portfolio_risk_session
from quant_platform.portfolio_risk.snapshots import PortfolioSnapshot, PriceSnapshot
from quant_platform.portfolio_risk.specs import PortfolioRiskSpec, compute_portfolio_risk_spec_id
from quant_platform.portfolio_risk.verification import verify_portfolio_risk_session

__all__ = [
    "PortfolioRiskGatewayContext",
    "authorize_portfolio_risk_dispatch",
    "consume_portfolio_risk_dispatch",
    "recover_portfolio_risk_dispatch_gate",
    "reserve_portfolio_risk_dispatch",
    "verify_execution_portfolio_risk_integration",
]

INTEGRATION_VERIFICATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PortfolioRiskGatewayContext:
    """Every value the gate needs beyond what `ExecutionIntent` already
    carries. `portfolio_snapshot`/`price_snapshot`/`risk_spec` are FIXED
    for the lifetime of one `run_execution_session` call -- Phase 2's own
    `evaluate_risk` is stateless and caller-supplied-everything by
    design (no portfolio-risk evaluator rewrite in this phase), so this
    context does not attempt to evolve the snapshot as fills accrue
    mid-session; see the architecture doc's Known Limitations for the
    explicit statement of this boundary. `price_snapshot.instrument_id`
    must match every intent's own `instrument_id` -- safe to assume
    because `ExecutionGatewaySpec` already scopes one whole execution
    session to exactly one instrument (Milestone 8's own existing
    invariant, Check 8 in `paper_bridge.execution_intent_from_paper_
    order`)."""

    store: PortfolioRiskLedgerStore
    portfolio_id: str
    portfolio_snapshot: PortfolioSnapshot
    price_snapshot: PriceSnapshot
    risk_spec: PortfolioRiskSpec
    portfolio_halted: bool
    consecutive_losses: int


def _count_entries(store: PortfolioRiskLedgerStore, portfolio_id: str, kind: RiskLedgerEntryKind) -> int:
    return sum(1 for e in store.read_events(portfolio_id) if e.entry_kind is kind)


_MAX_APPEND_RACE_RETRIES = 20
"""Bound on `authorize_portfolio_risk_dispatch`'s own optimistic-
concurrency retry loop -- see its docstring. Mirrors `portfolio_risk.
lifecycle._MAX_APPEND_RACE_RETRIES`'s identical rationale and value."""


def authorize_portfolio_risk_dispatch(*, intent: ExecutionIntent, context: PortfolioRiskGatewayContext, event_time: datetime) -> RiskAuthorization:
    """Evaluates portfolio risk for `intent` and, only if `APPROVED`,
    issues a `RiskAuthorization` bound to it. Every request/decision/
    issuance is durably recorded in `context.store` regardless of
    outcome -- a DENIED/HALTED decision is just as durably auditable as
    an APPROVED one, never silently dropped. Raises `ExecutionPortfolioRiskAuthorizationError`
    (chaining the underlying `portfolio_risk` exception) on DENIED/HALTED
    or any identity/binding failure -- never returns a usable value on
    refusal.

    RETRIES on a losing sequence-slot race (bounded, `_MAX_APPEND_RACE_
    RETRIES`): this function makes THREE separate appends to the SAME
    shared per-portfolio ledger (evaluation request, decision, and --
    only when approved -- authorization issuance). Two concurrent calls
    for two genuinely DIFFERENT `ExecutionIntent`s can each pass
    `portfolio_risk_lock` for their OWN first append, then race a LATER
    append against each other, since the three-append sequence is not
    atomic as a whole -- unlike `portfolio_risk.lifecycle`'s own
    single-append reserve/consume transactions (already race-safe from
    Phase 3), nothing in Phase 3 protects a multi-append flow like this
    one, which only exists in Phase 4. A confirmed, reproduced defect
    (found via this phase's own adversarial concurrency testing): the
    losing call previously surfaced a bare, confusing `RiskAuthorization
    ReuseError` about a ledger sequence conflict -- true but misleading,
    since the ACTUAL cause was a transient append race between two
    unrelated intents, not a genuine authorization conflict. Fixed by
    retrying the WHOLE evaluate-and-record sequence with freshly
    recomputed sequence numbers -- safe because every step is a pure
    function of its inputs (no wall-clock, no randomness), so a retry
    simply produces new, correctly-sequenced, non-conflicting objects."""
    risk_policy_id = compute_portfolio_risk_spec_id(context.risk_spec).portfolio_risk_spec_id

    for _ in range(_MAX_APPEND_RACE_RETRIES):
        try:
            requested_sequence = _count_entries(context.store, context.portfolio_id, RiskLedgerEntryKind.RISK_EVALUATION_REQUESTED)
            request = create_risk_evaluation_request(
                execution_intent_id=intent.execution_intent_id, execution_session_id=intent.execution_session_id, portfolio_id=context.portfolio_id,
                strategy_id=intent.strategy_candidate_id, instrument_id=intent.instrument_id, side=intent.side, quantity=intent.quantity,
                portfolio_snapshot_id=context.portfolio_snapshot.snapshot_id, price_snapshot_id=context.price_snapshot.price_snapshot_id,
                risk_policy_id=risk_policy_id, reduce_only=intent.reduce_only, requested_sequence=requested_sequence, event_time=event_time,
            )
            record_risk_evaluation_request(context.store, request, event_time=event_time)

            decision_sequence = _count_entries(context.store, context.portfolio_id, RiskLedgerEntryKind.RISK_DECISION_RECORDED)
            try:
                outcome = evaluate_risk(
                    request=request, portfolio=context.portfolio_snapshot, price=context.price_snapshot, spec=context.risk_spec, evaluation_time=event_time,
                    portfolio_halted=context.portfolio_halted, consecutive_losses=context.consecutive_losses, contract_multiplier=intent.contract_multiplier,
                    decision_sequence=decision_sequence,
                )
            except PortfolioRiskLockError:
                raise  # transient lock contention -- the caller is expected to retry; never reclassified as a business-level denial.
            except PortfolioRiskError as exc:
                raise ExecutionPortfolioRiskAuthorizationError(
                    f"portfolio risk evaluation raised for execution_intent_id={intent.execution_intent_id!r}: {exc}"
                ) from exc

            record_risk_decision(context.store, outcome.decision, portfolio_id=context.portfolio_id, event_time=event_time)

            if outcome.decision.kind is not RiskDecisionKind.APPROVED:
                reasons = ", ".join(sorted(r.value for r in outcome.decision.denial_reasons))
                raise ExecutionPortfolioRiskAuthorizationError(
                    f"portfolio risk {outcome.decision.kind.value} for execution_intent_id={intent.execution_intent_id!r} "
                    f"(risk_decision_id={outcome.decision.risk_decision_id!r}): {reasons}"
                )

            authorization_sequence = _count_entries(context.store, context.portfolio_id, RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED)
            try:
                authorization = issue_risk_authorization(request=request, decision=outcome.decision, authorization_sequence=authorization_sequence, event_time=event_time)
            except PortfolioRiskLockError:
                raise
            except PortfolioRiskError as exc:
                raise ExecutionPortfolioRiskAuthorizationError(f"could not issue a risk authorization for execution_intent_id={intent.execution_intent_id!r}: {exc}") from exc
            record_authorization_issuance(context.store, authorization, event_time=event_time)
            return authorization
        except RiskAuthorizationReuseError:
            continue  # lost a ledger append race against an unrelated intent -- retry with fresh sequence numbers.
    raise ExecutionPortfolioRiskAuthorizationError(
        f"could not resolve a ledger append race authorizing execution_intent_id={intent.execution_intent_id!r} after {_MAX_APPEND_RACE_RETRIES} attempts"
    )


def reserve_portfolio_risk_dispatch(
    *, authorization: RiskAuthorization, intent: ExecutionIntent, context: PortfolioRiskGatewayContext, event_time: datetime, expiry_time: datetime | None = None,
) -> None:
    """Reserves `authorization` for `intent`'s exact dispatch attempt,
    immediately before `dispatcher.dispatch_command` is called. Fail
    closed: any GENUINE `portfolio_risk` rejection (binding mismatch,
    expired, status does not permit use, conflicting consumption) becomes
    `ExecutionPortfolioRiskAuthorizationError` -- dispatch must never
    proceed past this call on any exception. `PortfolioRiskLockError`
    (transient lock contention, a documented, expected, RETRYABLE
    infrastructure condition -- never a business-level denial) propagates
    UNCHANGED rather than being reclassified as a gate refusal; a caller
    must not treat it as "never retry this intent." `expiry_time` is optional
    and caller-supplied (Phase 3's own `validate_authorization_use`
    convention, unchanged here) -- `None` (the default) means no expiry
    is enforced; `runner.py`'s own call site does not pass one today, but
    a future caller with a real authorization-validity policy can."""
    risk_policy_id = compute_portfolio_risk_spec_id(context.risk_spec).portfolio_risk_spec_id
    try:
        reserve_authorization(
            context.store, authorization, execution_intent_id=intent.execution_intent_id, execution_session_id=intent.execution_session_id,
            portfolio_id=context.portfolio_id, portfolio_snapshot_id=context.portfolio_snapshot.snapshot_id,
            price_snapshot_id=context.price_snapshot.price_snapshot_id, risk_policy_id=risk_policy_id, quantity=authorization.evaluated_quantity,
            price=authorization.evaluated_price, consumption_identity=intent.execution_intent_id, evaluation_time=event_time, expiry_time=expiry_time,
        )
    except PortfolioRiskLockError:
        raise  # transient lock contention -- the caller is expected to retry; never reclassified as a business-level denial.
    except PortfolioRiskError as exc:
        raise ExecutionPortfolioRiskAuthorizationError(
            f"could not reserve risk authorization {authorization.risk_authorization_id!r} for execution_intent_id={intent.execution_intent_id!r}: {exc}"
        ) from exc


def consume_portfolio_risk_dispatch(
    *, authorization: RiskAuthorization, intent: ExecutionIntent, context: PortfolioRiskGatewayContext, event_time: datetime, expiry_time: datetime | None = None,
) -> None:
    """Consumes `authorization` after `dispatcher.dispatch_command`
    returned `COMMAND_DISPATCH_SUCCEEDED` for `intent` -- the caller
    (`runner.py`) is responsible for only calling this on that exact
    outcome; every other dispatch outcome must leave the authorization
    RESERVED (see this module's own docstring). `expiry_time` -- see
    `reserve_portfolio_risk_dispatch`'s identical parameter."""
    risk_policy_id = compute_portfolio_risk_spec_id(context.risk_spec).portfolio_risk_spec_id
    try:
        consume_authorization(
            context.store, authorization, execution_intent_id=intent.execution_intent_id, execution_session_id=intent.execution_session_id,
            portfolio_id=context.portfolio_id, portfolio_snapshot_id=context.portfolio_snapshot.snapshot_id,
            price_snapshot_id=context.price_snapshot.price_snapshot_id, risk_policy_id=risk_policy_id, quantity=authorization.evaluated_quantity,
            price=authorization.evaluated_price, consumption_identity=intent.execution_intent_id, evaluation_time=event_time, expiry_time=expiry_time,
        )
    except PortfolioRiskLockError:
        raise  # transient lock contention -- the caller is expected to retry; never reclassified as a business-level denial.
    except PortfolioRiskError as exc:
        raise ExecutionPortfolioRiskAuthorizationError(
            f"could not consume risk authorization {authorization.risk_authorization_id!r} for execution_intent_id={intent.execution_intent_id!r}: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Recovery -- cross-references portfolio_risk's own recovery classification
# against execution_gateway's own order-state evidence to resolve what
# Phase 3's own recovery alone cannot: whether a RESERVED-but-unresolved
# authorization corresponds to a dispatch that (a) actually succeeded
# (broker-confirmed), (b) definitely never produced economic exposure, or
# (c) is still genuinely ambiguous even after execution_gateway's OWN
# recovery has run. Neither package's own recovery function is modified --
# this module only COMBINES their already-durable evidence.
# --------------------------------------------------------------------------
_ORDER_CONFIRMED_DISPATCHED_STATES = frozenset({"dispatched", "acknowledged", "partially_filled", "filled"})
_ORDER_CONFIRMED_NOT_LIVE_STATES = frozenset({"created", "validated", "rejected", "cancelled", "expired"})


@dataclass(frozen=True, slots=True)
class PortfolioRiskRecoveryCrossReference:
    execution_intent_id: str
    risk_authorization_id: str
    portfolio_risk_action: str
    """The underlying `portfolio_risk.recovery.RecoveryAction.action`
    this cross-reference was derived from."""
    execution_order_state: str | None
    """The resolved `ExecutionOrderState.value` for the order this
    execution_intent_id dispatched (via `compute_execution_order_id` on
    its own `SubmitOrderCommand`), or `None` if no matching command was
    ever created (the crash window between reservation and the FIRST
    dispatch-transaction ledger write)."""
    resolution: str
    """One of: `consumed_now` (order confirmed dispatched -- the missed
    consume call is now durably completed), `invalidated_now` (order
    confirmed never went live -- released so the authorization does not
    stay blocked forever), `remains_blocked` (still genuinely ambiguous,
    untouched), `not_applicable` (this authorization was not `reserved_
    unresolved_blocked` at all)."""


def _submit_command_for_intent(ledger: list[ExecutionLedgerEntry], *, execution_intent_id: str) -> SubmitOrderCommand | None:
    for entry in ledger:
        if entry.entry_kind is not ExecutionLedgerEntryKind.COMMAND_CREATED or entry.payload.get("command_type") != "submit_order":
            continue
        command = SubmitOrderCommand.from_json_dict(entry.payload)
        if command.execution_intent_id == execution_intent_id:
            return command
    return None


def recover_portfolio_risk_dispatch_gate(
    *, context: PortfolioRiskGatewayContext, execution_session_id: str, execution_ledger: list[ExecutionLedgerEntry], recovery_time: datetime,
) -> list[PortfolioRiskRecoveryCrossReference]:
    """Runs `portfolio_risk.recovery.recover_portfolio_risk_session` for
    `context.portfolio_id`, then resolves every `reserved_unresolved_
    blocked` authorization it reports by cross-checking the corresponding
    execution-gateway order's OWN already-recovered state (execution_
    gateway's own `recover_unknown_orders` must already have run --
    `runner.py` calls both at the same `ADAPTER_INITIALIZED` stage,
    execution_gateway's own recovery first). Never blindly reuses a
    RESERVED authorization and never consumes twice -- an order still
    `UNKNOWN` even after execution_gateway's own recovery leaves the
    authorization untouched (`remains_blocked`)."""
    portfolio_actions = recover_portfolio_risk_session(portfolio_id=context.portfolio_id, store=context.store, recovery_time=recovery_time)
    payload_index = build_authorization_payload_index(context.store.read_events(context.portfolio_id))

    results: list[PortfolioRiskRecoveryCrossReference] = []
    for action in portfolio_actions:
        if action.action != "reserved_unresolved_blocked":
            continue
        payload = payload_index.get(action.authorization_id)
        if payload is None:
            continue  # structurally impossible (payload_index is seeded from ISSUED entries) -- defensive skip only.
        execution_intent_id = str(payload["execution_intent_id"])
        if str(payload.get("execution_session_id")) != execution_session_id:
            continue  # this authorization belongs to a different execution session -- not this call's concern.

        command = _submit_command_for_intent(execution_ledger, execution_intent_id=execution_intent_id)
        if command is None:
            results.append(PortfolioRiskRecoveryCrossReference(
                execution_intent_id=execution_intent_id, risk_authorization_id=action.authorization_id, portfolio_risk_action=action.action,
                execution_order_state=None, resolution="remains_blocked",
            ))
            continue

        execution_order_id = compute_execution_order_id(command)
        order_state_events = [
            e for e in _order_state_events(execution_ledger) if e.execution_order_id == execution_order_id
        ]
        state = resolve_execution_order_state(execution_order_id, order_state_events)

        authorization = RiskAuthorization.from_json_dict(payload)
        if state.value in _ORDER_CONFIRMED_DISPATCHED_STATES:
            consume_authorization(
                context.store, authorization, execution_intent_id=execution_intent_id, execution_session_id=execution_session_id,
                portfolio_id=context.portfolio_id, portfolio_snapshot_id=str(payload["portfolio_snapshot_id"]),
                price_snapshot_id=str(payload["price_snapshot_id"]), risk_policy_id=str(payload["risk_policy_id"]),
                quantity=authorization.evaluated_quantity, price=authorization.evaluated_price, consumption_identity=execution_intent_id,
                evaluation_time=recovery_time,
            )
            resolution = "consumed_now"
        elif state.value in _ORDER_CONFIRMED_NOT_LIVE_STATES:
            invalidate_authorization(
                context.store, authorization, reason_code="execution_confirmed_not_dispatched",
                detail=f"execution_gateway order {execution_order_id!r} resolved to {state.value!r} -- no economic exposure resulted",
                evaluation_time=recovery_time,
            )
            resolution = "invalidated_now"
        else:
            resolution = "remains_blocked"

        results.append(PortfolioRiskRecoveryCrossReference(
            execution_intent_id=execution_intent_id, risk_authorization_id=action.authorization_id, portfolio_risk_action=action.action,
            execution_order_state=state.value, resolution=resolution,
        ))
    return results


def _order_state_events(ledger: list[ExecutionLedgerEntry]) -> list[ExecutionOrderStateEvent]:
    return [ExecutionOrderStateEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION]


# --------------------------------------------------------------------------
# Cross-milestone verification (Milestone 9 Phase 4). Combines each
# package's own INDEPENDENT verification -- `execution_gateway.
# verification.verify_execution_session` and `portfolio_risk.
# verification.verify_portfolio_risk_session`, NEITHER modified nor
# re-implemented here -- with cross-ledger checks neither package alone
# can perform, since neither has visibility into the other's own ledger.
# --------------------------------------------------------------------------
def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


_RESOLVED_DISPATCH_KINDS = frozenset({
    ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED, ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED,
    ExecutionLedgerEntryKind.COMMAND_MARKED_UNKNOWN, ExecutionLedgerEntryKind.COMMAND_REJECTED,
})


def verify_execution_portfolio_risk_integration(
    *, spec: ExecutionGatewaySpec, execution_session_id: str, execution_ledger: list[ExecutionLedgerEntry], context: PortfolioRiskGatewayContext,
    verification_time: datetime,
) -> ValidationReport:
    """Verifies: authorization binding (every accepted `ExecutionIntent`
    has a matching `RiskAuthorization` bound to the correct
    `portfolio_id`, and every `PORTFOLIO_RISK_AUTHORIZATION_BOUND` entry
    references a genuinely issued one), consumption/dispatch ordering (an
    authorization is `CONSUMED` if and only if its intent's command
    resolved to `COMMAND_DISPATCH_SUCCEEDED` -- never one without the
    other), and single economic execution (folded in from each package's
    own already-independent single-use/idempotency verification, never
    re-implemented here). `record=False` is used for the portfolio-risk
    half -- exactly like `replay.py`'s own comparison utility -- so
    calling this function itself is side-effect-free and repeatable."""
    issues: list[ValidationIssue] = []

    execution_report = verify_execution_session(spec, execution_session_id=execution_session_id, ledger=execution_ledger)
    issues.extend(execution_report.issues)

    portfolio_risk_report = verify_portfolio_risk_session(portfolio_id=context.portfolio_id, store=context.store, verification_time=verification_time, record=False)
    issues.extend(portfolio_risk_report.issues)

    portfolio_risk_ledger = context.store.read_events(context.portfolio_id)
    payload_index = build_authorization_payload_index(portfolio_risk_ledger)
    status_index = build_authorization_status_index(portfolio_risk_ledger)
    intent_index = build_execution_intent_index(portfolio_risk_ledger)

    accepted_intent_ids = {str(e.payload.get("execution_intent_id")) for e in execution_ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED}
    bound_entries = {str(e.payload.get("execution_intent_id")): str(e.payload.get("risk_authorization_id")) for e in execution_ledger if e.entry_kind is ExecutionLedgerEntryKind.PORTFOLIO_RISK_AUTHORIZATION_BOUND}

    submit_commands_by_intent: dict[str, SubmitOrderCommand] = {}
    for entry in execution_ledger:
        if entry.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and entry.payload.get("command_type") == "submit_order":
            command = SubmitOrderCommand.from_json_dict(entry.payload)
            submit_commands_by_intent[command.execution_intent_id] = command

    dispatch_resolution_by_command_id: dict[str, ExecutionLedgerEntryKind] = {
        str(e.payload.get("command_id")): e.entry_kind for e in execution_ledger if e.entry_kind in _RESOLVED_DISPATCH_KINDS and e.payload.get("command_id") is not None
    }

    for execution_intent_id in accepted_intent_ids:
        authorization_id = intent_index.get(execution_intent_id)
        if authorization_id is None:
            # Either genuinely never authorized (a bug), or the intent was
            # rejected before issuance ever completed -- indistinguishable
            # from here alone, so only flag it if a command was ALSO
            # created for it (proof the gate must have been passed).
            if execution_intent_id in submit_commands_by_intent:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "dispatched_intent_without_risk_authorization",
                    f"execution_intent_id={execution_intent_id!r} has a dispatched command but no corresponding RiskAuthorization exists.",
                ))
            continue

        payload = payload_index.get(authorization_id)
        if payload is None or str(payload.get("portfolio_id")) != context.portfolio_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "authorization_binding_mismatch",
                f"execution_intent_id={execution_intent_id!r}'s authorization {authorization_id!r} is not bound to portfolio_id={context.portfolio_id!r}.",
            ))

        bound_id = bound_entries.get(execution_intent_id)
        if bound_id is not None and bound_id != authorization_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "execution_ledger_authorization_binding_mismatch",
                f"execution_intent_id={execution_intent_id!r}'s PORTFOLIO_RISK_AUTHORIZATION_BOUND entry references {bound_id!r}, "
                f"but the portfolio_risk ledger's own execution_intent_id index resolves it to {authorization_id!r}.",
            ))

        status = status_index.get(authorization_id)
        submit_command = submit_commands_by_intent.get(execution_intent_id)
        resolution = dispatch_resolution_by_command_id.get(submit_command.command_id) if submit_command is not None else None
        was_dispatched_successfully = resolution is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED
        was_consumed = status is RiskAuthorizationStatus.CONSUMED

        if was_consumed and not was_dispatched_successfully:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "consumed_authorization_without_successful_dispatch",
                f"execution_intent_id={execution_intent_id!r}'s authorization {authorization_id!r} is CONSUMED, but its command's dispatch "
                f"resolution is {resolution.value if resolution is not None else 'unresolved'!r}, not COMMAND_DISPATCH_SUCCEEDED.",
            ))
        if was_dispatched_successfully and not was_consumed:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "successful_dispatch_without_consumed_authorization",
                f"execution_intent_id={execution_intent_id!r}'s command dispatched successfully, but its authorization {authorization_id!r} "
                f"is {status.value if status is not None else 'unknown'!r}, not CONSUMED.",
            ))

    return ValidationReport(
        schema_version=INTEGRATION_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(verification_time)),
    )
