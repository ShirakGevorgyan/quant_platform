"""Execution session runner (Milestone 8, Section 19/22). `run_execution_session`
is the ONE orchestrator that walks an `ExecutionSessionManifest` through
its full legal stage sequence -- creating (or transparently resuming) the
session named by `spec`'s own content-addressed identity, bridging source
paper orders into intents/commands, dispatching them (through the
centralized kill-switch gate, never around it), advancing the dummy
broker across the supplied market-event stream, reconciling, and only
then verifying completion.

COMPLETED is granted ONLY when every one of Section 19's gate conditions
holds -- no unresolved `UNKNOWN` order, no blocking/critical
reconciliation issue -- checked freshly every call, never inferred from
a persisted boolean.

EVERY new or resumed run MANDATORILY re-verifies source eligibility
(Section 6/23) -- this mirrors the Milestone 7 release audit's own
most severe finding (eligibility checked once at creation and never
again) and deliberately does NOT repeat that mistake here."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.exceptions import ExecutionPortfolioRiskAuthorizationError
from quant_platform.execution_gateway.adapter import ExecutionAdapter
from quant_platform.execution_gateway.commands import (
    CancelOrderCommand,
    ReplaceOrderCommand,
    SubmitOrderCommand,
    create_submit_order_command,
)
from quant_platform.execution_gateway.dispatcher import (
    _append,
    dispatch_command,
    process_broker_events,
)
from quant_platform.execution_gateway.kill_switch import (
    ExecutionKillSwitchTransitionEvent,
    authorize_dispatch,
    resolve_execution_kill_switch_state,
)
from quant_platform.execution_gateway.manifests import ExecutionSessionManifest, ExecutionSessionManifestStore
from quant_platform.execution_gateway.models import (
    ExecutionKillSwitchState,
    ExecutionLedgerEntryKind,
    ExecutionOrderState,
    ExecutionSessionStage,
)
from quant_platform.execution_gateway.paper_bridge import (
    PaperBridgeEnvironment,
    bind_portfolio_risk_authorization,
    execution_intent_from_paper_order,
)
from quant_platform.execution_gateway.persistence import (
    ExecutionLedgerEntry,
    ExecutionSessionEventStore,
    compute_execution_ledger_semantic_digest,
)
from quant_platform.execution_gateway.portfolio_risk_gate import (
    PortfolioRiskGatewayContext,
    authorize_portfolio_risk_dispatch,
    consume_portfolio_risk_dispatch,
    recover_portfolio_risk_dispatch_gate,
    reserve_portfolio_risk_dispatch,
)
from quant_platform.execution_gateway.reconciliation import (
    reconcile_execution_session,
    reconstruct_all_orders_from_ledger,
)
from quant_platform.execution_gateway.recovery import recover_unknown_orders
from quant_platform.execution_gateway.specs import (
    ExecutionGatewaySpec,
    compute_execution_gateway_spec_id,
    verify_execution_gateway_spec_identity,
)
from quant_platform.execution_gateway.state_machine import resolve_execution_order_state
from quant_platform.paper_trading.eligibility import require_paper_trading_eligibility
from quant_platform.paper_trading.events import MarketEvent
from quant_platform.paper_trading.orders import OrderRequest


@dataclass(frozen=True, slots=True)
class RunnerEnvironment:
    manifest_store: ExecutionSessionManifestStore
    event_store: ExecutionSessionEventStore
    paper_bridge_environment: PaperBridgeEnvironment
    portfolio_risk_context: PortfolioRiskGatewayContext
    """Milestone 9 Phase 4: REQUIRED, never optional -- `_run_intents_and_
    events` unconditionally gates every new `ExecutionIntent` through
    `portfolio_risk_gate.authorize_portfolio_risk_dispatch`/`reserve_
    portfolio_risk_dispatch` before it may reach `dispatcher.dispatch_
    command`. There is no configuration that disables this gate; a
    caller with no real portfolio-risk policy yet must still supply a
    context (e.g. an unconfigured `PortfolioRiskPolicy` where every limit
    is `None`), never omit one."""


def _kill_switch_events_from_ledger(ledger: list[ExecutionLedgerEntry]) -> list[ExecutionKillSwitchTransitionEvent]:
    return [ExecutionKillSwitchTransitionEvent.from_json_dict(e.payload) for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.KILL_SWITCH_TRIGGERED]


def run_execution_session(
    spec: ExecutionGatewaySpec, *, environment: RunnerEnvironment, adapter: ExecutionAdapter, paper_orders: list[OrderRequest], market_events: Iterable[MarketEvent],
    event_time: datetime,
) -> ExecutionSessionManifest:
    execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id
    manifest = environment.manifest_store.load_if_exists(execution_session_id)
    just_created = manifest is None

    if just_created:
        manifest = environment.manifest_store.create(
            execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id,
            adapter_id=adapter.adapter_id, execution_mode=spec.execution_mode,
        )
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_CREATED, payload=spec.to_json_dict(), event_time=event_time)
    else:
        environment.manifest_store.bump_resume_count(execution_session_id)
        assert manifest is not None

    if manifest.current_stage in (ExecutionSessionStage.COMPLETED, ExecutionSessionStage.FAILED, ExecutionSessionStage.TERMINATED):
        return manifest  # terminal -- resuming a completed session is an idempotent no-op.

    # Mandatory, fresh eligibility re-verification on EVERY call that
    # reaches this point -- unconditional, BEFORE any stage-dependent
    # branching, regardless of which stage the manifest is currently at.
    # CONFIRMED DEFECT, FIXED (found during this milestone's own
    # acceptance testing): this check used to live INSIDE the `if
    # manifest.current_stage is SPEC_VERIFIED:` block below, which only
    # executes on a session's very first pass through that exact stage --
    # a resume landing at ANY later stage (RUNNING, RECONCILING,
    # VERIFYING, or even a PAUSED resume, which jumps straight past
    # SPEC_VERIFIED to SOURCE_ELIGIBILITY_VERIFIED two lines below) never
    # reached that block again, so eligibility was in practice verified
    # exactly ONCE per session, never on any resume -- precisely the
    # Milestone 7 audit's own most severe finding, reintroduced here.
    # Fixed by mirroring `paper_trading.runner.run_paper_trading_session`'s
    # own unconditional top-level check exactly.
    loaded_paper_manifest = environment.paper_bridge_environment.manifest_store.load(spec.paper_session_id)
    from quant_platform.execution_gateway.paper_bridge import _load_paper_trading_spec

    paper_spec = _load_paper_trading_spec(environment.paper_bridge_environment, loaded_paper_manifest)
    require_paper_trading_eligibility(paper_spec, environment=environment.paper_bridge_environment.eligibility_environment)

    if manifest.current_stage is ExecutionSessionStage.PAUSED:
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_RESUMED, payload={}, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED)

    if manifest.current_stage is ExecutionSessionStage.CREATED:
        if not verify_execution_gateway_spec_identity(spec, execution_session_id):
            manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.FAILED, failure_category="spec_identity", failure_stage="spec_verified", failure_reason="spec does not reproduce its own execution_session_id", recoverable=False, safe_resume_stage=None)
            return manifest
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.SPEC_VERIFIED)

    if manifest.current_stage is ExecutionSessionStage.SPEC_VERIFIED:
        # The eligibility check itself already ran, unconditionally,
        # above -- this block only performs the first-time ledger/stage
        # bookkeeping for a brand-new session's own initial pass.
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.SOURCE_ELIGIBILITY_VERIFIED, payload={"paper_session_id": spec.paper_session_id}, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED)

    if manifest.current_stage is ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED:
        adapter.initialize(execution_session_id=execution_session_id, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.ADAPTER_INITIALIZED)

    if manifest.current_stage is ExecutionSessionStage.ADAPTER_INITIALIZED:
        recover_unknown_orders(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=event_time)
        # Milestone 9 Phase 4: execution_gateway's OWN order-state recovery
        # (above) must run FIRST -- portfolio-risk recovery's own cross-
        # reference (below) resolves a RESERVED-but-unresolved
        # authorization by reading the execution-gateway order state THAT
        # RECOVERY JUST ESTABLISHED, never a stale pre-recovery state.
        recover_portfolio_risk_dispatch_gate(
            context=environment.portfolio_risk_context, execution_session_id=execution_session_id,
            execution_ledger=environment.event_store.read_events(execution_session_id), recovery_time=event_time,
        )
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.RECOVERY_CHECKED)

    if manifest.current_stage is ExecutionSessionStage.RECOVERY_CHECKED:
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_STARTED, payload={}, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.RUNNING)

    if manifest.current_stage is ExecutionSessionStage.RUNNING:
        _run_intents_and_events(execution_session_id=execution_session_id, environment=environment, adapter=adapter, spec=spec, paper_orders=paper_orders, market_events=market_events, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.RECONCILING)

    if manifest.current_stage is ExecutionSessionStage.RECONCILING:
        ledger = environment.event_store.read_events(execution_session_id)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.RECONCILIATION_STARTED, payload={}, event_time=event_time)
        report = reconcile_execution_session(execution_session_id=execution_session_id, ledger=ledger, adapter=adapter, event_time=event_time, policy=spec.reconciliation_policy)
        if report.is_reconciled:
            _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.RECONCILIATION_COMPLETED, payload=report.to_json_dict(), event_time=event_time)
        else:
            _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.RECONCILIATION_FAILED, payload=report.to_json_dict(), event_time=event_time)
            codes = ", ".join(sorted({i.issue_code for i in report.blocking_issues}))
            manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.FAILED, failure_category="reconciliation", failure_stage="reconciling", failure_reason=f"blocking reconciliation issue(s): {codes}", recoverable=True, safe_resume_stage="running")
            return manifest
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.VERIFYING)

    if manifest.current_stage is ExecutionSessionStage.VERIFYING:
        ledger = environment.event_store.read_events(execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        unresolved_unknown = [oid for oid, (_c, state_events, _b, _f) in reconstructed.items() if resolve_execution_order_state(oid, state_events) is ExecutionOrderState.UNKNOWN]
        if unresolved_unknown:
            manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.FAILED, failure_category="unresolved_unknown", failure_stage="verifying", failure_reason=f"{len(unresolved_unknown)} order(s) remain UNKNOWN", recoverable=True, safe_resume_stage="adapter_initialized")
            return manifest
        digest = compute_execution_ledger_semantic_digest(ledger)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_COMPLETED, payload={"semantic_digest": digest}, event_time=event_time)
        manifest = environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.COMPLETED, semantic_digest=digest)

    return manifest


def _run_intents_and_events(
    *, execution_session_id: str, environment: RunnerEnvironment, adapter: ExecutionAdapter, spec: ExecutionGatewaySpec, paper_orders: list[OrderRequest],
    market_events: Iterable[MarketEvent], event_time: datetime,
) -> None:
    ledger = environment.event_store.read_events(execution_session_id)
    kill_switch_state = resolve_execution_kill_switch_state(_kill_switch_events_from_ledger(ledger))

    already_bridged_order_ids = {e.payload.get("source_paper_order_id") for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED}
    sequence = sum(1 for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED)

    for paper_order in paper_orders:
        if paper_order.order_id in already_bridged_order_ids:
            continue  # already bridged and dispatched on a prior call -- resume-safe, never re-bridged.
        intent, _authorization = execution_intent_from_paper_order(
            paper_order, execution_gateway_spec=spec, execution_session_id=execution_session_id, environment=environment.paper_bridge_environment,
            created_sequence=sequence, event_time=event_time,
        )
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED, payload=intent.to_json_dict(), event_time=event_time)

        # Milestone 9 Phase 4: NO execution intent may reach the dispatcher
        # without a valid Portfolio Risk Authorization -- mandatory,
        # fail-closed, never bypassed. A DENIED/HALTED decision (or any
        # other portfolio-risk refusal) is durably recorded via the
        # PRE-EXISTING (previously unused) EXECUTION_INTENT_REJECTED entry
        # kind before the exception propagates uncaught, exactly mirroring
        # `authorize_dispatch`'s own kill-switch-refusal precedent
        # immediately below (also uncaught here -- `run_execution_session`
        # itself has no try/except around this call, by existing design).
        try:
            risk_authorization = authorize_portfolio_risk_dispatch(intent=intent, context=environment.portfolio_risk_context, event_time=event_time)
        except ExecutionPortfolioRiskAuthorizationError as exc:
            _append(
                environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_INTENT_REJECTED,
                payload={"execution_intent_id": intent.execution_intent_id, "source_paper_order_id": paper_order.order_id, "reason": str(exc)},
                event_time=event_time,
            )
            raise
        intent = bind_portfolio_risk_authorization(intent, portfolio_risk_authorization_id=risk_authorization.risk_authorization_id)
        _append(
            environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.PORTFOLIO_RISK_AUTHORIZATION_BOUND,
            payload={"execution_intent_id": intent.execution_intent_id, "risk_authorization_id": risk_authorization.risk_authorization_id},
            event_time=event_time,
        )

        command = create_submit_order_command(
            execution_session_id=execution_session_id, execution_intent_id=intent.execution_intent_id, command_sequence=sequence, event_time=event_time,
            instrument_id=intent.instrument_id, side=intent.side, quantity=intent.quantity, order_type=intent.order_type, time_in_force=intent.time_in_force,
            reduce_only=intent.reduce_only, contract_multiplier=intent.contract_multiplier, limit_price=intent.limit_price, stop_price=intent.stop_price,
        )
        authorize_dispatch(kill_switch_state, command)
        reserve_portfolio_risk_dispatch(authorization=risk_authorization, intent=intent, context=environment.portfolio_risk_context, event_time=event_time)
        outcome = dispatch_command(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, command=command, event_time=event_time)
        if outcome.resolution is not None and outcome.resolution.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_SUCCEEDED:
            # Every other outcome (COMMAND_REJECTED, COMMAND_MARKED_UNKNOWN,
            # COMMAND_DISPATCH_REJECTED) leaves the authorization RESERVED,
            # never consumed and never auto-invalidated -- `recover_
            # portfolio_risk_dispatch_gate` resolves that ambiguity later,
            # using both packages' own durable evidence.
            consume_portfolio_risk_dispatch(authorization=risk_authorization, intent=intent, context=environment.portfolio_risk_context, event_time=event_time)
        sequence += 1

    advance = getattr(adapter, "advance_market_event", None)
    for market_event in market_events:
        if callable(advance):
            advance(market_event, event_time=event_time)
        process_broker_events(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, max_events=spec.recovery_policy.max_replay_events, event_time=event_time)

    # Deterministic DAY-order expiry once the supplied event stream is
    # exhausted (Section 13) -- unconditional, since `market_events` here
    # is a possibly-single-use iterator and cannot be safely re-scanned
    # for a `SessionCloseEvent` after the loop above has consumed it.
    _expire_day_orders_if_supported(adapter, event_time=event_time)
    process_broker_events(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, max_events=spec.recovery_policy.max_replay_events, event_time=event_time)


def _expire_day_orders_if_supported(adapter: ExecutionAdapter, *, event_time: datetime) -> None:
    expire = getattr(adapter, "expire_day_orders", None)
    if callable(expire):
        expire(event_time=event_time)


def pause_execution_session(*, execution_session_id: str, environment: RunnerEnvironment, event_time: datetime) -> ExecutionSessionManifest:
    """Durably pauses a `RUNNING` execution session (Section 19/31's
    `pause-execution-session` CLI command) -- a subsequent
    `run_execution_session` call continues from the ledger's own last
    completed step, re-verifying source eligibility fresh on the way."""
    environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.PAUSING)
    _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_SESSION_PAUSED, payload={}, event_time=event_time)
    return environment.manifest_store.transition(execution_session_id, target_stage=ExecutionSessionStage.PAUSED)


def authorize_cancel_or_reduce_only_submit(
    *, execution_session_id: str, environment: RunnerEnvironment, adapter: ExecutionAdapter, command: SubmitOrderCommand | CancelOrderCommand | ReplaceOrderCommand,
    event_time: datetime,
) -> None:
    """Exposed for `runner.py` callers (kill-switch escalation logic, the
    CLI's own operator-triggered cancel) that need to dispatch a command
    OUTSIDE the ordinary `run_execution_session` intent-bridging loop --
    still always through `authorize_dispatch` first, never around it.

    DELIBERATELY NOT gated through `portfolio_risk_gate` (Milestone 9
    Phase 4): every actual call site in this codebase passes only a
    `CancelOrderCommand`/`ReplaceOrderCommand` -- operating on an ALREADY-
    authorized, already-dispatched order, never a NEW economic
    submission. A cancel/replace does not correspond to a new
    `ExecutionIntent` at all, so "no execution intent may reach the
    dispatcher without a valid Portfolio Risk Authorization" does not
    apply here by definition. If a future caller ever passes a genuinely
    NEW `SubmitOrderCommand` through this function (the type signature
    technically allows it), that command would bypass the portfolio-risk
    gate entirely -- a gap worth closing before such a caller is added,
    not applicable to anything this milestone actually does."""
    ledger = environment.event_store.read_events(execution_session_id)
    kill_switch_state = resolve_execution_kill_switch_state(_kill_switch_events_from_ledger(ledger))
    authorize_dispatch(kill_switch_state, command)
    dispatch_command(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, command=command, event_time=event_time)


def current_kill_switch_state(*, execution_session_id: str, event_store: ExecutionSessionEventStore) -> ExecutionKillSwitchState:
    ledger = event_store.read_events(execution_session_id)
    return resolve_execution_kill_switch_state(_kill_switch_events_from_ledger(ledger))


__all__ = [
    "RunnerEnvironment",
    "authorize_cancel_or_reduce_only_submit",
    "current_kill_switch_state",
    "pause_execution_session",
    "run_execution_session",
]
