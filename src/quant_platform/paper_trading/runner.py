"""The session runner (Milestone 7, Section 22): `run_paper_trading_session`
orchestrates the 21-step pipeline (validate spec -> verify eligibility ->
acquire lock -> init manifest -> init account -> load strategy runtime ->
consume ordered events -> [compute features/model output -> create
decision -> risk checks -> create orders -> simulate execution -> apply
fills -> update account -> mark-to-market -> apply financing/session
events -> persist every event] -> reconcile -> finalize -> verify -> mark
COMPLETED) by calling into every other module's own pure functions --
this module adds no new financial/execution/accounting logic of its own.

FEATURE/MODEL COMPUTATION IS NOT THIS MODULE'S JOB: `strategy_runtime.
decide(context)` (Section 7's own `StrategyRuntime` Protocol) is handed a
`StrategyContext` and returns a `StrategyDecision` -- whatever feature
computation or model inference produced `context.model_output`/
`context.feature_snapshot` happened entirely INSIDE the caller-supplied
`StrategyRuntime` implementation, using the `features`/`ml` packages this
runner never touches directly.

RESUME SAFETY (Section 23), CORRECTED (release-audit finding -- this
paragraph previously overclaimed the actual behavior): recovery
granularity is WHOLE-EVENT-ONLY, never "mid-single-event crash-safe"
(Section 22's own "processed transactionally" language, which this
module satisfies only at the per-event grain). `_require_clean_event_
boundary` compares `MARKET_EVENT_ACCEPTED` against `ACCOUNT_SNAPSHOT`
counts on every resume and FAILS CLOSED (raises `PaperTradingStateError`,
mutates nothing) the instant they diverge -- a crash truly mid-event
does NOT "re-derive and skip already-persisted steps"; it is refused
outright and requires operator intervention (or a fresh session), by
design, because the cursor's ledger-length-based sequencing has no way
to safely resume a single interrupted event from an arbitrary partial
position. Resume only ever proceeds from a ledger that is a whole
number of fully-completed events; the four post-loop stage transitions
(`RUNNING -> END_OF_STREAM -> RECONCILING -> VERIFIED -> COMPLETED`) are
SEPARATELY made resume-safe by `_transition_with_ledger_entry` itself
(also a release-audit fix -- see that function's own docstring), since a
crash between one of ITS two durable writes is a distinct failure class
`_require_clean_event_boundary` does not cover (it happens strictly
after the last event's own `ACCOUNT_SNAPSHOT`). `PaperSessionEventStore.
append`'s own entry-id-keyed idempotency is a real, tested safety net,
but is not actually exercised by either of the two mechanisms above in
the normal case -- both resume paths avoid ever re-attempting an already-
persisted append rather than relying on it silently no-op'ing. Working
orders (including in-flight LIMIT/STOP orders) are reconstructed EXACTLY
on resume by `_reconstruct_working_orders` -- a release-audit finding,
fixed: every `ORDER_STATE_EVENT` ledger entry already embeds an order's
full economic detail (see `_order_state_payload`), so there was never a
real reason resume had to silently drop a still-working order.

KILL SWITCH (Section 18): after every mark-to-market, `evaluate_
continuous_risk` runs; if it recommends anything beyond `ALLOW` and the
in-memory kill switch is still `ACTIVE`, the runner walks it through the
legal transition graph (`ACTIVE -> HALTING -> {HALTED, FLATTENING ->
HALTED} -> TERMINATED`), persists a `HALT_TRIGGERED` ledger entry for
every step, sets `trading_halted=True` (blocking all further new orders
-- pre-trade risk's own mandatory check independently enforces this too),
and for `FLATTEN_SIMULATED_POSITIONS`/`TERMINATE_SESSION` synthesizes and
immediately executes a closing order for any open position. Once the
kill switch leaves `ACTIVE` it can never return to it, exactly like
`models.is_legal_kill_switch_transition`'s own construction guarantees --
"never silently auto-resume after a safety halt."

SINGLE-INSTRUMENT SCOPE: exactly like `portfolio.py`/`order_policy.py`,
this runner processes ONE instrument (`spec.instrument.symbol`) per
session, matching Section 13's own explicit permission for this
simplification."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

from quant_platform.backtesting.models import PositionDirection
from quant_platform.core.exceptions import PaperTradingStateError
from quant_platform.core.json import canonical_json_bytes
from quant_platform.ml.models import ArtifactCategory
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.paper_trading.accounting import apply_mark_to_position, flat_position
from quant_platform.paper_trading.clock import Clock, decision_time_for
from quant_platform.paper_trading.costs import compute_financing_cash_delta
from quant_platform.paper_trading.eligibility import (
    EligibilityVerificationEnvironment,
    require_paper_trading_eligibility,
)
from quant_platform.paper_trading.events import (
    BarEvent,
    EndOfStreamEvent,
    FinancingEvent,
    MarketEvent,
    QuoteEvent,
    SessionCloseEvent,
    SessionOpenEvent,
    TradingHaltEvent,
    TradingResumeEvent,
    market_event_id,
    market_event_time,
)
from quant_platform.paper_trading.execution import ExecutionOutcome, process_order_against_event
from quant_platform.paper_trading.manifests import PaperSessionManifest, PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    KillSwitchState,
    LedgerEntryKind,
    OrderSide,
    OrderState,
    OrderTypeKind,
    PaperSessionStage,
    PositionIntentKind,
    RejectReasonKind,
    RiskActionKind,
    RiskTriggerKind,
    SessionMode,
    TimeInForceKind,
)
from quant_platform.paper_trading.order_policy import OrderPolicyState, apply_order_policy
from quant_platform.paper_trading.orders import (
    OrderRequest,
    OrderStateEvent,
    create_order_request,
    create_order_state_event,
    resolve_order_state,
)
from quant_platform.paper_trading.persistence import LedgerEntry, PaperSessionEventStore, create_ledger_entry
from quant_platform.paper_trading.portfolio import (
    PortfolioState,
    apply_fill_to_portfolio,
    apply_financing_to_portfolio,
    apply_mark_to_portfolio,
    apply_order_created_to_portfolio,
    apply_order_rejected_to_portfolio,
    initial_portfolio,
)
from quant_platform.paper_trading.risk import (
    create_kill_switch_transition_event,
    evaluate_continuous_risk,
    evaluate_pre_trade_risk,
    most_severe_action,
)
from quant_platform.paper_trading.shadow import evaluate_shadow_decision
from quant_platform.paper_trading.specs import InstrumentSpec, PaperTradingSpec, compute_paper_session_spec_id
from quant_platform.paper_trading.strategy import (
    PortfolioSnapshot,
    RiskState,
    SessionState,
    StrategyContext,
    StrategyRuntime,
)

TERMINATED_ORDER_STATES = (OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED, OrderState.REJECTED)


@dataclass(frozen=True, slots=True)
class RunnerEnvironment:
    manifest_store: PaperSessionManifestStore
    event_store: PaperSessionEventStore
    eligibility_environment: EligibilityVerificationEnvironment


@dataclass
class _WorkingOrder:
    order: OrderRequest
    filled_quantity: float
    state: OrderState


@dataclass
class _Cursor:
    """Mutable ledger append cursor -- threaded through the whole run so
    every `_append` call sees the correct next `sequence`/`previous_hash`
    without every call site having to unpack/repack a tuple."""

    event_store: PaperSessionEventStore
    session_id: str
    sequence: int
    previous_hash: str | None

    def append(self, *, kind: LedgerEntryKind, payload: dict[str, object], event_time: datetime) -> LedgerEntry:
        entry = create_ledger_entry(session_id=self.session_id, sequence=self.sequence, kind=kind, payload=payload, event_time=event_time, previous_entry_hash=self.previous_hash)
        persisted = self.event_store.append(self.session_id, entry)
        self.sequence += 1
        self.previous_hash = persisted.entry_id
        return persisted


def _transition_with_ledger_entry(
    environment: RunnerEnvironment, paper_session_id: str, *, from_stage: PaperSessionStage, target_stage: PaperSessionStage, event_time: datetime, **transition_kwargs: object,
) -> PaperSessionManifest:
    """Every `PaperSessionStage` transition is persisted as its own
    `SESSION_TRANSITION` ledger entry (Section 21's 16-item required list)
    IN ADDITION TO the manifest's own current-stage field -- the manifest
    is a cache (Section 21 explicitly permits this for a fast "what stage
    are we at" read), the ledger is the source of truth `verify_paper_
    session` (Section 26) independently replays.

    RESUME-SAFE, release-audit finding, fixed: this function makes TWO
    separate durable writes (`manifest_store.transition` then `event_
    store.append`) with no atomicity between them. A crash after the
    first but before the second used to leave the session PERMANENTLY
    stuck -- `manifest.stage` was already `target_stage`, so every future
    call re-attempted the SAME `from_stage -> target_stage` transition,
    which `is_legal_paper_session_transition` rejects as an illegal
    self-transition, raising forever with no automatic recovery.

    Fixed: the manifest write only happens when `current.stage` still IS
    `from_stage`; any other value means a previous (interrupted) call
    already performed it -- possibly along with one or more LATER steps
    too, since `run_paper_trading_session`/`run_shadow_session` call this
    function for all 4 tail transitions unconditionally on every resume,
    so `current.stage` can legitimately be several steps ahead of what
    THIS particular call expects. Re-invoking `manifest_store.transition`
    in that case would be illegal (or would even ILLEGALLY REWIND the
    manifest backward, since `is_legal_paper_session_transition` also
    permits same-spine "rewind to an earlier stage" moves) -- so instead
    this only ensures THIS step's own `SESSION_TRANSITION` ledger entry
    exists (searching the whole ledger for its exact payload, not just
    the tail, since a manifest that's already 2+ steps ahead means this
    step's entry is no longer necessarily the ledger's last one)."""
    current = environment.manifest_store.load(paper_session_id)
    expected_payload: dict[str, object] = {"from_stage": from_stage.value, "to_stage": target_stage.value}
    if current.stage is from_stage:
        manifest = environment.manifest_store.transition(paper_session_id, target_stage=target_stage, **transition_kwargs)  # type: ignore[arg-type]
    else:
        if current.stage in (PaperSessionStage.FAILED, PaperSessionStage.TERMINATED):
            raise PaperTradingStateError(
                f"Cannot resume transition {from_stage.value!r} -> {target_stage.value!r} for paper session {paper_session_id!r}: "
                f"manifest is unexpectedly in terminal stage {current.stage.value!r}"
            )
        ledger = environment.event_store.read_events(paper_session_id)
        if any(e.kind is LedgerEntryKind.SESSION_TRANSITION and e.payload == expected_payload for e in ledger):
            return current
        manifest = current
    existing_ledger_length = environment.event_store.next_sequence(paper_session_id)
    previous_hash = environment.event_store.last_entry_hash(paper_session_id)
    entry = create_ledger_entry(
        session_id=paper_session_id, sequence=existing_ledger_length, kind=LedgerEntryKind.SESSION_TRANSITION,
        payload=expected_payload, event_time=event_time, previous_entry_hash=previous_hash,
    )
    environment.event_store.append(paper_session_id, entry)
    return manifest


def create_paper_session(spec: PaperTradingSpec, *, environment: RunnerEnvironment) -> PaperSessionManifest:
    """Steps 1-4: validate spec (already enforced by `PaperTradingSpec.
    __post_init__` at construction time), verify eligibility (fail-closed
    -- raises if not eligible), acquire the session lock and initialize
    the manifest at `CREATED` -> `ELIGIBILITY_VERIFIED`. Idempotent: if a
    manifest already exists for this spec's identity, returns it
    unchanged rather than re-verifying eligibility.

    Persists `spec` itself as a durable, content-addressed
    `PAPER_TRADING_SPEC` artifact and records the resulting reference on
    the manifest -- this is what lets a downstream consumer (Milestone
    8's execution gateway paper bridge) recover the full spec from just a
    `paper_session_id` later, and lets it cross-check the reference's own
    `content_hash` against an independently-declared identity, rather
    than needing the caller to already have the spec object in hand."""
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    existing = environment.manifest_store.load_if_exists(paper_session_id)
    if existing is not None:
        return existing

    require_paper_trading_eligibility(spec, environment=environment.eligibility_environment)
    spec_reference = environment.eligibility_environment.artifact_store.write_artifact(canonical_json_bytes(spec.to_json_dict()), category=ArtifactCategory.PAPER_TRADING_SPEC)
    environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=spec_reference)
    return _transition_with_ledger_entry(
        environment, paper_session_id, from_stage=PaperSessionStage.CREATED, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED, event_time=utc_now().to_pydatetime(),
    )


def pause_paper_session(environment: RunnerEnvironment, paper_session_id: str) -> PaperSessionManifest:
    """Section 23: an explicit, durable pause -- transitions a `RUNNING`
    session to `PAUSED`, persisted as its own `SESSION_TRANSITION` ledger
    entry exactly like every other stage transition (never merely an
    in-memory flag). A subsequent call to `run_paper_trading_session`
    transitions the manifest straight back to `RUNNING` (see that
    function's own `PaperSessionStage.PAUSED` handling) and continues
    from its own last completed event -- pausing and resuming a paper
    session is, from the ledger's perspective, just another interruption/
    resume cycle."""
    manifest = environment.manifest_store.load(paper_session_id)
    if manifest.stage is not PaperSessionStage.RUNNING:
        raise PaperTradingStateError(f"Cannot pause paper session {paper_session_id!r}: stage is {manifest.stage.value!r}, not RUNNING")
    return _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.RUNNING, target_stage=PaperSessionStage.PAUSED, event_time=utc_now().to_pydatetime())


def _reference_price(event: MarketEvent) -> float | None:
    if isinstance(event, QuoteEvent):
        return (event.bid + event.ask) / 2.0
    if isinstance(event, BarEvent):
        return event.close
    return None


def _order_state_payload(state_event: OrderStateEvent, order: OrderRequest) -> dict[str, object]:
    """Every `ORDER_STATE_EVENT` ledger entry embeds the FULL `OrderRequest`
    alongside its own state transition -- an `OrderStateEvent` on its own
    only records from_state/to_state/reason_code, never the order's own
    economic details (side/quantity/price), and `reconciliation.py`'s
    "order quantity = sum(fills)+remaining" check (and any ledger-only
    state reconstruction, e.g. `runner._reconstruct_resume_state`'s
    documented working-order gap) needs those details recoverable from
    ANY single order-related ledger entry, not just the very first one."""
    return {"order_state_event": state_event.to_json_dict(), "order": order.to_json_dict()}


def _apply_execution_outcome(outcome: ExecutionOutcome, working: _WorkingOrder, *, portfolio: PortfolioState, contract_multiplier: float, cursor: _Cursor, event_time: datetime) -> PortfolioState:
    for state_event in outcome.order_state_events:
        cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(state_event, working.order), event_time=event_time)
        working.state = state_event.to_state
    for fill in outcome.fills:
        working.filled_quantity += fill.quantity
        portfolio = apply_fill_to_portfolio(portfolio, fill, event_time=event_time, contract_multiplier=contract_multiplier)
        cursor.append(kind=LedgerEntryKind.FILL, payload=fill.to_json_dict(), event_time=event_time)
    return portfolio


def _create_and_submit_order(
    order: OrderRequest, *, portfolio: PortfolioState, instrument: InstrumentSpec, spec: PaperTradingSpec, event: MarketEvent, event_time: datetime,
    reference_price: float, trading_halted: bool, session_accepting_orders: bool, cursor: _Cursor, working_orders: dict[str, _WorkingOrder],
) -> PortfolioState:
    """Runs one freshly-policy-created `OrderRequest` through CREATED ->
    VALIDATED -> (REJECTED | ACCEPTED -> WORKING), persisting every
    transition, then immediately attempts a fill against `event` (the
    same event that produced the decision behind this order)."""
    risk_results = evaluate_pre_trade_risk(
        order, portfolio=portfolio, risk_limits=spec.risk_limits, reference_price=reference_price, contract_multiplier=instrument.contract_multiplier,
        event_identity=market_event_id(event), trading_halted=trading_halted, session_accepting_orders=session_accepting_orders, stale_data_seconds=None,
    )
    cursor.append(kind=LedgerEntryKind.RISK_DECISION, payload={"results": [r.to_json_dict() for r in risk_results]}, event_time=event_time)
    portfolio = apply_order_created_to_portfolio(portfolio, event_time=event_time)

    validated_event = create_order_state_event(order_id=order.order_id, session_id=order.session_id, from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=event_time, sequence=cursor.sequence)
    cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(validated_event, order), event_time=event_time)

    action = most_severe_action(risk_results)
    if action is not RiskActionKind.ALLOW:
        failed = next(r for r in risk_results if not r.passed)
        reason = failed.reason_code if isinstance(failed.reason_code, RejectReasonKind) else RejectReasonKind.RISK_HALT_ACTIVE
        rejected_event = create_order_state_event(order_id=order.order_id, session_id=order.session_id, from_state=OrderState.VALIDATED, to_state=OrderState.REJECTED, event_time=event_time, sequence=cursor.sequence, reason_code=reason)
        cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(rejected_event, order), event_time=event_time)
        return apply_order_rejected_to_portfolio(portfolio, event_time=event_time)

    accepted_event = create_order_state_event(order_id=order.order_id, session_id=order.session_id, from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=event_time, sequence=cursor.sequence)
    cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(accepted_event, order), event_time=event_time)
    working_event = create_order_state_event(order_id=order.order_id, session_id=order.session_id, from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=event_time, sequence=cursor.sequence)
    cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(working_event, order), event_time=event_time)

    working = _WorkingOrder(order=order, filled_quantity=0.0, state=OrderState.WORKING)
    working_orders[order.order_id] = working
    outcome = process_order_against_event(
        order, current_state=OrderState.WORKING, filled_quantity_so_far=0.0, event=event, event_time=event_time, sequence=cursor.sequence,
        instrument=instrument, execution_policy=spec.execution_policy, spread_policy=spec.spread_policy, slippage_policy=spec.slippage_policy,
        commission_policy=spec.commission_policy, fill_policy=spec.fill_policy, liquidity_policy=spec.liquidity_policy,
    )
    portfolio = _apply_execution_outcome(outcome, working, portfolio=portfolio, contract_multiplier=instrument.contract_multiplier, cursor=cursor, event_time=event_time)
    if working.state in TERMINATED_ORDER_STATES:
        del working_orders[order.order_id]
    return portfolio


def _require_clean_event_boundary(ledger: list[LedgerEntry], *, paper_session_id: str) -> None:
    """FAIL-CLOSED resume guard: every `MARKET_EVENT_ACCEPTED` entry must
    have a matching `ACCOUNT_SNAPSHOT` entry (unconditionally the LAST
    entry appended for any event's processing) before this function will
    allow a resume to proceed. If a crash happened truly MID-EVENT
    (the market event was accepted but its downstream decision/order/fill
    processing did not finish before the interruption), retrying that
    SAME event from scratch would try to re-append its entries at the
    WRONG ledger position (the cursor only knows the ledger's current
    total length, not where THAT event's own entries began) -- silently
    producing either duplicate or conflicting entries. Rather than risk
    that corruption, this function raises `PaperTradingStateError`
    outright: a session interrupted mid-event requires operator
    intervention (or a fresh session), never an automatic retry. Every
    event this runner itself completes always ends with exactly one
    `ACCOUNT_SNAPSHOT`, so this condition can only be reached by a REAL
    interruption mid-event, never by ordinary resume-after-completion."""
    market_event_count = sum(1 for entry in ledger if entry.kind is LedgerEntryKind.MARKET_EVENT_ACCEPTED)
    account_snapshot_count = sum(1 for entry in ledger if entry.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
    if market_event_count != account_snapshot_count:
        raise PaperTradingStateError(
            f"Paper session {paper_session_id!r} was interrupted mid-event (found {market_event_count} MARKET_EVENT_ACCEPTED "
            f"entries but only {account_snapshot_count} ACCOUNT_SNAPSHOT entries) -- automatic resume is refused to avoid "
            "corrupting the ledger; this session requires manual investigation."
        )


def _reconstruct_working_orders(ledger: list[LedgerEntry]) -> dict[str, _WorkingOrder]:
    """Rebuilds every order still non-terminal (`WORKING`/`PARTIALLY_
    FILLED`) at the moment of interruption, EXACTLY, from the ledger
    alone -- release-audit finding, fixed: every `ORDER_STATE_EVENT`
    entry already embeds the order's full economic detail (see
    `_order_state_payload`'s own docstring, written for exactly this
    purpose), so there was never a real reason `working_orders` had to
    reset to `{}` on resume. `resolve_order_state` (the SAME event-
    sourced derivation the forward runner and `verification.py` both
    already trust) determines each order's current state by replaying
    its own transitions; an order whose resolved state is not one of
    `TERMINATED_ORDER_STATES` is reconstructed with its ORIGINAL
    `OrderRequest` (side/quantity/price/type/time_in_force/create_time/
    submit_time all preserved byte-for-byte via `OrderRequest.from_json_
    dict`) and its `filled_quantity` summed from the ledger's own `FILL`
    entries -- never re-derived, never fabricated."""
    orders_raw: dict[str, tuple[dict[str, object], list[OrderStateEvent]]] = {}
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.ORDER_STATE_EVENT:
            continue
        order_json = entry.payload["order"]
        state_event = OrderStateEvent.from_json_dict(entry.payload["order_state_event"])  # type: ignore[arg-type]
        if state_event.order_id not in orders_raw:
            orders_raw[state_event.order_id] = (order_json, [])  # type: ignore[assignment]
        orders_raw[state_event.order_id][1].append(state_event)

    filled_quantity_by_order: dict[str, float] = {}
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.FILL:
            continue
        order_id = str(entry.payload["order_id"])
        filled_quantity_by_order[order_id] = filled_quantity_by_order.get(order_id, 0.0) + float(str(entry.payload["quantity"]))

    working_orders: dict[str, _WorkingOrder] = {}
    for order_id, (order_json, events) in orders_raw.items():
        final_state = resolve_order_state(order_id, events)
        if final_state in TERMINATED_ORDER_STATES:
            continue
        order = OrderRequest.from_json_dict(order_json)
        working_orders[order_id] = _WorkingOrder(order=order, filled_quantity=filled_quantity_by_order.get(order_id, 0.0), state=final_state)
    return working_orders


@dataclass
class _ResumeState:
    portfolio: PortfolioState
    working_orders: dict[str, _WorkingOrder]
    processed_event_count: int
    trading_halted: bool
    session_accepting_orders: bool
    kill_switch_state: KillSwitchState
    kill_switch_sequence: int


def _reconstruct_resume_state(ledger: list[LedgerEntry], *, paper_session_id: str, instrument: InstrumentSpec, starting_cash: float) -> _ResumeState:
    """Rebuilds in-memory runner state from a possibly-non-empty existing
    ledger prefix -- the trusted-cache resume path Section 21 explicitly
    permits (unlike `verification.py`'s from-scratch replay, which must
    NOT trust any cached snapshot). Working orders (including in-flight
    LIMIT/STOP orders) are reconstructed EXACTLY via `_reconstruct_
    working_orders` -- see that function's own docstring for why this is
    safe and exact, not an approximation.

    `processed_event_count` is deliberately counted from `ACCOUNT_
    SNAPSHOT` entries, not `MARKET_EVENT_ACCEPTED` ones -- an
    `ACCOUNT_SNAPSHOT` is unconditionally the LAST entry appended for any
    event's processing, so counting it (rather than the FIRST entry) is
    what makes `_require_clean_event_boundary`'s guard -- and therefore
    this function's own "always a whole number of fully-completed
    events" precondition -- correct."""
    processed_event_count = sum(1 for entry in ledger if entry.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)

    portfolio: PortfolioState | None = None
    for entry in reversed(ledger):
        if entry.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT:
            portfolio = PortfolioState.from_json_dict(entry.payload)
            break
    if portfolio is None:
        portfolio = initial_portfolio(paper_session_id, starting_cash=starting_cash)
        portfolio = replace(portfolio, positions={instrument.symbol: flat_position(instrument.symbol, contract_multiplier=instrument.contract_multiplier)})

    trading_halted = False
    session_accepting_orders = True
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.MARKET_EVENT_ACCEPTED:
            continue
        event_kind = entry.payload.get("kind")
        if event_kind == "session_open":
            session_accepting_orders = True
        elif event_kind == "session_close":
            session_accepting_orders = False
        elif event_kind == "trading_halt":
            trading_halted = True
        elif event_kind == "trading_resume":
            trading_halted = False

    kill_switch_state = KillSwitchState.ACTIVE
    kill_switch_sequence = 0
    for entry in ledger:
        if entry.kind is LedgerEntryKind.HALT_TRIGGERED:
            kill_switch_state = KillSwitchState(entry.payload["to_state"])
            kill_switch_sequence = int(str(entry.payload["sequence"])) + 1

    return _ResumeState(
        portfolio=portfolio, working_orders=_reconstruct_working_orders(ledger), processed_event_count=processed_event_count, trading_halted=trading_halted,
        session_accepting_orders=session_accepting_orders, kill_switch_state=kill_switch_state, kill_switch_sequence=kill_switch_sequence,
    )


def run_paper_trading_session(
    spec: PaperTradingSpec, *, environment: RunnerEnvironment, strategy_runtime: StrategyRuntime, clock: Clock, events: Iterable[MarketEvent],
) -> PaperSessionManifest:
    """Steps 5-21. Safe to call repeatedly (see module docstring) --
    `events` should be the SAME deterministic source every call for
    resume to reproduce identical results."""
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest = environment.manifest_store.load_if_exists(paper_session_id)
    just_created = manifest is None
    if manifest is None:
        manifest = create_paper_session(spec, environment=environment)
    if manifest.stage is PaperSessionStage.FAILED:
        raise PaperTradingStateError(f"Cannot run paper session {paper_session_id!r}: manifest is in terminal FAILED stage")
    if manifest.stage is PaperSessionStage.TERMINATED:
        return manifest
    if manifest.stage is PaperSessionStage.COMPLETED:
        # Release-audit finding, fixed: a crash between the tail sequence's
        # final `manifest_store.transition` (to COMPLETED) and its own
        # ledger append used to be invisible here -- this shortcut returned
        # immediately, permanently skipping the missing `SESSION_TRANSITION`
        # ledger entry. `_transition_with_ledger_entry` backfills it (or
        # no-ops if it's already there) instead.
        return _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.VERIFIED, target_stage=PaperSessionStage.COMPLETED, event_time=utc_now().to_pydatetime())

    if not just_created:
        # Release-audit finding, fixed: eligibility was previously
        # verified EXACTLY ONCE, inside `create_paper_session`, at the
        # very first call for this `paper_session_id` -- every subsequent
        # call (a resume, whether after a pause, a crash, or an operator-
        # initiated interruption) found an existing manifest and skipped
        # `create_paper_session` entirely, so the FULL eligibility chain
        # (promotion decision, robustness result, source backtest) was
        # NEVER re-verified again. A session whose underlying artifacts
        # were tampered, superseded, or invalidated AFTER it started could
        # resume and keep processing new market events indefinitely --
        # exactly the audit's own named blocker: "eligibility bypass on
        # start or resume." Fixed: every resume re-runs the SAME fail-
        # closed check `create_paper_session` itself uses, before this
        # call is allowed to process a single further market event.
        require_paper_trading_eligibility(spec, environment=environment.eligibility_environment)

    if manifest.stage is PaperSessionStage.ELIGIBILITY_VERIFIED:
        manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.ELIGIBILITY_VERIFIED, target_stage=PaperSessionStage.INITIALIZED, event_time=utc_now().to_pydatetime())
    if manifest.stage is PaperSessionStage.PAUSED:
        manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.PAUSED, target_stage=PaperSessionStage.RUNNING, event_time=utc_now().to_pydatetime())

    instrument = spec.instrument
    existing_ledger = environment.event_store.read_events(paper_session_id)
    _require_clean_event_boundary(existing_ledger, paper_session_id=paper_session_id)
    resume_state = _reconstruct_resume_state(existing_ledger, paper_session_id=paper_session_id, instrument=instrument, starting_cash=spec.starting_cash)
    portfolio = resume_state.portfolio
    working_orders = resume_state.working_orders
    already_processed_events = resume_state.processed_event_count
    order_policy_state = OrderPolicyState(bars_since_last_order=None, orders_created_in_rate_window=0)
    trading_halted = resume_state.trading_halted
    session_accepting_orders = resume_state.session_accepting_orders
    kill_switch_state = resume_state.kill_switch_state
    kill_switch_sequence = resume_state.kill_switch_sequence

    if manifest.stage is PaperSessionStage.INITIALIZED:
        manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.INITIALIZED, target_stage=PaperSessionStage.RUNNING, event_time=utc_now().to_pydatetime())

    cursor = _Cursor(event_store=environment.event_store, session_id=paper_session_id, sequence=environment.event_store.next_sequence(paper_session_id), previous_hash=environment.event_store.last_entry_hash(paper_session_id))

    stream_ended = False
    for event in itertools.islice(events, already_processed_events, None):
        event_time = market_event_time(event)
        clock.advance_to(event_time, sequence=event.sequence)
        cursor.append(kind=LedgerEntryKind.MARKET_EVENT_ACCEPTED, payload=event.to_json_dict(), event_time=event_time)

        if isinstance(event, SessionOpenEvent):
            session_accepting_orders = True
        elif isinstance(event, SessionCloseEvent):
            session_accepting_orders = False
        elif isinstance(event, TradingHaltEvent):
            trading_halted = True
        elif isinstance(event, TradingResumeEvent):
            trading_halted = False
        elif isinstance(event, FinancingEvent):
            position = portfolio.positions.get(instrument.symbol)
            if position is not None and position.signed_quantity != 0.0:
                direction = PositionDirection.LONG if position.signed_quantity > 0 else PositionDirection.SHORT
                notional = abs(position.signed_quantity) * (position.last_mark or position.average_entry_price or 0.0) * instrument.contract_multiplier
                cash_delta = compute_financing_cash_delta(spec.financing_policy, direction=direction, notional=notional, holding_days=1.0)
                portfolio = apply_financing_to_portfolio(portfolio, instrument=instrument.symbol, cash_delta=cash_delta, event_time=event_time)
                cursor.append(kind=LedgerEntryKind.FINANCING_APPLIED, payload={"instrument": instrument.symbol, "cash_delta": cash_delta}, event_time=event_time)
        elif isinstance(event, EndOfStreamEvent):
            stream_ended = True

        for order_id in list(working_orders):
            working = working_orders[order_id]
            outcome = process_order_against_event(
                working.order, current_state=working.state, filled_quantity_so_far=working.filled_quantity, event=event, event_time=event_time,
                sequence=cursor.sequence, instrument=instrument, execution_policy=spec.execution_policy, spread_policy=spec.spread_policy,
                slippage_policy=spec.slippage_policy, commission_policy=spec.commission_policy, fill_policy=spec.fill_policy,
                liquidity_policy=spec.liquidity_policy,
            )
            portfolio = _apply_execution_outcome(outcome, working, portfolio=portfolio, contract_multiplier=instrument.contract_multiplier, cursor=cursor, event_time=event_time)
            if working.state in TERMINATED_ORDER_STATES:
                del working_orders[order_id]

        reference_price = _reference_price(event)
        if reference_price is not None:
            portfolio = apply_mark_to_portfolio(portfolio, instrument=instrument.symbol, mark_price=reference_price, event_time=event_time)
            cursor.append(kind=LedgerEntryKind.MARK_APPLIED, payload={"instrument": instrument.symbol, "mark_price": reference_price}, event_time=event_time)

            if kill_switch_state is KillSwitchState.ACTIVE:
                continuous_results = evaluate_continuous_risk(
                    portfolio, risk_limits=spec.risk_limits, event_identity=market_event_id(event), rejected_order_count=portfolio.rejected_order_count,
                    consecutive_execution_failures=0, stale_data_seconds=None, reconciliation_discrepancy=None,
                )
                continuous_action = most_severe_action(continuous_results)
                if continuous_action is not RiskActionKind.ALLOW:
                    failed_trigger = next((r.reason_code for r in continuous_results if not r.passed and isinstance(r.reason_code, RiskTriggerKind)), RiskTriggerKind.LOSS_LIMIT)
                    trading_halted = True
                    halting_event = create_kill_switch_transition_event(session_id=paper_session_id, from_state=KillSwitchState.ACTIVE, to_state=KillSwitchState.HALTING, trigger=failed_trigger, event_time=event_time, sequence=kill_switch_sequence, detail="continuous risk check failed")
                    cursor.append(kind=LedgerEntryKind.HALT_TRIGGERED, payload=halting_event.to_json_dict(), event_time=event_time)
                    kill_switch_sequence += 1
                    kill_switch_state = KillSwitchState.HALTING

                    if continuous_action in (RiskActionKind.FLATTEN_SIMULATED_POSITIONS, RiskActionKind.TERMINATE_SESSION):
                        flattening_event = create_kill_switch_transition_event(session_id=paper_session_id, from_state=KillSwitchState.HALTING, to_state=KillSwitchState.FLATTENING, trigger=failed_trigger, event_time=event_time, sequence=kill_switch_sequence, detail="flattening open positions")
                        cursor.append(kind=LedgerEntryKind.HALT_TRIGGERED, payload=flattening_event.to_json_dict(), event_time=event_time)
                        kill_switch_sequence += 1
                        kill_switch_state = KillSwitchState.FLATTENING

                        # Release-audit finding, fixed: the working-order fill
                        # loop (top of this same iteration) runs UNCONDITIONALLY
                        # for every event regardless of kill_switch_state -- a
                        # pre-existing resting LIMIT/STOP order left WORKING
                        # through a flatten could still fill on a LATER event,
                        # silently reopening exposure the safety flatten exists
                        # to eliminate. Cancel every other still-working order
                        # FIRST, before creating the flatten's own closing
                        # order, exactly matching Section 9's own "cancel
                        # working orders first" requirement.
                        for stale_order_id in list(working_orders):
                            stale_working = working_orders[stale_order_id]
                            cancel_event = create_order_state_event(
                                order_id=stale_order_id, session_id=paper_session_id, from_state=stale_working.state, to_state=OrderState.CANCELLED,
                                event_time=event_time, sequence=cursor.sequence, reason_code=RejectReasonKind.RISK_HALT_ACTIVE,
                            )
                            cursor.append(kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=_order_state_payload(cancel_event, stale_working.order), event_time=event_time)
                            del working_orders[stale_order_id]

                        position = portfolio.positions.get(instrument.symbol)
                        if position is not None and position.signed_quantity != 0.0:
                            close_side = OrderSide.SELL if position.signed_quantity > 0 else OrderSide.BUY
                            close_order = create_order_request(
                                client_order_id=f"kill-switch-flatten:{event.event_id}", session_id=paper_session_id, strategy_decision_id="0" * 64,
                                instrument=instrument.symbol, side=close_side, order_type=OrderTypeKind.MARKET,
                                quantity=abs(position.signed_quantity), time_in_force=TimeInForceKind.IOC, create_time=event_time, submit_time=event_time,
                                reduce_only=True, position_intent=PositionIntentKind.CLOSE,
                            )
                            # The kill switch's OWN flattening order is a risk-REDUCING
                            # exit, never a new speculative trade -- it is deliberately
                            # exempt from the trading_halted/session_accepting_orders
                            # pre-trade checks that block everything else during a halt
                            # (those checks exist to stop NEW risk-taking, not to trap
                            # an open position open during the very halt meant to close it).
                            portfolio = _create_and_submit_order(close_order, portfolio=portfolio, instrument=instrument, spec=spec, event=event, event_time=event_time, reference_price=reference_price, trading_halted=False, session_accepting_orders=True, cursor=cursor, working_orders=working_orders)

                        halted_event = create_kill_switch_transition_event(session_id=paper_session_id, from_state=KillSwitchState.FLATTENING, to_state=KillSwitchState.HALTED, trigger=failed_trigger, event_time=event_time, sequence=kill_switch_sequence, detail="flattened")
                        cursor.append(kind=LedgerEntryKind.HALT_TRIGGERED, payload=halted_event.to_json_dict(), event_time=event_time)
                        kill_switch_sequence += 1
                        kill_switch_state = KillSwitchState.HALTED
                    else:
                        halted_event = create_kill_switch_transition_event(session_id=paper_session_id, from_state=KillSwitchState.HALTING, to_state=KillSwitchState.HALTED, trigger=failed_trigger, event_time=event_time, sequence=kill_switch_sequence, detail="halted, no flatten required")
                        cursor.append(kind=LedgerEntryKind.HALT_TRIGGERED, payload=halted_event.to_json_dict(), event_time=event_time)
                        kill_switch_sequence += 1
                        kill_switch_state = KillSwitchState.HALTED

                    if continuous_action is RiskActionKind.TERMINATE_SESSION:
                        terminated_event = create_kill_switch_transition_event(session_id=paper_session_id, from_state=KillSwitchState.HALTED, to_state=KillSwitchState.TERMINATED, trigger=failed_trigger, event_time=event_time, sequence=kill_switch_sequence, detail="terminated")
                        cursor.append(kind=LedgerEntryKind.HALT_TRIGGERED, payload=terminated_event.to_json_dict(), event_time=event_time)
                        kill_switch_sequence += 1
                        kill_switch_state = KillSwitchState.TERMINATED
                        stream_ended = True

            snapshot = portfolio.to_strategy_snapshot(instrument.symbol)
            decision_time = decision_time_for(event_time)
            context = StrategyContext(
                event=event, feature_snapshot={}, feature_snapshot_identity=None, model_output=0.0, model_output_identity=None,
                calibrated_probability=None, confidence=0.5, uncertainty=0.0, portfolio=snapshot,
                risk=RiskState(trading_halted=trading_halted, kill_switch_state=kill_switch_state), session=SessionState(paper_session_id=paper_session_id, stage=manifest.stage),
                decision_time=decision_time,
            )
            decision = strategy_runtime.decide(context)
            cursor.append(kind=LedgerEntryKind.STRATEGY_DECISION, payload=decision.to_json_dict(), event_time=event_time)

            if not decision.abstain and kill_switch_state is KillSwitchState.ACTIVE:
                new_orders = apply_order_policy(
                    decision, portfolio=snapshot, instrument=instrument, policy=spec.order_policy, risk_limits=spec.risk_limits,
                    latency_policy=spec.latency_policy, session_id=paper_session_id, create_time=decision_time, state=order_policy_state,
                )
                for order in new_orders:
                    portfolio = _create_and_submit_order(
                        order, portfolio=portfolio, instrument=instrument, spec=spec, event=event, event_time=event_time, reference_price=reference_price,
                        trading_halted=trading_halted, session_accepting_orders=session_accepting_orders, cursor=cursor, working_orders=working_orders,
                    )

        cursor.append(kind=LedgerEntryKind.ACCOUNT_SNAPSHOT, payload=portfolio.to_json_dict(), event_time=event_time)

        if stream_ended:
            break

    now = utc_now().to_pydatetime()
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.RUNNING, target_stage=PaperSessionStage.END_OF_STREAM, event_time=now)
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.END_OF_STREAM, target_stage=PaperSessionStage.RECONCILING, event_time=now)
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.RECONCILING, target_stage=PaperSessionStage.VERIFIED, event_time=now)
    return _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.VERIFIED, target_stage=PaperSessionStage.COMPLETED, event_time=now, completed_at=format_utc_timestamp(utc_now()))


def run_shadow_session(
    spec: PaperTradingSpec, *, environment: RunnerEnvironment, strategy_runtime: StrategyRuntime, clock: Clock, events: Iterable[MarketEvent],
) -> PaperSessionManifest:
    """Section 19: `SHADOW_OBSERVATION` session mode. Runs the same
    market-event consumption / strategy-decision loop as `run_paper_
    trading_session`, but every decision is evaluated via `shadow.
    evaluate_shadow_decision` against a private `PositionState` this
    function owns -- NEVER a real `PortfolioState` -- and every outcome is
    persisted as a `SHADOW_OBSERVATION` ledger entry, never `ORDER_STATE_
    EVENT`/`FILL`/`ACCOUNT_SNAPSHOT`. Fail-closed on `spec.session_mode`:
    a REPLAY_PAPER/FORWARD_PAPER spec must use `run_paper_trading_session`
    instead, never this function, so a shadow session can never be
    mistaken for one whose fills touch a real account.

    SIMPLIFIED (documented, not silently dropped): unlike the real
    runner, this loop does not model trading halts/session-close/
    financing/kill-switch escalation for the shadow position -- Section
    19 requires only that "strategy decisions are produced; hypothetical
    orders are produced; ... observations are persisted for later
    comparison," not a full parallel risk/session-lifecycle simulation.
    Every `MarketEvent` is still persisted (`MARKET_EVENT_ACCEPTED`) and
    every decision is still persisted (`STRATEGY_DECISION`), exactly like
    the real runner.

    DEFERRED (documented, not silently dropped): resuming an interrupted
    shadow session is not supported -- this function refuses (fail-closed)
    to proceed if the ledger already contains any `MARKET_EVENT_ACCEPTED`
    entry, rather than risk the same kind of mid-event corruption `run_
    paper_trading_session`'s `_require_clean_event_boundary`/`_reconstruct_
    resume_state` machinery exists to prevent for the real pipeline (that
    machinery is keyed on `ACCOUNT_SNAPSHOT`, which a shadow session never
    produces, so it cannot be reused here unmodified)."""
    if spec.session_mode is not SessionMode.SHADOW_OBSERVATION:
        raise PaperTradingStateError(f"run_shadow_session requires session_mode=SHADOW_OBSERVATION, got {spec.session_mode.value!r}")

    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest = environment.manifest_store.load_if_exists(paper_session_id)
    just_created = manifest is None
    if manifest is None:
        manifest = create_paper_session(spec, environment=environment)
    if manifest.stage is PaperSessionStage.FAILED:
        raise PaperTradingStateError(f"Cannot run shadow session {paper_session_id!r}: manifest is in terminal FAILED stage")
    if manifest.stage is PaperSessionStage.TERMINATED:
        return manifest
    if manifest.stage is PaperSessionStage.COMPLETED:
        # Same release-audit fix as `run_paper_trading_session`'s own
        # COMPLETED shortcut -- see that function's comment.
        return _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.VERIFIED, target_stage=PaperSessionStage.COMPLETED, event_time=utc_now().to_pydatetime())

    if not just_created:
        # Same release-audit fix as `run_paper_trading_session`'s own --
        # see that function's comment. Still reachable here even though
        # shadow sessions refuse to resume past their first event: a
        # crash between manifest creation and the first `MARKET_EVENT_
        # ACCEPTED` entry leaves a legitimate, eligible-at-the-time
        # resume path this check must still re-validate.
        require_paper_trading_eligibility(spec, environment=environment.eligibility_environment)

    if manifest.stage is PaperSessionStage.ELIGIBILITY_VERIFIED:
        manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.ELIGIBILITY_VERIFIED, target_stage=PaperSessionStage.INITIALIZED, event_time=utc_now().to_pydatetime())

    instrument = spec.instrument
    existing_ledger = environment.event_store.read_events(paper_session_id)
    if any(entry.kind is LedgerEntryKind.MARKET_EVENT_ACCEPTED for entry in existing_ledger):
        raise PaperTradingStateError(
            f"Shadow session {paper_session_id!r} has already begun processing events -- resuming an interrupted "
            "SHADOW_OBSERVATION session is not supported in this milestone; start a fresh session instead."
        )

    if manifest.stage is PaperSessionStage.INITIALIZED:
        manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.INITIALIZED, target_stage=PaperSessionStage.RUNNING, event_time=utc_now().to_pydatetime())

    shadow_position = flat_position(instrument.symbol, contract_multiplier=instrument.contract_multiplier)
    order_policy_state = OrderPolicyState(bars_since_last_order=None, orders_created_in_rate_window=0)
    cursor = _Cursor(event_store=environment.event_store, session_id=paper_session_id, sequence=environment.event_store.next_sequence(paper_session_id), previous_hash=environment.event_store.last_entry_hash(paper_session_id))

    stream_ended = False
    for event in events:
        event_time = market_event_time(event)
        clock.advance_to(event_time, sequence=event.sequence)
        cursor.append(kind=LedgerEntryKind.MARKET_EVENT_ACCEPTED, payload=event.to_json_dict(), event_time=event_time)

        if isinstance(event, EndOfStreamEvent):
            stream_ended = True

        reference_price = _reference_price(event)
        if reference_price is not None:
            shadow_position = apply_mark_to_position(shadow_position, mark_price=reference_price, event_time=event_time)

            snapshot = PortfolioSnapshot(
                instrument=instrument.symbol, signed_quantity=shadow_position.signed_quantity, average_entry_price=shadow_position.average_entry_price,
                cash=0.0, equity=0.0, unrealized_pnl=shadow_position.unrealized_pnl, realized_pnl=shadow_position.realized_pnl,
            )
            decision_time = decision_time_for(event_time)
            context = StrategyContext(
                event=event, feature_snapshot={}, feature_snapshot_identity=None, model_output=0.0, model_output_identity=None,
                calibrated_probability=None, confidence=0.5, uncertainty=0.0, portfolio=snapshot,
                risk=RiskState(trading_halted=False, kill_switch_state=KillSwitchState.ACTIVE), session=SessionState(paper_session_id=paper_session_id, stage=manifest.stage),
                decision_time=decision_time,
            )
            decision = strategy_runtime.decide(context)
            cursor.append(kind=LedgerEntryKind.STRATEGY_DECISION, payload=decision.to_json_dict(), event_time=event_time)

            observation, shadow_position = evaluate_shadow_decision(
                decision, shadow_position=shadow_position, instrument=instrument, order_policy=spec.order_policy, execution_policy=spec.execution_policy,
                spread_policy=spec.spread_policy, slippage_policy=spec.slippage_policy, commission_policy=spec.commission_policy,
                fill_policy=spec.fill_policy, liquidity_policy=spec.liquidity_policy, latency_policy=spec.latency_policy, risk_limits=spec.risk_limits,
                session_id=paper_session_id, event=event, order_policy_state=order_policy_state, sequence=cursor.sequence,
            )
            cursor.append(kind=LedgerEntryKind.SHADOW_OBSERVATION, payload=observation.to_json_dict(), event_time=event_time)

        if stream_ended:
            break

    now = utc_now().to_pydatetime()
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.RUNNING, target_stage=PaperSessionStage.END_OF_STREAM, event_time=now)
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.END_OF_STREAM, target_stage=PaperSessionStage.RECONCILING, event_time=now)
    manifest = _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.RECONCILING, target_stage=PaperSessionStage.VERIFIED, event_time=now)
    return _transition_with_ledger_entry(environment, paper_session_id, from_stage=PaperSessionStage.VERIFIED, target_stage=PaperSessionStage.COMPLETED, event_time=now, completed_at=format_utc_timestamp(utc_now()))


__all__ = ["RunnerEnvironment", "create_paper_session", "pause_paper_session", "run_paper_trading_session", "run_shadow_session"]
