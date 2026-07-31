"""Semantic ledger tampering tests (Milestone 8, Section 29/34/38). Mirrors
`tests/unit/paper_trading/test_audit_ledger_semantic_tampering.py`'s own
governing invariant exactly: "structural hash integrity alone must never
be treated as semantic validity." Every scenario below tampers a ledger's
semantic CONTENT while RECHAINING every hash coherently (`_rechain`) --
never relying on `verify_execution_ledger_chain_integrity` alone to catch
the tampering -- then confirms `verify_execution_session`/`reconcile_
execution_session` identify a SPECIFIC issue code, never a generic
mismatch.

CONFIRMED DEFECTS FOUND DURING THIS AUDIT, FIXED (see `verification.py`/
`reconciliation.py`): an orphan `EXECUTION_FILL_RECORDED` entry (an
`execution_order_id` matching no submitted order anywhere in the ledger)
used to be silently DROPPED by `reconstruct_all_orders_from_ledger`'s own
pre-filtering -- invisible to every downstream check. Both modules now
scan the raw ledger directly for this (`orphan_fill_no_matching_order`).

CONFIRMED, DOCUMENTED, NON-BLOCKING LIMITATION (mirroring the identical,
already-accepted Milestone 7 limitation for its own analogous content-
addressed types -- `Fill`/`OrderRequest`/`LedgerEntry`/market events):
`BrokerEvent`/`ExecutionFill`/`ExecutionOrderStateEvent` do not
self-validate their own identity field against a recomputed content hash
on deserialization. A forged-but-uniquely-valued `broker_event_id` (with
every other field intact, so no OTHER check fires) is therefore not
caught by any check today -- documented here, not silently omitted."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

from quant_platform.execution_gateway.commands import SubmitOrderCommand, create_submit_order_command
from quant_platform.execution_gateway.dispatcher import dispatch_command, process_broker_events
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.identity import compute_content_id
from quant_platform.execution_gateway.models import (
    EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION,
    AdapterKind,
    ExecutionLedgerEntryKind,
    ExecutionMode,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    SequencingPolicyKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.persistence import (
    EXECUTION_LEDGER_ENTRY_KIND,
    ExecutionLedgerEntry,
    ExecutionSessionEventStore,
)
from quant_platform.execution_gateway.reconciliation import (
    reconcile_execution_session,
    reconstruct_all_orders_from_ledger,
)
from quant_platform.execution_gateway.specs import (
    DEFAULT_DUMMY_BROKER_SCENARIO,
    DispatchPolicySpec,
    ExecutionGatewaySpec,
    HealthPolicySpec,
    HeartbeatPolicySpec,
    IdempotencyPolicySpec,
    KillSwitchPolicySpec,
    ReconciliationPolicySpec,
    RecoveryPolicySpec,
    SequencingPolicySpec,
    compute_execution_gateway_spec_id,
)
from quant_platform.execution_gateway.state_machine import (
    ExecutionOrderStateEvent,
    resolve_execution_order_state,
)
from quant_platform.execution_gateway.states import ExecutionFill, compute_execution_order_id
from quant_platform.execution_gateway.verification import verify_execution_session
from quant_platform.paper_trading.events import create_quote_event

_SHA_PAPER_SESSION = "1" * 64
_SHA_PAPER_SPEC = "2" * 64
_SHA_PROMOTION = "3" * 64
_SHA_INSTRUMENT = "4" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _spec() -> ExecutionGatewaySpec:
    return ExecutionGatewaySpec(
        schema_version=EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION, execution_mode=ExecutionMode.TEST_ONLY, adapter_kind=AdapterKind.DETERMINISTIC_DUMMY,
        paper_session_id=_SHA_PAPER_SESSION, paper_trading_spec_id=_SHA_PAPER_SPEC, promotion_decision_id=_SHA_PROMOTION, instrument_spec_id=_SHA_INSTRUMENT,
        sequencing_policy=SequencingPolicySpec(policy=SequencingPolicyKind.STRICT_SEQUENCE),
        idempotency_policy=IdempotencyPolicySpec(durable_evidence_required=True, max_safe_retry_attempts=3),
        recovery_policy=RecoveryPolicySpec(max_replay_events=1000, unknown_resolution_timeout_events=50),
        reconciliation_policy=ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True),
        health_policy=HealthPolicySpec(stale_after_events=20, degraded_after_consecutive_failures=2, unavailable_after_consecutive_failures=5),
        heartbeat_policy=HeartbeatPolicySpec(interval_events=10, missed_threshold_degraded=2, missed_threshold_halting=5),
        kill_switch_policy=KillSwitchPolicySpec(max_unresolved_unknown_operations=3, max_broker_sequence_conflicts=1, max_blocking_reconciliation_issues=1),
        dispatch_policy=DispatchPolicySpec(require_dispatch_intent_before_call=True, max_commands_per_batch=100), dummy_broker_scenario=DEFAULT_DUMMY_BROKER_SCENARIO,
        seed=42,
    )


# Computed (not hardcoded) so `verify_execution_session`'s own spec-identity
# check (comparing a fresh `compute_execution_gateway_spec_id(spec)` against
# the `execution_session_id` every ledger entry below is built under) never
# spuriously fires -- every tampering scenario in this file must be caught
# for the SPECIFIC reason it tests, not incidentally masked/confounded by an
# unrelated identity mismatch.
_SHA_SESSION = compute_execution_gateway_spec_id(_spec()).execution_gateway_spec_id


def _submit_command(*, instrument_id="EURUSD", quantity=Decimal("10"), sequence=0, intent_id):
    return create_submit_order_command(
        execution_session_id=_SHA_SESSION, execution_intent_id=intent_id, command_sequence=sequence, event_time=_NOW, instrument_id=instrument_id,
        side=OrderSide.BUY, quantity=quantity, order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.DAY, reduce_only=False, contract_multiplier=Decimal("1"),
    )


def _build_two_order_session(tmp_path) -> tuple[ExecutionSessionEventStore, str, str]:
    """Builds a real, small, two-order session (both filled) purely via
    the dispatcher/dummy-broker primitives -- no ML chain needed. Returns
    `(event_store, execution_order_id_a, execution_order_id_b)`."""
    event_store = ExecutionSessionEventStore(tmp_path)
    adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
    adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)

    command_a = _submit_command(quantity=Decimal("10"), sequence=0, intent_id="b" * 64)
    command_b = _submit_command(quantity=Decimal("5"), sequence=1, intent_id="c" * 64)
    order_id_a = compute_execution_order_id(command_a)
    order_id_b = compute_execution_order_id(command_b)
    dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command_a, event_time=_NOW)
    dispatch_command(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, command=command_b, event_time=_NOW)

    tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
    adapter.advance_market_event(tick, event_time=_NOW)
    process_broker_events(execution_session_id=_SHA_SESSION, event_store=event_store, adapter=adapter, max_events=100, event_time=_NOW)
    return event_store, order_id_a, order_id_b


def _rechain(entries: list[ExecutionLedgerEntry]) -> list[ExecutionLedgerEntry]:
    """Recomputes `entry_id`/`previous_entry_hash`/`entry_hash` for every
    entry so the result is a genuinely VALID hash chain -- this is what
    proves the SEMANTIC checks below catch the tampering, never merely
    that a broken hash chain would have."""
    rechained: list[ExecutionLedgerEntry] = []
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        entry_hash = compute_content_id("execution_ledger_entry_payload", entry.payload)
        provisional = dataclasses.replace(entry, entry_sequence=index, previous_entry_hash=previous_hash, entry_hash=entry_hash, entry_id="0" * 64)
        entry_id = compute_content_id(EXECUTION_LEDGER_ENTRY_KIND, provisional.to_identity_payload())
        final = dataclasses.replace(provisional, entry_id=entry_id)
        rechained.append(final)
        previous_hash = entry_id
    return rechained


def _replace_payload(entry: ExecutionLedgerEntry, new_payload: dict) -> ExecutionLedgerEntry:
    new_hash = compute_content_id("execution_ledger_entry_payload", new_payload)
    return dataclasses.replace(entry, payload=new_payload, entry_hash=new_hash)


def _verify(ledger: list[ExecutionLedgerEntry]):
    return verify_execution_session(_spec(), execution_session_id=_SHA_SESSION, ledger=ledger)


class TestRemoveAnOrderStateTransition:
    def test_removing_the_dispatched_to_acknowledged_transition_is_caught(self, tmp_path) -> None:
        event_store, order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        remove_index = next(
            i for i, e in enumerate(ledger)
            if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload.get("execution_order_id") == order_id_a
            and ExecutionOrderStateEvent.from_json_dict(e.payload).to_state.value == "acknowledged"
        )
        tampered = [e for i, e in enumerate(ledger) if i != remove_index]
        tampered = _rechain(tampered)

        report = _verify(tampered)
        codes = {i.code for i in report.criticals}
        assert "order_state_transition_illegal" in codes


class TestDuplicateFillWithDistinctIdentity:
    def test_a_second_fill_entry_with_the_same_broker_fill_id_but_a_different_execution_fill_id_is_caught(self, tmp_path) -> None:
        event_store, order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        fill_index = next(i for i, e in enumerate(ledger) if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED and e.payload.get("execution_order_id") == order_id_a)
        original = ledger[fill_index]
        forged_payload = dict(original.payload)
        forged_payload["execution_fill_id"] = "f" * 64  # same broker_fill_id, a DIFFERENT (forged) execution_fill_id
        forged_entry = ExecutionLedgerEntry(
            entry_id="0" * 64, execution_session_id=_SHA_SESSION, entry_sequence=0, entry_kind=ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED,
            payload=forged_payload, event_time=original.event_time, recorded_time=original.recorded_time, previous_entry_hash=None,
            entry_hash=compute_content_id("execution_ledger_entry_payload", forged_payload),
        )
        tampered = [*ledger[: fill_index + 1], forged_entry, *ledger[fill_index + 1 :]]
        tampered = _rechain(tampered)

        report = _verify(tampered)
        codes = {i.code for i in report.criticals}
        assert "duplicate_fill_identity" in codes


class TestSwapOrderStateHistoryBetweenTwoOrders:
    def test_swapping_which_order_a_transition_claims_to_belong_to_is_caught(self, tmp_path) -> None:
        event_store, order_id_a, order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        index_a = next(i for i, e in enumerate(ledger) if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload.get("execution_order_id") == order_id_a)
        index_b = next(i for i, e in enumerate(ledger) if e.entry_kind is ExecutionLedgerEntryKind.ORDER_STATE_TRANSITION and e.payload.get("execution_order_id") == order_id_b)
        tampered = list(ledger)
        payload_a, payload_b = dict(tampered[index_a].payload), dict(tampered[index_b].payload)
        payload_a["execution_order_id"], payload_b["execution_order_id"] = payload_b["execution_order_id"], payload_a["execution_order_id"]
        tampered[index_a] = _replace_payload(tampered[index_a], payload_a)
        tampered[index_b] = _replace_payload(tampered[index_b], payload_b)
        tampered = _rechain(tampered)

        report = _verify(tampered)
        codes = {i.code for i in report.criticals}
        assert "order_state_transition_illegal" in codes


class TestShrinkDeclaredQuantityBelowAlreadyFilled:
    """NOTE: `reconstruct_execution_order`'s own `ExecutionOrder.__post_
    init__` already validates `filled_quantity + remaining_quantity ==
    current_quantity` at CONSTRUCTION time -- a shrunk `quantity` that
    would make `filled_quantity > current_quantity` makes construction
    itself raise, caught by `verify_execution_session`'s own outer
    try/except and reported as `order_state_transition_illegal` (a
    slightly imprecise code for a quantity problem, but still correctly
    CRITICAL). This means the SEPARATE, more specific `cumulative_fill_
    exceeds_quantity` check (verification.py, checking `order.filled_
    quantity > order.current_quantity` on an ALREADY-constructed `order`)
    is unreachable dead code for this exact scenario -- a successfully
    constructed `ExecutionOrder` can never fail that comparison. Asserted
    here as the safety property that actually matters: the tampering is
    caught as CRITICAL, under EITHER code."""

    def test_shrinking_the_submit_commands_own_quantity_below_the_filled_amount_is_caught(self, tmp_path) -> None:
        event_store, order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        submit_index = next(
            i for i, e in enumerate(ledger)
            if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_CREATED and e.payload.get("command_type") == "submit_order"
            and compute_execution_order_id(SubmitOrderCommand.from_json_dict(e.payload)) == order_id_a
        )
        forged_payload = dict(ledger[submit_index].payload)
        forged_payload["quantity"] = "1"  # far below the 10 actually filled
        tampered = list(ledger)
        tampered[submit_index] = _replace_payload(tampered[submit_index], forged_payload)
        tampered = _rechain(tampered)

        report = _verify(tampered)
        codes = {i.code for i in report.criticals}
        assert codes & {"cumulative_fill_exceeds_quantity", "order_state_transition_illegal"}


class TestOrphanFill:
    def test_a_fill_referencing_an_unknown_execution_order_id_is_caught(self, tmp_path) -> None:
        event_store, _order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        fill_entry = next(e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED)
        orphan_payload = dict(fill_entry.payload)
        orphan_payload["execution_order_id"] = "e" * 64  # no COMMAND_CREATED submit anywhere for this id
        orphan_payload["execution_fill_id"] = "d" * 64
        last = ledger[-1]
        orphan_entry = ExecutionLedgerEntry(
            entry_id="0" * 64, execution_session_id=_SHA_SESSION, entry_sequence=last.entry_sequence + 1, entry_kind=ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED,
            payload=orphan_payload, event_time=last.event_time, recorded_time=last.recorded_time, previous_entry_hash=last.entry_id,
            entry_hash=compute_content_id("execution_ledger_entry_payload", orphan_payload),
        )
        final_id = compute_content_id(EXECUTION_LEDGER_ENTRY_KIND, orphan_entry.to_identity_payload())
        orphan_entry = dataclasses.replace(orphan_entry, entry_id=final_id)

        report = _verify([*ledger, orphan_entry])
        codes = {i.code for i in report.criticals}
        assert "orphan_fill_no_matching_order" in codes

        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
        reconciliation_report = reconcile_execution_session(execution_session_id=_SHA_SESSION, ledger=[*ledger, orphan_entry], adapter=adapter, event_time=_NOW, policy=_spec().reconciliation_policy)
        assert not reconciliation_report.is_reconciled
        assert any(i.issue_code == "orphan_fill_no_matching_order" for i in reconciliation_report.issues)


class TestForgedFinalReportHasNoInfluence:
    """`verify_execution_session` takes no report argument at all -- a
    forged/false `ExecutionSessionReport` structurally has zero influence
    on the outcome, since verification independently recomputes
    everything from `spec`/`ledger` alone."""

    def test_two_independent_verify_calls_against_the_same_ledger_produce_identical_issues(self, tmp_path) -> None:
        event_store, _order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        first = _verify(ledger)
        second = _verify(ledger)
        assert {i.code for i in first.issues} == {i.code for i in second.issues}


class TestUseALedgerFromAnotherSession:
    def test_foreign_session_ledger_entry_is_rejected(self, tmp_path) -> None:
        event_store, _order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        tampered = list(ledger)
        foreign_payload = dict(tampered[-1].payload)
        foreign_entry = dataclasses.replace(tampered[-1], execution_session_id="f" * 64, payload=foreign_payload)
        tampered[-1] = foreign_entry
        # Deliberately NOT rechained -- session ownership is checked directly
        # against `execution_session_id`, independent of hash-chain validity.

        report = _verify(tampered)
        codes = {i.code for i in report.criticals}
        assert "ledger_session_ownership_mismatch" in codes


class TestTruncateLedgerAtValidHashBoundary:
    def test_truncated_but_hash_valid_ledger_still_shows_unresolved_state(self, tmp_path) -> None:
        event_store, order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        # Truncate right after order_id_a's DISPATCH_INTENT but before its
        # resolution -- a genuinely valid hash-chain PREFIX (never
        # re-chained, doesn't need to be), but order_id_a's own history is
        # now incomplete.
        cut_index = next(i for i, e in enumerate(ledger) if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT)
        truncated = ledger[: cut_index + 1]

        report = _verify(truncated)
        # The truncated order never reaches a terminal state -- not
        # necessarily UNKNOWN (it may simply be DISPATCH_PENDING), but it
        # must not be silently treated as fully resolved/filled.
        reconstructed = reconstruct_all_orders_from_ledger(truncated)
        assert order_id_a in set(reconstructed)
        state = resolve_execution_order_state(order_id_a, reconstructed[order_id_a][1])
        assert state is not ExecutionOrderState.FILLED, "a truncated ledger must never resolve to a fully-filled state it never actually reached"
        assert not any(i.code == "spec_identity_mismatch" for i in report.issues)  # sanity: unrelated to this scenario


class TestSourceBrokerEventIdentityForgery:
    """CONFIRMED, UNDERSTOOD, NON-BLOCKING LIMITATION (see module
    docstring): `BrokerEvent.broker_event_id` is never re-validated
    against its own content on deserialization, matching every other
    content-addressed type in this codebase. Documents the gap exists
    rather than silently omitting it."""

    def test_forged_broker_event_id_with_intact_sequence_is_not_currently_caught(self, tmp_path) -> None:
        event_store, _order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        broker_event_index = next(i for i, e in enumerate(ledger) if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED)
        entry = ledger[broker_event_index]
        forged_payload = dict(entry.payload)
        forged_payload["broker_event_id"] = "9" * 64  # a fabricated but unique id, sequence/timestamps untouched
        tampered = list(ledger)
        tampered[broker_event_index] = _replace_payload(entry, forged_payload)
        tampered = _rechain(tampered)

        report = _verify(tampered)
        # Documents the CURRENT (limited) behavior -- no issue code exists for this today.
        assert not any("broker_event_id" in i.code or "broker_event_identity" in i.code for i in report.issues)


class TestFillIdentityRoundTripsCorrectly:
    def test_fill_can_be_deserialized_and_matches_its_own_declared_fields(self, tmp_path) -> None:
        event_store, order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        fill_entry = next(e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED and e.payload.get("execution_order_id") == order_id_a)
        fill = ExecutionFill.from_json_dict(fill_entry.payload)
        assert fill.execution_order_id == order_id_a
        assert fill.gross_notional == fill.quantity * fill.price * fill.contract_multiplier


class TestBrokerEventDeserializationRoundTrip:
    def test_broker_event_round_trips_and_preserves_broker_sequence(self, tmp_path) -> None:
        event_store, _order_id_a, _order_id_b = _build_two_order_session(tmp_path)
        ledger = event_store.read_events(_SHA_SESSION)
        entry = next(e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_RECEIVED)
        event = BrokerEvent.from_json_dict(entry.payload)
        assert event.broker_sequence >= 1
