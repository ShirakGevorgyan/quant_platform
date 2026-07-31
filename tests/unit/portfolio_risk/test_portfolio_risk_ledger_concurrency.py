"""Concurrency tests for `portfolio_risk.lifecycle`/`ledger`: real
`threading.Thread`s racing against a SHARED `PortfolioRiskLedgerStore`,
proving the required properties under a genuine race -- one economic
use wins, an exact duplicate is absorbed idempotently, a conflicting
attempt is rejected AND durably audited (never silently invisible --
regression coverage for the retry-and-reclassify fix in `lifecycle.py`
found during this phase's own adversarial concurrency testing), and the
ledger is never corrupted (chain integrity always holds afterward).

`PortfolioRiskLedgerStore`'s own lock (`ledger.portfolio_risk_lock`,
wrapping `historical.locking.DatasetLock`) is documented to FAIL FAST
rather than block/retry on contention -- so a losing thread may see
`PortfolioRiskLockError` (lock busy) as well as a genuine domain
`RiskAuthorizationReuseError` (a real, final conflict). `_retry_on_lock`
below simulates a well-behaved caller retrying ONLY on lock contention,
letting a genuine domain rejection propagate immediately -- this is what
actually drives two threads into a genuine race at the storage layer
reliably enough to exercise it under test."""

from __future__ import annotations

import random
import tempfile
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal

from quant_platform.core.exceptions import PortfolioRiskLockError, RiskAuthorizationReuseError
from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.ledger import (
    PortfolioRiskLedgerStore,
    RiskLedgerEntryKind,
    verify_risk_ledger_chain_integrity,
)
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    expire_authorization,
    record_authorization_issuance,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import RiskDecisionKind

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_MAX_RETRY_ATTEMPTS = 4000


def _authorization(**overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": "1" * 64, "execution_session_id": "2" * 64, "portfolio_id": "p1", "portfolio_snapshot_id": "3" * 64,
        "price_snapshot_id": "4" * 64, "risk_policy_id": "5" * 64, "risk_decision_id": "6" * 64, "decision_kind": RiskDecisionKind.APPROVED,
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "authorization_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_authorization(**base)  # type: ignore[arg-type]


def _use_kwargs(authorization, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id,
        "portfolio_id": authorization.portfolio_id, "portfolio_snapshot_id": authorization.portfolio_snapshot_id,
        "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id,
        "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price, "consumption_identity": "use-1",
        "evaluation_time": _T0,
    }
    base.update(overrides)
    return base


def _jittered_backoff() -> None:
    # Fixed-delay backoff makes every losing thread wake up and retry at
    # the SAME instant, which under heavy contention (many threads all
    # racing the same lock) can repeatedly re-collide (a thundering herd)
    # rather than converging -- random jitter desynchronizes retries so
    # contention actually drains.
    time.sleep(random.uniform(0.0002, 0.002))


def _retry_on_lock(fn):
    """Retries ONLY on `PortfolioRiskLockError` (fail-fast lock
    contention) -- a `RiskAuthorizationReuseError` is a genuine, final
    domain rejection and must propagate immediately, uncaught."""
    for _ in range(_MAX_RETRY_ATTEMPTS):
        try:
            return fn()
        except PortfolioRiskLockError:
            _jittered_backoff()
    raise AssertionError("exhausted lock-contention retries without ever acquiring the lock")


def _retry_on_lock_or_append_race(fn):
    """For scenarios with NO domain conflict at all (racing to append
    genuinely DIFFERENT, non-conflicting content) -- retrying the whole
    call on either a lock-busy signal or a lost sequence-slot race is the
    correct, expected caller behavior (there is nothing to classify; the
    loser simply re-attempts against fresh state)."""
    for _ in range(_MAX_RETRY_ATTEMPTS):
        try:
            return fn()
        except (PortfolioRiskLockError, RiskAuthorizationReuseError):
            _jittered_backoff()
    raise AssertionError("exhausted retries without ever winning the append race")


def _run_concurrently(callables: list) -> dict[str, tuple[str, object]]:
    results: dict[str, tuple[str, object]] = {}
    barrier = threading.Barrier(len(callables))

    def _wrap(name: str, fn) -> None:
        try:
            barrier.wait()
            results[name] = ("ok", fn())
        except RiskAuthorizationReuseError as exc:
            results[name] = ("reuse_error", exc)

    threads = [threading.Thread(target=_wrap, args=(f"t{i}", fn)) for i, fn in enumerate(callables)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class TestSameExactReservationRace:
    def test_two_threads_reserving_with_the_same_identity_both_succeed_with_exactly_one_ledger_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            kwargs = _use_kwargs(authorization, consumption_identity="use-1")

            results = _run_concurrently([
                lambda: _retry_on_lock(lambda: reserve_authorization(store, authorization, **kwargs)),
                lambda: _retry_on_lock(lambda: reserve_authorization(store, authorization, **kwargs)),
            ])

            assert all(outcome == "ok" for outcome, _ in results.values())
            events = store.read_events("p1")
            reserved = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED]
            assert len(reserved) == 1
            assert verify_risk_ledger_chain_integrity(events)


class TestConflictingReservationRace:
    def test_two_threads_reserving_with_different_identities_exactly_one_wins_and_the_loser_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            kwargs_a = _use_kwargs(authorization, consumption_identity="use-1")
            kwargs_b = _use_kwargs(authorization, consumption_identity="use-2")

            results = _run_concurrently([
                lambda: _retry_on_lock(lambda: reserve_authorization(store, authorization, **kwargs_a)),
                lambda: _retry_on_lock(lambda: reserve_authorization(store, authorization, **kwargs_b)),
            ])

            outcomes = [outcome for outcome, _ in results.values()]
            assert outcomes.count("ok") == 1
            assert outcomes.count("reuse_error") == 1

            events = store.read_events("p1")
            reserved = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED]
            rejected = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED]
            assert len(reserved) == 1
            # The loser's rejection must be DURABLY AUDITED, never silently
            # invisible -- this is the regression this test file exists to
            # cover (a real defect found and fixed during this phase).
            assert len(rejected) == 1
            assert rejected[0].payload["rejection_reason"] == "conflicting_consumption"
            assert verify_risk_ledger_chain_integrity(events)


class TestDuplicateExactConsumptionRace:
    def test_two_threads_consuming_with_the_same_identity_both_succeed_with_exactly_one_ledger_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            kwargs = _use_kwargs(authorization, consumption_identity="use-1")
            reserve_authorization(store, authorization, **kwargs)

            results = _run_concurrently([
                lambda: _retry_on_lock(lambda: consume_authorization(store, authorization, **kwargs)),
                lambda: _retry_on_lock(lambda: consume_authorization(store, authorization, **kwargs)),
            ])

            assert all(outcome == "ok" for outcome, _ in results.values())
            events = store.read_events("p1")
            consumed = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED]
            assert len(consumed) == 1
            assert verify_risk_ledger_chain_integrity(events)


class TestConflictingSecondConsumptionRace:
    def test_two_threads_consuming_with_different_identities_exactly_one_wins_and_the_loser_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorization = _authorization()
            record_authorization_issuance(store, authorization, event_time=_T0)
            reserve_kwargs = _use_kwargs(authorization, consumption_identity="use-1")
            reserve_authorization(store, authorization, **reserve_kwargs)

            kwargs_legit = _use_kwargs(authorization, consumption_identity="use-1")
            kwargs_conflict = _use_kwargs(authorization, consumption_identity="use-2")

            results = _run_concurrently([
                lambda: _retry_on_lock(lambda: consume_authorization(store, authorization, **kwargs_legit)),
                lambda: _retry_on_lock(lambda: consume_authorization(store, authorization, **kwargs_conflict)),
            ])

            outcomes = [outcome for outcome, _ in results.values()]
            assert outcomes.count("ok") == 1
            assert outcomes.count("reuse_error") == 1

            events = store.read_events("p1")
            consumed = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_CONSUMED]
            rejected = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED]
            assert len(consumed) == 1
            assert len(rejected) == 1
            assert verify_risk_ledger_chain_integrity(events)


class TestExpiryRace:
    def test_racing_expire_against_reserve_always_resolves_to_a_coherent_final_state(self) -> None:
        # Both orderings are individually LEGAL (ISSUED -> EXPIRED directly,
        # or ISSUED -> RESERVED -> EXPIRED), so this is not a "must always
        # reject" scenario like the conflicting-identity races above --
        # the invariant under test is that the race always resolves to
        # EXACTLY ONE coherent outcome (never both a rejected AND an
        # accepted reservation, never ledger corruption), regardless of
        # which side wins. Run many independent trials to sample both
        # orderings.
        for _ in range(20):
            with tempfile.TemporaryDirectory() as tmp:
                store = PortfolioRiskLedgerStore(tmp)
                authorization = _authorization()
                record_authorization_issuance(store, authorization, event_time=_T0)
                kwargs = _use_kwargs(authorization, consumption_identity="use-1")

                results = _run_concurrently([
                    lambda store=store, authorization=authorization, kwargs=kwargs: _retry_on_lock(
                        lambda: reserve_authorization(store, authorization, **kwargs)
                    ),
                    lambda store=store, authorization=authorization: _retry_on_lock(
                        lambda: expire_authorization(store, authorization, reason_code="timed_out", detail="race", evaluation_time=_T0)
                    ),
                ])

                outcomes = [outcome for outcome, _ in results.values()]
                events = store.read_events("p1")
                assert verify_risk_ledger_chain_integrity(events)

                reserved = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_RESERVED]
                expired = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_EXPIRED]
                assert len(expired) == 1  # expire always eventually applies, from ISSUED or from RESERVED
                if outcomes.count("ok") == 2:
                    # reserve won first, then EXPIRED legally followed RESERVED
                    assert len(reserved) == 1
                else:
                    # expire won first -- the reserve attempt was rejected
                    # and durably audited, never silently invisible
                    assert outcomes.count("reuse_error") == 1
                    assert len(reserved) == 0
                    rejected = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_USE_REJECTED]
                    assert len(rejected) == 1


class TestSequenceAppendRace:
    def test_many_threads_issuing_distinct_authorizations_concurrently_never_corrupts_the_ledger(self) -> None:
        # 4 concurrent writers is enough to reliably exercise a genuine
        # sequence-slot race against the ledger's own fail-fast (non-
        # blocking) lock within a bounded retry budget -- higher thread
        # counts make the total retries needed to drain contention grow
        # fast enough to make this specific test flaky by budget exhaustion
        # alone, without adding anything to what the property being
        # proven (no ledger corruption under concurrent writers) already
        # demonstrates at this width.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            authorizations = [
                _authorization(execution_intent_id=hex(0x10 + i)[2:].rjust(64, "0"), risk_decision_id=hex(0x20 + i)[2:].rjust(64, "0"))
                for i in range(4)
            ]

            def _issue(auth):
                return lambda: _retry_on_lock_or_append_race(lambda: record_authorization_issuance(store, auth, event_time=_T0))

            results = _run_concurrently([_issue(a) for a in authorizations])

            assert all(outcome == "ok" for outcome, _ in results.values())
            events = store.read_events("p1")
            issued = [e for e in events if e.entry_kind is RiskLedgerEntryKind.RISK_AUTHORIZATION_ISSUED]
            assert len(issued) == 4
            assert {e.payload["risk_authorization_id"] for e in issued} == {a.risk_authorization_id for a in authorizations}
            assert verify_risk_ledger_chain_integrity(events)
            # Contiguous, gapless sequence -- the defining property of "no
            # ledger corruption" under concurrent writers.
            assert [e.entry_sequence for e in events] == list(range(len(events)))
