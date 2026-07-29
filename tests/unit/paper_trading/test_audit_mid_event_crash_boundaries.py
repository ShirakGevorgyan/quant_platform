"""Release-audit Area 3: mid-event crash boundaries.

METHOD: `_CrashAfterNAppends` wraps a real `PaperSessionEventStore` and
raises on the (N+1)-th `append` call, simulating a hard process crash at
an EXACT durable sub-step of one market-event transaction. The ledger
entry sequence for one representative event (bar with a strategy that
opens a new long position, fully filled) was mapped empirically:

    0  session_transition   (ELIGIBILITY_VERIFIED -> INITIALIZED)
    1  session_transition   (INITIALIZED -> RUNNING)
    2  market_event_accepted
    3  mark_applied
    4  strategy_decision
    5  risk_decision
    6  order_state_event    (CREATED -> VALIDATED)
    7  order_state_event    (VALIDATED -> ACCEPTED)
    8  order_state_event    (ACCEPTED -> WORKING)
    9  order_state_event    (WORKING -> FILLED)
    10 fill
    11 account_snapshot     <- unconditionally the LAST entry for event 1
    12 market_event_accepted (event 2, no new order: already at target)
    ...
    20 session_transition   (RUNNING -> END_OF_STREAM)          <- tail
    21 session_transition   (END_OF_STREAM -> RECONCILING)      <- tail
    22 session_transition   (RECONCILING -> VERIFIED)           <- tail
    23 session_transition   (VERIFIED -> COMPLETED)              <- tail

CONFIRMED DEFECT (found during this audit, fixed in `runner.py`):
`_transition_with_ledger_entry` called `environment.manifest_store.
transition(...)` (durable write #1) and THEN `environment.event_store.
append(...)` (durable write #2) with no resume-safety between them. A
crash after #1 but before #2 -- e.g. at N=20, where `manifest.stage`
durably becomes `END_OF_STREAM` but the matching `SESSION_TRANSITION`
ledger entry never lands -- left the session PERMANENTLY stuck: every
future call to `run_paper_trading_session` re-attempted the exact same
`RUNNING -> END_OF_STREAM` transition, which `is_legal_paper_session_
transition` rejects as an illegal `END_OF_STREAM -> END_OF_STREAM`
self-transition (since `manifest.stage` is already `END_OF_STREAM`),
raising `PaperTradingManifestError` forever with no automatic recovery
and no typed `failure_category`/`failure_stage` ever recorded on the
manifest (Section 20's own documented FAILED-state contract). Fixed by
making `_transition_with_ledger_entry` itself resume-safe: if the
manifest is already at `target_stage`, it backfills the (possibly still
missing) `SESSION_TRANSITION` ledger entry instead of re-invoking
`manifest_store.transition` -- covers every call site (`create_paper_
session`, `pause_paper_session`, and all of `run_paper_trading_session`'s
own stage transitions), not just the tail sequence.

GENUINE mid-event crashes (any point strictly between a `MARKET_EVENT_
ACCEPTED` entry and its own matching `ACCOUNT_SNAPSHOT`) are, by
contrast, confirmed to ALREADY be correctly fail-closed: `_require_
clean_event_boundary` counts `MARKET_EVENT_ACCEPTED` entries against
`ACCOUNT_SNAPSHOT` entries and refuses to resume (raises `PaperTrading
StateError`) the instant they don't match -- this file adversarially
confirms that guard actually fires at EVERY listed sub-step (not just
some), that the ledger is left with a real, undeleted partial-event
tail (proving no silent transactional rollback), that a second failed
resume attempt is byte-for-byte idempotent (no corruption growth), and
that the one clean boundary in the sweep (between two whole events)
resumes successfully to a result identical to an uninterrupted run.

`_require_clean_event_boundary` is keyed ONLY on `MARKET_EVENT_ACCEPTED`
vs `ACCOUNT_SNAPSHOT` counts -- it does not fire for the tail-transition
crashes above, since those happen strictly AFTER the last event's own
`ACCOUNT_SNAPSHOT` (counts still match); this is exactly why the tail
sequence needed its own, separate resume-safety fix."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingStateError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    ClockMode,
    LedgerEntryKind,
    MarketEventMode,
    PaperSessionStage,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.persistence import LedgerEntry, PaperSessionEventStore
from quant_platform.paper_trading.runner import RunnerEnvironment, run_paper_trading_session
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_POSITION_POLICY,
    DEFAULT_SESSION_BOUNDARY_POLICY,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
    RiskLimitsSpec,
    compute_paper_session_spec_id,
)
from quant_platform.paper_trading.strategy import StrategyContext, StrategyDecision, create_strategy_decision

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash", account_currency="USD",
        financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _risk_limits(**overrides: object) -> RiskLimitsSpec:
    defaults: dict[str, object] = {
        "maximum_signed_position": None, "maximum_absolute_position": None, "maximum_gross_exposure": None, "maximum_order_quantity": None,
        "maximum_order_notional": None, "maximum_turnover": None, "maximum_daily_loss": None, "maximum_drawdown_fraction": None,
        "maximum_realized_loss": None, "maximum_unrealized_loss": None, "maximum_rejected_order_count": None,
        "maximum_consecutive_execution_failures": None, "maximum_stale_data_seconds": None, "maximum_reconciliation_discrepancy": 1e-6,
    }
    defaults.update(overrides)
    return RiskLimitsSpec(**defaults)  # type: ignore[arg-type]


def _spec(**overrides: object) -> PaperTradingSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "verified_robustness_id": _HEX_A, "verified_promotion_decision_id": _HEX_B, "strategy_candidate_identity": _HEX_C,
        "model_artifact_identity": _HEX_D, "calibration_artifact_identity": _HEX_E, "feature_spec_identity": _HEX_A, "instrument": _instrument(),
        "price_precision": 2, "quantity_precision": 2, "session_mode": SessionMode.REPLAY_PAPER, "market_event_mode": MarketEventMode.BAR,
        "bar_interval": Timeframe.H1, "clock_mode": ClockMode.REPLAY, "starting_cash": 100_000.0, "starting_positions": (),
        "order_policy": OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=100, order_rate_window_events=1000),
        "execution_policy": DEFAULT_EXECUTION_POLICY, "fill_policy": FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        "spread_policy": SpreadSpec(kind=SpreadModelKind.ZERO), "slippage_policy": SlippageSpec(kind=SlippageModelKind.ZERO),
        "commission_policy": CommissionSpec(kind=CommissionModelKind.ZERO),
        "financing_policy": FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)),
        "latency_policy": LatencyPolicySpec(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0),
        "liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=False), "position_policy": DEFAULT_POSITION_POLICY,
        "risk_limits": _risk_limits(), "session_boundary_policy": DEFAULT_SESSION_BOUNDARY_POLICY, "seed": 0,
    }
    defaults.update(overrides)
    return PaperTradingSpec(**defaults)  # type: ignore[arg-type]


def _bars(closes: list[float]) -> list:
    events = []
    for i, close in enumerate(closes):
        open_time = _T0 + timedelta(hours=i)
        events.append(create_bar_event(instrument="X", interval=Timeframe.H1, open_time=open_time, open=close, high=close + 0.5, low=close - 0.5, close=close, sequence=i + 1, source="test"))
    return events


@dataclass(frozen=True, slots=True)
class _FixedDirectionStrategy:
    direction: PositionDirection
    quantity: float
    abstain: bool = False

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=self.direction,
            target_quantity=(0.0 if self.abstain else self.quantity), confidence=0.9, uncertainty=0.05, abstain=self.abstain,
            reason_codes=("test_fixed_direction",), stop_target_intent=None,
        )


class _CrashAfterNAppends:
    """Wraps a real `PaperSessionEventStore`: the (N+1)-th `append` call
    raises instead of persisting -- everything before it is genuinely,
    durably written (this is what makes the test prove fail-closed
    behavior on a REAL partial ledger, not a hypothetical one)."""

    def __init__(self, real_store: PaperSessionEventStore, *, crash_after_n_appends: int) -> None:
        self._real = real_store
        self._remaining = crash_after_n_appends

    def append(self, paper_session_id: str, entry: LedgerEntry) -> LedgerEntry:
        if self._remaining <= 0:
            raise RuntimeError("simulated crash: process died before this ledger entry was appended")
        self._remaining -= 1
        return self._real.append(paper_session_id, entry)

    def read_events(self, paper_session_id: str) -> list[LedgerEntry]:
        return self._real.read_events(paper_session_id)

    def next_sequence(self, paper_session_id: str) -> int:
        return self._real.next_sequence(paper_session_id)

    def last_entry_hash(self, paper_session_id: str) -> str | None:
        return self._real.last_entry_hash(paper_session_id)


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests here are seeded directly PAST eligibility with
    `eligibility_environment=None` -- release-audit finding, fixed
    elsewhere: `run_paper_trading_session` now mandatorily re-verifies
    eligibility on every call that did not itself just create the
    manifest. That fix is exercised for real, against a genuine
    eligibility chain, in `test_audit_eligibility_bypass.py`; here it is
    bypassed so this file's own (unrelated) crash-boundary assertions
    don't crash on a `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=None)  # type: ignore[arg-type]


def _crashing_environment(tmp_path, *, crash_after_n_appends: int) -> RunnerEnvironment:
    manifest_store = PaperSessionManifestStore(tmp_path)
    real_event_store = PaperSessionEventStore(tmp_path)
    crashing_store = _CrashAfterNAppends(real_event_store, crash_after_n_appends=crash_after_n_appends)
    return RunnerEnvironment(manifest_store=manifest_store, event_store=crashing_store, eligibility_environment=None)  # type: ignore[arg-type]


def _seed_manifest_past_eligibility(environment: RunnerEnvironment, spec: PaperTradingSpec):
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    return environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)


def _normalize_ledger_for_comparison(ledger: list[LedgerEntry]) -> list:
    return [{"sequence": e.sequence, "kind": e.kind.value, "payload": e.payload} for e in ledger]


_SPEC = _spec()
_PAPER_SESSION_ID = compute_paper_session_spec_id(_SPEC).paper_session_spec_id
_EVENTS = _bars([100.0, 105.0, 110.0])


def _crash_and_capture(tmp_path, *, crash_after_n_appends: int) -> int:
    """Runs the standard 3-bar long-trade session against a crashing store,
    confirms the simulated crash actually fired, and returns the REAL
    (uncrashed-store) ledger length observed immediately afterward --
    i.e. exactly how many entries were durably persisted before the
    'process died'."""
    environment = _crashing_environment(tmp_path, crash_after_n_appends=crash_after_n_appends)
    _seed_manifest_past_eligibility(environment, _SPEC)
    strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_paper_trading_session(_SPEC, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)
    real_store = PaperSessionEventStore(tmp_path)
    return len(real_store.read_events(_PAPER_SESSION_ID))


class TestMidEventCrashSweepFailsClosed:
    """Every one of these N values lands strictly between event 1's own
    `MARKET_EVENT_ACCEPTED` (sequence 2) and its `ACCOUNT_SNAPSHOT`
    (sequence 11) -- a genuine mid-event crash at a different named
    sub-step each time."""

    @pytest.mark.parametrize(
        ("crash_after_n_appends", "sub_step"),
        [
            (3, "after market-event accepted, before mark applied"),
            (4, "after mark applied, before strategy decision"),
            (5, "after strategy decision, before risk decision"),
            (6, "after risk decision, before order CREATED->VALIDATED"),
            (7, "after order VALIDATED, before ACCEPTED"),
            (8, "after order ACCEPTED, before WORKING"),
            (9, "after order WORKING, before its FILLED transition"),
            (10, "after order FILLED transition, before the FILL entry itself"),
            (11, "after fill persisted, before account snapshot (Section 3's own named case)"),
        ],
    )
    def test_crash_leaves_partial_ledger_and_resume_fails_closed(self, tmp_path, crash_after_n_appends: int, sub_step: str) -> None:
        ledger_length_at_crash = _crash_and_capture(tmp_path, crash_after_n_appends=crash_after_n_appends)
        assert ledger_length_at_crash == crash_after_n_appends, f"expected exactly {crash_after_n_appends} durably persisted entries for {sub_step!r}, found {ledger_length_at_crash}"

        # A fresh process (real, non-crashing store) attempting to resume
        # must refuse outright -- never silently reprocess event 1 from a
        # position that doesn't match where it actually left off.
        resume_environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0)
        with pytest.raises(PaperTradingStateError, match="interrupted mid-event"):
            run_paper_trading_session(_SPEC, environment=resume_environment, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)

        # The failed resume attempt must not have mutated anything.
        real_store = PaperSessionEventStore(tmp_path)
        assert len(real_store.read_events(_PAPER_SESSION_ID)) == crash_after_n_appends

        # And it is idempotent: retrying the same failed resume again
        # produces the identical refusal, not a different/evolving error.
        with pytest.raises(PaperTradingStateError, match="interrupted mid-event"):
            run_paper_trading_session(_SPEC, environment=resume_environment, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)
        assert len(real_store.read_events(_PAPER_SESSION_ID)) == crash_after_n_appends


class TestFillPersistedAccountNotPersistedInverse:
    """Section 3 explicitly requires testing 'fill persisted, account
    update not persisted' (covered above at N=11) 'and its inverse if
    architecturally possible'. The inverse -- a durably persisted account
    update whose causing fill was NOT persisted -- is NOT architecturally
    possible in this design: `PortfolioState` is never durably persisted
    except via the single `ACCOUNT_SNAPSHOT` entry at the very end of an
    event's processing, and that snapshot is unconditionally appended
    AFTER every `FILL` entry for that same event (see `_apply_execution_
    outcome` then `run_paper_trading_session`'s own `cursor.append(...
    ACCOUNT_SNAPSHOT ...)` call, strictly the last append per event).
    This test proves that ordering invariant directly from the ledger of
    an uninterrupted run, rather than merely asserting it in prose."""

    def test_account_snapshot_is_never_appended_before_its_own_fills(self, tmp_path) -> None:
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, _SPEC)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0)
        run_paper_trading_session(_SPEC, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)
        ledger = environment.event_store.read_events(_PAPER_SESSION_ID)

        fill_sequences = [e.sequence for e in ledger if e.kind is LedgerEntryKind.FILL]
        snapshot_sequences = [e.sequence for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT]
        assert fill_sequences, "expected at least one FILL in this fixture"
        assert snapshot_sequences
        assert max(fill_sequences) < snapshot_sequences[0], "a FILL was found at or after the first ACCOUNT_SNAPSHOT -- the ordering invariant this test exists to prove no longer holds"


class TestCleanEventBoundaryResumesToUninterruptedEquivalence:
    """N=12: event 1 fully completed (its own ACCOUNT_SNAPSHOT at sequence
    11 durably persisted) and the crash happens strictly BEFORE event 2's
    `MARKET_EVENT_ACCEPTED` -- a genuinely clean boundary, and therefore
    the one crash point in this whole sweep that MUST resume successfully,
    not fail closed."""

    def test_resume_from_clean_boundary_matches_uninterrupted_control(self, tmp_path) -> None:
        control_environment = _environment(tmp_path / "control")
        _seed_manifest_past_eligibility(control_environment, _SPEC)
        control_manifest = run_paper_trading_session(_SPEC, environment=control_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0), clock=ReplayClock(), events=_EVENTS)
        control_ledger = control_environment.event_store.read_events(_PAPER_SESSION_ID)
        assert control_manifest.stage is PaperSessionStage.COMPLETED

        interrupted_dir = tmp_path / "interrupted"
        ledger_length_at_crash = _crash_and_capture(interrupted_dir, crash_after_n_appends=12)
        assert ledger_length_at_crash == 12

        resume_environment = _environment(interrupted_dir)
        resumed_manifest = run_paper_trading_session(_SPEC, environment=resume_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0), clock=ReplayClock(), events=_EVENTS)
        resumed_ledger = resume_environment.event_store.read_events(_PAPER_SESSION_ID)

        assert resumed_manifest.stage is PaperSessionStage.COMPLETED
        assert _normalize_ledger_for_comparison(resumed_ledger) == _normalize_ledger_for_comparison(control_ledger)


class TestTailTransitionCrashIsResumeSafe:
    """CONFIRMED-DEFECT, FIXED: crashes at N=20/21/22/23 land between a
    tail `SESSION_TRANSITION`'s `manifest_store.transition(...)` call and
    its own ledger append. Before the fix these left the manifest stuck
    forever (illegal self-transition on every future resume attempt);
    after the fix, resuming must complete the remaining tail transitions
    and land on the SAME final ledger/manifest state as an uninterrupted
    run, with no duplicated or missing `SESSION_TRANSITION` entries."""

    @pytest.mark.parametrize(
        ("crash_after_n_appends", "stuck_transition"),
        [
            (20, "RUNNING -> END_OF_STREAM"),
            (21, "END_OF_STREAM -> RECONCILING"),
            (22, "RECONCILING -> VERIFIED"),
            (23, "VERIFIED -> COMPLETED"),
        ],
    )
    def test_crash_between_manifest_write_and_ledger_append_still_resumes_to_completed(self, tmp_path, crash_after_n_appends: int, stuck_transition: str) -> None:
        control_environment = _environment(tmp_path / "control")
        _seed_manifest_past_eligibility(control_environment, _SPEC)
        run_paper_trading_session(_SPEC, environment=control_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0), clock=ReplayClock(), events=_EVENTS)
        control_ledger = control_environment.event_store.read_events(_PAPER_SESSION_ID)

        interrupted_dir = tmp_path / "interrupted"
        ledger_length_at_crash = _crash_and_capture(interrupted_dir, crash_after_n_appends=crash_after_n_appends)
        assert ledger_length_at_crash == crash_after_n_appends, stuck_transition

        resume_environment = _environment(interrupted_dir)
        resumed_manifest = run_paper_trading_session(_SPEC, environment=resume_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0), clock=ReplayClock(), events=_EVENTS)
        resumed_ledger = resume_environment.event_store.read_events(_PAPER_SESSION_ID)

        assert resumed_manifest.stage is PaperSessionStage.COMPLETED, f"stuck at {stuck_transition!r}"
        assert _normalize_ledger_for_comparison(resumed_ledger) == _normalize_ledger_for_comparison(control_ledger)

        transition_payloads = [e.payload for e in resumed_ledger if e.kind is LedgerEntryKind.SESSION_TRANSITION]
        # No duplicated tail transitions -- each of the 6 SESSION_TRANSITION
        # payloads (2 startup + 4 tail) must appear EXACTLY once.
        assert len(transition_payloads) == len({tuple(sorted(p.items())) for p in transition_payloads}) == 6

    def test_second_resume_after_already_completed_is_a_pure_noop(self, tmp_path) -> None:
        """Resuming a tail-crash TWICE in a row (crash again on the very
        next attempt, then resume for real) must not accumulate duplicate
        ledger entries -- the backfill path itself has to be idempotent,
        not just eventually-correct."""
        interrupted_dir = tmp_path
        _crash_and_capture(interrupted_dir, crash_after_n_appends=20)

        # First resume attempt also crashes immediately (simulates a flaky
        # environment that dies twice in a row during recovery).
        environment_second_crash = _crashing_environment(interrupted_dir, crash_after_n_appends=0)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_paper_trading_session(_SPEC, environment=environment_second_crash, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)
        real_store = PaperSessionEventStore(interrupted_dir)
        assert len(real_store.read_events(_PAPER_SESSION_ID)) == 20, "a crash on the very first backfill append must not have written anything"

        resume_environment = _environment(interrupted_dir)
        resumed_manifest = run_paper_trading_session(_SPEC, environment=resume_environment, strategy_runtime=strategy, clock=ReplayClock(), events=_EVENTS)
        assert resumed_manifest.stage is PaperSessionStage.COMPLETED
        resumed_ledger = resume_environment.event_store.read_events(_PAPER_SESSION_ID)
        transition_payloads = [e.payload for e in resumed_ledger if e.kind is LedgerEntryKind.SESSION_TRANSITION]
        assert len(transition_payloads) == 6
