"""Milestone 7, Sections 22-23: `run_paper_trading_session` orchestration.
Manifests are pre-seeded directly past `ELIGIBILITY_VERIFIED` (eligibility
itself is already exhaustively covered by `test_eligibility.py` -- these
tests exercise the RUNNER's own event-processing/ledger-persistence/kill-
switch logic, not the eligibility chain). Covers: a full happy-path
session ending COMPLETED with a filled long trade; abstention persisting
decisions without ever creating an order; resume-after-interruption
idempotency (re-running produces byte-identical final state, no duplicate
ledger entries); and a tight risk limit tripping the kill switch, halting
further trading and flattening the open position."""

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
from quant_platform.paper_trading.eligibility import EligibilityVerificationEnvironment
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    ClockMode,
    KillSwitchState,
    LedgerEntryKind,
    MarketEventMode,
    PaperSessionStage,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.runner import (
    RunnerEnvironment,
    pause_paper_session,
    run_paper_trading_session,
    run_shadow_session,
)
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
from quant_platform.paper_trading.strategy import (
    StrategyContext,
    StrategyDecision,
    create_strategy_decision,
)

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
    """Test double `StrategyRuntime`: always issues the SAME direction/
    quantity decision (or abstains), ignoring model/feature inputs
    entirely -- deterministic and trivially auditable."""

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


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """This file exercises the RUNNER's own event-processing/resume/
    kill-switch logic against manifests seeded directly PAST eligibility
    (with `eligibility_environment=None`, since eligibility itself is
    covered exhaustively by `test_eligibility.py`) -- release-audit
    finding, fixed elsewhere: `run_paper_trading_session`/`run_shadow_
    session` now mandatorily re-verify eligibility on every call that
    did not itself just create the manifest (previously resume skipped
    this entirely). That fix is exercised for real, against a genuine
    eligibility chain, in `test_audit_eligibility_bypass.py`; here it is
    bypassed so this file's own (unrelated) assertions don't crash on a
    `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)


def _seed_manifest_past_eligibility(environment: RunnerEnvironment, spec: PaperTradingSpec):
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    return environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)


def _normalize_ledger_for_comparison(ledger: list) -> list:
    """Compares only `(sequence, kind, payload)` -- `entry_id`/`checksum`/
    `previous_entry_hash` form a content-addressed HASH CHAIN, so even one
    entry with a wall-clock-derived field (a `SESSION_TRANSITION`'s own
    `event_time` -- Section 0's documented "excluding wall-clock/lock
    metadata" carve-out; there is no market event to attach a session-
    lifecycle transition to) cascades a difference through every
    subsequent entry's `previous_entry_hash`, even though nothing
    financially meaningful changed. Dropping the chain-linkage fields
    (and each entry's own `event_time`, deterministic or not, for
    uniformity) isolates the property that actually matters here: the
    exact same sequence of decisions/orders/fills/marks, in the same
    order, with the same values."""
    return [{"sequence": e.sequence, "kind": e.kind.value, "payload": e.payload} for e in ledger]


class TestHappyPathSession:
    def test_long_trade_completes_session(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=2.0)
        events = _bars([100.0, 105.0, 110.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        from quant_platform.paper_trading.models import PaperSessionStage

        assert manifest.stage is PaperSessionStage.COMPLETED
        assert manifest.completed_at is not None

        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        kinds = [e.kind for e in ledger]
        assert LedgerEntryKind.MARKET_EVENT_ACCEPTED in kinds
        assert LedgerEntryKind.STRATEGY_DECISION in kinds
        assert LedgerEntryKind.ORDER_STATE_EVENT in kinds
        assert LedgerEntryKind.FILL in kinds
        assert LedgerEntryKind.ACCOUNT_SNAPSHOT in kinds

        final_snapshot_payload = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot_payload["positions"]["X"]["signed_quantity"] == 2.0

    def test_abstention_persists_decisions_but_creates_no_orders(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.FLAT, quantity=0.0, abstain=True)
        events = _bars([100.0, 101.0, 102.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        from quant_platform.paper_trading.models import PaperSessionStage

        assert manifest.stage is PaperSessionStage.COMPLETED
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        kinds = [e.kind for e in ledger]
        assert LedgerEntryKind.STRATEGY_DECISION in kinds
        assert LedgerEntryKind.ORDER_STATE_EVENT not in kinds
        assert LedgerEntryKind.FILL not in kinds
        decision_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.STRATEGY_DECISION]
        assert all(p["abstain"] is True for p in decision_payloads)


class TestResumeIdempotency:
    def test_rerunning_completed_session_is_a_no_op(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0)
        events = _bars([100.0, 101.0])

        first_manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger_after_first = environment.event_store.read_events(paper_session_id)

        second_manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        ledger_after_second = environment.event_store.read_events(paper_session_id)

        assert second_manifest == first_manifest
        assert ledger_after_second == ledger_after_first

    def test_manifest_stuck_running_with_clean_event_boundary_resumes_and_matches_uninterrupted_run(self, tmp_path) -> None:
        """Genuinely simulates a process crash BETWEEN two events (a clean
        boundary: the manifest never reached END_OF_STREAM/COMPLETED, but
        every event so far has a matching ACCOUNT_SNAPSHOT) by manually
        appending the first 2 events' worth of ledger entries via one
        real run's own OWN ledger (copied into a fresh environment) with
        the manifest rolled back to RUNNING, then calling the runner AGAIN
        with the FULL event list. The result must match an uninterrupted
        single-call run byte-for-byte."""
        from quant_platform.paper_trading.models import PaperSessionStage

        spec = _spec()
        events = _bars([100.0, 102.0, 104.0, 106.0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        uninterrupted_environment = _environment(tmp_path / "uninterrupted")
        _seed_manifest_past_eligibility(uninterrupted_environment, spec)
        run_paper_trading_session(spec, environment=uninterrupted_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events)
        uninterrupted_ledger = uninterrupted_environment.event_store.read_events(paper_session_id)

        # Build a "crashed after 2 clean events" ledger prefix by running
        # the first 2 events for real (a clean boundary, since a bounded
        # sub-stream of a deterministic pipeline always ends on a
        # completed event), then splicing those persisted entries into a
        # FRESH environment whose manifest is manually rolled back to
        # RUNNING (undoing the first run's own COMPLETED finalization,
        # exactly what a real crash would have left behind).
        prefix_environment = _environment(tmp_path / "prefix_source")
        _seed_manifest_past_eligibility(prefix_environment, spec)
        run_paper_trading_session(spec, environment=prefix_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events[:2])
        prefix_ledger_full = prefix_environment.event_store.read_events(paper_session_id)
        # A truncated 2-event sub-stream still runs to (bogus) COMPLETED on
        # its own -- drop everything from the LAST ACCOUNT_SNAPSHOT onward
        # (its own END_OF_STREAM/RECONCILING/VERIFIED/COMPLETED finalization
        # path), keeping only the entries a REAL crash mid-RUNNING would
        # actually have left behind.
        last_snapshot_index = max(i for i, e in enumerate(prefix_ledger_full) if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
        prefix_ledger = prefix_ledger_full[: last_snapshot_index + 1]

        resumed_environment = _environment(tmp_path / "resumed")
        resumed_environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
        resumed_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        resumed_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        resumed_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)
        for entry in prefix_ledger:
            resumed_environment.event_store.append(paper_session_id, entry)

        resumed_manifest = run_paper_trading_session(spec, environment=resumed_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events)
        resumed_ledger = resumed_environment.event_store.read_events(paper_session_id)

        assert resumed_manifest.stage is PaperSessionStage.COMPLETED
        # SESSION_TRANSITION entries embed a real wall-clock event_time
        # (Section 0's own documented "excluding wall-clock/lock
        # metadata" determinism carve-out -- there is no market event
        # time to attach a session-lifecycle transition to) -- these two
        # SEPARATE runs will legitimately differ on that one field. Every
        # other entry kind is driven entirely by deterministic market-
        # event timestamps and must match exactly.
        assert _normalize_ledger_for_comparison(resumed_ledger) == _normalize_ledger_for_comparison(uninterrupted_ledger)

    def test_mid_event_interruption_refuses_automatic_resume(self, tmp_path) -> None:
        """A ledger with a `MARKET_EVENT_ACCEPTED` entry that has no
        matching `ACCOUNT_SNAPSHOT` (a genuine mid-event crash) must
        refuse to resume automatically rather than risk corrupting the
        ledger -- fail-closed, per `_require_clean_event_boundary`."""
        spec = _spec()
        events = _bars([100.0, 102.0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        environment = _environment(tmp_path)
        environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
        environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)

        # Manually append ONLY a MARKET_EVENT_ACCEPTED entry, simulating a
        # crash that occurred before this event's ACCOUNT_SNAPSHOT (its
        # final step) was ever persisted.
        from quant_platform.paper_trading.persistence import create_ledger_entry

        dangling_entry = create_ledger_entry(
            session_id=paper_session_id, sequence=0, kind=LedgerEntryKind.MARKET_EVENT_ACCEPTED, payload=events[0].to_json_dict(),
            event_time=events[0].close_time, previous_entry_hash=None,
        )
        environment.event_store.append(paper_session_id, dangling_entry)

        try:
            run_paper_trading_session(spec, environment=environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events)
            raised = False
        except PaperTradingStateError:
            raised = True
        assert raised, "expected run_paper_trading_session to refuse resuming a mid-event-interrupted ledger"


class TestKillSwitch:
    def test_tight_drawdown_limit_halts_and_flattens(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        # Sharp adverse move after opening should trip the tight drawdown limit.
        events = _bars([100.0, 100.0, 50.0, 50.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        assert manifest.stage is PaperSessionStage.COMPLETED
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        assert halt_payloads, "expected at least one HALT_TRIGGERED ledger entry"
        to_states = {p["to_state"] for p in halt_payloads}
        assert KillSwitchState.HALTED.value in to_states

        # Position should have been flattened (or never grown further) after halt.
        final_snapshot_payload = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot_payload["positions"]["X"]["signed_quantity"] == 0.0


class TestShadowSession:
    """Section 19: `run_shadow_session` must never touch a real account --
    every outcome lands in a `SHADOW_OBSERVATION` ledger entry, never
    `ORDER_STATE_EVENT`/`FILL`/`ACCOUNT_SNAPSHOT`, and the manifest still
    reaches `COMPLETED` through the same stage machine."""

    def test_wrong_session_mode_raises(self, tmp_path) -> None:
        spec = _spec(session_mode=SessionMode.REPLAY_PAPER)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        events = _bars([100.0, 103.0])

        with pytest.raises(PaperTradingStateError, match="SHADOW_OBSERVATION"):
            run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

    def test_long_direction_shadow_session_completes_and_never_touches_real_account(self, tmp_path) -> None:
        spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        events = _bars([100.0, 103.0, 106.0, 109.0])

        manifest = run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        assert manifest.stage is PaperSessionStage.COMPLETED
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        assert not [e for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT]
        assert not [e for e in ledger if e.kind is LedgerEntryKind.FILL]
        assert not [e for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT]

        observations = [e.payload for e in ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]
        assert len(observations) == 4
        assert any(obs["hypothetical_fill_id"] is not None for obs in observations)
        decisions = [e.payload for e in ledger if e.kind is LedgerEntryKind.STRATEGY_DECISION]
        assert len(decisions) == 4

    def test_abstain_strategy_produces_all_none_observations(self, tmp_path) -> None:
        spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0, abstain=True)
        events = _bars([100.0, 100.0, 100.0])

        run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        observations = [e.payload for e in ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]
        assert len(observations) == 3
        assert all(obs["hypothetical_order_id"] is None for obs in observations)
        assert all(obs["hypothetical_fill_id"] is None for obs in observations)
        assert all(obs["counterfactual_realized_pnl_delta"] is None for obs in observations)

    def test_resuming_an_already_started_shadow_session_refused(self, tmp_path) -> None:
        spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        events = _bars([100.0, 103.0])
        run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        # The session already reached COMPLETED, so a second call is a
        # legitimate idempotent no-op (returns unchanged) -- exercise the
        # actual "already begun, not yet finished" refusal by constructing
        # a FRESH session and truncating its ledger's own manifest stage
        # back to RUNNING without clearing its MARKET_EVENT_ACCEPTED history.
        second_spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION, seed=1)
        second_environment = _environment(tmp_path)
        manifest = _seed_manifest_past_eligibility(second_environment, second_spec)
        second_paper_session_id = compute_paper_session_spec_id(second_spec).paper_session_spec_id
        second_environment.manifest_store.transition(second_paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        manifest = second_environment.manifest_store.transition(second_paper_session_id, target_stage=PaperSessionStage.RUNNING)
        assert manifest.stage is PaperSessionStage.RUNNING
        from quant_platform.paper_trading.persistence import create_ledger_entry

        entry = create_ledger_entry(
            session_id=second_paper_session_id, sequence=second_environment.event_store.next_sequence(second_paper_session_id),
            kind=LedgerEntryKind.MARKET_EVENT_ACCEPTED, payload=events[0].to_json_dict(), event_time=events[0].close_time,
            previous_entry_hash=second_environment.event_store.last_entry_hash(second_paper_session_id),
        )
        second_environment.event_store.append(second_paper_session_id, entry)

        with pytest.raises(PaperTradingStateError, match="not supported"):
            run_shadow_session(second_spec, environment=second_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)


class TestPause:
    """Section 23: `pause_paper_session` durably transitions a RUNNING
    session to PAUSED; a subsequent `run_paper_trading_session` call
    transitions it straight back to RUNNING and continues from its own
    last completed event."""

    def test_pause_requires_running_stage(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        manifest = _seed_manifest_past_eligibility(environment, spec)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        assert manifest.stage is PaperSessionStage.ELIGIBILITY_VERIFIED

        with pytest.raises(PaperTradingStateError, match="not RUNNING"):
            pause_paper_session(environment, paper_session_id)

    def test_pause_then_resume_matches_uninterrupted_run(self, tmp_path) -> None:
        spec = _spec()
        events = _bars([100.0, 102.0, 104.0, 106.0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        uninterrupted_environment = _environment(tmp_path / "uninterrupted")
        _seed_manifest_past_eligibility(uninterrupted_environment, spec)
        run_paper_trading_session(spec, environment=uninterrupted_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events)
        uninterrupted_ledger = uninterrupted_environment.event_store.read_events(paper_session_id)

        # Build a clean "paused after 2 events" ledger prefix exactly like
        # TestResumeIdempotency's own crash-simulation technique.
        prefix_environment = _environment(tmp_path / "prefix_source")
        _seed_manifest_past_eligibility(prefix_environment, spec)
        run_paper_trading_session(spec, environment=prefix_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events[:2])
        prefix_ledger_full = prefix_environment.event_store.read_events(paper_session_id)
        last_snapshot_index = max(i for i, e in enumerate(prefix_ledger_full) if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
        prefix_ledger = prefix_ledger_full[: last_snapshot_index + 1]

        paused_environment = _environment(tmp_path / "paused")
        paused_environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
        paused_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        paused_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        paused_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)
        for entry in prefix_ledger:
            paused_environment.event_store.append(paper_session_id, entry)

        paused_manifest = pause_paper_session(paused_environment, paper_session_id)
        assert paused_manifest.stage is PaperSessionStage.PAUSED
        ledger_after_pause = paused_environment.event_store.read_events(paper_session_id)
        pause_transitions = [e for e in ledger_after_pause if e.kind is LedgerEntryKind.SESSION_TRANSITION and e.payload == {"from_stage": "running", "to_stage": "paused"}]
        assert len(pause_transitions) == 1

        resumed_manifest = run_paper_trading_session(spec, environment=paused_environment, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=1.0), clock=ReplayClock(), events=events)
        resumed_ledger = paused_environment.event_store.read_events(paper_session_id)

        assert resumed_manifest.stage is PaperSessionStage.COMPLETED
        # The pause/resume cycle legitimately adds exactly two extra
        # SESSION_TRANSITION entries (running->paused, paused->running)
        # the uninterrupted run never has -- filter those out on both
        # sides before comparing everything else (every financially
        # meaningful entry: market events, decisions, orders, fills,
        # marks, account snapshots) for an exact match.
        non_transition_resumed = [e for e in resumed_ledger if e.kind is not LedgerEntryKind.SESSION_TRANSITION]
        non_transition_uninterrupted = [e for e in uninterrupted_ledger if e.kind is not LedgerEntryKind.SESSION_TRANSITION]
        resumed_kinds_payloads = [(e.kind, e.payload) for e in non_transition_resumed]
        uninterrupted_kinds_payloads = [(e.kind, e.payload) for e in non_transition_uninterrupted]
        assert resumed_kinds_payloads == uninterrupted_kinds_payloads
        transition_count_resumed = sum(1 for e in resumed_ledger if e.kind is LedgerEntryKind.SESSION_TRANSITION)
        transition_count_uninterrupted = sum(1 for e in uninterrupted_ledger if e.kind is LedgerEntryKind.SESSION_TRANSITION)
        assert transition_count_resumed == transition_count_uninterrupted + 2
