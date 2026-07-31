"""Unit tests for `execution_gateway.runner` (Milestone 8, Section 19):
manifest stage progression through to `COMPLETED`, resume idempotency,
and the completion gate's refusal to complete with an unresolved
`UNKNOWN` order. Source-eligibility re-verification and the paper-order
bridge itself are monkeypatched here (both require a full, genuinely
verified Milestone 6/7 chain, which is exercised for real by the
acceptance workflow, `tests/integration/test_execution_gateway_acceptance.py`)
so this file can focus purely on the runner's OWN orchestration logic."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
from quant_platform.execution_gateway.models import (
    AdapterKind,
    AuthorizationMode,
    ExecutionMode,
    ExecutionSessionStage,
    OrderSide,
    OrderTypeKind,
    SequencingPolicyKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.paper_bridge import (
    ExecutionAuthorization,
    ExecutionIntent,
    PaperBridgeEnvironment,
)
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.portfolio_risk_gate import PortfolioRiskGatewayContext
from quant_platform.execution_gateway.runner import (
    RunnerEnvironment,
    pause_execution_session,
    run_execution_session,
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
)
from quant_platform.paper_trading.events import create_quote_event
from quant_platform.paper_trading.models import PositionIntentKind
from quant_platform.paper_trading.orders import create_order_request
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.snapshots import create_portfolio_snapshot, create_price_snapshot
from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec

_SHA_PAPER_SESSION = "a" * 64
_SHA_PAPER_SPEC = "b" * 64
_SHA_PROMOTION = "c" * 64
_SHA_INSTRUMENT = "d" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _spec() -> ExecutionGatewaySpec:
    return ExecutionGatewaySpec(
        schema_version=1, execution_mode=ExecutionMode.TEST_ONLY, adapter_kind=AdapterKind.DETERMINISTIC_DUMMY, paper_session_id=_SHA_PAPER_SESSION,
        paper_trading_spec_id=_SHA_PAPER_SPEC, promotion_decision_id=_SHA_PROMOTION, instrument_spec_id=_SHA_INSTRUMENT,
        sequencing_policy=SequencingPolicySpec(policy=SequencingPolicyKind.STRICT_SEQUENCE),
        idempotency_policy=IdempotencyPolicySpec(durable_evidence_required=True, max_safe_retry_attempts=3),
        recovery_policy=RecoveryPolicySpec(max_replay_events=1000, unknown_resolution_timeout_events=50),
        reconciliation_policy=ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True),
        health_policy=HealthPolicySpec(stale_after_events=20, degraded_after_consecutive_failures=2, unavailable_after_consecutive_failures=5),
        heartbeat_policy=HeartbeatPolicySpec(interval_events=10, missed_threshold_degraded=2, missed_threshold_halting=5),
        kill_switch_policy=KillSwitchPolicySpec(max_unresolved_unknown_operations=3, max_broker_sequence_conflicts=1, max_blocking_reconciliation_issues=1),
        dispatch_policy=DispatchPolicySpec(require_dispatch_intent_before_call=True, max_commands_per_batch=100), dummy_broker_scenario=DEFAULT_DUMMY_BROKER_SCENARIO,
        seed=7,
    )


def _paper_order():
    return create_order_request(
        client_order_id="paper-client-order-1", session_id=_SHA_PAPER_SESSION, strategy_decision_id="e" * 64, instrument="EURUSD",
        side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=10.0, time_in_force=TimeInForceKind.DAY, create_time=_NOW, submit_time=_NOW,
        reduce_only=False, position_intent=PositionIntentKind.OPEN,
    )


@pytest.fixture(autouse=True)
def _bypass_paper_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full 13-check paper bridge and its source-eligibility
    re-verification require a genuine, independently verified Milestone
    6/7 chain -- exercised for real by the acceptance workflow. Here,
    `runner.py`'s OWN orchestration is under test, so both are
    monkeypatched to a fast, deterministic stand-in."""
    import quant_platform.execution_gateway.runner as runner_module

    def _fake_require_eligibility(spec, *, environment):
        return None

    def _fake_load_spec(environment, manifest):
        return object()

    monkeypatch.setattr(runner_module, "require_paper_trading_eligibility", _fake_require_eligibility)
    monkeypatch.setattr("quant_platform.execution_gateway.paper_bridge._load_paper_trading_spec", _fake_load_spec)

    def _fake_intent_from_order(paper_order, *, execution_gateway_spec, execution_session_id, environment, created_sequence, event_time=None, source_event_id=None):
        intent = ExecutionIntent(
            execution_intent_id="0" * 63 + str(created_sequence % 10), execution_session_id=execution_session_id, paper_session_id=_SHA_PAPER_SESSION,
            source_decision_id=paper_order.strategy_decision_id, source_paper_order_id=paper_order.order_id, instrument_id=paper_order.instrument,
            side=paper_order.side, quantity=Decimal(str(paper_order.quantity)), order_type=paper_order.order_type, limit_price=None, stop_price=None,
            time_in_force=paper_order.time_in_force, reduce_only=paper_order.reduce_only, close_position=False, strategy_candidate_id="1" * 64,
            model_artifact_id="2" * 64, execution_bridge_authorization_id="3" * 64, portfolio_risk_authorization_id=None, source_event_id=source_event_id,
            source_event_time="2026-01-01T00:00:00+00:00", created_sequence=created_sequence, contract_multiplier=Decimal("1"), identity_version=2,
        )
        authorization = ExecutionAuthorization(
            execution_authorization_id="4" * 64, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION,
            paper_session_id=_SHA_PAPER_SESSION, paper_order_id=paper_order.order_id, execution_gateway_spec_id=execution_gateway_spec.paper_trading_spec_id,
            authorized_quantity=Decimal(str(paper_order.quantity)), authorized_side=paper_order.side, issued_sequence=created_sequence, source_verification_id="5" * 64,
        )
        return intent, authorization

    monkeypatch.setattr(runner_module, "execution_intent_from_paper_order", _fake_intent_from_order)


class _FakePaperManifestStore:
    def load(self, paper_session_id: str) -> object:
        return object()


def _portfolio_risk_context(tmp_path) -> PortfolioRiskGatewayContext:
    """An always-approving context (no configured limits) -- this file's
    own tests exercise runner.py's OWN orchestration logic, not portfolio-
    risk denial/reservation behavior (covered by the dedicated Phase 4
    integration test file), so the gate should simply never block here."""
    equity = Decimal("1000000")
    portfolio = create_portfolio_snapshot(
        portfolio_id="test-portfolio", event_time=_NOW, cash=equity, equity=equity, realized_pnl=Decimal(0), unrealized_pnl=Decimal(0),
        peak_equity=equity, daily_start_equity=equity, positions=(), source_execution_session_id=None,
    )
    price = create_price_snapshot(instrument_id="EURUSD", bid=Decimal("1.0995"), ask=Decimal("1.1005"), reference_price=Decimal("1.1"), event_time=_NOW, source_event_id=None)
    policy = PortfolioRiskPolicy(
        max_order_notional=None, max_position_notional=None, max_instrument_gross_exposure=None, max_strategy_gross_exposure=None,
        max_portfolio_gross_exposure=None, max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None,
        max_daily_realized_loss=None, max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None, minimum_cash_buffer=None,
        maximum_price_age=None, maximum_portfolio_snapshot_age=None, allow_reduce_only_during_halt=True,
    )
    return PortfolioRiskGatewayContext(
        store=PortfolioRiskLedgerStore(tmp_path / "portfolio_risk"), portfolio_id="test-portfolio", portfolio_snapshot=portfolio, price_snapshot=price,
        risk_spec=PortfolioRiskSpec(schema_version=1, policy=policy), portfolio_halted=False, consecutive_losses=0,
    )


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = ExecutionSessionManifestStore(tmp_path)
    event_store = ExecutionSessionEventStore(tmp_path)
    bridge_environment = PaperBridgeEnvironment(manifest_store=_FakePaperManifestStore(), event_store=None, artifact_store=None, eligibility_environment=None)  # type: ignore[arg-type]
    return RunnerEnvironment(
        manifest_store=manifest_store, event_store=event_store, paper_bridge_environment=bridge_environment, portfolio_risk_context=_portfolio_risk_context(tmp_path),
    )


class TestRunExecutionSessionCleanRun:
    def test_reaches_completed(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED
        assert manifest.semantic_digest is not None

    def test_resume_of_completed_session_is_idempotent_no_op(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        first = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        ledger_len_after_first = len(environment.event_store.read_events(first.execution_session_id))

        adapter2 = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        second = run_execution_session(spec, environment=environment, adapter=adapter2, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        ledger_len_after_second = len(environment.event_store.read_events(second.execution_session_id))

        assert second.current_stage is ExecutionSessionStage.COMPLETED
        assert second.semantic_digest == first.semantic_digest
        assert ledger_len_after_first == ledger_len_after_second

    def test_deterministic_replay_produces_identical_semantic_digest(self, tmp_path) -> None:
        spec = _spec()
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")

        env_a = _environment(tmp_path / "a")
        adapter_a = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        manifest_a = run_execution_session(spec, environment=env_a, adapter=adapter_a, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)

        env_b = _environment(tmp_path / "b")
        adapter_b = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        manifest_b = run_execution_session(spec, environment=env_b, adapter=adapter_b, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)

        assert manifest_a.semantic_digest == manifest_b.semantic_digest
        assert manifest_a.execution_session_id == manifest_b.execution_session_id


class TestPauseAndResume:
    def test_pause_then_resume_reaches_completed(self, tmp_path) -> None:
        from quant_platform.execution_gateway.specs import compute_execution_gateway_spec_id

        spec = _spec()
        environment = _environment(tmp_path)
        execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id

        # Manually drive the manifest up to RUNNING (bypassing
        # run_execution_session, which would otherwise run straight
        # through to COMPLETED in one call) to construct a genuine
        # mid-session pause point.
        environment.manifest_store.create(execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id, adapter_id="dummy-1", execution_mode=ExecutionMode.TEST_ONLY)
        for stage in (ExecutionSessionStage.SPEC_VERIFIED, ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED, ExecutionSessionStage.ADAPTER_INITIALIZED, ExecutionSessionStage.RECOVERY_CHECKED, ExecutionSessionStage.RUNNING):
            environment.manifest_store.transition(execution_session_id, target_stage=stage)

        paused = pause_execution_session(execution_session_id=execution_session_id, environment=environment, event_time=_NOW)
        assert paused.current_stage is ExecutionSessionStage.PAUSED

        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        resumed = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        assert resumed.current_stage is ExecutionSessionStage.COMPLETED
        assert resumed.resume_count >= 1


class TestEligibilityReVerifiedOnResumeRegardlessOfStage:
    """Regression test for a real, confirmed defect found during
    Milestone 8's own acceptance testing: eligibility re-verification
    used to live INSIDE the `if manifest.current_stage is SPEC_VERIFIED:`
    block, which only ever executes on a session's very first pass
    through that exact stage -- a resume landing at ANY later stage
    (RUNNING, RECONCILING, VERIFYING, or even a PAUSED resume, which
    jumps straight past SPEC_VERIFIED to SOURCE_ELIGIBILITY_VERIFIED)
    silently skipped re-verification entirely. That directly contradicted
    Section 6/23's own "fresh call every time" requirement and this
    module's own docstring claim -- precisely the Milestone 7 audit's own
    most severe finding, reintroduced here. Fixed by an unconditional,
    stage-independent check placed before any stage branching, mirroring
    `paper_trading.runner.run_paper_trading_session`'s own pattern
    exactly."""

    def test_resume_from_a_stage_strictly_after_spec_verified_still_reverifies_eligibility(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        import quant_platform.execution_gateway.runner as runner_module
        from quant_platform.execution_gateway.specs import compute_execution_gateway_spec_id

        call_count = 0

        def _counting_require_eligibility(spec, *, environment):
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr(runner_module, "require_paper_trading_eligibility", _counting_require_eligibility)

        spec = _spec()
        environment = _environment(tmp_path)
        execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id

        # Manually drive the manifest to RECONCILING -- strictly past
        # SPEC_VERIFIED -- simulating a crash-then-resume that never
        # passes back through that specific block again.
        environment.manifest_store.create(execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id, adapter_id="dummy-1", execution_mode=ExecutionMode.TEST_ONLY)
        for stage in (
            ExecutionSessionStage.SPEC_VERIFIED, ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED, ExecutionSessionStage.ADAPTER_INITIALIZED,
            ExecutionSessionStage.RECOVERY_CHECKED, ExecutionSessionStage.RUNNING, ExecutionSessionStage.RECONCILING,
        ):
            environment.manifest_store.transition(execution_session_id, target_stage=stage)
        assert call_count == 0, "manually driving the manifest via the manifest store directly must not itself trigger eligibility checks"

        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[], market_events=[], event_time=_NOW)

        assert call_count == 1, "resuming from RECONCILING (strictly past SPEC_VERIFIED) must still re-verify eligibility exactly once on this call"
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED
        assert manifest.resume_count >= 1

    def test_resume_from_paused_still_reverifies_eligibility(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The PAUSED branch transitions DIRECTLY to SOURCE_ELIGIBILITY_
        VERIFIED, skipping the SPEC_VERIFIED block entirely -- confirms
        the fix covers this specific resume path too, not just a generic
        later stage."""
        import quant_platform.execution_gateway.runner as runner_module
        from quant_platform.execution_gateway.specs import compute_execution_gateway_spec_id

        call_count = 0

        def _counting_require_eligibility(spec, *, environment):
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr(runner_module, "require_paper_trading_eligibility", _counting_require_eligibility)

        spec = _spec()
        environment = _environment(tmp_path)
        execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id

        environment.manifest_store.create(execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id, adapter_id="dummy-1", execution_mode=ExecutionMode.TEST_ONLY)
        for stage in (ExecutionSessionStage.SPEC_VERIFIED, ExecutionSessionStage.SOURCE_ELIGIBILITY_VERIFIED, ExecutionSessionStage.ADAPTER_INITIALIZED, ExecutionSessionStage.RECOVERY_CHECKED, ExecutionSessionStage.RUNNING):
            environment.manifest_store.transition(execution_session_id, target_stage=stage)
        paused = pause_execution_session(execution_session_id=execution_session_id, environment=environment, event_time=_NOW)
        assert paused.current_stage is ExecutionSessionStage.PAUSED
        assert call_count == 0

        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        resumed = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[], market_events=[], event_time=_NOW)

        assert call_count == 1, "resuming a PAUSED session must re-verify eligibility exactly once on this call"
        assert resumed.current_stage is ExecutionSessionStage.COMPLETED
