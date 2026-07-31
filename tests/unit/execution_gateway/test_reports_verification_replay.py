"""Unit tests for `execution_gateway.reports` (Section 27),
`execution_gateway.verification` (Section 28), and
`execution_gateway.replay` (Section 30). Reuses the same paper-bridge
monkeypatch pattern as `test_execution_gateway_runner.py` so these tests exercise the
REAL dispatch/fill/reconciliation pipeline without needing a full
Milestone 6/7 chain (covered for real by the acceptance workflow)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
from quant_platform.execution_gateway.models import (
    AdapterKind,
    AuthorizationMode,
    ExecutionMode,
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
from quant_platform.execution_gateway.replay import replay_execution_session
from quant_platform.execution_gateway.reports import generate_execution_session_report
from quant_platform.execution_gateway.runner import RunnerEnvironment, run_execution_session
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
from quant_platform.execution_gateway.verification import verify_execution_session
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
_POLICY = ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True)


def _spec() -> ExecutionGatewaySpec:
    return ExecutionGatewaySpec(
        schema_version=1, execution_mode=ExecutionMode.TEST_ONLY, adapter_kind=AdapterKind.DETERMINISTIC_DUMMY, paper_session_id=_SHA_PAPER_SESSION,
        paper_trading_spec_id=_SHA_PAPER_SPEC, promotion_decision_id=_SHA_PROMOTION, instrument_spec_id=_SHA_INSTRUMENT,
        sequencing_policy=SequencingPolicySpec(policy=SequencingPolicyKind.STRICT_SEQUENCE),
        idempotency_policy=IdempotencyPolicySpec(durable_evidence_required=True, max_safe_retry_attempts=3),
        recovery_policy=RecoveryPolicySpec(max_replay_events=1000, unknown_resolution_timeout_events=50), reconciliation_policy=_POLICY,
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


class _FakePaperManifestStore:
    def load(self, paper_session_id: str) -> object:
        return object()


@pytest.fixture(autouse=True)
def _bypass_paper_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_platform.execution_gateway.runner as runner_module

    monkeypatch.setattr(runner_module, "require_paper_trading_eligibility", lambda spec, *, environment: None)  # noqa: ARG005
    monkeypatch.setattr("quant_platform.execution_gateway.paper_bridge._load_paper_trading_spec", lambda environment, manifest: object())  # noqa: ARG005

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
            execution_authorization_id="4" * 64, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION, paper_session_id=_SHA_PAPER_SESSION,
            paper_order_id=paper_order.order_id, execution_gateway_spec_id=execution_gateway_spec.paper_trading_spec_id,
            authorized_quantity=Decimal(str(paper_order.quantity)), authorized_side=paper_order.side, issued_sequence=created_sequence, source_verification_id="5" * 64,
        )
        return intent, authorization

    monkeypatch.setattr(runner_module, "execution_intent_from_paper_order", _fake_intent_from_order)


def _portfolio_risk_context(store_root) -> PortfolioRiskGatewayContext:
    """An always-approving context (no configured limits) -- this file's
    own tests exercise reports/verification/replay, not portfolio-risk
    denial/reservation behavior (covered by the dedicated Phase 4
    integration test file)."""
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
        store=PortfolioRiskLedgerStore(store_root), portfolio_id="test-portfolio", portfolio_snapshot=portfolio, price_snapshot=price,
        risk_spec=PortfolioRiskSpec(schema_version=1, policy=policy), portfolio_halted=False, consecutive_losses=0,
    )


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = ExecutionSessionManifestStore(tmp_path)
    event_store = ExecutionSessionEventStore(tmp_path)
    bridge_environment = PaperBridgeEnvironment(manifest_store=_FakePaperManifestStore(), event_store=None, artifact_store=None, eligibility_environment=None)  # type: ignore[arg-type]
    return RunnerEnvironment(
        manifest_store=manifest_store, event_store=event_store, paper_bridge_environment=bridge_environment,
        portfolio_risk_context=_portfolio_risk_context(tmp_path / "portfolio_risk"),
    )


class TestReports:
    def test_report_reflects_a_real_fill(self, tmp_path) -> None:
        from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter

        environment = _environment(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        manifest = run_execution_session(_spec(), environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        ledger = environment.event_store.read_events(manifest.execution_session_id)
        report = generate_execution_session_report(execution_session_id=manifest.execution_session_id, ledger=ledger)
        assert report.sections["FillSummary"]["fill_count"] == 1  # type: ignore[index]
        assert report.sections["OrderSummary"]["final_state_counts"] == {"filled": 1}  # type: ignore[index]
        # The report's own digest is recomputed fresh from whatever ledger
        # state exists AT REPORT TIME (here, including the terminal
        # EXECUTION_SESSION_COMPLETED entry) -- it is deliberately a
        # DIFFERENT checkpoint than `manifest.semantic_digest` (computed
        # BEFORE that entry is appended, to avoid a self-referential
        # hash), but must itself be stable across repeated calls.
        report_again = generate_execution_session_report(execution_session_id=manifest.execution_session_id, ledger=ledger)
        assert report.sections["ExecutionSessionSummary"]["final_semantic_digest"] == report_again.sections["ExecutionSessionSummary"]["final_semantic_digest"]  # type: ignore[index]
        assert report.sections["ExecutionSessionSummary"]["final_semantic_digest"] != manifest.semantic_digest  # type: ignore[index]

    def test_report_never_raises_on_empty_ledger(self) -> None:
        report = generate_execution_session_report(execution_session_id="a" * 64, ledger=[])
        assert report.sections["OrderSummary"]["total_orders"] == 0  # type: ignore[index]


class TestVerification:
    def test_clean_session_verifies_with_no_critical_issues(self, tmp_path) -> None:
        from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter

        spec = _spec()
        environment = _environment(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        ledger = environment.event_store.read_events(manifest.execution_session_id)
        report = verify_execution_session(spec, execution_session_id=manifest.execution_session_id, ledger=ledger)
        assert not report.criticals, [i.message for i in report.criticals]

    def test_wrong_execution_session_id_flagged_as_critical(self, tmp_path) -> None:
        from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter

        spec = _spec()
        environment = _environment(tmp_path)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=DEFAULT_DUMMY_BROKER_SCENARIO, starting_cash=Decimal("100000"))
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[_paper_order()], market_events=[tick], event_time=_NOW)
        ledger = environment.event_store.read_events(manifest.execution_session_id)
        report = verify_execution_session(spec, execution_session_id="f" * 64, ledger=ledger)
        codes = {i.code for i in report.criticals}
        assert "spec_identity_mismatch" in codes


class TestReplayDeterminism:
    def test_two_independent_replays_produce_identical_digest(self, tmp_path) -> None:
        spec = _spec()
        bridge_environment = PaperBridgeEnvironment(manifest_store=_FakePaperManifestStore(), event_store=None, artifact_store=None, eligibility_environment=None)  # type: ignore[arg-type]
        tick = create_quote_event(instrument="EURUSD", event_time=_NOW, sequence=0, bid=1.0995, ask=1.1005, source="test")

        result_a = replay_execution_session(
            spec, storage_root=tmp_path / "a", paper_bridge_environment=bridge_environment, portfolio_risk_context=_portfolio_risk_context(tmp_path / "a" / "portfolio_risk"),
            paper_orders=[_paper_order()], market_events=[tick], adapter_id="dummy-1", event_time=_NOW,
        )
        result_b = replay_execution_session(
            spec, storage_root=tmp_path / "b", paper_bridge_environment=bridge_environment, portfolio_risk_context=_portfolio_risk_context(tmp_path / "b" / "portfolio_risk"),
            paper_orders=[_paper_order()], market_events=[tick], adapter_id="dummy-1", event_time=_NOW,
        )

        assert result_a.semantic_digest == result_b.semantic_digest
        assert result_a.execution_session_id == result_b.execution_session_id
        assert result_a.reconciliation_report.is_reconciled
        assert result_b.reconciliation_report.is_reconciled
        assert not result_a.verification_report.criticals
