"""Milestone 8, Section 33: real, bounded end-to-end acceptance workflow
for the broker-neutral deterministic execution gateway. Chains a REAL
Milestone 6/7 pipeline -- the identical `eligible_chain` fixture
`test_paper_trading_real_model_acceptance.py` already builds (a real
fitted logistic-regression model, walk-forward execution, calibration,
backtest, and Milestone 6's own statistical-robustness pipeline, reaching
a genuine `ELIGIBLE_FOR_PAPER_TRADING` promotion decision) -- through a
REAL, COMPLETED, independently-verified Milestone 7 paper session, then
bridges it into Milestone 8's execution gateway against the ONLY adapter
this milestone ships: `DeterministicDummyBrokerAdapter`. No MT5, no
FxPro, no real broker, no network call exists anywhere in this file or
what it exercises.

SCOPE NOTE, mirrored from `test_paper_trading_real_model_acceptance.py`'s
own convention: broker MECHANICS (fill semantics per order type/TIF,
sequencing classification, idempotent redispatch, ledger-tamper
detection, capability gating, ...) are already exhaustively covered,
corner case by corner case, by the unit suites under
`tests/unit/execution_gateway/` -- this file's unique job is proving
those mechanics compose correctly end-to-end against a GENUINE,
independently re-verified upstream chain, not re-deriving each mechanic
from scratch. To keep each of Section 33's 24 required scenarios crisp
and independently diagnosable, this file uses several small, isolated
execution sessions (distinguished by `seed`/scenario, each its own
content-addressed identity) rather than one large entangled run --
EVERY session still originates from the SAME real `paper_session_id` /
`paper_trading_spec_id` / instrument this module's real chain produced."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from tests.integration.test_paper_trading_real_model_acceptance import (
    _build_paper_spec,
    _build_replay_events,
    eligible_chain,  # noqa: F401 -- imported so pytest can resolve it as a fixture by name below, not directly referenced
)

from quant_platform.core.exceptions import ExecutionHaltError
from quant_platform.core.types import Timeframe
from quant_platform.execution_gateway.commands import (
    create_cancel_order_command,
    create_replace_order_command,
)
from quant_platform.execution_gateway.dispatcher import _append, process_broker_events
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.identity import compute_content_id
from quant_platform.execution_gateway.kill_switch import (
    ExecutionKillSwitchTriggerKind,
    authorize_dispatch,
    create_execution_kill_switch_transition_event,
    resolve_execution_kill_switch_state,
)
from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
from quant_platform.execution_gateway.models import (
    ExecutionKillSwitchState,
    ExecutionLedgerEntryKind,
    ExecutionOrderState,
    ExecutionSessionStage,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.paper_bridge import PaperBridgeEnvironment
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.portfolio_risk_gate import PortfolioRiskGatewayContext
from quant_platform.execution_gateway.reconciliation import (
    reconcile_execution_session,
    reconstruct_all_orders_from_ledger,
)
from quant_platform.execution_gateway.recovery import recover_unknown_orders
from quant_platform.execution_gateway.replay import replay_execution_session
from quant_platform.execution_gateway.runner import (
    RunnerEnvironment,
    authorize_cancel_or_reduce_only_submit,
    run_execution_session,
)
from quant_platform.execution_gateway.specs import (
    DispatchPolicySpec,
    DummyBrokerScenarioSpec,
    ExecutionGatewaySpec,
    HealthPolicySpec,
    HeartbeatPolicySpec,
    IdempotencyPolicySpec,
    KillSwitchPolicySpec,
    ReconciliationPolicySpec,
    RecoveryPolicySpec,
    RejectionRuleSpec,
    SequencingPolicySpec,
    compute_execution_gateway_spec_id,
)
from quant_platform.execution_gateway.state_machine import resolve_execution_order_state
from quant_platform.execution_gateway.states import compute_execution_order_id
from quant_platform.execution_gateway.verification import verify_execution_session
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml_cli import _load_paper_orders_for_session
from quant_platform.paper_trading.events import create_bar_event, create_end_of_stream_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import PositionIntentKind, SessionMode
from quant_platform.paper_trading.orders import create_order_request
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.runner import RunnerEnvironment as PaperRunnerEnvironment
from quant_platform.paper_trading.runner import create_paper_session, run_paper_trading_session
from quant_platform.paper_trading.specs import compute_paper_session_spec_id
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.snapshots import create_portfolio_snapshot, create_price_snapshot
from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec

_INSTRUMENT = "XAUUSD"
_STRATEGY_DECISION_ID = "a" * 64


def _default_sequencing_policy() -> SequencingPolicySpec:
    from quant_platform.execution_gateway.models import SequencingPolicyKind

    return SequencingPolicySpec(policy=SequencingPolicyKind.STRICT_SEQUENCE)


def _default_idempotency_policy() -> IdempotencyPolicySpec:
    return IdempotencyPolicySpec(durable_evidence_required=True, max_safe_retry_attempts=3)


def _default_recovery_policy() -> RecoveryPolicySpec:
    # `max_replay_events=100_000` matches `RecoveryPolicyConfigSchema`'s own
    # default exactly -- required so a spec built here via `_build_spec`
    # reproduces the IDENTICAL `execution_gateway_spec_id` a `--config` file
    # omitting this field would resolve to via the CLI (needed for the
    # cross-process determinism test below, which checks exactly that).
    return RecoveryPolicySpec(max_replay_events=100_000, unknown_resolution_timeout_events=100)


def _default_reconciliation_policy() -> ReconciliationPolicySpec:
    return ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True)


def _default_health_policy() -> HealthPolicySpec:
    return HealthPolicySpec(stale_after_events=20, degraded_after_consecutive_failures=2, unavailable_after_consecutive_failures=5)


def _default_heartbeat_policy() -> HeartbeatPolicySpec:
    return HeartbeatPolicySpec(interval_events=10, missed_threshold_degraded=2, missed_threshold_halting=5)


def _default_kill_switch_policy() -> KillSwitchPolicySpec:
    return KillSwitchPolicySpec(max_unresolved_unknown_operations=3, max_broker_sequence_conflicts=1, max_blocking_reconciliation_issues=1)


def _default_dispatch_policy() -> DispatchPolicySpec:
    return DispatchPolicySpec(require_dispatch_intent_before_call=True, max_commands_per_batch=500)


_DEFAULT_SCENARIO = DummyBrokerScenarioSpec(
    acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(),
    delayed_event_indices=(), out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None, heartbeat_failure_sequences=(),
    order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True, supports_idempotent_cancel=True,
    supports_idempotent_replace=True, seed=0,
)


@pytest.fixture(scope="module")
def real_paper_session(eligible_chain, tmp_path_factory: pytest.TempPathFactory) -> dict:  # noqa: F811 -- shadows the module-level import by design, this IS how pytest fixture parameter resolution works
    """Builds ONE real, COMPLETED, independently-verifiable Milestone 7
    paper session on top of the real `eligible_chain` -- every execution
    gateway scenario below bridges from THIS session's own identity."""
    chain = eligible_chain
    tmp_path = tmp_path_factory.mktemp("execution_gateway_acceptance_paper")
    # `create_paper_session` persists the source `PaperTradingSpec` via
    # `environment.eligibility_environment.artifact_store` (Milestone 7's
    # own single-shared-`ml_artifacts_root` convention, mirrored exactly
    # by every real CLI wiring in `ml_cli.py`) -- the paper session's OWN
    # manifest/event stores are rooted at THAT SAME artifact store's root
    # here too, so a later read-back (this module's own execution-gateway
    # bridge) finds the artifact where it was actually written, rather
    # than at a separate, unrelated tmp_path.
    shared_root = chain["eligibility_environment"].artifact_store.root
    paper_spec = _build_paper_spec(chain, session_mode=SessionMode.REPLAY_PAPER, maximum_order_quantity=1.2)
    paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id

    replay_path = tmp_path / "replay.jsonl"
    loaded_events = _build_replay_events(chain["df"], replay_path)

    manifest_store = PaperSessionManifestStore(shared_root)
    event_store = PaperSessionEventStore(shared_root)
    environment = PaperRunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=chain["eligibility_environment"])
    create_paper_session(paper_spec, environment=environment)
    from quant_platform.paper_trading.clock import ReplayClock
    from quant_platform.paper_trading.model_strategy import ModelStrategyRuntime

    strategy_runtime = ModelStrategyRuntime(
        strategy_identity=chain["backtest_id"], fitted_model=chain["fitted_model"], feature_names=("return_log_1", "return_log_5", "ma_distance_10", "candle_body_ratio"),
        long_threshold=0.55, short_threshold=0.45, target_quantity=1.0, confidence_scaled_sizing=True,
    )
    paper_manifest = run_paper_trading_session(paper_spec, environment=environment, strategy_runtime=strategy_runtime, clock=ReplayClock(), events=loaded_events)
    assert paper_manifest.stage.value == "completed"
    # `create_paper_session` now persists `spec` as a durable
    # `PAPER_TRADING_SPEC` artifact and records the reference on the
    # manifest (a real defect found and fixed during this milestone's own
    # acceptance testing -- see the final delivery report). `execution_
    # gateway.paper_bridge`'s own check 3 cross-checks `ExecutionGatewaySpec.
    # paper_trading_spec_id` against THIS EXACT persisted reference's
    # `content_hash` -- they must match precisely, so the real, persisted
    # value is used here, never an independently recomputed hash.
    assert paper_manifest.spec_reference is not None
    instrument_spec_id = compute_content_id("execution_gateway_instrument_spec", paper_spec.instrument.to_json_dict())
    paper_trading_spec_id = paper_manifest.spec_reference.content_hash

    return {
        "tmp_path": tmp_path, "paper_storage_root": shared_root, "paper_spec": paper_spec, "paper_session_id": paper_session_id,
        "paper_trading_spec_id": paper_trading_spec_id, "promotion_decision_id": chain["promotion_content_hash"],
        "instrument_spec_id": instrument_spec_id, "eligibility_environment": chain["eligibility_environment"], "df": chain["df"],
    }


def _paper_bridge_environment(real_paper_session: dict) -> PaperBridgeEnvironment:
    root = real_paper_session["paper_storage_root"]
    return PaperBridgeEnvironment(
        manifest_store=PaperSessionManifestStore(root), event_store=PaperSessionEventStore(root), artifact_store=MLArtifactStore(root),
        eligibility_environment=real_paper_session["eligibility_environment"],
    )


def _build_spec(real_paper_session: dict, *, seed: int, scenario: DummyBrokerScenarioSpec = _DEFAULT_SCENARIO) -> ExecutionGatewaySpec:
    from quant_platform.execution_gateway.models import (
        EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION,
        AdapterKind,
        ExecutionMode,
    )

    return ExecutionGatewaySpec(
        schema_version=EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION, execution_mode=ExecutionMode.TEST_ONLY, adapter_kind=AdapterKind.DETERMINISTIC_DUMMY,
        paper_session_id=real_paper_session["paper_session_id"], paper_trading_spec_id=real_paper_session["paper_trading_spec_id"],
        promotion_decision_id=real_paper_session["promotion_decision_id"], instrument_spec_id=real_paper_session["instrument_spec_id"],
        sequencing_policy=_default_sequencing_policy(), idempotency_policy=_default_idempotency_policy(), recovery_policy=_default_recovery_policy(),
        reconciliation_policy=_default_reconciliation_policy(), health_policy=_default_health_policy(), heartbeat_policy=_default_heartbeat_policy(),
        kill_switch_policy=_default_kill_switch_policy(), dispatch_policy=_default_dispatch_policy(), dummy_broker_scenario=scenario, seed=seed,
    )


def _portfolio_risk_context(storage_root: Path, *, positions: tuple = ()) -> PortfolioRiskGatewayContext:
    """An always-approving context (no configured limits) -- Milestone 9
    Phase 4's own dedicated integration test file exercises portfolio-
    risk denial/reservation/consumption behavior directly; this
    acceptance workflow's own job is proving the FULL Milestone 7->8
    bridge chain still works end to end with the mandatory gate now
    wired in, not re-testing the gate's own decision logic.

    `positions` -- see `_existing_positions_for_reduce_only_orders`'s own
    docstring: a real, unpredictable strategy can produce a genuine
    reduce-only order among the ones bridged here, and `evaluate_risk`'s
    own `missing_or_inconsistent_valuation_data`/`reduce_only_validity`
    checks correctly fail closed (`INCOHERENT_EVALUATION_STATE`) when no
    existing position backs a reduce-only request -- exactly the fail-
    closed behavior this milestone requires, not a defect in the gate.
    The FIXTURE must therefore supply a portfolio snapshot that honestly
    reflects a position exists wherever a reduce-only order needs one;
    defaulting to `()` remains correct for every other test in this file,
    which only ever bridges synthetic, always-opening orders. `cash` is
    adjusted down by the synthetic positions' own combined market value
    so `equity == cash + sum(position.market_value)` still reconciles
    (`PortfolioSnapshot`'s own required invariant, Phase 1) -- `equity`
    itself stays fixed at a comfortably large, constant figure regardless
    of how many positions are synthesized."""
    equity = Decimal("10000000")
    marked_position_value = sum((p.market_value for p in positions), start=Decimal(0))
    cash = equity - marked_position_value
    portfolio = create_portfolio_snapshot(
        portfolio_id="acceptance-portfolio", event_time=_EVENT_TIME, cash=cash, equity=equity, realized_pnl=Decimal(0), unrealized_pnl=Decimal(0),
        peak_equity=equity, daily_start_equity=equity, positions=positions, source_execution_session_id=None,
    )
    price = create_price_snapshot(
        instrument_id=_INSTRUMENT, bid=Decimal("1999"), ask=Decimal("2001"), reference_price=Decimal("2000"), event_time=_EVENT_TIME, source_event_id=None,
    )
    policy = PortfolioRiskPolicy(
        max_order_notional=None, max_position_notional=None, max_instrument_gross_exposure=None, max_strategy_gross_exposure=None,
        max_portfolio_gross_exposure=None, max_portfolio_net_exposure=None, max_concentration_fraction=None, max_leverage=None,
        max_daily_realized_loss=None, max_total_loss=None, max_drawdown_fraction=None, max_consecutive_losses=None, minimum_cash_buffer=None,
        maximum_price_age=None, maximum_portfolio_snapshot_age=None, allow_reduce_only_during_halt=True,
    )
    return PortfolioRiskGatewayContext(
        store=PortfolioRiskLedgerStore(storage_root), portfolio_id="acceptance-portfolio", portfolio_snapshot=portfolio, price_snapshot=price,
        risk_spec=PortfolioRiskSpec(schema_version=1, policy=policy), portfolio_halted=False, consecutive_losses=0,
    )


def _existing_positions_for_reduce_only_orders(paper_orders: list, *, strategy_id: str, contract_multiplier: Decimal) -> tuple:
    """Real, confirmed defect found and fixed via this phase's own
    acceptance-test run: a REAL strategy's own orders are unpredictable,
    and among them can be a genuine reduce-only close -- `evaluate_risk`
    correctly denies it (`INCOHERENT_EVALUATION_STATE`) against a
    synthetic ALWAYS-FLAT portfolio, since there is nothing to reduce.
    This is the gate working exactly as designed, not a bug in it -- the
    FIX belongs in the test fixture: synthesize a plausible pre-existing
    position, OPPOSITE side from the reduce-only order (a reduce-only
    order always reduces the OPPOSITE-side position), sized comfortably
    LARGER than the order's own quantity (so the trade classifies as
    REDUCING, never crossing through zero into a new direction, which
    Phase 2's own `classify_trade_risk` always counts as INCREASING
    regardless of resulting magnitude). `mark_price == average_entry_
    price` makes `unrealized_pnl == 0` trivially reconcile with
    `PositionSnapshot`'s own required invariant."""
    from quant_platform.portfolio_risk.snapshots import PositionSnapshot

    positions = []
    for order in paper_orders:
        if not order.reduce_only:
            continue
        opposite_side = OrderSide.SELL if order.side is OrderSide.BUY else OrderSide.BUY
        quantity = Decimal(str(order.quantity)) * Decimal(2)
        positions.append(PositionSnapshot(
            instrument_id=order.instrument, strategy_id=strategy_id, side=opposite_side, quantity=quantity,
            average_entry_price=Decimal("2000"), mark_price=Decimal("2000"), unrealized_pnl=Decimal(0), realized_pnl=Decimal(0),
            contract_multiplier=contract_multiplier,
        ))
    return tuple(positions)


def _runner_environment(real_paper_session: dict, execution_storage_root: Path, *, positions: tuple = ()) -> RunnerEnvironment:
    return RunnerEnvironment(
        manifest_store=ExecutionSessionManifestStore(execution_storage_root), event_store=ExecutionSessionEventStore(execution_storage_root),
        paper_bridge_environment=_paper_bridge_environment(real_paper_session),
        portfolio_risk_context=_portfolio_risk_context(execution_storage_root / "portfolio_risk", positions=positions),
    )


def _order(paper_session_id: str, *, client_order_id: str, side: OrderSide, order_type: OrderTypeKind, quantity: float, time_in_force: TimeInForceKind, event_time: datetime, reduce_only: bool = False, limit_price: float | None = None, stop_price: float | None = None):
    return create_order_request(
        client_order_id=client_order_id, session_id=paper_session_id, strategy_decision_id=_STRATEGY_DECISION_ID, instrument=_INSTRUMENT, side=side,
        order_type=order_type, quantity=quantity, time_in_force=time_in_force, create_time=event_time, submit_time=event_time, reduce_only=reduce_only,
        position_intent=(PositionIntentKind.CLOSE if reduce_only else PositionIntentKind.OPEN), limit_price=limit_price, stop_price=stop_price,
    )


def _bars(prices: list[float], *, start: datetime) -> list:
    events = []
    for i, price in enumerate(prices):
        open_time = start + timedelta(minutes=i)
        events.append(create_bar_event(instrument=_INSTRUMENT, interval=Timeframe.M1, open_time=open_time, open=price, high=price, low=price, close=price, sequence=i + 1, source="acceptance"))
    eos = create_end_of_stream_event(instrument=_INSTRUMENT, event_time=events[-1].close_time + timedelta(minutes=1), sequence=len(events) + 1, source="acceptance")
    return [*events, eos]


_EVENT_TIME = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)


# ==========================================================================
# Scenarios 1-2: long and short MARKET orders bridged from the REAL,
# independently-verified paper session's own real strategy decisions.
# Also covers scenario 22 (clean successful reconciliation) and lays the
# groundwork reused by scenario 23 (verification) below.
# ==========================================================================
class TestRealBridgeLongAndShortMarketOrders:
    def test_real_paper_orders_bridge_and_fill_end_to_end(self, real_paper_session: dict, tmp_path: Path) -> None:
        all_paper_orders = _load_paper_orders_for_session(
            type("Cfg", (), {"ml_artifacts_root": real_paper_session["paper_storage_root"]})(), real_paper_session["paper_session_id"],
        )
        assert len(all_paper_orders) > 0, "the real paper session must have produced at least one order"
        sides = {o.side for o in all_paper_orders}
        assert OrderSide.BUY in sides, "at least one real long (buy) order is required"
        assert OrderSide.SELL in sides, "at least one real short (sell) order is required"

        # PERFORMANCE FINDING (see the final delivery report's own
        # documented-limitation section): `execution_intent_from_paper_
        # order`'s check 4 fully re-verifies eligibility -- including
        # `verify_robustness`'s own from-scratch bootstrap/stress/regime
        # recomputation -- on EVERY bridged order, by design (Section 6,
        # mirroring the Milestone 7 audit's own "never cache eligibility"
        # fix). That is correct for a single dispatch, but bridging
        # DOZENS of real orders in one acceptance-test run multiplies an
        # already-nontrivial per-call cost by the order count, which
        # made the first attempt at this test run for 10+ hours without
        # finishing. Bridging one real BUY and one real SELL order here
        # (still genuinely FROM the real, independently-verified session
        # -- never synthesized) is enough to prove the bridge itself
        # works for both sides; it deliberately does not re-bridge every
        # order the real strategy produced.
        one_buy = next(o for o in all_paper_orders if o.side is OrderSide.BUY)
        one_sell = next(o for o in all_paper_orders if o.side is OrderSide.SELL)
        paper_orders = [one_buy, one_sell]

        market_events = _build_replay_events(real_paper_session["df"], tmp_path / "replay_for_bridge.jsonl")

        spec = _build_spec(real_paper_session, seed=1)
        storage_root = tmp_path / "session_root"
        # Real, confirmed defect found and fixed via this phase's own
        # acceptance-test run: the REAL strategy's own orders are
        # unpredictable and can include a genuine reduce-only close,
        # which `evaluate_risk` correctly denies against an always-flat
        # synthetic portfolio (nothing to reduce) -- see
        # `_existing_positions_for_reduce_only_orders`'s own docstring.
        paper_spec = real_paper_session["paper_spec"]
        positions = _existing_positions_for_reduce_only_orders(
            paper_orders, strategy_id=paper_spec.strategy_candidate_identity, contract_multiplier=Decimal(str(paper_spec.instrument.contract_multiplier)),
        )
        environment = _runner_environment(real_paper_session, storage_root, positions=positions)
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-1", scenario=_DEFAULT_SCENARIO)

        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=paper_orders, market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED, manifest.to_json_dict()

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        final_states = {oid: resolve_execution_order_state(oid, state_events) for oid, (_c, state_events, _b, _f) in reconstructed.items()}
        filled = [oid for oid, s in final_states.items() if s is ExecutionOrderState.FILLED]
        assert filled, "at least one bridged real order must have filled"

        # Scenario 22: clean successful reconciliation.
        report = reconcile_execution_session(execution_session_id=manifest.execution_session_id, ledger=ledger, adapter=adapter, event_time=_EVENT_TIME, policy=spec.reconciliation_policy)
        assert report.is_reconciled, [i.to_json_dict() for i in report.blocking_issues]

        # Scenario 23: verification.
        verification_report = verify_execution_session(spec, execution_session_id=manifest.execution_session_id, ledger=ledger)
        assert verification_report.is_ready, [i.to_json_dict() for i in verification_report.criticals]

        print(f"\nMilestone 8 acceptance: execution_session_id={manifest.execution_session_id} bridged_order_count={len(paper_orders)} filled_count={len(filled)}")


# ==========================================================================
# Scenario 3: LIMIT order resting, then filling once the market crosses.
# Scenario 4: STOP order triggering and becoming market-like.
# Scenario 5: partial fill via the scenario's own fill schedule.
# Scenario 6: full fill (subsumed by the LIMIT/STOP orders reaching FILLED).
# ==========================================================================
class TestLimitStopAndPartialFillMechanics:
    def test_limit_order_rests_then_fills(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-limit-1", side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME, limit_price=99.0)
        market_events = _bars([101.0, 100.0, 99.0], start=_EVENT_TIME)  # crosses down to the limit on the 3rd bar

        spec = _build_spec(real_paper_session, seed=2)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-2", scenario=_DEFAULT_SCENARIO)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        fills = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED]
        assert len(fills) == 1
        assert fills[0].payload["price"] == "99.0"  # LIMIT fills AT the limit price, never a better/worse price

    def test_stop_order_triggers_and_fills_market_like(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-stop-1", side=OrderSide.BUY, order_type=OrderTypeKind.STOP, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME, stop_price=101.0)
        # `_evaluate_fill_condition` triggers AND fills in the SAME call once
        # the reference price first reaches the stop price -- there is no
        # separate "triggered, waiting for the next event" state; the order
        # becomes market-like and fills immediately, at THAT SAME bar's own
        # reference price (101.0), never a later one.
        market_events = _bars([99.0, 100.0, 101.0, 102.0], start=_EVENT_TIME)

        spec = _build_spec(real_paper_session, seed=3)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-3", scenario=_DEFAULT_SCENARIO)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        fills = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.EXECUTION_FILL_RECORDED]
        assert len(fills) == 1
        assert fills[0].payload["price"] == "101.0"  # fills at the triggering bar's own reference, which IS the stop price here

    def test_partial_fill_then_completes_via_schedule(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-partial-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=2.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        market_events = _bars([100.0, 100.0], start=_EVENT_TIME)
        # Each entry is a fraction of the ORIGINAL quantity to fill AT THAT
        # STEP (not a cumulative target) -- the schedule must sum to <= 1
        # (`DummyBrokerScenarioSpec.__post_init__` enforces this). Two
        # 0.5 steps fill the order in exactly two equal partial fills.
        scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(Decimal("0.5"), Decimal("0.5")), rejection_rules=(),
            duplicate_event_indices=(), delayed_event_indices=(), out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None,
            heartbeat_failure_sequences=(), order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True,
            supports_idempotent_cancel=True, supports_idempotent_replace=True, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=4, scenario=scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-4", scenario=scenario)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        (_command, state_events, _broker_events, fills) = next(iter(reconstructed.values()))
        assert len(fills) == 2, "the order must have filled in exactly two partial steps per the schedule"
        assert resolve_execution_order_state(next(iter(reconstructed)), state_events) is ExecutionOrderState.FILLED


# ==========================================================================
# Scenario 7: synchronous broker rejection.
# ==========================================================================
class TestSynchronousBrokerRejection:
    def test_a_rejection_rule_produces_a_synchronous_reject(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-reject-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=5.0, time_in_force=TimeInForceKind.DAY, event_time=_EVENT_TIME)
        market_events = _bars([100.0], start=_EVENT_TIME)
        scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(RejectionRuleSpec(rule_index=0, reject_instrument_id=None, reject_quantity_above=Decimal("3.0"), reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False),),
            duplicate_event_indices=(), delayed_event_indices=(), out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None,
            heartbeat_failure_sequences=(), order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True,
            supports_idempotent_cancel=True, supports_idempotent_replace=True, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=5, scenario=scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-5", scenario=scenario)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        (_command, state_events, _b, _f) = next(iter(reconstructed.values()))
        assert resolve_execution_order_state(next(iter(reconstructed)), state_events) is ExecutionOrderState.REJECTED
        rejected_entries = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED]
        assert len(rejected_entries) == 1
        assert rejected_entries[0].payload["reason"] == "rejection_rule_0"


# ==========================================================================
# Scenario 8: cancellation before fill.
# Scenario 9: replacement.
# ==========================================================================
class TestCancellationAndReplacement:
    def test_cancellation_before_fill(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-cancel-1", side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME, limit_price=50.0)
        # Reference stays well above the limit -- never fillable within this test.
        market_events = _bars([100.0, 100.0], start=_EVENT_TIME)

        spec = _build_spec(real_paper_session, seed=6)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-6", scenario=_DEFAULT_SCENARIO)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        execution_order_id, (command, state_events, _b, _f) = next(iter(reconstructed.items()))
        assert resolve_execution_order_state(execution_order_id, state_events) is ExecutionOrderState.ACKNOWLEDGED

        cancel_command = create_cancel_order_command(
            execution_session_id=manifest.execution_session_id, execution_order_id=execution_order_id, client_order_id=command.client_order_id,
            cancellation_reason="acceptance_test_cancel_before_fill", command_sequence=100, event_time=_EVENT_TIME,
        )
        authorize_cancel_or_reduce_only_submit(execution_session_id=manifest.execution_session_id, environment=environment, adapter=adapter, command=cancel_command, event_time=_EVENT_TIME)
        process_broker_events(execution_session_id=manifest.execution_session_id, event_store=environment.event_store, adapter=adapter, max_events=100, event_time=_EVENT_TIME)

        ledger_after = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed_after = reconstruct_all_orders_from_ledger(ledger_after)
        (_c2, state_events_after, _b2, _f2) = reconstructed_after[execution_order_id]
        assert resolve_execution_order_state(execution_order_id, state_events_after) is ExecutionOrderState.CANCELLED

    def test_replacement_changes_the_resting_limit_price_and_it_later_fills_at_the_new_price(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-replace-1", side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME, limit_price=50.0)
        spec = _build_spec(real_paper_session, seed=7)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-7", scenario=_DEFAULT_SCENARIO)

        # First call: submit only, against a market that would never fill the ORIGINAL 50.0 limit.
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=_bars([100.0], start=_EVENT_TIME), event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED
        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        execution_order_id, (command, state_events, _b, _f) = next(iter(reconstructed.items()))
        assert resolve_execution_order_state(execution_order_id, state_events) is ExecutionOrderState.ACKNOWLEDGED

        replace_command = create_replace_order_command(
            execution_session_id=manifest.execution_session_id, execution_order_id=execution_order_id, client_order_id=command.client_order_id,
            command_sequence=101, event_time=_EVENT_TIME, replacement_limit_price=Decimal("99.0"),
        )
        authorize_cancel_or_reduce_only_submit(execution_session_id=manifest.execution_session_id, environment=environment, adapter=adapter, command=replace_command, event_time=_EVENT_TIME)
        process_broker_events(execution_session_id=manifest.execution_session_id, event_store=environment.event_store, adapter=adapter, max_events=100, event_time=_EVENT_TIME)

        ledger_after_replace = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed_after = reconstruct_all_orders_from_ledger(ledger_after_replace)
        (_c2, state_events_2, _b2, _f2) = reconstructed_after[execution_order_id]
        assert resolve_execution_order_state(execution_order_id, state_events_2) is ExecutionOrderState.ACKNOWLEDGED

        adapter.advance_market_event(_bars([99.0], start=_EVENT_TIME)[0], event_time=_EVENT_TIME)
        process_broker_events(execution_session_id=manifest.execution_session_id, event_store=environment.event_store, adapter=adapter, max_events=100, event_time=_EVENT_TIME)

        final_ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed_final = reconstruct_all_orders_from_ledger(final_ledger)
        (_c3, state_events_3, _b3, fills_3) = reconstructed_final[execution_order_id]
        assert resolve_execution_order_state(execution_order_id, state_events_3) is ExecutionOrderState.FILLED
        assert len(fills_3) == 1
        assert fills_3[0].price == Decimal("99.0"), "must fill at the REPLACED limit price, not the original 50.0"


# ==========================================================================
# Scenarios 10-12: duplicate / delayed / out-of-order broker event delivery.
# Scenarios 13-14: disconnect and reconnect.
# ==========================================================================
class TestBrokerEventDeliveryAnomaliesAndConnectivity:
    def test_duplicate_and_out_of_order_delivery_never_corrupt_economic_state(self, real_paper_session: dict, tmp_path: Path) -> None:
        """A single MARKET buy generates exactly 3 broker events in ONE
        poll batch (index 0 = informational `ORDER_RECEIVED`, index 1 =
        `ORDER_ACKNOWLEDGED`, index 2 = `ORDER_FILLED`). Swapping the
        delivery POSITIONS of indices 0/1 (rather than 0/2) is a
        deliberate choice: `ORDER_RECEIVED` is purely informational (no
        state-transition legality check applies to it at all -- Section
        14), so reordering it ahead of/behind `ORDER_ACKNOWLEDGED` proves
        genuine out-of-order handling without ever attempting the
        (illegal) `DISPATCHED -> FILLED` jump that reordering the FILL
        itself would create."""
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-anomaly-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        market_events = _bars([100.0], start=_EVENT_TIME)
        scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(2,),
            delayed_event_indices=(), out_of_order_event_groups=((0, 1),), disconnect_at_sequence=None, reconnect_at_sequence=None,
            heartbeat_failure_sequences=(), order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True,
            supports_idempotent_cancel=True, supports_idempotent_replace=True, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=8, scenario=scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-8", scenario=scenario)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        assert any(e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_DUPLICATE for e in ledger), "a duplicate delivery must be recorded"
        assert any(e.entry_kind is ExecutionLedgerEntryKind.BROKER_EVENT_OUT_OF_ORDER for e in ledger), "an out-of-order delivery must be recorded"

        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        execution_order_id, (_c, state_events, _b, fills) = next(iter(reconstructed.items()))
        assert resolve_execution_order_state(execution_order_id, state_events) is ExecutionOrderState.FILLED
        assert len(fills) == 1, "the duplicate delivery must never produce a second economic fill"

        report = reconcile_execution_session(execution_session_id=manifest.execution_session_id, ledger=ledger, adapter=adapter, event_time=_EVENT_TIME, policy=spec.reconciliation_policy)
        assert report.is_reconciled, [i.to_json_dict() for i in report.blocking_issues]

    def test_delayed_delivery_of_the_fill_event_still_eventually_resolves(self, real_paper_session: dict, tmp_path: Path) -> None:
        """Delays the `ORDER_FILLED` event's delivery by one poll cycle
        (it is generated immediately, but withheld from `poll_events`
        once) -- the order must still reach FILLED, with exactly one
        fill, once the SECOND poll cycle (the runner's own unconditional
        final `process_broker_events` call once the market-event stream
        is exhausted) delivers it."""
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-delay-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        market_events = _bars([100.0], start=_EVENT_TIME)
        scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(),
            delayed_event_indices=(2,), out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None,
            heartbeat_failure_sequences=(), order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True,
            supports_idempotent_cancel=True, supports_idempotent_replace=True, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=15, scenario=scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-15", scenario=scenario)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=market_events, event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        execution_order_id, (_c, state_events, _b, fills) = next(iter(reconstructed.items()))
        assert resolve_execution_order_state(execution_order_id, state_events) is ExecutionOrderState.FILLED
        assert len(fills) == 1

    def test_disconnect_then_reconnect(self, real_paper_session: dict, tmp_path: Path) -> None:
        order_before = _order(real_paper_session["paper_session_id"], client_order_id="acc-disc-before", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        order_during = _order(real_paper_session["paper_session_id"], client_order_id="acc-disc-during", side=OrderSide.SELL, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        # `submit_order`/`cancel_order`/`replace_order`/`poll_events` each
        # bump `_operation_counter` by 1: order_before's submit is
        # operation 1 (still connected); order_during's submit is
        # operation 2, landing exactly in the [disconnect_at_sequence,
        # reconnect_at_sequence) window and getting synchronously
        # rejected; the market-event loop's first `poll_events` call is
        # operation 3 -- already reconnected.
        scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(),
            delayed_event_indices=(), out_of_order_event_groups=(), disconnect_at_sequence=2, reconnect_at_sequence=3, heartbeat_failure_sequences=(),
            order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=True, supports_idempotent_cancel=True,
            supports_idempotent_replace=True, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=9, scenario=scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-9", scenario=scenario)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order_before, order_during], market_events=_bars([100.0], start=_EVENT_TIME), event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        # NOTE: `command.client_order_id` is the EXECUTION GATEWAY's own
        # deterministic-hash id (`derive_client_order_id`), never the
        # ORIGINAL paper order's own `client_order_id` field -- the two
        # orders are distinguished by SIDE instead (BUY vs SELL), which is
        # unambiguous here since each side was submitted exactly once.
        states = {command.side: resolve_execution_order_state(oid, se) for oid, (command, se, _b, _f) in reconstructed.items()}
        assert states[OrderSide.BUY] is ExecutionOrderState.FILLED, "the order submitted BEFORE the disconnect window must succeed normally"
        assert states[OrderSide.SELL] is ExecutionOrderState.REJECTED, "the order submitted DURING the disconnect window must be synchronously rejected"
        rejected_entries = [e for e in ledger if e.entry_kind is ExecutionLedgerEntryKind.COMMAND_DISPATCH_REJECTED]
        assert any(e.payload["reason"] == "adapter_disconnected" for e in rejected_entries)

        # Reconnect genuinely restores service -- a fresh health() check (which itself does not consume an operation slot) reports HEALTHY again.
        health = adapter.health(event_time=_EVENT_TIME)
        assert health.status.value == "healthy"
        assert health.can_submit is True


# ==========================================================================
# Scenarios 15-17: crash recovery after dispatch-intent / broker-
# acceptance / fill-persistence. Scenario 18: recovery-by-query.
# Scenario 19: an unresolved-ambiguity UNKNOWN case blocks completion.
# ==========================================================================
class TestCrashRecoveryAndUnresolvedUnknown:
    def test_crash_after_dispatch_intent_before_any_adapter_call_recovers_via_query(self, real_paper_session: dict, tmp_path: Path) -> None:
        """Simulates a process crash the instant AFTER `COMMAND_DISPATCH_
        INTENT` was durably recorded but BEFORE the adapter was ever
        called (Section 17's own ordering guarantee is what makes this
        crash point observable/recoverable at all). The broker genuinely
        has no record of the order -- `guarantees_idempotent_submit=True`
        on this scenario's capabilities means recovery must authorize a
        safe retry, never leave the order silently unresolved."""
        from quant_platform.execution_gateway.commands import create_submit_order_command
        from quant_platform.execution_gateway.paper_bridge import execution_intent_from_paper_order

        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-crash-intent-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        spec = _build_spec(real_paper_session, seed=10)
        storage_root = tmp_path / "session_root"
        environment = _runner_environment(real_paper_session, storage_root)
        execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id

        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-10", scenario=_DEFAULT_SCENARIO)
        adapter.initialize(execution_session_id=execution_session_id, event_time=_EVENT_TIME)
        environment.manifest_store.create(execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id, adapter_id=adapter.adapter_id, execution_mode=spec.execution_mode)

        intent, _auth = execution_intent_from_paper_order(order, execution_gateway_spec=spec, execution_session_id=execution_session_id, environment=environment.paper_bridge_environment, created_sequence=0, event_time=_EVENT_TIME)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED, payload=intent.to_json_dict(), event_time=_EVENT_TIME)
        command = create_submit_order_command(
            execution_session_id=execution_session_id, execution_intent_id=intent.execution_intent_id, command_sequence=0, event_time=_EVENT_TIME,
            instrument_id=intent.instrument_id, side=intent.side, quantity=intent.quantity, order_type=intent.order_type, time_in_force=intent.time_in_force,
            reduce_only=intent.reduce_only, contract_multiplier=intent.contract_multiplier,
        )
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_CREATED, payload=command.to_json_dict(), event_time=_EVENT_TIME)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT, payload={"command_id": command.command_id}, event_time=_EVENT_TIME)
        execution_order_id = compute_execution_order_id(command)
        from quant_platform.execution_gateway.dispatcher import _append_order_transition

        # The full legal CREATED -> VALIDATED -> DISPATCH_PENDING -> UNKNOWN
        # chain -- resolve_execution_order_state replays from an implicit
        # CREATED start, so a lone DISPATCH_PENDING->UNKNOWN event with no
        # preceding history would itself be an illegal (discontinuous)
        # transition. This is exactly the chain `dispatch_command` itself
        # appends before an adapter call -- reused here, not reinvented.
        pending_events: list = []
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.CREATED, to_state=ExecutionOrderState.VALIDATED, event_time=_EVENT_TIME)
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.VALIDATED, to_state=ExecutionOrderState.DISPATCH_PENDING, event_time=_EVENT_TIME)
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.DISPATCH_PENDING, to_state=ExecutionOrderState.UNKNOWN, event_time=_EVENT_TIME, reason_code="simulated_crash_after_dispatch_intent")
        # Deliberately NEVER call adapter.submit_order -- the broker genuinely has no record of this order.

        actions = recover_unknown_orders(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=_EVENT_TIME)
        assert len(actions) == 1
        assert actions[0].action == "safe_retry_authorized", actions[0].detail

    def test_crash_after_broker_acceptance_before_ledger_write_recovers_via_query_confirms(self, real_paper_session: dict, tmp_path: Path) -> None:
        """Simulates the broker having genuinely accepted (and even
        acknowledged) an order, but the process crashing before the
        gateway's own ledger recorded the successful outcome -- the
        adapter's in-process state (standing in for a durable, separately
        alive broker) still has the order; recovery must resolve UNKNOWN
        directly to the broker-confirmed state, never a blind retry."""
        from quant_platform.execution_gateway.commands import create_submit_order_command
        from quant_platform.execution_gateway.paper_bridge import execution_intent_from_paper_order

        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-crash-accept-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        spec = _build_spec(real_paper_session, seed=11)
        storage_root = tmp_path / "session_root"
        environment = _runner_environment(real_paper_session, storage_root)
        execution_session_id = compute_execution_gateway_spec_id(spec).execution_gateway_spec_id

        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-11", scenario=_DEFAULT_SCENARIO)
        adapter.initialize(execution_session_id=execution_session_id, event_time=_EVENT_TIME)
        environment.manifest_store.create(execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=spec.paper_session_id, adapter_id=adapter.adapter_id, execution_mode=spec.execution_mode)

        intent, _auth = execution_intent_from_paper_order(order, execution_gateway_spec=spec, execution_session_id=execution_session_id, environment=environment.paper_bridge_environment, created_sequence=0, event_time=_EVENT_TIME)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED, payload=intent.to_json_dict(), event_time=_EVENT_TIME)
        command = create_submit_order_command(
            execution_session_id=execution_session_id, execution_intent_id=intent.execution_intent_id, command_sequence=0, event_time=_EVENT_TIME,
            instrument_id=intent.instrument_id, side=intent.side, quantity=intent.quantity, order_type=intent.order_type, time_in_force=intent.time_in_force,
            reduce_only=intent.reduce_only, contract_multiplier=intent.contract_multiplier,
        )
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_CREATED, payload=command.to_json_dict(), event_time=_EVENT_TIME)
        _append(environment.event_store, execution_session_id, kind=ExecutionLedgerEntryKind.COMMAND_DISPATCH_INTENT, payload={"command_id": command.command_id}, event_time=_EVENT_TIME)
        execution_order_id = compute_execution_order_id(command)
        from quant_platform.execution_gateway.dispatcher import _append_order_transition

        pending_events: list = []
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.CREATED, to_state=ExecutionOrderState.VALIDATED, event_time=_EVENT_TIME)
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.VALIDATED, to_state=ExecutionOrderState.DISPATCH_PENDING, event_time=_EVENT_TIME)
        _append_order_transition(environment.event_store, execution_session_id, execution_order_id=execution_order_id, existing_events=pending_events, from_state=ExecutionOrderState.DISPATCH_PENDING, to_state=ExecutionOrderState.UNKNOWN, event_time=_EVENT_TIME, reason_code="simulated_crash_after_broker_acceptance")

        # The broker call itself DID happen (standing in for "the broker
        # genuinely accepted it") -- only the ledger write of the outcome
        # was lost.
        call_result = adapter.submit_order(command, event_time=_EVENT_TIME)
        assert call_result.accepted_for_processing

        actions = recover_unknown_orders(execution_session_id=execution_session_id, event_store=environment.event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=_EVENT_TIME)
        assert len(actions) == 1
        assert actions[0].action == "resolved_by_query", actions[0].detail

    def test_unresolved_unknown_blocks_completion(self, real_paper_session: dict, tmp_path: Path) -> None:
        """A submit whose adapter CALL itself raises (a genuinely
        ambiguous transport-layer failure -- Section 17's own "every
        exception from the adapter call is classified UNKNOWN, never a
        blanket failure" rule) must leave the order UNKNOWN, and
        `run_execution_session` must refuse COMPLETED while any order
        remains UNKNOWN (Section 19's own gate condition) -- confirmed
        via the REAL `run_execution_session`/`dispatch_command` path, not
        a hand-rolled ledger. A subsequent recovery attempt against a
        NON-idempotent-submit-guaranteeing adapter must then refuse to
        blindly retry (Section 16/23)."""
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-unknown-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        non_idempotent_scenario = DummyBrokerScenarioSpec(
            acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(),
            delayed_event_indices=(), out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None, heartbeat_failure_sequences=(),
            order_query_failure_sequences=(), account_query_failure_sequences=(), supports_idempotent_submit=False, supports_idempotent_cancel=False,
            supports_idempotent_replace=False, seed=0,
        )
        spec = _build_spec(real_paper_session, seed=12, scenario=non_idempotent_scenario)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-12", scenario=non_idempotent_scenario)

        def _raising_submit_order(command, *, event_time):
            raise RuntimeError("simulated ambiguous transport failure during submit")

        adapter.submit_order = _raising_submit_order  # type: ignore[method-assign]

        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=_bars([100.0], start=_EVENT_TIME), event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.FAILED
        # RECONCILING's own "unresolved_unknown_order" BLOCKING check (Section
        # 24) runs BEFORE VERIFYING's own equivalent check (Section 19) --
        # reconciliation reaches and reports the unresolved UNKNOWN order
        # FIRST, so `failure_category` is "reconciliation", not "unresolved_
        # unknown" (that category is only reached if reconciliation itself
        # somehow missed it). Both represent the SAME correct safety
        # property: a session with an unresolved UNKNOWN order must never
        # reach COMPLETED, regardless of which specific gate catches it first.
        assert manifest.failure_category in ("reconciliation", "unresolved_unknown"), manifest.failure_category

        ledger = environment.event_store.read_events(manifest.execution_session_id)
        reconstructed = reconstruct_all_orders_from_ledger(ledger)
        execution_order_id, (_c, state_events, _b, _f) = next(iter(reconstructed.items()))
        assert resolve_execution_order_state(execution_order_id, state_events) is ExecutionOrderState.UNKNOWN

        del adapter.submit_order  # restores the real (never-called-successfully) bound method for the recovery query below
        actions = recover_unknown_orders(execution_session_id=manifest.execution_session_id, event_store=environment.event_store, adapter=adapter, capabilities=adapter.capabilities(), event_time=_EVENT_TIME)
        assert len(actions) == 1
        assert actions[0].action == "remains_unknown", actions[0].detail


# ==========================================================================
# Scenario 20: kill-switch activation blocks new exposure.
# Scenario 21: blocking reconciliation mismatch prevents completion.
# ==========================================================================
class TestKillSwitchAndBlockingReconciliation:
    def test_kill_switch_activation_blocks_new_exposure_but_allows_queries(self, real_paper_session: dict) -> None:
        from quant_platform.execution_gateway.commands import (
            create_query_account_command,
            create_submit_order_command,
        )

        transition_1 = create_execution_kill_switch_transition_event(
            execution_session_id="1" * 64, from_state=ExecutionKillSwitchState.ACTIVE, to_state=ExecutionKillSwitchState.HALTING,
            trigger=ExecutionKillSwitchTriggerKind.OPERATOR_REQUEST, event_time=_EVENT_TIME, sequence=0, detail="acceptance test",
        )
        transition_2 = create_execution_kill_switch_transition_event(
            execution_session_id="1" * 64, from_state=ExecutionKillSwitchState.HALTING, to_state=ExecutionKillSwitchState.HALTED,
            trigger=ExecutionKillSwitchTriggerKind.OPERATOR_REQUEST, event_time=_EVENT_TIME, sequence=1, detail="acceptance test",
        )
        state = resolve_execution_kill_switch_state([transition_1, transition_2])
        assert state is ExecutionKillSwitchState.HALTED

        submit = create_submit_order_command(
            execution_session_id="1" * 64, execution_intent_id="2" * 64, command_sequence=0, event_time=_EVENT_TIME, instrument_id=_INSTRUMENT,
            side=OrderSide.BUY, quantity=Decimal("1"), order_type=OrderTypeKind.MARKET, time_in_force=TimeInForceKind.GTC, reduce_only=False,
            contract_multiplier=Decimal("1"),
        )
        with pytest.raises(ExecutionHaltError):
            authorize_dispatch(state, submit)

        query = create_query_account_command(execution_session_id="1" * 64, command_sequence=1, event_time=_EVENT_TIME)
        authorize_dispatch(state, query)  # must NOT raise -- queries remain permitted even while HALTED

    def test_blocking_reconciliation_mismatch_prevents_completion(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-mismatch-1", side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME)
        spec = _build_spec(real_paper_session, seed=13)
        environment = _runner_environment(real_paper_session, tmp_path / "session_root")
        adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-13", scenario=_DEFAULT_SCENARIO)
        manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=[order], market_events=_bars([100.0], start=_EVENT_TIME), event_time=_EVENT_TIME)
        assert manifest.current_stage is ExecutionSessionStage.COMPLETED

        # A SEPARATE, fresh adapter (no shared state -- standing in for
        # "the broker's own bookkeeping no longer agrees with what the
        # ledger reconstructs") is what makes this comparison genuinely
        # independent of the ledger it is checked against.
        fresh_adapter = DeterministicDummyBrokerAdapter(adapter_id="acceptance-adapter-13-fresh", scenario=_DEFAULT_SCENARIO)
        fresh_adapter.initialize(execution_session_id=manifest.execution_session_id, event_time=_EVENT_TIME)
        ledger = environment.event_store.read_events(manifest.execution_session_id)
        report = reconcile_execution_session(execution_session_id=manifest.execution_session_id, ledger=ledger, adapter=fresh_adapter, event_time=_EVENT_TIME, policy=spec.reconciliation_policy)
        assert not report.is_reconciled
        assert report.blocking_issues, "a broker with no memory of this order's fill must produce at least one BLOCKING/CRITICAL issue"


# ==========================================================================
# Scenario 24: a second, genuinely separate-process (distinct interpreter,
# distinct PYTHONHASHSEED, distinct temp storage path) deterministic run,
# proving the semantic digest is not an artifact of in-process state,
# hash seed, or storage layout.
# ==========================================================================
class TestDeterministicReplayAcrossProcessesAndHashSeeds:
    def test_two_fresh_in_process_runs_produce_identical_semantic_digest(self, real_paper_session: dict, tmp_path: Path) -> None:
        order = _order(real_paper_session["paper_session_id"], client_order_id="acc-determinism-1", side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, quantity=1.0, time_in_force=TimeInForceKind.GTC, event_time=_EVENT_TIME, limit_price=99.0)
        spec = _build_spec(real_paper_session, seed=14)
        market_events_factory = lambda: _bars([101.0, 100.0, 99.0], start=_EVENT_TIME)  # noqa: E731

        first = replay_execution_session(
            spec, storage_root=tmp_path / "replay_first", paper_bridge_environment=_paper_bridge_environment(real_paper_session),
            portfolio_risk_context=_portfolio_risk_context(tmp_path / "replay_first" / "portfolio_risk"), paper_orders=[order],
            market_events=market_events_factory(), adapter_id="replay-adapter-first", event_time=_EVENT_TIME,
        )
        second = replay_execution_session(
            spec, storage_root=tmp_path / "replay_second", paper_bridge_environment=_paper_bridge_environment(real_paper_session),
            portfolio_risk_context=_portfolio_risk_context(tmp_path / "replay_second" / "portfolio_risk"), paper_orders=[order],
            market_events=market_events_factory(), adapter_id="replay-adapter-second", event_time=_EVENT_TIME,
        )
        assert first.manifest.current_stage is ExecutionSessionStage.COMPLETED
        assert second.manifest.current_stage is ExecutionSessionStage.COMPLETED
        assert first.semantic_digest == second.semantic_digest

    def test_a_genuinely_separate_os_process_with_a_different_hashseed_reproduces_the_same_execution_gateway_spec_id(self, real_paper_session: dict) -> None:
        """The strongest available cross-process determinism check that
        does not require re-running the (expensive) real ML chain inside
        a subprocess: `create-execution-gateway-spec` recomputes THIS
        module's own real spec identity, from a config file alone, inside
        a genuinely separate Python process with a DIFFERENT
        `PYTHONHASHSEED` -- proving `ExecutionGatewaySpec` identity does
        not depend on in-process hash randomization."""
        import json

        config = {
            "ml_artifacts_root": str(real_paper_session["paper_storage_root"]), "research_storage_root": str(real_paper_session["paper_storage_root"]),
            "historical_storage_root": str(real_paper_session["paper_storage_root"]), "execution_mode": "test_only", "adapter_kind": "deterministic_dummy",
            "paper_session_id": real_paper_session["paper_session_id"], "paper_trading_spec_id": real_paper_session["paper_trading_spec_id"],
            "promotion_decision_id": real_paper_session["promotion_decision_id"], "instrument_spec_id": real_paper_session["instrument_spec_id"], "seed": 99,
        }
        config_path = real_paper_session["paper_storage_root"] / "cross_process_spec_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        expected_spec = _build_spec(real_paper_session, seed=99)
        expected_id = compute_execution_gateway_spec_id(expected_spec).execution_gateway_spec_id

        env_a = dict(os.environ, PYTHONHASHSEED="1")
        env_b = dict(os.environ, PYTHONHASHSEED="4242")
        result_a = subprocess.run([sys.executable, "-m", "quant_platform.ml_cli", "create-execution-gateway-spec", "--config", str(config_path)], capture_output=True, text=True, timeout=60, env=env_a)
        result_b = subprocess.run([sys.executable, "-m", "quant_platform.ml_cli", "create-execution-gateway-spec", "--config", str(config_path)], capture_output=True, text=True, timeout=60, env=env_b)
        assert result_a.returncode == 0, result_a.stderr
        assert result_b.returncode == 0, result_b.stderr
        assert f"execution_gateway_spec_id: {expected_id}" in result_a.stdout
        assert f"execution_gateway_spec_id: {expected_id}" in result_b.stdout
