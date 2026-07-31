"""Release-audit Area 9: risk halt and flatten behavior.

CONFIRMED DEFECT, FIXED: the working-order fill loop (top of every event
iteration in `run_paper_trading_session`) runs UNCONDITIONALLY regardless
of `kill_switch_state` -- a pre-existing resting LIMIT/STOP order left
WORKING through a kill-switch-triggered flatten could still fill on a
LATER event, silently reopening exposure the safety flatten exists to
eliminate. This directly contradicted Section 9's own "cancel working
orders first" / "no exposure increase" requirements. Fixed: every other
still-working order is now cancelled (CANCELLED, reason=RISK_HALT_ACTIVE)
BEFORE the flatten's own closing order is created.

CONFIRMED, DOCUMENTED, NON-BLOCKING LIMITATIONS (investigated, not fixed
in this audit -- see the final audit report for full reasoning):
  - `RiskActionKind.TERMINATE_SESSION` is unreachable via the live
    runner: `evaluate_continuous_risk` is called with `consecutive_
    execution_failures=0` and `reconciliation_discrepancy=None`
    hardcoded, and those are the ONLY two triggers mapped to
    TERMINATE_SESSION -- `maximum_consecutive_execution_failures`/
    `maximum_reconciliation_discrepancy` can never fire regardless of
    configuration. Implementing genuine live tracking for either is a
    feature addition (continuous mid-session reconciliation, execution-
    failure counting), not a surgical bug fix, and is out of this
    audit's safe scope.
  - `RiskActionKind.CANCEL_OPEN_ORDERS` is defined (with its own
    severity level) but never produced by any check in `evaluate_pre_
    trade_risk`/`evaluate_continuous_risk`, and the runner's own kill-
    switch escalation block has no dedicated handling for it either
    (it would collapse into the same HALTING->HALTED path as HALT_NEW_
    ORDERS). Dead code, not a safety gap: HALT_NEW_ORDERS/FLATTEN_
    SIMULATED_POSITIONS already cover the risk-reduction needs the
    configured limits actually exercise.
  - `PaperSessionManifest.stage` never reaches `PaperSessionStage.
    HALTING`/`HALTED`/`TERMINATED` -- only the SEPARATE, correctly-
    tracked `KillSwitchState` (in-memory during a run, correctly
    reconstructed on resume from `HALT_TRIGGERED` ledger entries) does.
    The actual SAFETY property (no new orders after a halt) holds
    regardless, gated on `kill_switch_state`, not `manifest.stage`; the
    session's own durable REPORT separately surfaces `final_kill_switch_
    state` correctly. A session that was safety-halted still ends with
    `manifest.stage == COMPLETED`, which is misleading for anyone
    reading ONLY the manifest stage rather than the ledger/report --
    tested explicitly below to confirm this is exactly the CURRENT
    (limited) behavor, not silently unmentioned."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.types import Timeframe
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    ClockMode,
    LedgerEntryKind,
    MarketEventMode,
    OrderSide,
    OrderState,
    OrderTypeKind,
    PaperSessionStage,
    PartialFillPolicyKind,
    PositionIntentKind,
    RejectReasonKind,
    SessionMode,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import (
    OrderStateEvent,
    create_order_request,
    create_order_state_event,
    resolve_order_state,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore, create_ledger_entry
from quant_platform.paper_trading.portfolio import apply_order_created_to_portfolio, initial_portfolio
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


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests here are seeded directly PAST eligibility with
    `eligibility_environment=None` -- release-audit finding, fixed
    elsewhere: `run_paper_trading_session` now mandatorily re-verifies
    eligibility on every call that did not itself just create the
    manifest. That fix is exercised for real in `test_audit_eligibility_
    bypass.py`; here it is bypassed so this file's own (unrelated) risk-
    halt/flatten assertions don't crash on a `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


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


@dataclasses.dataclass(frozen=True, slots=True)
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
            reason_codes=("test",), stop_target_intent=None,
        )


def _environment(tmp_path) -> RunnerEnvironment:
    """`eligibility_environment` is a minimal stub exposing only
    `.artifact_store` -- `require_paper_trading_eligibility` itself is
    monkeypatched to a no-op by this file's own autouse fixture, but
    `create_paper_session` also persists the spec via `environment.
    eligibility_environment.artifact_store` directly (Milestone 8's own
    paper-bridge read-back needs a real, durable `spec_reference` --
    a real defect found and fixed during Milestone 8 acceptance
    testing), so a genuine `MLArtifactStore` must be reachable here too."""
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    eligibility_environment = SimpleNamespace(artifact_store=MLArtifactStore(tmp_path))
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=eligibility_environment)  # type: ignore[arg-type]


def _seed_working_limit_order_prefix(environment: RunnerEnvironment, spec: PaperTradingSpec, *, side: OrderSide, limit_price: float, quantity: float, first_bar) -> str:
    """Same technique as `test_audit_resume_working_orders.py`'s own
    `_seed_working_order_prefix` -- builds a manifest at RUNNING plus a
    ledger prefix for a resting LIMIT order that does NOT trigger on
    `first_bar`, exactly the shape `_create_and_submit_order` itself
    leaves behind for an order that doesn't immediately fill. Returns
    the order_id."""
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
    environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
    environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)

    event_store = environment.event_store
    event_time = first_bar.close_time
    seq = event_store.next_sequence(paper_session_id)
    prev_hash = event_store.last_entry_hash(paper_session_id)

    def _append(kind: LedgerEntryKind, payload: dict) -> None:
        nonlocal seq, prev_hash
        entry = create_ledger_entry(session_id=paper_session_id, sequence=seq, kind=kind, payload=payload, event_time=event_time, previous_entry_hash=prev_hash)
        persisted = event_store.append(paper_session_id, entry)
        seq += 1
        prev_hash = persisted.entry_id

    _append(LedgerEntryKind.MARKET_EVENT_ACCEPTED, first_bar.to_json_dict())

    order = create_order_request(
        client_order_id="audit-seeded-stale-order", session_id=paper_session_id, strategy_decision_id="0" * 64, instrument=spec.instrument.symbol,
        side=side, order_type=OrderTypeKind.LIMIT, quantity=quantity, time_in_force=TimeInForceKind.GTC, create_time=event_time, submit_time=event_time,
        reduce_only=False, position_intent=PositionIntentKind.OPEN, limit_price=limit_price, stop_price=None,
    )

    def _order_state_payload(state_event: OrderStateEvent, order_request) -> dict[str, object]:
        return {"order_state_event": state_event.to_json_dict(), "order": order_request.to_json_dict()}

    validated = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=event_time, sequence=seq)
    _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(validated, order))
    accepted = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=event_time, sequence=seq)
    _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(accepted, order))
    working = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=event_time, sequence=seq)
    _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(working, order))

    _append(LedgerEntryKind.MARK_APPLIED, {"instrument": spec.instrument.symbol, "mark_price": first_bar.close})

    decision = create_strategy_decision(
        strategy_identity=_HEX_A, event=first_bar, decision_time=event_time, target_direction=PositionDirection.FLAT, target_quantity=0.0,
        confidence=0.5, uncertainty=0.5, abstain=True, reason_codes=("no_signal",), stop_target_intent=None,
    )
    _append(LedgerEntryKind.STRATEGY_DECISION, decision.to_json_dict())

    portfolio = apply_order_created_to_portfolio(initial_portfolio(paper_session_id, starting_cash=spec.starting_cash), event_time=event_time)
    _append(LedgerEntryKind.ACCOUNT_SNAPSHOT, portfolio.to_json_dict())

    return order.order_id


class TestFlattenCancelsStaleWorkingOrders:
    """The core release-audit fix: a resting LIMIT order that never
    triggered before a drawdown-triggered flatten must be CANCELLED, not
    left working -- otherwise a later bar crossing its limit price would
    silently reopen exposure the flatten was meant to eliminate."""

    def test_stale_buy_limit_is_cancelled_and_never_fills_after_flatten(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        bars = _bars([100.0, 100.0, 50.0, 30.0])  # bar2 (50.0) trips drawdown; bar3 (30.0) would cross a limit@40 if still working
        stale_order_id = _seed_working_limit_order_prefix(environment, spec, side=OrderSide.BUY, limit_price=40.0, quantity=1.0, first_bar=bars[0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=bars)
        assert manifest.stage.value == "completed"

        ledger = environment.event_store.read_events(paper_session_id)
        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        assert any(p["to_state"] == "flattening" for p in halt_payloads), "fixture must genuinely trigger a flatten"

        stale_order_events = [OrderStateEvent.from_json_dict(e.payload["order_state_event"]) for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and e.payload["order_state_event"]["order_id"] == stale_order_id]
        final_state = resolve_order_state(stale_order_id, stale_order_events)
        assert final_state is OrderState.CANCELLED, f"the stale resting LIMIT order must be CANCELLED by the flatten, got {final_state.value!r}"
        cancel_event = next(e for e in stale_order_events if e.to_state is OrderState.CANCELLED)
        assert cancel_event.reason_code is RejectReasonKind.RISK_HALT_ACTIVE

        stale_order_fills = [e.payload for e in ledger if e.kind is LedgerEntryKind.FILL and e.payload["order_id"] == stale_order_id]
        assert not stale_order_fills, "the cancelled stale order must NEVER fill, even though bar3's price would have crossed its limit"

        final_snapshot = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot["positions"]["X"]["signed_quantity"] == 0.0, "no exposure increase: the account must remain flat after the flatten, not reopen via the stale order"

    def test_stale_sell_limit_is_cancelled_when_flattening_a_short(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        bars = _bars([100.0, 100.0, 150.0, 200.0])  # bar2 (150) trips drawdown on the SHORT; bar3 (200) would cross a sell-limit@160 if still working
        stale_order_id = _seed_working_limit_order_prefix(environment, spec, side=OrderSide.SELL, limit_price=160.0, quantity=1.0, first_bar=bars[0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        strategy = _FixedDirectionStrategy(direction=PositionDirection.SHORT, quantity=50.0)
        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=bars)
        assert manifest.stage.value == "completed"

        ledger = environment.event_store.read_events(paper_session_id)
        stale_order_events = [OrderStateEvent.from_json_dict(e.payload["order_state_event"]) for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and e.payload["order_state_event"]["order_id"] == stale_order_id]
        assert resolve_order_state(stale_order_id, stale_order_events) is OrderState.CANCELLED

        final_snapshot = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot["positions"]["X"]["signed_quantity"] == 0.0


class TestFlattenMechanicsBySide:
    def test_long_position_flattened_with_correct_side_and_exact_quantity(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        events = _bars([100.0, 100.0, 50.0, 50.0])

        run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        flatten_orders = [e.payload["order"] for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and str(e.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")]
        assert flatten_orders
        flatten_order = flatten_orders[0]
        assert flatten_order["side"] == "sell", "closing a LONG position must SELL"
        assert flatten_order["quantity"] == pytest.approx(50.0), "must close the EXACT remaining quantity, never more or less"
        assert flatten_order["reduce_only"] is True
        assert flatten_order["position_intent"] == "close"

    def test_short_position_flattened_with_correct_side_and_exact_quantity(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.SHORT, quantity=50.0)
        events = _bars([100.0, 100.0, 150.0, 150.0])

        run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        flatten_orders = [e.payload["order"] for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and str(e.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")]
        assert flatten_orders
        flatten_order = flatten_orders[0]
        assert flatten_order["side"] == "buy", "closing a SHORT position must BUY"
        assert flatten_order["quantity"] == pytest.approx(50.0)
        assert flatten_order["reduce_only"] is True


@dataclasses.dataclass(frozen=True, slots=True)
class _OpenThenCloseStrategy:
    """Targets LONG on the first `open_bars` events (opening a position),
    then targets FLAT (a genuine non-abstain close decision, NOT an
    abstention -- `order_policy._signed_target_quantity`/`_position_
    intent_for` turn a FLAT-direction decision against an open position
    into a real CLOSE order) for every event after that."""

    quantity: float
    open_bars: int

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        is_open_phase = context.event.sequence <= self.open_bars
        direction = PositionDirection.LONG if is_open_phase else PositionDirection.FLAT
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=direction,
            target_quantity=(self.quantity if is_open_phase else 0.0), confidence=0.9, uncertainty=0.05, abstain=False,
            reason_codes=("test",), stop_target_intent=None,
        )


class TestAlreadyFlatAccountFlatten:
    def test_flatten_trigger_while_already_flat_creates_no_close_order(self, tmp_path) -> None:
        """A genuine round trip (open on bar1, close on bar2) pays
        commission both ways, permanently denting equity below the
        `peak_equity` recorded at the very start of the session -- a
        tight drawdown limit can therefore trip on a LATER bar while the
        account is truly, verifiably flat (no position at all). Confirms
        the flatten branch's own `position is not None and signed_
        quantity != 0.0` guard correctly produces NO closing order (and
        does not crash) when there is nothing left to close."""
        spec = _spec(
            # per_side_basis_points=500 on a 50*100=5000 notional pays 250 commission per side --
            # 0.25% drawdown after ONE trade (opening), 0.5% after TWO (open + close). A limit strictly
            # between the two (0.3%) does not trip until AFTER the round trip has already closed.
            risk_limits=_risk_limits(maximum_drawdown_fraction=0.003),
            commission_policy=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=500.0),
        )
        environment = _environment(tmp_path)
        strategy = _OpenThenCloseStrategy(quantity=50.0, open_bars=1)
        events = _bars([100.0, 100.0, 100.0, 100.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        assert manifest.stage.value == "completed"
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        assert any(p["to_state"] == "flattening" for p in halt_payloads), "fixture must genuinely trip the drawdown limit via pure commission drag"

        # The round trip (open bar1, close bar2) must have already flattened the
        # position BEFORE the drawdown limit trips on a later bar -- confirmed
        # directly from the snapshot immediately preceding the flatten trigger.
        flatten_index = next(i for i, e in enumerate(ledger) if e.kind is LedgerEntryKind.HALT_TRIGGERED and e.payload["to_state"] == "flattening")
        snapshot_before_flatten = next(e.payload for e in reversed(ledger[:flatten_index]) if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
        assert snapshot_before_flatten["positions"].get("X", {}).get("signed_quantity", 0.0) == 0.0, "fixture invariant: must already be flat before the flatten trigger fires"

        flatten_orders = [e.payload["order"] for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and str(e.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")]
        assert not flatten_orders, "no closing order may be created when the account is already flat at the moment the flatten triggers"

        final_snapshot = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot["positions"].get("X", {}).get("signed_quantity", 0.0) == 0.0


class TestOrdinaryHaltDoesNotBlockSafetyFlatten:
    def test_flatten_still_executes_while_trading_halted_flag_is_set(self, tmp_path) -> None:
        from quant_platform.paper_trading.events import create_trading_halt_event

        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)

        bars = _bars([100.0, 100.0])
        halt_event = create_trading_halt_event(instrument="X", event_time=bars[1].close_time + timedelta(minutes=1), sequence=3, reason="exchange_halt", source="test")
        drop_bar = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=bars[1].close_time + timedelta(hours=1), open=50.0, high=50.5, low=49.5, close=50.0, sequence=4, source="test")
        events = [*bars, halt_event, drop_bar]

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        assert manifest.stage.value == "completed"
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        assert any(p["to_state"] == "flattening" for p in halt_payloads), "the safety flatten must still trigger even while an exchange-level trading_halted flag is set"
        final_snapshot = [e.payload for e in ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1]
        assert final_snapshot["positions"]["X"]["signed_quantity"] == 0.0, "the flatten's own closing order must not be blocked by the ordinary trading_halted pre-trade check"


class TestFlattenCostsAppliedConsistently:
    def test_flatten_order_pays_the_same_commission_formula_as_ordinary_fills(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001), commission_policy=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=10.0))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        events = _bars([100.0, 100.0, 50.0, 50.0])

        run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        flatten_fills = [e.payload for e in ledger if e.kind is LedgerEntryKind.FILL and str(e.payload["order_id"]) in {str(oe.payload["order"]["order_id"]) for oe in ledger if oe.kind is LedgerEntryKind.ORDER_STATE_EVENT and str(oe.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")}]
        assert flatten_fills
        flatten_fill = flatten_fills[0]
        expected_commission = (10.0 / 10_000.0) * float(str(flatten_fill["gross_notional"]))
        assert float(str(flatten_fill["commission_cost"])) == pytest.approx(expected_commission), "the flatten's own fill must pay commission via the SAME formula as any ordinary fill, never zero/waived"


class TestNoAutomaticReturnToActive:
    def test_halted_session_never_accepts_new_strategy_orders_across_subsequent_events(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        # After the drawdown-triggered flatten on bar2, bars 3-6 all present a strong LONG signal again --
        # if the kill switch ever silently returned to ACTIVE, a new order would appear in the ledger.
        events = _bars([100.0, 100.0, 50.0, 55.0, 60.0, 65.0, 70.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        assert manifest.stage.value == "completed"
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)

        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        flatten_index = next(i for i, e in enumerate(ledger) if e.kind is LedgerEntryKind.HALT_TRIGGERED and e.payload["to_state"] == "halted")
        assert halt_payloads

        orders_created_after_halt = [
            e for e in ledger[flatten_index + 1 :]
            if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and OrderStateEvent.from_json_dict(e.payload["order_state_event"]).from_state is OrderState.CREATED
            and not str(e.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")
        ]
        assert not orders_created_after_halt, "no NEW strategy-originated order may ever appear after the kill switch has left ACTIVE"

    def test_manifest_stage_reaches_completed_not_terminated_documenting_current_limitation(self, tmp_path) -> None:
        """Documents the CURRENT (non-blocking, see module docstring)
        limitation directly: `manifest.stage` ends at COMPLETED even
        though the ledger's own `HALT_TRIGGERED` entries prove a safety
        halt occurred -- the durable, INDEPENDENTLY-CORRECT signal for
        'was this session halted' is the ledger/report, never `manifest.
        stage` alone."""
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        environment = _environment(tmp_path)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)
        events = _bars([100.0, 100.0, 50.0, 50.0])

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        assert manifest.stage is PaperSessionStage.COMPLETED
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        halt_payloads = [e.payload for e in ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        assert any(p["to_state"] == "halted" for p in halt_payloads), "the ledger's OWN record correctly shows the halt even though manifest.stage does not"


class TestKillSwitchStateNeverReturnsToActiveAcrossResume:
    def test_resumed_session_reconstructs_halted_state_and_still_refuses_new_orders(self, tmp_path) -> None:
        spec = _spec(risk_limits=_risk_limits(maximum_drawdown_fraction=0.001))
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=50.0)

        # Truncate a run right after the flatten sequence completes (mirrors the established
        # "run truncated substream then splice the clean prefix" technique used elsewhere in this audit).
        events_prefix = _bars([100.0, 100.0, 50.0])
        prefix_dir = tmp_path / "prefix"
        prefix_environment = _environment(prefix_dir)
        run_paper_trading_session(spec, environment=prefix_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events_prefix)
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        prefix_ledger_full = prefix_environment.event_store.read_events(paper_session_id)
        last_snapshot_index = max(i for i, e in enumerate(prefix_ledger_full) if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
        clean_prefix = prefix_ledger_full[: last_snapshot_index + 1]
        assert any(e.kind is LedgerEntryKind.HALT_TRIGGERED and e.payload["to_state"] == "halted" for e in clean_prefix), "fixture must genuinely reach HALTED within the truncated prefix"

        resume_environment = _environment(tmp_path)
        resume_environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
        resume_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        resume_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        resume_environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)
        for entry in clean_prefix:
            resume_environment.event_store.append(paper_session_id, entry)

        events_full = _bars([100.0, 100.0, 50.0, 60.0, 70.0, 80.0])
        manifest = run_paper_trading_session(spec, environment=resume_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events_full)
        assert manifest.stage.value == "completed"

        resumed_ledger = resume_environment.event_store.read_events(paper_session_id)
        halt_payloads = [e.payload for e in resumed_ledger if e.kind is LedgerEntryKind.HALT_TRIGGERED]
        to_states = [p["to_state"] for p in halt_payloads]
        assert "active" not in to_states, "the kill switch must never transition back to ACTIVE, even reconstructed across a resume"

        flatten_fills = [e for e in resumed_ledger if e.kind is LedgerEntryKind.FILL]
        flatten_only_fills = [f for f in flatten_fills if str(f.payload["order_id"]) in {str(oe.payload["order"]["order_id"]) for oe in resumed_ledger if oe.kind is LedgerEntryKind.ORDER_STATE_EVENT and str(oe.payload["order"]["client_order_id"]).startswith("kill-switch-flatten:")}]
        assert len(flatten_only_fills) == 1, "resuming after a halt must not repeat (double-fill) the flatten"

        orders_created_after_resume = [
            e for e in resumed_ledger[len(clean_prefix) :]
            if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and OrderStateEvent.from_json_dict(e.payload["order_state_event"]).from_state is OrderState.CREATED
        ]
        assert not orders_created_after_resume, "no new order (strategy-originated or otherwise) may be created after resuming a HALTED session"
