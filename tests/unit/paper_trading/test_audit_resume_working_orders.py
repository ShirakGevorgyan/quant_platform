"""Release-audit Area 2: in-flight LIMIT/STOP order resume semantics.

CONFIRMED DEFECT (found during this audit, fixed in `runner.py`):
`_reconstruct_resume_state` unconditionally reset `working_orders` to
`{}` on every resume, silently dropping any order still WORKING/
PARTIALLY_FILLED at the moment of interruption -- even though every
`ORDER_STATE_EVENT` ledger entry already embeds the order's own full
economic detail (`_order_state_payload`), so the ledger always had
enough information to reconstruct it exactly. Fixed via `_reconstruct_
working_orders`, which replays each order's own `ORDER_STATE_EVENT`
history through `resolve_order_state` (the SAME event-sourced derivation
the forward runner and `verification.py` both already trust) and
reconstructs any non-terminal order's `OrderRequest` byte-for-byte via
`OrderRequest.from_json_dict`, with `filled_quantity` summed from the
ledger's own `FILL` entries.

SCOPE NOTE: `order_policy.py` (the only production path from a
`StrategyDecision` to an `OrderRequest`) only ever emits `MARKET` orders
-- LIMIT/STOP order TYPES are fully modeled and validated (`orders.py`)
and execution.py's fill logic supports them, but the automated decision
pipeline never constructs one today. These tests therefore construct a
genuine WORKING LIMIT/STOP order directly via the public `orders`/
`persistence` API (exactly the ledger shape `_create_and_submit_order`
itself would have produced) and splice it into a session's ledger as a
"clean event boundary" prefix -- the same splicing technique `test_
paper_runner.py::TestResumeIdempotency` already established -- to prove
the RESUME MACHINERY correctly preserves and later resolves such an
order, independent of whether the current order-policy layer can
originate one itself."""

from __future__ import annotations

import dataclasses
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
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.eligibility import EligibilityVerificationEnvironment
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
    SessionMode,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import (
    create_order_request,
    create_order_state_event,
    resolve_order_state,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore, create_ledger_entry
from quant_platform.paper_trading.portfolio import apply_order_created_to_portfolio, initial_portfolio
from quant_platform.paper_trading.reconciliation import reconcile_session
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
        "liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=False), "position_policy": DEFAULT_POSITION_POLICY, "risk_limits": _risk_limits(),
        "session_boundary_policy": DEFAULT_SESSION_BOUNDARY_POLICY, "seed": 0,
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
class _AlwaysAbstainStrategy:
    """order_policy.py never creates LIMIT/STOP orders on its own -- this
    strategy always abstains, so the ONLY order in play across these
    tests is the one manually seeded directly into the ledger, and any
    fill observed can only have come from that seeded working order."""

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=PositionDirection.FLAT,
            target_quantity=0.0, confidence=0.5, uncertainty=0.5, abstain=True, reason_codes=("no_signal",), stop_target_intent=None,
        )


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests here are seeded directly PAST eligibility with
    `eligibility_environment=None` -- release-audit finding, fixed
    elsewhere: `run_paper_trading_session` now mandatorily re-verifies
    eligibility on every call that did not itself just create the
    manifest. That fix is exercised for real, against a genuine
    eligibility chain, in `test_audit_eligibility_bypass.py`; here it is
    bypassed so this file's own (unrelated) working-order-resume
    assertions don't crash on a `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)


def _seed_working_order_prefix(
    environment: RunnerEnvironment, spec: PaperTradingSpec, *, order_type: OrderTypeKind, side: OrderSide, limit_price: float | None, stop_price: float | None,
    quantity: float, time_in_force: TimeInForceKind, first_bar,
) -> str:
    """Builds a manifest at RUNNING plus a ledger prefix representing:
    bar1 accepted -> order CREATED->VALIDATED->ACCEPTED->WORKING (not
    filled by bar1) -> mark -> abstain decision -> account snapshot --
    EXACTLY the shape `_create_and_submit_order` itself would have left
    behind for an order that does not immediately trigger. Returns the
    order_id."""
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
        client_order_id="audit-seeded-order", session_id=paper_session_id, strategy_decision_id="0" * 64, instrument=spec.instrument.symbol,
        side=side, order_type=order_type, quantity=quantity, time_in_force=time_in_force, create_time=event_time, submit_time=event_time,
        reduce_only=False, position_intent=PositionIntentKind.OPEN, limit_price=limit_price, stop_price=stop_price,
    )

    def _order_state_payload(state_event, order_request) -> dict[str, object]:
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


class TestWorkingLimitOrderSurvivesResume:
    def test_buy_limit_not_yet_triggered_reconstructed_and_fills_on_later_bar(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        bars = _bars([100.0, 98.0, 97.0])  # bar1 low=99.5 (no trigger @ limit 99.0); bar2 low=97.5 (triggers)
        order_id = _seed_working_order_prefix(
            environment, spec, order_type=OrderTypeKind.LIMIT, side=OrderSide.BUY, limit_price=99.0, stop_price=None, quantity=2.0,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)
        assert manifest.stage.value == "completed"

        ledger = environment.event_store.read_events(paper_session_id)
        fills = [e.payload for e in ledger if e.kind is LedgerEntryKind.FILL]
        assert len(fills) == 1, "the seeded working LIMIT order must fill exactly once, on the bar that actually crosses its limit price"
        assert fills[0]["order_id"] == order_id
        assert fills[0]["quantity"] == 2.0
        assert fills[0]["side"] == "buy"

        order_events = [e for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT]
        from quant_platform.paper_trading.orders import OrderStateEvent

        this_order_events = [OrderStateEvent.from_json_dict(e.payload["order_state_event"]) for e in order_events if e.payload["order_state_event"]["order_id"] == order_id]
        final_state = resolve_order_state(order_id, this_order_events)
        assert final_state is OrderState.FILLED
        # Exactly one ACCEPTED and one WORKING transition -- never re-accepted, never re-entered WORKING.
        to_states = [e.to_state for e in this_order_events]
        assert to_states.count(OrderState.ACCEPTED) == 1
        assert to_states.count(OrderState.WORKING) == 1
        assert to_states.count(OrderState.FILLED) == 1

        reconciliation = reconcile_session(ledger, session_id=paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        assert reconciliation.is_reconciled, [c.check_identity for c in reconciliation.checks if not c.passed]

    def test_sell_stop_not_yet_triggered_reconstructed_and_fills_on_later_bar(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        bars = _bars([100.0, 100.0, 94.0])  # bar1/2 high=100.5 (no trigger @ stop 95.0); bar3 low=93.5 (triggers a sell-stop at 95.0)
        order_id = _seed_working_order_prefix(
            environment, spec, order_type=OrderTypeKind.STOP, side=OrderSide.SELL, limit_price=None, stop_price=95.0, quantity=1.5,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)
        assert manifest.stage.value == "completed"

        ledger = environment.event_store.read_events(paper_session_id)
        fills = [e.payload for e in ledger if e.kind is LedgerEntryKind.FILL]
        assert len(fills) == 1
        assert fills[0]["order_id"] == order_id
        assert fills[0]["side"] == "sell"
        assert fills[0]["quantity"] == 1.5

        reconciliation = reconcile_session(ledger, session_id=paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        assert reconciliation.is_reconciled, [c.check_identity for c in reconciliation.checks if not c.passed]

    def test_original_submission_time_preserved_exactly_across_resume(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        bars = _bars([100.0, 98.0])
        original_submit_time = bars[0].close_time
        order_id = _seed_working_order_prefix(
            environment, spec, order_type=OrderTypeKind.LIMIT, side=OrderSide.BUY, limit_price=99.0, stop_price=None, quantity=1.0,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)

        ledger = environment.event_store.read_events(paper_session_id)
        order_jsons = [e.payload["order"] for e in ledger if e.kind is LedgerEntryKind.ORDER_STATE_EVENT and e.payload["order"]["order_id"] == order_id]
        assert order_jsons, "expected at least one ORDER_STATE_EVENT for the seeded order after resume"
        for order_json in order_jsons:
            assert order_json["submit_time"] == original_submit_time.isoformat().replace("+00:00", "Z") or order_json["submit_time"].startswith(original_submit_time.strftime("%Y-%m-%dT%H:%M:%S"))
            assert order_json["limit_price"] == 99.0
            assert order_json["quantity"] == 1.0
            assert order_json["time_in_force"] == "gtc"

    def test_multiple_working_orders_all_survive_resume(self, tmp_path) -> None:
        """Two orders on the SAME instrument, at different trigger
        prices, both WORKING simultaneously at the moment of
        interruption (e.g. a limit buy far below market and a stop sell
        far below market, neither yet triggered) -- both must survive
        resume, independently, and neither may be lost or duplicated."""
        spec = _spec()
        bars = _bars([100.0, 100.0, 100.0, 90.0])
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        manifest_store = PaperSessionManifestStore(tmp_path)
        event_store = PaperSessionEventStore(tmp_path)
        manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
        manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.INITIALIZED)
        manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.RUNNING)

        event_time = bars[0].close_time
        seq = event_store.next_sequence(paper_session_id)
        prev_hash = event_store.last_entry_hash(paper_session_id)

        def _append(kind: LedgerEntryKind, payload: dict) -> None:
            nonlocal seq, prev_hash
            entry = create_ledger_entry(session_id=paper_session_id, sequence=seq, kind=kind, payload=payload, event_time=event_time, previous_entry_hash=prev_hash)
            persisted = event_store.append(paper_session_id, entry)
            seq += 1
            prev_hash = persisted.entry_id

        def _order_state_payload(state_event, order_request) -> dict[str, object]:
            return {"order_state_event": state_event.to_json_dict(), "order": order_request.to_json_dict()}

        def _seed_order(order: object) -> None:
            validated = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=event_time, sequence=seq)
            _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(validated, order))
            accepted = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.VALIDATED, to_state=OrderState.ACCEPTED, event_time=event_time, sequence=seq)
            _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(accepted, order))
            working = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.ACCEPTED, to_state=OrderState.WORKING, event_time=event_time, sequence=seq)
            _append(LedgerEntryKind.ORDER_STATE_EVENT, _order_state_payload(working, order))

        _append(LedgerEntryKind.MARKET_EVENT_ACCEPTED, bars[0].to_json_dict())

        order_1 = create_order_request(
            client_order_id="audit-seeded-order-1", session_id=paper_session_id, strategy_decision_id="0" * 64, instrument=spec.instrument.symbol,
            side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, quantity=1.0, time_in_force=TimeInForceKind.GTC, create_time=event_time,
            submit_time=event_time, reduce_only=False, position_intent=PositionIntentKind.OPEN, limit_price=91.0, stop_price=None,
        )
        _seed_order(order_1)
        order_2 = create_order_request(
            client_order_id="audit-seeded-order-2", session_id=paper_session_id, strategy_decision_id="0" * 64, instrument=spec.instrument.symbol,
            side=OrderSide.SELL, order_type=OrderTypeKind.STOP, quantity=2.0, time_in_force=TimeInForceKind.GTC, create_time=event_time,
            submit_time=event_time, reduce_only=False, position_intent=PositionIntentKind.OPEN, limit_price=None, stop_price=92.0,
        )
        _seed_order(order_2)

        _append(LedgerEntryKind.MARK_APPLIED, {"instrument": spec.instrument.symbol, "mark_price": bars[0].close})
        decision = create_strategy_decision(
            strategy_identity=_HEX_A, event=bars[0], decision_time=event_time, target_direction=PositionDirection.FLAT, target_quantity=0.0,
            confidence=0.5, uncertainty=0.5, abstain=True, reason_codes=("no_signal",), stop_target_intent=None,
        )
        _append(LedgerEntryKind.STRATEGY_DECISION, decision.to_json_dict())
        portfolio = initial_portfolio(paper_session_id, starting_cash=spec.starting_cash)
        portfolio = apply_order_created_to_portfolio(portfolio, event_time=event_time)
        portfolio = apply_order_created_to_portfolio(portfolio, event_time=event_time)
        _append(LedgerEntryKind.ACCOUNT_SNAPSHOT, portfolio.to_json_dict())

        environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=None)  # type: ignore[arg-type]
        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)
        assert manifest.stage.value == "completed"

        final_ledger = event_store.read_events(paper_session_id)
        fills = [e.payload for e in final_ledger if e.kind is LedgerEntryKind.FILL]
        filled_order_ids = {f["order_id"] for f in fills}
        assert order_1.order_id in filled_order_ids, "the seeded LIMIT buy must still fill once price crosses its limit"
        assert order_2.order_id in filled_order_ids, "the seeded STOP sell must still fill once price crosses its trigger"
        assert len(fills) == 2, f"expected exactly 2 fills (one per working order), got {len(fills)}: {fills}"

        reconciliation = reconcile_session(final_ledger, session_id=paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        assert reconciliation.is_reconciled, [c.check_identity for c in reconciliation.checks if not c.passed]

    def test_uninterrupted_versus_resumed_produce_equivalent_final_state(self, tmp_path) -> None:
        """The core required invariant: an uninterrupted continuation and
        a resumed continuation of the SAME seeded working order, against
        the SAME remaining events, must reach the SAME final reconciled
        account state."""
        spec = _spec()
        bars = _bars([100.0, 98.0, 96.0])

        # "Uninterrupted": seed the prefix and immediately run the FULL
        # remaining stream in one call (this already exercises the resume
        # machinery once, at the moment `run_paper_trading_session` is first
        # invoked -- there is no way to get a WORKING limit order into a
        # session without going through this exact path, since order_policy
        # itself never creates one; see module docstring).
        control_environment = _environment(tmp_path / "control")
        _seed_working_order_prefix(
            control_environment, spec, order_type=OrderTypeKind.LIMIT, side=OrderSide.BUY, limit_price=97.0, stop_price=None, quantity=1.0,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        control_manifest = run_paper_trading_session(spec, environment=control_environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)

        # "Resumed": identical seed, but call run_paper_trading_session TWICE
        # -- once for just the first remaining event, then again for the rest
        # -- simulating an operator-driven pause/resume cycle mid-stream.
        # A truncated sub-stream (`bars[:2]`) still runs all the way to its
        # OWN (bogus) COMPLETED, exactly like `test_paper_runner.py`'s own
        # `TestResumeIdempotency` documents -- truncate that bogus
        # END_OF_STREAM/RECONCILING/VERIFIED/COMPLETED tail back off (and
        # roll the manifest back to RUNNING) before the "resumed" call, so
        # only entries a REAL crash-then-resume would actually have left
        # behind survive.
        resumed_environment = _environment(tmp_path / "resumed")
        _seed_working_order_prefix(
            resumed_environment, spec, order_type=OrderTypeKind.LIMIT, side=OrderSide.BUY, limit_price=97.0, stop_price=None, quantity=1.0,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        paper_session_id_for_truncation = compute_paper_session_spec_id(spec).paper_session_spec_id
        run_paper_trading_session(spec, environment=resumed_environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars[:2])
        truncated_ledger = resumed_environment.event_store.read_events(paper_session_id_for_truncation)
        last_snapshot_index = max(i for i, e in enumerate(truncated_ledger) if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT)
        clean_prefix = truncated_ledger[: last_snapshot_index + 1]
        clean_root = tmp_path / "resumed_clean"
        clean_manifest_store = PaperSessionManifestStore(clean_root)
        clean_event_store = PaperSessionEventStore(clean_root)
        clean_manifest_store.create(paper_session_id=paper_session_id_for_truncation, session_mode=spec.session_mode, spec_reference=None)
        clean_manifest_store.transition(paper_session_id_for_truncation, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        clean_manifest_store.transition(paper_session_id_for_truncation, target_stage=PaperSessionStage.INITIALIZED)
        clean_manifest_store.transition(paper_session_id_for_truncation, target_stage=PaperSessionStage.RUNNING)
        for entry in clean_prefix:
            clean_event_store.append(paper_session_id_for_truncation, entry)
        clean_environment = RunnerEnvironment(manifest_store=clean_manifest_store, event_store=clean_event_store, eligibility_environment=None)  # type: ignore[arg-type]
        resumed_manifest = run_paper_trading_session(spec, environment=clean_environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)

        assert control_manifest.stage.value == "completed"
        assert resumed_manifest.stage.value == "completed"

        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        control_ledger = control_environment.event_store.read_events(paper_session_id)
        resumed_ledger = clean_event_store.read_events(paper_session_id)

        control_final_snapshot = [e for e in control_ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1].payload
        resumed_final_snapshot = [e for e in resumed_ledger if e.kind is LedgerEntryKind.ACCOUNT_SNAPSHOT][-1].payload
        assert control_final_snapshot["cash"] == resumed_final_snapshot["cash"]
        assert control_final_snapshot["positions"] == resumed_final_snapshot["positions"]
        assert control_final_snapshot["fill_count"] == resumed_final_snapshot["fill_count"]

        control_fills = [e.payload for e in control_ledger if e.kind is LedgerEntryKind.FILL]
        resumed_fills = [e.payload for e in resumed_ledger if e.kind is LedgerEntryKind.FILL]
        assert len(control_fills) == len(resumed_fills) == 1
        assert control_fills[0]["price"] == resumed_fills[0]["price"]
        assert control_fills[0]["quantity"] == resumed_fills[0]["quantity"]


class TestNoDuplicateFillOrAcceptanceAcrossResume:
    def test_resuming_an_already_completed_session_does_not_refill(self, tmp_path) -> None:
        spec = _spec()
        environment = _environment(tmp_path)
        bars = _bars([100.0, 98.0])
        _seed_working_order_prefix(
            environment, spec, order_type=OrderTypeKind.LIMIT, side=OrderSide.BUY, limit_price=99.0, stop_price=None, quantity=1.0,
            time_in_force=TimeInForceKind.GTC, first_bar=bars[0],
        )
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        first_manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)
        ledger_after_first = environment.event_store.read_events(paper_session_id)

        second_manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=_AlwaysAbstainStrategy(), clock=ReplayClock(), events=bars)
        ledger_after_second = environment.event_store.read_events(paper_session_id)

        assert second_manifest == first_manifest
        assert ledger_after_second == ledger_after_first
        fills = [e.payload for e in ledger_after_second if e.kind is LedgerEntryKind.FILL]
        assert len(fills) == 1, "re-running a COMPLETED session must never duplicate a fill"
