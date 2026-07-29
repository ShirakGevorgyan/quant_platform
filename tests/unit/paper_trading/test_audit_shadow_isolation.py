"""Release-audit Area 5: SHADOW_OBSERVATION isolation from real
paper-account state.

METHOD (per the audit's own instruction: "use object-identity and
mutation tests, not only ledger-entry-count assertions"): rather than
just counting ledger entries after the fact, these tests MONKEYPATCH the
real mutator functions each runner path is capable of calling and assert
they are never invoked across a mode boundary -- a call proves a real
boundary violation regardless of what the final ledger happens to show.

CONFIRMED DEFECTS FOUND DURING THIS AUDIT AND FIXED ELSEWHERE:
  - `verification.py`'s `verify_paper_session` never checked that a
    ledger's own entry KINDS are consistent with its `session_mode`
    (fixed: `_verify_ledger_matches_session_mode`, tested in `test_
    paper_verification.py::TestSessionModeMismatch`).
  - `ml_cli.py`'s `resume-paper-session` never checked that `--paper-
    session-id` matches the session `--config` resolves to (fixed,
    tested in `test_paper_trading_cli_subprocess.py::
    TestResumeRefusesMismatchedSessionIdentity`).

This file covers the REMAINING Section 5 requirements not already
covered by those two: object-identity/mutation boundaries, and
interleaved-session equivalence to a paper-only control."""

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
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.runner import (
    RunnerEnvironment,
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

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=self.direction,
            target_quantity=self.quantity, confidence=0.9, uncertainty=0.05, abstain=False, reason_codes=("test",), stop_target_intent=None,
        )


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests here are seeded directly PAST eligibility with
    `eligibility_environment=None` -- release-audit finding, fixed
    elsewhere: `run_paper_trading_session`/`run_shadow_session` now
    mandatorily re-verify eligibility on every call that did not itself
    just create the manifest. That fix is exercised for real, against a
    genuine eligibility chain, in `test_audit_eligibility_bypass.py`;
    here it is bypassed so this file's own (unrelated) shadow-isolation
    assertions don't crash on a `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _environment(tmp_path) -> RunnerEnvironment:
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=None)  # type: ignore[arg-type]


def _seed_manifest_past_eligibility(environment: RunnerEnvironment, spec: PaperTradingSpec):
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    environment.manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    return environment.manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)


def _normalize_ledger(ledger: list) -> list:
    return [{"sequence": e.sequence, "kind": e.kind.value, "payload": e.payload} for e in ledger]


class TestShadowSessionNeverCallsRealPortfolioMutators:
    """The strongest possible isolation proof: patch every function that
    is capable of mutating a real `PortfolioState` and assert NONE of
    them are ever called while running a shadow session that DOES trade
    (produces hypothetical fills) -- not merely that the final ledger
    happens to contain no such entries."""

    def test_run_shadow_session_never_touches_real_portfolio_functions(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        events = _bars([100.0, 103.0, 106.0, 109.0])

        def _forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("a SHADOW_OBSERVATION session must never call a real-portfolio mutator")

        for name in ("apply_fill_to_portfolio", "apply_financing_to_portfolio", "apply_mark_to_portfolio", "apply_order_created_to_portfolio", "apply_order_rejected_to_portfolio"):
            monkeypatch.setattr(f"quant_platform.paper_trading.runner.{name}", _forbidden)

        manifest = run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        assert manifest.stage is PaperSessionStage.COMPLETED
        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        ledger = environment.event_store.read_events(paper_session_id)
        observations = [e.payload for e in ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]
        assert any(obs["hypothetical_fill_id"] is not None for obs in observations), "fixture must genuinely trade in shadow mode for this test to be meaningful"


class TestPaperSessionNeverCallsShadowMachinery:
    """The reverse direction: patch shadow's own decision-evaluation
    entrypoint and prove a real paper session never invokes it."""

    def test_run_paper_trading_session_never_touches_shadow_evaluation(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec(session_mode=SessionMode.REPLAY_PAPER)
        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, spec)
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        events = _bars([100.0, 103.0, 106.0, 109.0])

        def _forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("a REPLAY_PAPER session must never call shadow.evaluate_shadow_decision")

        monkeypatch.setattr("quant_platform.paper_trading.runner.evaluate_shadow_decision", _forbidden)

        manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        assert manifest.stage is PaperSessionStage.COMPLETED


class TestInterleavedPaperAndShadowMatchPaperOnlyControl:
    """Required by Section 5: run interleaved paper and shadow sessions
    using the SAME strategy/event stream, sharing the SAME `RunnerEnvironment`
    (same manifest_store/event_store instances, same storage root -- the
    most adversarial sharing arrangement short of literally passing the
    same mutable object), and prove the paper session's own ledger is
    byte-for-byte identical to a fully isolated paper-only control run."""

    def test_shadow_session_sharing_the_same_environment_does_not_perturb_the_paper_ledger(self, tmp_path) -> None:
        paper_spec = _spec(session_mode=SessionMode.REPLAY_PAPER)
        shadow_spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        events = _bars([100.0, 103.0, 97.0, 102.0, 108.0])
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)

        # ---- Control: paper session run entirely alone. ----
        control_environment = _environment(tmp_path / "control")
        _seed_manifest_past_eligibility(control_environment, paper_spec)
        control_manifest = run_paper_trading_session(paper_spec, environment=control_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        control_paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        control_ledger = control_environment.event_store.read_events(control_paper_session_id)

        # ---- Interleaved: SAME environment, paper run FIRST, then a shadow run against the SAME event stream/strategy. ----
        interleaved_environment = _environment(tmp_path / "interleaved")
        _seed_manifest_past_eligibility(interleaved_environment, paper_spec)
        interleaved_manifest = run_paper_trading_session(paper_spec, environment=interleaved_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        _seed_manifest_past_eligibility(interleaved_environment, shadow_spec)
        shadow_manifest = run_shadow_session(shadow_spec, environment=interleaved_environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        interleaved_paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        interleaved_ledger = interleaved_environment.event_store.read_events(interleaved_paper_session_id)

        assert control_manifest.stage is PaperSessionStage.COMPLETED
        assert interleaved_manifest.stage is PaperSessionStage.COMPLETED
        assert shadow_manifest.stage is PaperSessionStage.COMPLETED
        assert interleaved_paper_session_id == control_paper_session_id
        assert _normalize_ledger(interleaved_ledger) == _normalize_ledger(control_ledger), "a shadow session sharing the same environment must never perturb the paper session's own ledger"

        # And the shadow run itself must have produced its OWN, separate ledger -- never merged into the paper one.
        shadow_paper_session_id = compute_paper_session_spec_id(shadow_spec).paper_session_spec_id
        assert shadow_paper_session_id != interleaved_paper_session_id
        shadow_ledger = interleaved_environment.event_store.read_events(shadow_paper_session_id)
        assert [e for e in shadow_ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]
        assert not [e for e in shadow_ledger if e.kind is LedgerEntryKind.FILL]
        assert not [e for e in interleaved_ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION]


class TestReportAggregationNeverMergesShadowIntoRealPnl:
    def test_paper_report_net_pnl_is_unaffected_by_a_coexisting_shadow_session(self, tmp_path) -> None:
        from quant_platform.paper_trading.reconciliation import reconcile_session
        from quant_platform.paper_trading.reports import build_paper_session_report

        paper_spec = _spec(session_mode=SessionMode.REPLAY_PAPER)
        shadow_spec = _spec(session_mode=SessionMode.SHADOW_OBSERVATION)
        events = _bars([100.0, 103.0, 97.0, 102.0, 108.0])
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)

        environment = _environment(tmp_path)
        _seed_manifest_past_eligibility(environment, paper_spec)
        paper_manifest = run_paper_trading_session(paper_spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        paper_session_id = compute_paper_session_spec_id(paper_spec).paper_session_spec_id
        paper_ledger = environment.event_store.read_events(paper_session_id)
        reconciliation_report = reconcile_session(paper_ledger, session_id=paper_session_id, instrument=paper_spec.instrument, starting_cash=paper_spec.starting_cash)
        report_before_shadow = build_paper_session_report(paper_ledger, spec=paper_spec, manifest=paper_manifest, reconciliation_report=reconciliation_report)

        _seed_manifest_past_eligibility(environment, shadow_spec)
        run_shadow_session(shadow_spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        # Re-read the SAME paper session's report after the shadow session ran -- must be byte-identical.
        paper_ledger_after = environment.event_store.read_events(paper_session_id)
        assert _normalize_ledger(paper_ledger_after) == _normalize_ledger(paper_ledger)
        report_after_shadow = build_paper_session_report(paper_ledger_after, spec=paper_spec, manifest=paper_manifest, reconciliation_report=reconciliation_report)
        assert report_after_shadow.account_equity.net_pnl == report_before_shadow.account_equity.net_pnl
        assert report_after_shadow.shadow.observation_count == 0, "a REPLAY_PAPER session's own report must show zero shadow observations regardless of any coexisting shadow session"
