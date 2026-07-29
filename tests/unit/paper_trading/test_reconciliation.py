"""Milestone 7, Section 25: the 11 required reconciliation checks. A real
session's ledger (built via `run_paper_trading_session`, reusing the same
fixtures as `test_runner.py`) must reconcile fully; targeted tampering of
a COPY of that ledger (never the store itself) must make the specific
check it violates fail, while leaving every other check unaffected."""

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
    MarketEventMode,
    PaperSessionStage,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore
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


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests here are seeded directly PAST eligibility with
    `eligibility_environment=None` -- release-audit finding, fixed
    elsewhere: `run_paper_trading_session` now mandatorily re-verifies
    eligibility on every call that did not itself just create the
    manifest. That fix is exercised for real in `test_audit_eligibility_
    bypass.py`; here it is bypassed so this file's own (unrelated)
    reconciliation-tampering assertions don't crash on a `None`
    environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash", account_currency="USD",
        financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _spec() -> PaperTradingSpec:
    risk_limits = RiskLimitsSpec(
        maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
        maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None, maximum_realized_loss=None,
        maximum_unrealized_loss=None, maximum_rejected_order_count=None, maximum_consecutive_execution_failures=None,
        maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
    )
    return PaperTradingSpec(
        schema_version=1, verified_robustness_id=_HEX_A, verified_promotion_decision_id=_HEX_B, strategy_candidate_identity=_HEX_C,
        model_artifact_identity=_HEX_D, calibration_artifact_identity=_HEX_E, feature_spec_identity=_HEX_A, instrument=_instrument(),
        price_precision=2, quantity_precision=2, session_mode=SessionMode.REPLAY_PAPER, market_event_mode=MarketEventMode.BAR,
        bar_interval=Timeframe.H1, clock_mode=ClockMode.REPLAY, starting_cash=100_000.0, starting_positions=(),
        order_policy=OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=100, order_rate_window_events=1000),
        execution_policy=DEFAULT_EXECUTION_POLICY, fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        spread_policy=SpreadSpec(kind=SpreadModelKind.ZERO), slippage_policy=SlippageSpec(kind=SlippageModelKind.ZERO),
        commission_policy=CommissionSpec(kind=CommissionModelKind.ZERO),
        financing_policy=FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)),
        latency_policy=LatencyPolicySpec(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0),
        liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False), position_policy=DEFAULT_POSITION_POLICY, risk_limits=risk_limits,
        session_boundary_policy=DEFAULT_SESSION_BOUNDARY_POLICY, seed=0,
    )


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

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=self.direction,
            target_quantity=self.quantity, confidence=0.9, uncertainty=0.05, abstain=False, reason_codes=("test",), stop_target_intent=None,
        )


def _run_real_session(tmp_path) -> tuple[list, str]:
    spec = _spec()
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
    strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
    events = _bars([100.0, 103.0, 106.0, 109.0])
    run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
    ledger = event_store.read_events(paper_session_id)
    return ledger, paper_session_id


class TestReconcileRealSession:
    def test_real_completed_session_reconciles_fully(self, tmp_path) -> None:
        ledger, paper_session_id = _run_real_session(tmp_path)
        report = reconcile_session(ledger, session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        failed = [c.check_identity for c in report.checks if not c.passed]
        assert failed == [], f"unexpected reconciliation failures: {failed}"
        assert report.is_reconciled

    def test_report_covers_every_named_check(self, tmp_path) -> None:
        ledger, paper_session_id = _run_real_session(tmp_path)
        report = reconcile_session(ledger, session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        check_identities = {c.check_identity for c in report.checks}
        expected = {
            "event_sequence_contiguous", "no_duplicate_identities", "order_state_transitions_legal", "order_quantity_equals_fills_plus_remaining",
            "filled_orders_have_zero_remaining", "no_fill_without_valid_order", "position_quantity_equals_signed_cumulative_fills",
            "realized_pnl_matches_closed_quantities", "cash_movements_match_fills_and_costs", "total_costs_equal_component_sums",
            "account_equity_reconciles",
        }
        assert expected.issubset(check_identities)

    def test_json_round_trip(self, tmp_path) -> None:
        from quant_platform.paper_trading.reconciliation import ReconciliationReport

        ledger, paper_session_id = _run_real_session(tmp_path)
        report = reconcile_session(ledger, session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        assert ReconciliationReport.from_json_dict(report.to_json_dict()) == report


class TestReconciliationTampering:
    def test_empty_ledger_reconciles_trivially(self) -> None:
        report = reconcile_session([], session_id="empty-session", instrument=_instrument(), starting_cash=100_000.0)
        assert report.is_reconciled

    def test_tampered_fill_quantity_breaks_position_and_cash_checks(self, tmp_path) -> None:
        ledger, paper_session_id = _run_real_session(tmp_path)
        tampered = list(ledger)
        for i, entry in enumerate(tampered):
            if entry.kind.value == "fill":
                tampered_payload = dict(entry.payload)
                tampered_payload["quantity"] = float(str(tampered_payload["quantity"])) + 100.0
                tampered_payload["gross_notional"] = float(str(tampered_payload["price"])) * float(str(tampered_payload["quantity"]))
                tampered[i] = dataclasses.replace(entry, payload=tampered_payload, checksum=_recompute_checksum(tampered_payload))
                break
        report = reconcile_session(tampered, session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        failed = {c.check_identity for c in report.checks if not c.passed}
        assert "position_quantity_equals_signed_cumulative_fills" in failed
        assert "cash_movements_match_fills_and_costs" in failed
        assert not report.is_reconciled

    def test_duplicate_ledger_entry_id_detected(self, tmp_path) -> None:
        ledger, paper_session_id = _run_real_session(tmp_path)
        duplicated = [*ledger, ledger[0]]
        report = reconcile_session(duplicated, session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        failed = {c.check_identity for c in report.checks if not c.passed}
        assert "no_duplicate_identities" in failed

    def test_orphaned_fill_detected(self, tmp_path) -> None:
        ledger, paper_session_id = _run_real_session(tmp_path)
        non_fill_entries = [e for e in ledger if e.kind.value != "fill"]
        fill_entries = [e for e in ledger if e.kind.value == "fill"]
        assert fill_entries, "fixture must produce at least one fill"
        orphaned_fill_payload = dict(fill_entries[0].payload)
        orphaned_fill_payload["order_id"] = "does-not-exist"
        orphaned_entry = dataclasses.replace(fill_entries[0], payload=orphaned_fill_payload, checksum=_recompute_checksum(orphaned_fill_payload))
        report = reconcile_session([*non_fill_entries, orphaned_entry], session_id=paper_session_id, instrument=_instrument(), starting_cash=100_000.0)
        failed = {c.check_identity for c in report.checks if not c.passed}
        assert "no_fill_without_valid_order" in failed


def _recompute_checksum(payload: dict) -> str:
    from quant_platform.paper_trading.identity import compute_content_id

    return compute_content_id("ledger_entry_payload", payload)
