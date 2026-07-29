"""Milestone 7, Section 27-28: durable session reports and the optional
backtest-comparison diagnostic. Reuses the real-session fixture pattern
established in `test_reconciliation.py`/`test_verification.py` -- a real
completed session (long entries only, `_FixedDirectionStrategy`) must
produce a report whose counts/sums match what the fixture actually did,
not merely "some report got built"."""

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
from quant_platform.paper_trading.reports import (
    BacktestComparisonMetrics,
    PaperSessionReport,
    build_paper_session_report,
    compare_paper_to_backtest,
)
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
    report-building assertions don't crash on a `None` environment."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class _AlwaysAbstainStrategy:
    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=PositionDirection.FLAT,
            target_quantity=0.0, confidence=0.5, uncertainty=0.5, abstain=True, reason_codes=("no_signal",), stop_target_intent=None,
        )


def _run_real_session(tmp_path, strategy, closes: list[float]) -> tuple[list, PaperTradingSpec, object]:
    spec = _spec()
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
    events = _bars(closes)
    run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
    ledger = event_store.read_events(paper_session_id)
    manifest = manifest_store.load(paper_session_id)
    return ledger, spec, manifest


class TestBuildPaperSessionReportRealSession:
    def test_long_round_trip_report_counts_match_fixture(self, tmp_path) -> None:
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

        assert report.session.session_id == manifest.paper_session_id
        assert report.session.session_mode == "replay_paper"
        assert report.session.instrument == "X"
        assert report.session.event_count == 4
        assert report.decisions.decision_count == 4
        assert report.decisions.abstention_count == 0
        assert report.orders.order_count >= 1
        assert report.fills.fill_count >= 1
        assert report.reconciliation.is_reconciled
        assert report.verification.was_run is False
        assert report.verification.is_ready is None
        assert report.shadow.observation_count == 0
        assert report.account_equity.starting_cash == 100_000.0
        assert report.disclaimer == "This report is diagnostic, not a promotion decision. Simulated paper fills are not broker fills; paper trading does not prove profitability."

    def test_all_abstentions_produce_zero_orders_and_fills(self, tmp_path) -> None:
        strategy = _AlwaysAbstainStrategy()
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 100.0, 100.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

        assert report.decisions.decision_count == 3
        assert report.decisions.abstention_count == 3
        assert report.orders.order_count == 0
        assert report.fills.fill_count == 0
        assert report.account_equity.final_equity == 100_000.0
        assert report.account_equity.gross_pnl == 0.0
        assert report.drawdown.maximum_drawdown_fraction == 0.0

    def test_json_round_trip(self, tmp_path) -> None:
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)
        assert PaperSessionReport.from_json_dict(report.to_json_dict()) == report

    def test_winning_long_round_trip_counts_one_winning_fill(self, tmp_path) -> None:
        """flat->long->flat with a strictly rising close: the closing fill
        realizes a gain, so it must be counted as a winning fill."""

        @dataclasses.dataclass(frozen=True, slots=True)
        class _EntryThenExit:
            @property
            def strategy_identity(self) -> str:
                return _HEX_A

            def decide(self, context: StrategyContext) -> StrategyDecision:
                target = PositionDirection.LONG if context.event.sequence <= 1 else PositionDirection.FLAT
                quantity = 3.0 if target is PositionDirection.LONG else 0.0
                return create_strategy_decision(
                    strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=target,
                    target_quantity=quantity, confidence=0.9, uncertainty=0.05, abstain=False, reason_codes=("test",), stop_target_intent=None,
                )

        ledger, spec, manifest = _run_real_session(tmp_path, _EntryThenExit(), [100.0, 103.0, 106.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)
        assert report.fills.winning_fill_count == 1
        assert report.fills.losing_fill_count == 0
        assert report.account_equity.realized_pnl > 0.0


class TestCompareToBacktest:
    def test_identical_metrics_all_match(self, tmp_path) -> None:
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

        backtest_metrics = BacktestComparisonMetrics(
            decision_count=report.decisions.decision_count, order_count=report.orders.order_count, gross_return=report.account_equity.gross_pnl / report.account_equity.starting_cash,
            net_return=report.account_equity.net_pnl / report.account_equity.starting_cash, total_costs=report.costs.total_costs, turnover=report.account_equity.turnover,
            max_drawdown_fraction=report.drawdown.maximum_drawdown_fraction, rejected_order_count=report.orders.rejected_count, abstention_count=report.decisions.abstention_count,
        )
        comparison = compare_paper_to_backtest(backtest_metrics, report, source_backtest_id=_HEX_C)
        assert all(c.matches for c in comparison.comparisons)
        assert comparison.source_backtest_id == _HEX_C
        assert comparison.disclaimer

    def test_decision_count_mismatch_classified_as_unexpected(self, tmp_path) -> None:
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

        backtest_metrics = BacktestComparisonMetrics(
            decision_count=report.decisions.decision_count + 1, order_count=report.orders.order_count, gross_return=0.0, net_return=0.0, total_costs=0.0,
            turnover=report.account_equity.turnover, max_drawdown_fraction=report.drawdown.maximum_drawdown_fraction, rejected_order_count=report.orders.rejected_count,
            abstention_count=report.decisions.abstention_count,
        )
        comparison = compare_paper_to_backtest(backtest_metrics, report, source_backtest_id=_HEX_C)
        decision_comparison = next(c for c in comparison.comparisons if c.metric_name == "decision_count")
        assert not decision_comparison.matches
        assert decision_comparison.classification == "unexpected_decision_mismatch"

    def test_cost_mismatch_classified_as_expected_due_to_spread(self, tmp_path) -> None:
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)

        backtest_metrics = BacktestComparisonMetrics(
            decision_count=report.decisions.decision_count, order_count=report.orders.order_count, gross_return=0.0, net_return=0.0, total_costs=report.costs.total_costs + 500.0,
            turnover=report.account_equity.turnover, max_drawdown_fraction=report.drawdown.maximum_drawdown_fraction, rejected_order_count=report.orders.rejected_count,
            abstention_count=report.decisions.abstention_count,
        )
        comparison = compare_paper_to_backtest(backtest_metrics, report, source_backtest_id=_HEX_C)
        cost_comparison = next(c for c in comparison.comparisons if c.metric_name == "total_costs")
        assert not cost_comparison.matches
        assert cost_comparison.classification == "expected_due_to_spread"

    def test_json_round_trip(self, tmp_path) -> None:
        from quant_platform.paper_trading.reports import BacktestComparisonReport

        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
        ledger, spec, manifest = _run_real_session(tmp_path, strategy, [100.0, 103.0, 106.0, 109.0])
        reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        report = build_paper_session_report(ledger, spec=spec, manifest=manifest, reconciliation_report=reconciliation_report)
        backtest_metrics = BacktestComparisonMetrics(
            decision_count=1, order_count=1, gross_return=0.0, net_return=0.0, total_costs=0.0, turnover=0.0, max_drawdown_fraction=0.0, rejected_order_count=0, abstention_count=0,
        )
        comparison = compare_paper_to_backtest(backtest_metrics, report, source_backtest_id=_HEX_C)
        assert BacktestComparisonReport.from_json_dict(comparison.to_json_dict()) == comparison
