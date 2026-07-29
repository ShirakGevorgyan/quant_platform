"""Milestone 7, Sections 17-18: risk engine pre-trade/continuous checks
and the event-sourced kill switch. Covers every configured limit firing
correctly, the "not configured = skipped" vs "mandatory = always
evaluated" distinction, `most_severe_action`'s severity ordering, the
full legal kill-switch transition graph, and the structural "no
transition back to ACTIVE" property."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import RiskHaltError, RiskLimitError
from quant_platform.paper_trading.fills import create_fill
from quant_platform.paper_trading.models import (
    ComparisonOperatorKind,
    KillSwitchState,
    OrderSide,
    OrderTypeKind,
    PartialFillPolicyKind,
    PositionIntentKind,
    RejectReasonKind,
    RiskActionKind,
    RiskTriggerKind,
    TimeInForceKind,
)
from quant_platform.paper_trading.orders import create_order_request
from quant_platform.paper_trading.portfolio import (
    apply_fill_to_portfolio,
    apply_mark_to_portfolio,
    initial_portfolio,
)
from quant_platform.paper_trading.risk import (
    KillSwitchTransitionEvent,
    RiskCheckResult,
    create_kill_switch_transition_event,
    evaluate_continuous_risk,
    evaluate_pre_trade_risk,
    most_severe_action,
    resolve_kill_switch_state,
)
from quant_platform.paper_trading.specs import DEFAULT_RISK_LIMITS, RiskLimitsSpec

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_EVENT = "a" * 64
_HEX_DECISION = "b" * 64


def _order(*, side: OrderSide = OrderSide.BUY, quantity: float = 1.0) -> object:
    return create_order_request(
        client_order_id="c1", session_id="session-1", strategy_decision_id=_HEX_DECISION, instrument="X", side=side, order_type=OrderTypeKind.MARKET,
        quantity=quantity, time_in_force=TimeInForceKind.DAY, create_time=_T0, submit_time=_T0, reduce_only=False, position_intent=PositionIntentKind.OPEN,
    )


def _limits(**overrides: object) -> RiskLimitsSpec:
    defaults: dict[str, object] = {
        "maximum_signed_position": None, "maximum_absolute_position": None, "maximum_gross_exposure": None, "maximum_order_quantity": None,
        "maximum_order_notional": None, "maximum_turnover": None, "maximum_daily_loss": None, "maximum_drawdown_fraction": None,
        "maximum_realized_loss": None, "maximum_unrealized_loss": None, "maximum_rejected_order_count": None,
        "maximum_consecutive_execution_failures": None, "maximum_stale_data_seconds": None, "maximum_reconciliation_discrepancy": 1e-6,
    }
    defaults.update(overrides)
    return RiskLimitsSpec(**defaults)  # type: ignore[arg-type]


class TestRiskCheckResultValidation:
    def test_passed_requires_allow_action(self) -> None:
        with pytest.raises(RiskLimitError, match="ALLOW"):
            RiskCheckResult(check_identity="c", measured_value=1.0, limit=2.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, passed=True, action=RiskActionKind.REJECT_ORDER, reason_code=None, event_identity=_HEX_EVENT)

    def test_failed_requires_reason_code(self) -> None:
        with pytest.raises(RiskLimitError, match="reason_code"):
            RiskCheckResult(check_identity="c", measured_value=3.0, limit=2.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, passed=False, action=RiskActionKind.REJECT_ORDER, reason_code=None, event_identity=_HEX_EVENT)


class TestEvaluatePreTradeRisk:
    def test_all_checks_pass_when_nothing_configured(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(), portfolio=portfolio, risk_limits=DEFAULT_RISK_LIMITS, reference_price=100.0, contract_multiplier=1.0, event_identity=_HEX_EVENT,
            trading_halted=False, session_accepting_orders=True, stale_data_seconds=None,
        )
        assert all(r.passed for r in results)
        assert most_severe_action(results) is RiskActionKind.ALLOW

    def test_trading_halted_always_evaluated_and_rejects(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(), portfolio=portfolio, risk_limits=DEFAULT_RISK_LIMITS, reference_price=100.0, contract_multiplier=1.0, event_identity=_HEX_EVENT,
            trading_halted=True, session_accepting_orders=True, stale_data_seconds=None,
        )
        failed = [r for r in results if not r.passed]
        assert len(failed) == 1
        assert failed[0].check_identity == "trading_halted"
        assert failed[0].reason_code is RejectReasonKind.TRADING_HALTED
        assert most_severe_action(results) is RiskActionKind.REJECT_ORDER

    def test_session_not_accepting_orders_rejects(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(), portfolio=portfolio, risk_limits=DEFAULT_RISK_LIMITS, reference_price=100.0, contract_multiplier=1.0, event_identity=_HEX_EVENT,
            trading_halted=False, session_accepting_orders=False, stale_data_seconds=None,
        )
        failed = [r for r in results if not r.passed]
        assert len(failed) == 1
        assert failed[0].reason_code is RejectReasonKind.SESSION_NOT_ACCEPTING_ORDERS

    def test_max_order_quantity_rejects_when_exceeded(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(quantity=10.0), portfolio=portfolio, risk_limits=_limits(maximum_order_quantity=5.0), reference_price=100.0, contract_multiplier=1.0,
            event_identity=_HEX_EVENT, trading_halted=False, session_accepting_orders=True, stale_data_seconds=None,
        )
        failed = [r for r in results if not r.passed]
        assert any(r.check_identity == "max_order_quantity" for r in failed)
        assert most_severe_action(results) is RiskActionKind.REJECT_ORDER

    def test_max_order_notional_rejects_when_exceeded(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(quantity=10.0), portfolio=portfolio, risk_limits=_limits(maximum_order_notional=500.0), reference_price=100.0, contract_multiplier=1.0,
            event_identity=_HEX_EVENT, trading_halted=False, session_accepting_orders=True, stale_data_seconds=None,
        )
        failed = [r.check_identity for r in results if not r.passed]
        assert "max_order_notional" in failed

    def test_max_gross_exposure_rejects_when_exceeded(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(quantity=10.0), portfolio=portfolio, risk_limits=_limits(maximum_gross_exposure=500.0), reference_price=100.0, contract_multiplier=1.0,
            event_identity=_HEX_EVENT, trading_halted=False, session_accepting_orders=True, stale_data_seconds=None,
        )
        failed = [r.check_identity for r in results if not r.passed]
        assert "max_gross_exposure" in failed

    def test_stale_data_not_evaluated_when_no_limit_configured(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(), portfolio=portfolio, risk_limits=DEFAULT_RISK_LIMITS, reference_price=100.0, contract_multiplier=1.0, event_identity=_HEX_EVENT,
            trading_halted=False, session_accepting_orders=True, stale_data_seconds=9999.0,
        )
        assert not any(r.check_identity == "stale_data" for r in results)

    def test_stale_data_rejects_when_limit_exceeded(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_pre_trade_risk(
            _order(), portfolio=portfolio, risk_limits=_limits(maximum_stale_data_seconds=5.0), reference_price=100.0, contract_multiplier=1.0,
            event_identity=_HEX_EVENT, trading_halted=False, session_accepting_orders=True, stale_data_seconds=10.0,
        )
        failed = [r.check_identity for r in results if not r.passed]
        assert "stale_data" in failed


class TestEvaluateContinuousRisk:
    def _lost_portfolio(self, loss: float) -> object:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        fill = create_fill(
            order_id="o1", session_id="session-1", instrument="X", side=OrderSide.BUY, quantity=10.0, price=100.0, contract_multiplier=1.0,
            spread_cost=0.0, slippage_cost=0.0, commission_cost=0.0, execution_time=_T0, source_market_event_identity=_HEX_EVENT,
            liquidity_assumption=PartialFillPolicyKind.FULL_FILL_ONLY, is_final=True,
        )
        portfolio = apply_fill_to_portfolio(portfolio, fill, event_time=_T0, contract_multiplier=1.0)
        mark_price = 100.0 - (loss / 10.0)
        return apply_mark_to_portfolio(portfolio, instrument="X", mark_price=mark_price, event_time=_T0)

    def test_max_daily_loss_triggers_halt_new_orders(self) -> None:
        portfolio = self._lost_portfolio(500.0)
        results = evaluate_continuous_risk(
            portfolio, risk_limits=_limits(maximum_daily_loss=200.0), event_identity=_HEX_EVENT, rejected_order_count=0, consecutive_execution_failures=0,
            stale_data_seconds=None, reconciliation_discrepancy=None,
        )
        failed = [r for r in results if not r.passed]
        assert any(r.check_identity == "max_daily_loss" for r in failed)
        assert most_severe_action(results) is RiskActionKind.HALT_NEW_ORDERS

    def test_max_drawdown_triggers_flatten(self) -> None:
        portfolio = self._lost_portfolio(500.0)
        results = evaluate_continuous_risk(
            portfolio, risk_limits=_limits(maximum_drawdown_fraction=0.001), event_identity=_HEX_EVENT, rejected_order_count=0,
            consecutive_execution_failures=0, stale_data_seconds=None, reconciliation_discrepancy=None,
        )
        failed = [r for r in results if not r.passed]
        assert any(r.check_identity == "max_drawdown" for r in failed)
        assert most_severe_action(results) is RiskActionKind.FLATTEN_SIMULATED_POSITIONS

    def test_max_reconciliation_discrepancy_triggers_terminate(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_continuous_risk(
            portfolio, risk_limits=_limits(maximum_reconciliation_discrepancy=0.01), event_identity=_HEX_EVENT, rejected_order_count=0,
            consecutive_execution_failures=0, stale_data_seconds=None, reconciliation_discrepancy=5.0,
        )
        failed = [r for r in results if not r.passed]
        assert any(r.check_identity == "max_reconciliation_discrepancy" for r in failed)
        assert most_severe_action(results) is RiskActionKind.TERMINATE_SESSION

    def test_max_consecutive_execution_failures_triggers_terminate(self) -> None:
        portfolio = initial_portfolio("session-1", starting_cash=100_000.0)
        results = evaluate_continuous_risk(
            portfolio, risk_limits=_limits(maximum_consecutive_execution_failures=3), event_identity=_HEX_EVENT, rejected_order_count=0,
            consecutive_execution_failures=5, stale_data_seconds=None, reconciliation_discrepancy=None,
        )
        assert most_severe_action(results) is RiskActionKind.TERMINATE_SESSION

    def test_no_configured_limits_all_pass(self) -> None:
        portfolio = self._lost_portfolio(500.0)
        results = evaluate_continuous_risk(
            portfolio, risk_limits=DEFAULT_RISK_LIMITS, event_identity=_HEX_EVENT, rejected_order_count=0, consecutive_execution_failures=0,
            stale_data_seconds=None, reconciliation_discrepancy=None,
        )
        assert all(r.passed for r in results)


class TestMostSevereAction:
    def test_empty_results_allow(self) -> None:
        assert most_severe_action(()) is RiskActionKind.ALLOW

    def test_severity_ordering_picks_most_severe(self) -> None:
        results = (
            RiskCheckResult(check_identity="a", measured_value=1.0, limit=0.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, passed=False, action=RiskActionKind.REJECT_ORDER, reason_code=RejectReasonKind.TRADING_HALTED, event_identity=_HEX_EVENT),
            RiskCheckResult(check_identity="b", measured_value=1.0, limit=0.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, passed=False, action=RiskActionKind.TERMINATE_SESSION, reason_code=RiskTriggerKind.RECONCILIATION_FAILURE, event_identity=_HEX_EVENT),
            RiskCheckResult(check_identity="c", measured_value=1.0, limit=2.0, comparison_operator=ComparisonOperatorKind.LESS_THAN_OR_EQUAL, passed=True, action=RiskActionKind.ALLOW, reason_code=None, event_identity=_HEX_EVENT),
        )
        assert most_severe_action(results) is RiskActionKind.TERMINATE_SESSION


class TestKillSwitchEventSourcing:
    def test_valid_transition_sequence(self) -> None:
        events = [
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.ACTIVE, to_state=KillSwitchState.HALTING, trigger=RiskTriggerKind.LOSS_LIMIT, event_time=_T0, sequence=1, detail="loss limit breached"),
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.HALTING, to_state=KillSwitchState.HALTED, trigger=RiskTriggerKind.LOSS_LIMIT, event_time=_T0, sequence=2, detail="halted"),
        ]
        assert resolve_kill_switch_state(events) is KillSwitchState.HALTED

    def test_full_flatten_then_terminate_path(self) -> None:
        events = [
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.ACTIVE, to_state=KillSwitchState.HALTING, trigger=RiskTriggerKind.DRAWDOWN_LIMIT, event_time=_T0, sequence=1, detail="drawdown"),
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.HALTING, to_state=KillSwitchState.FLATTENING, trigger=RiskTriggerKind.DRAWDOWN_LIMIT, event_time=_T0, sequence=2, detail="flattening"),
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.FLATTENING, to_state=KillSwitchState.HALTED, trigger=RiskTriggerKind.DRAWDOWN_LIMIT, event_time=_T0, sequence=3, detail="flattened"),
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.HALTED, to_state=KillSwitchState.TERMINATED, trigger=RiskTriggerKind.OPERATOR_REQUEST, event_time=_T0, sequence=4, detail="terminated"),
        ]
        assert resolve_kill_switch_state(events) is KillSwitchState.TERMINATED

    def test_illegal_transition_rejected_at_construction(self) -> None:
        with pytest.raises(RiskHaltError, match="Illegal"):
            create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.ACTIVE, to_state=KillSwitchState.HALTED, trigger=RiskTriggerKind.LOSS_LIMIT, event_time=_T0, sequence=1, detail="skip")

    def test_no_transition_back_to_active_from_any_state(self) -> None:
        """Structural property: `ACTIVE` never appears as a legal target
        from ANY other state -- never silently auto-resume."""
        for state in KillSwitchState:
            if state is KillSwitchState.ACTIVE:
                continue
            with pytest.raises(RiskHaltError, match="Illegal"):
                create_kill_switch_transition_event(session_id="s1", from_state=state, to_state=KillSwitchState.ACTIVE, trigger=RiskTriggerKind.OPERATOR_REQUEST, event_time=_T0, sequence=1, detail="attempted resume")

    def test_no_events_resolves_to_active(self) -> None:
        assert resolve_kill_switch_state([]) is KillSwitchState.ACTIVE

    def test_mismatched_from_state_rejected(self) -> None:
        events = [create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.HALTING, to_state=KillSwitchState.HALTED, trigger=RiskTriggerKind.LOSS_LIMIT, event_time=_T0, sequence=1, detail="skip active->halting")]
        with pytest.raises(RiskHaltError, match="from_state"):
            resolve_kill_switch_state(events)

    def test_json_round_trip(self) -> None:
        event = create_kill_switch_transition_event(session_id="s1", from_state=KillSwitchState.ACTIVE, to_state=KillSwitchState.HALTING, trigger=RiskTriggerKind.LOSS_LIMIT, event_time=_T0, sequence=1, detail="detail")
        assert KillSwitchTransitionEvent.from_json_dict(event.to_json_dict()) == event
