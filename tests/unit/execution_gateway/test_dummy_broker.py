"""Unit tests for `execution_gateway.dummy_broker` (Milestone 8, Section
12/13): MARKET/LIMIT/STOP fill semantics, IOC/FOK/DAY/GTC time-in-force
behavior, partial fills, cancellation, replacement, rejection rules,
duplicate/delayed/out-of-order event delivery, disconnect/reconnect, and
health/heartbeat."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_platform.execution_gateway.commands import (
    create_cancel_order_command,
    create_replace_order_command,
    create_submit_order_command,
)
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.models import (
    AdapterHealthStatus,
    BrokerEventType,
    ExecutionOrderState,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.specs import DEFAULT_DUMMY_BROKER_SCENARIO, RejectionRuleSpec
from quant_platform.paper_trading.events import create_quote_event

_SHA_SESSION = "a" * 64
_SHA_INTENT = "b" * 64
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _tick(seconds: int, bid: float, ask: float, sequence: int = 0):
    return create_quote_event(instrument="EURUSD", event_time=_NOW + timedelta(seconds=seconds), sequence=sequence, bid=bid, ask=ask, source="test")


def _adapter(scenario=DEFAULT_DUMMY_BROKER_SCENARIO) -> DeterministicDummyBrokerAdapter:
    adapter = DeterministicDummyBrokerAdapter(adapter_id="dummy-1", scenario=scenario, starting_cash=Decimal("100000"))
    adapter.initialize(execution_session_id=_SHA_SESSION, event_time=_NOW)
    return adapter


def _submit(adapter: DeterministicDummyBrokerAdapter, *, side=OrderSide.BUY, order_type=OrderTypeKind.MARKET, quantity=Decimal("10"), limit_price=None, stop_price=None, time_in_force=TimeInForceKind.DAY, sequence=0, intent_id=_SHA_INTENT):
    command = create_submit_order_command(
        execution_session_id=_SHA_SESSION, execution_intent_id=intent_id, command_sequence=sequence, event_time=_NOW, instrument_id="EURUSD", side=side,
        quantity=quantity, order_type=order_type, time_in_force=time_in_force, reduce_only=False, contract_multiplier=Decimal("1"), limit_price=limit_price,
        stop_price=stop_price,
    )
    result = adapter.submit_order(command, event_time=_NOW)
    return command, result


def _poll_all(adapter: DeterministicDummyBrokerAdapter, *, after=0, max_events=1000):
    return adapter.poll_events(after_sequence=after, max_events=max_events, event_time=_NOW)


class TestMarketOrderFill:
    def test_buy_market_fills_at_ask_on_next_tick(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY)
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert len(fill_events) == 1
        assert fill_events[0].fill_price == Decimal("1.1005")

    def test_sell_market_fills_at_bid(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.SELL)
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert fill_events[0].fill_price == Decimal("1.0995")


class TestLimitOrderFill:
    def test_buy_limit_does_not_trigger_above_limit(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("1.0900"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        assert not any(e.event_type in (BrokerEventType.ORDER_FILLED, BrokerEventType.ORDER_PARTIALLY_FILLED) for e in events)

    def test_buy_limit_triggers_at_or_below_limit_and_fills_at_limit_price(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("1.1000"))
        adapter.advance_market_event(_tick(0, bid=1.0895, ask=1.0995), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert len(fill_events) == 1
        assert fill_events[0].fill_price == Decimal("1.1000")

    def test_sell_limit_triggers_at_or_above_limit(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.SELL, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("1.1000"))
        adapter.advance_market_event(_tick(0, bid=1.1050, ask=1.1060), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert fill_events[0].fill_price == Decimal("1.1000")


class TestStopOrderFill:
    def test_buy_stop_does_not_trigger_below_stop(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=Decimal("1.1100"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        assert not any(e.event_type is BrokerEventType.ORDER_FILLED for e in events)

    def test_buy_stop_triggers_at_or_above_stop_and_fills_market_like(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.STOP, stop_price=Decimal("1.1000"))
        adapter.advance_market_event(_tick(0, bid=1.1095, ask=1.1105), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert len(fill_events) == 1
        assert fill_events[0].fill_price == Decimal("1.1105")  # market-like: fills at current ask, not stop price

    def test_sell_stop_triggers_at_or_below_stop(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.SELL, order_type=OrderTypeKind.STOP, stop_price=Decimal("1.0900"))
        adapter.advance_market_event(_tick(0, bid=1.0795, ask=1.0805), event_time=_NOW)
        events = _poll_all(adapter)
        fill_events = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert fill_events[0].fill_price == Decimal("1.0795")


class TestTimeInForce:
    def test_fok_cancels_when_partial_fill_schedule_would_only_partially_fill(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.5"),))
        adapter = _adapter(scenario)
        _submit(adapter, side=OrderSide.BUY, time_in_force=TimeInForceKind.FOK)
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        assert not any(e.event_type in (BrokerEventType.ORDER_FILLED, BrokerEventType.ORDER_PARTIALLY_FILLED) for e in events)
        assert any(e.event_type is BrokerEventType.CANCEL_ACKNOWLEDGED for e in events)

    def test_fok_fills_fully_when_fully_fillable(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, time_in_force=TimeInForceKind.FOK)
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        assert any(e.event_type is BrokerEventType.ORDER_FILLED for e in events)

    def test_ioc_fills_available_and_cancels_remainder(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.3"),))
        adapter = _adapter(scenario)
        _submit(adapter, side=OrderSide.BUY, time_in_force=TimeInForceKind.IOC, quantity=Decimal("10"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        partial = [e for e in events if e.event_type is BrokerEventType.ORDER_PARTIALLY_FILLED]
        cancel = [e for e in events if e.event_type is BrokerEventType.CANCEL_ACKNOWLEDGED]
        assert len(partial) == 1
        assert partial[0].filled_quantity == Decimal("3")
        assert len(cancel) == 1

    def test_day_order_expires_at_session_close(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("0.5"), time_in_force=TimeInForceKind.DAY)
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)  # never triggers (limit far below market)
        adapter.expire_day_orders(event_time=_NOW)
        events = _poll_all(adapter)
        assert any(e.event_type is BrokerEventType.ORDER_EXPIRED for e in events)

    def test_gtc_order_survives_many_non_triggering_ticks(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("0.5"), time_in_force=TimeInForceKind.GTC)
        for i in range(50):
            adapter.advance_market_event(_tick(i, bid=1.0995, ask=1.1005), event_time=_NOW)
        adapter.expire_day_orders(event_time=_NOW)  # GTC must NOT expire here
        events = _poll_all(adapter)
        assert not any(e.event_type is BrokerEventType.ORDER_EXPIRED for e in events)


class TestPartialFillSchedule:
    def test_two_step_schedule_produces_partial_then_full(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.4"), Decimal("0.6")))
        adapter = _adapter(scenario)
        _submit(adapter, side=OrderSide.BUY, quantity=Decimal("10"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        adapter.advance_market_event(_tick(1, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        partial = [e for e in events if e.event_type is BrokerEventType.ORDER_PARTIALLY_FILLED]
        full = [e for e in events if e.event_type is BrokerEventType.ORDER_FILLED]
        assert len(partial) == 1 and partial[0].filled_quantity == Decimal("4")
        assert len(full) == 1 and full[0].filled_quantity == Decimal("6")


class TestCancelAndReplace:
    def test_cancel_before_fill_prevents_later_fill(self) -> None:
        adapter = _adapter()
        submit_cmd, _ = _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("0.5"))
        cancel_cmd = create_cancel_order_command(execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, client_order_id=submit_cmd.client_order_id, cancellation_reason="operator_request", command_sequence=1, event_time=_NOW)
        result = adapter.cancel_order(cancel_cmd, event_time=_NOW)
        assert result.accepted_for_processing
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        events = _poll_all(adapter)
        assert any(e.event_type is BrokerEventType.CANCEL_ACKNOWLEDGED for e in events)
        assert not any(e.event_type is BrokerEventType.ORDER_FILLED for e in events)

    def test_cancel_is_idempotent_on_already_cancelled_order(self) -> None:
        adapter = _adapter()
        submit_cmd, _ = _submit(adapter)
        cancel_cmd = create_cancel_order_command(execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, client_order_id=submit_cmd.client_order_id, cancellation_reason="operator_request", command_sequence=1, event_time=_NOW)
        first = adapter.cancel_order(cancel_cmd, event_time=_NOW)
        second = adapter.cancel_order(cancel_cmd, event_time=_NOW)
        assert first.accepted_for_processing and second.accepted_for_processing

    def test_replace_changes_quantity(self) -> None:
        adapter = _adapter()
        submit_cmd, _ = _submit(adapter, side=OrderSide.BUY, order_type=OrderTypeKind.LIMIT, limit_price=Decimal("0.5"), quantity=Decimal("10"))
        replace_cmd = create_replace_order_command(execution_session_id=_SHA_SESSION, execution_order_id="c" * 64, client_order_id=submit_cmd.client_order_id, command_sequence=1, event_time=_NOW, replacement_quantity=Decimal("20"))
        result = adapter.replace_order(replace_cmd, event_time=_NOW)
        assert result.accepted_for_processing
        events = _poll_all(adapter)
        replace_events = [e for e in events if e.event_type is BrokerEventType.REPLACE_ACKNOWLEDGED]
        assert replace_events[0].original_quantity == Decimal("20")


class TestRejectionRules:
    def test_quantity_above_threshold_is_rejected_synchronously(self) -> None:
        from dataclasses import replace

        rule = RejectionRuleSpec(rule_index=0, reject_instrument_id=None, reject_quantity_above=Decimal("5"), reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, rejection_rules=(rule,))
        adapter = _adapter(scenario)
        _, result = _submit(adapter, quantity=Decimal("10"))
        assert not result.accepted_for_processing
        assert result.rejection_reason == "rejection_rule_0"


class TestDisconnectReconnect:
    def test_submit_rejected_while_disconnected(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, disconnect_at_sequence=0, reconnect_at_sequence=5)
        adapter = _adapter(scenario)
        _, result = _submit(adapter)
        assert not result.accepted_for_processing
        assert result.rejection_reason == "adapter_disconnected"

    def test_health_reports_unavailable_while_disconnected(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, disconnect_at_sequence=0, reconnect_at_sequence=5)
        adapter = _adapter(scenario)
        health = adapter.health(event_time=_NOW)
        assert health.status is AdapterHealthStatus.UNAVAILABLE
        assert not health.can_submit


class TestPollEventsDeliveryManipulation:
    def test_duplicate_event_index_delivers_event_twice_with_same_id(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, duplicate_event_indices=(0,))
        adapter = _adapter(scenario)
        _submit(adapter)  # generates ORDER_RECEIVED (index 0) then ORDER_ACKNOWLEDGED (index 1)
        events = _poll_all(adapter)
        received = [e for e in events if e.event_type is BrokerEventType.ORDER_RECEIVED]
        assert len(received) == 2
        assert received[0].broker_event_id == received[1].broker_event_id

    def test_delayed_event_index_is_withheld_on_first_poll(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, delayed_event_indices=(0,))
        adapter = _adapter(scenario)
        _submit(adapter)
        first_poll = _poll_all(adapter)
        assert not any(e.event_type is BrokerEventType.ORDER_RECEIVED for e in first_poll)
        second_poll = adapter.poll_events(after_sequence=0, max_events=1000, event_time=_NOW)
        assert any(e.event_type is BrokerEventType.ORDER_RECEIVED for e in second_poll)

    def test_out_of_order_group_reverses_delivery_order(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, out_of_order_event_groups=((0, 1),))
        adapter = _adapter(scenario)
        _submit(adapter)  # ORDER_RECEIVED (0), ORDER_ACKNOWLEDGED (1)
        events = _poll_all(adapter)
        assert events[0].event_type is BrokerEventType.ORDER_ACKNOWLEDGED
        assert events[1].event_type is BrokerEventType.ORDER_RECEIVED

    def test_after_sequence_excludes_already_seen_events(self) -> None:
        adapter = _adapter()
        _submit(adapter)
        first = _poll_all(adapter)
        last_seq = max(e.broker_sequence for e in first)
        again = adapter.poll_events(after_sequence=last_seq, max_events=1000, event_time=_NOW)
        assert len(again) == 0


class TestHeartbeatAndQueries:
    def test_heartbeat_succeeds_when_connected(self) -> None:
        adapter = _adapter()
        outcome = adapter.heartbeat(event_time=_NOW)
        assert outcome.success

    def test_heartbeat_fails_at_configured_sequence(self) -> None:
        from dataclasses import replace

        scenario = replace(DEFAULT_DUMMY_BROKER_SCENARIO, heartbeat_failure_sequences=(1,))
        adapter = _adapter(scenario)
        outcome = adapter.heartbeat(event_time=_NOW)
        assert not outcome.success

    def test_query_account_reflects_a_fill(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, quantity=Decimal("10"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        account = adapter.query_account(event_time=_NOW)
        assert account.cash == Decimal("100000") - Decimal("10") * Decimal("1.1005")

    def test_query_positions_reflects_a_fill(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, quantity=Decimal("10"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        positions = adapter.query_positions(event_time=_NOW)
        assert len(positions) == 1
        assert positions[0].signed_quantity == Decimal("10")

    def test_query_open_orders_excludes_filled_orders(self) -> None:
        adapter = _adapter()
        _submit(adapter, side=OrderSide.BUY, quantity=Decimal("10"))
        adapter.advance_market_event(_tick(0, bid=1.0995, ask=1.1005), event_time=_NOW)
        assert adapter.query_open_orders(event_time=_NOW) == ()

    def test_query_order_by_client_order_id(self) -> None:
        adapter = _adapter()
        submit_cmd, _ = _submit(adapter)
        snapshot = adapter.query_order(broker_order_id=None, client_order_id=submit_cmd.client_order_id, event_time=_NOW)
        assert snapshot is not None
        assert snapshot.state is ExecutionOrderState.ACKNOWLEDGED


class TestCapabilitiesReflectTheScenariosOwnIdempotencyDeclaration:
    """Regression test for a real defect found during Milestone 8
    acceptance testing: `capabilities()` used to return the static
    `DETERMINISTIC_DUMMY_CAPABILITIES` constant unconditionally --
    `DummyBrokerScenarioSpec.supports_idempotent_submit/cancel/replace`
    (Section 12's own scenario knob, meant to let a test simulate a
    LESS-capable adapter) had no effect on `capabilities()` at all, which
    meant `recovery.recover_unknown_orders`'s `remains_unknown`/never-
    blind-retry safety path (Section 16/23) could never actually be
    exercised against the dummy broker regardless of scenario
    configuration."""

    def test_default_scenario_declares_full_idempotency(self) -> None:
        adapter = _adapter()
        capabilities = adapter.capabilities()
        assert capabilities.guarantees_idempotent_submit is True
        assert capabilities.guarantees_idempotent_cancel is True
        assert capabilities.guarantees_idempotent_replace is True

    def test_scenario_declaring_no_idempotency_guarantee_is_reflected_in_capabilities(self) -> None:
        import dataclasses

        non_idempotent_scenario = dataclasses.replace(
            DEFAULT_DUMMY_BROKER_SCENARIO, supports_idempotent_submit=False, supports_idempotent_cancel=False, supports_idempotent_replace=False,
        )
        adapter = _adapter(scenario=non_idempotent_scenario)
        capabilities = adapter.capabilities()
        assert capabilities.guarantees_idempotent_submit is False
        assert capabilities.guarantees_idempotent_cancel is False
        assert capabilities.guarantees_idempotent_replace is False
        # every other capability (order-type support, query support, ...) is unaffected
        assert capabilities.supports_market_orders is True
        assert capabilities.supports_limit_orders is True
        assert capabilities.supports_order_query is True
