"""Unit tests for `execution_gateway.specs` (Milestone 8, Section 4):
`ExecutionGatewaySpec` validation, content-addressed identity, and the
fail-closed rejection list Section 4 requires."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExecutionGatewaySpecError
from quant_platform.execution_gateway.models import AdapterKind, ExecutionMode, SequencingPolicyKind
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
    RejectionRuleSpec,
    SequencingPolicySpec,
    compute_execution_gateway_spec_id,
    verify_execution_gateway_spec_identity,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _spec(**overrides: object) -> ExecutionGatewaySpec:
    base: dict[str, object] = {
        "schema_version": 1, "execution_mode": ExecutionMode.TEST_ONLY, "adapter_kind": AdapterKind.DETERMINISTIC_DUMMY, "paper_session_id": _SHA_A,
        "paper_trading_spec_id": _SHA_B, "promotion_decision_id": _SHA_C, "instrument_spec_id": _SHA_D,
        "sequencing_policy": SequencingPolicySpec(policy=SequencingPolicyKind.STRICT_SEQUENCE),
        "idempotency_policy": IdempotencyPolicySpec(durable_evidence_required=True, max_safe_retry_attempts=3),
        "recovery_policy": RecoveryPolicySpec(max_replay_events=10_000, unknown_resolution_timeout_events=50),
        "reconciliation_policy": ReconciliationPolicySpec(quantity_tolerance=Decimal("0.000001"), price_tolerance=Decimal("0.000001"), cash_tolerance=Decimal("0.01"), run_on_completion=True),
        "health_policy": HealthPolicySpec(stale_after_events=20, degraded_after_consecutive_failures=2, unavailable_after_consecutive_failures=5),
        "heartbeat_policy": HeartbeatPolicySpec(interval_events=10, missed_threshold_degraded=2, missed_threshold_halting=5),
        "kill_switch_policy": KillSwitchPolicySpec(max_unresolved_unknown_operations=3, max_broker_sequence_conflicts=1, max_blocking_reconciliation_issues=1),
        "dispatch_policy": DispatchPolicySpec(require_dispatch_intent_before_call=True, max_commands_per_batch=100),
        "dummy_broker_scenario": DEFAULT_DUMMY_BROKER_SCENARIO, "seed": 7,
    }
    base.update(overrides)
    return ExecutionGatewaySpec(**base)  # type: ignore[arg-type]


class TestExecutionGatewaySpecValidConstruction:
    def test_default_spec_constructs(self) -> None:
        spec = _spec()
        assert spec.execution_mode is ExecutionMode.TEST_ONLY
        assert spec.adapter_kind is AdapterKind.DETERMINISTIC_DUMMY

    def test_round_trips_through_json(self) -> None:
        spec = _spec()
        restored = ExecutionGatewaySpec.from_json_dict(spec.to_json_dict())
        assert restored.to_json_dict() == spec.to_json_dict()


class TestExecutionGatewaySpecIdentity:
    def test_identity_is_deterministic(self) -> None:
        a = compute_execution_gateway_spec_id(_spec()).execution_gateway_spec_id
        b = compute_execution_gateway_spec_id(_spec()).execution_gateway_spec_id
        assert a == b
        assert len(a) == 64

    def test_identity_is_a_pure_function_verify_matches(self) -> None:
        spec = _spec()
        identity = compute_execution_gateway_spec_id(spec)
        assert verify_execution_gateway_spec_identity(spec, identity.execution_gateway_spec_id)
        assert not verify_execution_gateway_spec_identity(spec, "0" * 64)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda s: replace(s, seed=s.seed + 1),
            lambda s: replace(s, paper_session_id=_SHA_D),
            lambda s: replace(s, dispatch_policy=replace(s.dispatch_policy, max_commands_per_batch=999)),
            lambda s: replace(s, recovery_policy=replace(s.recovery_policy, max_replay_events=1)),
            lambda s: replace(s, dummy_broker_scenario=replace(s.dummy_broker_scenario, seed=123)),
        ],
    )
    def test_changing_any_result_affecting_field_changes_identity(self, mutate) -> None:
        original = _spec()
        mutated = mutate(original)
        assert compute_execution_gateway_spec_id(original).execution_gateway_spec_id != compute_execution_gateway_spec_id(mutated).execution_gateway_spec_id

    def test_rejection_rule_declaration_order_does_not_affect_identity(self) -> None:
        rule_a = RejectionRuleSpec(rule_index=0, reject_instrument_id="EURUSD", reject_quantity_above=None, reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        rule_b = RejectionRuleSpec(rule_index=1, reject_instrument_id=None, reject_quantity_above=Decimal("10"), reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        forward = replace(DEFAULT_DUMMY_BROKER_SCENARIO, rejection_rules=(rule_a, rule_b))
        backward = replace(DEFAULT_DUMMY_BROKER_SCENARIO, rejection_rules=(rule_b, rule_a))
        spec_forward = _spec(dummy_broker_scenario=forward)
        spec_backward = _spec(dummy_broker_scenario=backward)
        assert compute_execution_gateway_spec_id(spec_forward).execution_gateway_spec_id == compute_execution_gateway_spec_id(spec_backward).execution_gateway_spec_id

    def test_partial_fill_schedule_order_affects_identity(self) -> None:
        forward = replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.3"), Decimal("0.7")))
        backward = replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.7"), Decimal("0.3")))
        spec_forward = _spec(dummy_broker_scenario=forward)
        spec_backward = _spec(dummy_broker_scenario=backward)
        assert compute_execution_gateway_spec_id(spec_forward).execution_gateway_spec_id != compute_execution_gateway_spec_id(spec_backward).execution_gateway_spec_id


class TestExecutionGatewaySpecFailClosedRejections:
    def test_rejects_non_sha256_paper_session_id(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            _spec(paper_session_id="not-a-hash")

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            _spec(seed=-1)

    def test_rejects_negative_acknowledgement_delay(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, acknowledgement_delay_events=-1)

    def test_rejects_negative_fill_delay(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, fill_delay_events=-1)

    def test_rejects_zero_partial_fill_fraction(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0"),))

    def test_rejects_negative_partial_fill_fraction(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("-0.1"),))

    def test_rejects_partial_fill_schedule_summing_above_one(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, partial_fill_schedule=(Decimal("0.6"), Decimal("0.6")))

    def test_rejects_non_finite_reject_quantity_above(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            RejectionRuleSpec(rule_index=0, reject_instrument_id=None, reject_quantity_above=Decimal("NaN"), reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)

    def test_rejects_duplicate_rejection_rule_indices(self) -> None:
        rule_a = RejectionRuleSpec(rule_index=0, reject_instrument_id="EURUSD", reject_quantity_above=None, reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        rule_b = RejectionRuleSpec(rule_index=0, reject_instrument_id="GBPUSD", reject_quantity_above=None, reject_command_sequence=None, reject_client_order_id=None, reject_unsupported_order_type=False, reject_when_disconnected=False)
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, rejection_rules=(rule_a, rule_b))

    def test_rejects_reconnect_before_disconnect(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, disconnect_at_sequence=10, reconnect_at_sequence=5)

    def test_rejects_reconnect_without_disconnect(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, disconnect_at_sequence=None, reconnect_at_sequence=5)

    def test_rejects_unsorted_duplicate_event_indices(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, duplicate_event_indices=(3, 1))

    def test_rejects_repeated_duplicate_event_indices(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, duplicate_event_indices=(1, 1, 2))

    def test_rejects_out_of_order_group_with_fewer_than_two_indices(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, out_of_order_event_groups=((1,),))

    def test_rejects_overlapping_out_of_order_groups(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            replace(DEFAULT_DUMMY_BROKER_SCENARIO, out_of_order_event_groups=((1, 2), (2, 3)))

    def test_rejects_non_strict_sequence_policy_for_dummy_adapter(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            _spec(sequencing_policy=SequencingPolicySpec(policy=SequencingPolicyKind.ARRIVAL_ORDER_ONLY))

    def test_rejects_idempotency_policy_without_durable_evidence(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            IdempotencyPolicySpec(durable_evidence_required=False, max_safe_retry_attempts=1)

    def test_rejects_dispatch_policy_without_intent_before_call(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            DispatchPolicySpec(require_dispatch_intent_before_call=False, max_commands_per_batch=10)

    def test_rejects_health_policy_unavailable_below_degraded(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            HealthPolicySpec(stale_after_events=10, degraded_after_consecutive_failures=5, unavailable_after_consecutive_failures=2)

    def test_rejects_heartbeat_policy_halting_below_degraded(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            HeartbeatPolicySpec(interval_events=10, missed_threshold_degraded=5, missed_threshold_halting=2)

    def test_rejects_zero_max_replay_events(self) -> None:
        with pytest.raises(ExecutionGatewaySpecError):
            RecoveryPolicySpec(max_replay_events=0, unknown_resolution_timeout_events=1)

    def test_rejects_unsupported_execution_mode_value_is_structurally_impossible(self) -> None:
        # There is no way to even reference a non-existent enum member --
        # this test documents that the allow-list exists as defense in
        # depth (Section 4) even though `ExecutionMode` is single-member.
        assert {m.value for m in ExecutionMode} == {"test_only"}
        assert {a.value for a in AdapterKind} == {"deterministic_dummy"}
