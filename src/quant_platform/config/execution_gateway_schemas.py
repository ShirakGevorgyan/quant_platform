"""Pydantic configuration schema for Milestone 8's execution gateway
(Section 4/29). Same conventions as `config.paper_trading_schemas`: every
model is frozen, `extra="forbid"` (no unknown field can ever be silently
accepted -- this is also what makes "reject broker credentials"/"reject
endpoint URLs" true BY CONSTRUCTION: no such field is ever defined here,
and none can be smuggled in through `extra`), every float that must be
finite declares `allow_inf_nan=False`, and `execution_mode`/`adapter_kind`
are `Literal["test_only"]`/`Literal["deterministic_dummy"]` -- there is
no LIVE-like value anywhere in this schema for a safety check to even
need to catch; pydantic's own `Literal` validation makes any other
string a structural parse error before it ever reaches
`ExecutionGatewaySpec.__post_init__`."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution_gateway.models import (
    EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION,
    AdapterKind,
    ExecutionMode,
    SequencingPolicyKind,
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
)

_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


def _sha256_hex_field() -> object:
    return Field(pattern=_SHA256_HEX_PATTERN)


class SequencingPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Literal["strict_sequence", "timestamp_and_id", "arrival_order_only"] = "strict_sequence"

    def build(self) -> SequencingPolicySpec:
        return SequencingPolicySpec(policy=SequencingPolicyKind(self.policy))


class IdempotencyPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    durable_evidence_required: bool = True
    max_safe_retry_attempts: int = Field(default=3, ge=1)

    def build(self) -> IdempotencyPolicySpec:
        return IdempotencyPolicySpec(durable_evidence_required=self.durable_evidence_required, max_safe_retry_attempts=self.max_safe_retry_attempts)


class RecoveryPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_replay_events: int = Field(default=100_000, ge=1)
    unknown_resolution_timeout_events: int = Field(default=100, ge=1)

    def build(self) -> RecoveryPolicySpec:
        return RecoveryPolicySpec(max_replay_events=self.max_replay_events, unknown_resolution_timeout_events=self.unknown_resolution_timeout_events)


class ReconciliationPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quantity_tolerance: str = "0.000001"
    price_tolerance: str = "0.000001"
    cash_tolerance: str = "0.01"
    run_on_completion: bool = True

    def build(self) -> ReconciliationPolicySpec:
        return ReconciliationPolicySpec(
            quantity_tolerance=Decimal(self.quantity_tolerance), price_tolerance=Decimal(self.price_tolerance), cash_tolerance=Decimal(self.cash_tolerance),
            run_on_completion=self.run_on_completion,
        )


class HealthPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stale_after_events: int = Field(default=20, ge=1)
    degraded_after_consecutive_failures: int = Field(default=2, ge=1)
    unavailable_after_consecutive_failures: int = Field(default=5, ge=1)

    def build(self) -> HealthPolicySpec:
        return HealthPolicySpec(
            stale_after_events=self.stale_after_events, degraded_after_consecutive_failures=self.degraded_after_consecutive_failures,
            unavailable_after_consecutive_failures=self.unavailable_after_consecutive_failures,
        )


class HeartbeatPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    interval_events: int = Field(default=10, ge=1)
    missed_threshold_degraded: int = Field(default=2, ge=1)
    missed_threshold_halting: int = Field(default=5, ge=1)

    def build(self) -> HeartbeatPolicySpec:
        return HeartbeatPolicySpec(
            interval_events=self.interval_events, missed_threshold_degraded=self.missed_threshold_degraded, missed_threshold_halting=self.missed_threshold_halting,
        )


class KillSwitchPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_unresolved_unknown_operations: int = Field(default=3, ge=1)
    max_broker_sequence_conflicts: int = Field(default=1, ge=1)
    max_blocking_reconciliation_issues: int = Field(default=1, ge=1)

    def build(self) -> KillSwitchPolicySpec:
        return KillSwitchPolicySpec(
            max_unresolved_unknown_operations=self.max_unresolved_unknown_operations, max_broker_sequence_conflicts=self.max_broker_sequence_conflicts,
            max_blocking_reconciliation_issues=self.max_blocking_reconciliation_issues,
        )


class DispatchPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    require_dispatch_intent_before_call: bool = True
    max_commands_per_batch: int = Field(default=500, ge=1)

    def build(self) -> DispatchPolicySpec:
        return DispatchPolicySpec(require_dispatch_intent_before_call=self.require_dispatch_intent_before_call, max_commands_per_batch=self.max_commands_per_batch)


class RejectionRuleConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_index: int = Field(ge=0)
    reject_instrument_id: str | None = None
    reject_quantity_above: str | None = None
    reject_command_sequence: int | None = Field(default=None, ge=0)
    reject_client_order_id: str | None = None
    reject_unsupported_order_type: bool = False
    reject_when_disconnected: bool = False

    def build(self) -> RejectionRuleSpec:
        return RejectionRuleSpec(
            rule_index=self.rule_index, reject_instrument_id=self.reject_instrument_id,
            reject_quantity_above=(None if self.reject_quantity_above is None else Decimal(self.reject_quantity_above)),
            reject_command_sequence=self.reject_command_sequence, reject_client_order_id=self.reject_client_order_id,
            reject_unsupported_order_type=self.reject_unsupported_order_type, reject_when_disconnected=self.reject_when_disconnected,
        )


class DummyBrokerScenarioConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    acknowledgement_delay_events: int = Field(default=0, ge=0)
    fill_delay_events: int = Field(default=0, ge=0)
    partial_fill_schedule: tuple[str, ...] = ()
    rejection_rules: tuple[RejectionRuleConfigSchema, ...] = ()
    duplicate_event_indices: tuple[int, ...] = ()
    delayed_event_indices: tuple[int, ...] = ()
    out_of_order_event_groups: tuple[tuple[int, ...], ...] = ()
    disconnect_at_sequence: int | None = Field(default=None, ge=0)
    reconnect_at_sequence: int | None = Field(default=None, ge=0)
    heartbeat_failure_sequences: tuple[int, ...] = ()
    order_query_failure_sequences: tuple[int, ...] = ()
    account_query_failure_sequences: tuple[int, ...] = ()
    supports_idempotent_submit: bool = True
    supports_idempotent_cancel: bool = True
    supports_idempotent_replace: bool = True
    seed: int = Field(default=0, ge=0)

    def build(self) -> DummyBrokerScenarioSpec:
        return DummyBrokerScenarioSpec(
            acknowledgement_delay_events=self.acknowledgement_delay_events, fill_delay_events=self.fill_delay_events,
            partial_fill_schedule=tuple(Decimal(f) for f in self.partial_fill_schedule), rejection_rules=tuple(r.build() for r in self.rejection_rules),
            duplicate_event_indices=self.duplicate_event_indices, delayed_event_indices=self.delayed_event_indices,
            out_of_order_event_groups=self.out_of_order_event_groups, disconnect_at_sequence=self.disconnect_at_sequence,
            reconnect_at_sequence=self.reconnect_at_sequence, heartbeat_failure_sequences=self.heartbeat_failure_sequences,
            order_query_failure_sequences=self.order_query_failure_sequences, account_query_failure_sequences=self.account_query_failure_sequences,
            supports_idempotent_submit=self.supports_idempotent_submit, supports_idempotent_cancel=self.supports_idempotent_cancel,
            supports_idempotent_replace=self.supports_idempotent_replace, seed=self.seed,
        )


class ExecutionGatewayConfigSchema(BaseModel):
    """Top-level operator-facing config. `ml_artifacts_root` reuses the
    same storage-root convention `config.paper_trading_schemas.
    PaperTradingConfig` uses -- execution-session manifests/ledgers live
    under it in their own `execution_sessions/` namespace
    (`manifests.ExecutionSessionManifestStore`/`persistence.
    ExecutionSessionEventStore`), never colliding with `paper_sessions/`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: str
    research_storage_root: str
    historical_storage_root: str

    execution_mode: Literal["test_only"] = "test_only"
    adapter_kind: Literal["deterministic_dummy"] = "deterministic_dummy"

    paper_session_id: str = _sha256_hex_field()  # type: ignore[assignment]
    paper_trading_spec_id: str = _sha256_hex_field()  # type: ignore[assignment]
    promotion_decision_id: str = _sha256_hex_field()  # type: ignore[assignment]
    instrument_spec_id: str = _sha256_hex_field()  # type: ignore[assignment]

    sequencing_policy: SequencingPolicyConfigSchema = SequencingPolicyConfigSchema()
    idempotency_policy: IdempotencyPolicyConfigSchema = IdempotencyPolicyConfigSchema()
    recovery_policy: RecoveryPolicyConfigSchema = RecoveryPolicyConfigSchema()
    reconciliation_policy: ReconciliationPolicyConfigSchema = ReconciliationPolicyConfigSchema()
    health_policy: HealthPolicyConfigSchema = HealthPolicyConfigSchema()
    heartbeat_policy: HeartbeatPolicyConfigSchema = HeartbeatPolicyConfigSchema()
    kill_switch_policy: KillSwitchPolicyConfigSchema = KillSwitchPolicyConfigSchema()
    dispatch_policy: DispatchPolicyConfigSchema = DispatchPolicyConfigSchema()
    dummy_broker_scenario: DummyBrokerScenarioConfigSchema = DummyBrokerScenarioConfigSchema()
    seed: int = Field(default=0, ge=0)

    def build(self) -> ExecutionGatewaySpec:
        return ExecutionGatewaySpec(
            schema_version=EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION, execution_mode=ExecutionMode(self.execution_mode), adapter_kind=AdapterKind(self.adapter_kind),
            paper_session_id=self.paper_session_id, paper_trading_spec_id=self.paper_trading_spec_id, promotion_decision_id=self.promotion_decision_id,
            instrument_spec_id=self.instrument_spec_id, sequencing_policy=self.sequencing_policy.build(), idempotency_policy=self.idempotency_policy.build(),
            recovery_policy=self.recovery_policy.build(), reconciliation_policy=self.reconciliation_policy.build(), health_policy=self.health_policy.build(),
            heartbeat_policy=self.heartbeat_policy.build(), kill_switch_policy=self.kill_switch_policy.build(), dispatch_policy=self.dispatch_policy.build(),
            dummy_broker_scenario=self.dummy_broker_scenario.build(), seed=self.seed,
        )


__all__ = [
    "DispatchPolicyConfigSchema",
    "DummyBrokerScenarioConfigSchema",
    "ExecutionGatewayConfigSchema",
    "HealthPolicyConfigSchema",
    "HeartbeatPolicyConfigSchema",
    "IdempotencyPolicyConfigSchema",
    "KillSwitchPolicyConfigSchema",
    "ReconciliationPolicyConfigSchema",
    "RecoveryPolicyConfigSchema",
    "RejectionRuleConfigSchema",
    "SequencingPolicyConfigSchema",
]
