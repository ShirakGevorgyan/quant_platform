"""Immutable specifications for `quant_platform.execution_gateway`
(Milestone 8, Section 4/12). Every spec here is a frozen, slotted
dataclass with an explicit `__post_init__` validator, following
`paper_trading.specs`'s identical convention exactly, including its
hard-won durable-order-versus-identity-canonicalization split:
`to_json_dict()` always preserves caller-declared order (the durable,
round-tripped representation); canonicalization of genuinely UNORDERED
fields (so declaration order never affects identity) happens ONLY inside
`to_identity_payload()`.

`ExecutionGatewaySpec` is the top-level, content-addressed specification.
Every field that can affect a session's economic or state-machine outcome
lives here -- including the dummy broker's own deterministic scenario and
seed, and every operational safety bound (Section 37: "Any operational
safety limit that can affect a result must be part of the relevant
specification identity")."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_platform.core.exceptions import ExecutionGatewaySpecError
from quant_platform.execution_gateway.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.execution_gateway.models import (
    EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION,
    AdapterKind,
    ExecutionMode,
    SequencingPolicyKind,
)

SUPPORTED_ADAPTER_KINDS: frozenset[AdapterKind] = frozenset({AdapterKind.DETERMINISTIC_DUMMY})
"""Explicit allow-list, defense-in-depth alongside `AdapterKind` being a
single-member enum today: if a future MT5-adapter milestone ever adds a
member to `AdapterKind` without also deliberately updating this
allow-list, `ExecutionGatewaySpec.__post_init__` still fails closed
rather than silently accepting it."""

SUPPORTED_EXECUTION_MODES: frozenset[ExecutionMode] = frozenset({ExecutionMode.TEST_ONLY})

_DUMMY_BROKER_REQUIRED_SEQUENCING_POLICY = SequencingPolicyKind.STRICT_SEQUENCE
"""Section 15: "Milestone 8 dummy broker must default to STRICT_SEQUENCE."
This milestone goes further and REQUIRES it whenever `adapter_kind is
DETERMINISTIC_DUMMY` (the only adapter kind that exists) -- `sequencing.py`
itself implements all three policies and is unit-tested against all three,
but nothing in this milestone constructs a spec that selects a
lower-assurance policy for the one adapter this milestone actually runs."""


def _finite_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite():
        raise ExecutionGatewaySpecError(f"{field_name} must be finite, got {value!r}")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    _finite_decimal(value, field_name=field_name)
    if value <= 0:
        raise ExecutionGatewaySpecError(f"{field_name} must be > 0, got {value!r}")


def _non_negative_decimal(value: Decimal, *, field_name: str) -> None:
    _finite_decimal(value, field_name=field_name)
    if value < 0:
        raise ExecutionGatewaySpecError(f"{field_name} must be >= 0, got {value!r}")


def _positive_int(value: int, *, field_name: str) -> None:
    if value < 1:
        raise ExecutionGatewaySpecError(f"{field_name} must be >= 1, got {value!r}")


def _non_negative_int(value: int, *, field_name: str) -> None:
    if value < 0:
        raise ExecutionGatewaySpecError(f"{field_name} must be >= 0, got {value!r}")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise ExecutionGatewaySpecError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _require_sorted_unique_ints(values: tuple[int, ...], *, field_name: str) -> None:
    for v in values:
        _non_negative_int(v, field_name=f"{field_name}[]")
    if list(values) != sorted(set(values)):
        raise ExecutionGatewaySpecError(f"{field_name} must be sorted, non-negative, and free of duplicates, got {values!r}")


# --------------------------------------------------------------------------
# Sub-policies (Section 4)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SequencingPolicySpec:
    policy: SequencingPolicyKind

    def to_json_dict(self) -> dict[str, object]:
        return {"policy": self.policy.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SequencingPolicySpec:
        return cls(policy=SequencingPolicyKind(raw["policy"]))


@dataclass(frozen=True, slots=True)
class IdempotencyPolicySpec:
    durable_evidence_required: bool
    max_safe_retry_attempts: int

    def __post_init__(self) -> None:
        if not self.durable_evidence_required:
            raise ExecutionGatewaySpecError("IdempotencyPolicySpec.durable_evidence_required must be True (Section 16: never rely only on in-memory evidence)")
        _positive_int(self.max_safe_retry_attempts, field_name="IdempotencyPolicySpec.max_safe_retry_attempts")

    def to_json_dict(self) -> dict[str, object]:
        return {"durable_evidence_required": self.durable_evidence_required, "max_safe_retry_attempts": self.max_safe_retry_attempts}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> IdempotencyPolicySpec:
        return cls(durable_evidence_required=bool(raw["durable_evidence_required"]), max_safe_retry_attempts=int(str(raw["max_safe_retry_attempts"])))


@dataclass(frozen=True, slots=True)
class RecoveryPolicySpec:
    max_replay_events: int
    """Section 37's operational maximum, separate from any financial risk
    limit -- caps how many ledger/broker events a single recovery or
    replay pass will process, independent of the event stream's own
    length, so a corrupted or maliciously long source can never drive an
    unbounded recovery."""
    unknown_resolution_timeout_events: int

    def __post_init__(self) -> None:
        _positive_int(self.max_replay_events, field_name="RecoveryPolicySpec.max_replay_events")
        _positive_int(self.unknown_resolution_timeout_events, field_name="RecoveryPolicySpec.unknown_resolution_timeout_events")

    def to_json_dict(self) -> dict[str, object]:
        return {"max_replay_events": self.max_replay_events, "unknown_resolution_timeout_events": self.unknown_resolution_timeout_events}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RecoveryPolicySpec:
        return cls(max_replay_events=int(str(raw["max_replay_events"])), unknown_resolution_timeout_events=int(str(raw["unknown_resolution_timeout_events"])))


@dataclass(frozen=True, slots=True)
class ReconciliationPolicySpec:
    quantity_tolerance: Decimal
    price_tolerance: Decimal
    cash_tolerance: Decimal
    run_on_completion: bool

    def __post_init__(self) -> None:
        _non_negative_decimal(self.quantity_tolerance, field_name="ReconciliationPolicySpec.quantity_tolerance")
        _non_negative_decimal(self.price_tolerance, field_name="ReconciliationPolicySpec.price_tolerance")
        _non_negative_decimal(self.cash_tolerance, field_name="ReconciliationPolicySpec.cash_tolerance")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "quantity_tolerance": decimal_to_json(self.quantity_tolerance), "price_tolerance": decimal_to_json(self.price_tolerance),
            "cash_tolerance": decimal_to_json(self.cash_tolerance), "run_on_completion": self.run_on_completion,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReconciliationPolicySpec:
        return cls(
            quantity_tolerance=parse_decimal(raw["quantity_tolerance"], field_name="quantity_tolerance"),
            price_tolerance=parse_decimal(raw["price_tolerance"], field_name="price_tolerance"),
            cash_tolerance=parse_decimal(raw["cash_tolerance"], field_name="cash_tolerance"), run_on_completion=bool(raw["run_on_completion"]),
        )


@dataclass(frozen=True, slots=True)
class HealthPolicySpec:
    stale_after_events: int
    degraded_after_consecutive_failures: int
    unavailable_after_consecutive_failures: int

    def __post_init__(self) -> None:
        _positive_int(self.stale_after_events, field_name="HealthPolicySpec.stale_after_events")
        _positive_int(self.degraded_after_consecutive_failures, field_name="HealthPolicySpec.degraded_after_consecutive_failures")
        _positive_int(self.unavailable_after_consecutive_failures, field_name="HealthPolicySpec.unavailable_after_consecutive_failures")
        if self.unavailable_after_consecutive_failures < self.degraded_after_consecutive_failures:
            raise ExecutionGatewaySpecError("HealthPolicySpec.unavailable_after_consecutive_failures must be >= degraded_after_consecutive_failures")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stale_after_events": self.stale_after_events, "degraded_after_consecutive_failures": self.degraded_after_consecutive_failures,
            "unavailable_after_consecutive_failures": self.unavailable_after_consecutive_failures,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HealthPolicySpec:
        return cls(
            stale_after_events=int(str(raw["stale_after_events"])), degraded_after_consecutive_failures=int(str(raw["degraded_after_consecutive_failures"])),
            unavailable_after_consecutive_failures=int(str(raw["unavailable_after_consecutive_failures"])),
        )


@dataclass(frozen=True, slots=True)
class HeartbeatPolicySpec:
    interval_events: int
    missed_threshold_degraded: int
    missed_threshold_halting: int

    def __post_init__(self) -> None:
        _positive_int(self.interval_events, field_name="HeartbeatPolicySpec.interval_events")
        _positive_int(self.missed_threshold_degraded, field_name="HeartbeatPolicySpec.missed_threshold_degraded")
        _positive_int(self.missed_threshold_halting, field_name="HeartbeatPolicySpec.missed_threshold_halting")
        if self.missed_threshold_halting < self.missed_threshold_degraded:
            raise ExecutionGatewaySpecError("HeartbeatPolicySpec.missed_threshold_halting must be >= missed_threshold_degraded")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "interval_events": self.interval_events, "missed_threshold_degraded": self.missed_threshold_degraded,
            "missed_threshold_halting": self.missed_threshold_halting,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HeartbeatPolicySpec:
        return cls(
            interval_events=int(str(raw["interval_events"])), missed_threshold_degraded=int(str(raw["missed_threshold_degraded"])),
            missed_threshold_halting=int(str(raw["missed_threshold_halting"])),
        )


@dataclass(frozen=True, slots=True)
class KillSwitchPolicySpec:
    max_unresolved_unknown_operations: int
    max_broker_sequence_conflicts: int
    max_blocking_reconciliation_issues: int

    def __post_init__(self) -> None:
        _positive_int(self.max_unresolved_unknown_operations, field_name="KillSwitchPolicySpec.max_unresolved_unknown_operations")
        _positive_int(self.max_broker_sequence_conflicts, field_name="KillSwitchPolicySpec.max_broker_sequence_conflicts")
        _positive_int(self.max_blocking_reconciliation_issues, field_name="KillSwitchPolicySpec.max_blocking_reconciliation_issues")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "max_unresolved_unknown_operations": self.max_unresolved_unknown_operations,
            "max_broker_sequence_conflicts": self.max_broker_sequence_conflicts,
            "max_blocking_reconciliation_issues": self.max_blocking_reconciliation_issues,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> KillSwitchPolicySpec:
        return cls(
            max_unresolved_unknown_operations=int(str(raw["max_unresolved_unknown_operations"])),
            max_broker_sequence_conflicts=int(str(raw["max_broker_sequence_conflicts"])),
            max_blocking_reconciliation_issues=int(str(raw["max_blocking_reconciliation_issues"])),
        )


@dataclass(frozen=True, slots=True)
class DispatchPolicySpec:
    require_dispatch_intent_before_call: bool
    max_commands_per_batch: int
    """Section 37's operational batch bound for polling/dispatch loops."""

    def __post_init__(self) -> None:
        if not self.require_dispatch_intent_before_call:
            raise ExecutionGatewaySpecError("DispatchPolicySpec.require_dispatch_intent_before_call must be True (Section 17: intent must be durable before any adapter call)")
        _positive_int(self.max_commands_per_batch, field_name="DispatchPolicySpec.max_commands_per_batch")

    def to_json_dict(self) -> dict[str, object]:
        return {"require_dispatch_intent_before_call": self.require_dispatch_intent_before_call, "max_commands_per_batch": self.max_commands_per_batch}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DispatchPolicySpec:
        return cls(require_dispatch_intent_before_call=bool(raw["require_dispatch_intent_before_call"]), max_commands_per_batch=int(str(raw["max_commands_per_batch"])))


# --------------------------------------------------------------------------
# Deterministic dummy broker scenario (Section 12)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RejectionRuleSpec:
    """One immutable, deterministic pre-dispatch/at-dispatch rejection
    predicate. `dummy_broker.py` evaluates every rule in a scenario as an
    OR (any matching rule rejects) -- rule evaluation order therefore
    never affects the outcome, which is what lets `rule_index` be the
    sole uniqueness key and `to_identity_payload` canonicalize this
    collection by sorting on it."""

    rule_index: int
    reject_instrument_id: str | None
    reject_quantity_above: Decimal | None
    reject_command_sequence: int | None
    reject_client_order_id: str | None
    reject_unsupported_order_type: bool
    reject_when_disconnected: bool

    def __post_init__(self) -> None:
        _non_negative_int(self.rule_index, field_name="RejectionRuleSpec.rule_index")
        if self.reject_quantity_above is not None:
            _positive_decimal(self.reject_quantity_above, field_name="RejectionRuleSpec.reject_quantity_above")
        if self.reject_command_sequence is not None:
            _non_negative_int(self.reject_command_sequence, field_name="RejectionRuleSpec.reject_command_sequence")
        if not any((
            self.reject_instrument_id, self.reject_quantity_above is not None, self.reject_command_sequence is not None, self.reject_client_order_id,
            self.reject_unsupported_order_type, self.reject_when_disconnected,
        )):
            raise ExecutionGatewaySpecError(f"RejectionRuleSpec[{self.rule_index}] must set at least one predicate")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "rule_index": self.rule_index, "reject_instrument_id": self.reject_instrument_id,
            "reject_quantity_above": (None if self.reject_quantity_above is None else decimal_to_json(self.reject_quantity_above)),
            "reject_command_sequence": self.reject_command_sequence, "reject_client_order_id": self.reject_client_order_id,
            "reject_unsupported_order_type": self.reject_unsupported_order_type, "reject_when_disconnected": self.reject_when_disconnected,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RejectionRuleSpec:
        return cls(
            rule_index=int(str(raw["rule_index"])), reject_instrument_id=(None if raw.get("reject_instrument_id") is None else str(raw["reject_instrument_id"])),
            reject_quantity_above=(None if raw.get("reject_quantity_above") is None else parse_decimal(raw["reject_quantity_above"], field_name="reject_quantity_above")),
            reject_command_sequence=(None if raw.get("reject_command_sequence") is None else int(str(raw["reject_command_sequence"]))),
            reject_client_order_id=(None if raw.get("reject_client_order_id") is None else str(raw["reject_client_order_id"])),
            reject_unsupported_order_type=bool(raw["reject_unsupported_order_type"]), reject_when_disconnected=bool(raw["reject_when_disconnected"]),
        )


@dataclass(frozen=True, slots=True)
class DummyBrokerScenarioSpec:
    acknowledgement_delay_events: int
    fill_delay_events: int
    partial_fill_schedule: tuple[Decimal, ...]
    """ORDER-SENSITIVE (Section 4: "declared durable order must be
    preserved"): the sequence in which partial fill fractions are applied
    over successive eligible events. Each entry in (0, 1]; the running sum
    must never exceed 1 (never invents more quantity than the order has).
    Empty means "no scheduled partial fills" -- an order fills for its
    full available quantity or not at all, subject to the rest of the
    scenario."""
    rejection_rules: tuple[RejectionRuleSpec, ...]
    duplicate_event_indices: tuple[int, ...]
    delayed_event_indices: tuple[int, ...]
    out_of_order_event_groups: tuple[tuple[int, ...], ...]
    disconnect_at_sequence: int | None
    reconnect_at_sequence: int | None
    heartbeat_failure_sequences: tuple[int, ...]
    order_query_failure_sequences: tuple[int, ...]
    account_query_failure_sequences: tuple[int, ...]
    supports_idempotent_submit: bool
    supports_idempotent_cancel: bool
    supports_idempotent_replace: bool
    seed: int

    def __post_init__(self) -> None:
        _non_negative_int(self.acknowledgement_delay_events, field_name="DummyBrokerScenarioSpec.acknowledgement_delay_events")
        _non_negative_int(self.fill_delay_events, field_name="DummyBrokerScenarioSpec.fill_delay_events")
        running_total = Decimal(0)
        for i, fraction in enumerate(self.partial_fill_schedule):
            _positive_decimal(fraction, field_name=f"DummyBrokerScenarioSpec.partial_fill_schedule[{i}]")
            if fraction > 1:
                raise ExecutionGatewaySpecError(f"DummyBrokerScenarioSpec.partial_fill_schedule[{i}]={fraction!r} must be <= 1")
            running_total += fraction
            if running_total > 1:
                raise ExecutionGatewaySpecError(f"DummyBrokerScenarioSpec.partial_fill_schedule cumulative fraction exceeds 1 at index {i} (running_total={running_total!r})")
        rule_indices = [r.rule_index for r in self.rejection_rules]
        if len(set(rule_indices)) != len(rule_indices):
            raise ExecutionGatewaySpecError(f"DummyBrokerScenarioSpec.rejection_rules must not repeat a rule_index, got {rule_indices!r}")
        _require_sorted_unique_ints(self.duplicate_event_indices, field_name="DummyBrokerScenarioSpec.duplicate_event_indices")
        _require_sorted_unique_ints(self.delayed_event_indices, field_name="DummyBrokerScenarioSpec.delayed_event_indices")
        seen_in_groups: set[int] = set()
        for group_i, group in enumerate(self.out_of_order_event_groups):
            if len(group) < 2:
                raise ExecutionGatewaySpecError(f"DummyBrokerScenarioSpec.out_of_order_event_groups[{group_i}] must contain at least 2 indices, got {group!r}")
            _require_sorted_unique_ints(group, field_name=f"DummyBrokerScenarioSpec.out_of_order_event_groups[{group_i}]")
            if seen_in_groups & set(group):
                raise ExecutionGatewaySpecError(f"DummyBrokerScenarioSpec.out_of_order_event_groups[{group_i}] overlaps a previous group: {sorted(seen_in_groups & set(group))!r}")
            seen_in_groups |= set(group)
        if self.disconnect_at_sequence is not None:
            _non_negative_int(self.disconnect_at_sequence, field_name="DummyBrokerScenarioSpec.disconnect_at_sequence")
        if self.reconnect_at_sequence is not None:
            _non_negative_int(self.reconnect_at_sequence, field_name="DummyBrokerScenarioSpec.reconnect_at_sequence")
            if self.disconnect_at_sequence is None:
                raise ExecutionGatewaySpecError("DummyBrokerScenarioSpec.reconnect_at_sequence requires disconnect_at_sequence to also be set")
            if self.reconnect_at_sequence <= self.disconnect_at_sequence:
                raise ExecutionGatewaySpecError(
                    f"DummyBrokerScenarioSpec.reconnect_at_sequence ({self.reconnect_at_sequence}) must be > disconnect_at_sequence ({self.disconnect_at_sequence})"
                )
        _require_sorted_unique_ints(self.heartbeat_failure_sequences, field_name="DummyBrokerScenarioSpec.heartbeat_failure_sequences")
        _require_sorted_unique_ints(self.order_query_failure_sequences, field_name="DummyBrokerScenarioSpec.order_query_failure_sequences")
        _require_sorted_unique_ints(self.account_query_failure_sequences, field_name="DummyBrokerScenarioSpec.account_query_failure_sequences")
        _non_negative_int(self.seed, field_name="DummyBrokerScenarioSpec.seed")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "acknowledgement_delay_events": self.acknowledgement_delay_events, "fill_delay_events": self.fill_delay_events,
            "partial_fill_schedule": [decimal_to_json(f) for f in self.partial_fill_schedule],
            "rejection_rules": [r.to_json_dict() for r in self.rejection_rules], "duplicate_event_indices": list(self.duplicate_event_indices),
            "delayed_event_indices": list(self.delayed_event_indices), "out_of_order_event_groups": [list(g) for g in self.out_of_order_event_groups],
            "disconnect_at_sequence": self.disconnect_at_sequence, "reconnect_at_sequence": self.reconnect_at_sequence,
            "heartbeat_failure_sequences": list(self.heartbeat_failure_sequences), "order_query_failure_sequences": list(self.order_query_failure_sequences),
            "account_query_failure_sequences": list(self.account_query_failure_sequences), "supports_idempotent_submit": self.supports_idempotent_submit,
            "supports_idempotent_cancel": self.supports_idempotent_cancel, "supports_idempotent_replace": self.supports_idempotent_replace, "seed": self.seed,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = self.to_json_dict()
        payload["rejection_rules"] = [r.to_json_dict() for r in sorted(self.rejection_rules, key=lambda r: r.rule_index)]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DummyBrokerScenarioSpec:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        return cls(
            acknowledgement_delay_events=int(str(raw["acknowledgement_delay_events"])), fill_delay_events=int(str(raw["fill_delay_events"])),
            partial_fill_schedule=tuple(parse_decimal(f, field_name="partial_fill_schedule[]") for f in as_json_list(raw.get("partial_fill_schedule") or [], field_name="partial_fill_schedule")),
            rejection_rules=tuple(
                RejectionRuleSpec.from_json_dict(as_json_dict(r, field_name="rejection_rules[]")) for r in as_json_list(raw.get("rejection_rules") or [], field_name="rejection_rules")
            ),
            duplicate_event_indices=tuple(int(str(i)) for i in as_json_list(raw.get("duplicate_event_indices") or [], field_name="duplicate_event_indices")),
            delayed_event_indices=tuple(int(str(i)) for i in as_json_list(raw.get("delayed_event_indices") or [], field_name="delayed_event_indices")),
            out_of_order_event_groups=tuple(
                tuple(int(str(i)) for i in as_json_list(g, field_name="out_of_order_event_groups[]")) for g in as_json_list(raw.get("out_of_order_event_groups") or [], field_name="out_of_order_event_groups")
            ),
            disconnect_at_sequence=(None if raw.get("disconnect_at_sequence") is None else int(str(raw["disconnect_at_sequence"]))),
            reconnect_at_sequence=(None if raw.get("reconnect_at_sequence") is None else int(str(raw["reconnect_at_sequence"]))),
            heartbeat_failure_sequences=tuple(int(str(i)) for i in as_json_list(raw.get("heartbeat_failure_sequences") or [], field_name="heartbeat_failure_sequences")),
            order_query_failure_sequences=tuple(int(str(i)) for i in as_json_list(raw.get("order_query_failure_sequences") or [], field_name="order_query_failure_sequences")),
            account_query_failure_sequences=tuple(int(str(i)) for i in as_json_list(raw.get("account_query_failure_sequences") or [], field_name="account_query_failure_sequences")),
            supports_idempotent_submit=bool(raw["supports_idempotent_submit"]), supports_idempotent_cancel=bool(raw["supports_idempotent_cancel"]),
            supports_idempotent_replace=bool(raw["supports_idempotent_replace"]), seed=int(str(raw["seed"])),
        )


DEFAULT_DUMMY_BROKER_SCENARIO = DummyBrokerScenarioSpec(
    acknowledgement_delay_events=0, fill_delay_events=0, partial_fill_schedule=(), rejection_rules=(), duplicate_event_indices=(), delayed_event_indices=(),
    out_of_order_event_groups=(), disconnect_at_sequence=None, reconnect_at_sequence=None, heartbeat_failure_sequences=(), order_query_failure_sequences=(),
    account_query_failure_sequences=(), supports_idempotent_submit=True, supports_idempotent_cancel=True, supports_idempotent_replace=True, seed=0,
)


# --------------------------------------------------------------------------
# Top-level spec (Section 4)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionGatewaySpec:
    """The top-level, content-addressed specification for one execution
    session. `execution_gateway_spec_id` (see
    `compute_execution_gateway_spec_id`) is a pure function of every field
    below except `schema_version`. `paper_session_id` is the source
    Milestone 7 paper session this execution session bridges from --
    identity/eligibility cross-checking against the ACTUAL paper session
    happens in `paper_bridge.py` (which has access to the paper session
    store this spec's own `__post_init__` does not), never here."""

    schema_version: int
    execution_mode: ExecutionMode
    adapter_kind: AdapterKind

    paper_session_id: str
    paper_trading_spec_id: str
    promotion_decision_id: str
    instrument_spec_id: str

    sequencing_policy: SequencingPolicySpec
    idempotency_policy: IdempotencyPolicySpec
    recovery_policy: RecoveryPolicySpec
    reconciliation_policy: ReconciliationPolicySpec
    health_policy: HealthPolicySpec
    heartbeat_policy: HeartbeatPolicySpec
    kill_switch_policy: KillSwitchPolicySpec
    dispatch_policy: DispatchPolicySpec

    dummy_broker_scenario: DummyBrokerScenarioSpec
    seed: int

    def __post_init__(self) -> None:
        if self.execution_mode not in SUPPORTED_EXECUTION_MODES:
            raise ExecutionGatewaySpecError(f"ExecutionGatewaySpec.execution_mode {self.execution_mode!r} is not supported -- only {sorted(m.value for m in SUPPORTED_EXECUTION_MODES)} exist in this milestone")
        if self.adapter_kind not in SUPPORTED_ADAPTER_KINDS:
            raise ExecutionGatewaySpecError(f"ExecutionGatewaySpec.adapter_kind {self.adapter_kind!r} is not supported -- only {sorted(a.value for a in SUPPORTED_ADAPTER_KINDS)} exist in this milestone")
        for field_name, value in (
            ("paper_session_id", self.paper_session_id), ("paper_trading_spec_id", self.paper_trading_spec_id),
            ("promotion_decision_id", self.promotion_decision_id), ("instrument_spec_id", self.instrument_spec_id),
        ):
            _require_sha256(value, field_name=f"ExecutionGatewaySpec.{field_name}")
        if self.adapter_kind is AdapterKind.DETERMINISTIC_DUMMY and self.sequencing_policy.policy is not _DUMMY_BROKER_REQUIRED_SEQUENCING_POLICY:
            raise ExecutionGatewaySpecError(
                f"ExecutionGatewaySpec.sequencing_policy must be {_DUMMY_BROKER_REQUIRED_SEQUENCING_POLICY.value!r} when adapter_kind=deterministic_dummy, "
                f"got {self.sequencing_policy.policy.value!r}"
            )
        _non_negative_int(self.seed, field_name="ExecutionGatewaySpec.seed")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "execution_mode": self.execution_mode.value, "adapter_kind": self.adapter_kind.value,
            "paper_session_id": self.paper_session_id, "paper_trading_spec_id": self.paper_trading_spec_id,
            "promotion_decision_id": self.promotion_decision_id, "instrument_spec_id": self.instrument_spec_id,
            "sequencing_policy": self.sequencing_policy.to_json_dict(), "idempotency_policy": self.idempotency_policy.to_json_dict(),
            "recovery_policy": self.recovery_policy.to_json_dict(), "reconciliation_policy": self.reconciliation_policy.to_json_dict(),
            "health_policy": self.health_policy.to_json_dict(), "heartbeat_policy": self.heartbeat_policy.to_json_dict(),
            "kill_switch_policy": self.kill_switch_policy.to_json_dict(), "dispatch_policy": self.dispatch_policy.to_json_dict(),
            "dummy_broker_scenario": self.dummy_broker_scenario.to_json_dict(), "seed": self.seed,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = self.to_json_dict()
        del payload["schema_version"]
        payload["dummy_broker_scenario"] = self.dummy_broker_scenario.to_identity_payload()
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionGatewaySpec:
        from quant_platform.ml.persistence import as_json_dict, require_schema_version

        require_schema_version(raw, supported=EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION, context="ExecutionGatewaySpec")
        return cls(
            schema_version=EXECUTION_GATEWAY_SPEC_SCHEMA_VERSION, execution_mode=ExecutionMode(raw["execution_mode"]), adapter_kind=AdapterKind(raw["adapter_kind"]),
            paper_session_id=str(raw["paper_session_id"]), paper_trading_spec_id=str(raw["paper_trading_spec_id"]),
            promotion_decision_id=str(raw["promotion_decision_id"]), instrument_spec_id=str(raw["instrument_spec_id"]),
            sequencing_policy=SequencingPolicySpec.from_json_dict(as_json_dict(raw["sequencing_policy"], field_name="sequencing_policy")),
            idempotency_policy=IdempotencyPolicySpec.from_json_dict(as_json_dict(raw["idempotency_policy"], field_name="idempotency_policy")),
            recovery_policy=RecoveryPolicySpec.from_json_dict(as_json_dict(raw["recovery_policy"], field_name="recovery_policy")),
            reconciliation_policy=ReconciliationPolicySpec.from_json_dict(as_json_dict(raw["reconciliation_policy"], field_name="reconciliation_policy")),
            health_policy=HealthPolicySpec.from_json_dict(as_json_dict(raw["health_policy"], field_name="health_policy")),
            heartbeat_policy=HeartbeatPolicySpec.from_json_dict(as_json_dict(raw["heartbeat_policy"], field_name="heartbeat_policy")),
            kill_switch_policy=KillSwitchPolicySpec.from_json_dict(as_json_dict(raw["kill_switch_policy"], field_name="kill_switch_policy")),
            dispatch_policy=DispatchPolicySpec.from_json_dict(as_json_dict(raw["dispatch_policy"], field_name="dispatch_policy")),
            dummy_broker_scenario=DummyBrokerScenarioSpec.from_json_dict(as_json_dict(raw["dummy_broker_scenario"], field_name="dummy_broker_scenario")),
            seed=int(str(raw["seed"])),
        )


@dataclass(frozen=True, slots=True)
class ExecutionGatewaySpecIdentity:
    schema_version: int
    execution_gateway_spec_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "execution_gateway_spec_id": self.execution_gateway_spec_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionGatewaySpecIdentity:
        from quant_platform.ml.persistence import require_schema_version

        require_schema_version(raw, supported=1, context="ExecutionGatewaySpecIdentity")
        return cls(schema_version=1, execution_gateway_spec_id=str(raw["execution_gateway_spec_id"]))


def compute_execution_gateway_spec_id(spec: ExecutionGatewaySpec) -> ExecutionGatewaySpecIdentity:
    execution_gateway_spec_id = compute_content_id("execution_gateway_spec", spec.to_identity_payload())
    return ExecutionGatewaySpecIdentity(schema_version=1, execution_gateway_spec_id=execution_gateway_spec_id)


def verify_execution_gateway_spec_identity(spec: ExecutionGatewaySpec, expected_id: str) -> bool:
    """Pure recomputation-and-compare -- never trusts a caller-supplied id
    without recomputing it fresh from `spec`."""
    return compute_execution_gateway_spec_id(spec).execution_gateway_spec_id == expected_id


__all__ = [
    "DEFAULT_DUMMY_BROKER_SCENARIO",
    "SUPPORTED_ADAPTER_KINDS",
    "SUPPORTED_EXECUTION_MODES",
    "DispatchPolicySpec",
    "DummyBrokerScenarioSpec",
    "ExecutionGatewaySpec",
    "ExecutionGatewaySpecIdentity",
    "HealthPolicySpec",
    "HeartbeatPolicySpec",
    "IdempotencyPolicySpec",
    "KillSwitchPolicySpec",
    "ReconciliationPolicySpec",
    "RecoveryPolicySpec",
    "RejectionRuleSpec",
    "SequencingPolicySpec",
    "compute_execution_gateway_spec_id",
    "verify_execution_gateway_spec_identity",
]
