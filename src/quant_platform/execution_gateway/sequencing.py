"""Broker-event sequencing (Milestone 8, Section 15). `classify_broker_
event` is the single place every incoming `BrokerEvent` is checked
against what has already been durably recorded for its session, BEFORE
`dispatcher.py` decides whether to append it, absorb it as an idempotent
duplicate, or escalate it. Processing NEVER silently skips a sequence
gap -- a gap is reported (`BROKER_SEQUENCE_GAP`), not swallowed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.execution_gateway.events import BrokerEvent
from quant_platform.execution_gateway.models import SequencingPolicyKind

ISSUE_CODE_BROKER_SEQUENCE_GAP = "BROKER_SEQUENCE_GAP"
ISSUE_CODE_BROKER_SEQUENCE_CONFLICT = "BROKER_SEQUENCE_CONFLICT"
ISSUE_CODE_BROKER_EVENT_DUPLICATE = "BROKER_EVENT_DUPLICATE"
ISSUE_CODE_BROKER_EVENT_PAYLOAD_CONFLICT = "BROKER_EVENT_PAYLOAD_CONFLICT"
ISSUE_CODE_BROKER_EVENT_OUT_OF_ORDER = "BROKER_EVENT_OUT_OF_ORDER"


class SequenceClassification(Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    SEQUENCE_CONFLICT = "sequence_conflict"


@dataclass(frozen=True, slots=True)
class SequencingResult:
    classification: SequenceClassification
    issue_code: str | None
    gap_sequences: tuple[int, ...]

    @property
    def is_critical(self) -> bool:
        """Section 15: a conflicting event reusing an already-used
        sequence is CRITICAL -- everything else here (gap, out-of-order,
        duplicate) is recoverable/expected and handled through ordinary
        reconciliation, never a structural failure on its own."""
        return self.classification is SequenceClassification.SEQUENCE_CONFLICT


def classify_broker_event(
    event: BrokerEvent, *, policy: SequencingPolicyKind, events_by_sequence: dict[int, BrokerEvent], max_seen_sequence: int,
) -> SequencingResult:
    """`events_by_sequence`/`max_seen_sequence` describe what is ALREADY
    durably recorded for `event.execution_session_id` -- never what is
    merely in-flight/unpersisted, so this classification is safe to base
    an idempotent-append decision on."""
    if policy is SequencingPolicyKind.ARRIVAL_ORDER_ONLY:
        # Section 15: lower assurance by design -- duplicate-by-id
        # detection only, no gap/order tracking. Documented, never the
        # dummy-broker default.
        existing = next((e for e in events_by_sequence.values() if e.broker_event_id == event.broker_event_id), None)
        if existing is not None:
            return SequencingResult(SequenceClassification.DUPLICATE, ISSUE_CODE_BROKER_EVENT_DUPLICATE, ())
        return SequencingResult(SequenceClassification.NEW, None, ())

    # STRICT_SEQUENCE and TIMESTAMP_AND_ID share identical duplicate/gap/
    # conflict detection here -- TIMESTAMP_AND_ID additionally imposes a
    # deterministic (broker_timestamp, broker_event_id) sort at the
    # persistence layer (Section 15), which does not change this
    # per-event classification.
    existing = events_by_sequence.get(event.broker_sequence)
    if existing is not None:
        if existing.broker_event_id == event.broker_event_id:
            return SequencingResult(SequenceClassification.DUPLICATE, ISSUE_CODE_BROKER_EVENT_DUPLICATE, ())
        return SequencingResult(SequenceClassification.SEQUENCE_CONFLICT, ISSUE_CODE_BROKER_SEQUENCE_CONFLICT, ())
    if event.broker_sequence <= max_seen_sequence:
        return SequencingResult(SequenceClassification.OUT_OF_ORDER, ISSUE_CODE_BROKER_EVENT_OUT_OF_ORDER, ())
    gap = tuple(range(max_seen_sequence + 1, event.broker_sequence))
    if gap:
        return SequencingResult(SequenceClassification.NEW, ISSUE_CODE_BROKER_SEQUENCE_GAP, gap)
    return SequencingResult(SequenceClassification.NEW, None, ())


__all__ = [
    "ISSUE_CODE_BROKER_EVENT_DUPLICATE",
    "ISSUE_CODE_BROKER_EVENT_OUT_OF_ORDER",
    "ISSUE_CODE_BROKER_EVENT_PAYLOAD_CONFLICT",
    "ISSUE_CODE_BROKER_SEQUENCE_CONFLICT",
    "ISSUE_CODE_BROKER_SEQUENCE_GAP",
    "SequenceClassification",
    "SequencingResult",
    "classify_broker_event",
]
