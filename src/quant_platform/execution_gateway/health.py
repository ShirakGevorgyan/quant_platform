"""Adapter health (Milestone 8, Section 21). `AdapterHealthSnapshot` is
the immutable value `adapter.py`'s `health()` method returns and
`kill_switch.py`'s dispatch gate consults before permitting a mutating
command. Every threshold in `health_status_for` is evaluated against
SUPPLIED event time / event-count counters (Section 21: "deterministic
thresholds based on supplied event time, not wall clock") -- nothing
here ever calls `datetime.now()`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import ExecutionHealthError
from quant_platform.execution_gateway.models import AdapterHealthStatus
from quant_platform.execution_gateway.specs import HealthPolicySpec
from quant_platform.ml.persistence import as_json_list, format_utc_timestamp, parse_utc_timestamp


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionHealthError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime | None, *, field_name: str) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        raise ExecutionHealthError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")
    return format_utc_timestamp(pd.Timestamp(ts))


def _deserialize_optional_timestamp(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExecutionHealthError(f"{field_name} must be a string or None, got {type(value).__name__}")
    return parse_utc_timestamp(value).to_pydatetime()


@dataclass(frozen=True, slots=True)
class AdapterHealthSnapshot:
    adapter_id: str
    status: AdapterHealthStatus

    last_successful_contact_event_time: datetime | None
    last_event_received_event_time: datetime | None
    last_heartbeat_event_time: datetime | None

    consecutive_failures: int
    event_lag: int
    heartbeat_lag: int

    can_submit: bool
    can_cancel: bool
    can_replace: bool
    can_query: bool

    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise ExecutionHealthError("AdapterHealthSnapshot.adapter_id must not be empty")
        for ts, field_name in (
            (self.last_successful_contact_event_time, "last_successful_contact_event_time"), (self.last_event_received_event_time, "last_event_received_event_time"),
            (self.last_heartbeat_event_time, "last_heartbeat_event_time"),
        ):
            if ts is not None:
                _require_tz_aware(ts, field_name=f"AdapterHealthSnapshot.{field_name}")
        if self.consecutive_failures < 0:
            raise ExecutionHealthError(f"AdapterHealthSnapshot.consecutive_failures must be >= 0, got {self.consecutive_failures}")
        if self.event_lag < 0:
            raise ExecutionHealthError(f"AdapterHealthSnapshot.event_lag must be >= 0, got {self.event_lag}")
        if self.heartbeat_lag < 0:
            raise ExecutionHealthError(f"AdapterHealthSnapshot.heartbeat_lag must be >= 0, got {self.heartbeat_lag}")
        if self.status is AdapterHealthStatus.UNAVAILABLE and self.can_submit:
            raise ExecutionHealthError("AdapterHealthSnapshot: can_submit must be False when status=UNAVAILABLE")
        if self.status is AdapterHealthStatus.STALE and self.can_submit:
            raise ExecutionHealthError("AdapterHealthSnapshot: can_submit must be False when status=STALE")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id, "status": self.status.value,
            "last_successful_contact_event_time": _serialize_timestamp(self.last_successful_contact_event_time, field_name="last_successful_contact_event_time"),
            "last_event_received_event_time": _serialize_timestamp(self.last_event_received_event_time, field_name="last_event_received_event_time"),
            "last_heartbeat_event_time": _serialize_timestamp(self.last_heartbeat_event_time, field_name="last_heartbeat_event_time"),
            "consecutive_failures": self.consecutive_failures, "event_lag": self.event_lag, "heartbeat_lag": self.heartbeat_lag, "can_submit": self.can_submit,
            "can_cancel": self.can_cancel, "can_replace": self.can_replace, "can_query": self.can_query, "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AdapterHealthSnapshot:
        return cls(
            adapter_id=str(raw["adapter_id"]), status=AdapterHealthStatus(raw["status"]),
            last_successful_contact_event_time=_deserialize_optional_timestamp(raw.get("last_successful_contact_event_time"), field_name="last_successful_contact_event_time"),
            last_event_received_event_time=_deserialize_optional_timestamp(raw.get("last_event_received_event_time"), field_name="last_event_received_event_time"),
            last_heartbeat_event_time=_deserialize_optional_timestamp(raw.get("last_heartbeat_event_time"), field_name="last_heartbeat_event_time"),
            consecutive_failures=int(str(raw["consecutive_failures"])), event_lag=int(str(raw["event_lag"])), heartbeat_lag=int(str(raw["heartbeat_lag"])),
            can_submit=bool(raw["can_submit"]), can_cancel=bool(raw["can_cancel"]), can_replace=bool(raw["can_replace"]), can_query=bool(raw["can_query"]),
            reason_codes=tuple(str(c) for c in as_json_list(raw.get("reason_codes") or [], field_name="reason_codes")),
        )


def compute_health_status(
    *, adapter_id: str, policy: HealthPolicySpec, consecutive_failures: int, event_lag: int, heartbeat_lag: int, disconnected: bool,
    last_successful_contact_event_time: datetime | None, last_event_received_event_time: datetime | None, last_heartbeat_event_time: datetime | None,
) -> AdapterHealthSnapshot:
    """Pure, deterministic threshold evaluation (Section 21) -- every
    input is a caller-supplied counter/flag derived from EVENT time, never
    read from a live clock here."""
    reason_codes: list[str] = []
    if disconnected:
        reason_codes.append("adapter_disconnected")
        status = AdapterHealthStatus.UNAVAILABLE
    elif consecutive_failures >= policy.unavailable_after_consecutive_failures:
        reason_codes.append("consecutive_failures_exceeds_unavailable_threshold")
        status = AdapterHealthStatus.UNAVAILABLE
    elif event_lag >= policy.stale_after_events:
        reason_codes.append("event_lag_exceeds_stale_threshold")
        status = AdapterHealthStatus.STALE
    elif consecutive_failures >= policy.degraded_after_consecutive_failures:
        reason_codes.append("consecutive_failures_exceeds_degraded_threshold")
        status = AdapterHealthStatus.DEGRADED
    else:
        status = AdapterHealthStatus.HEALTHY

    can_submit = status is AdapterHealthStatus.HEALTHY
    can_cancel = status in (AdapterHealthStatus.HEALTHY, AdapterHealthStatus.DEGRADED)
    can_replace = status is AdapterHealthStatus.HEALTHY
    can_query = status is not AdapterHealthStatus.UNAVAILABLE

    return AdapterHealthSnapshot(
        adapter_id=adapter_id, status=status, last_successful_contact_event_time=last_successful_contact_event_time,
        last_event_received_event_time=last_event_received_event_time, last_heartbeat_event_time=last_heartbeat_event_time,
        consecutive_failures=consecutive_failures, event_lag=event_lag, heartbeat_lag=heartbeat_lag, can_submit=can_submit, can_cancel=can_cancel,
        can_replace=can_replace, can_query=can_query, reason_codes=tuple(reason_codes),
    )


__all__ = ["AdapterHealthSnapshot", "compute_health_status"]
