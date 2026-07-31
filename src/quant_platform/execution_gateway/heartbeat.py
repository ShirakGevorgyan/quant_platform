"""Heartbeat tracking (Milestone 8, Section 21). `HeartbeatOutcome` is the
immutable value `adapter.py`'s `heartbeat()` method returns; `heartbeat_
lag_status` applies `HeartbeatPolicySpec`'s thresholds to a running missed-
count, deterministically, from a caller-supplied count -- never a live
timer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import ExecutionHealthError
from quant_platform.execution_gateway.specs import HeartbeatPolicySpec
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ExecutionHealthError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


@dataclass(frozen=True, slots=True)
class HeartbeatOutcome:
    adapter_id: str
    success: bool
    event_time: datetime

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise ExecutionHealthError("HeartbeatOutcome.adapter_id must not be empty")
        _require_tz_aware(self.event_time, field_name="HeartbeatOutcome.event_time")

    def to_json_dict(self) -> dict[str, object]:
        return {"adapter_id": self.adapter_id, "success": self.success, "event_time": format_utc_timestamp(pd.Timestamp(self.event_time))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HeartbeatOutcome:
        return cls(adapter_id=str(raw["adapter_id"]), success=bool(raw["success"]), event_time=parse_utc_timestamp(str(raw["event_time"])).to_pydatetime())


class HeartbeatLagStatus(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    HALTING = "halting"


def heartbeat_lag_status(*, policy: HeartbeatPolicySpec, consecutive_missed: int) -> HeartbeatLagStatus:
    if consecutive_missed < 0:
        raise ExecutionHealthError(f"consecutive_missed must be >= 0, got {consecutive_missed}")
    if consecutive_missed >= policy.missed_threshold_halting:
        return HeartbeatLagStatus.HALTING
    if consecutive_missed >= policy.missed_threshold_degraded:
        return HeartbeatLagStatus.DEGRADED
    return HeartbeatLagStatus.NORMAL


__all__ = ["HeartbeatLagStatus", "HeartbeatOutcome", "heartbeat_lag_status"]
