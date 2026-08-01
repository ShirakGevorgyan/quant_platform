"""Release availability policy (Milestone 10, Phase 4B) -- THE critical
anti-look-ahead-leakage boundary for curated macro data.

Four distinct times are in play, and this module exists specifically
because they must never be confused:
1. `observation_date` -- the economic PERIOD a value describes (e.g.
   `"2024-01-01"` for January CPI).
2. `realtime_start`/`realtime_end` -- FRED/ALFRED's own real-time
   validity metadata for a specific vintage of a value.
3. `availability_time` -- THIS module's output: the earliest instant
   the platform permits a point-in-time feature join to use the value.
4. `ingestion_time` -- caller-supplied, purely operational (when the
   platform happened to collect it).

`observation_date` ALONE is never proof a value was available then --
using it that way is exactly the look-ahead bias this module exists to
prevent (see `test_collectors_curated_point_in_time.py`'s own explicit
proof that a value is invisible before its resolved `availability_time`).

FAIL-CLOSED IS STRUCTURAL, NOT CONFIGURABLE: `resolve_availability_time`
has no "silently treat as immediately available" fallback path at all
-- every policy kind either resolves a concrete `availability_time` from
its declared, REQUIRED inputs, or raises `AvailabilityUnresolvedError`.
There is no field anywhere in `AvailabilityPolicy` that can select a
different, less-safe behavior.

HONEST APPROXIMATION, DISCLOSED: FRED provides only DATES (never
intraday release timestamps) for `realtime_start`. Every policy below
that anchors on a date therefore combines it with a caller-declared,
IDENTITY-RELEVANT `(timezone_key, availability_hour, availability_minute)`
-- an explicit, versioned, conservative approximation, never a
fabricated precise timestamp FRED never actually provided."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import AvailabilityPolicyError, AvailabilityUnresolvedError
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "AVAILABILITY_POLICY_KIND",
    "AvailabilityPolicy",
    "AvailabilityPolicyKind",
    "create_availability_policy",
    "resolve_availability_time",
]

AVAILABILITY_POLICY_KIND = "curated_availability_policy"

_WEEKEND_WEEKDAYS = frozenset({5, 6})


class AvailabilityPolicyKind(Enum):
    OBSERVATION_DATE_END_OF_DAY = "observation_date_end_of_day"
    """Anchors on `observation_date` itself -- appropriate ONLY for a
    series genuinely published same-day (e.g. a daily market rate)."""

    NEXT_BUSINESS_DAY_CONSERVATIVE = "next_business_day_conservative"
    """Anchors on `realtime_start`'s OWN next business day (a fixed,
    non-configurable +1 business-day push) -- extra conservatism beyond
    `REALTIME_START_DATE_CONSERVATIVE` for series where same-day receipt
    is not a safe assumption."""

    EXPLICIT_RELEASE_TIMESTAMP = "explicit_release_timestamp"
    """A single, fully-specified, caller-declared tz-aware timestamp --
    used when a curator has independently confirmed the exact release
    time for a specific case."""

    RELEASE_CALENDAR_REFERENCE = "release_calendar_reference"
    """RESERVED for a future phase: resolving availability against an
    external, named release CALENDAR (e.g. a BLS/Fed publication
    schedule). `resolve_availability_time` raises `AvailabilityPolicyError`
    for this kind in Phase 4B -- see Known Non-Blocking Limitations in
    the delivery report."""

    REALTIME_START_DATE_CONSERVATIVE = "realtime_start_date_conservative"
    """Anchors on FRED's own `realtime_start` date -- the general-purpose
    default: uses FRED's own point-in-time metadata directly, at a
    declared conservative time of day."""

    MANUAL_CURATED_RELEASE_RULE = "manual_curated_release_rule"
    """Anchors on `observation_date` plus an explicit, curator-declared
    `delay_days` (calendar days) -- for a series where neither
    `realtime_start` nor same-day receipt is trusted, and a curator has
    hand-verified a fixed publication lag instead."""


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    availability_policy_id: str
    kind: AvailabilityPolicyKind
    policy_version: int
    timezone_key: str | None
    availability_hour: int | None
    availability_minute: int | None
    delay_days: int
    business_day_calendar: str | None
    explicit_release_timestamp: datetime | None

    def __post_init__(self) -> None:
        if self.policy_version < 1:
            raise AvailabilityPolicyError(f"AvailabilityPolicy.policy_version must be >= 1, got {self.policy_version}")
        if self.delay_days < 0:
            raise AvailabilityPolicyError(f"AvailabilityPolicy.delay_days must be >= 0, got {self.delay_days}")

        needs_time_of_day = self.kind in (
            AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, AvailabilityPolicyKind.NEXT_BUSINESS_DAY_CONSERVATIVE,
            AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, AvailabilityPolicyKind.MANUAL_CURATED_RELEASE_RULE,
        )
        if needs_time_of_day:
            require_non_empty(self.timezone_key or "", field_name="AvailabilityPolicy.timezone_key")
            if self.availability_hour is None or not (0 <= self.availability_hour <= 23):
                raise AvailabilityPolicyError(f"AvailabilityPolicy.availability_hour must be in [0, 23] for kind {self.kind.value!r}, got {self.availability_hour}")
            if self.availability_minute is None or not (0 <= self.availability_minute <= 59):
                raise AvailabilityPolicyError(f"AvailabilityPolicy.availability_minute must be in [0, 59] for kind {self.kind.value!r}, got {self.availability_minute}")
        else:
            if self.timezone_key is not None or self.availability_hour is not None or self.availability_minute is not None:
                raise AvailabilityPolicyError(f"AvailabilityPolicy.timezone_key/availability_hour/availability_minute must be None for kind {self.kind.value!r}")

        if self.kind is AvailabilityPolicyKind.MANUAL_CURATED_RELEASE_RULE:
            pass  # delay_days may legitimately be > 0; already validated >= 0 above.
        elif self.delay_days != 0:
            raise AvailabilityPolicyError(f"AvailabilityPolicy.delay_days must be 0 for kind {self.kind.value!r} (only MANUAL_CURATED_RELEASE_RULE uses it)")

        if self.kind is AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE:
            require_non_empty(self.business_day_calendar or "", field_name="AvailabilityPolicy.business_day_calendar")
        elif self.business_day_calendar is not None:
            raise AvailabilityPolicyError(f"AvailabilityPolicy.business_day_calendar must be None for kind {self.kind.value!r}")

        if self.kind is AvailabilityPolicyKind.EXPLICIT_RELEASE_TIMESTAMP:
            if self.explicit_release_timestamp is None:
                raise AvailabilityPolicyError("AvailabilityPolicy.explicit_release_timestamp is required for kind EXPLICIT_RELEASE_TIMESTAMP")
            require_tz_aware(self.explicit_release_timestamp, field_name="AvailabilityPolicy.explicit_release_timestamp")
        elif self.explicit_release_timestamp is not None:
            raise AvailabilityPolicyError(f"AvailabilityPolicy.explicit_release_timestamp must be None for kind {self.kind.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": AVAILABILITY_POLICY_KIND, "availability_policy_id": self.availability_policy_id, "policy_kind": self.kind.value,
            "policy_version": self.policy_version, "timezone_key": self.timezone_key, "availability_hour": self.availability_hour,
            "availability_minute": self.availability_minute, "delay_days": self.delay_days, "business_day_calendar": self.business_day_calendar,
            "explicit_release_timestamp": (None if self.explicit_release_timestamp is None else serialize_timestamp(self.explicit_release_timestamp, field_name="explicit_release_timestamp")),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["availability_policy_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AvailabilityPolicy:
        raw_ts = raw.get("explicit_release_timestamp")
        raw_hour = raw.get("availability_hour")
        raw_minute = raw.get("availability_minute")
        return cls(
            availability_policy_id=str(raw["availability_policy_id"]), kind=AvailabilityPolicyKind(raw["policy_kind"]),
            policy_version=int(str(raw["policy_version"])), timezone_key=(None if raw.get("timezone_key") is None else str(raw["timezone_key"])),
            availability_hour=(None if raw_hour is None else int(str(raw_hour))), availability_minute=(None if raw_minute is None else int(str(raw_minute))),
            delay_days=int(str(raw["delay_days"])), business_day_calendar=(None if raw.get("business_day_calendar") is None else str(raw["business_day_calendar"])),
            explicit_release_timestamp=(None if raw_ts is None else deserialize_timestamp(raw_ts, field_name="explicit_release_timestamp")),
        )


def create_availability_policy(
    *, kind: AvailabilityPolicyKind, policy_version: int = 1, timezone_key: str | None = None, availability_hour: int | None = None,
    availability_minute: int | None = None, delay_days: int = 0, business_day_calendar: str | None = None,
    explicit_release_timestamp: datetime | None = None,
) -> AvailabilityPolicy:
    provisional = AvailabilityPolicy(
        availability_policy_id="0" * 64, kind=kind, policy_version=policy_version, timezone_key=timezone_key, availability_hour=availability_hour,
        availability_minute=availability_minute, delay_days=delay_days, business_day_calendar=business_day_calendar,
        explicit_release_timestamp=explicit_release_timestamp,
    )
    availability_policy_id = compute_content_id(AVAILABILITY_POLICY_KIND, provisional.to_identity_payload())
    return AvailabilityPolicy(
        availability_policy_id=availability_policy_id, kind=kind, policy_version=policy_version, timezone_key=timezone_key,
        availability_hour=availability_hour, availability_minute=availability_minute, delay_days=delay_days,
        business_day_calendar=business_day_calendar, explicit_release_timestamp=explicit_release_timestamp,
    )


def _localize_date_at_time(date_text: str, *, hour: int, minute: int, timezone_key: str, field_name: str) -> datetime:
    import pandas as pd

    try:
        naive = pd.Timestamp(date_text) + pd.Timedelta(hours=hour, minutes=minute)
        localized = naive.tz_localize(timezone_key)
    except (ValueError, TypeError) as exc:
        raise AvailabilityUnresolvedError(f"could not localize {field_name}={date_text!r} at {hour:02d}:{minute:02d} in {timezone_key!r}: {exc}") from exc
    result: datetime = localized.tz_convert("UTC").to_pydatetime()
    return result


def _next_business_day_text(date_text: str) -> str:
    import pandas as pd

    ts = pd.Timestamp(date_text)
    next_day = ts + pd.Timedelta(days=1)
    while next_day.weekday() in _WEEKEND_WEEKDAYS:
        next_day += pd.Timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")


def _add_calendar_days_text(date_text: str, days: int) -> str:
    import pandas as pd

    ts = pd.Timestamp(date_text) + pd.Timedelta(days=days)
    return ts.strftime("%Y-%m-%d")


def resolve_availability_time(policy: AvailabilityPolicy, *, observation_date_text: str, realtime_start_text: str | None) -> datetime:
    """Pure -- never reads the wall clock. Raises `AvailabilityUnresolvedError`
    whenever the declared policy's REQUIRED input is missing (e.g.
    `REALTIME_START_DATE_CONSERVATIVE` with no `realtime_start_text`
    supplied) -- there is no other fallback."""
    require_non_empty(observation_date_text, field_name="observation_date_text")

    if policy.kind is AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY:
        assert policy.timezone_key is not None and policy.availability_hour is not None and policy.availability_minute is not None
        return _localize_date_at_time(observation_date_text, hour=policy.availability_hour, minute=policy.availability_minute, timezone_key=policy.timezone_key, field_name="observation_date")

    if policy.kind is AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE:
        if not realtime_start_text:
            raise AvailabilityUnresolvedError(f"AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE requires realtime_start, none was supplied for observation_date={observation_date_text!r}")
        assert policy.timezone_key is not None and policy.availability_hour is not None and policy.availability_minute is not None
        return _localize_date_at_time(realtime_start_text, hour=policy.availability_hour, minute=policy.availability_minute, timezone_key=policy.timezone_key, field_name="realtime_start")

    if policy.kind is AvailabilityPolicyKind.NEXT_BUSINESS_DAY_CONSERVATIVE:
        if not realtime_start_text:
            raise AvailabilityUnresolvedError(f"AvailabilityPolicyKind.NEXT_BUSINESS_DAY_CONSERVATIVE requires realtime_start, none was supplied for observation_date={observation_date_text!r}")
        assert policy.timezone_key is not None and policy.availability_hour is not None and policy.availability_minute is not None
        anchor = _next_business_day_text(realtime_start_text)
        return _localize_date_at_time(anchor, hour=policy.availability_hour, minute=policy.availability_minute, timezone_key=policy.timezone_key, field_name="next_business_day(realtime_start)")

    if policy.kind is AvailabilityPolicyKind.MANUAL_CURATED_RELEASE_RULE:
        assert policy.timezone_key is not None and policy.availability_hour is not None and policy.availability_minute is not None
        anchor = _add_calendar_days_text(observation_date_text, policy.delay_days)
        return _localize_date_at_time(anchor, hour=policy.availability_hour, minute=policy.availability_minute, timezone_key=policy.timezone_key, field_name="observation_date+delay_days")

    if policy.kind is AvailabilityPolicyKind.EXPLICIT_RELEASE_TIMESTAMP:
        assert policy.explicit_release_timestamp is not None
        return policy.explicit_release_timestamp

    raise AvailabilityPolicyError(f"AvailabilityPolicyKind.{policy.kind.name} is not implemented in Phase 4B -- see delivery report's Known Non-Blocking Limitations")
