"""Point-in-time availability policy for market bars (Milestone 10, Phase
4C) -- the market-bar analogue of Phase 4B's `curated.availability`.

A candle cannot be used by a point-in-time consumer before its CLOSE,
and typically not until some conservative delay after close (a provider
publication/availability lag). `resolve_bar_availability_time` is
STRUCTURALLY fail-closed: no branch returns "available at candle open"
or any other silent default -- every branch either resolves a concrete,
timezone-aware datetime or raises `MarketAvailabilityUnresolvedError`.
For delayed-provider data, the delay is modeled explicitly; this module
never claims real-time availability it cannot back with a declared,
identity-relevant policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from quant_platform.core.exceptions import MarketAvailabilityPolicyError, MarketAvailabilityUnresolvedError
from quant_platform.market_data.identity import compute_content_id, require_non_empty, require_tz_aware

__all__ = [
    "BAR_AVAILABILITY_POLICY_KIND",
    "BarAvailabilityPolicy",
    "BarAvailabilityPolicyKind",
    "create_bar_availability_policy",
    "resolve_bar_availability_time",
]

BAR_AVAILABILITY_POLICY_KIND = "cross_asset_bar_availability_policy"


class BarAvailabilityPolicyKind(Enum):
    CLOSE_PLUS_CONSERVATIVE_DELAY = "close_plus_conservative_delay"
    """Bar close time, in the driver's own session timezone, plus an
    explicit, configured conservative delay (minutes) -- the default,
    honest policy for end-of-day bars this phase's shipped provider
    adapter uses; a delay of `0` still requires close to have actually
    passed, never candle-open."""

    NEXT_SESSION_OPEN_CONSERVATIVE = "next_session_open_conservative"
    """Available only at the NEXT trading session's open -- for
    providers with a materially delayed end-of-day publication (e.g. a
    provider that only finalizes daily data well after the close)."""

    EXPLICIT_PUBLICATION_DELAY_MINUTES = "explicit_publication_delay_minutes"
    """A caller-supplied, provider-documented publication delay (in
    minutes) applied after bar close -- distinct from `CLOSE_PLUS_
    CONSERVATIVE_DELAY` only in that the delay is asserted to be a
    KNOWN provider fact rather than a conservative platform-chosen
    buffer; both resolve identically, the distinction is disclosure-only."""


@dataclass(frozen=True, slots=True)
class BarAvailabilityPolicy:
    availability_policy_id: str
    kind: BarAvailabilityPolicyKind
    policy_version: int
    timezone_key: str
    delay_minutes: int
    notes: str

    def __post_init__(self) -> None:
        if self.policy_version < 1:
            raise MarketAvailabilityPolicyError(f"BarAvailabilityPolicy.policy_version must be >= 1, got {self.policy_version}")
        require_non_empty(self.timezone_key, field_name="BarAvailabilityPolicy.timezone_key")
        try:
            ZoneInfo(self.timezone_key)
        except Exception as exc:
            raise MarketAvailabilityPolicyError(f"BarAvailabilityPolicy.timezone_key {self.timezone_key!r} is not a resolvable IANA timezone: {exc}") from exc
        if self.delay_minutes < 0:
            raise MarketAvailabilityPolicyError(f"BarAvailabilityPolicy.delay_minutes must be >= 0, got {self.delay_minutes}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": BAR_AVAILABILITY_POLICY_KIND, "availability_policy_id": self.availability_policy_id, "policy_kind": self.kind.value,
            "policy_version": self.policy_version, "timezone_key": self.timezone_key, "delay_minutes": self.delay_minutes, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["availability_policy_id"]
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BarAvailabilityPolicy:
        return cls(
            availability_policy_id=str(raw["availability_policy_id"]), kind=BarAvailabilityPolicyKind(raw["policy_kind"]),
            policy_version=int(str(raw["policy_version"])), timezone_key=str(raw["timezone_key"]),
            delay_minutes=int(str(raw["delay_minutes"])), notes=str(raw.get("notes", "")),
        )


def create_bar_availability_policy(
    *, kind: BarAvailabilityPolicyKind, timezone_key: str, delay_minutes: int, policy_version: int = 1, notes: str = "",
) -> BarAvailabilityPolicy:
    provisional = BarAvailabilityPolicy(
        availability_policy_id="0" * 64, kind=kind, policy_version=policy_version, timezone_key=timezone_key, delay_minutes=delay_minutes, notes=notes,
    )
    availability_policy_id = compute_content_id(BAR_AVAILABILITY_POLICY_KIND, provisional.to_identity_payload())
    return BarAvailabilityPolicy(
        availability_policy_id=availability_policy_id, kind=kind, policy_version=policy_version, timezone_key=timezone_key,
        delay_minutes=delay_minutes, notes=notes,
    )


def resolve_bar_availability_time(policy: BarAvailabilityPolicy, *, bar_close_time: datetime, next_session_open_time: datetime | None = None) -> datetime:
    """PURE and structurally fail-closed. `bar_close_time` must already
    be resolved (tz-aware) by the caller from the bar's own
    `TimezoneSessionPolicy` -- this function never derives a close time
    itself, only the AVAILABILITY delay on top of one."""
    require_tz_aware(bar_close_time, field_name="bar_close_time")
    if policy.kind in (BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, BarAvailabilityPolicyKind.EXPLICIT_PUBLICATION_DELAY_MINUTES):
        return bar_close_time + timedelta(minutes=policy.delay_minutes)
    assert policy.kind is BarAvailabilityPolicyKind.NEXT_SESSION_OPEN_CONSERVATIVE
    if next_session_open_time is None:
        raise MarketAvailabilityUnresolvedError("NEXT_SESSION_OPEN_CONSERVATIVE requires next_session_open_time, none was supplied")
    require_tz_aware(next_session_open_time, field_name="next_session_open_time")
    if next_session_open_time <= bar_close_time:
        raise MarketAvailabilityUnresolvedError(
            f"next_session_open_time ({next_session_open_time}) must be after bar_close_time ({bar_close_time})"
        )
    return next_session_open_time + timedelta(minutes=policy.delay_minutes)
