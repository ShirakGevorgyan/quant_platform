"""Timezone and trading-session policy (Milestone 10, Phase 4C).

Not all markets are 24/7, and daily-bar "date" boundaries are a PROVIDER
CONVENTION, never a universal fact -- a US-listed ETF's daily bar closes
at the NYSE close in `America/New_York`; an OTC XAUUSD spot session may
run continuously through a broker-specific "day" cutoff that has no
relationship to any centralized exchange's calendar. `TimezoneSessionPolicy`
exists so every curated driver's daily-bar date carries an EXPLICIT,
disclosed session convention rather than an implicit, silently-assumed
one -- two component datasets built under DIFFERENT session policies
must never be joined as if their daily bars were aligned without
explicit normalization (never performed automatically in this phase --
see `datasets.py`'s own "no automatic alignment" discipline)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum

from quant_platform.core.exceptions import SessionPolicyError
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "SESSION_POLICY_KIND",
    "CandleTimestampConvention",
    "TimezoneSessionPolicy",
    "create_timezone_session_policy",
]

SESSION_POLICY_KIND = "cross_asset_session_policy"

_VALID_TIMEZONE_KEYS = frozenset({
    "UTC", "America/New_York", "America/Chicago", "Europe/London", "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney",
})


class CandleTimestampConvention(Enum):
    OPEN_LABELED = "open_labeled"
    """The provider's own daily-bar timestamp/date labels the SESSION
    OPEN (this phase's own shipped provider adapter's convention -- see
    `providers/alpha_vantage.py`, whose `TIME_SERIES_DAILY` date key
    corresponds to the trading day, and this platform conservatively
    treats it as bounding the OPEN of that day's session)."""

    CLOSE_LABELED = "close_labeled"
    """The provider's own daily-bar timestamp/date labels the SESSION
    CLOSE."""


@dataclass(frozen=True, slots=True)
class TimezoneSessionPolicy:
    session_policy_id: str
    policy_version: int
    timezone_key: str
    session_open_time: time | None
    session_close_time: time | None
    is_24_hour_session: bool
    """True for a genuinely continuous (e.g. OTC FX/spot-metals-style)
    session -- `session_open_time`/`session_close_time` are then both
    `None`, since there is no single daily open/close to name."""
    trading_week_note: str
    holiday_calendar_reference: str | None
    timestamp_convention: CandleTimestampConvention
    provider_session_note: str
    """Free-text disclosure of the PROVIDER's own documented session
    semantics -- never an invented centralized-exchange truth."""

    def __post_init__(self) -> None:
        if self.policy_version < 1:
            raise SessionPolicyError(f"TimezoneSessionPolicy.policy_version must be >= 1, got {self.policy_version}")
        require_non_empty(self.timezone_key, field_name="TimezoneSessionPolicy.timezone_key")
        if self.timezone_key not in _VALID_TIMEZONE_KEYS:
            raise SessionPolicyError(f"TimezoneSessionPolicy.timezone_key {self.timezone_key!r} is not a supported timezone key: {sorted(_VALID_TIMEZONE_KEYS)!r}")
        if self.is_24_hour_session:
            if self.session_open_time is not None or self.session_close_time is not None:
                raise SessionPolicyError("TimezoneSessionPolicy.session_open_time/session_close_time must be None when is_24_hour_session=True")
        else:
            if self.session_open_time is None or self.session_close_time is None:
                raise SessionPolicyError("TimezoneSessionPolicy.session_open_time/session_close_time are required when is_24_hour_session=False")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": SESSION_POLICY_KIND, "session_policy_id": self.session_policy_id, "policy_version": self.policy_version,
            "timezone_key": self.timezone_key, "session_open_time": (None if self.session_open_time is None else self.session_open_time.isoformat()),
            "session_close_time": (None if self.session_close_time is None else self.session_close_time.isoformat()),
            "is_24_hour_session": self.is_24_hour_session, "trading_week_note": self.trading_week_note,
            "holiday_calendar_reference": self.holiday_calendar_reference, "timestamp_convention": self.timestamp_convention.value,
            "provider_session_note": self.provider_session_note,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["session_policy_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TimezoneSessionPolicy:
        raw_open = raw.get("session_open_time")
        raw_close = raw.get("session_close_time")
        return cls(
            session_policy_id=str(raw["session_policy_id"]), policy_version=int(str(raw["policy_version"])),
            timezone_key=str(raw["timezone_key"]), session_open_time=(None if raw_open is None else time.fromisoformat(str(raw_open))),
            session_close_time=(None if raw_close is None else time.fromisoformat(str(raw_close))),
            is_24_hour_session=bool(raw["is_24_hour_session"]), trading_week_note=str(raw["trading_week_note"]),
            holiday_calendar_reference=(None if raw.get("holiday_calendar_reference") is None else str(raw["holiday_calendar_reference"])),
            timestamp_convention=CandleTimestampConvention(raw["timestamp_convention"]), provider_session_note=str(raw["provider_session_note"]),
        )


def create_timezone_session_policy(
    *, timezone_key: str, is_24_hour_session: bool, timestamp_convention: CandleTimestampConvention, provider_session_note: str,
    policy_version: int = 1, session_open_time: time | None = None, session_close_time: time | None = None,
    trading_week_note: str = "Monday-Friday", holiday_calendar_reference: str | None = None,
) -> TimezoneSessionPolicy:
    provisional = TimezoneSessionPolicy(
        session_policy_id="0" * 64, policy_version=policy_version, timezone_key=timezone_key, session_open_time=session_open_time,
        session_close_time=session_close_time, is_24_hour_session=is_24_hour_session, trading_week_note=trading_week_note,
        holiday_calendar_reference=holiday_calendar_reference, timestamp_convention=timestamp_convention, provider_session_note=provider_session_note,
    )
    session_policy_id = compute_content_id(SESSION_POLICY_KIND, provisional.to_identity_payload())
    return TimezoneSessionPolicy(
        session_policy_id=session_policy_id, policy_version=policy_version, timezone_key=timezone_key, session_open_time=session_open_time,
        session_close_time=session_close_time, is_24_hour_session=is_24_hour_session, trading_week_note=trading_week_note,
        holiday_calendar_reference=holiday_calendar_reference, timestamp_convention=timestamp_convention, provider_session_note=provider_session_note,
    )
