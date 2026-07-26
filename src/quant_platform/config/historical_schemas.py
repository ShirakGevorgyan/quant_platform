"""Pydantic configuration schemas for the historical data pipeline
(Milestone 2). Extends the existing `config` subpackage -- same
conventions as `config.schemas` (frozen, `extra="forbid"`, a `.build()`
factory per schema turning validated config into the runtime object it
describes) -- rather than replacing or duplicating it.

Secrets discipline: `MT5SourceConfig.login`/`password`/`server` are always
optional and default to `None`. A checked-in config file should never set
them; `resolve_mt5_credentials_from_env` reads them from environment
variables at runtime instead, and the CLI (`quant_platform.data_cli`) is
the only place that calls it and merges the result in -- so a config file
committed to version control can be fully populated except for credentials
and still be safe to check in (see `examples/ingestion_config.example.json`).
"""

from __future__ import annotations

import os
from datetime import time as time_of_day
from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.core.types import Timeframe
from quant_platform.historical.calendar import (
    DailyMaintenanceBreak,
    HolidayClosure,
    TradingCalendar,
    WeeklySession,
)
from quant_platform.historical.canonical_store import CompressionCodec
from quant_platform.historical.mt5_adapter import MT5AdapterConfig
from quant_platform.historical.quality import QualityThresholds
from quant_platform.historical.repair import SeverityPolicy
from quant_platform.historical.resampling import DerivedBarPolicy
from quant_platform.historical.timezones import FixedOffsetTimezone, NamedZoneTimezone, SourceTimezone
from quant_platform.historical.update_pipeline import RevisionPolicy

_DERIVABLE_TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "H12", "D1")


class TimezoneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed_offset", "named_zone"]
    offset_minutes: int | None = None
    label: str = "FIXED"
    zone_key: str | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> TimezoneConfig:
        if self.kind == "fixed_offset" and self.offset_minutes is None:
            raise ValueError("offset_minutes is required when kind='fixed_offset'")
        if self.kind == "named_zone" and not self.zone_key:
            raise ValueError("zone_key is required when kind='named_zone'")
        return self

    def build(self) -> SourceTimezone:
        if self.kind == "fixed_offset":
            assert self.offset_minutes is not None
            return FixedOffsetTimezone(timedelta(minutes=self.offset_minutes), name=self.label)
        assert self.zone_key is not None
        return NamedZoneTimezone(self.zone_key)


class MT5SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str = Field(min_length=1)
    source_symbol: str = Field(min_length=1)
    server_timezone: TimezoneConfig
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = Field(default=None, repr=False)
    server: str | None = None

    def build(self) -> MT5AdapterConfig:
        return MT5AdapterConfig(
            broker=self.broker, source_symbol=self.source_symbol,
            server_timezone=self.server_timezone.build(), terminal_path=self.terminal_path,
            login=self.login, password=self.password, server=self.server,
        )

    def with_credentials_from_env(self, *, prefix: str = "MT5_") -> MT5SourceConfig:
        """Return a copy with `login`/`password`/`server` filled in from
        environment variables (`{prefix}LOGIN`/`{prefix}PASSWORD`/
        `{prefix}SERVER`) wherever this config left them unset -- the only
        sanctioned way credentials enter this object. Never call this on a
        config a caller intends to serialize/log afterward."""
        login_env = os.environ.get(f"{prefix}LOGIN")
        return self.model_copy(
            update={
                "login": self.login if self.login is not None else (int(login_env) if login_env else None),
                "password": self.password if self.password is not None else os.environ.get(f"{prefix}PASSWORD"),
                "server": self.server if self.server is not None else os.environ.get(f"{prefix}SERVER"),
            }
        )


class WeeklySessionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    open_weekday: int = Field(ge=0, le=6)
    open_time: time_of_day
    close_weekday: int = Field(ge=0, le=6)
    close_time: time_of_day

    def build(self) -> WeeklySession:
        return WeeklySession(
            open_weekday=self.open_weekday, open_time=self.open_time,
            close_weekday=self.close_weekday, close_time=self.close_time,
        )


class MaintenanceBreakConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: time_of_day
    end: time_of_day
    weekdays: frozenset[int] = Field(default_factory=lambda: frozenset(range(7)))

    def build(self) -> DailyMaintenanceBreak:
        return DailyMaintenanceBreak(start=self.start, end=self.end, weekdays=self.weekdays)


class HolidayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    closure_date: str  # ISO "YYYY-MM-DD", kept as a plain string for simple JSON round-tripping
    start: time_of_day = time_of_day(0, 0)
    end: time_of_day = time_of_day(23, 59, 59, 999999)
    description: str = ""

    def build(self) -> HolidayClosure:
        from datetime import date

        return HolidayClosure(
            closure_date=date.fromisoformat(self.closure_date), start=self.start, end=self.end,
            description=self.description,
        )


class SessionCalendarConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default"
    local_timezone: TimezoneConfig
    weekly_sessions: list[WeeklySessionConfig] = Field(min_length=1)
    maintenance_breaks: list[MaintenanceBreakConfig] = Field(default_factory=list)
    holidays: list[HolidayConfig] = Field(default_factory=list)

    def build(self) -> TradingCalendar:
        return TradingCalendar(
            local_tz=self.local_timezone.build(),
            weekly_sessions=tuple(s.build() for s in self.weekly_sessions),
            maintenance_breaks=tuple(b.build() for b in self.maintenance_breaks),
            holidays=tuple(h.build() for h in self.holidays),
            name=self.name,
        )


class ValidationPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    severity_policy: Literal["STRICT", "WARN_ONLY", "QUARANTINE"] = "STRICT"
    allow_sort: bool = False
    allow_exact_duplicate_removal: bool = True
    max_price_jump_fraction: float = Field(default=0.05, gt=0.0)
    max_spread_points: float = Field(default=500.0, gt=0.0)
    frozen_sequence_min_length: int = Field(default=5, ge=2)
    extreme_range_multiple: float = Field(default=10.0, gt=0.0)
    volume_spike_multiple: float = Field(default=10.0, gt=0.0)

    def build_policy(self) -> SeverityPolicy:
        return SeverityPolicy(self.severity_policy)

    def build_thresholds(self) -> QualityThresholds:
        return QualityThresholds(
            max_price_jump_fraction=self.max_price_jump_fraction, max_spread_points=self.max_spread_points,
            frozen_sequence_min_length=self.frozen_sequence_min_length,
            extreme_range_multiple=self.extreme_range_multiple, volume_spike_multiple=self.volume_spike_multiple,
        )


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_root: Path
    compression: CompressionCodec = "zstd"


class ResamplingOutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_timeframes: list[Literal["M5", "M15", "M30", "H1", "H4", "H12", "D1"]] = Field(min_length=1)
    policy: Literal["REJECT_INCOMPLETE", "RETAIN_INCOMPLETE"] = "REJECT_INCOMPLETE"

    def build_targets(self) -> tuple[Timeframe, ...]:
        return tuple(Timeframe(v) for v in self.target_timeframes)

    def build_policy(self) -> DerivedBarPolicy:
        return DerivedBarPolicy(self.policy)


class IngestionConfig(BaseModel):
    """The top-level config for one symbol's historical ingestion
    pipeline run -- everything `quant_platform.data_cli` needs, all in one
    validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    source_name: Literal["mt5"] = "mt5"
    mt5: MT5SourceConfig | None = None
    requested_timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1"] = "M1"
    extraction_chunk_size_days: int = Field(default=20, gt=0, le=90)
    """Bounds any single source request's date width -- see
    `historical.mt5_adapter`'s module docstring for the empirically
    observed ~25-day MT5 `copy_rates_range` ceiling this guards against.
    """
    update_overlap_bars: int = Field(default=5, ge=0)
    storage: StorageConfig
    session_calendar: SessionCalendarConfig | None = None
    validation: ValidationPolicyConfig = Field(default_factory=ValidationPolicyConfig)
    resampling: ResamplingOutputConfig | None = None
    revision_policy: Literal["REJECT_CONFLICTS", "ACCEPT_NEWER_SOURCE"] = "REJECT_CONFLICTS"

    @model_validator(mode="after")
    def _check_source_specific_config_present(self) -> IngestionConfig:
        if self.source_name == "mt5" and self.mt5 is None:
            raise ValueError("mt5 config section is required when source_name='mt5'")
        return self

    def build_requested_timeframe(self) -> Timeframe:
        return Timeframe(self.requested_timeframe)

    def build_revision_policy(self) -> RevisionPolicy:
        return RevisionPolicy(self.revision_policy)


def resolve_mt5_credentials_from_env(*, prefix: str = "MT5_") -> dict[str, str | int | None]:
    """Read MT5 credentials from environment variables
    (`{prefix}LOGIN`/`{prefix}PASSWORD`/`{prefix}SERVER`). This is the only
    place in the config layer that reads credentials -- never from a
    checked-in file, never hardcoded, never logged (callers must not log
    the returned dict's `password` entry)."""
    login_env = os.environ.get(f"{prefix}LOGIN")
    return {
        "login": int(login_env) if login_env else None,
        "password": os.environ.get(f"{prefix}PASSWORD"),
        "server": os.environ.get(f"{prefix}SERVER"),
    }


__all__ = [
    "HolidayConfig",
    "IngestionConfig",
    "MT5SourceConfig",
    "MaintenanceBreakConfig",
    "ResamplingOutputConfig",
    "SessionCalendarConfig",
    "StorageConfig",
    "TimezoneConfig",
    "ValidationPolicyConfig",
    "WeeklySessionConfig",
    "resolve_mt5_credentials_from_env",
]
