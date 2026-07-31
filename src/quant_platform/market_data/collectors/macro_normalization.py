"""Deterministic mapping of raw FRED observation TEXT into
`macro.MacroEvent`'s existing schema (Milestone 10, Phase 4A) --
mirrors `source_normalization.py`'s own "raw text in, strict typed value
out, never through float" discipline exactly, and `mappings.py`'s own
versioned, content-addressed spec pattern for `UnitMappingSpec`.

POINT-IN-TIME MAPPING, THE SINGLE MOST IMPORTANT DESIGN DECISION HERE:
`MacroEvent.event_time` is documented (Phase 1, `macro.py`) as "the
instant the value BECOMES known." FRED's own `date` field is the
OBSERVATION PERIOD a value describes (e.g. `"2024-01-01"` for January's
CPI print) -- NOT when it became known; CPI for January is not published
until mid-February. Using `date` as `event_time` would be exactly the
kind of look-ahead bias `core.exceptions.PointInTimeViolationError`
exists to catch. This module instead maps `event_time` from FRED's own
`realtime_start` field -- the date THIS SPECIFIC VINTAGE of the value
became FRED's officially current one -- the closest honest analogue
FRED's schema actually provides to a release timestamp. Both are always
parsed as UTC MIDNIGHT (FRED dates carry no intraday precision; a daily
series is NEVER falsely given an intraday timestamp here). The actual
economic OBSERVATION PERIOD (`date`) is preserved separately, in
`source_event_id` (`f"fred:{series_id}:date={date}"`), so a monthly
CPI observation's own monthly meaning is never lost even though
`event_time` reflects the (potentially much later) vintage date.

A response format lacking per-row vintage information (this module's own
CSV parsing -- see `fred_schemas.py`) cannot honestly derive `event_time`
this way; `normalize_macro_row` accepts an explicit
`default_realtime_start` for exactly that case, and quarantines
(`EMPTY_TIMESTAMP`) any row with neither a per-row `realtime_start` nor a
caller-supplied default -- it never silently assumes the observation
date IS the release date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.core.exceptions import CollectorError, HistoricalIngestionError
from quant_platform.market_data.collectors.fred_schemas import is_missing_value
from quant_platform.market_data.identity import compute_content_id, require_non_empty
from quant_platform.market_data.quarantine import (
    EMPTY_TIMESTAMP,
    INVALID_DECIMAL,
    MISSING_OBSERVATION_VALUE,
    MISSING_REQUIRED_COLUMN,
    UNKNOWN_SYMBOL,
)
from quant_platform.market_data.source_normalization import normalize_signed_zero, parse_source_decimal

__all__ = [
    "UNIT_MAPPING_KIND",
    "MacroUnit",
    "NormalizedMacroObservation",
    "UnitMappingEntry",
    "UnitMappingSpec",
    "apply_unit_scale",
    "create_unit_mapping_spec",
    "fred_timezone_policy_id",
    "normalize_macro_row",
    "observation_date_to_event_time",
    "resolve_unit",
]

UNIT_MAPPING_KIND = "macro_unit_mapping_spec"


def fred_timezone_policy_id() -> str:
    """The single shared identity for "FRED calendar dates are parsed as
    UTC midnight, never DST-disambiguated" -- both `orchestration.
    run_fred_macro_ingestion_operation` (when building a `SourceManifest`)
    and `verification.verify_fred_macro_operation` (when REDERIVING one)
    call this SAME function, so the two can never silently drift apart by
    each hard-coding an equivalent-looking literal."""
    return compute_content_id("fred_calendar_date_policy", {"note": "FRED dates parsed as UTC midnight, no DST"})

# FRED dates are always `YYYY-MM-DD`, always parsed as UTC midnight. A
# naive `%Y-%m-%d` parse would normally require an explicit
# `source_timezone` (see `source_normalization.parse_source_timestamp`);
# `observation_date_to_event_time` below instead builds a UTC timestamp
# directly for FRED dates specifically, since "midnight UTC" is a fixed,
# unambiguous, DST-free convention for a pure calendar date -- never a
# genuine broker-local wall-clock time requiring DST disambiguation.


class MacroUnit(Enum):
    PERCENT = "percent"
    INDEX = "index"
    RATE = "rate"
    BASIS_POINTS = "basis_points"


@dataclass(frozen=True, slots=True)
class UnitMappingEntry:
    series_id: str
    unit: MacroUnit
    scale_factor: Decimal = Decimal(1)
    """Applied to the raw FRED value (via Decimal multiplication, never
    float) to reach the CANONICAL stored value -- `1` (the default)
    means "store exactly as FRED reports it, only labeled with the
    correct `MacroUnit`." A future series needing an actual unit
    conversion (e.g. percent -> basis points, `* 100`) sets this
    explicitly."""

    def __post_init__(self) -> None:
        require_non_empty(self.series_id, field_name="UnitMappingEntry.series_id")

    def to_json_dict(self) -> dict[str, object]:
        return {"series_id": self.series_id, "unit": self.unit.value, "scale_factor": str(self.scale_factor)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> UnitMappingEntry:
        return cls(series_id=str(raw["series_id"]), unit=MacroUnit(raw["unit"]), scale_factor=Decimal(str(raw["scale_factor"])))


@dataclass(frozen=True, slots=True)
class UnitMappingSpec:
    unit_mapping_id: str
    unit_mapping_version: int
    entries: tuple[UnitMappingEntry, ...]

    def __post_init__(self) -> None:
        if self.unit_mapping_version < 1:
            raise CollectorError(f"UnitMappingSpec.unit_mapping_version must be >= 1, got {self.unit_mapping_version}")
        seen: set[str] = set()
        for entry in self.entries:
            if entry.series_id in seen:
                raise CollectorError(f"UnitMappingSpec has a duplicate entry for series_id={entry.series_id!r}")
            seen.add(entry.series_id)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": UNIT_MAPPING_KIND, "unit_mapping_id": self.unit_mapping_id, "unit_mapping_version": self.unit_mapping_version,
            "entries": [e.to_json_dict() for e in self.entries],
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["unit_mapping_id"]
        entries = [e.to_json_dict() for e in self.entries]
        payload["entries"] = sorted(entries, key=lambda e: str(e["series_id"]))
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> UnitMappingSpec:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        entries = tuple(UnitMappingEntry.from_json_dict(as_json_dict(e, field_name="entry")) for e in as_json_list(raw["entries"], field_name="entries"))
        return cls(unit_mapping_id=str(raw["unit_mapping_id"]), unit_mapping_version=int(str(raw["unit_mapping_version"])), entries=entries)


def create_unit_mapping_spec(*, unit_mapping_version: int, entries: tuple[UnitMappingEntry, ...]) -> UnitMappingSpec:
    provisional = UnitMappingSpec(unit_mapping_id="0" * 64, unit_mapping_version=unit_mapping_version, entries=entries)
    unit_mapping_id = compute_content_id(UNIT_MAPPING_KIND, provisional.to_identity_payload())
    return UnitMappingSpec(unit_mapping_id=unit_mapping_id, unit_mapping_version=unit_mapping_version, entries=entries)


def resolve_unit(spec: UnitMappingSpec, *, series_id: str) -> UnitMappingEntry:
    for entry in spec.entries:
        if entry.series_id == series_id:
            return entry
    raise CollectorError(f"No unit mapping for series_id={series_id!r} in mapping {spec.unit_mapping_id!r}")


def apply_unit_scale(value: Decimal, scale_factor: Decimal) -> Decimal:
    return normalize_signed_zero(value * scale_factor)


def observation_date_to_event_time(date_text: str, *, field_name: str = "date") -> datetime:
    """FRED calendar dates only -- always UTC midnight, never a genuine
    broker-local wall-clock time, so this bypasses
    `parse_source_timestamp`'s own DST-disambiguation machinery (not
    applicable to a bare calendar date) and localizes directly."""
    import pandas as pd

    require_non_empty(date_text, field_name=field_name)
    try:
        ts = pd.Timestamp(date_text, tz="UTC")
    except (ValueError, TypeError) as exc:
        raise CollectorError(f"{field_name}={date_text!r} is not a valid FRED calendar date: {exc}") from exc
    result: datetime = ts.to_pydatetime()
    return result


@dataclass(frozen=True, slots=True)
class NormalizedMacroObservation:
    event_time: datetime
    value: Decimal
    unit: MacroUnit
    source_event_id: str


def normalize_macro_row(
    raw_fields: dict[str, str], *, series_id: str, unit_mapping: UnitMappingSpec, default_realtime_start: datetime | None = None,
) -> tuple[NormalizedMacroObservation | None, tuple[str, ...]]:
    """Pure: given one row's `raw_fields` (`"date"`, `"value"`, optional
    `"realtime_start"`) plus the target series/unit mapping, returns
    either `(observation, ())` on success or `(None, issue_codes)` on
    failure -- the caller (the collector-side orchestration row
    processor) decides whether that means quarantine or fail-fast,
    mirroring `market_data.orchestration._process_row`'s own contract
    exactly."""
    issue_codes: list[str] = []
    missing_keys = [k for k in ("date", "value") if k not in raw_fields]
    if missing_keys:
        return None, (MISSING_REQUIRED_COLUMN,)

    value_text = raw_fields["value"]
    if is_missing_value(value_text):
        issue_codes.append(MISSING_OBSERVATION_VALUE)

    value: Decimal | None = None
    if not issue_codes:
        try:
            value = parse_source_decimal(value_text, field_name="value")
        except HistoricalIngestionError:
            issue_codes.append(INVALID_DECIMAL)

    realtime_start_text = raw_fields.get("realtime_start")
    event_time: datetime | None = None
    if realtime_start_text:
        try:
            event_time = observation_date_to_event_time(realtime_start_text, field_name="realtime_start")
        except CollectorError:
            issue_codes.append(EMPTY_TIMESTAMP)
    elif default_realtime_start is not None:
        event_time = default_realtime_start
    else:
        issue_codes.append(EMPTY_TIMESTAMP)

    try:
        entry = resolve_unit(unit_mapping, series_id=series_id)
    except CollectorError:
        issue_codes.append(UNKNOWN_SYMBOL)
        entry = None

    if issue_codes:
        return None, tuple(issue_codes)

    assert value is not None and event_time is not None and entry is not None
    scaled_value = apply_unit_scale(value, entry.scale_factor)
    source_event_id = f"fred:{series_id}:date={raw_fields['date']}"
    return NormalizedMacroObservation(event_time=event_time, value=scaled_value, unit=entry.unit, source_event_id=source_event_id), ()
