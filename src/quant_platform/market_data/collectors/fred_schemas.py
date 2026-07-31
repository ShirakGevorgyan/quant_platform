"""Raw FRED observation model and strict JSON/CSV parsing (Milestone 10,
Phase 4A) -- the boundary between raw FRED response bytes and this
package's own text-only `FredObservation`. Mirrors `market_data.
csv_adapter`/`jsonl_adapter`'s own discipline exactly: a value is read
as TEXT, never coerced to a typed value here (never through `float`,
never a date parsed to a `datetime`) -- that is `macro_normalization.py`'s
job. `FredObservation.value_text` may legitimately be `"."`, FRED's own
documented convention for "no value published for this date"
(`is_missing_value` checks for exactly this, never silently treated as
`0`).

Only the OFFICIAL FRED `/fred/series/observations` JSON schema shape is
supported for JSON (`{"observations": [{"date": ..., "value": ...,
"realtime_start": ..., "realtime_end": ...}, ...]}`); CSV support
targets FRED's documented two-column `DATE,<SERIES_ID>` downloadable-
series export shape -- an explicit, defensible, DOCUMENTED assumption
about that export's exact column layout (this phase performs zero live
FRED requests, so it is never verified against a live response; a
caller integrating a real network collector should confirm this against
FRED's own current documentation before trusting CSV parsing in
production)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from quant_platform.core.exceptions import MalformedFredResponseError, UnsupportedFredSchemaError
from quant_platform.core.json import parse_json_strict

__all__ = ["FRED_MISSING_VALUE_TEXT", "FredObservation", "is_missing_value", "parse_fred_csv_response", "parse_fred_json_response"]

FRED_MISSING_VALUE_TEXT = "."

_REQUIRED_OBSERVATION_KEYS = ("date", "value")
_ALLOWED_OBSERVATION_KEYS = ("date", "value", "realtime_start", "realtime_end")


@dataclass(frozen=True, slots=True)
class FredObservation:
    series_id: str
    row_index: int
    observation_date: str
    """Raw `"YYYY-MM-DD"` text exactly as FRED sent it -- NOT parsed to a
    `datetime` here (see module docstring)."""
    value_text: str
    """Raw text exactly as FRED sent it -- may be `"."` (see
    `is_missing_value`), never parsed to `Decimal`/`float` here."""
    realtime_start: str | None
    realtime_end: str | None


def is_missing_value(value_text: str) -> bool:
    return value_text.strip() == FRED_MISSING_VALUE_TEXT


def parse_fred_json_response(raw_bytes: bytes, *, series_id: str, encoding: str = "utf-8") -> list[FredObservation]:
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise MalformedFredResponseError(f"FRED JSON response is not valid {encoding}: {exc}") from exc
    try:
        parsed = parse_json_strict(text)
    except ValueError as exc:
        raise MalformedFredResponseError(f"FRED JSON response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MalformedFredResponseError(f"FRED JSON response must decode to a JSON object, got {type(parsed).__name__}")
    if "observations" not in parsed:
        raise UnsupportedFredSchemaError("FRED JSON response has no 'observations' key -- not a recognized series/observations schema")
    raw_observations = parsed["observations"]
    if not isinstance(raw_observations, list):
        raise MalformedFredResponseError(f"FRED JSON response 'observations' must be a list, got {type(raw_observations).__name__}")

    observations: list[FredObservation] = []
    for row_index, raw_obs in enumerate(raw_observations):
        if not isinstance(raw_obs, dict):
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index} must be a JSON object, got {type(raw_obs).__name__}")
        missing = [k for k in _REQUIRED_OBSERVATION_KEYS if k not in raw_obs]
        if missing:
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index} is missing required key(s): {missing}")
        extra = [k for k in raw_obs if k not in _ALLOWED_OBSERVATION_KEYS]
        if extra:
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index} has undeclared key(s): {extra}")
        date_value = raw_obs["date"]
        value_value = raw_obs["value"]
        if not isinstance(date_value, str) or not isinstance(value_value, str):
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index}: 'date'/'value' must be JSON strings, never numbers")
        realtime_start = raw_obs.get("realtime_start")
        realtime_end = raw_obs.get("realtime_end")
        if realtime_start is not None and not isinstance(realtime_start, str):
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index}: 'realtime_start' must be a string")
        if realtime_end is not None and not isinstance(realtime_end, str):
            raise MalformedFredResponseError(f"FRED JSON observation at index {row_index}: 'realtime_end' must be a string")
        observations.append(FredObservation(
            series_id=series_id, row_index=row_index, observation_date=date_value, value_text=value_value,
            realtime_start=realtime_start, realtime_end=realtime_end,
        ))
    return observations


def parse_fred_csv_response(raw_bytes: bytes, *, series_id: str, encoding: str = "utf-8") -> list[FredObservation]:
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise MalformedFredResponseError(f"FRED CSV response is not valid {encoding}: {exc}") from exc
    if text.startswith("﻿"):
        text = text[1:]
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise MalformedFredResponseError("FRED CSV response has no header row") from None
    normalized_header = [h.strip().upper() for h in header]
    if len(normalized_header) != 2 or normalized_header[0] != "DATE":
        raise UnsupportedFredSchemaError(f"FRED CSV response header must be exactly ['DATE', <SERIES_COLUMN>], got {header!r}")

    observations: list[FredObservation] = []
    for row_index, row in enumerate(reader):
        if len(row) != 2:
            raise MalformedFredResponseError(f"FRED CSV response row {row_index} has {len(row)} field(s), expected 2")
        observations.append(FredObservation(
            series_id=series_id, row_index=row_index, observation_date=row[0], value_text=row[1], realtime_start=None, realtime_end=None,
        ))
    return observations
