"""FRED series-METADATA collector (Milestone 10, Phase 4B) -- the
official `/fred/series` endpoint (distinct from `/fred/series/
observations`, which `fred.py` already covers). Mirrors `fred.py`'s own
discipline exactly: strict, text-only parsing (never through float),
zero duplication of the transport/retry/rate-limit attempt loop.

REUSE, NOT DUPLICATION: `execute_request.build_transport_request`
(Milestone 10, Phase 4C: extracted from `fred.py`, where it originally
lived as a private helper) already builds its URL from `manifest.
endpoint_host`/`manifest.endpoint_path`/`manifest.canonical_query_params`
-- nothing about it is observations-specific -- and `fred.
execute_fred_request`'s attempt loop only ever touches `request_manifest`/
`response_manifest`, never an observations-specific field. Both are
reused UNCHANGED here; this module supplies only what is genuinely
different: the metadata endpoint path, its own query-parameter shape,
and its own strict response schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.exceptions import MalformedFredResponseError, UnsupportedFredSchemaError
from quant_platform.core.json import parse_json_strict
from quant_platform.market_data.collectors.fred import (
    FRED_COLLECTOR_VERSION,
    FRED_ENDPOINT_HOST,
    execute_fred_request,
)
from quant_platform.market_data.collectors.request_manifest import (
    CollectorRequestManifest,
    CredentialMode,
    create_request_manifest,
)

__all__ = [
    "FRED_SERIES_ENDPOINT_PATH",
    "FredSeriesMetadata",
    "build_fred_series_metadata_request_manifest",
    "execute_fred_series_metadata_request",
    "parse_fred_series_metadata_response",
]

FRED_SERIES_ENDPOINT_PATH = "/fred/series"

_REQUIRED_SERIES_KEYS = (
    "id", "realtime_start", "realtime_end", "title", "observation_start", "observation_end",
    "frequency", "frequency_short", "units", "units_short", "seasonal_adjustment", "seasonal_adjustment_short",
    "last_updated",
)
_OPTIONAL_SERIES_KEYS = ("notes", "popularity")
_ALLOWED_SERIES_KEYS = frozenset(_REQUIRED_SERIES_KEYS) | frozenset(_OPTIONAL_SERIES_KEYS)


def build_fred_series_metadata_request_manifest(
    *, series_id: str, timeout_policy_id: str, retry_policy_id: str, rate_limit_policy_id: str, credential_mode: CredentialMode,
    request_time: datetime, response_format: str = "json", realtime_start: datetime | None = None, realtime_end: datetime | None = None,
    collector_version: str = FRED_COLLECTOR_VERSION,
) -> CollectorRequestManifest:
    query_params: dict[str, str] = {"series_id": series_id, "file_type": response_format}
    if realtime_start is not None:
        query_params["realtime_start"] = realtime_start.strftime("%Y-%m-%d")
    if realtime_end is not None:
        query_params["realtime_end"] = realtime_end.strftime("%Y-%m-%d")
    return create_request_manifest(
        collector_name="fred_series_metadata", collector_version=collector_version, endpoint_host=FRED_ENDPOINT_HOST,
        endpoint_path=FRED_SERIES_ENDPOINT_PATH, canonical_query_params=query_params, canonical_headers={},
        requested_series_or_dataset=series_id, response_format=response_format, timeout_policy_id=timeout_policy_id,
        retry_policy_id=retry_policy_id, rate_limit_policy_id=rate_limit_policy_id, credential_mode=credential_mode,
        request_time=request_time,
    )


# `execute_fred_series_metadata_request` is a thin, explicitly-named alias:
# `execute_fred_request` already does everything needed (transport + retry +
# rate-limit coordination over any FRED request_manifest); this alias exists
# purely so a caller reading `curated/metadata.py` sees a name that matches
# what it is fetching, not because the underlying logic differs at all.
execute_fred_series_metadata_request = execute_fred_request


@dataclass(frozen=True, slots=True)
class FredSeriesMetadata:
    """Raw FRED series metadata -- TEXT fields exactly as FRED sent
    them, never coerced to a typed value here (mirrors `fred_schemas.
    FredObservation`'s own discipline). `curated/metadata.py` is the
    layer that compares these against a curated spec's own
    expectations and decides fail-closed vs. warning."""

    series_id: str
    response_realtime_start: str
    response_realtime_end: str
    title: str
    observation_start: str
    observation_end: str
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment: str
    seasonal_adjustment_short: str
    last_updated: str
    notes: str | None
    popularity: str | None


def parse_fred_series_metadata_response(raw_bytes: bytes, *, encoding: str = "utf-8") -> FredSeriesMetadata:
    """Reports exactly what FRED returned, structurally -- deciding
    whether the returned `series_id` matches what was REQUESTED (the
    "wrong series id: fail closed" drift rule) is `curated/metadata.
    verify_series_metadata`'s job, not this parser's, mirroring
    `fred_schemas.py`'s own "parse first, compare later" layering."""
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise MalformedFredResponseError(f"FRED series metadata response is not valid {encoding}: {exc}") from exc
    try:
        parsed = parse_json_strict(text)
    except ValueError as exc:
        raise MalformedFredResponseError(f"FRED series metadata response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MalformedFredResponseError(f"FRED series metadata response must decode to a JSON object, got {type(parsed).__name__}")
    if "seriess" not in parsed:
        raise UnsupportedFredSchemaError("FRED series metadata response has no 'seriess' key -- not a recognized /fred/series schema")
    if "realtime_start" not in parsed or "realtime_end" not in parsed:
        raise MalformedFredResponseError("FRED series metadata response is missing top-level 'realtime_start'/'realtime_end'")
    response_realtime_start = parsed["realtime_start"]
    response_realtime_end = parsed["realtime_end"]
    if not isinstance(response_realtime_start, str) or not isinstance(response_realtime_end, str):
        raise MalformedFredResponseError("FRED series metadata response top-level 'realtime_start'/'realtime_end' must be strings")

    seriess = parsed["seriess"]
    if not isinstance(seriess, list):
        raise MalformedFredResponseError(f"FRED series metadata response 'seriess' must be a list, got {type(seriess).__name__}")
    if len(seriess) != 1:
        raise MalformedFredResponseError(f"FRED series metadata response 'seriess' must contain exactly one entry for a single series_id request, got {len(seriess)}")
    entry = seriess[0]
    if not isinstance(entry, dict):
        raise MalformedFredResponseError(f"FRED series metadata 'seriess[0]' must be a JSON object, got {type(entry).__name__}")

    missing = [k for k in _REQUIRED_SERIES_KEYS if k not in entry]
    if missing:
        raise MalformedFredResponseError(f"FRED series metadata entry is missing required key(s): {missing}")
    extra = [k for k in entry if k not in _ALLOWED_SERIES_KEYS]
    if extra:
        raise MalformedFredResponseError(f"FRED series metadata entry has undeclared key(s): {extra}")

    for key in _REQUIRED_SERIES_KEYS:
        if not isinstance(entry[key], str):
            raise MalformedFredResponseError(f"FRED series metadata field {key!r} must be a JSON string, got {type(entry[key]).__name__}")
    notes = entry.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise MalformedFredResponseError("FRED series metadata field 'notes' must be a string or absent")
    popularity_raw = entry.get("popularity")
    popularity: str | None
    if popularity_raw is None:
        popularity = None
    elif isinstance(popularity_raw, str):
        popularity = popularity_raw
    elif isinstance(popularity_raw, int) and not isinstance(popularity_raw, bool):
        popularity = str(popularity_raw)
    else:
        raise MalformedFredResponseError(f"FRED series metadata field 'popularity' must be a string, an integer, or absent, got {type(popularity_raw).__name__}")

    return FredSeriesMetadata(
        series_id=str(entry["id"]), response_realtime_start=response_realtime_start, response_realtime_end=response_realtime_end,
        title=str(entry["title"]), observation_start=str(entry["observation_start"]), observation_end=str(entry["observation_end"]),
        frequency=str(entry["frequency"]), frequency_short=str(entry["frequency_short"]), units=str(entry["units"]),
        units_short=str(entry["units_short"]), seasonal_adjustment=str(entry["seasonal_adjustment"]),
        seasonal_adjustment_short=str(entry["seasonal_adjustment_short"]), last_updated=str(entry["last_updated"]),
        notes=notes, popularity=popularity,
    )
