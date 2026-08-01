"""Strict FRED historical-series collector (Milestone 10, Phase 4A) --
the one place transport, retry, and rate-limiting are coordinated
together for a single logical request. `execute_fred_request` is the
attempt LOOP (see `retry.py`'s own docstring for why the loop lives here
and not there): it calls a `HistoricalHttpTransport`, classifies each
failure via `retry.classify_failure`, and -- in real, non-test use --
sleeps between attempts via an INJECTABLE `sleep_fn` (default
`time.sleep`), never hard-coded, so a test can inject a zero-cost
recorder and assert the exact deterministic attempt sequence without
ever actually waiting.

`FredSourceAdapter` implements `adapters.HistoricalSourceAdapter`
(Phase 3's own Protocol) over an ALREADY-DOWNLOADED, ALREADY-PERSISTED
raw response -- it performs ZERO network I/O itself (mirrors
`CsvCandleAdapter` reading a local file); `execute_fred_request` is the
only network-capable function in this whole module, and it is never
called from inside the adapter. `content_digest()` is bound to the exact
SAME `raw_content_digest` the response manifest itself records, which is
what makes `orchestration.run_ingestion_operation`'s own `SOURCE_VERIFIED`
check (`adapter.content_digest() != source_manifest.content_digest`)
a meaningful, real integrity check for FRED data too."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.exceptions import CollectorRequestManifestError, MalformedFredResponseError
from quant_platform.market_data.adapters import RawSourceRecord
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.execute_request import (
    CollectorRequestExecution,
    execute_collector_request,
)
from quant_platform.market_data.collectors.fred_schemas import (
    FredObservation,
    parse_fred_csv_response,
    parse_fred_json_response,
)
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import (
    CollectorRequestManifest,
    CredentialMode,
    create_request_manifest,
)
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.source_manifests import RecordKind, SourceKind

__all__ = [
    "FRED_ALLOWED_HOSTS",
    "FRED_COLLECTOR_NAME",
    "FRED_COLLECTOR_VERSION",
    "FRED_ENDPOINT_HOST",
    "FRED_ENDPOINT_PATH",
    "FRED_EXAMPLE_SERIES",
    "FredRequestExecution",
    "FredSourceAdapter",
    "build_fred_request_manifest",
    "execute_fred_request",
    "load_fred_adapter_from_cache",
]

FRED_COLLECTOR_NAME = "fred"
FRED_COLLECTOR_VERSION = "1.0.0"
FRED_ENDPOINT_HOST = "api.stlouisfed.org"
FRED_ENDPOINT_PATH = "/fred/series/observations"
FRED_ALLOWED_HOSTS = frozenset({FRED_ENDPOINT_HOST})

FRED_EXAMPLE_SERIES = ("DFII10", "DGS10", "CPIAUCSL", "DFF")
"""Configured EXAMPLES only -- `build_fred_request_manifest` accepts any
`series_id`; this is never enforced as an allowlist."""

_VALID_SORT_ORDERS = frozenset({"asc", "desc"})
_VALID_FREQUENCY_CODES = frozenset({
    "d", "w", "bw", "m", "q", "sa", "a", "wef", "weth", "wew", "wetu", "wem", "wesu", "wesa", "bwew", "bwem",
})
_VALID_AGGREGATION_METHODS = frozenset({"avg", "sum", "eop"})
_VALID_OUTPUT_TYPES = frozenset({1, 2, 3, 4})


def build_fred_request_manifest(
    *, series_id: str, observation_start: datetime | None, observation_end: datetime | None, response_format: str,
    timeout_policy_id: str, retry_policy_id: str, rate_limit_policy_id: str, credential_mode: CredentialMode, request_time: datetime,
    collector_version: str = FRED_COLLECTOR_VERSION, realtime_start: datetime | None = None, realtime_end: datetime | None = None,
    limit: int | None = None, offset: int | None = None, sort_order: str = "asc", units: str = "lin",
    frequency: str | None = None, aggregation_method: str | None = None, output_type: int | None = None,
    vintage_dates: tuple[datetime, ...] | None = None,
) -> CollectorRequestManifest:
    """Milestone 10, Phase 4B extends this with the remaining OFFICIAL
    `/fred/series/observations` parameters the spec requires be modeled
    explicitly rather than left to an undocumented FRED default:
    `realtime_start`/`realtime_end` (ALFRED vintage window),
    `limit`/`offset` (pagination), `sort_order` (ALWAYS included,
    default `"asc"` -- never left to FRED's own undocumented default),
    `units` (ALWAYS included, default `"lin"` -- a transformation code,
    never omitted), `frequency`/`aggregation_method` (aggregation is
    rejected if supplied without a frequency -- meaningless combination
    per FRED's own API contract), `output_type`, `vintage_dates`. All
    are OPTIONAL kwargs with values that reproduce the exact prior
    query shape when omitted, EXCEPT `sort_order`/`units`, which are
    now always present -- a deliberate, disclosed identity-affecting
    change (see the Phase 4B delivery report): omitting a
    result-affecting parameter and relying on FRED's own undocumented
    default is exactly what this phase's spec forbids."""
    if sort_order not in _VALID_SORT_ORDERS:
        raise CollectorRequestManifestError(f"sort_order must be one of {sorted(_VALID_SORT_ORDERS)!r}, got {sort_order!r}")
    if limit is not None and not (1 <= limit <= 100_000):
        raise CollectorRequestManifestError(f"limit must be in [1, 100000], got {limit}")
    if offset is not None and offset < 0:
        raise CollectorRequestManifestError(f"offset must be >= 0, got {offset}")
    if frequency is not None and frequency not in _VALID_FREQUENCY_CODES:
        raise CollectorRequestManifestError(f"frequency {frequency!r} is not a recognized FRED frequency code")
    if aggregation_method is not None and frequency is None:
        raise CollectorRequestManifestError("aggregation_method requires frequency to also be set -- meaningless otherwise per FRED's own API contract")
    if aggregation_method is not None and aggregation_method not in _VALID_AGGREGATION_METHODS:
        raise CollectorRequestManifestError(f"aggregation_method {aggregation_method!r} is not one of {sorted(_VALID_AGGREGATION_METHODS)!r}")
    if output_type is not None and output_type not in _VALID_OUTPUT_TYPES:
        raise CollectorRequestManifestError(f"output_type must be one of {sorted(_VALID_OUTPUT_TYPES)!r}, got {output_type}")
    if vintage_dates is not None and not vintage_dates:
        raise CollectorRequestManifestError("vintage_dates must be non-empty when supplied, or omitted entirely (None)")

    query_params: dict[str, str] = {"series_id": series_id, "file_type": response_format, "sort_order": sort_order, "units": units}
    if observation_start is not None:
        query_params["observation_start"] = observation_start.strftime("%Y-%m-%d")
    if observation_end is not None:
        query_params["observation_end"] = observation_end.strftime("%Y-%m-%d")
    if realtime_start is not None:
        query_params["realtime_start"] = realtime_start.strftime("%Y-%m-%d")
    if realtime_end is not None:
        query_params["realtime_end"] = realtime_end.strftime("%Y-%m-%d")
    if limit is not None:
        query_params["limit"] = str(limit)
    if offset is not None:
        query_params["offset"] = str(offset)
    if frequency is not None:
        query_params["frequency"] = frequency
    if aggregation_method is not None:
        query_params["aggregation_method"] = aggregation_method
    if output_type is not None:
        query_params["output_type"] = str(output_type)
    if vintage_dates is not None:
        query_params["vintage_dates"] = ",".join(d.strftime("%Y-%m-%d") for d in vintage_dates)

    return create_request_manifest(
        collector_name=FRED_COLLECTOR_NAME, collector_version=collector_version, endpoint_host=FRED_ENDPOINT_HOST,
        endpoint_path=FRED_ENDPOINT_PATH, canonical_query_params=query_params, canonical_headers={}, requested_series_or_dataset=series_id,
        response_format=response_format, timeout_policy_id=timeout_policy_id, retry_policy_id=retry_policy_id,
        rate_limit_policy_id=rate_limit_policy_id, credential_mode=credential_mode, request_time=request_time,
        requested_interval_start=observation_start, requested_interval_end=observation_end,
    )


FredRequestExecution = CollectorRequestExecution
"""Milestone 10, Phase 4C: `execute_request.CollectorRequestExecution`
under its original Phase 4A name -- preserved for backward compatibility
(nothing outside this module actually names the type, but it remains
part of this module's own public `__all__`)."""


def execute_fred_request(
    *,
    transport: HistoricalHttpTransport,
    request_manifest: CollectorRequestManifest,
    api_key: str | None,
    retry_policy: RetryPolicy,
    rate_limit_policy: RateLimitPolicy,
    rate_limit_state: TokenBucketState,
    connect_timeout: float,
    read_timeout: float,
    max_response_bytes: int,
    operation_time: datetime,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[FredRequestExecution, TokenBucketState]:
    """Milestone 10, Phase 4C: a THIN ALIAS of `execute_request.
    execute_collector_request` (bound to `FRED_ALLOWED_HOSTS`) -- zero
    duplication. The attempt loop itself was always fully provider-
    neutral (see that module's own docstring for the extraction
    rationale); this wrapper exists only so every existing caller of
    `fred.execute_fred_request` keeps working unchanged."""
    return execute_collector_request(
        transport=transport, request_manifest=request_manifest, api_key=api_key, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=rate_limit_state, connect_timeout=connect_timeout, read_timeout=read_timeout, max_response_bytes=max_response_bytes,
        operation_time=operation_time, allowed_hosts=FRED_ALLOWED_HOSTS, sleep_fn=sleep_fn,
    )


# --------------------------------------------------------------------------
# Source adapter bridge -- zero network I/O, reads only persisted bytes.
# --------------------------------------------------------------------------
def _observation_to_raw_fields(obs: FredObservation) -> dict[str, str]:
    fields = {"date": obs.observation_date, "value": obs.value_text}
    if obs.realtime_start is not None:
        fields["realtime_start"] = obs.realtime_start
    if obs.realtime_end is not None:
        fields["realtime_end"] = obs.realtime_end
    return fields


@dataclass(frozen=True, slots=True)
class FredSourceAdapter:
    _series_id: str
    _content_digest: str
    _byte_size: int
    _records: tuple[RawSourceRecord, ...]
    _metadata: dict[str, object]

    def source_kind(self) -> SourceKind:
        return SourceKind.FRED_API

    def source_schema_version(self) -> int:
        return 1

    def record_kind(self) -> RecordKind:
        return RecordKind.MACRO_OBSERVATION

    def content_digest(self) -> str:
        return self._content_digest

    def byte_size(self) -> int:
        return self._byte_size

    def describe(self) -> dict[str, object]:
        return dict(self._metadata)

    def iter_records(self) -> Iterator[RawSourceRecord]:
        return iter(self._records)


def load_fred_adapter_from_cache(cache: RawResponseCache, response_manifest_id: str, *, series_id: str, response_format: str) -> FredSourceAdapter:
    """Builds a `FredSourceAdapter` from an ALREADY-CACHED response --
    zero network I/O (`RawResponseCache.read_bytes` re-hashes on every
    read; a corrupted or tampered cache entry fails closed via
    `CacheCorruptionError` here, never silently). This is the ONLY
    supported way to obtain a `FredSourceAdapter`: there is no
    constructor that accepts raw bytes directly from a live transport
    call, so an adapter can never exist without first being durably
    cached -- "persist raw downloaded bytes before parsing" is enforced
    structurally, not by convention."""
    manifest = cache.read_manifest(response_manifest_id)
    if manifest is None:
        raise MalformedFredResponseError(f"no cached response manifest for response_manifest_id {response_manifest_id!r}")
    raw_bytes = cache.read_bytes(response_manifest_id, verify=True)

    if response_format == "json":
        observations = parse_fred_json_response(raw_bytes, series_id=series_id)
    elif response_format == "csv":
        observations = parse_fred_csv_response(raw_bytes, series_id=series_id)
    else:
        raise MalformedFredResponseError(f"unsupported response_format {response_format!r}, expected 'json' or 'csv'")

    records = tuple(
        RawSourceRecord(row_index=obs.row_index, raw_fields=_observation_to_raw_fields(obs), raw_text=f"{obs.observation_date},{obs.value_text}")
        for obs in observations
    )
    metadata: dict[str, object] = {
        "series_id": series_id, "response_manifest_id": response_manifest_id, "response_format": response_format,
        "observation_count": len(records),
    }
    return FredSourceAdapter(
        _series_id=series_id, _content_digest=manifest.raw_content_digest, _byte_size=manifest.byte_length, _records=records, _metadata=metadata,
    )
