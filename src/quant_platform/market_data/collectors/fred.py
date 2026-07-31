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
from urllib.parse import urlencode

from quant_platform.core.exceptions import (
    CollectorError,
    MalformedFredResponseError,
    RateLimitUnavailableError,
    RetryExhaustedError,
)
from quant_platform.market_data.adapters import RawSourceRecord
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred_schemas import (
    FredObservation,
    parse_fred_csv_response,
    parse_fred_json_response,
)
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport, TransportRequest
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState, try_acquire
from quant_platform.market_data.collectors.request_manifest import (
    CollectorRequestManifest,
    CredentialMode,
    create_request_manifest,
)
from quant_platform.market_data.collectors.response_manifest import (
    CollectorResponseManifest,
    CompletionStatus,
    create_response_manifest,
)
from quant_platform.market_data.collectors.retry import (
    RetryAttemptRecord,
    RetryFailureKind,
    RetryOutcome,
    RetryPolicy,
    classify_failure,
    parse_retry_after,
    plan_next_wait_seconds,
)
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

_RETRYABLE_TRANSPORT_EXCEPTIONS = ("TransportTimeoutError",)


def build_fred_request_manifest(
    *, series_id: str, observation_start: datetime | None, observation_end: datetime | None, response_format: str,
    timeout_policy_id: str, retry_policy_id: str, rate_limit_policy_id: str, credential_mode: CredentialMode, request_time: datetime,
    collector_version: str = FRED_COLLECTOR_VERSION,
) -> CollectorRequestManifest:
    query_params: dict[str, str] = {"series_id": series_id, "file_type": response_format}
    if observation_start is not None:
        query_params["observation_start"] = observation_start.strftime("%Y-%m-%d")
    if observation_end is not None:
        query_params["observation_end"] = observation_end.strftime("%Y-%m-%d")
    return create_request_manifest(
        collector_name=FRED_COLLECTOR_NAME, collector_version=collector_version, endpoint_host=FRED_ENDPOINT_HOST,
        endpoint_path=FRED_ENDPOINT_PATH, canonical_query_params=query_params, canonical_headers={}, requested_series_or_dataset=series_id,
        response_format=response_format, timeout_policy_id=timeout_policy_id, retry_policy_id=retry_policy_id,
        rate_limit_policy_id=rate_limit_policy_id, credential_mode=credential_mode, request_time=request_time,
        requested_interval_start=observation_start, requested_interval_end=observation_end,
    )


def _build_transport_request(
    manifest: CollectorRequestManifest, *, api_key: str | None, connect_timeout: float, read_timeout: float, max_response_bytes: int,
    request_time: datetime,
) -> TransportRequest:
    query = dict(manifest.canonical_query_params)
    if api_key is not None:
        query["api_key"] = api_key
    url = f"https://{manifest.endpoint_host}{manifest.endpoint_path}?{urlencode(query)}"
    return TransportRequest(
        url=url, headers=dict(manifest.canonical_headers), connect_timeout=connect_timeout, read_timeout=read_timeout,
        max_response_bytes=max_response_bytes, allow_redirects=False, max_redirects=0, allowed_hosts=FRED_ALLOWED_HOSTS,
        request_time=request_time,
    )


@dataclass(frozen=True, slots=True)
class FredRequestExecution:
    response_manifest: CollectorResponseManifest
    raw_bytes: bytes
    attempts: tuple[RetryAttemptRecord, ...]


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
    """The attempt loop: transport + retry + rate-limit coordinated
    together. `operation_time` is the ONLY time value used to drive rate
    limiting/`Retry-After` math -- this function never reads the wall
    clock itself; `sleep_fn` is the only place real waiting happens, and
    it is fully injectable."""
    attempts: list[RetryAttemptRecord] = []
    state = rate_limit_state

    for attempt_number in range(1, retry_policy.max_attempts + 1):
        acquired, state = try_acquire(state, rate_limit_policy, now=operation_time)
        if not acquired:
            raise RateLimitUnavailableError(
                f"no rate-limit token available for request_manifest_id {request_manifest.request_manifest_id!r} at attempt {attempt_number}"
            )

        transport_request = _build_transport_request(
            request_manifest, api_key=api_key, connect_timeout=connect_timeout, read_timeout=read_timeout,
            max_response_bytes=max_response_bytes, request_time=operation_time,
        )

        try:
            response = transport.get(transport_request)
        except CollectorError as exc:
            failure_kind = RetryFailureKind.READ_TIMEOUT if type(exc).__name__ in _RETRYABLE_TRANSPORT_EXCEPTIONS else RetryFailureKind.MALFORMED_RESPONSE
            outcome = classify_failure(kind=failure_kind, status_code=None, policy=retry_policy)
            if outcome is RetryOutcome.STOP or attempt_number == retry_policy.max_attempts:
                attempts.append(RetryAttemptRecord(
                    attempt_number=attempt_number, outcome="exhausted" if outcome is RetryOutcome.RETRY else "non_retryable_failure",
                    status_code=None, failure_kind=failure_kind.value, wait_seconds_before_next=None, detail=type(exc).__name__,
                ))
                raise RetryExhaustedError(f"request_manifest_id {request_manifest.request_manifest_id!r} exhausted retries: {exc}") from exc
            wait = plan_next_wait_seconds(retry_policy, attempt_number=attempt_number, retry_after_seconds=None)
            attempts.append(RetryAttemptRecord(
                attempt_number=attempt_number, outcome="retryable_failure", status_code=None, failure_kind=failure_kind.value,
                wait_seconds_before_next=wait, detail=type(exc).__name__,
            ))
            sleep_fn(wait)
            continue

        if response.status_code == 200:
            attempts.append(RetryAttemptRecord(attempt_number=attempt_number, outcome="success", status_code=200, failure_kind=None, wait_seconds_before_next=None))
            response_manifest = create_response_manifest(
                request_manifest_id=request_manifest.request_manifest_id, http_status=response.status_code, raw_headers=response.headers,
                raw_bytes=response.body, content_type=response.headers.get("Content-Type") or response.headers.get("content-type"),
                encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=operation_time, transport_attempt_count=attempt_number,
            )
            return FredRequestExecution(response_manifest=response_manifest, raw_bytes=response.body, attempts=tuple(attempts)), state

        outcome = classify_failure(kind=RetryFailureKind.HTTP_STATUS, status_code=response.status_code, policy=retry_policy)
        retry_after_header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        retry_after_seconds = parse_retry_after(retry_after_header, now=operation_time) if retry_after_header else None

        if outcome is RetryOutcome.STOP:
            attempts.append(RetryAttemptRecord(
                attempt_number=attempt_number, outcome="non_retryable_failure", status_code=response.status_code, failure_kind="http_status",
                wait_seconds_before_next=None,
            ))
            raise RetryExhaustedError(
                f"request_manifest_id {request_manifest.request_manifest_id!r} received non-retryable status {response.status_code}"
            )
        if attempt_number == retry_policy.max_attempts:
            attempts.append(RetryAttemptRecord(
                attempt_number=attempt_number, outcome="exhausted", status_code=response.status_code, failure_kind="http_status",
                wait_seconds_before_next=None,
            ))
            raise RetryExhaustedError(
                f"request_manifest_id {request_manifest.request_manifest_id!r} exhausted {retry_policy.max_attempts} attempts, last status {response.status_code}"
            )
        wait = plan_next_wait_seconds(retry_policy, attempt_number=attempt_number, retry_after_seconds=retry_after_seconds)
        attempts.append(RetryAttemptRecord(
            attempt_number=attempt_number, outcome="retryable_failure", status_code=response.status_code, failure_kind="http_status",
            wait_seconds_before_next=wait,
        ))
        sleep_fn(wait)

    raise RetryExhaustedError(f"request_manifest_id {request_manifest.request_manifest_id!r} exhausted all attempts")


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
