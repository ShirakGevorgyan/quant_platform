"""Provider-neutral collector request execution (Milestone 10, Phase 4A
originally, extracted in Phase 4C).

`execute_collector_request` is the attempt LOOP (see `retry.py`'s own
docstring for why the loop lives here and not there): it calls a
`HistoricalHttpTransport`, classifies each failure via `retry.
classify_failure`, and -- in real, non-test use -- sleeps between
attempts via an INJECTABLE `sleep_fn` (default `time.sleep`).

EXTRACTED FROM `fred.py`, NOT DUPLICATED: this attempt loop was already
100% provider-neutral in its Phase 4A implementation (no FRED-specific
logic anywhere in it -- it builds a URL generically from `manifest.
endpoint_host`/`endpoint_path`/`canonical_query_params`, coordinates
transport/retry/rate-limit generically, and records generic
`RetryAttemptRecord`s). Phase 4C's own provider-neutral cross-asset
collectors need the EXACT same coordination for a materially different
provider (`providers/alpha_vantage.py`), so this module promotes it to
a shared home rather than duplicating ~100 lines a second time --
`fred.execute_fred_request` is now a thin alias of `execute_collector_
request` here, exactly mirroring `fred_series_metadata.
execute_fred_series_metadata_request`'s own established alias
precedent. The one genuine generalization this extraction required:
`allowed_hosts` becomes an explicit PARAMETER (was hardcoded to
`fred.FRED_ALLOWED_HOSTS`) -- each provider adapter supplies its own."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode

from quant_platform.core.exceptions import CollectorError, RateLimitUnavailableError, RetryExhaustedError
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport, TransportRequest
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState, try_acquire
from quant_platform.market_data.collectors.request_manifest import CollectorRequestManifest
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

__all__ = ["CollectorRequestExecution", "build_transport_request", "execute_collector_request"]

_RETRYABLE_TRANSPORT_EXCEPTIONS = ("TransportTimeoutError",)


def build_transport_request(
    manifest: CollectorRequestManifest, *, api_key: str | None, connect_timeout: float, read_timeout: float, max_response_bytes: int,
    request_time: datetime, allowed_hosts: frozenset[str],
) -> TransportRequest:
    query = dict(manifest.canonical_query_params)
    if api_key is not None:
        query["api_key"] = api_key
    url = f"https://{manifest.endpoint_host}{manifest.endpoint_path}?{urlencode(query)}"
    return TransportRequest(
        url=url, headers=dict(manifest.canonical_headers), connect_timeout=connect_timeout, read_timeout=read_timeout,
        max_response_bytes=max_response_bytes, allow_redirects=False, max_redirects=0, allowed_hosts=allowed_hosts, request_time=request_time,
    )


@dataclass(frozen=True, slots=True)
class CollectorRequestExecution:
    response_manifest: CollectorResponseManifest
    raw_bytes: bytes
    attempts: tuple[RetryAttemptRecord, ...]


def execute_collector_request(
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
    allowed_hosts: frozenset[str],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[CollectorRequestExecution, TokenBucketState]:
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

        transport_request = build_transport_request(
            request_manifest, api_key=api_key, connect_timeout=connect_timeout, read_timeout=read_timeout,
            max_response_bytes=max_response_bytes, request_time=operation_time, allowed_hosts=allowed_hosts,
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
                # Deliberately NEVER interpolate `{exc}` (the raw exception text) here: a
                # `TransportRequest.url` MAY legitimately carry a real `api_key` query
                # parameter for the in-flight call (see protocols.py), and a transport
                # implementation's own exception message may echo that URL back (e.g. in
                # a timeout message) -- only the exception's CLASS NAME is safe to surface,
                # exactly mirroring `RetryAttemptRecord.detail` immediately above.
                raise RetryExhaustedError(
                    f"request_manifest_id {request_manifest.request_manifest_id!r} exhausted retries: {type(exc).__name__}"
                ) from exc
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
            return CollectorRequestExecution(response_manifest=response_manifest, raw_bytes=response.body, attempts=tuple(attempts)), state

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
