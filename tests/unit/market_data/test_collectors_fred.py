"""FRED collector tests (Milestone 10, Phase 4A) -- `execute_fred_request`'s
attempt loop (retry + rate-limit coordination), secret handling in the
real transport call vs. the durable manifest, and `FredSourceAdapter`'s
zero-network-I/O construction from an already-cached response."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
    fred_json_body,
)

from quant_platform.core.exceptions import RateLimitUnavailableError, RetryExhaustedError
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred import (
    build_fred_request_manifest,
    execute_fred_request,
    load_fred_adapter_from_cache,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import (
    create_rate_limit_policy,
    initial_bucket_state,
    try_acquire,
)
from quant_platform.market_data.collectors.request_manifest import CredentialMode

SECRET = "SECRET_KEY_VALUE"


def _request_manifest(credential_mode: CredentialMode = CredentialMode.ANONYMOUS):
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    return build_fred_request_manifest(
        series_id="DGS10", observation_start=T0, observation_end=datetime(2024, 1, 31, tzinfo=timezone.utc), response_format="json",
        timeout_policy_id="a" * 64, retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
        credential_mode=credential_mode, request_time=T0,
    ), retry_policy, rate_limit_policy


class TestSecretRedactionInRequestManifest:
    def test_api_key_absent_from_canonical_query_params(self) -> None:
        manifest, *_ = _request_manifest(CredentialMode.API_KEY)
        assert "api_key" not in manifest.canonical_query_params

    def test_api_key_absent_from_serialized_manifest(self) -> None:
        import json

        manifest, *_ = _request_manifest(CredentialMode.API_KEY)
        assert SECRET not in json.dumps(manifest.to_json_dict())


class TestExecuteFredRequestHappyPath:
    def test_successful_first_attempt(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
            ]), final_url=f"https://{manifest.endpoint_host}{manifest.endpoint_path}"),
        ])
        sleep_calls = []
        execution, _new_state = execute_fred_request(
            transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
            max_response_bytes=1_000_000, operation_time=T0, sleep_fn=sleep_calls.append,
        )
        assert len(execution.attempts) == 1
        assert execution.attempts[0].outcome == "success"
        assert sleep_calls == []

    def test_real_secret_reaches_transport_but_never_the_manifest(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest(CredentialMode.API_KEY)
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02"},
            ]), final_url=f"https://{manifest.endpoint_host}{manifest.endpoint_path}"),
        ])
        import json

        execution, _ = execute_fred_request(
            transport=transport, request_manifest=manifest, api_key=SECRET, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
            max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
        )
        assert f"api_key={SECRET}" in transport.calls[0].url
        assert SECRET not in json.dumps(execution.response_manifest.to_json_dict())


class TestExecuteFredRequestRetry:
    def test_retry_after_429_then_success_with_deterministic_sequence(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=429, headers={"Retry-After": "5"}, body=b"", final_url="x"),
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([{"date": "2024-01-02", "value": "4.02"}]), final_url="x"),
        ])
        sleep_calls: list[float] = []
        execution, _ = execute_fred_request(
            transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
            max_response_bytes=1_000_000, operation_time=T0, sleep_fn=sleep_calls.append,
        )
        assert len(execution.attempts) == 2
        assert execution.attempts[0].outcome == "retryable_failure" and execution.attempts[0].status_code == 429
        assert execution.attempts[1].outcome == "success"
        assert sleep_calls == [5.0]

    def test_non_retryable_401_stops_after_one_attempt(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[TransportResponse(status_code=401, headers={}, body=b"", final_url="x")])
        with pytest.raises(RetryExhaustedError):
            execute_fred_request(
                transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
                max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
            )
        assert len(transport.calls) == 1

    def test_retry_exhaustion_after_max_attempts(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=503, headers={}, body=b"", final_url="x"),
            TransportResponse(status_code=503, headers={}, body=b"", final_url="x"),
            TransportResponse(status_code=503, headers={}, body=b"", final_url="x"),
        ])
        with pytest.raises(RetryExhaustedError):
            execute_fred_request(
                transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
                max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
            )
        assert len(transport.calls) == retry_policy.max_attempts == 3

    def test_no_real_sleep_ever_happens_in_this_test(self) -> None:
        """Confirms the injected `sleep_fn` is the ONLY place waiting
        could happen -- this whole module runs in well under a second
        despite exercising multiple retry sequences."""
        import time

        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=503, headers={}, body=b"", final_url="x"),
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([{"date": "2024-01-02", "value": "4.02"}]), final_url="x"),
        ])
        start = time.monotonic()
        execute_fred_request(
            transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
            max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
        )
        assert time.monotonic() - start < 1.0


class TestRateLimitUnavailable:
    def test_fails_closed_before_ever_calling_transport(self) -> None:
        manifest, retry_policy, _rate_limit_policy = _request_manifest()
        policy_tiny = create_rate_limit_policy(max_tokens=Decimal(1), refill_rate_per_second=Decimal("0.001"))
        state_tiny = initial_bucket_state(policy_tiny, now=T0)
        _acquired, state_tiny = try_acquire(state_tiny, policy_tiny, now=T0)  # drain the single token
        transport = FakeTransport(responses=[])
        with pytest.raises(RateLimitUnavailableError):
            execute_fred_request(
                transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=policy_tiny,
                rate_limit_state=state_tiny, connect_timeout=5.0, read_timeout=10.0, max_response_bytes=1_000_000, operation_time=T0,
                sleep_fn=lambda _seconds: None,
            )
        assert transport.calls == []


class TestFredSourceAdapterFromCache:
    def test_loaded_with_zero_network_calls(self) -> None:
        manifest, retry_policy, rate_limit_policy = _request_manifest()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02"},
                {"date": "2024-01-03", "value": "4.05"},
            ]), final_url="x"),
        ])
        execution, _ = execute_fred_request(
            transport=transport, request_manifest=manifest, api_key=None, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
            max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
        )
        cache = RawResponseCache(Path(tempfile.mkdtemp()))
        cache.store(execution.response_manifest, execution.raw_bytes)

        adapter = load_fred_adapter_from_cache(cache, execution.response_manifest.response_manifest_id, series_id="DGS10", response_format="json")
        assert adapter.content_digest() == execution.response_manifest.raw_content_digest
        records = list(adapter.iter_records())
        assert len(records) == 2
        assert records[0].raw_fields["value"] == "4.02"
        assert len(transport.calls) == 1  # only the ORIGINAL fetch touched the network; loading the adapter did not
