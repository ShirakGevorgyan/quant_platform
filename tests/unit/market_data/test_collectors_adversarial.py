"""Adversarial tests (Milestone 10, Phase 4A) -- scenarios not already
covered by `test_collectors_transport_security.py` (redirect-to-
localhost/metadata-endpoint), `test_collectors_retry.py` (malformed
Retry-After), `test_collectors_verification.py`/`test_collectors_
reconciliation.py` (coherent rehash after tampering), or the safety scan
(hidden float conversion, wall-clock/uuid4 semantic input): a secret
injected into an error message or exception, a secret's fate across a
rejected redirect, and a cached response manually swapped between two
DIFFERENT response identities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
)

from quant_platform.core.exceptions import CacheCorruptionError, DisallowedUrlError, RetryExhaustedError
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred import build_fred_request_manifest, execute_fred_request
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.response_manifest import CompletionStatus, create_response_manifest
from quant_platform.market_data.collectors.transport import StdlibHttpsTransport

SECRET = "SECRET_KEY_VALUE"
HOST = "api.stlouisfed.org"
ALLOWED = frozenset({HOST})


class TestSecretInjectedIntoErrorBody:
    def test_a_secret_reflected_in_a_failed_response_body_never_leaks_into_the_raised_exception(self) -> None:
        """A malicious/misbehaving server could echo request parameters
        (including a query-string secret) back in an ERROR response
        body. `RetryExhaustedError`'s own message must never embed
        `response.body` -- confirmed here with a body that DOES contain
        the secret, checking the raised exception's message stays clean."""
        retry_policy = default_retry_policy()
        rate_limit_policy = default_rate_limit_policy()
        manifest = build_fred_request_manifest(
            series_id="DGS10", observation_start=T0, observation_end=T0, response_format="json", timeout_policy_id="a" * 64,
            retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
            credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        transport = FakeTransport(responses=[
            TransportResponse(status_code=401, headers={}, body=f"invalid api_key={SECRET}".encode(), final_url="x"),
        ])
        with pytest.raises(RetryExhaustedError) as exc_info:
            execute_fred_request(
                transport=transport, request_manifest=manifest, api_key=SECRET, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), connect_timeout=5.0, read_timeout=10.0,
                max_response_bytes=1_000_000, operation_time=T0, sleep_fn=lambda _seconds: None,
            )
        assert SECRET not in str(exc_info.value)


class TestSecretAcrossRejectedRedirect:
    def test_redirect_to_a_disallowed_host_is_rejected_without_leaking_the_original_url_secret(self) -> None:
        """The original request URL legitimately embeds the real secret
        (in-flight, in-memory only -- see `protocols.py`'s own
        docstring). A redirect to a disallowed host must be rejected,
        and the resulting exception must never echo the ORIGINAL
        request's secret-bearing URL."""
        transport = StdlibHttpsTransport()
        original_url = f"https://{HOST}/fred/series/observations?series_id=DGS10&api_key={SECRET}"
        request = TransportRequest(url=original_url, allowed_hosts=ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(DisallowedUrlError) as exc_info:
            transport._follow_redirect(302, {"Location": "https://evil.example.com/steal"}, request, redirect_count=0, current_url=original_url)
        assert SECRET not in str(exc_info.value)

    def test_redirect_location_itself_carrying_the_secret_does_not_propagate_it_into_the_exception(self) -> None:
        """A compromised/malicious FRED-lookalike could redirect to a
        THIRD-PARTY host with the secret embedded in the Location's own
        query string (attempting exfiltration via the collector's own
        error reporting). Still rejected (host not allowlisted), and the
        raised exception must not embed that Location's secret either
        -- it reports only the host, not the disallowed URL wholesale in
        a way that would be routinely logged with sensitive info."""
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{HOST}/x", allowed_hosts=ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        malicious_location = f"https://evil.example.com/exfiltrate?leaked_key={SECRET}"
        with pytest.raises(DisallowedUrlError) as exc_info:
            transport._follow_redirect(302, {"Location": malicious_location}, request, redirect_count=0, current_url=f"https://{HOST}/x")
        # The "host not on allowlist" rejection message embeds only the
        # HOSTNAME, never the full URL/query string -- the secret never
        # reaches the exception at all.
        assert SECRET not in str(exc_info.value)


class TestCachedResponseSwappedBetweenRequests:
    def test_manually_swapping_two_stored_bodies_is_detected_by_rehash(self) -> None:
        """Simulates an attacker (or filesystem corruption) swapping the
        on-disk bytes of two DIFFERENT, already-cached responses --
        each's own re-hash against its OWN manifest must independently
        detect the mismatch; neither swapped file silently passes as
        the other's legitimate content."""
        cache = RawResponseCache(Path(tempfile.mkdtemp()))
        manifest_a = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={}, raw_bytes=b"response-A-content", content_type="application/json",
            encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        manifest_b = create_response_manifest(
            request_manifest_id="b" * 64, http_status=200, raw_headers={}, raw_bytes=b"response-B-content-different-length", content_type="application/json",
            encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        cache.store(manifest_a, b"response-A-content")
        cache.store(manifest_b, b"response-B-content-different-length")

        # Swap the raw bytes on disk directly.
        path_a = cache._body_path(manifest_a.response_manifest_id)
        path_b = cache._body_path(manifest_b.response_manifest_id)
        bytes_a, bytes_b = path_a.read_bytes(), path_b.read_bytes()
        path_a.write_bytes(bytes_b)
        path_b.write_bytes(bytes_a)

        with pytest.raises(CacheCorruptionError):
            cache.read_bytes(manifest_a.response_manifest_id, verify=True)
        with pytest.raises(CacheCorruptionError):
            cache.read_bytes(manifest_b.response_manifest_id, verify=True)


class TestDnsRebindingAssumptionsAreDocumented:
    def test_transport_module_documents_the_residual_dns_rebinding_limitation(self) -> None:
        """The two-phase validation closes the classic TOCTOU gap (see
        `test_collectors_transport_security.py::TestDnsRebindingResistantResolution`)
        but a compromised, ALREADY-VALIDATED IP's own routing at the OS/
        network level is explicitly out of this layer's control -- this
        must remain documented, not silently assumed away."""
        import quant_platform.market_data.collectors.transport as transport_module

        assert transport_module.__doc__ is not None
        assert "DNS rebinding" in transport_module.__doc__
        assert "residual" in transport_module.__doc__.lower() or "honestly-disclosed" in transport_module.__doc__.lower()
