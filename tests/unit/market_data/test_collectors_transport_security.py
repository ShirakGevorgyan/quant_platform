"""Transport security tests (Milestone 10, Phase 4A) -- the SSRF/URL
security matrix required by the Phase 4A specification. Tests the PURE,
network-free security-critical functions directly (`_validate_url_static`,
`_resolve_and_validate_address`, `_is_globally_routable`,
`_looks_like_ip_literal`) plus the redirect-validation and size-bounded
reading logic in `StdlibHttpsTransport`, none of which requires touching
a real network. `StdlibHttpsTransport`'s own wire-level pinned-IP
connection/response-construction path is exercised indirectly by every
`FakeTransport`-based collector test elsewhere in this suite (the
collector layer depends only on the `HistoricalHttpTransport` Protocol,
never on this concrete implementation) -- see
`docs/milestone10_phase4a_delivery_report.md`'s Known Non-Blocking
Limitations for why a full offline mock of stdlib `http.client`'s
wire-level behavior is out of scope here."""

from __future__ import annotations

import socket
from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import (
    DisallowedUrlError,
    RedirectViolationError,
    ResponseTooLargeError,
    SsrfTargetError,
    TransportTimeoutError,
)
from quant_platform.market_data.collectors.fred import FRED_ALLOWED_HOSTS
from quant_platform.market_data.collectors.protocols import TransportRequest
from quant_platform.market_data.collectors.transport import (
    ForbiddenTransport,
    StdlibHttpsTransport,
    _is_globally_routable,
    _looks_like_ip_literal,
    _read_bounded,
    _resolve_and_validate_address,
    _validate_url_static,
)

_HOST = "api.stlouisfed.org"
_ALLOWED = frozenset({_HOST})
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


class TestStaticUrlValidation:
    def test_https_allowed_host_passes(self) -> None:
        parts = _validate_url_static(f"https://{_HOST}/fred/series/observations?series_id=DGS10", allowed_hosts=_ALLOWED)
        assert parts.hostname == _HOST

    def test_http_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static(f"http://{_HOST}/x", allowed_hosts=_ALLOWED)

    def test_ftp_scheme_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static(f"ftp://{_HOST}/x", allowed_hosts=_ALLOWED)

    def test_file_scheme_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("file:///etc/passwd", allowed_hosts=_ALLOWED)

    def test_data_scheme_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("data:text/plain,hello", allowed_hosts=_ALLOWED)

    def test_userinfo_in_url_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static(f"https://user:pass@{_HOST}/x", allowed_hosts=_ALLOWED)

    def test_non_allowlisted_host_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("https://evil.example.com/x", allowed_hosts=_ALLOWED)

    def test_host_allowlist_match_is_case_insensitive(self) -> None:
        _validate_url_static(f"https://{_HOST.upper()}/x", allowed_hosts=_ALLOWED)

    def test_ipv4_literal_host_is_rejected_even_if_allowlisted(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("https://93.184.216.34/x", allowed_hosts=frozenset({"93.184.216.34"}))

    def test_ipv6_literal_host_is_rejected_even_if_allowlisted(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("https://[2001:db8::1]/x", allowed_hosts=frozenset({"2001:db8::1"}))

    def test_localhost_is_rejected_when_not_allowlisted(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("https://localhost/x", allowed_hosts=_ALLOWED)

    def test_no_host_is_rejected(self) -> None:
        with pytest.raises(DisallowedUrlError):
            _validate_url_static("https:///x", allowed_hosts=_ALLOWED)


class TestIpLiteralDetection:
    def test_plain_ipv4_is_detected(self) -> None:
        assert _looks_like_ip_literal("127.0.0.1")

    def test_bracketed_ipv6_is_detected(self) -> None:
        assert _looks_like_ip_literal("[::1]")

    def test_hostname_is_not_detected_as_ip_literal(self) -> None:
        assert not _looks_like_ip_literal(_HOST)


class TestGlobalRoutabilityCheck:
    """Direct unit tests of `_is_globally_routable`, the resolved-IP
    check that closes the DNS-rebinding gap -- covers every category the
    spec's test matrix requires, with zero network dependency (pure
    `ipaddress` logic)."""

    @pytest.mark.parametrize(
        "ip_text",
        [
            "127.0.0.1",  # loopback
            "::1",  # IPv6 loopback
            "10.0.0.5",  # private IPv4
            "172.16.0.5",  # private IPv4
            "192.168.1.5",  # private IPv4
            "fd00::1",  # unique local IPv6 (private)
            "169.254.169.254",  # link-local / cloud metadata service
            "fe80::1",  # IPv6 link-local
            "224.0.0.1",  # multicast
            "0.0.0.0",  # unspecified
            "192.0.0.1",  # reserved (IETF protocol assignments)
        ],
    )
    def test_non_global_addresses_are_rejected(self, ip_text: str) -> None:
        import ipaddress

        assert not _is_globally_routable(ipaddress.ip_address(ip_text))

    def test_a_real_global_address_is_accepted(self) -> None:
        import ipaddress

        assert _is_globally_routable(ipaddress.ip_address("93.184.216.34"))


class TestDnsRebindingResistantResolution:
    """`_resolve_and_validate_address` resolves ONCE and validates the
    ACTUAL resolved IP -- "localhost" is not an IP literal (so it would
    pass `_validate_url_static`'s allowlist check if ever allowlisted by
    mistake) but resolves only to loopback addresses, which this catches."""

    def test_localhost_resolves_only_to_non_global_addresses_and_is_rejected(self) -> None:
        with pytest.raises(SsrfTargetError):
            _resolve_and_validate_address("localhost", 443)

    def test_unresolvable_host_raises_ssrf_target_error(self) -> None:
        with pytest.raises(SsrfTargetError):
            _resolve_and_validate_address("this-host-does-not-exist.invalid.example", 443)


class TestRedirectValidation:
    def test_redirect_without_allow_redirects_is_rejected(self) -> None:
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=False, max_redirects=0, request_time=T0)
        with pytest.raises(RedirectViolationError):
            transport._follow_redirect(302, {"Location": "https://evil.example.com/y"}, request, redirect_count=0, current_url=f"https://{_HOST}/x")

    def test_redirect_exceeding_max_redirects_is_rejected(self) -> None:
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=1, request_time=T0)
        with pytest.raises(RedirectViolationError):
            transport._follow_redirect(302, {"Location": f"https://{_HOST}/y"}, request, redirect_count=1, current_url=f"https://{_HOST}/x")

    def test_redirect_scheme_downgrade_is_rejected(self) -> None:
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(RedirectViolationError):
            transport._follow_redirect(302, {"Location": f"http://{_HOST}/y"}, request, redirect_count=0, current_url=f"https://{_HOST}/x")

    def test_redirect_with_no_location_header_is_rejected(self) -> None:
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(RedirectViolationError):
            transport._follow_redirect(302, {}, request, redirect_count=0, current_url=f"https://{_HOST}/x")

    def test_redirect_to_disallowed_host_is_rejected_on_the_recursive_get(self) -> None:
        """A redirect to an allowed SCHEME but a host outside the
        allowlist must still be rejected -- `_follow_redirect` recurses
        into `_get`, which re-runs the full `_validate_url_static` check
        against the redirect TARGET, not just the original URL."""
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(DisallowedUrlError):
            transport._follow_redirect(302, {"Location": "https://evil.example.com/y"}, request, redirect_count=0, current_url=f"https://{_HOST}/x")

    def test_redirect_to_localhost_is_rejected(self) -> None:
        """Adversarial category: a compromised/malicious upstream
        redirecting to a loopback address must not be followed."""
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(DisallowedUrlError):
            transport._follow_redirect(302, {"Location": "https://127.0.0.1/y"}, request, redirect_count=0, current_url=f"https://{_HOST}/x")

    def test_redirect_to_cloud_metadata_endpoint_is_rejected(self) -> None:
        transport = StdlibHttpsTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, allow_redirects=True, max_redirects=3, request_time=T0)
        with pytest.raises(DisallowedUrlError):
            transport._follow_redirect(302, {"Location": "https://169.254.169.254/latest/meta-data/"}, request, redirect_count=0, current_url=f"https://{_HOST}/x")


class TestSizeBoundedReading:
    class _FakeStreamingResponse:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = list(chunks)

        def read(self, _chunk_size: int) -> bytes:
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    def test_response_within_bound_is_read_fully(self) -> None:
        fake = self._FakeStreamingResponse([b"abc", b"def", b""])
        body = _read_bounded(fake, max_bytes=100, url="https://x/y")  # type: ignore[arg-type]
        assert body == b"abcdef"

    def test_oversized_response_is_rejected(self) -> None:
        fake = self._FakeStreamingResponse([b"a" * 60, b"a" * 60])
        with pytest.raises(ResponseTooLargeError):
            _read_bounded(fake, max_bytes=100, url="https://x/y")  # type: ignore[arg-type]

    def test_read_timeout_during_streaming_is_mapped(self) -> None:
        class _TimeoutOnRead:
            def read(self, _chunk_size: int) -> bytes:
                raise TimeoutError("simulated read timeout")

        with pytest.raises(TransportTimeoutError):
            _read_bounded(_TimeoutOnRead(), max_bytes=100, url="https://x/y")  # type: ignore[arg-type]


class TestConnectTimeoutMapping:
    """No real network needed: connecting to an unused LOOPBACK port
    fails almost instantly with a local `ConnectionRefusedError`
    (`OSError`), exercising the exact same mapping path a real remote
    connect-timeout would take."""

    def test_connection_failure_is_mapped_to_transport_timeout_error(self) -> None:
        import ssl

        from quant_platform.market_data.collectors.transport import _PinnedHTTPSConnection

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            unused_port = probe.getsockname()[1]
        # The probe socket is now closed, so this port is (almost certainly) refused.
        conn = _PinnedHTTPSConnection(
            "example.invalid", "127.0.0.1", port=unused_port, connect_timeout=1.0, read_timeout=1.0, context=ssl.create_default_context(),
        )
        with pytest.raises(TransportTimeoutError):
            conn.connect()


class TestForbiddenTransport:
    def test_get_raises_immediately(self) -> None:
        transport = ForbiddenTransport()
        request = TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, request_time=T0)
        with pytest.raises(AssertionError):
            transport.get(request)


class TestTransportRequestValidation:
    def test_empty_allowed_hosts_is_rejected(self) -> None:
        from quant_platform.core.exceptions import CollectorError

        with pytest.raises(CollectorError):
            TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=frozenset(), request_time=T0)

    def test_non_positive_connect_timeout_is_rejected(self) -> None:
        from quant_platform.core.exceptions import CollectorError

        with pytest.raises(CollectorError):
            TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, connect_timeout=0.0, request_time=T0)

    def test_non_positive_read_timeout_is_rejected(self) -> None:
        from quant_platform.core.exceptions import CollectorError

        with pytest.raises(CollectorError):
            TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, read_timeout=-1.0, request_time=T0)

    def test_non_positive_max_response_bytes_is_rejected(self) -> None:
        from quant_platform.core.exceptions import CollectorError

        with pytest.raises(CollectorError):
            TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, max_response_bytes=0, request_time=T0)

    def test_negative_max_redirects_is_rejected(self) -> None:
        from quant_platform.core.exceptions import CollectorError

        with pytest.raises(CollectorError):
            TransportRequest(url=f"https://{_HOST}/x", allowed_hosts=_ALLOWED, max_redirects=-1, request_time=T0)


def test_fred_allowlist_matches_transport_expectations() -> None:
    assert FRED_ALLOWED_HOSTS == _ALLOWED
