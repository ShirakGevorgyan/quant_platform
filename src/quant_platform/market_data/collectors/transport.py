"""SSRF-hardened stdlib HTTPS transport (Milestone 10, Phase 4A).

No third-party HTTP library is an existing dependency of this repository
(confirmed: `pyproject.toml` declares none), and one is not justified for
a single "historical HTTPS GET" use case -- this implementation uses only
`http.client`, `ssl`, `socket`, and `ipaddress` from the standard library,
which also makes the security-critical logic below fully auditable
without trusting a third party's own redirect/proxy/cookie handling.

TWO-PHASE HOST VALIDATION, defense in depth against SSRF:
1. `_validate_url_static` -- a pure, pre-DNS check on the URL STRING
   itself: scheme must be `https`, no userinfo, host present, host on the
   caller's explicit allowlist (case-insensitive exact match), host is
   NOT an IP literal (an IP-literal host is always rejected outright,
   regardless of allowlist -- this alone blocks `127.0.0.1`, private
   ranges, and metadata-service addresses like `169.254.169.254` if a
   caller ever mistakenly allowlisted one).
2. `_resolve_and_validate_address` -- resolves the (already
   allowlisted) HOSTNAME via `socket.getaddrinfo` and validates the
   ACTUAL RESOLVED IP is global/public (`ipaddress.*.is_global`) --
   defending against DNS rebinding, where a hostname that legitimately
   resolves to a public address at allowlist-check time could resolve to
   a private address by connect time. This transport resolves ONCE,
   validates every candidate, and connects DIRECTLY to a validated IP
   (never re-resolving the hostname a second time internally), closing
   the classic time-of-check/time-of-use gap -- see the module docstring
   of `market_data_architecture.md`'s Phase 4A section for the residual,
   honestly-disclosed limitation this still carries (a compromised
   *validated* IP's own routing at the OS/network level is out of this
   layer's control)."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

from quant_platform.core.exceptions import (
    DisallowedUrlError,
    RedirectViolationError,
    ResponseTooLargeError,
    SsrfTargetError,
    TransportTimeoutError,
)
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse

__all__ = ["ForbiddenTransport", "StdlibHttpsTransport"]

_ALLOWED_SCHEME = "https"


def _validate_url_static(url: str, *, allowed_hosts: frozenset[str]) -> SplitResult:
    parts = urlsplit(url)
    if parts.scheme.lower() != _ALLOWED_SCHEME:
        raise DisallowedUrlError(f"URL scheme must be {_ALLOWED_SCHEME!r}, got {parts.scheme!r} in {url!r}")
    if "@" in (parts.netloc or ""):
        raise DisallowedUrlError(f"URL must not carry userinfo: {url!r}")
    host = parts.hostname
    if not host:
        raise DisallowedUrlError(f"URL has no host: {url!r}")
    if _looks_like_ip_literal(host):
        raise DisallowedUrlError(f"IP-literal hosts are never allowed: {host!r} in {url!r}")
    allowed_lower = {h.lower() for h in allowed_hosts}
    if host.lower() not in allowed_lower:
        raise DisallowedUrlError(f"host {host!r} is not on the allowlist {sorted(allowed_hosts)!r}")
    return parts


def _looks_like_ip_literal(host: str) -> bool:
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


def _resolve_and_validate_address(host: str, port: int) -> str:
    try:
        candidates = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise SsrfTargetError(f"could not resolve host {host!r}: {exc}") from exc
    for _family, _type, _proto, _canonname, sockaddr in candidates:
        raw_ip = str(sockaddr[0])
        try:
            ip_obj = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if _is_globally_routable(ip_obj):
            return raw_ip
    raise SsrfTargetError(f"host {host!r} resolved only to non-global address(es): {[str(c[4][0]) for c in candidates]}")


def _is_globally_routable(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast
        or ip_obj.is_reserved or ip_obj.is_unspecified
    )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects directly to a PRE-VALIDATED IP address (never re-resolving
    the hostname internally), while still sending the original hostname
    via SNI/`Host` header for correct TLS certificate validation --
    closing the DNS-rebinding time-of-check/time-of-use gap described in
    this module's own docstring. Applies `connect_timeout` to the socket
    connect phase and switches to `read_timeout` immediately afterward,
    which `http.client`'s own single `timeout=` constructor argument
    cannot express."""

    def __init__(self, original_host: str, pinned_ip: str, *, port: int, connect_timeout: float, read_timeout: float, context: ssl.SSLContext) -> None:
        super().__init__(original_host, port=port, timeout=connect_timeout, context=context)
        self._pinned_ip = pinned_ip
        self._read_timeout = read_timeout
        self._ssl_context = context

    def connect(self) -> None:
        try:
            raw_sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)
        except TimeoutError as exc:
            raise TransportTimeoutError(f"connect timeout to {self.host!r} ({self._pinned_ip!r}:{self.port}): {exc}") from exc
        except OSError as exc:
            raise TransportTimeoutError(f"could not connect to {self.host!r} ({self._pinned_ip!r}:{self.port}): {exc}") from exc
        self.sock = self._ssl_context.wrap_socket(raw_sock, server_hostname=self.host)
        self.sock.settimeout(self._read_timeout)


class StdlibHttpsTransport:
    """The one concrete `HistoricalHttpTransport` implementation. No
    implicit global session state: every `.get()` call resolves,
    validates, connects, and tears down its own connection -- no
    connection pooling, no cookie jar, no proxy-environment inheritance
    (`http.client` never reads `HTTP_PROXY`/`HTTPS_PROXY`, unlike
    `urllib.request`), no credential read from the environment."""

    def get(self, request: TransportRequest) -> TransportResponse:
        return self._get(request.url, request, redirect_count=0)

    def _get(self, url: str, request: TransportRequest, *, redirect_count: int) -> TransportResponse:
        parts = _validate_url_static(url, allowed_hosts=request.allowed_hosts)
        host = parts.hostname
        assert host is not None
        port = parts.port or 443
        pinned_ip = _resolve_and_validate_address(host, port)
        context = ssl.create_default_context()
        path = urlunsplit(("", "", parts.path or "/", parts.query, ""))

        conn = _PinnedHTTPSConnection(host, pinned_ip, port=port, connect_timeout=request.connect_timeout, read_timeout=request.read_timeout, context=context)
        try:
            try:
                conn.request("GET", path, headers=dict(request.headers))
                response = conn.getresponse()
            except TimeoutError as exc:
                raise TransportTimeoutError(f"read timeout for {url!r}: {exc}") from exc
            except (http.client.HTTPException, OSError) as exc:
                raise TransportTimeoutError(f"transport failure for {url!r}: {exc}") from exc

            content_encoding = response.getheader("Content-Encoding", "identity")
            if content_encoding.lower() not in ("identity", ""):
                response.close()
                raise ResponseTooLargeError(
                    f"refusing compressed response (Content-Encoding={content_encoding!r}) for {url!r} -- decompression is "
                    "not supported and decompressed size cannot be bounded"
                )

            body = _read_bounded(response, max_bytes=request.max_response_bytes, url=url)
            headers = dict(response.getheaders())
            status_code = response.status
        finally:
            conn.close()

        if status_code in (301, 302, 303, 307, 308):
            return self._follow_redirect(status_code, headers, request, redirect_count=redirect_count, current_url=url)

        return TransportResponse(status_code=status_code, headers=headers, body=body, final_url=url)

    def _follow_redirect(
        self, status_code: int, headers: dict[str, str], request: TransportRequest, *, redirect_count: int, current_url: str,
    ) -> TransportResponse:
        if not request.allow_redirects:
            raise RedirectViolationError(f"received redirect status {status_code} for {current_url!r} but allow_redirects=False")
        if redirect_count >= request.max_redirects:
            raise RedirectViolationError(f"redirect count exceeded max_redirects={request.max_redirects} at {current_url!r}")
        location = headers.get("Location") or headers.get("location")
        if not location:
            raise RedirectViolationError(f"redirect status {status_code} for {current_url!r} carried no Location header")
        target = urljoin(current_url, location)
        target_scheme = urlsplit(target).scheme.lower()
        if target_scheme != _ALLOWED_SCHEME:
            raise RedirectViolationError(f"redirect from {current_url!r} to {target!r} would downgrade scheme to {target_scheme!r}")
        return self._get(target, request, redirect_count=redirect_count + 1)


def _read_bounded(response: http.client.HTTPResponse, *, max_bytes: int, url: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    chunk_size = 65536
    while True:
        try:
            chunk = response.read(chunk_size)
        except TimeoutError as exc:
            raise TransportTimeoutError(f"read timeout while streaming body for {url!r}: {exc}") from exc
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError(f"response for {url!r} exceeded max_response_bytes={max_bytes} while streaming")
        chunks.append(chunk)
    return b"".join(chunks)


class ForbiddenTransport:
    """A `HistoricalHttpTransport` that raises immediately on `.get()` --
    used to PROVE a code path (offline replay, a cached-response test)
    performs zero network calls, rather than merely asserting it by
    inspection. Legitimate production use too: any caller that wants a
    hard, structural guarantee "this operation must not touch the
    network" can inject this instead of a real transport."""

    def get(self, request: TransportRequest) -> TransportResponse:
        raise AssertionError(f"ForbiddenTransport.get() was called for {request.url!r} -- network access is not permitted here")
