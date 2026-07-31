"""Minimal historical HTTP transport protocol (Milestone 10, Phase 4A).

`HistoricalHttpTransport` is a `Protocol` (structural typing, mirroring
`market_data.adapters.HistoricalSourceAdapter`'s own convention) -- every
collector depends on THIS shape, never on a concrete HTTP library
directly, so a test can inject a fully deterministic fake transport (no
network, no timing dependency) without any monkeypatching of a real
library's internals, and so swapping the underlying implementation later
never touches collector logic.

`TransportRequest`/`TransportResponse` are plain, ephemeral, in-memory
value objects for ONE call -- neither is ever serialized directly into a
durable artifact (that is `request_manifest.py`/`response_manifest.py`'s
own job, and each is deliberately a NARROWER, secret-free view of the
same call). A `TransportRequest.headers`/its URL's query string MAY
legitimately carry a real credential for the single in-flight call; nothing
in this module ever logs, prints, or persists a `TransportRequest`/
`TransportResponse` object as a whole."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.identity import require_non_empty, require_tz_aware

__all__ = ["HistoricalHttpTransport", "TransportRequest", "TransportResponse"]


@dataclass(frozen=True, slots=True)
class TransportRequest:
    url: str
    """The full URL INCLUDING query string, already encoded -- e.g.
    `https://api.stlouisfed.org/fred/series/observations?series_id=...`.
    May legitimately embed a real secret (`api_key=...`) for this one
    in-flight call; the request MANIFEST records a secret-free canonical
    view separately."""
    headers: dict[str, str] = field(default_factory=dict)
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_response_bytes: int = 10_000_000
    allow_redirects: bool = False
    max_redirects: int = 0
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    request_time: datetime | None = None
    """Caller-supplied, for manifest/logging correlation ONLY -- the
    transport itself must never read the wall clock to decide anything
    about the request it sends."""

    def __post_init__(self) -> None:
        require_non_empty(self.url, field_name="TransportRequest.url")
        if self.connect_timeout <= 0:
            raise CollectorError(f"TransportRequest.connect_timeout must be > 0, got {self.connect_timeout}")
        if self.read_timeout <= 0:
            raise CollectorError(f"TransportRequest.read_timeout must be > 0, got {self.read_timeout}")
        if self.max_response_bytes <= 0:
            raise CollectorError(f"TransportRequest.max_response_bytes must be > 0, got {self.max_response_bytes}")
        if self.max_redirects < 0:
            raise CollectorError(f"TransportRequest.max_redirects must be >= 0, got {self.max_redirects}")
        if not self.allowed_hosts:
            raise CollectorError("TransportRequest.allowed_hosts must not be empty -- an explicit host allowlist is mandatory")
        if self.request_time is not None:
            require_tz_aware(self.request_time, field_name="TransportRequest.request_time")


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    final_url: str
    """The URL the body actually came from -- identical to
    `TransportRequest.url` unless a redirect was followed."""
    attempt_count: int = 1


class HistoricalHttpTransport(Protocol):
    """Structural contract every collector depends on. An implementation
    MUST: perform HTTPS GET only; validate `request.url`'s host against
    `request.allowed_hosts` (and every redirect hop's host again, if
    `allow_redirects`) before connecting; enforce `connect_timeout`/
    `read_timeout` separately; enforce `max_response_bytes` incrementally
    while reading (never buffer an unbounded body first); raise
    `DisallowedUrlError`/`SsrfTargetError`/`RedirectViolationError`/
    `TransportTimeoutError`/`ResponseTooLargeError` (never a bare
    `Exception`) for the corresponding failure; carry no implicit global
    session state, cookies, proxy inheritance, or environment-derived
    credentials between calls."""

    def get(self, request: TransportRequest) -> TransportResponse: ...
