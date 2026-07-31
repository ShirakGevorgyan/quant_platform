"""Shared, explicitly-imported test doubles/fixtures for the Phase 4A
`collectors` test suite (Milestone 10) -- NOT a `conftest.py` (this repo
has none anywhere under `tests/`; every existing `test_market_data_*.py`
file is self-contained), so every test file imports what it needs from
here explicitly rather than relying on pytest fixture auto-discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, create_rate_limit_policy
from quant_platform.market_data.collectors.retry import RetryPolicy, create_retry_policy

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class FakeTransport:
    """A deterministic, in-memory `HistoricalHttpTransport` double --
    never touches the network. `responses` is a queue of
    `TransportResponse` (or `Exception` instances) handed back in order,
    one per `.get()` call."""

    responses: list[object] = field(default_factory=list)
    calls: list[TransportRequest] = field(default_factory=list)

    def get(self, request: TransportRequest) -> TransportResponse:
        self.calls.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, TransportResponse)
        return item


def fred_json_body(observations: list[dict]) -> bytes:
    return json.dumps({"observations": observations}).encode("utf-8")


def default_retry_policy() -> RetryPolicy:
    return create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0, 2.0))


def default_rate_limit_policy() -> RateLimitPolicy:
    return create_rate_limit_policy(max_tokens=Decimal(10), refill_rate_per_second=Decimal(1))
