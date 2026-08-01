"""Shared, explicitly-imported test doubles/fixtures for the Phase 4B
curated-universe test suite (Milestone 10) -- mirrors `_collectors_
test_helpers.py`'s own "no conftest.py, explicit imports" convention."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.curated.availability import (
    AvailabilityPolicyKind,
    create_availability_policy,
)
from quant_platform.market_data.collectors.curated.registry import (
    CuratedFredRegistry,
    create_curated_registry,
    default_core_series_specs,
)
from quant_platform.market_data.collectors.curated.revision_policy import (
    RevisionPolicyKind,
    create_revision_policy,
)
from quant_platform.market_data.collectors.rate_limit import create_rate_limit_policy
from quant_platform.market_data.collectors.retry import create_retry_policy
from quant_platform.market_data.repository import MarketDataRepository

T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
OBS_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
OBS_END = datetime(2024, 3, 1, tzinfo=timezone.utc)


def fresh_repository_and_cache() -> tuple[Path, MarketDataRepository, RawResponseCache]:
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def default_daily_policy(hour: int = 17, minute: int = 0):
    return create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="America/New_York", availability_hour=hour, availability_minute=minute)


def default_monthly_policy(hour: int = 8, minute: int = 30):
    return create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="America/New_York", availability_hour=hour, availability_minute=minute)


def default_availability_policies() -> dict[str, object]:
    daily = default_daily_policy()
    monthly = default_monthly_policy()
    return {"DFII10": daily, "DGS10": daily, "DFF": daily, "CPIAUCSL": monthly}


def default_revision_policy():
    return create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE)


def default_core_registry(observation_start: datetime = OBS_START) -> CuratedFredRegistry:
    specs = default_core_series_specs(
        registry_version=1, revision_policy_id="a" * 64, release_availability_policy_id_daily="b" * 64,
        release_availability_policy_id_monthly="c" * 64, default_observation_start=observation_start,
    )
    return create_curated_registry(registry_version=1, specs=specs)


def default_retry_policy():
    return create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0, 2.0))


def default_rate_limit_policy():
    return create_rate_limit_policy(max_tokens=Decimal(20), refill_rate_per_second=Decimal(5))


def metadata_body(series_id: str, freq: str, freq_short: str, units: str, units_short: str, sa: str, sa_short: str) -> bytes:
    return json.dumps({
        "realtime_start": "2024-06-01", "realtime_end": "2024-06-01",
        "seriess": [{
            "id": series_id, "realtime_start": "2024-06-01", "realtime_end": "2024-06-01", "title": f"Title for {series_id}",
            "observation_start": "1962-01-02", "observation_end": "2024-06-01", "frequency": freq, "frequency_short": freq_short,
            "units": units, "units_short": units_short, "seasonal_adjustment": sa, "seasonal_adjustment_short": sa_short,
            "last_updated": "2024-06-01 10:00:00-05",
        }],
    }).encode()


def observations_body(rows: list[dict]) -> bytes:
    return json.dumps({"observations": rows}).encode()


CORE_METADATA_BODIES = {
    "DFII10": metadata_body("DFII10", "Daily", "D", "Percent", "%", "Not Seasonally Adjusted", "NSA"),
    "DGS10": metadata_body("DGS10", "Daily", "D", "Percent", "%", "Not Seasonally Adjusted", "NSA"),
    "DFF": metadata_body("DFF", "Daily", "D", "Percent", "%", "Not Seasonally Adjusted", "NSA"),
    "CPIAUCSL": metadata_body("CPIAUCSL", "Monthly", "M", "Index 1982-1984=100", "Index 1982-1984=100", "Seasonally Adjusted", "SA"),
}
