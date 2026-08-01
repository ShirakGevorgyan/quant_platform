"""Pure raw-to-canonical market-bar normalization (Milestone 10, Phase
4C) -- the provider-neutral layer every concrete adapter's parsed
`RawMarketRecord`s flow through on the way to a durable `MarketDriverBar`.
Mirrors `curated.orchestration._normalize_curated_row`'s own discipline:
a pure row processor, reused UNCHANGED by both orchestration and
independent verification, never trusting a cached parse."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicy,
    resolve_bar_availability_time,
)
from quant_platform.market_data.collectors.cross_asset.futures import RollProvenance
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_record import (
    MarketDriverBar,
    RawMarketRecord,
    create_market_driver_bar,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    TimezoneSessionPolicy,
)
from quant_platform.market_data.source_normalization import parse_source_decimal

__all__ = ["INVALID_MARKET_RECORD", "MISSING_MARKET_VOLUME", "normalize_raw_market_record", "resolve_bar_open_time"]

INVALID_MARKET_RECORD = "invalid_market_record"
MISSING_MARKET_VOLUME = "missing_market_volume"


def resolve_bar_open_time(date_text: str, *, session_policy: TimezoneSessionPolicy, timeframe: Timeframe) -> datetime:
    """PURE. Interprets a provider's own daily-bar date text as this
    bar's OPEN time, honoring the session policy's declared timestamp
    convention -- never assumes a generic midnight-UTC open regardless
    of the actual session (see module docstring's own point-in-time
    discipline)."""
    calendar_date = date.fromisoformat(date_text)
    tz = ZoneInfo(session_policy.timezone_key)
    if session_policy.is_24_hour_session:
        local_dt = datetime.combine(calendar_date, time(0, 0), tzinfo=tz)
        return local_dt.astimezone(timezone.utc)
    assert session_policy.session_open_time is not None and session_policy.session_close_time is not None
    if session_policy.timestamp_convention is CandleTimestampConvention.OPEN_LABELED:
        local_open = datetime.combine(calendar_date, session_policy.session_open_time, tzinfo=tz)
        return local_open.astimezone(timezone.utc)
    local_close = datetime.combine(calendar_date, session_policy.session_close_time, tzinfo=tz)
    return local_close.astimezone(timezone.utc) - timeframe.duration


def normalize_raw_market_record(
    raw: RawMarketRecord, *, canonical_driver_id: str, instrument_form: InstrumentForm, timeframe: Timeframe,
    session_policy: TimezoneSessionPolicy, availability_policy: BarAvailabilityPolicy, adjustment_policy_id: str, request_manifest_id: str,
    response_manifest_id: str, source_manifest_id: str, source_row_index: int, contract_metadata_id: str | None = None,
    roll_provenance: RollProvenance | None = None,
) -> tuple[MarketDriverBar | None, tuple[str, ...]]:
    """PURE. Returns `(bar_or_none, quarantine_issue_codes)` -- never
    raises for an ordinary malformed row (quarantines instead), matching
    `curated._normalize_curated_row`'s own contract."""
    try:
        open_ = parse_source_decimal(raw.open_text, field_name="open")
        high = parse_source_decimal(raw.high_text, field_name="high")
        low = parse_source_decimal(raw.low_text, field_name="low")
        close = parse_source_decimal(raw.close_text, field_name="close")
    except Exception:
        return None, (INVALID_MARKET_RECORD,)

    volume: Decimal | None
    if raw.volume_text is None:
        volume = None
    else:
        try:
            volume = parse_source_decimal(raw.volume_text, field_name="volume")
        except Exception:
            return None, (MISSING_MARKET_VOLUME,)

    try:
        open_time = resolve_bar_open_time(raw.provider_timestamp_text, session_policy=session_policy, timeframe=timeframe)
    except Exception:
        return None, (INVALID_MARKET_RECORD,)

    close_time = compute_close_time(open_time, timeframe)
    try:
        availability_time = resolve_bar_availability_time(availability_policy, bar_close_time=close_time)
    except Exception:
        return None, (INVALID_MARKET_RECORD,)

    try:
        bar = create_market_driver_bar(
            canonical_driver_id=canonical_driver_id, provider=raw.provider, provider_symbol=raw.provider_symbol, instrument_form=instrument_form,
            open_time=open_time, timeframe=timeframe, open=open_, high=high, low=low, close=close, volume=volume, volume_unit="native",
            availability_time=availability_time, availability_policy_id=availability_policy.availability_policy_id,
            session_policy_id=session_policy.session_policy_id, adjustment_policy_id=adjustment_policy_id, request_manifest_id=request_manifest_id,
            response_manifest_id=response_manifest_id, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
            contract_metadata_id=contract_metadata_id, roll_provenance=roll_provenance,
        )
    except Exception:
        return None, (INVALID_MARKET_RECORD,)
    return bar, ()
