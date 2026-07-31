"""Deterministic, locale-independent normalization of RAW SOURCE TEXT
(Milestone 10, Phase 3) -- the boundary between untyped strings read from
a CSV/JSON Lines file and this package's own strict, typed domain
values. Distinct from Phase 1's `normalization.py` (which tolerates
already-typed `str`/`int`/`float`/`Decimal` VALUES from a programmatic
caller): this module parses raw TEXT, a stricter concern -- a source
file's own timestamp/number FORMAT must be explicitly declared, never
guessed via locale-dependent auto-detection (`datetime.fromisoformat`'s
own permissiveness, or `Decimal`'s tolerance for a bare thousands-comma
in some locales, are exactly the kind of implicit format-guessing this
module refuses to do)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import HistoricalIngestionError, MarketDataError, TimezoneError
from quant_platform.historical.timezones import SourceTimezone, localize_broker_timestamps
from quant_platform.market_data.identity import parse_decimal

__all__ = ["TimestampParsingPolicy", "normalize_signed_zero", "normalize_volume", "parse_source_decimal", "parse_source_timestamp"]


@dataclass(frozen=True, slots=True)
class TimestampParsingPolicy:
    formats: tuple[str, ...]
    """Explicit `datetime.strptime` format strings, tried in order -- the
    FIRST one that parses the raw text successfully is used. Never
    empty: an ingestion operation must explicitly declare which
    format(s) its source uses (see module docstring)."""
    source_timezone: SourceTimezone | None
    """`None` means every format in `formats` is expected to capture an
    explicit UTC offset itself (e.g. `"%Y-%m-%dT%H:%M:%S%z"`) -- the
    parsed value is then already unambiguous and is used as-is (converted
    to UTC). A non-`None` value means `formats` parse a NAIVE wall-clock
    string, and this timezone is applied via `historical.timezones.
    localize_broker_timestamps` (DST-ambiguous/nonexistent rejection
    included -- see that module's own docstring)."""

    def __post_init__(self) -> None:
        if not self.formats:
            raise HistoricalIngestionError("TimestampParsingPolicy.formats must not be empty -- locale-dependent auto-parsing is forbidden")


def parse_source_timestamp(raw: str, *, policy: TimestampParsingPolicy, field_name: str = "timestamp") -> datetime:
    if not raw:
        raise HistoricalIngestionError(f"{field_name} must not be empty")
    parsed: datetime | None = None
    for fmt in policy.formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        raise HistoricalIngestionError(f"{field_name}={raw!r} does not match any of the configured formats: {list(policy.formats)}")

    if parsed.tzinfo is not None:
        # Already carries an explicit offset -- convert directly, no
        # ambiguity possible, `source_timezone` is irrelevant here.
        return pd.Timestamp(parsed).tz_convert("UTC").to_pydatetime()

    if policy.source_timezone is None:
        raise TimezoneError(
            f"{field_name}={raw!r} parsed as a naive (timezone-less) timestamp, but no source_timezone policy was "
            "configured -- a naive timestamp is never silently assumed to be UTC in this package."
        )
    localized = localize_broker_timestamps(pd.Series([pd.Timestamp(parsed)]), policy.source_timezone)
    result: datetime = localized.iloc[0].to_pydatetime()
    return result


def normalize_signed_zero(value: Decimal) -> Decimal:
    """`Decimal("-0")` and `Decimal("0")` compare equal but serialize to
    DIFFERENT strings (`str(Decimal("-0")) == "-0"`), which would
    silently produce a different content digest for economically
    identical data -- e.g. a source reporting a flat `price_delta` as
    `"-0.00"`. Every source-parsed Decimal is normalized to canonical
    (non-negative-zero) form."""
    return Decimal(0) if value == 0 else value


def parse_source_decimal(raw: str, *, field_name: str) -> Decimal:
    """Rejects a locale-dependent thousands-separator comma outright
    (the source schema must pre-normalize it, never guessed here) and
    never routes through `float`. Every failure mode -- malformed text,
    NaN/Infinity, a comma -- raises `HistoricalIngestionError` (never the
    more generic `identity.parse_decimal`'s own `MarketDataError`): a
    caller of THIS module's own public functions gets one consistent
    exception type for "this source text boundary rejected the value",
    regardless of which underlying check caught it."""
    stripped = raw.strip()
    if "," in stripped:
        raise HistoricalIngestionError(f"{field_name}={raw!r} contains a comma -- locale-dependent thousands separators are not supported; the source schema must pre-normalize them")
    try:
        value = parse_decimal(stripped, field_name=field_name)
    except MarketDataError as exc:
        raise HistoricalIngestionError(str(exc)) from exc
    return normalize_signed_zero(value)


def normalize_volume(raw: str, *, unit_scale: Decimal = Decimal(1), field_name: str = "volume") -> Decimal:
    value = parse_source_decimal(raw, field_name=field_name)
    if value < 0:
        raise HistoricalIngestionError(f"{field_name} must be >= 0, got {value}")
    return normalize_signed_zero(value * unit_scale)
