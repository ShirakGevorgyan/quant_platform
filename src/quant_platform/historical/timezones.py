"""Timezone normalization policy for the historical data ingestion pipeline.

THIS IS THE SINGLE MOST SAFETY-CRITICAL MODULE IN THE HISTORICAL DATA
PIPELINE. Milestone 1's `TimeframeCursor` already rejects naive timestamps
for exactly this reason (see `multiframe/cursor.py`): two series that are
each individually "monotonic" but disagree about which real-world instant
their timestamps represent can silently desynchronize a cross-timeframe
clock and leak future data into a signal evaluation. That risk is *larger*
here, one layer upstream, because broker historical-data APIs (MT5's
`copy_rates_range` foremost among them) hand back naive wall-clock
timestamps in the *broker/trade server's* local time -- which is routinely
neither UTC nor the timezone of the machine running this code -- with
absolutely nothing in the returned data to say so.

Policy enforced by this module:

  * Canonical timestamps are always tz-aware UTC. Nothing downstream of
    `localize_broker_timestamps` ever sees a naive timestamp.
  * A naive timestamp is never silently assumed to be UTC (contrast with
    `core.time_utils.ensure_utc`, which deliberately *does* assume naive
    means UTC -- that function exists for internal/synthetic data that this
    platform itself produced and already knows to be UTC; it must never be
    used on broker-sourced timestamps, which is precisely why this separate,
    stricter module exists for the ingestion boundary).
  * The source timezone is always an explicit, configured value: either a
    fixed UTC offset (most MT5 brokers run a fixed-offset "server time" with
    no DST, e.g. UTC+2/UTC+3 for the trading-session convention some brokers
    use) or a standards-based IANA zone via `zoneinfo` (for the less common
    case of a broker/vendor that reports true local time in a named zone).
  * DST-ambiguous wall-clock times (an hour repeated at a "fall back"
    transition) and DST-nonexistent wall-clock times (an hour skipped at a
    "spring forward" transition) are rejected outright rather than guessed.
    This platform has no way to recover the correct UTC instant for an
    ambiguous/nonexistent broker timestamp from the bar data alone (MT5's
    historical API provides no DST-fold disambiguation metadata), so
    "reject and let the operator resolve it" is the only honest option --
    silently picking one of the two candidate instants (as a naive `fold=0`
    default would) is exactly the kind of silent timezone assumption this
    module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from datetime import tzinfo as _TzInfo  # noqa: N812
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from quant_platform.core.exceptions import TimezoneError

UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class FixedOffsetTimezone:
    """A fixed UTC offset with no DST transitions -- the common case for
    MT5 broker "server time" (e.g. a broker publishing candles on a
    constant UTC+2 or UTC+3 server clock year-round, or one that shifts
    between two fixed offsets on its own non-IANA schedule, in which case
    model each side of that shift as a separate `FixedOffsetTimezone` valid
    over its own date range at the config layer -- this class itself makes
    no DST-transition decision, which is the point: a fixed offset has none
    to make, and is therefore unambiguous for every wall-clock timestamp."""

    offset: timedelta
    name: str = "FIXED"

    def __post_init__(self) -> None:
        if abs(self.offset) >= timedelta(hours=24):
            raise ValueError(f"offset must be within +/-24h, got {self.offset}")

    def to_tzinfo(self) -> _TzInfo:
        return timezone(self.offset, self.name)

    def __str__(self) -> str:
        total_minutes = int(self.offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        h, m = divmod(abs(total_minutes), 60)
        return f"{self.name}(UTC{sign}{h:02d}:{m:02d})"


@dataclass(frozen=True, slots=True)
class NamedZoneTimezone:
    """A standards-based IANA timezone (e.g. 'Europe/Berlin'), resolved via
    the stdlib `zoneinfo` module. Requires the `tzdata` package to be
    installed on platforms (notably Windows) that do not ship an IANA
    timezone database with the OS."""

    key: str

    def to_tzinfo(self) -> _TzInfo:
        try:
            return ZoneInfo(self.key)
        except ZoneInfoNotFoundError as exc:
            raise TimezoneError(
                f"Unknown or unavailable IANA timezone key: {self.key!r}. On Windows this "
                "usually means the 'tzdata' package is not installed.",
                context={"zone_key": self.key},
            ) from exc

    def __str__(self) -> str:
        return f"NamedZone({self.key})"


SourceTimezone = FixedOffsetTimezone | NamedZoneTimezone


def _is_tz_aware(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) and getattr(series.dt, "tz", None) is not None


def localize_broker_timestamps(naive_timestamps: pd.Series, source_tz: SourceTimezone) -> pd.Series:
    """Convert a series of NAIVE broker wall-clock timestamps to tz-aware
    UTC, using the explicitly configured `source_tz`.

    Rejects (does not silently accept):
      * an already tz-aware input series (localization must happen exactly
        once, at this boundary; a caller passing an already-aware series
        indicates a bug upstream, not a value to pass through untouched)
      * any wall-clock time that is ambiguous (DST fall-back) or
        nonexistent (DST spring-forward) under `source_tz` -- see module
        docstring for why this cannot be safely guessed.
    """
    if not pd.api.types.is_datetime64_any_dtype(naive_timestamps):
        raise TimezoneError(
            f"Expected a datetime64 series of broker timestamps, got dtype {naive_timestamps.dtype}"
        )
    if _is_tz_aware(naive_timestamps):
        raise TimezoneError(
            "Expected timezone-naive broker wall-clock timestamps but received an "
            "already tz-aware series. Localization must be applied exactly once, at "
            "the raw source boundary -- localizing twice (or localizing data that was "
            "already correctly converted) silently shifts every timestamp."
        )

    tzinfo_obj = source_tz.to_tzinfo()
    try:
        localized = naive_timestamps.dt.tz_localize(tzinfo_obj, ambiguous="raise", nonexistent="raise")
    except Exception as exc:
        raise TimezoneError(
            f"Cannot unambiguously localize broker timestamps to {source_tz}: {exc}. "
            "This series contains a wall-clock time that is either ambiguous (repeated "
            "during a DST fall-back transition) or nonexistent (skipped during a DST "
            "spring-forward transition) under the configured source timezone. This "
            "pipeline has no metadata from the broker to disambiguate such a time, so "
            "it is rejected rather than guessed; a fixed-offset source timezone (the "
            "common case for MT5 broker server time) never raises this because a fixed "
            "offset has no DST transitions.",
            context={"source_tz": str(source_tz)},
        ) from exc

    result: pd.Series = localized.dt.tz_convert("UTC")
    return result


def require_utc(timestamps: pd.Series, *, context: str) -> None:
    """Raise `TimezoneError` unless `timestamps` is already tz-aware UTC.

    This is the enforcement point used throughout the rest of the pipeline
    (raw store, validation, resampling, canonical storage, loader) -- none
    of those stages infer or silently localize a timezone; they all require
    it to have already happened, at the single boundary
    (`localize_broker_timestamps`) where the source timezone is known.
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise TimezoneError(f"{context}: expected a datetime64 series, got dtype {timestamps.dtype}")
    tz = timestamps.dt.tz
    if tz is None:
        raise TimezoneError(
            f"{context}: timestamps are timezone-naive. Naive timestamps are ambiguous "
            "about which real-world instant they represent and are never silently "
            "assumed to be UTC in the historical data pipeline; localize explicitly via "
            "`localize_broker_timestamps` at the source boundary first."
        )
    # Normalize both sides of the comparison through a UTC instant rather
    # than a string/identity check, so semantically-UTC-equivalent tzinfo
    # objects (datetime.timezone.utc vs zoneinfo.ZoneInfo("UTC") vs a
    # FixedOffsetTimezone(timedelta(0))) are all accepted -- rejecting any
    # of them merely because their tzinfo object type differs from
    # whichever one a caller happened to construct would be an arbitrary,
    # unjustified restriction unrelated to the actual correctness property
    # this function exists to guarantee (that the instant is unambiguous
    # and already expressed in UTC).
    sample = timestamps.iloc[0]
    if sample.utcoffset() != timedelta(0):
        raise TimezoneError(
            f"{context}: timestamps must be normalized to UTC (found offset "
            f"{sample.utcoffset()} from tz {tz}); convert explicitly via `.dt.tz_convert('UTC')` first."
        )


__all__ = [
    "UTC",
    "FixedOffsetTimezone",
    "NamedZoneTimezone",
    "SourceTimezone",
    "localize_broker_timestamps",
    "require_utc",
]
