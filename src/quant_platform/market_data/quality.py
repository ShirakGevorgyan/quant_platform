"""Structured data-quality reporting for raw candle rows (Milestone 10,
Phase 1). Reuses `ml.models.ValidationIssue`/`ValidationReport`/
`ValidationSeverity` (the same reporting vocabulary `portfolio_risk.
verification`/`execution_gateway.verification` already use) rather than
inventing a new report shape.

Operates on RAW rows (plain mappings), not already-constructed `Candle`
objects, because `Candle.__post_init__` already rejects an invalid OHLC
relationship/negative volume/non-finite value at construction time --
which is exactly right for a single, already-trusted candle, but wrong
for quality reporting, whose whole point is to examine a BATCH of
untrusted rows and report EVERY problem found rather than stop at the
first bad one (mirrors `historical.quality`'s identical "every check is
independent and additive" philosophy).

Never mutates its input and never repairs anything -- detection only.
`assert_quality_gate` is the explicit, opt-in fail-closed step for a
caller (e.g. `feature_generation.py`) that wants a CRITICAL finding to
raise rather than merely be reported."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import MarketDataError, MarketDataQualityError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.calendar import TradingCalendar, enumerate_expected_open_times
from quant_platform.market_data.candles import Candle
from quant_platform.market_data.identity import parse_decimal, require_tz_aware
from quant_platform.market_data.normalization import normalize_candle_row
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["QUALITY_REPORT_SCHEMA_VERSION", "assert_quality_gate", "run_candle_quality_checks"]

QUALITY_REPORT_SCHEMA_VERSION = 1
_MAX_SAMPLE_TIMESTAMPS = 10


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _classify_numeric(value: object, *, field_name: str) -> tuple[Decimal | None, str | None]:
    """Returns `(parsed_value, None)` on success, or `(None, error_code)`
    where `error_code` is one of `"nan"`/`"infinity"`/`"invalid"` --
    classified BEFORE delegating to `parse_decimal` so a NaN and an
    Infinity get distinct, spec-required issue codes rather than both
    collapsing into one generic "invalid number" message."""
    if isinstance(value, float):
        if math.isnan(value):
            return None, "nan"
        if math.isinf(value):
            return None, "infinity"
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "nan":
            return None, "nan"
        if stripped in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
            return None, "infinity"
    try:
        return parse_decimal(value, field_name=field_name), None
    except MarketDataError:
        return None, "invalid"


def run_candle_quality_checks(
    rows: Sequence[Mapping[str, object]], *, provider: str, symbol: str, timeframe: Timeframe, as_of: datetime,
    calendar: TradingCalendar | None = None, instrument_id: str | None = None,
) -> ValidationReport:
    """Each row is expected to provide `open_time` (tz-aware `datetime`)
    plus `open`/`high`/`low`/`close` (and optionally `volume`), in any
    type `market_data.identity.parse_decimal` accepts (str/int/float/
    Decimal). `as_of` is the caller-supplied reference time future-
    timestamp checks are judged against -- never an internal wall-clock
    read (mirrors this repository's established staleness convention,
    e.g. `StalePriceError`'s docstring)."""
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    generated_at = format_utc_timestamp(pd.Timestamp(as_of))

    valid_candles: list[Candle] = []
    raw_open_times: list[datetime] = []
    for index, row in enumerate(rows):
        open_time = row.get("open_time")
        if not isinstance(open_time, datetime) or open_time.tzinfo is None:
            issues.append(_issue(
                ValidationSeverity.ERROR, "missing_or_invalid_open_time",
                f"Row {index} has a missing or non-timezone-aware open_time: {open_time!r}.",
            ))
            continue
        raw_open_times.append(open_time)

        row_ok = True
        parsed: dict[str, Decimal | None] = {}
        for field_name in ("open", "high", "low", "close"):
            value, error_code = _classify_numeric(row.get(field_name), field_name=field_name)
            if error_code is not None:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, f"{error_code}_value",
                    f"Row {index} (open_time={open_time}) field {field_name!r} is {error_code}: {row.get(field_name)!r}.",
                ))
                row_ok = False
            parsed[field_name] = value
        volume_raw = row.get("volume")
        parsed_volume: Decimal | None = None
        if volume_raw is not None:
            volume_value, volume_error = _classify_numeric(volume_raw, field_name="volume")
            if volume_error is not None:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, f"{volume_error}_value",
                    f"Row {index} (open_time={open_time}) field 'volume' is {volume_error}: {volume_raw!r}.",
                ))
                row_ok = False
            else:
                parsed_volume = volume_value
                if volume_value is not None and volume_value < 0:
                    issues.append(_issue(
                        ValidationSeverity.CRITICAL, "negative_volume",
                        f"Row {index} (open_time={open_time}) has negative volume: {volume_value}.",
                    ))
                    row_ok = False
        if not row_ok:
            continue

        open_, high, low, close = parsed["open"], parsed["high"], parsed["low"], parsed["close"]
        assert open_ is not None and high is not None and low is not None and close is not None
        if high < low or not (low <= open_ <= high) or not (low <= close <= high):
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "invalid_ohlc",
                f"Row {index} (open_time={open_time}) has an invalid OHLC relationship: open={open_} high={high} low={low} close={close}.",
            ))
            continue

        if open_time > as_of:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "future_timestamp",
                f"Row {index} has open_time {open_time} after the as_of reference time {as_of}.",
            ))
            continue

        try:
            candle = normalize_candle_row(
                provider=provider, symbol=symbol, event_time=open_time, timeframe=timeframe, sequence=index, open=open_, high=high,
                low=low, close=close, volume=parsed_volume, instrument_id=instrument_id,
            )
        except MarketDataError as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "candle_construction_failed", f"Row {index} (open_time={open_time}) could not be constructed: {exc}"))
            continue
        valid_candles.append(candle)

    if len(raw_open_times) >= 2:
        for previous, current in itertools.pairwise(raw_open_times):
            if current < previous:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "timestamp_disorder",
                    f"open_time {current} follows {previous} out of order (input rows are not chronologically ordered).",
                ))
                break

    seen_open_times: dict[datetime, int] = {}
    for open_time in raw_open_times:
        seen_open_times[open_time] = seen_open_times.get(open_time, 0) + 1
    duplicate_open_times = sorted(ts for ts, count in seen_open_times.items() if count > 1)
    if duplicate_open_times:
        issues.append(_issue(
            ValidationSeverity.CRITICAL, "duplicate_candle",
            f"{len(duplicate_open_times)} open_time value(s) appear more than once: "
            f"{[format_utc_timestamp(pd.Timestamp(t)) for t in duplicate_open_times[:_MAX_SAMPLE_TIMESTAMPS]]}.",
        ))

    # `Candle.event_id` bakes in `sequence` (see `candles.py`), and every
    # row here is assigned its own unique `sequence=index` a few lines
    # above -- so comparing `event_id` values could never detect a
    # byte-identical repeated row at this pre-sequencing quality-scan
    # stage (that check belongs to `verification.verify_market_event_store`,
    # which runs AFTER real sequence numbers are assigned at store-append
    # time). "duplicate ids" here instead means: two rows whose full
    # OHLCV content is identical at the same open_time -- the row-level
    # analogue of an id collision, detectable before sequencing exists.
    seen_content: dict[tuple[datetime, Decimal, Decimal, Decimal, Decimal, Decimal | None], int] = {}
    for candle in valid_candles:
        content_key = (candle.event_time, candle.open, candle.high, candle.low, candle.close, candle.volume)
        seen_content[content_key] = seen_content.get(content_key, 0) + 1
    duplicate_content_count = sum(1 for count in seen_content.values() if count > 1)
    if duplicate_content_count:
        issues.append(_issue(
            ValidationSeverity.WARNING, "duplicate_id",
            f"{duplicate_content_count} row(s) are byte-identical repeats (same open_time and OHLCV values) -- "
            "harmless idempotent duplicates, not a content conflict.",
        ))

    if calendar is not None and valid_candles:
        sorted_open_times = sorted(c.event_time for c in valid_candles)
        expected = enumerate_expected_open_times(
            calendar, timeframe=timeframe, start=sorted_open_times[0], end=sorted_open_times[-1] + timeframe.duration,
        )
        present = set(sorted_open_times)
        missing = sorted(t for t in expected if t not in present)
        if missing:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "missing_candle",
                f"{len(missing)} expected {timeframe.value} bar(s) are missing per the supplied calendar: "
                f"{[format_utc_timestamp(pd.Timestamp(t)) for t in missing[:_MAX_SAMPLE_TIMESTAMPS]]}.",
            ))

    return ValidationReport(schema_version=QUALITY_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=generated_at)


def assert_quality_gate(report: ValidationReport) -> None:
    if report.criticals:
        codes = sorted({issue.code for issue in report.criticals})
        raise MarketDataQualityError(f"Quality gate failed: {len(report.criticals)} critical issue(s) ({codes}).")
