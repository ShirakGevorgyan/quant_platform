"""Unit tests for `market_data.quality`: every check named in Milestone
10's "Detect:" list -- missing candles, duplicate candles, timestamp
disorder, future timestamps, negative volume, invalid OHLC, NaN,
Infinity, duplicate ids, timeframe gaps -- plus the fail-closed gate."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest

from quant_platform.core.exceptions import MarketDataQualityError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import FixedOffsetTimezone
from quant_platform.market_data.calendar import TradingCalendar, WeeklySession
from quant_platform.market_data.quality import assert_quality_gate, run_candle_quality_checks

_T0 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # Monday
_AS_OF = _T0 + timedelta(days=1)
_UTC_TZ = FixedOffsetTimezone(offset=timedelta(0))


def _row(hour: int, *, open_="2000", high="2005", low="1995", close="2001", volume="10") -> dict[str, object]:
    return {"open_time": _T0 + timedelta(hours=hour), "open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _clean_rows(count: int) -> list[dict[str, object]]:
    return [_row(h) for h in range(count)]


def _codes(report, severity=None) -> set[str]:
    if severity is None:
        return {i.code for i in report.issues}
    return {i.code for i in report.issues if i.severity is severity}


class TestCleanData:
    def test_no_issues_for_well_formed_rows(self) -> None:
        report = run_candle_quality_checks(_clean_rows(10), provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert report.criticals == ()

    def test_assert_quality_gate_does_not_raise_on_a_clean_report(self) -> None:
        report = run_candle_quality_checks(_clean_rows(5), provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert_quality_gate(report)  # must not raise


class TestInvalidOHLC:
    def test_high_less_than_low_is_flagged(self) -> None:
        rows = [_row(0, high="1990", low="1995")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "invalid_ohlc" in _codes(report)


class TestNegativeVolume:
    def test_negative_volume_is_flagged(self) -> None:
        rows = [_row(0, volume="-5")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "negative_volume" in _codes(report)


class TestNaNAndInfinity:
    def test_nan_close_is_flagged(self) -> None:
        rows = [_row(0, close=float("nan"))]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "nan_value" in _codes(report)

    def test_infinity_high_is_flagged(self) -> None:
        rows = [_row(0, high=float("inf"))]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "infinity_value" in _codes(report)

    def test_string_nan_is_flagged(self) -> None:
        rows = [_row(0, open_="NaN")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "nan_value" in _codes(report)


class TestTimestampDisorder:
    def test_out_of_order_rows_are_flagged(self) -> None:
        rows = [_row(2), _row(1), _row(3)]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "timestamp_disorder" in _codes(report)


class TestFutureTimestamps:
    def test_open_time_after_as_of_is_flagged(self) -> None:
        rows = [_row(0), {"open_time": _AS_OF + timedelta(days=10), "open": "2000", "high": "2005", "low": "1995", "close": "2001", "volume": "1"}]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "future_timestamp" in _codes(report)


class TestDuplicateCandles:
    def test_repeated_open_time_is_flagged(self) -> None:
        rows = [_row(0), _row(0, close="2002")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "duplicate_candle" in _codes(report)


class TestDuplicateIds:
    def test_byte_identical_repeated_row_is_flagged_as_a_warning(self) -> None:
        # A repeated row necessarily shares its open_time too, so this
        # also trips `duplicate_candle` (CRITICAL) -- `duplicate_id`'s own
        # distinct signal is that the repeat is a harmless, byte-identical
        # resubmission rather than a genuine content conflict, which is
        # why IT is reported at WARNING, not CRITICAL, severity.
        rows = [_row(0), _row(1), _row(1)]  # last two rows are byte-identical
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        from quant_platform.ml.models import ValidationSeverity

        assert "duplicate_id" in _codes(report, ValidationSeverity.WARNING)
        assert "duplicate_candle" in _codes(report, ValidationSeverity.CRITICAL)

    def test_conflicting_values_at_the_same_timestamp_have_no_duplicate_id_finding(self) -> None:
        # Different content at the same open_time can never share an
        # event_id -- only `duplicate_candle` fires here, never `duplicate_id`.
        rows = [_row(0), _row(0, close="2002")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "duplicate_candle" in _codes(report)
        assert "duplicate_id" not in _codes(report)


class TestMissingCandlesAndTimeframeGaps:
    def _always_open_calendar(self) -> TradingCalendar:
        return TradingCalendar(local_tz=_UTC_TZ, weekly_sessions=(WeeklySession(open_weekday=0, open_time=time(0, 0), close_weekday=6, close_time=time(23, 59, 59)),), name="always_open")

    def test_a_missing_bar_in_the_middle_is_flagged(self) -> None:
        rows = [_row(0), _row(1), _row(3)]  # hour 2 missing
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF, calendar=self._always_open_calendar())
        assert "missing_candle" in _codes(report)

    def test_no_calendar_supplied_skips_the_gap_check(self) -> None:
        rows = [_row(0), _row(1), _row(3)]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "missing_candle" not in _codes(report)

    def test_complete_series_has_no_missing_candle_finding(self) -> None:
        rows = _clean_rows(10)
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF, calendar=self._always_open_calendar())
        assert "missing_candle" not in _codes(report)


class TestFailClosedGate:
    def test_a_critical_finding_raises_via_assert_quality_gate(self) -> None:
        rows = [_row(0, high="1990", low="1995")]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        with pytest.raises(MarketDataQualityError):
            assert_quality_gate(report)


class TestMissingOrInvalidOpenTime:
    def test_missing_open_time_is_flagged_as_an_error(self) -> None:
        rows = [{"open": "2000", "high": "2005", "low": "1995", "close": "2001"}]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "missing_or_invalid_open_time" in _codes(report)

    def test_naive_open_time_is_flagged(self) -> None:
        rows = [{"open_time": datetime(2026, 1, 5), "open": "2000", "high": "2005", "low": "1995", "close": "2001"}]
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert "missing_or_invalid_open_time" in _codes(report)


class TestReportIsAdditiveNotFailFast:
    def test_multiple_independent_issues_are_all_reported(self) -> None:
        rows = [_row(0, volume="-1"), _row(0, close="2002")]  # negative volume AND duplicate open_time
        report = run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        codes = _codes(report)
        assert "negative_volume" in codes
        assert "duplicate_candle" in codes

    def test_report_never_mutates_input_rows(self) -> None:
        rows = _clean_rows(5)
        snapshot = [dict(r) for r in rows]
        run_candle_quality_checks(rows, provider="mt5", symbol="XAUUSD", timeframe=Timeframe.H1, as_of=_AS_OF)
        assert rows == snapshot
