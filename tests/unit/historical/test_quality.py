"""Tests for `historical.quality.run_quality_checks`."""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.calendar import TradingCalendar, WeeklySession
from quant_platform.historical.quality import (
    IssueType,
    QualityThresholds,
    Severity,
    run_quality_checks,
)
from quant_platform.historical.timezones import FixedOffsetTimezone

UTC0 = FixedOffsetTimezone(timedelta(0), name="UTC0")


def _frame(n: int = 30, start: str = "2024-01-03T00:00:00", seed: int = 3) -> pd.DataFrame:
    """Realistic (non-constant-price) synthetic bars in the raw historical
    schema shape, reusing the existing M1 synthetic generator rather than
    hand-rolling a second random-walk implementation."""
    sd = generate_ohlcv(
        SyntheticDataConfig(
            start=datetime.fromisoformat(start).replace(tzinfo=dt_timezone.utc),
            periods=n, timeframe=Timeframe.M1, seed=seed,
        )
    )
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["real_volume"] = 0
    sd["spread"] = 15
    return sd


def _no_break_calendar() -> TradingCalendar:
    """A weekly-session-only calendar with NO daily maintenance break, so
    tests isolating weekly-gap classification aren't also tripped by a
    maintenance-break window that happens to coincide with the reopen
    instant (an artifact of the illustrative default calendar's own
    example parameters, not something these tests need to exercise)."""
    from datetime import time

    return TradingCalendar(
        local_tz=UTC0,
        weekly_sessions=(WeeklySession(open_weekday=6, open_time=time(23, 0), close_weekday=4, close_time=time(23, 0)),),
        name="no_break_test_calendar",
    )


class TestCleanData:
    def test_clean_synthetic_data_has_no_issues(self) -> None:
        report = run_quality_checks(_frame(), symbol="XAUUSD", timeframe=Timeframe.M1)
        assert report.is_valid
        assert report.issues == []


class TestSchemaAndInvariantChecks:
    def test_missing_column_is_critical(self) -> None:
        df = _frame().drop(columns=["spread"])
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert not report.is_valid
        assert report.critical_issues[0].issue_type is IssueType.MISSING_COLUMN

    def test_null_price_is_critical(self) -> None:
        df = _frame()
        df.loc[3, "close"] = np.nan
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.NULL_VALUE for i in report.critical_issues)

    def test_non_finite_value_is_critical(self) -> None:
        df = _frame()
        df.loc[3, "close"] = np.inf
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.NON_FINITE_VALUE for i in report.critical_issues)

    def test_non_positive_price_is_critical(self) -> None:
        df = _frame()
        df.loc[0, "close"] = 0.0
        df.loc[0, "low"] = 0.0
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.NON_POSITIVE_PRICE for i in report.critical_issues)

    def test_high_less_than_low_is_critical(self) -> None:
        df = _frame()
        df.loc[5, "high"] = df.loc[5, "low"] - 1.0
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        types = {i.issue_type for i in report.critical_issues}
        assert IssueType.HIGH_LESS_THAN_LOW in types

    def test_open_outside_range_is_critical(self) -> None:
        df = _frame()
        df.loc[3, "open"] = df.loc[3, "high"] + 5.0
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.OPEN_OUTSIDE_RANGE for i in report.critical_issues)

    def test_close_outside_range_is_critical(self) -> None:
        df = _frame()
        df.loc[3, "close"] = df.loc[3, "low"] - 5.0
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.CLOSE_OUTSIDE_RANGE for i in report.critical_issues)

    def test_negative_volume_is_critical(self) -> None:
        df = _frame()
        df.loc[0, "tick_volume"] = -1
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.NEGATIVE_VOLUME for i in report.critical_issues)

    def test_negative_spread_is_critical(self) -> None:
        df = _frame()
        df.loc[0, "spread"] = -1
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.NEGATIVE_SPREAD for i in report.critical_issues)

    def test_duplicate_timestamp_is_critical(self) -> None:
        df = _frame()
        df.loc[1, "open_time"] = df.loc[0, "open_time"]
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.DUPLICATE_TIMESTAMP for i in report.critical_issues)

    def test_unordered_timestamp_is_critical(self) -> None:
        df = _frame()
        df.loc[2, "open_time"] = df.loc[0, "open_time"] - timedelta(minutes=1)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.UNORDERED_TIMESTAMP for i in report.critical_issues)

    def test_misaligned_timestamp_is_critical(self) -> None:
        df = _frame()
        df.loc[3, "open_time"] = df.loc[3, "open_time"] + pd.Timedelta(seconds=30)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.MISALIGNED_TIMESTAMP for i in report.critical_issues)

    def test_naive_open_time_is_critical(self) -> None:
        df = _frame()
        df["open_time"] = df["open_time"].dt.tz_localize(None)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.MIXED_TZ_AWARENESS for i in report.critical_issues)


class TestTemporalChecks:
    def test_overlapping_bars_is_critical(self) -> None:
        df = _frame(10)
        df.loc[5, "open_time"] = df.loc[4, "open_time"] + timedelta(seconds=30)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.OVERLAPPING_BARS for i in report.critical_issues)

    def test_gap_without_calendar_is_a_warning(self) -> None:
        before = _frame(5, "2024-01-03T00:00:00")
        after = _frame(5, "2024-01-03T01:00:00")
        df = pd.concat([before, after]).reset_index(drop=True)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        gap_issues = [i for i in report.issues if i.issue_type is IssueType.UNEXPECTED_SESSION_GAP]
        assert len(gap_issues) == 1
        assert gap_issues[0].severity is Severity.WARNING

    def test_expected_weekend_gap_with_calendar_is_info_not_warning(self) -> None:
        calendar = _no_break_calendar()
        before = _frame(5, "2024-01-05T22:55:00")  # Friday, ends just before close
        after = _frame(5, "2024-01-07T23:00:00")   # Sunday, exact reopen
        df = pd.concat([before, after]).reset_index(drop=True)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1, calendar=calendar)
        assert report.is_valid
        assert [i.issue_type for i in report.issues] == [IssueType.MISSING_BARS_GAP]
        assert report.issues[0].severity is Severity.INFO

    def test_unexpected_closed_session_bar_is_flagged(self) -> None:
        calendar = _no_break_calendar()
        # A bar sitting squarely inside Saturday (fully closed all day).
        df = _frame(5, "2024-01-06T12:00:00")
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1, calendar=calendar)
        assert any(i.issue_type is IssueType.UNEXPECTED_CLOSED_SESSION_BAR for i in report.issues)


class TestMarketQualityChecks:
    def test_impossible_price_jump_is_critical(self) -> None:
        df = _frame()
        # Scale the ENTIRE bar (not just close) so it stays internally
        # OHLC-valid (high >= open/close >= low) while still creating a
        # genuine cross-bar jump relative to bar 9's close -- scaling only
        # `close` would instead (and did, in an earlier version of this
        # test) trip CLOSE_OUTSIDE_RANGE first and mask the jump check.
        df.loc[10, ["open", "high", "low", "close"]] *= 2
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.IMPOSSIBLE_PRICE_JUMP for i in report.critical_issues)

    def test_impossible_price_jump_threshold_is_configurable(self) -> None:
        df = _frame()
        df.loc[10, ["open", "high", "low", "close"]] *= 1.10  # a 10% jump
        lenient = run_quality_checks(
            df, symbol="XAUUSD", timeframe=Timeframe.M1, thresholds=QualityThresholds(max_price_jump_fraction=0.5)
        )
        strict = run_quality_checks(
            df, symbol="XAUUSD", timeframe=Timeframe.M1, thresholds=QualityThresholds(max_price_jump_fraction=0.01)
        )
        assert not any(i.issue_type is IssueType.IMPOSSIBLE_PRICE_JUMP for i in lenient.issues)
        assert any(i.issue_type is IssueType.IMPOSSIBLE_PRICE_JUMP for i in strict.issues)

    def test_frozen_price_sequence_is_flagged(self) -> None:
        df = _frame()
        frozen_values = df.loc[5, ["open", "high", "low", "close"]].to_numpy()
        df.loc[5:12, ["open", "high", "low", "close"]] = frozen_values
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        frozen_issues = [i for i in report.issues if i.issue_type is IssueType.FROZEN_PRICE_SEQUENCE]
        assert len(frozen_issues) == 1
        assert frozen_issues[0].affected_row_count == 8

    def test_extreme_spread_is_flagged(self) -> None:
        df = _frame(40)
        df.loc[20, "spread"] = 10_000
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.EXTREME_SPREAD for i in report.issues)

    def test_volume_spike_is_flagged(self) -> None:
        df = _frame(40)
        df.loc[20, "tick_volume"] = int(df["tick_volume"].median() * 50)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.VOLUME_SPIKE for i in report.issues)

    def test_incomplete_edge_bar_is_flagged(self) -> None:
        df = _frame()
        df.loc[0, ["open", "high", "low", "close"]] = 2000.0
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert any(i.issue_type is IssueType.INCOMPLETE_EDGE_BAR for i in report.issues)

    def test_batch_boundary_artifact_is_flagged(self) -> None:
        df = _frame(10)
        # Introduce a 1-bar gap exactly at the declared boundary (the bar
        # that WOULD sit there is simply missing, as if two paginated
        # fetches were stitched together one bar apart) -- non-critical
        # (an ordinary gap), so the market-quality check block still runs
        # and can additionally flag it as sitting at a known seam.
        boundary_time = df["open_time"].iloc[5] + timedelta(minutes=1)
        df.loc[5:, "open_time"] = df.loc[5:, "open_time"] + timedelta(minutes=1)
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1, batch_boundaries=(boundary_time,))
        assert report.is_valid  # a single-bar gap alone is not critical
        assert any(i.issue_type is IssueType.BATCH_BOUNDARY_ARTIFACT for i in report.issues)


class TestAffectedRowIndices:
    def test_indices_are_complete_and_positional_not_just_the_capped_sample(self) -> None:
        df = _frame(20)
        df.loc[3, "tick_volume"] = -5
        df.loc[9, "tick_volume"] = -5
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        issue = next(i for i in report.critical_issues if i.issue_type is IssueType.NEGATIVE_VOLUME)
        assert issue.affected_row_indices == (3, 9)
        assert issue.affected_row_count == 2


class TestReportSummary:
    def test_summary_lists_each_issue(self) -> None:
        df = _frame()
        df.loc[0, "tick_volume"] = -1
        report = run_quality_checks(df, symbol="XAUUSD", timeframe=Timeframe.M1)
        text = report.summary()
        assert "XAUUSD" in text
        assert "NEGATIVE_VOLUME" in text
        assert "valid=False" in text


class TestQualityThresholdsValidation:
    def test_rejects_non_positive_jump_fraction(self) -> None:
        with pytest.raises(ValueError, match="max_price_jump_fraction"):
            QualityThresholds(max_price_jump_fraction=0.0)

    def test_rejects_frozen_sequence_length_below_two(self) -> None:
        with pytest.raises(ValueError, match="frozen_sequence_min_length"):
            QualityThresholds(frozen_sequence_min_length=1)
