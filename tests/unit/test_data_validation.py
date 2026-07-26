"""Tests for OHLCV data-quality validation and gap detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import DataQualityError
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.data.validation import detect_gaps, detect_overlaps, validate_ohlcv

UTC = timezone.utc


def _valid_frame(periods: int = 50) -> pd.DataFrame:
    return generate_ohlcv(
        SyntheticDataConfig(
            start=datetime(2024, 1, 1, tzinfo=UTC), periods=periods, timeframe=Timeframe.M15, seed=1
        )
    )


class TestValidData:
    def test_clean_synthetic_data_is_valid(self) -> None:
        report = validate_ohlcv(_valid_frame(), symbol="TEST", timeframe=Timeframe.M15)
        assert report.is_valid
        assert not report.critical_issues

    def test_raise_if_invalid_is_a_noop_for_valid_data(self) -> None:
        report = validate_ohlcv(_valid_frame(), symbol="TEST", timeframe=Timeframe.M15)
        report.raise_if_invalid()  # must not raise


class TestSummary:
    def test_summary_of_valid_report_mentions_no_issues_sections(self) -> None:
        report = validate_ohlcv(_valid_frame(), symbol="TEST", timeframe=Timeframe.M15)
        text = report.summary()
        assert "TEST" in text
        assert "valid=True" in text
        assert "Critical issues" not in text

    def test_summary_of_invalid_report_lists_each_critical_issue(self) -> None:
        df = _valid_frame()
        df.loc[0, "volume"] = -1.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        text = report.summary()
        assert "valid=False" in text
        assert "Critical issues (1)" in text
        assert "negative volume" in text

    def test_summary_reports_largest_gap(self) -> None:
        df = _valid_frame(periods=20)
        df = pd.concat([df.iloc[:5], df.iloc[8:]]).reset_index(drop=True)
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        text = report.summary()
        assert "Gaps detected: 1" in text
        assert "3 bars missing after" in text

    def test_summary_includes_warnings_section(self) -> None:
        df = _valid_frame(periods=20)
        df.loc[0, "volume"] = 0.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        text = report.summary()
        assert "Warnings (1)" in text
        assert "zero volume" in text


class TestCriticalIssueDetection:
    def test_detects_high_less_than_low(self) -> None:
        df = _valid_frame()
        df.loc[5, "high"] = df.loc[5, "low"] - 1.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("high < low" in issue for issue in report.critical_issues)

    def test_detects_open_outside_range(self) -> None:
        df = _valid_frame()
        df.loc[3, "open"] = df.loc[3, "high"] + 5.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("open outside" in issue for issue in report.critical_issues)

    def test_detects_close_outside_range(self) -> None:
        df = _valid_frame()
        df.loc[3, "close"] = df.loc[3, "low"] - 5.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("close outside" in issue for issue in report.critical_issues)

    def test_detects_non_positive_price(self) -> None:
        df = _valid_frame()
        df.loc[0, "close"] = 0.0
        df.loc[0, "low"] = 0.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("non-positive price" in issue for issue in report.critical_issues)

    def test_detects_negative_volume(self) -> None:
        df = _valid_frame()
        df.loc[0, "volume"] = -100.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("negative volume" in issue for issue in report.critical_issues)

    def test_detects_duplicate_timestamps(self) -> None:
        df = _valid_frame()
        df.loc[1, "open_time"] = df.loc[0, "open_time"]
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("duplicate" in issue for issue in report.critical_issues)

    def test_detects_non_monotonic_timestamps(self) -> None:
        df = _valid_frame()
        df.loc[2, "open_time"] = df.loc[0, "open_time"] - timedelta(minutes=1)
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("monotonically increasing" in issue for issue in report.critical_issues)

    def test_detects_null_values(self) -> None:
        df = _valid_frame()
        df.loc[0, "close"] = float("nan")
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("null value" in issue for issue in report.critical_issues)

    def test_raise_if_invalid_raises_data_quality_error(self) -> None:
        df = _valid_frame()
        df.loc[0, "volume"] = -1.0
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        with pytest.raises(DataQualityError):
            report.raise_if_invalid()

    def test_missing_columns_reported_as_critical(self) -> None:
        df = _valid_frame().drop(columns=["volume"])
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("missing required columns" in issue for issue in report.critical_issues)

    def test_empty_dataframe_is_a_warning_not_a_critical_issue(self) -> None:
        df = _valid_frame(periods=1).iloc[0:0]
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert report.is_valid
        assert report.row_count == 0
        assert any("empty" in warning for warning in report.warnings)


class TestGapDetection:
    def test_no_gaps_in_contiguous_series(self) -> None:
        df = _valid_frame(periods=100)
        gaps = detect_gaps(df["open_time"], Timeframe.M15)
        assert gaps == []

    def test_detects_a_single_gap(self) -> None:
        df = _valid_frame(periods=20)
        # Remove bars 5-7 (inclusive) to create a 3-bar gap.
        df = pd.concat([df.iloc[:5], df.iloc[8:]]).reset_index(drop=True)
        gaps = detect_gaps(df["open_time"], Timeframe.M15)
        assert len(gaps) == 1
        assert gaps[0].missing_bars == 3

    def test_detects_multiple_gaps(self) -> None:
        df = _valid_frame(periods=30)
        df = pd.concat([df.iloc[:5], df.iloc[7:15], df.iloc[20:]]).reset_index(drop=True)
        gaps = detect_gaps(df["open_time"], Timeframe.M15)
        assert len(gaps) == 2

    def test_gaps_surface_as_warnings_not_critical_issues(self) -> None:
        df = _valid_frame(periods=20)
        df = pd.concat([df.iloc[:5], df.iloc[8:]]).reset_index(drop=True)
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert report.is_valid
        assert len(report.gaps) == 1
        assert any("gap" in warning for warning in report.warnings)

    def test_single_row_or_empty_has_no_gaps(self) -> None:
        assert detect_gaps(pd.Series([], dtype="datetime64[ns, UTC]"), Timeframe.M15) == []
        assert detect_gaps(_valid_frame(periods=1)["open_time"], Timeframe.M15) == []


class TestOverlapDetection:
    """Golden-master regression tests for a data-quality gap found during
    adversarial audit: bars spaced STRICTLY LESS than the nominal timeframe
    duration apart (irregular/overlapping spacing) were completely
    invisible to validation -- `detect_gaps` only fires when delta > duration
    (a real gap), so a malformed series with e.g. M15 bars only 10 minutes
    apart passed as perfectly valid. Exact duplicates (delta == 0) are a
    separate, already-covered critical issue; this covers the "too close
    but not identical" case specifically.
    """

    def test_no_overlaps_in_regularly_spaced_series(self) -> None:
        df = _valid_frame(periods=50)
        assert detect_overlaps(df["open_time"], Timeframe.M15) == []

    def test_detects_a_single_overlap(self) -> None:
        df = _valid_frame(periods=10)
        # Hand-construct one bar spaced only 10 minutes after its
        # predecessor, on an M15 (15-minute) series -- an irregular gap of
        # 10 < 15 minutes, which must be flagged as an overlap.
        df.loc[5, "open_time"] = df.loc[4, "open_time"] + timedelta(minutes=10)
        overlaps = detect_overlaps(df["open_time"], Timeframe.M15)
        assert len(overlaps) == 1
        assert overlaps[0].actual_spacing == timedelta(minutes=10)

    def test_overlap_is_a_critical_issue_not_a_warning(self) -> None:
        df = _valid_frame(periods=10)
        df.loc[5, "open_time"] = df.loc[4, "open_time"] + timedelta(minutes=10)
        report = validate_ohlcv(df, symbol="TEST", timeframe=Timeframe.M15)
        assert not report.is_valid
        assert any("overlapping" in issue for issue in report.critical_issues)
        assert len(report.overlaps) == 1

    def test_exact_duplicate_is_not_double_reported_as_an_overlap(self) -> None:
        # Duplicates (delta == 0) are the OTHER critical check's job; detect_overlaps
        # requires a STRICTLY positive (but sub-duration) delta, so a duplicate
        # should not also show up in the overlaps list.
        df = _valid_frame(periods=10)
        df.loc[5, "open_time"] = df.loc[4, "open_time"]  # exact duplicate, delta == 0
        overlaps = detect_overlaps(df["open_time"], Timeframe.M15)
        assert overlaps == []

    def test_single_row_or_empty_has_no_overlaps(self) -> None:
        assert detect_overlaps(pd.Series([], dtype="datetime64[ns, UTC]"), Timeframe.M15) == []
        assert detect_overlaps(_valid_frame(periods=1)["open_time"], Timeframe.M15) == []
