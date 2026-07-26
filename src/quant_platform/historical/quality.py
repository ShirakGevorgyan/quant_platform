"""Typed data-quality reporting for the historical ingestion pipeline.

This is deliberately a much richer, separate system from the existing
`data.validation.validate_ohlcv` (which remains untouched and still serves
Milestone 1's simpler CSV/Parquet `DataSource` path): every issue here is a
typed `QualityIssue` carrying a severity, a category, affected row
count/sample timestamps, and (where meaningful) summary statistics --
never just a boolean or a bare string -- so a caller can make policy
decisions (`historical.repair`) and structured log entries
(`historical.pipeline`) from the report without re-parsing message text.

Every check is independent and additive: a frame with multiple simultaneous
problems reports all of them, not just the first one found. Detection
never mutates the input and never repairs anything itself -- that is
`historical.repair`'s job, operating on the report this module produces.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.historical.calendar import TradingCalendar
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS

logger = logging.getLogger(__name__)

_MAX_SAMPLE_TIMESTAMPS = 10


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class IssueType(Enum):
    MISSING_COLUMN = "MISSING_COLUMN"
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    NULL_VALUE = "NULL_VALUE"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    HIGH_LESS_THAN_LOW = "HIGH_LESS_THAN_LOW"
    OPEN_OUTSIDE_RANGE = "OPEN_OUTSIDE_RANGE"
    CLOSE_OUTSIDE_RANGE = "CLOSE_OUTSIDE_RANGE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    NEGATIVE_SPREAD = "NEGATIVE_SPREAD"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    UNORDERED_TIMESTAMP = "UNORDERED_TIMESTAMP"
    MISSING_BARS_GAP = "MISSING_BARS_GAP"
    OVERLAPPING_BARS = "OVERLAPPING_BARS"
    IRREGULAR_SPACING = "IRREGULAR_SPACING"
    UNEXPECTED_CLOSED_SESSION_BAR = "UNEXPECTED_CLOSED_SESSION_BAR"
    UNEXPECTED_SESSION_GAP = "UNEXPECTED_SESSION_GAP"
    INCOMPLETE_EDGE_BAR = "INCOMPLETE_EDGE_BAR"
    MISALIGNED_TIMESTAMP = "MISALIGNED_TIMESTAMP"
    MIXED_TZ_AWARENESS = "MIXED_TZ_AWARENESS"
    FROZEN_PRICE_SEQUENCE = "FROZEN_PRICE_SEQUENCE"
    IMPOSSIBLE_PRICE_JUMP = "IMPOSSIBLE_PRICE_JUMP"
    EXTREME_SPREAD = "EXTREME_SPREAD"
    EXTREME_RANGE = "EXTREME_RANGE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    BATCH_BOUNDARY_ARTIFACT = "BATCH_BOUNDARY_ARTIFACT"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    issue_type: IssueType
    severity: Severity
    message: str
    affected_row_count: int
    affected_timestamps: tuple[pd.Timestamp, ...] = ()
    """A capped (at `_MAX_SAMPLE_TIMESTAMPS`) sample for human-readable
    reporting/logging -- NOT the complete set. See `affected_row_indices`
    for the complete, uncapped set a repair/quarantine policy needs to act
    on precisely."""
    affected_row_indices: tuple[int, ...] = ()
    """The COMPLETE set of positional row indices this issue applies to
    (never capped/sampled), consumed by `historical.repair` to quarantine
    or reject exactly the affected rows -- never an approximation."""
    stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Configurable thresholds for the market-quality heuristics. Every
    default here is a documented, deliberately conservative starting point
    for XAUUSD-scale intraday data -- NOT a claim of universal correctness
    for every instrument/timeframe; callers with better domain knowledge
    of their specific broker/symbol should override these."""

    max_price_jump_fraction: float = 0.05
    """A bar-to-bar close return whose absolute value exceeds this fraction
    (5% default) is flagged as an impossible/implausible jump for an
    intraday XAUUSD timeframe."""
    max_spread_points: float = 500.0
    frozen_sequence_min_length: int = 5
    """Minimum run length of bit-identical OHLC bars before being flagged
    as a frozen/stale price sequence (a feed outage masquerading as quiet
    trading, not genuine zero-volatility, is the failure mode this
    detects)."""
    extreme_range_multiple: float = 10.0
    """A bar whose (high - low) exceeds this multiple of the trailing
    median range is flagged."""
    volume_spike_multiple: float = 10.0

    def __post_init__(self) -> None:
        if self.max_price_jump_fraction <= 0:
            raise ValueError("max_price_jump_fraction must be positive")
        if self.max_spread_points <= 0:
            raise ValueError("max_spread_points must be positive")
        if self.frozen_sequence_min_length < 2:
            raise ValueError("frozen_sequence_min_length must be >= 2")
        if self.extreme_range_multiple <= 0:
            raise ValueError("extreme_range_multiple must be positive")
        if self.volume_spike_multiple <= 0:
            raise ValueError("volume_spike_multiple must be positive")


@dataclass(slots=True)
class QualityReport:
    symbol: str
    timeframe: Timeframe
    row_count: int
    generated_at: datetime
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is Severity.CRITICAL]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return not self.critical_issues

    def summary(self) -> str:
        lines = [
            f"QualityReport(symbol={self.symbol}, timeframe={self.timeframe.value}, "
            f"rows={self.row_count}, valid={self.is_valid}, "
            f"critical={len(self.critical_issues)}, warnings={len(self.warnings)}, info={len(self.infos)})",
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.value}] {issue.issue_type.value}: {issue.message}")
        return "\n".join(lines)


def _sample_timestamps(open_time: pd.Series, mask: pd.Series) -> tuple[pd.Timestamp, ...]:
    return tuple(open_time.loc[mask].iloc[:_MAX_SAMPLE_TIMESTAMPS])


def _all_indices(mask: pd.Series) -> tuple[int, ...]:
    """The complete, uncapped set of positional row indices where `mask`
    is True -- see `QualityIssue.affected_row_indices`."""
    return tuple(int(i) for i in np.flatnonzero(mask.to_numpy()))


def run_quality_checks(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    calendar: TradingCalendar | None = None,
    thresholds: QualityThresholds | None = None,
    batch_boundaries: tuple[pd.Timestamp, ...] = (),
) -> QualityReport:
    """Run the full market-data quality check catalog against `df` and
    return a `QualityReport`. Never raises on data-quality grounds (that is
    `historical.repair`'s decision, driven by this report) and never
    mutates `df`. `calendar`, if provided, is used to avoid misclassifying
    expected weekly/maintenance/holiday closures as anomalous gaps -- see
    `historical.calendar`."""
    started_at = time.perf_counter()
    thresholds = thresholds or QualityThresholds()
    issues: list[QualityIssue] = []
    generated_at = pd.Timestamp.now(tz="UTC").to_pydatetime()

    required_columns = set(RAW_HISTORICAL_COLUMNS) if _looks_like_raw_schema(df) else set(OHLCV_COLUMNS)
    missing = required_columns - set(df.columns)
    if missing:
        issues.append(
            QualityIssue(
                IssueType.MISSING_COLUMN, Severity.CRITICAL,
                f"missing required columns: {sorted(missing)}", affected_row_count=len(df),
            )
        )
        report = QualityReport(symbol=symbol, timeframe=timeframe, row_count=len(df), generated_at=generated_at, issues=issues)
        _log_quality_report(report, started_at)
        return report

    if df.empty:
        report = QualityReport(symbol=symbol, timeframe=timeframe, row_count=0, generated_at=generated_at, issues=issues)
        _log_quality_report(report, started_at)
        return report

    price_columns = [c for c in ("open", "high", "low", "close") if c in df.columns]
    open_time = df["open_time"]

    # Tier 1: per-value sanity. These make individual cells untrustworthy
    # for any further arithmetic, so everything below is gated on them.
    _check_tz_awareness(df, issues)
    _check_dtypes(df, price_columns, issues)
    _check_nulls(df, price_columns, issues)
    _check_non_finite(df, price_columns, issues)
    _check_non_positive_prices(df, price_columns, issues)
    _check_ohlc_invariants(df, issues)
    _check_volume_and_spread_sign(df, issues)

    # Tier 2: temporal shape (ordering, alignment, spacing). These are
    # deliberately NOT gated on each other -- ordering, duplicate,
    # misalignment, overlap, and gap findings are independent facets of the
    # same open_time arithmetic and a malformed series routinely trips
    # several simultaneously (e.g. any sub-duration-spaced "overlap" bar is,
    # by definition, also off the timeframe's alignment grid). Gating one
    # behind another here would silently hide real, independent findings --
    # confirmed via a failing test where an overlap was never reported
    # because it happened to also be flagged as misaligned first. They are
    # gated only on `open_time` itself being usable at all (tz-aware, no
    # nulls): a naive or null timestamp column makes delta/grid arithmetic
    # meaningless, not just imprecise.
    open_time_is_usable = (
        not any(i.issue_type is IssueType.MIXED_TZ_AWARENESS for i in issues)
        and not open_time.isna().any()
    )
    if open_time_is_usable:
        _check_timestamp_ordering(df, open_time, issues)
        _check_alignment(open_time, timeframe, issues)
        _check_gaps_and_overlaps(open_time, timeframe, calendar, issues)
        _check_unexpected_closed_session_bars(open_time, timeframe, calendar, issues)

    # Tier 3: market-quality heuristics (percent-change, rolling medians).
    # These genuinely do need Tiers 1+2 to be clean first -- they are
    # statistics computed ACROSS rows, and a single bad value or
    # out-of-order timestamp would corrupt every window touching it.
    if not any(i.severity is Severity.CRITICAL for i in issues):
        _check_frozen_sequences(df, thresholds, issues)
        _check_impossible_jumps(df, thresholds, issues)
        _check_extreme_spread(df, thresholds, issues)
        _check_extreme_range(df, thresholds, issues)
        _check_volume_spikes(df, thresholds, issues)
        _check_incomplete_edge_bars(df, issues)
        if batch_boundaries:
            _check_batch_boundary_artifacts(open_time, timeframe, batch_boundaries, issues)

    report = QualityReport(symbol=symbol, timeframe=timeframe, row_count=len(df), generated_at=generated_at, issues=issues)
    _log_quality_report(report, started_at)
    return report


def _log_quality_report(report: QualityReport, started_at: float) -> None:
    duration_s = time.perf_counter() - started_at
    logger.info(
        "quality check complete: symbol=%s timeframe=%s rows=%d critical=%d warning=%d info=%d "
        "valid=%s duration_s=%.3f",
        report.symbol, report.timeframe.value, report.row_count, len(report.critical_issues),
        len(report.warnings), len(report.infos), report.is_valid, duration_s,
    )


def _looks_like_raw_schema(df: pd.DataFrame) -> bool:
    return "tick_volume" in df.columns or "spread" in df.columns


def _check_tz_awareness(df: pd.DataFrame, issues: list[QualityIssue]) -> None:
    open_time = df["open_time"]
    if not pd.api.types.is_datetime64_any_dtype(open_time) or open_time.dt.tz is None:
        issues.append(
            QualityIssue(
                IssueType.MIXED_TZ_AWARENESS, Severity.CRITICAL,
                "open_time is not tz-aware UTC", affected_row_count=len(df),
            )
        )


def _check_dtypes(df: pd.DataFrame, price_columns: list[str], issues: list[QualityIssue]) -> None:
    for column in price_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(
                QualityIssue(
                    IssueType.UNSUPPORTED_DTYPE, Severity.CRITICAL,
                    f"column {column!r} has non-numeric dtype {df[column].dtype}", affected_row_count=len(df),
                )
            )


def _check_nulls(df: pd.DataFrame, price_columns: list[str], issues: list[QualityIssue]) -> None:
    columns_to_check = [*price_columns, "open_time"]
    for column in columns_to_check:
        null_mask = df[column].isna()
        if null_mask.any():
            issues.append(
                QualityIssue(
                    IssueType.NULL_VALUE, Severity.CRITICAL,
                    f"{int(null_mask.sum())} null value(s) in column {column!r}",
                    affected_row_count=int(null_mask.sum()),
                    affected_timestamps=_sample_timestamps(df["open_time"], null_mask),
                    affected_row_indices=_all_indices(null_mask),
                )
            )


def _check_non_finite(df: pd.DataFrame, price_columns: list[str], issues: list[QualityIssue]) -> None:
    for column in price_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        finite = np.isfinite(df[column].astype(np.float64).to_numpy())
        mask = pd.Series(~finite, index=df.index) & df[column].notna()
        if mask.any():
            issues.append(
                QualityIssue(
                    IssueType.NON_FINITE_VALUE, Severity.CRITICAL,
                    f"{int(mask.sum())} non-finite (inf) value(s) in column {column!r}",
                    affected_row_count=int(mask.sum()),
                    affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
                )
            )


def _check_non_positive_prices(df: pd.DataFrame, price_columns: list[str], issues: list[QualityIssue]) -> None:
    if not price_columns:
        return
    mask = (df[price_columns] <= 0).any(axis=1)
    if mask.any():
        issues.append(
            QualityIssue(
                IssueType.NON_POSITIVE_PRICE, Severity.CRITICAL,
                f"{int(mask.sum())} bar(s) with non-positive price", affected_row_count=int(mask.sum()),
                affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
            )
        )


def _check_ohlc_invariants(df: pd.DataFrame, issues: list[QualityIssue]) -> None:
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return
    high_low_mask = df["high"] < df["low"]
    if high_low_mask.any():
        issues.append(
            QualityIssue(
                IssueType.HIGH_LESS_THAN_LOW, Severity.CRITICAL,
                f"{int(high_low_mask.sum())} bar(s) with high < low", affected_row_count=int(high_low_mask.sum()),
                affected_timestamps=_sample_timestamps(df["open_time"], high_low_mask),
                    affected_row_indices=_all_indices(high_low_mask),
            )
        )
    open_mask = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if open_mask.any():
        issues.append(
            QualityIssue(
                IssueType.OPEN_OUTSIDE_RANGE, Severity.CRITICAL,
                f"{int(open_mask.sum())} bar(s) with open outside [low, high]", affected_row_count=int(open_mask.sum()),
                affected_timestamps=_sample_timestamps(df["open_time"], open_mask),
                    affected_row_indices=_all_indices(open_mask),
            )
        )
    close_mask = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if close_mask.any():
        issues.append(
            QualityIssue(
                IssueType.CLOSE_OUTSIDE_RANGE, Severity.CRITICAL,
                f"{int(close_mask.sum())} bar(s) with close outside [low, high]", affected_row_count=int(close_mask.sum()),
                affected_timestamps=_sample_timestamps(df["open_time"], close_mask),
                    affected_row_indices=_all_indices(close_mask),
            )
        )


def _check_volume_and_spread_sign(df: pd.DataFrame, issues: list[QualityIssue]) -> None:
    for column, issue_type in (
        ("tick_volume", IssueType.NEGATIVE_VOLUME),
        ("real_volume", IssueType.NEGATIVE_VOLUME),
        ("volume", IssueType.NEGATIVE_VOLUME),
        ("spread", IssueType.NEGATIVE_SPREAD),
    ):
        if column not in df.columns:
            continue
        mask = df[column] < 0
        if mask.any():
            issues.append(
                QualityIssue(
                    issue_type, Severity.CRITICAL,
                    f"{int(mask.sum())} bar(s) with negative {column}", affected_row_count=int(mask.sum()),
                    affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
                )
            )


def _check_timestamp_ordering(df: pd.DataFrame, open_time: pd.Series, issues: list[QualityIssue]) -> None:
    if not open_time.is_monotonic_increasing:
        issues.append(
            QualityIssue(
                IssueType.UNORDERED_TIMESTAMP, Severity.CRITICAL,
                "open_time is not monotonically increasing", affected_row_count=len(df),
            )
        )
    duplicated = open_time.duplicated()
    if duplicated.any():
        issues.append(
            QualityIssue(
                IssueType.DUPLICATE_TIMESTAMP, Severity.CRITICAL,
                f"{int(duplicated.sum())} duplicate open_time value(s)", affected_row_count=int(duplicated.sum()),
                affected_timestamps=_sample_timestamps(open_time, duplicated),
                    affected_row_indices=_all_indices(duplicated),
            )
        )


def _check_alignment(open_time: pd.Series, timeframe: Timeframe, issues: list[QualityIssue]) -> None:
    duration_seconds = int(timeframe.duration.total_seconds())
    # `open_time`'s datetime64 storage unit (ns/us/ms/s) is not guaranteed --
    # pandas selects it based on how the series was constructed (observed:
    # microsecond resolution from `pd.date_range(..., tz=...)` on this
    # pandas version) -- so an `.astype("int64")` without first pinning a
    # known unit would silently divide by the wrong power of ten and mis-
    # classify every timestamp's alignment. Casting to a fixed ns unit first
    # makes the epoch-seconds conversion below correct regardless of the
    # input's original resolution.
    epoch_seconds = open_time.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000
    misaligned = (epoch_seconds % duration_seconds) != 0
    if misaligned.any():
        issues.append(
            QualityIssue(
                IssueType.MISALIGNED_TIMESTAMP, Severity.CRITICAL,
                f"{int(misaligned.sum())} bar(s) not aligned to the {timeframe.value} grid",
                affected_row_count=int(misaligned.sum()),
                affected_timestamps=_sample_timestamps(open_time, misaligned),
                    affected_row_indices=_all_indices(misaligned),
            )
        )


def _check_gaps_and_overlaps(
    open_time: pd.Series, timeframe: Timeframe, calendar: TradingCalendar | None, issues: list[QualityIssue]
) -> None:
    if len(open_time) < 2:
        return
    duration = pd.Timedelta(timeframe.duration)
    deltas = open_time.diff()
    zero = pd.Timedelta(0)

    overlap_mask = (deltas > zero) & (deltas < duration)
    if overlap_mask.any():
        issues.append(
            QualityIssue(
                IssueType.OVERLAPPING_BARS, Severity.CRITICAL,
                f"{int(overlap_mask.sum())} bar(s) spaced closer together than {timeframe.value}",
                affected_row_count=int(overlap_mask.sum()),
                affected_timestamps=_sample_timestamps(open_time, overlap_mask),
                    affected_row_indices=_all_indices(overlap_mask),
            )
        )

    duration_ns = int(duration.value)
    deltas_ns = deltas.astype("timedelta64[ns]").astype("int64")
    irregular_mask = (deltas > duration) & ((deltas_ns % duration_ns) != 0)
    if irregular_mask.any():
        issues.append(
            QualityIssue(
                IssueType.IRREGULAR_SPACING, Severity.WARNING,
                f"{int(irregular_mask.sum())} bar(s) spaced at a non-integer multiple of {timeframe.value}",
                affected_row_count=int(irregular_mask.sum()),
                affected_timestamps=_sample_timestamps(open_time, irregular_mask),
                    affected_row_indices=_all_indices(irregular_mask),
            )
        )

    gap_mask = deltas > duration
    gap_positions = np.flatnonzero(gap_mask.to_numpy())
    for position in gap_positions:
        gap_start = pd.Timestamp(open_time.iloc[position - 1]) + duration
        gap_end = pd.Timestamp(open_time.iloc[position])
        if calendar is not None and calendar.is_expected_closure(gap_start, gap_end):
            issues.append(
                QualityIssue(
                    IssueType.MISSING_BARS_GAP, Severity.INFO,
                    f"expected closure gap from {gap_start} to {gap_end}", affected_row_count=1,
                    affected_timestamps=(gap_start,),
                )
            )
        else:
            issues.append(
                QualityIssue(
                    IssueType.UNEXPECTED_SESSION_GAP, Severity.WARNING,
                    f"unexplained gap from {gap_start} to {gap_end} (no calendar closure covers it)",
                    affected_row_count=1, affected_timestamps=(gap_start,),
                )
            )


def _check_unexpected_closed_session_bars(
    open_time: pd.Series, timeframe: Timeframe, calendar: TradingCalendar | None, issues: list[QualityIssue]
) -> None:
    if calendar is None:
        return
    closed_mask = pd.Series(
        [
            calendar.is_expected_closure(ts, ts + timeframe.duration)
            for ts in open_time
        ],
        index=open_time.index,
    )
    if closed_mask.any():
        issues.append(
            QualityIssue(
                IssueType.UNEXPECTED_CLOSED_SESSION_BAR, Severity.WARNING,
                f"{int(closed_mask.sum())} bar(s) present during a configured closed-session window",
                affected_row_count=int(closed_mask.sum()),
                affected_timestamps=_sample_timestamps(open_time, closed_mask),
                    affected_row_indices=_all_indices(closed_mask),
            )
        )


def _check_frozen_sequences(df: pd.DataFrame, thresholds: QualityThresholds, issues: list[QualityIssue]) -> None:
    if not {"open", "high", "low", "close"}.issubset(df.columns) or len(df) < thresholds.frozen_sequence_min_length:
        return
    identical_to_prev = (
        (df["open"] == df["open"].shift())
        & (df["high"] == df["high"].shift())
        & (df["low"] == df["low"].shift())
        & (df["close"] == df["close"].shift())
    )
    run_id = (~identical_to_prev).cumsum()
    run_lengths = identical_to_prev.groupby(run_id).transform("sum") + 1
    frozen_mask = run_lengths >= thresholds.frozen_sequence_min_length
    if frozen_mask.any():
        issues.append(
            QualityIssue(
                IssueType.FROZEN_PRICE_SEQUENCE, Severity.WARNING,
                f"{int(frozen_mask.sum())} bar(s) part of a frozen/repeated OHLC run "
                f"(>= {thresholds.frozen_sequence_min_length} identical consecutive bars)",
                affected_row_count=int(frozen_mask.sum()),
                affected_timestamps=_sample_timestamps(df["open_time"], frozen_mask),
                    affected_row_indices=_all_indices(frozen_mask),
            )
        )


def _check_impossible_jumps(df: pd.DataFrame, thresholds: QualityThresholds, issues: list[QualityIssue]) -> None:
    if "close" not in df.columns or len(df) < 2:
        return
    returns = df["close"].pct_change().abs()
    mask = returns > thresholds.max_price_jump_fraction
    if mask.any():
        issues.append(
            QualityIssue(
                IssueType.IMPOSSIBLE_PRICE_JUMP, Severity.CRITICAL,
                f"{int(mask.sum())} bar(s) with a close-to-close move exceeding "
                f"{thresholds.max_price_jump_fraction:.1%}",
                affected_row_count=int(mask.sum()), affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
                stats={"max_abs_return": float(returns.max())},
            )
        )


def _check_extreme_spread(df: pd.DataFrame, thresholds: QualityThresholds, issues: list[QualityIssue]) -> None:
    if "spread" not in df.columns:
        return
    mask = df["spread"] > thresholds.max_spread_points
    if mask.any():
        issues.append(
            QualityIssue(
                IssueType.EXTREME_SPREAD, Severity.WARNING,
                f"{int(mask.sum())} bar(s) with spread exceeding {thresholds.max_spread_points} points",
                affected_row_count=int(mask.sum()), affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
            )
        )


def _check_extreme_range(df: pd.DataFrame, thresholds: QualityThresholds, issues: list[QualityIssue]) -> None:
    if not {"high", "low"}.issubset(df.columns) or len(df) < 10:
        return
    bar_range = df["high"] - df["low"]
    median_range = bar_range.rolling(window=20, min_periods=5).median()
    mask = (median_range > 0) & (bar_range > median_range * thresholds.extreme_range_multiple)
    if mask.any():
        issues.append(
            QualityIssue(
                IssueType.EXTREME_RANGE, Severity.WARNING,
                f"{int(mask.sum())} bar(s) with range exceeding {thresholds.extreme_range_multiple}x "
                "the trailing median range",
                affected_row_count=int(mask.sum()), affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
            )
        )


def _check_volume_spikes(df: pd.DataFrame, thresholds: QualityThresholds, issues: list[QualityIssue]) -> None:
    volume_column = "tick_volume" if "tick_volume" in df.columns else ("volume" if "volume" in df.columns else None)
    if volume_column is None or len(df) < 10:
        return
    volume = df[volume_column]
    median_volume = volume.rolling(window=20, min_periods=5).median()
    mask = (median_volume > 0) & (volume > median_volume * thresholds.volume_spike_multiple)
    if mask.any():
        issues.append(
            QualityIssue(
                IssueType.VOLUME_SPIKE, Severity.INFO,
                f"{int(mask.sum())} bar(s) with {volume_column} exceeding {thresholds.volume_spike_multiple}x "
                "the trailing median",
                affected_row_count=int(mask.sum()), affected_timestamps=_sample_timestamps(df["open_time"], mask),
                    affected_row_indices=_all_indices(mask),
            )
        )


def _check_incomplete_edge_bars(df: pd.DataFrame, issues: list[QualityIssue]) -> None:
    if not {"open", "high", "low", "close"}.issubset(df.columns) or len(df) == 0:
        return
    for position in (0, len(df) - 1):
        row = df.iloc[position]
        if row["high"] == row["low"] == row["open"] == row["close"]:
            issues.append(
                QualityIssue(
                    IssueType.INCOMPLETE_EDGE_BAR, Severity.INFO,
                    f"edge bar at position {position} has zero range (open==high==low==close); "
                    "possibly a partially-formed bar at the extraction boundary",
                    affected_row_count=1, affected_timestamps=(df["open_time"].iloc[position],),
                )
            )


def _check_batch_boundary_artifacts(
    open_time: pd.Series, timeframe: Timeframe, batch_boundaries: tuple[pd.Timestamp, ...], issues: list[QualityIssue]
) -> None:
    """Flag bars sitting exactly at a caller-supplied pagination boundary
    whose spacing from the preceding bar is anything other than exactly one
    bar-width. Any such anomaly (duplicate, overlap, or gap) is already
    reported under its own primary `IssueType` by the general checks above
    -- this adds a second, WARNING-level annotation specifically because
    "the anomaly happens to sit exactly at a known batch seam" is itself
    operationally useful signal (it points at the ingestion pipeline's own
    pagination/stitching logic as a likely root cause, rather than at the
    upstream source)."""
    boundary_set = set(batch_boundaries)
    mask = open_time.isin(boundary_set)
    positions = np.flatnonzero(mask.to_numpy())
    duration = timeframe.duration
    flagged: list[pd.Timestamp] = []
    for position in positions:
        if position == 0:
            continue
        delta = open_time.iloc[position] - open_time.iloc[position - 1]
        if delta != duration:
            flagged.append(pd.Timestamp(open_time.iloc[position]))
    if flagged:
        issues.append(
            QualityIssue(
                IssueType.BATCH_BOUNDARY_ARTIFACT, Severity.WARNING,
                f"{len(flagged)} anomalous bar(s) found exactly at a pagination batch boundary",
                affected_row_count=len(flagged), affected_timestamps=tuple(flagged[:_MAX_SAMPLE_TIMESTAMPS]),
            )
        )


__all__ = [
    "IssueType",
    "QualityIssue",
    "QualityReport",
    "QualityThresholds",
    "Severity",
    "run_quality_checks",
]
