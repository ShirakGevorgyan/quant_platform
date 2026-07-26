"""OHLCV data-quality validation: OHLC invariants, gap detection, and
duplicate/monotonicity checks, independent of where the data came from.

This is deliberately separate from `DataSource._finalize` (basic hygiene
every source enforces unconditionally) -- validation here is an explicit,
opt-in *report* a caller requests when they want to know whether the data
is trustworthy enough for research, not a silent gate that could hide
problems by quietly dropping rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_platform.core.exceptions import DataQualityError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe


@dataclass(frozen=True, slots=True)
class Gap:
    """A detected discontinuity between two consecutive bars."""

    after_open_time: pd.Timestamp
    before_open_time: pd.Timestamp
    missing_bars: int


@dataclass(frozen=True, slots=True)
class Overlap:
    """Two consecutive bars spaced STRICTLY LESS than one timeframe
    duration apart -- irregular/overlapping spacing. Distinct from an
    exact duplicate open_time (spacing == 0), which is rejected as its own
    critical issue; this covers the case where bars are merely too close
    together (e.g. an M15 series with bars only 10 minutes apart), which
    never happens legitimately in a well-formed fixed-timeframe series and
    indicates a genuine upstream data problem (bad feed, merge error,
    incorrect resampling)."""

    after_open_time: pd.Timestamp
    before_open_time: pd.Timestamp
    actual_spacing: pd.Timedelta


@dataclass(slots=True)
class DataQualityReport:
    """Aggregated findings from `validate_ohlcv`. `is_valid` reflects only
    critical structural issues (OHLC invariant violations, non-monotonic,
    duplicate, or overlapping timestamps, non-positive prices/volume);
    gaps are informational since real markets close on weekends/holidays
    and a gap alone does not make the data unusable."""

    symbol: str
    timeframe: Timeframe
    row_count: int
    critical_issues: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    overlaps: list[Overlap] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.critical_issues

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise DataQualityError(
                f"Data quality validation failed for {self.symbol}/{self.timeframe.value}: "
                f"{'; '.join(self.critical_issues)}",
                context={"symbol": self.symbol, "timeframe": self.timeframe.value},
            )

    def summary(self) -> str:
        lines = [
            f"DataQualityReport(symbol={self.symbol}, timeframe={self.timeframe.value}, "
            f"rows={self.row_count}, valid={self.is_valid})",
        ]
        if self.critical_issues:
            lines.append(f"  Critical issues ({len(self.critical_issues)}):")
            lines.extend(f"    - {issue}" for issue in self.critical_issues)
        if self.gaps:
            lines.append(f"  Gaps detected: {len(self.gaps)} (largest: {self._largest_gap()})")
        if self.overlaps:
            lines.append(f"  Overlapping/irregular spacing detected: {len(self.overlaps)}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            lines.extend(f"    - {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def _largest_gap(self) -> str:
        if not self.gaps:
            return "none"
        largest = max(self.gaps, key=lambda g: g.missing_bars)
        return f"{largest.missing_bars} bars missing after {largest.after_open_time}"


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    detect_gaps_flag: bool = True,
) -> DataQualityReport:
    """Run structural and statistical checks on a canonical-schema OHLCV
    DataFrame and return a report. Does not raise on its own -- call
    `report.raise_if_invalid()` if the caller wants strict enforcement."""
    critical: list[str] = []
    warnings: list[str] = []

    missing_columns = set(OHLCV_COLUMNS) - set(df.columns)
    if missing_columns:
        critical.append(f"missing required columns: {sorted(missing_columns)}")
        return DataQualityReport(
            symbol=symbol, timeframe=timeframe, row_count=len(df), critical_issues=critical
        )

    if df.empty:
        warnings.append("DataFrame is empty")
        return DataQualityReport(
            symbol=symbol, timeframe=timeframe, row_count=0, warnings=warnings
        )

    if not df["open_time"].is_monotonic_increasing:
        critical.append("open_time is not monotonically increasing")
    if df["open_time"].duplicated().any():
        duplicate_count = int(df["open_time"].duplicated().sum())
        critical.append(f"{duplicate_count} duplicate open_time value(s)")

    invalid_range = (df["high"] < df["low"]).sum()
    if invalid_range:
        critical.append(f"{int(invalid_range)} bar(s) with high < low")

    invalid_open = ((df["open"] < df["low"]) | (df["open"] > df["high"])).sum()
    if invalid_open:
        critical.append(f"{int(invalid_open)} bar(s) with open outside [low, high]")

    invalid_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
    if invalid_close:
        critical.append(f"{int(invalid_close)} bar(s) with close outside [low, high]")

    non_positive_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if non_positive_price:
        critical.append(f"{int(non_positive_price)} bar(s) with non-positive price")

    negative_volume = (df["volume"] < 0).sum()
    if negative_volume:
        critical.append(f"{int(negative_volume)} bar(s) with negative volume")

    null_counts = df[list(OHLCV_COLUMNS)].isna().sum()
    for column, count in null_counts.items():
        if count:
            critical.append(f"{int(count)} null value(s) in column '{column}'")

    zero_volume = int((df["volume"] == 0).sum())
    if zero_volume:
        warnings.append(f"{zero_volume} bar(s) with zero volume (illiquid or session gap)")

    overlaps: list[Overlap] = []
    if detect_gaps_flag and not critical:
        overlaps = detect_overlaps(df["open_time"], timeframe)
        if overlaps:
            critical.append(
                f"{len(overlaps)} bar(s) spaced closer together than the {timeframe.value} "
                "timeframe implies (overlapping/irregular spacing)"
            )

    gaps: list[Gap] = []
    if detect_gaps_flag and not critical:
        gaps = detect_gaps(df["open_time"], timeframe)
        if gaps:
            warnings.append(f"{len(gaps)} gap(s) detected between consecutive bars")

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        row_count=len(df),
        critical_issues=critical,
        gaps=gaps,
        overlaps=overlaps,
        warnings=warnings,
    )


def detect_gaps(open_times: pd.Series, timeframe: Timeframe) -> list[Gap]:
    """Flag every place two consecutive bars are separated by more than one
    timeframe duration. Weekends/holidays routinely produce gaps for
    intraday FX/commodity data; this is informational, not an error."""
    if len(open_times) < 2:
        return []

    ordered = open_times.reset_index(drop=True)
    duration = timeframe.duration
    deltas = ordered.diff()

    gaps: list[Gap] = []
    for position in range(1, len(ordered)):
        delta = deltas.iloc[position]
        if delta > duration:
            missing_bars = int(delta / duration) - 1
            gaps.append(
                Gap(
                    after_open_time=ordered.iloc[position - 1],
                    before_open_time=ordered.iloc[position],
                    missing_bars=missing_bars,
                )
            )
    return gaps


def detect_overlaps(open_times: pd.Series, timeframe: Timeframe) -> list[Overlap]:
    """Flag every place two consecutive bars are spaced STRICTLY LESS than
    one timeframe duration apart. Exact duplicates (spacing == 0) are
    reported separately as a critical duplicate-timestamp issue; this
    covers merely-too-close/irregular spacing, which -- unlike a gap --
    never happens legitimately in a well-formed fixed-timeframe series."""
    if len(open_times) < 2:
        return []

    ordered = open_times.reset_index(drop=True)
    duration = timeframe.duration
    zero = pd.Timedelta(0)
    deltas = ordered.diff()

    overlaps: list[Overlap] = []
    for position in range(1, len(ordered)):
        delta = deltas.iloc[position]
        if zero < delta < duration:
            overlaps.append(
                Overlap(
                    after_open_time=ordered.iloc[position - 1],
                    before_open_time=ordered.iloc[position],
                    actual_spacing=delta,
                )
            )
    return overlaps


__all__ = ["DataQualityReport", "Gap", "Overlap", "detect_gaps", "detect_overlaps", "validate_ohlcv"]
