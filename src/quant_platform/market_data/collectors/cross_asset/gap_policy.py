"""Gap and missing-bar policy (Milestone 10, Phase 4C, spec Section 21).

Detects, for one component's bar sequence: missing expected bars,
conflicting duplicate coordinates, and out-of-order source sequencing.
Deliberately conservative about "missing": with no authoritative
holiday-calendar data loaded this phase, a business-day (Mon-Fri)
heuristic is the ONLY calendar assurance available for non-24-hour
sessions -- `GapAnalysisReport.calendar_assurance` is always
`"limited"`, an honest disclosure rather than false precision (a
legitimate market holiday on a weekday will be flagged as a candidate
missing bar; this is a KNOWN, DISCLOSED limitation, never silently
presented as a verified holiday-aware gap analysis). A genuine weekend
closure is never reported as missing -- the business-day heuristic
already excludes Saturday/Sunday.

Conflicting-duplicate-coordinate detection (same `open_time`, different
`bar_id`) is NOT a `GapPolicy` choice -- it is always reported and is
always treated as a hard integrity failure by the orchestration layer
that calls `analyze_bar_gaps`, regardless of which `GapPolicy` a
backfill run is configured with. `GapPolicy` governs only how MISSING
bars are handled."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum

from quant_platform.core.exceptions import GapPolicyError
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBar
from quant_platform.market_data.collectors.cross_asset.sessions import TimezoneSessionPolicy

__all__ = [
    "GapAnalysisReport",
    "GapPolicy",
    "analyze_bar_gaps",
]

_MAX_REPORTED_MISSING_DAYS = 200
"""Report bound -- a wildly wrong coverage window (e.g. an accidental
multi-decade span) must never produce an unbounded report payload."""


class GapPolicy(Enum):
    ALLOW_AND_REPORT = "allow_and_report"
    """Missing bars are recorded in the report but never block
    completion -- the default for optional/secondary drivers."""

    QUARANTINE_COMPONENT = "quarantine_component"
    """The component is marked incomplete but the universe as a whole
    may still complete -- other components are unaffected."""

    FAIL_REQUIRED_DRIVER = "fail_required_driver"
    """If the component belongs to a REQUIRED driver and its missing-bar
    rate exceeds the caller's configured tolerance, that driver fails --
    the universe cannot reach COMPLETED with a silently missing required
    driver (spec Section 18)."""

    FAIL_UNIVERSE = "fail_universe"
    """Any missing bar in this component fails the entire universe
    backfill outright -- the strictest policy, for callers that need a
    hard guarantee of complete coverage."""


@dataclass(frozen=True, slots=True)
class GapAnalysisReport:
    canonical_driver_id: str
    provider: str
    provider_symbol: str
    bar_count: int
    missing_business_day_count: int
    missing_business_days: tuple[str, ...]
    """ISO date strings, capped at `_MAX_REPORTED_MISSING_DAYS`."""
    conflicting_coordinate_count: int
    conflicting_coordinates: tuple[str, ...]
    """ISO datetime strings of `open_time`s with >1 distinct `bar_id`."""
    out_of_order_count: int
    calendar_assurance: str
    """Always `"limited"` this phase -- see module docstring."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "canonical_driver_id": self.canonical_driver_id, "provider": self.provider, "provider_symbol": self.provider_symbol,
            "bar_count": self.bar_count, "missing_business_day_count": self.missing_business_day_count,
            "missing_business_days": list(self.missing_business_days), "conflicting_coordinate_count": self.conflicting_coordinate_count,
            "conflicting_coordinates": list(self.conflicting_coordinates), "out_of_order_count": self.out_of_order_count,
            "calendar_assurance": self.calendar_assurance,
        }

    @property
    def has_conflicting_coordinates(self) -> bool:
        return self.conflicting_coordinate_count > 0


def _business_days_between(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def analyze_bar_gaps(bars: tuple[MarketDriverBar, ...], *, session_policy: TimezoneSessionPolicy) -> GapAnalysisReport:
    """PURE. `bars` need not be pre-sorted or pre-deduplicated -- this
    function performs its own analysis directly against whatever was
    read back from `MarketDriverBarStore` (or an in-memory candidate
    batch, prior to a commit decision)."""
    if not bars:
        raise GapPolicyError("analyze_bar_gaps requires at least one bar")
    driver_ids = {b.canonical_driver_id for b in bars}
    providers = {b.provider for b in bars}
    symbols = {b.provider_symbol for b in bars}
    if len(driver_ids) != 1 or len(providers) != 1 or len(symbols) != 1:
        raise GapPolicyError(
            f"analyze_bar_gaps requires a single (canonical_driver_id, provider, provider_symbol) scope, got "
            f"driver_ids={sorted(driver_ids)!r} providers={sorted(providers)!r} symbols={sorted(symbols)!r}"
        )

    ordered = sorted(bars, key=lambda b: (b.open_time, b.bar_id))

    # Conflicting duplicate coordinates: same open_time, DIFFERENT ECONOMIC CONTENT.
    # Deliberately NOT a `bar_id` comparison: `bar_id` embeds provenance
    # (request/response/source manifest ids, source_row_index), so the SAME
    # economic bar re-fetched under a fresh response (e.g. a FORCE_FRESH refetch
    # whose response envelope happens to include one more trailing row) gets a
    # DIFFERENT bar_id despite identical OHLCV/volume/classification -- comparing
    # by bar_id would misclassify that as a data-integrity conflict. Two bars are
    # only a genuine conflict when they disagree on what actually happened in the
    # market (price/volume) or how the bar is classified (adjustment/availability/
    # session/futures identity).
    by_open_time: dict[datetime, set[tuple[object, ...]]] = {}
    for bar in ordered:
        content = (
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.volume_unit, bar.adjustment_policy_id, bar.availability_time,
            bar.availability_policy_id, bar.session_policy_id, bar.contract_metadata_id, bar.roll_provenance,
        )
        by_open_time.setdefault(bar.open_time, set()).add(content)
    conflicting = sorted(ot.isoformat() for ot, contents in by_open_time.items() if len(contents) > 1)

    # Out-of-order source sequencing: source_row_index should be non-decreasing
    # when bars are read in the provider's own committed order (source_sequence
    # was assigned deterministically by the adapter; a genuine re-ordering here
    # indicates the store or an upstream stage reordered rows unexpectedly).
    by_source_order = bars
    out_of_order_count = 0
    previous_index: int | None = None
    for bar in by_source_order:
        if previous_index is not None and bar.source_row_index < previous_index:
            out_of_order_count += 1
        previous_index = bar.source_row_index

    present_dates = {b.open_time.date() for b in ordered}
    coverage_start = ordered[0].open_time.date()
    coverage_end = ordered[-1].open_time.date()

    missing_days: list[str] = []
    if not session_policy.is_24_hour_session:
        for day in _business_days_between(coverage_start, coverage_end):
            if day not in present_dates:
                missing_days.append(day.isoformat())
                if len(missing_days) >= _MAX_REPORTED_MISSING_DAYS:
                    break
    # 24-hour sessions (e.g. OTC spot) have no weekday-closure assumption to lean
    # on -- this phase reports zero missing-bar candidates for them rather than
    # inventing an unverified continuous-session expectation (honest limitation,
    # not a false "complete" signal; see module docstring).

    return GapAnalysisReport(
        canonical_driver_id=next(iter(driver_ids)), provider=next(iter(providers)), provider_symbol=next(iter(symbols)),
        bar_count=len(bars), missing_business_day_count=len(missing_days), missing_business_days=tuple(missing_days),
        conflicting_coordinate_count=len(conflicting), conflicting_coordinates=tuple(conflicting), out_of_order_count=out_of_order_count,
        calendar_assurance="limited",
    )
