"""Source-coverage policy (Milestone 10, Phase 4D, spec Section 10):
decides, BEFORE the expensive `features.engine.FeatureEngine.compute`
call, whether every bound source has enough coverage of the requested
`[start, end)` range to safely proceed -- and, for `QUARANTINE_INTERVAL`,
inspects an already-computed missing-value indicator AFTER the engine
has run to identify runs that exceed a configured tolerance.

FOUR POLICY KINDS (spec's own vocabulary, no others invented):

  * `FAIL_REQUIRED_SOURCE` -- any REQUIRED source (a `MacroDatasetBinding`/
    `CrossAssetDatasetBinding` with `required=True`) whose own coverage
    does not overlap the requested range at all, or overlaps less than
    `minimum_observation_coverage_fraction`, raises `SourceCoverageError`
    immediately. This is the default-safe, fail-closed policy.
  * `ALLOW_OPTIONAL_MISSING_AND_REPORT` -- identical check, but an
    OPTIONAL source's shortfall is recorded as a `CoverageFinding` with
    `status="insufficient"` and never raises.
  * `TRIM_TO_COMMON_SAFE_RANGE` -- computes the intersection of every
    REQUIRED source's own coverage with the requested range and returns
    it as `CoverageReport.safe_start`/`safe_end`, with `trimmed=True` and
    an explicit `trim_reason` recording exactly which source forced the
    trim and by how much. NEVER trims silently -- the report is always
    the caller's evidence for what happened and why.
  * `QUARANTINE_INTERVAL` -- evaluated separately, post-hoc, by
    `evaluate_missing_runs` against an already-computed boolean missing
    indicator (e.g. `features.engine.FeatureComputationResult.
    missing_indicators[name]`): identifies every maximal run of
    consecutive `True` values whose length exceeds
    `SourceCoveragePolicy.max_consecutive_missing_aligned_rows` and
    returns them as explicit `(start_index, end_index)` intervals for the
    caller to act on (drop, flag, or refuse) -- this module never drops
    rows itself.

Coverage-range computation is source-kind-specific: base-asset coverage
is `[base_df.open_time.min(), base_df.open_time.max() + timeframe.duration)`
(the LAST bar's own close time, since that bar's data is fully covered
through its close); macro coverage is
`[macro_df.release_time.min(), release_time.max()]`; cross-asset coverage
is `[cross_df.open_time.min(), open_time.max()]` (already the
availability-shifted coordinate `cross_asset_adapter.py` produces).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import SourceCoverageError
from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.bindings import CrossAssetDatasetBinding, MacroDatasetBinding

__all__ = [
    "CoverageFinding",
    "CoverageReport",
    "QuarantineInterval",
    "SourceCoveragePolicy",
    "SourceCoveragePolicyKind",
    "evaluate_missing_runs",
    "evaluate_source_coverage",
]


class SourceCoveragePolicyKind(Enum):
    FAIL_REQUIRED_SOURCE = "fail_required_source"
    ALLOW_OPTIONAL_MISSING_AND_REPORT = "allow_optional_missing_and_report"
    TRIM_TO_COMMON_SAFE_RANGE = "trim_to_common_safe_range"
    QUARANTINE_INTERVAL = "quarantine_interval"


@dataclass(frozen=True, slots=True)
class SourceCoveragePolicy:
    kind: SourceCoveragePolicyKind
    minimum_observation_coverage_fraction: float = 0.0
    max_consecutive_missing_aligned_rows: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum_observation_coverage_fraction <= 1.0):
            raise ValueError(
                f"minimum_observation_coverage_fraction must be in [0, 1], got {self.minimum_observation_coverage_fraction}"
            )
        if self.max_consecutive_missing_aligned_rows is not None and self.max_consecutive_missing_aligned_rows < 0:
            raise ValueError("max_consecutive_missing_aligned_rows must be >= 0 when set")


@dataclass(frozen=True, slots=True)
class CoverageFinding:
    source_kind: str
    """`"base"`, `"macro"`, or `"cross_asset"`."""
    source_name: str
    required: bool
    covered_start: pd.Timestamp | None
    covered_end: pd.Timestamp | None
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    coverage_fraction: float
    status: str
    """`"ok"`, `"insufficient"`, or `"missing"` (zero overlap, or the
    source frame was empty)."""


@dataclass(frozen=True, slots=True)
class CoverageReport:
    findings: tuple[CoverageFinding, ...]
    safe_start: pd.Timestamp
    safe_end: pd.Timestamp
    trimmed: bool
    trim_reason: str | None = None

    @property
    def has_insufficient_optional(self) -> bool:
        return any(f.status != "ok" and not f.required for f in self.findings)


def _overlap_fraction(
    covered_start: pd.Timestamp | None, covered_end: pd.Timestamp | None, requested_start: pd.Timestamp, requested_end: pd.Timestamp,
) -> float:
    if covered_start is None or covered_end is None or covered_end < covered_start:
        return 0.0
    overlap_start = max(covered_start, requested_start)
    overlap_end = min(covered_end, requested_end)
    requested_span = (requested_end - requested_start).total_seconds()
    if requested_span <= 0:
        return 0.0
    overlap_span = max(0.0, (overlap_end - overlap_start).total_seconds())
    return overlap_span / requested_span


def _finding(
    *, source_kind: str, source_name: str, required: bool, covered_start: pd.Timestamp | None, covered_end: pd.Timestamp | None,
    requested_start: pd.Timestamp, requested_end: pd.Timestamp, minimum_fraction: float,
) -> CoverageFinding:
    fraction = _overlap_fraction(covered_start, covered_end, requested_start, requested_end)
    if covered_start is None or covered_end is None:
        status = "missing"
    elif fraction < minimum_fraction:
        status = "insufficient"
    else:
        status = "ok"
    return CoverageFinding(
        source_kind=source_kind, source_name=source_name, required=required, covered_start=covered_start, covered_end=covered_end,
        requested_start=requested_start, requested_end=requested_end, coverage_fraction=fraction, status=status,
    )


def evaluate_source_coverage(
    *, base_df: pd.DataFrame, base_timeframe: Timeframe, macro_frames: dict[str, pd.DataFrame],
    macro_bindings: dict[str, MacroDatasetBinding], cross_asset_frames: dict[str, pd.DataFrame],
    cross_asset_bindings: dict[str, CrossAssetDatasetBinding], requested_start: pd.Timestamp, requested_end: pd.Timestamp,
    policy: SourceCoveragePolicy,
) -> CoverageReport:
    """Pre-flight coverage evaluation over already-resolved source frames
    (as produced by `base_asset_adapter`/`macro_adapter`/
    `cross_asset_adapter`) -- never re-reads `market_data` itself."""
    findings: list[CoverageFinding] = []

    base_covered_start = pd.Timestamp(base_df["open_time"].min()) if len(base_df) else None
    base_covered_end = (
        pd.Timestamp(base_df["open_time"].max()) + base_timeframe.duration if len(base_df) else None
    )
    findings.append(
        _finding(
            source_kind="base", source_name="base", required=True, covered_start=base_covered_start, covered_end=base_covered_end,
            requested_start=requested_start, requested_end=requested_end, minimum_fraction=policy.minimum_observation_coverage_fraction,
        )
    )

    for name, frame in sorted(macro_frames.items()):
        macro_binding = macro_bindings.get(name)
        required = macro_binding.required if macro_binding is not None else True
        covered_start = pd.Timestamp(frame["release_time"].min()) if len(frame) else None
        covered_end = pd.Timestamp(frame["release_time"].max()) if len(frame) else None
        findings.append(
            _finding(
                source_kind="macro", source_name=name, required=required, covered_start=covered_start, covered_end=covered_end,
                requested_start=requested_start, requested_end=requested_end, minimum_fraction=policy.minimum_observation_coverage_fraction,
            )
        )

    for name, frame in sorted(cross_asset_frames.items()):
        cross_binding = cross_asset_bindings.get(name)
        required = cross_binding.required if cross_binding is not None else False
        covered_start = pd.Timestamp(frame["open_time"].min()) if len(frame) else None
        covered_end = pd.Timestamp(frame["open_time"].max()) if len(frame) else None
        findings.append(
            _finding(
                source_kind="cross_asset", source_name=name, required=required, covered_start=covered_start, covered_end=covered_end,
                requested_start=requested_start, requested_end=requested_end, minimum_fraction=policy.minimum_observation_coverage_fraction,
            )
        )

    required_bad = [f for f in findings if f.required and f.status != "ok"]

    if policy.kind is SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE and required_bad:
        worst = required_bad[0]
        raise SourceCoverageError(
            f"Required source {worst.source_kind}:{worst.source_name!r} has {worst.status} coverage of the "
            f"requested range [{requested_start}, {requested_end}) (covered=[{worst.covered_start}, "
            f"{worst.covered_end}], fraction={worst.coverage_fraction:.3f}, minimum required="
            f"{policy.minimum_observation_coverage_fraction:.3f}).",
            context={
                "source_kind": worst.source_kind, "source_name": worst.source_name, "status": worst.status,
                "coverage_fraction": worst.coverage_fraction,
            },
        )

    if policy.kind is SourceCoveragePolicyKind.TRIM_TO_COMMON_SAFE_RANGE:
        safe_start, safe_end = requested_start, requested_end
        trim_reason: str | None = None
        for finding in findings:
            if not finding.required or finding.covered_start is None or finding.covered_end is None:
                continue
            new_start = max(safe_start, finding.covered_start)
            new_end = min(safe_end, finding.covered_end)
            if new_start != safe_start or new_end != safe_end:
                trim_reason = (
                    f"trimmed to [{new_start}, {new_end}) because required source "
                    f"{finding.source_kind}:{finding.source_name!r} only covers [{finding.covered_start}, {finding.covered_end}]"
                )
            safe_start, safe_end = new_start, new_end
        if safe_end <= safe_start:
            raise SourceCoverageError(
                f"TRIM_TO_COMMON_SAFE_RANGE: required sources have no common overlap within the requested range "
                f"[{requested_start}, {requested_end})",
                context={"requested_start": str(requested_start), "requested_end": str(requested_end)},
            )
        return CoverageReport(
            findings=tuple(findings), safe_start=safe_start, safe_end=safe_end, trimmed=trim_reason is not None, trim_reason=trim_reason
        )

    return CoverageReport(findings=tuple(findings), safe_start=requested_start, safe_end=requested_end, trimmed=False, trim_reason=None)


@dataclass(frozen=True, slots=True)
class QuarantineInterval:
    start_index: int
    end_index: int
    """Inclusive `[start_index, end_index]` row-position range within the
    missing indicator's own index."""
    run_length: int


def evaluate_missing_runs(
    missing_indicator: pd.Series, *, max_consecutive_missing_aligned_rows: int | None,
) -> tuple[QuarantineInterval, ...]:
    """`QUARANTINE_INTERVAL` policy evaluation: every maximal run of
    consecutive `True` values in `missing_indicator` (row-position order,
    never label/index-value order) whose length exceeds
    `max_consecutive_missing_aligned_rows`. Returns an empty tuple (never
    raises) if `max_consecutive_missing_aligned_rows` is `None` -- the
    caller decides whether a non-empty result blocks the build."""
    if max_consecutive_missing_aligned_rows is None:
        return ()
    values = missing_indicator.reset_index(drop=True).to_numpy(dtype=bool)
    intervals: list[QuarantineInterval] = []
    run_start: int | None = None
    for position, is_missing in enumerate(values):
        if is_missing and run_start is None:
            run_start = position
        elif not is_missing and run_start is not None:
            run_length = position - run_start
            if run_length > max_consecutive_missing_aligned_rows:
                intervals.append(QuarantineInterval(start_index=run_start, end_index=position - 1, run_length=run_length))
            run_start = None
    if run_start is not None:
        run_length = len(values) - run_start
        if run_length > max_consecutive_missing_aligned_rows:
            intervals.append(QuarantineInterval(start_index=run_start, end_index=len(values) - 1, run_length=run_length))
    return tuple(intervals)
