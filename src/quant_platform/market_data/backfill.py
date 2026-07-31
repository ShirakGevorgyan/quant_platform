"""Pure, deterministic backfill planning (Milestone 10, Phase 3). Given a
source manifest, a target dataset, a requested interval, and the
CALLER-SUPPLIED current durable coverage of that dataset (a plain value
-- this module performs NO filesystem or repository I/O of its own; a
caller reads current coverage via `MarketDataRepository`/
`DatasetManifestStore`/`PartitionStore` first and hands the result in),
`create_backfill_plan` always produces the identical `BackfillPlan`
(same `backfill_plan_id`, same ordered `batches`) for the identical
inputs -- the specification's own "same inputs -> same plan identity and
ordered batches" requirement, satisfied structurally the same way every
other identity in this package is: `to_identity_payload()` excludes only
`backfill_plan_id` itself and `creation_time` (operational).

GRANULARITY: every interval this module reports (`missing_intervals`,
`overlapping_intervals`, `already_covered_intervals`, each `BackfillBatch`)
is aligned to a whole PARTITION -- never a sub-partition slice -- because
Phase 2's own ingestion model (`ingestion.rebuild_touched_partitions`)
always rebuilds an entire touched partition from its current full
membership; a plan describing anything finer would misrepresent what
actually gets rebuilt. "Existing durable dataset state" is therefore
expressed as `existing_covered_partition_keys: frozenset[str]` -- the set
of partition keys the target dataset already has non-empty data for --
not a list of exact event timestamps.

OVERLAP POLICY, AS IMPLEMENTED: `REJECT_ANY_OVERLAP` is the only overlap
policy fully decidable at PURE PLANNING time (a plan touching any already-
covered partition is inadmissible, no row content needed). The other two
policies both keep the plan admissible but differ in the RUNTIME CONTRACT
`orchestration.py` must honor when it later processes an overlapping
partition's rows: `EXACT_DUPLICATES_ONLY` requires orchestration to
verify every row landing in an overlapping partition is byte-identical to
already-durable data (any genuine difference must fail closed there, not
here -- planning cannot know row content in advance); `ALLOW_LATE_ARRIVAL_
NEW_VERSION` permits new, non-duplicate content in an overlapping
partition, expected to produce Phase 2's own late-arrival new manifest
version. `EXACT_DUPLICATES_ONLY` overlap is flagged with a warning noting
this deferred contract; `ALLOW_LATE_ARRIVAL_NEW_VERSION` is not (it is
the designed-for case).

GAP POLICY, AS IMPLEMENTED: a "gap" is the portion of this plan's MISSING
(not-yet-covered) partitions that ALSO falls entirely outside the source
manifest's own declared `[expected_start, expected_end)` range -- i.e. a
hole this specific source cannot fill at all, computed purely from
`SourceManifest` fields (no row data needed). If the source declares no
expected range, gap analysis is skipped (nothing to compare against).
`allow_and_report` reports every such gap as a warning without blocking;
`reject` blocks the plan on any such gap; `require_expected_market_
calendar` re-checks each gap against `calendar.TradingCalendar` (reused
directly, never re-derived -- see `calendar.py`'s own docstring) via
`enumerate_expected_open_times`, using `SourceManifest.expected_timeframe`
(required for this policy) -- only a gap the market was actually expected
to be OPEN and producing bars during blocks the plan; a gap entirely
within market-closed time (a weekend, a holiday) is not a real problem.
This still does not assume 24/7 trading for every instrument, and it
inherits every documented limitation of `default_xauusd_calendar`/
`TradingCalendar` for OTC/provider-specific sessions (see `calendar.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import BackfillPlanError
from quant_platform.market_data.calendar import TradingCalendar, enumerate_expected_open_times
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.manifests import DatasetKey, PartitioningSpec
from quant_platform.market_data.partitions import partition_bounds, partition_key_for
from quant_platform.market_data.source_manifests import SourceManifest

__all__ = [
    "BACKFILL_PLAN_KIND",
    "GAP_CALENDAR_OPEN",
    "GAP_REJECTED",
    "OVERLAP_REJECTED",
    "BackfillBatch",
    "BackfillPlan",
    "BatchOrderingPolicy",
    "GapPolicy",
    "OverlapPolicy",
    "create_backfill_plan",
]

BACKFILL_PLAN_KIND = "backfill_plan"
_MAX_TOUCHED_PARTITIONS = 100_000
"""Mirrors `calendar.enumerate_expected_open_times`'s own finite-bound
philosophy: a plan spanning this many partitions is almost certainly a
caller error (a mis-specified interval), and must fail loudly rather
than hang."""

OVERLAP_REJECTED = "overlap_rejected"
GAP_REJECTED = "gap_rejected"
GAP_CALENDAR_OPEN = "gap_calendar_open"


class OverlapPolicy(Enum):
    EXACT_DUPLICATES_ONLY = "exact_duplicates_only"
    REJECT_ANY_OVERLAP = "reject_any_overlap"
    ALLOW_LATE_ARRIVAL_NEW_VERSION = "allow_late_arrival_new_version"


class GapPolicy(Enum):
    ALLOW_AND_REPORT = "allow_and_report"
    REJECT = "reject"
    REQUIRE_EXPECTED_MARKET_CALENDAR = "require_expected_market_calendar"


class BatchOrderingPolicy(Enum):
    CHRONOLOGICAL_ASCENDING = "chronological_ascending"
    CHRONOLOGICAL_DESCENDING = "chronological_descending"


@dataclass(frozen=True, slots=True)
class BackfillBatch:
    batch_index: int
    partition_key: str
    start_time: datetime
    end_time: datetime
    already_covered: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index, "partition_key": self.partition_key,
            "start_time": serialize_timestamp(self.start_time, field_name="start_time"),
            "end_time": serialize_timestamp(self.end_time, field_name="end_time"), "already_covered": self.already_covered,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BackfillBatch:
        return cls(
            batch_index=int(str(raw["batch_index"])), partition_key=str(raw["partition_key"]),
            start_time=deserialize_timestamp(raw["start_time"], field_name="start_time"),
            end_time=deserialize_timestamp(raw["end_time"], field_name="end_time"), already_covered=bool(raw["already_covered"]),
        )


def _interval_json(interval: tuple[datetime, datetime]) -> dict[str, object]:
    return {"start": serialize_timestamp(interval[0], field_name="start"), "end": serialize_timestamp(interval[1], field_name="end")}


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    backfill_plan_id: str
    source_manifest_id: str
    target_dataset_key: DatasetKey
    requested_start: datetime
    requested_end: datetime
    overlap_policy: OverlapPolicy
    gap_policy: GapPolicy
    ordering_policy: BatchOrderingPolicy
    partitioning: PartitioningSpec
    instrument_mapping_id: str
    timeframe_mapping_id: str | None
    timezone_policy_id: str
    already_covered_intervals: tuple[tuple[datetime, datetime], ...]
    missing_intervals: tuple[tuple[datetime, datetime], ...]
    overlapping_intervals: tuple[tuple[datetime, datetime], ...]
    gap_intervals: tuple[tuple[datetime, datetime], ...]
    expected_partitions_touched: tuple[str, ...]
    estimated_row_count: int | None
    batches: tuple[BackfillBatch, ...]
    warnings: tuple[str, ...]
    blocking_issue_codes: tuple[str, ...]
    is_admissible: bool
    creation_time: datetime

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": BACKFILL_PLAN_KIND, "backfill_plan_id": self.backfill_plan_id, "source_manifest_id": self.source_manifest_id,
            "target_dataset_key": self.target_dataset_key.to_json_dict(),
            "requested_start": serialize_timestamp(self.requested_start, field_name="requested_start"),
            "requested_end": serialize_timestamp(self.requested_end, field_name="requested_end"),
            "overlap_policy": self.overlap_policy.value, "gap_policy": self.gap_policy.value,
            "ordering_policy": self.ordering_policy.value, "partitioning": self.partitioning.to_json_dict(),
            "instrument_mapping_id": self.instrument_mapping_id,
            "timeframe_mapping_id": self.timeframe_mapping_id, "timezone_policy_id": self.timezone_policy_id,
            "already_covered_intervals": [_interval_json(i) for i in self.already_covered_intervals],
            "missing_intervals": [_interval_json(i) for i in self.missing_intervals],
            "overlapping_intervals": [_interval_json(i) for i in self.overlapping_intervals],
            "gap_intervals": [_interval_json(i) for i in self.gap_intervals],
            "expected_partitions_touched": list(self.expected_partitions_touched), "estimated_row_count": self.estimated_row_count,
            "batches": [b.to_json_dict() for b in self.batches], "warnings": list(self.warnings),
            "blocking_issue_codes": list(self.blocking_issue_codes), "is_admissible": self.is_admissible,
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["backfill_plan_id"]
        del payload["creation_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BackfillPlan:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        return cls(
            backfill_plan_id=str(raw["backfill_plan_id"]), source_manifest_id=str(raw["source_manifest_id"]),
            target_dataset_key=DatasetKey.from_json_dict(as_json_dict(raw["target_dataset_key"], field_name="target_dataset_key")),
            requested_start=deserialize_timestamp(raw["requested_start"], field_name="requested_start"),
            requested_end=deserialize_timestamp(raw["requested_end"], field_name="requested_end"),
            overlap_policy=OverlapPolicy(raw["overlap_policy"]), gap_policy=GapPolicy(raw["gap_policy"]),
            ordering_policy=BatchOrderingPolicy(raw["ordering_policy"]),
            partitioning=PartitioningSpec.from_json_dict(as_json_dict(raw["partitioning"], field_name="partitioning")),
            instrument_mapping_id=str(raw["instrument_mapping_id"]),
            timeframe_mapping_id=(None if raw.get("timeframe_mapping_id") is None else str(raw["timeframe_mapping_id"])),
            timezone_policy_id=str(raw["timezone_policy_id"]),
            already_covered_intervals=tuple(
                _parse_interval_json(as_json_dict(i, field_name="interval")) for i in as_json_list(raw["already_covered_intervals"], field_name="already_covered_intervals")
            ),
            missing_intervals=tuple(
                _parse_interval_json(as_json_dict(i, field_name="interval")) for i in as_json_list(raw["missing_intervals"], field_name="missing_intervals")
            ),
            overlapping_intervals=tuple(
                _parse_interval_json(as_json_dict(i, field_name="interval")) for i in as_json_list(raw["overlapping_intervals"], field_name="overlapping_intervals")
            ),
            gap_intervals=tuple(
                _parse_interval_json(as_json_dict(i, field_name="interval")) for i in as_json_list(raw["gap_intervals"], field_name="gap_intervals")
            ),
            expected_partitions_touched=tuple(str(p) for p in as_json_list(raw["expected_partitions_touched"], field_name="expected_partitions_touched")),
            estimated_row_count=(None if raw.get("estimated_row_count") is None else int(str(raw["estimated_row_count"]))),
            batches=tuple(BackfillBatch.from_json_dict(as_json_dict(b, field_name="batch")) for b in as_json_list(raw["batches"], field_name="batches")),
            warnings=tuple(str(w) for w in as_json_list(raw["warnings"], field_name="warnings")),
            blocking_issue_codes=tuple(str(c) for c in as_json_list(raw["blocking_issue_codes"], field_name="blocking_issue_codes")),
            is_admissible=bool(raw["is_admissible"]), creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
        )


def _parse_interval_json(raw: dict[str, object]) -> tuple[datetime, datetime]:
    return (
        deserialize_timestamp(raw["start"], field_name="start"),
        deserialize_timestamp(raw["end"], field_name="end"),
    )


def _touched_partition_keys(start: datetime, end: datetime, partitioning: PartitioningSpec) -> tuple[str, ...]:
    keys: list[str] = []
    cursor = start
    iterations = 0
    while cursor < end:
        key = partition_key_for(cursor, partitioning)
        keys.append(key)
        _, next_cursor = partition_bounds(key, partitioning)
        if next_cursor <= cursor:
            raise BackfillPlanError(f"partition_bounds did not advance past cursor {cursor} for partition_key {key!r}")
        cursor = next_cursor
        iterations += 1
        if iterations > _MAX_TOUCHED_PARTITIONS:
            raise BackfillPlanError(
                f"requested interval [{start}, {end}) would touch more than {_MAX_TOUCHED_PARTITIONS} partitions -- "
                "this is almost certainly a mis-specified interval"
            )
    return tuple(keys)


def _subtract_interval(
    window: tuple[datetime, datetime], covering: tuple[datetime, datetime],
) -> tuple[tuple[datetime, datetime], ...]:
    """`window` minus `covering` -- 0, 1, or 2 resulting sub-intervals of
    `window` not covered by `covering`."""
    window_start, window_end = window
    cover_start, cover_end = covering
    if cover_end <= window_start or cover_start >= window_end:
        return (window,)
    remainders: list[tuple[datetime, datetime]] = []
    if cover_start > window_start:
        remainders.append((window_start, min(cover_start, window_end)))
    if cover_end < window_end:
        remainders.append((max(cover_end, window_start), window_end))
    return tuple(remainders)


def create_backfill_plan(
    *,
    source_manifest: SourceManifest,
    target_dataset_key: DatasetKey,
    requested_start: datetime,
    requested_end: datetime,
    existing_covered_partition_keys: frozenset[str],
    partitioning: PartitioningSpec,
    overlap_policy: OverlapPolicy,
    gap_policy: GapPolicy,
    instrument_mapping_id: str,
    timeframe_mapping_id: str | None,
    timezone_policy_id: str,
    creation_time: datetime,
    ordering_policy: BatchOrderingPolicy = BatchOrderingPolicy.CHRONOLOGICAL_ASCENDING,
    calendar: TradingCalendar | None = None,
) -> BackfillPlan:
    require_tz_aware(requested_start, field_name="requested_start")
    require_tz_aware(requested_end, field_name="requested_end")
    if requested_end <= requested_start:
        raise BackfillPlanError(f"requested_end ({requested_end}) must be after requested_start ({requested_start})")
    require_tz_aware(creation_time, field_name="creation_time")
    if gap_policy is GapPolicy.REQUIRE_EXPECTED_MARKET_CALENDAR:
        if calendar is None:
            raise BackfillPlanError("gap_policy=REQUIRE_EXPECTED_MARKET_CALENDAR requires a calendar to be supplied")
        if source_manifest.expected_timeframe is None:
            raise BackfillPlanError("gap_policy=REQUIRE_EXPECTED_MARKET_CALENDAR requires source_manifest.expected_timeframe to be set")
    elif calendar is not None:
        raise BackfillPlanError(f"a calendar was supplied but gap_policy is {gap_policy.value!r}, not REQUIRE_EXPECTED_MARKET_CALENDAR")

    touched_keys = _touched_partition_keys(requested_start, requested_end, partitioning)
    missing_intervals: list[tuple[datetime, datetime]] = []
    overlapping_intervals: list[tuple[datetime, datetime]] = []
    batches: list[BackfillBatch] = []
    for index, key in enumerate(touched_keys):
        bounds = partition_bounds(key, partitioning)
        already_covered = key in existing_covered_partition_keys
        (overlapping_intervals if already_covered else missing_intervals).append(bounds)
        batches.append(BackfillBatch(batch_index=index, partition_key=key, start_time=bounds[0], end_time=bounds[1], already_covered=already_covered))

    if ordering_policy is BatchOrderingPolicy.CHRONOLOGICAL_DESCENDING:
        batches = list(reversed(batches))
        batches = [BackfillBatch(batch_index=i, partition_key=b.partition_key, start_time=b.start_time, end_time=b.end_time, already_covered=b.already_covered) for i, b in enumerate(batches)]

    gap_intervals: list[tuple[datetime, datetime]] = []
    if source_manifest.expected_start is not None and source_manifest.expected_end is not None:
        source_covered = (source_manifest.expected_start, source_manifest.expected_end)
        for missing_window in missing_intervals:
            gap_intervals.extend(_subtract_interval(missing_window, source_covered))

    warnings: list[str] = []
    blocking_issue_codes: list[str] = []

    if overlap_policy is OverlapPolicy.REJECT_ANY_OVERLAP and overlapping_intervals:
        blocking_issue_codes.append(OVERLAP_REJECTED)
        warnings.append(f"{len(overlapping_intervals)} partition(s) already covered; overlap_policy=REJECT_ANY_OVERLAP forbids any overlap")
    elif overlap_policy is OverlapPolicy.EXACT_DUPLICATES_ONLY and overlapping_intervals:
        warnings.append(
            f"{len(overlapping_intervals)} partition(s) already covered; overlap_policy=EXACT_DUPLICATES_ONLY requires "
            "orchestration to verify every row in these partitions is byte-identical to already-durable data"
        )

    if gap_intervals:
        if gap_policy is GapPolicy.REJECT:
            blocking_issue_codes.append(GAP_REJECTED)
            warnings.append(f"{len(gap_intervals)} gap interval(s) fall outside the source's declared expected range; gap_policy=REJECT forbids any gap")
        elif gap_policy is GapPolicy.ALLOW_AND_REPORT:
            warnings.append(f"{len(gap_intervals)} gap interval(s) fall outside the source's declared expected range")
        elif gap_policy is GapPolicy.REQUIRE_EXPECTED_MARKET_CALENDAR:
            assert calendar is not None and source_manifest.expected_timeframe is not None
            calendar_open_gaps = [
                gap for gap in gap_intervals
                if enumerate_expected_open_times(calendar, timeframe=source_manifest.expected_timeframe, start=gap[0], end=gap[1])
            ]
            if calendar_open_gaps:
                blocking_issue_codes.append(GAP_CALENDAR_OPEN)
                warnings.append(f"{len(calendar_open_gaps)} gap interval(s) fall within expected market-open time per the supplied calendar")

    is_admissible = not blocking_issue_codes

    provisional = BackfillPlan(
        backfill_plan_id="0" * 64, source_manifest_id=source_manifest.source_manifest_id, target_dataset_key=target_dataset_key,
        requested_start=requested_start, requested_end=requested_end, overlap_policy=overlap_policy, gap_policy=gap_policy,
        ordering_policy=ordering_policy, partitioning=partitioning, instrument_mapping_id=instrument_mapping_id,
        timeframe_mapping_id=timeframe_mapping_id, timezone_policy_id=timezone_policy_id, already_covered_intervals=tuple(overlapping_intervals),
        missing_intervals=tuple(missing_intervals), overlapping_intervals=tuple(overlapping_intervals),
        gap_intervals=tuple(gap_intervals), expected_partitions_touched=touched_keys, estimated_row_count=source_manifest.row_count,
        batches=tuple(batches), warnings=tuple(warnings), blocking_issue_codes=tuple(blocking_issue_codes), is_admissible=is_admissible,
        creation_time=creation_time,
    )
    backfill_plan_id = compute_content_id(BACKFILL_PLAN_KIND, provisional.to_identity_payload())
    return BackfillPlan(
        backfill_plan_id=backfill_plan_id, source_manifest_id=source_manifest.source_manifest_id, target_dataset_key=target_dataset_key,
        requested_start=requested_start, requested_end=requested_end, overlap_policy=overlap_policy, gap_policy=gap_policy,
        ordering_policy=ordering_policy, partitioning=partitioning, instrument_mapping_id=instrument_mapping_id,
        timeframe_mapping_id=timeframe_mapping_id, timezone_policy_id=timezone_policy_id, already_covered_intervals=tuple(overlapping_intervals),
        missing_intervals=tuple(missing_intervals), overlapping_intervals=tuple(overlapping_intervals),
        gap_intervals=tuple(gap_intervals), expected_partitions_touched=touched_keys, estimated_row_count=source_manifest.row_count,
        batches=tuple(batches), warnings=tuple(warnings), blocking_issue_codes=tuple(blocking_issue_codes), is_admissible=is_admissible,
        creation_time=creation_time,
    )
