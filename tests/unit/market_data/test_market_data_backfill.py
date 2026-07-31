"""Unit tests for `market_data.backfill` (Milestone 10, Phase 3): pure
backfill plan determinism, overlap policy, gap policy (including
calendar-aware gap filtering), and batch ordering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import BackfillPlanError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import NamedZoneTimezone
from quant_platform.market_data.backfill import (
    GAP_CALENDAR_OPEN,
    GAP_REJECTED,
    OVERLAP_REJECTED,
    BackfillPlan,
    BatchOrderingPolicy,
    GapPolicy,
    OverlapPolicy,
    create_backfill_plan,
)
from quant_platform.market_data.calendar import default_xauusd_calendar
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, create_source_manifest

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="offline_csv")
_PARTITIONING = PartitioningSpec(granularity=PartitionGranularity.DAILY)


def _manifest(expected_start: datetime | None = None, expected_end: datetime | None = None) -> object:
    return create_source_manifest(
        source_name="test", source_kind=SourceKind.CSV_CANDLES, source_schema_version=1, record_kind=RecordKind.CANDLE,
        source_label="x.csv", content_digest="a" * 64, byte_size=10, encoding="utf-8", instrument_mapping_id="b" * 64,
        timezone_policy_id="c" * 64, unit_normalization_version=1, creation_time=_T0, expected_timeframe=Timeframe.D1,
        expected_start=expected_start, expected_end=expected_end,
    )


def _plan(**overrides: object) -> BackfillPlan:
    kwargs: dict[str, object] = {
        "source_manifest": _manifest(), "target_dataset_key": _KEY, "requested_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "requested_end": datetime(2024, 1, 4, tzinfo=timezone.utc), "existing_covered_partition_keys": frozenset(),
        "partitioning": _PARTITIONING, "overlap_policy": OverlapPolicy.REJECT_ANY_OVERLAP, "gap_policy": GapPolicy.ALLOW_AND_REPORT,
        "instrument_mapping_id": "b" * 64, "timeframe_mapping_id": None, "timezone_policy_id": "c" * 64, "creation_time": _T0,
    }
    kwargs.update(overrides)
    return create_backfill_plan(**kwargs)  # type: ignore[arg-type]


class TestPlanIdentity:
    def test_deterministic_for_same_inputs(self) -> None:
        assert _plan().backfill_plan_id == _plan().backfill_plan_id

    def test_creation_time_excluded_from_identity(self) -> None:
        p1 = _plan(creation_time=_T0)
        p2 = _plan(creation_time=datetime(2030, 1, 1, tzinfo=timezone.utc))
        assert p1.backfill_plan_id == p2.backfill_plan_id

    def test_different_overlap_policy_changes_id(self) -> None:
        p1 = _plan(overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP)
        p2 = _plan(overlap_policy=OverlapPolicy.ALLOW_LATE_ARRIVAL_NEW_VERSION)
        assert p1.backfill_plan_id != p2.backfill_plan_id

    def test_same_batches_ordered_deterministically(self) -> None:
        p1 = _plan()
        p2 = _plan()
        assert [b.partition_key for b in p1.batches] == [b.partition_key for b in p2.batches]

    def test_round_trip(self) -> None:
        plan = _plan()
        assert BackfillPlan.from_json_dict(plan.to_json_dict()) == plan


class TestFreshDatasetPlan:
    def test_no_overlap_all_missing(self) -> None:
        plan = _plan()
        assert plan.is_admissible
        assert plan.expected_partitions_touched == ("2024-01-01", "2024-01-02", "2024-01-03")
        assert len(plan.batches) == 3
        assert all(not b.already_covered for b in plan.batches)
        assert plan.overlapping_intervals == ()

    def test_invalid_interval_rejected(self) -> None:
        with pytest.raises(BackfillPlanError):
            _plan(requested_start=datetime(2024, 1, 4, tzinfo=timezone.utc), requested_end=datetime(2024, 1, 1, tzinfo=timezone.utc))


class TestOverlapPolicy:
    def test_reject_any_overlap_blocks_plan(self) -> None:
        plan = _plan(existing_covered_partition_keys=frozenset({"2024-01-02"}), overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP)
        assert not plan.is_admissible
        assert OVERLAP_REJECTED in plan.blocking_issue_codes

    def test_exact_duplicates_only_allows_with_warning(self) -> None:
        plan = _plan(existing_covered_partition_keys=frozenset({"2024-01-02"}), overlap_policy=OverlapPolicy.EXACT_DUPLICATES_ONLY)
        assert plan.is_admissible
        assert len(plan.warnings) == 1

    def test_allow_late_arrival_allows_with_no_warning(self) -> None:
        plan = _plan(existing_covered_partition_keys=frozenset({"2024-01-02"}), overlap_policy=OverlapPolicy.ALLOW_LATE_ARRIVAL_NEW_VERSION)
        assert plan.is_admissible
        assert plan.warnings == ()

    def test_overlap_policy_participates_in_plan_identity(self) -> None:
        p1 = _plan(existing_covered_partition_keys=frozenset({"2024-01-02"}), overlap_policy=OverlapPolicy.EXACT_DUPLICATES_ONLY)
        p2 = _plan(existing_covered_partition_keys=frozenset({"2024-01-02"}), overlap_policy=OverlapPolicy.ALLOW_LATE_ARRIVAL_NEW_VERSION)
        assert p1.backfill_plan_id != p2.backfill_plan_id


class TestGapPolicy:
    def test_gap_reported_but_not_blocking(self) -> None:
        manifest = _manifest(expected_start=datetime(2024, 1, 1, tzinfo=timezone.utc), expected_end=datetime(2024, 1, 2, tzinfo=timezone.utc))
        plan = _plan(source_manifest=manifest, gap_policy=GapPolicy.ALLOW_AND_REPORT)
        assert plan.is_admissible
        assert len(plan.gap_intervals) == 2

    def test_gap_rejected_blocks_plan(self) -> None:
        manifest = _manifest(expected_start=datetime(2024, 1, 1, tzinfo=timezone.utc), expected_end=datetime(2024, 1, 2, tzinfo=timezone.utc))
        plan = _plan(source_manifest=manifest, gap_policy=GapPolicy.REJECT)
        assert not plan.is_admissible
        assert GAP_REJECTED in plan.blocking_issue_codes

    def test_no_expected_range_means_no_gap_analysis(self) -> None:
        plan = _plan(source_manifest=_manifest(), gap_policy=GapPolicy.REJECT)
        assert plan.is_admissible
        assert plan.gap_intervals == ()

    def test_calendar_required_when_policy_needs_it(self) -> None:
        with pytest.raises(BackfillPlanError):
            _plan(gap_policy=GapPolicy.REQUIRE_EXPECTED_MARKET_CALENDAR)

    def test_calendar_forbidden_when_policy_does_not_need_it(self) -> None:
        calendar = default_xauusd_calendar(NamedZoneTimezone(key="UTC"))
        with pytest.raises(BackfillPlanError):
            _plan(gap_policy=GapPolicy.ALLOW_AND_REPORT, calendar=calendar)

    def test_weekend_gap_not_blocking_under_calendar_policy(self) -> None:
        calendar = default_xauusd_calendar(NamedZoneTimezone(key="UTC"))
        manifest = _manifest(expected_start=datetime(2024, 1, 1, tzinfo=timezone.utc), expected_end=datetime(2024, 1, 6, tzinfo=timezone.utc))
        plan = _plan(
            source_manifest=manifest, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc), requested_end=datetime(2024, 1, 8, tzinfo=timezone.utc),
            gap_policy=GapPolicy.REQUIRE_EXPECTED_MARKET_CALENDAR, calendar=calendar,
        )
        # 2024-01-06/07 is a weekend -- the XAUUSD calendar is closed then.
        assert plan.is_admissible
        assert GAP_CALENDAR_OPEN not in plan.blocking_issue_codes

    def test_weekday_gap_blocking_under_calendar_policy(self) -> None:
        calendar = default_xauusd_calendar(NamedZoneTimezone(key="UTC"))
        manifest = _manifest(expected_start=datetime(2024, 1, 1, tzinfo=timezone.utc), expected_end=datetime(2024, 1, 2, tzinfo=timezone.utc))
        plan = _plan(
            source_manifest=manifest, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc), requested_end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            gap_policy=GapPolicy.REQUIRE_EXPECTED_MARKET_CALENDAR, calendar=calendar,
        )
        # 2024-01-02 is a Tuesday -- the market is expected to be open.
        assert not plan.is_admissible
        assert GAP_CALENDAR_OPEN in plan.blocking_issue_codes


class TestBatchOrdering:
    def test_chronological_ascending_is_default(self) -> None:
        plan = _plan()
        assert [b.partition_key for b in plan.batches] == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert [b.batch_index for b in plan.batches] == [0, 1, 2]

    def test_chronological_descending(self) -> None:
        plan = _plan(ordering_policy=BatchOrderingPolicy.CHRONOLOGICAL_DESCENDING)
        assert [b.partition_key for b in plan.batches] == ["2024-01-03", "2024-01-02", "2024-01-01"]
        assert [b.batch_index for b in plan.batches] == [0, 1, 2]


class TestNoFilesystemWritesInPlanning:
    def test_planning_touches_no_filesystem(self) -> None:
        # create_backfill_plan takes no filesystem-facing parameter at
        # all (existing_covered_partition_keys is a plain frozenset), so
        # there is structurally nothing for it to write -- this test just
        # confirms the call succeeds without any repository/root argument.
        plan = _plan()
        assert plan.backfill_plan_id
