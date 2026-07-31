"""Unit tests for `market_data.partitions`: daily/monthly boundary
correctness, deterministic (order-independent) partition membership,
identity mutation, and `PartitionStore`'s current-version-only storage."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import MarketDataPathSecurityError, PartitionError
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.partitions import (
    PartitionStore,
    build_partition,
    partition_bounds,
    partition_key_for,
)

_DAILY = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_MONTHLY = PartitioningSpec(granularity=PartitionGranularity.MONTHLY)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")


class TestPartitionKeyFor:
    def test_daily_key_format(self) -> None:
        assert partition_key_for(datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc), _DAILY) == "2026-01-05"

    def test_monthly_key_format(self) -> None:
        assert partition_key_for(datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc), _MONTHLY) == "2026-01"

    def test_exact_midnight_boundary_belongs_to_the_new_day(self) -> None:
        assert partition_key_for(datetime(2026, 1, 6, 0, 0, 0, tzinfo=timezone.utc), _DAILY) == "2026-01-06"

    def test_one_microsecond_before_midnight_belongs_to_the_previous_day(self) -> None:
        assert partition_key_for(datetime(2026, 1, 5, 23, 59, 59, 999999, tzinfo=timezone.utc), _DAILY) == "2026-01-05"

    def test_month_end_boundary(self) -> None:
        assert partition_key_for(datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc), _MONTHLY) == "2026-01"
        assert partition_key_for(datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc), _MONTHLY) == "2026-02"

    def test_non_utc_timezone_is_converted_before_bucketing(self) -> None:
        from datetime import timezone as tz

        # 23:30 UTC-5 == 04:30 UTC the next day.
        eastern = datetime(2026, 1, 5, 23, 30, tzinfo=tz(timedelta(hours=-5)))
        assert partition_key_for(eastern, _DAILY) == "2026-01-06"


class TestPartitionBounds:
    def test_daily_bounds_are_half_open_24_hours(self) -> None:
        start, end = partition_bounds("2026-01-05", _DAILY)
        assert start == datetime(2026, 1, 5, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 6, tzinfo=timezone.utc)

    def test_monthly_bounds_handle_variable_month_length(self) -> None:
        start, end = partition_bounds("2026-02", _MONTHLY)
        assert start == datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert end == datetime(2026, 3, 1, tzinfo=timezone.utc)

    def test_invalid_key_is_rejected(self) -> None:
        with pytest.raises(PartitionError):
            partition_bounds("not-a-date", _DAILY)


class TestBuildPartitionDeterministicMembership:
    def test_member_order_is_by_time_not_input_order(self) -> None:
        t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
        members = [("z", t0 + timedelta(hours=2)), ("a", t0), ("m", t0 + timedelta(hours=1))]
        partition = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=members)
        assert partition.ordered_member_ids == ("a", "m", "z")

    def test_identical_membership_in_different_input_order_produces_identical_id(self) -> None:
        t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
        members_a = [("1", t0), ("2", t0 + timedelta(hours=1))]
        members_b = [("2", t0 + timedelta(hours=1)), ("1", t0)]
        a = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=members_a)
        b = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=members_b)
        assert a.partition_id == b.partition_id

    def test_a_member_outside_bounds_is_rejected(self) -> None:
        outside = datetime(2026, 1, 6, 1, 0, tzinfo=timezone.utc)
        with pytest.raises(PartitionError):
            build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("a", outside)])

    def test_empty_members_is_rejected(self) -> None:
        with pytest.raises(PartitionError):
            build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[])

    def test_a_different_member_set_changes_the_partition_id(self) -> None:
        t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
        a = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0)])
        b = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0), ("2", t0 + timedelta(hours=1))])
        assert a.partition_id != b.partition_id  # partition identity mutation is detectable

    def test_round_trips_through_json(self) -> None:
        t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
        partition = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0)])
        from quant_platform.market_data.partitions import Partition

        assert Partition.from_json_dict(partition.to_json_dict()) == partition


class TestPartitionStore:
    def test_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
            partition = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0)])
            store.write(partition)
            assert store.read(_KEY, "2026-01-05") == partition

    def test_missing_partition_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            assert store.read(_KEY, "2026-01-05") is None

    def test_rewriting_atomically_replaces_the_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
            first = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0)])
            second = build_partition(dataset_key=_KEY, partition_key="2026-01-05", spec=_DAILY, members=[("1", t0), ("2", t0 + timedelta(hours=1))])
            store.write(first)
            store.write(second)
            assert store.read(_KEY, "2026-01-05") == second

    def test_list_partition_keys_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            for day in (7, 5, 6):
                key = f"2026-01-0{day}"
                store.write(build_partition(dataset_key=_KEY, partition_key=key, spec=_DAILY, members=[("1", datetime(2026, 1, day, tzinfo=timezone.utc))]))
            assert store.list_partition_keys(_KEY) == ("2026-01-05", "2026-01-06", "2026-01-07")

    def test_unsafe_partition_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            with pytest.raises(MarketDataPathSecurityError):
                store.read(_KEY, "../../etc/passwd")
