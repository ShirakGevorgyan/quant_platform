"""Tests for `historical.timezones` -- the ingestion pipeline's single
timezone-normalization boundary. These are the "critical tests" the
Milestone 2 spec calls out by name: naive timestamps rejected, DST
ambiguous/nonexistent times rejected, and equivalent instants expressed in
different source timezones normalize to the identical UTC value.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_platform.core.exceptions import TimezoneError
from quant_platform.historical.timezones import (
    FixedOffsetTimezone,
    NamedZoneTimezone,
    localize_broker_timestamps,
    require_utc,
)


def _series(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


class TestFixedOffsetTimezone:
    def test_rejects_offset_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="offset must be within"):
            FixedOffsetTimezone(timedelta(hours=25))

    def test_localizes_positive_offset_to_utc(self) -> None:
        out = localize_broker_timestamps(_series("2024-01-01 10:00:00"), FixedOffsetTimezone(timedelta(hours=2)))
        # 10:00 at UTC+2 is 08:00 UTC -- hand-computed, not re-derived from the function under test.
        assert out.iloc[0] == pd.Timestamp("2024-01-01 08:00:00", tz="UTC")

    def test_localizes_negative_offset_to_utc(self) -> None:
        out = localize_broker_timestamps(_series("2024-01-01 10:00:00"), FixedOffsetTimezone(timedelta(hours=-5)))
        # 10:00 at UTC-5 is 15:00 UTC -- hand-computed.
        assert out.iloc[0] == pd.Timestamp("2024-01-01 15:00:00", tz="UTC")


class TestNamedZoneTimezone:
    def test_unknown_zone_key_raises_timezone_error(self) -> None:
        with pytest.raises(TimezoneError, match="Unknown or unavailable"):
            NamedZoneTimezone("Not/AZone").to_tzinfo()


class TestLocalizeBrokerTimestampsRejectsBadInput:
    def test_rejects_already_tz_aware_series(self) -> None:
        aware = pd.Series(pd.to_datetime(["2024-01-01 10:00:00"])).dt.tz_localize("UTC")
        with pytest.raises(TimezoneError, match="already tz-aware"):
            localize_broker_timestamps(aware, FixedOffsetTimezone(timedelta(0)))

    def test_rejects_non_datetime_series(self) -> None:
        with pytest.raises(TimezoneError, match="datetime64 series"):
            localize_broker_timestamps(pd.Series([1, 2, 3]), FixedOffsetTimezone(timedelta(0)))


class TestDstAmbiguousAndNonexistent:
    """America/New_York, 2024: DST fall-back on 2024-11-03 (02:00 local
    becomes 01:00 local again, so 01:30 is ambiguous), DST spring-forward
    on 2024-03-10 (02:00 local jumps to 03:00, so 02:30 never occurs).
    Both hand-picked from the published US DST transition dates, not
    derived from the code under test.
    """

    def test_ambiguous_fall_back_time_is_rejected(self) -> None:
        with pytest.raises(TimezoneError, match="ambiguous"):
            localize_broker_timestamps(_series("2024-11-03 01:30:00"), NamedZoneTimezone("America/New_York"))

    def test_nonexistent_spring_forward_time_is_rejected(self) -> None:
        with pytest.raises(TimezoneError, match="ambiguous"):
            localize_broker_timestamps(_series("2024-03-10 02:30:00"), NamedZoneTimezone("America/New_York"))

    def test_fixed_offset_never_raises_on_these_same_wall_clock_values(self) -> None:
        # A fixed-offset source timezone has no DST transitions, so the
        # identical wall-clock values that are ambiguous/nonexistent for a
        # named US zone must localize cleanly under a fixed offset.
        localize_broker_timestamps(_series("2024-11-03 01:30:00"), FixedOffsetTimezone(timedelta(hours=-5)))
        localize_broker_timestamps(_series("2024-03-10 02:30:00"), FixedOffsetTimezone(timedelta(hours=-5)))


class TestEquivalentInstantsAcrossTimezones:
    def test_same_instant_different_source_tz_normalizes_identically(self) -> None:
        # 12:00 at UTC+0 and 14:00 at UTC+2 are the same real-world instant.
        a = localize_broker_timestamps(_series("2024-06-01 12:00:00"), FixedOffsetTimezone(timedelta(hours=0)))
        b = localize_broker_timestamps(_series("2024-06-01 14:00:00"), FixedOffsetTimezone(timedelta(hours=2)))
        assert a.iloc[0] == b.iloc[0]

    @given(
        hour=st.integers(min_value=0, max_value=23),
        offset_hours=st.integers(min_value=-11, max_value=11),
    )
    def test_property_wall_clock_shifted_by_offset_yields_same_utc_instant(
        self, hour: int, offset_hours: int
    ) -> None:
        base = localize_broker_timestamps(
            pd.Series([pd.Timestamp(2024, 6, 1, hour)]), FixedOffsetTimezone(timedelta(hours=0))
        )
        shifted_wall_clock = pd.Timestamp(2024, 6, 1, hour) + timedelta(hours=offset_hours)
        shifted = localize_broker_timestamps(
            pd.Series([shifted_wall_clock]), FixedOffsetTimezone(timedelta(hours=offset_hours))
        )
        assert base.iloc[0] == shifted.iloc[0]


class TestRequireUtc:
    def test_rejects_naive(self) -> None:
        with pytest.raises(TimezoneError, match="timezone-naive"):
            require_utc(_series("2024-01-01 00:00:00"), context="ctx")

    def test_rejects_non_utc_offset(self) -> None:
        aware = _series("2024-01-01 00:00:00").dt.tz_localize(FixedOffsetTimezone(timedelta(hours=2)).to_tzinfo())
        with pytest.raises(TimezoneError, match="must be normalized to UTC"):
            require_utc(aware, context="ctx")

    def test_accepts_utc(self) -> None:
        aware = _series("2024-01-01 00:00:00").dt.tz_localize("UTC")
        require_utc(aware, context="ctx")  # must not raise

    def test_accepts_zero_offset_fixed_timezone_as_utc_equivalent(self) -> None:
        aware = _series("2024-01-01 00:00:00").dt.tz_localize(FixedOffsetTimezone(timedelta(0)).to_tzinfo())
        require_utc(aware, context="ctx")  # must not raise despite a different tzinfo object identity
