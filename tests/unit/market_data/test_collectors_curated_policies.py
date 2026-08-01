"""Revision-policy and availability-policy tests (Milestone 10, Phase 4B)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import (
    AvailabilityPolicyError,
    AvailabilityUnresolvedError,
    RevisionPolicyError,
)
from quant_platform.market_data.collectors.curated.availability import (
    AvailabilityPolicyKind,
    create_availability_policy,
    resolve_availability_time,
)
from quant_platform.market_data.collectors.curated.revision_policy import (
    RevisionPolicyKind,
    create_revision_policy,
    resolve_fred_request_overrides,
)


class TestRevisionPolicyConstruction:
    def test_latest_available(self) -> None:
        p = create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE)
        overrides = resolve_fred_request_overrides(p)
        assert overrides == {"realtime_start": None, "realtime_end": None, "output_type": None}

    def test_first_release_only(self) -> None:
        p = create_revision_policy(kind=RevisionPolicyKind.FIRST_RELEASE_ONLY)
        overrides = resolve_fred_request_overrides(p)
        assert overrides["output_type"] == 4

    def test_as_of_realtime_date_requires_date(self) -> None:
        with pytest.raises(RevisionPolicyError):
            create_revision_policy(kind=RevisionPolicyKind.AS_OF_REALTIME_DATE)

    def test_as_of_realtime_date_resolves_window(self) -> None:
        as_of = datetime(2024, 3, 1, tzinfo=timezone.utc)
        p = create_revision_policy(kind=RevisionPolicyKind.AS_OF_REALTIME_DATE, as_of_realtime_date=as_of)
        overrides = resolve_fred_request_overrides(p)
        assert overrides["realtime_start"] == as_of
        assert overrides["realtime_end"] == as_of

    def test_vintage_series(self) -> None:
        p = create_revision_policy(kind=RevisionPolicyKind.VINTAGE_SERIES)
        overrides = resolve_fred_request_overrides(p)
        assert overrides["output_type"] == 2

    def test_invalid_combination_rejected_as_of_date_on_non_as_of_kind(self) -> None:
        with pytest.raises(RevisionPolicyError):
            create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE, as_of_realtime_date=datetime(2024, 1, 1, tzinfo=timezone.utc))

    def test_distinct_vintage_identities(self) -> None:
        d1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        d2 = datetime(2024, 2, 1, tzinfo=timezone.utc)
        p1 = create_revision_policy(kind=RevisionPolicyKind.AS_OF_REALTIME_DATE, as_of_realtime_date=d1)
        p2 = create_revision_policy(kind=RevisionPolicyKind.AS_OF_REALTIME_DATE, as_of_realtime_date=d2)
        assert p1.revision_policy_id != p2.revision_policy_id

    def test_deterministic_id(self) -> None:
        p1 = create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE)
        p2 = create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE)
        assert p1.revision_policy_id == p2.revision_policy_id

    def test_round_trip(self) -> None:
        from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicy

        p = create_revision_policy(kind=RevisionPolicyKind.VINTAGE_SERIES)
        assert RevisionPolicy.from_json_dict(p.to_json_dict()) == p


class TestAvailabilityPolicyConstruction:
    def test_observation_date_end_of_day_requires_time_fields(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        with pytest.raises(MarketDataError):  # require_non_empty raises the base type
            create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY)

    def test_explicit_release_timestamp_requires_timestamp(self) -> None:
        with pytest.raises(AvailabilityPolicyError):
            create_availability_policy(kind=AvailabilityPolicyKind.EXPLICIT_RELEASE_TIMESTAMP)

    def test_explicit_release_timestamp_forbids_time_of_day_fields(self) -> None:
        with pytest.raises(AvailabilityPolicyError):
            create_availability_policy(
                kind=AvailabilityPolicyKind.EXPLICIT_RELEASE_TIMESTAMP, explicit_release_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                timezone_key="UTC",
            )

    def test_delay_days_forbidden_outside_manual_curated_rule(self) -> None:
        with pytest.raises(AvailabilityPolicyError):
            create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="UTC", availability_hour=0, availability_minute=0, delay_days=1)

    def test_manual_curated_rule_allows_delay_days(self) -> None:
        p = create_availability_policy(kind=AvailabilityPolicyKind.MANUAL_CURATED_RELEASE_RULE, timezone_key="UTC", availability_hour=0, availability_minute=0, delay_days=45)
        assert p.delay_days == 45

    def test_release_calendar_reference_requires_calendar_name(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        with pytest.raises(MarketDataError):  # require_non_empty raises the base type
            create_availability_policy(kind=AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE)

    def test_policy_change_changes_identity(self) -> None:
        p1 = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="UTC", availability_hour=0, availability_minute=0)
        p2 = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="UTC", availability_hour=1, availability_minute=0)
        assert p1.availability_policy_id != p2.availability_policy_id

    def test_round_trip(self) -> None:
        from quant_platform.market_data.collectors.curated.availability import AvailabilityPolicy

        p = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="UTC", availability_hour=12, availability_minute=30)
        assert AvailabilityPolicy.from_json_dict(p.to_json_dict()) == p


class TestDailyRateAvailability:
    def test_daily_rate_available_end_of_observation_day(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="America/New_York", availability_hour=17, availability_minute=0)
        t = resolve_availability_time(policy, observation_date_text="2024-01-02", realtime_start_text=None)
        assert t.date().isoformat() == "2024-01-02" or t.date().isoformat() == "2024-01-03"  # UTC conversion may roll to next calendar day

    def test_daily_rate_does_not_need_realtime_start(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="UTC", availability_hour=23, availability_minute=59)
        t = resolve_availability_time(policy, observation_date_text="2024-01-02", realtime_start_text=None)
        assert t == datetime(2024, 1, 2, 23, 59, tzinfo=timezone.utc)


class TestMonthlyCpiAvailability:
    def test_monthly_cpi_conservative_release_availability(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="America/New_York", availability_hour=8, availability_minute=30)
        t = resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="2024-02-13")
        assert t.month == 2 and t.day in (13,)  # published mid-February, not January

    def test_monthly_cpi_requires_realtime_start(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="America/New_York", availability_hour=8, availability_minute=30)
        with pytest.raises(AvailabilityUnresolvedError):
            resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text=None)


class TestRealtimeStartBasedPolicy:
    def test_value_visible_at_or_after_availability(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="UTC", availability_hour=0, availability_minute=0)
        t = resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="2024-02-13")
        assert t == datetime(2024, 2, 13, tzinfo=timezone.utc)


class TestNextBusinessDayConservative:
    def test_friday_pushes_to_monday(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.NEXT_BUSINESS_DAY_CONSERVATIVE, timezone_key="UTC", availability_hour=0, availability_minute=0)
        t = resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="2024-01-05")  # Friday
        assert t == datetime(2024, 1, 8, tzinfo=timezone.utc)  # Monday

    def test_wednesday_pushes_to_thursday(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.NEXT_BUSINESS_DAY_CONSERVATIVE, timezone_key="UTC", availability_hour=0, availability_minute=0)
        t = resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="2024-01-03")  # Wednesday
        assert t == datetime(2024, 1, 4, tzinfo=timezone.utc)  # Thursday


class TestTimezoneAndBusinessDayHandling:
    def test_different_timezones_produce_different_utc_availability(self) -> None:
        policy_ny = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="America/New_York", availability_hour=17, availability_minute=0)
        policy_tokyo = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="Asia/Tokyo", availability_hour=17, availability_minute=0)
        t_ny = resolve_availability_time(policy_ny, observation_date_text="2024-01-02", realtime_start_text=None)
        t_tokyo = resolve_availability_time(policy_tokyo, observation_date_text="2024-01-02", realtime_start_text=None)
        assert t_ny != t_tokyo


class TestUnresolvedAvailabilityFailsClosed:
    def test_unresolvable_availability_raises_not_silently_available(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="UTC", availability_hour=0, availability_minute=0)
        with pytest.raises(AvailabilityUnresolvedError):
            resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="")

    def test_release_calendar_reference_is_not_implemented_this_phase(self) -> None:
        policy = create_availability_policy(kind=AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE, business_day_calendar="bls_release_calendar")
        with pytest.raises(AvailabilityPolicyError):
            resolve_availability_time(policy, observation_date_text="2024-01-01", realtime_start_text="2024-02-13")
