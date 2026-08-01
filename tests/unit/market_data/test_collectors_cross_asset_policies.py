"""Tests for session/timezone, futures/continuation, and availability
policies (Milestone 10, Phase 4C, spec Section 30 "Sessions"/"Futures"/
"Prices" availability sub-items)."""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from quant_platform.core.exceptions import (
    ContinuationPolicyError,
    FuturesContractError,
    MarketAvailabilityPolicyError,
    MarketAvailabilityUnresolvedError,
    SessionPolicyError,
)
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicyKind,
    create_bar_availability_policy,
    resolve_bar_availability_time,
)
from quant_platform.market_data.collectors.cross_asset.futures import (
    ContinuationPolicyKind,
    RollProvenance,
    create_continuation_policy,
    create_futures_contract_metadata,
    require_adjustment_evidence,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    create_timezone_session_policy,
)


class TestTimezoneSessionPolicy:
    def test_non_24h_requires_open_and_close_times(self) -> None:
        with pytest.raises(SessionPolicyError):
            create_timezone_session_policy(
                timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
                provider_session_note="x",
            )

    def test_24h_forbids_open_and_close_times(self) -> None:
        with pytest.raises(SessionPolicyError):
            create_timezone_session_policy(
                timezone_key="UTC", is_24_hour_session=True, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
                provider_session_note="x", session_open_time=time(0, 0), session_close_time=time(23, 59),
            )

    def test_unsupported_timezone_key_rejected(self) -> None:
        with pytest.raises(SessionPolicyError):
            create_timezone_session_policy(
                timezone_key="Mars/Olympus_Mons", is_24_hour_session=True, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
                provider_session_note="x",
            )

    def test_valid_non_24h_policy_constructs(self) -> None:
        policy = create_timezone_session_policy(
            timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
            provider_session_note="NYSE ETF session", session_open_time=time(9, 30), session_close_time=time(16, 0),
        )
        assert policy.session_policy_id

    def test_identity_deterministic(self) -> None:
        kwargs: dict[str, object] = {
            "timezone_key": "Europe/London", "is_24_hour_session": False, "timestamp_convention": CandleTimestampConvention.CLOSE_LABELED,
            "provider_session_note": "LSE session", "session_open_time": time(8, 0), "session_close_time": time(16, 30),
        }
        policy_a = create_timezone_session_policy(**kwargs)  # type: ignore[arg-type]
        policy_b = create_timezone_session_policy(**kwargs)  # type: ignore[arg-type]
        assert policy_a.session_policy_id == policy_b.session_policy_id

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.sessions import TimezoneSessionPolicy

        policy = create_timezone_session_policy(
            timezone_key="Asia/Tokyo", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
            provider_session_note="test", session_open_time=time(9, 0), session_close_time=time(15, 0),
        )
        restored = TimezoneSessionPolicy.from_json_dict(policy.to_json_dict())
        assert restored == policy


class TestFuturesContractMetadata:
    def test_valid_metadata_constructs(self) -> None:
        meta = create_futures_contract_metadata(
            root_symbol="CL", full_contract_symbol="CLF25", exchange="NYMEX", expiry=date(2025, 1, 20),
            contract_month=1, contract_year=2025, contract_multiplier=Decimal(1000), quote_unit="usd_per_barrel", currency="USD",
            tick_size=Decimal("0.01"), session_timezone_key="America/New_York",
        )
        assert meta.futures_contract_metadata_id

    def test_negative_multiplier_rejected(self) -> None:
        with pytest.raises(FuturesContractError):
            create_futures_contract_metadata(
                root_symbol="CL", full_contract_symbol="CLF25", exchange="NYMEX", expiry=date(2025, 1, 20),
                contract_month=1, contract_year=2025, contract_multiplier=Decimal(-1), quote_unit="usd_per_barrel", currency="USD",
                tick_size=Decimal("0.01"), session_timezone_key="America/New_York",
            )

    def test_first_notice_date_after_expiry_rejected(self) -> None:
        with pytest.raises(FuturesContractError):
            create_futures_contract_metadata(
                root_symbol="CL", full_contract_symbol="CLF25", exchange="NYMEX", expiry=date(2025, 1, 20),
                contract_month=1, contract_year=2025, contract_multiplier=Decimal(1000), quote_unit="usd_per_barrel", currency="USD",
                tick_size=Decimal("0.01"), session_timezone_key="America/New_York", first_notice_date=date(2025, 2, 1),
            )

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.futures import FuturesContractMetadata

        meta = create_futures_contract_metadata(
            root_symbol="GC", full_contract_symbol="GCG25", exchange="COMEX", expiry=date(2025, 2, 26),
            contract_month=2, contract_year=2025, contract_multiplier=Decimal(100), quote_unit="usd_per_troy_ounce", currency="USD",
            tick_size=Decimal("0.10"), session_timezone_key="America/New_York",
        )
        restored = FuturesContractMetadata.from_json_dict(meta.to_json_dict())
        assert restored == meta


class TestContinuationPolicy:
    def test_roll_on_fixed_days_requires_days_value(self) -> None:
        with pytest.raises(ContinuationPolicyError):
            create_continuation_policy(kind=ContinuationPolicyKind.ROLL_ON_FIXED_DAYS_BEFORE_EXPIRY, roll_days_before_expiry=None)

    def test_other_kinds_forbid_days_value(self) -> None:
        with pytest.raises(ContinuationPolicyError):
            create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS, roll_days_before_expiry=5)

    def test_back_adjusted_requires_evidence(self) -> None:
        policy = create_continuation_policy(kind=ContinuationPolicyKind.BACK_ADJUSTED_DIFFERENCE)
        provenance = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol="CLF25", next_contract_symbol=None, roll_timestamp="2025-01-15",
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=policy.continuation_policy_id,
        )
        with pytest.raises(ContinuationPolicyError):
            require_adjustment_evidence(policy, provenance)

    def test_ratio_adjusted_requires_evidence(self) -> None:
        policy = create_continuation_policy(kind=ContinuationPolicyKind.RATIO_ADJUSTED)
        provenance = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol="CLF25", next_contract_symbol=None, roll_timestamp="2025-01-15",
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=policy.continuation_policy_id,
        )
        with pytest.raises(ContinuationPolicyError):
            require_adjustment_evidence(policy, provenance)

    def test_back_adjusted_with_evidence_passes(self) -> None:
        policy = create_continuation_policy(kind=ContinuationPolicyKind.BACK_ADJUSTED_DIFFERENCE)
        provenance = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol="CLF25", next_contract_symbol=None, roll_timestamp="2025-01-15",
            adjustment_amount=Decimal("0.15"), adjustment_ratio=None, continuation_policy_id=policy.continuation_policy_id,
        )
        require_adjustment_evidence(policy, provenance)

    def test_provider_native_continuous_does_not_require_evidence(self) -> None:
        policy = create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS)
        provenance = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol=None, next_contract_symbol=None, roll_timestamp=None,
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id=policy.continuation_policy_id,
        )
        require_adjustment_evidence(policy, provenance)

    def test_continuation_policy_constructs(self) -> None:
        policy = create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS)
        assert policy.continuation_policy_id

    def test_roll_provenance_json_round_trip(self) -> None:
        roll = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol="CLF25", next_contract_symbol="CLH25",
            roll_timestamp="2025-01-15T00:00:00+00:00", adjustment_amount=Decimal("0.15"), adjustment_ratio=None,
            continuation_policy_id="c" * 64,
        )
        restored = RollProvenance.from_json_dict(roll.to_json_dict())
        assert restored == roll


class TestBarAvailabilityPolicy:
    def test_negative_delay_rejected(self) -> None:
        with pytest.raises(MarketAvailabilityPolicyError):
            create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="UTC", delay_minutes=-1)

    def test_invalid_timezone_key_rejected(self) -> None:
        with pytest.raises(MarketAvailabilityPolicyError):
            create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="Not/A/Zone", delay_minutes=0)

    def test_close_plus_delay_zero_still_requires_close_passed(self) -> None:
        """delay_minutes=0 resolves to EXACTLY close_time, never before it."""
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="UTC", delay_minutes=0)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        resolved = resolve_bar_availability_time(policy, bar_close_time=close_time)
        assert resolved == close_time

    def test_close_plus_delay_adds_delay(self) -> None:
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="UTC", delay_minutes=60)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        resolved = resolve_bar_availability_time(policy, bar_close_time=close_time)
        assert resolved == datetime(2024, 1, 5, 22, 0, tzinfo=timezone.utc)
        assert resolved > close_time

    def test_naive_close_time_rejected(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="UTC", delay_minutes=0)
        with pytest.raises(MarketDataError):
            resolve_bar_availability_time(policy, bar_close_time=datetime(2024, 1, 5, 21, 0))  # type: ignore[arg-type]

    def test_next_session_open_requires_the_argument(self) -> None:
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.NEXT_SESSION_OPEN_CONSERVATIVE, timezone_key="UTC", delay_minutes=0)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        with pytest.raises(MarketAvailabilityUnresolvedError):
            resolve_bar_availability_time(policy, bar_close_time=close_time)

    def test_next_session_open_before_close_rejected(self) -> None:
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.NEXT_SESSION_OPEN_CONSERVATIVE, timezone_key="UTC", delay_minutes=0)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        with pytest.raises(MarketAvailabilityUnresolvedError):
            resolve_bar_availability_time(policy, bar_close_time=close_time, next_session_open_time=datetime(2024, 1, 5, 20, 0, tzinfo=timezone.utc))

    def test_next_session_open_valid(self) -> None:
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.NEXT_SESSION_OPEN_CONSERVATIVE, timezone_key="UTC", delay_minutes=5)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        next_open = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
        resolved = resolve_bar_availability_time(policy, bar_close_time=close_time, next_session_open_time=next_open)
        assert resolved == datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)

    def test_never_available_at_or_before_open(self) -> None:
        """Structural PIT guarantee: for ANY policy kind, resolved
        availability must be strictly >= close_time, which is itself
        always > open_time -- so availability can never be candle-open."""
        policy = create_bar_availability_policy(kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="UTC", delay_minutes=0)
        open_time = datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc)
        close_time = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        resolved = resolve_bar_availability_time(policy, bar_close_time=close_time)
        assert resolved > open_time
        assert resolved >= close_time
