"""Tests for `MarketDriverBar`/`MarketDriverBarStore`, raw-to-canonical
normalization, and gap analysis (Milestone 10, Phase 4C, spec Section 30
"Prices"/"Sessions" sub-items, Section 21 gap policy)."""

from __future__ import annotations

import sys
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _cross_asset_test_helpers import (
    default_availability_policy,
    default_nyse_session_policy,
    fresh_repository_and_cache,
)

from quant_platform.core.exceptions import MarketRecordError
from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cross_asset.futures import RollProvenance
from quant_platform.market_data.collectors.cross_asset.gap_policy import analyze_bar_gaps
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_normalization import (
    normalize_raw_market_record,
    resolve_bar_open_time,
)
from quant_platform.market_data.collectors.cross_asset.market_record import (
    MarketDriverBarStore,
    RawMarketRecord,
    create_market_driver_bar,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    create_timezone_session_policy,
)

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


def _valid_bar(**overrides: object):
    defaults: dict[str, object] = {
        "canonical_driver_id": "gold_reference", "provider": "alpha_vantage", "provider_symbol": "GLD", "instrument_form": InstrumentForm.ETF,
        "open_time": datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc), "timeframe": Timeframe.D1, "open": Decimal("191.00"),
        "high": Decimal("193.00"), "low": Decimal("189.80"), "close": Decimal("192.50"), "volume": Decimal("1000000"), "volume_unit": "shares",
        "availability_policy_id": "a" * 64, "session_policy_id": "s" * 64, "adjustment_policy_id": "adj" + "0" * 61,
        "request_manifest_id": "r" * 64, "response_manifest_id": "resp" + "0" * 60, "source_manifest_id": "src" + "0" * 61,
        "source_row_index": 0,
    }
    defaults.update(overrides)
    close_time = compute_close_time(defaults["open_time"], defaults["timeframe"])  # type: ignore[arg-type]
    defaults.setdefault("availability_time", close_time)
    return create_market_driver_bar(**defaults)  # type: ignore[arg-type]


class TestMarketDriverBar:
    def test_valid_bar_constructs(self) -> None:
        bar = _valid_bar()
        assert bar.bar_id

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(high=Decimal("10"), low=Decimal("20"))

    def test_open_outside_range_rejected(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(open=Decimal("500"))

    def test_close_outside_range_rejected(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(close=Decimal("500"))

    def test_zero_or_negative_low_rejected(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(low=Decimal("0"), open=Decimal("0"), close=Decimal("0"), high=Decimal("0.01"))

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(volume=Decimal("-1"))

    def test_zero_volume_accepted(self) -> None:
        bar = _valid_bar(volume=Decimal("0"))
        assert bar.volume == Decimal("0")

    def test_none_volume_accepted(self) -> None:
        bar = _valid_bar(volume=None)
        assert bar.volume is None

    def test_availability_before_close_rejected(self) -> None:
        open_time = datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc)
        close_time = compute_close_time(open_time, Timeframe.D1)
        with pytest.raises(MarketRecordError):
            _valid_bar(open_time=open_time, availability_time=open_time)
        assert close_time > open_time

    def test_futures_form_requires_contract_metadata_id(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(instrument_form=InstrumentForm.EXCHANGE_FUTURES_CONTRACT, contract_metadata_id=None)

    def test_futures_form_with_contract_metadata_id_accepted(self) -> None:
        bar = _valid_bar(instrument_form=InstrumentForm.EXCHANGE_FUTURES_CONTRACT, contract_metadata_id="c" * 64)
        assert bar.contract_metadata_id == "c" * 64

    def test_provider_continuous_requires_roll_provenance(self) -> None:
        with pytest.raises(MarketRecordError):
            _valid_bar(instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, roll_provenance=None)

    def test_provider_continuous_with_roll_provenance_accepted(self) -> None:
        roll = RollProvenance(
            active_contract_symbol="CLG25", prior_contract_symbol=None, next_contract_symbol=None, roll_timestamp=None,
            adjustment_amount=None, adjustment_ratio=None, continuation_policy_id="c" * 64,
        )
        bar = _valid_bar(instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES, roll_provenance=roll)
        assert bar.roll_provenance == roll

    def test_close_time_property(self) -> None:
        bar = _valid_bar()
        assert bar.close_time == compute_close_time(bar.open_time, bar.timeframe)

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBar

        bar = _valid_bar()
        restored = MarketDriverBar.from_json_dict(bar.to_json_dict())
        assert restored == bar

    def test_decimal_exactness_no_float(self) -> None:
        bar = _valid_bar(open=Decimal("191.123456789"), close=Decimal("192.987654321"), low=Decimal("189.000000001"), high=Decimal("193.999999999"))
        assert isinstance(bar.open, Decimal)
        assert str(bar.open) == "191.123456789"


class TestMarketDriverBarStore:
    def test_append_many_and_read_all_dedupes_by_bar_id(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = MarketDriverBarStore(root)
        bar = _valid_bar()
        result_1 = store.append_many_and_read_all("alpha_vantage", "gold_reference", InstrumentForm.ETF, [bar])
        result_2 = store.append_many_and_read_all("alpha_vantage", "gold_reference", InstrumentForm.ETF, [bar])
        assert len(result_1) == 1
        assert len(result_2) == 1
        assert result_1 == result_2

    def test_different_instrument_forms_never_share_scope(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = MarketDriverBarStore(root)
        bar_etf = _valid_bar(instrument_form=InstrumentForm.ETF)
        bar_spot = _valid_bar(instrument_form=InstrumentForm.SPOT, adjustment_policy_id="b" * 64)
        store.append_many_and_read_all("alpha_vantage", "gold_reference", InstrumentForm.ETF, [bar_etf])
        store.append_many_and_read_all("alpha_vantage", "gold_reference", InstrumentForm.SPOT, [bar_spot])
        assert len(store.read_bars("alpha_vantage", "gold_reference", InstrumentForm.ETF)) == 1
        assert len(store.read_bars("alpha_vantage", "gold_reference", InstrumentForm.SPOT)) == 1

    def test_empty_store_returns_empty_list(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = MarketDriverBarStore(root)
        assert store.read_bars("alpha_vantage", "does_not_exist", InstrumentForm.ETF) == []


class TestResolveBarOpenTime:
    def test_open_labeled_non_24h(self) -> None:
        session = create_timezone_session_policy(
            timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
            provider_session_note="x", session_open_time=time(9, 30), session_close_time=time(16, 0),
        )
        open_time = resolve_bar_open_time("2024-01-05", session_policy=session, timeframe=Timeframe.D1)
        assert open_time == datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc)

    def test_close_labeled_non_24h(self) -> None:
        session = create_timezone_session_policy(
            timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.CLOSE_LABELED,
            provider_session_note="x", session_open_time=time(9, 30), session_close_time=time(16, 0),
        )
        open_time = resolve_bar_open_time("2024-01-05", session_policy=session, timeframe=Timeframe.D1)
        expected_close = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        assert open_time == expected_close - Timeframe.D1.duration

    def test_24h_session(self) -> None:
        session = create_timezone_session_policy(
            timezone_key="UTC", is_24_hour_session=True, timestamp_convention=CandleTimestampConvention.OPEN_LABELED, provider_session_note="x",
        )
        open_time = resolve_bar_open_time("2024-01-05", session_policy=session, timeframe=Timeframe.D1)
        assert open_time == datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)

    def test_different_timezones_produce_different_utc_opens(self) -> None:
        ny_session = create_timezone_session_policy(
            timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
            provider_session_note="x", session_open_time=time(9, 30), session_close_time=time(16, 0),
        )
        tokyo_session = create_timezone_session_policy(
            timezone_key="Asia/Tokyo", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
            provider_session_note="x", session_open_time=time(9, 0), session_close_time=time(15, 0),
        )
        ny_open = resolve_bar_open_time("2024-01-05", session_policy=ny_session, timeframe=Timeframe.D1)
        tokyo_open = resolve_bar_open_time("2024-01-05", session_policy=tokyo_session, timeframe=Timeframe.D1)
        assert ny_open != tokyo_open


class TestNormalizeRawMarketRecord:
    def _valid_raw(self, **overrides: object) -> RawMarketRecord:
        defaults: dict[str, object] = {
            "provider": "alpha_vantage", "provider_symbol": "GLD", "provider_timestamp_text": "2024-01-05", "interval": "1d",
            "open_text": "191.00", "high_text": "193.00", "low_text": "189.80", "close_text": "192.50", "volume_text": "1000000",
            "adjusted_close_text": None, "trade_count_text": None, "source_sequence": 0, "contract_symbol": None,
        }
        defaults.update(overrides)
        return RawMarketRecord(**defaults)  # type: ignore[arg-type]

    def test_valid_record_normalizes(self) -> None:
        session = default_nyse_session_policy()
        availability = default_availability_policy()
        bar, issues = normalize_raw_market_record(
            self._valid_raw(), canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF, timeframe=Timeframe.D1,
            session_policy=session, availability_policy=availability, adjustment_policy_id="a" * 64, request_manifest_id="r" * 64,
            response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert issues == ()
        assert bar is not None
        assert bar.close == Decimal("192.50")

    def test_invalid_decimal_quarantined(self) -> None:
        session = default_nyse_session_policy()
        availability = default_availability_policy()
        bar, issues = normalize_raw_market_record(
            self._valid_raw(close_text="not-a-number"), canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF,
            timeframe=Timeframe.D1, session_policy=session, availability_policy=availability, adjustment_policy_id="a" * 64,
            request_manifest_id="r" * 64, response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert bar is None
        assert "invalid_market_record" in issues

    def test_missing_volume_quarantined(self) -> None:
        session = default_nyse_session_policy()
        availability = default_availability_policy()
        bar, issues = normalize_raw_market_record(
            self._valid_raw(volume_text="garbage"), canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF,
            timeframe=Timeframe.D1, session_policy=session, availability_policy=availability, adjustment_policy_id="a" * 64,
            request_manifest_id="r" * 64, response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert bar is None
        assert "missing_market_volume" in issues

    def test_none_volume_text_normalizes_to_none_not_zero(self) -> None:
        session = default_nyse_session_policy()
        availability = default_availability_policy()
        bar, issues = normalize_raw_market_record(
            self._valid_raw(volume_text=None), canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF, timeframe=Timeframe.D1,
            session_policy=session, availability_policy=availability, adjustment_policy_id="a" * 64, request_manifest_id="r" * 64,
            response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert issues == ()
        assert bar is not None
        assert bar.volume is None

    def test_availability_time_after_close(self) -> None:
        session = default_nyse_session_policy()
        availability = default_availability_policy(delay_minutes=60)
        bar, _issues = normalize_raw_market_record(
            self._valid_raw(), canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF, timeframe=Timeframe.D1,
            session_policy=session, availability_policy=availability, adjustment_policy_id="a" * 64, request_manifest_id="r" * 64,
            response_manifest_id="p" * 64, source_manifest_id="s" * 64, source_row_index=0,
        )
        assert bar is not None
        assert bar.availability_time > bar.close_time


class TestGapAnalysis:
    def test_no_gaps_for_consecutive_business_days(self) -> None:
        session = default_nyse_session_policy()
        bars = [
            _valid_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc), source_row_index=0),
            _valid_bar(open_time=datetime(2024, 1, 4, 14, 30, tzinfo=timezone.utc), source_row_index=1),
            _valid_bar(open_time=datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc), source_row_index=2),
        ]
        report = analyze_bar_gaps(tuple(bars), session_policy=session)
        assert report.missing_business_day_count == 0
        assert report.conflicting_coordinate_count == 0

    def test_weekend_never_reported_as_missing(self) -> None:
        """Jan 5 2024 is a Friday, Jan 8 2024 is the following Monday --
        no Saturday/Sunday gap should ever be flagged."""
        session = default_nyse_session_policy()
        bars = [
            _valid_bar(open_time=datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc), source_row_index=0),
            _valid_bar(open_time=datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc), source_row_index=1),
        ]
        report = analyze_bar_gaps(tuple(bars), session_policy=session)
        assert report.missing_business_day_count == 0

    def test_missing_business_day_detected(self) -> None:
        """Jan 3 and Jan 5 are both weekdays; Jan 4 (a Thursday) is
        missing between them."""
        session = default_nyse_session_policy()
        bars = [
            _valid_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc), source_row_index=0),
            _valid_bar(open_time=datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc), source_row_index=1),
        ]
        report = analyze_bar_gaps(tuple(bars), session_policy=session)
        assert report.missing_business_day_count == 1
        assert report.missing_business_days == ("2024-01-04",)

    def test_exact_duplicate_bar_not_a_conflict(self) -> None:
        """Two bars with the SAME bar_id (identical content) at the
        SAME open_time are never flagged as conflicting -- only
        DIFFERENT bar_ids at the same coordinate are."""
        session = default_nyse_session_policy()
        bar = _valid_bar()
        report = analyze_bar_gaps((bar, bar), session_policy=session)
        assert report.conflicting_coordinate_count == 0

    def test_conflicting_duplicate_bar_detected(self) -> None:
        session = default_nyse_session_policy()
        bar_a = _valid_bar(close=Decimal("192.50"))
        bar_b = _valid_bar(close=Decimal("190.10"))
        assert bar_a.open_time == bar_b.open_time
        assert bar_a.bar_id != bar_b.bar_id
        report = analyze_bar_gaps((bar_a, bar_b), session_policy=session)
        assert report.conflicting_coordinate_count == 1
        assert report.has_conflicting_coordinates

    def test_24h_session_reports_zero_missing_days(self) -> None:
        session = create_timezone_session_policy(
            timezone_key="UTC", is_24_hour_session=True, timestamp_convention=CandleTimestampConvention.OPEN_LABELED, provider_session_note="x",
        )
        bars = [
            _valid_bar(open_time=datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc), source_row_index=0),
            _valid_bar(open_time=datetime(2024, 1, 10, 0, 0, tzinfo=timezone.utc), source_row_index=1),
        ]
        report = analyze_bar_gaps(tuple(bars), session_policy=session)
        assert report.missing_business_day_count == 0
        assert report.calendar_assurance == "limited"

    def test_out_of_order_source_sequence_detected(self) -> None:
        session = default_nyse_session_policy()
        bars = [
            _valid_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc), source_row_index=5),
            _valid_bar(open_time=datetime(2024, 1, 4, 14, 30, tzinfo=timezone.utc), source_row_index=2),
        ]
        report = analyze_bar_gaps(tuple(bars), session_policy=session)
        assert report.out_of_order_count == 1

    def test_mixed_scope_rejected(self) -> None:
        from quant_platform.core.exceptions import GapPolicyError

        session = default_nyse_session_policy()
        bar_a = _valid_bar(canonical_driver_id="gold_reference")
        bar_b = _valid_bar(canonical_driver_id="silver", open_time=datetime(2024, 1, 6, 14, 30, tzinfo=timezone.utc))
        with pytest.raises(GapPolicyError):
            analyze_bar_gaps((bar_a, bar_b), session_policy=session)
