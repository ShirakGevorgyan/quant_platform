"""Macro normalization tests (Milestone 10, Phase 4A) -- unit mapping
specs, exact Decimal preservation, missing-value policy, point-in-time
event_time derivation, and monthly-observation-meaning preservation."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    UnitMappingSpec,
    apply_unit_scale,
    create_unit_mapping_spec,
    normalize_macro_row,
    observation_date_to_event_time,
    resolve_unit,
)
from quant_platform.market_data.quarantine import (
    EMPTY_TIMESTAMP,
    INVALID_DECIMAL,
    MISSING_OBSERVATION_VALUE,
    MISSING_REQUIRED_COLUMN,
    UNKNOWN_SYMBOL,
)


def _mapping(*entries: UnitMappingEntry) -> UnitMappingSpec:
    return create_unit_mapping_spec(unit_mapping_version=1, entries=entries)


DGS10_MAPPING = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT))
DFII10_MAPPING = _mapping(UnitMappingEntry(series_id="DFII10", unit=MacroUnit.PERCENT))
CPI_MAPPING = _mapping(UnitMappingEntry(series_id="CPIAUCSL", unit=MacroUnit.INDEX))
DFF_MAPPING = _mapping(UnitMappingEntry(series_id="DFF", unit=MacroUnit.RATE))


class TestUnitMappingSpecIdentity:
    def test_deterministic_id(self) -> None:
        a = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT))
        b = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT))
        assert a.unit_mapping_id == b.unit_mapping_id

    def test_entry_order_is_irrelevant_to_identity(self) -> None:
        a = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT), UnitMappingEntry(series_id="DFF", unit=MacroUnit.RATE))
        b = _mapping(UnitMappingEntry(series_id="DFF", unit=MacroUnit.RATE), UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT))
        assert a.unit_mapping_id == b.unit_mapping_id

    def test_unit_change_changes_id(self) -> None:
        a = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT))
        b = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.RATE))
        assert a.unit_mapping_id != b.unit_mapping_id

    def test_scale_factor_change_changes_id(self) -> None:
        a = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT, scale_factor=Decimal(1)))
        b = _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT, scale_factor=Decimal(100)))
        assert a.unit_mapping_id != b.unit_mapping_id

    def test_duplicate_series_id_is_rejected(self) -> None:
        with pytest.raises(CollectorError):
            _mapping(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT), UnitMappingEntry(series_id="DGS10", unit=MacroUnit.RATE))

    def test_round_trip_through_json(self) -> None:
        spec = DGS10_MAPPING
        assert UnitMappingSpec.from_json_dict(spec.to_json_dict()) == spec


class TestResolveUnit:
    def test_known_series_resolves(self) -> None:
        entry = resolve_unit(DGS10_MAPPING, series_id="DGS10")
        assert entry.unit is MacroUnit.PERCENT

    def test_unknown_series_fails_closed(self) -> None:
        with pytest.raises(CollectorError):
            resolve_unit(DGS10_MAPPING, series_id="UNKNOWN_SERIES")


class TestApplyUnitScale:
    def test_exact_decimal_preserved_with_default_scale(self) -> None:
        assert apply_unit_scale(Decimal("4.02"), Decimal(1)) == Decimal("4.02")

    def test_scale_factor_applied_via_decimal_multiplication(self) -> None:
        assert apply_unit_scale(Decimal("4.02"), Decimal(100)) == Decimal("402.00")

    def test_signed_zero_is_normalized(self) -> None:
        result = apply_unit_scale(Decimal("-0.00"), Decimal(1))
        assert result == Decimal("0.00")
        assert not result.is_signed()


class TestObservationDateToEventTime:
    def test_calendar_date_parsed_as_utc_midnight(self) -> None:
        result = observation_date_to_event_time("2024-01-02")
        assert result == datetime(2024, 1, 2, tzinfo=timezone.utc)
        assert result.hour == 0 and result.minute == 0 and result.second == 0

    def test_monthly_date_semantics_preserved(self) -> None:
        result = observation_date_to_event_time("2024-01-01")
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_invalid_date_text_raises(self) -> None:
        with pytest.raises(CollectorError):
            observation_date_to_event_time("not-a-date")

    def test_empty_date_text_raises(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        with pytest.raises(MarketDataError):
            observation_date_to_event_time("")


class TestNormalizeMacroRowHappyPaths:
    def test_dgs10_style_percent_observation(self) -> None:
        row = {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert issues == ()
        assert observation is not None
        assert observation.value == Decimal("4.02")
        assert observation.unit is MacroUnit.PERCENT
        assert observation.event_time == datetime(2024, 1, 2, tzinfo=timezone.utc)
        assert observation.source_event_id == "fred:DGS10:date=2024-01-02"

    def test_dfii10_style_percent_observation(self) -> None:
        row = {"date": "2024-01-02", "value": "1.85", "realtime_start": "2024-01-02"}
        observation, issues = normalize_macro_row(row, series_id="DFII10", unit_mapping=DFII10_MAPPING)
        assert issues == ()
        assert observation is not None
        assert observation.value == Decimal("1.85")

    def test_cpi_index_observation_preserves_monthly_meaning(self) -> None:
        row = {"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13"}
        observation, issues = normalize_macro_row(row, series_id="CPIAUCSL", unit_mapping=CPI_MAPPING)
        assert issues == ()
        assert observation is not None
        assert observation.unit is MacroUnit.INDEX
        # event_time reflects the VINTAGE (realtime_start), not the observation month:
        assert observation.event_time == datetime(2024, 2, 13, tzinfo=timezone.utc)
        # the observation month itself is preserved separately, in source_event_id:
        assert observation.source_event_id == "fred:CPIAUCSL:date=2024-01-01"

    def test_dff_style_rate_observation(self) -> None:
        row = {"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-03"}
        observation, issues = normalize_macro_row(row, series_id="DFF", unit_mapping=DFF_MAPPING)
        assert issues == ()
        assert observation is not None
        assert observation.unit is MacroUnit.RATE

    def test_exact_decimal_preservation_no_float_roundoff(self) -> None:
        row = {"date": "2024-01-02", "value": "0.1", "realtime_start": "2024-01-02"}
        observation, _issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is not None
        assert observation.value == Decimal("0.1")  # would NOT be exact if it had ever passed through float

    def test_default_realtime_start_fallback_is_used_when_absent(self) -> None:
        row = {"date": "2024-01-02", "value": "4.02"}
        fallback = datetime(2024, 1, 5, tzinfo=timezone.utc)
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING, default_realtime_start=fallback)
        assert issues == ()
        assert observation is not None
        assert observation.event_time == fallback


class TestNormalizeMacroRowFailurePaths:
    def test_missing_required_column_is_quarantined(self) -> None:
        observation, issues = normalize_macro_row({"date": "2024-01-02"}, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert issues == (MISSING_REQUIRED_COLUMN,)

    def test_fred_dot_missing_value_is_explicit_never_coerced_to_zero(self) -> None:
        row = {"date": "2024-01-02", "value": ".", "realtime_start": "2024-01-02"}
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert issues == (MISSING_OBSERVATION_VALUE,)

    def test_genuinely_malformed_value_is_invalid_decimal_not_missing(self) -> None:
        """The real correctness distinction: "." (explicit absence) and
        "not_a_number" (malformed presence) must be classified
        DIFFERENTLY -- both quarantine the row, but under distinct issue
        codes, since one means "resubmitting won't help" (missing) and
        the other flags a genuine parse failure."""
        row = {"date": "2024-01-02", "value": "not_a_number", "realtime_start": "2024-01-02"}
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert issues == (INVALID_DECIMAL,)

    def test_no_realtime_start_and_no_default_is_empty_timestamp(self) -> None:
        row = {"date": "2024-01-02", "value": "4.02"}
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert issues == (EMPTY_TIMESTAMP,)

    def test_unmapped_series_is_unknown_symbol(self) -> None:
        row = {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}
        observation, issues = normalize_macro_row(row, series_id="NOT_MAPPED", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert issues == (UNKNOWN_SYMBOL,)

    def test_multiple_simultaneous_issues_are_all_reported(self) -> None:
        row = {"date": "2024-01-02", "value": "."}  # missing value AND no realtime_start
        observation, issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is None
        assert MISSING_OBSERVATION_VALUE in issues
        assert EMPTY_TIMESTAMP in issues

    def test_daily_series_never_falsely_given_intraday_timestamp(self) -> None:
        row = {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"}
        observation, _issues = normalize_macro_row(row, series_id="DGS10", unit_mapping=DGS10_MAPPING)
        assert observation is not None
        assert observation.event_time.hour == 0
        assert observation.event_time.minute == 0
        assert observation.event_time.second == 0
        assert observation.event_time.microsecond == 0
