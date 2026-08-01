"""Series-metadata drift verification tests (Milestone 10, Phase 4B)."""

from __future__ import annotations

from _curated_test_helpers import OBS_START

from quant_platform.market_data.collectors.curated.metadata import verify_series_metadata
from quant_platform.market_data.collectors.curated.registry import SeriesTier, create_curated_series_spec
from quant_platform.market_data.collectors.fred_series_metadata import FredSeriesMetadata
from quant_platform.market_data.collectors.macro_normalization import MacroUnit


def _spec(**overrides):
    defaults = {
        "series_id": "DGS10", "canonical_series_name": "us_10y_nominal_yield", "registry_version": 1, "tier": SeriesTier.CORE_XAUUSD_DRIVER,
        "economic_category": "rates", "expected_native_frequency": "D", "expected_units": ("%",), "expected_seasonal_adjustment": "NSA",
        "target_macro_instrument_id": "us_10y_nominal_yield", "normalization_kind": MacroUnit.PERCENT, "revision_policy_id": "a" * 64,
        "release_availability_policy_id": "b" * 64, "default_observation_start": OBS_START,
    }
    defaults.update(overrides)
    return create_curated_series_spec(**defaults)


def _metadata(**overrides) -> FredSeriesMetadata:
    defaults = {
        "series_id": "DGS10", "response_realtime_start": "2024-06-01", "response_realtime_end": "2024-06-01", "title": "10-Year Treasury",
        "observation_start": "1962-01-02", "observation_end": "2024-06-01", "frequency": "Daily", "frequency_short": "D", "units": "Percent",
        "units_short": "%", "seasonal_adjustment": "Not Seasonally Adjusted", "seasonal_adjustment_short": "NSA", "last_updated": "2024-06-01 10:00:00-05",
        "notes": None, "popularity": None,
    }
    defaults.update(overrides)
    return FredSeriesMetadata(**defaults)


class TestExactMatch:
    def test_matching_metadata_passes(self) -> None:
        result = verify_series_metadata(_spec(), _metadata())
        assert result.passed
        assert not any(f.severity == "fail_closed" for f in result.findings)


class TestFailClosedDrift:
    def test_wrong_series_id_fails_closed(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(series_id="DGS30"))
        assert not result.passed
        assert any(f.code == "unexpected_series_id" and f.severity == "fail_closed" for f in result.findings)

    def test_wrong_frequency_fails_closed(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(frequency_short="M"))
        assert not result.passed
        assert any(f.code == "incompatible_frequency" for f in result.findings)

    def test_wrong_units_fails_closed(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(units_short="Index"))
        assert not result.passed
        assert any(f.code == "incompatible_units" for f in result.findings)

    def test_wrong_seasonal_adjustment_fails_closed_when_declared(self) -> None:
        result = verify_series_metadata(_spec(expected_seasonal_adjustment="SA"), _metadata(seasonal_adjustment_short="NSA"))
        assert not result.passed
        assert any(f.code == "changed_seasonal_adjustment" for f in result.findings)

    def test_seasonal_adjustment_not_checked_when_spec_declares_none(self) -> None:
        result = verify_series_metadata(_spec(expected_seasonal_adjustment=None), _metadata(seasonal_adjustment_short="SA"))
        assert result.passed


class TestHarmlessChanges:
    def test_title_change_is_informational_only(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(title="A Completely Different Title"))
        assert result.passed
        assert any(f.code == "title_reported" and f.severity == "info" for f in result.findings)


class TestObservationRangeChange:
    def test_requested_range_within_supported_range_permitted(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(observation_start="1962-01-02", observation_end="2024-06-01"), requested_observation_start="2020-01-01", requested_observation_end="2024-01-01")
        assert result.passed

    def test_requested_range_exceeding_supported_end_is_warning(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(observation_end="2024-01-01"), requested_observation_start="2020-01-01", requested_observation_end="2024-06-01")
        assert result.passed  # a warning, not fail-closed
        assert any(f.code == "requested_range_after_supported_end" and f.severity == "warning" for f in result.findings)

    def test_requested_range_before_supported_start_fails_closed(self) -> None:
        result = verify_series_metadata(_spec(), _metadata(observation_start="1962-01-02"), requested_observation_start="1900-01-01", requested_observation_end="2024-01-01")
        assert not result.passed
        assert any(f.code == "requested_range_before_supported_start" for f in result.findings)


class TestForgedMetadataIdentity:
    def test_a_metadata_object_with_swapped_series_id_is_caught_as_drift(self) -> None:
        """A "forged" metadata response (claiming to be for a different
        series while everything else matches) is exactly the
        `unexpected_series_id` case -- proving the check is non-vacuous
        against a deliberately mismatched artifact."""
        forged = _metadata(series_id="CPIAUCSL")
        result = verify_series_metadata(_spec(), forged)
        assert not result.passed
