"""Curated registry tests (Milestone 10, Phase 4B)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from _curated_test_helpers import OBS_START, default_core_registry

from quant_platform.core.exceptions import CuratedRegistryError
from quant_platform.market_data.collectors.curated.registry import (
    CuratedFredRegistry,
    MissingValuePolicy,
    SeriesTier,
    create_curated_registry,
    create_curated_series_spec,
    default_extended_series_specs,
)
from quant_platform.market_data.collectors.macro_normalization import MacroUnit


def _spec(series_id: str, canonical_name: str, **overrides):
    defaults = {
        "series_id": series_id, "canonical_series_name": canonical_name, "registry_version": 1, "tier": SeriesTier.CORE_XAUUSD_DRIVER,
        "economic_category": "test", "expected_native_frequency": "D", "expected_units": ("%",), "target_macro_instrument_id": canonical_name,
        "normalization_kind": MacroUnit.PERCENT, "revision_policy_id": "a" * 64, "release_availability_policy_id": "b" * 64,
        "default_observation_start": OBS_START,
    }
    defaults.update(overrides)
    return create_curated_series_spec(**defaults)


class TestRequiredCoreSeries:
    def test_all_four_core_series_present(self) -> None:
        registry = default_core_registry()
        assert {s.series_id for s in registry.specs} == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}

    def test_core_canonical_names_match_spec(self) -> None:
        registry = default_core_registry()
        by_id = {s.series_id: s.canonical_series_name for s in registry.specs}
        assert by_id["DFII10"] == "us_10y_real_yield"
        assert by_id["DGS10"] == "us_10y_nominal_yield"
        assert by_id["CPIAUCSL"] == "us_cpi_all_urban"
        assert by_id["DFF"] == "effective_federal_funds_rate"

    def test_core_series_are_all_enabled(self) -> None:
        registry = default_core_registry()
        assert set(registry.enabled_series_ids()) == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}


class TestExtendedUniverse:
    def test_extended_series_are_disabled_by_default(self) -> None:
        extended = default_extended_series_specs(
            registry_version=1, revision_policy_id="a" * 64, release_availability_policy_id_daily="b" * 64,
            release_availability_policy_id_monthly="c" * 64, release_availability_policy_id_weekly="d" * 64, default_observation_start=OBS_START,
        )
        assert len(extended) == 14
        assert all(not s.enabled for s in extended)

    def test_extended_universe_covers_documented_candidates(self) -> None:
        extended = default_extended_series_specs(
            registry_version=1, revision_policy_id="a" * 64, release_availability_policy_id_daily="b" * 64,
            release_availability_policy_id_monthly="c" * 64, release_availability_policy_id_weekly="d" * 64, default_observation_start=OBS_START,
        )
        expected = {"T10YIE", "T5YIE", "DGS2", "DGS5", "DGS30", "DTWEXBGS", "UNRATE", "PAYEMS", "PCEPI", "PCEPILFE", "INDPRO", "VIXCLS", "WALCL", "M2SL"}
        assert {s.series_id for s in extended} == expected


class TestRegistryIdentity:
    def test_duplicate_series_id_rejected(self) -> None:
        a = _spec("DGS10", "name_a")
        b = _spec("DGS10", "name_b")
        with pytest.raises(CuratedRegistryError):
            create_curated_registry(registry_version=1, specs=(a, b))

    def test_duplicate_canonical_name_rejected(self) -> None:
        a = _spec("DGS10", "same_name")
        b = _spec("DFF", "same_name")
        with pytest.raises(CuratedRegistryError):
            create_curated_registry(registry_version=1, specs=(a, b))

    def test_declaration_order_independence(self) -> None:
        a = _spec("DGS10", "name_a")
        b = _spec("DFF", "name_b")
        r1 = create_curated_registry(registry_version=1, specs=(a, b))
        r2 = create_curated_registry(registry_version=1, specs=(b, a))
        assert r1.registry_id == r2.registry_id
        assert [s.series_id for s in r1.specs] == [s.series_id for s in r2.specs]

    def test_material_field_change_changes_identity(self) -> None:
        a = _spec("DGS10", "name_a", economic_category="rates")
        b = _spec("DGS10", "name_a", economic_category="different")
        r1 = create_curated_registry(registry_version=1, specs=(a,))
        r2 = create_curated_registry(registry_version=1, specs=(b,))
        assert r1.registry_id != r2.registry_id

    def test_notes_change_does_not_change_identity(self) -> None:
        a = _spec("DGS10", "name_a", notes="original note")
        b = _spec("DGS10", "name_a", notes="a completely different note")
        r1 = create_curated_registry(registry_version=1, specs=(a,))
        r2 = create_curated_registry(registry_version=1, specs=(b,))
        assert r1.registry_id == r2.registry_id

    def test_round_trip_through_json(self) -> None:
        registry = default_core_registry()
        restored = CuratedFredRegistry.from_json_dict(registry.to_json_dict())
        assert restored == registry


class TestSpecValidation:
    def test_unsupported_frequency_rejected(self) -> None:
        with pytest.raises(CuratedRegistryError):
            _spec("DGS10", "x", expected_native_frequency="NOT_A_CODE")

    def test_empty_expected_units_rejected(self) -> None:
        with pytest.raises(CuratedRegistryError):
            _spec("DGS10", "x", expected_units=())

    def test_enabled_series_requires_target_instrument_id(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        # Goes through the generic `require_non_empty` helper (raises the
        # base `MarketDataError`, not the `CuratedRegistryError` subclass) --
        # same pattern as every other `require_non_empty` call in this codebase.
        with pytest.raises(MarketDataError):
            _spec("DGS10", "x", target_macro_instrument_id="", enabled=True)

    def test_disabled_series_selection_is_rejected_at_backfill_time(self) -> None:
        # Registry construction itself allows a disabled series to exist (with or
        # without a target id); rejection of SELECTING a disabled series happens in
        # backfill.py, covered by test_collectors_curated_backfill.py.
        spec = _spec("DGS10", "x", enabled=False, target_macro_instrument_id="")
        registry = create_curated_registry(registry_version=1, specs=(spec,))
        assert registry.get("DGS10") is not None
        assert not registry.get("DGS10").enabled

    def test_aggregation_method_requires_frequency(self) -> None:
        with pytest.raises(CuratedRegistryError):
            _spec("DGS10", "x", aggregation_method="avg", request_frequency=None)

    def test_negative_unit_conversion_rejected(self) -> None:
        with pytest.raises(CuratedRegistryError):
            _spec("DGS10", "x", unit_conversion=Decimal(-1))

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 -- require_tz_aware's own exception type
            _spec("DGS10", "x", default_observation_start=datetime(2020, 1, 1))

    def test_missing_value_policy_default_is_quarantine(self) -> None:
        spec = _spec("DGS10", "x")
        assert spec.missing_value_policy is MissingValuePolicy.QUARANTINE


class TestRegistryLookup:
    def test_get_unknown_series_returns_none(self) -> None:
        registry = default_core_registry()
        assert registry.get("NOPE") is None

    def test_get_known_series(self) -> None:
        registry = default_core_registry()
        spec = registry.get("DGS10")
        assert spec is not None and spec.series_id == "DGS10"
