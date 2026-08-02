"""`features.market_data_bridge.bindings`: immutable source bindings pin
exact, content-addressed versions and structurally reject mutable
aliases (spec Section 4)."""

from __future__ import annotations

import pytest

from quant_platform.core.exceptions import SourceBindingError
from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.bindings import (
    create_base_asset_binding,
    create_cross_asset_dataset_binding,
    create_macro_dataset_binding,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind


def _base_kwargs(**overrides: object) -> dict:
    kwargs = {"canonical_instrument_id": "XAUUSD", "provider": "mt5", "pinned_dataset_id": "a" * 64, "timeframe": Timeframe.H1}
    kwargs.update(overrides)
    return kwargs


def _macro_kwargs(**overrides: object) -> dict:
    kwargs = {
        "curated_registry_id": "r" * 64, "combined_universe_manifest_id": "c" * 64, "series_id": "DFII10",
        "canonical_series_name": "10Y TIPS", "provider": "fred", "component_manifest_id": "d" * 64, "revision_policy_id": "p" * 64,
        "revision_policy_kind": RevisionPolicyKind.VINTAGE_SERIES, "availability_policy_id": "ap1", "native_frequency": "daily",
        "normalized_unit": "percent",
    }
    kwargs.update(overrides)
    return kwargs


def _cross_kwargs(**overrides: object) -> dict:
    proxy = create_proxy_policy(is_proxy=True, proxy_for="us_dollar_strength", proxy_quality=ProxyQuality.MODERATE)
    kwargs = {
        "curated_registry_id": "r" * 64, "combined_manifest_id": "c" * 64, "canonical_driver_id": "us_dollar_strength",
        "mapping_id": "m" * 64, "provider": "alpha_vantage", "provider_symbol": "UUP", "component_manifest_id": "e" * 64,
        "instrument_form": InstrumentForm.ETF, "proxy_policy": proxy, "adjustment_policy_id": "raw", "continuation_policy_id": None,
        "session_policy_id": "nyse", "availability_policy_id": "close_plus_1d", "timeframe": Timeframe.D1,
    }
    kwargs.update(overrides)
    return kwargs


class TestBaseAssetDatasetBinding:
    def test_construction_is_deterministic(self) -> None:
        a = create_base_asset_binding(**_base_kwargs())
        b = create_base_asset_binding(**_base_kwargs())
        assert a.binding_id == b.binding_id

    def test_different_pinned_dataset_id_changes_binding_id(self) -> None:
        a = create_base_asset_binding(**_base_kwargs())
        b = create_base_asset_binding(**_base_kwargs(pinned_dataset_id="b" * 64))
        assert a.binding_id != b.binding_id

    @pytest.mark.parametrize("alias", ["latest", "current", "newest", "active", "default", "LATEST", " current "])
    def test_rejects_mutable_alias_in_pinned_dataset_id(self, alias: str) -> None:
        with pytest.raises(SourceBindingError):
            create_base_asset_binding(**_base_kwargs(pinned_dataset_id=alias))

    def test_round_trip_json(self) -> None:
        from quant_platform.features.market_data_bridge.bindings import BaseAssetDatasetBinding

        original = create_base_asset_binding(**_base_kwargs())
        restored = BaseAssetDatasetBinding.from_json_dict(original.to_json_dict())
        assert restored == original

    def test_rejects_inverted_coverage_range(self) -> None:
        import pandas as pd

        with pytest.raises(Exception):  # noqa: B017
            create_base_asset_binding(
                **_base_kwargs(
                    required_coverage_start=pd.Timestamp("2024-06-01", tz="UTC"), required_coverage_end=pd.Timestamp("2024-01-01", tz="UTC")
                )
            )


class TestMacroDatasetBinding:
    def test_construction_is_deterministic(self) -> None:
        a = create_macro_dataset_binding(**_macro_kwargs())
        b = create_macro_dataset_binding(**_macro_kwargs())
        assert a.binding_id == b.binding_id

    def test_different_component_manifest_id_changes_binding_id(self) -> None:
        a = create_macro_dataset_binding(**_macro_kwargs())
        b = create_macro_dataset_binding(**_macro_kwargs(component_manifest_id="f" * 64))
        assert a.binding_id != b.binding_id

    def test_different_revision_policy_kind_changes_binding_id(self) -> None:
        a = create_macro_dataset_binding(**_macro_kwargs(revision_policy_kind=RevisionPolicyKind.VINTAGE_SERIES))
        b = create_macro_dataset_binding(**_macro_kwargs(revision_policy_kind=RevisionPolicyKind.FIRST_RELEASE_ONLY))
        assert a.binding_id != b.binding_id

    @pytest.mark.parametrize("alias", ["latest", "current", "newest"])
    def test_rejects_mutable_alias_in_component_manifest_id(self, alias: str) -> None:
        with pytest.raises(SourceBindingError):
            create_macro_dataset_binding(**_macro_kwargs(component_manifest_id=alias))

    def test_round_trip_json(self) -> None:
        from quant_platform.features.market_data_bridge.bindings import MacroDatasetBinding

        original = create_macro_dataset_binding(**_macro_kwargs())
        restored = MacroDatasetBinding.from_json_dict(original.to_json_dict())
        assert restored == original


class TestCrossAssetDatasetBinding:
    def test_construction_is_deterministic(self) -> None:
        a = create_cross_asset_dataset_binding(**_cross_kwargs())
        b = create_cross_asset_dataset_binding(**_cross_kwargs())
        assert a.binding_id == b.binding_id

    def test_different_component_manifest_id_changes_binding_id(self) -> None:
        a = create_cross_asset_dataset_binding(**_cross_kwargs())
        b = create_cross_asset_dataset_binding(**_cross_kwargs(component_manifest_id="f" * 64))
        assert a.binding_id != b.binding_id

    def test_different_proxy_quality_changes_binding_id(self) -> None:
        proxy_high = create_proxy_policy(is_proxy=True, proxy_for="us_dollar_strength", proxy_quality=ProxyQuality.HIGH)
        proxy_low = create_proxy_policy(is_proxy=True, proxy_for="us_dollar_strength", proxy_quality=ProxyQuality.LOW)
        a = create_cross_asset_dataset_binding(**_cross_kwargs(proxy_policy=proxy_high))
        b = create_cross_asset_dataset_binding(**_cross_kwargs(proxy_policy=proxy_low))
        assert a.binding_id != b.binding_id

    def test_etf_form_requires_is_proxy_true(self) -> None:
        not_a_proxy = create_proxy_policy(is_proxy=False)
        with pytest.raises(SourceBindingError):
            create_cross_asset_dataset_binding(**_cross_kwargs(proxy_policy=not_a_proxy))

    @pytest.mark.parametrize("alias", ["latest", "current", "newest"])
    def test_rejects_mutable_alias_in_combined_manifest_id(self, alias: str) -> None:
        with pytest.raises(SourceBindingError):
            create_cross_asset_dataset_binding(**_cross_kwargs(combined_manifest_id=alias))

    def test_round_trip_json(self) -> None:
        from quant_platform.features.market_data_bridge.bindings import CrossAssetDatasetBinding

        original = create_cross_asset_dataset_binding(**_cross_kwargs())
        restored = CrossAssetDatasetBinding.from_json_dict(original.to_json_dict())
        assert restored == original
