"""Tests for the curated cross-asset driver registry and provider symbol
mappings (Milestone 10, Phase 4C, spec Section 30 "Registry"/"Mappings")."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _cross_asset_test_helpers import (
    build_default_alpha_vantage_mappings,
    build_default_registry_and_mappings,
)

from quant_platform.core.exceptions import (
    AdjustmentPolicyError,
    InstrumentFormError,
    MarketDriverRegistryError,
    SymbolMappingError,
)
from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicyKind,
    create_adjustment_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.registry import (
    DriverTier,
    create_curated_market_driver_registry,
    create_curated_market_driver_spec,
    default_core_driver_ids,
    default_optional_driver_ids,
)
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    create_provider_symbol_mapping,
    create_symbol_mapping_set,
)


def _minimal_spec(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "canonical_driver_id": "test_driver", "canonical_name": "Test Driver", "registry_version": 1, "tier": DriverTier.SECONDARY_MARKET_DRIVER,
        "economic_role": "test role", "is_required": False, "asset_class": "test_asset", "preferred_instrument_form": InstrumentForm.SPOT,
        "allowed_instrument_forms": (InstrumentForm.SPOT,), "canonical_currency": "USD", "canonical_quote_unit": "usd_per_unit",
        "expected_frequency": "daily", "session_policy_id": "s" * 64, "adjustment_policy": create_adjustment_policy(kind=AdjustmentPolicyKind.NOT_APPLICABLE),
        "availability_policy_id": "a" * 64, "enabled": False,
    }
    defaults.update(overrides)
    return create_curated_market_driver_spec(**defaults)  # type: ignore[arg-type]


class TestCuratedMarketDriverSpec:
    def test_minimal_spec_constructs(self) -> None:
        spec = _minimal_spec()
        assert spec.canonical_driver_id == "test_driver"

    def test_preferred_form_must_be_in_allowed_forms(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(preferred_instrument_form=InstrumentForm.ETF, allowed_instrument_forms=(InstrumentForm.SPOT,))

    def test_empty_allowed_forms_rejected(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(allowed_instrument_forms=())

    def test_duplicate_allowed_forms_rejected(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.SPOT), preferred_instrument_form=InstrumentForm.SPOT)

    def test_futures_form_without_continuation_policy_rejected(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(
                preferred_instrument_form=InstrumentForm.EXCHANGE_FUTURES_CONTRACT, allowed_instrument_forms=(InstrumentForm.EXCHANGE_FUTURES_CONTRACT,),
                continuation_policy_id=None,
            )

    def test_continuation_policy_without_futures_form_rejected(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(continuation_policy_id="c" * 64)

    def test_futures_form_with_continuation_policy_accepted(self) -> None:
        spec = _minimal_spec(
            preferred_instrument_form=InstrumentForm.EXCHANGE_FUTURES_CONTRACT, allowed_instrument_forms=(InstrumentForm.EXCHANGE_FUTURES_CONTRACT,),
            continuation_policy_id="c" * 64,
        )
        assert spec.continuation_policy_id == "c" * 64

    def test_enabled_requires_provider_mapping_ids(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(enabled=True, provider_mapping_ids=())

    def test_enabled_with_mapping_ids_accepted(self) -> None:
        spec = _minimal_spec(enabled=True, provider_mapping_ids=("m" * 64,))
        assert spec.enabled

    def test_duplicate_provider_mapping_ids_rejected(self) -> None:
        with pytest.raises(MarketDriverRegistryError):
            _minimal_spec(provider_mapping_ids=("m" * 64, "m" * 64))

    def test_etf_form_without_equity_like_adjustment_rejected(self) -> None:
        with pytest.raises(AdjustmentPolicyError):
            _minimal_spec(
                preferred_instrument_form=InstrumentForm.ETF, allowed_instrument_forms=(InstrumentForm.ETF,),
                adjustment_policy=create_adjustment_policy(kind=AdjustmentPolicyKind.NOT_APPLICABLE),
            )

    def test_etf_form_with_equity_like_adjustment_accepted(self) -> None:
        spec = _minimal_spec(
            preferred_instrument_form=InstrumentForm.ETF, allowed_instrument_forms=(InstrumentForm.ETF,),
            adjustment_policy=create_adjustment_policy(kind=AdjustmentPolicyKind.RAW_UNADJUSTED),
        )
        assert spec.adjustment_policy_id

    def test_identity_excludes_notes(self) -> None:
        spec_a = _minimal_spec(notes="note A")
        spec_b = _minimal_spec(notes="note B")
        # Notes are excluded from to_identity_payload -- registry identity built from these two must match.
        registry_a = create_curated_market_driver_registry(registry_version=1, specs=(spec_a,))
        registry_b = create_curated_market_driver_registry(registry_version=1, specs=(spec_b,))
        assert registry_a.registry_id == registry_b.registry_id

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverSpec

        spec = _minimal_spec(enabled=True, provider_mapping_ids=("m" * 64,))
        restored = CuratedMarketDriverSpec.from_json_dict(spec.to_json_dict())
        assert restored == spec


class TestCuratedMarketDriverRegistry:
    def test_duplicate_canonical_driver_id_rejected(self) -> None:
        spec_a = _minimal_spec(canonical_driver_id="dup", canonical_name="A")
        spec_b = _minimal_spec(canonical_driver_id="dup", canonical_name="B")
        with pytest.raises(MarketDriverRegistryError):
            create_curated_market_driver_registry(registry_version=1, specs=(spec_a, spec_b))

    def test_duplicate_canonical_name_rejected(self) -> None:
        spec_a = _minimal_spec(canonical_driver_id="a", canonical_name="Same Name")
        spec_b = _minimal_spec(canonical_driver_id="b", canonical_name="Same Name")
        with pytest.raises(MarketDriverRegistryError):
            create_curated_market_driver_registry(registry_version=1, specs=(spec_a, spec_b))

    def test_registry_version_mismatch_rejected(self) -> None:
        spec = _minimal_spec(registry_version=2)
        with pytest.raises(MarketDriverRegistryError):
            create_curated_market_driver_registry(registry_version=1, specs=(spec,))

    def test_deterministic_ordering_independent_of_declaration_order(self) -> None:
        spec_a = _minimal_spec(canonical_driver_id="zzz", canonical_name="Z")
        spec_b = _minimal_spec(canonical_driver_id="aaa", canonical_name="A")
        registry_1 = create_curated_market_driver_registry(registry_version=1, specs=(spec_a, spec_b))
        registry_2 = create_curated_market_driver_registry(registry_version=1, specs=(spec_b, spec_a))
        assert registry_1.registry_id == registry_2.registry_id
        assert [s.canonical_driver_id for s in registry_1.specs] == ["aaa", "zzz"]

    def test_get_returns_none_for_unknown(self) -> None:
        registry = create_curated_market_driver_registry(registry_version=1, specs=(_minimal_spec(),))
        assert registry.get("does_not_exist") is None

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry

        registry = create_curated_market_driver_registry(registry_version=1, specs=(_minimal_spec(),))
        restored = CuratedMarketDriverRegistry.from_json_dict(registry.to_json_dict())
        assert restored.registry_id == registry.registry_id


class TestDefaultPopulatedRegistry:
    def test_all_5_mandatory_core_concepts_present(self) -> None:
        registry, _mapping_set, _session, _avail = build_default_registry_and_mappings()
        required = set(registry.required_driver_ids())
        assert required == set(default_core_driver_ids())
        assert required == {"us_dollar_strength", "wti_crude", "brent_crude", "silver", "gold_reference"}

    def test_all_5_optional_concepts_present_even_if_unsupported(self) -> None:
        registry, _mapping_set, _session, _avail = build_default_registry_and_mappings()
        all_ids = {s.canonical_driver_id for s in registry.specs}
        assert set(default_optional_driver_ids()) <= all_ids

    def test_treasury_volatility_unsupported_and_fails_closed(self) -> None:
        registry, _mapping_set, _session, _avail = build_default_registry_and_mappings()
        spec = registry.get("treasury_volatility")
        assert spec is not None
        assert not spec.enabled
        assert spec.provider_mapping_ids == ()

    def test_9_of_10_concepts_enabled(self) -> None:
        registry, _mapping_set, _session, _avail = build_default_registry_and_mappings()
        assert len(registry.enabled_driver_ids()) == 9

    def test_every_alpha_vantage_mapping_is_proxy(self) -> None:
        mappings = build_default_alpha_vantage_mappings()
        assert all(m.proxy_policy.is_proxy for m in mappings)
        assert all(m.instrument_form is InstrumentForm.ETF for m in mappings)

    def test_gold_reference_mapping_never_labeled_as_spot(self) -> None:
        mappings = build_default_alpha_vantage_mappings(driver_ids=("gold_reference",))
        assert mappings[0].instrument_form is not InstrumentForm.SPOT
        assert mappings[0].proxy_policy.is_proxy


class TestProviderSymbolMapping:
    def test_etf_form_requires_is_proxy_true(self) -> None:
        not_proxy = create_proxy_policy(is_proxy=False)
        with pytest.raises(SymbolMappingError):
            create_provider_symbol_mapping(
                provider="alpha_vantage", provider_symbol="GLD", canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF,
                currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=not_proxy,
            )

    def test_futures_form_requires_continuation_policy_id(self) -> None:
        proxy = create_proxy_policy(is_proxy=False)
        with pytest.raises(SymbolMappingError):
            create_provider_symbol_mapping(
                provider="test", provider_symbol="CL1", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.EXCHANGE_FUTURES_CONTRACT,
                currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.NOT_APPLICABLE, proxy_policy=proxy, continuation_policy_id=None,
            )

    def test_non_futures_form_with_continuation_policy_id_rejected(self) -> None:
        proxy = create_proxy_policy(is_proxy=False)
        with pytest.raises(SymbolMappingError):
            create_provider_symbol_mapping(
                provider="test", provider_symbol="XYZ", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.SPOT,
                currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.NOT_APPLICABLE, proxy_policy=proxy, continuation_policy_id="c" * 64,
            )

    def test_ambiguous_mapping_rejected(self) -> None:
        proxy_a = create_proxy_policy(is_proxy=True, proxy_for="driver_a", proxy_quality=ProxyQuality.HIGH)
        proxy_b = create_proxy_policy(is_proxy=True, proxy_for="driver_b", proxy_quality=ProxyQuality.HIGH)
        mapping_a = create_provider_symbol_mapping(
            provider="p", provider_symbol="SAME", canonical_driver_id="driver_a", instrument_form=InstrumentForm.ETF, currency="USD",
            adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy_a,
        )
        mapping_b = create_provider_symbol_mapping(
            provider="p", provider_symbol="SAME", canonical_driver_id="driver_b", instrument_form=InstrumentForm.ETF, currency="USD",
            adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy_b,
        )
        with pytest.raises(SymbolMappingError):
            create_symbol_mapping_set((mapping_a, mapping_b))

    def test_different_mapping_version_allows_different_driver(self) -> None:
        """An alias change is a NEW mapping VERSION, never ambiguous within one version."""
        proxy_a = create_proxy_policy(is_proxy=True, proxy_for="driver_a", proxy_quality=ProxyQuality.HIGH)
        proxy_b = create_proxy_policy(is_proxy=True, proxy_for="driver_b", proxy_quality=ProxyQuality.HIGH)
        mapping_v1 = create_provider_symbol_mapping(
            provider="p", provider_symbol="SAME", canonical_driver_id="driver_a", instrument_form=InstrumentForm.ETF, currency="USD",
            adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy_a, mapping_version=1, enabled=False,
        )
        mapping_v2 = create_provider_symbol_mapping(
            provider="p", provider_symbol="SAME", canonical_driver_id="driver_b", instrument_form=InstrumentForm.ETF, currency="USD",
            adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy_b, mapping_version=2,
        )
        mapping_set = create_symbol_mapping_set((mapping_v1, mapping_v2))
        assert len(mapping_set.mappings) == 2

    def test_duplicate_mapping_id_rejected(self) -> None:
        proxy = create_proxy_policy(is_proxy=True, proxy_for="driver_a", proxy_quality=ProxyQuality.HIGH)
        mapping = create_provider_symbol_mapping(
            provider="p", provider_symbol="SAME", canonical_driver_id="driver_a", instrument_form=InstrumentForm.ETF, currency="USD",
            adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy,
        )
        with pytest.raises(SymbolMappingError):
            create_symbol_mapping_set((mapping, mapping))

    def test_for_driver_returns_only_matching(self) -> None:
        mappings = build_default_alpha_vantage_mappings()
        mapping_set = create_symbol_mapping_set(mappings)
        gold_mappings = mapping_set.for_driver("gold_reference")
        assert len(gold_mappings) == 1
        assert gold_mappings[0].provider_symbol == "GLD"

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.symbol_mapping import ProviderSymbolMapping

        mapping = build_default_alpha_vantage_mappings(driver_ids=("gold_reference",))[0]
        restored = ProviderSymbolMapping.from_json_dict(mapping.to_json_dict())
        assert restored == mapping


class TestInstrumentFormProxyPolicy:
    def test_is_proxy_requires_proxy_for(self) -> None:
        with pytest.raises(InstrumentFormError):
            create_proxy_policy(is_proxy=True, proxy_for=None, proxy_quality=ProxyQuality.HIGH)

    def test_is_proxy_requires_proxy_quality(self) -> None:
        with pytest.raises(InstrumentFormError):
            create_proxy_policy(is_proxy=True, proxy_for="x", proxy_quality=None)

    def test_not_proxy_forbids_proxy_for(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.instrument_form import ProxyPolicy

        with pytest.raises(InstrumentFormError):
            ProxyPolicy(
                is_proxy=False, proxy_for="x", proxy_quality=None, known_basis_risk="", roll_risk="", tracking_error_risk="",
                currency_difference_note="", session_difference_note="", adjustment_difference_note="",
            )

    def test_not_proxy_forbids_proxy_quality(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.instrument_form import ProxyPolicy

        with pytest.raises(InstrumentFormError):
            ProxyPolicy(
                is_proxy=False, proxy_for=None, proxy_quality=ProxyQuality.HIGH, known_basis_risk="", roll_risk="", tracking_error_risk="",
                currency_difference_note="", session_difference_note="", adjustment_difference_note="",
            )
