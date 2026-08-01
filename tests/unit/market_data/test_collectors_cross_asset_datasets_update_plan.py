"""Tests for `ComponentMarketDatasetManifest`/`CombinedCrossAssetManifest`
and incremental update planning (Milestone 10, Phase 4C, spec Section 30
"Datasets"/"Point-in-time" completeness sub-items, Section 22)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _cross_asset_test_helpers import (
    build_default_registry_and_mappings,
    fresh_repository_and_cache,
)

from quant_platform.core.exceptions import MarketCombinedManifestError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cross_asset.datasets import (
    CombinedCrossAssetManifestStore,
    CompletenessStatus,
    ComponentMarketDatasetManifestStore,
    create_combined_cross_asset_manifest,
    create_component_market_dataset_manifest,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_record import (
    create_market_driver_bar,
)
from quant_platform.market_data.collectors.cross_asset.update_plan import (
    MappingUpdateAction,
    create_cross_asset_update_plan,
)

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


def _bar(*, open_time: datetime, mapping_symbol: str = "GLD", driver_id: str = "gold_reference"):
    from quant_platform.core.time_utils import compute_close_time

    close_time = compute_close_time(open_time, Timeframe.D1)
    return create_market_driver_bar(
        canonical_driver_id=driver_id, provider="alpha_vantage", provider_symbol=mapping_symbol, instrument_form=InstrumentForm.ETF,
        open_time=open_time, timeframe=Timeframe.D1, open=Decimal("100"), high=Decimal("101"), low=Decimal("99"), close=Decimal("100.5"),
        volume=Decimal("1000"), volume_unit="shares", availability_time=close_time, availability_policy_id="a" * 64, session_policy_id="s" * 64,
        adjustment_policy_id="j" * 64, request_manifest_id="r" * 64, response_manifest_id="p" * 64, source_manifest_id="c" * 64, source_row_index=0,
    )


class TestComponentMarketDatasetManifest:
    def test_conflicting_coordinate_count_must_be_zero(self) -> None:
        with pytest.raises(MarketCombinedManifestError):
            create_component_market_dataset_manifest(
                mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD",
                instrument_form="etf", timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64,
                bars=(), missing_business_day_count=0, conflicting_coordinate_count=1, creation_time=T0,
            )

    def test_empty_bars_produces_none_coverage(self) -> None:
        manifest = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        assert manifest.coverage_start is None
        assert manifest.coverage_end is None
        assert manifest.bar_count == 0

    def test_coverage_derived_from_bars(self) -> None:
        bars = (_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)), _bar(open_time=datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc)))
        manifest = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=bars,
            missing_business_day_count=1, conflicting_coordinate_count=0, creation_time=T0,
        )
        assert manifest.bar_count == 2
        assert manifest.coverage_start == bars[0].open_time.isoformat()
        assert manifest.coverage_end == bars[1].open_time.isoformat()

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.datasets import ComponentMarketDatasetManifest

        manifest = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        restored = ComponentMarketDatasetManifest.from_json_dict(manifest.to_json_dict())
        assert restored == manifest


class TestComponentMarketDatasetManifestStore:
    def test_idempotent_append_never_mints_new_version(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = ComponentMarketDatasetManifestStore(root)
        manifest = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        store.append(manifest)
        store.append(manifest)
        assert store.current_version("m" * 64) == 1

    def test_different_content_mints_new_version(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = ComponentMarketDatasetManifestStore(root)
        bars_1 = (_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)),)
        bars_2 = (_bar(open_time=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)), _bar(open_time=datetime(2024, 1, 4, 14, 30, tzinfo=timezone.utc)))
        manifest_1 = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=bars_1,
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest_2 = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=bars_2,
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        store.append(manifest_1)
        store.append(manifest_2)
        assert store.current_version("m" * 64) == 2


class TestCombinedCrossAssetManifest:
    def test_missing_required_driver_forces_partial(self) -> None:
        component = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={"m" * 64: component},
            required_driver_ids=("gold_reference", "silver"), creation_time=T0,
        )
        assert manifest.completeness_status == CompletenessStatus.PARTIAL
        assert manifest.missing_required_driver_ids == ("silver",)

    def test_all_required_present_is_complete(self) -> None:
        component = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={"m" * 64: component},
            required_driver_ids=("gold_reference",), creation_time=T0,
        )
        assert manifest.completeness_status == CompletenessStatus.COMPLETE
        assert manifest.missing_required_driver_ids == ()

    def test_empty_components_rejected(self) -> None:
        with pytest.raises(MarketCombinedManifestError):
            create_combined_cross_asset_manifest(
                curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={},
                required_driver_ids=(), creation_time=T0,
            )

    def test_component_swap_changes_identity(self) -> None:
        """Combined-manifest-component-swap detection (adversarial audit
        item): binding a DIFFERENT component_manifest_id for the same
        mapping_id must produce a DIFFERENT combined_manifest_id."""
        component_a = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        component_b = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(_bar(open_time=T0),),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest_a = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={"m" * 64: component_a},
            required_driver_ids=("gold_reference",), creation_time=T0,
        )
        manifest_b = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={"m" * 64: component_b},
            required_driver_ids=("gold_reference",), creation_time=T0,
        )
        assert manifest_a.combined_manifest_id != manifest_b.combined_manifest_id

    def test_json_round_trip(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.datasets import CombinedCrossAssetManifest

        component = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns", component_manifests={"m" * 64: component},
            required_driver_ids=("gold_reference",), creation_time=T0,
        )
        restored = CombinedCrossAssetManifest.from_json_dict(manifest.to_json_dict())
        assert restored.combined_manifest_id == manifest.combined_manifest_id


class TestCombinedCrossAssetManifestStore:
    def test_idempotent_append(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        store = CombinedCrossAssetManifestStore(root)
        component = create_component_market_dataset_manifest(
            mapping_id="m" * 64, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="a" * 64, session_policy_id="s" * 64, availability_policy_id="v" * 64, bars=(),
            missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        manifest = create_combined_cross_asset_manifest(
            curated_registry_id="r" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="test_ns", component_manifests={"m" * 64: component},
            required_driver_ids=("gold_reference",), creation_time=T0,
        )
        store.append(manifest)
        store.append(manifest)
        assert store.current_version("test_ns") == 1


class TestCrossAssetUpdatePlan:
    def test_no_existing_component_yields_append(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(root)
        plan = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=T0, planning_time=T0,
        )
        assert not plan.is_exact_no_op()
        assert plan.entries[0].action is MappingUpdateAction.APPEND_BARS

    def test_up_to_date_component_yields_no_op(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        driver_spec = registry.get("gold_reference")
        assert driver_spec is not None
        component_store = ComponentMarketDatasetManifestStore(root)
        bar = _bar(open_time=T0)
        component = create_component_market_dataset_manifest(
            mapping_id=mapping.mapping_id, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id=driver_spec.adjustment_policy_id, session_policy_id="s" * 64, availability_policy_id="v" * 64,
            bars=(bar,), missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0, continuation_policy_id=mapping.continuation_policy_id,
        )
        component_store.append(component)
        plan = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=bar.open_time, planning_time=T0,
        )
        assert plan.is_exact_no_op()
        assert plan.entries[0].action is MappingUpdateAction.NO_UPDATE_NEEDED

    def test_policy_change_triggers_refresh(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(root)
        bar = _bar(open_time=T0)
        component = create_component_market_dataset_manifest(
            mapping_id=mapping.mapping_id, canonical_driver_id="gold_reference", provider="alpha_vantage", provider_symbol="GLD", instrument_form="etf",
            timeframe="D1", adjustment_policy_id="STALE_POLICY_ID" + "0" * 49, session_policy_id="s" * 64, availability_policy_id="v" * 64,
            bars=(bar,), missing_business_day_count=0, conflicting_coordinate_count=0, creation_time=T0,
        )
        component_store.append(component)
        plan = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=bar.open_time, planning_time=T0,
        )
        assert plan.entries[0].action is MappingUpdateAction.POLICY_REFRESH

    def test_unknown_mapping_reported_as_issue(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        component_store = ComponentMarketDatasetManifestStore(root)
        plan = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=("f" * 64,), target_dataset_namespace="ns", desired_end_time=T0, planning_time=T0,
        )
        assert any("unknown_mapping" in issue for issue in plan.plan_issues)
        assert not plan.entries

    def test_futures_form_flags_roll_refresh_need(self) -> None:
        from quant_platform.market_data.collectors.cross_asset.adjustment import (
            AdjustmentPolicyKind,
            create_adjustment_policy,
        )
        from quant_platform.market_data.collectors.cross_asset.futures import (
            ContinuationPolicyKind,
            create_continuation_policy,
        )
        from quant_platform.market_data.collectors.cross_asset.instrument_form import create_proxy_policy
        from quant_platform.market_data.collectors.cross_asset.registry import (
            DriverTier,
            create_curated_market_driver_registry,
            create_curated_market_driver_spec,
        )
        from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
            create_provider_symbol_mapping,
            create_symbol_mapping_set,
        )

        root, _repo, _cache = fresh_repository_and_cache()
        continuation = create_continuation_policy(kind=ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS)
        proxy = create_proxy_policy(is_proxy=False)
        mapping = create_provider_symbol_mapping(
            provider="fixture", provider_symbol="CL1!", canonical_driver_id="wti_crude", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
            currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.NOT_APPLICABLE, proxy_policy=proxy, continuation_policy_id=continuation.continuation_policy_id,
        )
        mapping_set = create_symbol_mapping_set((mapping,))
        adjustment = create_adjustment_policy(kind=AdjustmentPolicyKind.NOT_APPLICABLE)
        spec = create_curated_market_driver_spec(
            canonical_driver_id="wti_crude", canonical_name="WTI", registry_version=1, tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER, economic_role="x",
            is_required=True, asset_class="energy", preferred_instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
            allowed_instrument_forms=(InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,), canonical_currency="USD", canonical_quote_unit="usd_per_barrel",
            expected_frequency="daily", session_policy_id="s" * 64, adjustment_policy=adjustment, availability_policy_id="v" * 64,
            continuation_policy_id=continuation.continuation_policy_id, provider_mapping_ids=(mapping.mapping_id,), enabled=True,
        )
        registry = create_curated_market_driver_registry(registry_version=1, specs=(spec,))
        component_store = ComponentMarketDatasetManifestStore(root)
        plan = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=T0, planning_time=T0,
        )
        assert plan.entries[0].needs_futures_roll_refresh

    def test_plan_identity_deterministic(self) -> None:
        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(root)
        plan_1 = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=T0, planning_time=T0,
        )
        plan_2 = create_cross_asset_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
            selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=T0, planning_time=T0,
        )
        assert plan_1.update_plan_id == plan_2.update_plan_id

    def test_naive_desired_end_time_rejected(self) -> None:
        from quant_platform.core.exceptions import MarketDataError

        root, _repo, _cache = fresh_repository_and_cache()
        registry, mapping_set, _session, _avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        component_store = ComponentMarketDatasetManifestStore(root)
        with pytest.raises(MarketDataError):
            create_cross_asset_update_plan(
                existing_combined_manifest=None, component_store=component_store, registry=registry, mapping_set=mapping_set,
                selected_mapping_ids=(mapping.mapping_id,), target_dataset_namespace="ns", desired_end_time=datetime(2024, 1, 5), planning_time=T0,  # type: ignore[arg-type]
            )
