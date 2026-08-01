"""Curated multi-series backfill spec tests (Milestone 10, Phase 4B)."""

from __future__ import annotations

import pytest
from _curated_test_helpers import OBS_END, OBS_START, default_core_registry

from quant_platform.core.exceptions import CuratedBackfillSpecError
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, create_curated_backfill_spec


class TestFourCoreSeries:
    def test_all_four_core_series_selectable(self) -> None:
        registry = default_core_registry()
        plan = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DFII10", "DGS10", "CPIAUCSL", "DFF"), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id="a" * 64, target_dataset_namespace="ns1",
        )
        assert set(plan.selected_series_ids) == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}


class TestDeterministicOrdering:
    def test_ordering_is_alphabetical_regardless_of_input_order(self) -> None:
        registry = default_core_registry()
        plan = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DGS10", "DFF", "DFII10", "CPIAUCSL"), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id="a" * 64, target_dataset_namespace="ns1",
        )
        assert plan.selected_series_ids == ("CPIAUCSL", "DFF", "DFII10", "DGS10")

    def test_identity_independent_of_declaration_order(self) -> None:
        registry = default_core_registry()
        p1 = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10", "DFF"), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        p2 = create_curated_backfill_spec(registry=registry, selected_series_ids=("DFF", "DGS10"), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        assert p1.backfill_plan_id == p2.backfill_plan_id


class TestBoundedIntervals:
    def test_invalid_range_rejected(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_END, observation_end=OBS_START, revision_policy_id="a" * 64, target_dataset_namespace="ns1")

    def test_no_wall_clock_default_end(self) -> None:
        """`observation_end` is a REQUIRED explicit argument -- there is
        no way to construct a spec that implicitly defaults to "today"."""
        import inspect

        sig = inspect.signature(create_curated_backfill_spec)
        assert sig.parameters["observation_end"].default is inspect.Parameter.empty


class TestUnknownAndDisabledSeries:
    def test_unknown_series_rejected(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("NOPE",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")

    def test_disabled_series_rejected(self) -> None:
        from quant_platform.market_data.collectors.curated.registry import (
            create_curated_registry,
            default_core_series_specs,
        )

        specs = default_core_series_specs(registry_version=1, revision_policy_id="a" * 64, release_availability_policy_id_daily="b" * 64, release_availability_policy_id_monthly="c" * 64, default_observation_start=OBS_START)
        # All core series are enabled by construction; simulate a disabled one via the registry factory directly.
        from quant_platform.market_data.collectors.curated.registry import (
            SeriesTier,
            create_curated_series_spec,
        )
        from quant_platform.market_data.collectors.macro_normalization import MacroUnit

        disabled_spec = create_curated_series_spec(
            series_id="DISABLEDX", canonical_series_name="disabled_x", registry_version=1, tier=SeriesTier.EXPERIMENTAL, economic_category="x",
            expected_native_frequency="D", expected_units=("%",), target_macro_instrument_id="", normalization_kind=MacroUnit.PERCENT,
            revision_policy_id="a" * 64, release_availability_policy_id="b" * 64, default_observation_start=OBS_START, enabled=False,
        )
        registry2 = create_curated_registry(registry_version=1, specs=(*specs, disabled_spec))
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry2, selected_series_ids=("DISABLEDX",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")


class TestExactRetry:
    def test_same_params_same_id(self) -> None:
        registry = default_core_registry()
        p1 = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        p2 = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        assert p1.backfill_plan_id == p2.backfill_plan_id


class TestPartialSuccessVsFailFast:
    def test_fail_fast_default_true(self) -> None:
        registry = default_core_registry()
        plan = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        assert plan.fail_fast is True

    def test_partial_success_explicit(self) -> None:
        registry = default_core_registry()
        plan = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", fail_fast=False)
        assert plan.fail_fast is False


class TestPaginationAndLimits:
    def test_page_size_bounds(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", page_size=0)
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", page_size=100_001)

    def test_max_series_count_bound(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10", "DFF"), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", max_series_count=1)

    def test_max_observations_and_bytes_must_be_positive(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", max_observations_per_series=0)
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1", max_total_raw_bytes=0)


class TestEmptySelection:
    def test_empty_selection_rejected(self) -> None:
        registry = default_core_registry()
        with pytest.raises(CuratedBackfillSpecError):
            create_curated_backfill_spec(registry=registry, selected_series_ids=(), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")


class TestCachePolicyChoice:
    def test_default_is_prefer_cache(self) -> None:
        registry = default_core_registry()
        plan = create_curated_backfill_spec(registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, revision_policy_id="a" * 64, target_dataset_namespace="ns1")
        assert plan.cache_policy is CachePolicy.PREFER_CACHE
