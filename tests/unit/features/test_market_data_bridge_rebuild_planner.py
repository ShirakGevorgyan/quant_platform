"""`features.market_data_bridge.rebuild_planner` (spec Section 18)."""

from __future__ import annotations

import pandas as pd

from quant_platform.features.market_data_bridge.rebuild_planner import (
    RebuildPlanKind,
    SourceChangeEvidence,
    plan_rebuild,
)


def _evidence(*, old_first, old_last, old_count, new_first, new_last, new_count) -> SourceChangeEvidence:
    return SourceChangeEvidence(
        source_kind="base", source_name="base", old_first_covered_time=old_first, old_last_covered_time=old_last,
        old_observation_count=old_count, new_first_covered_time=new_first, new_last_covered_time=new_last, new_observation_count=new_count,
    )


class TestPlanRebuild:
    def test_no_lineage_forces_full_rebuild(self) -> None:
        plan = plan_rebuild(
            existing_lineage=None, existing_dataset_id=None, recipe_unchanged=True, new_base_pinned_dataset_id="a", old_base_pinned_dataset_id=None,
            new_macro_component_ids={}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
        )
        assert plan.kind is RebuildPlanKind.FULL_REBUILD_REQUIRED

    def test_recipe_change_forces_full_rebuild_with_no_expected_dataset_id(self) -> None:
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=False, new_base_pinned_dataset_id="a", old_base_pinned_dataset_id="a",
            new_macro_component_ids={}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
        )
        assert plan.kind is RebuildPlanKind.FULL_REBUILD_REQUIRED
        assert plan.expected_output_dataset_id is None

    def test_nothing_changed_is_no_op(self) -> None:
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="a", old_base_pinned_dataset_id="a",
            new_macro_component_ids={"m": "1"}, old_macro_component_ids={"m": "1"}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
        )
        assert plan.kind is RebuildPlanKind.NO_OP
        assert plan.expected_output_dataset_id == "d1"

    def test_append_only_evidence_yields_append_only_plan(self) -> None:
        ev = _evidence(
            old_first=pd.Timestamp("2024-01-01", tz="UTC"), old_last=pd.Timestamp("2024-01-05", tz="UTC"), old_count=5,
            new_first=pd.Timestamp("2024-01-01", tz="UTC"), new_last=pd.Timestamp("2024-01-10", tz="UTC"), new_count=10,
        )
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="b", old_base_pinned_dataset_id="a",
            new_macro_component_ids={}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
            evidence_by_source_name={"base": ev},
        )
        assert plan.kind is RebuildPlanKind.APPEND_ONLY_SAFE_EXTENSION
        assert plan.required_warmup_from is None

    def test_correction_evidence_yields_partial_recomputation(self) -> None:
        ev = _evidence(
            old_first=pd.Timestamp("2024-01-01", tz="UTC"), old_last=pd.Timestamp("2024-01-05", tz="UTC"), old_count=5,
            new_first=pd.Timestamp("2024-01-02", tz="UTC"), new_last=pd.Timestamp("2024-01-05", tz="UTC"), new_count=5,
        )
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="b", old_base_pinned_dataset_id="a",
            new_macro_component_ids={}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
            evidence_by_source_name={"base": ev},
        )
        assert plan.kind is RebuildPlanKind.PARTIAL_RECOMPUTATION_REQUIRED
        assert plan.required_warmup_from == pd.Timestamp("2024-01-01", tz="UTC")

    def test_changed_without_evidence_forces_full_rebuild(self) -> None:
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="b", old_base_pinned_dataset_id="a",
            new_macro_component_ids={}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
        )
        assert plan.kind is RebuildPlanKind.FULL_REBUILD_REQUIRED

    def test_plan_id_is_deterministic(self) -> None:
        kwargs = {
            "existing_lineage": {"x": 1}, "existing_dataset_id": "d1", "recipe_unchanged": True, "new_base_pinned_dataset_id": "a", "old_base_pinned_dataset_id": "a",
            "new_macro_component_ids": {}, "old_macro_component_ids": {}, "new_cross_asset_component_ids": {}, "old_cross_asset_component_ids": {},
        }
        plan_a = plan_rebuild(**kwargs)
        plan_b = plan_rebuild(**kwargs)
        assert plan_a.plan_id == plan_b.plan_id

    def test_new_optional_macro_source_is_a_change(self) -> None:
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="a", old_base_pinned_dataset_id="a",
            new_macro_component_ids={"m": "1"}, old_macro_component_ids={}, new_cross_asset_component_ids={}, old_cross_asset_component_ids={},
        )
        assert plan.kind is RebuildPlanKind.FULL_REBUILD_REQUIRED
        assert "m" in plan.affected_source_names
