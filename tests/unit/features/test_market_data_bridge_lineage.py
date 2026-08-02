"""`features.market_data_bridge.lineage` (spec Section 16): deterministic
assembly + content fingerprint stability."""

from __future__ import annotations

from dataclasses import replace

from _market_data_bridge_test_helpers import (
    make_base_binding,
    make_cross_asset_fixture,
    make_macro_fixture,
    open_repository,
)

from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage, lineage_content_id


class TestBuildMarketDataLineage:
    def test_deterministic_regardless_of_dict_order(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        macro_a = make_macro_fixture(tmp_path / "macro_a", series_id="DFII10", days=3)
        macro_b = make_macro_fixture(tmp_path / "macro_b", series_id="DGS10", days=3)

        lineage_1 = build_market_data_lineage(
            base_binding=base_binding, macro_bindings={"DFII10": macro_a.binding, "DGS10": macro_b.binding}, cross_asset_bindings={},
        )
        lineage_2 = build_market_data_lineage(
            base_binding=base_binding, macro_bindings={"DGS10": macro_b.binding, "DFII10": macro_a.binding}, cross_asset_bindings={},
        )
        assert lineage_1 == lineage_2
        assert lineage_content_id(lineage_1) == lineage_content_id(lineage_2)

    def test_different_binding_changes_content_id(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        cross = make_cross_asset_fixture(tmp_path / "cross")

        lineage_with = build_market_data_lineage(base_binding=base_binding, macro_bindings={}, cross_asset_bindings={"DXY": cross.binding})
        lineage_without = build_market_data_lineage(base_binding=base_binding, macro_bindings={}, cross_asset_bindings={})
        assert lineage_content_id(lineage_with) != lineage_content_id(lineage_without)

    def test_different_base_pin_changes_content_id(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        other_binding = replace(base_binding, pinned_dataset_id="9" * 64, binding_id="")

        lineage_a = build_market_data_lineage(base_binding=base_binding, macro_bindings={}, cross_asset_bindings={})
        lineage_b = build_market_data_lineage(base_binding=other_binding, macro_bindings={}, cross_asset_bindings={})
        assert lineage_content_id(lineage_a) != lineage_content_id(lineage_b)

    def test_schema_version_present(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        lineage = build_market_data_lineage(base_binding=base_binding, macro_bindings={}, cross_asset_bindings={})
        assert lineage["schema_version"] == 1
