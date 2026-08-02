"""`features.market_data_bridge.reports` (spec Section 25): deterministic,
stable-ordered rendering."""

from __future__ import annotations

from _market_data_bridge_test_helpers import (
    make_base_binding,
    make_cross_asset_fixture,
    make_macro_fixture,
    open_repository,
)

from quant_platform.features.market_data_bridge.reports import render_binding_inventory_report


class TestRenderBindingInventoryReport:
    def test_deterministic_across_dict_orderings(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        macro_a = make_macro_fixture(tmp_path / "macro_a", series_id="DFII10", days=2)
        macro_b = make_macro_fixture(tmp_path / "macro_b", series_id="DGS10", days=2)
        cross = make_cross_asset_fixture(tmp_path / "cross", days=2)

        report_1 = render_binding_inventory_report(
            base_binding=base_binding, macro_bindings={"DFII10": macro_a.binding, "DGS10": macro_b.binding}, cross_asset_bindings={"DXY": cross.binding}
        )
        report_2 = render_binding_inventory_report(
            base_binding=base_binding, macro_bindings={"DGS10": macro_b.binding, "DFII10": macro_a.binding}, cross_asset_bindings={"DXY": cross.binding}
        )
        assert report_1 == report_2
        assert "XAUUSD" in report_1
        assert "DFII10" in report_1
        assert "DGS10" in report_1
        assert "us_dollar_strength" in report_1
