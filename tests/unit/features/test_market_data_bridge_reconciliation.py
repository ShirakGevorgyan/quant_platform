"""`features.market_data_bridge.reconciliation` (spec Section 19)."""

from __future__ import annotations

import pandas as pd
from _market_data_bridge_test_helpers import make_cross_asset_fixture, make_macro_fixture

from quant_platform.core.exceptions import BridgeReconciliationError
from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.reconciliation import (
    ReconciliationIssueCode,
    reconcile_binding_source,
    reconcile_no_pre_availability_macro_leakage,
    reconcile_no_pre_close_cross_asset_leakage,
)


class TestReconcileBindingSource:
    def test_clean_macro_binding_reports_no_issues(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=5)
        report = reconcile_binding_source(macro=(fixture.observation_store, fixture.manifest_store, fixture.binding))
        assert report.is_clean

    def test_clean_cross_asset_binding_reports_no_issues(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=5)
        report = reconcile_binding_source(cross_asset=(fixture.bar_store, fixture.manifest_store, fixture.binding))
        assert report.is_clean

    def test_broken_binding_is_reported_not_raised(self, tmp_path) -> None:
        from dataclasses import replace

        fixture = make_macro_fixture(tmp_path, days=3)
        broken = replace(fixture.binding, component_manifest_id="9" * 64, binding_id="")
        report = reconcile_binding_source(macro=(fixture.observation_store, fixture.manifest_store, broken))
        assert not report.is_clean
        assert report.issues[0].code is ReconciliationIssueCode.WRONG_COMPONENT_VERSION

    def test_requires_exactly_one_source_kind(self) -> None:
        try:
            reconcile_binding_source()
            raise AssertionError("expected BridgeReconciliationError")
        except BridgeReconciliationError:
            pass


class TestLeakageReconciliation:
    def test_clean_macro_alignment_reports_no_leakage(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-05T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        report = reconcile_no_pre_availability_macro_leakage(base_avail, macro_df, source_name="dfii10")
        assert report.is_clean

    def test_clean_cross_asset_alignment_reports_no_leakage(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-05T00:00Z"], utc=True))
        cross_df = pd.DataFrame({"open_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True), "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
        report = reconcile_no_pre_close_cross_asset_leakage(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1)
        assert report.is_clean
