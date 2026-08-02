"""`features.market_data_bridge.staleness` (spec Section 11)."""

from __future__ import annotations

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.staleness import (
    evaluate_cross_asset_staleness,
    evaluate_macro_staleness,
)


class TestEvaluateMacroStaleness:
    def test_fresh_value_has_no_stale_rows(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-01T00:30Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        finding = evaluate_macro_staleness(base_avail, macro_df, source_name="dfii10", threshold=pd.Timedelta(days=1))
        assert finding.stale_row_count == 0
        assert finding.unavailable_row_count == 0

    def test_no_release_yet_counts_as_unavailable_not_stale(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2023-12-01T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        finding = evaluate_macro_staleness(base_avail, macro_df, source_name="dfii10", threshold=pd.Timedelta(days=1))
        assert finding.unavailable_row_count == 1
        assert finding.stale_row_count == 0

    def test_old_value_beyond_threshold_is_stale(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-02-01T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        finding = evaluate_macro_staleness(base_avail, macro_df, source_name="dfii10", threshold=pd.Timedelta(days=3))
        assert finding.stale_row_count == 1
        assert finding.unavailable_row_count == 0

    def test_no_threshold_never_flags_stale(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2030-01-01T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        finding = evaluate_macro_staleness(base_avail, macro_df, source_name="dfii10", threshold=None)
        assert finding.stale_row_count == 0
        assert finding.threshold_seconds is None


class TestEvaluateCrossAssetStaleness:
    def test_fresh_bar_has_no_stale_rows(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-02T01:00Z"], utc=True))
        cross_df = pd.DataFrame({"open_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True), "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
        finding = evaluate_cross_asset_staleness(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1, threshold=pd.Timedelta(days=5))
        assert finding.stale_row_count == 0

    def test_stale_bar_beyond_threshold(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-02-01T00:00Z"], utc=True))
        cross_df = pd.DataFrame({"open_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True), "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
        finding = evaluate_cross_asset_staleness(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1, threshold=pd.Timedelta(days=3))
        assert finding.stale_row_count == 1
