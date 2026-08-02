"""`features.market_data_bridge.coverage`: source-coverage policy
enforcement (spec Section 10)."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import SourceCoverageError
from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.coverage import (
    SourceCoveragePolicy,
    SourceCoveragePolicyKind,
    evaluate_missing_runs,
    evaluate_source_coverage,
)


def _base_df(hours: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")})


class TestFailRequiredSource:
    def test_raises_when_required_macro_source_has_zero_coverage(self) -> None:
        policy = SourceCoveragePolicy(kind=SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE)
        with pytest.raises(SourceCoverageError):
            evaluate_source_coverage(
                base_df=_base_df(), base_timeframe=Timeframe.H1, macro_frames={"dfii10": pd.DataFrame({"release_time": pd.to_datetime([], utc=True), "value": []})},
                macro_bindings={}, cross_asset_frames={}, cross_asset_bindings={},
                requested_start=pd.Timestamp("2024-01-01", tz="UTC"), requested_end=pd.Timestamp("2024-01-01T10:00Z"), policy=policy,
            )

    def test_passes_when_all_required_sources_covered(self) -> None:
        policy = SourceCoveragePolicy(kind=SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE)
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        report = evaluate_source_coverage(
            base_df=_base_df(), base_timeframe=Timeframe.H1, macro_frames={"dfii10": macro_df}, macro_bindings={}, cross_asset_frames={},
            cross_asset_bindings={}, requested_start=pd.Timestamp("2024-01-01", tz="UTC"), requested_end=pd.Timestamp("2024-01-01T10:00Z"), policy=policy,
        )
        assert all(f.status == "ok" for f in report.findings)


class TestAllowOptionalMissingAndReport:
    def test_optional_source_shortfall_is_reported_not_raised(self) -> None:
        policy = SourceCoveragePolicy(kind=SourceCoveragePolicyKind.ALLOW_OPTIONAL_MISSING_AND_REPORT)
        report = evaluate_source_coverage(
            base_df=_base_df(), base_timeframe=Timeframe.H1, macro_frames={}, macro_bindings={},
            cross_asset_frames={"dxy": pd.DataFrame({"open_time": pd.to_datetime([], utc=True), "open": [], "high": [], "low": [], "close": [], "volume": []})},
            cross_asset_bindings={}, requested_start=pd.Timestamp("2024-01-01", tz="UTC"), requested_end=pd.Timestamp("2024-01-01T10:00Z"), policy=policy,
        )
        assert report.has_insufficient_optional


class TestTrimToCommonSafeRange:
    def test_trims_to_the_narrower_required_source_range(self) -> None:
        policy = SourceCoveragePolicy(kind=SourceCoveragePolicyKind.TRIM_TO_COMMON_SAFE_RANGE)
        from quant_platform.features.market_data_bridge.bindings import create_macro_dataset_binding
        from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind

        binding = create_macro_dataset_binding(
            curated_registry_id="r" * 64, combined_universe_manifest_id="c" * 64, series_id="DFII10", canonical_series_name="x",
            provider="fred", component_manifest_id="d" * 64, revision_policy_id="p" * 64, revision_policy_kind=RevisionPolicyKind.VINTAGE_SERIES,
            availability_policy_id="ap1", native_frequency="daily", normalized_unit="percent", required=True,
        )
        macro_df = pd.DataFrame({"value": [1.0, 1.1], "release_time": pd.to_datetime(["2024-01-01T02:00Z", "2024-01-01T05:00Z"], utc=True)})
        report = evaluate_source_coverage(
            base_df=_base_df(hours=10), base_timeframe=Timeframe.H1, macro_frames={"dfii10": macro_df}, macro_bindings={"dfii10": binding},
            cross_asset_frames={}, cross_asset_bindings={}, requested_start=pd.Timestamp("2024-01-01", tz="UTC"),
            requested_end=pd.Timestamp("2024-01-01T10:00Z"), policy=policy,
        )
        assert report.trimmed
        assert report.safe_end == pd.Timestamp("2024-01-01T05:00Z", tz="UTC")
        assert report.trim_reason is not None

    def test_no_overlap_raises(self) -> None:
        policy = SourceCoveragePolicy(kind=SourceCoveragePolicyKind.TRIM_TO_COMMON_SAFE_RANGE)
        from quant_platform.features.market_data_bridge.bindings import create_macro_dataset_binding
        from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind

        binding = create_macro_dataset_binding(
            curated_registry_id="r" * 64, combined_universe_manifest_id="c" * 64, series_id="DFII10", canonical_series_name="x",
            provider="fred", component_manifest_id="d" * 64, revision_policy_id="p" * 64, revision_policy_kind=RevisionPolicyKind.VINTAGE_SERIES,
            availability_policy_id="ap1", native_frequency="daily", normalized_unit="percent", required=True,
        )
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2025-06-01T00:00Z"], utc=True)})
        with pytest.raises(SourceCoverageError):
            evaluate_source_coverage(
                base_df=_base_df(hours=10), base_timeframe=Timeframe.H1, macro_frames={"dfii10": macro_df}, macro_bindings={"dfii10": binding},
                cross_asset_frames={}, cross_asset_bindings={}, requested_start=pd.Timestamp("2024-01-01", tz="UTC"),
                requested_end=pd.Timestamp("2024-01-01T10:00Z"), policy=policy,
            )


class TestEvaluateMissingRuns:
    def test_flags_runs_exceeding_the_threshold(self) -> None:
        missing = pd.Series([False, True, True, True, False, True, True, True, True, True])
        intervals = evaluate_missing_runs(missing, max_consecutive_missing_aligned_rows=2)
        assert len(intervals) == 2
        assert intervals[0].run_length == 3
        assert intervals[1].run_length == 5

    def test_none_threshold_never_flags(self) -> None:
        missing = pd.Series([True] * 100)
        assert evaluate_missing_runs(missing, max_consecutive_missing_aligned_rows=None) == ()

    def test_run_touching_the_end_of_series_is_flagged(self) -> None:
        missing = pd.Series([False, True, True, True])
        intervals = evaluate_missing_runs(missing, max_consecutive_missing_aligned_rows=1)
        assert len(intervals) == 1
        assert intervals[0].end_index == 3
