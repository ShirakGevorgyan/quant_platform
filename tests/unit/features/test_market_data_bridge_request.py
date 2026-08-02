"""`features.market_data_bridge.request`: the orchestration entry point,
wired to the REAL, UNMODIFIED `features.dataset_builder.
ResearchDatasetBuilder` (spec Section 21's own "must use the REAL
existing ResearchDatasetBuilder, not a test replacement")."""

from __future__ import annotations

import pandas as pd
from _market_data_bridge_test_helpers import (
    make_base_binding,
    make_cross_asset_fixture,
    make_macro_fixture,
    open_repository,
)

from quant_platform.core.types import Timeframe
from quant_platform.features.cross_asset.cross_asset import register_cross_asset_features
from quant_platform.features.dataset_builder import ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.macro.macro_features import MacroSourceConfig, register_macro_features
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.market_data_bridge.coverage import SourceCoveragePolicy, SourceCoveragePolicyKind
from quant_platform.features.market_data_bridge.request import (
    CrossAssetRepository,
    MacroRepository,
    MarketDataResearchDatasetRequest,
    build_research_dataset_from_market_data,
)
from quant_platform.features.registry import FeatureRegistry


def _build(tmp_path, *, coverage_policy_kind: SourceCoveragePolicyKind = SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE):
    repo = open_repository(tmp_path / "md")
    base_binding = make_base_binding(repo, hours=200)
    macro = make_macro_fixture(tmp_path / "macro", days=10)
    cross = make_cross_asset_fixture(tmp_path / "cross", days=10)

    registry = FeatureRegistry()
    register_macro_features(registry, base_timeframe=Timeframe.H1, config=MacroSourceConfig(source_name="DFII10"))
    register_cross_asset_features(
        registry, base_timeframe=Timeframe.H1, base_momentum_window=5, base_volatility_window=10, cross_asset_symbol="DXY", cross_asset_timeframe=Timeframe.D1
    )
    feature_names = tuple(s.name for s in registry.list_features())

    research_store = ResearchDatasetStore(str(tmp_path / "research"))
    manifest_store = ResearchManifestStore(str(tmp_path / "research"))

    build_request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.H1, start=pd.Timestamp("2024-01-03T00:00Z"), end=pd.Timestamp("2024-01-08T00:00Z"),
        feature_names=feature_names, label_definition=LabelDefinition(name="fwd_ret_5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological",
    )
    req = MarketDataResearchDatasetRequest(
        base_binding=base_binding, macro_bindings={"DFII10": macro.binding}, cross_asset_bindings={"DXY": cross.binding},
        coverage_policy=SourceCoveragePolicy(kind=coverage_policy_kind), build_request=build_request,
    )
    return build_research_dataset_from_market_data(
        market_data_repository=repo, macro_repository=MacroRepository(macro.observation_store, macro.manifest_store),
        cross_asset_repository=CrossAssetRepository(cross.bar_store, cross.manifest_store), registry=registry, research_store=research_store,
        manifest_store=manifest_store, request=req,
    )


class TestBuildResearchDatasetFromMarketData:
    def test_produces_a_real_manifest_with_market_data_lineage(self, tmp_path) -> None:
        manifest, coverage_report = _build(tmp_path)
        assert manifest.market_data_lineage is not None
        assert manifest.dataset_id
        assert manifest.version
        assert sum(manifest.row_counts.values()) > 0
        assert all(f.status == "ok" for f in coverage_report.findings)

    def test_manifest_carries_market_data_lineage_content_id_in_input_hashes(self, tmp_path) -> None:
        manifest, _ = _build(tmp_path)
        assert "market_data_lineage_content_id" in manifest.input_content_hashes
        assert "historical_dataset_content_checksum" in manifest.input_content_hashes

    def test_rebuild_with_identical_bindings_is_idempotent(self, tmp_path) -> None:
        manifest_a, _ = _build(tmp_path)
        # A second call against the same (unmodified) fixtures should resolve
        # to the SAME dataset_id/version (ResearchManifestStore.save's own
        # content-duplicate no-op detection).
        repo_root = tmp_path
        import shutil

        second_root = repo_root.parent / (repo_root.name + "_copy")
        shutil.copytree(repo_root, second_root)
        manifest_b, _ = _build(second_root)
        assert manifest_a.dataset_id == manifest_b.dataset_id
        assert manifest_a.version == manifest_b.version
        assert manifest_a.content_id == manifest_b.content_id
