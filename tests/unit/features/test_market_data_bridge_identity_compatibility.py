"""Golden dataset-identity compatibility (Milestone 10, Phase 4D, spec
Section 17): proves that changing a material market-data binding and
REBUILDING AGAINST THE SAME MANIFEST HISTORY changes the resulting
research dataset's VERSION, while `compute_dataset_id`'s own "recipe id"
(symbol/timeframe/feature registry fingerprint/label/split/
preprocessing) -- deliberately untouched by this phase, see
`features.manifests`'s own module docstring -- stays stable across a
pure market-data-source-content change.

THE EXACT MECHANISM (worth stating precisely, since it is easy to get
subtly wrong): `ResearchDatasetManifest.version` is
`f"{sequence:06d}-{content_id_prefix}"` (`ResearchManifestStore.save`).
`content_id` is a checksum of the WRITTEN feature/label/split Parquet
bytes only -- it does NOT depend on `market_data_lineage` directly, so
two builds whose underlying `market_data` bindings differ but whose
resulting FEATURE OUTPUT happens to be byte-identical (e.g. a macro
revision released outside the requested date range) can share the same
`content_id_prefix`. What DOES always change is `sequence`: `save()`
compares the new manifest's `_identity_fields()` (which includes
`market_data_lineage`, hence any binding change) against the latest
existing version's; any difference -- including a lineage-only
difference with identical output bytes -- forces `next_seq` to advance,
so the full version string still changes, AS LONG AS the rebuild targets
the SAME manifest history (the same `ResearchManifestStore` root, same
`dataset_id`) rather than an isolated one. This is exactly the realistic
"rebuild after a source revision" scenario incremental rebuild planning
targets, and is what these tests exercise -- two ISOLATED stores would
each independently produce `sequence=1` and could coincidentally share
an identical version string despite genuinely different lineage, which
is why every test below deliberately reuses one store across both
builds."""

from __future__ import annotations

from dataclasses import replace

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
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    ProxyQuality,
    create_proxy_policy,
)


def _registry() -> tuple[FeatureRegistry, tuple[str, ...]]:
    registry = FeatureRegistry()
    register_macro_features(registry, base_timeframe=Timeframe.H1, config=MacroSourceConfig(source_name="DFII10"))
    register_cross_asset_features(
        registry, base_timeframe=Timeframe.H1, base_momentum_window=5, base_volatility_window=10, cross_asset_symbol="DXY", cross_asset_timeframe=Timeframe.D1
    )
    return registry, tuple(s.name for s in registry.list_features())


def _run(*, market_data_repository, base_binding, macro_binding, cross_binding, macro_store, cross_store, research_store, manifest_store):
    registry, feature_names = _registry()
    build_request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.H1, start=pd.Timestamp("2024-01-03T00:00Z"), end=pd.Timestamp("2024-01-08T00:00Z"),
        feature_names=feature_names, label_definition=LabelDefinition(name="fwd_ret_5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological",
    )
    req = MarketDataResearchDatasetRequest(
        base_binding=base_binding, macro_bindings={"DFII10": macro_binding}, cross_asset_bindings={"DXY": cross_binding},
        coverage_policy=SourceCoveragePolicy(kind=SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE), build_request=build_request,
    )
    manifest, _ = build_research_dataset_from_market_data(
        market_data_repository=market_data_repository, macro_repository=macro_store, cross_asset_repository=cross_store,
        registry=registry, research_store=research_store, manifest_store=manifest_store, request=req,
    )
    return manifest


class TestMarketDataBindingChangesPropagateToVersionNotDatasetId:
    def test_different_macro_binding_changes_version_but_not_dataset_id(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=200)
        cross = make_cross_asset_fixture(tmp_path / "cross", days=10)
        cross_repo = CrossAssetRepository(cross.bar_store, cross.manifest_store)
        research_store = ResearchDatasetStore(str(tmp_path / "research"))
        manifest_store = ResearchManifestStore(str(tmp_path / "research"))

        macro_a = make_macro_fixture(tmp_path / "macro_a", days=10)
        macro_b = make_macro_fixture(tmp_path / "macro_b", days=10, with_revision_on_day=3)

        manifest_a = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro_a.binding, cross_binding=cross.binding,
            macro_store=MacroRepository(macro_a.observation_store, macro_a.manifest_store), cross_store=cross_repo,
            research_store=research_store, manifest_store=manifest_store,
        )
        manifest_b = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro_b.binding, cross_binding=cross.binding,
            macro_store=MacroRepository(macro_b.observation_store, macro_b.manifest_store), cross_store=cross_repo,
            research_store=research_store, manifest_store=manifest_store,
        )
        assert manifest_a.dataset_id == manifest_b.dataset_id
        assert manifest_a.version != manifest_b.version
        assert manifest_b.version.startswith("000002-")
        assert manifest_a.market_data_lineage != manifest_b.market_data_lineage
        assert manifest_a.input_content_hashes["market_data_lineage_content_id"] != manifest_b.input_content_hashes["market_data_lineage_content_id"]

    def test_different_cross_asset_proxy_quality_changes_version(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=200)
        macro = make_macro_fixture(tmp_path / "macro", days=10)
        macro_repo = MacroRepository(macro.observation_store, macro.manifest_store)
        research_store = ResearchDatasetStore(str(tmp_path / "research"))
        manifest_store = ResearchManifestStore(str(tmp_path / "research"))

        cross = make_cross_asset_fixture(tmp_path / "cross", days=10)
        cross_repo = CrossAssetRepository(cross.bar_store, cross.manifest_store)
        different_proxy_binding = replace(
            cross.binding, proxy_policy=create_proxy_policy(is_proxy=True, proxy_for="us_dollar_strength", proxy_quality=ProxyQuality.LOW), binding_id="",
        )

        manifest_a = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro.binding, cross_binding=cross.binding,
            macro_store=macro_repo, cross_store=cross_repo, research_store=research_store, manifest_store=manifest_store,
        )
        manifest_b = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro.binding, cross_binding=different_proxy_binding,
            macro_store=macro_repo, cross_store=cross_repo, research_store=research_store, manifest_store=manifest_store,
        )
        assert manifest_a.dataset_id == manifest_b.dataset_id
        assert manifest_a.version != manifest_b.version
        assert manifest_b.version.startswith("000002-")

    def test_rebuild_with_identical_bindings_is_a_true_no_op(self, tmp_path) -> None:
        """The control case: rebuilding with the EXACT same bindings
        against the same manifest history must NOT mint a new version --
        proves the mechanism above is sensitive to genuine changes only,
        never merely to being called twice."""
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=200)
        macro = make_macro_fixture(tmp_path / "macro", days=10)
        cross = make_cross_asset_fixture(tmp_path / "cross", days=10)
        research_store = ResearchDatasetStore(str(tmp_path / "research"))
        manifest_store = ResearchManifestStore(str(tmp_path / "research"))
        macro_repo = MacroRepository(macro.observation_store, macro.manifest_store)
        cross_repo = CrossAssetRepository(cross.bar_store, cross.manifest_store)

        manifest_a = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro.binding, cross_binding=cross.binding,
            macro_store=macro_repo, cross_store=cross_repo, research_store=research_store, manifest_store=manifest_store,
        )
        manifest_b = _run(
            market_data_repository=repo, base_binding=base_binding, macro_binding=macro.binding, cross_binding=cross.binding,
            macro_store=macro_repo, cross_store=cross_repo, research_store=research_store, manifest_store=manifest_store,
        )
        assert manifest_a.version == manifest_b.version
        assert manifest_a.version.startswith("000001-")

    def test_compute_dataset_id_hash_inputs_are_untouched_by_bridge(self) -> None:
        """`compute_dataset_id` (features/manifests.py) is deliberately
        NOT modified by Phase 4D -- confirms its signature still accepts
        only the pre-existing recipe fields, no market_data-specific
        parameter was added to it."""
        import inspect

        from quant_platform.features.manifests import compute_dataset_id

        params = set(inspect.signature(compute_dataset_id).parameters)
        assert params == {"symbol", "base_timeframe", "feature_registry_fingerprint", "label_definition", "split_definition", "preprocessing_definition"}
