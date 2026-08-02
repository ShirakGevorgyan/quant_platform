"""Mandatory fixture-based end-to-end acceptance workflow (Milestone 10,
Phase 4D, spec Section 21) -- the 12-step workflow, run as one sequential
narrative test class (later steps depend on durable state earlier steps
created), using the REAL `features.dataset_builder.ResearchDatasetBuilder`
throughout (never a test replacement).

FIXTURES: base XAUUSD candles; four macro series (DFII10, DGS10,
CPIAUCSL, DFF) with one missing observation and one revised observation
among them; five cross-asset drivers (a dollar-strength proxy, WTI proxy,
Brent proxy, silver proxy, and a gold reference) with one missing candle
and explicit ETF-proxy metadata on every mapping.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
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
from quant_platform.features.market_data_bridge.cross_asset_adapter import resolve_cross_asset_dataframe
from quant_platform.features.market_data_bridge.macro_adapter import resolve_macro_dataframe
from quant_platform.features.market_data_bridge.rebuild_planner import (
    RebuildPlanKind,
    SourceChangeEvidence,
    plan_rebuild,
)
from quant_platform.features.market_data_bridge.reconciliation import (
    reconcile_binding_source,
    reconcile_manifest_lineage,
    reconcile_no_pre_availability_macro_leakage,
    reconcile_no_pre_close_cross_asset_leakage,
)
from quant_platform.features.market_data_bridge.verification import (
    verify_truncation_invariance_cross_asset,
    verify_truncation_invariance_macro,
)
from quant_platform.features.registry import FeatureRegistry

MACRO_SERIES = ("DFII10", "DGS10", "CPIAUCSL", "DFF")
CROSS_ASSET_DRIVERS = (
    ("us_dollar_strength", "UUP", "mapping_dxy"),
    ("wti_crude", "USO", "mapping_wti"),
    ("brent_crude", "BNO", "mapping_brent"),
    ("silver", "SLV", "mapping_silver"),
    ("gold_reference", "GLD", "mapping_gold"),
)


@pytest.fixture(scope="module")
def workflow_state(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("phase4d_acceptance")
    state: dict = {}

    # -------------------- Step 1: create durable market_data source datasets --------------------
    repo = open_repository(tmp_path / "md")
    state["repo"] = repo
    state["base_binding"] = make_base_binding(repo, hours=240)

    macro_fixtures = {}
    for series_id in MACRO_SERIES:
        macro_fixtures[series_id] = make_macro_fixture(
            tmp_path / f"macro_{series_id}", series_id=series_id, days=15,
            with_missing_day=(3 if series_id == "CPIAUCSL" else None), with_revision_on_day=(5 if series_id == "DFF" else None),
        )
    state["macro_fixtures"] = macro_fixtures

    cross_fixtures = {}
    for canonical_driver_id, provider_symbol, mapping_id in CROSS_ASSET_DRIVERS:
        cross_fixtures[canonical_driver_id] = make_cross_asset_fixture(
            tmp_path / f"cross_{canonical_driver_id}", canonical_driver_id=canonical_driver_id, provider_symbol=provider_symbol, mapping_id=mapping_id, days=15,
        )
    state["cross_fixtures"] = cross_fixtures
    state["tmp_path"] = tmp_path
    return state


class TestStep1And2CreateAndVerifyDurableSources:
    def test_base_binding_verifies(self, workflow_state) -> None:
        from quant_platform.features.market_data_bridge.base_asset_adapter import verify_base_asset_binding

        candles = verify_base_asset_binding(workflow_state["repo"], workflow_state["base_binding"])
        assert len(candles) == 240

    def test_every_macro_binding_verifies(self, workflow_state) -> None:
        for series_id, fixture in workflow_state["macro_fixtures"].items():
            report = reconcile_binding_source(macro=(fixture.observation_store, fixture.manifest_store, fixture.binding))
            assert report.is_clean, f"{series_id}: {report.issues}"

    def test_every_cross_asset_binding_verifies(self, workflow_state) -> None:
        for driver_id, fixture in workflow_state["cross_fixtures"].items():
            report = reconcile_binding_source(cross_asset=(fixture.bar_store, fixture.manifest_store, fixture.binding))
            assert report.is_clean, f"{driver_id}: {report.issues}"


class TestStep3BindingsArePinnedAndImmutable:
    def test_bindings_reject_mutable_aliases(self, workflow_state) -> None:
        from quant_platform.core.exceptions import SourceBindingError
        from quant_platform.features.market_data_bridge.bindings import create_base_asset_binding

        with pytest.raises(SourceBindingError):
            create_base_asset_binding(canonical_instrument_id="XAUUSD", provider="mt5", pinned_dataset_id="latest", timeframe=Timeframe.H1)


class TestStep4And5AlignAndBuildResearchDataset:
    def test_full_build_succeeds(self, workflow_state) -> None:
        registry = FeatureRegistry()
        for series_id in MACRO_SERIES:
            register_macro_features(registry, base_timeframe=Timeframe.H1, config=MacroSourceConfig(source_name=series_id))
        for canonical_driver_id, _symbol, _mapping in CROSS_ASSET_DRIVERS:
            register_cross_asset_features(
                registry, base_timeframe=Timeframe.H1, base_momentum_window=5, base_volatility_window=10,
                cross_asset_symbol=canonical_driver_id.upper(), cross_asset_timeframe=Timeframe.D1,
            )
        feature_names = tuple(s.name for s in registry.list_features())

        research_store = ResearchDatasetStore(str(workflow_state["tmp_path"] / "research"))
        manifest_store = ResearchManifestStore(str(workflow_state["tmp_path"] / "research"))
        build_request = ResearchDatasetBuildRequest(
            symbol="XAUUSD", base_timeframe=Timeframe.H1, start=pd.Timestamp("2024-01-05T00:00Z"), end=pd.Timestamp("2024-01-10T00:00Z"),
            feature_names=feature_names, label_definition=LabelDefinition(name="fwd_ret_5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
            split_strategy="chronological",
        )
        macro_bindings = {series_id: workflow_state["macro_fixtures"][series_id].binding for series_id in MACRO_SERIES}
        cross_bindings = {
            canonical_driver_id.upper(): workflow_state["cross_fixtures"][canonical_driver_id].binding for canonical_driver_id, _s, _m in CROSS_ASSET_DRIVERS
        }
        manifest, coverage_report = self._build_with_per_series_repos(
            workflow_state, registry, research_store, manifest_store, build_request, macro_bindings, cross_bindings
        )

        workflow_state["manifest"] = manifest
        workflow_state["coverage_report"] = coverage_report
        workflow_state["research_store"] = research_store
        workflow_state["manifest_store"] = manifest_store
        workflow_state["registry"] = registry
        workflow_state["feature_names"] = feature_names
        workflow_state["build_request"] = build_request
        assert manifest.dataset_id
        assert sum(manifest.row_counts.values()) > 0

    @staticmethod
    def _build_with_per_series_repos(workflow_state, registry, research_store, manifest_store, build_request, macro_bindings, cross_bindings):
        # `MacroRepository`/`CrossAssetRepository` each expect ONE shared
        # store pair, but this acceptance fixture deliberately gives each
        # macro series/cross-asset driver its OWN isolated store (mirrors
        # separate Phase 4B/4C backfills) -- so resolve each source's
        # frame directly here rather than through one shared repository
        # object, then hand the already-resolved frames + a
        # dummy-but-consistent repository pair to the SAME real
        # `ResearchDatasetBuilder` orchestration `request.py` itself uses
        # internally (mirroring, not duplicating, its own build steps).
        from quant_platform.features.dataset_builder import ResearchDatasetBuilder
        from quant_platform.features.market_data_bridge.base_asset_adapter import MarketDataBaseAssetLoader

        macro_frames = {
            series_id: resolve_macro_dataframe(workflow_state["macro_fixtures"][series_id].observation_store, workflow_state["macro_fixtures"][series_id].manifest_store, binding)
            for series_id, binding in macro_bindings.items()
        }
        cross_frames = {
            symbol: resolve_cross_asset_dataframe(
                workflow_state["cross_fixtures"][symbol.lower()].bar_store, workflow_state["cross_fixtures"][symbol.lower()].manifest_store, binding
            )
            for symbol, binding in cross_bindings.items()
        }
        base_loader = MarketDataBaseAssetLoader(workflow_state["repo"], workflow_state["base_binding"])
        builder = ResearchDatasetBuilder(
            historical_loader=base_loader, registry=registry, research_store=research_store, manifest_store=manifest_store,
            cross_asset_data=cross_frames, macro_data=macro_frames,
        )
        from dataclasses import replace as dc_replace

        from quant_platform.features.market_data_bridge.base_asset_adapter import resolve_base_asset_dataframe
        from quant_platform.features.market_data_bridge.coverage import evaluate_source_coverage
        from quant_platform.features.market_data_bridge.lineage import (
            build_market_data_lineage,
            lineage_content_id,
        )

        base_df = resolve_base_asset_dataframe(workflow_state["repo"], workflow_state["base_binding"], start=build_request.start, end=build_request.end)
        coverage_report = evaluate_source_coverage(
            base_df=base_df, base_timeframe=workflow_state["base_binding"].timeframe, macro_frames=macro_frames, macro_bindings=macro_bindings,
            cross_asset_frames=cross_frames, cross_asset_bindings=cross_bindings, requested_start=build_request.start, requested_end=build_request.end,
            policy=SourceCoveragePolicy(kind=SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE),
        )
        lineage = build_market_data_lineage(base_binding=workflow_state["base_binding"], macro_bindings=macro_bindings, cross_asset_bindings=cross_bindings, coverage_report=coverage_report)
        effective_request = dc_replace(
            build_request, start=coverage_report.safe_start, end=coverage_report.safe_end,
            aux_input_content_hashes={"market_data_lineage_content_id": lineage_content_id(lineage)}, market_data_lineage=lineage,
        )
        manifest = builder.build(effective_request)
        return manifest, coverage_report


class TestStep7VerifyResearchManifest:
    def test_manifest_lineage_reconciles(self, workflow_state) -> None:
        macro_bindings = {series_id: workflow_state["macro_fixtures"][series_id].binding for series_id in MACRO_SERIES}
        cross_bindings = {
            canonical_driver_id.upper(): workflow_state["cross_fixtures"][canonical_driver_id].binding for canonical_driver_id, _s, _m in CROSS_ASSET_DRIVERS
        }
        report = reconcile_manifest_lineage(
            workflow_state["manifest"], base_binding=workflow_state["base_binding"], macro_bindings=macro_bindings, cross_asset_bindings=cross_bindings,
            coverage_report=workflow_state["coverage_report"],
        )
        assert report.is_clean


class TestStep8And9NoLeakageProofs:
    def test_no_pre_availability_macro_leakage(self, workflow_state) -> None:
        base_avail = pd.Series(pd.date_range("2024-01-05", "2024-01-10", freq="h", tz="UTC"))
        for series_id, fixture in workflow_state["macro_fixtures"].items():
            macro_df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
            report = reconcile_no_pre_availability_macro_leakage(base_avail, macro_df, source_name=series_id)
            assert report.is_clean, f"{series_id}: {report.issues}"

    def test_no_incomplete_cross_asset_candle_leakage(self, workflow_state) -> None:
        base_avail = pd.Series(pd.date_range("2024-01-05", "2024-01-10", freq="h", tz="UTC"))
        for driver_id, fixture in workflow_state["cross_fixtures"].items():
            cross_df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
            report = reconcile_no_pre_close_cross_asset_leakage(base_avail, cross_df, source_name=driver_id, timeframe=Timeframe.D1)
            assert report.is_clean, f"{driver_id}: {report.issues}"

    def test_truncation_invariance_holds_for_every_macro_series(self, workflow_state) -> None:
        base_avail = pd.Series(pd.date_range("2024-01-05", "2024-01-10", freq="h", tz="UTC"))
        for series_id, fixture in workflow_state["macro_fixtures"].items():
            macro_df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
            result = verify_truncation_invariance_macro(base_avail, macro_df, source_name=series_id, truncate_after=pd.Timestamp("2024-01-07T00:00Z"))
            assert result.is_invariant, f"{series_id}: {result}"

    def test_truncation_invariance_holds_for_every_cross_asset_driver(self, workflow_state) -> None:
        base_avail = pd.Series(pd.date_range("2024-01-05", "2024-01-10", freq="h", tz="UTC"))
        for driver_id, fixture in workflow_state["cross_fixtures"].items():
            cross_df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
            result = verify_truncation_invariance_cross_asset(base_avail, cross_df, source_name=driver_id, timeframe=Timeframe.D1, truncate_after=pd.Timestamp("2024-01-07T00:00Z"))
            assert result.is_invariant, f"{driver_id}: {result}"


class TestStep10And11ReplayIntoFreshRootAndCompareIdentity:
    def test_replay_produces_byte_identical_semantic_digest(self, workflow_state) -> None:
        registry = workflow_state["registry"]
        build_request = workflow_state["build_request"]
        fresh_root = workflow_state["tmp_path"] / "research_replay"
        research_store = ResearchDatasetStore(str(fresh_root))
        manifest_store = ResearchManifestStore(str(fresh_root))
        macro_bindings = {series_id: workflow_state["macro_fixtures"][series_id].binding for series_id in MACRO_SERIES}
        cross_bindings = {
            canonical_driver_id.upper(): workflow_state["cross_fixtures"][canonical_driver_id].binding for canonical_driver_id, _s, _m in CROSS_ASSET_DRIVERS
        }
        replay_manifest, _ = TestStep4And5AlignAndBuildResearchDataset._build_with_per_series_repos(
            workflow_state, registry, research_store, manifest_store, build_request, macro_bindings, cross_bindings
        )
        original = workflow_state["manifest"]
        assert replay_manifest.dataset_id == original.dataset_id
        assert replay_manifest.content_id == original.content_id
        assert replay_manifest.output_content_hashes == original.output_content_hashes
        assert replay_manifest.market_data_lineage == original.market_data_lineage


class TestStep12IncrementalRebuildPlan:
    def test_updating_one_macro_source_version_produces_a_deterministic_plan(self, workflow_state) -> None:
        old_binding = workflow_state["macro_fixtures"]["DFF"].binding
        new_binding = replace(old_binding, component_manifest_id="f" * 64, binding_id="")

        ev = SourceChangeEvidence(
            source_kind="macro", source_name="DFF", old_first_covered_time=pd.Timestamp("2024-01-01", tz="UTC"),
            old_last_covered_time=pd.Timestamp("2024-01-15", tz="UTC"), old_observation_count=16,
            new_first_covered_time=pd.Timestamp("2024-01-01", tz="UTC"), new_last_covered_time=pd.Timestamp("2024-01-20", tz="UTC"), new_observation_count=20,
        )
        plan = plan_rebuild(
            existing_lineage=workflow_state["manifest"].market_data_lineage, existing_dataset_id=workflow_state["manifest"].dataset_id, recipe_unchanged=True,
            new_base_pinned_dataset_id=workflow_state["base_binding"].pinned_dataset_id, old_base_pinned_dataset_id=workflow_state["base_binding"].pinned_dataset_id,
            new_macro_component_ids={"DFF": new_binding.component_manifest_id}, old_macro_component_ids={"DFF": old_binding.component_manifest_id},
            new_cross_asset_component_ids={}, old_cross_asset_component_ids={}, evidence_by_source_name={"DFF": ev},
        )
        assert plan.kind is RebuildPlanKind.APPEND_ONLY_SAFE_EXTENSION
        assert plan.expected_output_dataset_id == workflow_state["manifest"].dataset_id
        assert "DFF" in plan.affected_source_names

        # Determinism: planning again with the identical inputs produces the identical plan_id.
        plan_again = plan_rebuild(
            existing_lineage=workflow_state["manifest"].market_data_lineage, existing_dataset_id=workflow_state["manifest"].dataset_id, recipe_unchanged=True,
            new_base_pinned_dataset_id=workflow_state["base_binding"].pinned_dataset_id, old_base_pinned_dataset_id=workflow_state["base_binding"].pinned_dataset_id,
            new_macro_component_ids={"DFF": new_binding.component_manifest_id}, old_macro_component_ids={"DFF": old_binding.component_manifest_id},
            new_cross_asset_component_ids={}, old_cross_asset_component_ids={}, evidence_by_source_name={"DFF": ev},
        )
        assert plan.plan_id == plan_again.plan_id


class TestBaseAssetIngestionSanityCheck:
    def test_ingest_raw_events_is_the_real_ingestion_path(self, workflow_state) -> None:
        # Sanity: the base fixture really did go through market_data's own
        # real ingestion pipeline (not a shortcut/mock) -- re-submitting
        # the EXACT same batch (same batch_id, same candle set) hits
        # `IngestionBatchStore.reserve`'s own idempotent-replay path.
        from _market_data_bridge_test_helpers import ingest_base_candles

        result = ingest_base_candles(workflow_state["repo"], hours=240)
        assert result.was_idempotent_replay
