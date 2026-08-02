from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.feature_discovery.graph import (
    FeatureDependencyGraph,
    FeatureGraphNodeKind,
    build_feature_dependency_graph,
)
from quant_platform.feature_discovery.metadata import FeatureMetadata
from quant_platform.feature_discovery.registry_snapshot import capture_feature_registry_snapshot
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.features.models import FeatureCategory, FeatureSpec


class TestBuildFeatureDependencyGraph:
    def test_clean_graph_has_correct_nodes_and_edges(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        graph = build_feature_dependency_graph(snapshot)
        assert graph.is_valid is True
        assert graph.cycles == ()
        assert graph.missing_parents == ()
        assert any(n.node_id == "input:close" and n.kind is FeatureGraphNodeKind.RAW_SOURCE for n in graph.nodes)
        assert any(e.source == "input:close" and e.target == "trend@1" for e in graph.edges)
        assert any(e.source == "trend@1" and e.target == "trend_double@1" for e in graph.edges)

    def test_json_round_trip(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        graph = build_feature_dependency_graph(snapshot)
        assert FeatureDependencyGraph.from_json_dict(graph.to_json_dict()) == graph


class TestCycleDetection:
    def test_injected_cycle_is_detected(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        tampered_metadata = tuple(
            replace(m, dependencies=("trend_double",)) if m.feature_name == "trend" else m for m in snapshot.metadata
        )
        tampered_snapshot = replace(snapshot, metadata=tampered_metadata)
        graph = build_feature_dependency_graph(tampered_snapshot)
        assert graph.cycles != ()
        assert graph.is_valid is False


class TestMissingParentDetection:
    def test_dependency_on_a_nonexistent_feature_is_detected(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        tampered_metadata = tuple(
            replace(m, dependencies=("trend", "does_not_exist")) if m.feature_name == "trend_double" else m for m in snapshot.metadata
        )
        tampered_snapshot = replace(snapshot, metadata=tampered_metadata)
        graph = build_feature_dependency_graph(tampered_snapshot)
        assert ("trend_double", "does_not_exist") in graph.missing_parents
        assert graph.is_valid is False


class TestOrphanDetection:
    def test_disconnected_injected_feature_is_flagged_as_orphan(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        orphan_entry = FeatureMetadata(
            schema_version=1, feature_id="ghost@1", feature_name="ghost", feature_group="price", origin_dataset=graph_manifest.dataset_id,
            origin_manifest=graph_manifest.version, creation_stage="raw_source", availability_rule="n/a", warmup_requirement=0,
            dependencies=(), deterministic_identity="deadbeef",
        )
        tampered_snapshot = replace(snapshot, metadata=(*snapshot.metadata, orphan_entry))
        graph = build_feature_dependency_graph(tampered_snapshot, declared_feature_names=frozenset(graph_manifest.feature_names))
        assert graph.orphan_features == ("ghost",)

    def test_default_declared_names_never_produces_a_false_positive_orphan(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        graph = build_feature_dependency_graph(snapshot)  # declared_feature_names defaults to snapshot's own metadata
        assert graph.orphan_features == ()


class TestDuplicateDerivationDetection:
    def test_two_features_with_the_same_recipe_signature_are_flagged(self, tmp_path, seeded_loader, research_store, graph_registry) -> None:
        dup_spec = FeatureSpec(
            name="trend2", version="1", description="different name, same recipe", category=FeatureCategory.PRICE,
            required_inputs=("close",), source_symbols=("XAUUSD",), source_timeframe=Timeframe.M1, output_dtype="float64",
            lookback_bars=0, warmup_bars=3,
        )
        graph_registry.register(FeatureDefinition(spec=dup_spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64"))))

        builder = ResearchDatasetBuilder(
            historical_loader=seeded_loader, registry=graph_registry, research_store=research_store,
            manifest_store=ResearchManifestStore(tmp_path / "research"),
        )
        request = ResearchDatasetBuildRequest(
            symbol="XAUUSD", base_timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01", tz="UTC") + Timeframe.M1.duration * 2000, feature_names=("trend", "trend_double", "trend2"),
            label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5), split_strategy="chronological",
            split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        )
        manifest = builder.build(request)
        snapshot = capture_feature_registry_snapshot(graph_registry, manifest)
        graph = build_feature_dependency_graph(snapshot)
        assert ("trend", "trend2") in graph.duplicate_derivations
