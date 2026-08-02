from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.feature_discovery.metadata import (
    FeatureMetadata,
    FeatureProvenance,
    FeatureVersionHistory,
    compute_feature_metadata,
    compute_feature_provenance,
    compute_feature_version_history,
)
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec


class TestComputeFeatureMetadata:
    def test_base_feature_is_derived_feature_stage(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        metadata = compute_feature_metadata(spec, dataset_id=graph_manifest.dataset_id, manifest_version=graph_manifest.version)
        assert metadata.creation_stage == "derived_feature"
        assert metadata.feature_id == "trend@1"
        assert metadata.feature_group == "price"
        assert metadata.warmup_requirement == 3
        assert metadata.dependencies == ()

    def test_feature_with_feature_dependencies_is_higher_order_stage(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend_double", graph_manifest.feature_versions["trend_double"]).spec
        metadata = compute_feature_metadata(spec, dataset_id=graph_manifest.dataset_id, manifest_version=graph_manifest.version)
        assert metadata.creation_stage == "higher_order_feature"
        assert metadata.dependencies == ("trend",)

    def test_json_round_trip(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        metadata = compute_feature_metadata(spec, dataset_id=graph_manifest.dataset_id, manifest_version=graph_manifest.version)
        assert FeatureMetadata.from_json_dict(metadata.to_json_dict()) == metadata

    def test_deterministic_identity_matches_spec_fingerprint(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        metadata = compute_feature_metadata(spec, dataset_id=graph_manifest.dataset_id, manifest_version=graph_manifest.version)
        assert metadata.deterministic_identity == spec.fingerprint()

    def test_availability_rule_reflects_delay(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        delayed_spec = replace(spec, availability_delay=pd.Timedelta(seconds=30))
        metadata = compute_feature_metadata(delayed_spec, dataset_id=graph_manifest.dataset_id, manifest_version=graph_manifest.version)
        assert "30s" in metadata.availability_rule


class TestComputeFeatureProvenance:
    def test_captures_manifest_and_spec_fields(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        provenance = compute_feature_provenance(spec, graph_manifest)
        assert provenance.origin_dataset == graph_manifest.dataset_id
        assert provenance.source_historical_dataset_id == graph_manifest.source_historical_dataset_id
        assert provenance.has_market_data_lineage is False

    def test_json_round_trip(self, graph_registry, graph_manifest) -> None:
        spec = graph_registry.get("trend", graph_manifest.feature_versions["trend"]).spec
        provenance = compute_feature_provenance(spec, graph_manifest)
        assert FeatureProvenance.from_json_dict(provenance.to_json_dict()) == provenance


class TestComputeFeatureVersionHistory:
    def test_single_version_registry(self, graph_registry, graph_manifest) -> None:
        history = compute_feature_version_history(graph_registry, "trend", current_version=graph_manifest.feature_versions["trend"])
        assert history.current_version == "1"
        assert len(history.versions) == 1
        assert history.versions[0].version == "1"

    def test_multiple_versions_all_captured(self, graph_registry, graph_manifest) -> None:
        v2_spec = FeatureSpec(
            name="trend", version="2", description="row index v2", category=FeatureCategory.PRICE, required_inputs=("close",),
            source_symbols=("XAUUSD",), source_timeframe=Timeframe.M1, output_dtype="float64", lookback_bars=0, warmup_bars=5,
        )
        graph_registry.register(FeatureDefinition(spec=v2_spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64"))))
        history = compute_feature_version_history(graph_registry, "trend", current_version="1")
        assert [v.version for v in history.versions] == ["1", "2"]
        assert history.versions[0].spec_fingerprint != history.versions[1].spec_fingerprint

    def test_json_round_trip(self, graph_registry, graph_manifest) -> None:
        history = compute_feature_version_history(graph_registry, "trend", current_version=graph_manifest.feature_versions["trend"])
        assert FeatureVersionHistory.from_json_dict(history.to_json_dict()) == history
