from __future__ import annotations

import pytest

from quant_platform.feature_discovery.catalog import (
    FeatureCatalog,
    FeatureInfrastructureBundle,
    FeatureInventory,
    FeatureManifest,
    build_feature_catalog,
    build_feature_infrastructure_bundle,
    build_feature_inventory,
    build_feature_manifest,
)
from quant_platform.feature_discovery.registry_snapshot import capture_feature_registry_snapshot


class TestBuildFeatureCatalog:
    def test_entries_sorted_by_name(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        names = [e.feature_name for e in catalog.entries]
        assert names == sorted(names)

    def test_entry_lookup(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        assert catalog.entry("trend").feature_name == "trend"
        with pytest.raises(KeyError):
            catalog.entry("does_not_exist")

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        assert FeatureCatalog.from_json_dict(catalog.to_json_dict()) == catalog


class TestBuildFeatureInventory:
    def test_five_views_present(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        inventory = build_feature_inventory(catalog)
        assert set(inventory.grouped_catalog["price"]) == {"trend", "const", "trend_copy"}
        assert set(inventory.origin_catalog) <= {"raw_source", "derived_feature", "higher_order_feature"}
        assert inventory.dataset_catalog[discovered_manifest.dataset_id] == tuple(sorted(discovered_manifest.feature_names))

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        catalog = build_feature_catalog(snapshot)
        inventory = build_feature_inventory(catalog)
        assert FeatureInventory.from_json_dict(inventory.to_json_dict()) == inventory


class TestBuildFeatureManifest:
    def test_deterministic_regardless_of_metadata_order(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        manifest_a = build_feature_manifest(snapshot)
        reversed_snapshot = snapshot.__class__(
            schema_version=snapshot.schema_version, dataset_id=snapshot.dataset_id, manifest_version=snapshot.manifest_version,
            captured_at=snapshot.captured_at, metadata=tuple(reversed(snapshot.metadata)), lineages=snapshot.lineages, provenances=snapshot.provenances,
        )
        manifest_b = build_feature_manifest(reversed_snapshot)
        assert manifest_a.manifest_id == manifest_b.manifest_id

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        feature_manifest = build_feature_manifest(snapshot)
        assert FeatureManifest.from_json_dict(feature_manifest.to_json_dict()) == feature_manifest


class TestBuildFeatureInfrastructureBundle:
    def test_bundle_assembles_every_component_consistently(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        assert bundle.snapshot.dataset_id == discovered_manifest.dataset_id
        assert bundle.catalog.dataset_id == discovered_manifest.dataset_id
        assert bundle.graph.dataset_id == discovered_manifest.dataset_id
        assert bundle.manifest.dataset_id == discovered_manifest.dataset_id
        assert {e.feature_name for e in bundle.catalog.entries} == set(discovered_manifest.feature_names)

    def test_json_round_trip(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        assert FeatureInfrastructureBundle.from_json_dict(bundle.to_json_dict()) == bundle
