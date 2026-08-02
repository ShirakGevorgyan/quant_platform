"""Milestone 11, Phase 2, Part 2: the 13 named adversarial test
scenarios, each run against the real infrastructure pipeline -- never
a mock. Some scenarios are also exercised (from a different angle) in
`test_feature_discovery_graph.py`/`test_feature_discovery_infra_
verification.py`/`test_feature_discovery_infra_reconciliation.py`; this
file exists for direct, unambiguous traceability against the spec's
own 13-item list, one test class per item."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import (
    DuplicateFeatureError,
    FeatureDiscoveryReconciliationError,
    SchemaVersionError,
)
from quant_platform.feature_discovery.catalog import (
    FeatureCatalog,
    FeatureInfrastructureBundle,
    FeatureManifest,
    build_feature_catalog,
    build_feature_infrastructure_bundle,
    build_feature_inventory,
    build_feature_manifest,
)
from quant_platform.feature_discovery.graph import build_feature_dependency_graph
from quant_platform.feature_discovery.infra_reconciliation import FeatureInfrastructureReconciliation
from quant_platform.feature_discovery.infra_verification import (
    FeatureInfrastructureVerifier,
    verify_bundle_self_consistency,
)
from quant_platform.feature_discovery.metadata import FeatureMetadata
from quant_platform.feature_discovery.registry_snapshot import capture_feature_registry_snapshot
from quant_platform.features.interfaces import FeatureDefinition


def _rebuild_bundle(bundle: FeatureInfrastructureBundle, snapshot) -> FeatureInfrastructureBundle:
    catalog = build_feature_catalog(snapshot)
    return FeatureInfrastructureBundle(
        snapshot=snapshot, graph=bundle.graph, catalog=catalog, inventory=build_feature_inventory(catalog), manifest=build_feature_manifest(snapshot),
    )


class Test01FeatureIdTampering:
    def test_tampered_feature_id_is_caught_by_self_consistency(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(replace(m, feature_id="trend@999") if m.feature_name == "trend" else m for m in bundle.snapshot.metadata)
        tampered = _rebuild_bundle(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        consistent, issues = verify_bundle_self_consistency(tampered, registry=discovered_registry)
        assert consistent is False
        assert any("identity" in i or "manifest_id" in i or "feature_ids" in i for i in issues)


class Test02MetadataTampering:
    def test_metadata_tampering_is_detected_by_reconciliation(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(replace(m, feature_group="macro") if m.feature_name == "trend" else m for m in bundle.snapshot.metadata)
        tampered = _rebuild_bundle(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered)
        assert any(i.kind == "metadata_drift" and i.feature_name == "trend" for i in result.issues)


class Test03DependencyCorruption:
    def test_corrupted_dependencies_produce_a_missing_parent(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        corrupted = tuple(replace(m, dependencies=("nonexistent_upstream",)) if m.feature_name == "trend_double" else m for m in snapshot.metadata)
        graph = build_feature_dependency_graph(replace(snapshot, metadata=corrupted))
        assert ("trend_double", "nonexistent_upstream") in graph.missing_parents
        assert graph.is_valid is False


class Test04LineageCorruption:
    def test_corrupted_lineage_is_detected_by_reconciliation(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        corrupted_lineages = tuple(
            replace(ln, required_inputs=("corrupted_column",)) if ln.feature_name == "trend" else ln for ln in bundle.snapshot.lineages
        )
        corrupted = replace(bundle, snapshot=replace(bundle.snapshot, lineages=corrupted_lineages))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, corrupted)
        assert any(i.kind == "lineage_drift" and i.feature_name == "trend" for i in result.issues)


class Test05ManifestCorruption:
    def test_corrupted_manifest_id_fails_self_consistency(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        corrupted = replace(bundle, manifest=replace(bundle.manifest, manifest_id="0" * 16))
        consistent, issues = verify_bundle_self_consistency(corrupted)
        assert consistent is False
        assert any("manifest_id" in i for i in issues)


class Test06CycleInjection:
    def test_injected_cycle_makes_the_graph_invalid(self, graph_registry, graph_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(graph_registry, graph_manifest)
        cyclic = tuple(replace(m, dependencies=("trend_double",)) if m.feature_name == "trend" else m for m in snapshot.metadata)
        graph = build_feature_dependency_graph(replace(snapshot, metadata=cyclic))
        assert graph.cycles != ()
        assert graph.is_valid is False


class Test07OrphanFeature:
    def test_disconnected_extra_feature_is_flagged_orphan(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        ghost = FeatureMetadata(
            schema_version=1, feature_id="ghost@1", feature_name="ghost", feature_group="price", origin_dataset=discovered_manifest.dataset_id,
            origin_manifest=discovered_manifest.version, creation_stage="raw_source", availability_rule="n/a", warmup_requirement=0,
            dependencies=(), deterministic_identity="deadbeef",
        )
        graph = build_feature_dependency_graph(
            replace(snapshot, metadata=(*snapshot.metadata, ghost)), declared_feature_names=frozenset(discovered_manifest.feature_names),
        )
        assert "ghost" in graph.orphan_features


class Test08DuplicateFeature:
    def test_registering_the_same_name_and_version_twice_is_refused_by_the_real_registry(self, discovered_registry) -> None:
        existing = discovered_registry.get("trend").spec
        with pytest.raises(DuplicateFeatureError):
            discovered_registry.register(FeatureDefinition(spec=existing, compute=lambda ctx: ctx.base_df["close"]))


class Test09MissingParent:
    def test_dependency_referencing_an_unregistered_feature_is_flagged(self, discovered_registry, discovered_manifest) -> None:
        snapshot = capture_feature_registry_snapshot(discovered_registry, discovered_manifest)
        corrupted = tuple(replace(m, dependencies=("phantom_feature",)) if m.feature_name == "trend" else m for m in snapshot.metadata)
        graph = build_feature_dependency_graph(replace(snapshot, metadata=corrupted))
        assert ("trend", "phantom_feature") in graph.missing_parents


class Test10SchemaMismatch:
    def test_wrong_schema_version_is_rejected(self) -> None:
        with pytest.raises(SchemaVersionError):
            FeatureCatalog.from_json_dict({"schema_version": 999, "dataset_id": "abc", "entries": []})

    def test_wrong_schema_version_rejected_for_manifest(self) -> None:
        with pytest.raises(SchemaVersionError):
            FeatureManifest.from_json_dict({"schema_version": 999, "manifest_id": "x", "dataset_id": "abc", "origin_manifest": "v1", "generated_at": "t", "feature_count": 0, "feature_ids": []})


class Test11AvailabilityMismatch:
    def test_tampered_availability_rule_is_detected_by_reconciliation(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(
            replace(m, availability_rule="available at open_time + 999s delay (tampered)") if m.feature_name == "trend" else m
            for m in bundle.snapshot.metadata
        )
        tampered = _rebuild_bundle(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered)
        assert any(i.kind == "metadata_drift" and i.feature_name == "trend" for i in result.issues)


class Test12WarmupCorruption:
    def test_tampered_warmup_requirement_is_detected_by_reconciliation(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(replace(m, warmup_requirement=-1) if m.feature_name == "trend" else m for m in bundle.snapshot.metadata)
        tampered = _rebuild_bundle(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        result = FeatureInfrastructureReconciliation().reconcile(bundle, tampered)
        assert any(i.kind == "metadata_drift" and i.feature_name == "trend" for i in result.issues)


class Test13IdentityCorruption:
    def test_tampered_deterministic_identity_fails_full_verification(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        tampered_metadata = tuple(
            replace(m, deterministic_identity="0" * 64) if m.feature_name == "trend" else m for m in bundle.snapshot.metadata
        )
        tampered = _rebuild_bundle(bundle, replace(bundle.snapshot, metadata=tampered_metadata))
        result = FeatureInfrastructureVerifier().verify(tampered, discovered_registry, discovered_manifest)
        assert result.verified is False
        assert result.self_consistent is False


class TestCrossDatasetReconciliationStillRaises:
    """Not one of the 13 named items, but a necessary structural guard
    the adversarial audit should not silently regress: reconciling two
    genuinely unrelated bundles must never be silently accepted."""

    def test_different_dataset_ids_raise(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        other = replace(bundle, snapshot=replace(bundle.snapshot, dataset_id="f" * 16))
        with pytest.raises(FeatureDiscoveryReconciliationError):
            FeatureInfrastructureReconciliation().reconcile(bundle, other)
