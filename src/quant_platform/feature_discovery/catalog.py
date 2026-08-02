"""`FeatureCatalog`, `FeatureInventory`, `FeatureManifest` (Milestone
11, Phase 2, Part 2) -- the "what features exist" views, all built
FROM an already-captured `FeatureRegistrySnapshot`, never by re-reading
a registry. `FeatureCatalog` is the flat, complete listing;
`FeatureInventory` bundles the 5 named groupings (complete/grouped/
dataset/origin/availability) over that same listing; `FeatureManifest`
is this package's own deterministic, content-addressed bundle
identity for a snapshot's exact feature set -- distinct from
`features.manifests.ResearchDatasetManifest` (the dataset's own
manifest), never a replacement for it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from quant_platform.feature_discovery.graph import FeatureDependencyGraph, build_feature_dependency_graph
from quant_platform.feature_discovery.metadata import FeatureMetadata
from quant_platform.feature_discovery.registry_snapshot import (
    FeatureRegistrySnapshot,
    capture_feature_registry_snapshot,
)
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.registry import FeatureRegistry
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "FEATURE_CATALOG_SCHEMA_VERSION",
    "FEATURE_INVENTORY_SCHEMA_VERSION",
    "FEATURE_MANIFEST_SCHEMA_VERSION",
    "FeatureCatalog",
    "FeatureInfrastructureBundle",
    "FeatureInventory",
    "FeatureManifest",
    "build_feature_catalog",
    "build_feature_infrastructure_bundle",
    "build_feature_inventory",
    "build_feature_manifest",
]

FEATURE_CATALOG_SCHEMA_VERSION = 1
FEATURE_INVENTORY_SCHEMA_VERSION = 1
FEATURE_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    schema_version: int
    dataset_id: str
    entries: tuple[FeatureMetadata, ...]
    """Sorted by `feature_name` -- the complete, flat catalog."""

    def entry(self, feature_name: str) -> FeatureMetadata:
        for e in self.entries:
            if e.feature_name == feature_name:
                return e
        raise KeyError(feature_name)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "entries": [e.to_json_dict() for e in self.entries],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureCatalog:
        require_schema_version(raw, supported=FEATURE_CATALOG_SCHEMA_VERSION, context="FeatureCatalog")
        return cls(
            schema_version=FEATURE_CATALOG_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            entries=tuple(
                FeatureMetadata.from_json_dict(as_json_dict(e, field_name="entries[]"))
                for e in as_json_list(raw.get("entries") or [], field_name="entries")
            ),
        )


def build_feature_catalog(snapshot: FeatureRegistrySnapshot) -> FeatureCatalog:
    entries = tuple(sorted(snapshot.metadata, key=lambda m: m.feature_name))
    return FeatureCatalog(schema_version=FEATURE_CATALOG_SCHEMA_VERSION, dataset_id=snapshot.dataset_id, entries=entries)


def _grouped(catalog: FeatureCatalog, key: str) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for entry in catalog.entries:
        value = getattr(entry, key)
        groups.setdefault(str(value), []).append(entry.feature_name)
    return {k: tuple(sorted(v)) for k, v in sorted(groups.items())}


@dataclass(frozen=True, slots=True)
class FeatureInventory:
    schema_version: int
    dataset_id: str
    complete_catalog: FeatureCatalog
    grouped_catalog: dict[str, tuple[str, ...]]
    """`feature_group` -> feature names."""
    dataset_catalog: dict[str, tuple[str, ...]]
    """`origin_dataset` -> feature names (trivially one key for a single-dataset catalog; meaningful once
    two inventories are merged/reconciled)."""
    origin_catalog: dict[str, tuple[str, ...]]
    """`creation_stage` -> feature names."""
    availability_catalog: dict[str, tuple[str, ...]]
    """`availability_rule` -> feature names."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "complete_catalog": self.complete_catalog.to_json_dict(),
            "grouped_catalog": {k: list(v) for k, v in self.grouped_catalog.items()},
            "dataset_catalog": {k: list(v) for k, v in self.dataset_catalog.items()},
            "origin_catalog": {k: list(v) for k, v in self.origin_catalog.items()},
            "availability_catalog": {k: list(v) for k, v in self.availability_catalog.items()},
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureInventory:
        require_schema_version(raw, supported=FEATURE_INVENTORY_SCHEMA_VERSION, context="FeatureInventory")

        def _dict_of_tuples(field_name: str) -> dict[str, tuple[str, ...]]:
            return {str(k): tuple(str(s) for s in v) for k, v in as_json_dict(raw.get(field_name) or {}, field_name=field_name).items()}

        return cls(
            schema_version=FEATURE_INVENTORY_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            complete_catalog=FeatureCatalog.from_json_dict(as_json_dict(raw["complete_catalog"], field_name="complete_catalog")),
            grouped_catalog=_dict_of_tuples("grouped_catalog"), dataset_catalog=_dict_of_tuples("dataset_catalog"),
            origin_catalog=_dict_of_tuples("origin_catalog"), availability_catalog=_dict_of_tuples("availability_catalog"),
        )


def build_feature_inventory(catalog: FeatureCatalog) -> FeatureInventory:
    return FeatureInventory(
        schema_version=FEATURE_INVENTORY_SCHEMA_VERSION, dataset_id=catalog.dataset_id, complete_catalog=catalog,
        grouped_catalog=_grouped(catalog, "feature_group"), dataset_catalog=_grouped(catalog, "origin_dataset"),
        origin_catalog=_grouped(catalog, "creation_stage"), availability_catalog=_grouped(catalog, "availability_rule"),
    )


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    """This package's OWN deterministic, content-addressed bundle
    identity for a snapshot's exact feature set -- distinct from, and
    never a replacement for, `features.manifests.
    ResearchDatasetManifest`."""

    schema_version: int
    manifest_id: str
    dataset_id: str
    origin_manifest: str
    generated_at: str
    feature_count: int
    feature_ids: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "manifest_id": self.manifest_id, "dataset_id": self.dataset_id,
            "origin_manifest": self.origin_manifest, "generated_at": self.generated_at, "feature_count": self.feature_count,
            "feature_ids": list(self.feature_ids),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureManifest:
        require_schema_version(raw, supported=FEATURE_MANIFEST_SCHEMA_VERSION, context="FeatureManifest")
        return cls(
            schema_version=FEATURE_MANIFEST_SCHEMA_VERSION, manifest_id=str(raw["manifest_id"]), dataset_id=str(raw["dataset_id"]),
            origin_manifest=str(raw["origin_manifest"]), generated_at=str(raw["generated_at"]), feature_count=int(str(raw["feature_count"])),
            feature_ids=tuple(str(s) for s in as_json_list(raw.get("feature_ids") or [], field_name="feature_ids")),
        )


def build_feature_manifest(snapshot: FeatureRegistrySnapshot) -> FeatureManifest:
    feature_ids = tuple(sorted(m.feature_id for m in snapshot.metadata))
    manifest_id = hashlib.sha256(f"{snapshot.dataset_id}|{','.join(feature_ids)}".encode()).hexdigest()[:16]
    return FeatureManifest(
        schema_version=FEATURE_MANIFEST_SCHEMA_VERSION, manifest_id=manifest_id, dataset_id=snapshot.dataset_id,
        origin_manifest=snapshot.manifest_version, generated_at=format_utc_timestamp(utc_now()), feature_count=len(feature_ids),
        feature_ids=feature_ids,
    )


@dataclass(frozen=True, slots=True)
class FeatureInfrastructureBundle:
    """Every infrastructure artifact this part of `feature_discovery`
    produces for one `(registry, manifest)` pair, assembled together --
    the single object `infra_verification.py`/`infra_reconciliation.py`
    operate on so neither has to re-derive the graph/catalog/inventory/
    manifest from the snapshot itself."""

    snapshot: FeatureRegistrySnapshot
    graph: FeatureDependencyGraph
    catalog: FeatureCatalog
    inventory: FeatureInventory
    manifest: FeatureManifest

    def to_json_dict(self) -> dict[str, object]:
        return {
            "snapshot": self.snapshot.to_json_dict(), "graph": self.graph.to_json_dict(), "catalog": self.catalog.to_json_dict(),
            "inventory": self.inventory.to_json_dict(), "manifest": self.manifest.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureInfrastructureBundle:
        return cls(
            snapshot=FeatureRegistrySnapshot.from_json_dict(as_json_dict(raw["snapshot"], field_name="snapshot")),
            graph=FeatureDependencyGraph.from_json_dict(as_json_dict(raw["graph"], field_name="graph")),
            catalog=FeatureCatalog.from_json_dict(as_json_dict(raw["catalog"], field_name="catalog")),
            inventory=FeatureInventory.from_json_dict(as_json_dict(raw["inventory"], field_name="inventory")),
            manifest=FeatureManifest.from_json_dict(as_json_dict(raw["manifest"], field_name="manifest")),
        )


def build_feature_infrastructure_bundle(registry: FeatureRegistry, manifest: ResearchDatasetManifest) -> FeatureInfrastructureBundle:
    snapshot = capture_feature_registry_snapshot(registry, manifest)
    graph = build_feature_dependency_graph(snapshot, declared_feature_names=frozenset(manifest.feature_names))
    catalog = build_feature_catalog(snapshot)
    inventory = build_feature_inventory(catalog)
    feature_manifest = build_feature_manifest(snapshot)
    return FeatureInfrastructureBundle(snapshot=snapshot, graph=graph, catalog=catalog, inventory=inventory, manifest=feature_manifest)
