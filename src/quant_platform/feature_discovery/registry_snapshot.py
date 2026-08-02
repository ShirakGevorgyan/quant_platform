"""`FeatureRegistrySnapshot` (Milestone 11, Phase 2, Part 2): a
point-in-time capture of every feature a `ResearchDatasetManifest`
actually used, read from the caller's own live, real, unmodified
`features.registry.FeatureRegistry` -- never a second registry. This is
the ONE place `metadata.py`'s `compute_feature_metadata`/
`compute_feature_provenance` and `features.lineage.build_lineage` are
called for every feature in a dataset; everything else in this part of
`feature_discovery` (the dependency graph, catalog, inventory, health
reports) is built FROM the resulting snapshot, never by re-reading the
registry a second time."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import FeatureDiscoveryVerificationError, UnknownFeatureError
from quant_platform.feature_discovery.metadata import (
    FeatureMetadata,
    FeatureProvenance,
    compute_feature_metadata,
    compute_feature_provenance,
)
from quant_platform.features.lineage import FeatureLineage, build_lineage
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.registry import FeatureRegistry
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["FEATURE_REGISTRY_SNAPSHOT_SCHEMA_VERSION", "FeatureRegistrySnapshot", "capture_feature_registry_snapshot"]

FEATURE_REGISTRY_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureRegistrySnapshot:
    schema_version: int
    dataset_id: str
    manifest_version: str
    captured_at: str
    metadata: tuple[FeatureMetadata, ...]
    lineages: tuple[FeatureLineage, ...]
    provenances: tuple[FeatureProvenance, ...]

    def metadata_for(self, feature_name: str) -> FeatureMetadata:
        for entry in self.metadata:
            if entry.feature_name == feature_name:
                return entry
        raise FeatureDiscoveryVerificationError(
            f"No FeatureMetadata captured for feature_name={feature_name!r}", context={"dataset_id": self.dataset_id, "feature_name": feature_name},
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "manifest_version": self.manifest_version,
            "captured_at": self.captured_at, "metadata": [m.to_json_dict() for m in self.metadata],
            "lineages": [ln.to_json_dict() for ln in self.lineages], "provenances": [p.to_json_dict() for p in self.provenances],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureRegistrySnapshot:
        require_schema_version(raw, supported=FEATURE_REGISTRY_SNAPSHOT_SCHEMA_VERSION, context="FeatureRegistrySnapshot")
        return cls(
            schema_version=FEATURE_REGISTRY_SNAPSHOT_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), manifest_version=str(raw["manifest_version"]),
            captured_at=str(raw["captured_at"]),
            metadata=tuple(
                FeatureMetadata.from_json_dict(as_json_dict(m, field_name="metadata[]"))
                for m in as_json_list(raw.get("metadata") or [], field_name="metadata")
            ),
            lineages=tuple(
                FeatureLineage.from_json_dict(as_json_dict(ln, field_name="lineages[]"))
                for ln in as_json_list(raw.get("lineages") or [], field_name="lineages")
            ),
            provenances=tuple(
                FeatureProvenance.from_json_dict(as_json_dict(p, field_name="provenances[]"))
                for p in as_json_list(raw.get("provenances") or [], field_name="provenances")
            ),
        )


def capture_feature_registry_snapshot(registry: FeatureRegistry, manifest: ResearchDatasetManifest) -> FeatureRegistrySnapshot:
    metadata: list[FeatureMetadata] = []
    lineages: list[FeatureLineage] = []
    provenances: list[FeatureProvenance] = []
    for name in sorted(manifest.feature_names):
        try:
            spec = registry.get(name, manifest.feature_versions.get(name)).spec
        except UnknownFeatureError as exc:
            raise FeatureDiscoveryVerificationError(
                f"manifest declares feature {name!r} (version {manifest.feature_versions.get(name)!r}) which is not "
                "registered in the supplied FeatureRegistry -- cannot capture a snapshot without it",
                context={"dataset_id": manifest.dataset_id, "feature_name": name},
            ) from exc
        metadata.append(compute_feature_metadata(spec, dataset_id=manifest.dataset_id, manifest_version=manifest.version))
        lineages.append(build_lineage(spec, source_dataset_manifest_id=manifest.source_historical_dataset_id, transformation=spec.description))
        provenances.append(compute_feature_provenance(spec, manifest))

    return FeatureRegistrySnapshot(
        schema_version=FEATURE_REGISTRY_SNAPSHOT_SCHEMA_VERSION, dataset_id=manifest.dataset_id, manifest_version=manifest.version,
        captured_at=format_utc_timestamp(utc_now()), metadata=tuple(metadata), lineages=tuple(lineages), provenances=tuple(provenances),
    )
