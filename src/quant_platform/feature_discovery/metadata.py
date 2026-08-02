"""Feature Metadata, Feature Provenance, Feature Version History
(Milestone 11, Phase 2, Part 2) -- infrastructure only, no statistic
that depends on a prediction target anywhere in this module.

Every field is derived directly from the REAL, already-existing
`features.models.FeatureSpec` (read via the caller's own live
`features.registry.FeatureRegistry`, exactly as `feature_cli.py`'s own
`report-lineage` command already does -- `registry.get(name, manifest.
feature_versions.get(name)).spec`) and `features.manifests.
ResearchDatasetManifest` -- never fabricated. `FeatureSpec` already
carries `feature_dependencies` (the dependency-graph edge data),
`availability_delay`/`warmup_bars` (the availability/warmup contract),
`category` (the feature group), and `fingerprint()` (a deterministic
sha256 over the whole spec) -- this module's own job is assembling
those EXISTING fields into the shapes the spec's Feature Metadata
section names, never recomputing any of them a second way."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.models import FeatureSpec
from quant_platform.features.registry import FeatureRegistry
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "FEATURE_METADATA_SCHEMA_VERSION",
    "FEATURE_PROVENANCE_SCHEMA_VERSION",
    "FEATURE_VERSION_HISTORY_SCHEMA_VERSION",
    "FeatureMetadata",
    "FeatureProvenance",
    "FeatureVersionEntry",
    "FeatureVersionHistory",
    "compute_feature_metadata",
    "compute_feature_provenance",
    "compute_feature_version_history",
]

FEATURE_METADATA_SCHEMA_VERSION = 1
FEATURE_PROVENANCE_SCHEMA_VERSION = 1
FEATURE_VERSION_HISTORY_SCHEMA_VERSION = 1


def _availability_rule(spec: FeatureSpec) -> str:
    delay_seconds = spec.availability_delay.total_seconds()
    base = f"available at open_time + {spec.source_timeframe.value} bar duration"
    return f"{base} + {delay_seconds:.0f}s release delay" if delay_seconds > 0 else base


def _creation_stage(spec: FeatureSpec) -> str:
    """`"higher_order_feature"` if this feature depends on OTHER
    features (`feature_dependencies`); `"derived_feature"` if it only
    reads raw input columns (`required_inputs`); `"raw_source"` only for
    a feature needing no inputs of any kind (rare -- e.g. a pure
    row-position feature)."""
    if spec.feature_dependencies:
        return "higher_order_feature"
    if spec.required_inputs:
        return "derived_feature"
    return "raw_source"


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    schema_version: int
    feature_id: str
    """`f"{name}@{version}"` (`FeatureSpec.qualified_name`) -- stable across every dataset the same
    feature appears in, exactly matching `FeatureSpec`'s own established `(name, version)` identity."""
    feature_name: str
    feature_group: str
    """`FeatureSpec.category.value`."""
    origin_dataset: str
    origin_manifest: str
    creation_stage: str
    availability_rule: str
    warmup_requirement: int
    dependencies: tuple[str, ...]
    """`FeatureSpec.feature_dependencies` -- other FEATURE names (not raw input columns)."""
    deterministic_identity: str
    """`FeatureSpec.fingerprint()` -- unchanged if and only if every field of the spec is unchanged."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "feature_id": self.feature_id, "feature_name": self.feature_name,
            "feature_group": self.feature_group, "origin_dataset": self.origin_dataset, "origin_manifest": self.origin_manifest,
            "creation_stage": self.creation_stage, "availability_rule": self.availability_rule,
            "warmup_requirement": self.warmup_requirement, "dependencies": list(self.dependencies),
            "deterministic_identity": self.deterministic_identity,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureMetadata:
        require_schema_version(raw, supported=FEATURE_METADATA_SCHEMA_VERSION, context="FeatureMetadata")
        return cls(
            schema_version=FEATURE_METADATA_SCHEMA_VERSION, feature_id=str(raw["feature_id"]), feature_name=str(raw["feature_name"]),
            feature_group=str(raw["feature_group"]), origin_dataset=str(raw["origin_dataset"]), origin_manifest=str(raw["origin_manifest"]),
            creation_stage=str(raw["creation_stage"]), availability_rule=str(raw["availability_rule"]),
            warmup_requirement=int(str(raw["warmup_requirement"])),
            dependencies=tuple(str(s) for s in as_json_list(raw.get("dependencies") or [], field_name="dependencies")),
            deterministic_identity=str(raw["deterministic_identity"]),
        )


def compute_feature_metadata(spec: FeatureSpec, *, dataset_id: str, manifest_version: str) -> FeatureMetadata:
    return FeatureMetadata(
        schema_version=FEATURE_METADATA_SCHEMA_VERSION, feature_id=spec.qualified_name, feature_name=spec.name,
        feature_group=spec.category.value, origin_dataset=dataset_id, origin_manifest=manifest_version,
        creation_stage=_creation_stage(spec), availability_rule=_availability_rule(spec), warmup_requirement=spec.warmup_bars,
        dependencies=spec.feature_dependencies, deterministic_identity=spec.fingerprint(),
    )


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """The chain-of-custody a `FeatureMetadata` record alone doesn't
    carry: which historical dataset/manifest the base data came from,
    what code revision built it, and whether a `market_data_bridge`
    lineage exists for the whole dataset."""

    schema_version: int
    feature_name: str
    origin_dataset: str
    origin_manifest: str
    source_historical_dataset_id: str
    source_historical_manifest_version: str
    source_symbols: tuple[str, ...]
    source_timeframe: str
    code_revision: str
    has_market_data_lineage: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "feature_name": self.feature_name, "origin_dataset": self.origin_dataset,
            "origin_manifest": self.origin_manifest, "source_historical_dataset_id": self.source_historical_dataset_id,
            "source_historical_manifest_version": self.source_historical_manifest_version, "source_symbols": list(self.source_symbols),
            "source_timeframe": self.source_timeframe, "code_revision": self.code_revision,
            "has_market_data_lineage": self.has_market_data_lineage,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureProvenance:
        require_schema_version(raw, supported=FEATURE_PROVENANCE_SCHEMA_VERSION, context="FeatureProvenance")
        return cls(
            schema_version=FEATURE_PROVENANCE_SCHEMA_VERSION, feature_name=str(raw["feature_name"]), origin_dataset=str(raw["origin_dataset"]),
            origin_manifest=str(raw["origin_manifest"]), source_historical_dataset_id=str(raw["source_historical_dataset_id"]),
            source_historical_manifest_version=str(raw["source_historical_manifest_version"]),
            source_symbols=tuple(str(s) for s in as_json_list(raw.get("source_symbols") or [], field_name="source_symbols")),
            source_timeframe=str(raw["source_timeframe"]), code_revision=str(raw["code_revision"]),
            has_market_data_lineage=bool(raw["has_market_data_lineage"]),
        )


def compute_feature_provenance(spec: FeatureSpec, manifest: ResearchDatasetManifest) -> FeatureProvenance:
    return FeatureProvenance(
        schema_version=FEATURE_PROVENANCE_SCHEMA_VERSION, feature_name=spec.name, origin_dataset=manifest.dataset_id,
        origin_manifest=manifest.version, source_historical_dataset_id=manifest.source_historical_dataset_id,
        source_historical_manifest_version=manifest.source_historical_manifest_version, source_symbols=spec.source_symbols,
        source_timeframe=spec.source_timeframe.value, code_revision=manifest.code_revision,
        has_market_data_lineage=manifest.market_data_lineage is not None,
    )


@dataclass(frozen=True, slots=True)
class FeatureVersionEntry:
    version: str
    spec_fingerprint: str
    description: str

    def to_json_dict(self) -> dict[str, object]:
        return {"version": self.version, "spec_fingerprint": self.spec_fingerprint, "description": self.description}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureVersionEntry:
        return cls(version=str(raw["version"]), spec_fingerprint=str(raw["spec_fingerprint"]), description=str(raw["description"]))


@dataclass(frozen=True, slots=True)
class FeatureVersionHistory:
    schema_version: int
    feature_name: str
    current_version: str
    versions: tuple[FeatureVersionEntry, ...]
    """Every version of `feature_name` registered in the `FeatureRegistry` this history was captured
    from, sorted by version string -- not merely the one version the current dataset happens to use."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "feature_name": self.feature_name, "current_version": self.current_version,
            "versions": [v.to_json_dict() for v in self.versions],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureVersionHistory:
        require_schema_version(raw, supported=FEATURE_VERSION_HISTORY_SCHEMA_VERSION, context="FeatureVersionHistory")
        return cls(
            schema_version=FEATURE_VERSION_HISTORY_SCHEMA_VERSION, feature_name=str(raw["feature_name"]),
            current_version=str(raw["current_version"]),
            versions=tuple(
                FeatureVersionEntry.from_json_dict(as_json_dict(v, field_name="versions[]"))
                for v in as_json_list(raw.get("versions") or [], field_name="versions")
            ),
        )


def compute_feature_version_history(registry: FeatureRegistry, feature_name: str, *, current_version: str) -> FeatureVersionHistory:
    matching = sorted((spec for spec in registry.list_features() if spec.name == feature_name), key=lambda s: s.version)
    versions = tuple(FeatureVersionEntry(version=s.version, spec_fingerprint=s.fingerprint(), description=s.description) for s in matching)
    return FeatureVersionHistory(
        schema_version=FEATURE_VERSION_HISTORY_SCHEMA_VERSION, feature_name=feature_name, current_version=current_version, versions=versions,
    )
