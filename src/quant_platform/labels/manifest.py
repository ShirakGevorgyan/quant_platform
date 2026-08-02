"""`LabelManifest` (Milestone 11, Phase 3, Part A): the self-contained,
content-addressed summary of one `models.LabelSpecification`'s full
lineage -- everything a consumer needs to know without cross-referencing
`registry.LabelRegistry` or re-reading the specification itself.

`feature_identity`/`qualification_identity` are plain, caller-supplied,
optional identity strings (e.g. a `ResearchDatasetManifest.
feature_registry_fingerprint`, or a hash of a `DatasetQualificationReport`)
-- this module never imports `features`/`qualification`/`feature_discovery`
to compute them itself, for the same dependency-direction reason
`models.py`'s own docstring gives. `dependency_chain` renders the
Market Data -> Features -> Qualification -> Feature Discovery -> Labels
lineage as an ordered tuple of identity references -- a linear chain,
never a branching graph, since exactly one specification produces
exactly one manifest with exactly one upstream lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from quant_platform.labels.models import LabelSpecification
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["LABEL_MANIFEST_SCHEMA_VERSION", "LabelManifest", "build_label_manifest", "compute_manifest_checksum"]

LABEL_MANIFEST_SCHEMA_VERSION = 1


def compute_manifest_checksum(
    *, label_specification_id: str, dataset_identity: str, manifest_identity: str, feature_identity: str | None,
    qualification_identity: str | None, generation_parameters: dict[str, object], availability_semantics: str,
    dependency_chain: tuple[str, ...],
) -> str:
    """Deterministic sha256 over every OTHER field of a `LabelManifest` --
    deliberately EXCLUDES `generation_timestamp` (wall-clock, non-identity)
    so two manifests built at different times for the identical
    specification/lineage always checksum identically."""
    payload = {
        "label_specification_id": label_specification_id, "dataset_identity": dataset_identity, "manifest_identity": manifest_identity,
        "feature_identity": feature_identity, "qualification_identity": qualification_identity,
        "generation_parameters": generation_parameters, "availability_semantics": availability_semantics,
        "dependency_chain": list(dependency_chain),
    }
    blob = json.dumps(payload, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelManifest:
    schema_version: int
    manifest_checksum: str
    label_specification_id: str
    dataset_identity: str
    manifest_identity: str
    feature_identity: str | None
    qualification_identity: str | None
    generation_parameters: dict[str, object]
    generation_timestamp: str
    availability_semantics: str
    dependency_chain: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "manifest_checksum": self.manifest_checksum,
            "label_specification_id": self.label_specification_id, "dataset_identity": self.dataset_identity,
            "manifest_identity": self.manifest_identity, "feature_identity": self.feature_identity,
            "qualification_identity": self.qualification_identity, "generation_parameters": self.generation_parameters,
            "generation_timestamp": self.generation_timestamp, "availability_semantics": self.availability_semantics,
            "dependency_chain": list(self.dependency_chain),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelManifest:
        require_schema_version(raw, supported=LABEL_MANIFEST_SCHEMA_VERSION, context="LabelManifest")
        return cls(
            schema_version=LABEL_MANIFEST_SCHEMA_VERSION, manifest_checksum=str(raw["manifest_checksum"]),
            label_specification_id=str(raw["label_specification_id"]), dataset_identity=str(raw["dataset_identity"]),
            manifest_identity=str(raw["manifest_identity"]),
            feature_identity=(None if raw.get("feature_identity") is None else str(raw["feature_identity"])),
            qualification_identity=(None if raw.get("qualification_identity") is None else str(raw["qualification_identity"])),
            generation_parameters=as_json_dict(raw.get("generation_parameters") or {}, field_name="generation_parameters"),
            generation_timestamp=str(raw["generation_timestamp"]), availability_semantics=str(raw["availability_semantics"]),
            dependency_chain=tuple(str(s) for s in as_json_list(raw.get("dependency_chain") or [], field_name="dependency_chain")),
        )

    def verify_self_consistency(self) -> tuple[bool, tuple[str, ...]]:
        recomputed = compute_manifest_checksum(
            label_specification_id=self.label_specification_id, dataset_identity=self.dataset_identity, manifest_identity=self.manifest_identity,
            feature_identity=self.feature_identity, qualification_identity=self.qualification_identity,
            generation_parameters=self.generation_parameters, availability_semantics=self.availability_semantics,
            dependency_chain=self.dependency_chain,
        )
        if recomputed != self.manifest_checksum:
            return False, (f"manifest_checksum: claims {self.manifest_checksum!r}, recomputed is {recomputed!r}",)
        return True, ()


def build_label_manifest(
    specification: LabelSpecification, *, generation_timestamp: str, feature_identity: str | None = None,
    qualification_identity: str | None = None,
) -> LabelManifest:
    dependency_chain = (
        f"dataset:{specification.created_from_dataset}",
        f"manifest:{specification.created_from_manifest}",
        f"features:{feature_identity or 'unavailable'}",
        f"qualification:{qualification_identity or 'unavailable'}",
        f"label_specification:{specification.label_specification_id}",
    )
    manifest_checksum = compute_manifest_checksum(
        label_specification_id=specification.label_specification_id, dataset_identity=specification.created_from_dataset,
        manifest_identity=specification.created_from_manifest, feature_identity=feature_identity, qualification_identity=qualification_identity,
        generation_parameters=specification.parameters, availability_semantics=specification.availability_rule, dependency_chain=dependency_chain,
    )
    return LabelManifest(
        schema_version=LABEL_MANIFEST_SCHEMA_VERSION, manifest_checksum=manifest_checksum, label_specification_id=specification.label_specification_id,
        dataset_identity=specification.created_from_dataset, manifest_identity=specification.created_from_manifest, feature_identity=feature_identity,
        qualification_identity=qualification_identity, generation_parameters=specification.parameters, generation_timestamp=generation_timestamp,
        availability_semantics=specification.availability_rule, dependency_chain=dependency_chain,
    )
