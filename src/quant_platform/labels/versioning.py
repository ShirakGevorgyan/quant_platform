"""`LabelVersion` and `LabelVersionHistory` (Milestone 11, Phase 3, Part
A) -- mirrors `feature_discovery.metadata.FeatureVersionEntry`/
`FeatureVersionHistory` exactly, one layer up: every specification ever
registered for a given `models.LabelFamily`, in append-only order, never
merely the one a caller happens to be looking at. `registered_at` is a
disclosed, non-identity-bearing audit timestamp -- excluded from every
hash in this package, present here purely for a human reading a version
history report."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["LABEL_VERSION_HISTORY_SCHEMA_VERSION", "LabelVersion", "LabelVersionHistory"]

LABEL_VERSION_HISTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelVersion:
    generation_version: str
    label_specification_id: str
    parameter_hash: str
    registered_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "generation_version": self.generation_version, "label_specification_id": self.label_specification_id,
            "parameter_hash": self.parameter_hash, "registered_at": self.registered_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelVersion:
        return cls(
            generation_version=str(raw["generation_version"]), label_specification_id=str(raw["label_specification_id"]),
            parameter_hash=str(raw["parameter_hash"]), registered_at=str(raw["registered_at"]),
        )


@dataclass(frozen=True, slots=True)
class LabelVersionHistory:
    schema_version: int
    label_family: str
    versions: tuple[LabelVersion, ...]
    """Every specification registered for `label_family`, sorted by
    `(generation_version, parameter_hash)` -- deterministic regardless of
    registration order."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_family": self.label_family,
            "versions": [v.to_json_dict() for v in self.versions],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelVersionHistory:
        require_schema_version(raw, supported=LABEL_VERSION_HISTORY_SCHEMA_VERSION, context="LabelVersionHistory")
        return cls(
            schema_version=LABEL_VERSION_HISTORY_SCHEMA_VERSION, label_family=str(raw["label_family"]),
            versions=tuple(
                LabelVersion.from_json_dict(as_json_dict(v, field_name="versions[]"))
                for v in as_json_list(raw.get("versions") or [], field_name="versions")
            ),
        )
