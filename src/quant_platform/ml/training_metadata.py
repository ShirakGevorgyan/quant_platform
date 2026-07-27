"""`TrainingMetadata` (Milestone 4C, "TRAINING METADATA") -- one immutable,
content-addressed provenance record per fitted model, persisted as its
own `ArtifactCategory.TRAINING_METADATA` artifact, distinct from the
fitted model's own serialized bytes (which carry `ModelMetadata` --
objective/feature-schema/capabilities/hyperparameters -- not this
record's execution-provenance fields: WHEN it was fit, HOW LONG it took,
which library version actually did the fitting, and which experiment/
fold/dataset it belongs to).

WHY A SEPARATE ARTIFACT, NOT MORE `ModelMetadata` FIELDS
--------------------------------------------------------------------------
`ModelMetadata` is IDENTITY-ADJACENT: it is embedded inside the model's
own serialized bytes and round-trips through `ModelDeserializer.
deserialize`'s `expected_metadata` check, so it must stay scientifically
minimal (name/version/objective/feature_schema/capabilities/
hyperparameters -- inputs that determine WHAT would be fit). Training
duration, wall-clock fit time, and the exact installed library version
are execution-provenance facts about ONE fit run, not inputs to it --
mixing them into `ModelMetadata` would make two scientifically identical
fits (same data, same hyperparameters, same seed) compare as "different
models" the moment they were timed differently. Keeping them in a
separate artifact keeps `ModelMetadata` a pure function of identity-
relevant inputs while still recording everything Section "TRAINING
METADATA" asks for, exactly the same separation Milestone 4B already
drew between `FoldResult` (per-fold outcome) and `ExecutionManifest`
(current run state).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_platform.ml.models import JsonPrimitive, validate_json_primitive_mapping
from quant_platform.ml.persistence import parse_utc_timestamp, require_schema_version

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    schema_version: int
    experiment_id: str
    fold_index: int
    model_name: str
    model_version: str
    library_name: str
    library_version: str
    seed: int
    training_duration_seconds: float
    feature_schema_fingerprint: str
    dataset_content_id: str
    fitted_at: str
    hyperparameters: dict[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError(f"TrainingMetadata.fold_index must be >= 0, got {self.fold_index}")
        if not self.model_name:
            raise ValueError("TrainingMetadata.model_name must not be empty")
        if not self.model_version:
            raise ValueError("TrainingMetadata.model_version must not be empty")
        if not self.library_name:
            raise ValueError("TrainingMetadata.library_name must not be empty")
        if not self.library_version:
            raise ValueError("TrainingMetadata.library_version must not be empty")
        if self.seed < 0:
            raise ValueError(f"TrainingMetadata.seed must be >= 0, got {self.seed}")
        if self.training_duration_seconds < 0:
            raise ValueError(f"TrainingMetadata.training_duration_seconds must be >= 0, got {self.training_duration_seconds}")
        if not self.feature_schema_fingerprint:
            raise ValueError("TrainingMetadata.feature_schema_fingerprint must not be empty")
        if not self.dataset_content_id:
            raise ValueError("TrainingMetadata.dataset_content_id must not be empty")
        parse_utc_timestamp(self.fitted_at)
        validate_json_primitive_mapping(self.hyperparameters, field_name="TrainingMetadata.hyperparameters")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "fold_index": self.fold_index,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "library_name": self.library_name,
            "library_version": self.library_version,
            "seed": self.seed,
            "training_duration_seconds": self.training_duration_seconds,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "dataset_content_id": self.dataset_content_id,
            "fitted_at": self.fitted_at,
            "hyperparameters": dict(sorted(self.hyperparameters.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TrainingMetadata:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="TrainingMetadata")
        hyperparameters_raw = raw.get("hyperparameters") or {}
        if not isinstance(hyperparameters_raw, dict):
            raise ValueError("TrainingMetadata.hyperparameters must be a JSON object")
        return cls(
            schema_version=_SCHEMA_VERSION,
            experiment_id=str(raw["experiment_id"]),
            fold_index=int(str(raw["fold_index"])),
            model_name=str(raw["model_name"]),
            model_version=str(raw["model_version"]),
            library_name=str(raw["library_name"]),
            library_version=str(raw["library_version"]),
            seed=int(str(raw["seed"])),
            training_duration_seconds=float(str(raw["training_duration_seconds"])),
            feature_schema_fingerprint=str(raw["feature_schema_fingerprint"]),
            dataset_content_id=str(raw["dataset_content_id"]),
            fitted_at=str(raw["fitted_at"]),
            hyperparameters=dict(hyperparameters_raw),
        )


__all__ = ["TrainingMetadata"]
