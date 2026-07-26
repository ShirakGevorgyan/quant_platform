"""Strongly typed, immutable domain models for the ML research platform.

Every model here is a frozen dataclass with an explicit `__post_init__`
validator and `to_json_dict`/`from_json_dict` canonical serialization
(mirroring `features.models.FeatureSpec`'s established convention) --
nothing important is represented as an arbitrary unvalidated dict.

IDENTITY-RELEVANT VS. DESCRIPTIVE
--------------------------------------------------------------------------
Every binding model in this module (`DatasetBinding`, `FeatureBinding`,
`LabelBinding`, `SplitBinding`, `PreprocessingBinding`, `CodeRevisionBinding`)
is IDENTITY-RELEVANT -- it participates in `experiment_identity.
compute_experiment_identity`. `EnvironmentSnapshot` is the one model in
this file that is deliberately, permanently INFORMATIONAL ONLY: it never
participates in experiment identity (see its own docstring and
`ml/environment.py`'s module docstring for why).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from quant_platform.core.exceptions import ArtifactCorruptionError, UnsupportedObjectiveError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

_SCHEMA_VERSION = 1

JsonPrimitive = str | int | float | bool | None


def validate_json_primitive_mapping(mapping: Mapping[str, object], *, field_name: str) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name}: keys must be non-empty strings, got {key!r}")
        if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{field_name}[{key!r}] must be finite, got {value!r}")
            continue
        raise ValueError(f"{field_name}[{key!r}] has unsupported type {type(value).__name__}")


# --------------------------------------------------------------------------
# Objective / label compatibility
# --------------------------------------------------------------------------
class ObjectiveType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class LabelType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    MULTICLASS = "multiclass"


_COMPATIBLE_LABEL_TYPES: dict[ObjectiveType, tuple[LabelType, ...]] = {
    ObjectiveType.BINARY_CLASSIFICATION: (LabelType.BINARY,),
    ObjectiveType.MULTICLASS_CLASSIFICATION: (LabelType.MULTICLASS,),
    ObjectiveType.REGRESSION: (LabelType.CONTINUOUS,),
}


def objective_supports_label_type(objective: ObjectiveType, label_type: LabelType) -> bool:
    return label_type in _COMPATIBLE_LABEL_TYPES[objective]


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ArtifactCategory(Enum):
    MODEL = "model"
    PREDICTIONS = "predictions"
    PROBABILITIES = "probabilities"
    METRICS = "metrics"
    FEATURE_IMPORTANCE = "feature_importance"
    CALIBRATION = "calibration"
    REPORT = "report"
    EXPERIMENT_MANIFEST = "experiment_manifest"
    LOGS = "logs"


class ExperimentStatus(Enum):
    CREATED = "created"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Legal status transitions -- see manifests.py's module docstring for the
# full state-machine rationale.
_LEGAL_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.CREATED: frozenset({ExperimentStatus.VALIDATING, ExperimentStatus.CANCELLED}),
    ExperimentStatus.VALIDATING: frozenset(
        {ExperimentStatus.READY, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED}
    ),
    ExperimentStatus.READY: frozenset({ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED}),
    ExperimentStatus.RUNNING: frozenset({ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}),
    ExperimentStatus.COMPLETED: frozenset(),
    ExperimentStatus.FAILED: frozenset(),
    ExperimentStatus.CANCELLED: frozenset(),
}


def is_legal_transition(current: ExperimentStatus, target: ExperimentStatus) -> bool:
    return target in _LEGAL_TRANSITIONS[current]


# --------------------------------------------------------------------------
# Model capabilities / hyperparameters
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    supported_objectives: tuple[ObjectiveType, ...]
    supports_predict_proba: bool = False
    supports_feature_importance: bool = False
    supports_incremental_fit: bool = False

    def __post_init__(self) -> None:
        if not self.supported_objectives:
            raise ValueError("ModelCapabilities.supported_objectives must not be empty")
        if len(set(self.supported_objectives)) != len(self.supported_objectives):
            raise ValueError("ModelCapabilities.supported_objectives must not contain duplicates")
        if self.supports_predict_proba and ObjectiveType.REGRESSION in self.supported_objectives and len(self.supported_objectives) == 1:
            raise ValueError("A regression-only model cannot declare supports_predict_proba=True")

    def supports(self, objective: ObjectiveType) -> bool:
        return objective in self.supported_objectives

    def require_predict_proba(self, objective: ObjectiveType) -> None:
        if not self.supports_predict_proba or objective is ObjectiveType.REGRESSION:
            raise UnsupportedObjectiveError(
                f"predict_proba is not supported for objective {objective.value!r}",
                context={"objective": objective.value},
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "supported_objectives": [o.value for o in self.supported_objectives],
            "supports_predict_proba": self.supports_predict_proba,
            "supports_feature_importance": self.supports_feature_importance,
            "supports_incremental_fit": self.supports_incremental_fit,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ModelCapabilities:
        return cls(
            supported_objectives=tuple(
                ObjectiveType(o) for o in as_json_list(raw["supported_objectives"], field_name="supported_objectives")
            ),
            supports_predict_proba=bool(raw.get("supports_predict_proba", False)),
            supports_feature_importance=bool(raw.get("supports_feature_importance", False)),
            supports_incremental_fit=bool(raw.get("supports_incremental_fit", False)),
        )


@dataclass(frozen=True, slots=True)
class ModelHyperparameters:
    """A validated, JSON-primitive-only mapping of hyperparameter values.
    Deliberately generic (no per-model schema) at this foundational layer,
    since Milestone 4A registers no real model -- but never an
    unvalidated arbitrary dict: keys must be non-empty strings, values must
    be JSON primitives, and NaN/Infinity floats are rejected outright."""

    values: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_json_primitive_mapping(self.values, field_name="ModelHyperparameters.values")

    def to_json_dict(self) -> dict[str, object]:
        return {"values": dict(sorted(self.values.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ModelHyperparameters:
        values_raw = raw.get("values", {})
        if not isinstance(values_raw, dict):
            raise ValueError("ModelHyperparameters.values must be a JSON object")
        return cls(values=dict(values_raw))


# --------------------------------------------------------------------------
# Identity-relevant bindings
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DatasetBinding:
    """Binds an experiment to one exact, immutable research dataset
    artifact -- everything needed to reconstruct precisely which bytes an
    experiment was built from (see `features.manifests.
    ResearchDatasetManifest`, which this binding references but never
    duplicates storage logic for)."""

    dataset_id: str
    manifest_version: str
    content_id: str
    symbol: str
    base_timeframe: str
    source_historical_dataset_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "manifest_version", "content_id", "symbol", "base_timeframe"):
            if not getattr(self, field_name):
                raise ValueError(f"DatasetBinding.{field_name} must not be empty")
        if not is_valid_sha256_hex(self.content_id):
            raise ValueError(
                f"DatasetBinding.content_id must be a 64-character lowercase hex SHA-256 digest, got {self.content_id!r}"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "manifest_version": self.manifest_version,
            "content_id": self.content_id,
            "symbol": self.symbol,
            "base_timeframe": self.base_timeframe,
            "source_historical_dataset_id": self.source_historical_dataset_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DatasetBinding:
        return cls(
            dataset_id=str(raw["dataset_id"]), manifest_version=str(raw["manifest_version"]),
            content_id=str(raw["content_id"]), symbol=str(raw["symbol"]), base_timeframe=str(raw["base_timeframe"]),
            source_historical_dataset_id=(
                None if raw.get("source_historical_dataset_id") is None else str(raw["source_historical_dataset_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class FeatureBinding:
    """The EXACT, ORDERED set of features an experiment depends on. Order
    is preserved (never sorted) because a model consuming a fixed-width
    feature vector treats column order as semantically meaningful."""

    feature_names: tuple[str, ...]
    feature_versions: Mapping[str, str]
    feature_registry_fingerprint: str

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("FeatureBinding.feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureBinding.feature_names must not contain duplicates")
        if not is_valid_sha256_hex(self.feature_registry_fingerprint):
            raise ValueError(
                f"FeatureBinding.feature_registry_fingerprint must be a 64-character lowercase hex SHA-256 "
                f"digest, got {self.feature_registry_fingerprint!r}"
            )
        missing = [n for n in self.feature_names if n not in self.feature_versions]
        if missing:
            raise ValueError(f"FeatureBinding: no version recorded for feature(s) {missing}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_versions": dict(self.feature_versions),
            "feature_registry_fingerprint": self.feature_registry_fingerprint,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureBinding:
        return cls(
            feature_names=tuple(str(n) for n in as_json_list(raw["feature_names"], field_name="feature_names")),
            feature_versions={
                str(k): str(v) for k, v in as_json_dict(raw["feature_versions"], field_name="feature_versions").items()
            },
            feature_registry_fingerprint=str(raw["feature_registry_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class LabelBinding:
    name: str
    kind: str
    horizon_bars: int
    label_type: LabelType
    params: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LabelBinding.name must not be empty")
        if not self.kind:
            raise ValueError("LabelBinding.kind must not be empty")
        if self.horizon_bars <= 0:
            raise ValueError(f"LabelBinding.horizon_bars must be positive, got {self.horizon_bars}")
        validate_json_primitive_mapping(self.params, field_name="LabelBinding.params")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "kind": self.kind, "horizon_bars": self.horizon_bars,
            "label_type": self.label_type.value, "params": dict(sorted(self.params.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelBinding:
        return cls(
            name=str(raw["name"]), kind=str(raw["kind"]), horizon_bars=int(str(raw["horizon_bars"])),
            label_type=LabelType(raw["label_type"]),
            params=dict(as_json_dict(raw.get("params") or {}, field_name="LabelBinding.params")),
        )

    def require_compatible(self, objective: ObjectiveType) -> None:
        if not objective_supports_label_type(objective, self.label_type):
            raise UnsupportedObjectiveError(
                f"Label type {self.label_type.value!r} is not compatible with objective {objective.value!r}",
                context={"label_type": self.label_type.value, "objective": objective.value},
            )


@dataclass(frozen=True, slots=True)
class SplitBinding:
    strategy: str
    params: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("SplitBinding.strategy must not be empty")
        validate_json_primitive_mapping(self.params, field_name="SplitBinding.params")

    def to_json_dict(self) -> dict[str, object]:
        return {"strategy": self.strategy, "params": dict(sorted(self.params.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SplitBinding:
        return cls(
            strategy=str(raw["strategy"]),
            params=dict(as_json_dict(raw.get("params") or {}, field_name="SplitBinding.params")),
        )


@dataclass(frozen=True, slots=True)
class PreprocessingBinding:
    preprocessing_definition: Mapping[str, str] = field(default_factory=dict)
    fitted_preprocessing_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.fitted_preprocessing_fingerprint is not None and not is_valid_sha256_hex(self.fitted_preprocessing_fingerprint):
            raise ValueError(
                f"PreprocessingBinding.fitted_preprocessing_fingerprint must be a 64-character lowercase hex "
                f"SHA-256 digest or None, got {self.fitted_preprocessing_fingerprint!r}"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "preprocessing_definition": dict(sorted(self.preprocessing_definition.items())),
            "fitted_preprocessing_fingerprint": self.fitted_preprocessing_fingerprint,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PreprocessingBinding:
        return cls(
            preprocessing_definition={
                str(k): str(v)
                for k, v in as_json_dict(
                    raw.get("preprocessing_definition") or {}, field_name="preprocessing_definition"
                ).items()
            },
            fitted_preprocessing_fingerprint=(
                None if raw.get("fitted_preprocessing_fingerprint") is None
                else str(raw["fitted_preprocessing_fingerprint"])
            ),
        )


@dataclass(frozen=True, slots=True)
class CodeRevisionBinding:
    revision: str
    source: Literal["git", "content"]
    is_dirty: bool | None = None

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("CodeRevisionBinding.revision must not be empty")
        if self.source not in ("git", "content"):
            raise ValueError(f"CodeRevisionBinding.source must be 'git' or 'content', got {self.source!r}")
        if self.source == "content" and self.is_dirty is not None:
            raise ValueError("CodeRevisionBinding.is_dirty must be None when source='content' (no working tree to check)")

    def to_json_dict(self) -> dict[str, object]:
        return {"revision": self.revision, "source": self.source, "is_dirty": self.is_dirty}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CodeRevisionBinding:
        source = str(raw["source"])
        if source not in ("git", "content"):
            raise ValueError(f"Unknown CodeRevisionBinding.source {source!r}")
        return cls(
            revision=str(raw["revision"]), source=source,  # type: ignore[arg-type]
            is_dirty=(None if raw.get("is_dirty") is None else bool(raw["is_dirty"])),
        )


# --------------------------------------------------------------------------
# Environment snapshot -- INFORMATIONAL ONLY, never identity-relevant.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Captured actual runtime environment -- Python/OS/package versions,
    optional CPU metadata. **This entire model is informational.** It is
    recorded in every experiment manifest for debugging/provenance, but
    `experiment_identity.compute_experiment_identity` never reads it: two
    scientifically identical experiments prepared on different machines
    (different OS, different CPU, even different pandas patch version)
    get the SAME experiment ID. If environment compatibility should
    actually gate an experiment, declare it explicitly and separately via
    `ExperimentSpec.environment_requirements` (identity-relevant precisely
    because it is an explicit declaration, not a detected fact) -- see
    `ml/environment.py`'s module docstring."""

    schema_version: int
    python_version: str
    platform_system: str
    platform_release: str
    architecture: str
    package_versions: Mapping[str, str | None]
    cpu_count: int | None
    captured_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "architecture": self.architecture,
            "package_versions": dict(sorted(self.package_versions.items())),
            "cpu_count": self.cpu_count,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EnvironmentSnapshot:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="EnvironmentSnapshot")
        return cls(
            schema_version=_SCHEMA_VERSION, python_version=str(raw["python_version"]),
            platform_system=str(raw["platform_system"]), platform_release=str(raw["platform_release"]),
            architecture=str(raw["architecture"]),
            package_versions={
                str(k): (None if v is None else str(v))
                for k, v in as_json_dict(raw["package_versions"], field_name="package_versions").items()
            },
            cpu_count=(None if raw.get("cpu_count") is None else int(str(raw["cpu_count"]))),
            captured_at=str(raw["captured_at"]),
        )


# --------------------------------------------------------------------------
# Artifact references / metadata
# --------------------------------------------------------------------------
def _validate_sha256_hex(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise ValueError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    category: ArtifactCategory
    content_hash: str
    size_bytes: int
    created_at: str

    def __post_init__(self) -> None:
        _validate_sha256_hex(self.content_hash, field_name="ArtifactReference.content_hash")
        if self.size_bytes < 0:
            raise ValueError(f"ArtifactReference.size_bytes must be >= 0, got {self.size_bytes}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value, "content_hash": self.content_hash,
            "size_bytes": self.size_bytes, "created_at": self.created_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ArtifactReference:
        return cls(
            category=ArtifactCategory(raw["category"]), content_hash=str(raw["content_hash"]),
            size_bytes=int(str(raw["size_bytes"])), created_at=str(raw["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    model_name: str
    model_version: str
    objective: ObjectiveType
    feature_names: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name, "model_version": self.model_version,
            "objective": self.objective.value, "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ModelArtifactMetadata:
        return cls(
            model_name=str(raw["model_name"]), model_version=str(raw["model_version"]),
            objective=ObjectiveType(raw["objective"]),
            feature_names=tuple(str(n) for n in as_json_list(raw["feature_names"], field_name="feature_names")),
        )


@dataclass(frozen=True, slots=True)
class PredictionArtifactMetadata:
    row_count: int
    split_name: str

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError(f"PredictionArtifactMetadata.row_count must be >= 0, got {self.row_count}")
        if not self.split_name:
            raise ValueError("PredictionArtifactMetadata.split_name must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {"row_count": self.row_count, "split_name": self.split_name}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PredictionArtifactMetadata:
        return cls(row_count=int(str(raw["row_count"])), split_name=str(raw["split_name"]))


@dataclass(frozen=True, slots=True)
class MetricsArtifactMetadata:
    split_name: str
    metric_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.split_name:
            raise ValueError("MetricsArtifactMetadata.split_name must not be empty")
        if not self.metric_names:
            raise ValueError("MetricsArtifactMetadata.metric_names must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {"split_name": self.split_name, "metric_names": list(self.metric_names)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MetricsArtifactMetadata:
        return cls(
            split_name=str(raw["split_name"]),
            metric_names=tuple(str(n) for n in as_json_list(raw["metric_names"], field_name="metric_names")),
        )


# --------------------------------------------------------------------------
# Validation report
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    context: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("ValidationIssue.code must not be empty")
        if not self.message:
            raise ValueError("ValidationIssue.message must not be empty")
        validate_json_primitive_mapping(self.context, field_name="ValidationIssue.context")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity.value, "code": self.code, "message": self.message,
            "context": dict(sorted(self.context.items())),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ValidationIssue:
        return cls(
            severity=ValidationSeverity(raw["severity"]), code=str(raw["code"]), message=str(raw["message"]),
            context=dict(as_json_dict(raw.get("context") or {}, field_name="ValidationIssue.context")),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_version: int
    issues: tuple[ValidationIssue, ...]
    generated_at: str

    @property
    def criticals(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.CRITICAL)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)

    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.INFO)

    @property
    def is_ready(self) -> bool:
        """True if nothing at ERROR or CRITICAL severity is present --
        warnings never block a transition to `ready`."""
        return not self.criticals and not self.errors

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "issues": [i.to_json_dict() for i in self.issues],
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ValidationReport:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="ValidationReport")
        return cls(
            schema_version=_SCHEMA_VERSION,
            issues=tuple(
                ValidationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw["issues"], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


def validate_artifact_hash(value: str, *, field_name: str = "content_hash") -> None:
    """Public helper for other ML modules (`artifacts.py`) to validate a
    referenced content hash's format, raising the ARTIFACT-specific
    exception type rather than a bare `ValueError`."""
    try:
        _validate_sha256_hex(value, field_name=field_name)
    except ValueError as exc:
        raise ArtifactCorruptionError(str(exc), context={"value": value}) from exc


__all__ = [
    "ArtifactCategory",
    "ArtifactReference",
    "CodeRevisionBinding",
    "DatasetBinding",
    "EnvironmentSnapshot",
    "ExperimentStatus",
    "FeatureBinding",
    "JsonPrimitive",
    "LabelBinding",
    "LabelType",
    "MetricsArtifactMetadata",
    "ModelArtifactMetadata",
    "ModelCapabilities",
    "ModelHyperparameters",
    "ObjectiveType",
    "PredictionArtifactMetadata",
    "PreprocessingBinding",
    "SplitBinding",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "is_legal_transition",
    "objective_supports_label_type",
    "validate_artifact_hash",
    "validate_json_primitive_mapping",
]
