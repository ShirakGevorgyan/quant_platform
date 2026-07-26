"""ML core infrastructure and artifact foundation (Milestone 4A).

Deterministic experiment identity, content-addressed artifact storage,
append-only event tracking, pre-run validation, and experiment
preparation -- with NO real predictive model trained anywhere in this
package. See `docs/ml_infrastructure.md` for the full architecture and
`examples/xauusd_ml_experiment.example.json` for a worked configuration.

`ml.testing.ConstantTestModel` (a deliberately fake, clearly-named
test-only model) is intentionally NOT re-exported here -- it stays
scoped to `quant_platform.ml.testing`, so nothing that looks like part
of this package's main public surface could be mistaken for a real
predictive algorithm.
"""

from __future__ import annotations

from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_code_revision_binding, capture_environment_snapshot
from quant_platform.ml.experiment_identity import (
    IDENTITY_SCHEMA_VERSION,
    ExperimentIdentity,
    compute_experiment_identity,
    verify_experiment_identity,
)
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.interfaces import (
    FeatureColumnPolicy,
    FeatureSchema,
    FittedModel,
    ModelDeserializer,
    ModelFactory,
    ModelMetadata,
    ModelSerializer,
    Predictor,
    ProbabilisticPredictor,
    TrainableModel,
)
from quant_platform.ml.manifests import MANIFEST_SCHEMA_VERSION, ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.models import (
    ArtifactCategory,
    ArtifactReference,
    CodeRevisionBinding,
    DatasetBinding,
    EnvironmentSnapshot,
    ExperimentStatus,
    FeatureBinding,
    LabelBinding,
    LabelType,
    ModelCapabilities,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    is_legal_transition,
    objective_supports_label_type,
)
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.reporting import build_report_json, render_report_markdown
from quant_platform.ml.seeds import MAX_SEED, SeedConfiguration, SeedDomain, derive_seed
from quant_platform.ml.tracking import EventRecord, EventType, ExperimentEventStore
from quant_platform.ml.validation import validate_experiment_spec

ML_INFRASTRUCTURE_VERSION = "1.0.0"
"""Version of this package's own identity/artifact/manifest semantics --
recorded informationally alongside `ExperimentSpec`'s own
`schema_version`/`IDENTITY_SCHEMA_VERSION`/`MANIFEST_SCHEMA_VERSION`
fields. Bump on any change to those semantics, independent of
`quant_platform.__version__`."""

__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_SEED",
    "ML_INFRASTRUCTURE_VERSION",
    "ArtifactCategory",
    "ArtifactReference",
    "CodeRevisionBinding",
    "DatasetBinding",
    "EnvironmentSnapshot",
    "EventRecord",
    "EventType",
    "ExperimentEventStore",
    "ExperimentIdentity",
    "ExperimentManifest",
    "ExperimentManifestStore",
    "ExperimentPreparer",
    "ExperimentSpec",
    "ExperimentStatus",
    "FeatureBinding",
    "FeatureColumnPolicy",
    "FeatureSchema",
    "FittedModel",
    "LabelBinding",
    "LabelType",
    "MLArtifactStore",
    "ModelCapabilities",
    "ModelDefinition",
    "ModelDeserializer",
    "ModelFactory",
    "ModelHyperparameters",
    "ModelMetadata",
    "ModelRegistry",
    "ModelSerializer",
    "ObjectiveType",
    "Predictor",
    "PreprocessingBinding",
    "ProbabilisticPredictor",
    "SeedConfiguration",
    "SeedDomain",
    "SplitBinding",
    "TrainableModel",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "build_report_json",
    "capture_code_revision_binding",
    "capture_environment_snapshot",
    "compute_experiment_identity",
    "derive_seed",
    "is_legal_transition",
    "objective_supports_label_type",
    "render_report_markdown",
    "validate_experiment_spec",
    "verify_experiment_identity",
]
