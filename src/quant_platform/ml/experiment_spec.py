"""`ExperimentSpec`: the complete, immutable binding of an experiment to
its exact scientific inputs.

IDENTITY-RELEVANT VS. DESCRIPTIVE FIELDS (read this before changing
anything in this file)
--------------------------------------------------------------------------
`to_identity_payload()` returns ONLY the fields that participate in
`experiment_identity.compute_experiment_identity` -- dataset/feature/label/
split/preprocessing bindings, model name+version, hyperparameters,
objective, seed configuration, code revision, and `environment_requirements`
(an explicit DECLARATION, see below). `to_json_dict()` returns the FULL
record (identity payload plus `primary_metric`, `tags`, `notes`) used for
the durable manifest/spec record -- changing `notes`, `tags`, or
`primary_metric` never changes `to_identity_payload()`'s output and
therefore never changes the experiment ID.

Why `primary_metric` is descriptive, not identity-relevant: which metric a
human chooses to call "primary" for REPORTING/ranking purposes does not
change what the underlying fit computation would produce given identical
features/labels/hyperparameters/seeds -- it is absent from Milestone 4A
Section 6's identity-field list for exactly this reason. This has been
audited (post-4A correctness review) and confirmed still true as of this
milestone: `primary_metric` is read in exactly three places in this
package -- `ExperimentSpec.__post_init__`'s non-empty check,
`to_json_dict()`'s round-trip, and `ml_cli.py`'s config passthrough -- and
NOWHERE does it drive a computation, comparison, selection, threshold,
fold choice, or model/hyperparameter decision. **This decision is
conditional on that remaining true.** The moment any future milestone uses
`primary_metric` to choose between models, hyperparameters, thresholds, or
folds (e.g. "pick the run with the best `primary_metric`"), it MUST move
into `to_identity_payload()` at that point -- two experiments that would
be selected differently because of this field are no longer
scientifically interchangeable, and `IDENTITY_SCHEMA_VERSION` must bump
alongside the change (see `experiment_identity.py`) so old and new IDs
never silently collide. Do not defer that migration; a metric that
silently started affecting outcomes while still being identity-exempt
would silently break reproducibility guarantees this whole module exists
to provide.

Why `environment_requirements` IS identity-relevant (the "unless explicitly
configured" escape hatch from Section 8): by default it is empty, so it
contributes the same neutral value to every experiment's identity payload
regardless of machine -- identical scientific experiments on different
machines still get the same ID. Only when a human explicitly declares a
requirement (e.g. `{"numpy": ">=2.0"}`) does it start to matter, and even
then it is the DECLARED value (the same across machines unless a human
deliberately writes a different declaration on each one, which is exactly
the explicit-configuration case Section 8 describes).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from quant_platform.ml.models import (
    CodeRevisionBinding,
    DatasetBinding,
    FeatureBinding,
    LabelBinding,
    ModelHyperparameters,
    ObjectiveType,
    PreprocessingBinding,
    SplitBinding,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    dataset_binding: DatasetBinding
    feature_binding: FeatureBinding
    label_binding: LabelBinding
    split_binding: SplitBinding
    preprocessing_binding: PreprocessingBinding
    model_name: str
    model_version: str
    hyperparameters: ModelHyperparameters
    objective: ObjectiveType
    seed_configuration: SeedConfiguration
    code_revision_binding: CodeRevisionBinding
    primary_metric: str
    environment_requirements: Mapping[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("ExperimentSpec.model_name must not be empty")
        if not self.model_version:
            raise ValueError("ExperimentSpec.model_version must not be empty")
        if not self.primary_metric:
            raise ValueError("ExperimentSpec.primary_metric must not be empty")
        validate_json_primitive_mapping(self.environment_requirements, field_name="ExperimentSpec.environment_requirements")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("ExperimentSpec.tags must not contain duplicates")
        # Reject incompatible label/objective declarations up front.
        self.label_binding.require_compatible(self.objective)

    def to_identity_payload(self) -> dict[str, object]:
        """ONLY the scientifically-relevant fields -- see module docstring."""
        return {
            "schema_version": self.schema_version,
            "dataset_binding": self.dataset_binding.to_json_dict(),
            "feature_binding": self.feature_binding.to_json_dict(),
            "label_binding": self.label_binding.to_json_dict(),
            "split_binding": self.split_binding.to_json_dict(),
            "preprocessing_binding": self.preprocessing_binding.to_json_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hyperparameters": self.hyperparameters.to_json_dict(),
            "objective": self.objective.value,
            "seed_configuration": self.seed_configuration.to_json_dict(),
            "code_revision_binding": self.code_revision_binding.to_json_dict(),
            "environment_requirements": dict(sorted(self.environment_requirements.items())),
        }

    def to_json_dict(self) -> dict[str, object]:
        """The FULL canonical record -- identity payload plus descriptive
        fields. Used for the durable spec/manifest record, never for
        identity computation."""
        payload = self.to_identity_payload()
        payload["primary_metric"] = self.primary_metric
        payload["tags"] = list(self.tags)
        payload["notes"] = self.notes
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExperimentSpec:
        require_schema_version(raw, supported=SCHEMA_VERSION, context="ExperimentSpec")
        return cls(
            dataset_binding=DatasetBinding.from_json_dict(raw["dataset_binding"]),  # type: ignore[arg-type]
            feature_binding=FeatureBinding.from_json_dict(raw["feature_binding"]),  # type: ignore[arg-type]
            label_binding=LabelBinding.from_json_dict(raw["label_binding"]),  # type: ignore[arg-type]
            split_binding=SplitBinding.from_json_dict(raw["split_binding"]),  # type: ignore[arg-type]
            preprocessing_binding=PreprocessingBinding.from_json_dict(raw["preprocessing_binding"]),  # type: ignore[arg-type]
            model_name=str(raw["model_name"]), model_version=str(raw["model_version"]),
            hyperparameters=ModelHyperparameters.from_json_dict(raw["hyperparameters"]),  # type: ignore[arg-type]
            objective=ObjectiveType(raw["objective"]),
            seed_configuration=SeedConfiguration.from_json_dict(raw["seed_configuration"]),  # type: ignore[arg-type]
            code_revision_binding=CodeRevisionBinding.from_json_dict(raw["code_revision_binding"]),  # type: ignore[arg-type]
            primary_metric=str(raw["primary_metric"]),
            environment_requirements={
                str(k): str(v)
                for k, v in as_json_dict(
                    raw.get("environment_requirements") or {}, field_name="environment_requirements"
                ).items()
            },
            tags=tuple(str(t) for t in as_json_list(raw.get("tags") or [], field_name="tags")),
            notes=str(raw.get("notes", "")),
            schema_version=SCHEMA_VERSION,
        )


__all__ = ["SCHEMA_VERSION", "ExperimentSpec"]
