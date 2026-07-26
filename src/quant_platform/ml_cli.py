"""Command-line interface for the ML core infrastructure and artifact
foundation (Milestone 4A).

    python -m quant_platform.ml_cli list-model-definitions
    python -m quant_platform.ml_cli describe-model-definition --name constant_test_model --version 1
    python -m quant_platform.ml_cli prepare-experiment --config config.json
    python -m quant_platform.ml_cli validate-experiment --config config.json
    python -m quant_platform.ml_cli inspect-experiment --config config.json --experiment-id ID
    python -m quant_platform.ml_cli inspect-experiment-manifest --config config.json --experiment-id ID
    python -m quant_platform.ml_cli verify-artifact --config config.json --content-hash HASH
    python -m quant_platform.ml_cli list-experiment-events --config config.json --experiment-id ID

Same operability conventions as `data_cli`/`feature_cli`: every command
returns 0 on success, non-zero on failure, and prints an actionable
stderr message -- never a raw traceback. `prepare-experiment` and
`validate-experiment` return 2 (not 1) when the experiment/spec itself
is not ready -- distinct from 1, which means the COMMAND itself failed
(bad config, missing dataset, etc).

NO TRAIN/PREDICT COMMANDS
--------------------------------------------------------------------------
There is deliberately no `ml train` or `ml predict` here -- this
milestone prepares and validates experiments; it does not fit models.
The only model ever registered by `build_model_registry()` below is
`ml.testing.ConstantTestModelFactory`, explicitly labeled as a test-only
model, never a real predictive algorithm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from quant_platform.config.ml_schemas import MLExperimentConfig
from quant_platform.core.exceptions import QuantPlatformError
from quant_platform.features.manifests import ResearchDatasetManifest, ResearchManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.environment import capture_code_revision_binding
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.models import (
    DatasetBinding,
    ExperimentStatus,
    FeatureBinding,
    ModelCapabilities,
    ObjectiveType,
    PreprocessingBinding,
    ValidationReport,
)
from quant_platform.ml.persistence import parse_json_strict
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.reporting import build_report_json, render_report_markdown
from quant_platform.ml.testing import TEST_MODEL_NAME, TEST_MODEL_VERSION, ConstantTestModelFactory
from quant_platform.ml.tracking import ExperimentEventStore
from quant_platform.ml.validation import validate_experiment_spec


def build_model_registry() -> ModelRegistry:
    """The only registry this milestone ships -- one TEST-ONLY model
    (`ml.testing.ConstantTestModelFactory`), explicitly labeled as such.
    A future milestone that adds a real model registers it here too;
    nothing about this function's shape needs to change."""
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            name=TEST_MODEL_NAME, version=TEST_MODEL_VERSION,
            description="TEST-ONLY deterministic model (predicts the training label mean/positive-rate "
            "regardless of input) -- NOT a real predictive algorithm. See ml.testing module docstring.",
            capabilities=ModelCapabilities(
                supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION),
                supports_predict_proba=True,
            ),
            factory=ConstantTestModelFactory(), serializer_id="constant_test_model_json_v1",
        )
    )
    return registry


def _load_config(path: Path) -> MLExperimentConfig:
    return MLExperimentConfig.model_validate_json(path.read_text())


def _load_dataset_manifest(config: MLExperimentConfig) -> ResearchDatasetManifest:
    store = ResearchManifestStore(config.dataset.research_storage_root)
    return store.load(config.dataset.dataset_id, config.dataset.manifest_version)


def build_experiment_spec(config: MLExperimentConfig) -> tuple[ExperimentSpec, ResearchDatasetManifest]:
    """Assembles an `ExperimentSpec` from validated config plus whatever
    the referenced research dataset manifest actually records -- the
    dataset's own `feature_versions`/`feature_registry_fingerprint`/
    `preprocessing_definition`/`fitted_preprocessing_fingerprint` are
    used directly, never re-typed into the config (see
    `config.ml_schemas`'s module docstring)."""
    dataset_manifest = _load_dataset_manifest(config)

    dataset_binding = DatasetBinding(
        dataset_id=dataset_manifest.dataset_id, manifest_version=dataset_manifest.version,
        content_id=dataset_manifest.content_id, symbol=dataset_manifest.symbol,
        base_timeframe=dataset_manifest.base_timeframe.value,
        source_historical_dataset_id=dataset_manifest.source_historical_dataset_id,
    )
    feature_names = tuple(config.dataset.feature_names) if config.dataset.feature_names is not None else dataset_manifest.feature_names
    feature_binding = FeatureBinding(
        feature_names=feature_names,
        feature_versions=dict(dataset_manifest.feature_versions),
        feature_registry_fingerprint=dataset_manifest.feature_registry_fingerprint,
    )
    preprocessing_binding = PreprocessingBinding(
        preprocessing_definition=dict(dataset_manifest.preprocessing_definition),
        fitted_preprocessing_fingerprint=dataset_manifest.fitted_preprocessing_fingerprint,
    )
    code_revision_binding = capture_code_revision_binding()

    spec = ExperimentSpec(
        dataset_binding=dataset_binding, feature_binding=feature_binding, label_binding=config.label.build(),
        split_binding=config.split.build(), preprocessing_binding=preprocessing_binding,
        model_name=config.model.name, model_version=config.model.version,
        hyperparameters=config.model.build_hyperparameters(), objective=config.model.build_objective(),
        seed_configuration=config.seeds.build(), code_revision_binding=code_revision_binding,
        primary_metric=config.primary_metric, environment_requirements=dict(config.environment_requirements),
        tags=tuple(config.tags), notes=config.notes,
    )
    return spec, dataset_manifest


def cmd_list_model_definitions(args: argparse.Namespace) -> int:  # noqa: ARG001 -- uniform handler signature
    registry = build_model_registry()
    for definition in registry.list_definitions():
        deprecated = " [DEPRECATED]" if definition.deprecated else ""
        print(f"{definition.qualified_name}{deprecated} -- {definition.description}")
    return 0


def cmd_describe_model_definition(args: argparse.Namespace) -> int:
    registry = build_model_registry()
    definition = registry.get(args.name, args.version)
    for key, value in definition.to_json_dict().items():
        print(f"{key}: {value}")
    print(f"fingerprint: {definition.fingerprint()}")
    return 0


def _build_preparer(config: MLExperimentConfig) -> ExperimentPreparer:
    return ExperimentPreparer(
        ml_artifacts_root=config.ml_artifacts_root, model_registry=build_model_registry(),
        research_manifest_store=ResearchManifestStore(config.dataset.research_storage_root),
    )


def cmd_prepare_experiment(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    spec, _ = build_experiment_spec(config)
    manifest = _build_preparer(config).prepare(spec)
    print(f"experiment_id: {manifest.identity.experiment_id}")
    print(f"status: {manifest.status.value}")
    if manifest.failure_summary:
        print(f"failure_summary: {manifest.failure_summary}")
    return 0 if manifest.status is ExperimentStatus.READY else 2


def cmd_validate_experiment(args: argparse.Namespace) -> int:
    """Dry run: builds the spec and runs preflight validation WITHOUT
    calling `ExperimentPreparer.prepare` -- no manifest, artifact, or
    event is written. Useful for iterating on a config before committing."""
    config = _load_config(Path(args.config))
    spec, dataset_manifest = build_experiment_spec(config)
    registry = build_model_registry()
    report = validate_experiment_spec(
        spec, model_registry=registry, dataset_manifest=dataset_manifest, ml_artifacts_root=config.ml_artifacts_root,
    )
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
    print(f"is_ready: {report.is_ready}")
    return 0 if report.is_ready else 2


def _load_manifest(config: MLExperimentConfig, experiment_id: str) -> ExperimentManifest:
    return ExperimentManifestStore(config.ml_artifacts_root).load(experiment_id)


def cmd_inspect_experiment(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = _load_manifest(config, args.experiment_id)
    validation_report = None
    if manifest.validation_report_reference is not None:
        raw = MLArtifactStore(config.ml_artifacts_root).read_artifact(manifest.validation_report_reference.content_hash)
        validation_report = ValidationReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    if args.format == "json":
        import json

        print(json.dumps(build_report_json(manifest, validation_report=validation_report), indent=2, sort_keys=True))
    else:
        print(render_report_markdown(manifest, validation_report=validation_report))
    return 0


def cmd_inspect_experiment_manifest(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = _load_manifest(config, args.experiment_id)
    for key, value in manifest.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def cmd_verify_artifact(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    store = MLArtifactStore(config.ml_artifacts_root)
    ref = store.artifact_reference(args.content_hash)
    print(f"content_hash: {ref.content_hash}")
    print(f"category: {ref.category.value}")
    print(f"size_bytes: {ref.size_bytes}")
    print(f"created_at: {ref.created_at}")
    print("verified: OK (content hash matches, metadata sidecar consistent)")
    return 0


def cmd_list_experiment_events(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    events = ExperimentEventStore(config.ml_artifacts_root).read_events(args.experiment_id)
    if not events:
        print(f"No events recorded for experiment_id={args.experiment_id!r}", file=sys.stderr)
        return 1
    for event in events:
        print(f"{event.sequence:04d} {event.occurred_at} {event.event_type.value} {dict(sorted(event.details.items()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.ml_cli",
        description="ML core infrastructure and artifact foundation CLI (Milestone 4A) -- prepares and "
        "validates experiments; trains nothing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-model-definitions", help="List every registered model definition.")
    list_parser.set_defaults(handler=cmd_list_model_definitions)

    describe_parser = subparsers.add_parser("describe-model-definition", help="Print one model definition's full metadata.")
    describe_parser.add_argument("--name", required=True)
    describe_parser.add_argument("--version", required=True)
    describe_parser.set_defaults(handler=cmd_describe_model_definition)

    prepare_parser = subparsers.add_parser("prepare-experiment", help="Prepare (validate + create manifest for) an experiment.")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.set_defaults(handler=cmd_prepare_experiment)

    validate_parser = subparsers.add_parser("validate-experiment", help="Dry-run preflight validation; writes nothing.")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(handler=cmd_validate_experiment)

    inspect_parser = subparsers.add_parser("inspect-experiment", help="Print a human-readable (or JSON) experiment report.")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--experiment-id", required=True)
    inspect_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    inspect_parser.set_defaults(handler=cmd_inspect_experiment)

    manifest_parser = subparsers.add_parser("inspect-experiment-manifest", help="Print the raw experiment manifest.")
    manifest_parser.add_argument("--config", required=True)
    manifest_parser.add_argument("--experiment-id", required=True)
    manifest_parser.set_defaults(handler=cmd_inspect_experiment_manifest)

    verify_parser = subparsers.add_parser("verify-artifact", help="Verify a content-addressed artifact's integrity.")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--content-hash", required=True)
    verify_parser.set_defaults(handler=cmd_verify_artifact)

    events_parser = subparsers.add_parser("list-experiment-events", help="Print an experiment's append-only event log.")
    events_parser.add_argument("--config", required=True)
    events_parser.add_argument("--experiment-id", required=True)
    events_parser.set_defaults(handler=cmd_list_experiment_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (QuantPlatformError, ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
