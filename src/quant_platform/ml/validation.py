"""Pre-run validation: does an `ExperimentSpec` actually describe a
preparable experiment, checked against everything it claims to bind to
-- the model registry and the research dataset subsystem -- before any
manifest is marked `ready`.

WHAT IS CHECKED HERE VS. WHAT `ExperimentSpec.__post_init__` ALREADY
GUARANTEES
--------------------------------------------------------------------------
Some checks Section 13 of this milestone lists (label/objective
compatibility, non-duplicate feature names, non-empty code revision,
well-formed hash fields) are already enforced STRUCTURALLY by the
domain models themselves (`ml/models.py`) at construction time -- an
`ExperimentSpec` that violates them cannot exist as a Python object at
all. This module still reports them (at `INFO` severity, confirmed-not-
violated) for a complete, honest `ValidationReport` rather than silently
omitting checks that "can't fail" -- but the checks that are NOT already
guaranteed structurally are the ones that matter most here, since they
require looking OUTSIDE the spec itself: does the referenced model
definition actually exist and support this objective; does the
referenced research dataset manifest actually contain these exact
features, in this order, under this exact registry fingerprint; does its
recorded `label_definition` actually agree with this experiment's
declared `LabelBinding` (`_validate_label_binding` -- a mismatch here is
possible even when the dataset identity itself matches, since nothing
else confirms a human-declared label agrees with what a `dataset_id`
actually points to); is the declared preprocessing binding consistent
with what the dataset was actually built with.

THIS MODULE DEPENDS ON THE RESEARCH DATASET SUBSYSTEM THROUGH ITS PUBLIC
INTERFACE ONLY
--------------------------------------------------------------------------
`features.manifests.ResearchDatasetManifest` is read but never
constructed, mutated, or re-stored here -- validation is a pure function
of an already-resolved spec, model registry, and dataset manifest, with
no side effects and no duplicated storage logic (Section 1's dependency-
inversion requirement).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from quant_platform.core.exceptions import UnknownModelDefinitionError
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.ml.experiment_identity import ExperimentIdentity, compute_experiment_identity
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifest
from quant_platform.ml.models import (
    ExperimentStatus,
    JsonPrimitive,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    objective_supports_label_type,
)
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.registry import ModelRegistry

_TERMINAL_STATUSES = (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
_SCHEMA_VERSION = 1


def validate_experiment_spec(
    spec: ExperimentSpec,
    *,
    model_registry: ModelRegistry,
    dataset_manifest: ResearchDatasetManifest,
    ml_artifacts_root: Path | str,
    expected_identity: ExperimentIdentity | None = None,
    existing_manifest: ExperimentManifest | None = None,
) -> ValidationReport:
    """Every check contributes exactly one `ValidationIssue`. Nothing here
    raises on a failed check -- failures accumulate as `ERROR`/`CRITICAL`
    issues in the returned report; `report.is_ready` is the single
    authoritative readiness gate (see `models.ValidationReport.is_ready`).
    Only genuinely unusable input (e.g. an empty `ml_artifacts_root`)
    surfaces as a `CRITICAL` issue rather than a raised exception --
    validation never raises for a bad EXPERIMENT, only reports."""
    issues: list[ValidationIssue] = []
    issues += _validate_model_definition(spec, model_registry)
    issues += _validate_dataset_binding(spec, dataset_manifest)
    issues += _validate_feature_binding(spec, dataset_manifest)
    issues += _validate_label_binding(spec, dataset_manifest)
    issues += _validate_label_objective(spec)
    issues += _validate_preprocessing_binding(spec, dataset_manifest)
    issues += _validate_split_binding(spec, dataset_manifest)
    issues += _validate_seeds_and_code_revision(spec)
    issues += _validate_artifact_root(ml_artifacts_root)
    issues += _validate_identity(spec, expected_identity)
    issues += _validate_no_accidental_overwrite(existing_manifest)
    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    """`ValidationIssue.context` is restricted to JSON-primitive values
    (see `models.validate_json_primitive_mapping`) -- callers needing to
    report a list or nested mapping must flatten it to a scalar (e.g. a
    comma-joined string) before passing it here; the full detail always
    belongs in `message`, which has no such restriction."""
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def _join(values: Iterable[object]) -> str:
    return ", ".join(str(v) for v in values)


def _flatten_mapping(mapping: Mapping[str, object]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in sorted(mapping.items(), key=lambda kv: kv[0]))


def _validate_model_definition(spec: ExperimentSpec, model_registry: ModelRegistry) -> list[ValidationIssue]:
    try:
        definition = model_registry.get(spec.model_name, spec.model_version)
    except UnknownModelDefinitionError:
        return [
            _issue(
                ValidationSeverity.CRITICAL, "unknown_model_definition",
                f"No model definition registered for name={spec.model_name!r} version={spec.model_version!r}",
                model_name=spec.model_name, model_version=spec.model_version,
            )
        ]
    if not definition.capabilities.supports(spec.objective):
        return [
            _issue(
                ValidationSeverity.CRITICAL, "model_does_not_support_objective",
                f"Model {definition.qualified_name!r} does not support objective {spec.objective.value!r}",
                model_name=spec.model_name, model_version=spec.model_version, objective=spec.objective.value,
            )
        ]
    return [
        _issue(
            ValidationSeverity.INFO, "model_definition_resolved",
            f"Resolved model definition {definition.qualified_name!r} supporting objective {spec.objective.value!r}",
            model_name=spec.model_name, model_version=spec.model_version,
            model_definition_fingerprint=definition.fingerprint(),
        )
    ]


def _validate_dataset_binding(spec: ExperimentSpec, dataset_manifest: ResearchDatasetManifest) -> list[ValidationIssue]:
    binding = spec.dataset_binding
    mismatches = {
        "dataset_id": (binding.dataset_id, dataset_manifest.dataset_id),
        "manifest_version": (binding.manifest_version, dataset_manifest.version),
        "content_id": (binding.content_id, dataset_manifest.content_id),
    }
    bad = {k: v for k, v in mismatches.items() if v[0] != v[1]}
    if bad:
        detail = "; ".join(f"{k}: expected={v[0]!r} actual={v[1]!r}" for k, v in sorted(bad.items(), key=lambda kv: kv[0]))
        return [
            _issue(
                ValidationSeverity.CRITICAL, "dataset_manifest_mismatch",
                f"DatasetBinding does not match the resolved research dataset manifest: {detail}",
                mismatched_fields=_join(sorted(bad)),
            )
        ]
    return [
        _issue(
            ValidationSeverity.INFO, "dataset_manifest_consistent",
            f"DatasetBinding matches research dataset manifest {dataset_manifest.dataset_id!r} "
            f"version {dataset_manifest.version!r}",
            dataset_id=dataset_manifest.dataset_id, content_id=dataset_manifest.content_id,
        )
    ]


def _validate_feature_binding(spec: ExperimentSpec, dataset_manifest: ResearchDatasetManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    requested = spec.feature_binding.feature_names
    available = dataset_manifest.feature_names
    missing = sorted(set(requested) - set(available))
    if missing:
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "feature_set_mismatch",
                f"Feature(s) {missing} declared in FeatureBinding are not present in the research dataset manifest",
                missing=_join(missing), available=_join(available),
            )
        )
    else:
        available_in_requested_order = tuple(f for f in available if f in set(requested))
        if available_in_requested_order != requested:
            issues.append(
                _issue(
                    ValidationSeverity.ERROR, "feature_order_mismatch",
                    "FeatureBinding.feature_names order does not match the research dataset manifest's column order",
                    requested=_join(requested), available=_join(available),
                )
            )
        else:
            issues.append(
                _issue(
                    ValidationSeverity.INFO, "feature_set_consistent",
                    f"All {len(requested)} requested feature(s) are present in the research dataset manifest in matching order",
                    feature_count=len(requested),
                )
            )

    if spec.feature_binding.feature_registry_fingerprint != dataset_manifest.feature_registry_fingerprint:
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "feature_registry_fingerprint_mismatch",
                "FeatureBinding.feature_registry_fingerprint does not match the research dataset manifest's "
                "recorded fingerprint -- feature definitions may have changed since the dataset was built",
                expected=dataset_manifest.feature_registry_fingerprint, actual=spec.feature_binding.feature_registry_fingerprint,
            )
        )
    else:
        issues.append(
            _issue(ValidationSeverity.INFO, "feature_registry_fingerprint_consistent", "Feature registry fingerprint matches the research dataset manifest")
        )
    return issues


def _validate_label_binding(spec: ExperimentSpec, dataset_manifest: ResearchDatasetManifest) -> list[ValidationIssue]:
    """Cross-validates `ExperimentSpec.label_binding` against the actual,
    immutable `label_definition` recorded in the research dataset manifest
    this experiment's `DatasetBinding` claims to reference -- so an
    experiment cannot reach `ready` believing it targets one label (e.g. a
    20-bar binary direction call) while the referenced dataset was actually
    built against a different one (e.g. a 5-bar future return). A mismatch
    here is possible even when `_validate_dataset_binding` passes: that
    check only confirms `dataset_id`/`manifest_version`/`content_id` agree,
    never that a human-declared `LabelBinding` agrees with what those IDs
    actually point to.

    COMPARISON SEMANTICS (exact, and deliberately conservative -- two
    genuinely different labels must never be silently normalized into
    equality):
      - `name`, `kind`: compared as opaque, case-sensitive strings, byte
        for byte. No trimming, case-folding, or synonym normalization.
      - `horizon_bars`: compared as `int`.
      - `params`: compared as a `dict` via Python `==`. A manifest with no
        `"params"` key at all is treated as an explicit empty dict `{}`,
        consistent with `LabelDefinition.params`/`LabelBinding.params`
        both defaulting to `{}` -- this is NOT missing metadata. This
        inherits the same int/float/bool JSON-equality quirk (e.g.
        `1 == 1.0 == True`) `_validate_preprocessing_binding` above already
        accepts for comparing definition dicts -- a pre-existing,
        documented platform limitation, not something newly introduced by
        this check.
      - `name`/`kind`/`horizon_bars` ARE required (never defaulted) in
        both `features.labels.LabelDefinition` and `LabelBinding`. If the
        manifest's `label_definition` is missing any of them outright (a
        malformed or foreign-schema manifest), that is reported as
        `research_dataset_label_binding_incomplete` (CRITICAL) rather than
        compared, since there is nothing reliable to compare against --
        never a raw `KeyError`/`ValueError`.
      - `LabelBinding.label_type` (the ML-layer-only CONTINUOUS/BINARY/
        MULTICLASS classification) has NO analogous field in
        `ResearchDatasetManifest.label_definition` -- Milestone 3 never
        records it -- so it is deliberately NOT cross-validated here. It
        remains independently governed by `_validate_label_objective`'s
        objective-compatibility check below. A "target column" concept
        similarly does not exist anywhere in `ResearchDatasetManifest`'s
        schema (labels are written under a fixed `"label"` column name by
        `dataset_builder.py`, an implementation detail, never a manifest
        field), so there is nothing to cross-validate there either. A
        separate "label version"/label-definition-fingerprint concept,
        distinct from the fields compared above, does not exist in
        Milestone 3's schema -- `label_definition` is already baked into
        `dataset_id`'s hash at dataset-build time (`features.manifests.
        compute_dataset_id`), so a label-definition change necessarily
        changes `dataset_id`, already independently enforced by
        `_validate_dataset_binding`'s exact-match check above. All three
        omissions are documented decisions, not oversights -- see
        `docs/ml_infrastructure.md`'s "Pre-run validation" section.
    """
    issues: list[ValidationIssue] = []
    manifest_label = dataset_manifest.label_definition
    missing_keys = [k for k in ("name", "kind", "horizon_bars") if manifest_label.get(k) is None]
    if missing_keys:
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "research_dataset_label_binding_incomplete",
                f"Research dataset manifest's label_definition is missing required field(s) "
                f"{missing_keys} -- cannot verify that this experiment's LabelBinding matches the "
                "dataset it claims to bind to",
                missing_fields=_join(missing_keys),
            )
        )

    mismatched: list[str] = []

    if "name" not in missing_keys:
        manifest_name = str(manifest_label.get("name"))
        if spec.label_binding.name != manifest_name:
            mismatched.append("name")
            issues.append(
                _issue(
                    ValidationSeverity.CRITICAL, "label_name_mismatch",
                    f"LabelBinding.name {spec.label_binding.name!r} does not match the research dataset "
                    f"manifest's recorded label name {manifest_name!r}",
                    declared=spec.label_binding.name, actual=manifest_name,
                )
            )

    if "kind" not in missing_keys:
        manifest_kind = str(manifest_label.get("kind"))
        if spec.label_binding.kind != manifest_kind:
            mismatched.append("kind")
            issues.append(
                _issue(
                    ValidationSeverity.CRITICAL, "label_kind_mismatch",
                    f"LabelBinding.kind {spec.label_binding.kind!r} does not match the research dataset "
                    f"manifest's recorded label kind {manifest_kind!r}",
                    declared=spec.label_binding.kind, actual=manifest_kind,
                )
            )

    if "horizon_bars" not in missing_keys:
        manifest_horizon_raw = manifest_label.get("horizon_bars")
        try:
            manifest_horizon = int(str(manifest_horizon_raw))
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    ValidationSeverity.CRITICAL, "research_dataset_label_binding_incomplete",
                    f"Research dataset manifest's label_definition.horizon_bars is not a valid integer: "
                    f"{manifest_horizon_raw!r}",
                    horizon_bars=str(manifest_horizon_raw),
                )
            )
        else:
            if spec.label_binding.horizon_bars != manifest_horizon:
                mismatched.append("horizon_bars")
                issues.append(
                    _issue(
                        ValidationSeverity.CRITICAL, "label_horizon_mismatch",
                        f"LabelBinding.horizon_bars {spec.label_binding.horizon_bars} does not match the "
                        f"research dataset manifest's recorded horizon_bars {manifest_horizon}",
                        declared=spec.label_binding.horizon_bars, actual=manifest_horizon,
                    )
                )

    manifest_params_raw = manifest_label.get("params")
    manifest_params: dict[str, object] | None
    if manifest_params_raw is None:
        manifest_params = {}
    elif isinstance(manifest_params_raw, dict):
        manifest_params = manifest_params_raw
    else:
        manifest_params = None
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "research_dataset_label_binding_incomplete",
                f"Research dataset manifest's label_definition.params is not a JSON object: {manifest_params_raw!r}",
            )
        )
    if manifest_params is not None:
        declared_params = dict(spec.label_binding.params)
        if declared_params != manifest_params:
            mismatched.append("params")
            issues.append(
                _issue(
                    ValidationSeverity.CRITICAL, "label_params_mismatch",
                    "LabelBinding.params does not match the research dataset manifest's recorded label params",
                    declared=_flatten_mapping(declared_params), actual=_flatten_mapping(manifest_params),
                )
            )

    if mismatched:
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "research_dataset_label_binding_mismatch",
                f"LabelBinding disagrees with the research dataset manifest's recorded label_definition "
                f"in field(s) {mismatched} -- this experiment would train/evaluate against a different "
                "label than the one actually present in the referenced dataset",
                mismatched_fields=_join(mismatched),
            )
        )
    elif not missing_keys:
        issues.append(
            _issue(
                ValidationSeverity.INFO, "research_dataset_label_binding_consistent",
                f"LabelBinding matches the research dataset manifest's recorded label_definition "
                f"(name={spec.label_binding.name!r}, kind={spec.label_binding.kind!r}, "
                f"horizon_bars={spec.label_binding.horizon_bars})",
            )
        )
    return issues


def _validate_label_objective(spec: ExperimentSpec) -> list[ValidationIssue]:
    # Structurally guaranteed by `ExperimentSpec.__post_init__` (via
    # `LabelBinding.require_compatible`) -- unreachable in practice, but
    # reported explicitly rather than silently assumed.
    if objective_supports_label_type(spec.objective, spec.label_binding.label_type):
        return [
            _issue(
                ValidationSeverity.INFO, "label_objective_compatible",
                f"Label type {spec.label_binding.label_type.value!r} is compatible with objective {spec.objective.value!r}",
            )
        ]
    return [  # pragma: no cover - unreachable given ExperimentSpec's own construction-time guarantee
        _issue(
            ValidationSeverity.CRITICAL, "label_objective_incompatible",
            f"Label type {spec.label_binding.label_type.value!r} is not compatible with objective {spec.objective.value!r}",
        )
    ]


def _validate_preprocessing_binding(spec: ExperimentSpec, dataset_manifest: ResearchDatasetManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    declared = dict(spec.preprocessing_binding.preprocessing_definition)
    actual = dict(dataset_manifest.preprocessing_definition)
    if declared != actual:
        issues.append(
            _issue(
                ValidationSeverity.ERROR, "preprocessing_definition_mismatch",
                "PreprocessingBinding.preprocessing_definition does not match the research dataset manifest's "
                "recorded preprocessing definition",
                declared=_flatten_mapping(declared), actual=_flatten_mapping(actual),
            )
        )
    else:
        issues.append(
            _issue(ValidationSeverity.INFO, "preprocessing_definition_consistent", "Preprocessing definition matches the research dataset manifest")
        )

    declared_fp = spec.preprocessing_binding.fitted_preprocessing_fingerprint
    actual_fp = dataset_manifest.fitted_preprocessing_fingerprint
    if declared_fp is not None and actual_fp is not None and declared_fp != actual_fp:
        issues.append(
            _issue(
                ValidationSeverity.ERROR, "fitted_preprocessing_fingerprint_mismatch",
                "PreprocessingBinding.fitted_preprocessing_fingerprint does not match the research dataset "
                "manifest's recorded fitted preprocessing fingerprint",
                expected=actual_fp, actual=declared_fp,
            )
        )
    elif (declared_fp is None) != (actual_fp is None):
        issues.append(
            _issue(
                ValidationSeverity.WARNING, "fitted_preprocessing_presence_mismatch",
                "PreprocessingBinding declares a fitted preprocessing fingerprint that the research dataset "
                "manifest does not (or vice versa)",
                declared_present=declared_fp is not None, actual_present=actual_fp is not None,
            )
        )
    return issues


def _validate_split_binding(spec: ExperimentSpec, dataset_manifest: ResearchDatasetManifest) -> list[ValidationIssue]:
    manifest_strategy = dataset_manifest.split_definition.get("strategy")
    if manifest_strategy is not None and manifest_strategy != spec.split_binding.strategy:
        return [
            _issue(
                ValidationSeverity.WARNING, "split_strategy_mismatch",
                f"SplitBinding.strategy {spec.split_binding.strategy!r} does not match the research dataset "
                f"manifest's recorded split strategy {manifest_strategy!r}",
                declared=spec.split_binding.strategy, actual=str(manifest_strategy),
            )
        ]
    return [_issue(ValidationSeverity.INFO, "split_binding_present", f"SplitBinding.strategy={spec.split_binding.strategy!r}")]


def _validate_seeds_and_code_revision(spec: ExperimentSpec) -> list[ValidationIssue]:
    # `SeedConfiguration` and `CodeRevisionBinding`/`FeatureBinding` already
    # validate their own invariants (seed range; non-empty revision;
    # no duplicate feature names) at construction -- reported here for a
    # complete, explicit audit trail, not because they could fail now.
    return [
        _issue(ValidationSeverity.INFO, "seed_configuration_valid", f"master_seed={spec.seed_configuration.master_seed}"),
        _issue(
            ValidationSeverity.INFO, "code_revision_present",
            f"code_revision source={spec.code_revision_binding.source!r} is_dirty={spec.code_revision_binding.is_dirty}",
        ),
        _issue(ValidationSeverity.INFO, "feature_names_unique", "FeatureBinding.feature_names contains no duplicates (enforced at construction)"),
    ]


def _validate_artifact_root(ml_artifacts_root: Path | str) -> list[ValidationIssue]:
    text = str(ml_artifacts_root).strip()
    if not text:
        return [_issue(ValidationSeverity.CRITICAL, "invalid_artifact_root", "ml_artifacts_root must not be empty")]
    resolved = Path(ml_artifacts_root).resolve()
    return [_issue(ValidationSeverity.INFO, "artifact_root_resolved", f"ML artifact output root resolves to {resolved}", path=str(resolved))]


def _validate_identity(spec: ExperimentSpec, expected_identity: ExperimentIdentity | None) -> list[ValidationIssue]:
    computed = compute_experiment_identity(spec)
    if expected_identity is not None and computed != expected_identity:
        return [
            _issue(
                ValidationSeverity.CRITICAL, "identity_mismatch",
                f"Computed experiment identity {computed.experiment_id!r} does not match the expected "
                f"identity {expected_identity.experiment_id!r}",
                computed=computed.experiment_id, expected=expected_identity.experiment_id,
            )
        ]
    return [_issue(ValidationSeverity.INFO, "identity_computed", f"experiment_id={computed.experiment_id}", experiment_id=computed.experiment_id)]


def _validate_no_accidental_overwrite(existing_manifest: ExperimentManifest | None) -> list[ValidationIssue]:
    if existing_manifest is None:
        return []
    if existing_manifest.status in _TERMINAL_STATUSES:
        return [
            _issue(
                ValidationSeverity.INFO, "experiment_already_finalized",
                f"An experiment manifest with this identity already exists in terminal status "
                f"{existing_manifest.status.value!r} -- preparing an identical spec is idempotent, not an overwrite",
                status=existing_manifest.status.value,
            )
        ]
    return [
        _issue(
            ValidationSeverity.INFO, "experiment_already_in_progress",
            f"An experiment manifest with this identity already exists with status {existing_manifest.status.value!r}",
            status=existing_manifest.status.value,
        )
    ]


__all__ = ["validate_experiment_spec"]
