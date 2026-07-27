"""Pre-training model/data compatibility validation (Milestone 4C,
"MODEL VALIDATION"/"ADVERSARIAL AUDIT" sections) -- does the ACTUAL
feature/label data about to be handed to `TrainableModel.fit` actually
satisfy what the resolved `ml.registry.ModelDefinition` declares it can
handle. Distinct from `ml.validation.validate_experiment_spec` (Milestone
4A): that module checks an `ExperimentSpec` against a model REGISTRY
entry and a research dataset MANIFEST, entirely at the level of declared
metadata, and never touches a single row of actual feature/label data.
This module is the complementary, DATA-level gate that must run
immediately before `fit` is ever called -- the same "every check
contributes a `ValidationIssue`, never raises for a bad but loadable
input" philosophy `ml.validation`/`execution.execution_validation`
already established, reused here rather than re-invented.

WHY THIS RUNS AGAIN EVEN THOUGH `ml.validation` ALREADY CHECKED THE SPEC
--------------------------------------------------------------------------
`validate_experiment_spec` can only ever confirm the DECLARED feature
names/label type/objective are internally consistent with the dataset
MANIFEST's metadata -- it has no access to (and no business inspecting)
the actual per-row values a fold's train partition will contain. A
dataset can pass every 4A/4B check and still hand a MODEL an all-NaN
column, a single training sample, or a fold whose train partition
happens to contain only one label value (see the walk-forward property:
folds are carved from a time series, and a sufficiently short or
unlucky partition can legitimately end up label-degenerate even when the
FULL dataset is not). This module is what catches that, per fold, per
model, right before `fit`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FeatureSchemaMismatchError
from quant_platform.ml.interfaces import ModelMetadata
from quant_platform.ml.models import (
    JsonPrimitive,
    ObjectiveType,
    PreprocessingBinding,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_SCHEMA_VERSION = 1
_MIN_RECOMMENDED_SAMPLES = 10
"""Below this row count a WARNING (never a CRITICAL -- a genuinely tiny
fold is a real, sometimes deliberate, walk-forward configuration choice,
not necessarily an error) is reported."""

_SCALING_TRANSFORM_KINDS = frozenset({"standard_scale", "robust_scale"})
"""The `PreprocessingBinding.preprocessing_definition` value strings that
count as "this feature was scaled to a comparable magnitude" -- matches
`features.normalization.TransformKind.STANDARD_SCALE`/`ROBUST_SCALE`'s
`.value`s exactly, duplicated as plain strings here (rather than importing
`features.normalization`) so `ml.model_validation` stays decoupled from
Milestone 3's transform catalog -- `WINSORIZE`/`SIGNED_LOG1P` are real
preprocessing steps but do not make features comparably scaled."""


def _issue(severity: ValidationSeverity, code: str, message: str, **context: JsonPrimitive) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, context=context)


def _qualified_name(metadata: ModelMetadata) -> str:
    return f"{metadata.name}@{metadata.version}"


def validate_training_data(
    *, metadata: ModelMetadata, features: pd.DataFrame, labels: pd.Series,
    preprocessing_binding: PreprocessingBinding | None = None,
) -> ValidationReport:
    """`metadata` is a freshly-created `TrainableModel.metadata` (or an
    already-`FittedModel.metadata`) -- everything this function needs
    (`capabilities`, `objective`, `feature_schema`) is already embedded
    there, exactly what is actually available inside a `FoldExecutor`
    right after calling `ModelFactory.create(...)`; there is no need to
    separately thread through the resolved `ml.registry.ModelDefinition`
    just to re-derive the same three fields.

    NO SEPARATE `label_type` PARAMETER: `ml.models._COMPATIBLE_LABEL_TYPES`
    maps every `ObjectiveType` to EXACTLY ONE `LabelType` today (a strict
    1:1 relationship) -- so `metadata.objective` alone already determines
    every check this module needs to make about label TYPE. An
    independently-declared `LabelBinding.label_type` disagreeing with
    `objective` is a different, spec-level concern `ml.validation.
    validate_experiment_spec`'s `_validate_label_objective` already
    checks (and `ExperimentSpec.__post_init__` already prevents from
    existing at all); this module only ever sees an objective that has
    already passed that gate.

    `preprocessing_binding`, when omitted (`None`), is treated as "no
    evidence of any preprocessing" -- the same fail-closed default as an
    explicit, empty `PreprocessingBinding()`. A caller that does not care
    about the "REQUIRED PREPROCESSING MUST BE ENFORCED" check at all (most
    of this module's own unit tests, which exercise sample-count/missing-
    value/label checks in isolation) may simply omit it.

    Every check below contributes exactly one or more `ValidationIssue`s
    to the returned report; nothing here raises for bad-but-inspectable
    data. `report.is_ready` is the single authoritative gate a caller
    (`execution.executor`'s production `FoldExecutor`) must check before
    calling `TrainableModel.fit` at all."""
    issues: list[ValidationIssue] = []
    issues += _validate_objective_compatibility(metadata)
    issues += _validate_feature_schema(metadata, features)
    issues += _validate_sample_count(features, labels)
    issues += _validate_missing_values(metadata, features)
    issues += _validate_feature_types(metadata, features)
    issues += _validate_labels(labels, metadata.objective)
    issues += _validate_preprocessing_requirements(metadata, preprocessing_binding)
    return ValidationReport(schema_version=_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def _validate_objective_compatibility(metadata: ModelMetadata) -> list[ValidationIssue]:
    if not metadata.capabilities.supports(metadata.objective):
        return [  # pragma: no cover - ModelMetadata.__post_init__ already guarantees capabilities.supports(objective)
            _issue(
                ValidationSeverity.CRITICAL, "model_objective_incompatible",
                f"Model {_qualified_name(metadata)!r} does not support objective {metadata.objective.value!r} "
                f"(supports: {[o.value for o in metadata.capabilities.supported_objectives]})",
                model_name=metadata.name, objective=metadata.objective.value,
            )
        ]
    return [_issue(ValidationSeverity.INFO, "model_objective_compatible", f"{_qualified_name(metadata)!r} supports objective {metadata.objective.value!r}")]


def _validate_feature_schema(metadata: ModelMetadata, features: pd.DataFrame) -> list[ValidationIssue]:
    feature_schema = metadata.feature_schema
    try:
        feature_schema.validate_frame(features)
    except FeatureSchemaMismatchError as exc:
        return [_issue(ValidationSeverity.CRITICAL, "feature_schema_incompatible", str(exc))]
    return [
        _issue(
            ValidationSeverity.INFO, "feature_schema_compatible",
            f"features frame provides exactly the {len(feature_schema.feature_names)} declared column(s)",
            feature_count=len(feature_schema.feature_names),
        )
    ]


def _validate_sample_count(features: pd.DataFrame, labels: pd.Series) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(features) != len(labels):
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "feature_label_row_count_mismatch",
                f"features ({len(features)} rows) and labels ({len(labels)} rows) have different lengths",
                feature_rows=len(features), label_rows=len(labels),
            )
        )
        return issues
    n = len(features)
    if n == 0:
        issues.append(_issue(ValidationSeverity.CRITICAL, "zero_training_samples", "features/labels contain zero rows -- there is nothing to fit"))
    elif n == 1:
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "single_training_sample",
                "features/labels contain exactly 1 row -- a model cannot be meaningfully fit (and, for "
                "classification, cannot possibly see more than one class)",
                sample_count=1,
            )
        )
    elif n < _MIN_RECOMMENDED_SAMPLES:
        issues.append(
            _issue(
                ValidationSeverity.WARNING, "small_sample_count",
                f"Only {n} training sample(s) -- below the recommended minimum of {_MIN_RECOMMENDED_SAMPLES}; "
                "a fitted model's parameters may be highly unstable",
                sample_count=n,
            )
        )
    else:
        issues.append(_issue(ValidationSeverity.INFO, "sample_count_sufficient", f"{n} training sample(s)", sample_count=n))

    n_features = features.shape[1]
    if n_features > n:
        issues.append(
            _issue(
                ValidationSeverity.WARNING, "high_dimensional_features",
                f"Feature count ({n_features}) exceeds sample count ({n}) -- a high-dimensional (p > n) "
                "regime; some models (e.g. plain least-squares) are ill-posed here, though Elastic Net and "
                "tree ensembles handle it by design",
                feature_count=n_features, sample_count=n,
            )
        )
    return issues


def _validate_missing_values(metadata: ModelMetadata, features: pd.DataFrame) -> list[ValidationIssue]:
    feature_schema = metadata.feature_schema
    if not set(feature_schema.feature_names) <= set(features.columns):
        return []  # already reported by _validate_feature_schema; avoid a confusing duplicate KeyError here
    columns = features[list(feature_schema.feature_names)]
    null_counts = columns.isna().sum()
    columns_with_nulls = [str(name) for name, count in null_counts.items() if count > 0]
    if not columns_with_nulls:
        return [_issue(ValidationSeverity.INFO, "no_missing_values", "No missing (NaN) feature values present")]
    if not metadata.capabilities.supports_missing_values:
        return [
            _issue(
                ValidationSeverity.CRITICAL, "missing_values_unsupported",
                f"Feature column(s) {columns_with_nulls} contain missing (NaN) values, but "
                f"{_qualified_name(metadata)!r} does not declare supports_missing_values=True",
                columns=", ".join(columns_with_nulls),
            )
        ]
    return [
        _issue(
            ValidationSeverity.INFO, "missing_values_supported",
            f"Feature column(s) {columns_with_nulls} contain missing values; "
            f"{_qualified_name(metadata)!r} declares native support for them",
            columns=", ".join(columns_with_nulls),
        )
    ]


def _validate_feature_types(metadata: ModelMetadata, features: pd.DataFrame) -> list[ValidationIssue]:
    feature_schema = metadata.feature_schema
    if not set(feature_schema.feature_names) <= set(features.columns):
        return []
    columns = features[list(feature_schema.feature_names)]
    non_numeric = [str(name) for name, dtype in columns.dtypes.items() if not pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)]
    if not non_numeric:
        return [_issue(ValidationSeverity.INFO, "all_features_numeric", "All declared feature columns are numeric (or boolean)")]
    if not metadata.capabilities.supports_categorical_features:
        return [
            _issue(
                ValidationSeverity.CRITICAL, "categorical_features_unsupported",
                f"Feature column(s) {non_numeric} are non-numeric, but {_qualified_name(metadata)!r} "
                "does not declare supports_categorical_features=True",
                columns=", ".join(non_numeric),
            )
        ]
    return [
        _issue(
            ValidationSeverity.INFO, "categorical_features_supported",
            f"Feature column(s) {non_numeric} are non-numeric; {_qualified_name(metadata)!r} declares "
            "native support for categorical features",
            columns=", ".join(non_numeric),
        )
    ]


def _validate_labels(labels: pd.Series, objective: ObjectiveType) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    non_null = labels.dropna()
    if len(non_null) == 0:
        issues.append(_issue(ValidationSeverity.CRITICAL, "all_labels_missing", "Every label value is missing (NaN) -- there is nothing to fit against"))
        return issues
    if len(non_null) != len(labels):
        issues.append(
            _issue(
                ValidationSeverity.CRITICAL, "labels_contain_missing_values",
                f"{len(labels) - len(non_null)} of {len(labels)} label value(s) are missing (NaN) -- no model "
                "in this milestone's registry supports fitting against a partially-missing target",
                missing_count=len(labels) - len(non_null),
            )
        )

    unique_values = non_null.unique()
    n_unique = len(unique_values)
    if n_unique == 1:
        severity = ValidationSeverity.CRITICAL if objective is not ObjectiveType.REGRESSION else ValidationSeverity.WARNING
        issues.append(
            _issue(
                severity, "constant_labels",
                f"All non-missing label values are identical ({unique_values[0]!r}) -- "
                + (
                    "a classification model cannot be fit against a single observed class"
                    if objective is not ObjectiveType.REGRESSION
                    else "a regression model can technically still be fit, but the result is scientifically "
                    "degenerate (equivalent to a constant predictor)"
                ),
                unique_label_count=1,
            )
        )
    else:
        issues.append(_issue(ValidationSeverity.INFO, "labels_not_constant", f"{n_unique} distinct label value(s) observed", unique_label_count=n_unique))

    if objective is ObjectiveType.BINARY_CLASSIFICATION:
        as_float = np.asarray(non_null, dtype="float64")
        non_binary = sorted({v for v in as_float if v not in (0.0, 1.0)})
        if non_binary:
            issues.append(
                _issue(
                    ValidationSeverity.CRITICAL, "non_binary_label_values",
                    f"Objective is BINARY_CLASSIFICATION but label value(s) {non_binary} outside {{0, 1}} were "
                    "observed",
                    example_values=", ".join(str(v) for v in non_binary[:5]),
                )
            )
        elif n_unique > 1:
            issues.append(_issue(ValidationSeverity.INFO, "binary_labels_valid", "All observed label values are in {0, 1}"))
    return issues


def _validate_preprocessing_requirements(
    metadata: ModelMetadata, preprocessing_binding: PreprocessingBinding | None,
) -> list[ValidationIssue]:
    """"REQUIRED PREPROCESSING MUST BE ENFORCED": a `required_preprocessing`
    free-text string is documentation, not enforcement -- this is the
    enforcement, keyed off the machine-checkable `ModelCapabilities.
    requires_scaled_numeric_features` flag instead.

    FAIL CLOSED, NOT SILENTLY-UNSCALED: a model declaring this flag `True`
    (Logistic Regression, Elastic Net) is REJECTED (CRITICAL) unless
    `preprocessing_binding` is typed evidence that a scaling transform was
    actually FIT (`fitted_preprocessing_fingerprint is not None`, never
    just declared-but-unfit) for EVERY feature this model's schema names.
    A full fold-local preprocessing-refitting framework is out of scope
    for this milestone (see `execution.runner.
    assert_preprocessing_is_safe_for_execution`'s own module docstring):
    that function already fails an entire EXECUTION closed the moment a
    bound dataset shows ANY sign of fitted preprocessing (a global fit
    cannot be proven safe against this engine's own, independently-chosen
    fold boundaries) -- so today, no `PreprocessingBinding` this module
    could ever actually see has a non-`None` fingerprint, and a model
    requiring scaled features can never reach `fit` through the real,
    orchestrated execution path. It remains directly usable
    (`ModelFactory.create().fit(...)`) for unit-level testing of the
    model's OWN fit/predict logic, which never goes through this gate."""
    if not metadata.capabilities.requires_scaled_numeric_features:
        return [
            _issue(
                ValidationSeverity.INFO, "preprocessing_requirement_none",
                f"{_qualified_name(metadata)!r} does not require scaled numeric features",
            )
        ]

    binding = preprocessing_binding if preprocessing_binding is not None else PreprocessingBinding()
    if binding.fitted_preprocessing_fingerprint is not None:
        unscaled = [
            name for name in metadata.feature_schema.feature_names
            if binding.preprocessing_definition.get(name) not in _SCALING_TRANSFORM_KINDS
        ]
        if not unscaled:
            return [
                _issue(
                    ValidationSeverity.INFO, "preprocessing_requirement_satisfied",
                    f"a fitted scaling transform is proven for every feature {_qualified_name(metadata)!r} declares",
                )
            ]
        return [
            _issue(
                ValidationSeverity.CRITICAL, "required_preprocessing_unproven",
                f"{_qualified_name(metadata)!r} requires scaled numeric features, but no proven (fitted) "
                f"scaling transform is declared for feature(s) {unscaled}",
                columns=", ".join(unscaled),
            )
        ]
    return [
        _issue(
            ValidationSeverity.CRITICAL, "required_preprocessing_unproven",
            f"{_qualified_name(metadata)!r} declares capabilities.requires_scaled_numeric_features=True, but "
            "no fitted preprocessing binding proves a safe, train-only scaling transform was applied to its "
            "declared features -- this milestone implements no fold-local preprocessing-refitting framework, "
            "so this model cannot be trained through the real execution engine until one exists",
        )
    ]


__all__ = ["validate_training_data"]
