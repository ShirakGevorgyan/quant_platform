"""Structured exception hierarchy for the quant platform.

Every exception carries an optional `context` mapping so callers building
structured logs can attach machine-readable diagnostic fields (symbol,
timeframe, timestamps, offending values) without parsing message strings.
"""

from __future__ import annotations

from typing import Any


class QuantPlatformError(Exception):
    """Base class for every exception raised by this package."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered_context = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({rendered_context})"


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
class DataError(QuantPlatformError):
    """Base class for all data-layer failures."""


class SchemaError(DataError):
    """A DataFrame did not conform to the required canonical OHLCV schema."""


class DataQualityError(DataError):
    """Raised when strict data validation detects a critical integrity issue
    (non-monotonic timestamps, invalid OHLC relationships, null prices)."""


class DataSourceError(DataError):
    """Raised when a concrete `DataSource` fails to load/read underlying data
    (missing file, unreadable format, empty result set)."""


# --------------------------------------------------------------------------
# Engine / execution
# --------------------------------------------------------------------------
class EngineError(QuantPlatformError):
    """Base class for backtest engine runtime failures."""


class LookaheadViolationError(EngineError):
    """Raised by a defensive runtime assertion when point-in-time data access
    would have (or did) expose information from beyond the current instant.

    This should be structurally unreachable given a correct `TimeframeCursor`
    implementation; its presence is a deliberate paranoia check protecting
    the engine's core correctness guarantee, not expected error-path code.
    """


class InsufficientDataError(EngineError):
    """Raised when a strategy or indicator requires more warm-up history
    than is currently available at the point-in-time cursor position."""


# --------------------------------------------------------------------------
# Risk / position sizing
# --------------------------------------------------------------------------
class RiskLimitExceededError(QuantPlatformError):
    """Raised when a computed position size or resulting exposure would
    breach a configured risk limit (max leverage, max position, max risk %)."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class ConfigurationError(QuantPlatformError):
    """Raised when a configuration object fails semantic validation beyond
    what Pydantic's field-level validators already enforce."""


# --------------------------------------------------------------------------
# Validation / walk-forward splitting
# --------------------------------------------------------------------------
class ValidationSplitError(QuantPlatformError):
    """Raised when a walk-forward/purged cross-validation split cannot be
    constructed from the given data (insufficient samples, incompatible
    purge/embargo parameters, non-monotonic index)."""


# --------------------------------------------------------------------------
# Historical data pipeline (Milestone 2)
# --------------------------------------------------------------------------
class HistoricalDataError(QuantPlatformError):
    """Base class for every failure in the historical data ingestion,
    storage, and loading pipeline (`quant_platform.historical`)."""


class TimezoneError(HistoricalDataError):
    """Raised when a timestamp's timezone cannot be established
    unambiguously: a naive timestamp with no explicit source timezone
    configured, an unresolvable DST-ambiguous or nonexistent local time, or
    an attempt to mix tz-aware and tz-naive timestamps in the same series.
    Never silently defaulted -- see `historical.timezones`."""


class MissingDependencyError(HistoricalDataError):
    """Raised when functionality that requires an optional third-party
    package (e.g. the `MetaTrader5` package for the MT5 adapter) is invoked
    but that package is not installed. The package this platform ships is
    importable and testable without any such optional dependency present."""


class SourceError(HistoricalDataError):
    """Raised when a `HistoricalSource` fails to retrieve data: connection
    failure, authentication failure, invalid symbol, unavailable timeframe,
    malformed response, or an empty/partial result the caller did not
    explicitly request."""


class SnapshotError(HistoricalDataError):
    """Raised for raw immutable snapshot store failures: corruption,
    checksum mismatch, schema-version mismatch, an attempt to overwrite an
    existing completed snapshot, or an unsafe/traversal-prone identifier."""


class PathSecurityError(HistoricalDataError):
    """Raised when a symbol, dataset ID, snapshot ID, or other
    caller-influenced identifier would resolve to a filesystem path outside
    the intended storage root (path traversal), or otherwise contains
    characters unsafe to use as a path component."""


class QuarantineError(HistoricalDataError):
    """Raised when strict validation policy rejects data outright rather
    than quarantining or repairing it (see `historical.repair`)."""


class ResamplingError(HistoricalDataError):
    """Raised when leak-free resampling to a derived timeframe cannot be
    performed safely: misaligned source timeframe, a source timeframe not
    evenly dividing the target, or a requested aggregation that would
    require information not yet available at the derived bar's close."""


class ManifestError(HistoricalDataError):
    """Raised when a dataset manifest is missing, malformed, references an
    incompatible schema/checksum, or does not match the data it describes."""


class UpdateConflictError(HistoricalDataError):
    """Raised when an incremental update encounters source bars that
    conflict with already-canonicalized historical bars (a revised OHLCV
    value at a timestamp previously stored) and the configured revision
    policy does not permit silently accepting the revision."""


class DatasetLockError(HistoricalDataError):
    """Raised when a dataset lock (`historical.locking.DatasetLock`) is
    already held by another non-stale process, or a lock file is
    corrupted/unreadable in a way that prevents safe reclamation."""


# --------------------------------------------------------------------------
# Feature engineering / research dataset platform (Milestone 3)
# --------------------------------------------------------------------------
class FeatureError(QuantPlatformError):
    """Base class for every failure in the feature engineering and research
    dataset platform (`quant_platform.features`)."""


class PointInTimeViolationError(FeatureError):
    """Raised when a feature, alignment, or dataset-building operation
    would use (or did use) information that would not actually have been
    available at the timestamp it is being attached to: a future base-
    timeframe candle, an incomplete higher-timeframe bar, a macro value
    before its release timestamp, or any other future-information leak.
    This is the feature-engineering platform's analogue of
    `LookaheadViolationError` -- a defensive, structural guard, not an
    expected error path in correct usage."""


class LabelLeakageError(FeatureError):
    """Raised when label/target information would enter feature
    computation -- e.g. a caller passes a DataFrame already containing
    reserved `label_`/`target_`-prefixed columns into `FeatureEngine.compute`.
    Features and labels are built by entirely separate code paths
    (`quant_platform.features.labels` is never imported by any feature
    module) specifically so this class of bug is structurally difficult to
    introduce; this exception is the last-line runtime guard."""


class DuplicateFeatureError(FeatureError):
    """Raised when `FeatureRegistry.register` is called with a
    (name, version) pair that is already registered. Registrations are
    append-only and never silently overwrite one another."""


class UnknownFeatureError(FeatureError):
    """Raised when a requested feature name/version is not present in a
    `FeatureRegistry`."""


class CyclicFeatureDependencyError(FeatureError):
    """Raised when `FeatureRegistry.resolve_dependency_order` detects a
    cycle among registered features' `feature_dependencies`."""


class FeatureComputationError(FeatureError):
    """Raised when a registered feature's `compute` callable fails, or
    returns output that violates its own declared `FeatureSpec` (wrong
    length, wrong dtype)."""


class PreprocessingLeakageError(FeatureError):
    """Raised when a fitted preprocessing transform (scaler, imputer) would
    be (or was) refit on data outside its designated training partition --
    e.g. calling `.fit()` a second time on an already-fitted
    `TransformPipeline` without an explicit `allow_refit=True` override."""


class ResearchDatasetError(FeatureError):
    """Raised for research dataset manifest/storage failures: a manifest
    referencing content that no longer exists, a corrupted artifact, or an
    attempt to reconstruct a dataset from an incompatible manifest."""


# --------------------------------------------------------------------------
# ML core infrastructure (Milestone 4A)
# --------------------------------------------------------------------------
class MLError(QuantPlatformError):
    """Base class for every failure in the ML core infrastructure and
    artifact foundation (`quant_platform.ml`). This milestone trains no
    real model -- these exceptions govern experiment identity, artifact
    storage, and manifest lifecycle, not model quality."""


class DuplicateModelDefinitionError(MLError):
    """Raised when `ModelRegistry.register` is called with a
    (name, version) pair that is already registered."""


class UnknownModelDefinitionError(MLError):
    """Raised when a requested model name/version is not present in a
    `ModelRegistry`."""


class ModelNotFittedError(MLError):
    """Raised when `predict`/`predict_proba` is called on a `TrainableModel`
    that has not been fit yet -- 'predict before fit' must fail loudly, not
    silently return an arbitrary result."""


class UnsupportedObjectiveError(MLError):
    """Raised when an operation is incompatible with a model's declared
    `ModelCapabilities` or `ObjectiveType` -- e.g. `predict_proba` on a
    regression-only model, or a label type that does not match the
    declared objective."""


class FeatureSchemaMismatchError(MLError):
    """Raised when input feature columns do not match a fitted model's
    recorded feature schema -- missing columns, extra columns (unless an
    explicit policy allows them), or a different column order when order is
    semantically relevant."""


class InvalidSeedError(MLError):
    """Raised when a seed value or `SeedConfiguration` is out of the
    supported range, non-deterministic, or otherwise invalid."""


class ExperimentIdentityError(MLError):
    """Raised when a deterministic experiment identity cannot be computed
    from an `ExperimentSpec` -- e.g. non-canonicalizable fields."""


class ExperimentValidationError(MLError):
    """Raised when an experiment is asked to proceed (e.g. to 'ready') while
    its `ValidationReport` still contains an ERROR- or CRITICAL-severity
    issue. Warnings alone never block a transition."""


class ArtifactCorruptionError(MLError):
    """Raised for ML artifact store failures: a checksum mismatch, a missing
    `_SUCCESS` marker, or malformed sidecar metadata -- mirrors
    `historical.raw_store`/`features.manifests`'s identical concern for the
    ML artifact layer."""


class ArtifactNotFoundError(MLError):
    """Raised when a referenced artifact (by content hash) does not exist
    in the store -- distinguishes 'never existed' from 'exists but
    corrupted' (`ArtifactCorruptionError`)."""


class UntrustedArtifactError(MLError):
    """Raised when a generic artifact-layer caller attempts to deserialize
    an artifact using a mechanism capable of executing arbitrary code
    (e.g. pickle) -- the generic artifact layer never does this itself;
    this is the last-line guard for a caller that tries anyway."""


class ExperimentStateError(MLError):
    """Raised for illegal `ExperimentStatus` transitions (e.g. `completed`
    -> `running`), or an attempt to silently overwrite an already
    `ready`/`completed` experiment manifest with inconsistent content."""


class ExperimentLockError(MLError):
    """Raised when an experiment-preparation OR experiment-execution lock
    is already held by another non-stale process -- see
    `historical.locking.DatasetLock`, reused (via `ml.concurrency.
    experiment_lock`) directly for both purposes, under distinct lock
    files per experiment_id."""


class SchemaVersionError(MLError):
    """Raised when a durable ML JSON structure (spec, manifest, event,
    artifact metadata) declares a schema version this code does not know
    how to read."""


# --------------------------------------------------------------------------
# Time-safe validation and experiment execution engine (Milestone 4B)
# --------------------------------------------------------------------------
class ExecutionError(MLError):
    """Base class for every failure in the experiment execution engine
    (`quant_platform.execution`) -- fold splitting, walk-forward
    execution, resume, and execution-lifecycle bookkeeping. Distinct from
    `MLError`'s other subclasses, which govern Milestone 4A's identity/
    artifact/manifest/preparation concerns rather than actually running
    an experiment's folds."""


class ExecutionStateError(ExecutionError):
    """Raised for illegal `ExecutionStage` transitions (e.g. `completed`
    -> `running_fold`), or an attempt to silently overwrite an already
    `completed`/terminal execution manifest with inconsistent content --
    the execution-stage analogue of `ExperimentStateError`."""


class FoldValidationError(ExecutionError):
    """Raised when a generated or reconstructed fold plan fails a
    structural time-safety check that must never be silently accepted:
    non-chronological or duplicate timestamps, overlapping folds, an
    empty train/test split, or a purge/embargo gap that does not actually
    separate train from test. Reported as a `ValidationIssue` wherever a
    report is expected instead; raised only where there is no reasonable
    way to continue (e.g. reconstructing a dataset's timeline)."""


class ExecutionResumeError(ExecutionError):
    """Raised when a resume attempt cannot proceed safely: a completed
    fold's recorded artifact is missing or corrupted, the requested
    execution has no prior manifest to resume, or an already-terminal
    execution is asked to resume without an explicit force restart."""


# --------------------------------------------------------------------------
# Baseline predictive model framework (Milestone 4C)
# --------------------------------------------------------------------------
class TrainingDataValidationError(ExecutionError):
    """Raised when `ml.model_validation.validate_training_data`'s
    `ValidationReport` for one fold's actual train partition contains an
    ERROR- or CRITICAL-severity issue (missing values a model does not
    support, constant/degenerate labels, a single training sample, etc.)
    -- the per-fold, data-level analogue of `ExperimentValidationError`.
    Raised by `execution.executor`'s production `FoldExecutor`
    immediately before `TrainableModel.fit` would otherwise be called;
    caught by `execution.runner` exactly like any other fold-level
    exception (that fold is marked `FAILED`, the run continues to the
    remaining folds)."""


# --------------------------------------------------------------------------
# Leakage-safe feature selection and hyperparameter optimization (Milestone 4D)
# --------------------------------------------------------------------------
class OptimizationError(QuantPlatformError):
    """Base class for every failure in the leakage-safe feature-selection
    and hyperparameter-optimization engine (`quant_platform.optimization`).
    Distinct from `MLError`/`ExecutionError` -- this milestone governs
    nested (outer/inner) search over an already-`ready`, already-
    executable `ExperimentSpec`, never experiment identity or plain
    walk-forward execution directly."""


class OptimizationIdentityError(OptimizationError):
    """Raised when a deterministic `optimization_id` cannot be computed
    from an `OptimizationSpec` -- the optimization-level analogue of
    `ExperimentIdentityError`."""


class OptimizationStateError(OptimizationError):
    """Raised for illegal `OptimizationStage` transitions, or an attempt
    to silently overwrite an already-terminal `OptimizationManifest` with
    inconsistent content -- the optimization-level analogue of
    `ExecutionStateError`."""


class TrialExecutionError(OptimizationError):
    """Raised when a trial cannot be executed at all due to a structural
    problem (a sampled hyperparameter combination the model wrapper
    rejects outright, a malformed search space, a search-space/sampled-
    value mismatch) -- distinct from a trial that runs to completion but
    is legitimately marked `INVALID`/`FAILED`/`PRUNED` as a normal,
    expected outcome of a bad-but-well-formed candidate."""


class OptimizationResumeError(OptimizationError):
    """Raised when an optimization resume attempt cannot proceed safely:
    a completed trial's recorded artifact is missing or corrupted, the
    requested optimization has no prior manifest to resume, an already-
    terminal optimization is asked to resume, or a resumed TPE sampler's
    replayed trial sequence cannot be reconstructed deterministically."""


class OptimizationVerificationError(OptimizationError):
    """Raised by `optimization.verification.verify_optimization` (or a
    caller consuming its report) when a FATAL cross-consistency check
    fails and the caller has asked for that to raise rather than merely
    be reported as an issue."""
