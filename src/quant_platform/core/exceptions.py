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


# --------------------------------------------------------------------------
# Leakage-safe prediction calibration, thresholding, confidence, and
# uncertainty framework (Milestone 4E)
# --------------------------------------------------------------------------
class CalibrationError(QuantPlatformError):
    """Base class for every failure in the calibration/threshold/
    confidence/uncertainty framework (`quant_platform.calibration`).
    Distinct from `OptimizationError` -- this milestone governs
    post-processing an already-selected, already-refit base model's raw
    outputs, never model/feature/hyperparameter selection itself."""


class CalibrationStateError(CalibrationError):
    """Raised for illegal `CalibrationStage` transitions, or an attempt to
    silently overwrite an already-terminal `CalibrationManifest` with
    inconsistent content -- the calibration-level analogue of
    `OptimizationStateError`."""


class CalibrationValidationError(CalibrationError):
    """Raised when a `CalibrationSpec`/`ThresholdSpec`/`ConfidenceSpec`/
    `UncertaintySpec`/`AbstentionSpec` (or a raw prediction contract) is
    structurally invalid -- an unknown method, an impossible class
    requirement, an invalid probability/threshold/bucket bound, a
    non-finite numeric field, or an inconsistent source identity. Never
    silently repaired."""


class CalibrationDataError(CalibrationError):
    """Raised when raw or inner out-of-fold prediction data violates the
    raw prediction contract: row-count mismatch, timestamp-order
    mismatch, duplicate sample identity, non-finite score, a probability
    outside `[0, 1]`, class-order mismatch, fold mismatch, or source
    identity mismatch."""


class CalibrationFitError(CalibrationError):
    """Raised when a calibration-method candidate cannot be fit at all
    due to a structural problem (insufficient samples, a missing class,
    a malformed input representation) -- distinct from a candidate that
    fits but is legitimately rejected during selection (see
    `CalibrationSelectionError`)."""


class CalibrationSelectionError(CalibrationError):
    """Raised when calibrator selection cannot produce ANY usable
    candidate -- every candidate (including the always-available identity
    baseline) failed or emitted invalid probabilities. Should be rare:
    identity can virtually never legitimately fail."""


class ThresholdSelectionError(CalibrationError):
    """Raised when decision-threshold selection cannot produce a valid
    threshold: no feasible candidate satisfies a declared constraint and
    the spec's fallback policy has no further recourse, or a supplied
    cost matrix is invalid."""


class ConfidencePolicyError(CalibrationError):
    """Raised when a `ConfidenceSpec` cannot be applied to a given
    prediction -- an undefined component, an out-of-range boundary, or a
    category configuration inconsistent with the declared boundaries."""


class UncertaintyPolicyError(CalibrationError):
    """Raised when an `UncertaintySpec` cannot be applied to a given
    prediction -- an undefined component, an inconsistent inner-model
    ensemble (mismatched class ordering, missing model identity), or an
    out-of-range aggregate."""


class CalibrationResumeError(CalibrationError):
    """Raised when a calibration resume attempt cannot proceed safely: a
    completed outer fold's recorded artifact is missing or corrupted, the
    requested calibration has no prior manifest to resume, an already-
    terminal calibration is asked to resume, or the recorded environment
    is incompatible with the current one under a STRICT determinism
    policy."""


class CalibrationVerificationError(CalibrationError):
    """Raised by `calibration.verification.verify_calibration` (or a
    caller consuming its report) when a FATAL cross-consistency check
    fails and the caller has asked for that to raise rather than merely
    be reported as an issue."""


# --------------------------------------------------------------------------
# Leakage-safe financial evaluation, signal simulation, transaction-cost
# modeling, and backtesting framework (Milestone 5)
# --------------------------------------------------------------------------
class BacktestError(QuantPlatformError):
    """Base class for every failure in the financial evaluation/backtesting
    framework (`quant_platform.backtesting`). Distinct from
    `CalibrationError` -- this milestone governs turning already-verified,
    already-calibrated outer-fold predictions into signals, fills, trades,
    returns, and financial metrics under explicitly modeled execution
    assumptions; it never re-selects a model, re-fits a calibrator, or
    re-derives a threshold."""


class BacktestValidationError(BacktestError):
    """Raised when a `BacktestSpec` (or an embedded cost/entry/exit/
    signal-mapping/overlap policy) is structurally invalid -- a non-finite
    numeric field, a negative cost without an explicit rebate policy, an
    impossible price basis, an invalid holding period, an unsupported
    overlap behavior, an incompatible long/short mapping, invalid
    confidence/uncertainty bounds, a missing source identity, an
    inconsistent dataset/calibration binding, non-positive initial
    notional, an invalid timezone, an unsupported bar interval, or an
    entry policy that would fill before prediction availability. Never
    silently repaired."""


class BacktestStateError(BacktestError):
    """Raised for illegal `BacktestStage` transitions, or an attempt to
    silently overwrite an already-terminal `BacktestManifest` with
    inconsistent content -- the backtest-level analogue of
    `CalibrationStateError`."""


class MarketDataBindingError(BacktestError):
    """Raised when market-bar data cannot be bound to a verified
    prediction set safely: a missing bar at a required decision/entry/exit
    timestamp, a non-finite or non-positive price, an OHLC relationship
    violation (`high < max(open, close, low)` or similar), `ask < bid`, a
    negative spread, non-chronological or duplicate timestamps, an
    unexpected bar-interval gap not explicitly permitted, or an
    ambiguous/naive timestamp the market-data contract does not permit."""


class SignalGenerationError(BacktestError):
    """Raised when deterministic signal mapping cannot proceed safely: an
    unsupported predicted class, an invalid probability/confidence/
    uncertainty value, a signal-mapping policy inconsistent with the
    declared position mode, or a reason code that cannot be assigned
    unambiguously."""


class ExecutionSimulationError(BacktestError):
    """Raised when chronological entry/exit simulation encounters a
    structural violation: an entry requested before its prediction was
    available, an exit requested before its entry, a position crossing a
    forbidden outer-fold boundary, or an overlap-policy state that cannot
    be resolved deterministically."""


class FillCalculationError(BacktestError):
    """Raised when a fill price cannot be computed safely: a missing
    observed market price, an invalid side/price-basis combination, a
    spread applied more than once, or a resulting non-finite/non-positive
    effective fill price."""


class CostModelError(BacktestError):
    """Raised when a spread/commission/slippage/financing model is
    invalid or cannot be applied: a non-finite value, a negative cost
    without explicit rebate support, an inconsistent unit convention, or
    an instrument-incompatible cost shape."""


class TradeConstructionError(BacktestError):
    """Raised when a trade record cannot be constructed or is internally
    inconsistent: a non-deterministic or colliding trade identity, an
    exit before an entry, a non-finite return, or a cost breakdown that
    does not sum to the persisted total."""


class FinancialMetricError(BacktestError):
    """Raised when a financial metric cannot be computed safely -- never
    raised merely because a metric is mathematically undefined (that case
    reports an explicit unavailable status/reason instead); raised only
    for a genuine structural problem, e.g. annualizing without a valid bar
    frequency."""


class BacktestResumeError(BacktestError):
    """Raised when a backtest resume attempt cannot proceed safely: a
    completed outer fold's recorded artifact is missing or corrupted, the
    requested backtest has no prior manifest to resume, an already-
    terminal backtest is asked to resume, or the recorded environment is
    incompatible with the current one under a STRICT determinism policy."""


class BacktestVerificationError(BacktestError):
    """Raised by `backtesting.verification.verify_backtest` (or a caller
    consuming its report) when a FATAL cross-consistency check fails and
    the caller has asked for that to raise rather than merely be reported
    as an issue."""


# --------------------------------------------------------------------------
# Leakage-safe statistical robustness, strategy selection, and promotion-gate
# framework (Milestone 6)
# --------------------------------------------------------------------------
class RobustnessError(QuantPlatformError):
    """Base class for every failure in `quant_platform.robustness`. This
    package NEVER refits a model, never re-simulates fills, and never
    claims profitability -- it only statistically characterizes an
    already-verified, already-COMPLETED backtest's own persisted results,
    and decides whether that evidence clears configurable, fail-closed
    promotion gates for paper trading (never live trading)."""


class RobustnessValidationError(RobustnessError):
    """Raised when a `RobustnessSpec` (or an embedded bootstrap/stress/
    regime/promotion policy) is structurally invalid -- a non-finite
    numeric field, an empty perturbation grid, an unsupported bootstrap
    method, a non-positive repetition count, a missing mandatory gate, or
    an internally inconsistent threshold. Never silently repaired."""


class RobustnessStateError(RobustnessError):
    """Raised for illegal `RobustnessStage` transitions, or an attempt to
    silently overwrite an already-terminal `RobustnessManifest` with
    inconsistent content."""


class RobustnessSourceVerificationError(RobustnessError):
    """Raised when the source backtest this robustness run analyzes
    cannot be trusted: `verify_backtest` did not return `is_ready=True`,
    a source identity (dataset/split-plan/instrument) does not match the
    declared `RobustnessSpec`, or a required fold/stitched artifact is
    missing or fails decode-time validation. A robustness run must NEVER
    proceed past this check on an unverified backtest."""


class ReturnSeriesError(RobustnessError):
    """Raised when a requested analysis return series cannot be built
    safely: an unsupported `ReturnSeriesKind`, an attempt to mix bar-level
    and trade-level observations, an empty series, or a source artifact
    that does not match the series' own declared sampling frequency."""


class BootstrapError(RobustnessError):
    """Raised when a dependence-aware bootstrap procedure cannot proceed
    safely: an unsupported method, a block length that does not fit the
    series length, a non-deterministic seed request, or every repetition
    for a statistic failing (leaving no valid resample to report)."""


class MultipleTestingError(RobustnessError):
    """Raised when a multiple-testing correction or a probabilistic/
    deflated Sharpe estimate cannot proceed safely: an empty candidate
    family, a p-value outside `[0, 1]`, or (for deflated Sharpe) missing
    required assumptions this package refuses to silently fabricate."""


class StabilityAnalysisError(RobustnessError):
    """Raised when fold-stability or concentration-risk analysis cannot
    proceed safely: fewer folds than `RobustnessSpec.minimum_fold_count`,
    or a fold-level input inconsistent with the source backtest's own
    persisted fold count."""


class SensitivityAnalysisError(RobustnessError):
    """Raised when parameter/decision-stability (perturbation) analysis
    cannot proceed safely: an undeclared perturbation axis, or an attempt
    to re-optimize rather than merely perturb around the already-selected
    operating point."""


class StressAnalysisError(RobustnessError):
    """Raised when cost/latency/execution stress analysis cannot proceed
    safely: an invalid stress multiplier, or a break-even search whose
    declared bounds are internally inconsistent."""


class RegimeAnalysisError(RobustnessError):
    """Raised when regime-robustness analysis cannot proceed safely: a
    regime definition that would require information not yet available
    at or before the evaluated bar (a leakage violation), or a regime
    requiring source data this platform does not yet persist time-aligned."""


class SelectionError(RobustnessError):
    """Raised when champion/challenger candidate selection cannot proceed
    safely: an empty `StrategyFamily`, a candidate whose robustness
    analysis is not itself COMPLETED/verified, or an ill-defined
    eligibility-gate/ranking configuration."""


class PromotionError(RobustnessError):
    """Raised when promotion-gate evaluation cannot proceed safely: a
    mandatory gate with no measurable input (must fail closed, never be
    silently skipped), or an attempt to construct a `PromotionDecision`
    of `ELIGIBLE_FOR_LIVE_TRADING` (never a decision this milestone may
    output)."""


class RobustnessResumeError(RobustnessError):
    """Raised when a robustness run cannot be legally resumed: no
    manifest exists, the manifest already reached a terminal stage, or a
    provided `RobustnessSpec` does not reproduce the `robustness_id`
    being resumed."""


class RobustnessVerificationError(RobustnessError):
    """Raised by `robustness.verification.verify_robustness` (or a
    caller consuming its report) when a FATAL cross-consistency check
    fails and the caller has asked for that to raise rather than merely
    be reported as an issue."""


# --------------------------------------------------------------------------
# Deterministic paper trading and shadow execution (Milestone 7)
# --------------------------------------------------------------------------
class PaperTradingError(QuantPlatformError):
    """Base class for every failure in `quant_platform.paper_trading`.
    This package NEVER transmits an order to a broker, exchange, MT5
    terminal, or any live-trading API -- it only deterministically
    simulates the trading lifecycle (decisions, orders, fills, positions,
    accounting) for a strategy already promoted `ELIGIBLE_FOR_PAPER_
    TRADING` by Milestone 6. There is no `ELIGIBLE_FOR_LIVE_TRADING` and
    no live-execution code path anywhere in this package."""


class PaperTradingSpecError(PaperTradingError):
    """Raised when a `PaperTradingSpec` (or an embedded instrument/order/
    execution/risk policy) is structurally invalid: a non-finite numeric
    field, an unsupported enum value, a LIVE-like mode string, a broker
    credential/endpoint field, or an internally inconsistent policy.
    Never silently repaired."""


class PaperTradingIdentityError(PaperTradingError):
    """Raised when a `paper_session_spec_id` cannot be computed or a
    provided spec does not reproduce the identity being resumed/verified
    against."""


class PaperTradingEligibilityError(PaperTradingError):
    """Raised when the eligibility chain (verified `PromotionDecision` ->
    verified robustness result -> verified source backtest) cannot be
    established: a missing/tampered decision, a decision kind other than
    `ELIGIBLE_FOR_PAPER_TRADING`, a decision referencing a different
    candidate, or any upstream artifact that fails independent
    re-verification. A paper session must NEVER start without this check
    passing."""


class PaperTradingStateError(PaperTradingError):
    """Raised for illegal `PaperSessionStage` transitions, or an attempt
    to silently overwrite an already-terminal session manifest with
    inconsistent content."""


class PaperTradingManifestError(PaperTradingError):
    """Raised when a `PaperSessionManifest` is structurally invalid or
    references an artifact kind this package does not recognize."""


class PaperTradingArtifactError(PaperTradingError):
    """Raised when a durable paper-trading artifact (ledger record,
    snapshot, report) cannot be read, decoded, or reconstructed safely."""


class PaperTradingVerificationError(PaperTradingError):
    """Raised by `paper_trading.verification.verify_paper_session` (or a
    caller consuming its report) when a FATAL cross-consistency check
    fails and the caller has asked for that to raise rather than merely
    be reported as an issue."""


class MarketEventError(PaperTradingError):
    """Raised when a normalized market event is structurally invalid:
    non-finite/non-positive prices, `ask < bid`, a bar whose high/low do
    not bound open/close, a naive (non-timezone-aware) timestamp, or an
    unsupported interval."""


class MarketEventOrderError(MarketEventError):
    """Raised when a market-event sequence violates strict ordering (a
    forward/live stream may never be silently reordered), contains an
    unresolved duplicate, or otherwise breaks this package's event-time
    correctness guarantee."""


class ClockError(PaperTradingError):
    """Raised for an illegal clock operation: requesting a time before
    the clock has been advanced, stepping a `ReplayClock` past the end of
    its source sequence, or supplying a non-monotonic manual time to a
    clock mode that requires monotonicity."""


class StrategyRuntimeError(PaperTradingError):
    """Raised when a `StrategyRuntime` cannot safely produce a decision:
    a feature/model artifact identity mismatch against the eligibility
    chain, a non-JSON-safe diagnostic value, or an internal strategy
    failure that must halt rather than fabricate a decision."""


class OrderValidationError(PaperTradingError):
    """Raised when an `OrderRequest` is structurally invalid or violates
    a pre-trade policy: non-positive/mis-quantized quantity, missing
    required price for its order type, unsupported order type/time-in-
    force, or a duplicate `client_order_id`."""


class OrderStateError(PaperTradingError):
    """Raised for an illegal `OrderState` transition, or an attempt to
    mutate an order outside its defined lifecycle."""


class FillValidationError(PaperTradingError):
    """Raised when a `Fill` is structurally invalid: non-positive
    quantity, cumulative fill exceeding order quantity, a non-finite cost
    component, a side mismatched with its order, or an attempt to mutate
    an already-persisted fill."""


class PositionAccountingError(PaperTradingError):
    """Raised when position accounting cannot be reconciled: a fill that
    would drive quantity/cost-basis inconsistent with its own recorded
    formula for LONG/SHORT/scale-in/reversal."""


class PortfolioReconciliationError(PaperTradingError):
    """Raised when portfolio/account reconciliation fails: `equity !=
    cash + marked_position_value - liabilities - accrued_costs` (or an
    equivalent exact-arithmetic check) outside tolerance."""


class RiskLimitError(PaperTradingError):
    """Raised when a risk check cannot be evaluated safely (e.g. a
    mandatory limit with no measurable input) -- fails closed, never
    silently skipped."""


class RiskHaltError(PaperTradingError):
    """Raised to signal a risk-driven session halt/flatten action -- the
    kill switch's own typed escalation, never a silent no-op."""


class DuplicateEventError(PaperTradingError):
    """Raised when an event ledger append would introduce a duplicate
    deterministic event identity."""


class DuplicateOrderError(PaperTradingError):
    """Raised when an order create would introduce a duplicate
    `client_order_id` or duplicate deterministic order identity."""


class DuplicateFillError(PaperTradingError):
    """Raised when a fill apply would introduce a duplicate deterministic
    fill identity -- recognized and rejected idempotently, never applied
    twice."""


class SessionLockError(PaperTradingError):
    """Raised when a second writer attempts to run/resume the same
    `paper_session_id` concurrently, or a stale-lock recovery is
    attempted without the required explicit force."""


class ResumeError(PaperTradingError):
    """Raised when a paper session cannot be legally resumed: no manifest
    exists, the manifest already reached a terminal, non-recoverable
    stage, or a provided `PaperTradingSpec` does not reproduce the
    `paper_session_spec_id` being resumed."""
