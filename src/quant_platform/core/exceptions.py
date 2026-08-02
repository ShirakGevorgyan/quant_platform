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


# --------------------------------------------------------------------------
# Broker-neutral deterministic execution gateway (Milestone 8)
# --------------------------------------------------------------------------
class ExecutionGatewayError(QuantPlatformError):
    """Base class for every failure in `quant_platform.execution_gateway`.
    This package is TEST-ONLY: it never transmits an order to a broker,
    exchange, MT5 terminal, or any live-trading API. Every execution in
    this package is dispatched to a deterministic, in-process dummy
    broker adapter only -- there is no network client, no broker
    credential field, no MT5/FxPro/real-broker adapter, and no `LIVE`
    execution mode anywhere in this package."""


class ExecutionGatewaySpecError(ExecutionGatewayError):
    """Raised when an `ExecutionGatewaySpec` (or an embedded policy) is
    structurally invalid: a non-finite/non-positive numeric field, an
    unsupported enum value, a live/non-dummy adapter value, or a
    cross-artifact identity mismatch. Never silently repaired."""


class ExecutionGatewayIdentityError(ExecutionGatewayError):
    """Raised when an `execution_gateway_spec_id`/`execution_session_id`
    cannot be computed, or a provided spec does not reproduce the
    identity being resumed/verified against."""


class ExecutionGatewayEligibilityError(ExecutionGatewayError):
    """Raised when the source Milestone 7 paper session's eligibility
    chain, verified session report, or execution authorization cannot be
    established. An execution session must NEVER start or resume without
    this check passing."""


class ExecutionGatewayStateError(ExecutionGatewayError):
    """Raised for an illegal `ExecutionSessionStage` transition, or an
    attempt to mutate an already-terminal execution session manifest."""


class ExecutionGatewayManifestError(ExecutionGatewayError):
    """Raised when an `ExecutionSessionManifest` is structurally invalid
    or references an artifact kind this package does not recognize."""


class ExecutionGatewayArtifactError(ExecutionGatewayError):
    """Raised when a durable execution-gateway artifact (ledger record,
    snapshot, report) cannot be read, decoded, or reconstructed safely."""


class ExecutionGatewayVerificationError(ExecutionGatewayError):
    """Raised by `execution_gateway.verification.verify_execution_session`
    (or a caller consuming its report) when a FATAL cross-consistency
    check fails and the caller has asked for that to raise rather than
    merely be reported as an issue."""


class ExecutionIntentError(ExecutionGatewayError):
    """Raised when an `ExecutionIntent` is structurally invalid, or when
    the paper-session bridge cannot establish that an intent's economic
    fields genuinely match its declared source paper order."""


class ExecutionCommandError(ExecutionGatewayError):
    """Base class for command-model failures."""


class ExecutionCommandValidationError(ExecutionCommandError):
    """Raised when a command is structurally invalid or violates a
    pre-dispatch policy: a missing required price, an unsupported
    order/time-in-force combination, or an inconsistent close/reduce-only
    combination."""


class ExecutionCommandDispatchError(ExecutionCommandError):
    """Raised when a command definitely could not be dispatched (a
    capability check failed closed before any adapter call, or the
    adapter synchronously and unambiguously refused the call)."""


class ExecutionCommandAmbiguousError(ExecutionCommandError):
    """Raised to signal that whether a command was accepted by the broker
    is genuinely UNKNOWN -- callers must resolve via query/reconciliation,
    never by blindly retrying a non-idempotent operation."""


class ExecutionOrderStateError(ExecutionGatewayError):
    """Raised for an illegal `ExecutionOrderState` transition, or an
    attempt to reconstruct an aggregate from a durable event history that
    does not chain legally."""


class ExecutionFillError(ExecutionGatewayError):
    """Raised when an `ExecutionFill` is structurally invalid: non-positive
    quantity/price/contract multiplier, a gross-notional mismatch, or an
    attempt to double-count an already-recorded broker fill."""


class BrokerEventError(ExecutionGatewayError):
    """Raised when a normalized broker event is structurally invalid or
    cannot be classified against the order/command it claims to belong
    to."""


class BrokerEventSequenceError(BrokerEventError):
    """Raised for a broker-sequence violation this package's active
    `SequencingPolicy` classifies as CRITICAL rather than an ordinary,
    recoverable gap/out-of-order condition: a conflicting event reusing
    an already-used sequence number, or a same-ID event with a changed
    payload."""


class BrokerCapabilityError(ExecutionGatewayError):
    """Raised when a command would require an `AdapterCapabilities` flag
    the active adapter does not declare -- capability checks fail closed
    before dispatch, never silently downgrading requested semantics."""


class BrokerSnapshotError(ExecutionGatewayError):
    """Raised when a broker order/position/account snapshot is
    structurally invalid or cannot be attributed to the requesting
    execution session."""


class ExecutionIdempotencyError(ExecutionGatewayError):
    """Raised when the same `command_id`/`client_order_id`/
    `broker_event_id` is presented with a payload that differs from the
    durable evidence already on record -- never silently overwritten or
    merged."""


class ExecutionRecoveryError(ExecutionGatewayError):
    """Raised when deterministic crash recovery cannot safely reconstruct
    execution-session state from the spec, manifest, ledger, and broker
    snapshots -- surfaced rather than guessed."""


class ExecutionReconciliationError(ExecutionGatewayError):
    """Raised when internal/broker/ledger reconciliation cannot complete
    structurally (as opposed to an ordinary `ReconciliationIssue`, which
    is a normal, expected outcome even for corrupted state)."""


class ExecutionHealthError(ExecutionGatewayError):
    """Raised when adapter health cannot be evaluated safely, or a
    command is attempted while health policy forbids it."""


class ExecutionHaltError(ExecutionGatewayError):
    """Raised to signal a kill-switch-driven dispatch refusal -- the
    execution gateway's own typed escalation, never a silent no-op."""


class ExecutionSessionLockError(ExecutionGatewayError):
    """Raised when a second writer attempts to run/resume the same
    `execution_session_id` concurrently, or two workers race to dispatch
    the same pending command."""


class ExecutionPortfolioRiskAuthorizationError(ExecutionGatewayError):
    """Raised to signal a Milestone-9-portfolio-risk-driven dispatch
    refusal (Milestone 9 Phase 4) -- the execution gateway's own typed
    escalation for this refusal reason, mirroring `ExecutionHaltError`'s
    identical role for a kill-switch refusal. Raised by `execution_
    gateway.portfolio_risk_gate` whenever a `RiskDecision` is not
    `APPROVED`, a `RiskAuthorization` cannot be reserved/consumed for the
    exact execution intent attempting to dispatch, or any other
    portfolio-risk-side rejection occurs -- always chaining the
    underlying `portfolio_risk` exception via `from exc` so no diagnostic
    detail is lost, while still giving `runner.py`'s own caller ONE
    uniform exception type to catch for every portfolio-risk dispatch
    refusal, regardless of which specific `portfolio_risk` exception
    produced it. Never a silent no-op -- exactly like `ExecutionHaltError`,
    this is the fail-closed escalation, not a bypassable warning."""


# --------------------------------------------------------------------------
# Portfolio risk and capital management engine (Milestone 9)
# --------------------------------------------------------------------------
class PortfolioRiskError(QuantPlatformError):
    """Base class for every failure in `quant_platform.portfolio_risk`.
    This package is TEST-ONLY: it never opens a network connection, never
    imports a broker SDK, never defines a credential field, and never
    claims profitability, broker readiness, or live-trading readiness.
    Every risk authorization it produces gates deterministic, in-process
    dispatch decisions only -- it never itself transmits an order. Unknown
    or incomplete risk state always fails CLOSED (denied/halted), never
    silently approved."""


class PortfolioRiskPolicyError(PortfolioRiskError):
    """Raised when a `PortfolioRiskPolicy` or `PortfolioRiskSpec` is
    structurally invalid: a non-finite/non-positive limit, a fraction
    outside its legal range, or a malformed low-level Decimal/identity
    primitive shared across this package's models."""


class PortfolioRiskSpecIdentityError(PortfolioRiskError):
    """Raised when a `portfolio_risk_spec_id` cannot be computed, or a
    provided spec does not reproduce the identity it is being verified
    against."""


class PortfolioSnapshotValidationError(PortfolioRiskError):
    """Raised when a `PortfolioSnapshot` (or an embedded `PositionSnapshot`)
    is structurally invalid or internally incoherent: a duplicate
    instrument/strategy position identity, a negative gross exposure, an
    equity figure that does not reconcile with cash and position
    valuation under the documented accounting model, or a non-Decimal/
    non-finite monetary or quantity field."""


class StalePortfolioSnapshotError(PortfolioRiskError):
    """Raised when a `PortfolioSnapshot` is older, relative to a caller-
    supplied reference time, than `PortfolioRiskPolicy.
    maximum_portfolio_snapshot_age` permits. Staleness is always judged
    against an explicit caller-supplied reference time -- never an
    internal wall-clock read."""


class StalePriceError(PortfolioRiskError):
    """Raised when a `PriceSnapshot` is older, relative to a caller-
    supplied reference time, than `PortfolioRiskPolicy.maximum_price_age`
    permits, or when a `PriceSnapshot` is structurally invalid (bid > ask,
    a non-positive price)."""


class ExposureCalculationError(PortfolioRiskError):
    """Raised when instrument/strategy/portfolio gross or net exposure
    cannot be derived safely from a `PortfolioSnapshot` -- e.g. a position
    referencing a price/contract multiplier that is missing or
    non-finite. Exposure is always DERIVED, never independently trusted
    from a stored figure."""


class PositionSizingError(PortfolioRiskError):
    """Raised when a `PositionSizeProposal` or `CapitalAllocation` is
    structurally invalid: a non-positive proposed quantity/reference
    price, or utilized capital exceeding allocated capital."""


class RiskEvaluationError(PortfolioRiskError):
    """Raised when a `RiskEvaluationRequest` cannot be safely evaluated at
    all -- as opposed to an ordinary `RiskDecision` with kind=DENIED,
    which is ALWAYS a valid, structurally sound outcome, never this
    exception. Incomplete or contradictory evaluation inputs fail closed
    via this exception, never a silent APPROVED."""


class RiskDenialError(PortfolioRiskError):
    """Raised by a caller that asked a DENIED/HALTED `RiskDecision` to
    raise rather than merely be returned/reported -- the decision object
    itself never raises on construction; denial is always a normal,
    valid, structurally complete outcome."""


class RiskAuthorizationIdentityError(PortfolioRiskError):
    """Raised when a `risk_authorization_id` cannot be computed, or a
    provided `RiskAuthorization` does not reproduce the identity it is
    being verified against."""


class RiskAuthorizationMismatchError(PortfolioRiskError):
    """Raised when a `RiskAuthorization` is checked against an execution
    intent, session, portfolio snapshot, price snapshot, policy, quantity,
    or price it was not actually issued for. A `RiskAuthorization` binds
    to the EXACT tuple of fields captured in its own content identity --
    it is never valid for a different one, even one that differs in only
    a single field."""


class RiskAuthorizationReuseError(PortfolioRiskError):
    """Raised when a `RiskAuthorization` already `RESERVED`/`CONSUMED` (per
    the durable, ledger-derived single-use lifecycle) is presented again
    to gate a conflicting second economic use -- an exact, identical
    retry of the SAME use is idempotently absorbed instead (never raises);
    only a genuine CONFLICT (a different `consumption_identity`, or use
    after a terminal status) raises this."""


class PortfolioHaltError(PortfolioRiskError):
    """Raised to signal a portfolio-level halt -- the risk engine's own
    typed escalation for a `RiskDecisionKind.HALTED` outcome, never a
    silent no-op."""


class PortfolioRiskReconciliationError(PortfolioRiskError):
    """Raised when portfolio-risk-side reconciliation cannot complete
    structurally (e.g. the ledger itself cannot be reconstructed), as
    opposed to an ordinary `ReconciliationIssue` finding, which is a
    normal, expected, non-raising outcome."""


class PortfolioRiskVerificationError(PortfolioRiskError):
    """Raised by `verification.verify_portfolio_risk_session` (or a
    caller consuming its report) when a FATAL cross-consistency check
    fails and the caller has asked for that to raise rather than merely
    be reported as an issue."""


class PortfolioRiskPersistenceError(PortfolioRiskError):
    """Raised when a durable portfolio-risk artifact (a risk ledger
    entry, a chain-integrity check, a report) cannot be read, decoded, or
    reconstructed safely."""


class PortfolioRiskRecoveryError(PortfolioRiskError):
    """Raised when crash recovery cannot safely reconstruct
    portfolio-risk authorization lifecycle state from durable ledger
    evidence -- surfaced rather than guessed. Never used to justify a
    blind reuse of an ambiguous authorization."""


class PortfolioRiskLockError(PortfolioRiskError):
    """Raised when a second writer attempts to append to, or mutate the
    lifecycle of, the same `portfolio_id`'s risk ledger concurrently --
    mirrors `execution_gateway`'s own `ExecutionSessionLockError`
    exactly, reusing the same underlying `ml.concurrency.experiment_lock`
    infrastructure."""


# --------------------------------------------------------------------------
# Deterministic market data platform and feature store (Milestone 10)
# --------------------------------------------------------------------------
class MarketDataError(QuantPlatformError):
    """Base class for every failure in `quant_platform.market_data`. This
    package is the single authoritative source for market, macro,
    calendar, and derived feature data consumed by research, ML,
    backtesting, portfolio risk, execution, and replay -- it never opens
    a network connection, never imports a broker SDK, and never claims
    live-feed connectivity. Every event and feature it produces is
    immutable and content-addressed; the same input data always produces
    identical output, or this package raises rather than silently
    diverging."""


class MarketDataEventError(MarketDataError):
    """Raised when a `Tick`/`Quote`/`Trade`/`Candle` is structurally
    invalid: a non-finite/non-positive price, a negative volume, `ask <
    bid`, an OHLC relationship violation, a naive (non-timezone-aware)
    timestamp, `arrival_time` before `event_time`, or a `timeframe` that
    is present where not applicable (tick/quote/trade) or absent where
    required (candle)."""


class MarketDataOrderError(MarketDataError):
    """Raised when a sequence of market data events violates strict
    ordering: a non-monotonic `event_time`/`sequence`, a future
    timestamp relative to a caller-supplied reference time, or an
    unresolved duplicate id -- the market-data analogue of
    `MarketEventOrderError`."""


class MarketDataIdentityError(MarketDataError):
    """Raised when an `event_id`/`feature_id` cannot be computed, or a
    provided object does not reproduce the identity it is being verified
    against -- a forged or tampered record."""


class MarketDataPersistenceError(MarketDataError):
    """Raised for durable market-data/feature store failures: corruption,
    a sequence gap, a conflicting append at an already-occupied
    coordinate, or a malformed stored record."""


class MarketDataLockError(MarketDataError):
    """Raised when a second writer attempts to append to the same event
    or feature store partition concurrently -- mirrors
    `PortfolioRiskLockError` exactly, reusing the same underlying
    `ml.concurrency.experiment_lock` infrastructure."""


class MarketCalendarError(MarketDataError):
    """Raised when a market calendar specification or session-expectation
    query is structurally invalid or cannot be evaluated safely."""


class MacroDataError(MarketDataError):
    """Raised when a macro/economic-calendar event is structurally
    invalid, or is used before its own release (`event_time`) in a
    context that requires point-in-time safety."""


class MarketDataQualityError(MarketDataError):
    """Raised by a caller that asks a quality report containing a
    CRITICAL issue to fail closed rather than merely be reported --
    never raised by report generation itself, which always returns a
    complete report regardless of how many issues it found."""


class FeatureStoreError(MarketDataError):
    """Raised when a `FeatureRecord` is structurally invalid, or an
    append would silently overwrite an already-stored value at the same
    (feature_name, feature_version, instrument, timeframe, timestamp)
    coordinate with a DIFFERENT value -- feature history is append-only
    and never mutated; an identical re-append is idempotently absorbed
    instead of raising."""


class FeatureIdentityError(MarketDataError):
    """Raised when a `feature_id` cannot be computed, or a provided
    `FeatureRecord` does not reproduce the identity it is being verified
    against."""


class FeatureGenerationError(MarketDataError):
    """Raised when a deterministic feature computation cannot proceed
    safely: mismatched input lengths, a non-positive window, or a result
    that would be non-finite -- never raised merely because a window's
    warm-up period has insufficient history (that case yields `None` for
    the affected points, which callers skip rather than store)."""


class MarketDataReplayError(MarketDataError):
    """Raised when replaying raw events into a fresh feature store fails
    to reproduce identical feature ids, ordering, values, or semantic
    digest -- the market-data analogue of
    `PortfolioRiskVerificationError`'s replay-divergence role."""


class MarketDataVerificationError(MarketDataError):
    """Raised by `market_data.verification` (or a caller consuming its
    report) when a FATAL cross-consistency check fails and the caller
    has asked for that to raise rather than merely be reported as an
    issue."""


# --------------------------------------------------------------------------
# Durable repository, dataset versioning, incremental ingestion,
# partitioning, and reconciliation (Milestone 10, Phase 2)
# --------------------------------------------------------------------------
class MarketDataPathSecurityError(MarketDataError):
    """Raised when a dataset/partition/checkpoint key would resolve to a
    filesystem path outside the intended storage root (path traversal),
    or otherwise contains characters unsafe to use as a path component --
    the market-data analogue of `PathSecurityError`."""


class DatasetManifestError(MarketDataError):
    """Raised when a `DatasetManifest` is structurally invalid, or an
    operation would advance one to a state inconsistent with its own
    durable evidence (a partition it references is missing/unverified, a
    digest mismatch, an incoherent event/row count or time range)."""


class DatasetIdentityError(MarketDataError):
    """Raised when a `dataset_id` cannot be computed, or a provided
    manifest does not reproduce the identity it is being verified
    against -- a forged or tampered manifest."""


class PartitionError(MarketDataError):
    """Raised when a `Partition` is structurally invalid, its recorded
    content digest does not match its own member list, or a requested
    partitioning operation cannot proceed safely (an out-of-range
    boundary, an empty partition, a member assigned to the wrong
    partition key)."""


class IngestionError(MarketDataError):
    """Base class for ingestion-batch failures: a structurally invalid
    batch, a sequence/partition-membership inconsistency, or a batch that
    cannot be safely committed."""


class IngestionConflictError(IngestionError):
    """Raised when an ingestion batch is submitted under a `batch_id`
    that already has a durably recorded result for DIFFERENT content
    (a different ordered event-id list or ingestion_time) -- an exact
    repeat of the SAME content under the SAME `batch_id` is idempotently
    absorbed instead and never raises this."""


class CheckpointError(MarketDataError):
    """Raised when a checkpoint is structurally invalid, or a provided
    checkpoint does not reproduce the identity it is being verified
    against -- a forged or tampered checkpoint."""


class StaleCheckpointError(CheckpointError):
    """Raised when a checkpoint's own recorded coordinate/digest does not
    match the durable data it claims to describe (behind OR ahead of the
    actual store) -- a checkpoint is never trusted without this
    independent re-validation."""


class MarketDataRecoveryError(MarketDataError):
    """Raised when deterministic repository recovery cannot safely
    reconstruct dataset/partition/manifest/checkpoint state from durable
    evidence alone -- surfaced rather than guessed. Never used to justify
    fabricating data that was never actually durably committed."""


class MarketDataReconciliationError(MarketDataError):
    """Raised when repository reconciliation cannot complete
    structurally (e.g. a referenced store cannot be read at all), as
    opposed to an ordinary structured reconciliation issue (missing
    partition, wrong digest, ...), which is a normal, expected,
    non-raising finding."""


class RepositoryCorruptionError(MarketDataError):
    """Raised for structural repository corruption a reconciliation
    ISSUE would understate: an unrecoverable trailing record that is not
    a clean truncation, a manifest whose own digest does not match its
    identity payload, or evidence that has been tampered with rather than
    merely incomplete. Distinct from `MarketDataPersistenceError`
    (Phase 1's I/O-level corruption signal, still raised for a single
    corrupted store file) -- this is the repository-level escalation for
    corruption that recovery/reconciliation determined it must not
    silently quarantine past."""


class ExportError(MarketDataError):
    """Raised when a deterministic export cannot be produced safely: a
    non-finite/non-Decimal value slipping into an exported field, an
    inconsistent column set across rows, or a requested export format
    this package does not support."""


# --------------------------------------------------------------------------
# Historical ingestion orchestration and offline source adapters
# (Milestone 10, Phase 3)
# --------------------------------------------------------------------------
class HistoricalIngestionError(MarketDataError):
    """Base class for every failure in `quant_platform.market_data`'s
    historical ingestion layer. This layer is OFFLINE-ONLY: it never
    opens a network connection, never imports a broker/vendor SDK, and
    never defines a credential field -- it imports already-acquired
    local files (CSV/JSON Lines) or in-memory fixtures into the durable
    repository built in Phase 2."""


class SourceAdapterError(HistoricalIngestionError):
    """Raised when a source adapter cannot safely produce raw records: a
    malformed file, an unsupported/unrecognized schema, a declared
    encoding the file does not actually match, or a request for a record
    kind the adapter does not support. Adapters raise this for
    STRUCTURAL problems with the source itself -- a single bad ROW is a
    validation-layer concern (`RowValidationError`/quarantine), never
    this."""


class SourceManifestError(HistoricalIngestionError):
    """Raised when a `SourceManifest` is structurally invalid: an
    inconsistent field combination, a non-finite/negative byte size, or
    an attempt to construct one from content that does not match its own
    declared digest."""


class SourceIdentityError(HistoricalIngestionError):
    """Raised when a `source_manifest_id` cannot be computed, or a
    provided `SourceManifest` does not reproduce the identity it is
    being verified against -- a forged or tampered manifest."""


class InstrumentMappingError(HistoricalIngestionError):
    """Raised when an instrument/symbol mapping cannot be safely
    resolved: an unmapped source symbol under a fail-closed policy, an
    ambiguous mapping (the same source symbol bound to two different
    `instrument_id` values within one mapping spec), or a structurally
    invalid mapping spec."""


class TimeframeMappingError(HistoricalIngestionError):
    """Raised when a source-declared timeframe label cannot be safely
    resolved to a canonical `Timeframe`: an unknown label, or a
    structurally invalid mapping spec."""


class RowValidationError(HistoricalIngestionError):
    """Raised when a raw source row fails validation AND the active
    policy is fail-fast (never quarantine) -- the row-level analogue of
    `MarketDataEventError`. Under quarantine policy, the identical
    validation failure produces a `QuarantineRecord` instead of raising;
    this exception exists for the caller that explicitly asked
    (`fail_fast=True`) not to have invalid rows silently diverted."""


class SourceQuarantineError(HistoricalIngestionError):
    """Raised when a quarantine append would silently overwrite an
    already-quarantined source coordinate with DIFFERENT evidence -- an
    identical re-append (same source coordinate, same content) is
    idempotently absorbed instead and never raises this. Named distinctly
    from `historical.repair`'s own, unrelated `QuarantineError` (a
    different package's different quarantine concept -- historical bar
    repair vs. this layer's source-row ingestion quarantine)."""


class BackfillPlanError(HistoricalIngestionError):
    """Raised when a backfill plan cannot be safely constructed: an
    inverted or non-finite requested interval, a structurally invalid
    overlap/gap policy combination, or (under `REJECT_ANY_OVERLAP`) a
    requested interval that genuinely overlaps already-covered durable
    data. An admissible-but-imperfect plan (gaps present under
    `allow_and_report`, for instance) is never this -- it is reported via
    `BackfillPlan.issues`/`is_admissible` instead."""


class OrchestrationError(HistoricalIngestionError):
    """Base class for historical-ingestion stage-machine failures."""


class OrchestrationStateError(OrchestrationError):
    """Raised for an illegal `IngestionStage` transition, or an attempt
    to advance an operation past a stage its own durable evidence does
    not yet support (e.g. claiming `VERIFIED` before `REPOSITORY_
    COMMITTED` and `PROVENANCE_COMMITTED` both durably agree)."""


class OrchestrationConflictError(OrchestrationError):
    """Raised when an ingestion operation is resubmitted under the same
    operation/batch identity with DIFFERENT inputs (a changed source
    content digest, mapping id, or normalization spec id) -- fails
    closed rather than silently continuing a differently-sourced
    operation under an old identity. An exact retry (identical inputs)
    is idempotently absorbed instead and never raises this."""


class ProvenanceError(HistoricalIngestionError):
    """Raised when a `ProvenanceRecord` is structurally invalid, its own
    identity cannot be reproduced from its recorded fields (forged), or
    an index built from durable provenance evidence finds a genuine
    conflict: one source row bound to two different events, or one event
    bound to two different source rows, within what should be a single
    coherent operation's provenance."""


class HistoricalReconciliationError(HistoricalIngestionError):
    """Raised when historical-ingestion reconciliation cannot complete
    structurally (a referenced store/source cannot even be read), as
    opposed to an ordinary structured reconciliation issue, which is a
    normal, expected, non-raising finding."""


class HistoricalVerificationError(HistoricalIngestionError):
    """Raised by historical-ingestion verification (or a caller
    consuming its report) when a FATAL cross-consistency check fails and
    the caller has asked for that to raise rather than merely be
    reported as an issue."""


# --------------------------------------------------------------------------
# Secure external historical collector infrastructure and FRED integration
# (Milestone 10, Phase 4A). `CollectorError` is a SIBLING of
# `HistoricalIngestionError` (both `MarketDataError` directly), not a
# subclass of it -- the collector layer is network-capable (for HISTORICAL
# data only, never live), a materially different trust boundary than
# Phase 3's purely offline ingestion layer, and callers must be able to
# catch "something about talking to an external collector went wrong"
# separately from "something about the purely offline ingestion pipeline
# went wrong."
# --------------------------------------------------------------------------
class CollectorError(MarketDataError):
    """Base class for every failure in
    `quant_platform.market_data.collectors`. This layer fetches HISTORICAL
    data only, over HTTPS, from an explicit host allowlist -- it never
    opens a live/streaming connection, never imports a broker SDK, and
    never executes an order."""


class DisallowedUrlError(CollectorError):
    """Raised when a URL fails STATIC security validation before any
    connection is attempted: a non-HTTPS scheme, userinfo in the URL, an
    IP-literal host, a host not on the caller-supplied allowlist, or any
    other structural rule a URL must satisfy before a transport is even
    permitted to resolve it."""


class SsrfTargetError(CollectorError):
    """Raised when a hostname RESOLVES (via DNS) to a loopback, private,
    link-local, multicast, unspecified, or otherwise non-global address --
    including when this is discovered only at connect time (the
    DNS-rebinding case: a hostname that legitimately resolved to a public
    address during allowlist validation resolves to a private address by
    the time of actual connection). Distinct from `DisallowedUrlError`,
    which is a purely static, pre-DNS check on the URL string itself."""


class RedirectViolationError(CollectorError):
    """Raised when a redirect cannot be safely followed: the maximum
    redirect count was exceeded, a redirect target fails the SAME full
    security validation the original URL underwent, or a redirect
    attempts a scheme downgrade (HTTPS -> HTTP)."""


class TransportTimeoutError(CollectorError):
    """Raised when a connect or read timeout elapses. Never conflated
    with a data-quality failure -- a timeout is purely a transport-layer,
    infrastructure-level fact, and the retry layer classifies it as
    retryable."""


class ResponseTooLargeError(CollectorError):
    """Raised when a response body would exceed the caller-configured
    maximum size -- checked incrementally while reading, so an
    oversized response is rejected before its full body is ever held in
    memory."""


class RetryExhaustedError(CollectorError):
    """Raised when a request has been retried per its `RetryPolicy` up
    to `max_attempts` and every attempt still failed."""


class RateLimitUnavailableError(CollectorError):
    """Raised when a caller requests a rate-limit token be acquired in a
    mode that fails closed rather than waits, and no token is currently
    available."""


class MalformedFredResponseError(CollectorError):
    """Raised when a FRED response cannot be parsed as valid JSON/CSV
    matching FRED's own documented schema shape at all -- a structural
    parse failure, distinct from a single malformed OBSERVATION within an
    otherwise well-formed response (which is `fred_schemas.py`'s own
    per-row quarantine concern, not this)."""


class UnsupportedFredSchemaError(CollectorError):
    """Raised when a FRED response's own declared/inferred schema
    version, file type, or top-level shape is not one this collector
    knows how to parse."""


class ResponseIntegrityError(CollectorError):
    """Raised when raw response bytes do not match their own recorded
    `raw_content_digest` (on re-hash), or a response manifest's declared
    `byte_length`/`content_type` disagrees with the actual persisted
    bytes -- a tamper/corruption signal, never silently repaired."""


class CacheCorruptionError(CollectorError):
    """Raised when the raw-response cache's own on-disk state cannot be
    trusted: bytes missing for a manifest that claims to exist, bytes
    present but re-hashing does not reproduce the recorded digest, or a
    conflicting write is attempted under an identity the cache already
    durably holds different content for."""


class SecretExposureError(CollectorError):
    """Raised when a caller-supplied credential (an API key) is detected
    somewhere it must never appear -- a request manifest, a response
    manifest, a report, a durable artifact, or an exception message.
    Exists primarily so the safety-scan/redaction tests have a specific,
    unambiguous failure mode to assert against."""


class CollectorRequestManifestError(CollectorError):
    """Raised when a `CollectorRequestManifest` is structurally invalid,
    or its own identity cannot be reproduced from its recorded fields."""


class CollectorResponseManifestError(CollectorError):
    """Raised when a `CollectorResponseManifest` is structurally invalid
    (e.g. a partial/incomplete response marked complete), or its own
    identity cannot be reproduced from its recorded fields."""


class CollectorOrchestrationError(CollectorError):
    """Base class for collector-operation stage-machine failures --
    the collector-side analogue of `OrchestrationError`."""


class CollectorOrchestrationStateError(CollectorOrchestrationError):
    """Raised for an illegal collector-operation stage transition, or an
    attempt to advance past a stage its own durable evidence does not
    yet support."""


class CollectorOrchestrationConflictError(CollectorOrchestrationError):
    """Raised when a collector operation is resubmitted under the same
    operation identity with DIFFERENT inputs. An exact retry is
    idempotently absorbed instead and never raises this."""


class CollectorReconciliationError(CollectorError):
    """Raised when collector reconciliation cannot complete structurally
    (a referenced store/artifact cannot even be read), as opposed to an
    ordinary structured reconciliation issue, which is a normal,
    expected, non-raising finding."""


class CollectorVerificationError(CollectorError):
    """Raised by collector verification (or a caller consuming its
    report) when a FATAL cross-consistency check fails and the caller
    has asked for that to raise rather than merely be reported as an
    issue."""


# --------------------------------------------------------------------------
# Milestone 10, Phase 4B: curated FRED macro universe.
# --------------------------------------------------------------------------
class CuratedRegistryError(CollectorError):
    """Raised when a curated FRED series registry or an individual
    `CuratedFredSeriesSpec` is structurally invalid: a duplicate FRED
    series id or canonical name, an ambiguous target instrument
    identity, an unsupported normalization/unit/frequency combination,
    a missing release-availability policy, or an enabled series with no
    target identity."""


class SeriesMetadataError(CollectorError):
    """Base class for FRED series-METADATA failures (the `/fred/series`
    endpoint, distinct from `/fred/series/observations`) -- a structural
    parse failure of the metadata response itself, as opposed to a
    drift between that (valid) metadata and a curated spec's own
    expectations (`MetadataDriftError`)."""


class MetadataDriftError(SeriesMetadataError):
    """Raised when officially-returned FRED series metadata disagrees
    with a curated spec's own declared expectations in a way this
    phase's drift policy classifies as FAIL CLOSED: a different series
    id than requested, an incompatible frequency, incompatible units, or
    a seasonal-adjustment change that is analytically meaningful. A
    harmless title/notes-only change is a WARNING, never this."""


class AvailabilityPolicyError(CollectorError):
    """Raised when a `ReleaseAvailabilityPolicy` is structurally invalid
    (e.g. a negative delay, an unsupported policy kind, or a missing
    field its own kind requires)."""


class AvailabilityUnresolvedError(AvailabilityPolicyError):
    """Raised when an observation's `availability_time` cannot be
    resolved under its declared `ReleaseAvailabilityPolicy` -- the
    record must be quarantined or rejected, NEVER silently treated as
    immediately available (which would be a point-in-time leak)."""


class RevisionPolicyError(CollectorError):
    """Raised when a requested/declared revision policy
    (`LATEST_AVAILABLE`/`FIRST_RELEASE_ONLY`/`AS_OF_REALTIME_DATE`/
    `VINTAGE_SERIES`) is invalid for the requesting context, or two
    series combined into one curated universe declare incompatible
    revision policies that this phase does not support combining."""


class CuratedBackfillSpecError(CollectorError):
    """Raised when a `CuratedBackfillSpec` is structurally invalid: an
    empty or duplicate series selection, an unknown or disabled series,
    an invalid interval, an unbounded request, or a wall-clock-dependent
    default this phase forbids."""


class CombinedManifestError(CollectorError):
    """Raised when a combined curated-universe manifest is structurally
    invalid, or its own identity cannot be reproduced from its recorded
    component dataset references."""


class UpdatePlanError(CollectorError):
    """Raised when incremental update-plan construction receives
    invalid inputs (e.g. a caller-supplied planning time that is not
    tz-aware, or an existing manifest that does not belong to the
    curated registry/universe being planned against)."""


# --------------------------------------------------------------------------
# Milestone 10, Phase 4C: provider-neutral cross-asset historical market
# collectors and curated XAUUSD market-driver universe.
# `CollectorOrchestrationStateError`/`CollectorOrchestrationConflictError`/
# `CollectorReconciliationError`/`CollectorVerificationError` (Phase 4A,
# above) are REUSED directly for this phase's own multi-driver stage
# machine, reconciliation, and verification -- exactly as Phase 4B already
# did for its own curated FRED universe; they are already generic enough
# that a phase-specific subclass would add no distinguishing information.
# --------------------------------------------------------------------------
class MarketDriverRegistryError(CollectorError):
    """Raised when a curated cross-asset market-driver registry or an
    individual `CuratedMarketDriverSpec` is structurally invalid: a
    duplicate canonical driver id or name, a provider mapping without an
    instrument form, a proxy mapping without an explicit proxy target, a
    futures mapping without contract/continuation semantics, an equity/
    ETF mapping without an adjustment policy, an enabled driver with no
    supported mapping, or a result-affecting field excluded from
    identity."""


class ProviderCapabilityError(CollectorError):
    """Raised when a request would exceed a `MarketCollectorCapabilities`
    declaration a `HistoricalMarketCollector` provider adapter actually
    made: an interval/granularity/adjustment/instrument-form the provider
    does not support, a request exceeding the declared maximum
    interval/rows-per-page, or a runtime-credential requirement not
    satisfied. The orchestrator fails closed here rather than silently
    downgrading the request."""


class InstrumentFormError(CollectorError):
    """Raised when an instrument-form/proxy classification is structurally
    invalid: an unsupported `InstrumentForm`, a proxy mapping missing its
    required `proxy_for`/`proxy_quality` classification, or code that
    would label a proxy instrument (an ETF, a continuous futures series)
    as the underlying spot/cash instrument it merely approximates."""


class SymbolMappingError(CollectorError):
    """Raised when a `ProviderSymbolMapping` is structurally invalid, or a
    mapping-set would let one provider symbol resolve to two DIFFERENT
    active canonical instruments within the same mapping version -- an
    ambiguous mapping is rejected at construction, never resolved by
    silent precedence."""


class AdjustmentPolicyError(CollectorError):
    """Raised when a price-`AdjustmentPolicy` is structurally invalid for
    its declared `AdjustmentPolicyKind`, or a normalization step would mix
    raw and adjusted price semantics within one series (e.g. an
    adjusted-close value substituted into an otherwise-raw OHLC bar)."""


class SessionPolicyError(CollectorError):
    """Raised when a `TimezoneSessionPolicy` is structurally invalid: an
    unresolvable timezone, an invalid session-open/close/break
    combination, or a policy that cannot honestly represent the provider's
    own documented session semantics."""


class FuturesContractError(CollectorError):
    """Raised when `FuturesContractMetadata` is structurally invalid or
    missing a result-critical field (root symbol, exchange, expiry,
    contract month/year, multiplier, quote unit, currency, tick size) --
    a provider that cannot supply this metadata must be classified as
    provider-generated continuous data instead of asserting individual
    contract knowledge it does not actually have."""


class ContinuationPolicyError(CollectorError):
    """Raised when a `ContinuationPolicy` is structurally invalid, or a
    continuous-series value is produced without the roll provenance
    (active/prior/next contract, roll timestamp, adjustment amount or
    ratio) its declared policy kind requires -- continuation is NEVER
    silently stitched without this evidence."""


class MarketAvailabilityPolicyError(CollectorError):
    """Raised when a market-bar `BarAvailabilityPolicy` is structurally
    invalid (e.g. a negative delay, an unsupported policy kind, or a
    missing field its own kind requires) -- the market-bar analogue of
    Phase 4B's `AvailabilityPolicyError`."""


class MarketAvailabilityUnresolvedError(MarketAvailabilityPolicyError):
    """Raised when a bar's `availability_time` cannot be resolved under
    its declared `BarAvailabilityPolicy` -- the record must be quarantined
    or rejected, NEVER silently treated as available at candle open (which
    would be a point-in-time leak) -- the market-bar analogue of Phase
    4B's `AvailabilityUnresolvedError`."""


class MarketProviderResponseError(CollectorError):
    """Raised when an ENTIRE provider response cannot be parsed as the
    provider's own documented schema shape at all -- a structural parse
    failure (unexpected top-level JSON shape, a provider error/rate-
    limit envelope in place of the expected data envelope), distinct
    from `MarketRecordError` (a single malformed ROW within an
    otherwise well-formed response)."""


class MarketRecordError(CollectorError):
    """Raised when a raw or normalized market record is structurally
    invalid: an OHLC relationship violation, a non-finite/non-positive
    price where economically required, negative volume, a timestamp not
    aligned with its declared interval, or an overlapping duplicate
    coordinate carrying a conflicting payload."""


class MarketBackfillSpecError(CollectorError):
    """Raised when a `CuratedMarketBackfillSpec` is structurally invalid:
    an empty or duplicate driver selection, an unsupported provider
    mapping, an invalid interval, an unbounded request, an implicit
    current-date default, an unsupported adjustment/futures policy, or
    mixing semantically incompatible series into one component dataset."""


class MarketCombinedManifestError(CollectorError):
    """Raised when a combined cross-asset market-driver universe manifest
    is structurally invalid, or its own identity cannot be reproduced
    from its recorded component dataset references."""


class MarketUpdatePlanError(CollectorError):
    """Raised when cross-asset incremental update-plan construction
    receives invalid inputs (a non-tz-aware planning time, or an existing
    manifest that does not belong to the curated registry/universe being
    planned against) -- the market-driver analogue of Phase 4B's
    `UpdatePlanError`."""


class GapPolicyError(CollectorError):
    """Raised when a `GapPolicy` is structurally invalid, or a required
    driver's missing-bar rate exceeds its configured tolerance under a
    policy that must prevent a COMPLETE universe status as a result --
    an ordinary, expected gap finding is reported via `GapReport`
    instead and never raises this."""


# --------------------------------------------------------------------------
# Milestone 10, Phase 4D: point-in-time multi-source alignment bridge
# between `quant_platform.market_data` and `quant_platform.features`.
# `MarketDataBridgeError` is a `FeatureError` subclass (not a
# `MarketDataError` one) because the bridge package itself lives under
# `quant_platform.features.market_data_bridge` and only ever READS from
# `market_data` -- the same "exceptions are grouped by the package that
# raises them" convention every other block in this file already follows.
# --------------------------------------------------------------------------
class MarketDataBridgeError(FeatureError):
    """Base class for every failure in
    `quant_platform.features.market_data_bridge`. This package never
    computes a feature value itself (all feature-family computation
    still happens in the unmodified Milestone 3 `features` modules), never
    writes to `quant_platform.market_data`, and never opens a network
    connection -- it only reads already-durable `market_data` evidence and
    reshapes/aligns it into the exact input shapes `features.engine.
    FeatureEngine`/`features.dataset_builder.ResearchDatasetBuilder`
    already accept."""


class SourceBindingError(MarketDataBridgeError):
    """Raised when a `bindings.BaseAssetDatasetBinding`/`MacroDatasetBinding`/
    `CrossAssetDatasetBinding` is structurally invalid -- including an
    attempt to pin a mutable alias (`"latest"`/`"current"`/`"newest"`/
    `"active"`/a provider default) in a field that must be an exact,
    immutable, content-addressed identity string."""


class SourceVerificationError(MarketDataBridgeError):
    """Raised when a pinned source binding cannot be independently
    re-verified against the durable `market_data` evidence it claims to
    reference: the referenced dataset/manifest/partition/component no
    longer exists, a recomputed semantic digest does not match the pinned
    one, or the underlying durable store has been appended to since the
    binding was pinned (making the exact pinned version no longer
    reconstructible from the store's current, append-only-grown state).
    Always fails closed -- never silently substitutes the current/latest
    state for the pinned one."""


class AlignmentPolicyError(MarketDataBridgeError):
    """Raised when a requested source-alignment, revision/vintage, or
    availability policy is structurally invalid for the requesting
    context -- most commonly, a macro revision policy that is explicitly
    NOT point-in-time-safe (`RevisionPolicyKind.LATEST_AVAILABLE` or
    `AS_OF_REALTIME_DATE`) bound to a training/research dataset request,
    which this bridge always refuses rather than silently building a
    leaky dataset."""


class SourceCoverageError(MarketDataBridgeError):
    """Raised when a bound source's available coverage cannot satisfy its
    declared `coverage.SourceCoveragePolicy` under `FAIL_REQUIRED_SOURCE`
    -- a required source with insufficient coverage, or a requested range
    with no safe common overlap across every required source. An optional
    source's shortfall, or a policy that permits trimming/quarantine, is
    reported via `CoverageReport` instead and never raises this."""


class MarketDataLineageError(MarketDataBridgeError):
    """Raised when a `ResearchDatasetManifest.market_data_lineage` payload
    is structurally invalid, references a lineage schema version this code
    does not know how to read, or cannot be reproduced from its own
    recorded bindings -- never silently reinterpreted as a different
    schema version."""


class RebuildPlanError(MarketDataBridgeError):
    """Raised when `rebuild_planner`'s pure planner cannot safely
    construct a rebuild plan from its inputs: an existing research
    manifest whose recorded market-data lineage is missing or malformed,
    or a proposed binding set that is not a well-formed successor to the
    manifest's recorded one."""


class BridgeReconciliationError(MarketDataBridgeError):
    """Raised when bridge-side reconciliation cannot complete structurally
    (a referenced binding/manifest/store cannot even be read), as opposed
    to an ordinary structured `ReconciliationIssue` finding, which is a
    normal, expected, non-raising outcome."""


class BridgeVerificationError(MarketDataBridgeError):
    """Raised by `verification.verify_market_data_bridge` (or a caller
    consuming its report) when a FATAL cross-consistency check fails and
    the caller has asked for that to raise rather than merely be reported
    as an issue."""
