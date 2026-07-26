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
