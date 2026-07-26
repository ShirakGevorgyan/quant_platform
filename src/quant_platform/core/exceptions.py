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
