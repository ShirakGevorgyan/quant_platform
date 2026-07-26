"""Core domain types, exceptions, and pure time utilities.

Nothing in this subpackage performs I/O or depends on any other
`quant_platform` subpackage -- it is the dependency-free foundation every
other layer builds on.
"""

from quant_platform.core.exceptions import (
    ConfigurationError,
    DataError,
    DataQualityError,
    DataSourceError,
    EngineError,
    InsufficientDataError,
    LookaheadViolationError,
    QuantPlatformError,
    RiskLimitExceededError,
    SchemaError,
    ValidationSplitError,
)
from quant_platform.core.time_utils import compute_close_time, ensure_utc, to_pandas_freq
from quant_platform.core.types import (
    OHLCV_COLUMNS,
    Bar,
    EquityPoint,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    Signal,
    SignalAction,
    Timeframe,
    Trade,
)

__all__ = [
    "OHLCV_COLUMNS",
    "Bar",
    "ConfigurationError",
    "DataError",
    "DataQualityError",
    "DataSourceError",
    "EngineError",
    "EquityPoint",
    "Fill",
    "InsufficientDataError",
    "LookaheadViolationError",
    "Order",
    "OrderSide",
    "OrderType",
    "Position",
    "QuantPlatformError",
    "RiskLimitExceededError",
    "SchemaError",
    "Signal",
    "SignalAction",
    "Timeframe",
    "Trade",
    "ValidationSplitError",
    "compute_close_time",
    "ensure_utc",
    "to_pandas_freq",
]
