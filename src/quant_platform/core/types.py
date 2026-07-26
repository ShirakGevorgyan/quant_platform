"""Core domain types shared across the entire platform.

Design notes
------------
Bulk time-series data always flows through the system as `pandas.DataFrame`
objects conforming to `OHLCV_COLUMNS` -- this is what makes the engine
tractable at the "millions of rows" scale the platform targets, since every
per-bar computation that can be vectorized, is. Lightweight, immutable
dataclasses such as `Bar`, `Signal`, `Order`, and `Fill` exist only for the
O(1)-per-step values passed between engine components at a single point in
time; they are never used to represent bulk series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class Timeframe(Enum):
    """Supported OHLCV bar durations.

    The associated `timedelta` is the single source of truth for computing
    a bar's close time from its open time (`open_time + duration`) -- the
    invariant the entire no-look-ahead guarantee of this engine rests on.
    """

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    H12 = "H12"
    D1 = "D1"

    @property
    def duration(self) -> timedelta:
        return _TIMEFRAME_DURATIONS[self]

    @property
    def minutes(self) -> float:
        return self.duration.total_seconds() / 60.0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.duration < other.duration

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.duration <= other.duration


_TIMEFRAME_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H12: timedelta(hours=12),
    Timeframe.D1: timedelta(days=1),
}

# Canonical OHLCV schema. Every DataSource implementation must return a
# DataFrame with exactly these columns (extra columns are permitted and
# preserved, but these must be present with these exact names/order).
OHLCV_COLUMNS: tuple[str, ...] = ("open_time", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Bar:
    """A single, fully-closed OHLCV candle exposed at one point in time."""

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Bar high ({self.high}) < low ({self.low}) at {self.open_time}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Bar open ({self.open}) outside [low, high] at {self.open_time}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Bar close ({self.close}) outside [low, high] at {self.open_time}")
        if self.close_time <= self.open_time:
            raise ValueError(
                f"Bar close_time ({self.close_time}) must be after open_time ({self.open_time})"
            )


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class SignalAction(Enum):
    """A strategy's directional intent for the current bar."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"   # close any open position, stay out
    HOLD = "HOLD"   # no change to current position


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy output for a single point in time.

    Carries no *sizing* details (quantity, account-equity considerations)
    -- that remains the risk layer's responsibility. It MAY optionally
    carry `stop_loss`/`take_profit`/`max_hold`, since those are typically
    derived from the strategy's own view of market structure (e.g. "stop
    below the recent swing low") rather than from account size, and
    `current_volatility` (e.g. an ATR the strategy already computed) so
    volatility-aware position sizing and cost models don't force every
    caller to recompute it independently.
    """

    timestamp: datetime
    action: SignalAction
    confidence: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    max_hold: timedelta | None = None
    current_volatility: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Signal confidence must be in [0, 1], got {self.confidence}")
        if self.stop_loss is not None and self.take_profit is not None and self.stop_loss == self.take_profit:
            raise ValueError("stop_loss and take_profit must not be equal")


@dataclass(frozen=True, slots=True)
class Order:
    """An execution request produced from a `Signal` after position sizing."""

    timestamp: datetime
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    max_hold: timedelta | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require a limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require a stop_price")


@dataclass(frozen=True, slots=True)
class Fill:
    """The realized execution outcome of an `Order`.

    Deliberately holds only `price` (the actual, cost-adjusted fill price)
    and `commission` (a dollar amount). Slippage is not a separate field
    here: it is already embedded in how far `price` sits from whatever
    reference price the caller compared it to, and mixing a price-unit
    slippage figure with a dollar-unit commission in one "total_cost" was a
    real unit-consistency bug caught during code review before any other
    component came to depend on it. Callers that want to report slippage
    should compute `fill.price - reference_price` themselves, in whichever
    unit (price or dollars via point_value) their reporting needs.
    """

    timestamp: datetime
    side: OrderSide
    quantity: float
    price: float
    commission: float


@dataclass(slots=True)
class Position:
    """Mutable running state for one open position in one symbol.

    Deliberately holds only state and pure queries derived from that state;
    all mutation happens exclusively through `Portfolio`, which owns the
    authoritative mapping of symbol -> Position.
    """

    symbol: str
    quantity: float = 0.0
    average_entry_price: float = 0.0
    entry_commission: float = 0.0
    opened_at: datetime | None = None

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0

    def unrealized_pnl(self, current_price: float, point_value: float = 1.0) -> float:
        if self.is_flat:
            return 0.0
        return (current_price - self.average_entry_price) * self.quantity * point_value


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One snapshot of total account equity, for the equity curve."""

    timestamp: datetime
    cash: float
    equity: float
    drawdown_pct: float


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round-trip trade, as reported to the user/analytics layer."""

    entry_time: datetime
    exit_time: datetime
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    total_cost: float
    exit_reason: str

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_cost

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0.0

    @property
    def holding_period(self) -> timedelta:
        return self.exit_time - self.entry_time
