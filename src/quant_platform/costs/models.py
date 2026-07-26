"""Transaction cost models.

Strategies never reason about spread, slippage, or commission -- a
`CostModel` is the single place that turns a "reference" market price
(typically a bar's close) into a realistic fill price, keeping cost
assumptions auditable and swappable independent of signal logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quant_platform.core.types import OrderSide


class CostModel(ABC):
    """Determines realistic fill prices and commission for order execution.

    `current_volatility` is accepted by every implementation (even ones
    that ignore it) so callers can pass it uniformly and swap cost models
    without branching on which one is in use (Liskov substitution).
    """

    @abstractmethod
    def entry_fill_price(
        self, reference_price: float, side: OrderSide, *, current_volatility: float | None = None
    ) -> float:
        """Realistic fill price for OPENING a position via `side` at
        `reference_price` (typically the signal bar's close)."""
        raise NotImplementedError

    @abstractmethod
    def exit_fill_price(
        self, reference_price: float, position_side: OrderSide, *, current_volatility: float | None = None
    ) -> float:
        """Realistic fill price for CLOSING a position that was originally
        opened via `position_side` (BUY = closing a long, SELL = closing a
        short), at `reference_price`."""
        raise NotImplementedError

    @abstractmethod
    def commission(self, quantity: float, price: float) -> float:
        """Commission owed for a single fill of `quantity` units at `price`.
        Charged once per fill (i.e. once on entry, once on exit)."""
        raise NotImplementedError


class FixedSpreadCostModel(CostModel):
    """Constant spread + constant adverse slippage on entry, expressed in
    "points" (the instrument's smallest quoted price increment) and
    converted to price units via `point_value`.

    This mirrors the assumptions used for the reference XAUUSD backtest
    this platform's cost modeling supersedes: half the spread applied on
    each side of the market, slippage applied only on entry (a stop/limit
    exit is assumed to fill at its exact level; no additional slippage is
    modeled there), and a flat per-unit commission.
    """

    def __init__(
        self,
        spread_points: float,
        slippage_points: float,
        point_value: float,
        commission_per_unit: float = 0.0,
    ) -> None:
        if spread_points < 0:
            raise ValueError(f"spread_points must be non-negative, got {spread_points}")
        if slippage_points < 0:
            raise ValueError(f"slippage_points must be non-negative, got {slippage_points}")
        if point_value <= 0:
            raise ValueError(f"point_value must be positive, got {point_value}")
        if commission_per_unit < 0:
            raise ValueError(f"commission_per_unit must be non-negative, got {commission_per_unit}")

        self._half_spread_price = (spread_points / 2.0) * point_value
        self._slippage_price = slippage_points * point_value
        self._commission_per_unit = commission_per_unit

    def entry_fill_price(
        self,
        reference_price: float,
        side: OrderSide,
        *,
        current_volatility: float | None = None,  # noqa: ARG002 - part of the uniform CostModel interface
    ) -> float:
        adverse = self._half_spread_price + self._slippage_price
        return reference_price + side.sign * adverse

    def exit_fill_price(
        self,
        reference_price: float,
        position_side: OrderSide,
        *,
        current_volatility: float | None = None,  # noqa: ARG002 - part of the uniform CostModel interface
    ) -> float:
        # Closing a long (position_side=BUY) means selling at the bid: price
        # moves against the position by half the spread. Closing a short
        # means buying at the ask: same adverse direction, opposite sign.
        return reference_price - position_side.sign * self._half_spread_price

    def commission(self, quantity: float, price: float) -> float:  # noqa: ARG002 - flat-rate model ignores price
        return abs(quantity) * self._commission_per_unit


class VolatilityScaledSlippageModel(CostModel):
    """Like `FixedSpreadCostModel`, but entry slippage scales with a
    supplied volatility measure (e.g. ATR) relative to a reference level --
    a fixed slippage assumption understates cost in fast markets and
    overstates it in quiet ones; this model lets that assumption vary with
    the regime the backtest is actually simulating.
    """

    def __init__(
        self,
        spread_points: float,
        base_slippage_points: float,
        reference_volatility: float,
        point_value: float,
        commission_per_unit: float = 0.0,
        min_slippage_multiplier: float = 0.25,
        max_slippage_multiplier: float = 5.0,
    ) -> None:
        if spread_points < 0:
            raise ValueError(f"spread_points must be non-negative, got {spread_points}")
        if base_slippage_points < 0:
            raise ValueError(f"base_slippage_points must be non-negative, got {base_slippage_points}")
        if reference_volatility <= 0:
            raise ValueError(f"reference_volatility must be positive, got {reference_volatility}")
        if point_value <= 0:
            raise ValueError(f"point_value must be positive, got {point_value}")
        if commission_per_unit < 0:
            raise ValueError(f"commission_per_unit must be non-negative, got {commission_per_unit}")
        if min_slippage_multiplier <= 0 or min_slippage_multiplier > max_slippage_multiplier:
            raise ValueError(
                "min_slippage_multiplier must be positive and <= max_slippage_multiplier "
                f"(got min={min_slippage_multiplier}, max={max_slippage_multiplier})"
            )

        self._half_spread_price = (spread_points / 2.0) * point_value
        self._base_slippage_price = base_slippage_points * point_value
        self._reference_volatility = reference_volatility
        self._commission_per_unit = commission_per_unit
        self._min_multiplier = min_slippage_multiplier
        self._max_multiplier = max_slippage_multiplier

    def _slippage_multiplier(self, current_volatility: float | None) -> float:
        if current_volatility is None or current_volatility <= 0:
            return 1.0
        raw_multiplier = current_volatility / self._reference_volatility
        return min(max(raw_multiplier, self._min_multiplier), self._max_multiplier)

    def entry_fill_price(
        self, reference_price: float, side: OrderSide, *, current_volatility: float | None = None
    ) -> float:
        slippage = self._base_slippage_price * self._slippage_multiplier(current_volatility)
        adverse = self._half_spread_price + slippage
        return reference_price + side.sign * adverse

    def exit_fill_price(
        self,
        reference_price: float,
        position_side: OrderSide,
        *,
        current_volatility: float | None = None,  # noqa: ARG002 - part of the uniform CostModel interface
    ) -> float:
        return reference_price - position_side.sign * self._half_spread_price

    def commission(self, quantity: float, price: float) -> float:  # noqa: ARG002 - flat-rate model ignores price
        return abs(quantity) * self._commission_per_unit
