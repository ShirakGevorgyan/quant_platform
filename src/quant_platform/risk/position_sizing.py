"""Position sizing / risk management.

`PositionSizer` implements the Template Method pattern: the base class
owns input validation and the max-leverage safety clamp (or hard rejection)
so every concrete sizing rule gets that protection for free and cannot
accidentally omit it; subclasses only implement the sizing formula itself
via `_raw_size`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from quant_platform.core.exceptions import RiskLimitExceededError

LimitPolicy = Literal["clamp", "raise"]


class PositionSizer(ABC):
    def __init__(self, max_position_fraction: float = 1.0, on_limit_exceeded: LimitPolicy = "clamp") -> None:
        if not (0.0 < max_position_fraction <= 1.0):
            raise ValueError(
                f"max_position_fraction must be in (0, 1], got {max_position_fraction}"
            )
        if on_limit_exceeded not in ("clamp", "raise"):
            raise ValueError(f"on_limit_exceeded must be 'clamp' or 'raise', got {on_limit_exceeded!r}")

        self._max_position_fraction = max_position_fraction
        self._on_limit_exceeded: LimitPolicy = on_limit_exceeded

    @abstractmethod
    def _raw_size(
        self,
        *,
        account_equity: float,
        entry_price: float,
        point_value: float,
        stop_loss_price: float | None,
        current_volatility: float | None,
    ) -> float:
        """The sizing formula itself, before the max-leverage safeguard is
        applied. Must return a non-negative quantity."""
        raise NotImplementedError

    def size(
        self,
        *,
        account_equity: float,
        entry_price: float,
        point_value: float = 1.0,
        stop_loss_price: float | None = None,
        current_volatility: float | None = None,
    ) -> float:
        """Compute a risk-managed position quantity. Never returns a
        quantity whose notional exposure (quantity * entry_price *
        point_value) exceeds `max_position_fraction` of `account_equity`."""
        if account_equity <= 0:
            raise ValueError(f"account_equity must be positive, got {account_equity}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if point_value <= 0:
            raise ValueError(f"point_value must be positive, got {point_value}")

        raw_quantity = self._raw_size(
            account_equity=account_equity,
            entry_price=entry_price,
            point_value=point_value,
            stop_loss_price=stop_loss_price,
            current_volatility=current_volatility,
        )
        if raw_quantity < 0:
            raise RiskLimitExceededError(
                f"Sizing formula produced a negative quantity ({raw_quantity}); this "
                "indicates a bug in the sizer, not a legitimate risk limit.",
                context={"sizer": type(self).__name__},
            )

        max_quantity = (account_equity * self._max_position_fraction) / (entry_price * point_value)
        if raw_quantity > max_quantity:
            if self._on_limit_exceeded == "raise":
                raise RiskLimitExceededError(
                    f"Computed quantity {raw_quantity:.6f} exceeds the max_position_fraction "
                    f"limit ({max_quantity:.6f} at {self._max_position_fraction:.0%} of equity)",
                    context={
                        "sizer": type(self).__name__,
                        "raw_quantity": raw_quantity,
                        "max_quantity": max_quantity,
                    },
                )
            return max_quantity
        return raw_quantity


class FixedFractionalSizer(PositionSizer):
    """Risk a fixed percentage of account equity per trade, sized from the
    distance to the stop-loss level."""

    def __init__(
        self,
        risk_percent: float,
        max_position_fraction: float = 1.0,
        on_limit_exceeded: LimitPolicy = "clamp",
    ) -> None:
        super().__init__(max_position_fraction, on_limit_exceeded)
        if not (0.0 < risk_percent <= 100.0):
            raise ValueError(f"risk_percent must be in (0, 100], got {risk_percent}")
        self._risk_percent = risk_percent

    def _raw_size(
        self,
        *,
        account_equity: float,
        entry_price: float,
        point_value: float,
        stop_loss_price: float | None,
        current_volatility: float | None,  # noqa: ARG002 - part of the uniform PositionSizer interface
    ) -> float:
        if stop_loss_price is None:
            raise ValueError("FixedFractionalSizer requires stop_loss_price")
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            raise ValueError("stop_loss_price must differ from entry_price")

        risk_amount = account_equity * (self._risk_percent / 100.0)
        return risk_amount / (stop_distance * point_value)


class VolatilityTargetSizer(PositionSizer):
    """Size a position so its expected P&L volatility over one holding
    period (approximated by `current_volatility`, e.g. an ATR value in
    price units) equals a fixed percentage of account equity."""

    def __init__(
        self,
        target_volatility_percent: float,
        max_position_fraction: float = 1.0,
        on_limit_exceeded: LimitPolicy = "clamp",
    ) -> None:
        super().__init__(max_position_fraction, on_limit_exceeded)
        if not (0.0 < target_volatility_percent <= 100.0):
            raise ValueError(
                f"target_volatility_percent must be in (0, 100], got {target_volatility_percent}"
            )
        self._target_volatility_percent = target_volatility_percent

    def _raw_size(
        self,
        *,
        account_equity: float,
        entry_price: float,  # noqa: ARG002 - part of the uniform PositionSizer interface
        point_value: float,
        stop_loss_price: float | None,  # noqa: ARG002 - part of the uniform PositionSizer interface
        current_volatility: float | None,
    ) -> float:
        if current_volatility is None or current_volatility <= 0:
            raise ValueError("VolatilityTargetSizer requires a positive current_volatility")

        target_amount = account_equity * (self._target_volatility_percent / 100.0)
        return target_amount / (current_volatility * point_value)


class KellyCriterionSizer(PositionSizer):
    """Fractional (typically half-) Kelly criterion sizing from a
    historical win rate and average win/loss ratio.

    `full_kelly = win_rate - (1 - win_rate) / win_loss_ratio` is the
    fraction of equity a full-Kelly bettor would allocate; `kelly_fraction`
    scales that down (0.5 = "half-Kelly", the common practitioner choice
    to reduce sensitivity to estimation error in `win_rate`/`win_loss_ratio`).
    A negative full-Kelly value (a losing edge) sizes to zero rather than
    going short the opposite trade.
    """

    def __init__(
        self,
        win_rate: float,
        win_loss_ratio: float,
        kelly_fraction: float = 0.5,
        max_position_fraction: float = 1.0,
        on_limit_exceeded: LimitPolicy = "clamp",
    ) -> None:
        super().__init__(max_position_fraction, on_limit_exceeded)
        if not (0.0 < win_rate < 1.0):
            raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
        if win_loss_ratio <= 0:
            raise ValueError(f"win_loss_ratio must be positive, got {win_loss_ratio}")
        if not (0.0 < kelly_fraction <= 1.0):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")

        self._win_rate = win_rate
        self._win_loss_ratio = win_loss_ratio
        self._kelly_fraction = kelly_fraction

    @property
    def full_kelly(self) -> float:
        return self._win_rate - (1.0 - self._win_rate) / self._win_loss_ratio

    def _raw_size(
        self,
        *,
        account_equity: float,
        entry_price: float,
        point_value: float,
        stop_loss_price: float | None,  # noqa: ARG002 - part of the uniform PositionSizer interface
        current_volatility: float | None,  # noqa: ARG002 - part of the uniform PositionSizer interface
    ) -> float:
        fractional_kelly = max(self.full_kelly, 0.0) * self._kelly_fraction
        notional = account_equity * fractional_kelly
        return notional / (entry_price * point_value)
