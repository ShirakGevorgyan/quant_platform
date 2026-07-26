"""Label/target construction -- deliberately isolated from
`quant_platform.features`'s feature modules. Labels are allowed to use
FUTURE data (that is their entire purpose); features are never allowed to.
This module is never imported by `engine.py`, `technical/`, `temporal/`,
`multi_timeframe.py`, `cross_asset/`, or `macro/` -- and `engine.
FeatureEngine.compute` structurally refuses any input DataFrame carrying a
`label_`/`target_`-prefixed column (`LabelLeakageError`), so even a caller
mistake (accidentally joining labels into a features DataFrame before
computing features) is caught at the boundary rather than silently
succeeding.

Every label function returns `NaN` (`LabelResult.is_valid=False`) for a row
without enough future data to resolve -- never a fabricated value. Whether
those rows are dropped or kept (with an explicit "unlabeled" marker) is the
caller's (`dataset_builder`) decision, made explicitly via its own
`drop_unlabeled_rows` config flag; this module never drops a row itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FeatureError
from quant_platform.features.technical.price import simple_return, trailing_rolling


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeatureError(f"Expected a JSON object, got {type(value).__name__}")
    return value


class LabelKind(Enum):
    FUTURE_RETURN = "future_return"
    FUTURE_LOG_RETURN = "future_log_return"
    BINARY_DIRECTION = "binary_direction"
    VOL_ADJUSTED_RETURN = "vol_adjusted_return"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    name: str
    kind: LabelKind
    horizon_bars: int
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError(f"horizon_bars must be positive, got {self.horizon_bars}")

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind.value, "horizon_bars": self.horizon_bars, "params": self.params}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDefinition:
        return cls(
            name=str(raw["name"]), kind=LabelKind(raw["kind"]), horizon_bars=int(str(raw["horizon_bars"])),
            params=_as_dict(raw.get("params") or {}),
        )


@dataclass(frozen=True, slots=True)
class LabelResult:
    values: pd.Series
    is_valid: pd.Series
    embargo_bars: int
    definition: LabelDefinition


def compute_future_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    return close.shift(-horizon_bars) / close - 1.0


def compute_future_log_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    return pd.Series(np.log(close.shift(-horizon_bars) / close), index=close.index)


def compute_binary_direction(close: pd.Series, horizon_bars: int, *, threshold: float = 0.0) -> pd.Series:
    future_return = compute_future_return(close, horizon_bars)
    direction = (future_return > threshold).astype("float64")
    return direction.where(future_return.notna())


def compute_vol_adjusted_return(close: pd.Series, horizon_bars: int, *, vol_window: int) -> pd.Series:
    """Future return scaled by trailing (PAST-only, up to and including the
    current bar) realized volatility -- never a volatility computed from
    the same future window the return itself spans, which would smuggle
    future information into the denominator."""
    future_return = compute_future_return(close, horizon_bars)
    past_volatility = trailing_rolling(simple_return(close, 1), vol_window).std()
    result: pd.Series = (future_return / past_volatility).where(past_volatility != 0)
    return result


def build_triple_barrier_labels(
    high: pd.Series, low: pd.Series, close: pd.Series, *, horizon_bars: int, upper_pct: float, lower_pct: float
) -> pd.Series:
    """+1 if the upper barrier (`entry * (1 + upper_pct)`) is touched
    before the lower one within `horizon_bars`; -1 if the lower barrier is
    touched first (or both are touched within the same forward bar --
    OHLC alone cannot say which happened first within one bar, so this
    resolves the tie toward the stop-like outcome, the same conservative
    convention `engine.broker_simulator` already uses for intrabar
    ambiguity); 0 (the "time barrier") if neither is touched within
    `horizon_bars` -- using the SIGN of the actual terminal return, itself
    only computed where the full horizon's data genuinely exists. `NaN`
    where the row has neither a barrier touch nor a resolvable time-barrier
    outcome (insufficient future data)."""
    if horizon_bars <= 0:
        raise ValueError(f"horizon_bars must be positive, got {horizon_bars}")
    if upper_pct <= 0 or lower_pct <= 0:
        raise ValueError(f"upper_pct/lower_pct must be positive, got {upper_pct}/{lower_pct}")

    n = len(close)
    entry = close.to_numpy()
    upper = entry * (1.0 + upper_pct)
    lower = entry * (1.0 - lower_pct)
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()

    label = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)

    for offset in range(1, horizon_bars + 1):
        valid_range = n - offset
        if valid_range <= 0:
            break
        fwd_high = np.full(n, np.nan)
        fwd_low = np.full(n, np.nan)
        fwd_high[:valid_range] = high_arr[offset : offset + valid_range]
        fwd_low[:valid_range] = low_arr[offset : offset + valid_range]

        touches_upper = (~resolved) & (fwd_high >= upper)
        touches_lower = (~resolved) & (fwd_low <= lower)
        both = touches_upper & touches_lower
        only_upper = touches_upper & ~both
        only_lower = touches_lower & ~both

        label[only_upper] = 1.0
        label[only_lower] = -1.0
        label[both] = -1.0
        resolved |= touches_upper | touches_lower

    terminal_return = compute_future_return(close, horizon_bars)
    unresolved = ~resolved
    time_barrier_label = np.sign(terminal_return.to_numpy())
    label = np.where(unresolved, time_barrier_label, label)

    insufficient = terminal_return.isna().to_numpy() & unresolved
    label = np.where(insufficient, np.nan, label)

    return pd.Series(label, index=close.index)


def build_label(
    definition: LabelDefinition, *, close: pd.Series, high: pd.Series | None = None, low: pd.Series | None = None
) -> LabelResult:
    """Compute `definition` against `close` (and, for `TRIPLE_BARRIER`,
    `high`/`low`). This is the ONLY function in this module a caller
    (`dataset_builder`) needs -- it dispatches on `definition.kind`."""
    if definition.kind is LabelKind.FUTURE_RETURN:
        values = compute_future_return(close, definition.horizon_bars)
    elif definition.kind is LabelKind.FUTURE_LOG_RETURN:
        values = compute_future_log_return(close, definition.horizon_bars)
    elif definition.kind is LabelKind.BINARY_DIRECTION:
        threshold = float(definition.params.get("threshold", 0.0))  # type: ignore[arg-type]
        values = compute_binary_direction(close, definition.horizon_bars, threshold=threshold)
    elif definition.kind is LabelKind.VOL_ADJUSTED_RETURN:
        vol_window = int(str(definition.params.get("vol_window", 20)))
        values = compute_vol_adjusted_return(close, definition.horizon_bars, vol_window=vol_window)
    elif definition.kind is LabelKind.TRIPLE_BARRIER:
        if high is None or low is None:
            raise ValueError("build_label: TRIPLE_BARRIER requires both `high` and `low` series")
        upper_pct = float(definition.params.get("upper_pct", 0.01))  # type: ignore[arg-type]
        lower_pct = float(definition.params.get("lower_pct", 0.01))  # type: ignore[arg-type]
        values = build_triple_barrier_labels(
            high, low, close, horizon_bars=definition.horizon_bars, upper_pct=upper_pct, lower_pct=lower_pct
        )
    else:  # pragma: no cover - exhaustive over LabelKind
        raise AssertionError(f"Unhandled LabelKind: {definition.kind}")

    return LabelResult(
        values=values, is_valid=values.notna(), embargo_bars=definition.horizon_bars, definition=definition
    )


__all__ = [
    "LabelDefinition",
    "LabelKind",
    "LabelResult",
    "build_label",
    "build_triple_barrier_labels",
    "compute_binary_direction",
    "compute_future_log_return",
    "compute_future_return",
    "compute_vol_adjusted_return",
]
