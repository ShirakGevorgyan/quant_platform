"""Pluggable volatility estimator contract (Milestone 11, Phase 3, Part
B) -- shared by `triple_barrier.py` (optional volatility-scaled barrier
width) and `forward_volatility.py` (the label itself). No estimator is
privileged: `VolatilityEstimatorFn` is a plain, structural contract, and
this module ships two independent, equally-valid implementations to
prove genuine pluggability rather than a single hardcoded formula
dressed up as configurable.

Both estimators are PAST-only when applied directly (a trailing
`.rolling()` window ending at row `t`) -- `forward_volatility.py` is
responsible for shifting the result to express "volatility over the
NEXT `horizon_bars` returns," never this module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelRequestError

__all__ = [
    "REALIZED_PARKINSON_ESTIMATOR_NAME",
    "REALIZED_STDDEV_ESTIMATOR_NAME",
    "VolatilityEstimatorFn",
    "realized_parkinson_estimator",
    "realized_stddev_estimator",
    "resolve_estimator_by_name",
]

VolatilityEstimatorFn = Callable[[pd.DataFrame, int], pd.Series]
"""`(source_data, window_bars) -> rolling volatility series`, aligned
1:1 with `source_data`'s row order, trailing-window (row `t`'s value
depends only on rows `<= t`)."""

REALIZED_STDDEV_ESTIMATOR_NAME = "realized_stddev_v1"
REALIZED_PARKINSON_ESTIMATOR_NAME = "realized_parkinson_v1"


def realized_stddev_estimator(source_data: pd.DataFrame, window_bars: int) -> pd.Series:
    """Trailing standard deviation of simple close-to-close returns over
    `window_bars` -- the textbook realized-volatility estimator."""
    if window_bars <= 0:
        raise LabelRequestError(f"window_bars must be positive, got {window_bars}", context={"window_bars": window_bars})
    if "close" not in source_data.columns:
        raise LabelRequestError("source_data is missing required column 'close'", context={"missing_columns": ["close"]})
    returns = source_data["close"].pct_change()
    return returns.rolling(window=window_bars, min_periods=window_bars).std()


def realized_parkinson_estimator(source_data: pd.DataFrame, window_bars: int) -> pd.Series:
    """Parkinson (1980) range-based estimator: `sqrt(mean((ln(high/low))^2)
    / (4 * ln(2)))` over a trailing window -- uses the high/low RANGE
    rather than close-to-close returns, a genuinely different (not a
    disguised copy of) estimator, demonstrating real pluggability."""
    if window_bars <= 0:
        raise LabelRequestError(f"window_bars must be positive, got {window_bars}", context={"window_bars": window_bars})
    missing = [c for c in ("high", "low") if c not in source_data.columns]
    if missing:
        raise LabelRequestError(f"source_data is missing column(s) {missing} required for the Parkinson estimator", context={"missing_columns": missing})
    log_range_sq = pd.Series(np.log(source_data["high"] / source_data["low"]) ** 2, index=source_data.index)
    mean_log_range_sq = log_range_sq.rolling(window=window_bars, min_periods=window_bars).mean()
    result: pd.Series = np.sqrt(mean_log_range_sq / (4.0 * np.log(2.0)))
    return result


_ESTIMATORS_BY_NAME: dict[str, VolatilityEstimatorFn] = {
    REALIZED_STDDEV_ESTIMATOR_NAME: realized_stddev_estimator,
    REALIZED_PARKINSON_ESTIMATOR_NAME: realized_parkinson_estimator,
}


def resolve_estimator_by_name(name: str) -> VolatilityEstimatorFn:
    """Looks up a SHIPPED reference estimator by its stable name (as
    recorded in `LabelSpecification.parameters["volatility_estimator_reference"]`)
    -- a convenience for tests/replay that reconstruct a `LabelDefinition`
    from a specification alone. A caller supplying their own estimator
    callable directly (never registered here) is equally valid; this
    registry is not the only sanctioned source of one."""
    try:
        return _ESTIMATORS_BY_NAME[name]
    except KeyError:
        raise LabelRequestError(
            f"Unknown volatility estimator name {name!r}; known names: {sorted(_ESTIMATORS_BY_NAME)}", context={"estimator_name": name},
        ) from None
