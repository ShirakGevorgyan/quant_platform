"""Core price/technical features. A carefully selected initial library --
not an attempt at exhaustive indicator coverage (Milestone 3 Section 4
explicitly asks for a small, deliberate set, not hundreds of arbitrary
indicators).

EVERY rolling computation in this module goes through `trailing_rolling`, which
pins `center=False` explicitly (never relying on pandas' default) and
`min_periods=window` (so a feature's warm-up NaN row count always matches
its declared `FeatureSpec.warmup_bars` exactly, rather than silently
returning a partial-window value for the first `window - 1` rows). No
function in this module ever calls `.shift()` with a negative period, and
none uses `pandas.DataFrame.resample` -- both are the concrete mechanisms
Section 2 calls out as centered-window/forward-looking leak vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicySpec
from quant_platform.features.registry import FeatureRegistry


def trailing_rolling(series: pd.Series, window: int) -> pd.api.typing.Rolling[pd.Series]:
    return series.rolling(window=window, min_periods=window, center=False)


def simple_return(close: pd.Series, window: int) -> pd.Series:
    result: pd.Series = close.pct_change(periods=window)
    return result


def log_return(close: pd.Series, window: int) -> pd.Series:
    return pd.Series(np.log(close / close.shift(window)), index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    result: pd.Series = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return result


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    result: pd.Series = trailing_rolling(true_range(high, low, close), window).mean()
    return result


def rolling_high_low_distance(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Position of `close` within the trailing `[window]`-bar high/low
    range, in `[0, 1]` (0 = at the rolling low, 1 = at the rolling high);
    NaN when the rolling range is zero (a frozen/flat market)."""
    roll_high = trailing_rolling(high, window).max()
    roll_low = trailing_rolling(low, window).min()
    span = roll_high - roll_low
    result: pd.Series = ((close - roll_low) / span).where(span != 0)
    return result


def candle_body_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    span = high - low
    result: pd.Series = ((close - open_).abs() / span).where(span != 0)
    return result


def candle_upper_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    span = high - low
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    result: pd.Series = ((high - body_top) / span).where(span != 0)
    return result


def candle_lower_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    span = high - low
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    result: pd.Series = ((body_bottom - low) / span).where(span != 0)
    return result


def momentum(close: pd.Series, window: int) -> pd.Series:
    result: pd.Series = close - close.shift(window)
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = trailing_rolling(series, window).mean()
    std = trailing_rolling(series, window).std()
    result: pd.Series = ((series - mean) / std).where(std != 0)
    return result


def moving_average_distance(close: pd.Series, window: int) -> pd.Series:
    ma = trailing_rolling(close, window).mean()
    result: pd.Series = ((close - ma) / ma).where(ma != 0)
    return result


def rolling_volume_mean(volume: pd.Series, window: int) -> pd.Series:
    result: pd.Series = trailing_rolling(volume, window).mean()
    return result


def rolling_volume_std(volume: pd.Series, window: int) -> pd.Series:
    result: pd.Series = trailing_rolling(volume, window).std()
    return result


def rolling_spread_mean(spread: pd.Series, window: int) -> pd.Series:
    result: pd.Series = trailing_rolling(spread, window).mean()
    return result


@dataclass(frozen=True, slots=True)
class TechnicalWindows:
    """Configurable window lengths for the core technical feature family --
    part of `config.feature_schemas`'s typed configuration surface (Section
    15). Defaults are deliberately modest, matching Section 4's "carefully
    selected", not exhaustive, feature library."""

    return_windows: tuple[int, ...] = (1, 5, 15)
    momentum_windows: tuple[int, ...] = (10, 20)
    volatility_window: int = 20
    zscore_window: int = 20
    ma_distance_windows: tuple[int, ...] = (20, 50)
    high_low_distance_window: int = 20
    atr_window: int = 14
    volume_window: int = 20
    include_spread: bool = False
    spread_window: int = 20
    null_policy: MissingPolicySpec = field(default_factory=lambda: MissingPolicySpec())


def register_core_technical_features(
    registry: FeatureRegistry, *, timeframe: Timeframe, windows: TechnicalWindows | None = None
) -> None:
    """Register the full core technical/price feature family against
    `timeframe`. Each feature reads only `ctx.base_df`'s OHLCV(+volume,
    +optional spread) columns up to and including the current row --
    nothing here ever touches `ctx.higher_timeframe_data`, `ctx.
    cross_asset_data`, or `ctx.macro_data`."""
    w = windows if windows is not None else TechnicalWindows()

    def _register(
        name: str, *, required_inputs: tuple[str, ...], lookback: int, compute_fn: object, description: str
    ) -> None:
        spec = FeatureSpec(
            name=name, version="1", description=description, category=FeatureCategory.PRICE,
            required_inputs=required_inputs, source_symbols=(), source_timeframe=timeframe,
            output_dtype="float64", lookback_bars=lookback, warmup_bars=lookback, null_policy=w.null_policy,
            deterministic_params={"window": lookback} if lookback else {},
        )
        registry.register(FeatureDefinition(spec=spec, compute=compute_fn))  # type: ignore[arg-type]

    for window in w.return_windows:
        _register(
            f"return_simple_{window}", required_inputs=("close",), lookback=window,
            compute_fn=lambda ctx, _w=window: simple_return(ctx.base_df["close"], _w),
            description=f"Simple percentage return over {window} bar(s).",
        )
        _register(
            f"return_log_{window}", required_inputs=("close",), lookback=window,
            compute_fn=lambda ctx, _w=window: log_return(ctx.base_df["close"], _w),
            description=f"Log return over {window} bar(s).",
        )

    for window in w.momentum_windows:
        _register(
            f"momentum_{window}", required_inputs=("close",), lookback=window,
            compute_fn=lambda ctx, _w=window: momentum(ctx.base_df["close"], _w),
            description=f"close - close.shift({window}).",
        )

    _register(
        f"rolling_volatility_{w.volatility_window}", required_inputs=("close",), lookback=w.volatility_window + 1,
        compute_fn=lambda ctx, _w=w.volatility_window: trailing_rolling(
            simple_return(ctx.base_df["close"], 1), _w
        ).std(),
        description=f"Rolling std of 1-bar returns over {w.volatility_window} bars.",
    )
    _register(
        f"rolling_zscore_close_{w.zscore_window}", required_inputs=("close",), lookback=w.zscore_window,
        compute_fn=lambda ctx, _w=w.zscore_window: rolling_zscore(ctx.base_df["close"], _w),
        description=f"Rolling z-score of close over {w.zscore_window} bars.",
    )
    for window in w.ma_distance_windows:
        _register(
            f"ma_distance_{window}", required_inputs=("close",), lookback=window,
            compute_fn=lambda ctx, _w=window: moving_average_distance(ctx.base_df["close"], _w),
            description=f"Fractional distance of close from its {window}-bar moving average.",
        )
    _register(
        f"high_low_distance_{w.high_low_distance_window}", required_inputs=("high", "low", "close"),
        lookback=w.high_low_distance_window,
        compute_fn=lambda ctx, _w=w.high_low_distance_window: rolling_high_low_distance(
            ctx.base_df["close"], ctx.base_df["high"], ctx.base_df["low"], _w
        ),
        description=f"Position of close within its {w.high_low_distance_window}-bar high/low range.",
    )
    _register(
        f"atr_{w.atr_window}", required_inputs=("high", "low", "close"), lookback=w.atr_window + 1,
        compute_fn=lambda ctx, _w=w.atr_window: average_true_range(
            ctx.base_df["high"], ctx.base_df["low"], ctx.base_df["close"], _w
        ),
        description=f"Average True Range over {w.atr_window} bars.",
    )
    _register(
        "candle_body_ratio", required_inputs=("open", "high", "low", "close"), lookback=0,
        compute_fn=lambda ctx: candle_body_ratio(
            ctx.base_df["open"], ctx.base_df["high"], ctx.base_df["low"], ctx.base_df["close"]
        ),
        description="abs(close - open) / (high - low).",
    )
    _register(
        "candle_upper_wick_ratio", required_inputs=("open", "high", "low", "close"), lookback=0,
        compute_fn=lambda ctx: candle_upper_wick_ratio(
            ctx.base_df["open"], ctx.base_df["high"], ctx.base_df["low"], ctx.base_df["close"]
        ),
        description="(high - max(open, close)) / (high - low).",
    )
    _register(
        "candle_lower_wick_ratio", required_inputs=("open", "high", "low", "close"), lookback=0,
        compute_fn=lambda ctx: candle_lower_wick_ratio(
            ctx.base_df["open"], ctx.base_df["high"], ctx.base_df["low"], ctx.base_df["close"]
        ),
        description="(min(open, close) - low) / (high - low).",
    )
    _register(
        f"rolling_volume_mean_{w.volume_window}", required_inputs=("volume",), lookback=w.volume_window,
        compute_fn=lambda ctx, _w=w.volume_window: rolling_volume_mean(ctx.base_df["volume"], _w),
        description=f"Rolling mean volume over {w.volume_window} bars.",
    )
    _register(
        f"rolling_volume_std_{w.volume_window}", required_inputs=("volume",), lookback=w.volume_window,
        compute_fn=lambda ctx, _w=w.volume_window: rolling_volume_std(ctx.base_df["volume"], _w),
        description=f"Rolling std of volume over {w.volume_window} bars.",
    )
    if w.include_spread:
        _register(
            f"rolling_spread_mean_{w.spread_window}", required_inputs=("spread",), lookback=w.spread_window,
            compute_fn=lambda ctx, _w=w.spread_window: rolling_spread_mean(ctx.base_df["spread"], _w),
            description=f"Rolling mean quoted spread over {w.spread_window} bars.",
        )


__all__ = [
    "TechnicalWindows",
    "average_true_range",
    "candle_body_ratio",
    "candle_lower_wick_ratio",
    "candle_upper_wick_ratio",
    "log_return",
    "momentum",
    "moving_average_distance",
    "register_core_technical_features",
    "rolling_high_low_distance",
    "rolling_spread_mean",
    "rolling_volume_mean",
    "rolling_volume_std",
    "rolling_zscore",
    "simple_return",
    "trailing_rolling",
    "true_range",
]
