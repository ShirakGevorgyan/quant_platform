"""Multi-timeframe features: higher-timeframe completed-bar OHLCV, returns,
volatility, trend distance, and elapsed time since the last completed
higher-timeframe bar -- all reached through `alignment.align_higher_timeframe`,
never a raw join.

ORDERING MATTERS: return/volatility/trend-distance are computed on the
higher-timeframe bar sequence FIRST (at its own native granularity, using
the same trailing-window functions as `features.technical.price`), and only
THEN aligned into the base timeframe. Computing them the other way around
-- aligning the raw higher-timeframe close into the base timeframe (which
necessarily repeats the same value across every base bar until the next
higher-timeframe bar closes) and THEN taking a rolling return/diff on that
upsampled series -- would measure change over N BASE bars of repeated
values, not N higher-timeframe bars, which is a different (and for
"higher-timeframe return", wrong) quantity entirely.

Each registered feature independently recomputes the augmented/aligned
frame (not cached across the ~8 features in this family) -- a deliberate
simplicity-over-micro-optimization tradeoff; `alignment.align_higher_timeframe`
is an O(n log m) vectorized operation, so the repeated cost is modest (see
`tests/performance/test_feature_throughput.py` for measured numbers), and
every feature's `compute` stays a pure, independent function of `ctx`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import FeatureComputationError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.features.alignment import align_higher_timeframe
from quant_platform.features.interfaces import FeatureContext, FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicySpec
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import simple_return, trailing_rolling


@dataclass(frozen=True, slots=True)
class MultiTimeframeWindows:
    return_window: int = 3
    volatility_window: int = 10
    trend_window: int = 10


def _augmented_higher_df(higher_df: pd.DataFrame, windows: MultiTimeframeWindows) -> pd.DataFrame:
    missing = set(OHLCV_COLUMNS) - set(higher_df.columns)
    if missing:
        raise FeatureComputationError(
            f"Multi-timeframe features require higher_timeframe_data to be projected to "
            f"core.types.OHLCV_COLUMNS (e.g. via historical.loader.DatasetLoader.load_for_engine) -- "
            f"missing column(s) {sorted(missing)}. A raw historical.models.RAW_HISTORICAL_COLUMNS "
            "frame (with tick_volume/real_volume/spread instead of volume) will not work directly.",
            context={"missing_columns": sorted(missing)},
        )
    augmented = higher_df.copy()
    close = higher_df["close"]
    augmented[f"return_{windows.return_window}"] = simple_return(close, windows.return_window)
    one_bar_return = simple_return(close, 1)
    augmented[f"volatility_{windows.volatility_window}"] = trailing_rolling(
        one_bar_return, windows.volatility_window
    ).std()
    ma = trailing_rolling(close, windows.trend_window).mean()
    augmented[f"trend_distance_{windows.trend_window}"] = ((close - ma) / ma).where(ma != 0)
    return augmented


def _aligned(ctx: FeatureContext, *, higher_timeframe: Timeframe, windows: MultiTimeframeWindows) -> pd.DataFrame:
    higher_df = ctx.higher_timeframe_data[higher_timeframe]
    augmented = _augmented_higher_df(higher_df, windows)
    return align_higher_timeframe(ctx.availability_times, augmented, higher_timeframe)


def register_multi_timeframe_features(
    registry: FeatureRegistry,
    *,
    base_timeframe: Timeframe,
    higher_timeframe: Timeframe,
    windows: MultiTimeframeWindows | None = None,
) -> None:
    """Register the higher-timeframe feature family for one
    `(base_timeframe, higher_timeframe)` pair. `higher_timeframe` must be
    strictly coarser than `base_timeframe` (the same invariant
    `historical.resampling.resample_ohlcv` enforces)."""
    if higher_timeframe.duration <= base_timeframe.duration:
        raise ValueError(
            f"higher_timeframe ({higher_timeframe.value}) must be strictly coarser than "
            f"base_timeframe ({base_timeframe.value})"
        )
    w = windows if windows is not None else MultiTimeframeWindows()
    htf = higher_timeframe.value
    bars_per_htf = math.ceil(higher_timeframe.duration / base_timeframe.duration)
    trend_warmup = (w.trend_window + 1) * bars_per_htf
    null_policy = MissingPolicySpec()

    def _register(suffix: str, *, warmup_bars: int, description: str, compute_fn: object) -> None:
        spec = FeatureSpec(
            name=f"htf_{htf}_{suffix}", version="1", description=description,
            category=FeatureCategory.MULTI_TIMEFRAME, required_inputs=("open_time", "open", "high", "low", "close"),
            source_symbols=(), source_timeframe=base_timeframe, output_dtype="float64",
            lookback_bars=warmup_bars, warmup_bars=warmup_bars, null_policy=null_policy,
            deterministic_params={"higher_timeframe": htf},
        )
        registry.register(FeatureDefinition(spec=spec, compute=compute_fn))  # type: ignore[arg-type]

    for field_name in ("open", "high", "low", "close", "volume"):
        _register(
            field_name, warmup_bars=bars_per_htf,
            description=f"Most recently CLOSED {htf} bar's {field_name}, as of each base row's availability instant.",
            compute_fn=lambda ctx, _f=field_name: _aligned(ctx, higher_timeframe=higher_timeframe, windows=w)[
                f"htf_{htf}_{_f}"
            ],
        )
    _register(
        f"return_{w.return_window}", warmup_bars=(w.return_window + 1) * bars_per_htf,
        description=f"Return of the last completed {htf} bar vs. {w.return_window} {htf} bars earlier.",
        compute_fn=lambda ctx: _aligned(ctx, higher_timeframe=higher_timeframe, windows=w)[
            f"htf_{htf}_return_{w.return_window}"
        ],
    )
    _register(
        f"volatility_{w.volatility_window}", warmup_bars=(w.volatility_window + 1) * bars_per_htf,
        description=f"Rolling std of {htf}-bar returns over {w.volatility_window} {htf} bars.",
        compute_fn=lambda ctx: _aligned(ctx, higher_timeframe=higher_timeframe, windows=w)[
            f"htf_{htf}_volatility_{w.volatility_window}"
        ],
    )
    _register(
        f"trend_distance_{w.trend_window}", warmup_bars=trend_warmup,
        description=f"Fractional distance of the last completed {htf} close from its {w.trend_window}-bar moving average.",
        compute_fn=lambda ctx: _aligned(ctx, higher_timeframe=higher_timeframe, windows=w)[
            f"htf_{htf}_trend_distance_{w.trend_window}"
        ],
    )
    _register(
        "seconds_since_close", warmup_bars=bars_per_htf,
        description=f"Seconds elapsed since the last completed {htf} bar's close, as of each base row.",
        compute_fn=lambda ctx: _aligned(ctx, higher_timeframe=higher_timeframe, windows=w)[
            f"htf_{htf}_seconds_since_close"
        ],
    )


__all__ = ["MultiTimeframeWindows", "register_multi_timeframe_features"]
