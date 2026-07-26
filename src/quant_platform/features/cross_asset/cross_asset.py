"""Cross-asset features: another instrument's (DXY, oil, silver, ...)
returns/momentum/volatility, backward-aligned into the base symbol's
timeframe via `alignment.align_higher_timeframe` (the same close-time-based
alignment mechanism used for genuine higher-timeframe data -- a cross-asset
feed is just another `(symbol, timeframe)` series that needs the identical
"never reveal before its own bar has closed" treatment).

Rolling correlation is computed over the BASE timeframe's own bar cadence,
using the cross-asset return aligned (and necessarily repeated, since a
same-cadence cross-asset bar that hasn't updated yet keeps its last-known
value) into that cadence. This is a documented, deliberate simplification:
if the cross-asset series' native cadence is coarser than the base
timeframe, the repeated values mechanically inflate the apparent rolling
correlation relative to a "true" same-cadence comparison. It is still
POINT-IN-TIME CORRECT (never reflects a value before it was actually
available) -- the caveat is about correlation magnitude interpretation,
not leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import FeatureComputationError
from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe
from quant_platform.features.interfaces import FeatureContext, FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicySpec
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import momentum, simple_return, trailing_rolling


@dataclass(frozen=True, slots=True)
class CrossAssetWindows:
    return_window: int = 5
    momentum_window: int = 10
    volatility_window: int = 20
    correlation_window: int = 30


def _augmented_cross_asset_df(df: pd.DataFrame, windows: CrossAssetWindows) -> pd.DataFrame:
    if "close" not in df.columns:
        raise FeatureComputationError(
            "Cross-asset features require the cross-asset DataFrame to have a 'close' column "
            "(e.g. via historical.loader.DatasetLoader.load_for_engine).",
            context={"columns": list(df.columns)},
        )
    augmented = df.copy()
    close = df["close"]
    one_bar_return = simple_return(close, 1)
    augmented["return_1"] = one_bar_return
    augmented[f"return_{windows.return_window}"] = simple_return(close, windows.return_window)
    augmented[f"momentum_{windows.momentum_window}"] = momentum(close, windows.momentum_window)
    augmented[f"volatility_{windows.volatility_window}"] = trailing_rolling(
        one_bar_return, windows.volatility_window
    ).std()
    return augmented


def _aligned(
    ctx: FeatureContext, *, cross_asset_symbol: str, cross_asset_timeframe: Timeframe, windows: CrossAssetWindows
) -> pd.DataFrame:
    df = ctx.cross_asset_data[cross_asset_symbol]
    augmented = _augmented_cross_asset_df(df, windows)
    return align_higher_timeframe(ctx.availability_times, augmented, cross_asset_timeframe)


def register_cross_asset_features(
    registry: FeatureRegistry,
    *,
    base_timeframe: Timeframe,
    base_momentum_window: int,
    base_volatility_window: int,
    cross_asset_symbol: str,
    cross_asset_timeframe: Timeframe,
    windows: CrossAssetWindows | None = None,
) -> None:
    """Register the cross-asset feature family for one `cross_asset_symbol`
    (e.g. "DXY", "XAGUSD", "USOIL"). Call once per cross-asset instrument
    the research dataset should include."""
    w = windows if windows is not None else CrossAssetWindows()
    prefix = f"htf_{cross_asset_timeframe.value}_"
    null_policy = MissingPolicySpec()

    def _register(suffix: str, *, warmup_bars: int, description: str, compute_fn: object) -> None:
        spec = FeatureSpec(
            name=f"cross_{cross_asset_symbol.lower()}_{suffix}", version="1", description=description,
            category=FeatureCategory.CROSS_ASSET, required_inputs=("close",),
            source_symbols=(cross_asset_symbol,), source_timeframe=base_timeframe, output_dtype="float64",
            lookback_bars=warmup_bars, warmup_bars=warmup_bars, null_policy=null_policy,
            deterministic_params={"cross_asset_symbol": cross_asset_symbol},
        )
        registry.register(FeatureDefinition(spec=spec, compute=compute_fn))  # type: ignore[arg-type]

    _register(
        f"return_{w.return_window}", warmup_bars=w.return_window,
        description=f"{cross_asset_symbol} return over {w.return_window} of its own bars, aligned backward.",
        compute_fn=lambda ctx: _aligned(
            ctx, cross_asset_symbol=cross_asset_symbol, cross_asset_timeframe=cross_asset_timeframe, windows=w
        )[f"{prefix}return_{w.return_window}"],
    )

    def _relative_momentum(ctx: FeatureContext) -> pd.Series:
        base_mom = momentum(ctx.base_df["close"], base_momentum_window)
        cross_mom = _aligned(
            ctx, cross_asset_symbol=cross_asset_symbol, cross_asset_timeframe=cross_asset_timeframe, windows=w
        )[f"{prefix}momentum_{w.momentum_window}"]
        return base_mom - cross_mom

    _register(
        "relative_momentum", warmup_bars=max(base_momentum_window, w.momentum_window),
        description=f"Base symbol momentum minus {cross_asset_symbol} momentum.",
        compute_fn=_relative_momentum,
    )

    def _volatility_ratio(ctx: FeatureContext) -> pd.Series:
        base_ret = simple_return(ctx.base_df["close"], 1)
        base_vol = trailing_rolling(base_ret, base_volatility_window).std()
        cross_vol = _aligned(
            ctx, cross_asset_symbol=cross_asset_symbol, cross_asset_timeframe=cross_asset_timeframe, windows=w
        )[f"{prefix}volatility_{w.volatility_window}"]
        result: pd.Series = (base_vol / cross_vol).where(cross_vol != 0)
        return result

    _register(
        "volatility_ratio", warmup_bars=max(base_volatility_window, w.volatility_window) + 1,
        description=f"Base symbol rolling volatility divided by {cross_asset_symbol} rolling volatility.",
        compute_fn=_volatility_ratio,
    )

    def _rolling_correlation(ctx: FeatureContext) -> pd.Series:
        base_ret = simple_return(ctx.base_df["close"], 1)
        cross_ret = _aligned(
            ctx, cross_asset_symbol=cross_asset_symbol, cross_asset_timeframe=cross_asset_timeframe, windows=w
        )[f"{prefix}return_1"]
        return base_ret.rolling(window=w.correlation_window, min_periods=w.correlation_window).corr(cross_ret)

    _register(
        f"rolling_correlation_{w.correlation_window}", warmup_bars=w.correlation_window + 1,
        description=(
            f"Rolling correlation over {w.correlation_window} base bars between base 1-bar returns and "
            f"{cross_asset_symbol}'s aligned 1-bar returns."
        ),
        compute_fn=_rolling_correlation,
    )


__all__ = ["CrossAssetWindows", "register_cross_asset_features"]
