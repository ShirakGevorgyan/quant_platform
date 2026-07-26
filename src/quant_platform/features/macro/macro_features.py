"""Macro/external-data features: rate/yield levels, period-over-period
changes, release-age, and staleness -- every value point-in-time-joined via
`alignment.as_of_join_external`, keyed on the observation's RELEASE
timestamp, never its observation/reference period. This is what makes "a
value observed for January but published in February must not appear in
January feature rows" true by construction -- see
`tests/unit/features/test_macro_alignment.py` for the direct proof.

Staleness is controlled by the as-of join's own `tolerance` parameter, not
by a second, redundant `MissingPolicySpec` forward-fill: the as-of join
already carries a released value forward until either a newer release
supersedes it or `tolerance` is exceeded, so layering a feature-level
forward-fill on top would just duplicate (and potentially contradict) that
behavior. Callers who want a hard staleness cutoff should set `tolerance`;
`macro_{source}_is_stale` and `macro_{source}_release_age_days` make that
state explicit either way -- see Milestone 3 Section 4's "missing/stale
flags" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import as_of_join_external
from quant_platform.features.interfaces import FeatureContext, FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicySpec
from quant_platform.features.registry import FeatureRegistry


@dataclass(frozen=True, slots=True)
class MacroSourceConfig:
    """One external/macro data source's identity and alignment settings.
    `source_name` is the key into `FeatureContext.macro_data`; `change_lookback`
    is expressed in RELEASES of this source (e.g. `1` = month-over-month for
    a monthly series), never in base-timeframe bars."""

    source_name: str
    change_lookback: int = 1
    tolerance: pd.Timedelta | None = None


def _augmented_macro_df(df: pd.DataFrame, change_lookback: int) -> pd.DataFrame:
    augmented = df.copy()
    augmented["value_change"] = df["value"].diff(change_lookback)
    return augmented


def _joined(ctx: FeatureContext, *, config: MacroSourceConfig) -> pd.DataFrame:
    df = ctx.macro_data[config.source_name]
    augmented = _augmented_macro_df(df, config.change_lookback)
    level = as_of_join_external(
        ctx.availability_times, augmented, value_column="value", tolerance=config.tolerance, output_name="level"
    )
    change = as_of_join_external(
        ctx.availability_times, augmented, value_column="value_change", tolerance=config.tolerance,
        output_name="change",
    )
    return pd.concat([level, change], axis=1)


def register_macro_features(registry: FeatureRegistry, *, base_timeframe: Timeframe, config: MacroSourceConfig) -> None:
    """Register the level/change/release-age/staleness feature family for
    one macro source. Call once per external series (Fed Funds rate, CPI
    YoY, 10Y yield, ...)."""
    null_policy = MissingPolicySpec()
    source = config.source_name

    def _register(suffix: str, *, description: str, compute_fn: object, output_dtype: str = "float64") -> None:
        spec = FeatureSpec(
            name=f"macro_{source}_{suffix}", version="1", description=description, category=FeatureCategory.MACRO,
            required_inputs=("value", "release_time"), source_symbols=(), source_timeframe=base_timeframe,
            output_dtype=output_dtype, lookback_bars=0, warmup_bars=0, null_policy=null_policy,
            deterministic_params={"source_name": source, "change_lookback": config.change_lookback},
        )
        registry.register(FeatureDefinition(spec=spec, compute=compute_fn))  # type: ignore[arg-type]

    _register(
        "level", description=f"Most recently RELEASED {source} value as of each row's availability instant.",
        compute_fn=lambda ctx: _joined(ctx, config=config)["level"],
    )
    _register(
        f"change_{config.change_lookback}",
        description=f"{source} value change over {config.change_lookback} release(s), as of each row's availability instant.",
        compute_fn=lambda ctx: _joined(ctx, config=config)["change"],
    )
    _register(
        "release_age_days", description=f"Days since the {source} value currently visible was released.",
        compute_fn=lambda ctx: _joined(ctx, config=config)["level_age_seconds"] / 86400.0,
    )
    _register(
        "is_stale", description=f"True if no {source} release has ever occurred as of this row (or none within tolerance).",
        compute_fn=lambda ctx: _joined(ctx, config=config)["level_is_stale"], output_dtype="bool",
    )


__all__ = ["MacroSourceConfig", "register_macro_features"]
