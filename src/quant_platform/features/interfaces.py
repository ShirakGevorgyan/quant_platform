"""Core interfaces for the feature engineering platform: the input
context every feature computes against, and the `Feature`/`FeatureDefinition`
shape a registered feature takes.

THE POINT-IN-TIME CONTRACT
--------------------------------------------------------------------------
Every row of `FeatureContext.base_df` has an implicit "availability
instant" -- `FeatureContext.availability_times[i]`, always
`base_df.open_time[i] + timeframe.duration` (the bar's CLOSE time, mirroring
`core.time_utils.compute_close_time` and `multiframe.cursor.TimeframeCursor`'s
own reveal rule). A feature's `compute` implementation may use:

  * `ctx.base_df` rows up to and including the current row (trailing
    windows only -- never `center=True`, never a negative `shift`).
  * `ctx.higher_timeframe_data` bars whose CLOSE time is `<=` the current
    row's availability instant (use `features.alignment.align_higher_timeframe`
    -- never a raw join against `open_time`).
  * `ctx.cross_asset_data`/`ctx.macro_data` observations whose OWN
    availability (a cross-asset bar's close time; a macro observation's
    RELEASE time, never its observation/reference period) is `<=` the
    current row's availability instant (use
    `features.alignment.as_of_join_external`).
  * `ctx.resolved_features` -- other already-computed features' OUTPUT
    series, which by construction (the engine computes features in
    dependency-sorted order, see `registry.FeatureRegistry.
    resolve_dependency_order`) are already point-in-time correct
    themselves.

A feature's `compute` must NEVER receive label/target data -- see
`engine.FeatureEngine.compute`'s reserved-column-prefix guard
(`core.exceptions.LabelLeakageError`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.models import FeatureSpec


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Everything a feature's `compute` callable is given. Constructed once
    per `FeatureEngine.compute` call and shared (read-mostly) across every
    feature in the resolved computation order -- `resolved_features` is the
    one field that accumulates as the engine proceeds."""

    base_df: pd.DataFrame
    symbol: str
    timeframe: Timeframe
    availability_times: pd.Series
    higher_timeframe_data: Mapping[Timeframe, pd.DataFrame] = field(default_factory=dict)
    cross_asset_data: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    macro_data: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    resolved_features: dict[str, pd.Series] = field(default_factory=dict)

    def require_feature(self, name: str) -> pd.Series:
        """Fetch an already-computed dependency's output series. Raising
        here (rather than a bare `KeyError`) gives a feature author an
        actionable message if they declared a dependency incorrectly or the
        engine's ordering guarantee were ever violated."""
        if name not in self.resolved_features:
            raise KeyError(
                f"Feature dependency {name!r} has not been computed yet -- this indicates the "
                "computation order did not respect declared feature_dependencies."
            )
        return self.resolved_features[name]


@runtime_checkable
class Feature(Protocol):
    """Structural interface a registered feature satisfies. Most features
    are constructed via `FeatureDefinition` (a plain callable + spec)
    rather than a hand-written class implementing this Protocol directly,
    but a class-based feature is equally valid as long as it exposes these
    two attributes/methods."""

    @property
    def spec(self) -> FeatureSpec: ...

    def compute(self, ctx: FeatureContext) -> pd.Series: ...


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """The concrete, registrable unit: immutable metadata (`spec`) plus the
    pure function that computes it. `compute` must return a `pd.Series`
    aligned 1:1 with `ctx.base_df`'s row order (same length, same implicit
    index position) -- `engine.FeatureEngine` validates this."""

    spec: FeatureSpec
    compute: Callable[[FeatureContext], pd.Series]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def version(self) -> str:
        return self.spec.version


__all__ = ["Feature", "FeatureContext", "FeatureDefinition"]
