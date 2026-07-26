"""`FeatureEngine`: resolves a requested feature set's dependency order and
computes every feature against one shared `FeatureContext`.

STRUCTURAL LABEL-LEAKAGE GUARD
--------------------------------------------------------------------------
`compute` refuses (raising `LabelLeakageError`) any `base_df` that already
contains a column named `label_*` or `target_*` -- these prefixes are
reserved for `features.labels` output. Since `features.labels` is never
imported by this module or by any feature-group module, and this guard
exists at the single entry point every feature computation passes through,
label information cannot reach a feature's `compute` callable even by
caller error.

AVAILABILITY DELAY
--------------------------------------------------------------------------
A feature's `FeatureSpec.availability_delay` (e.g. a conservative padding
for a data source with a known reporting lag not otherwise captured by an
as-of join) is applied by shifting the computed series forward by
`ceil(availability_delay / timeframe.duration)` rows -- coarser than the
requested delay would allow if it does not evenly divide the timeframe,
which is the conservative (never LESS available than requested) direction
to round.

TRAINING_STATISTIC_FILL is intentionally NOT applied here -- it requires a
fitted statistic that can only be computed from a training split, which
does not exist yet at feature-computation time (features are computed
across the FULL requested range, before splitting). Those columns are left
null by the engine; `dataset_builder.ResearchDatasetBuilder` applies them
after splitting, using only the training partition -- see its module
docstring.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from quant_platform.core.exceptions import (
    FeatureComputationError,
    LabelLeakageError,
    PointInTimeViolationError,
)
from quant_platform.core.types import Timeframe
from quant_platform.features.interfaces import FeatureContext
from quant_platform.features.lineage import FeatureLineage, build_lineage
from quant_platform.features.missing import apply_missing_policy
from quant_platform.features.models import MissingPolicyKind
from quant_platform.features.registry import FeatureRegistry
from quant_platform.historical.timezones import require_utc

_RESERVED_PREFIXES = ("label_", "target_")


def _reject_label_columns(df: pd.DataFrame) -> None:
    reserved = [c for c in df.columns if any(str(c).startswith(p) for p in _RESERVED_PREFIXES)]
    if reserved:
        raise LabelLeakageError(
            f"Input DataFrame contains reserved label/target column(s) {reserved} -- features must "
            "never be computed from a DataFrame that already includes label/target data. Build "
            "features and labels separately (features.labels) and join them only after both are "
            "independently computed.",
            context={"reserved_columns": reserved},
        )


@dataclass(frozen=True, slots=True)
class FeatureComputationResult:
    features: pd.DataFrame
    lineages: dict[str, FeatureLineage]
    warmup_bars: int
    missing_indicators: dict[str, pd.Series] = field(default_factory=dict)
    rows_to_drop: pd.Index = field(default_factory=lambda: pd.Index([], dtype="int64"))


class FeatureEngine:
    def __init__(self, registry: FeatureRegistry) -> None:
        self._registry = registry

    def compute(
        self,
        *,
        base_df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        feature_names: Sequence[str],
        higher_timeframe_data: Mapping[Timeframe, pd.DataFrame] | None = None,
        cross_asset_data: Mapping[str, pd.DataFrame] | None = None,
        macro_data: Mapping[str, pd.DataFrame] | None = None,
        source_dataset_manifest_id: str | None = None,
    ) -> FeatureComputationResult:
        _reject_label_columns(base_df)
        if "open_time" not in base_df.columns:
            raise FeatureComputationError("base_df is missing an 'open_time' column")
        require_utc(base_df["open_time"], context="FeatureEngine.compute")
        if len(base_df) > 1:
            if not base_df["open_time"].is_monotonic_increasing:
                raise PointInTimeViolationError(
                    "FeatureEngine.compute: base_df.open_time must be sorted ascending -- computing "
                    "features against unsorted data would silently mix past and future rows in every "
                    "rolling window."
                )
            if base_df["open_time"].duplicated().any():
                raise PointInTimeViolationError(
                    "FeatureEngine.compute: base_df.open_time contains duplicate values."
                )

        base_reset = base_df.reset_index(drop=True)
        availability_times = (base_reset["open_time"] + timeframe.duration).reset_index(drop=True)
        ctx = FeatureContext(
            base_df=base_reset, symbol=symbol, timeframe=timeframe, availability_times=availability_times,
            higher_timeframe_data=higher_timeframe_data or {}, cross_asset_data=cross_asset_data or {},
            macro_data=macro_data or {},
        )

        order = self._registry.resolve_dependency_order(feature_names)
        n = len(base_reset)
        features = pd.DataFrame(index=range(n))
        lineages: dict[str, FeatureLineage] = {}
        missing_indicators: dict[str, pd.Series] = {}
        drop_rows_union: set[int] = set()
        max_warmup = 0

        for name in order:
            definition = self._registry.get(name)
            spec = definition.spec
            raw_output = definition.compute(ctx)
            raw_series = pd.Series(raw_output).reset_index(drop=True)
            if len(raw_series) != n:
                raise FeatureComputationError(
                    f"Feature {name!r} returned {len(raw_series)} value(s), expected {n} (one per base_df row)",
                    context={"feature": name, "expected": n, "actual": len(raw_series)},
                )

            if spec.availability_delay > pd.Timedelta(0):
                shift_bars = math.ceil(spec.availability_delay / pd.Timedelta(timeframe.duration))
                raw_series = raw_series.shift(shift_bars)

            ctx.resolved_features[name] = raw_series

            if spec.null_policy.kind is MissingPolicyKind.TRAINING_STATISTIC_FILL:
                final_series = raw_series
            else:
                missing_result = apply_missing_policy(raw_series, policy=spec.null_policy)
                final_series = missing_result.values
                if missing_result.missing_indicator is not None:
                    missing_indicators[name] = missing_result.missing_indicator
                if missing_result.rows_to_drop is not None:
                    drop_rows_union.update(int(i) for i in missing_result.rows_to_drop)

            features[name] = final_series
            lineages[name] = build_lineage(
                spec, source_dataset_manifest_id=source_dataset_manifest_id, transformation=spec.description
            )
            max_warmup = max(max_warmup, spec.warmup_bars)

        # Only feature columns explicitly requested (not transitive
        # dependencies pulled in purely to satisfy the DAG) are returned --
        # a dependency-only feature is an implementation detail, not
        # something the caller asked for.
        requested = [name for name in order if name in set(feature_names)]
        return FeatureComputationResult(
            features=features[requested], lineages={n: lineages[n] for n in requested}, warmup_bars=max_warmup,
            missing_indicators=missing_indicators, rows_to_drop=pd.Index(sorted(drop_rows_union), dtype="int64"),
        )


__all__ = ["FeatureComputationResult", "FeatureEngine"]
