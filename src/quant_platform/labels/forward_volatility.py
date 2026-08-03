"""Forward Volatility Labels (Milestone 11, Phase 3, Part B): the
realized volatility of the NEXT `horizon_bars` bars, via a pluggable
`volatility.VolatilityEstimatorFn` -- no estimator is privileged; this
module works identically regardless of which shipped (or caller-
supplied) estimator `volatility_estimator_reference` names.

Computed by taking the estimator's own trailing rolling statistic and
shifting it `-horizon_bars`: `estimator(source_data, horizon_bars)` at
row `t + horizon_bars` covers exactly rows `[t+1, t+horizon_bars]`, so
after the shift, row `t`'s label is "volatility of the NEXT
`horizon_bars` returns" -- never reaching beyond the configured horizon,
never including row `t` itself."""

from __future__ import annotations

import pandas as pd

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.labels.volatility import resolve_estimator_by_name

__all__ = ["FORWARD_VOLATILITY_GENERATION_VERSION", "build_forward_volatility_specification", "generate_forward_volatility_labels"]

FORWARD_VOLATILITY_GENERATION_VERSION = "v1"


def generate_forward_volatility_labels(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    horizon_bars = int(str(specification.parameters["horizon_bars"]))
    estimator = resolve_estimator_by_name(str(specification.parameters["volatility_estimator_reference"]))
    trailing = estimator(source_data, horizon_bars)
    return trailing.shift(-horizon_bars)


def build_forward_volatility_specification(
    *, horizon_bars: int, volatility_estimator_reference: str, created_from_dataset: str, created_from_manifest: str,
    generation_version: str = FORWARD_VOLATILITY_GENERATION_VERSION,
) -> LabelSpecification:
    if horizon_bars <= 0:
        raise LabelRequestError(f"horizon_bars must be positive, got {horizon_bars}", context={"horizon_bars": horizon_bars})
    resolve_estimator_by_name(volatility_estimator_reference)  # fail fast on an unknown reference

    return build_label_specification(
        label_family=LabelFamily.FORWARD_VOLATILITY, generation_version=generation_version, price_basis="close",
        prediction_horizon=f"{horizon_bars} bars", reference_price="estimator-dependent (returns or high/low range)",
        availability_rule=f"available at event_time + {horizon_bars} bars", event_time_rule="bar close time",
        generation_rule=f"volatility of the next {horizon_bars} bars' returns via estimator={volatility_estimator_reference}",
        created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
        parameters={"horizon_bars": horizon_bars, "volatility_estimator_reference": volatility_estimator_reference},
    )
