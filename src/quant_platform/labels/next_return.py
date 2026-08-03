"""Next Return Labels (Milestone 11, Phase 3, Part B): the forward
return over a single, fixed horizon, computed via `pricing.
compute_forward_return`. `price_basis` is a REQUIRED, explicit argument
of `build_next_return_specification` -- there is no default; a caller
must always name Close->Close, Open->Close, Close->Open, or Mid->Mid."""

from __future__ import annotations

import pandas as pd

from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.labels.pricing import PriceBasis, compute_forward_return

__all__ = ["NEXT_RETURN_GENERATION_VERSION", "build_next_return_specification", "generate_next_return_labels"]

NEXT_RETURN_GENERATION_VERSION = "v1"


def generate_next_return_labels(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    price_basis = PriceBasis(specification.parameters["price_basis"])
    horizon_bars = int(str(specification.parameters["horizon_bars"]))
    return compute_forward_return(source_data, price_basis, horizon_bars)


def build_next_return_specification(
    *, price_basis: PriceBasis, horizon_bars: int, created_from_dataset: str, created_from_manifest: str,
    generation_version: str = NEXT_RETURN_GENERATION_VERSION,
) -> LabelSpecification:
    return build_label_specification(
        label_family=LabelFamily.NEXT_RETURN, generation_version=generation_version, price_basis=price_basis.value,
        prediction_horizon=f"{horizon_bars} bars", availability_rule=f"available at event_time + {horizon_bars} bars",
        reference_price=f"{price_basis.value} entry/exit price", event_time_rule="bar close time",
        generation_rule=f"forward return: exit_price[t+{horizon_bars}] / entry_price[t] - 1, price_basis={price_basis.value}",
        created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
        parameters={"price_basis": price_basis.value, "horizon_bars": horizon_bars},
    )
