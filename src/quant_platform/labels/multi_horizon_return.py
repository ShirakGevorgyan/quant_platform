"""Multi Horizon Return Labels (Milestone 11, Phase 3, Part B): the SAME
forward-return computation `next_return.py` uses (`pricing.
compute_forward_return`), independently parameterized per horizon --
"horizons belong to `LabelSpecification`" means each requested horizon
produces its OWN, independently identified `LabelSpecification` (a
different `horizon_bars` changes `parameters` -> `parameter_hash` ->
`label_specification_id`), never one label smeared across many
horizons. Reusing the same pure return-computation primitive
`next_return.py` also uses is NOT "depending on Next Return's output" --
neither family ever reads the other's GENERATED VALUES; both compute
independently, in parallel, from the same raw `source_data`.

`MULTI_HORIZON_RETURN_MINIMUM_HORIZONS` names the 6 horizons the
governing specification requires be SUPPORTED at minimum -- callers are
never restricted to only these; "no hardcoded assumptions" means
`build_multi_horizon_return_specifications` accepts ANY horizon
sequence."""

from __future__ import annotations

import pandas as pd

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.labels.pricing import PriceBasis, compute_forward_return

__all__ = [
    "MULTI_HORIZON_RETURN_GENERATION_VERSION",
    "MULTI_HORIZON_RETURN_MINIMUM_HORIZONS",
    "build_multi_horizon_return_specifications",
    "generate_multi_horizon_return_labels",
]

MULTI_HORIZON_RETURN_GENERATION_VERSION = "v1"
MULTI_HORIZON_RETURN_MINIMUM_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 50, 100)


def generate_multi_horizon_return_labels(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    """One horizon's worth of forward return -- a single
    `LabelSpecification` (and therefore a single `LabelDefinition`/
    `LabelBundle`) covers exactly one horizon; a caller wanting the full
    multi-horizon set builds one `LabelDefinition` per specification
    returned by `build_multi_horizon_return_specifications` and calls
    `builder.LabelBuilder.build` once per horizon."""
    price_basis = PriceBasis(specification.parameters["price_basis"])
    horizon_bars = int(str(specification.parameters["horizon_bars"]))
    return compute_forward_return(source_data, price_basis, horizon_bars)


def build_multi_horizon_return_specifications(
    *, horizons: tuple[int, ...], price_basis: PriceBasis, created_from_dataset: str, created_from_manifest: str,
    generation_version: str = MULTI_HORIZON_RETURN_GENERATION_VERSION,
) -> tuple[LabelSpecification, ...]:
    if not horizons:
        raise LabelRequestError("horizons must not be empty", context={"horizons": horizons})
    return tuple(
        build_label_specification(
            label_family=LabelFamily.MULTI_HORIZON_RETURN, generation_version=generation_version, price_basis=price_basis.value,
            prediction_horizon=f"{horizon_bars} bars", availability_rule=f"available at event_time + {horizon_bars} bars",
            reference_price=f"{price_basis.value} entry/exit price", event_time_rule="bar close time",
            generation_rule=f"forward return: exit_price[t+{horizon_bars}] / entry_price[t] - 1, price_basis={price_basis.value}",
            created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
            parameters={"price_basis": price_basis.value, "horizon_bars": horizon_bars},
        )
        for horizon_bars in horizons
    )
