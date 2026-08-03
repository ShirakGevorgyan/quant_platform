"""Direction Labels (Milestone 11, Phase 3, Part B): UP/DOWN/NEUTRAL,
thresholded off the same `pricing.compute_forward_return` primitive
`next_return.py`/`multi_horizon_return.py` use -- independently computed
from raw `source_data`, never by reading either family's generated
output. `neutral_threshold` is a REQUIRED, explicit argument (no
default) and is always part of `LabelSpecification.parameters`, so it
participates in `parameter_hash` -> `label_specification_id` by
construction: any change to the threshold is automatically a new,
independently identified specification."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.labels.pricing import PriceBasis, compute_forward_return

__all__ = ["DIRECTION_GENERATION_VERSION", "Direction", "build_direction_specification", "generate_direction_labels"]

DIRECTION_GENERATION_VERSION = "v1"


class Direction(Enum):
    DOWN = -1.0
    NEUTRAL = 0.0
    UP = 1.0


def generate_direction_labels(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    price_basis = PriceBasis(specification.parameters["price_basis"])
    horizon_bars = int(str(specification.parameters["horizon_bars"]))
    neutral_threshold = float(specification.parameters["neutral_threshold"])  # type: ignore[arg-type]

    forward_return = compute_forward_return(source_data, price_basis, horizon_bars)
    direction = np.select(
        [forward_return > neutral_threshold, forward_return < -neutral_threshold],
        [Direction.UP.value, Direction.DOWN.value],
        default=Direction.NEUTRAL.value,
    )
    return pd.Series(direction, index=forward_return.index).where(forward_return.notna())


def build_direction_specification(
    *, price_basis: PriceBasis, horizon_bars: int, neutral_threshold: float, created_from_dataset: str, created_from_manifest: str,
    generation_version: str = DIRECTION_GENERATION_VERSION,
) -> LabelSpecification:
    if neutral_threshold < 0:
        raise LabelRequestError(f"neutral_threshold must be >= 0, got {neutral_threshold}", context={"neutral_threshold": neutral_threshold})
    return build_label_specification(
        label_family=LabelFamily.DIRECTION, generation_version=generation_version, price_basis=price_basis.value,
        prediction_horizon=f"{horizon_bars} bars", availability_rule=f"available at event_time + {horizon_bars} bars",
        reference_price=f"{price_basis.value} entry/exit price", event_time_rule="bar close time",
        generation_rule=(
            f"UP if forward_return > {neutral_threshold}, DOWN if forward_return < -{neutral_threshold}, else NEUTRAL; "
            f"price_basis={price_basis.value}"
        ),
        created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
        parameters={"price_basis": price_basis.value, "horizon_bars": horizon_bars, "neutral_threshold": neutral_threshold},
    )
