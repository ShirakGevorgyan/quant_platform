"""Triple Barrier Labels (Milestone 11, Phase 3, Part B): +1 if the
upper barrier is touched before the lower one within `max_holding_bars`;
-1 if the lower barrier is touched first (or both are touched within the
same forward bar -- OHLC alone cannot say which happened first within
one bar, so this resolves the tie toward the stop-like outcome); the
SIGN of the terminal return (the "time barrier" outcome) if neither is
touched within `max_holding_bars`. `NaN` where trailing volatility is
unavailable (insufficient warmup) or the full holding horizon's data
does not exist.

Deliberately reimplemented natively rather than importing `features.
labels.build_triple_barrier_labels` -- `labels/` imports nothing from
`features` (see `models.py`'s module docstring); the two systems are
independent by design (see `docs/labels_architecture.md`'s "Relationship
to features.labels" section)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.models import LabelFamily, LabelSpecification, build_label_specification
from quant_platform.labels.pricing import PriceBasis, compute_forward_return
from quant_platform.labels.volatility import resolve_estimator_by_name

__all__ = ["TRIPLE_BARRIER_GENERATION_VERSION", "build_triple_barrier_specification", "generate_triple_barrier_labels"]

TRIPLE_BARRIER_GENERATION_VERSION = "v1"


def generate_triple_barrier_labels(source_data: pd.DataFrame, specification: LabelSpecification) -> pd.Series:
    profit_multiplier = float(specification.parameters["profit_multiplier"])  # type: ignore[arg-type]
    loss_multiplier = float(specification.parameters["loss_multiplier"])  # type: ignore[arg-type]
    max_holding_bars = int(str(specification.parameters["max_holding_bars"]))
    volatility_window_bars = int(str(specification.parameters["volatility_window_bars"]))
    estimator = resolve_estimator_by_name(str(specification.parameters["volatility_estimator_reference"]))

    for column in ("close", "high", "low"):
        if column not in source_data.columns:
            raise LabelRequestError(f"source_data is missing required column {column!r}", context={"missing_columns": [column]})

    close, high, low = source_data["close"], source_data["high"], source_data["low"]
    trailing_vol = estimator(source_data, volatility_window_bars)  # PAST-only: row t depends only on rows <= t

    n = len(source_data)
    entry = close.to_numpy()
    vol = trailing_vol.to_numpy()
    vol_available = ~np.isnan(vol)
    upper = entry * (1.0 + profit_multiplier * vol)
    lower = entry * (1.0 - loss_multiplier * vol)
    high_arr, low_arr = high.to_numpy(), low.to_numpy()

    label = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)

    for offset in range(1, max_holding_bars + 1):
        valid_range = n - offset
        if valid_range <= 0:
            break
        fwd_high, fwd_low = np.full(n, np.nan), np.full(n, np.nan)
        fwd_high[:valid_range] = high_arr[offset : offset + valid_range]
        fwd_low[:valid_range] = low_arr[offset : offset + valid_range]

        touches_upper = (~resolved) & vol_available & (fwd_high >= upper)
        touches_lower = (~resolved) & vol_available & (fwd_low <= lower)
        both = touches_upper & touches_lower
        only_upper = touches_upper & ~both
        only_lower = touches_lower & ~both

        label[only_upper] = 1.0
        label[only_lower] = -1.0
        label[both] = -1.0
        resolved |= touches_upper | touches_lower

    terminal_return = compute_forward_return(source_data, PriceBasis.CLOSE_TO_CLOSE, max_holding_bars)
    unresolved = ~resolved & vol_available
    time_barrier_label = np.sign(terminal_return.to_numpy())
    label = np.where(unresolved, time_barrier_label, label)
    insufficient = terminal_return.isna().to_numpy() & unresolved
    label = np.where(insufficient, np.nan, label)
    label = np.where(~vol_available, np.nan, label)

    return pd.Series(label, index=source_data.index)


def build_triple_barrier_specification(
    *, profit_multiplier: float, loss_multiplier: float, max_holding_bars: int, volatility_window_bars: int,
    volatility_estimator_reference: str, created_from_dataset: str, created_from_manifest: str,
    generation_version: str = TRIPLE_BARRIER_GENERATION_VERSION,
) -> LabelSpecification:
    if profit_multiplier <= 0 or loss_multiplier <= 0:
        raise LabelRequestError(
            f"profit_multiplier/loss_multiplier must be positive, got {profit_multiplier}/{loss_multiplier}",
            context={"profit_multiplier": profit_multiplier, "loss_multiplier": loss_multiplier},
        )
    if max_holding_bars <= 0:
        raise LabelRequestError(f"max_holding_bars must be positive, got {max_holding_bars}", context={"max_holding_bars": max_holding_bars})
    resolve_estimator_by_name(volatility_estimator_reference)  # fail fast on an unknown reference

    return build_label_specification(
        label_family=LabelFamily.TRIPLE_BARRIER, generation_version=generation_version, price_basis="close",
        prediction_horizon=f"up to {max_holding_bars} bars", reference_price="close at event_time (entry); high/low for barrier touches",
        availability_rule=f"available no later than event_time + {max_holding_bars} bars (may resolve earlier via barrier touch)",
        event_time_rule="bar close time",
        generation_rule=(
            f"+1 if upper barrier touched first, -1 if lower touched first (or same-bar tie), else sign(terminal_return) at "
            f"the time barrier; barriers = close * (1 +/- multiplier * trailing_volatility); profit_multiplier={profit_multiplier}, "
            f"loss_multiplier={loss_multiplier}, volatility_estimator_reference={volatility_estimator_reference}"
        ),
        created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
        parameters={
            "profit_multiplier": profit_multiplier, "loss_multiplier": loss_multiplier, "max_holding_bars": max_holding_bars,
            "volatility_window_bars": volatility_window_bars, "volatility_estimator_reference": volatility_estimator_reference,
        },
    )
