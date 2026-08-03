"""Shared, point-in-time-safe price-basis resolution and forward-return
computation (Milestone 11, Phase 3, Part B) -- reused by `next_return.py`,
`multi_horizon_return.py`, and `direction.py`. Not a label family of its
own; a pure, family-agnostic computational primitive, exactly like
`features.labels.compute_future_return` is reused across multiple
`LabelKind` values in the pre-existing (Milestone 3) label system. Reusing
this helper across multiple families is NOT "one family depending on
another family's output" -- no family ever reads another family's
GENERATED VALUES; they independently compute from the same raw
`source_data` using the same pure math.

PRICE BASIS -- explicit, never a hidden default. Every family that uses
this module requires the caller to name a `PriceBasis` explicitly."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import LabelRequestError

__all__ = ["PriceBasis", "compute_forward_return", "resolve_entry_exit_series"]


class PriceBasis(Enum):
    CLOSE_TO_CLOSE = "close_to_close"
    OPEN_TO_CLOSE = "open_to_close"
    CLOSE_TO_OPEN = "close_to_open"
    MID_TO_MID = "mid_to_mid"


_REQUIRED_COLUMNS: dict[PriceBasis, tuple[str, ...]] = {
    PriceBasis.CLOSE_TO_CLOSE: ("close",),
    PriceBasis.OPEN_TO_CLOSE: ("open", "close"),
    PriceBasis.CLOSE_TO_OPEN: ("open", "close"),
    PriceBasis.MID_TO_MID: ("high", "low"),
}


def resolve_entry_exit_series(source_data: pd.DataFrame, price_basis: PriceBasis) -> tuple[pd.Series, pd.Series]:
    """Returns `(entry_series, exit_series)` -- the ENTRY price observed
    at row `t`, and the price a forward return's EXIT leg reads at row
    `t + horizon_bars` (via `.shift(-horizon_bars)` in
    `compute_forward_return`, never here). Raises `LabelRequestError`
    if `source_data` is missing a column this basis requires -- a
    structurally invalid request, not a generation-output-contract
    violation (`builder.LabelGenerationContractError`'s concern)."""
    missing = [c for c in _REQUIRED_COLUMNS[price_basis] if c not in source_data.columns]
    if missing:
        raise LabelRequestError(
            f"source_data is missing column(s) {missing} required for price_basis={price_basis.value!r}",
            context={"price_basis": price_basis.value, "missing_columns": missing},
        )

    if price_basis is PriceBasis.CLOSE_TO_CLOSE:
        return source_data["close"], source_data["close"]
    if price_basis is PriceBasis.OPEN_TO_CLOSE:
        return source_data["open"], source_data["close"]
    if price_basis is PriceBasis.CLOSE_TO_OPEN:
        return source_data["close"], source_data["open"]
    if price_basis is PriceBasis.MID_TO_MID:
        mid = (source_data["high"] + source_data["low"]) / 2.0
        return mid, mid
    raise AssertionError(f"Unhandled PriceBasis: {price_basis}")  # pragma: no cover - exhaustive over PriceBasis


def compute_forward_return(source_data: pd.DataFrame, price_basis: PriceBasis, horizon_bars: int) -> pd.Series:
    """`exit_series.shift(-horizon_bars) / entry_series - 1.0` --
    point-in-time safe by construction: row `t`'s value reads EXACTLY
    row `t + horizon_bars`'s exit price, never anything beyond it. The
    trailing `horizon_bars` rows (where the shift has no data to read)
    are `NaN` -- "not enough future data yet," never a fabricated
    value, matching the trailing-NaN-tail shape Part A's own
    `diagnostics.py` already reasons about."""
    if horizon_bars <= 0:
        raise LabelRequestError(f"horizon_bars must be positive, got {horizon_bars}", context={"horizon_bars": horizon_bars})
    entry, exit_ = resolve_entry_exit_series(source_data, price_basis)
    return exit_.shift(-horizon_bars) / entry - 1.0
