"""Shared low-level primitives every content-addressed object in
`quant_platform.market_data` uses: deterministic content identity (reused
directly from `paper_trading.identity` -- the same shared building block
`portfolio_risk.identity`/`execution_gateway.identity` already reuse; the
`kind` namespace tag prevents a market-data object from ever colliding
with an object of the same shape from another package), canonical
Decimal<->JSON round-tripping (every price, size, and feature value in
this package is `Decimal` -- see `docs/market_data_architecture.md`'s
"Why Decimal everywhere" section), and the small handful of field
validators every event/feature module in this package needs.

The validators below are centralized here -- rather than each of
`events.py`/`candles.py`/`ticks.py`/`macro.py`/`feature_store.py`
re-declaring its own private copies, which is this repository's usual
per-file convention (see e.g. `portfolio_risk.ledger`'s own private
`_require_tz_aware`) -- specifically because `events.py` imports `Candle`
from `candles.py` and `Tick` from `ticks.py` to build the `MarketDataEvent`
union; if `candles.py`/`ticks.py` in turn imported these helpers from
`events.py` that would be a circular import. Centralizing in `identity.py`
(which every module already depends on for `compute_content_id`, and which
itself depends on nothing else in this package) avoids the cycle without
duplicating the same ~20 lines three times."""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd

from quant_platform.core.exceptions import MarketDataError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id

__all__ = [
    "compute_content_id",
    "decimal_from_float",
    "decimal_to_json",
    "deserialize_timestamp",
    "is_valid_sha256_hex",
    "parse_decimal",
    "require_non_empty",
    "require_non_negative_sequence",
    "require_tz_aware",
    "serialize_timestamp",
]


# --------------------------------------------------------------------------
# Decimal <-> JSON. Never `float` -- `float(Decimal("0.1"))` is not exactly
# `0.1`, and content identity in this package must never depend on binary
# floating-point rounding (mirrors `portfolio_risk.identity` exactly).
# --------------------------------------------------------------------------
def decimal_to_json(value: Decimal) -> str:
    if not value.is_finite():
        raise MarketDataError(f"Decimal value must be finite, got {value!r}")
    return str(value)


def decimal_from_float(value: float, *, field_name: str) -> Decimal:
    if not math.isfinite(value):
        raise MarketDataError(f"{field_name} must be finite, got {value!r}")
    return Decimal(str(value))


def parse_decimal(raw: object, *, field_name: str) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float, Decimal)):
        raise MarketDataError(f"{field_name} must be a string/int/float/Decimal, got {type(raw).__name__}")
    if isinstance(raw, float) and not math.isfinite(raw):
        raise MarketDataError(f"{field_name} must be finite, got {raw!r}")
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise MarketDataError(f"{field_name}={raw!r} is not a valid decimal number") from exc
    if not value.is_finite():
        raise MarketDataError(f"{field_name} must be finite, got {raw!r}")
    return value


# --------------------------------------------------------------------------
# Shared field validators.
# --------------------------------------------------------------------------
def require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise MarketDataError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise MarketDataError(f"{field_name} must not be empty")


def require_non_negative_sequence(sequence: int, *, field_name: str = "sequence") -> None:
    if sequence < 0:
        raise MarketDataError(f"{field_name} must be >= 0, got {sequence}")


def serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    require_tz_aware(ts, field_name=field_name)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise MarketDataError(f"{field_name}: {exc}") from exc


def deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise MarketDataError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise MarketDataError(f"{field_name}: {exc}") from exc
