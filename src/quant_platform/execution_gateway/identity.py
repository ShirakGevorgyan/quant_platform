"""Shared low-level primitives every content-addressed object in
`quant_platform.execution_gateway` uses: deterministic content
identity (reused directly from `paper_trading.identity` -- Section 2's
own instruction not to duplicate infrastructure that already exists;
the `kind` namespace tag already prevents an execution-gateway object
from ever colliding with a paper-trading object of the same shape) and
canonical Decimal<->JSON round-tripping (Section 4: "Use Decimal for
quantities, prices, monetary values, rates, tolerances, and financial
thresholds whenever practical" -- `json.dumps` has no native Decimal
support, so every `to_json_dict`/`from_json_dict` in this package
serializes a `Decimal` as its exact `str()` form, never as a `float`,
which would silently reintroduce binary floating-point error)."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation

from quant_platform.core.exceptions import ExecutionGatewaySpecError
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.paper_trading.identity import compute_content_id

__all__ = ["compute_content_id", "decimal_from_float", "decimal_to_json", "is_valid_sha256_hex", "parse_decimal"]


def decimal_to_json(value: Decimal) -> str:
    """Exact, lossless string form of `value` for a JSON payload. Never a
    `float` -- `float(Decimal("0.1"))` is not exactly `0.1`, and content
    identity/reconciliation in this package must never depend on binary
    floating-point rounding."""
    if not value.is_finite():
        raise ExecutionGatewaySpecError(f"Decimal value must be finite, got {value!r}")
    return str(value)


def decimal_from_float(value: float, *, field_name: str) -> Decimal:
    """Converts an already-validated `float` field from an upstream
    Milestone 7 object (`OrderRequest.quantity`, `InstrumentSpec.
    contract_multiplier`, ...) into an exact `Decimal` via its `str()`
    form -- NEVER `Decimal(value)` directly, which reproduces the
    binary float's own imprecision (`Decimal(0.1) != Decimal("0.1")`)."""
    if not math.isfinite(value):
        raise ExecutionGatewaySpecError(f"{field_name} must be finite, got {value!r}")
    return Decimal(str(value))


def parse_decimal(raw: object, *, field_name: str) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, Decimal)):
        raise ExecutionGatewaySpecError(f"{field_name} must be a string/int/Decimal, got {type(raw).__name__}")
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ExecutionGatewaySpecError(f"{field_name}={raw!r} is not a valid decimal number") from exc
    if not value.is_finite():
        raise ExecutionGatewaySpecError(f"{field_name} must be finite, got {raw!r}")
    return value
