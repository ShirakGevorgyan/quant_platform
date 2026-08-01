"""Instrument-form and proxy semantics (Milestone 10, Phase 4C) -- the
shared vocabulary `registry.py`/`symbol_mapping.py` both depend on.

THE CENTRAL DISCIPLINE THIS MODULE EXISTS TO ENFORCE: an economic
CONCEPT (e.g. "WTI crude oil") is never the same thing as a specific
TRADABLE INSTRUMENT FORM that approximates it (spot assessment vs. an
exchange-traded futures contract vs. a provider-generated continuous
series vs. an ETF). Conceptually different objects must never be
silently merged: a DXY cash index is not a dollar ETF; a WTI spot
assessment is not front-month CL futures; spot silver is not SI futures
and is not a silver ETF. `InstrumentForm` names the SHAPE of the
tradable object; `ProxyQuality` (declared alongside `is_proxy`/
`proxy_for` wherever a mapping is not the literal underlying) names how
faithfully that shape approximates the economic concept it stands in
for. No code anywhere in this subpackage may label a proxy instrument
as the underlying it approximates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.core.exceptions import InstrumentFormError

__all__ = [
    "InstrumentForm",
    "ProxyPolicy",
    "ProxyQuality",
    "create_proxy_policy",
]


class InstrumentForm(Enum):
    SPOT = "spot"
    """An OTC/interbank spot price assessment (e.g. XAUUSD spot,
    provider-specific -- see `sessions.py`'s own discussion of why
    OTC spot session semantics are never centralized-exchange truth)."""

    CASH_INDEX = "cash_index"
    """A calculated cash/spot INDEX (e.g. a dollar index) -- not itself
    directly tradable, distinct from any ETF/futures product that
    tracks it."""

    EXCHANGE_FUTURES_CONTRACT = "exchange_futures_contract"
    """One SPECIFIC, individually identified exchange-traded futures
    contract (a real expiry, e.g. "front-month CL for delivery next
    month") -- requires `futures.FuturesContractMetadata`."""

    PROVIDER_CONTINUOUS_FUTURES = "provider_continuous_futures"
    """A provider-GENERATED continuous futures series -- NOT raw
    individual-contract history; requires `futures.ContinuationPolicy`
    and roll provenance. Never conflated with
    `EXCHANGE_FUTURES_CONTRACT`."""

    ETF = "etf"
    """An exchange-traded fund tracking (never identical to) an
    underlying concept -- always `is_proxy=True` wherever mapped."""

    EQUITY = "equity"
    """A single company's equity (e.g. a gold-miner equity proxy)."""

    SYNTHETIC_INDEX = "synthetic_index"
    """A provider-constructed synthetic/composite index with no single
    underlying tradable instrument (e.g. a broad commodity index)."""

    ECONOMIC_PROXY = "economic_proxy"
    """A catch-all for an instrument that approximates an economic
    concept through neither a direct spot/futures/ETF/equity relationship
    (reserved for a future, explicitly documented case; unused by any
    mapping this phase ships)."""


class ProxyQuality(Enum):
    HIGH = "high"
    """A tight, well-understood tracking relationship (e.g. a
    physically-backed, single-underlying ETF like GLD for spot gold)."""

    MODERATE = "moderate"
    """A useful but imperfect proxy carrying material, disclosed basis/
    tracking-error risk (e.g. a futures-based commodity ETF subject to
    roll cost, like USO for WTI)."""

    LOW = "low"
    """A weak or indirect proxy -- included for REGIME CONTEXT only,
    never as a primary driver signal (e.g. an equity-market stress ETF
    standing in for implied volatility)."""


@dataclass(frozen=True, slots=True)
class ProxyPolicy:
    is_proxy: bool
    proxy_for: str | None
    """The `canonical_driver_id` this instrument approximates -- required
    whenever `is_proxy=True`, forbidden otherwise."""
    proxy_quality: ProxyQuality | None
    known_basis_risk: str
    roll_risk: str
    tracking_error_risk: str
    currency_difference_note: str
    session_difference_note: str
    adjustment_difference_note: str

    def __post_init__(self) -> None:
        if self.is_proxy:
            if not self.proxy_for:
                raise InstrumentFormError("ProxyPolicy.proxy_for is required when is_proxy=True")
            if self.proxy_quality is None:
                raise InstrumentFormError("ProxyPolicy.proxy_quality is required when is_proxy=True")
        else:
            if self.proxy_for is not None:
                raise InstrumentFormError("ProxyPolicy.proxy_for must be None when is_proxy=False")
            if self.proxy_quality is not None:
                raise InstrumentFormError("ProxyPolicy.proxy_quality must be None when is_proxy=False")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "is_proxy": self.is_proxy, "proxy_for": self.proxy_for,
            "proxy_quality": (None if self.proxy_quality is None else self.proxy_quality.value),
            "known_basis_risk": self.known_basis_risk, "roll_risk": self.roll_risk,
            "tracking_error_risk": self.tracking_error_risk, "currency_difference_note": self.currency_difference_note,
            "session_difference_note": self.session_difference_note, "adjustment_difference_note": self.adjustment_difference_note,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ProxyPolicy:
        return cls(
            is_proxy=bool(raw["is_proxy"]), proxy_for=(None if raw.get("proxy_for") is None else str(raw["proxy_for"])),
            proxy_quality=(None if raw.get("proxy_quality") is None else ProxyQuality(raw["proxy_quality"])),
            known_basis_risk=str(raw["known_basis_risk"]), roll_risk=str(raw["roll_risk"]),
            tracking_error_risk=str(raw["tracking_error_risk"]), currency_difference_note=str(raw["currency_difference_note"]),
            session_difference_note=str(raw["session_difference_note"]), adjustment_difference_note=str(raw["adjustment_difference_note"]),
        )


def create_proxy_policy(
    *, is_proxy: bool, proxy_for: str | None = None, proxy_quality: ProxyQuality | None = None, known_basis_risk: str = "none declared",
    roll_risk: str = "none declared", tracking_error_risk: str = "none declared", currency_difference_note: str = "none declared",
    session_difference_note: str = "none declared", adjustment_difference_note: str = "none declared",
) -> ProxyPolicy:
    """`ProxyPolicy.__post_init__` already enforces `proxy_for`/
    `proxy_quality` being required iff `is_proxy=True` (raising
    `InstrumentFormError`) -- this factory does not duplicate that check
    itself, so every violation surfaces through exactly ONE exception
    type regardless of construction path."""
    return ProxyPolicy(
        is_proxy=is_proxy, proxy_for=proxy_for, proxy_quality=proxy_quality, known_basis_risk=known_basis_risk, roll_risk=roll_risk,
        tracking_error_risk=tracking_error_risk, currency_difference_note=currency_difference_note,
        session_difference_note=session_difference_note, adjustment_difference_note=adjustment_difference_note,
    )
